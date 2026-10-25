"""Figure 7 — physics simulation and walk-intent gate

gate — the EEG state classifier component of the three-part architecture

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os
import sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gate_features import (BANDS, WIN_S, BASE_N, SESSIONS,
                                    load_session, windows, bandpower, build,
                                    calib_split)

HOP_S = 1.0
BASE_MIN_S = 30.0
IQR_K = 0.25
N_ON = 1
CLOSE_M, CLOSE_N = 3, 3
N_OFF = CLOSE_M
OFF_Q = 90
DWELL_S = 8.0
SPEC_ON = 0.90
SPEC_OFF = 0.50


def session_baseline(root, sess, min_s=BASE_MIN_S):
    """The session's calibration segment: P_baseline plus everything needed to score it."""
    X, fs, ch, ev = load_session(root, sess)
    rest = [e for e in ev if e["code"] == "rest"]
    calib, _ = calib_split(rest, fs, min_s)
    bw = [bandpower(w, fs) for e in calib for w in windows(X, e["a"], e["b"], fs)]
    if not bw:
        raise RuntimeError(f"{sess}: no calibration windows")
    return np.median(np.vstack(bw), axis=0), X, fs, ch, ev, calib


def calib_scores(root, sess, model, min_s=BASE_MIN_S):
    """Gate scores over the calibration segment, plus its ledger (K, seconds, window count)."""
    P_base, X, fs, ch, ev, calib = session_baseline(root, sess, min_s)
    S = []
    for e in calib:
        for w in windows(X, e["a"], e["b"], fs):
            f = np.log10((bandpower(w, fs) + 1e-20) / (P_base + 1e-20))
            S.append(float(model["clf"].decision_function(
                model["scaler"].transform(f[None, :]))[0]))
    S = np.asarray(S)
    led = dict(K=len(calib), cum_s=round(sum(e["b"] - e["a"] for e in calib) / fs, 1),
               n_win=len(S))
    return S, led


def set_thresholds(model, S):
    """theta_on = calib max + IQR_K * IQR(calib); theta_off = p90 of the calibration scores."""
    iqr = float(np.percentile(S, 75) - np.percentile(S, 25))
    model["theta_on"] = float(S.max() + IQR_K * iqr)
    model["theta_off"] = float(np.percentile(S, OFF_Q))
    model["calib_iqr"] = iqr
    model["calib_max"] = float(S.max())
    return model


def fit_gate(rows, demo_session, spec_on=SPEC_ON, spec_off=SPEC_OFF, n_on=N_ON, offset=0.0,
             name="OP-A"):
    """LOSO fold model for `demo_session` + a hysteresis operating point."""
    tr = [r for r in rows if r["session"] != demo_session]
    X = np.vstack([r["corrected"] for r in tr])
    y = np.array([r["label"] for r in tr])
    sc = StandardScaler().fit(X)
    clf = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(X), y)
    d_rest = clf.decision_function(sc.transform(X[y == 0]))
    return dict(scaler=sc, clf=clf, name=name, offset=float(offset),
                spec_on=spec_on, spec_off=spec_off, n_on=n_on,
                theta_on=float(np.quantile(d_rest, spec_on)),
                theta_off=float(np.quantile(d_rest, spec_off)),
                n_train=len(tr), n_train_rest=int((y == 0).sum()),
                train_sessions=sorted({r["session"] for r in tr}))


def episode_spec(n_windows, alpha=0.05):
    """Per-window specificity needed for an episode-level false-open probability <= alpha over
    a rest episode of n_windows independent decisions: (1-alpha)**(1/n)."""
    return float((1.0 - alpha) ** (1.0 / max(n_windows, 1)))


def baseline_offset(root, sess, model):
    """Median gate score over the session's own reserved baseline rest windows."""
    P_base, X, fs, ch, ev, base_ev = session_baseline(root, sess)
    S = []
    for e in base_ev:
        for w in windows(X, e["a"], e["b"], fs):
            f = np.log10((bandpower(w, fs) + 1e-20) / (P_base + 1e-20))
            S.append(float(model["clf"].decision_function(
                model["scaler"].transform(f[None, :]))[0]))
    return float(np.median(S)) if S else 0.0, len(S)


def score_stream(X, fs, P_base, model, splices=(), segments=None):
    """Slide the gate over the EEG stream."""
    n = int(round(WIN_S * fs))
    hop = int(round(HOP_S * fs))
    if segments is None:
        segments = [(0, len(X), "stream")]
    T, S, K = [], [], []
    for gi, (a, b, _kind) in enumerate(segments):
        s = a
        while s + n <= b:
            P = bandpower(X[s:s + n], fs)
            f = np.log10((P + 1e-20) / (P_base + 1e-20))
            S.append(float(model["clf"].decision_function(
                model["scaler"].transform(f[None, :]))[0]) - model.get("offset", 0.0))
            T.append((s + n) / fs)
            K.append(gi)
            s += hop
    o = np.argsort(T)
    return np.array(T)[o], np.array(S)[o], np.array(K, int)[o]


def hysteresis(t_end, score, model, total_s, fs, force_open=False):
    """Per-sample gate state (0 idle / 1 move) over the whole timeline."""
    n_samp = int(round(total_s * fs))
    if force_open:
        return np.ones(n_samp, int), []
    state = np.zeros(n_samp, int)
    cur, above, t_open = 0, 0, -1e9
    hist = []
    n_on = model.get("n_on", N_ON)
    events = []
    for i, (te, sc) in enumerate(zip(t_end, score)):
        if cur == 0:
            above = above + 1 if sc > model["theta_on"] else 0
            if above >= n_on:
                cur, t_open, above, hist = 1, te, 0, []
                events.append((te, "open", sc))
        else:
            hist.append(bool(sc < model["theta_off"]))
            if len(hist) > CLOSE_N:
                hist.pop(0)
            if sum(hist) >= CLOSE_M and (te - t_open) >= DWELL_S:
                cur, hist = 0, []
                events.append((te, "close", sc))
        j0 = int(round(te * fs))
        j1 = int(round(t_end[i + 1] * fs)) if i + 1 < len(t_end) else n_samp
        state[max(j0, 0):max(j1, 0)] = cur
    return state, events


def params_block(model=None):
    if model is None:
        return (f"gate: {WIN_S:.0f} s window, {HOP_S:.0f} s hop; theta_on at {SPEC_ON:.0%} / "
                f"theta_off at {SPEC_OFF:.0%} specificity on TRAINING rest; open after {N_ON} "
                f"window, close after {N_OFF} consecutive, minimum dwell {DWELL_S:.0f} s")
    return (f"{WIN_S:.0f} s window, {HOP_S:.0f} s hop (zero overlap); theta_on = calib max + "
            f"{IQR_K} x IQR = {model['theta_on']:+.3f}, theta_off = calib p90 = "
            f"{model['theta_off']:+.3f} (calib p{OFF_Q}); open after "
            f"{model.get('n_on', N_ON)} window, close on {CLOSE_M} of the last {CLOSE_N}, "
            f"minimum dwell {DWELL_S:.0f} s")
