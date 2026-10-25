"""Figure 7 — physics simulation and walk-intent gate

gate_features — band-power classification of imagined walking vs rest in patient EEG (leave-one-session-out).

Reads:
  the aligned per-session recordings (not distributed)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import scipy.io as sio
import h5py
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SESSIONS = ["pat01_sess04", "pat01_sess06", "pat01_sess07", "pat01_sess09",
            "pat02_sess03", "pat02_sess05", "pat02_sess06"]
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "mu": (8.0, 13.0), "beta": (13.0, 30.0)}
WIN_S = 4.0
BASE_N = 4
FS_EXPECT = 250.0


def load_session(root, sess):
    d = os.path.join(root, sess)
    with h5py.File(os.path.join(d, f"{sess}_eeg.mat"), "r") as h:
        fs = float(np.array(h["eeg/s"]).ravel()[0])
        X = np.array(h["eeg/x"])
        lab = h["eeg/c/labels"]
        ch = ["".join(chr(c) for c in np.array(h[lab[i, 0]]).ravel())
              for i in range(lab.shape[0])]
    m = sio.loadmat(os.path.join(d, f"{sess}_event.mat"), squeeze_me=True,
                    struct_as_record=False)
    ev = m["event"] if "event" in m else m["events"]
    out = []
    for e in ev:
        if str(e.qc) != "good":
            continue
        out.append(dict(code=str(e.code), a=int(round(float(e.etstart))),
                        b=int(round(float(e.etstop)))))
    out.sort(key=lambda r: r["a"])
    return X, fs, ch, out


def windows(X, a, b, fs):
    """Non-overlapping WIN_S windows tiled from the event start."""
    n = int(round(WIN_S * fs))
    out = []
    s = a
    while s + n <= min(b, len(X)):
        out.append(X[s:s + n])
        s += n
    return out


def bandpower(W, fs):
    """[n_ch * n_band] Welch band power of one window, band order = BANDS order."""
    f, P = welch(W, fs=fs, nperseg=min(len(W), int(fs * 2)), axis=0)
    feats = []
    for lo, hi in BANDS.values():
        m = (f >= lo) & (f < hi)
        feats.append(P[m].mean(axis=0))
    return np.concatenate(feats)


def calib_split(rest, fs, min_s=None):
    """(calibration events, evaluated events)."""
    if min_s is None:
        return rest[:BASE_N], rest[BASE_N:]
    cum, k = 0.0, 0
    while k < len(rest) and cum < min_s:
        cum += (rest[k]["b"] - rest[k]["a"]) / fs
        k += 1
    return rest[:k], rest[k:]


def build(root, base_min_s=None):
    rows, meta = [], {}
    for sess in SESSIONS:
        X, fs, ch, ev = load_session(root, sess)
        assert abs(fs - FS_EXPECT) < 1e-6, f"{sess}: fs={fs}"
        rest = [e for e in ev if e["code"] == "rest"]
        iw = [e for e in ev if e["code"] == "gaitimagery"]
        base_ev, eval_rest = calib_split(rest, fs, base_min_s)
        bw = [bandpower(w, fs) for e in base_ev for w in windows(X, e["a"], e["b"], fs)]
        if not bw:
            raise RuntimeError(f"{sess}: no baseline windows")
        P_base = np.median(np.vstack(bw), axis=0)
        n_skip = 0
        for lab, evs in (("iw", iw), ("rest", eval_rest)):
            for k, e in enumerate(evs):
                ws = windows(X, e["a"], e["b"], fs)
                if not ws:
                    n_skip += 1
                    continue
                for wi, w in enumerate(ws):
                    P = bandpower(w, fs)
                    rows.append(dict(session=sess, label=1 if lab == "iw" else 0,
                                     event=f"{lab}{k}", win=wi,
                                     raw=np.log10(P + 1e-20),
                                     corrected=np.log10((P + 1e-20) / (P_base + 1e-20))))
        span = (ev[0]["a"], ev[-1]["b"])
        meta[sess] = dict(
            n_ch=len(ch), fs=fs, channels=ch,
            n_iw_ev=len(iw), n_rest_ev=len(rest), n_rest_eval=len(eval_rest),
            baseline_events=len(base_ev),
            baseline_s=round(sum(e["b"] - e["a"] for e in base_ev) / fs, 1),
            baseline_span_frac=round((np.mean([e["a"] for e in base_ev]) - span[0])
                                     / max(span[1] - span[0], 1), 3),
            n_win_iw=sum(1 for r in rows if r["session"] == sess and r["label"] == 1),
            n_win_rest=sum(1 for r in rows if r["session"] == sess and r["label"] == 0),
            events_too_short=n_skip)
        print(f"{sess}: IW {len(iw)} ev / {meta[sess]['n_win_iw']} win, "
              f"rest {len(eval_rest)}(+{BASE_N} baseline) ev / {meta[sess]['n_win_rest']} win, "
              f"baseline {meta[sess]['baseline_s']}s at session fraction "
              f"{meta[sess]['baseline_span_frac']}, short events skipped {n_skip}", flush=True)
    return rows, meta


def loso(rows, key, model="lda", labels=None):
    """Leave-one-session-out. Returns per-session dicts and the fold coefficient matrix."""
    y_all = np.array([r["label"] for r in rows]) if labels is None else np.asarray(labels)
    X_all = np.vstack([r[key] for r in rows])
    sess = np.array([r["session"] for r in rows])
    ev = np.array([r["session"] + "|" + r["event"] for r in rows])
    per, coefs = [], []
    for s in SESSIONS:
        te = sess == s
        tr = ~te
        if len(np.unique(y_all[tr])) < 2 or len(np.unique(y_all[te])) < 2:
            continue
        sc = StandardScaler().fit(X_all[tr])
        clf = (LDA(solver="lsqr", shrinkage="auto") if model == "lda"
               else LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(sc.transform(X_all[tr]), y_all[tr])
        d = clf.decision_function(sc.transform(X_all[te]))
        coefs.append(np.ravel(clf.coef_))
        auc_w = roc_auc_score(y_all[te], d)
        acc_w = float(np.mean((d > 0).astype(int) == y_all[te]))
        te_ev = ev[te]
        ue = np.unique(te_ev)
        dt = np.array([d[te_ev == e].mean() for e in ue])
        yt = np.array([y_all[te][te_ev == e][0] for e in ue])
        per.append(dict(session=s, n_win=int(te.sum()), n_ev=len(ue),
                        auc_win=auc_w, acc_win=acc_w,
                        auc_trial=roc_auc_score(yt, dt) if len(np.unique(yt)) > 1 else np.nan,
                        acc_trial=float(np.mean((dt > 0).astype(int) == yt))))
    return per, (np.vstack(coefs) if coefs else np.zeros((0, X_all.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="${GAIT_PROJECT_ROOT}/patient_prep_restgood_v4_7")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "result"))
    ap.add_argument("--nperm", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, meta = build(args.root)
    ch = meta[SESSIONS[0]]["channels"]
    names = [f"{b}:{c}" for b in BANDS for c in ch]

    res, coefs = {}, {}
    for key in ("raw", "corrected"):
        for model in ("lda", "logreg"):
            per, C = loso(rows, key, model)
            res[(key, model)] = per
            coefs[(key, model)] = C

    rng = np.random.default_rng(20260829)
    y = np.array([r["label"] for r in rows])
    sess = np.array([r["session"] for r in rows])
    null = []
    for _ in range(args.nperm):
        yp = y.copy()
        for s in SESSIONS:
            m = sess == s
            yp[m] = rng.permutation(yp[m])
        per, _ = loso(rows, "corrected", "lda", labels=yp)
        if per:
            null.append(float(np.mean([p["auc_win"] for p in per])))
    null = np.array(null)

    L = []
    
    with open(os.path.join(args.out, "gate_auc.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["features", "model", "session", "n_win", "n_ev",
                                           "auc_win", "acc_win", "auc_trial", "acc_trial"])
        w.writeheader()
        for (key, model), per in res.items():
            for p in per:
                w.writerow(dict(p, features=key, model=model))
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "channels"}
               for k, v in meta.items()},
              open(os.path.join(args.out, "gate_inventory.json"), "w"), indent=1)
    print("\n".join(L[:60]), flush=True)




if __name__ == "__main__":
    main()
