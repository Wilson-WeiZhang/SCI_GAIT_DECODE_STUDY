"""Figure 7 — physics simulation and walk-intent gate

gate_fusion — fused band-power (LDA) and Riemannian tangent-space scores of the walk-intent gate, leave-one-session-out.

Reads:
  the aligned per-session recordings (not distributed)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import csv
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gate_features import (BANDS, SESSIONS, bandpower, build,
                                    calib_split, load_session, windows)

ROOT = "${GAIT_PROJECT_ROOT}/patient_prep_restgood_v4_7"
RES = os.path.join(HERE, "result")

BASE_MIN_S = 30.0
BASE_N = 4
WIN_S, HOP_S = 4.0, 1.0
IQR_K, OFF_Q = 0.25, 90
LAM_RA = 1e-3
LAM_SWEEP = 1e-10
LAM_SENS = (1e-6, 1e-4, 1e-3, 1e-2)
C_GRID = [0.01, 0.1, 1.0]

GATE_A_MEAN = 0.9037
GATE_A_PER = {"pat01_sess04": 0.931, "pat01_sess06": 0.947, "pat01_sess07": 0.961,
              "pat01_sess09": 0.941, "pat02_sess03": 0.877, "pat02_sess05": 0.904,
              "pat02_sess06": 0.766}
SWEEP_CSV = os.path.join(RES, "gate_model_sweep.csv")
DENSE_CSV = os.path.join(RES, "gate_dense_hop_scores.csv")


def _eig(C):
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    return np.maximum(w, 1e-20), V


def logm_sym(C):
    w, V = _eig(C)
    return (V * np.log(w)) @ V.T


def powm_sym(C, p):
    w, V = _eig(C)
    return (V * (w ** p)) @ V.T


def _expm_sym(S):
    w, V = np.linalg.eigh((S + S.T) / 2.0)
    return (V * np.exp(w)) @ V.T


def riemann_mean(Cs, tol=1e-10, itmax=60):
    """AIRM (Karcher) mean by the standard fixed-point iteration, init = arithmetic mean."""
    R = np.mean(Cs, axis=0)
    for _ in range(itmax):
        Rm, Rp = powm_sym(R, -0.5), powm_sym(R, 0.5)
        S = np.mean([logm_sym(Rm @ C @ Rm) for C in Cs], axis=0)
        R = Rp @ _expm_sym(S) @ Rp
        if np.linalg.norm(S, "fro") < tol:
            break
    return (R + R.T) / 2.0


def shrink(C, lam):
    n = C.shape[0]
    return (1.0 - lam) * C + lam * (np.trace(C) / n) * np.eye(n)


def sweep_ridge(C):
    """The sweep's exact regulariser, so gate B is a reproduction and not an approximation."""
    return C + LAM_SWEEP * np.trace(C) / C.shape[0] * np.eye(C.shape[0])


def vec(L):
    return L[np.triu_indices(L.shape[0])]


_SOS = {b: butter(4, (lo, hi), btype="band", fs=250.0) for b, (lo, hi) in BANDS.items()}


def band_covs(w):
    """The 4 band covariances of one window, band order = BANDS order."""
    out = []
    for bb, ab in _SOS.values():
        wf = filtfilt(bb, ab, w, axis=0)
        out.append((wf.T @ wf) / len(wf))
    return out


def sweep_feat(covs):
    return np.concatenate([vec(logm_sym(sweep_ridge(C))) for C in covs])


def ra_feat(covs, whit, lam=LAM_RA):
    """Riemannian-aligned log map: vec(log(W C W)) per band, W = R^-1/2 of the rest reference."""
    out = []
    for C, W in zip(covs, whit):
        Cs = shrink(C, lam)
        out.append(vec(logm_sym(W @ Cs @ W)))
    return np.concatenate(out)


def session_layout(root, sess, base_min_s):
    """(X, fs, calib_events, eval_events) with eval_events in build()'s exact order/naming."""
    X, fs, _ch, ev = load_session(root, sess)
    rest = [e for e in ev if e["code"] == "rest"]
    iw = [e for e in ev if e["code"] == "gaitimagery"]
    if base_min_s is None:
        cal, eval_rest = rest[:BASE_N], rest[BASE_N:]
    else:
        cal, eval_rest = calib_split(rest, fs, base_min_s)
    ordered_eval = [("iw", k, e) for k, e in enumerate(iw)] + \
                   [("rest", k, e) for k, e in enumerate(eval_rest)]
    return X, fs, cal, ordered_eval


def dense_windows(X, a, b, fs):
    """4 s history windows at 1 s hop from the event start -> (t_end_from_onset, window)."""
    n, hop = int(round(WIN_S * fs)), int(round(HOP_S * fs))
    out, s, k = [], a, 0
    while s + n <= min(b, len(X)):
        out.append((k * HOP_S + WIN_S, X[s:s + n]))
        s += hop
        k += 1
    return out


def cov_inventory(root, base_min_s):
    """Covariance stacks on the 4 s zero-overlap grid, in build()'s order."""
    keys, covs, calib = [], [], {}
    for sess in SESSIONS:
        X, fs, cal, ordered_eval = session_layout(root, sess, base_min_s)
        calib[sess] = [band_covs(w) for e in cal for w in windows(X, e["a"], e["b"], fs)]
        for lab, k, e in ordered_eval:
            for wi, w in enumerate(windows(X, e["a"], e["b"], fs)):
                keys.append((sess, f"{lab}{k}", wi))
                covs.append(band_covs(w))
        print(f"  [cov {'demobase' if base_min_s else 'dumpbase'}] {sess}: "
              f"{len(calib[sess])} calib win, {len(keys)} eval win cumulative", flush=True)
    return keys, covs, calib


def whiteners(calib_covs, lam=LAM_RA):
    """Per band: R = Karcher mean of the shrunk calibration covariances, W = R^-1/2."""
    nb = len(BANDS)
    out = []
    for b in range(nb):
        Cs = [shrink(c[b], lam) for c in calib_covs]
        out.append(powm_sym(riemann_mean(Cs), -0.5))
    return out


def zsc(d):
    return (d - d.mean()) / (d.std() + 1e-12)


def fit_ts(Xtr, ytr, gtr):
    gs = GridSearchCV(LogisticRegression(max_iter=5000), {"C": C_GRID}, scoring="roc_auc",
                      cv=GroupKFold(n_splits=3), n_jobs=8)
    gs.fit(Xtr, ytr, groups=gtr)
    return gs.best_estimator_, gs.best_params_["C"]


def gate_a(rows_demo):
    y = np.array([r["label"] for r in rows_demo])
    X = np.vstack([r["corrected"] for r in rows_demo])
    sess = np.array([r["session"] for r in rows_demo])
    ev = np.array([r["session"] + "|" + r["event"] for r in rows_demo])
    out, aucs = [], []
    for s in SESSIONS:
        te = sess == s
        sc = StandardScaler().fit(X[~te])
        clf = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(X[~te]), y[~te])
        d = clf.decision_function(sc.transform(X[te]))
        a = roc_auc_score(y[te], d)
        ue = np.unique(ev[te])
        at = roc_auc_score([y[te][ev[te] == e][0] for e in ue],
                           [d[ev[te] == e].mean() for e in ue])
        aucs.append(a)
        out.append(dict(block="gateA_demobase_bp_lda", variant="-", session=s,
                        n_win=int(te.sum()), auc_win=round(a, 6), auc_trial=round(at, 6)))
        assert abs(a - GATE_A_PER[s]) < 5e-4, f"GATE A {s}: {a:.4f} != frozen {GATE_A_PER[s]}"
    m = float(np.mean(aucs))
    assert abs(m - GATE_A_MEAN) < 5e-4, f"GATE A mean {m:.4f} != {GATE_A_MEAN}"
    out.append(dict(block="gateA_demobase_bp_lda", variant="-", session="MEAN7",
                    auc_win=round(m, 6),
                    auc_trial=round(float(np.mean([r["auc_trial"] for r in out])), 6)))
    print(f"[GATE A] demobase bp-LDA LOSO window AUC {m:.4f} == frozen {GATE_A_MEAN} OK",
          flush=True)
    return out


def gate_b(rows_dump, keys_dump, covs_dump):
    assert [(r["session"], r["event"], r["win"]) for r in rows_dump] == keys_dump, \
        "gate B: cov inventory diverged from build() (dumpbase)"
    frozen = {r["session"]: float(r["auc_win"])
              for r in csv.DictReader(open(SWEEP_CSV, encoding="utf-8"))
              if r["method"] == "fuse_bp_ts"}
    y = np.array([r["label"] for r in rows_dump])
    Xbp = np.vstack([r["corrected"] for r in rows_dump])
    sess = np.array([r["session"] for r in rows_dump])
    ev = np.array([r["session"] + "|" + r["event"] for r in rows_dump])
    Xcov = np.vstack([sweep_feat(c) for c in covs_dump])
    Xrc = Xcov.copy()
    for s in SESSIONS:
        m = sess == s
        Xrc[m] -= Xcov[m].mean(axis=0, keepdims=True)

    out, aucs = [], []
    for s in SESSIONS:
        te, tr = sess == s, sess != s
        sa = StandardScaler().fit(Xbp[tr])
        dl = LDA(solver="lsqr", shrinkage="auto").fit(
            sa.transform(Xbp[tr]), y[tr]).decision_function(sa.transform(Xbp[te]))
        sb = StandardScaler().fit(Xrc[tr])
        est, _ = fit_ts(sb.transform(Xrc[tr]), y[tr], sess[tr])
        dt = est.decision_function(sb.transform(Xrc[te]))
        df = (zsc(dl) + zsc(dt)) / 2
        a = roc_auc_score(y[te], df)
        ue = np.unique(ev[te])
        at = roc_auc_score([y[te][ev[te] == e][0] for e in ue],
                           [df[ev[te] == e].mean() for e in ue])
        assert abs(a - frozen[s]) < 6e-5, f"GATE B {s}: {a:.6f} != sweep {frozen[s]}"
        aucs.append(a)
        out.append(dict(block="gateB_sweep_fuse2_dumpbase", variant="-", session=s,
                        n_win=int(te.sum()), auc_win=round(a, 6), auc_trial=round(at, 6)))
    m = float(np.mean(aucs))
    out.append(dict(block="gateB_sweep_fuse2_dumpbase", variant="-", session="MEAN7",
                    auc_win=round(m, 6)))
    print(f"[GATE B] sweep fuse2 reproduced: mean window AUC {m:.4f}  OK",
          flush=True)
    return out


def deployable(rows_demo, keys_demo, covs_demo, calib_demo, lam=LAM_RA, dense=True):
    """Fit the two conditions per LOSO fold on the RA-rest representation and emit every z variant."""
    assert [(r["session"], r["event"], r["win"]) for r in rows_demo] == keys_demo, \
        "cov inventory diverged from build() (demobase)"
    y = np.array([r["label"] for r in rows_demo])
    Xbp = np.vstack([r["corrected"] for r in rows_demo])
    sess = np.array([r["session"] for r in rows_demo])
    ev = np.array([r["session"] + "|" + r["event"] for r in rows_demo])

    whit = {s: whiteners(calib_demo[s], lam) for s in SESSIONS}
    Xra = np.vstack([ra_feat(c, whit[k[0]], lam) for c, k in zip(covs_demo, keys_demo)])
    Xcal = {s: np.vstack([ra_feat(c, whit[s], lam) for c in calib_demo[s]]) for s in SESSIONS}
    Pbase = {}
    Bcal = {}
    for s in SESSIONS:
        X, fs, cal, _ = session_layout(ROOT, s, BASE_MIN_S)
        bw = np.vstack([bandpower(w, fs) for e in cal for w in windows(X, e["a"], e["b"], fs)])
        Pbase[s] = np.median(bw, axis=0)
        Bcal[s] = np.log10((bw + 1e-20) / (Pbase[s] + 1e-20))

    auc_rows, dense_rows, thr_rows, calib_rows = [], [], [], []
    per = {v: [] for v in ("V1", "V2", "V3")}
    for s in SESSIONS:
        te, tr = sess == s, sess != s
        sa = StandardScaler().fit(Xbp[tr])
        cla = LDA(solver="lsqr", shrinkage="auto").fit(sa.transform(Xbp[tr]), y[tr])
        sb = StandardScaler().fit(Xra[tr])
        clb, Cbest = fit_ts(sb.transform(Xra[tr]), y[tr], sess[tr])

        dA = cla.decision_function(sa.transform(Xbp[te]))
        dB = clb.decision_function(sb.transform(Xra[te]))
        cA = cla.decision_function(sa.transform(Bcal[s]))
        cB = clb.decision_function(sb.transform(Xcal[s]))
        sdA_tr = float(cla.decision_function(sa.transform(Xbp[tr])).std())
        sdB_tr = float(clb.decision_function(sb.transform(Xra[tr])).std())

        SC = dict(V1=(sdA_tr, sdB_tr),
                  V2=(float(cA.std()) + 1e-12, float(cB.std()) + 1e-12),
                  V3=(float(dA.std()) + 1e-12, float(dB.std()) + 1e-12))
        mA, mB = float(np.median(cA)), float(np.median(cB))
        for v, (sA, sB) in SC.items():
            zA, zB = (dA - mA) / sA, (dB - mB) / sB
            f = (zA + zB) / 2
            a = roc_auc_score(y[te], f)
            ue = np.unique(ev[te])
            at = roc_auc_score([y[te][ev[te] == e][0] for e in ue],
                               [f[ev[te] == e].mean() for e in ue])
            per[v].append(a)
            auc_rows.append(dict(block=f"deployable_fuse2_ra_lam{lam:g}", variant=v, session=s,
                                 n_win=int(te.sum()), auc_win=round(a, 6), auc_trial=round(at, 6),
                                 lr_C=Cbest, sd_A=round(sA, 6), sd_B=round(sB, 6),
                                 sd_ratio=round(sA / sB, 6)))
        for nm, d in (("armA_bp_lda", dA), ("classifierB_ts_lr_ra", dB)):
            auc_rows.append(dict(block=f"deployable_condition_lam{lam:g}", variant=nm, session=s,
                                 n_win=int(te.sum()), auc_win=round(roc_auc_score(y[te], d), 6)))
        if not dense:
            continue

        for v, (sA, sB) in SC.items():
            fc = ((cA - mA) / sA + (cB - mB) / sB) / 2
            iqr = float(np.percentile(fc, 75) - np.percentile(fc, 25))
            thr_rows.append(dict(session=s, variant=v, n_calib_win=len(fc),
                                 calib_max=float(fc.max()), calib_iqr=iqr,
                                 theta_on=float(fc.max() + IQR_K * iqr),
                                 theta_off=float(np.percentile(fc, OFF_Q)),
                                 sd_A=sA, sd_B=sB, centre_A=mA, centre_B=mB))
            for i, x in enumerate(fc):
                calib_rows.append(dict(session=s, variant=v, win=i, score_fuse=float(x)))
        for i, (a_, b_) in enumerate(zip(cA, cB)):
            calib_rows.append(dict(session=s, variant="conditions", win=i, score_lda=float(a_),
                                   score_ts=float(b_)))

        X, fs, _cal, ordered_eval = session_layout(ROOT, s, BASE_MIN_S)
        for lab, k, e in ordered_eval:
            dw = dense_windows(X, e["a"], e["b"], fs)
            if not dw:
                continue
            fb = np.vstack([np.log10((bandpower(w, fs) + 1e-20) / (Pbase[s] + 1e-20))
                            for _t, w in dw])
            fr = np.vstack([ra_feat(band_covs(w), whit[s], lam) for _t, w in dw])
            gA = cla.decision_function(sa.transform(fb))
            gB = clb.decision_function(sb.transform(fr))
            for (t_end, _w), va, vb in zip(dw, gA, gB):
                r = dict(session=s, event=f"{lab}{k}", label=1 if lab == "iw" else 0,
                         t_end_ev_s=round(t_end, 4),
                         ev_dur_s=round((e["b"] - e["a"]) / fs, 4), n_dec_ev=len(dw),
                         score_lda=float(va), score_ts=float(vb))
                for v, (sA, sB) in SC.items():
                    r[f"fuse_{v}"] = float(((va - mA) / sA + (vb - mB) / sB) / 2)
                dense_rows.append(r)
        print(f"  [deployable lam={lam:g}] {s}: LR C={Cbest}, "
              f"AUC V1 {per['V1'][-1]:.4f} V2 {per['V2'][-1]:.4f} V3 {per['V3'][-1]:.4f}",
              flush=True)

    for v in ("V1", "V2", "V3"):
        auc_rows.append(dict(block=f"deployable_fuse2_ra_lam{lam:g}", variant=v, session="MEAN7",
                             auc_win=round(float(np.mean(per[v])), 6)))
        print(f"[deployable lam={lam:g}] {v} mean window AUC {np.mean(per[v]):.4f}", flush=True)
    return auc_rows, dense_rows, thr_rows, calib_rows


def gate_c(dense_rows):
    """The band-power condition of the dense stream must BE the frozen densegrid LDA stream."""
    frozen = {(r["session"], r["event"], round(float(r["t_end_ev_s"]), 4)): float(r["score"])
              for r in csv.DictReader(open(DENSE_CSV, encoding="utf-8"))}
    worst, n = 0.0, 0
    for r in dense_rows:
        key = (r["session"], r["event"], round(r["t_end_ev_s"], 4))
        assert key in frozen, f"GATE C: dense window {key} absent from the frozen stream"
        worst = max(worst, abs(r["score_lda"] - frozen[key]))
        n += 1
    assert n == len(frozen), f"GATE C: covered {n} of {len(frozen)} frozen dense windows"
    assert worst < 1e-9, f"GATE C: band-power condition deviates from frozen densegrid ({worst:.3e})"
    print(f"[GATE C] dense band-power condition == frozen densegrid stream: n={n}, "
          f"max |delta| {worst:.3e} OK", flush=True)
    return dict(block="gateC_dense_lda_identity", variant="-", session="ALL7", n_win=n,
                auc_win="", auc_trial="", max_abs_delta=worst)


def write(path, rows, cols=None):
    if cols is None:
        cols, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows", flush=True)


def main():
    os.makedirs(RES, exist_ok=True)
    print("== building demobase band-power rows ==", flush=True)
    rows_demo, _ = build(ROOT, base_min_s=BASE_MIN_S)
    print("== building dumpbase band-power rows (gate B only) ==", flush=True)
    rows_dump, _ = build(ROOT)

    auc = gate_a(rows_demo)

    print("== covariance inventories ==", flush=True)
    keys_dump, covs_dump, _cal_dump = cov_inventory(ROOT, None)
    auc += gate_b(rows_dump, keys_dump, covs_dump)
    del covs_dump

    keys_demo, covs_demo, calib_demo = cov_inventory(ROOT, BASE_MIN_S)
    a2, dense_rows, thr_rows, calib_rows = deployable(rows_demo, keys_demo, covs_demo,
                                                      calib_demo, LAM_RA, dense=True)
    auc += a2
    auc.append(gate_c(dense_rows))

    print("== RA-shrinkage sensitivity (declared grid) ==", flush=True)
    for lam in LAM_SENS:
        if lam == LAM_RA:
            continue
        auc += deployable(rows_demo, keys_demo, covs_demo, calib_demo, lam, dense=False)[0]

    write(os.path.join(RES, "gate_fused_auc.csv"), auc)
    write(os.path.join(RES, "gate_fused_scores.csv"), dense_rows)
    write(os.path.join(RES, "gate_fused_thresholds.csv"), thr_rows)
    write(os.path.join(RES, "gate_fused_calib.csv"), calib_rows)


if __name__ == "__main__":
    main()
