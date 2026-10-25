"""Figure 7 — physics simulation and walk-intent gate

gate_window3_within_session — gate scores for 3 s and 4 s decision windows, leave-one-session-out and within-session 10-fold.

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import csv
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gate_features as pgf
import gate_fusion as PF
from gate_dense_hop import run_consec
from gate_transition import run_clean
import gate_dense_hop as PDH
import gate as GATE

RES = os.path.join(HERE, "result")
ROOT = PF.ROOT
SESSIONS = list(pgf.SESSIONS)
P1 = [s for s in SESSIONS if s.startswith("pat01")]
P2 = [s for s in SESSIONS if s.startswith("pat02")]
LAM = PF.LAM_RA
BASE_MIN_S = PF.BASE_MIN_S
HOP_S = 1.0
NFOLD = 10
SEED = 0
B_BOOT = 1000

TS = (4.0, 3.0)
REGIMES = ("loso", "within_random", "within_block")
CLFS = ("fuse2_V1", "bp_lda")


def auc_np(y, x):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_weighted(y, x, w):
    """Event-weighted (dense 'moment') AUC: every event contributes weight 1 in total."""
    y = np.asarray(y)
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    o = np.argsort(x, kind="mergesort")
    y, x, w = y[o], x[o], w[o]
    pos = y == 1
    W1, W0 = w[pos].sum(), w[~pos].sum()
    if W1 <= 0 or W0 <= 0:
        return np.nan
    cneg = np.cumsum(np.where(pos, 0.0, w))
    below = np.concatenate([[0.0], cneg[:-1]])
    num = 0.0
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[j + 1] == x[i]:
            j += 1
        blk = slice(i, j + 1)
        neg_in = w[blk][~pos[blk]].sum()
        for k in range(i, j + 1):
            if pos[k]:
                num += w[k] * (below[i] + 0.5 * neg_in)
        i = j + 1
    return float(num / (W1 * W0))


def dense_windows(X, a, b, fs, win_s, hop_s=HOP_S):
    n, hop = int(round(win_s * fs)), int(round(hop_s * fs))
    out, s, k = [], a, 0
    while s + n <= min(b, len(X)):
        out.append((k * hop_s + win_s, X[s:s + n]))
        s += hop
        k += 1
    return out


def rest_stream_T(rest_events, win_s, hop_s=HOP_S):
    """gate_dense_hop.rest_stream, with the window length parameterised."""
    t, s, x0 = [], [], 0.0
    for _name, rs in rest_events:
        for j, r in enumerate(rs):
            t.append(x0 + win_s + j * hop_s)
            s.append(r["s"])
        x0 += (len(rs) - 1) * hop_s + win_s
    return t, s, x0


def ordered(ev):
    return [(k, ev[k]) for k in sorted(ev, key=lambda e: int(e.lstrip("irestw")))]


def label_pure_T(dec, win_s):
    iw_end, ra, rb = dec[0]["iw_end"], dec[0]["rest_a"], dec[0]["rest_b"]
    iw = [d for d in dec if d["t"] - win_s >= -1e-9 and d["t"] <= iw_end + 1e-9]
    rest = [d for d in dec if d["t"] - win_s >= ra - 1e-9 and d["t"] <= rb + 1e-9]
    return iw, rest


def prepare(T):
    """Everything that depends only on T (never on the fold): features, calibration, dense."""
    t0 = time.time()
    pgf.WIN_S = T
    PF.WIN_S = T
    print(f"\n===== prepare T = {T} s =====", flush=True)
    rows, _meta = pgf.build(ROOT, base_min_s=BASE_MIN_S)
    keys, covs, calib = PF.cov_inventory(ROOT, BASE_MIN_S)
    assert [(r["session"], r["event"], r["win"]) for r in rows] == keys, "inventory mismatch"

    whit = {s: PF.whiteners(calib[s], LAM) for s in SESSIONS}
    y = np.array([r["label"] for r in rows])
    Xbp = np.vstack([r["corrected"] for r in rows])
    Xra = np.vstack([PF.ra_feat(c, whit[k[0]], LAM) for c, k in zip(covs, keys)])
    sess = np.array([r["session"] for r in rows])
    evn = np.array([r["event"] for r in rows])

    Bcal, Xcal, Pbase, onset, evdur = {}, {}, {}, {}, {}
    dense = {}
    for s in SESSIONS:
        X, fs, cal, ordered_eval = PF.session_layout(ROOT, s, BASE_MIN_S)
        bw = np.vstack([pgf.bandpower(w, fs)
                        for e in cal for w in pgf.windows(X, e["a"], e["b"], fs)])
        Pbase[s] = np.median(bw, axis=0)
        Bcal[s] = np.log10((bw + 1e-20) / (Pbase[s] + 1e-20))
        Xcal[s] = np.vstack([PF.ra_feat(c, whit[s], LAM) for c in calib[s]])
        meta, fb, fr = [], [], []
        for lab, k, e in ordered_eval:
            onset[(s, f"{lab}{k}")] = e["a"]
            evdur[(s, f"{lab}{k}")] = (e["b"] - e["a"]) / fs
            dw = dense_windows(X, e["a"], e["b"], fs, T)
            if not dw:
                continue
            for t_end, w in dw:
                meta.append(dict(session=s, event=f"{lab}{k}", label=1 if lab == "iw" else 0,
                                 t_end_ev_s=round(t_end, 4),
                                 ev_dur_s=round((e["b"] - e["a"]) / fs, 4), n_dec_ev=len(dw)))
                fb.append(np.log10((pgf.bandpower(w, fs) + 1e-20) / (Pbase[s] + 1e-20)))
                fr.append(PF.ra_feat(PF.band_covs(w), whit[s], LAM))
        dense[s] = dict(meta=meta, bp=np.vstack(fb), ra=np.vstack(fr))
        print(f"  [dense T={T}] {s}: {len(meta)} decisions, "
              f"{len(calib[s])} calib windows", flush=True)
    print(f"  prepare T={T} done in {time.time() - t0:.0f} s "
          f"({len(rows)} tiled windows)", flush=True)
    return dict(T=T, y=y, Xbp=Xbp, Xra=Xra, sess=sess, evn=evn, Bcal=Bcal, Xcal=Xcal,
                Pbase=Pbase, whit=whit, onset=onset, evdur=evdur, dense=dense, calib=calib)


def fit_fold(D, tr_idx, groups, test_sess):
    """Fit both conditions on tr_idx and return everything needed to score anything of test_sess."""
    sa = StandardScaler().fit(D["Xbp"][tr_idx])
    cla = LDA(solver="lsqr", shrinkage="auto").fit(sa.transform(D["Xbp"][tr_idx]),
                                                   D["y"][tr_idx])
    sb = StandardScaler().fit(D["Xra"][tr_idx])
    clb, Cbest = PF.fit_ts(sb.transform(D["Xra"][tr_idx]), D["y"][tr_idx], groups)
    cA = cla.decision_function(sa.transform(D["Bcal"][test_sess]))
    cB = clb.decision_function(sb.transform(D["Xcal"][test_sess]))
    sdA = float(cla.decision_function(sa.transform(D["Xbp"][tr_idx])).std())
    sdB = float(clb.decision_function(sb.transform(D["Xra"][tr_idx])).std())
    mA, mB = float(np.median(cA)), float(np.median(cB))
    return dict(sa=sa, cla=cla, sb=sb, clb=clb, C=Cbest, cA=cA, cB=cB,
                sdA=sdA, sdB=sdB, mA=mA, mB=mB)


def score(M, Fbp, Fra):
    dA = M["cla"].decision_function(M["sa"].transform(Fbp))
    dB = M["clb"].decision_function(M["sb"].transform(Fra))
    fu = ((dA - M["mA"]) / M["sdA"] + (dB - M["mB"]) / M["sdB"]) / 2.0
    return dA, dB, fu


def event_table(D, s):
    """The session's evaluated events in canonical row order, with label and onset."""
    m = D["sess"] == s
    names, seen = [], set()
    for e in D["evn"][m]:
        if e not in seen:
            seen.add(e)
            names.append(e)
    lab = np.array([1 if n.startswith("iw") else 0 for n in names])
    ons = np.array([D["onset"][(s, n)] for n in names], float)
    return names, lab, ons


def folds_for(D, s, regime):
    names, lab, ons = event_table(D, s)
    n = len(names)
    if regime == "within_random":
        skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
        return names, [(tr, te) for tr, te in skf.split(np.zeros(n), lab)]
    if regime == "within_block":
        order = np.argsort(ons, kind="mergesort")
        blocks = np.array_split(order, NFOLD)
        out = []
        for b in blocks:
            te = np.asarray(b, int)
            tr = np.array([i for i in range(n) if i not in set(te.tolist())], int)
            out.append((tr, te))
        return names, out
    raise ValueError(regime)


def run_cell(D, regime, fold_rows):
    """Return per-session out-of-fold scores for both classifiers, tiled and dense."""
    T = D["T"]
    out = {}
    for s in SESSIONS:
        te_m = D["sess"] == s
        idx_te = np.where(te_m)[0]
        dn = D["dense"][s]
        dn_ev = np.array([r["event"] for r in dn["meta"]])
        sc_tiled = {c: np.full(idx_te.size, np.nan) for c in CLFS}
        sc_dense = {c: np.full(len(dn["meta"]), np.nan) for c in CLFS}

        if regime == "loso":
            tr_idx = np.where(~te_m)[0]
            M = fit_fold(D, tr_idx, D["sess"][tr_idx], s)
            dA, _dB, fu = score(M, D["Xbp"][idx_te], D["Xra"][idx_te])
            sc_tiled["bp_lda"][:] = dA
            sc_tiled["fuse2_V1"][:] = fu
            gA, _gB, gf = score(M, dn["bp"], dn["ra"])
            sc_dense["bp_lda"][:] = gA
            sc_dense["fuse2_V1"][:] = gf
            tr_ev = {f"{a}|{b}" for a, b in zip(D["sess"][tr_idx], D["evn"][tr_idx])}
            fold_rows.append(dict(T=T, regime=regime, session=s, fold=0,
                                  n_train_events=len(tr_ev),
                                  n_train_events_iw=sum(1 for e in tr_ev if "|iw" in e),
                                  n_train_events_rest=sum(1 for e in tr_ev if "|rest" in e),
                                  n_train_win=int(tr_idx.size),
                                  n_test_events=int(np.unique(D["evn"][idx_te]).size),
                                  n_test_win=int(idx_te.size), lr_C=M["C"]))
            out[s] = dict(model=M, tiled=sc_tiled, dense=sc_dense, meta=dn["meta"])
            print(f"  [{regime} T={T}] {s}: C={M['C']}", flush=True)
            continue

        names, fl = folds_for(D, s, regime)
        ev_of_row = D["evn"][idx_te]
        for k, (tr_e, te_e) in enumerate(fl):
            tr_names = {names[i] for i in tr_e}
            te_names = {names[i] for i in te_e}
            tr_idx = idx_te[np.isin(ev_of_row, list(tr_names))]
            row_te = np.isin(ev_of_row, list(te_names))
            M = fit_fold(D, tr_idx, D["evn"][tr_idx], s)
            dA, _dB, fu = score(M, D["Xbp"][idx_te][row_te], D["Xra"][idx_te][row_te])
            sc_tiled["bp_lda"][row_te] = dA
            sc_tiled["fuse2_V1"][row_te] = fu
            dm = np.isin(dn_ev, list(te_names))
            if dm.any():
                gA, _gB, gf = score(M, dn["bp"][dm], dn["ra"][dm])
                sc_dense["bp_lda"][dm] = gA
                sc_dense["fuse2_V1"][dm] = gf
            fold_rows.append(dict(T=T, regime=regime, session=s, fold=k,
                                  n_train_events=len(tr_names),
                                  n_train_events_iw=sum(1 for n in tr_names
                                                        if n.startswith("iw")),
                                  n_train_events_rest=sum(1 for n in tr_names
                                                          if n.startswith("rest")),
                                  n_train_win=int(tr_idx.size),
                                  n_test_events=len(te_names),
                                  n_test_win=int(row_te.sum()), lr_C=M["C"]))
        assert not np.isnan(sc_tiled["fuse2_V1"]).any(), f"{s}: unscored tiled windows"
        out[s] = dict(model=None, tiled=sc_tiled, dense=sc_dense, meta=dn["meta"])
        ntr = [r["n_train_events"] for r in fold_rows
               if r["T"] == T and r["regime"] == regime and r["session"] == s]
        print(f"  [{regime} T={T}] {s}: {len(fl)} folds, "
              f"train events {min(ntr)}-{max(ntr)} (mean {np.mean(ntr):.1f})", flush=True)
    return out


def cell_stats(D, cellout, clf, rng):
    """Per-session tiled / dense / trial AUC, mean and pooled, and the event bootstrap CI."""
    per, rows = [], []
    boot_mat = np.zeros((B_BOOT, len(SESSIONS)))
    pooled_y, pooled_x = [], []
    for si, s in enumerate(SESSIONS):
        m = D["sess"] == s
        y = D["y"][m]
        x = cellout[s]["tiled"][clf]
        ev = D["evn"][m]
        a_t = auc_np(y, x)
        ue = np.unique(ev)
        a_tr = auc_np(np.array([y[ev == e][0] for e in ue]),
                      np.array([x[ev == e].mean() for e in ue]))
        meta = cellout[s]["meta"]
        yd = np.array([r["label"] for r in meta])
        xd = cellout[s]["dense"][clf]
        wd = np.array([1.0 / r["n_dec_ev"] for r in meta])
        a_d = auc_weighted(yd, xd, wd)
        per.append(a_t)
        pooled_y.append(y)
        pooled_x.append(x)
        rows.append(dict(session=s, n_win=int(m.sum()), n_ev=len(ue),
                         n_dec_dense=len(meta), auc_tiled=a_t, auc_dense=a_d, auc_trial=a_tr))
        idx_iw = [np.where(ev == e)[0] for e in ue if e.startswith("iw")]
        idx_rs = [np.where(ev == e)[0] for e in ue if e.startswith("rest")]
        for b in range(B_BOOT):
            pick = ([idx_iw[i] for i in rng[b][si][0]] + [idx_rs[i] for i in rng[b][si][1]])
            ii = np.concatenate(pick)
            boot_mat[b, si] = auc_np(y[ii], x[ii])
    mean7 = float(np.mean(per))
    bm = np.nanmean(boot_mat, axis=1)
    lo, hi = np.nanpercentile(bm, [2.5, 97.5])
    pooled = auc_np(np.concatenate(pooled_y), np.concatenate(pooled_x))
    ci_per = np.nanpercentile(boot_mat, [2.5, 97.5], axis=0)
    for i, r in enumerate(rows):
        r["ci_lo"] = float(ci_per[0, i])
        r["ci_hi"] = float(ci_per[1, i])
    return rows, mean7, float(lo), float(hi), pooled, bm


def boot_index(D):
    """Shared bootstrap resampling indices (seed 0), identical across every cell."""
    rs = np.random.RandomState(SEED)
    counts = []
    for s in SESSIONS:
        m = D["sess"] == s
        ue = np.unique(D["evn"][m])
        counts.append((sum(1 for e in ue if e.startswith("iw")),
                       sum(1 for e in ue if e.startswith("rest"))))
    return [[(rs.randint(0, n1, n1), rs.randint(0, n0, n0)) for (n1, n0) in counts]
            for _ in range(B_BOOT)]


def symdg_thresholds(D, models):
    thr = {}
    for s in SESSIONS:
        M = models[s]
        t = ((0.0 - M["mA"]) / M["sdA"] + (0.0 - M["mB"]) / M["sdB"]) / 2.0
        fc = ((M["cA"] - M["mA"]) / M["sdA"] + (M["cB"] - M["mB"]) / M["sdB"]) / 2.0
        iqr = float(np.percentile(fc, 75) - np.percentile(fc, 25))
        d = GATE.IQR_K * iqr
        thr[s] = dict(theta_sym=t, delta=d, theta_on=t + d, theta_off=t - d,
                      calib_iqr=iqr, n_calib_win=len(fc), sd_A=M["sdA"], sd_B=M["sdB"],
                      centre_A=M["mA"], centre_B=M["mB"])
    return thr


def working_point(D, cellout, thr, tag, rows):
    T = D["T"]
    agg = dict(det=0, tot=0, lat=[], nfp=0, true_rest=0.0, nrest=0, first_ok=0, dec_iw=0)
    per = {}
    for s in SESSIONS:
        meta = cellout[s]["meta"]
        sc = cellout[s]["dense"]["fuse2_V1"]
        by = {}
        for r, v in zip(meta, sc):
            by.setdefault(r["label"] == 1, {}).setdefault(r["event"], []).append(
                dict(t=r["t_end_ev_s"], s=float(v), dur=r["ev_dur_s"]))
        for d in by.values():
            for rs in d.values():
                rs.sort(key=lambda r: r["t"])
        on, off = thr[s]["theta_on"], thr[s]["theta_off"]
        iw = ordered(by[True])
        ndet, lat, dec = 0, [], 0
        for _n, rs in iw:
            dec += len(rs)
            op = [e[0] for e in run_consec([r["t"] for r in rs], [r["s"] for r in rs],
                                           on, off, GATE.N_ON) if e[1] == "open"]
            if op:
                ndet += 1
                lat.append(op[0])
        rest = ordered(by[False])
        t, v, _span = rest_stream_T(rest, T)
        nfp = sum(1 for e in run_consec(t, v, on, off, GATE.N_ON) if e[1] == "open")
        trs = sum(rs[0]["dur"] for _n, rs in rest)
        fok = sum(1 for _n, rs in rest if rs[0]["s"] <= on)
        per[s] = dict(ndet=ndet, ntr=len(iw), nfp=nfp, trs=trs, nrest=len(rest), fok=fok)
        agg["det"] += ndet
        agg["tot"] += len(iw)
        agg["lat"] += lat
        agg["nfp"] += nfp
        agg["true_rest"] += trs
        agg["nrest"] += len(rest)
        agg["first_ok"] += fok
        agg["dec_iw"] += dec
        rows.append(dict(row_type="session", tag=tag, T=T, session=s, n_trials=len(iw),
                         n_detected=ndet, tpr=ndet / len(iw), n_fp=nfp,
                         true_rest_s=round(trs, 3),
                         fp_per_min_true=nfp / (trs / 60.0), n_rest_events=len(rest),
                         first_correct=fok, latency_median_s=(float(np.median(lat))
                                                              if lat else np.nan),
                         theta_on=on, theta_off=off))
    lat = agg["lat"]
    A = dict(row_type="overall", tag=tag, T=T, session="ALL7", n_trials=agg["tot"],
             n_detected=agg["det"], tpr=agg["det"] / agg["tot"], n_fp=agg["nfp"],
             true_rest_s=round(agg["true_rest"], 3),
             fp_per_min_true=agg["nfp"] / (agg["true_rest"] / 60.0),
             n_rest_events=agg["nrest"], first_correct=agg["first_ok"],
             latency_median_s=float(np.median(lat)), latency_mean_s=float(np.mean(lat)),
             latency_min_s=float(np.min(lat)), latency_max_s=float(np.max(lat)),
             dec_per_trial=agg["dec_iw"] / agg["tot"])
    rows.append(A)
    for nm, ss in (("P1", P1), ("P2", P2)):
        d = sum(per[s]["ndet"] for s in ss)
        n = sum(per[s]["ntr"] for s in ss)
        f = sum(per[s]["nfp"] for s in ss)
        tr = sum(per[s]["trs"] for s in ss)
        rows.append(dict(row_type="overall", tag=tag, T=T, session=nm, n_trials=n,
                         n_detected=d, tpr=d / n, n_fp=f, true_rest_s=round(tr, 3),
                         fp_per_min_true=f / (tr / 60.0)))
    hist = {v: lat.count(v) for v in sorted(set(lat))}
    for v, c in hist.items():
        rows.append(dict(row_type="latency_hist", tag=tag, T=T, latency_s=v, n_trials=c))
    print(f"[WP {tag} T={T}] TPR {A['tpr']:.4f} ({A['n_detected']}/{A['n_trials']})  "
          f"FP {A['fp_per_min_true']:.3f}/min ({A['n_fp']} ev, "
          f"{A['true_rest_s'] / 60:.2f} min true rest)  lat med "
          f"{A['latency_median_s']:.1f}s mean {A['latency_mean_s']:.3f}s  "
          f"rest first-dec correct {A['first_correct']}/{A['n_rest_events']}", flush=True)
    return A


def transition_span(D, models, T):
    """The IW -> rest continuous span at window T, deployable fuse2 V1 (LOSO models)."""
    out = []
    for s in SESSIONS:
        M = models[s]
        X, fs, cal, _oe = PF.session_layout(ROOT, s, BASE_MIN_S)
        _X2, fs2, _ch, ev = pgf.load_session(ROOT, s)
        assert abs(fs - fs2) < 1e-9
        calib_a = {e["a"] for e in cal}
        n_win, hop = int(round(T * fs)), int(round(HOP_S * fs))
        iw_evs = [e for e in ev if e["code"] == "gaitimagery"]
        pid = 0
        for k_iw, e in enumerate(iw_evs):
            nxt = [x for x in ev if x["a"] > e["a"]]
            if not nxt or nxt[0]["code"] != "rest":
                continue
            r = nxt[0]
            pid += 1
            s0 = e["a"] - int(round(4.0 * fs))
            if s0 < 0:
                continue
            meta, fb, fr = [], [], []
            ss, m = s0, 0
            while ss + n_win <= min(r["b"], len(X)):
                t_abs = ss + n_win
                t_rel = (t_abs - e["a"]) / fs
                zone = ("pre" if t_abs <= e["a"] else "iw" if t_abs <= e["b"] else
                        "gap" if t_abs <= r["a"] else "rest")
                w = X[ss:ss + n_win]
                fb.append(np.log10((pgf.bandpower(w, fs) + 1e-20) / (D["Pbase"][s] + 1e-20)))
                fr.append(PF.ra_feat(PF.band_covs(w), D["whit"][s], LAM))
                meta.append((m, round(t_rel, 4), zone))
                ss += hop
                m += 1
            if not meta:
                continue
            gA, _gB, gf = score(M, np.vstack(fb), np.vstack(fr))
            for (m_i, t_rel, zone), va, vf in zip(meta, gA, gf):
                out.append(dict(pair_id=f"{s}|p{pid:03d}", session=s, iw_event=f"iw{k_iw}",
                                rest_is_calibration=int(r["a"] in calib_a), m=m_i,
                                t_rel_iw_s=t_rel, zone=zone, score_lda=float(va),
                                score_fuse=float(vf),
                                gap_s=round((r["a"] - e["b"]) / fs, 4),
                                iw_end_rel_s=round((e["b"] - e["a"]) / fs, 4),
                                rest_start_rel_s=round((r["a"] - e["a"]) / fs, 4),
                                rest_end_rel_s=round((r["b"] - e["a"]) / fs, 4)))
        print(f"  [trans T={T}] {s}: {pid} pairs, {len(out)} decisions cumulative", flush=True)
    return out


def close_fraction(span, thr, T):
    pairs = {}
    for r in span:
        pairs.setdefault(r["pair_id"], []).append(
            dict(t=float(r["t_rel_iw_s"]), e=float(r["score_fuse"]),
                 session=r["session"], iw_end=float(r["iw_end_rel_s"]),
                 rest_a=float(r["rest_start_rel_s"]), rest_b=float(r["rest_end_rel_s"])))
    for v in pairs.values():
        v.sort(key=lambda d: d["t"])
    nopen = nclose = 0
    for _pid, dec in sorted(pairs.items()):
        s = dec[0]["session"]
        on, off = thr[s]["theta_on"], thr[s]["theta_off"]
        iw, rest = label_pure_T(dec, T)
        evs = run_clean(iw, rest, "e", on, off)
        if any(x[1] == "open" and x[2] == "iw" for x in evs):
            nopen += 1
            if any(x[1] == "close" and x[2] == "rest" for x in evs):
                nclose += 1
    return len(pairs), nopen, nclose


def write(path, rows, head=None):
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    if head:
        cols = head + [c for c in cols if c not in head]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows", flush=True)


def main():
    t00 = time.time()
    print(f"[config] gate.py N_ON={GATE.N_ON} DWELL={GATE.DWELL_S} IQR_K={GATE.IQR_K} "
          f"N_OFF={PDH.N_OFF} HOP={HOP_S}", flush=True)
    D = {T: prepare(T) for T in TS}

    auc_rows, fold_rows, wp_rows, cf_rows = [], [], [], []
    cells, models = {}, {}
    for T in TS:
        for reg in REGIMES:
            cells[(T, reg)] = run_cell(D[T], reg, fold_rows)
        models[T] = {s: cells[(T, "loso")][s]["model"] for s in SESSIONS}

    boots = {T: boot_index(D[T]) for T in TS}
    draws = {}
    for T in TS:
        for reg in REGIMES:
            for clf in CLFS:
                rows, m7, lo, hi, pooled, bm = cell_stats(D[T], cells[(T, reg)], clf, boots[T])
                draws[(T, reg, clf)] = bm
                for r in rows:
                    auc_rows.append(dict(T=T, classifier=clf, regime=reg, **r))
                auc_rows.append(dict(T=T, classifier=clf, regime=reg, session="MEAN7",
                                     auc_tiled=m7, ci_lo=lo, ci_hi=hi))
                auc_rows.append(dict(T=T, classifier=clf, regime=reg, session="POOLED",
                                     auc_tiled=pooled))
                print(f"[cell] T={T} {clf:9s} {reg:13s} mean7 {m7:.4f} "
                      f"[{lo:.4f}, {hi:.4f}] pooled {pooled:.4f}", flush=True)

    def m7(T, reg, clf):
        return [r for r in auc_rows if r["T"] == T and r["regime"] == reg
                and r["classifier"] == clf and r["session"] == "MEAN7"][0]

    cmp_rows = []
    for clf in CLFS:
        c1 = m7(3.0, "loso", clf)["auc_tiled"] - m7(4.0, "loso", clf)["auc_tiled"]
        d1 = draws[(3.0, "loso", clf)] - draws[(4.0, "loso", clf)]
        cmp_rows.append(dict(comparison="C1_window_cost", classifier=clf, T="3-4",
                             regime="loso", delta=c1,
                             ci_lo=float(np.percentile(d1, 2.5)),
                             ci_hi=float(np.percentile(d1, 97.5))))
        for T in TS:
            for reg in ("within_random", "within_block"):
                d = m7(T, reg, clf)["auc_tiled"] - m7(T, "loso", clf)["auc_tiled"]
                dd = draws[(T, reg, clf)] - draws[(T, "loso", clf)]
                cmp_rows.append(dict(comparison=f"C2_{reg}_minus_loso", classifier=clf, T=T,
                                     regime=reg, delta=d,
                                     ci_lo=float(np.percentile(dd, 2.5)),
                                     ci_hi=float(np.percentile(dd, 97.5))))
            d = m7(T, "within_random", clf)["auc_tiled"] - m7(T, "within_block", clf)["auc_tiled"]
            bl = m7(T, "within_block", clf)
            half = (bl["ci_hi"] - bl["ci_lo"]) / 2.0
            cmp_rows.append(dict(comparison="C2_random_minus_block", classifier=clf, T=T,
                                 regime="within", delta=d, block_ci_half_width=half,
                                 optimistic=int(d > half)))
    for r in cmp_rows:
        print(f"[cmp] {r['comparison']:26s} {r['classifier']:9s} T={r['T']} "
              f"delta {r['delta']:+.4f}", flush=True)

    thr_rows = []
    thr = {}
    for T in TS:
        thr[T] = symdg_thresholds(D[T], models[T])
        for s in SESSIONS:
            thr_rows.append(dict(T=T, session=s, **thr[T][s]))
        working_point(D[T], cells[(T, "loso")], thr[T], "symdg", wp_rows)

    for T in TS:
        span = transition_span(D[T], models[T], T)
        npair, nopen, nclose = close_fraction(span, thr[T], T)
        cf_rows.append(dict(T=T, source="recomputed", n_pairs=npair, n_open_pairs=nopen,
                            n_close_pairs=nclose,
                            close_frac=(nclose / nopen if nopen else np.nan)))
        print(f"[close T={T}] {nclose}/{nopen} of {npair} pairs = "
              f"{nclose / nopen if nopen else float('nan'):.4f}", flush=True)
        if T == 3.0:
            write(os.path.join(RES, "gate_win3_transition_scores_T3.csv"), span)

    d3 = []
    for s in SESSIONS:
        for r, va, vf in zip(cells[(3.0, "loso")][s]["meta"],
                             cells[(3.0, "loso")][s]["dense"]["bp_lda"],
                             cells[(3.0, "loso")][s]["dense"]["fuse2_V1"]):
            d3.append(dict(session=s, event=r["event"], label=r["label"],
                           t_end_ev_s=r["t_end_ev_s"], ev_dur_s=r["ev_dur_s"],
                           n_dec_ev=r["n_dec_ev"], score_lda=float(va), fuse_V1=float(vf)))
    write(os.path.join(RES, "gate_win3_scores_T3.csv"), d3)
    write(os.path.join(RES, "gate_win3_withinsession.csv"), auc_rows,
          head=["T", "classifier", "regime", "session"])
    write(os.path.join(RES, "gate_win3_comparisons.csv"), cmp_rows)
    write(os.path.join(RES, "gate_win3_folds.csv"), fold_rows)
    write(os.path.join(RES, "gate_win3_thresholds.csv"), thr_rows)
    write(os.path.join(RES, "gate_win3_workingpoint.csv"), wp_rows,
          head=["row_type", "tag", "T", "session"])
    write(os.path.join(RES, "gate_win3_closefrac.csv"), cf_rows)
    print(f"\nALL DONE in {(time.time() - t00) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
