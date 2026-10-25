"""Figure 7 — physics simulation and walk-intent gate

gate_window3_fold_reference — within-session 10-fold gate scores whose rest reference is drawn inside each fold from that fold's training rest events.

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gate_window3_within_session as M
import gate_features as pgf
import gate_fusion as PF
import gate as GATE
from gate_transition import run_clean

RES, ROOT, SESSIONS = M.RES, M.ROOT, M.SESSIONS
LAM, BASE_MIN_S, HOP_S = M.LAM, M.BASE_MIN_S, M.HOP_S
SEED, B_BOOT, REF_MIN_S = 0, 1000, 30.0
REF = os.path.join(RES, "gate_win3_withinsession.csv")


def ref_auc_A():
    out = {}
    for r in csv.DictReader(open(REF, encoding="utf-8")):
        if r["regime"] == "within_random" and r["session"] in SESSIONS:
            out[(float(r["T"]), r["classifier"], r["session"])] = float(r["auc_tiled"])
    return out


def prepare_raw(T):
    """Everything that does NOT depend on the fold: raw band power and band covariances."""
    t0 = time.time()
    pgf.WIN_S = T
    PF.WIN_S = T
    out = {}
    for s in SESSIONS:
        X, fs, cal, ordered_eval = PF.session_layout(ROOT, s, BASE_MIN_S)
        calP, calC = [], []
        for e in cal:
            for w in pgf.windows(X, e["a"], e["b"], fs):
                calP.append(pgf.bandpower(w, fs))
                calC.append(PF.band_covs(w))
        tiled, dense, dur = {}, {}, {}
        for lab, k, e in ordered_eval:
            name = f"{lab}{k}"
            dur[name] = (e["b"] - e["a"]) / fs
            ws = pgf.windows(X, e["a"], e["b"], fs)
            if ws:
                tiled[name] = (np.vstack([pgf.bandpower(w, fs) for w in ws]),
                               [PF.band_covs(w) for w in ws])
            dw = M.dense_windows(X, e["a"], e["b"], fs, T)
            if dw:
                dense[name] = ([t for t, _w in dw],
                               np.vstack([pgf.bandpower(w, fs) for _t, w in dw]),
                               [PF.band_covs(w) for _t, w in dw])
        out[s] = dict(calP=np.vstack(calP), calC=calC, tiled=tiled, dense=dense, dur=dur)
        print(f"  [raw T={T}] {s}: {len(tiled)} tiled events, "
              f"{sum(len(v[0]) for v in dense.values())} dense decisions, "
              f"{len(calC)} head-calib windows", flush=True)
    print(f"  prepare_raw T={T} done in {time.time() - t0:.0f} s", flush=True)
    return out


N_DAMPED = 0


def riemann_mean_damped(Cs, tau=0.5, tol=1e-10, itmax=3000):
    R = np.mean(Cs, axis=0)
    for _ in range(itmax):
        Rm, Rp = PF.powm_sym(R, -0.5), PF.powm_sym(R, 0.5)
        S = np.mean([PF.logm_sym(Rm @ C @ Rm) for C in Cs], axis=0)
        n = float(np.linalg.norm(S, "fro"))
        R = Rp @ PF._expm_sym(tau * S) @ Rp
        R = (R + R.T) / 2.0
        if n < tol:
            return R
    raise RuntimeError(f"damped Karcher mean did not converge (|S|_F {n:.3e})")


def whiteners_safe(covs, lam=LAM):
    """PF.whiteners, with the damped solver as a counted fallback."""
    global N_DAMPED
    out = []
    for b in range(len(pgf.BANDS)):
        Cs = [PF.shrink(c[b], lam) for c in covs]
        try:
            Rb = PF.riemann_mean(Cs)
        except np.linalg.LinAlgError:
            Rb = riemann_mean_damped(Cs)
            N_DAMPED += 1
        out.append(PF.powm_sym(Rb, -0.5))
    return out


def karcher_residual(R, Cs):
    """|mean_i log(R^-1/2 C_i R^-1/2)|_F -- zero exactly at the AIRM (Karcher) mean."""
    Rm = PF.powm_sym(R, -0.5)
    S = np.mean([PF.logm_sym(Rm @ C @ Rm) for C in Cs], axis=0)
    return float(np.linalg.norm(S, "fro"))


def self_check_damped(RAW):
    """The two Karcher-mean solvers compared on every head-calibration band solve."""
    worst_conv, n_conv, n_tot, worst_all = 0.0, 0, 0, 0.0
    r1s, r2s = [], []
    for s in SESSIONS:
        for b in range(len(pgf.BANDS)):
            Cs = [PF.shrink(c[b], LAM) for c in RAW[s]["calC"]]
            R1 = PF.riemann_mean(Cs)
            R2 = riemann_mean_damped(Cs)
            r1, r2 = karcher_residual(R1, Cs), karcher_residual(R2, Cs)
            rel = float(np.max(np.abs(R1 - R2)) / max(np.max(np.abs(R1)), 1e-30))
            r1s.append(r1)
            r2s.append(r2)
            worst_all = max(worst_all, rel)
            n_tot += 1
            if r1 < 1e-8:
                n_conv += 1
                worst_conv = max(worst_conv, rel)
    print(f"[SELF-CHECK] {n_tot} head-calibration band solves: frozen residual "
          f"{min(r1s):.2e}..{max(r1s):.2e} (converged, <1e-8: {n_conv}); damped residual "
          f"{min(r2s):.2e}..{max(r2s):.2e}; max rel |d| overall {worst_all:.2e}, "
          f"on the converged subset {worst_conv:.2e}", flush=True)
    assert n_conv > 0, ""
    assert worst_conv <= 1e-6, \
        f"damped solver disagrees with a CONVERGED frozen solve by {worst_conv:.3e}"


def ra_stack(covs_list, whit):
    return np.vstack([PF.ra_feat(c, whit, LAM) for c in covs_list])


def rel(P, Pbase):
    return np.log10((P + 1e-20) / (Pbase + 1e-20))


def draw_reference(rest_names, dur, onset, scheme, rs):
    """Rest events from the training fold until cumulative duration >= REF_MIN_S."""
    if scheme == "random":
        order = list(np.asarray(rest_names)[rs.permutation(len(rest_names))])
    elif scheme == "earliest":
        order = sorted(rest_names, key=lambda n: onset[n])
    else:
        raise ValueError(scheme)
    cum, pick = 0.0, []
    for n in order:
        pick.append(n)
        cum += dur[n]
        if cum >= REF_MIN_S:
            break
    return pick, cum


def run_scheme(D, RAW, T, scheme, want_dense=True):
    """scheme: 'random' | 'earliest' | 'calib' (the regression gate)."""
    streams, thr_rows, tiled_sc, models = [], [], {}, {}
    for s in SESSIONS:
        R = RAW[s]
        te_m = D["sess"] == s
        idx_te = np.where(te_m)[0]
        ev_row = D["evn"][idx_te]
        names, folds = M.folds_for(D, s, "within_random")
        onset = {n: D["onset"][(s, n)] for n in names}
        rs = np.random.RandomState(SEED)
        sc_t = {c: np.full(idx_te.size, np.nan) for c in ("fuse2_V1", "bp_lda")}
        mods = {}
        for k, (tr_e, te_e) in enumerate(folds):
            tr_names = [names[i] for i in tr_e]
            te_names = [names[i] for i in te_e]
            tr_rest = [n for n in tr_names if n.startswith("rest")]
            if scheme == "calib":
                refP, refC, ref_names, ref_dur = R["calP"], R["calC"], [], 0.0
                fit_names = tr_names
            else:
                ref_names, ref_dur = draw_reference(tr_rest, R["dur"], onset, scheme, rs)
                refP = np.vstack([R["tiled"][n][0] for n in ref_names])
                refC = [c for n in ref_names for c in R["tiled"][n][1]]
                fit_names = [n for n in tr_names if n not in set(ref_names)]
            Pbase = np.median(refP, axis=0)
            whit = whiteners_safe(refC, LAM)

            Xb = np.vstack([rel(R["tiled"][n][0], Pbase) for n in fit_names])
            Xr = np.vstack([ra_stack(R["tiled"][n][1], whit) for n in fit_names])
            y = np.concatenate([np.full(len(R["tiled"][n][0]), 1 if n.startswith("iw") else 0)
                                for n in fit_names])
            g = np.concatenate([np.full(len(R["tiled"][n][0]), n) for n in fit_names])
            sa = StandardScaler().fit(Xb)
            cla = LDA(solver="lsqr", shrinkage="auto").fit(sa.transform(Xb), y)
            sb = StandardScaler().fit(Xr)
            clb, Cbest = PF.fit_ts(sb.transform(Xr), y, g)

            cA = cla.decision_function(sa.transform(rel(refP, Pbase)))
            cB = clb.decision_function(sb.transform(ra_stack(refC, whit)))
            sdA = float(cla.decision_function(sa.transform(Xb)).std())
            sdB = float(clb.decision_function(sb.transform(Xr)).std())
            mA, mB = float(np.median(cA)), float(np.median(cB))
            fc = ((cA - mA) / sdA + (cB - mB) / sdB) / 2.0
            th = ((0.0 - mA) / sdA + (0.0 - mB) / sdB) / 2.0
            dl = GATE.IQR_K * float(np.percentile(fc, 75) - np.percentile(fc, 25))
            mod = dict(sa=sa, cla=cla, sb=sb, clb=clb, Pbase=Pbase, whit=whit,
                       mA=mA, mB=mB, sdA=sdA, sdB=sdB, theta_sym=th, delta=dl, fold=k)
            for n in te_names:
                mods[n] = mod

            for n in te_names:
                P, C = R["tiled"][n]
                dA = cla.decision_function(sa.transform(rel(P, Pbase)))
                dB = clb.decision_function(sb.transform(ra_stack(C, whit)))
                fu = ((dA - mA) / sdA + (dB - mB) / sdB) / 2.0
                m_rows = ev_row == n
                sc_t["fuse2_V1"][m_rows] = fu
                sc_t["bp_lda"][m_rows] = dA
            thr_rows.append(dict(scheme=scheme, T=T, session=s, fold=k,
                                 n_ref_events=len(ref_names), ref_dur_s=round(ref_dur, 2),
                                 ref_events=";".join(str(x) for x in ref_names), n_ref_win=len(refP),
                                 n_fit_events=len(fit_names), n_fit_win=int(len(y)),
                                 lr_C=Cbest, sd_A=sdA, sd_B=sdB, centre_A=mA, centre_B=mB,
                                 theta_sym=th, delta=dl, calib_iqr=dl / GATE.IQR_K))
        assert not np.isnan(sc_t["fuse2_V1"]).any(), f"{s}: unscored tiled windows"
        tiled_sc[s] = sc_t
        models[s] = mods
        if want_dense:
            for n, (ts, P, C) in R["dense"].items():
                mod = mods[n]
                dA = mod["cla"].decision_function(mod["sa"].transform(rel(P, mod["Pbase"])))
                dB = mod["clb"].decision_function(mod["sb"].transform(ra_stack(C, mod["whit"])))
                fu = ((dA - mod["mA"]) / mod["sdA"] + (dB - mod["mB"]) / mod["sdB"]) / 2.0
                for t, va, vb, vf in zip(ts, dA, dB, fu):
                    streams.append(dict(session=s, event=n,
                                        label=1 if n.startswith("iw") else 0,
                                        t_end_ev_s=round(t, 4),
                                        ev_dur_s=round(R["dur"][n], 4), n_dec_ev=len(ts),
                                        score_lda=float(va), score_ts=float(vb),
                                        fuse_V1=float(vf), fold=mod["fold"],
                                        theta_sym=mod["theta_sym"], delta=mod["delta"],
                                        e_sigma=float((vf - mod["theta_sym"]) / mod["delta"])))
        a1 = M.auc_np(D["y"][te_m], sc_t["fuse2_V1"])
        a0 = M.auc_np(D["y"][te_m], sc_t["bp_lda"])
        print(f"  [{scheme} T={T}] {s}: fuse2 {a1:.6f}  lda {a0:.6f}", flush=True)
    return streams, tiled_sc, thr_rows, models


def cell(D, tiled_sc, clf, boots):
    per, rows, bm = [], [], np.zeros((B_BOOT, len(SESSIONS)))
    for si, s in enumerate(SESSIONS):
        m = D["sess"] == s
        y, x, ev = D["y"][m], tiled_sc[s][clf], D["evn"][m]
        a = M.auc_np(y, x)
        per.append(a)
        ue = np.unique(ev)
        at = M.auc_np(np.array([y[ev == e][0] for e in ue]),
                      np.array([x[ev == e].mean() for e in ue]))
        rows.append(dict(session=s, n_win=int(m.sum()), auc_tiled=a, auc_trial=at))
        idx_iw = [np.where(ev == e)[0] for e in ue if e.startswith("iw")]
        idx_rs = [np.where(ev == e)[0] for e in ue if e.startswith("rest")]
        for b in range(B_BOOT):
            ii = np.concatenate([idx_iw[i] for i in boots[b][si][0]]
                                + [idx_rs[i] for i in boots[b][si][1]])
            bm[b, si] = M.auc_np(y[ii], x[ii])
    mean7 = float(np.mean(per))
    lo, hi = np.nanpercentile(np.nanmean(bm, axis=1), [2.5, 97.5])
    ci = np.nanpercentile(bm, [2.5, 97.5], axis=0)
    for i, r in enumerate(rows):
        r["ci_lo"], r["ci_hi"] = float(ci[0, i]), float(ci[1, i])
    return rows, mean7, float(lo), float(hi)


def transition_span(D, RAW_unused, models, T, RAWs):
    out = []
    for s in SESSIONS:
        mods = models[s]
        X, fs, cal, _oe = PF.session_layout(ROOT, s, BASE_MIN_S)
        _X, _fs, _ch, ev = pgf.load_session(ROOT, s)
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
            name = f"iw{k_iw}"
            if s0 < 0 or name not in mods:
                continue
            mod = mods[name]
            meta, fb, fr = [], [], []
            ss, m = s0, 0
            while ss + n_win <= min(r["b"], len(X)):
                t_abs = ss + n_win
                zone = ("pre" if t_abs <= e["a"] else "iw" if t_abs <= e["b"] else
                        "gap" if t_abs <= r["a"] else "rest")
                w = X[ss:ss + n_win]
                fb.append(rel(pgf.bandpower(w, fs), mod["Pbase"]))
                fr.append(PF.ra_feat(PF.band_covs(w), mod["whit"], LAM))
                meta.append((m, round((t_abs - e["a"]) / fs, 4), zone))
                ss += hop
                m += 1
            if not meta:
                continue
            dA = mod["cla"].decision_function(mod["sa"].transform(np.vstack(fb)))
            dB = mod["clb"].decision_function(mod["sb"].transform(np.vstack(fr)))
            fu = ((dA - mod["mA"]) / mod["sdA"] + (dB - mod["mB"]) / mod["sdB"]) / 2.0
            for (m_i, t_rel, zone), va, vf in zip(meta, dA, fu):
                out.append(dict(pair_id=f"{s}|p{pid:03d}", session=s, iw_event=name, m=m_i,
                                rest_is_calibration=int(r["a"] in calib_a),
                                t_rel_iw_s=t_rel, zone=zone, score_lda=float(va),
                                score_fuse=float(vf),
                                e_sigma=float((vf - mod["theta_sym"]) / mod["delta"]),
                                iw_end_rel_s=round((e["b"] - e["a"]) / fs, 4),
                                rest_start_rel_s=round((r["a"] - e["a"]) / fs, 4),
                                rest_end_rel_s=round((r["b"] - e["a"]) / fs, 4)))
        print(f"  [trans-foldref T={T}] {s}: {pid} pairs, {len(out)} decisions", flush=True)
    return out


def close_fraction(span, T):
    pairs = {}
    for r in span:
        pairs.setdefault(r["pair_id"], []).append(
            dict(t=r["t_rel_iw_s"], e=r["e_sigma"], iw_end=r["iw_end_rel_s"],
                 rest_a=r["rest_start_rel_s"], rest_b=r["rest_end_rel_s"]))
    for v in pairs.values():
        v.sort(key=lambda d: d["t"])
    nopen = nclose = 0
    for _pid, dec in sorted(pairs.items()):
        iw, rest = M.label_pure_T(dec, T)
        evs = run_clean(iw, rest, "e", 1.0, -1.0)
        if any(x[1] == "open" and x[2] == "iw" for x in evs):
            nopen += 1
            if any(x[1] == "close" and x[2] == "rest" for x in evs):
                nclose += 1
    return len(pairs), nopen, nclose


def main():
    t0 = time.time()
    AA = ref_auc_A()
    auc_rows, thr_all = [], []
    keep = {}
    for T in (3.0, 4.0):
        D = M.prepare(T)
        for s in SESSIONS:
            D["dense"][s].pop("bp", None)
            D["dense"][s].pop("ra", None)
        D.pop("Xra", None)
        RAW = prepare_raw(T)
        if T == 3.0:
            self_check_damped(RAW)
        boots = M.boot_index(D)

        _st, tsc, _thr, _mo = run_scheme(D, RAW, T, "calib", want_dense=False)
        worst = 0.0
        for clf in ("fuse2_V1", "bp_lda"):
            for s in SESSIONS:
                a = M.auc_np(D["y"][D["sess"] == s], tsc[s][clf])
                worst = max(worst, abs(a - AA[(T, clf, s)]))
                assert abs(a - AA[(T, clf, s)]) <= 1e-6, \
                    f"REGRESSION T={T} {clf} {s}: {a:.9f} vs {AA[(T, clf, s)]:.9f}"
        print(f"[REGRESSION GATE T={T}] calib-reference fold pipeline == the pre-specified protocol, "
              f"max |d| {worst:.2e}  (damped-Karcher fallbacks so far: {N_DAMPED})", flush=True)

        for scheme in ("random", "earliest"):
            st, tsc, thr, mods = run_scheme(D, RAW, T, scheme,
                                            want_dense=(scheme == "random" or T == 3.0))
            thr_all += thr
            for clf in ("fuse2_V1", "bp_lda"):
                rows, m7, lo, hi = cell(D, tsc, clf, boots)
                for r in rows:
                    auc_rows.append(dict(scheme=scheme, T=T, classifier=clf, **r))
                auc_rows.append(dict(scheme=scheme, T=T, classifier=clf, session="MEAN7",
                                     auc_tiled=m7, ci_lo=lo, ci_hi=hi))
                print(f"[cell] {scheme} T={T} {clf:9s} mean7 {m7:.4f} [{lo:.4f}, {hi:.4f}]",
                      flush=True)
            if scheme == "random":
                M.write(os.path.join(RES, f"gate_win{T:.0f}_scores"
                                          + ("_T3" if T == 3.0 else "")
                                          + "_within_foldref.csv"), st)
                if T == 3.0:
                    keep = dict(D=D, RAW=RAW, mods=mods)
            elif T == 3.0:
                M.write(os.path.join(RES, "gate_win3_scores_T3_within_foldref_earliest.csv"),
                        st)
        del RAW

    span = transition_span(keep["D"], None, keep["mods"], 3.0, None)
    npair, nopen, nclose = close_fraction(span, 3.0)
    print(f"[close foldref T=3] {nclose}/{nopen} of {npair} pairs = {nclose / nopen:.4f}",
          flush=True)
    M.write(os.path.join(RES, "gate_win3_transition_scores_T3_within_foldref.csv"), span)
    M.write(os.path.join(RES, "gate_win3_closefrac_within_foldref.csv"),
            [dict(T=3.0, scheme="random", n_pairs=npair, n_open_pairs=nopen,
                  n_close_pairs=nclose, close_frac=nclose / nopen)])
    M.write(os.path.join(RES, "gate_win3_auc_within_foldref.csv"), auc_rows,
            head=["scheme", "T", "classifier", "session"])
    M.write(os.path.join(RES, "gate_win3_thresholds_within_foldref.csv"), thr_all)
    print(f"[numerics] damped-Karcher fallbacks used: {N_DAMPED} of "
          f"{len(thr_all) * len(pgf.BANDS)} band-reference solves", flush=True)
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
