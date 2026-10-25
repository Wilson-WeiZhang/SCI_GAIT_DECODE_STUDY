"""Figure 7 — physics simulation and walk-intent gate

gate_scores_within_session — the gate under WITHIN-SESSION 10-FOLD cross-validation on the post-operative patient sessions

Reads:
  the aligned per-session recordings (not distributed)

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
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import karcher_checked as KC
import gate_features as pgf
import gate_fusion as PF
import gate_window3_within_session as M
import gate_window3_fold_reference as FR

RES = os.path.join(HERE, "result")
ROOT_V47 = "${GAIT_PROJECT_ROOT}/patient_prep_restgood_v4_7"
ROOT_G11 = "${GAIT_PROJECT_ROOT}/patient_prep_gate"

SESS7 = ["pat01_sess04", "pat01_sess06", "pat01_sess07", "pat01_sess09",
         "pat02_sess03", "pat02_sess05", "pat02_sess06"]
SESS11 = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
          "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06",
          "pat02_sess08", "pat02_sess09"]

LAM = PF.LAM_RA
REF_MIN_S = FR.REF_MIN_S
SEED, B_BOOT, NFOLD = M.SEED, M.B_BOOT, M.NFOLD
HOP_S = M.HOP_S

FOLDREF_AUC = os.path.join(RES, "gate_win3_auc_within_foldref.csv")
COMPARATOR = {3.0: 0.9230661334861122, 4.0: 0.9460975578597545}
TOL_MEAN, TOL_SESS, TOL_HARNESS = 5e-4, 1e-4, 1e-9

KARCHER_LOG = []


def log(m):
    print(m, flush=True)


def checked_whiteners(covs, lam=LAM):
    """FR.whiteners_safe's contract, with the CONVERGED solver and convergence recorded."""
    res = KC.band_karcher(covs, lam, "checked")
    KARCHER_LOG.extend(res)
    return [PF.powm_sym(r.mean, -0.5) for r in res]


SOLVERS = {"frozen_safe": FR.whiteners_safe, "checked": checked_whiteners}


def prepare_raw2(sessions, root, T, base_min_s):
    """FR.prepare_raw, parameterised over (sessions, root, base_min_s) and tolerant of an
    EMPTY calibration list (which is what base_min_s = 0 produces -- the allrest pool)."""
    t0 = time.time()
    pgf.WIN_S = T
    PF.WIN_S = T
    out, anchors, inv = {}, [], []
    for s in sessions:
        X, fs, cal, ordered_eval = PF.session_layout(root, s, base_min_s)
        calP, calC = [], []
        for e in cal:
            for w in pgf.windows(X, e["a"], e["b"], fs):
                calP.append(pgf.bandpower(w, fs))
                calC.append(PF.band_covs(w))
        tiled, dense, dur, order = {}, {}, {}, []
        n_t = n_d = n_zero = 0
        for lab, k, e in ordered_eval:
            name = f"{lab}{k}"
            order.append(name)
            dur[name] = (e["b"] - e["a"]) / fs
            anchors.append(dict(session=s, event=name, role="evaluated",
                                label=1 if lab == "iw" else 0, a=e["a"], b=e["b"],
                                t_start_s=e["a"] / fs, t_stop_s=e["b"] / fs,
                                dur_s=(e["b"] - e["a"]) / fs))
            ws = pgf.windows(X, e["a"], e["b"], fs)
            if ws:
                tiled[name] = (np.vstack([pgf.bandpower(w, fs) for w in ws]),
                               [PF.band_covs(w) for w in ws])
                n_t += len(ws)
            else:
                n_zero += 1
            dw = M.dense_windows(X, e["a"], e["b"], fs, T)
            if dw:
                dense[name] = ([t for t, _w in dw],
                               np.vstack([pgf.bandpower(w, fs) for _t, w in dw]),
                               [PF.band_covs(w) for _t, w in dw])
                n_d += len(dw)
        anom = []
        if abs(fs - 250.0) > 1e-9:
            anom.append(f"fs={fs}")
        if n_zero:
            anom.append(f"events_without_a_{T:.0f}s_window={n_zero} "
                        f"({','.join(n for n in order if n not in tiled)})")
        out[s] = dict(calP=(np.vstack(calP) if calP else None), calC=calC,
                      tiled=tiled, dense=dense, dur=dur, order=order)
        inv.append(dict(session=s, T=T, root=os.path.basename(root), base_min_s=base_min_s,
                        n_iw_ev=sum(1 for n in order if n.startswith("iw")),
                        n_rest_ev=sum(1 for n in order if n.startswith("rest")),
                        n_calib_ev=len(cal), n_calib_win=len(calC),
                        n_events_with_windows=len(tiled), n_tiled_win=n_t, n_dense_dec=n_d,
                        anomalies="; ".join(anom)))
        log(f"  [raw T={T} {os.path.basename(root)} base={base_min_s:g}] {s}: "
            f"{len(order)} events ({len(tiled)} windowable), {n_t} tiled, {n_d} dense"
            + (f"   ANOMALY: {'; '.join(anom)}" if anom else ""))
    log(f"  prepare_raw2 done in {time.time() - t0:.0f} s")
    return out, anchors, inv


def minimal_D(RAW, sessions, T):
    """The (T, y, sess, evn, onset) frame FR.run_scheme and M.folds_for consume."""
    sess, evn, y = [], [], []
    for s in sessions:
        for n in RAW[s]["order"]:
            if n not in RAW[s]["tiled"]:
                continue
            k = len(RAW[s]["tiled"][n][0])
            sess += [s] * k
            evn += [n] * k
            y += [1 if n.startswith("iw") else 0] * k
    onset = {}
    return dict(T=T, y=np.array(y), sess=np.array(sess), evn=np.array(evn), onset=onset)


def attach_onsets(D, anchors):
    for r in anchors:
        D["onset"][(r["session"], r["event"])] = r["a"]
    return D


def fold_table(D, sessions, T, tag):
    """The frozen fold assignment, and the (session, event) -> fold map, from M.folds_for."""
    rows, fmap = [], {}
    for s in sessions:
        names, folds = M.folds_for(D, s, "within_random")
        assert len(folds) == NFOLD, f"{s}: {len(folds)} folds"
        seen = set()
        for k, (tr, te) in enumerate(folds):
            for i in te:
                assert names[i] not in seen, f"{s}: {names[i]} in two test folds"
                seen.add(names[i])
                fmap[(s, names[i])] = k
                rows.append(dict(tag=tag, T=T, session=s, fold=k, event=names[i],
                                 label=1 if names[i].startswith("iw") else 0, role="test"))
            assert set(tr).isdisjoint(set(te)), f"{s} fold {k}: train/test overlap"
            assert len(tr) + len(te) == len(names), f"{s} fold {k}: partition incomplete"
        assert seen == set(names), f"{s}: test folds do not cover the event list"
    return rows, fmap


def run_rung(name, sessions, root, T, solver, base_min_s, scheme, want_dense, chk,
             RAWCACHE, fold_rows=None):
    """One rung of the ladder. Returns (per-session AUC dict, mean, ci_lo, ci_hi, stream,
    thr_rows, D, fmap)."""
    log(f"\n=========== RUNG {name}: {len(sessions)} sess, {os.path.basename(root)}, T={T}, "
        f"solver={solver}, base_min_s={base_min_s:g}, ref={scheme} ===")
    key = (tuple(sessions), root, T, base_min_s)
    if key not in RAWCACHE:
        RAWCACHE[key] = prepare_raw2(sessions, root, T, base_min_s)
    RAW, anchors, inv = RAWCACHE[key]
    D = attach_onsets(minimal_D(RAW, sessions, T), anchors)
    frows, fmap = fold_table(D, sessions, T, name)
    if fold_rows is not None:
        fold_rows.extend(frows)

    n0 = len(KARCHER_LOG)
    FR.SESSIONS = list(sessions)
    M.SESSIONS = list(sessions)
    FR.whiteners_safe = SOLVERS[solver]
    FR.REF_MIN_S = REF_MIN_S
    stream, tsc, thr_rows, _mods = FR.run_scheme(D, RAW, T, scheme, want_dense=want_dense)
    nonconv = sum(1 for r in KARCHER_LOG[n0:] if not r.converged)
    if solver == "checked":
        its = [r.n_iter for r in KARCHER_LOG[n0:]]
        log(f"  [karcher {name}] {len(its)} checked solves, iters "
            f"{min(its)}..{max(its)} (median {int(np.median(its))}), non-converged {nonconv}")
        chk.append(dict(check=f"{name}_karcher", key="non_converged", n=len(its),
                        max_abs_delta=nonconv, tol=0, passed=int(nonconv == 0)))

    boots = M.boot_index(D)
    rows, mean_, lo, hi = FR.cell(D, tsc, "fuse2_V1", boots)
    per = {r["session"]: r["auc_tiled"] for r in rows}
    for r in thr_rows:
        r["rung"] = name
    log(f"[RUNG {name}] fuse2 V1 tiled mean-of-{len(sessions)} {mean_:.6f} [{lo:.4f}, {hi:.4f}]")
    return per, mean_, lo, hi, stream, thr_rows, rows, D, fmap, inv


def check_prepare_raw2_equivalence(chk):
    """prepare_raw2 must be FR.prepare_raw in the demobase configuration ."""
    FR.SESSIONS = list(SESS7)
    FR.ROOT = ROOT_V47
    FR.BASE_MIN_S = 30.0
    ref = FR.prepare_raw(3.0)
    got, _anch, _inv = prepare_raw2(SESS7, ROOT_V47, 3.0, 30.0)
    worst = 0.0
    for s in SESS7:
        worst = max(worst, float(np.max(np.abs(ref[s]["calP"] - got[s]["calP"]))))
        assert set(ref[s]["tiled"]) == set(got[s]["tiled"]), f"{s}: tiled event sets differ"
        assert set(ref[s]["dense"]) == set(got[s]["dense"]), f"{s}: dense event sets differ"
        for n in ref[s]["tiled"]:
            worst = max(worst, float(np.max(np.abs(ref[s]["tiled"][n][0]
                                                  - got[s]["tiled"][n][0]))))
            for a, b in zip(ref[s]["tiled"][n][1], got[s]["tiled"][n][1]):
                worst = max(worst, float(np.max(np.abs(np.array(a) - np.array(b)))))
        for n in ref[s]["dense"]:
            assert ref[s]["dense"][n][0] == got[s]["dense"][n][0], f"{s}|{n}: dense grid differs"
            worst = max(worst, float(np.max(np.abs(ref[s]["dense"][n][1]
                                                  - got[s]["dense"][n][1]))))
    log(f"[SELF-CHECK prepare_raw2] max |delta| vs FR.prepare_raw = {worst:.3e}")
    chk.append(dict(check="harness_prepare_raw2", key="T3_demobase_7", n=len(SESS7),
                    max_abs_delta=worst, tol=0.0, passed=int(worst == 0.0)))
    assert worst == 0.0, "prepare_raw2 is not FR.prepare_raw"
    del ref
    return got


def check_minimal_D(chk):
    """minimal_D must reproduce M.prepare()'s (sess, evn, y, onset) exactly."""
    M.SESSIONS = list(SESS7)
    M.ROOT = ROOT_V47
    M.BASE_MIN_S = 30.0
    Dref = M.prepare(3.0)
    RAW, anchors, _inv = prepare_raw2(SESS7, ROOT_V47, 3.0, 30.0)
    Dgot = attach_onsets(minimal_D(RAW, SESS7, 3.0), anchors)
    ok = (list(Dref["sess"]) == list(Dgot["sess"]) and list(Dref["evn"]) == list(Dgot["evn"])
          and list(Dref["y"]) == list(Dgot["y"])
          and all(Dref["onset"][k] == Dgot["onset"][k] for k in Dref["onset"]))
    log(f"[SELF-CHECK minimal_D] identical to M.prepare frame: {ok} "
        f"({len(Dref['sess'])} vs {len(Dgot['sess'])} rows)")
    chk.append(dict(check="harness_minimal_D", key="T3_demobase_7", n=len(Dref["sess"]),
                    max_abs_delta=0 if ok else 1, tol=0, passed=int(ok)))
    assert ok, "minimal_D diverged from M.prepare's frame"
    del Dref
    return RAW, anchors


def check_no_leakage(name, thr_rows, D, sessions, fmap, chk):
    """Protocol 6.5: for every fold, reference and fit events are TRAINING events only."""
    bad = 0
    for r in thr_rows:
        s, k = r["session"], r["fold"]
        refs = [x for x in str(r["ref_events"]).split(";") if x]
        for n in refs:
            if fmap[(s, n)] == k:
                bad += 1
        assert r["n_ref_events"] + r["n_fit_events"] == \
            sum(1 for (ss, nn), kk in fmap.items() if ss == s and kk != k), \
            f"{name} {s} fold {k}: ref+fit != training-event count"
    chk.append(dict(check=f"{name}_no_leakage", key="ref_events_in_test_fold",
                    n=len(thr_rows), max_abs_delta=bad, tol=0, passed=int(bad == 0)))
    assert bad == 0, f"{name}: {bad} reference events came from the test fold"
    log(f"[SELF-CHECK {name} leakage] 0 reference events from a test fold, "
        f"ref+fit == training count on all {len(thr_rows)} folds")


def check_stream(name, stream, RAWinv, fmap, chk):
    """Protocol 6.3 / 6.6: dense row counts and one-fold-per-event."""
    per, evfold = {}, {}
    for r in stream:
        per[r["session"]] = per.get(r["session"], 0) + 1
        k = (r["session"], r["event"])
        evfold.setdefault(k, set()).add(r["fold"])
    worst = 0
    for d in RAWinv:
        got, want = per.get(d["session"], 0), d["n_dense_dec"]
        worst = max(worst, abs(got - want))
        chk.append(dict(check=f"{name}_dense_row_count", key=d["session"], n=got,
                        max_abs_delta=abs(got - want), tol=0, passed=int(got == want)))
        assert got == want, f"{name} {d['session']}: {got} dense rows, loader says {want}"
    multi = [k for k, v in evfold.items() if len(v) != 1]
    assert not multi, f"{name}: events scored from more than one fold: {multi[:5]}"
    mism = [k for k, v in evfold.items() if list(v)[0] != fmap[k]]
    assert not mism, f"{name}: fold column disagrees with the fold map: {mism[:5]}"
    chk.append(dict(check=f"{name}_one_fold_per_event", key="all", n=len(evfold),
                    max_abs_delta=len(multi) + len(mism), tol=0,
                    passed=int(not multi and not mism)))
    log(f"[SELF-CHECK {name} stream] {len(stream)} rows, {len(evfold)} events, "
        f"max |delta| vs loader {worst}, one fold per event, fold column consistent")


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    chk, auc_rows, fold_rows, thr_all = [], [], [], {3.0: [], 4.0: []}
    RAWCACHE = {}

    ref_pub = {}
    for r in csv.DictReader(open(FOLDREF_AUC, encoding="utf-8")):
        if r["scheme"] == "random" and r["classifier"] == "fuse2_V1":
            ref_pub[(float(r["T"]), r["session"])] = float(r["auc_tiled"])
    R0 = {}
    for T in (3.0, 4.0):
        FR.SESSIONS = list(SESS7)
        FR.ROOT = ROOT_V47
        FR.BASE_MIN_S = 30.0
        FR.whiteners_safe = SOLVERS["frozen_safe"]
        M.SESSIONS = list(SESS7)
        M.ROOT = ROOT_V47
        M.BASE_MIN_S = 30.0
        D = M.prepare(T)
        for s in SESS7:
            D["dense"][s].pop("bp", None)
            D["dense"][s].pop("ra", None)
        D.pop("Xra", None)
        RAW = FR.prepare_raw(T)
        boots = M.boot_index(D)
        _st, tsc, _thr, _mo = FR.run_scheme(D, RAW, T, "random", want_dense=False)
        rows, m7, lo, hi = FR.cell(D, tsc, "fuse2_V1", boots)
        per = {r["session"]: r["auc_tiled"] for r in rows}
        dmean = abs(m7 - COMPARATOR[T])
        dsess = max(abs(per[s] - ref_pub[(T, s)]) for s in SESS7)
        log(f"[R0 T={T}] mean7 {m7:.9f} vs published {COMPARATOR[T]:.9f} "
            f"(|delta| {dmean:.3e}); worst per-session |delta| {dsess:.3e}")
        chk.append(dict(check="R0_comparator_mean", key=f"T{T:.0f}", n=7,
                        max_abs_delta=dmean, tol=TOL_MEAN, passed=int(dmean <= TOL_MEAN)))
        chk.append(dict(check="R0_comparator_session", key=f"T{T:.0f}", n=7,
                        max_abs_delta=dsess, tol=TOL_SESS, passed=int(dsess <= TOL_SESS)))
        assert dmean <= TOL_MEAN and dsess <= TOL_SESS, f"R0 T={T} comparator FAILED"
        R0[T] = per
        auc_rows.append(dict(rung="R0_published_foldref", T=T, sessions=7, root="restgood_v4_7",
                             solver="frozen_safe", pool="demobase", ref_scheme="random",
                             session=f"MEAN7", auc_tiled=m7, ci_lo=lo, ci_hi=hi))
        for s in SESS7:
            auc_rows.append(dict(rung="R0_published_foldref", T=T, sessions=7,
                                 root="restgood_v4_7", solver="frozen_safe", pool="demobase",
                                 ref_scheme="random", session=s, auc_tiled=per[s]))
        del RAW, D
    log("[R0] PASS -- the published foldref comparator is reproduced at T = 3 and T = 4")

    check_prepare_raw2_equivalence(chk)
    RAW7, anch7 = check_minimal_D(chk)
    inv7 = [dict(session=s, T=3.0, root="restgood_v4_7", base_min_s=30.0,
                 n_iw_ev=sum(1 for n in RAW7[s]["order"] if n.startswith("iw")),
                 n_rest_ev=sum(1 for n in RAW7[s]["order"] if n.startswith("rest")),
                 n_calib_ev="", n_calib_win=len(RAW7[s]["calC"]),
                 n_events_with_windows=len(RAW7[s]["tiled"]),
                 n_tiled_win=sum(len(v[0]) for v in RAW7[s]["tiled"].values()),
                 n_dense_dec=sum(len(v[0]) for v in RAW7[s]["dense"].values()),
                 anomalies="") for s in SESS7]
    RAWCACHE[(tuple(SESS7), ROOT_V47, 3.0, 30.0)] = (RAW7, anch7, inv7)

    LADDER = [
        ("R1_harness",    SESS7,  ROOT_V47, "frozen_safe", 30.0, "random",   [3.0, 4.0]),
        ("L1_solver",     SESS7,  ROOT_V47, "checked",     30.0, "random",   [3.0]),
        ("L2_loading",    SESS7,  ROOT_G11, "checked",     30.0, "random",   [3.0]),
        ("L3_sessions",   SESS11, ROOT_G11, "checked",     30.0, "random",   [3.0]),
        ("L4_pool",       SESS11, ROOT_G11, "checked",      0.0, "random",   [3.0, 4.0]),
        ("L5_headline",   SESS11, ROOT_G11, "checked",      0.0, "earliest", [3.0, 4.0]),
    ]
    head = {}
    inv_all = []
    plan = [(i, name, sess, root, solver, bms, scheme, T)
            for i, (name, sess, root, solver, bms, scheme, Ts) in enumerate(LADDER) for T in Ts]
    last_use = {}
    for j, (_i, _n, sess, root, _s, bms, _sc, T) in enumerate(plan):
        last_use[(tuple(sess), root, T, bms)] = j
    for j, (_i, name, sess, root, solver, bms, scheme, T) in enumerate(plan):
        if 1:
            want_dense = name in ("L4_pool", "L5_headline")
            fr = fold_rows if (name == "L5_headline") else None
            per, m_, lo, hi, stream, thr_rows, rows, D, fmap, inv = run_rung(
                name, sess, root, T, solver, bms, scheme, want_dense, chk, RAWCACHE,
                fold_rows=fr)
            check_no_leakage(f"{name}_T{T:.0f}", thr_rows, D, sess, fmap, chk)
            if want_dense:
                check_stream(f"{name}_T{T:.0f}", stream, inv, fmap, chk)
            if name == "R1_harness":
                d = max(abs(per[s] - R0[T][s]) for s in sess)
                log(f"[R1 T={T}] harness vs R0: worst per-session |delta| {d:.3e}")
                chk.append(dict(check="R1_harness_equivalence", key=f"T{T:.0f}", n=len(sess),
                                max_abs_delta=d, tol=TOL_HARNESS, passed=int(d <= TOL_HARNESS)))
                assert d <= TOL_HARNESS, f"R1 harness diverged from R0 by {d:.3e}"
            auc_rows.append(dict(rung=name, T=T, sessions=len(sess),
                                 root=os.path.basename(root), solver=solver,
                                 pool=("allrest" if bms == 0.0 else "demobase"),
                                 ref_scheme=scheme, session=f"MEAN{len(sess)}",
                                 auc_tiled=m_, ci_lo=lo, ci_hi=hi))
            for r in rows:
                auc_rows.append(dict(rung=name, T=T, sessions=len(sess),
                                     root=os.path.basename(root), solver=solver,
                                     pool=("allrest" if bms == 0.0 else "demobase"),
                                     ref_scheme=scheme, session=r["session"],
                                     auc_tiled=r["auc_tiled"], ci_lo=r["ci_lo"],
                                     ci_hi=r["ci_hi"], n_win=r["n_win"],
                                     auc_trial=r.get("auc_trial")))
            thr_all[T] += thr_rows
            if name == "L5_headline":
                head[T] = dict(stream=stream, per=per, mean=m_, lo=lo, hi=hi,
                               anchors=RAWCACHE[(tuple(sess), root, T, bms)][1])
            inv_all += inv
            del D
            ck = (tuple(sess), root, T, bms)
            if last_use.get(ck) == j and ck in RAWCACHE:
                del RAWCACHE[ck]
                log(f"  [cache] dropped {os.path.basename(root)} T={T} base={bms:g} "
                    f"({len(sess)} sess)")
    inv_all += inv7

    for d in inv_all:
        chk.append(dict(check="loader_inventory", **d))
    for T in (3.0, 4.0):
        M.write(os.path.join(RES, f"gate_within_session_scores_T{T:.0f}.csv"),
                head[T]["stream"])
        M.write(os.path.join(RES, f"gate_within_session_thresholds_T{T:.0f}.csv"), thr_all[T],
                head=["rung", "scheme", "T", "session", "fold"])
    M.write(os.path.join(RES, "gate_within_session_folds.csv"), fold_rows)
    M.write(os.path.join(RES, "gate_within_session_events.csv"), head[3.0]["anchors"])
    M.write(os.path.join(RES, "gate_within_session_auc.csv"), auc_rows,
            head=["rung", "T", "sessions", "root", "solver", "pool", "ref_scheme", "session"])
    M.write(os.path.join(RES, "gate_within_session_selfcheck.csv"), chk,
            head=["check", "key"])
    log(f"\nHEADLINE within10 / 3 s: {head[3.0]['mean']:.6f} "
        f"[{head[3.0]['lo']:.4f}, {head[3.0]['hi']:.4f}]   (11 sessions)")
    log(f"           within10 / 4 s: {head[4.0]['mean']:.6f} "
        f"[{head[4.0]['lo']:.4f}, {head[4.0]['hi']:.4f}]")
    log(f"\nDONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
