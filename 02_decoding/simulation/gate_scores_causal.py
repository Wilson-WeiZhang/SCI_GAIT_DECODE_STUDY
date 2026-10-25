"""Figure 7 — physics simulation and walk-intent gate

The walk-intent gate on the causal preprocessing: gate_scores_within_session.py (within-session 10-fold by
event, converged Karcher mean, all-rest pool, earliest-rest reference, fusion V1) run with its headline
configuration on the gate session set of build_gate_session_set_causal.py, decision windows of 3 s (the
reported gate) and 4 s, with the scorer's own leakage and stream self-checks. First, as a regression check,
the same configuration on the zero-phase gate session set must reproduce that run's score stream (every
column within 1e-6 relative, every per-session AUC within 1e-6, every decision's call and fold identical).
Writes gate_scores_within_session-style tables (scores, thresholds, folds, events, AUC, self-checks).

Reads:
  decoder output in data/decoder_output/ (not distributed)
  the aligned per-session recordings (not distributed)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import csv
import sys
import time

PW = os.environ.get("GAIT_PROJECT_ROOT", "") + "/02_decoding/simulation"
sys.path.insert(0, PW)
import gate_scores_within_session as G

MAIN_ROOT = os.environ.get("GAIT_PROJECT_ROOT", "") + "/data/decoder_output/main/triples/pat_gate"
OUT = os.path.join(PW, "result", "main")
OFFICIAL = os.path.join(PW, "result", "gate_within_session_scores_T3.csv")
OFFICIAL_AUC = "the zero-phase run"
TOL = 1e-9


def rung(tag, root, T, chk, cache, fold_rows=None):
    per, m_, lo, hi, stream, thr, rows, D, fmap, inv = G.run_rung(
        tag, G.SESS11, root, T, "checked", 0.0, "earliest", True, chk, cache, fold_rows=fold_rows)
    G.check_no_leakage(f"{tag}_T{T:.0f}", thr, D, G.SESS11, fmap, chk)
    G.check_stream(f"{tag}_T{T:.0f}", stream, inv, fmap, chk)
    anchors = cache[(tuple(G.SESS11), root, T, 0.0)][1]
    del cache[(tuple(G.SESS11), root, T, 0.0)]
    return dict(per=per, mean=m_, lo=lo, hi=hi, stream=stream, thr=thr, rows=rows, anchors=anchors, inv=inv)


def regression(res, chk):
    off = list(csv.DictReader(open(OFFICIAL, encoding="utf-8")))
    new = res["stream"]
    key = lambda r: (str(r["session"]), str(r["event"]), float(r["t_end_ev_s"]))
    ko = {key(r): r for r in off}
    kn = {key(r): r for r in new}
    same_rows = set(ko) == set(kn) and len(ko) == len(off) and len(kn) == len(new)
    worst = {}
    agree = n = 0
    for k, ro in ko.items():
        rn = kn.get(k)
        if rn is None:
            continue
        for c, v in ro.items():
            if c in ("session", "event"):
                continue
            d = abs(float(v) - float(rn[c])) / max(1.0, abs(float(v)))
            worst[c] = max(worst.get(c, 0.0), d)
        n += 1
        agree += int((float(ro["e_sigma"]) >= 0) == (float(rn["e_sigma"]) >= 0) and int(float(ro["fold"])) == int(rn["fold"]))
    auc_off = {}
    for r in csv.DictReader(open(os.path.join(PW, "result", "gate_within_session_auc.csv"), encoding="utf-8")):
        if r["rung"] == "L5_headline" and float(r["T"]) == 3.0 and r["session"] != "MEAN11":
            auc_off[r["session"]] = float(r["auc_tiled"])
    dauc = max(abs(res["per"][s] - auc_off[s]) for s in auc_off)
    wmax = max(worst.values())
    ok = same_rows and wmax <= 1e-6 and dauc <= 1e-6 and agree == n == len(off)
    G.log(f"[REGRESSION zero-phase T=3] rows identical sets {same_rows} ({len(off)} / {len(new)}); max relative "
          f"|d| per column " + ", ".join(f"{c} {v:.2e}" for c, v in sorted(worst.items())) + f"; per-session AUC max |d| "
          f"{dauc:.2e}; S0 call + fold agreement {agree}/{n}; MEAN11 {res['mean']:.9f} (zero-phase run) -> "
          f"REGRESSION {'PASS' if ok else 'FAIL'} ")
    chk.append(dict(check="_regression_asrun_stream", key="T3", n=len(off), max_abs_delta=wmax, tol=1e-6,
                    passed=int(wmax <= 1e-6 and same_rows)))
    chk.append(dict(check="_regression_asrun_session_auc", key="T3", n=len(auc_off), max_abs_delta=dauc, tol=1e-6,
                    passed=int(dauc <= 1e-6)))
    chk.append(dict(check="_regression_asrun_s0_call_fold", key="T3", n=n, max_abs_delta=n - agree, tol=0,
                    passed=int(agree == n == len(off))))
    os.makedirs(os.path.join(OUT, "_regression"), exist_ok=True)
    G.M.write(os.path.join(OUT, "_regression", "gate_within_session_scores_T3_asrun_rerun.csv"), new)
    return ok


def main():
    t0 = time.time()
    assert os.path.exists(os.path.join(MAIN_ROOT, "READY")), "causal gate root not built (build_gate_session_set_causal.py)"
    if os.path.exists(os.path.join(OUT, "gate_within_session_scores_T3.csv")):
        sys.exit(f"{OUT} already holds a stream; refusing to overwrite")
    chk, cache, fold_rows = [], {}, []
    os.makedirs(OUT, exist_ok=True)
    if os.path.isdir(G.ROOT_G11) and os.path.exists(OFFICIAL):
        asrun = rung("L5_asrun_regression", G.ROOT_G11, 3.0, chk, cache)
        reg_ok = regression(asrun, chk)
    else:
        asrun, reg_ok = None, None
        G.log("internal regression check skipped: the zero-phase gate session set and its stream are not distributed")
    current = {T: rung("L5_causal", MAIN_ROOT, T, chk, cache, fold_rows=(fold_rows if T == 3.0 else None)) for T in (3.0, 4.0)}
    os.makedirs(OUT, exist_ok=True)
    auc_rows = []
    for tag, res, T, root in [("L5_asrun_regression", asrun, 3.0, "patient_prep_gate"),
                              ("L5_causal", current[3.0], 3.0, "pat_gate_causal"), ("L5_causal", current[4.0], 4.0, "pat_gate_causal")]:
        if res is None:
            continue
        base = dict(rung=tag, T=T, sessions=11, root=root, solver="checked", pool="allrest", ref_scheme="earliest")
        auc_rows.append(dict(base, session="MEAN11", auc_tiled=res["mean"], ci_lo=res["lo"], ci_hi=res["hi"]))
        for r in res["rows"]:
            auc_rows.append(dict(base, session=r["session"], auc_tiled=r["auc_tiled"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                 n_win=r["n_win"], auc_trial=r.get("auc_trial")))
    for d in current[3.0]["inv"] + current[4.0]["inv"]:
        chk.append(dict(check="loader_inventory", **d))
    for T in (3.0, 4.0):
        for r in current[T]["thr"]:
            r["rung"] = "L5_causal"
        G.M.write(os.path.join(OUT, f"gate_within_session_scores_T{T:.0f}.csv"), current[T]["stream"])
        G.M.write(os.path.join(OUT, f"gate_within_session_thresholds_T{T:.0f}.csv"), current[T]["thr"],
                  head=["rung", "scheme", "T", "session", "fold"])
    G.M.write(os.path.join(OUT, "gate_within_session_folds.csv"), fold_rows)
    G.M.write(os.path.join(OUT, "gate_within_session_events.csv"), current[3.0]["anchors"])
    G.M.write(os.path.join(OUT, "gate_within_session_auc.csv"), auc_rows,
              head=["rung", "T", "sessions", "root", "solver", "pool", "ref_scheme", "session"])
    G.M.write(os.path.join(OUT, "gate_within_session_selfcheck.csv"), chk, head=["check", "key"])
    G.log(f"\nHEADLINE within10 / 3 s: {current[3.0]['mean']:.6f} [{current[3.0]['lo']:.4f}, {current[3.0]['hi']:.4f}] (11 sessions)")
    G.log(f"        within10 / 4 s: {current[4.0]['mean']:.6f} [{current[4.0]['lo']:.4f}, {current[4.0]['hi']:.4f}]")
    G.log(f"REGRESSION verdict: {'skipped' if reg_ok is None else 'PASS' if reg_ok else 'FAIL'}")
    G.log(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
