"""Figure 7 — physics simulation and walk-intent gate

gate_scores_karcher_checked — the fused (V1) dense decision stream recomputed with the convergence-checked Karcher-mean solver.

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import csv
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import karcher_checked as KC
import gate_fusion as PF
from gate_features import SESSIONS, build

RES = os.path.join(HERE, "result")
KARCHER_CSV = os.path.join(RES, "karcher_convergence.csv")

OUT_SCORES = os.path.join(RES, "gate_fused_checked_scores.csv")
OUT_THR = os.path.join(RES, "gate_fused_checked_thresholds.csv")
OUT_CHK = os.path.join(RES, "gate_fused_checked_selfcheck.csv")

TOL_S2_SESS = 1e-6

SCORE_COLS = ["score_lda", "score_ts", "fuse_V1", "fuse_V2", "fuse_V3"]


def log(m):
    print(m, flush=True)


def tiled_auc(dense_rows):
    """Per-session AUC on the 4 s zero-overlap grid (t_end = 4, 8, ...), the frozen convention."""
    out = {}
    for s in SESSIONS:
        y = [r["label"] for r in dense_rows
             if r["session"] == s and abs(r["t_end_ev_s"] % 4.0) < 1e-9]
        d = [r["fuse_V1"] for r in dense_rows
             if r["session"] == s and abs(r["t_end_ev_s"] % 4.0) < 1e-9]
        out[s] = float(roc_auc_score(y, d))
    return out


def read_karcher_auc():
    out = {}
    for r in csv.DictReader(open(KARCHER_CSV, encoding="utf-8")):
        if r["block"] == "B2_auc":
            out[r["session"]] = float(r["auc_checked"])
    if not out:
        raise RuntimeError(f"no B2_auc rows in {KARCHER_CSV}")
    return out


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    chk_rows = []

    log("== building demobase band-power rows + covariance inventory ==")
    rows_demo, _m = build(PF.ROOT, base_min_s=PF.BASE_MIN_S)
    keys, covs, calib = PF.cov_inventory(PF.ROOT, PF.BASE_MIN_S)
    assert [(r["session"], r["event"], r["win"]) for r in rows_demo] == keys, \
        "cov inventory diverged from build() (demobase)"

    log("\n== checked-solver condition ==")
    PF.whiteners = KC.make_whiteners("checked")
    _auc_c, dense_c, thr_c, _cal_c = PF.deployable(rows_demo, keys, covs, calib, PF.LAM_RA,
                                                   dense=True)

    log("\n== S2  tiled AUC consistency with karcher_convergence.csv ==")
    a_c = tiled_auc(dense_c)
    ref = read_karcher_auc()
    worst_s = 0.0
    for s in SESSIONS:
        d = abs(a_c[s] - ref[s])
        worst_s = max(worst_s, d)
        log(f"  {s}: checked stream {a_c[s]:.6f}  probe {ref[s]:.6f}  |delta| {d:.3e}")
        chk_rows.append(dict(check="S2_tiled_auc", key=s, auc_stream=a_c[s], auc_probe=ref[s],
                             max_abs_delta=d, tol=TOL_S2_SESS, passed=int(d <= TOL_S2_SESS)))
        assert d <= TOL_S2_SESS, f"S2 FAIL {s}: {a_c[s]:.9f} vs probe {ref[s]:.9f}"
    mean7 = float(np.mean([a_c[s] for s in SESSIONS]))
    log(f"[S2] PASS  per-session max |delta| {worst_s:.3e}, mean-of-7 {mean7:.6f}")

    PF.write(OUT_SCORES, dense_c)
    PF.write(OUT_THR, thr_c)
    PF.write(OUT_CHK, chk_rows)
    log(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
