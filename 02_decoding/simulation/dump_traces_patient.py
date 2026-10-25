"""Figure 7 — physics simulation and walk-intent gate

Dump the per-rollout hip-x TRACE for the four f5n main-cell conditions (post_operative).

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os, sys, csv, gzip, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import data_io as IO
import replay_rig as RR
from rhythm_metrics import NULLS, LEVEL, PHASE_SRC, LAM
import analysis_config as ec

GRID_DT = 0.04


def _job(J):
    t, sp, tg, tn = J["trial"], J["setpoint"], J["tmpl_g"], J["tmpl_n"]
    Y = t["Y"]; f0 = IO.f0_of(Y)
    if IO.F7_FROM2S:
        W, Y = IO.f7_from2s_inputs(IO.F7_FROM2S_COND[J["condition"]], t)
        r = RR.rollout(IO.f5_targets(Y, tg, tn, J["delta"], f0, PHASE_SRC, LAM, sp, J["session"], warm=W), LEVEL)
    else:
        if J["null"]:
            Y = NULLS[J["null"]](Y, seed=t["trial_idx"])
        r = RR.rollout(IO.f5_targets(Y, tg, tn, J["delta"], f0, PHASE_SRC, LAM, sp, J["session"]), LEVEL)
    tr = r["trace"]
    if tr.size == 0:
        return []
    ix = 1 + 2 * list(r["site_order"]).index("hip")
    t_s, x_m = tr[:, 0], tr[:, ix] - tr[0, ix]
    grid = np.arange(0.0, t_s[-1] + 1e-9, GRID_DT)
    xg = np.interp(grid, t_s, x_m)
    return [(J["stage"], J["condition"], J["session"], t["trial_idx"], round(float(a), 3), round(float(b), 5))
            for a, b in zip(grid, xg)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ec.path("p_dump_dir"))); ap.add_argument("--npz", default=str(ec.path("p_cycavg")))
    ap.add_argument("--out", required=True); ap.add_argument("--sessions", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    IO.build_id_inputs(dump=a.dump, npz=a.npz)
    if ec.SCOPED:
        a.out = str(ec.scoped(a.out))
    tmpl_n = IO.fukuchi_template(); jobs = []
    for s in [x for x in a.sessions.split(",") if x]:
        sp = IO.device_stance_ankle(IO.loso_setpoint(a.npz, s)[0]); tg = IO.loso_template(a.npz, s)
        delta, _ = IO.global_delta(tg, tmpl_n)
        ctx = dict(setpoint=sp, tmpl_g=tg, tmpl_n=tmpl_n, delta=delta, session=s)
        for t in ([] if IO.F7_FROM2S else IO.load_session_trials(a.dump, s, stream="Y_true")):
            jobs.append(dict(stage="calib", condition="wphase", trial=t, null=None, **ctx))
        for t in IO.load_session_trials(a.dump, s, stream="C_pred"):
            for condition, nl in (("dphase", None), ("dphase_iscr", "iscr"), ("dphase_uscr", "uscr")):
                jobs.append(dict(stage="confirm", condition=condition, trial=t, null=nl, **ctx))
    print(f"enumerated {len(jobs)} rollouts", flush=True)
    import multiprocessing as mp
    rows = []
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        for i, rr in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            rows.extend(rr)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    with gzip.open(a.out, "wt", newline="") as fh:
        w = csv.writer(fh); w.writerow(["stage", "condition", "session", "trial_idx", "t_s", "x_m"]); w.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} samples from {len(jobs)} rollouts")


if __name__ == "__main__":
    main()
