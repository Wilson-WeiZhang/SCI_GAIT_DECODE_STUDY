"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

panel c — per-session, per-trial decoding r (Hip/Knee/Ankle/All)

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


def _quantile(vals, q):
    """Linear-interpolation quantile (numpy 'linear' convention) on a sorted copy."""
    v = sorted(vals)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)

ROOT = Path(__file__).resolve().parents[1]

import analysis_config as _ec
GAIT_BUILD_ID = "main"
_ERA_TAG = "main"

SRC = _ec.path("h_iw_trial_r_long")
OUTDIR = ROOT / "04_source_data" / "fig2"
GROUPS = {"Hip": ["LHipW X", "RHipW X"], "Knee": ["LKneW X", "RKneW X"],
          "Ankle": ["LAnkW X", "RAnkW X"],
          "All": ["LHipW X", "RHipW X", "LKneW X", "RKneW X", "LAnkW X", "RAnkW X"]}


def nanmean(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def main() -> None:
    per = defaultdict(dict)
    meta = {}
    for row in csv.DictReader(SRC.open(newline="")):
        if row["model"] != "calib":
            continue
        key = (row["session"], int(row["trial_idx"]))
        try:
            v = float(row["rpt_r"])
            if v != v:
                v = None
        except ValueError:
            v = None
        per[key][row["joint"]] = v
        meta[key] = (int(row["test_ordinal"]), int(row["fold"]))

    trials = []
    for (sess, tidx), jr in per.items():
        rec = {"subject": sess.split("_sess")[0], "session": sess, "trial_idx": tidx,
               "test_ordinal": meta[(sess, tidx)][0], "fold": meta[(sess, tidx)][1]}
        for g, js in GROUPS.items():
            rec[g] = nanmean([jr.get(j) for j in js])
        if rec["All"] is None:
            continue
        trials.append(rec)
    trials.sort(key=lambda r: (r["session"], r["trial_idx"]))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with _ec.sd("fig2g_trial_r_long.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "session", "trial_idx", "test_ordinal", "fold", "Hip", "Knee", "Ankle", "All"])
        for r in trials:
            w.writerow([r["subject"], r["session"], r["trial_idx"], r["test_ordinal"], r["fold"]] +
                       [("" if r[g] is None else f"{r[g]:.6f}") for g in ("Hip", "Knee", "Ankle", "All")])

    bysess = defaultdict(list)
    for r in trials:
        bysess[r["session"]].append(r)
    sess_all_median = {s: median([r["All"] for r in rs]) for s, rs in bysess.items()}
    sess_all_mean = {s: mean([r["All"] for r in rs]) for s, rs in bysess.items()}
    sessions = sorted(bysess, key=lambda s: -sess_all_median[s])

    with _ec.sd("fig2g_session_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        head = ["session", "subject", "n_trials"]
        for g in ("Hip", "Knee", "Ankle", "All"):
            head += [f"{g}_median", f"{g}_q25", f"{g}_q75", f"{g}_mean", f"{g}_sd"]
        w.writerow(head)
        for s in sessions:
            rs = bysess[s]
            row = [s, s.split("_sess")[0], len(rs)]
            for g in ("Hip", "Knee", "Ankle", "All"):
                vals = [r[g] for r in rs if r[g] is not None]
                row += [f"{median(vals):.6f}", f"{_quantile(vals, 0.25):.6f}",
                        f"{_quantile(vals, 0.75):.6f}", f"{mean(vals):.6f}",
                        f"{pstdev(vals):.6f}" if len(vals) > 1 else "0.000000"]
            w.writerow(row)

    for g in ("Hip", "Knee", "Ankle", "All"):
        g_median = {s: median([r[g] for r in bysess[s] if r[g] is not None]) for s in bysess}
        g_sessions = sorted(bysess, key=lambda s: -g_median[s])
        cols = {s: [r[g] for r in sorted(bysess[s], key=lambda r: r["trial_idx"])] for s in g_sessions}
        nmax = max(len(v) for v in cols.values())
        with _ec.sd(f"fig2g_trial_r_matrix_{g}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(g_sessions)
            sessions = g_sessions
            for i in range(nmax):
                w.writerow([("" if i >= len(cols[s]) or cols[s][i] is None else f"{cols[s][i]:.6f}")
                            for s in sessions])

    rest_src = _ec.path("h_rest_trial_r_long")
    per_r = defaultdict(dict)
    for row in csv.DictReader(rest_src.open(newline="")):
        if row["model"] != "calib" or row["rpt_r"] in ("", "NaN"):
            continue
        per_r[(row["session"], int(row["trial_idx"]))][row["joint"]] = float(row["rpt_r"])
    rest_bysess = defaultdict(lambda: defaultdict(list))
    for (s_, _t), jr in per_r.items():
        for g, js in GROUPS.items():
            vals = [jr[j] for j in js if j in jr]
            if vals:
                rest_bysess[s_][g].append(mean(vals))
    with _ec.sd("fig2g_rest_null_band.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "convention", "rest_center", "band_lo_p2p5", "band_hi_p97p5",
                    "n_sessions", "n_iw_sessions_above_hi"])
        for g in ("Hip", "Knee", "Ankle", "All"):
            for conv, agg in (("session_median", median), ("session_mean", mean)):
                rv = sorted(agg(rest_bysess[s_][g]) for s_ in rest_bysess)
                lo = _quantile(rv, 0.025); hi = _quantile(rv, 0.975)
                iw = [agg([r[g] for r in bysess[s_] if r[g] is not None]) for s_ in bysess]
                w.writerow([g, conv, f"{median(rv):.6f}", f"{lo:.6f}", f"{hi:.6f}",
                            len(rv), sum(1 for v in iw if v > hi)])
    print(f"rest null band: {len(rest_bysess)} rest sessions -> fig2g_rest_null_band.csv")

    cohort_anchor = mean(sess_all_mean.values())
    print(f"trials={len(trials)}  sessions={len(sessions)}  subjects={len({r['subject'] for r in trials})}")
    exp = _ec.anchor("iw_session_mean")[0]
    flag = "" if exp is None else ("  OK" if abs(cohort_anchor - exp) < 5e-4 else f"  <-- MISMATCH vs {exp:.6f}")
    print(f"cohort session-mean All = {cohort_anchor:.6f}{flag}")

    print("cohort pooled-trial medians (Supplementary Note 1 convention):")
    for g in ("All", "Hip", "Knee", "Ankle"):
        vals = [r[g] for r in trials if r[g] is not None]
        print(f"  {g:5s} median {median(vals):.4f} "
              f"[IQR {_quantile(vals, 0.25):.4f}, {_quantile(vals, 0.75):.4f}]  n={len(vals)} trials")

    print("wrote fig2g_trial_r_long.csv, fig2g_session_summary.csv, fig2g_trial_r_matrix_{Hip,Knee,Ankle,All}.csv -> 04_source_data/")


if __name__ == "__main__":
    main()
