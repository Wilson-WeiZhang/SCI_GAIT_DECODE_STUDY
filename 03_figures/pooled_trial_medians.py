#!/usr/bin/env python3
"""Numbers reported in the Supplementary Information: pooled trial-wise medians (both cohorts, imagined walking vs resting-state EEG input)

Computes numbers quoted in the text; it performs no preprocessing and
fits no model.
"""

import csv, glob, statistics as st
from collections import defaultdict
from pathlib import Path

import analysis_config as _ec
_ec.require("main", "main")

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data/supplementary"

def trial_means_from_long(path, model="calib"):
    per = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != model:
                continue
            try:
                v = float(row["rpt_r"])
            except ValueError:
                continue
            if v != v:
                continue
            per[(row["session"], row["fold"], row["trial_idx"])].append(v)
    return {k: st.fmean(v) for k, v in per.items()}

def summarize(vals):
    q = st.quantiles(vals, n=4)
    return dict(n_trials=len(vals), median=st.median(vals), q25=q[0], q75=q[2])

rows = []
with open(_ec.sd("fig2g_trial_r_long.csv"), newline="") as f:
    hiw = [float(r["All"]) for r in csv.DictReader(f) if r["All"] not in ("", "nan")]
    f.seek(0); hs = {r["session"] for r in csv.DictReader(f)}
rows.append(("healthy", "imagined_walking", summarize(hiw), len(hs)))
hr = trial_means_from_long(_ec.path("h_rest_trial_r_long"))
rows.append(("healthy", "resting_input", summarize(list(hr.values())), len({k[0] for k in hr})))
with open(_ec.sd("fig5b_trial_matrix.csv"), newline="") as f:
    rd = csv.DictReader(f); cols = [c for c in rd.fieldnames if "post" in c]
    piw = [float(r[c]) for r in rd for c in cols if r[c] not in ("", "nan")]
rows.append(("patient_post_operative", "imagined_walking", summarize(piw), len(cols)))
pr = {}
for fn in sorted(glob.glob(str(_ec.path("p_rest_dump_dir") / "*_trial_r.csv"))):
    pr.update(trial_means_from_long(fn))
pr = {k: v for k, v in pr.items() if _ec.keep_trial(k[0], "REST", k[2])}
rows.append(("patient_post_operative", "resting_input", summarize(list(pr.values())), len({k[0] for k in pr})))

with open(_ec.sd("pooled_trial_medians_source_data.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["cohort", "input", "n_trials", "n_sessions", "median_r", "q25", "q75"])
    for coh, inp, s, ns in rows:
        w.writerow([coh, inp, s["n_trials"], ns, f"{s['median']:.6f}", f"{s['q25']:.6f}", f"{s['q75']:.6f}"])
        print(f"{coh:17s} {inp:17s} n={s['n_trials']:5d} sessions={ns:3d} median={s['median']:.4f} IQR=[{s['q25']:.3f}, {s['q75']:.3f}]")
with open(_ec.sd("pooled_trial_medians_metadata.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["key", "value"])
    for k, v in [
        ("paper_location", "Supplementary Note 1 (pooled trial medians; both cohorts, imagined walking vs resting-state EEG input)"),
        ("statistic", "median over TRIALS of the per-trial all-joint mean rpt_r (model calib), pooled across sessions; IQR = 25th/75th percentile over trials"),
        ("healthy_iw_source", f"{_ec.sd('fig2g_trial_r_long.csv').relative_to(ROOT).as_posix()} column All (mean of six joints, model calib, rpt_r)"),
        ("healthy_rest_source", f"{_ec.rel('h_rest_trial_r_long')}, model calib, rpt_r averaged over the six joints per trial (INPUT_CONTROL_MODE=resting_random_window_test_only, seed 20260820)"),
        ("patient_iw_source", f"{_ec.sd('fig5b_trial_matrix.csv').relative_to(ROOT).as_posix()}, the 11 post-operative columns (post_operative)"),
        ("scope", "descriptive; healthy full cohort 80 sessions / 42 participants; patient n = 2, case-level"),
        ("producer", "03_figures/pooled_trial_medians.py"),
    ]:
        w.writerow([k, v])
print("wrote", _ec.sd("pooled_trial_medians_source_data.csv"))
