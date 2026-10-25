#!/usr/bin/env python3
"""Numbers reported in the text (Abstract, Results, Supplementary Notes)

Headline decoding accuracy in the session convention: the median across sessions of each session's median trial r

Computes numbers quoted in the text; it performs no preprocessing and
fits no model.
"""

import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

import analysis_config as _ec

_ec.require("main", "main")

ROOT = Path(__file__).resolve().parents[1]


def from_long(path, model="calib", cond=None):
    """{session: [per-trial mean over joints]} from a long per-joint table; cond = scoring-mask condition of a raw per-session trial_r.csv"""
    per = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model", model) != model:
                continue
            if cond and not _ec.keep_trial(row["session"], cond, row["trial_idx"]):
                continue
            try:
                v = float(row["rpt_r"])
            except ValueError:
                continue
            if v == v:
                per[(row["session"], row["fold"], row["trial_idx"])].append(v)
    out = defaultdict(list)
    for (s, _, _), v in per.items():
        out[s].append(float(np.mean(v)))
    return out


def stat(by_session):
    med = np.array([np.median(v) for v in by_session.values()])
    return dict(n_sessions=len(med), n_trials=sum(len(v) for v in by_session.values()), median=float(np.median(med)),
                q25=float(np.percentile(med, 25)), q75=float(np.percentile(med, 75)), lo=float(med.min()), hi=float(med.max()))


rows = []
hiw = {g: defaultdict(list) for g in ("All", "Hip", "Knee", "Ankle")}
with open(_ec.sd("fig2g_trial_r_long.csv"), newline="") as f:
    for r in csv.DictReader(f):
        for g in hiw:
            if r[g] not in ("", "nan"):
                hiw[g][r["session"]].append(float(r[g]))
for g in hiw:
    rows.append(("healthy", "imagined_walking", g, stat(hiw[g])))
rows.append(("healthy", "real_walking", "All", stat(from_long(_ec.path("h_rw_trial_r_long")))))
rows.append(("healthy", "resting_input", "All", stat(from_long(_ec.path("h_rest_trial_r_long")))))
pat = defaultdict(list)
with open(_ec.sd("fig5b_trial_matrix.csv"), newline="") as f:
    for r in csv.DictReader(f):
        for c, v in r.items():
            if v not in ("", "nan"):
                pat[c].append(float(v))
for name, key in (("patient_post_operative", "post"), ("patient_pre_operative", "pre")):
    sel = {c: v for c, v in pat.items() if f" {key} " in c}
    rows.append((name, "imagined_walking", "All", stat(sel)))
    for p in ("P1", "P2"):
        rows.append((f"{name}_{p}", "imagined_walking", "All", stat({c: v for c, v in sel.items() if c.startswith(p)})))
pr = defaultdict(list)
for fn in sorted(glob.glob(str(_ec.path("p_rest_dump_dir") / "*_trial_r.csv"))):
    for s, v in from_long(fn, cond="REST").items():
        pr[s].extend(v)
rows.append(("patient_post_operative", "resting_input", "All", stat(pr)))

out_sd = _ec.sd("headline_session_medians_source_data.csv")
tmp = out_sd.with_suffix(".tmp")
with open(tmp, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["cohort", "input", "joints", "n_sessions", "n_trials", "median_of_session_medians", "q25", "q75", "min", "max"])
    for coh, inp, g, s in rows:
        w.writerow([coh, inp, g, s["n_sessions"], s["n_trials"]] + [f"{s[k]:.6f}" for k in ("median", "q25", "q75", "lo", "hi")])
        print(f"{coh:20s} {inp:17s} {g:5s} sessions={s['n_sessions']:3d} trials={s['n_trials']:5d} "
              f"median={s['median']:.4f} IQR=[{s['q25']:.4f}, {s['q75']:.4f}] range=[{s['lo']:.2f}, {s['hi']:.2f}]")
tmp.replace(out_sd)
out_md = _ec.sd("headline_session_medians_metadata.csv")
tmp = out_md.with_suffix(".tmp")
with open(tmp, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["key", "value"])
    for k, v in [
        ("paper_location", "Abstract; Results (Fig. 2c, Fig. 4b and Fig. 5b sentences); Discussion; Supplementary "
                           "Note 1, Supplementary Note 6 and Supplementary Video 1 legend"),
        ("statistic", "median over SESSIONS of each session's median trial r (per trial: mean over the joints of the group, "
                      "model calib); q25 / q75 / min / max over sessions, linear quantile"),
        ("healthy_iw_source", f"{_ec.sd('fig2g_trial_r_long.csv').relative_to(ROOT).as_posix()}"),
        ("healthy_rw_source", _ec.rel("h_rw_trial_r_long")),
        ("healthy_rest_source", _ec.rel("h_rest_trial_r_long")),
        ("patient_iw_source", f"{_ec.sd('fig5b_trial_matrix.csv').relative_to(ROOT).as_posix()} (columns 'P? post ?' = post_operative, 'P? pre ?' = pre_operative)"),
        ("patient_rest_source", f"{_ec.rel('p_rest_dump_dir')}/*_trial_r.csv"),
        ("scope", "descriptive; healthy full cohort 80 sessions / 42 participants; patients n = 2, case-level"),
        ("producer", "03_figures/headline_session_medians.py"),
    ]:
        w.writerow([k, v])
tmp.replace(out_md)
print("wrote", out_sd)
