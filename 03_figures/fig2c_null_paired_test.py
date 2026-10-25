#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

Paired real-vs-null test behind the Fig. 2c sentence in Results

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from scipy import stats

import analysis_config as _ec
_ec.require("main", "main")

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "04_source_data/fig2"
IW_SRC = _ec.path("h_iw_trial_r_long")
REST_SRC = _ec.path("h_rest_trial_r_long")

GROUPS = {"Hip": ["LHipW X", "RHipW X"], "Knee": ["LKneW X", "RKneW X"],
          "Ankle": ["LAnkW X", "RAnkW X"],
          "All": ["LHipW X", "RHipW X", "LKneW X", "RKneW X", "LAnkW X", "RAnkW X"]}


def session_medians(path):
    """{session: {group: median over trials of the per-trial joint-group mean}}"""
    per_trial = defaultdict(dict)
    for row in csv.DictReader(path.open(newline="")):
        if row["model"] != "calib" or row["rpt_r"] in ("", "NaN"):
            continue
        per_trial[(row["session"], int(row["trial_idx"]))][row["joint"]] = float(row["rpt_r"])
    bysess = defaultdict(lambda: defaultdict(list))
    for (sess, _t), jr in per_trial.items():
        for g, js in GROUPS.items():
            vals = [jr[j] for j in js if j in jr]
            if vals:
                bysess[sess][g].append(mean(vals))
    return {s: {g: median(v) for g, v in gv.items()} for s, gv in bysess.items()}


def pooled_trials(path):
    """{group: [one value per held-out trial]} - the cohort convention the Results quote."""
    per_trial = defaultdict(dict)
    for row in csv.DictReader(path.open(newline="")):
        if row["model"] != "calib" or row["rpt_r"] in ("", "NaN"):
            continue
        per_trial[(row["session"], int(row["trial_idx"]))][row["joint"]] = float(row["rpt_r"])
    out = defaultdict(list)
    for jr in per_trial.values():
        for g, js in GROUPS.items():
            vals = [jr[j] for j in js if j in jr]
            if vals:
                out[g].append(mean(vals))
    return out


def main():
    iw, rest = session_medians(IW_SRC), session_medians(REST_SRC)
    iw_tr, rest_tr = pooled_trials(IW_SRC), pooled_trials(REST_SRC)
    sessions = sorted(set(iw) & set(rest))
    assert len(sessions) == 80, f"expected 80 paired sessions, got {len(sessions)}"

    rows = []
    for g in ["Hip", "Knee", "Ankle", "All"]:
        a = [iw[s][g] for s in sessions]
        b = [rest[s][g] for s in sessions]
        d = [x - y for x, y in zip(a, b)]
        w, p = stats.wilcoxon(a, b, alternative="two-sided")
        rows.append(dict(joint_group=g, n_paired_sessions=len(sessions),
                         real_median=f"{median(a):.6f}", null_median=f"{median(b):.6f}",
                         median_paired_difference=f"{median(d):.6f}",
                         n_sessions_real_above_null=sum(1 for v in d if v > 0),
                         wilcoxon_W=f"{w:.0f}", wilcoxon_p=f"{p:.3e}"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with _ec.sd("fig2c_null_paired_test_source_data.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)

    rest_trial, rest_sess = median(rest_tr['All']), median([rest[s]['All'] for s in sessions])
    if _ec.IS_DEFAULT:
        trial_note = ("pooled over all held-out trials: "
                      "imagined "
                      f"walking All {median(iw_tr['All']):.4f} "
                      f"(n={len(iw_tr['All'])} trials), resting input All "
                      f"{rest_trial:.4f} (n={len(rest_tr['All'])}). "
                      "Resting input All "
                      f"(trial {rest_trial:.4f}, session "
                      f"{rest_sess:.4f})"
                      "; the paired test "
                      "beside it stays session-paired because the pairing unit is "
                      "the session")
        rest_note = (" "
                     "whether the correspondence-matrix diagonal differs from "
                     "zero (see fig2e_correspondence_metadata.csv, rest_vs_zero)")
    else:
        t2, s2 = f"{rest_trial:.2f}", f"{rest_sess:.2f}"
        same = (f"The resting value reads {t2} under both conventions" if t2 == s2 else
                f"The resting value reads {t2} pooled over trials but "
                f"{s2} as the median of session medians")
        trial_note = ("pooled over all held-out trials: "
                      f"imagined walking All {median(iw_tr['All']):.4f} (n={len(iw_tr['All'])} "
                      f"trials), resting input All {rest_trial:.4f} (n={len(rest_tr['All'])}). "
                      f"{same} (trial {rest_trial:.4f}, session {rest_sess:.4f}) - quote it with "
                      "its convention; the paired test beside it stays session-paired because the "
                      "pairing unit is the session")
        rest_note = (f"the resting-input condition's cohort median is {rest_sess:+.4f} (median of session "
                     f"medians; {rest_trial:+.4f} pooled over trials); whether the "
                     "correspondence-matrix diagonal differs from zero is recorded in "
                     "fig2e_correspondence_metadata.csv, rest_vs_zero")

    if _ec.IS_DEFAULT:
        build_id_note = "healthy cohort, "
        quoted = ("All row: "
                  "median r 0.01, lower than for "
                  "imagined walking in all 80 sessions (80 of 80 "
                  "same-sign pairs)"
                  "")
        legend_note = ("the legend's '74 of 80 sessions' counts session medians "
                       "above the cohort-level 97.5th-percentile null band drawn "
                       "in the panel; this file's test is session-paired")
    else:
        build_id_note = "healthy cohort, causal complex-demodulation decoder (random 10-fold readout)"
        if _ec.BUILD_ID in _ec.DECODER_BUILDS:
            build_id_note = ("healthy cohort, causal complex-demodulation decoder with a causal 6 Hz low-pass on the "
                        "stage-1 output (random 10-fold readout)")
        ra = rows[-1]
        quoted = ("the Results Fig. 2 paragraph quotes the resting-input median of session medians "
                  "(headline_session_medians_source_data.csv; pooled trial median: cohort_trial_medians) and compares imagined-walking with resting input by a "
                  "trial-level linear mixed-effects model (sup_restinput_trial_lme_metadata.csv); "
                  " (All: median paired "
                  f"difference {float(ra['median_paired_difference']):+.4f}, W = {ra['wilcoxon_W']}, "
                  f"P = {ra['wilcoxon_p']}, {ra['n_sessions_real_above_null']} of "
                  f"{ra['n_paired_sessions']} sessions above their own resting-input median)")
        band = {r["convention"]: r for r in csv.DictReader(
            _ec.sd_in("fig2g_rest_null_band.csv").open(newline="")) if r["panel"] == "All"}["session_median"]
        n_above = sum(1 for s in sessions if iw[s]["All"] > float(band["band_hi_p97p5"]))
        assert n_above == int(band["n_iw_sessions_above_hi"]), (n_above, band)
        legend_note = (f"the legend's '{n_above} of {band['n_sessions']} sessions' (joint average) "
                       "counts session medians above the cohort-level 97.5th-percentile null band "
                       "drawn in the panel (fig2g_rest_null_band.csv); this file's test is "
                       "session-paired")

    meta = [
        ("iw_source", IW_SRC.relative_to(ROOT).as_posix()),
        ("rest_source", REST_SRC.relative_to(ROOT).as_posix()),
        ("aggregation", "model=calib, rpt_r; per-trial mean over the joints of a group; "
                        "per-session median over trials (identical to fig2g_trial_r_export.py)"),
        ("test", "two-sided Wilcoxon signed-rank on the 80 session-median pairs"),
        ("quoted_in_main_text", quoted),
        ("rest_condition_note", rest_note),
    ]
    with _ec.sd("fig2c_null_paired_test_metadata.csv").open("w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["field", "value"]); wr.writerows(meta)

    for r in rows:
        print(f"{r['joint_group']:6s} real {r['real_median']} null {r['null_median']} "
              f"d {r['median_paired_difference']} W {r['wilcoxon_W']} p {r['wilcoxon_p']} "
              f"{r['n_sessions_real_above_null']}/{r['n_paired_sessions']}")


if __name__ == "__main__":
    main()
