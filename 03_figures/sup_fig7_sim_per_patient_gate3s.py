"""Supplementary item (Figure 7 companion)

sup_fig7_sim_per_patient_gate3s — the Fig. 7c-e simulation outcomes repeated PER PATIENT on the
3 s usable-duration, window-matched-walker build

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

import csv
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import sup_table_sim_outcomes as T
os.environ.setdefault("GAIT_F7_SIM", "causal_from2s")
import analysis_config as ec
ec.require("main", "main")

SD = os.path.join(ROOT, "04_source_data", "supplementary")
STEM = "sup_fig7_sim_per_patient_gate3s"
GROUPS = [("Pooled", None), ("Patient 1", "pat01"), ("Patient 2", "pat02")]
SCHEME = ("paired difference d = A - B per trial (session x trial_idx), NaN dropped; the test_rule "
          "model of the Fig. 7 panels (LME d ~ 1 + (1 | session), REML, session random intercept), pooled and within each patient; Wald p = p_used in both (Shapiro-Wilk and Wilcoxon reported in the Source Data only); star from p_used")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(name, header, rows):
    p = os.path.join(SD, f"{STEM}_{name}.csv")
    if ec.SCOPED:
        p = str(ec.scoped(p))
    tmp = p + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, p)
    return p


def main():
    tab = T.load()
    tab = tab.assign(x3_m=tab.disp_3s_m)

    med_rows, test_rows = [], []
    for gname, pat in GROUPS:
        g = tab if pat is None else tab[tab.patient == pat]
        A = {c: g[g.condition == c] for c in T.ORDER}
        for key, quant, unit, _ in T.METRICS:
            for c in T.ORDER:
                v = A[c][key].astype(float).dropna()
                med_rows.append([gname, quant, unit, T.DISPLAY[c], len(v),
                                 A[c].session.nunique(), f"{v.median():.6f}",
                                 f"{v.quantile(.25):.6f}", f"{v.quantile(.75):.6f}"])
        mixed = g.patient.nunique() >= 2
        for name, a, b in T.CONTRASTS:
            for key, quant, unit, _ in T.METRICS:
                r = T.lme_paired_test(A[a], A[b], key, mixed=mixed)
                test_rows.append([gname, quant, unit, name,
                                  f"{T.DISPLAY[a]} minus {T.DISPLAY[b]}", r["n"],
                                  r["n_subjects"], r["n_sessions"],
                                  f"{r['mean_difference_lme']:.6f}", f"{r['se_lme']:.6g}",
                                  f"{r['median_difference']:.6f}", r["model"], r["optimizer"],
                                  f"{r['shapiro_p']:.6g}", f"{r['p_lme']:.6g}",
                                  f"{r['p_wilcoxon']:.6g}", r["test_used"], f"{r['p_used']:.6g}",
                                  r["star"], SCHEME])
        for c in T.ORDER:
            r = T.lme_paired_test(A[c], None, "disp_3s_m", mixed=mixed)
            test_rows.append([gname, T.DISP_Q, "m", f"{c}_vs_zero",
                              f"{T.DISPLAY[c]} versus zero", r["n"], r["n_subjects"],
                              r["n_sessions"], f"{r['mean_difference_lme']:.6f}",
                              f"{r['se_lme']:.6g}", f"{r['median_difference']:.6f}", r["model"],
                              r["optimizer"], f"{r['shapiro_p']:.6g}", f"{r['p_lme']:.6g}",
                              f"{r['p_wilcoxon']:.6g}", r["test_used"], f"{r['p_used']:.6g}",
                              r["star"], "one-sample: d is the per-rollout 3.0 s endpoint itself; "
                                         + SCHEME.split("; ", 1)[1]])

    p_sd = write("source_data",
                 ["condition", "patient", "session", "trial_idx", "stage", "condition", "dur_s", "x3_m",
                  "cadence_steps_per_s", "step_length_m", "lr_asymmetry", "dx_m", "n_adv_events",
                  "n_adv_l", "n_adv_r", "fell"],
                 [[T.DISPLAY[r.condition], r.patient, r.session, int(r.trial_idx), r.stage, r.condition,
                   f"{float(r.dur_s):.4f}", f"{float(r.x3_m):.6f}",
                   f"{float(r.cadence_steps_per_s):.6f}",
                   "" if pd.isna(r.step_length_m) else f"{float(r.step_length_m):.6f}",
                   "" if pd.isna(r.lr_asymmetry) else f"{float(r.lr_asymmetry):.6f}",
                   f"{float(r.dx_m):.6f}", int(r.n_adv_events), int(r.n_adv_l), int(r.n_adv_r),
                   int(r.fell)] for r in tab.itertuples(index=False)])
    p_st = write("stats",
                 ["row_type", "group", "metric", "unit", "condition_or_comparison", "comparison_key",
                  "n", "n_subjects", "n_sessions", "median", "q25", "q75", "mean_difference", "se",
                  "median_difference", "model", "optimizer", "shapiro_p", "p_wald", "p_wilcoxon",
                  "test_used", "p_used", "star", "scheme"],
                 [["median", g, m, u, c, "", n, "", ns, med, q1, q3, "", "", "", "", "", "", "", "",
                   "", "", "", "median and interquartile range over rollouts"]
                  for g, m, u, c, n, ns, med, q1, q3 in med_rows] +
                 [["test", g, m, u, lab, key, n, nsub, ns, "", "", "", mdf, se, mdm, mod, opt, sw,
                   pw, pwx, used, pu, star, sch]
                  for g, m, u, key, lab, n, nsub, ns, mdf, se, mdm, mod, opt, sw, pw, pwx, used,
                  pu, star, sch in test_rows])

    cad = T.panel_md("fig7_sim_cadence")
    sess = {g: "; ".join(sorted(set(tab[tab.patient == p].session))) for g, p in GROUPS if p}
    n_by = tab.groupby(["condition", "patient"]).size().unstack(fill_value=0)
    p_md = write("metadata", ["key", "value"], [
        *([list(ec.F7_SIM_MD)] if ec.F7_SIM else []),
        ["producer", "03_figures/sup_fig7_sim_per_patient_gate3s.py (definitions, loader and test "
                     "imported from 03_figures/sup_table_sim_outcomes.py)"],
        ["build_id", cad["build_id"]],
        ["build_id_note", cad["build_id_note"]],
        ["cohort", cad["cohort"]],
        ["usable_duration_gate", cad["usable_duration_gate"]],
        ["walker_window", cad["walker_window"]],
        ["test_rule", cad["test_rule"]],
        ["metric_definitions", " | ".join(f"{q} ({u}): {d}" for q, u, d in T.DEFS[:4])],
        ["sessions_per_patient", "; ".join(f"{g}: {s}" for g, s in sess.items())],
        ["n_rollouts", "; ".join(f"{T.DISPLAY[c]} P1 {int(n_by.loc[c].get('pat01', 0))} / P2 "
                                 f"{int(n_by.loc[c].get('pat02', 0))}" for c in T.ORDER)],
        ["per_patient_model", "Per-patient contrasts fit the same test_rule model to that patient's trials "
                              "(d ~ 1 + (1 | session), REML, Wald p on the intercept), reported as "
                              "test_used 'LME (within patient)' with no Shapiro switch."],
        ["relation_to_table_S4", "The pooled rows here are the same numbers as the pooled rows of "
                                  "Supplementary Table 4 and, through that table's cross-check, "
                                  "the same numbers as the Fig. 7c-e panel Source Data. This file "
                                  "adds no analysis; it is the per-patient cut."],
        ["input_files", "; ".join(f"{os.path.relpath(p, ROOT)} (md5 {md5(p)})"
                                  for p in (T.RHY, T.WRHY, T.TRA, T.WTRA))],
        ["scope", "n = 2 patients, case-level. 'Replicates in each patient' is a within-person "
                  "consistency statement over trials, never a population-level one. MuJoCo "
                  "behavioural outcomes only: no ground reaction force, no contact force, no balance / "
                  "stability / efficacy claim. METRES, never m/s."],
    ])

    for p in (p_sd, p_st, p_md):
        print("wrote", os.path.relpath(p, ROOT))
    print(n_by.to_string())
    for g, m, u, c, n, ns, med, q1, q3 in med_rows:
        print(f"  {g:9s} | {m[:28]:28s} | {c:23s} n={n:3d} s={ns} "
              f"median {float(med):+.3f} [{float(q1):+.3f}, {float(q3):+.3f}]")
    for row in test_rows:
        g, m, _u, key, _lab, n = row[0], row[1], row[2], row[3], row[4], row[5]
        print(f"  {g:9s} | {m[:28]:28s} | {key:24s} n={n:3d} mean {float(row[8]):+.4f} "
              f"{row[16]:28s} p {float(row[17]):.3g} {row[18]}")


if __name__ == "__main__":
    main()
