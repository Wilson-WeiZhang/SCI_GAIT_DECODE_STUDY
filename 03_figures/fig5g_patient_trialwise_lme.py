#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 5 — Decoding imagined walking after spinal cord injury

Patient trial-wise brain-behaviour LME vs healthy reference

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and does not fit the decoder. It fits a linear mixed model
for the statistics reported with that panel or table.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

import analysis_config as ec

plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.linewidth": 2.2,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic"})

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "decoder_output"
PC = RES / "patient_pertrial_c.csv"
PD_ = ec.path("p_trial_r_long") if ec.IS_DEFAULT else ec.path("p_trial_r_long")
HC = RES / "persession_pertrial_c_main.csv"
HD = ec.path("h_iw_trial_r_long")
SD = ROOT / "04_source_data" / "fig5"
OUT = ec.scoped(ROOT / "figures_out" / "fig5" / "fig5g_patient_trialwise_lme_forest.png")

BUILD_ID = ec.TAG
POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
            "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
            "pat02_sess09"]
PAT_SIDECARS = {
    "pat02_sess08": (RES / "patient_pertrial_c_sess08.csv",
                     ec.path("p_trial_r_long_sess08") if ec.IS_DEFAULT else None),
    "pat02_sess09": (RES / "patient_pertrial_c_sess09.csv",
                     ec.path("p_trial_r_long_sess09") if ec.IS_DEFAULT else None),
}
for _p in (PD_, HD) + tuple(d for _, d in PAT_SIDECARS.values() if d is not None):
    ec.check_tag(_p, "decoder", "decode-r input")
for _p in (PC, HC) + tuple(c for c, _ in PAT_SIDECARS.values()):
    ec.check_tag(_p, "neural", "|c| input")

BLUE, GREY = "#0072B2", "#9aa0a6"
COH = {"healthy": ("#7a7a7a", "Healthy"), "patient": ("#962014", "Patient")}
KDISP = {1: r"$\mathrm{k}_1$" + "\n(stride)", 2: r"$\mathrm{k}_2$" + "\n(step)", 3: r"$\mathrm{k}_3$"}


def _concat_sidecars(main_path, which):
    """Read a main table plus every sidecar's `which` (0 = |c|, 1 = trial r) table."""
    parts = [pd.read_csv(main_path)]
    seen = set(parts[0].session.unique())
    for sess, paths in PAT_SIDECARS.items():
        d = pd.read_csv(paths[which])
        ids = set(d.session.unique())
        if ids != {sess}:
            raise SystemExit(f"{paths[which].name} must hold exactly {sess}, got: {sorted(ids)}")
        if ids & seen:
            raise SystemExit(f"{sess} present in BOTH {Path(main_path).name} and its sidecar "
                             f"- would double-weight")
        seen |= ids
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def alljoint_r(dpath, sessions=None, sidecars=False):
    d = _concat_sidecars(dpath, 1) if sidecars and ec.IS_DEFAULT else pd.read_csv(dpath)
    dc = d[d.model == "calib"].copy()
    if sessions is not None:
        missing = [s for s in sessions if s not in set(dc.session)]
        if missing:
            raise SystemExit(f"cohort sessions absent from {Path(dpath).name}: {missing}")
        dc = dc[dc.session.isin(sessions)]
    return (dc.dropna(subset=["rpt_r"])
              .groupby(["session", "trial_idx"], as_index=False)
              .rpt_r.mean()
              .rename(columns={"rpt_r": "decode_r"}))


def run_cohort(cpath, dpath, subj_re, groups_by, sessions=None, sidecars=False):
    c = _concat_sidecars(cpath, 0) if sidecars else pd.read_csv(cpath)
    if sessions is not None:
        missing = [s for s in sessions if s not in set(c.session)]
        if missing:
            raise SystemExit(f"cohort sessions absent from {Path(cpath).name}: {missing}")
        c = c[c.session.isin(sessions)]
    r = alljoint_r(dpath, sessions, sidecars=sidecars)
    res = {}
    for k in (1, 2, 3):
        ch = ec.filter_trials(c[c.harmonic_k == k], "IW").copy()
        df = ch.merge(r, on=["session", "trial_idx"], how="inner")
        if not len(df) == len(ch) == len(r):
            raise SystemExit(f"trial_idx join |c| x r is not 1:1 for k{k}: |c| {len(ch)}, r {len(r)}, "
                             f"joined {len(df)} ({Path(cpath).name} x {Path(dpath).name})")
        df["subject"] = df.session.map(lambda s: re.match(subj_re, s).group(1))
        x = df.copy()
        x["cval"] = x.c_mi.values
        smean = x.groupby("session").cval.transform("mean")
        x["within_c"] = x.cval - smean
        x["between_c"] = smean
        if groups_by == "subject":
            m = smf.mixedlm("decode_r ~ within_c + between_c", x,
                            groups=x["subject"], re_formula="~1",
                            vc_formula={"session": "0 + C(session)"})
        else:
            m = smf.mixedlm("decode_r ~ within_c + between_c", x, groups=x["session"])
        fit = m.fit(reml=True, method="lbfgs")
        res[k] = dict(within_b=float(fit.params["within_c"]),
                      within_se=float(fit.bse["within_c"]),
                      within_p=float(fit.pvalues["within_c"]),
                      between_b=float(fit.params["between_c"]),
                      between_p=float(fit.pvalues["between_c"]),
                      n_trial=int(len(df)), n_sess=int(df.session.nunique()),
                      n_subj=int(df.subject.nunique()))
    return res


healthy = run_cohort(HC, HD, r"(subj\d+)", "subject")
patient = run_cohort(PC, PD_, r"(pat\d+)", "session", sessions=POST_OPERATIVE, sidecars=True)

print(f"[build_id] {BUILD_ID}  patient scope = post_operative ({len(POST_OPERATIVE)} sessions)")
print(f"[N] healthy: {healthy[1]['n_subj']} subj / {healthy[1]['n_sess']} sess / "
      f"{healthy[1]['n_trial']} trial   patient: {patient[1]['n_subj']} subj / "
      f"{patient[1]['n_sess']} sess / {patient[1]['n_trial']} trial")
print("\n=== within-session slope (raw trial |c| -> decode r) ===")
print(f"{'k':<3}{'cohort':<9}{'within_b':>10}{'within_se':>11}{'within_p':>12}")
for k in (1, 2, 3):
    for nm, res in [("healthy", healthy), ("patient", patient)]:
        o = res[k]
        print(f"{k:<3}{nm:<9}{o['within_b']:>10.4f}{o['within_se']:>11.4f}{o['within_p']:>12.2e}")

rows = []
for k in (1, 2, 3):
    for nm, res in [("healthy", healthy), ("patient", patient)]:
        o = res[k]
        rows.append(dict(
            harmonic_k=k, cohort=nm, version="raw",
            within_slope=o["within_b"], within_se=o["within_se"], within_p=o["within_p"],
            between_slope=o["between_b"], between_p=o["between_p"],
            n_trials=o["n_trial"], n_sessions=o["n_sess"], n_subjects=o["n_subj"]))
sd = pd.DataFrame(rows)
SD.mkdir(parents=True, exist_ok=True)
sd.to_csv(ec.sd("fig5g_patient_trialwise_lme_source_data.csv"), index=False)

meta = [
    ("analysis", "patient trial-level brain-behaviour LME (Mundlak within/between), RAW |c|; healthy reference"),
    ("patient_c_input", PC.relative_to(ROOT).as_posix()),
    ("patient_r_input", PD_.relative_to(ROOT).as_posix()),
    ("patient_sidecars", "; ".join(f"{sess}: {c.relative_to(ROOT).as_posix()} + "
                                   f"{d.relative_to(ROOT).as_posix()}" if d is not None else
                                   f"{sess}: {c.relative_to(ROOT).as_posix()} (|c| only; r in patient_r_input)"
                                   for sess, (c, d) in PAT_SIDECARS.items())),
    ("healthy_c_input", HC.relative_to(ROOT).as_posix()),
    ("healthy_r_input", HD.relative_to(ROOT).as_posix()),
    ("predictor", "c_mi = per-trial gait-phase-locking magnitude (k1/k2/k3); RAW (no de-chance)"),
    ("outcome", "per-trial decode r (model=='calib', all-joint mean)"),
    ("patient_model", "decode_r ~ within_c + between_c; groups=session (n=2 subjects -> session RE); post_operative"),
    ("post_operative_set", "pat01 sess04-09 + pat02 sess03/05/06/08/09 (11 sessions)"),
    ("healthy_N", f"{healthy[1]['n_subj']} subj / {healthy[1]['n_sess']} sess / "
                  f"{healthy[1]['n_trial']} trial per harmonic"),
    ("patient_N", f"{patient[1]['n_subj']} patients / {patient[1]['n_sess']} sess / "
                  f"{patient[1]['n_trial']} trial per harmonic"),
    ("within_b_healthy", "; ".join(f"k{k} b={healthy[k]['within_b']:.4f}" for k in (1, 2, 3))),
    ("within_p_healthy", "; ".join(f"k{k} p={healthy[k]['within_p']:.2e}" for k in (1, 2, 3))),
    ("within_b_patient", "; ".join(f"k{k} b={patient[k]['within_b']:.4f}" for k in (1, 2, 3))),
    ("within_p_patient", "; ".join(f"k{k} p={patient[k]['within_p']:.3f}" for k in (1, 2, 3))),
    ("marker_semantics", f"DATA-DRIVEN from within_p: filled = p<.05, open = n.s.; "
                         f"healthy {COH['healthy'][0]}, patient {COH['patient'][0]}"),
    ("scope", "n=2 SCI patients, case-level, post-operative, descriptive; NOT a population claim"),
    ("paper_location", "No panel of its own: input table of Fig. 6g (the patient within-session k2 "
                       "slope read by 03_figures/fig6_locking_vs_transfer.py)"),
    ("source_script", "03_figures/fig5g_patient_trialwise_lme.py"),
]
pd.DataFrame(meta, columns=["key", "value"]).to_csv(
    ec.sd("fig5g_patient_trialwise_lme_metadata.csv"), index=False)

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.axvline(0, color=GREY, lw=1.4, ls=(0, (4, 3)), zorder=0)

yoff = {"healthy": 0.20, "patient": -0.20}
y_of_k = {1: 2, 2: 1, 3: 0}
for k in (1, 2, 3):
    for nm, res in [("healthy", healthy), ("patient", patient)]:
        o = res[k]
        b, se, p = o["within_b"], o["within_se"], o["within_p"]
        lo, hi = b - 1.96 * se, b + 1.96 * se
        y = y_of_k[k] + yoff[nm]
        sig = p < 0.05
        col = BLUE if sig else GREY
        ax.plot([lo, hi], [y, y], color=col, lw=2.6, solid_capstyle="round", zorder=3)
        ax.plot(b, y, marker=("o" if nm == "healthy" else "s"), ms=10,
                mfc=(col if sig else "white"), mec=col, mew=2.0, lw=0, zorder=5)

ax.set_yticks([2, 1, 0])
ax.set_yticklabels([KDISP[1], KDISP[2], KDISP[3]], fontsize=15)
ax.set_ylim(-0.75, 2.75)
ax.set_xlabel(r"Within-session slope: trial $|c|$ $\rightarrow$ decode $r$  ($\pm$95% CI)",
              fontsize=14, labelpad=6)
ax.tick_params(axis="x", direction="out", length=5, width=2.2, labelsize=13)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

leg = [
    Line2D([0], [0], marker="o", color="none", mfc=BLUE, mec=BLUE, mew=2.0, ms=10,
           label=f"Healthy (n={healthy[1]['n_subj']})"),
    Line2D([0], [0], marker="s", color="none", mfc=GREY, mec=GREY, mew=2.0, ms=10,
           label="Patient (n=2)"),
    Line2D([0], [0], marker="o", color="none", mfc="white", mec=GREY, mew=2.0, ms=10,
           label="n.s. (open)"),
]
ax.legend(handles=leg, frameon=False, fontsize=11, loc="lower center",
          bbox_to_anchor=(0.5, -0.03), ncol=3, handletextpad=0.6, columnspacing=1.4,
          handlelength=1.2)

fig.subplots_adjust(left=0.155, right=0.985, top=0.985, bottom=0.185)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, facecolor="white")
plt.close(fig)

print(f"\n[wrote] {SD / 'fig5g_patient_trialwise_lme_source_data.csv'}")
print(f"[wrote] {SD / 'fig5g_patient_trialwise_lme_metadata.csv'}")
print(f"[wrote] {OUT}")
