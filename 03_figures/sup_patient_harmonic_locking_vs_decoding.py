#!/usr/bin/env python3
"""Supplementary item

Patient post-operative sessions: gait-phase locking at each harmonic (k1, k2, k3) against decoding accuracy

Renders one panel or table; it performs no
preprocessing and does not fit the decoder. It fits a linear mixed model
for the statistics reported with that panel or table.
"""

import csv
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

import analysis_config as ec
import fig6_locking_vs_transfer as f6
from safe_write import safe_write_csv

ec.require("main")
OUT_SD = ec.sd("sup_patient_harmonic_locking_vs_decoding_source_data.csv")
OUT_MD = ec.sd("sup_patient_harmonic_locking_vs_decoding_metadata.csv")

dec = {r["session"]: float(r["median"]) for r in csv.DictReader(open(f6.DEC_SD))
       if r["cohort"] == f6.COHORT}
ys = np.array([dec[s] for s in f6.POST_OPERATIVE])
pats = ["P1" if s.startswith("pat01") else "P2" for s in f6.POST_OPERATIVE]

rows = []
for k in (1, 2, 3):
    f6.KH = k
    lock = f6.load_locking()
    xs = np.array([lock[s][0] for s in f6.POST_OPERATIVE])
    df = pd.DataFrame(dict(x=xs, y=ys, patient=pats))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lme = smf.mixedlm("y ~ x", df, groups=df["patient"]).fit(reml=True)
    b, se, p = float(lme.params["x"]), float(lme.bse["x"]), float(lme.pvalues["x"])
    pr, pp = pearsonr(xs, ys)
    perm_p, n_perm = f6.exact_within_patient_perm_p(xs, ys, pats, pr)
    rows.append([k, len(xs), round(b, 6), round(se, 6), round(b / se, 4), f"{p:.3g}",
                 round(pr, 6), f"{pp:.3g}", f"{perm_p:.4g}", n_perm])

sd = pd.DataFrame(rows, columns=["harmonic_k", "n_sessions", "lme_slope", "lme_se", "lme_wald_z", "lme_p",
                                 "pearson_r", "pearson_p", "within_patient_perm_p", "n_perm"])
k2 = sd[sd.harmonic_k == 2].iloc[0]
assert abs(k2.lme_slope - 2.115588) < 5e-5 and abs(k2.lme_se - 0.451658) < 5e-5, k2

safe_write_csv(OUT_SD, list(sd.columns), sd.values.tolist())
meta = [
    ["analysis", "patient post-operative session-level gait-phase locking |c| at k1 / k2 / k3 vs decoding accuracy"],
    ["build_id", ec.BUILD_ID],
    ["cohort", f"{f6.COHORT} (11 post-operative sessions, 2 patients; case-level)"],
    ["x_metric", "session mean of zero-phase c_mi at harmonic_k over IW trials (neural tag main, build_id-invariant)"],
    ["y_metric", "session trial-median decoding r, fig5b_session_box_source_data.csv cohort post_operative"],
    ["statistics", "LME y ~ x, patient random intercept, REML (headline as in Fig. 6g); Pearson r with exact "
                   "within-patient permutation p (86,400 relabellings)"],
    ["guard", "k2 row reproduces Fig. 6g LME slope 2.115588 and s.e. 0.451658 within 5e-5"],
    ["script", "03_figures/sup_patient_harmonic_locking_vs_decoding.py (reuses 03_figures/fig6_locking_vs_transfer.py loaders)"],
]
safe_write_csv(OUT_MD, ["key", "value"], meta)
print(sd.to_string(index=False))
