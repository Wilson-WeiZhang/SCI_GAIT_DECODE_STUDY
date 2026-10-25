#!/usr/bin/env python3
"""Supplementary item

Healthy-cohort sensitivity: drop sessions with more than 10 interpolated channels

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import analysis_config as ec
from safe_write import safe_write_csv

ec.require("main")
ROOT = Path(__file__).resolve().parents[1]
OUT_SD = ec.sd("sup_interp_channel_sensitivity_source_data.csv")
OUT_MD = ec.sd("sup_interp_channel_sensitivity_metadata.csv")
THRESH = 10


def key(qc_session):
    m = re.fullmatch(r"SUB_(\d+)_sess(\d+)", qc_session)
    return f"subj{m.group(1)}_sess{m.group(2)}"


qc = pd.read_csv(ec.sd("sup_prep_qc_table_source_data.csv"))
qc = qc[(qc.row_type == "session") & qc.cohort.str.startswith("healthy")].copy()
assert len(qc) == 80, len(qc)
qc["skey"] = qc.session.map(key)
excluded = sorted(qc.loc[qc.n_bad_ch > THRESH, "skey"])
assert len(excluded) == 7, excluded

tr = pd.read_csv(ec.sd("fig2g_trial_r_long.csv"))
tr["skey"] = tr.session.str.replace(r"_[a-z]+$", "", regex=True)
sess_med = tr.groupby("skey").All.median()
assert len(sess_med) == 80 and len(tr) == 3172, (len(sess_med), len(tr))

m = pd.read_csv(ec.sd("fig3_master_source_data.csv"))
resp = pd.read_csv(ec.sd("fig3_def_resnonres_source_data.csv"))[["session", "resp"]]
m = m.merge(resp, on="session", how="left")
m["resp"] = m.resp.fillna(False).astype(bool)
assert len(m) == 80


def bh(ps):
    ps = np.array(ps)
    o = np.argsort(ps)
    r = np.empty(len(ps), int)
    r[o] = np.arange(1, len(ps) + 1)
    return np.minimum(1.0, ps * len(ps) / r)


def boot_rho(x, y, rng, n=1000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r0, p = spearmanr(x, y)
    i = np.arange(len(x))
    rs = np.array([spearmanr(x[s], y[s])[0] for s in (rng.choice(i, len(i), True) for _ in range(n))])
    return r0, np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5), p, len(x)


rows = []
for label, keep in (("all80", lambda s: True), ("le10_interp", lambda s: s not in excluded)):
    med = sess_med[[s for s in sess_med.index if keep(s)]]
    mm = m[m.session.map(keep)]
    n_part = mm.session.str.extract(r"subj(\d+)_")[0].nunique()
    rows += [
        [label, "n_sessions", len(med), "", "", "", ""],
        [label, "n_participants", n_part, "", "", "", ""],
        [label, "n_trials", int(tr.skey.map(keep).sum()), "", "", "", ""],
        [label, "iw_median_of_session_medians", round(med.median(), 6),
         round(med.quantile(0.25), 6), round(med.quantile(0.75), 6), "", ""],
        [label, "n_responders", int(mm.resp.sum()), "", "", "", ""],
    ]
    rng = np.random.default_rng(0)
    fam = [(f"rho_excess_k{k}_vs_r",) + boot_rho(mm[f"excess_k{k}"], mm.r_reference, rng) for k in (1, 2, 3)]
    for _k in (1, 2, 3):
        boot_rho(mm[f"rest_k{_k}"], mm.r_reference, rng)
    fam.append(("rho_mubeta_erd_vs_r",) + boot_rho(mm.mubeta_erd_mean, mm.r_reference, rng))
    q = list(bh([f[4] for f in fam[:3]])) + [float("nan")]
    for f, qq in zip(fam, q):
        rows.append([label, f[0], round(f[1], 6), round(f[2], 6), round(f[3], 6), f"{f[4]:.3g}", f"{qq:.3g}"])

sd = pd.DataFrame(rows, columns=["set", "metric", "value", "ci_or_q25", "ci_or_q75", "p", "p_bh_fdr"])
a = sd[(sd.set == "all80") & (sd.metric == "iw_median_of_session_medians")].value.iloc[0]
assert abs(a - 0.465586) < 5e-5, a
k1 = sd[(sd.set == "all80") & (sd.metric == "rho_excess_k1_vs_r")].value.iloc[0]
assert abs(k1 - 0.832841) < 5e-5, k1

safe_write_csv(OUT_SD, list(sd.columns), sd.values.tolist())
meta = [
    ["analysis", "healthy imagined-walking sensitivity to excluding sessions with > 10 interpolated channels"],
    ["build_id", ec.BUILD_ID],
    ["cohort", "healthy cohort (80 sessions / 42 participants), scored trials only (3,172)"],
    ["excluded_sessions", "|".join(excluded)],
    ["exclusion_rule", f"n_bad_ch > {THRESH} in sup_prep_qc_table_source_data.csv (causal-prep calibration decomposition; "
                       "the count Supplementary Table 6 prints)"],
    ["headline_def", "median over sessions of each session's median trial r (six joints averaged per trial); IQR 25th-75th"],
    ["fig3c_def", "Spearman rho, excess_k1..3 and mu/beta ERD vs r_reference (session mean of rpt_calib cells); "
                  "1,000-resample session bootstrap seed 0; BH-FDR over k1-k3, ERD uncorrected (= fig3_d_forest_stats.csv q_bh_fdr)"],
    ["guards", "all80 reproduces headline 0.465586 and Fig. 3c k1 rho 0.832841 within 5e-5"],
    ["script", "03_figures/sup_interp_channel_sensitivity.py"],
    ["paper_location", "Supplementary Note 9, sentence after the Supplementary Table 6 census"],
]
safe_write_csv(OUT_MD, ["key", "value"], meta)
print(sd.to_string(index=False))
