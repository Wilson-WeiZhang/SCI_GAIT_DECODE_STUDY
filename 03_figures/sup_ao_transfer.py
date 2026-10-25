#!/usr/bin/env python3
"""Supplementary item

Cross-condition transfer of the decoder between imagined walking with observation (MI+AO) and observation only (AO),
six supplementary participants

Renders one panel or table; it performs no
preprocessing and does not fit the decoder. It fits a linear mixed model
for the statistics reported with that panel or table.
"""

import glob
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import analysis_config as ec
from safe_write import safe_write_csv

ec.require("main")
FOLDS = ec.ROOT / "gait_prep_supplementary/result/main/cross_transfer/folds"
LME_DIR = ec.ROOT / "gait_prep_supplementary/result/main/lme_trialwise"
OUT_SD = ec.sd("sup_ao_transfer_source_data.csv")
OUT_MD = ec.sd("sup_ao_transfer_metadata.csv")
CONDITION = {"miao": "MI+AO", "ao": "AO"}

js = [json.load(open(f)) for f in sorted(glob.glob(str(FOLDS / "*.json")))]
assert len(js) == 360 and all(j["gate_pass"] for j in js if j["feat"] == "all"), "fold reproduction gate"

d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(FOLDS / "*.csv")))], ignore_index=True)
assert (d.groupby(["key", "train_condition", "test_condition", "feat", "trial_idx"]).joint.nunique() == 6).all()
t = (d.groupby(["key", "train_condition", "test_condition", "feat", "kind", "trial_idx"]).rpt_r.mean()
     .rename("r").reset_index())
t = t[t.trial_idx > 10].copy()
t["participant"] = t.key.str.replace("sub", "SUB_", regex=False)

ref = pd.read_csv(LME_DIR / "sc_trials_all.csv")
ref["participant"] = ref.participant.str.extract(r"^(SUB_\d+)")[0]
ref["test_condition"] = ref.cond.map({"MI+AO": "miao", "AO": "ao"})
own = t[(t.kind == "within") & (t.feat == "all")].merge(ref, on=["participant", "test_condition", "trial_idx"], how="outer")
assert len(own) == 240 and own.r.notna().all() and own.rpt_calib.notna().all(), "trial sets differ"
assert (own.r - own.rpt_calib).abs().max() < 1e-6, (own.r - own.rpt_calib).abs().max()


def lme(df):
    diff = df[df.c > 0].r.mean() - df[df.c < 0].r.mean()
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in ("lbfgs", "powell", "cg"):
            try:
                f = smf.mixedlm("r ~ c", df, groups=df["participant"], re_formula="~c").fit(reml=True, method=m)
            except Exception:
                continue
            if f.converged and np.isfinite(f.bse["c"]) and abs(f.params["c"] - diff) < 1e-6 \
                    and (best is None or f.llf > best.llf):
                best = f
        if best is not None:
            return best, "intercept and slope"
        return smf.mixedlm("r ~ c", df, groups=df["participant"]).fit(reml=True), "intercept only (no admissible slope fit)"


rows = []
for feat in ("all", "k1", "k2"):
    for test in ("miao", "ao"):
        other = "ao" if test == "miao" else "miao"
        sub = t[(t.feat == feat) & (t.test_condition == test)].copy()
        sub["c"] = np.where(sub.kind == "cross", 0.5, -0.5)
        med = sub.groupby(["participant", "kind"]).r.median().unstack()
        for p, v in med.iterrows():
            for kind, trained in (("within", test), ("cross", other)):
                rows.append(["participant", feat, CONDITION[test], CONDITION[trained], "", p, 20, round(v[kind], 6),
                             "", "", "", "", "", ""])
        f, spec = lme(sub)
        b, se, pv = float(f.params["c"]), float(f.bse["c"]), float(f.pvalues["c"])
        rows.append(["contrast", feat, CONDITION[test], CONDITION[other], f"transferred - own decoder on {CONDITION[test]} trials", "all6",
                     len(sub), "", round(med["within"].median(), 6), round(med["cross"].median(), 6),
                     int((med["cross"] < med["within"]).sum()), round(b, 6), round(se, 6), f"{pv:.3g}|{spec}"])
    for kind in ("within", "cross"):
        sub = t[(t.feat == feat) & (t.kind == kind)].copy()
        sub["c"] = np.where(sub.test_condition == "miao", 0.5, -0.5)
        med = sub.groupby(["participant", "test_condition"]).r.median().unstack()
        f, spec = lme(sub)
        b, se, pv = float(f.params["c"]), float(f.bse["c"]), float(f.pvalues["c"])
        if feat == "all" and kind == "within":
            lj = json.load(open(LME_DIR / "s5m_trialwise_lme.json"))["all"]["M2_random_slope"]
            assert abs(b - lj["beta_miao_minus_ao"]) < 5e-5, (b, lj["beta_miao_minus_ao"])
        rows.append(["contrast", feat, "MI+AO vs AO", "own condition" if kind == "within" else "other condition",
                     f"MI+AO - AO test trials, {'own' if kind == 'within' else 'transferred'} decoders", "all6",
                     len(sub), "", round(med["ao"].median(), 6), round(med["miao"].median(), 6),
                     int((med["miao"] < med["ao"]).sum()), round(b, 6), round(se, 6), f"{pv:.3g}|{spec}"])

cols = ["row_type", "features", "test_condition", "decoder_trained_on", "contrast", "participant", "n_trials",
        "median_r", "reference_median_of_participant_medians", "compared_median_of_participant_medians",
        "n_participants_compared_lower", "lme_estimate", "lme_se", "lme_p|random_effects"]
safe_write_csv(OUT_SD, cols, rows)
meta = [
    ["analysis", "supplementary decoder transfer between imagined walking with observation (MI+AO) and observation only (AO)"],
    ["build_id", ec.BUILD_ID],
    ["cohort", "supplementary experiment, 6 participants (SUB_01/09/12/23/29/35), 20 scored trials per condition"],
    ["decoder", " supplementary fold models (27 training trials), applied to own-condition fold trials "
                "(own decoder) and to the other condition's same-fold trials (transferred decoder)"],
    ["features", "all = 360 features; k1 / k2 = the 120 features of one harmonic, PCA 30 of the restricted LSO basis"],
    ["scoring", "trial r = mean over six walker joints of calib rpt_r; trial_idx > 10 (cycles 11-30)"],
    ["statistics", "trial-level REML LME r ~ c + (1 + c | participant), c = +-0.5 (compared = +0.5), "
                   "best of lbfgs / powell / cg; random-intercept fallback labelled in lme_p|random_effects"],
    ["columns", "contrast rows: reference_ = own decoder (transfer rows) or AO test trials (MI+AO vs AO rows); "
                "compared_ = transferred decoder or MI+AO test trials; lme_estimate = compared - reference"],
    ["guards", "fold reproduction gates pass; own-decoder trial r = sc_trials_all.csv rpt_calib (240 trials, < 1e-6); "
               "own-decoder MI+AO - AO LME = s5m_trialwise_lme.json M2 0.33513355 within 5e-5"],
    ["inputs", "gait_prep_supplementary/result/main/cross_transfer/folds/"],
    ["script", "03_figures/sup_ao_transfer.py"],
]
safe_write_csv(OUT_MD, ["key", "value"], meta)
pd.set_option("display.width", 250)
print(pd.DataFrame([r for r in rows if r[0] == "contrast"], columns=cols).drop(columns=["row_type", "participant",
                                                                                     "median_r"]).to_string(index=False))
