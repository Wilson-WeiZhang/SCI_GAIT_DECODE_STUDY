#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary item

Session-order, participant and session variance components — numbers-only SI producer

Renders one panel or table; it performs no
preprocessing and does not fit the decoder. It fits a linear mixed model
for the statistics reported with that panel or table.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import analysis_config as ec
ec.require("main", "main")

ROOT = Path(__file__).resolve().parents[1]
SRC = ec.sd("fig2g_trial_r_long.csv")
SDDIR = ROOT / "04_source_data" / "supplementary"
STEM = "sup_session_order_lme"
BUILD_ID_TAG = ec.TAG

EXPECT = {"trials": 4756, "sessions": 80, "participants": 42}
if ec.SCORE_MASK is not None:
    EXPECT = {"trials": 3172, "sessions": 80, "participants": 42}

OPTIMIZER = "lbfgs"
SESS_RE = re.compile(r"^(subj\d+)_sess(\d+)")


def pfmt(p: float) -> str:
    """House style: three decimals, leading zero, floored at p < 0.001."""
    return "p < 0.001" if p < 0.0005 else f"p = {p:.3f}"


def load() -> pd.DataFrame:
    if not SRC.exists():
        raise SystemExit(f"[{STEM}] missing input {SRC} -- run 03_figures/fig2g_trial_r_export.py first")
    df = pd.read_csv(SRC)
    df = df[np.isfinite(df["All"])].copy()
    m = df["session"].str.extract(SESS_RE)
    if m[0].isna().any():
        bad = sorted(df.loc[m[0].isna(), "session"].unique())
        raise SystemExit(f"[{STEM}] unparseable session labels: {bad}")
    df["subject"] = m[0]
    df["session_num"] = m[1].astype(int)
    if set(df["session_num"].unique()) - {1, 2}:
        raise SystemExit(f"[{STEM}] expected session numbers 1/2, got "
                         f"{sorted(df.session_num.unique())}")
    df["session2"] = (df["session_num"] == 2).astype(float)
    df["r"] = df["All"].astype(float)
    got = {"trials": len(df), "sessions": df["session"].nunique(),
           "participants": df["subject"].nunique()}
    if got != EXPECT:
        raise SystemExit(f"[{STEM}] input is not the {BUILD_ID_TAG} cohort: got {got}, "
                         f"expected {EXPECT}.")
    return df


def fit_finite(model, reml: bool):
    """OPTIMIZER first; fall back only when it returns a non-finite log-likelihood."""
    for method in (OPTIMIZER, "bfgs", "cg", "nm"):
        res = model.fit(reml=reml, method=method)
        if np.isfinite(res.llf):
            return res
    raise SystemExit(f"[{STEM}] no optimizer returned a finite log-likelihood")


def fit(df: pd.DataFrame, reml: bool):
    return fit_finite(smf.mixedlm("r ~ session2", df, groups=df["subject"], re_formula="1",
                                  vc_formula={"sess": "0 + C(session)"}), reml)


def nlme_crosscheck(df: pd.DataFrame) -> str:
    """Independent refit in R/nlme (containment df); statsmodels reports a Wald z only."""
    rbin = "/usr/local/bin/R"
    if not Path(rbin).exists():
        return "skipped (R not found)"
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"{STEM}_nlme.csv"
    df[["r", "session2", "subject", "session"]].to_csv(tmp, index=False)
    script = tmp.with_suffix(".R")
    script.write_text(
        'suppressMessages(library(nlme))\n'
        f'd <- read.csv("{tmp}")\n'
        'd$subject <- factor(d$subject); d$session <- factor(d$session)\n'
        'm <- lme(r ~ session2, random = ~1 | subject/session, data = d, method = "REML")\n'
        'tt <- summary(m)$tTable\n'
        'cat(sprintf("NLME beta=%.6f se=%.6f df=%d t=%.3f p=%.4g\\n",\n'
        '            tt["session2","Value"], tt["session2","Std.Error"],\n'
        '            as.integer(tt["session2","DF"]), tt["session2","t-value"],\n'
        '            tt["session2","p-value"]))\n'
    )
    try:
        out = subprocess.run([rbin, "-q", "--no-save", "-f", str(script)],
                             capture_output=True, text=True, timeout=900)
        for line in out.stdout.splitlines():
            if line.strip().startswith("NLME beta="):
                return line.strip()
        return f"failed (no NLME line; rc={out.returncode})"
    except Exception as exc:
        return f"failed ({type(exc).__name__})"
    finally:
        for p in (tmp, script):
            p.unlink(missing_ok=True)


def main() -> None:
    df = load()

    sess = (df.groupby(["subject", "session", "session_num"], as_index=False)
              .agg(n_trials=("r", "size"), sess_mean=("r", "mean")))
    n_subj = sess["subject"].nunique()
    wide = sess.pivot_table(index="subject", columns="session_num", values="sess_mean")
    paired = wide.dropna()
    only_one = sorted(set(sess["subject"]) - set(paired.index))
    anchor = float(sess["sess_mean"].mean())
    n_trials_mean = float(sess["n_trials"].mean())

    mdf = fit(df, reml=True)
    beta = float(mdf.fe_params["session2"])
    se = float(mdf.bse["session2"])
    z = float(mdf.tvalues["session2"])
    p_wald = float(mdf.pvalues["session2"])
    ci_lo, ci_hi = beta - 1.96 * se, beta + 1.96 * se
    intercept = float(mdf.fe_params["Intercept"])

    var_subj = float(mdf.cov_re.iloc[0, 0])
    var_sess = float(mdf.vcomp[0]) if len(mdf.vcomp) else 0.0
    var_res = float(mdf.scale)
    sd_subj, sd_sess, sd_res = (float(np.sqrt(v)) for v in (var_subj, var_sess, var_res))
    icc_trial = var_subj / (var_subj + var_sess + var_res)
    icc_sessmean = var_subj / (var_subj + var_sess)

    cov = mdf.cov_params()
    se_m1 = float(np.sqrt(cov.loc["Intercept", "Intercept"]))
    se_m2 = float(np.sqrt(cov.loc["Intercept", "Intercept"] + cov.loc["session2", "session2"]
                          + 2 * cov.loc["Intercept", "session2"]))

    blup = {g: float(v.get("Group", v.iloc[0])) for g, v in mdf.random_effects.items()}
    blup_s = pd.Series(blup).sort_values()

    d1, d2 = paired[1].to_numpy(), paired[2].to_numpy()
    dd = d2 - d1
    n_pair = len(paired)
    t_stat, t_p = stats.ttest_rel(d2, d1)
    sd_d = float(dd.std(ddof=1))
    se_d = sd_d / np.sqrt(n_pair)
    tost = {b: float(max(stats.t.sf((dd.mean() + b) / se_d, n_pair - 1),
                         stats.t.cdf((dd.mean() - b) / se_d, n_pair - 1)))
            for b in (0.10, 0.05)}
    mde = float((stats.t.ppf(.975, n_pair - 1) + stats.t.ppf(.80, n_pair - 1)) * se_d)
    r_tt, p_tt = stats.pearsonr(d1, d2)
    rho_tt, _ = stats.spearmanr(d1, d2)
    a2 = np.c_[d1, d2]
    ms_b = a2.mean(1).var(ddof=1) * 2
    ms_w = ((a2 - a2.mean(1, keepdims=True)) ** 2).sum() / n_pair
    icc11 = float((ms_b - ms_w) / (ms_b + ms_w))

    full = fit(df, reml=False)
    red = fit_finite(smf.mixedlm("r ~ session2", df, groups=df["session"], re_formula="1"), False)
    lrt = float(2 * (full.llf - red.llf))
    lrt_p = float(0.5 * stats.chi2.sf(lrt, 1))

    subj_mean = df.groupby("subject")["r"].mean()
    subj_idx = subj_mean.index.str.extract(r"subj(\d+)")[0].astype(int).to_numpy()
    rho_enrol, p_enrol = stats.spearmanr(subj_idx, subj_mean.to_numpy())
    half1 = float(subj_mean.to_numpy()[subj_idx <= 21].mean())
    half2 = float(subj_mean.to_numpy()[subj_idx > 21].mean())

    nlme_line = nlme_crosscheck(df)

    SDDIR.mkdir(parents=True, exist_ok=True)
    with ec.sd(f"{STEM}_source_data.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "subject", "session", "session_num", "n_trials",
                    "value", "se", "ci_lo", "ci_hi", "term"])
        for _, r_ in sess.sort_values(["subject", "session_num"]).iterrows():
            w.writerow(["session_point", r_.subject, r_.session, int(r_.session_num),
                        int(r_.n_trials), f"{r_.sess_mean:.6f}", "", "", "",
                        "mean r_All over trials"])
        for subj, v in blup_s.items():
            w.writerow(["subject_blup", subj, "", "", "", f"{v:.6f}", "", "", "",
                        "LME random intercept (BLUP)"])
        for val, s_, lo, hi, term in [
            (intercept, se_m1, intercept - 1.96 * se_m1, intercept + 1.96 * se_m1,
             "Intercept = marginal mean, session 1"),
            (beta, se, ci_lo, ci_hi, "session2 (fixed effect, session 2 vs 1)"),
            (intercept + beta, se_m2, intercept + beta - 1.96 * se_m2,
             intercept + beta + 1.96 * se_m2, "marginal mean, session 2 (full cov matrix)"),
        ]:
            w.writerow(["model_term", "", "", "", "", f"{val:.6f}", f"{s_:.6f}",
                        f"{lo:.6f}", f"{hi:.6f}", term])
        for term, val in [("Var(subject)", var_subj), ("Var(session|subject)", var_sess),
                          ("Var(residual/trial)", var_res), ("SD(subject)", sd_subj),
                          ("SD(session|subject)", sd_sess), ("SD(residual/trial)", sd_res),
                          ("ICC(subject), session-mean level", icc_sessmean),
                          ("ICC(subject), single-trial level", icc_trial)]:
            w.writerow(["variance_component", "", "", "", "", f"{val:.6f}", "", "", "", term])
        for term, val in [
            ("z (session2, Wald)", z), ("p (session2, Wald)", p_wald),
            ("paired t (n=%d pairs)" % n_pair, float(t_stat)), ("paired p", float(t_p)),
            ("paired mean diff (sess2-sess1)", float(dd.mean())),
            ("paired SD of diff", sd_d),
            ("TOST p, equivalence bound +-0.10 r", tost[0.10]),
            ("TOST p, equivalence bound +-0.05 r", tost[0.05]),
            ("minimum detectable effect, 80% power", mde),
            ("test-retest Pearson r (session means, n=%d)" % n_pair, float(r_tt)),
            ("test-retest p", float(p_tt)),
            ("test-retest Spearman rho", float(rho_tt)),
            ("ICC(1,1), one-way, model-free", icc11),
            ("LRT chi2(1) for participant random effect", lrt),
            ("LRT p, 0.5*chi2(1) boundary-corrected", lrt_p),
            ("mean trials per session", n_trials_mean),
            ("enrolment-order Spearman rho (participant index vs mean r)", float(rho_enrol)),
            ("enrolment-order p", float(p_enrol)),
            ("mean r, subj01-21", half1), ("mean r, subj22-42", half2),
            ("cohort session-mean r_All (anchor)", anchor),
        ]:
            w.writerow(["statistic", "", "", "", "", f"{val:.6g}", "", "", "", term])

    meta = [
        ("renderer", "03_figures/sup_session_order_lme.py"),
        ("paper_location", "Main Results, session-order sentence after the Fig. 2 results "
                           "; Supplementary Note 10, paragraph "
                           "'Session-order, participant and session variance components' "
                           "(method only, no values). "
                           "TEXT ONLY - no figure, no panel, no S-number."),
        ("input_source", SRC.relative_to(ROOT).as_posix()),
        ("outcome", "per-trial calibrated r, mean over 6 joints (rpt_r, model=calib)"),
        ("model", "r ~ 1 + session2 + (1|subject) + (1|subject:session)"),
        ("estimation", "REML for the fixed effect and the variance components; both models "
                       "refitted by ML for the likelihood-ratio test"),
        ("engine", "statsmodels MixedLM (Wald z; no Satterthwaite/KR denominator df)"),
        ("optimizer", OPTIMIZER),
        ("nlme_crosscheck", nlme_line),
        ("robustness_check", "paired t-test on per-session means, complete pairs only"),
        ("n_trials", len(df)), ("n_sessions", len(sess)), ("n_participants", n_subj),
        ("n_participants_paired", n_pair),
        ("participants_single_session", "|".join(only_one)),
        ("mean_trials_per_session", f"{n_trials_mean:.2f}"),
        ("reported_beta_session2", f"{beta:.6f}"),
        ("reported_beta_se", f"{se:.6f}"),
        ("reported_beta_ci95", f"[{ci_lo:.6f}, {ci_hi:.6f}]"),
        ("reported_beta_wald_z", f"{z:.6f}"),
        ("reported_beta_wald_p", f"{p_wald:.6f}"),
        ("reported_lrt_chi2_1df", f"{lrt:.6f}"),
        ("reported_lrt_p_boundary_corrected", f"{lrt_p:.6g}"),
        ("reported_icc_session_mean_level", f"{icc_sessmean:.6f}"),
        ("reported_icc_single_trial_level", f"{icc_trial:.6f}"),
        ("reported_icc11_model_free", f"{icc11:.6f}"),
        ("reported_test_retest_pearson_r", f"{r_tt:.6f}"),
        ("reported_test_retest_p", f"{p_tt:.6g}"),
        ("reported_n_pairs", n_pair),
        ("reported_tost_p_bound_0.10", f"{tost[0.10]:.6g}"),
        ("reported_tost_p_bound_0.05", f"{tost[0.05]:.6g}"),
        ("note_sentence_as_printed",
         f"beta = {beta:.3f}, 95% CI {ci_lo:.3f} to {ci_hi:.3f}, {pfmt(p_wald)}; "
         f"participant variance component likelihood-ratio {pfmt(lrt_p)}; "
         f"two session means correlate at r = {r_tt:.2f} ({pfmt(p_tt)}, n = {n_pair} pairs)"),
        ("source_script", "03_figures/sup_session_order_lme.py"),
    ]
    with ec.sd(f"{STEM}_metadata.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(meta)

    print(f"[{STEM}] trials={len(df)} sessions={len(sess)} participants={n_subj} "
          f"(both sessions: {n_pair}; single: {only_one})  build_id {BUILD_ID_TAG}")
    print(f"[{STEM}] anchor session-mean r_All = {anchor:.6f}")
    print(f"[{STEM}] session2: beta={beta:+.6f} se={se:.6f} 95%CI[{ci_lo:+.6f},{ci_hi:+.6f}] "
          f"z={z:.3f} {pfmt(p_wald)} (raw {p_wald:.6f})")
    print(f"[{STEM}] paired t (n={n_pair}): mean diff={dd.mean():+.6f} t={t_stat:.3f} "
          f"{pfmt(float(t_p))}")
    print(f"[{STEM}] {nlme_line}")
    print(f"[{STEM}] SD participant={sd_subj:.4f} session|participant={sd_sess:.4f} "
          f"trial={sd_res:.4f}")
    print(f"[{STEM}] ICC session-mean level={icc_sessmean:.4f} single-trial={icc_trial:.4f} "
          f"ICC(1,1) model-free={icc11:.4f}")
    print(f"[{STEM}] LRT chi2(1)={lrt:.4f} {pfmt(lrt_p)} (raw {lrt_p:.6g}, 0.5*chi2 "
          f"boundary-corrected)")
    print(f"[{STEM}] test-retest r={r_tt:.6f} ({pfmt(float(p_tt))}, n={n_pair} pairs)")
    print(f"[{STEM}] TOST +-0.10 {pfmt(tost[0.10])} | +-0.05 {pfmt(tost[0.05])} | "
          f"MDE={mde:.4f}")
    print(f"[{STEM}] wrote {ec.sd(STEM).parent.relative_to(ROOT).as_posix()}/{STEM}_{{source_data,metadata}}.csv "
          f"(TEXT-ONLY: no figure)")


if __name__ == "__main__":
    main()
