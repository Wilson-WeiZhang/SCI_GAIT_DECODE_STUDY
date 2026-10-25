#!/usr/bin/env python3
"""Numbers reported in the text (Abstract, Results, Supplementary Notes)

Statistics for the Results section

Reads:
  04_source_data/fig7/fig7_sim_trajectories_source_data.csv
  04_source_data/fig7/fig7_sim_cadence_stats.csv
  04_source_data/fig7/fig7_sim_cadence_source_data.csv
  04_source_data/fig7/fig7_sim_step_symmetry_stats.csv
  04_source_data/fig7/fig7_sim_step_symmetry_source_data.csv

Computes numbers quoted in the text from the inputs above; it performs no preprocessing and
does not fit the decoder. It fits a linear mixed model.
"""

import os
import re
import re as _re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

os.environ.setdefault("GAIT_F7_SIM", "causal_from2s")
import analysis_config as ec

SD = Path(__file__).resolve().parent / "source_data"
if ec.BUILD_ID == "zero_phase_prep" and ec.F7_SIM != "causal_from2s":
    raise SystemExit("results_statistics: this build needs the Fig. 7 simulation; run with GAIT_F7_SIM=causal_from2s")
OUT = ec.sd("results_statistics_source_data.csv")
OUT_META = ec.sd("results_statistics_metadata.csv")
B = 10000
SEED = 20260930

ROWS = []
CMP = []


def meta(name):
    d = pd.read_csv(ec.sd_in(name), dtype=str)
    return dict(zip(d.iloc[:, 0], d.iloc[:, 1]))


def median_iqr(text):
    """'0.4055 [0.2406, 0.5730]' -> (value, lo, hi) as floats plus the value token as printed."""
    m = re.match(r"\s*(-?[0-9.]+)\s*\[\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\]", text)
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(1)


def add(item, para, fig, quantity, value, ci=(np.nan, np.nan), n=np.nan, test="",
        side="", family="", src="", note=""):
    ROWS.append(dict(item_id=item, paragraph=para, figure=fig, quantity=quantity,
                     value=value, ci_lo=ci[0], ci_hi=ci[1], n=n, test=test,
                     sidedness=side, family=family, source_file=src, note=note))


def cmp(item, q, reported, recomputed):
    CMP.append((item, q, reported, recomputed))


def bh(p):
    return multipletests(np.asarray(p, float), alpha=0.05, method="fdr_bh")[1]


def fisher_ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.959964 * se)), float(np.tanh(z + 1.959964 * se))


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    rk = stats.rankdata(np.abs(d))
    tp, tm = rk[d > 0].sum(), rk[d < 0].sum()
    return (tp - tm) / (tp + tm), tp, tm


def boot(d, fn, rng):
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    v = np.array([fn(d[i]) for i in idx])
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_block(item, para, fig, label, d, family, src, rng, unit, side="two-sided"):
    """Wilcoxon W, p, rank-biserial (+boot CI), median difference (+boot CI) for paired d."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    w = stats.wilcoxon(d)
    r, tp, tm = rank_biserial(d)
    add(item, para, fig, f"{label}: Wilcoxon W", float(w.statistic), n=len(d),
        test="Wilcoxon signed-rank (scipy default method)", side=side, family=family, src=src,
        note=f"derived from per-{unit} differences; T+={tp:g}, T-={tm:g}")
    add(item, para, fig, f"{label}: p", float(w.pvalue), n=len(d),
        test="Wilcoxon signed-rank", side=side, family=family, src=src, note="derived")
    add(item, para, fig, f"{label}: rank-biserial r", float(r),
        boot(d, lambda x: rank_biserial(x)[0], rng), n=len(d),
        test="matched-pairs rank-biserial", side="", family=family, src=src,
        note=f"derived; CI = percentile bootstrap over {unit}s, B={B}, seed {SEED}"
             + ("; degenerate CI because every difference has the same sign" if min(tp, tm) == 0 else ""))
    add(item, para, fig, f"{label}: median difference", float(np.median(d)),
        boot(d, np.median, rng), n=len(d), test="median of paired differences",
        family=family, src=src, note=f"derived; CI = percentile bootstrap over {unit}s, B={B}, seed {SEED}")
    return w, r


def exact_signrank_counts(n):
    c = np.zeros(n * (n + 1) // 2 + 1)
    c[0] = 1
    for k in range(1, n + 1):
        c[k:] = c[k:] + c[:-k].copy()
    return c


def invert_exact_W(p_stored, n, decimals):
    c = exact_signrank_counts(n)
    cdf = np.cumsum(c) / c.sum()
    hits = [w for w in range(len(c) // 2 + 1)
            if round(min(1.0, 2 * cdf[w]), decimals) == round(p_stored, decimals)]
    return hits


def main():
    rng = np.random.default_rng(SEED)

    f = "sup_session_order_lme_metadata.csv"
    m = meta(f)
    r, n = float(m["reported_test_retest_pearson_r"]), int(m["reported_n_pairs"])
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    p_t = 2 * stats.t.sf(abs(t), n - 2)
    add("1", "", "Fig. 2 text", "test-retest Pearson r (session 1 vs session 2 mean r)", r,
        fisher_ci(r, n), n, "Pearson correlation", "two-sided", "", f,
        "copied r; CI derived = Fisher z, 1.96/sqrt(n-3)")
    add("1", "", "Fig. 2 text", "test-retest t statistic", float(t), n=n,
        test=f"t({n - 2}) for Pearson r", side="two-sided", src=f, note="derived t = r*sqrt((n-2)/(1-r^2))")
    add("1", "", "Fig. 2 text", "test-retest p", float(m["reported_test_retest_p"]), n=n,
        test=f"t({n - 2})", side="two-sided", src=f, note="copied")
    cmp("1", "test-retest p from t(36)", float(m["reported_test_retest_p"]), p_t)
    add("1", "", "Fig. 2 text", "participant variance LRT chi2", float(m["reported_lrt_chi2_1df"]),
        n=int(m["n_participants"]), test="likelihood-ratio chi2(1), ML refit", side="boundary-corrected",
        src=f, note="copied")
    add("1", "", "Fig. 2 text", "participant variance LRT p", float(m["reported_lrt_p_boundary_corrected"]),
        n=int(m["n_participants"]), test="LRT chi2(1), boundary-corrected (chi-bar2)", side="one-sided (boundary)",
        src=f, note="copied")
    lo, hi = [float(x) for x in m["reported_beta_ci95"].strip("[]").split(",")]
    add("1", "", "Fig. 2 text", "session-order beta (session 2 - session 1)", float(m["reported_beta_session2"]),
        (lo, hi), int(m["n_trials"]), "LME r ~ session2 + (1|subject) + (1|subject:session), REML", "two-sided",
        src=f, note=f"copied; se {m['reported_beta_se']}")
    add("1", "", "Fig. 2 text", "session-order beta Wald z", float(m["reported_beta_wald_z"]),
        n=int(m["n_trials"]), test="Wald z (statsmodels MixedLM, no df)", side="two-sided", src=f, note="copied")
    cmp("1", "beta Wald z = beta/se", float(m["reported_beta_wald_z"]),
        float(m["reported_beta_session2"]) / float(m["reported_beta_se"]))
    add("1", "", "Fig. 2 text", "session-order beta Wald p", float(m["reported_beta_wald_p"]),
        n=int(m["n_trials"]), test="Wald z", side="two-sided", src=f, note="copied")
    add("1", "", "Fig. 2 text", "session-order TOST p (bounds +-0.10 r)", float(m["reported_tost_p_bound_0.10"]),
        n=int(m["n_trials"]), test="TOST on LME beta", side="two one-sided", src=f,
        note="copied")

    f = "sup_restinput_trial_lme_metadata.csv"
    m = meta(f)
    lo, hi = [float(x) for x in m["lme_r_ci95"].split(" to ")]
    add("2", "", "Fig. 2 text", "own-EEG minus resting-input r (LME mean difference)",
        float(m["lme_r_mean_difference"]), (lo, hi), int(m["n_trials"]),
        "LME d ~ 1 + (1|participant) + (1|participant:session), REML", "two-sided", src=f,
        note=f"copied; se {m['lme_r_se']}; n_sessions 80, n_participants 42")
    add("2", "", "Fig. 2 text", "rest-input LME Wald z", float(m["lme_r_z"]), n=int(m["n_trials"]),
        test="Wald z", side="two-sided", src=f, note="copied")
    add("2", "", "Fig. 2 text", "rest-input LME p", float(m["lme_r_p"]), n=int(m["n_trials"]),
        test="Wald z", side="two-sided", src=f, note="copied")
    cmp("2", "z = diff/se", float(m["lme_r_z"]), float(m["lme_r_mean_difference"]) / float(m["lme_r_se"]))

    f = "fig2e_correspondence_metadata.csv"
    m = meta(f)
    iw_v, iw_lo, iw_hi, iw_tok = median_iqr(m["iw_diag_median_iqr"])
    re_v, re_lo, re_hi, re_tok = median_iqr(m["rest_diag_median_iqr"])
    pdm = _re.search(r"(\d+)/(\d+) sessions.*?W = (\d+(?:\.\d+)?).*?exact ([0-9.eE+-]+)\)", m["paired_drop"])
    n_up, n_d, w_tok, p_d = int(pdm.group(1)), int(pdm.group(2)), pdm.group(3), float(pdm.group(4))
    add("3", "", "Fig. 2i", "same-joint diagonal, imagined-walking input: median [IQR]", iw_v,
        (iw_lo, iw_hi), n_d, "median [IQR] (ci_lo/ci_hi = IQR, not CI)", src=f, note="copied iw_diag_median_iqr")
    add("3", "", "Fig. 2i", "same-joint diagonal, resting input: median [IQR]", re_v,
        (re_lo, re_hi), n_d, "median [IQR] (ci_lo/ci_hi = IQR, not CI)", src=f, note="copied rest_diag_median_iqr")
    add("3", "", "Fig. 2i", "IW vs REST diagonal: Wilcoxon W", float(w_tok), n=n_d, test="Wilcoxon signed-rank, exact",
        side="two-sided", src=f, note=f"copied paired_drop (W = {w_tok}, {n_up}/{n_d} sessions)")
    add("3", "", "Fig. 2i", "IW vs REST diagonal: p", p_d, n=n_d, test="Wilcoxon signed-rank, exact",
        side="two-sided", src=f, note="copied paired_drop exact p")
    if float(w_tok) == 0 and n_up == n_d:
        add("3", "", "Fig. 2i", "IW vs REST diagonal: rank-biserial r", 1.0, (np.nan, np.nan), n_d,
            "matched-pairs rank-biserial", src=f,
            note=f"derived from W = {w_tok} with {n_up}/{n_d} sessions IW > REST (T- = 0 => r = +1); bootstrap CI NA")
    else:
        add("3", "", "Fig. 2i", "IW vs REST diagonal: rank-biserial r", np.nan, (np.nan, np.nan), n_d,
            "matched-pairs rank-biserial", src=f,
            note=f"NA: W = {w_tok} with {n_up}/{n_d} sessions IW > REST")
    add("3", "", "Fig. 2i", "IW vs REST diagonal: median difference", np.nan, n=n_d,
        test="median of paired differences", src=f,
        note=f"NA; group medians {iw_tok} / {re_tok}")

    f = "fig3_d_forest_stats.csv"
    fs = pd.read_csv(ec.sd_in(f))
    master = pd.read_csv(ec.sd_in("fig3_master_source_data.csv"))
    fam4 = fs[fs.in_panel == 1]
    rec_p = []
    for _, row in fam4.iterrows():
        col = {"k1_taskevoked": "excess_k1", "k2_taskevoked": "excess_k2", "k3_taskevoked": "excess_k3",
               "mubeta_erd": "mubeta_erd_mean"}[row.predictor]
        rho, p = stats.spearmanr(master.r_reference, master[col], nan_policy="omit")
        rec_p.append(p)
        cmp("4", f"rho {row.predictor}", row.rho, rho)
    harm = [i for i, pr in enumerate(fam4.predictor) if pr != "mubeta_erd"]
    q_harm = dict(zip(harm, bh([rec_p[i] for i in harm])))
    for i, (_, row) in enumerate(fam4.iterrows()):
        is_erd = row.predictor == "mubeta_erd"
        fam_lab = ("none: Fig. 3c mu/beta ERD negative control, outside the k1-k3 family, uncorrected p" if is_erd
                   else "F4: Fig. 3c k1/k2/k3 task-evoked locking (BH over the 3 harmonics)")
        add("4", "", "Fig. 3c", f"Spearman rho(decoding r, {row.predictor})", row.rho, (row.ci_lo, row.ci_hi),
            int(row.n), "Spearman", "two-sided", fam_lab,
            f, "copied rho + CI (1000-resample session bootstrap, rng seed 0)")
        add("4", "", "Fig. 3c", f"{row.predictor}: p" + (" (uncorrected)" if is_erd else ""), row.p_spearman,
            n=int(row.n), test="Spearman", side="two-sided", family="none" if is_erd else "F4", src=f, note="copied")
        if is_erd:
            continue
        add("4", "", "Fig. 3c", f"{row.predictor}: pFDR", row.q_bh_fdr, n=int(row.n), test="Spearman, BH",
            side="two-sided", family="F4", src=f, note="copied")
        cmp("4", f"pFDR {row.predictor}", row.q_bh_fdr, q_harm[i])

    f = "sup_fig3_crosssession_predictor_source_data.csv"
    cs = pd.read_csv(ec.sd_in(f))
    rho, p = stats.spearmanr(cs.exA, cs.rB)
    m = meta("sup_fig3_crosssession_predictor_metadata.csv")
    cmp("5", "rho exA->rB", float(m["cross_excess_A_to_rB_rho"]), rho)
    cmp("5", "p exA->rB", float(m["cross_excess_A_to_rB_p"]), p)
    add("5", "", "Supp. Note 4 (Supplementary Fig. 3d)", "Spearman rho(session-1 k1 locking excess, session-2 decoding r)",
        float(rho), fisher_ci(rho, len(cs)), len(cs), "Spearman", "two-sided", "", f,
        "derived from per-participant rows; CI = Fisher z on rho, 1.96/sqrt(n-3) "
        "(approximate for Spearman)")
    add("5", "", "Supp. Note 4", "cross-session rho: p", float(p), n=len(cs), test="Spearman",
        side="two-sided", src=f, note="derived (scipy t-approximation)")

    f = "fig3d_locking_paired_source_data.csv"
    d3 = pd.read_csv(ec.sd_in(f))
    m = meta("fig3d_locking_paired_metadata.csv")
    w, r = paired_block("6", "", "Fig. 3d", "k1 |c| IW minus rest control",
                        d3.IW_c_k1 - d3.rest_c_k1, "", f, rng, "session")
    cmp("6", "W", float(m["wilcoxon_W"]), w.statistic)
    cmp("6", "p", float(m["wilcoxon_p"]), w.pvalue)
    cmp("6", "median difference", 0.0467, float(np.median(d3.IW_minus_rest)))
    add("6", "", "Fig. 3d", "k1 |c| median difference (as printed in legend)", 0.0467, (0.0399, 0.0600), 80,
        "median of paired differences", src="fig3d_locking_paired_metadata.csv",
        note="copied; CI = percentile bootstrap over sessions, B = 1000, seed 0 (the printed CI)")

    f = "sup_responder_participant_consistency_source_data.csv"
    rc = pd.read_csv(ec.sd_in(f))
    m = meta("sup_responder_participant_consistency_metadata.csv")
    a1, a2 = rc.resp_1.astype(str) == "True", rc.resp_2.astype(str) == "True"
    tab = np.array([[np.sum(a1 & a2), np.sum(a1 & ~a2)], [np.sum(~a1 & a2), np.sum(~a1 & ~a2)]])
    orr, pf = stats.fisher_exact(tab)
    po = np.trace(tab) / tab.sum()
    pe = (tab[0].sum() * tab[:, 0].sum() + tab[1].sum() * tab[:, 1].sum()) / tab.sum() ** 2
    kappa = (po - pe) / (1 - pe)
    cmp("7", "kappa", float(m["kappa"]), kappa)
    cmp("7", "Fisher OR (sample)", float(m["fisher_odds_ratio"]), orr)
    cmp("7", "Fisher p", float(m["fisher_p"]), pf)
    lw = np.log(orr)
    se = np.sqrt((1 / tab).sum())
    add("7", "", "Fig. 3e text", "responder status concordance (both sessions same)", int(np.trace(tab)),
        n=int(tab.sum()), test="count", src=f,
        note=f"derived 2x2 [s1 resp: s2 resp {tab[0,0]}, s2 non {tab[0,1]}; s1 non: s2 resp {tab[1,0]}, s2 non {tab[1,1]}]")
    add("7", "", "Fig. 3e text", "Cohen's kappa", float(kappa), n=int(tab.sum()), test="Cohen's kappa", src=f,
        note="derived from 2x2")
    add("7", "", "Fig. 3e text", "odds ratio (sample, ad/bc)", float(orr),
        (float(np.exp(lw - 1.959964 * se)), float(np.exp(lw + 1.959964 * se))), int(tab.sum()),
        "Fisher's exact test", "two-sided", src=f, note="derived; CI = Woolf (log-OR) 95% CI")
    try:
        from scipy.stats.contingency import odds_ratio
        cor = odds_ratio(tab, kind="conditional")
        ci = cor.confidence_interval(0.95)
        add("7", "", "Fig. 3e text", "odds ratio (conditional MLE)", float(cor.statistic),
            (float(ci.low), float(ci.high)), int(tab.sum()), "Fisher's exact test", "two-sided", src=f,
            note="derived; conditional MLE with exact (Fisher) 95% CI, scipy.stats.contingency.odds_ratio")
    except ImportError:
        add("7", "", "Fig. 3e text", "odds ratio (conditional MLE)", np.nan, n=int(tab.sum()), src=f,
            note="NA: scipy.stats.contingency.odds_ratio unavailable")
    add("7", "", "Fig. 3e text", "Fisher's exact p", float(pf), n=int(tab.sum()), test="Fisher's exact test",
        side="two-sided", src=f, note="derived")

    f = "fig3_def_resnonres_source_data.csv"
    rn = pd.read_csv(ec.sd_in(f))
    x = rn.loc[rn.resp.astype(str) == "True", "r_reference"].to_numpy()
    y = rn.loc[rn.resp.astype(str) == "False", "r_reference"].to_numpy()
    cliff = lambda a, b: float(np.mean(np.sign(a[:, None] - b[None, :])))
    ix = rng.integers(0, len(x), size=(B, len(x)))
    iy = rng.integers(0, len(y), size=(B, len(y)))
    md_b = np.median(x[ix], axis=1) - np.median(y[iy], axis=1)
    cd_b = np.array([cliff(x[i], y[j]) for i, j in zip(ix, iy)])
    obs = np.median(x) - np.median(y)
    pooled = np.concatenate([x, y])
    perm = np.empty(B)
    for b in range(B):
        s = rng.permutation(pooled)
        perm[b] = np.median(s[:len(x)]) - np.median(s[len(x):])
    p_perm = (1 + np.sum(np.abs(perm) >= abs(obs))) / (B + 1)
    mw = stats.mannwhitneyu(x, y, alternative="two-sided")
    add("7", "", "Fig. 3f", "responder median session decoding r", float(np.median(x)), n=len(x),
        test="median", src=f, note="derived (r_reference column =  session-mean rpt_calib)")
    add("7", "", "Fig. 3f", "non-responder median session decoding r", float(np.median(y)), n=len(y),
        test="median", src=f, note="derived")
    add("7", "", "Fig. 3f", "median difference responder - non-responder", float(obs),
        tuple(np.percentile(md_b, [2.5, 97.5])), len(x) + len(y), "difference of medians", src=f,
        note=f"derived; CI = percentile bootstrap resampling sessions within group, B={B}, seed {SEED}")
    add("7", "", "Fig. 3f", "Cliff's delta (= Mann-Whitney rank-biserial)", cliff(x, y),
        tuple(np.percentile(cd_b, [2.5, 97.5])), len(x) + len(y), "Cliff's delta", src=f,
        note=f"derived; CI = stratified percentile bootstrap, B={B}, seed {SEED}")
    add("7", "", "Fig. 3f", "Mann-Whitney U", float(mw.statistic), n=len(x) + len(y),
        test="Mann-Whitney U (responder first)", side="two-sided", src=f, note="derived")
    add("7", "", "Fig. 3f", "Mann-Whitney p", float(mw.pvalue), n=len(x) + len(y),
        test="Mann-Whitney U", side="two-sided", src=f, note="derived")
    add("7", "", "Fig. 3f", "permutation p (as stored)", np.nan, n=len(x) + len(y), test="permutation",
        src=f, note="NA: see the recomputed "
                    "permutation p")
    add("7", "", "Fig. 3f", "permutation p (recomputed here, difference of medians)", float(p_perm),
        n=len(x) + len(y), test=f"label permutation, fixed 55/25, B={B}, seed {SEED}", side="two-sided", src=f,
        note="derived; p = (1 + #|perm| >= |obs|)/(B + 1); floor 1/(B+1)")

    f = "fig3_h_forest_stats.csv"
    hs = pd.read_csv(ec.sd_in(f))
    sp = pd.read_csv(ec.sd_in("fig3_h_scatter_points.csv"))
    rp = []
    for it in hs.item:
        rho, p = stats.spearmanr(sp[it], sp.excess_k1, nan_policy="omit")
        rp.append(p)
        if it == "Q6_avoid":
            cmp("8", "rho Q6 vs k1 locking", float(hs.loc[hs.item == it, "rho"].iloc[0]), rho)
    qq = bh(rp)
    cmp("8", "pFDR Q6 (6 items)", float(hs.loc[hs.item == "Q6_avoid", "q_bh_fdr"].iloc[0]),
        float(qq[list(hs.item).index("Q6_avoid")]))
    q6 = hs[hs.item == "Q6_avoid"].iloc[0]
    add("8", "", "Fig. 3h,i", "Spearman rho(volitional control Q6, k1 locking excess)", q6.rho,
        (q6.ci_lo, q6.ci_hi), int(q6.n), "Spearman", "two-sided", "F8: six questionnaire items vs k1 locking", f,
        "copied rho + CI (session bootstrap, 1000 resamples, rng seed 0)")
    add("8", "", "Fig. 3h,i", "Q6 rho: p", q6.p_spearman, n=int(q6.n), test="Spearman", side="two-sided",
        family="F8", src=f, note="copied")
    add("8", "", "Fig. 3h,i", "Q6 rho: pFDR", q6.q_bh_fdr, n=int(q6.n), test="Spearman, BH over 6 items",
        side="two-sided", family="F8", src=f, note="copied")
    add("8", "", "Fig. 3h,i", "Q6 rho: Fisher-z CI (alternative)", q6.rho, fisher_ci(q6.rho, int(q6.n)),
        int(q6.n), "Spearman", src=f, note="derived; Fisher z CI, 1.96/sqrt(n-3), for comparison with the bootstrap CI")
    f = "fig3_mediation_source_data.csv"
    md = pd.read_csv(ec.sd_in(f)).set_index("path")
    n_med = int(meta("fig3_mediation_metadata.csv")["n_analysed"])
    cp_ci = (md.loc["cprime_CI_lo", "beta"], md.loc["cprime_CI_hi", "beta"])
    add("8", "", "Fig. 3j", "indirect effect a x b (standardized)", md.loc["indirect_axb", "beta"],
        (md.loc["indirect_CI_lo", "beta"], md.loc["indirect_CI_hi", "beta"]), n_med,
        "OLS mediation, session-level percentile bootstrap B=1000 seed 0", "two-sided", src=f,
        note="copied; bootstrap p stored as 0.0 (0 of 1000 resamples cross zero) -> report p < 0.001")
    add("8", "", "Fig. 3j", "indirect effect bootstrap p", np.nan, n=n_med, test="bootstrap, B=1000",
        side="two-sided", src=f, note="stored 0.0 -> write p < 0.001 (= < 1/B); no exact value exists")
    add("8", "", "Fig. 3j", "direct effect c' (standardized)", md.loc["cprime_direct_X_to_Y_given_M", "beta"],
        cp_ci, n_med, "OLS Y ~ X + M (z-scored)", "two-sided", src=f,
        note="copied beta + CI (session-level percentile bootstrap B=1000 seed 0, the same resamples as the "
             "a x b CI)")
    add("8", "", "Fig. 3j", "direct effect c' p", md.loc["cprime_direct_X_to_Y_given_M", "p"], n=n_med,
        test="OLS t-test", side="two-sided", src=f, note="copied")

    f = "fig4_trialmedian_source_data.csv"
    tm = pd.read_csv(ec.sd_in(f))
    m = meta("fig4_trialmedian_metadata.csv")
    w, r = paired_block("9", "", "Fig. 4b", "RW minus IW per-session median r", tm.rw_median - tm.iw_median,
                        "", f, rng, "session")
    cmp("9", "W", float(m["sessions_wilcoxon_W"]), w.statistic)
    cmp("9", "p", float(m["sessions_wilcoxon_p"]), w.pvalue)

    f = "fig4_phase_delay_persession_metadata.csv"
    m = meta(f)
    for k in (1, 2, 3):
        add("10", "", "Fig. 4d", f"k{k} per-session circular mean offset (% cycle)",
            float(m[f"k{k}_circ_mean_Delta_pct"]), (float(m[f"k{k}_circ_mean_CI95_lo"]),
                                                    float(m[f"k{k}_circ_mean_CI95_hi"])), 80,
            "circular mean", src=f, note="copied; CI = session bootstrap B=1000 seed 0")
        add("10", "", "Fig. 4d", f"k{k} Rayleigh z (= n R_bar^2)", float(m[f"k{k}_rayleigh_z"]), n=80,
            test="Rayleigh test", side="", family="F10a: Rayleigh over k1-k3", src=f,
            note=f"copied; R_bar = {m[f'k{k}_R_bar']}")
        add("10", "", "Fig. 4d", f"k{k} Rayleigh p", float(m[f"k{k}_rayleigh_P"]), n=80, test="Rayleigh",
            family="F10a", src=f, note="copied")
        add("10", "", "Fig. 4d", f"k{k} Rayleigh pFDR", float(m[f"k{k}_rayleigh_P_BH"]), n=80,
            test="Rayleigh, BH over 3", family="F10a", src=f, note="copied")
    cmp("10", "Rayleigh z k2 = n R^2", float(m["k2_rayleigh_z"]), 80 * float(m["k2_R_bar"]) ** 2)
    cmp("10", "Rayleigh BH k1..k3", str([m[f"k{k}_rayleigh_P_BH"] for k in (1, 2, 3)]),
        str(np.round(bh([float(m[f"k{k}_rayleigh_P"]) for k in (1, 2, 3)]), 5)))

    f = "fig4d_rsa_persession_spin_metadata.csv"
    m = meta(f)
    sz = pd.read_csv(ec.sd_in("fig4d_rsa_persession_spin_source_data.csv"))
    for k in (1, 2, 3):
        add("10", "", "Fig. 4e", f"k{k} cohort-mean per-session spatial RSA rho (observed)",
            float(m[f"k{k}_cohort_mean_rho_obs"]), n=80, test="Spearman RSA", src=f,
            note=f"copied; spin-null mean {m[f'k{k}_cohort_mean_null_mean']}")
        add("10", "", "Fig. 4e", f"k{k} spin-test p", float(m[f"k{k}_P_spin"]), n=80,
            test="spin permutation, 1000 rotations", side="one-sided", family="F10b: spin RSA over k1-k3",
            src=f, note="copied; P = (1 + #null >= obs)/(1000 + 1)")
        add("10", "", "Fig. 4e", f"k{k} spin-test pFDR", float(m[f"k{k}_P_spin_BH"]), n=80,
            test="spin permutation, BH over 3", side="one-sided", family="F10b", src=f, note="copied")
        z = sz.loc[sz.harmonic_k == k, "z"].to_numpy()
        w = stats.wilcoxon(z)
        rr, tp, tmn = rank_biserial(z)
        cmp("10", f"k{k} Wilcoxon spin-z p", float(m[f"k{k}_wilcoxon_z_P"]), w.pvalue)
        add("10", "", "Fig. 4e", f"k{k} per-session spin-z vs 0: Wilcoxon W", float(w.statistic), n=len(z),
            test="Wilcoxon signed-rank (secondary test)", side="two-sided", src="fig4d_rsa_persession_spin_source_data.csv",
            note=f"derived; T+={tp:g}, T-={tmn:g}")
        add("10", "", "Fig. 4e", f"k{k} per-session spin-z vs 0: rank-biserial r", float(rr),
            boot(z, lambda v: rank_biserial(v)[0], rng), len(z), "matched-pairs rank-biserial",
            src="fig4d_rsa_persession_spin_source_data.csv", note=f"derived; bootstrap over sessions B={B} seed {SEED}")

    f = "fig4_correspondence_amplitude_source_data.csv"
    am = pd.read_csv(ec.sd_in(f)).pivot(index="session", columns="harmonic_k", values="correspondence_amplitude")
    pairs = [(2, 1), (2, 3), (1, 3)]
    res = []
    for a, b_ in pairs:
        dd = (am[a] - am[b_]).to_numpy()
        res.append((a, b_, stats.wilcoxon(dd), rank_biserial(dd), dd))
    qa = bh([x[2].pvalue for x in res])
    mm = meta("fig4_correspondence_amplitude_metadata.csv")
    cmp("10", "amp k1 vs k2 raw p", 0.000113, res[0][2].pvalue)
    cmp("10", "amp k3 vs k2 raw p", 4.12e-05, res[1][2].pvalue)
    cmp("10", "amp k1 vs k2 BH q", 0.000169, qa[0])
    cmp("10", "amp k3 vs k2 BH q", 0.000124, qa[1])
    for (a, b_, w, (rr, tp, tmn), dd), q in zip(res, qa):
        lab = f"amplitude A k{a} minus k{b_}"
        add("10", "", "Fig. 4f", f"{lab}: Wilcoxon W", float(w.statistic), n=len(dd),
            test="Wilcoxon signed-rank", side="two-sided", family="F10c: amplitude pairs (3)", src=f,
            note=f"derived; T+={tp:g}, T-={tmn:g}")
        add("10", "", "Fig. 4f", f"{lab}: p", float(w.pvalue), n=len(dd), test="Wilcoxon signed-rank",
            side="two-sided", family="F10c", src=f, note="derived")
        add("10", "", "Fig. 4f", f"{lab}: pFDR", float(q), n=len(dd), test="Wilcoxon, BH over 3 pairs",
            side="two-sided", family="F10c", src=f,
            note="derived")
        add("10", "", "Fig. 4f", f"{lab}: rank-biserial r", float(rr), boot(dd, lambda v: rank_biserial(v)[0], rng),
            len(dd), "matched-pairs rank-biserial", family="F10c", src=f,
            note=f"derived; bootstrap over sessions B={B} seed {SEED}")

    f = "fig5_efgh_patient_reconstruction_source_data.csv"
    h = pd.read_csv(ec.sd_in(f))
    h = h[h.panel == "h"].set_index("quantity")
    p1, p2 = float(h.loc["wilcoxon_p_IW_diag_vs_offdiag", "value"]), float(h.loc["wilcoxon_p_diag_IW_vs_REST_paired", "value"])
    q5 = bh([p1, p2])
    n5a = int(h.loc["wilcoxon_W_IW_diag_vs_offdiag", "n_sessions_contributing"])
    n5b = int(h.loc["wilcoxon_W_diag_IW_vs_REST_paired", "n_sessions_contributing"])
    for lab, wk, pk, n_, q in [("IW same-joint diagonal vs mean |off-diagonal|", "wilcoxon_W_IW_diag_vs_offdiag", p1, n5a, q5[0]),
                               ("same-joint diagonal IW vs resting input (paired)", "wilcoxon_W_diag_IW_vs_REST_paired", p2, n5b, q5[1])]:
        W_ = float(h.loc[wk, "value"])
        add("11", "", "Fig. 5h", f"{lab}: Wilcoxon W", W_, n=n_, test="Wilcoxon signed-rank, exact",
            side="two-sided", family="F11: Fig. 5h two tests", src=f, note="copied")
        add("11", "", "Fig. 5h", f"{lab}: p", pk, n=n_, test="Wilcoxon signed-rank, exact", side="two-sided",
            family="F11", src=f, note="copied")
        add("11", "", "Fig. 5h", f"{lab}: pFDR", float(q), n=n_, test="BH over the 2 tests", side="two-sided",
            family="F11", src=f, note="derived")
        add("11", "", "Fig. 5h", f"{lab}: rank-biserial r", 1.0 if W_ == 0 else np.nan, n=n_,
            test="matched-pairs rank-biserial", family="F11", src=f,
            note="derived from W = 0 (every session same sign, first-named larger per the stored medians) -> r = +1")
    cmp("11", "pFDR both", "0.0078/0.0078", f"{q5[0]:.6f}/{q5[1]:.6f}")

    f = "fig6_trial_excess_source_data.csv"
    te = pd.read_csv(ec.sd_in(f))
    exp_pct = {1: 60.6, 2: 91.5, 3: 66.8}
    sess_p = []
    for k in (1, 2, 3):
        t_ = te[te.harmonic_k == k]
        qk = bh(t_.null_p)
        cnt = int(np.sum(qk < 0.05))
        pct = 100 * cnt / len(t_)
        cmp("12", f"k{k} % trials BH<0.05", exp_pct[k], round(pct, 1))
        add("12", "", "Fig. 6b", f"k{k} trials above own null after BH (count)", cnt, n=len(t_),
            test="per-trial surrogate p, BH within harmonic", side="one-sided", src=f, note=f"derived; {pct:.1f}%")
        add("12", "", "Fig. 6b", f"k{k} trials above own null after BH (%)", round(pct, 1), n=len(t_),
            test="per-trial surrogate p, BH within harmonic", side="one-sided", src=f, note="derived")
        sm = t_.groupby("session").excess.mean().to_numpy()
        w = stats.wilcoxon(sm)
        sess_p.append(w.pvalue)
        rr, tp, tmn = rank_biserial(sm)
        add("12", "", "Fig. 6b", f"k{k} session-mean excess vs 0: Wilcoxon W", float(w.statistic), n=len(sm),
            test="Wilcoxon signed-rank, exact", side="two-sided", family="F12: k1-k3 session-level", src=f,
            note=f"derived; T+={tp:g}, T-={tmn:g}; median session excess {np.median(sm):.4f}")
        add("12", "", "Fig. 6b", f"k{k} session-mean excess vs 0: p", float(w.pvalue), n=len(sm),
            test="Wilcoxon signed-rank, exact", side="two-sided", family="F12", src=f, note="derived")
        add("12", "", "Fig. 6b", f"k{k} session-mean excess vs 0: rank-biserial r", float(rr),
            boot(sm, lambda v: rank_biserial(v)[0], rng), len(sm), "matched-pairs rank-biserial", family="F12",
            src=f, note=f"derived; bootstrap over sessions B={B} seed {SEED}")
    for k, q in zip((1, 2, 3), bh(sess_p)):
        add("12", "", "Fig. 6b", f"k{k} session-mean excess vs 0: pFDR", float(q), n=11,
            test="Wilcoxon, BH over 3 harmonics", side="two-sided", family="F12", src=f, note="derived")

    f = "fig6_patient_healthy_harmonic_similarity_source_data.csv"
    hsim = pd.read_csv(ec.sd_in(f))
    g = hsim[hsim.row_type == "group"].set_index("harmonic_k")
    ch = pd.read_csv(ec.sd_in("fig6_cross_harmonic_specificity_source_data.csv"))
    pe_ = []
    for k in (1, 2, 3):
        diag = float(ch[(ch.patient_harmonic == k) & (ch.healthy_harmonic == k)].coherence.iloc[0])
        cmp("13", f"k{k} diagonal = similarity group row", g.loc[k, "similarity"], diag)
        p_ = max(float(g.loc[k, "empirical_p"]), 1 / 1001)
        pe_.append(p_)
        add("13", "", "Fig. 6d", f"k{k} matching-harmonic coherence", diag, n=11, test="complex spatial coherence",
            src="fig6_cross_harmonic_specificity_source_data.csv",
            note=f"copied; channel-shuffle null 95th pct {g.loc[k, 'null_p95']}")
        add("13", "", "Fig. 6d", f"k{k} matching-harmonic empirical p (upper bound)", p_, n=11,
            test="channel-shuffle permutation, 1000 perms", side="one-sided", family="F13: matching harmonics k1-k3",
            src=f, note="stored empirical_p = 0.0000 (no permutation >= observed) -> written as p <= 1/1001")
    for k, q in zip((1, 2, 3), bh(pe_)):
        add("13", "", "Fig. 6d", f"k{k} matching-harmonic pFDR (upper bound)", float(q), n=11,
            test="BH over 3", side="one-sided", family="F13", src=f, note="derived from the 1/1001 bounds -> report pFDR < 0.001")

    f = "fig6_locking_vs_transfer_metadata.csv"
    m = meta(f)
    b_, s_ = float(m["lme_slope_patient_random_intercept"]), float(m["lme_slope_se"])
    z = b_ / s_
    pz = 2 * stats.norm.sf(abs(z))
    cmp("14", "LME slope Wald p", float(m["lme_slope_wald_p"]), pz)
    add("14", "", "Fig. 6g", "LME slope (k2 locking -> transfer r)", b_, (b_ - 1.959964 * s_, b_ + 1.959964 * s_), 11,
        "LME y ~ x + (1|patient), REML", "two-sided", src=f, note=f"copied slope, se {s_}; CI derived = Wald 1.96*se")
    add("14", "", "Fig. 6g", "LME slope Wald z", float(z), n=11, test="Wald z", side="two-sided", src=f, note="derived beta/se")
    add("14", "", "Fig. 6g", "LME slope Wald p", float(m["lme_slope_wald_p"]), n=11, test="Wald z", side="two-sided",
        src=f, note="copied")
    add("14", "", "Fig. 6g", "within-patient exact permutation p", float(m["perm_within_patient_p"]), n=11,
        test="exact label permutation within patient (86,400)", side="two-sided", src=f, note="copied")

    f = "fig6_movement_free_dissociation_source_data.csv"
    mf = pd.read_csv(ec.sd_in(f))
    exp_surv = {1: 9, 2: 10}
    for k in (1, 2):
        rows = mf[(mf.subpanel == f"left_phaselock_persession_k{k}") & (mf.task == "IW")]
        p_ = np.maximum(rows.p_value.to_numpy(float), 1 / 1001)
        q = bh(p_)
        cnt = int(np.sum(q < 0.05))
        cmp("14", f"IW k{k} sessions surviving BH", exp_surv[k], cnt)
        add("14", "", "Fig. 6h (phase-lock sub-panel)", f"IW k{k} sessions above surrogate after BH", cnt, n=len(rows),
            test="per-session phase-shift surrogate (1000), BH within harmonic over sessions", side="one-sided",
            src=f, note=f"derived; stored p = 0 replaced by 1/1001; max pFDR among survivors "
                        f"{q[q < 0.05].max():.4f}; raw p < 0.05 in {int(np.sum(p_ < 0.05))}/{len(rows)}")

    md = meta("fig6_movement_free_dissociation_metadata.csv")
    sig = md["right_significance"]
    import re
    erd_rows = mf[(mf.session == "cohort_mean") & mf.subpanel.str.startswith("right_erd")]
    pe, labs = [], []
    for _, row in erd_rows.iterrows():
        task, band, n_, p_st = row.task, row.harmonic_or_band, int(row.n_sessions), float(row.p_value)
        mneg = re.search(rf"{task} {band} P=[0-9.]+ \((\d+)/(\d+) neg\)", sig)
        nneg = int(mneg.group(1)) if mneg else None
        hits = invert_exact_W(p_st, n_, 4)
        lab = f"{task} {band} ERD vs 0"
        pe.append(p_st)
        labs.append((lab, n_))
        add("14", "", "Fig. 6h", f"{lab}: mean dB", float(row.value), n=n_, test="mean +- SEM", src=f,
            note=f"copied; SEM {row['sem']}")
        add("14", "", "Fig. 6h", f"{lab}: p", p_st, n=n_, test="Wilcoxon signed-rank vs 0, exact",
            side="two-sided", family="F14: ERD IW/RW/S2S x mu/beta (6)", src=f, note="copied (4 d.p. as stored)")
        if len(hits) == 1 and nneg is not None:
            W_ = hits[0]
            S = n_ * (n_ + 1) / 2
            rr = (2 * W_ - S) / S if nneg > n_ / 2 else (S - 2 * W_) / S
            add("14", "", "Fig. 6h", f"{lab}: Wilcoxon W", float(W_), n=n_, test="Wilcoxon signed-rank, exact",
                side="two-sided", family="F14", src=f,
                note="derived from the exact null distribution of W at the stored p (unique solution)")
            add("14", "", "Fig. 6h", f"{lab}: rank-biserial r", float(rr), n=n_, test="matched-pairs rank-biserial",
                family="F14", src=f,
                note=f"derived from W and {nneg}/{n_} sessions negative (negative r = desynchronization)")
        else:
            add("14", "", "Fig. 6h", f"{lab}: Wilcoxon W", np.nan, n=n_, family="F14", src=f,
                note=f"NA: exact-p inversion not unique (candidates {hits}) or sign count missing")
    for (lab, n_), q in zip(labs, bh(pe)):
        add("14", "", "Fig. 6h", f"{lab}: pFDR", float(q), n=n_, test="BH over 6", side="two-sided",
            family="F14", src=f, note="derived from the stored 4-d.p. p values")

    fams = {"fig7_sim_trajectories_stats.csv": ("Fig. 7c", "displacement", 7, "fig7_sim_trajectories_source_data.csv"),
            "fig7_sim_cadence_stats.csv": ("Fig. 7d", "cadence", 3, "fig7_sim_cadence_source_data.csv"),
            "fig7_sim_step_symmetry_stats.csv": ("Fig. 7e", "asymmetry", 3, "fig7_sim_step_symmetry_source_data.csv")}
    exp_bh = {("displacement", "Walker_minus_Decoded"): 2.7e-18, ("cadence", "Walker_minus_Decoded"): 0.347,
              ("asymmetry", "Walker_minus_Decoded"): 0.090}
    def _f7(name):
        return ec.sd_in(name)
    for fn, (fig, fam, k_, raw) in fams.items():
        st = pd.read_csv(_f7(fn)).iloc[:k_]
        q = bh(st.p_used)
        rd = pd.read_csv(_f7(raw))
        if fam == "displacement":
            rd = (rd[rd.row_type == "trace"].sort_values("t_s")
                  .groupby(["condition_label", "session", "trial_idx"], as_index=False).last()
                  .rename(columns={"x_m": "value"}))
        wide = rd.pivot_table(index=["session", "trial_idx"], columns="condition_label", values="value", aggfunc="first")
        para = ""
        for (_, row), qq in zip(st.iterrows(), q):
            c = row.comparison
            fm = f"F15-{fam}: Fig. 7 {fam} rows ({k_})"
            if (fam, c) in exp_bh:
                cmp("15", f"{fam} {c} BH", exp_bh[(fam, c)], qq)
            add("15", para, fig, f"{fam} {c}: p_used", row.p_used, n=int(row.n), test=row.test_used, side="two-sided",
                family=fm, src=fn, note="copied")
            add("15", para, fig, f"{fam} {c}: pFDR", float(qq), n=int(row.n), test=f"{row.test_used}, BH over {k_}",
                side="two-sided", family=fm, src=fn, note="derived")
            if np.isfinite(row.mean_difference_lme):
                add("15", para, fig, f"{fam} {c}: LME mean difference", row.mean_difference_lme,
                    (row.mean_difference_lme - 1.959964 * row.se_lme, row.mean_difference_lme + 1.959964 * row.se_lme),
                    int(row.n), "LME d ~ 1 + (1|session), REML", family=fm, src=fn,
                    note=f"copied mean + se {row.se_lme}; CI derived = Wald 1.96*se")
            add("15", para, fig, f"{fam} {c}: median difference", row.median_difference, n=int(row.n),
                test="median", family=fm, src=fn, note="copied")
            if row.test_used.startswith("Wilcoxon"):
                if "_minus_" in c:
                    a_, b2 = c.split("_minus_")
                    d = (wide[a_] - wide[b2]).to_numpy()
                else:
                    d = wide[c.replace("_vs_zero", "")].to_numpy()
                d = d[np.isfinite(d)]
                w = stats.wilcoxon(d)
                cmp("15", f"{fam} {c} Wilcoxon p (n={len(d)} vs {int(row.n)})", row.p_wilcoxon, w.pvalue)
                rr, tp, tmn = rank_biserial(d)
                add("15", para, fig, f"{fam} {c}: Wilcoxon W", float(w.statistic), n=len(d),
                    test="Wilcoxon signed-rank", side="two-sided", family=fm, src=raw,
                    note=f"derived from per-rollout Source Data (values exported at 6 d.p.); T+={tp:g}, T-={tmn:g}; "
                         f"recomputed p {w.pvalue:.4g} vs stored p_wilcoxon {row.p_wilcoxon:.4g} (p_used is the copied value)")
                add("15", para, fig, f"{fam} {c}: rank-biserial r", float(rr),
                    boot(d, lambda v: rank_biserial(v)[0], rng), len(d), "matched-pairs rank-biserial",
                    family=fm, src=raw, note=f"derived; bootstrap over rollouts B={B} seed {SEED}")

    out = pd.DataFrame(ROWS)
    write_atomic(out, OUT)
    fam_lines = [
        ("F8", "Fig. 3i: six questionnaire items vs k1 locking excess; BH over 6 (as stored)"),
        ("F10a", "Fig. 4d: Rayleigh tests k1/k2/k3; BH over 3 (as stored)"),
        ("F10b", "Fig. 4e: cohort-mean spin tests k1/k2/k3, one-sided; BH over 3 (as stored)"),
        ("F10c", "Fig. 4f: amplitude pairs k2-k1, k2-k3, k1-k3 Wilcoxon; BH over 3 (reproduces stored q)"),
        ("F11", f"Fig. 5h: IW diag vs off-diag (n={n5a}) and diag IW vs REST paired (n={n5b}); BH over 2"),
        ("F12", "Fig. 6b: session-mean excess vs 0, k1/k2/k3; BH over 3. Trial counts: BH within harmonic over 449 trials"),
        ("F13", "Fig. 6d: matching-harmonic channel-shuffle p (bounded at 1/1001), k1/k2/k3; BH over 3"),
        ("F14", "Fig. 6h: ERD vs 0 for IW/RW/S2S x mu/beta; BH over 6. Phase-lock sessions: BH within harmonic over 11 sessions"),
        ("F15-displacement", "Fig. 7c: first 7 rows of fig7_sim_trajectories_stats.csv (p_used); BH over 7"),
        ("F15-cadence", "Fig. 7d: fig7_sim_cadence_stats.csv 3 rows (p_used); BH over 3"),
        ("F15-asymmetry", "Fig. 7e: fig7_sim_step_symmetry_stats.csv 3 rows (p_used); BH over 3"),
    ]
    md_rows = [("build_id", f"{ec.TAG} healthy / {ec.COHORTS['post_operative']} patient"
                       + ("" if ec.IS_DEFAULT else f" (neural zero-phase rows: {ec.NEURAL_TAG})")),
               ("producer", "03_figures/results_statistics.py"),
               ("bh_method", "statsmodels.stats.multitest.multipletests method=fdr_bh alpha=0.05"),
               ("bootstrap", f"percentile, B={B}, numpy default_rng seed {SEED} (one stream, items in script order); "
                             "copied CIs keep their own B/seed as stated in the row note"),
               ("rank_biserial", "matched pairs (T+ - T-)/(T+ + T-), zeros dropped; positive = first-named larger"),
               ("fisher_z_ci", "tanh(atanh(r) +- 1.959964/sqrt(n-3))"),
               ("ci_columns_note", "rows labelled 'median [IQR]' carry the IQR, not a CI, in ci_lo/ci_hi"),
               ("upstream_rerun", "none: no prep, decoder or simulation re-run")]
    md_rows += [(f"family_{k}", v) for k, v in fam_lines]
    if ec.F7_SIM == "causal_from2s":
        md_rows += [("fig7_source", "F15 rows read from the Fig. 7 Source Data of the published simulation build (GAIT_F7_SIM=causal_from2s); "
                                    "all other rows from the decoder and decoder-independent Source Data in 04_source_data/"), ec.F7_SIM_MD]
    write_atomic(pd.DataFrame(md_rows, columns=["key", "value"]), OUT_META)

    print(f"{'item':<5}{'quantity':<52}{'reported':>22}{'recomputed':>22}")
    for it, q, rep, rec in CMP:
        fmt = lambda v: f"{v:.6g}" if isinstance(v, (float, int, np.floating, np.integer)) else str(v)
        print(f"{it:<5}{q[:51]:<52}{fmt(rep):>22}{fmt(rec):>22}")
    back = pd.read_csv(OUT)
    print(f"\nwrote {OUT} ({len(back)} rows) and {OUT_META} ({len(pd.read_csv(OUT_META))} rows)")


def write_atomic(df, path):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    df.to_csv(tmp, index=False)
    assert len(pd.read_csv(tmp)) == len(df)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
