#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 6 — Representational geometry of patient imagined walking

_locking_vs_transfer.py

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and does not fit the decoder. It fits a linear mixed model and a least-squares regression
for the statistics reported with that panel or table.
"""
r"""Fig. 6 panel g — per-session k2 gait-phase-locking |c| vs transfer decoding.

               
   
         

CHAIN
  in : data/decoder_output/patient_pertrial_c.csv   (per-trial |c|)
       + pat02_sess08 sidecar patient_pertrial_c_sess08.csv
       + pat02_sess09 sidecar patient_pertrial_c_sess09.csv
       04_source_data/fig5/fig5b_session_box_source_data.csv                      (trial-median r)
  out: figures_out/fig6/fig6_locking_vs_transfer.png
       figures_out/fig6/fig6_locking_vs_transfer_legend.png
       figures_out/fig6/fig6_locking_vs_transfer_stats.txt
       04_source_data/fig6_locking_vs_transfer_{source_data,metadata}.csv
"""
import csv
import warnings
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

import fig6_panel_style as S
import project_palette as P
from sup_panel_export import PANEL_MODE, PANEL_DIR, export_stats
import analysis_config as ec


def pfmt(p):
    """House style: lower-case italic p, leading zero, floor at 0.001."""
    return "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "decoder_output"
PC = RES / "patient_pertrial_c.csv"
DEC_SD = ec.sd("fig5b_session_box_source_data.csv")
DEC_META = ec.sd("fig5b_session_box_metadata.csv")
OUT_PNG = ec.scoped(ROOT / "figures_out/fig6/fig6_locking_vs_transfer.png")
STATS = ec.scoped(ROOT / "figures_out/fig6/fig6_locking_vs_transfer_stats.txt")
SRC = ec.sd("fig6_locking_vs_transfer_source_data.csv")
META = ec.sd("fig6_locking_vs_transfer_metadata.csv")
TRIAL_LME_SD = ec.sd("fig5g_patient_trialwise_lme_source_data.csv")

def _ms_mm(v):
    return (round(v / 25.4 * 600) + 0.25) / 600 * 25.4


BUILD_ID = ec.TAG
COHORT = ec.COHORTS["post_operative"]
KH = 2

POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
            "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
            "pat02_sess09"]
SIDECARS = {
    "pat02_sess08": RES / "patient_pertrial_c_sess08.csv",
    "pat02_sess09": RES / "patient_pertrial_c_sess09.csv",
}

P1C, P2C = P.PATIENT_P1, P.PATIENT_P2

S.apply(plt)


def load_locking():
    """session -> (mean |c| at k=KH over that session's IW trials, n_trials)."""
    parts = [pd.read_csv(PC)]
    seen = set(parts[0].session.unique())
    for sess, path in SIDECARS.items():
        d = pd.read_csv(path)
        ids = set(d.session.unique())
        if ids != {sess}:
            raise SystemExit(f"{path.name} must hold exactly {sess}, got: {sorted(ids)}")
        if ids & seen:
            raise SystemExit(f"{sess} present in BOTH {PC.name} and its sidecar "
                             f"- would double-weight")
        seen |= ids
        parts.append(d)
    c = pd.concat(parts, ignore_index=True)
    c = c[c.harmonic_k == KH]
    missing = [s for s in POST_OPERATIVE if s not in set(c.session)]
    if missing:
        raise SystemExit(f"post_operative sessions absent from the |c| tables: {missing}")
    g = c[c.session.isin(POST_OPERATIVE)].groupby("session").c_mi
    return {s: (float(g.mean()[s]), int(g.count()[s])) for s in POST_OPERATIVE}


def exact_within_patient_perm_p(x, y, patient, r_obs):
    """Exact label permutation of y WITHIN each patient (6! x 5! = 86,400), two-sided on |r|."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    i1 = np.flatnonzero(np.asarray(patient) == "P1")
    i2 = np.flatnonzero(np.asarray(patient) == "P2")
    p1 = np.array(list(permutations(y[i1])))
    p2 = np.array(list(permutations(y[i2])))
    n1, n2 = len(p1), len(p2)
    Y = np.empty((n1, n2, len(y)))
    Y[:, :, i1] = p1[:, None, :]
    Y[:, :, i2] = p2[None, :, :]
    Y = Y.reshape(n1 * n2, len(y))
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=1, keepdims=True)
    r = (Yc @ xc) / (np.linalg.norm(xc) * np.linalg.norm(Yc, axis=1))
    n_perm = r.size
    return float((np.abs(r) >= abs(r_obs) - 1e-12).sum() / n_perm), int(n_perm)


def main():
    for _p in [PC, *SIDECARS.values()]:
        ec.check_tag(_p, "neural", "|c| input")
    for _m in (DEC_META, ec.sd("fig5g_patient_trialwise_lme_metadata.csv")):
        _dec_build_id = next(r[1] for r in csv.reader(open(_m, encoding="utf-8-sig"))
                        if r and r[0] == "build_id")
        ec.check_tag(_dec_build_id, "decoder", f"{_m.name} build_id")
    lock = load_locking()
    dec = {r["session"]: float(r["median"]) for r in csv.DictReader(open(DEC_SD))
           if r["cohort"] == COHORT}
    missing = [s for s in POST_OPERATIVE if s not in dec]
    if missing:
        raise SystemExit(f"{DEC_SD.name} has no {COHORT} row for: {missing}")

    xs = np.array([lock[s][0] for s in POST_OPERATIVE])
    ns = [lock[s][1] for s in POST_OPERATIVE]
    ys = np.array([dec[s] for s in POST_OPERATIVE])
    pats = ["P1" if s.startswith("pat01") else "P2" for s in POST_OPERATIVE]

    pr, pp = pearsonr(xs, ys)
    sr, sp = spearmanr(xs, ys)
    df = pd.DataFrame(dict(session=POST_OPERATIVE, x=xs, y=ys, patient=pats))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols_fe = smf.ols("y ~ x + C(patient)", df).fit()
        lme = smf.mixedlm("y ~ x", df, groups=df["patient"]).fit(reml=True)
    b_fe, p_fe = float(ols_fe.params["x"]), float(ols_fe.pvalues["x"])
    b_lme = float(lme.params["x"])
    se_lme = float(lme.bse["x"])
    p_lme = float(lme.pvalues["x"])

    loo = []
    for i, s in enumerate(POST_OPERATIVE):
        k = np.arange(len(POST_OPERATIVE)) != i
        r_i, p_i = pearsonr(xs[k], ys[k])
        loo.append((s, float(r_i), float(p_i)))
    loo_r = [t[1] for t in loo]
    loo_p = [t[2] for t in loo]

    per_pat = {}
    for pid in ("P1", "P2"):
        k = np.array([p == pid for p in pats])
        rho_i, p_i = spearmanr(xs[k], ys[k])
        per_pat[pid] = (float(rho_i), float(p_i), int(k.sum()))

    perm_p, n_perm = exact_within_patient_perm_p(xs, ys, pats, pr)

    tl = pd.read_csv(TRIAL_LME_SD)
    tlr = tl[(tl.cohort == "patient") & (tl.harmonic_k == KH)].iloc[0]
    tl_txt = (f"patient k{KH} within-session trial-level slope "
              f"{tlr.within_slope:+.2f} $\\pm$ {tlr.within_se:.2f} (s.e.), "
              f"{pfmt(float(tlr.within_p))} "
              f"({int(tlr.n_trials)} trials, {int(tlr.n_sessions)} sessions)")

    print(f"[build_id] {BUILD_ID}  cohort = {COHORT} ({len(POST_OPERATIVE)} sessions, 2 patients)")
    print(f"Pearson  r = {pr:.4f}, P = {pp:.4g}")
    print(f"Spearman rho = {sr:.4f}, P = {sp:.4g}")
    print(f"patient-adjusted OLS slope = {b_fe:+.4f}, P = {p_fe:.4g}")
    print(f"LME slope (patient RE) = {b_lme:+.4f} +/- {se_lme:.4f}, Wald P = {p_lme:.4g}")
    print(f"LOO Pearson: r {min(loo_r):.3f}-{max(loo_r):.3f}, P {min(loo_p):.4f}-{max(loo_p):.4f}")
    print(f"within-patient exact permutation ({n_perm}): P = {perm_p:.4f}")
    print(f"per-patient Spearman: P1 {per_pat['P1'][0]:+.2f} (n={per_pat['P1'][2]}), "
          f"P2 {per_pat['P2'][0]:+.2f} (n={per_pat['P2'][2]})")
    print(f"trial-level slope: {tl_txt}")

    if PANEL_MODE:
        import si_panel_style as SI
        SI.apply(plt)
        plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
        fig, ax = SI.panel(plt, "quarter", 28.0, left_mm=11.6, bottom_mm=8.6, top_mm=1.0, right_mm=1.8)
        _S_PT, _EW, _LW_FIT, _FS_LAB = 6.0, 0.3, 0.6, SI.TITLE_PT
    else:
        fig, ax = S.panel_axes(plt)
        _S_PT, _EW, _LW_FIT, _FS_LAB = 22, 0.6, 1.1, S.BASE_PT
    for s, x, y in zip(POST_OPERATIVE, xs, ys):
        c = P1C if s.startswith("pat01") else P2C
        ax.scatter(x, y, s=_S_PT, facecolor=c, edgecolor=S.INK, lw=_EW, zorder=3)
    xr = np.linspace(xs.min() - 0.01, xs.max() + 0.01, 100)
    V = np.asarray(lme.cov_params())[:2, :2]
    a0 = float(lme.params["Intercept"])
    yhat = a0 + b_lme * xr
    Xd = np.column_stack([np.ones_like(xr), xr])
    se_line = np.sqrt(np.einsum("ij,jk,ik->i", Xd, V, Xd))
    ax.fill_between(xr, yhat - 1.96 * se_line, yhat + 1.96 * se_line,
                    color=S.MID, alpha=0.16, lw=0, zorder=1)
    ax.plot(xr, yhat, color=S.MID, lw=_LW_FIT, ls=(0, (3, 2)), zorder=2)
    ax.set_xlabel(r"$\mathrm{k}_2$ phase locking $|c|$", fontsize=_FS_LAB, labelpad=1.5)
    ax.set_ylabel("Decoding\naccuracy ($r$)" if PANEL_MODE else "Decoding accuracy ($r$)", fontsize=_FS_LAB,
                  labelpad=2)
    (SI.finish if (PANEL_MODE) else S.finish)(ax)
    if not PANEL_MODE:
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=S.DPI, facecolor="white")
        plt.close(fig)

    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    figL = plt.figure(figsize=(34 * S.MM, 12 * S.MM), dpi=S.DPI)
    ms = 2 * (22 ** 0.5) / 2.0
    _mew, _llw, _lfs, _ldpi = 0.6, 1.1, S.NOTE_PT, S.DPI
    hh = [(Line2D([0], [0], marker="o", color="none", mfc=P1C, mec=S.INK, mew=_mew, ms=ms),
           Line2D([0], [0], marker="o", color="none", mfc=P2C, mec=S.INK, mew=_mew, ms=ms)),
          Line2D([0], [0], color=S.MID, lw=_llw, ls=(0, (3, 2)))]
    figL.legend(handles=hh, labels=["sessions, P1 · P2", "LME fit ± 95% CI"], loc="center",
                fontsize=_lfs, frameon=True, edgecolor="#1A1A1A", facecolor="white",
                fancybox=False, framealpha=1.0, handletextpad=0.5, labelspacing=0.35,
                handlelength=2.0, borderpad=0.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    figL.legends[-1].get_frame().set_linewidth(0.5)
    LEG_OUT = (ec.scoped(PANEL_DIR / "S1h_locking_vs_transfer_legend.png") if PANEL_MODE
               else OUT_PNG.with_name("fig6_locking_vs_transfer_legend.png"))
    LEG_OUT.parent.mkdir(parents=True, exist_ok=True)
    figL.savefig(LEG_OUT, bbox_inches="tight", pad_inches=0.02, dpi=_ldpi, facecolor="white")
    plt.close(figL)
    print("wrote", LEG_OUT.name, "(legend strip)")

    HEAD = [
        f"LME slope = {b_lme:+.2f} $\\pm$ {se_lme:.2f} (s.e.), {pfmt(p_lme)}",
        f"Pearson $r$ = {pr:+.2f}, {pfmt(pp)}",
        f"Within-patient exact permutation {pfmt(perm_p)}",
        f"$n$ = {len(POST_OPERATIVE)} sessions",
    ]
    STATS_LINES = [
        f"Fig.6 panel g (k{KH} phase locking |c| vs transfer decoding) "
        "",
        f"build_id {BUILD_ID} · cohort {COHORT} ({len(POST_OPERATIVE)} sessions / 2 patients) · DESCRIPTIVE",
        "",
        "HEADLINE (the fitted line drawn in the panel):",
        f"  LME slope (patient random intercept) = {b_lme:+.2f} $\\pm$ {se_lme:.2f} (s.e.), "
        f"Wald {pfmt(p_lme)}",
        
        
        "",
        "supporting:",
        f"Pearson  $r$   = {pr:+.2f}, {pfmt(pp)}",
        f"Spearman $\\rho$ = {sr:+.2f}, {pfmt(sp)}",
        f"patient-adjusted OLS slope (y ~ x + patient) = {b_fe:+.2f}, {pfmt(p_fe)}",
        f"within-patient exact permutation ({n_perm} = 6! x 5!): {pfmt(perm_p)}",
        f"leave-one-session-out Pearson: r {min(loo_r):+.2f} to {max(loo_r):+.2f}, "
        f"p {max(min(loo_p), 0.001):.3f} to {max(loo_p):.3f}",
        f"per-patient Spearman: P1 {per_pat['P1'][0]:+.2f} (n={per_pat['P1'][2]}), "
        f"P2 {per_pat['P2'][0]:+.2f} (n={per_pat['P2'][2]})",
        "",
        
        
        
        
        
        "  ",
        
        "",
        f"per-session points (x = k{KH} |c| session mean, y = trial-median r, n = trials):",
        *[f"  {s:<14s} {lock[s][0]:.4f}  {dec[s]:+.4f}  n={lock[s][1]:d}" for s in POST_OPERATIVE],
        "",
        "leave-one-session-out Pearson (session dropped, r, p):",
        *[f"  {s:<14s} {r_i:+.4f}  {p_i:.4f}" for s, r_i, p_i in loo],
    ]
    if not PANEL_MODE:
        S.write_stats(STATS, STATS_LINES)

    if PANEL_MODE:
        _lo, _hi = ax.get_ylim()
        ax.set_ylim(_lo, _hi + 0.22 * (_hi - _lo))
        S.stat_note(ax, [f"$\\it{{r}}$ = {pr:+.2f}, {S.pstr(pp)}"], loc="upper left", fontsize=SI.TICK_PT)
        PANEL_DIR.mkdir(parents=True, exist_ok=True)
        _pp = ec.scoped(PANEL_DIR / "S1h_locking_vs_transfer.png")
        SI.save(fig, _pp)
        plt.close(fig)
        export_stats(HEAD, "S1", "h", "locking_vs_transfer", width=4.4)
        ec.scoped(PANEL_DIR / "S1h_locking_vs_transfer_stats.txt").write_text(
            "\n".join(HEAD) + "\n", encoding="utf-8")
        print("wrote", ec.scoped(PANEL_DIR / "S1h_locking_vs_transfer_stats.txt"))
        print("[panel mode] figures_out/fig6 outputs and both Source Data CSVs left untouched")
        return

    with open(SRC, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "patient", "cohort", "k2_locking_c_mean", "n_trials",
                    "decode_trial_median_r"])
        for s, pid, n in zip(POST_OPERATIVE, pats, ns):
            w.writerow([s, pid, COHORT, f"{lock[s][0]:.6f}", n, f"{dec[s]:.6f}"])

    with open(META, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerows([
            ["key", "value"],
            ["analysis", f"per-session k{KH} gait-phase-locking |c| vs transfer decoding "
                         f"trial-median r, {len(POST_OPERATIVE)} post-operative sessions / 2 SCI patients"],
            ["build_id", BUILD_ID],
            ["cohort", COHORT],
            ["cohort_definition", "post_operative = pat01 sess04-09 + pat02 sess03/05/06/08/09 "
                                  "(11 sessions). "
                                  
                                  
                                  "18 recorded "
                                  "- 3 excluded = 15 retained = pre_operative (4) + post_operative (11)."],
            ["x_metric", f"session-mean c_mi at harmonic_k == {KH} over that session's "
                         f"imagined-walking trials (gait-phase-locking magnitude |c|)"],
            ["y_metric", "transfer decoding trial-median r (frozen all-80-healthy pooled shared "
                         f"PCA30 basis + patient-trained 10-fold-CV readout)"],
            ["x_source", f"{PC.relative_to(ROOT).as_posix()} + sidecars "
                         + "; ".join(f"{s}: {p.relative_to(ROOT).as_posix()}"
                                     for s, p in SIDECARS.items())],
            ["y_source", f"{DEC_SD.relative_to(ROOT).as_posix()}, column `median`, "
                         "cohort == post_operative"],
            ["n_sessions", len(POST_OPERATIVE)],
            ["n_patients", 2],
            ["headline_statistic", "LME slope (patient random intercept) — the fitted line drawn "
                                   "in the panel. Pearson / Spearman / "
                                   "patient-adjusted OLS / LOO / exact permutation are SUPPORTING."],
            ["lme_slope_patient_random_intercept", f"{b_lme:.6f}"],
            ["lme_slope_se", f"{se_lme:.6f}"],
            ["lme_slope_wald_p", f"{p_lme:.6f}"],
            ["lme_model", "statsmodels MixedLM y ~ x, groups = patient (random intercept), REML; "
                          "2 clusters -> descriptive, like every number in this panel"],
            ["lme_drawn_line", "fixed-effect (population) line: intercept = the mean of the two "
                               "patient random intercepts (statsmodels `Intercept`), slope = the "
                               "fixed-effect slope above; band = 95% CI of that fixed-effects mean "
                               "line, var(y_hat) = x' V x with V the (Intercept, x) block of "
                               "lme.cov_params(), normal quantile 1.96 (statsmodels reports Wald z "
                               "for MixedLM). Same construction as the sibling panel's fit."],
            ["pearson_r", f"{pr:.6f}"],
            ["pearson_p", f"{pp:.6f}"],
            ["spearman_rho", f"{sr:.6f}"],
            ["spearman_p", f"{sp:.6f}"],
            ["ols_patient_adjusted_slope", f"{b_fe:.6f}"],
            ["ols_patient_adjusted_slope_p", f"{p_fe:.6f}"],
            ["ols_patient_adjusted_model", "y ~ x + C(patient)"],
            ["loo_pearson_r_range", f"{min(loo_r):.6f} to {max(loo_r):.6f}"],
            ["loo_pearson_p_range", f"{min(loo_p):.6f} to {max(loo_p):.6f}"],
            ["loo_pearson_per_session", "; ".join(f"{s} r={r_i:.4f} P={p_i:.4f}"
                                                  for s, r_i, p_i in loo)],
            ["perm_within_patient_p", f"{perm_p:.6f}"],
            ["perm_within_patient_scheme", f"EXACT label permutation of y across sessions WITHIN "
                                           f"each patient (6! x 5! = {n_perm}), two-sided on "
                                           f"|Pearson r| >= |observed|; holds the between-patient "
                                           f"offset fixed"],
            ["per_patient_spearman", "; ".join(
                f"{pid} rho={per_pat[pid][0]:.4f} P={per_pat[pid][1]:.4f} n={per_pat[pid][2]}"
                for pid in ("P1", "P2"))],
            ["scope", f"n = 2 SCI patients, {len(POST_OPERATIVE)} post_operative sessions, case-level, "
                      f"post-operative, DESCRIPTIVE; NOT a population claim"],
            ["story_link", "patient counterpart of Fig.3f / Fig.4c (locking strength indexes "
                           "decodability), asked at the harmonic that transfers to patients "
                           "(k2, Fig.6f)"],
            ["source_script", "03_figures/fig6_locking_vs_transfer.py"],
        ])
    print(f"wrote {SRC.name} and {META.name}")


if __name__ == "__main__":
    main()
