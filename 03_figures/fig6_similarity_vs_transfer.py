"""Figure 6 — Representational geometry of patient imagined walking

Supplementary panel: k2 phase-map similarity vs transfer decoding. 

Reads:
  04_source_data/fig6/fig6_patient_in_healthy_similarity_source_data.csv
  04_source_data/fig6/fig6_patient_in_healthy_similarity_metadata.csv

Writes:
  figures_out/fig6/fig6_similarity_vs_transfer.png
  04_source_data/fig6_similarity_vs_transfer_{source_data,metadata}.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and does not fit the decoder. It fits a linear mixed model and a least-squares regression
for the statistics reported with that panel or table.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fig6_panel_style as S
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sup_panel_export import PANEL_MODE, PANEL_DIR, export_stats
import analysis_config as ec


def pfmt(p):
    """House style: lower-case italic p, leading zero, floor at 0.001."""
    return "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"

ROOT = Path(__file__).resolve().parents[1]
PIH_SD = ROOT / "04_source_data/fig6/fig6_patient_in_healthy_similarity_source_data.csv"
PIH_META = ROOT / "04_source_data/fig6/fig6_patient_in_healthy_similarity_metadata.csv"
DEC_SD = ec.sd("fig5b_session_box_source_data.csv")
DEC_META = ec.sd("fig5b_session_box_metadata.csv")
OUT_PNG = ec.scoped(ROOT / "figures_out/fig6/fig6_similarity_vs_transfer.png")
STATS = ec.scoped(ROOT / "figures_out/fig6/fig6_similarity_vs_transfer_stats.txt")
SRC = ec.sd("fig6_similarity_vs_transfer_source_data.csv")
META = ec.sd("fig6_similarity_vs_transfer_metadata.csv")

POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
            "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
            "pat02_sess09"]
P1C, P2C = "#D3B8A2", "#E5E7CD"

S.apply(plt)


def main():
    for _m, _kind in ((PIH_META, "neural"), (DEC_META, "decoder")):
        _build_id = next(r[1] for r in csv.reader(open(_m, encoding="utf-8-sig")) if r and r[0] == "build_id")
        ec.check_tag(_build_id, _kind, f"{_m.name} build_id")
    sim = {r["session"]: float(r["value"]) for r in csv.DictReader(open(PIH_SD))
           if r["row_type"] == "patient_session" and r["harmonic_k"] == "2"
           and r["cohort"] == "post_operative"}
    dec = {r["session"]: float(r["median"]) for r in csv.DictReader(open(DEC_SD))
           if r["cohort"] == "post_operative"}
    xs = [sim[s] for s in POST_OPERATIVE]
    ys = [dec[s] for s in POST_OPERATIVE]
    _sp = spearmanr(xs, ys)
    rho, rho_p = float(_sp.statistic), float(_sp.pvalue)
    keep = [s for s in POST_OPERATIVE if s != "pat02_sess03"]
    rho_wo = spearmanr([sim[s] for s in keep], [dec[s] for s in keep]).statistic
    df = pd.DataFrame(dict(session=POST_OPERATIVE, sim=xs, dec=ys))
    df["patient"] = np.where(df.session.str.startswith("pat01"), "P1", "P2")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lme = smf.mixedlm("dec ~ sim", df, groups=df["patient"]).fit(reml=True)
        ols = smf.ols("dec ~ sim + C(patient)", df).fit()
    b, se, pw = (float(lme.params["sim"]), float(lme.bse["sim"]), float(lme.pvalues["sim"]))

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
    xr = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 100)
    V = np.asarray(lme.cov_params())[:2, :2]
    a0 = float(lme.params["Intercept"])
    yhat = a0 + b * xr
    Xd = np.column_stack([np.ones_like(xr), xr])
    se_line = np.sqrt(np.einsum("ij,jk,ik->i", Xd, V, Xd))
    ax.fill_between(xr, yhat - 1.96 * se_line, yhat + 1.96 * se_line,
                    color=S.MID, alpha=0.16, lw=0, zorder=1)
    ax.plot(xr, yhat, color=S.MID, lw=_LW_FIT, ls=(0, (3, 2)), zorder=2)
    ax.set_xlabel(r"$\mathrm{k}_2$ phase-map similarity", fontsize=_FS_LAB, labelpad=1.5)
    ax.set_ylabel("Decoding\naccuracy ($r$)" if PANEL_MODE else "Decoding accuracy ($r$)", fontsize=_FS_LAB,
                  labelpad=2)
    (SI.finish if PANEL_MODE else S.finish)(ax)
    if not PANEL_MODE:
        fig.savefig(OUT_PNG, dpi=S.DPI, facecolor="white")

    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    figL = plt.figure(figsize=(34 * S.MM, 12 * S.MM), dpi=S.DPI)
    ms = 2 * (22 ** 0.5) / 2.0
    hh = [(Line2D([0], [0], marker="o", color="none", mfc=P1C, mec=S.INK, mew=0.6, ms=ms),
           Line2D([0], [0], marker="o", color="none", mfc=P2C, mec=S.INK, mew=0.6, ms=ms)),
          Line2D([0], [0], color=S.MID, lw=1.1, ls=(0, (3, 2)))]
    figL.legend(handles=hh, labels=["sessions, P1 · P2", "LME fit ± 95% CI"], loc="center",
                fontsize=S.NOTE_PT, frameon=True, edgecolor="#1A1A1A", facecolor="white",
                fancybox=False, framealpha=1.0, handletextpad=0.5, labelspacing=0.35,
                handlelength=2.0, borderpad=0.5, handler_map={tuple: HandlerTuple(ndivide=None)})
    figL.legends[-1].get_frame().set_linewidth(0.5)
    LEG_OUT = (ec.scoped(PANEL_DIR / "S1g_similarity_vs_transfer_legend.png") if PANEL_MODE
               else OUT_PNG.with_name("fig6_similarity_vs_transfer_legend.png"))
    LEG_OUT.parent.mkdir(parents=True, exist_ok=True)
    figL.savefig(LEG_OUT, bbox_inches="tight", pad_inches=0.02, dpi=S.DPI, facecolor="white")
    plt.close(figL)
    print("wrote", LEG_OUT.name, "(legend strip)")
    HEAD = [
        f"LME slope = +{b:.2f} $\\pm$ {se:.2f} (s.e.), {pfmt(pw)} (patients as clusters)",
        f"Spearman $\\rho$ = {rho:.2f}, {pfmt(rho_p)}, $n$ = {len(POST_OPERATIVE)} sessions "
        f"($\\rho$ = {rho_wo:.2f} without one Patient 2 session)",
        "Descriptive: the two axes share the harmonic machinery, so this is an association "
        "display, not an independent-biomarker claim",
    ]
    if not PANEL_MODE:
        S.write_stats(STATS, [
            "Supplementary panel (k2 similarity vs transfer decoding)",
            *HEAD,
            f"{len(POST_OPERATIVE)} sessions / 2 patients",
            "",
            "per-session points (x = k2 similarity, y = trial-median r):",
            *[f"  {s:<14s} {sim[s]:.4f}  {dec[s]:.4f}" for s in POST_OPERATIVE],
        ])

    if PANEL_MODE:
        _lo, _hi = ax.get_ylim()
        ax.set_ylim(_lo, _hi + 0.22 * (_hi - _lo))
        S.stat_note(ax, [f"$\\it{{\\rho}}$ = {rho:.2f}, {S.pstr(rho_p)}"], loc="upper left", fontsize=SI.TICK_PT)
        PANEL_DIR.mkdir(parents=True, exist_ok=True)
        _pp = ec.scoped(PANEL_DIR / "S1g_similarity_vs_transfer.png")
        SI.save(fig, _pp)
        plt.close(fig)
        export_stats(HEAD, "S1", "g", "similarity_vs_transfer", width=6.4)
        ec.scoped(PANEL_DIR / "S1g_similarity_vs_transfer_stats.txt").write_text(
            "\n".join(HEAD) + "\n", encoding="utf-8")
        print("wrote", ec.scoped(PANEL_DIR / "S1g_similarity_vs_transfer_stats.txt"))
        print(f"[panel mode] figures_out/fig6 outputs and both Source Data CSVs left untouched; "
              f"rho {rho:.3f} p {rho_p:.3f}")
        return

    with open(SRC, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "patient", "cohort", "k2_similarity", "decode_trial_median_r"])
        for s in POST_OPERATIVE:
            w.writerow([s, "P1" if s.startswith("pat01") else "P2", "post_operative",
                        f"{sim[s]:.6f}", f"{dec[s]:.6f}"])
    with open(META, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerows([
            ["key", "value"],
            ["build_id", ec.TAG],
            ["cohort", ec.COHORTS["post_operative"]],
            [f"spearman_rho_all{len(POST_OPERATIVE)}", f"{rho:.6f}"],
            ["spearman_rho_without_pat02_sess03", f"{rho_wo:.6f}"],
            ["lme_slope_patient_cluster", f"{b:.6f}"],
            ["lme_slope_se", f"{se:.6f}"],
            ["lme_slope_wald_p", f"{pw:.6f}"],
            ["ols_patientFE_slope_p", f"{float(ols.params['sim']):.6f} / {float(ols.pvalues['sim']):.6f}"],
            ["x_source", "fig6_patient_in_healthy_similarity_source_data.csv (panel d; not recomputed)"],
            ["y_source", "fig5b_session_box_source_data.csv (Fig.5b; not recomputed)"],
            ["scope", f"n=2 SCI patients, {len(POST_OPERATIVE)} post_operative sessions, case-level, DESCRIPTIVE; "
             "shared harmonic machinery — association display, not a biomarker claim"],
            ["paper_location", "Supplementary Fig. 1g (panel S1g_similarity_vs_transfer.png, "
                               "first cited by Supplementary Note 6)"
                               
                               
                               ],
        ])
    print(f"wrote {OUT_PNG.name}; rho all{len(POST_OPERATIVE)} {rho:.3f} / wo sess03 {rho_wo:.3f}")


if __name__ == "__main__":
    main()
