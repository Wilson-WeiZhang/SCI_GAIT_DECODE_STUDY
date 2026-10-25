"""Figure 6 — Representational geometry of patient imagined walking

panel b — trial-level healthy-likeness (per-trial excess similarity)

Reads:
  decoder output in data/decoder_output/ (not distributed)

Writes:
  figures_out/fig6/fig6_trial_excess.png
  04_source_data/fig6_trial_excess_{source_data,metadata}.csv
  

Renders one panel or table from the inputs above; it performs no
preprocessing and does not fit the decoder. It fits a linear mixed model
for the statistics reported with that panel or table.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

import fig6_panel_style as S

ROOT = Path(__file__).resolve().parents[1]
CSV = (ROOT / "data/decoder_output/diag_pertrial_phasor_sim_pat14"
       / "pertrial_phasor_similarity.csv")
CSV_S09 = (ROOT / "data/decoder_output/diag_pertrial_phasor_sim_sess09"
           / "pertrial_phasor_similarity.csv")
DIAG_TAG = ("diag_pertrial_phasor_sim_pat14 + pat02_sess09 sidecar "
            "diag_pertrial_phasor_sim_sess09")
POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
            "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
            "pat02_sess09"]
OUT = ROOT / "figures_out/fig6/fig6_trial_excess.png"
STATS = ROOT / "figures_out/fig6/fig6_trial_excess_stats.txt"
SRC = ROOT / "04_source_data/fig6/fig6_trial_excess_source_data.csv"
META = ROOT / "04_source_data/fig6/fig6_trial_excess_metadata.csv"
P1C, P2C, GREY = "#D3B8A2", "#E5E7CD", "#9aa0a6"
KD = {1: r"$\mathrm{k}_1$", 2: r"$\mathrm{k}_2$", 3: r"$\mathrm{k}_3$"}

S.apply(plt)

_S_TRIAL, _S_MEAN, _LW_BOX, _LW_MED, _LW_WHI, _LW_DIA, _PT_STAR = (
    (2.0, 11, 1.1, 1.4, 1.0, 1.0, S.STAR_PT))

_main = pd.read_csv(CSV)
_side = pd.read_csv(CSV_S09)
if list(_main.columns) != list(_side.columns):
    raise SystemExit(f"{CSV_S09.name} header differs from {CSV.name} - cannot concatenate")
_side_ids = set(_side.session.unique())
if _side_ids != {"pat02_sess09"}:
    raise SystemExit(f"{CSV_S09.name} must hold exactly pat02_sess09, got: {sorted(_side_ids)}")
if _side_ids & set(_main.session.unique()):
    raise SystemExit("pat02_sess09 is present in BOTH dumps - would double-count")
df = pd.concat([_main, _side], ignore_index=True)
df = df[df.session.isin(POST_OPERATIVE)].copy()
_got = sorted(df.session.unique())
if _got != sorted(POST_OPERATIVE):
    raise SystemExit(f"post_operative sessions missing from the dumps: "
                     f"{sorted(set(POST_OPERATIVE) - set(_got))}")
df["excess"] = df["sim"] - df["null_mean"]
df["above"] = df["null_p"] < 0.05
df["patient"] = np.where(df.session.str.startswith("pat01"), "P1", "P2")

rng = np.random.default_rng(20260831)
fig, ax = S.panel_axes(plt)
stat_lines =["Fig. 6 panel b (trial-level excess similarity)"]
for i, k in enumerate((1, 2, 3)):
    sub = df[df.harmonic_k == k]
    assert len(sub) == len(df) // 3, f"k{k}: {len(sub)} trials != {len(df) // 3}"
    assert sorted(sub.session.unique()) == sorted(POST_OPERATIVE), f"k{k}: session set incomplete"
    x0 = i
    for pat, c, dx in (("P1", P1C, -0.13), ("P2", P2C, 0.13)):
        v = sub[sub.patient == pat]["excess"].values
        ax.scatter(x0 + dx + rng.uniform(-0.08, 0.08, len(v)), v, s=_S_TRIAL, color=c,
                   alpha=0.7, lw=0, zorder=2)
    v = sub["excess"].values
    q10, q25, med, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
    ax.add_patch(plt.Rectangle((x0 - 0.26, q25), 0.52, q75 - q25, fill=False,
                               edgecolor=S.INK, lw=_LW_BOX, zorder=4))
    ax.hlines(med, x0 - 0.26, x0 + 0.26, color=S.INK, lw=_LW_MED, zorder=4)
    for q in (q10, q90):
        ax.hlines(q, x0 - 0.10, x0 + 0.10, color=S.INK, lw=_LW_WHI, zorder=4)
    ax.vlines([x0], q10, q25, color=S.INK, lw=_LW_WHI, zorder=4)
    ax.vlines([x0], q75, q90, color=S.INK, lw=_LW_WHI, zorder=4)
    sm_ = sub.groupby("session")["excess"].mean()
    for s_, m_ in sm_.items():
        ax.scatter(x0 + (-0.13 if s_.startswith("pat01") else 0.13), m_, marker="D", s=_S_MEAN,
                   facecolor=P1C if s_.startswith("pat01") else P2C, edgecolor=S.INK,
                   lw=_LW_DIA, zorder=5)
    md = smf.mixedlm("excess ~ 1", sub, groups=sub["session"]).fit(reml=True)
    b = float(md.params["Intercept"])
    wt = wilcoxon(sm_.values, alternative="two-sided", mode="exact")
    assert (sm_.values > 0).all(), f"k{k}: not every session mean excess is positive"
    assert wt.pvalue < 0.05, f"k{k}: session-level exact Wilcoxon P = {wt.pvalue:.4f}"
    pct = 100 * sub["above"].mean()
    _star = "***" if wt.pvalue < 1e-3 else "**" if wt.pvalue < 1e-2 else "*" if wt.pvalue < 0.05 else "n.s."
    ax.text(x0, 0.995, _star, ha="center", va="top", fontsize=_PT_STAR,
            transform=ax.get_xaxis_transform())
    _n_sess = len(sm_)
    stat_lines.append(f"k{k}: {pct:.0f}% of trials > own null (exact permutation P<.05); "
                      f"LME intercept +{b:.3f}; session-level exact Wilcoxon P = {wt.pvalue:.4f} "
                      f"({_n_sess}/{_n_sess} positive — exact-test floor at n = {_n_sess} is "
                      f"{2 / 2 ** _n_sess:.5f}; star {_star} by the exact P)")

ax.axhline(0, color=S.PALE, lw=1.1, ls=(0, (2, 2)), zorder=1)
K_TICK_SCALE = 33 / 27
ax.set_xticks([0, 1, 2], [KD[k] for k in (1, 2, 3)], fontsize=S.TICK_PT * K_TICK_SCALE)
ax.set_ylabel("Trial excess similarity", fontsize=S.BASE_PT, labelpad=2)
ax.set_xlim(-0.55, 2.55)
S.finish(ax)
fig.savefig(OUT, dpi=S.DPI, facecolor="white")

figL, axL = plt.subplots(figsize=(40 * S.MM, 14 * S.MM), dpi=S.DPI)
axL.axis("off")
hh = [axL.scatter([], [], s=10, color=P1C, lw=0, label="P1 trials"),
      axL.scatter([], [], s=10, color=P2C, lw=0, label="P2 trials"),
      axL.scatter([], [], marker="D", s=12, facecolor="#D9D9D9", edgecolor=S.INK, lw=1.0,
                  label="session mean (patient colour)")]
axL.legend(handles=hh, loc="center", ncol=3, fontsize=S.NOTE_PT, frameon=False,
           handletextpad=0.4, labelspacing=0.35, columnspacing=1.6)
figL.savefig(OUT.with_name("fig6_trial_excess_legend.png"), bbox_inches="tight", dpi=S.DPI,
             facecolor="white")
plt.close(figL)
print("wrote fig6_trial_excess_legend.png (legend strip)")
stat_lines.append("zero line = each trial's own channel-shuffle chance level")
S.write_stats(STATS, stat_lines)

rows_out = df[["session", "patient", "harmonic_k", "trial_ord", "sim", "null_mean",
               "null_p95", "null_p", "excess", "above"]]
rows_out.to_csv(SRC, index=False, float_format="%.6f")
stats = []
for k in (1, 2, 3):
    sub = df[df.harmonic_k == k]
    import statsmodels.formula.api as _smf
    md = _smf.mixedlm("excess ~ 1", sub, groups=sub["session"]).fit(reml=True)
    sm_ = sub.groupby("session")["excess"].mean()
    wt = wilcoxon(sm_.values, alternative="two-sided", mode="exact")
    stats.append((k, float(md.params["Intercept"]), float(md.bse["Intercept"]),
                  float(md.pvalues["Intercept"]), float(wt.pvalue),
                  100 * sub["above"].mean(), 100 * (sub["sim"] > sub["null_p95"]).mean()))
with open(META, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["key", "value"])
    w.writerow(["build_id", "main"])
    _n_tr = len(df) // 3
    _n_se = df.session.nunique()
    w.writerow(["cohort", f"post_operative ({_n_tr} trials / {_n_se} sessions / 2 patients)"])
    w.writerow(["diag_tag", DIAG_TAG])
    for k, b, se, pw, wp, pct, pct95 in stats:
        w.writerow([f"k{k}_lme_intercept_se_waldp", f"+{b:.6f} / {se:.6f} / {pw:.3e}"])
        w.writerow([f"k{k}_wilcoxon_session_exact_p",
                    f"{wp:.6f} ({_n_se}/{_n_se} positive; exact-test FLOOR at n={_n_se} is "
                    f"{2 / 2 ** _n_se:.5f} -> *** since the exact P < .001)"])
        w.writerow([f"k{k}_pct_above_own_null", f"{pct:.1f} (criterion null_p<.05; "
                    f"sim>null_p95 gives {pct95:.1f})"])
    w.writerow(["stars_convention", f"two-sided exact Wilcoxon on {_n_se} session means vs 0, "
                "sessions as units, case-level descriptive (movement-free panel idiom); "
                f"LME Wald P caption/metadata only, scoped {_n_se} clusters / 2 patients"])
    
                
    w.writerow(["claim_note", "first-order sharing at ALL three harmonics (largest k2); "
                "k2-selectivity belongs to second-order geometry (Fig. 6f, on the matrices of Fig. 6e) and "
                "cross-harmonic specificity (Fig. 6d)"])
    
    w.writerow(["scope", "n=2 SCI patients, case-level, descriptive; decode-independent"])
    w.writerow(["paper_location", "Fig. 6 panel b"])
print(f"wrote {OUT} + Source Data")
