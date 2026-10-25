"""Figure 3 — Gait-phase locking of the imagined-walking representation

panels i + h — self-report 6-item forest + Q6 carrier scatter

Reads:
  04_source_data/fig3/fig3_master_source_data.csv
  04_source_data/fig3/fig3_def_resnonres_source_data.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and does not fit the decoder. It fits a least-squares regression
for the statistics reported with that panel or table.
"""

from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, false_discovery_control

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 1.3})
ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data" / "fig3"
OUT = ROOT / "figures_out" / "fig3" / "fig3_h_selfreport_forest.png"
GREY = "#9aa0a6"; SIG = "#00568A"
import project_palette as _P
RESP, NON, RESP_EDGE = _P.RESPONDER_FACE, _P.NON_RESPONDER_FACE, _P.IW_HUE
def p_mathtext(p):
    """Format a p value for on-figure display as mathtext: 1.1 x 10^-27, or P < 0.001 style."""
    import math
    if p == 0 or not math.isfinite(p):
        return r"$<10^{-300}$"
    e = int(math.floor(math.log10(abs(p))))
    m = p / 10 ** e
    if e >= -2:
        return f"{p:.3f}".rstrip("0")
    return rf"${m:.1f}\times10^{{{e}}}$"


rng = np.random.default_rng(0)

_SD_M = SD / "fig3_master_source_data.csv"
_SD_R = SD / "fig3_def_resnonres_source_data.csv"
m = pd.read_csv(_SD_M)
resp = pd.read_csv(_SD_R)[["session", "resp"]]
m = m.merge(resp, on="session", how="left"); m["resp"] = m.resp.fillna(False).astype(bool)
ITEMS = [("Q1_intent", "intent to move"), ("Q2_sync", "sync with walker"), ("Q3_still", "stayed still (0-1)"),
         ("Q4_pace", "matched pace"), ("Q5_relax", "felt relaxed"), ("Q6_avoid", "volitional control")]


def boot_rho(x, y, n=1000):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    r0, p = spearmanr(x, y); i = np.arange(len(x))
    rs = np.array([spearmanr(x[s], y[s])[0] for s in (rng.choice(i, len(i), True) for _ in range(n))])
    rs = rs[np.isfinite(rs)]
    lo, hi = (np.percentile(rs, [2.5, 97.5]) if rs.size else (np.nan, np.nan))
    return r0, lo, hi, p, len(x)


rows = [(col, lab, *boot_rho(m[col], m.excess_k1)) for col, lab in ITEMS]
rows.sort(key=lambda r: r[2])
qvals = false_discovery_control([r[5] for r in rows], method="bh")

OUT_K = ROOT / "figures_out" / "fig3" / "fig3_k_q6_scatter.png"

MM  = 1/25.4
FS  = 1.2
INK, MID, PALE = "#1A1A1A", "#666666", "#B3B3B3"
plt.rcParams.update({"font.family": "Arial", "font.size": 7*FS, "axes.linewidth": 1.4,
                     "xtick.major.width": 1.4, "ytick.major.width": 1.4,
                     "xtick.major.size": 3.0, "ytick.major.size": 3.0, "pdf.fonttype": 42})
def _stars(q): return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"
SHORT = {"volitional control": "Volitional control", "felt relaxed": "Felt relaxed",
         "stayed still (0-1)": "Stayed still", "intent to move": "Intent to walk",
         "matched pace": "Matched pace", "sync with walker": "Sync with walker"}

_CI_LW, _CAP, _DOT_MS, _DOT_MEW, _ZERO = ((1.6, .16, 5.0, 1.4, (1.1, (0, (2, 2)))))
figf, axf = plt.subplots(figsize=(56*MM, 56*MM), dpi=400)
axf.axvline(0, color=PALE, lw=_ZERO[0], ls=_ZERO[1], zorder=0)
for i, (col, lab, r0, lo, hi, p, n) in enumerate(rows):
    sig = np.isfinite(qvals[i]) and qvals[i] < 0.05
    c = INK if sig else MID
    if np.isfinite(lo):
        axf.plot([lo, hi], [i, i], color=c, lw=_CI_LW, solid_capstyle="butt", zorder=2)
        for xx in (lo, hi):
            axf.plot([xx, xx], [i - _CAP, i + _CAP], color=c, lw=_CI_LW, solid_capstyle="butt", zorder=2)
    axf.plot(r0, i, "o", ms=_DOT_MS, mfc=c if sig else "white", mec=c, mew=_DOT_MEW, zorder=3)
    _st = _stars(qvals[i])
    axf.text(1.04, i, _st, transform=axf.get_yaxis_transform(),
             ha="left", va="center",
             fontsize=11.3*FS if _st != "ns" else 11.3*FS*0.65, color=INK)
axf.set_yticks(range(len(rows)))
axf.set_yticklabels([SHORT.get(r[1], r[1]) for r in rows], fontsize=7*FS)
axf.set_xlim(-0.34, 0.74); axf.set_ylim(-0.55, len(rows) - 0.45)
axf.set_xticks([0, 0.25, 0.5])
axf.spines[["top", "right"]].set_visible(False)
axf.set_xlabel("Spearman $\\rho$ with\ngait-phase locking |c| ($\\mathrm{k}_1$)", fontsize=7*FS, labelpad=1.5)
axf.set_box_aspect(1)
figf.savefig(OUT, dpi=400, bbox_inches="tight", facecolor="white", pad_inches=0.04)
plt.close(figf)

figs, axs = plt.subplots(figsize=(56*MM, 56*MM), dpi=400)
_FIT_LW, _FIT_LS, _PT_S, _PT_LW = ((1.3, (0, (4, 2)), 14, .5))
x, y, rp = m.Q6_avoid.values, m.excess_k1.values, m.resp.values
ok = np.isfinite(x) & np.isfinite(y); x, y, rp = x[ok], y[ok], rp[ok]
xs = np.linspace(x.min(), x.max(), 60); idx = np.arange(len(x))
lines = np.array([np.polyval(np.polyfit(x[s], y[s], 1), xs)
                  for s in (rng.choice(idx, len(idx), True) for _ in range(1000))])
axs.fill_between(xs, np.percentile(lines, 2.5, 0), np.percentile(lines, 97.5, 0),
                 color="#9A9A9A", alpha=0.20, lw=0, zorder=1)
axs.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=INK, lw=_FIT_LW, ls=_FIT_LS, zorder=2)
axs.scatter(x[~rp], y[~rp], s=_PT_S, facecolor=NON, edgecolor=RESP_EDGE, lw=_PT_LW, zorder=3)
axs.scatter(x[rp],  y[rp],  s=_PT_S, facecolor=RESP, edgecolor=RESP_EDGE, lw=_PT_LW, zorder=4)
q6 = boot_rho(m.Q6_avoid, m.excess_k1)
axs.set_xlabel("Volitional imagery control", fontsize=7*FS, labelpad=1.5)
axs.set_ylabel("Rest-corrected locking |c| ($\\mathrm{k}_1$)", fontsize=7*FS, labelpad=2)
axs.set_xticks([5, 6, 7]); axs.set_yticks([0, .10, .20])
axs.spines[["top", "right"]].set_visible(False)
axs.set_box_aspect(1)
figs.savefig(OUT_K, dpi=400, bbox_inches="tight", facecolor="white", pad_inches=0.04)
plt.close(figs)

pd.DataFrame([(c, l, r, lo, hi, p, q, n, int(np.isfinite(q) and q < 0.05)) for (c, l, r, lo, hi, p, n), q in zip(rows, qvals)],
            columns=["item", "label", "rho", "ci_lo", "ci_hi", "p_spearman", "q_bh_fdr", "n", "sig_fdr"]).to_csv(
    SD / "fig3_h_forest_stats.csv", index=False)
m[["session", "subj_int", "Q1_intent", "Q2_sync", "Q3_still", "Q4_pace", "Q5_relax", "Q6_avoid", "excess_k1", "resp"]].rename(
    columns={"subj_int": "subject", "resp": "responder"}).to_csv(SD / "fig3_h_scatter_points.csv", index=False)
print(f"wrote {OUT}")
print(f"  scatter stats: rho = {q6[0]:.2f} [{q6[1]:.2f}, {q6[2]:.2f}], "
      f"P = {p_mathtext(q6[3])}  (exact rho {q6[0]:.6f}, raw P {q6[3]:.4g})")
print("  forest (BH-FDR):", ", ".join(f"{r[0]}={r[2]:+.2f}(q={q:.3f}){'*' if (np.isfinite(q) and q<0.05) else ''}" for r, q in zip(rows, qvals)))
