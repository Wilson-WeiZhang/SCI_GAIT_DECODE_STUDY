"""Supplementary item (Figure 4 companion)

Each point = one participant (mean over that participant's sessions) - the same per-subject values
panel a aggregates (fig4_decoding_gain_source_data.csv (not distributed), written by
fig4_decoding_gain.py)

Renders one panel or table; it performs no
preprocessing and does not fit the decoder. It fits a least-squares regression
for the statistics reported with that panel or table.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import panel_style as PS
PS.apply(plt)
from sup_panel_export import PANEL_MODE, PANEL_DIR, export_stats
import analysis_config as ec

plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.linewidth": 2.2,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
if PANEL_MODE:
    import si_panel_style as SI
    SI.apply(plt)
ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data/supplementary"
OUT = ec.scoped(ROOT / "figures_out/supplementary/sup_fig4_participant_iw_rw.png")
BLUE, GREY = "#0072B2", "#9A9A9A"

IN = ec.sd("fig4_decoding_gain_source_data.csv")
d = pd.read_csv(IN, comment="#")
x, y = d.IW.to_numpy(), d.RW.to_numpy(); n = len(d)
rho, p = stats.spearmanr(x, y)
col = BLUE if p < 0.05 else GREY
rng = np.random.default_rng(0)
xs = np.linspace(x.min(), x.max(), 50); idx = np.arange(n)
lines = np.array([np.polyval(np.polyfit(x[s], y[s], 1), xs) for s in (rng.choice(idx, n, True) for _ in range(1000))])

if PANEL_MODE:
    fig, ax = SI.panel(plt, "third", 48.0, left_mm=11.0)
else:
    fig, ax = PS.panel_axes(plt, left_mm=15.5, bottom_mm=13.0)
ax.fill_between(xs, np.percentile(lines, 2.5, 0), np.percentile(lines, 97.5, 0),
                color="#9A9A9A", alpha=0.20, lw=0, zorder=2)
ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color="#1A1A1A",
        lw=SI.LW_MAIN if PANEL_MODE else 1.3, ls=(0, (4, 2)), zorder=4)
ax.scatter(x, y, s=4.0 if PANEL_MODE else 5, color=col, alpha=0.85, edgecolors="#1A1A1A",
           lw=SI.LW_MINOR if PANEL_MODE else 0.4, zorder=3)
_ylo, _yhi = ax.get_ylim()
ax.set_ylim(_ylo, _yhi + 0.28 * (_yhi - _ylo))
STAT_NOTE = [f"$\\rho$ = {rho:.2f}", f"{PS.pstr(p)}, $n$ = {n}"]
PS.stat_note(ax, STAT_NOTE, loc="upper left",
             fontsize=SI.TICK_PT if PANEL_MODE else 12)
ax.set_xlabel("Imagined walking $r$")
ax.set_ylabel("Real walking $r$")
PS.finish(ax)
if PANEL_MODE:
    SI.finish(ax)
    assert ax.get_title() == "" and ax.get_legend() is None, "panel carries a title or a legend"
    assert [t.get_text() for t in ax.texts] == ["\n".join(STAT_NOTE)], \
        f"panel carries unexpected in-axes text: {[t.get_text() for t in ax.texts]}"
    assert list(fig.texts) == [], f"panel carries figure text: {[t.get_text() for t in fig.texts]}"
    print("spec-check S5b: title empty, no legend, ax.texts == [rho/p note], fig.texts == []  -> PASS")
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    _pp = ec.scoped(PANEL_DIR / "S5b_participant_iw_rw.png")
    SI.save(fig, _pp)
    _lines = [
        "Spearman $\\rho$ = %.3f, %s, $n$ = %d participants"
        % (rho, "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}", n),
        "Dashed line: ordinary-least-squares fit. Grey band: 95 % percentile interval of that",
        "fit over 1000 bootstrap resamples of the participants (seed 0).",
    ]
    export_stats(_lines, "S5", "b", "participant_iw_rw", width=5.6)
    ec.scoped(PANEL_DIR / "S5b_participant_iw_rw_stats.txt").write_text(
        "\n".join(_lines) + "\n", encoding="utf-8")
    print("wrote", ec.scoped(PANEL_DIR / "S5b_participant_iw_rw_stats.txt"))
else:
    fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

if PANEL_MODE:
    print(f"n={n} rho={rho:.3f} p={p:.3g}")
    raise SystemExit(0)
d.to_csv(ec.sd("sup_fig4_participant_iw_rw_source_data.csv"), index=False)
pd.DataFrame([
    ("input", IN.relative_to(ROOT).as_posix() + " (fig4_decoding_gain.py)"),
    ("spearman_rho", f"{rho:.4f}"), ("spearman_p", f"{p:.3g}"), ("n_participants", n),
    ("band", "OLS line + 95% bootstrap band (1000 resamples, seed 0)"),
], columns=["key", "value"]).to_csv(ec.sd("sup_fig4_participant_iw_rw_metadata.csv"), index=False)
print(f"n={n} rho={rho:.3f} p={p:.3g}"); print("wrote", OUT.relative_to(ROOT))

if ec.BUILD_ID in ec.DECODER_BUILDS:
    PS.apply(plt)
    figj, axj = PS.panel_axes(plt, left_mm=15.0, bottom_mm=12.0, top_mm=6.0)
    _lo, _hi = np.percentile(lines, 2.5, 0), np.percentile(lines, 97.5, 0)
    _fit = np.polyval(np.polyfit(x, y, 1), xs)
    axj.fill_between(xs, _lo, _hi, color="#9A9A9A", alpha=0.20, lw=0, zorder=2)
    axj.plot(xs, _fit, color=PS.INK, lw=1.3, ls=(0, (4, 2)), zorder=4)
    axj.scatter(x, y, s=5, color=PS.MID, alpha=0.85, lw=0, zorder=3)
    _ylo, _yhi = axj.get_ylim()
    axj.set_ylim(_ylo, _yhi + 0.28 * (_yhi - _ylo))
    PS.stat_note(axj, [f"Spearman $\\rho$ = {rho:.2f}", f"{PS.pstr(p)}, $n$ = {n}"], loc="upper left")
    axj.set_xlabel("Imagined walking $r$")
    axj.set_ylabel("Real walking $r$")
    PS.finish(axj)
    _pj = ec.scoped(ROOT / "figures_out/fig4/fig4_participant_iw_rw.png")
    figj.savefig(_pj, dpi=PS.DPI, facecolor="white")
    plt.close(figj)
    d[["subject", "IW", "RW"]].to_csv(ec.sd("fig4_participant_iw_rw_source_data.csv"), index=False)
    pd.DataFrame({"x_IW_r": xs, "fit_RW_r": _fit, "ci95_lo": _lo, "ci95_hi": _hi}).to_csv(
        ec.sd("fig4_participant_iw_rw_fit.csv"), index=False, float_format="%.6f")
    pd.DataFrame([
        ("output", _pj.relative_to(ROOT).as_posix()),
        ("input", IN.relative_to(ROOT).as_posix() + " (fig4_decoding_gain.py)"),
        ("r_convention", "per-participant mean of that participant's per-session TRIAL MEDIANS (per-session "
                         "median over trials of the six-joint trial r; the input's subject_unit), i.e. the "
                         "Fig. 4b convention"),
        ("spearman_rho", f"{rho:.4f}"), ("spearman_p", f"{p:.3g}"), ("n_participants", n),
        ("band", "dashed OLS fit + 95% percentile band over 1000 bootstrap resamples of the participants "
                 "(seed 0); values in fig4_participant_iw_rw_fit.csv"),
    ], columns=["key", "value"]).to_csv(ec.sd("fig4_participant_iw_rw_metadata.csv"), index=False)
    print("wrote", _pj.relative_to(ROOT), "+ fig4_participant_iw_rw_* Source Data")
