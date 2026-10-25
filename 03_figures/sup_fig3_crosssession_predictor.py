"""Supplementary item (Figure 3 companion)

cross-session control: locking in session 1 against decoding in session 2

Renders one panel or table; it performs no
preprocessing and does not fit the decoder. It fits a least-squares regression
for the statistics reported with that panel or table.
"""

from pathlib import Path
import numpy as np
from sup_panel_export import PANEL_MODE, PANEL_DIR
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import panel_style as PS
PS.apply(plt)
import si_panel_style as SI
from scipy.stats import spearmanr
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from project_palette import KNEE, HIP
import analysis_config as ec

ec.require("main", "main")
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
if PANEL_MODE:
    SI.apply(plt)
    SI.set_line_weight(plt, SI.LW_AXIS_R2)
PANEL_NAME = "S3d_crosssession_predictor"
PANEL_W, PANEL_H = 37.0, 44.0
SUBSCRIPT_PT = 7.0
SD = ROOT / "04_source_data" / "supplementary"
OUT = ec.scoped(ROOT / "figures_out" / "supplementary" / "sup_fig3_crosssession_predictor.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

NAME = "sup_fig3_crosssession_predictor"
MASTER = ec.sd("fig3_master_source_data.csv")
if not MASTER.exists():
    raise SystemExit(
        "\n[{}] INPUT NOT FOUND:\n  {}\n\n".format(NAME, MASTER)
        + "Rebuild it with 03_figures/fig3_build_master.py (healthy cohort).\n"
        
        + "\n"
    )
M = pd.read_csv(MASTER).dropna(subset=["mi_k1", "excess_k1", "r_reference"]).copy()
M["subj"] = M.session.str.extract(r"(subj\d+)")
M["sess"] = M.session.str.extract(r"sess0*(\d+)").astype(int)

rows = []
for s, d in M.groupby("subj"):
    if d.sess.nunique() < 2:
        continue
    d = d.sort_values("sess")
    a, b = d.iloc[0], d.iloc[1]
    rows.append((s, a.mi_k1, a.excess_k1, a.r_reference, b.mi_k1, b.excess_k1, b.r_reference))
P = pd.DataFrame(rows, columns=["subj", "miA", "exA", "rA", "miB", "exB", "rB"])
n = len(P)

rho_ex, p_ex = spearmanr(P.exA, P.rB)
rho_mi, p_mi = spearmanr(P.miA, P.rB)
exX, rX = np.r_[P.exA, P.exB], np.r_[P.rB, P.rA]
rho_pool, p_pool = spearmanr(exX, rX)
rho_within = spearmanr(M.excess_k1, M.r_reference)[0]

if PANEL_MODE:
    fig, ax = SI.panel(plt, PANEL_W, PANEL_H, left_mm=8.1, bottom_mm=9.8,
                       top_mm=1.3, right_mm=1.6)
else:
    fig, ax = PS.panel_axes(plt, left_mm=15.5, bottom_mm=13.0)
fig.patch.set_facecolor("white")
_sc = dict(s=9, linewidths=0.3) if PANEL_MODE else dict(s=5, linewidths=0.4)
ax.scatter(P.exA, P.rB, c=KNEE, alpha=.85, edgecolors="#1A1A1A", zorder=3, **_sc)
_grid = np.linspace(P.exA.min(), P.exA.max(), 200)
_bx = np.polyfit(P.exA, P.rB, 1)
_rng = np.random.default_rng(20260901)
_B = 1000
_draws = np.empty((_B, _grid.size))
for _b in range(_B):
    _pick = _rng.integers(0, len(P), len(P))
    _draws[_b] = np.polyval(np.polyfit(P.exA.to_numpy()[_pick], P.rB.to_numpy()[_pick], 1), _grid)
_lo, _hi = np.percentile(_draws, [2.5, 97.5], axis=0)
ax.fill_between(_grid, _lo, _hi, color="#9A9A9A", alpha=0.20, lw=0, zorder=2)
if PANEL_MODE:
    ax.plot(_grid, np.polyval(_bx, _grid), color=SI.INK, lw=0.75, zorder=4)
else:
    ax.plot(_grid, np.polyval(_bx, _grid), color="#1A1A1A", lw=1.3, ls=(0, (4, 2)), zorder=4)
if PANEL_MODE:
    ax.set_xlabel("Excess |c| at $k_1$\n(session 1)", fontsize=SUBSCRIPT_PT, labelpad=1.5,
                  linespacing=1.0)
    ax.set_ylabel("Decoding $r$ (session 2)", fontsize=SI.TITLE_PT, labelpad=1.5)
else:
    ax.set_xlabel("Session 1 excess |c| ($k_1$)")
    ax.set_ylabel("Session 2 decoding $r$")
if not PANEL_MODE:
    ax.text(.04, .96, f"cross-session ρ={rho_ex:+.2f}\np={p_ex:.1g}  (n={n} subj)\n"
                      f"(within-session ρ={rho_within:+.2f})",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=.3", fc="white", ec="0.8", lw=.6))
if not PANEL_MODE:
    ax.text(.96, .05, "independent sessions",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color="0.4")
_NOTE = rf"$\rho$ = {rho_ex:+.2f}, {PS.pstr(p_ex)}"
if PANEL_MODE:
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
    PS.stat_note(ax, [_NOTE], loc="upper left", fontsize=SI.TICK_PT)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(pad=2.0)
    SI.finish(ax)
else:
    PS.finish(ax)


def _assert_panel_clean(f, axes, name, allow=()):
    """SI panel spec: data + axes + axis titles only (colour-bar tick text exempted; none here)."""
    allow = set(allow)
    bad = []
    for i, a_ in enumerate(axes):
        if any(a_.get_title(loc) for loc in ("center", "left", "right")):
            bad.append(f"axes[{i}] carries a title")
        if a_.get_legend() is not None:
            bad.append(f"axes[{i}] carries a legend")
        stray = [t.get_text() for t in a_.texts if t.get_text() not in allow]
        if stray:
            bad.append(f"axes[{i}] carries text {stray}")
    if list(f.texts):
        bad.append(f"figure carries text {[t.get_text() for t in f.texts]}")
    if bad:
        raise SystemExit(f"PANEL SPEC VIOLATION [{name}]: " + "; ".join(bad))
    print(f"panel spec OK [{name}]: no title, no legend, no figure text, no in-axes text "
          f"beyond the {len(allow)} allowed statistic mark(s)")


if PANEL_MODE:
    from sup_panel_export import export_stats as _stats
    _assert_panel_clean(fig, [ax], PANEL_NAME, allow=[_NOTE])
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    _pp = SI.save(fig, PANEL_DIR / f"{PANEL_NAME}.png")
    _pfmt = r"$p$ < 0.001" if p_ex < 0.001 else rf"$p$ = {p_ex:.3f}"
    _lines = [
        rf"cross-session Spearman $\rho$ = {rho_ex:+.2f}, {_pfmt}  ($n$ = {n} participants)",
        rf"within-session $\rho$ = {rho_within:+.2f}",
    ]
    _stats(_lines, "S3", "d", "crosssession_predictor", width=4.6)
    (PANEL_DIR / f"{PANEL_NAME}_stats.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print("wrote", PANEL_DIR / f"{PANEL_NAME}_stats.txt")
    for _ln in _lines:
        print("   ", _ln)
    plt.close(fig)
    raise SystemExit(0)

fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.06)
plt.close(fig)

P.to_csv(ec.sd("sup_fig3_crosssession_predictor_source_data.csv"), index=False)
pd.DataFrame([
    ("paper_location", "Supplementary Fig. 3d - Cross-session independence: locking (session 1) "
                       "predicts decoding (session 2) (first cited in Supplementary Note 4)"),
    ("n_sessions", int(M.shape[0])),
    ("n_participants", int(M.subj.nunique())),
    ("input_master", MASTER.relative_to(ROOT).as_posix()),
    ("n_two_session_subjects", n),
    ("cross_excess_A_to_rB_rho", round(rho_ex, 4)), ("cross_excess_A_to_rB_p", f"{p_ex:.2g}"),
    ("cross_mi_A_to_rB_rho", round(rho_mi, 4)),
    ("cross_excess_pooled_2n_rho", round(rho_pool, 4)), ("cross_excess_pooled_p", f"{p_pool:.2g}"),
    ("within_session_excess_rho", round(rho_within, 4)),
], columns=["key", "value"]).to_csv(ec.sd("sup_fig3_crosssession_predictor_metadata.csv"), index=False)

print(f"n={n} two-session subjects")
print(f"cross excess A->rB: rho={rho_ex:+.3f} p={p_ex:.2g} | mi A->rB: rho={rho_mi:+.3f}")
print(f"pooled 2n: rho={rho_pool:+.3f} p={p_pool:.2g} | within: rho={rho_within:+.3f}")
print(f"wrote {OUT}")
