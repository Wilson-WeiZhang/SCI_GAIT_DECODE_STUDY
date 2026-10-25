"""Figure 5 — Decoding imagined walking after spinal cord injury

per-session patient imagined-walking decoding, drawn as Fig. 2c-style boxes

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import analysis_config as ec

TFS = 1.5
plt.rcParams.update({"font.family": "Arial", "font.size": 13 * TFS, "axes.linewidth": 2.2,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic"})

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data" / "fig5"
SRC = ec.sd("fig5b_session_box_source_data.csv")
META = ec.sd("fig5b_session_box_metadata.csv")
OUT_PNG = ec.scoped(ROOT / "figures_out" / "fig5" / "fig5b_session_box.png")

PATIENT = "#962014"
PCOL = {"P1": "#D3B8A2", "P2": "#E5E7CD"}
INK = "#1A1A1A"
REST = "#9aa0a6"


def read_band(meta_path: Path) -> tuple[float, float]:
    """Take the rest-null band from the metadata, so the panel cannot drift from the CSV."""
    for row in csv.reader(open(meta_path, newline="")):
        if row and row[0] == "rest_band_all":
            lo, hi = row[1].split("]")[0].lstrip("[").split(",")
            return float(lo), float(hi)
    raise SystemExit("rest_band_all not found in metadata")


def main() -> None:
    rows = list(csv.DictReader(open(SRC, newline="")))
    band_lo, band_hi = read_band(META)

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    _LW_BAND, _LW_ZERO, _LW_WHI, _LW_BOX, _LW_MED, _LW_BR = (
        (1.4, 1.2, 1.6, 1.8, 2.4, 1.6))

    ax.axhspan(band_lo, band_hi, color=REST, alpha=0.28, lw=0, zorder=0)
    for y in (band_lo, band_hi):
        ax.axhline(y, color=REST, lw=_LW_BAND, ls=(0, (5, 3)), zorder=1)
    ax.axhline(0, color="0.35", lw=_LW_ZERO, ls=(0, (1, 2.5)), zorder=1)

    gap_after = max(i for i, r in enumerate(rows) if r["patient"] == "P1")
    xs, x = [], 0.0
    for i in range(len(rows)):
        xs.append(x)
        x += 1.0 + (1.0 if i == gap_after else 0.0)

    half, cap = 0.30, 0.13
    for xi, r in zip(xs, rows):
        post = r["cohort"] == "post_operative"
        pc = PCOL[r["patient"]]
        q25, q75 = float(r["q25"]), float(r["q75"])
        med, p10, p90 = float(r["median"]), float(r["p10"]), float(r["p90"])
        ax.plot([xi, xi], [p10, q25], color=INK, lw=_LW_WHI, zorder=3, solid_capstyle="butt")
        ax.plot([xi, xi], [q75, p90], color=INK, lw=_LW_WHI, zorder=3, solid_capstyle="butt")
        for c in (p10, p90):
            ax.plot([xi - cap, xi + cap], [c, c], color=INK, lw=_LW_WHI, zorder=3)
        ax.add_patch(Rectangle((xi - half, q25), 2 * half, q75 - q25,
                               facecolor=pc if post else "white",
                               edgecolor=INK, lw=_LW_BOX, zorder=4))
        ax.plot([xi - half, xi + half], [med, med], color=INK, lw=_LW_MED, zorder=5,
                solid_capstyle="butt")

    ax.set_xlim(xs[0] - 0.85, xs[-1] + 0.85)
    ax.set_ylim(-0.45, 0.90)
    tx = ax.get_xaxis_transform()
    BR_Y, BR_LAB_Y = -0.315, -0.355
    for lo, hi, name in ((0, gap_after, "Patient 1"), (gap_after + 1, len(rows) - 1, "Patient 2")):
        ax.plot([xs[lo], xs[hi]], [BR_Y, BR_Y], color="0.25", lw=1.6,
                transform=tx, clip_on=False)
        ax.text((xs[lo] + xs[hi]) / 2, BR_LAB_Y, name, ha="center", va="top",
                transform=tx, fontsize=13 * TFS)

    ax.set_xticks(xs)
    ax.set_xticklabels([r["x_label"] for r in rows], fontsize=11 * TFS,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75])
    ax.set_ylabel("Decoding accuracy ($r$)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=2.2, length=5)

    handles = [
        Patch(facecolor=PCOL["P1"], edgecolor=INK, lw=1.8, label="Patient 1, post-operative"),
        Patch(facecolor=PCOL["P2"], edgecolor=INK, lw=1.8, label="Patient 2, post-operative"),
        Patch(facecolor="white", edgecolor=INK, lw=1.8, label="pre-operative"),
        Patch(facecolor=REST, alpha=0.28, lw=0, label="resting-input null (healthy, $n$ = 80)"),
    ]
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, facecolor="white", bbox_inches="tight")

    lfig = plt.figure(figsize=(11.0, 0.5))
    lfig.legend(handles=handles, loc="center", ncol=4, frameon=False,
                fontsize=11 * TFS, handlelength=1.5, columnspacing=1.6,
                borderpad=0.2, handletextpad=0.6)
    LEG_PNG = OUT_PNG.with_name(OUT_PNG.stem + "_legend.png")
    lfig.savefig(LEG_PNG, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(lfig)
    print(f"wrote {LEG_PNG}  (place at the panel's own scale)")
    n_post = sum(1 for r in rows if r["cohort"] == "post_operative")
    n_above_post = sum(1 for r in rows
                       if r["cohort"] == "post_operative" and r["median_above_rest_band"] == "True")
    print(f"wrote {OUT_PNG}")
    print(f"  {len(rows)} boxes = pre_operative + post_operative")
    print(f"  rest band [{band_lo:.6f}, {band_hi:.6f}]")
    print(f"  session median above band: {n_above_post}/{n_post} post-operative")


if __name__ == "__main__":
    main()
