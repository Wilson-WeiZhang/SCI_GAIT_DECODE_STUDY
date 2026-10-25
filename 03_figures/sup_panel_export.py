# -*- coding: utf-8 -*-
"""Shared figure settings

Figure is delivered as its own PNG, with data, axes and axis titles only —
no statistics text, no legend, no panel letter inside the image

Imported by the figure scripts; it renders no panel of its own.
"""

from pathlib import Path
import os

import analysis_config as _ec

PANEL_MODE = os.environ.get("SI_PANEL_SPEC") == "panel"
PANEL_DIR = Path(__file__).resolve().parents[1] / "figures_out/supplementary/_panels"
if _ec.SCOPED:
    PANEL_DIR = PANEL_DIR / _ec.OUT_DIRNAME


def export_panels(fig, axes, figno, letters, slug, pad=1.06, dpi=300):
    """Save each axes of `fig` as its own PNG: S<figno><letter>_<slug>.png."""
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax, lab in zip(axes, letters):
        bb = ax.get_tightbbox(r).transformed(fig.dpi_scale_trans.inverted())
        p = PANEL_DIR / f"{figno}{lab}_{slug}.png"
        fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches=bb.expanded(pad, pad))
        print("wrote", p)
        out.append(p)
    return out


def export_legend(handles, figno, letter, slug, ncol=1, fontsize=8, figsize=(2.6, 1.0),
                  labels=None):
    """Save a standalone legend PNG for one panel."""
    import matplotlib.pyplot as plt
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    f = plt.figure(figsize=figsize)
    f.legend(handles=handles, labels=labels, loc="center", frameon=False,
             fontsize=fontsize, ncol=ncol)
    p = PANEL_DIR / f"{figno}{letter}_{slug}_legend.png"
    f.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    print("wrote", p)
    plt.close(f)
    return p


def export_panel_groups(fig, groups, figno, letters, slug, pad=1.06, dpi=300):
    """Save each GROUP of axes as one panel PNG (for grid panels: one row = one panel)."""
    from matplotlib.transforms import Bbox
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for group, lab in zip(groups, letters):
        bbs = [ax.get_tightbbox(r) for ax in group if ax.get_visible()]
        bb = Bbox.union(bbs).transformed(fig.dpi_scale_trans.inverted())
        p = PANEL_DIR / f"{figno}{lab}_{slug}.png"
        fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches=bb.expanded(pad, pad))
        print("wrote", p)
        out.append(p)
    return out


def export_stats(lines, figno, letter, slug, fontsize=9, dpi=300, width=3.4):
    """Save a panel's statistics as their own small PNG, one line per entry."""
    import matplotlib.pyplot as plt
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    n = len(lines)
    f = plt.figure(figsize=(width, 0.24 * n + 0.08))
    for i, ln in enumerate(lines):
        f.text(0.0, 1.0 - (i + 0.6) / n, ln, fontsize=fontsize, ha="left", va="center")
    p = PANEL_DIR / f"{figno}{letter}_{slug}_stats.png"
    f.savefig(p, dpi=dpi, bbox_inches="tight", transparent=True)
    print("wrote", p)
    plt.close(f)
    return p
