#!/usr/bin/env python
"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

causal-FIR calibration glyph: observed vs before-FIR vs after-FIR

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42

import analysis_config as _ec
_ec.require("main", "main")

ROOT = Path(__file__).resolve().parents[1]
DUMPDIR = _ec.path("h_iw_dump_dir")
OUT = ROOT / "figures_out" / "fig2"

INK = "#1a1a1a"
GREY = "#8c8c8c"
OBS = "#1a1a1a"
RED = "#f34142"
JOINT = 0
LEGEND_FS = 11

EXAMPLES = [
    ("fig2b_glyph_fir_before_after",           "subj08_sess01", 8, 2, 4.0, 11.09),
    ("fig2b_glyph_fir_subj25_sess01_f1_o2",    "subj25_sess01", 1, 2, 4.0, 10.78),
    ("fig2b_glyph_fir_subj08_sess01_f4_o5",    "subj08_sess01", 4, 5, 4.0, 10.0),
    ("fig2b_glyph_fir_subj08_sess01_f4_o2",    "subj08_sess01", 4, 2, 4.0, 10.2),
    ("fig2b_glyph_fir_subj06_sess02_f8_o5",    "subj06_sess02", 8, 5, 4.0, 11.1),
]

MAIN_EXAMPLES = [
    ("fig2b_glyph_fir_before_after",           "subj13_sess01", None, 46, 4.0, 10.296),
    ("fig2b_glyph_fir_subj06_sess02_t28",      "subj06_sess02", None, 28, 4.0, 11.296),
]
if _ec.BUILD_ID in _ec.DECODER_BUILDS:
    MAIN_EXAMPLES = [
        ("fig2b_glyph_fir_before_after",           "subj11_sess02", None, 22, 4.0, 10.188),
        ("fig2b_glyph_fir_subj09_sess01_t49",      "subj09_sess01", None, 49, 4.0, 10.388),
    ]
if _ec.BUILD_ID in _ec.DECODER_BUILDS:
    MAIN_EXAMPLES = [
        ("fig2b_glyph_fir_before_after",           "subj11_sess02", None, 22, 4.0, 10.188),
        ("fig2b_glyph_fir_subj13_sess01_t54",      "subj13_sess01", None, 54, 4.0, 10.144),
    ]


def z(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def load_trial(session, fold, ordinal):
    if _ec.DUMP_FORMAT == "pyport_session_h5":
        import dump_h5
        tr = dump_h5.load_trial(DUMPDIR / f"{session}_dump.h5", ordinal, ("Y_true", "A_pred", "C_pred"))
        return tr["trial_idx"], tr["time_sec"], tr["Y_true"][JOINT], tr["A_pred"][JOINT], tr["C_pred"][JOINT]
    _ec.require_dump_format("matlab_fold_mat")
    path = DUMPDIR / f"{session}_fold{fold}_trial_dump.mat"
    with h5py.File(path, "r") as f:
        i = ordinal - 1
        idx = int(np.array(f["meta/original_trial_indices"]).ravel()[i])
        t = np.array(f[f["time_sec"][i, 0]]).ravel()
        obs = np.array(f[f["Y_true"][i, 0]])[JOINT]
        bef = np.array(f[f["F_pred"][i, 0]])[JOINT]
        aft = np.array(f[f["C_pred"][i, 0]])[JOINT]
    return idx, t, obs, bef, aft


def glyph(name, session, fold, ordinal, t0, t1, with_legend=True):
    idx, t, obs, bef, aft = load_trial(session, fold, ordinal)
    m = (t >= t0) & (t <= t1)
    t, obs, bef, aft = t[m], z(obs[m]), z(bef[m]), z(aft[m])
    r_bef = float(np.corrcoef(obs, bef)[0, 1])
    r_aft = float(np.corrcoef(obs, aft)[0, 1])

    fig, ax = plt.subplots(figsize=(2.95, 1.30))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.87)
    ax.plot(t, obs, color=OBS, lw=2.1, label="reference", solid_capstyle="round")
    ax.plot(t, bef, color=GREY, lw=2.1, ls=(0, (3.2, 1.8)), label="before")
    ax.plot(t, aft, color=RED, lw=3.2, label="after", solid_capstyle="round")
    ax.axis("off")

    ax.set_ylim(min(obs.min(), bef.min(), aft.min()) - 0.15,
                max(obs.max(), bef.max(), aft.max()) + 0.15)

    if with_legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                  fontsize=LEGEND_FS, handlelength=1.5, handletextpad=0.45, columnspacing=1.0,
                  borderpad=0, borderaxespad=0, labelcolor=INK)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(_ec.scoped(OUT / f"{name}.png"), dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"{name}: {session} fold{fold} ordinal {ordinal} (trial_idx {idx}), LHipW X, "
          f"window {t0:g}-{t1:g} s, r before {r_bef:.3f} after {r_aft:.3f}")


def legend_strip(name="fig2b_glyph_fir_legend", fontsize=LEGEND_FS, ncol=1):
    """Stand-alone observed/before/after legend."""
    fig, ax = plt.subplots(figsize=(1.2, 0.95))
    ax.axis("off")
    h = [
        ax.plot([], [], color=OBS, lw=2.1, solid_capstyle="round", label="reference")[0],
        ax.plot([], [], color=GREY, lw=2.1, ls=(0, (3.2, 1.8)), label="before")[0],
        ax.plot([], [], color=RED, lw=3.2, solid_capstyle="round", label="after")[0],
    ]
    ax.legend(handles=h, loc="center", ncol=ncol, frameon=False, fontsize=fontsize,
              handlelength=1.5, handletextpad=0.45, columnspacing=1.0,
              labelspacing=0.55, borderpad=0, borderaxespad=0, labelcolor=INK)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(_ec.scoped(OUT / f"{name}.png"), dpi=600, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"{name}: stand-alone legend, fontsize {fontsize}, ncol={ncol} "
          f"({3 // max(ncol, 1)} rows), 600 dpi")


def main():
    examples = EXAMPLES if _ec.IS_DEFAULT else MAIN_EXAMPLES
    for e in examples:
        glyph(*e)
    glyph(examples[0][0] + "_nolegend", *examples[0][1:], with_legend=False)
    legend_strip()


if __name__ == "__main__":
    main()
