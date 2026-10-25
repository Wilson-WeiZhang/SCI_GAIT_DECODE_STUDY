#!/usr/bin/env python
"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

decoded-output glyph: the three decoded joint trajectories

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

EXAMPLES = [
    ("fig2b_glyph_output_joints",                "subj08_sess01", 8, 2, 4.0, 11.09),
    ("fig2b_glyph_output_joints_subj25_f1_o2",   "subj25_sess01", 1, 2, 4.0, 10.78),
    ("fig2b_glyph_output_joints_subj08_f4_o5",   "subj08_sess01", 4, 5, 4.0, 10.0),
    ("fig2b_glyph_output_joints_subj08_f4_o2",   "subj08_sess01", 4, 2, 4.0, 10.2),
    ("fig2b_glyph_output_joints_subj06_f8_o5",   "subj06_sess02", 8, 5, 4.0, 11.1),
]

MAIN_EXAMPLES = [
    ("fig2b_glyph_output_joints",                "subj13_sess01", None, 46, 4.0, 10.296),
    ("fig2b_glyph_output_joints_subj06_t28",     "subj06_sess02", None, 28, 4.0, 11.296),
]
if _ec.BUILD_ID in _ec.DECODER_BUILDS:
    MAIN_EXAMPLES = [
        ("fig2b_glyph_output_joints",                "subj11_sess02", None, 22, 4.0, 10.188),
        ("fig2b_glyph_output_joints_subj09_t49",     "subj09_sess01", None, 49, 4.0, 10.388),
    ]
if _ec.BUILD_ID in _ec.DECODER_BUILDS:
    MAIN_EXAMPLES = [
        ("fig2b_glyph_output_joints",                "subj11_sess02", None, 22, 4.0, 10.188),
        ("fig2b_glyph_output_joints_subj13_t54",     "subj13_sess01", None, 54, 4.0, 10.144),
    ]
COLORS = ("#f34142", "#1e6fd9", "#3aaf6a")
NAMES = ("Hip", "Knee", "Ankle")
OFFSET = 4.6
INK = "#1a1a1a"


def z(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def load(session, fold, ordinal, t0, t1):
    if _ec.DUMP_FORMAT == "pyport_session_h5":
        import dump_h5
        tr = dump_h5.load_trial(DUMPDIR / f"{session}_dump.h5", ordinal, ("Y_true", "C_pred"))
        t = tr["time_sec"]; m = (t >= t0) & (t <= t1)
        return tr["trial_idx"], t[m], tr["Y_true"][:3, m], tr["C_pred"][:3, m]
    _ec.require_dump_format("matlab_fold_mat")
    with h5py.File(DUMPDIR / f"{session}_fold{fold}_trial_dump.mat", "r") as f:
        i = ordinal - 1
        idx = int(np.array(f["meta/original_trial_indices"]).ravel()[i])
        t = np.array(f[f["time_sec"][i, 0]]).ravel()
        Y = np.array(f[f["Y_true"][i, 0]])
        C = np.array(f[f["C_pred"][i, 0]])
    m = (t >= t0) & (t <= t1)
    return idx, t[m], Y[:3, m], C[:3, m]


def glyph(name, session, fold, ordinal, t0, t1):
    idx, tt, Y, C = load(session, fold, ordinal, t0, t1)
    fig, ax = plt.subplots(figsize=(2.0, 1.46))
    rs = []
    for j in range(3):
        y, c = z(Y[j]), z(C[j])
        rs.append(float(np.corrcoef(y, c)[0, 1]))
        ax.plot(tt, c - j * OFFSET, color=COLORS[j], lw=2.0, solid_capstyle="round",
                solid_joinstyle="round")
    ax.axis("off")
    ax.set_ylim(-2 * OFFSET - 2.55, 2.55)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(_ec.scoped(OUT / f"{name}.png"), dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"{name}: {session} fold{fold} ordinal {ordinal} (trial_idx {idx}), window "
          f"{t0:g}-{tt[-1]:.1f} s, decoded r vs observed "
          + " / ".join(f"{n} {r:.3f}" for n, r in zip(NAMES, rs)))


if __name__ == "__main__":
    for e in (EXAMPLES if _ec.IS_DEFAULT else MAIN_EXAMPLES):
        glyph(*e)
