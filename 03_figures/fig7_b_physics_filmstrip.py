#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 7 — Intent gate and physics simulation of the decoded gait phase

Physics filmstrip and forward-displacement traces of the four simulation conditions

Reads:
  simulation and gate output in 02_decoding/simulation/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import project_palette as P
import os
os.environ.setdefault("GAIT_F7_SIM", "causal_from2s")
import analysis_config as ec
ec.require("main", "main")

TRACES = ROOT / "02_decoding/simulation/simulation_rollouts/traces_x.csv.gz"
CONDITIONS = [("wphase", "Walker", P.WALKER_HUE, 1.2, "-"),
        ("dphase", "Decoded", P.IW_HUE, 1.2, "-"),
        ("dphase_iscr", "Coordination null", P.MID, 0.8, "--"),
        ("dphase_uscr", "In-phase null", P.MID, 0.8, ":")]
DPI = 300

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--n-frames", type=int, default=7, help="frame instants = linspace(0, tmax, n)")
    ap.add_argument("--row-png", default=str(ROOT / "04_source_data/fig7/fig7_b_physics_frames.png"),
                    help="the frames row (n cells side by side, no gutter); shipped as an image, "
                         "its MuJoCo renderer is not part of this repository")
    ap.add_argument("--gutter-px", type=int, default=300)
    ap.add_argument("--axes-ratio", type=float, default=2 / 3,
                    help="y-axis (axes box) height as a fraction of the frame-row height (3:2 -> 2/3)")
    ap.add_argument("--fs", type=float, default=1.15, help="font/line scale (the row is ~12 in wide at 300 dpi)")
    ap.add_argument("--out", default=str(ROOT / "figures_out/fig7/fig7_b_filmstrip"), help="output stem")
    ap.add_argument("--traces", default=str(TRACES), help="batch traces_x.csv.gz to plot")
    ap.add_argument("--walker-condition", default="wphase_matched", help="condition name of the Walker condition (wphase_matched for the gate3s build)")
    ap.add_argument("--frame-times", default="", help="explicit frame instants (s, comma list): dotted lines at these times; the strip then spans the full row width instead of cell-centre alignment")
    ap.add_argument("--standalone", default="", help="also write the displacement strip as its own panel PNG (+ _legend.png) at this stem, Fig. 7 panel typography (4.4 x 1.7 in, 300 dpi), no frame-row gutter")
    args = ap.parse_args()
    if args.session is None or args.trial is None:
        args.session = "pat02_sess05" if args.session is None else args.session
        args.trial = 8 if args.trial is None else args.trial
    if ec.SCOPED:
        args.traces, args.row_png, args.out = (str(ec.scoped(p)) for p in (args.traces, args.row_png, args.out))
        if args.standalone:
            args.standalone = str(ec.scoped(args.standalone))

    row = Image.open(args.row_png).convert("RGB")
    RW, RH = row.size
    cell_w = RW / args.n_frames
    frames = np.linspace(0.0, args.tmax, args.n_frames)
    explicit = bool(args.frame_times)
    if explicit:
        frames = np.array([float(x) for x in args.frame_times.split(",")])
    conditions = [(args.walker_condition if i == 0 else condition, label, col, lw, ls) for i, (condition, label, col, lw, ls) in enumerate(CONDITIONS)]
    G = args.gutter_px
    W = G + RW
    fs = args.fs
    plt.rcParams.update({"font.family": "Arial", "font.size": 7 * fs, "axes.linewidth": 0.5 * fs,
                         "xtick.major.width": 0.5 * fs, "ytick.major.width": 0.5 * fs, "pdf.fonttype": 42})

    t = pd.read_csv(args.traces)
    t = t[(t.session == args.session) & (t.trial_idx == args.trial) & (t.t_s <= args.tmax + 1e-9)]
    assert not t.empty, "trial not in traces_x"
    axes_h = args.axes_ratio * RH
    ends = {}

    def draw(SH, bottom_px):
        fig = plt.figure(figsize=(W / DPI, SH / DPI), dpi=DPI)
        if explicit:
            ax = fig.add_axes([G / W, bottom_px / SH, RW / W, axes_h / SH])
        else:
            ax = fig.add_axes([(G + cell_w / 2) / W, bottom_px / SH, (RW - cell_w) / W, axes_h / SH])
        for condition, label, col, lw, ls in conditions:
            s = t[t.condition == condition].sort_values("t_s")
            assert not s.empty, condition
            ax.plot(s.t_s, s.x_m, color=col, lw=lw * fs, ls=ls, solid_capstyle="round")
            ends[label] = float(s.x_m.iloc[-1])
        for tf in frames:
            ax.axvline(tf, color=P.INK, lw=0.4 * fs, ls=(0, (1.5, 2.5)), alpha=0.7, zorder=0)
        ax.axhline(0, color=P.INK, lw=0.4 * fs, ls=(0, (1, 2)), alpha=0.6)
        ax.set_xlim(0, args.tmax)
        ax.set_xlabel("Time (s)", labelpad=1.5 * fs)
        ax.set_ylabel("Forward\ndisplacement (m)", labelpad=2 * fs)
        ax.set_xticks(np.arange(0, args.tmax + 1e-9, 1.0))
        ax.tick_params(length=2.5 * fs, pad=1.5 * fs)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        return fig, ax

    fig, ax = draw(int(axes_h + 500), 300)
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    yl_h = ax.yaxis.label.get_window_extent(rd).height
    xl_y0 = min(ax.xaxis.label.get_window_extent(rd).y0, ax.xaxis.get_tightbbox(rd).y0)
    ax_y0 = ax.get_window_extent(rd).y0
    plt.close(fig)
    pad = 12
    overhang = max(0.0, (yl_h - axes_h) / 2)
    bottom_px = int(round(max(ax_y0 - xl_y0 + pad, overhang + pad)))
    top_px = int(round(overhang + pad))
    SH = bottom_px + int(round(axes_h)) + top_px
    fig, ax = draw(SH, bottom_px)
    print(f"strip: row {RH} px, axes box {axes_h:.0f} px (ratio {args.axes_ratio:.3f}), y label {yl_h:.0f} px, strip {SH} px")
    out_strip = Path(args.out + "_xtrace.png")
    fig.savefig(out_strip, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("wrote", out_strip, "endpoints at tmax:", {k: round(v, 3) for k, v in ends.items()})

    figL = plt.figure(figsize=(3.6 * fs, 0.32 * fs), dpi=DPI)
    handles = [plt.Line2D([], [], color=col, lw=lw * fs, ls=ls, label=label) for _, label, col, lw, ls in conditions]
    handles.append(plt.Line2D([], [], color=P.INK, lw=0.4 * fs, ls=(0, (1.5, 2.5)), label="frame instants"))
    figL.legend(handles=handles, loc="center", ncol=5, frameon=False, fontsize=6 * fs, handlelength=2.2, columnspacing=1.2)
    out_leg = Path(args.out + "_xtrace_legend.png")
    figL.savefig(out_leg, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(figL)
    print("wrote", out_leg)

    if args.standalone:
        plt.rcParams.update({"font.family": "Arial", "font.size": 6.5 * 1.35, "axes.linewidth": 0.8,
                             "xtick.major.width": 0.8, "ytick.major.width": 0.8, "pdf.fonttype": 42})
        fs2 = 1.0
        f3 = plt.figure(figsize=(4.4, 1.7), dpi=DPI)
        ax3 = f3.add_axes([0.13, 0.26, 0.84, 0.70])
        for condition, label, col, lw, ls in conditions:
            s3 = t[t.condition == condition].sort_values("t_s")
            ax3.plot(s3.t_s, s3.x_m, color=col, lw=lw * 1.1, ls=ls, solid_capstyle="round")
        for tf in frames:
            ax3.axvline(tf, color=P.INK, lw=0.5, ls=(0, (1.5, 2.5)), alpha=0.7, zorder=0)
        ax3.axhline(0, color=P.INK, lw=0.5, ls=(0, (1, 2)), alpha=0.6)
        ax3.set_xlim(0, args.tmax); ax3.set_xticks(np.arange(0, args.tmax + 1e-9, 1.0))
        ax3.set_xlabel("Time (s)", labelpad=2); ax3.set_ylabel("Forward\ndisplacement (m)", labelpad=2)
        ax3.tick_params(length=2.5, pad=1.5)
        for sp in ("top", "right"):
            ax3.spines[sp].set_visible(False)
        f3.savefig(args.standalone + ".png", dpi=DPI, facecolor="white"); plt.close(f3)
        fL = plt.figure(figsize=(4.4, 0.32), dpi=DPI)
        hs = [plt.Line2D([], [], color=col, lw=lw * 1.1, ls=ls, label=label) for _, label, col, lw, ls in conditions]
        hs.append(plt.Line2D([], [], color=P.INK, lw=0.5, ls=(0, (1.5, 2.5)), label="frame instants"))
        fL.legend(handles=hs, loc="center", ncol=5, frameon=False, fontsize=6.5, handlelength=2.2, columnspacing=1.2)
        fL.savefig(args.standalone + "_legend.png", dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.02); plt.close(fL)
        print("wrote", args.standalone + ".png", "+ _legend.png (standalone strip)")

    strip = Image.open(out_strip).convert("RGB")
    assert strip.width == W, (strip.width, W)
    canvas = Image.new("RGB", (W, RH + strip.height), "white")
    canvas.paste(row, (G, 0))
    canvas.paste(strip, (0, RH))
    out = Path(args.out + ".png")
    canvas.save(out, dpi=(DPI, DPI))
    print("wrote", out, canvas.size)


if __name__ == "__main__":
    main()
