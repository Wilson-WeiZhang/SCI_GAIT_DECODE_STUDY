"""Figure 7 — Intent gate and physics simulation of the decoded gait phase

_gate_method_schematic — Fig. 7 gate panel: METHOD SCHEMATIC of the deployable input-side gate

Reads:
  simulation and gate output in 02_decoding/simulation/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_style as PS
from project_palette import INK, IW, MID, PALE, REST, REST_EDGE

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = f"{ROOT}/figures_out/fig7/fig7_gate_method_schematic.png"
MD = f"{ROOT}/04_source_data/fig7/fig7_gate_method_schematic_metadata.csv"

W_MM, H_MM = 183.0, 72.0
HEAD_PT, BODY_PT = 7.0, 6.2
SI_PANEL = os.environ.get("SI_PANEL_SPEC") == "panel"
SI_W_MM, SI_H_MM = 160.0, 63.0
LW_K = 1.0

ROW_A = (50.0, 70.0)
ROW_B = (26.0, 46.0)
ROW_F = (2.0, 20.0)
BUS_Y = 23.0


def box(ax, x, y, w, h, head, lines, edge=MID, face="#FFFFFF", lw=0.9, ls="solid", center=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                linewidth=lw * LW_K, edgecolor=edge, facecolor=face, linestyle=ls,
                                mutation_aspect=1.0, zorder=2))
    if center:
        block = (4.0 if head else 0.0) + 3.4 * len(lines)
        ty = y + h / 2 + block / 2
    else:
        ty = y + h - 4.2
    if head:
        ax.text(x + w / 2, ty, head, ha="center", va="center", fontsize=HEAD_PT, color=INK,
                zorder=3)
        ty -= 4.0
    for ln in lines:
        ax.text(x + w / 2, ty, ln, ha="center", va="center", fontsize=BODY_PT, color=INK,
                zorder=3)
        ty -= 3.4


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=MID, linewidth=0.9 * LW_K,
                                shrinkA=0, shrinkB=0, mutation_scale=6 if LW_K == 1.0 else 5))


def main():
    global HEAD_PT, BODY_PT, LW_K
    PS.apply(plt)
    if SI_PANEL:
        import si_panel_style as SI
        SI.apply(plt)
        plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                             "mathtext.it": "Arial:italic"})
        HEAD_PT, BODY_PT, LW_K = SI.TITLE_PT, SI.TICK_PT, 0.55
        fig = plt.figure(figsize=(SI_W_MM * PS.MM, SI_H_MM * PS.MM), dpi=SI.DPI)
    else:
        fig = plt.figure(figsize=(W_MM * PS.MM, H_MM * PS.MM), dpi=PS.DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_MM)
    ax.set_ylim(0, H_MM)
    ax.axis("off")

    ix, iw_, iy, ih = 1.0, 29.0, ROW_B[0], ROW_A[1] - ROW_B[0]
    box(ax, ix, iy, iw_, ih, "EEG window",
        ["3 s of history", "60 channels, 250 Hz", "one decision every 1 s"],
        edge=INK, lw=1.1, center=True)

    ax_a0, ax_b0 = ROW_A[0], ROW_B[0]
    ha, hb = ROW_A[1] - ROW_A[0], ROW_B[1] - ROW_B[0]
    xs = [34.0, 84.0, 134.0]
    bw = 47.0

    box(ax, xs[0], ax_a0, bw, ha, "A   Band power (Welch)",
        ["Delta, theta, mu, beta (0.5–30 Hz)",
         "60 channels × 4 bands = 240 features"])
    box(ax, xs[1], ax_a0, bw, ha, "Rest-relative correction",
        ["$\\log_{10}\\,(\\mathit{P}\\,/\\,\\mathit{P}_{\\mathrm{rest}})$",
         "$\\mathit{P}_{\\mathrm{rest}}$ = median over training-fold rest"])
    box(ax, xs[2], ax_a0, bw, ha, "Shrinkage LDA",
        ["Ledoit–Wolf shrinkage",
         "fit on the training folds → score $\\mathit{d}_\\mathrm{A}$"])

    box(ax, xs[0], ax_b0, bw, hb, "B   Channel covariance",
        ["per band, 60 × 60",
         "trace-normalised, shrinkage $\\lambda$ = $10^{-3}$"])
    box(ax, xs[1], ax_b0, bw, hb, "Riemannian alignment",
        ["whiten by $\\mathit{R}^{-1/2}$",
         "$\\mathit{R}$ = Karcher mean of training-fold rest"])
    box(ax, xs[2], ax_b0, bw, hb, "Tangent space → logistic regression",
        ["log map at $\\mathit{R}$, upper triangle",
         "fit on the training folds → score $\\mathit{d}_\\mathrm{B}$"])

    for r, y0, h in ((ROW_A, ax_a0, ha), (ROW_B, ax_b0, hb)):
        arrow(ax, ix + iw_, y0 + h / 2, xs[0], y0 + h / 2)
        for k in range(2):
            arrow(ax, xs[k] + bw, y0 + h / 2, xs[k + 1], y0 + h / 2)

    fy0, fh = ROW_F[0], ROW_F[1] - ROW_F[0]
    join_x = xs[2] + bw + 1.0
    fx = [1.0, 46.0, 68.0, 111.0]
    fw = [43.0, 20.0, 41.0, 47.0]
    ya, yb = ax_a0 + ha / 2, ax_b0 + hb / 2
    f1cx = fx[0] + fw[0] / 2
    arrow(ax, xs[2] + bw, ya, join_x, ya)
    arrow(ax, xs[2] + bw, yb, join_x, yb)
    ax.plot([join_x, join_x], [BUS_Y, ya], color=MID, lw=0.9 * LW_K, zorder=1)
    ax.plot([f1cx, join_x], [BUS_Y, BUS_Y], color=MID, lw=0.9 * LW_K, zorder=1)
    arrow(ax, f1cx, BUS_Y, f1cx, ROW_F[1])

    box(ax, fx[0], fy0, fw[0], fh, "Standardise A and B",
        ["centre = training-fold median",
         "scale = training-fold s.d."])
    box(ax, fx[1], fy0, fw[1], fh, "Fuse",
        ["$\\mathit{z}$ = mean($\\mathit{z}_\\mathrm{A}$, $\\mathit{z}_\\mathrm{B}$)"])
    box(ax, fx[2], fy0, fw[2], fh, "Thresholds from training folds",
        ["$\\theta_{\\mathrm{sym}}$ = zero of the fused score",
         "$\\delta$ = 0.25 × IQR, "
         "$\\theta_{\\mathrm{on/off}}$ = $\\theta_{\\mathrm{sym}}$ ± $\\delta$"])
    box(ax, fx[3], fy0, fw[3], fh, "State machine",
        ["open at the first decision ≥ $\\theta_{\\mathrm{on}}$",
         "close at the first decision < $\\theta_{\\mathrm{off}}$"])
    for k in range(3):
        arrow(ax, fx[k] + fw[k], fy0 + fh / 2, fx[k + 1], fy0 + fh / 2)

    tx = fx[3] + fw[3]
    arrow(ax, tx, fy0 + fh / 2, tx + 4.0, fy0 + fh / 2)
    box(ax, tx + 4.0, fy0 + fh / 2 + 0.6, 19.0, 7.4, "", ["WALK"], edge=INK, face=IW, lw=0.9)
    box(ax, tx + 4.0, fy0 + fh / 2 - 8.0, 19.0, 7.4, "", ["STOP"], edge=REST_EDGE, face=REST,
        lw=0.9)


    if SI_PANEL:
        SI.save(fig, f"{ROOT}/figures_out/supplementary/_panels/S1d_gate_method_schematic.png")
        plt.close(fig)
        return
    fig.savefig(OUT, dpi=PS.DPI)
    plt.close(fig)

    meta = [
        ("colours", "greys only (project_palette INK/MID/PALE); terminals use the condition code "
                    "WALK = IW #8C8C8C, STOP = REST white with REST_EDGE. No decorative colour."),
        ("sampling_rate_hz", "250 (gate_features.py, FS_EXPECT)"),
        ("n_channels", "60 (59 brain + restored FCz) — the patient prep root's step3-clean "
                       "_eeg.mat"),
        ("bands_hz", "delta 0.5-4, theta 4-8, mu 8-13, beta 13-30 — gate_features.py "
                     "(BANDS); condition B filters them with a 4th-order Butterworth, "
                     "gate_fusion.py"),
        ("armA_features", "240 = 60 ch x 4 bands, Welch band power (gate_features.py)"),
        ("armA_correction", "log10(P / P_rest), P_rest = median band power of the calibration-rest "
                            "windows (gate_fusion.py)"),
        ("armA_model", "LDA(solver='lsqr', shrinkage='auto') (Ledoit-Wolf) — "
                       "gate_fusion.py"),
        ("armB_covariance", "per-band 60 x 60 channel covariance of the filtered window — "
                            "gate_fusion.py"),
        ("armB_shrinkage", "trace-normalised, lambda = 1e-3 — gate_fusion.py (LAM_RA, shrink)"),
        ("armB_alignment", "whiten by R^-1/2 with R = Karcher (AIRM) mean of the session's "
                           "calibration-rest covariances — gate_fusion.py (whiteners, ra_feat)"),
        ("armB_calibration_length_s", ">= 30 — gate_fusion.py and gate.py (BASE_MIN_S); split rule gate_features.py (calib_split)"),
        ("armB_readout", "tangent-space log map (upper triangle) -> logistic regression, "
                         "C in {0.01, 0.1, 1} chosen by inner 3-fold GroupKFold — "
                         "gate_fusion.py (C_GRID, ra_feat, fit_ts)"),
        ("fusion_rule", "fused score = mean of the two z scores (gate_fusion.py)"),
        ("theta_sym", "the S1a NATURAL ZERO of the fused calibration score: "
                      "((0 - centre_A)/sd_A + (0 - centre_B)/sd_B)/2, i.e. the two conditions' own "
                      "decision boundaries mapped through the V1 z — "
                      "02_decoding/simulation/gate_working_point_symmetric.py, "
                      "re-derived in 03_figures/fig7_gate_accuracy_table.py (read_thresholds); "
                      "column theta_sym of "
                      "02_decoding/simulation/result/gate_event_set_thresholds.csv"),
        ("delta", "0.25 x IQR(session calibration fused scores) — "
                  "gate_working_point_symmetric.py (GATE.IQR_K x calib_iqr; IQR_K = 0.25 in gate.py); "
                  
                  "column delta of the same CSV"),
        ("theta_on", "theta_sym + delta — gate_working_point_symmetric.py; column "
                     "theta_on_s1 of gate_event_set_thresholds.csv"),
        ("theta_off", "theta_sym - delta — gate_working_point_symmetric.py; column "
                      "theta_off_s1 of gate_event_set_thresholds.csv"),
        ("karcher_solver", "checked (step-halving fixed point, tol 1e-10, itmax 20000)."),
        ("gate_set", "post_operative"),
        ("companion_panel", "03_figures/fig7_gate_roc.py -> figures_out/fig7/fig7_gate_roc.png (the ROC this "
                            "schematic explains)"),
        ("producer", "03_figures/fig7_gate_method_schematic.py"),
    ]
    with open(MD, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(meta)
    print("wrote", OUT, MD)


if __name__ == "__main__":
    main()
