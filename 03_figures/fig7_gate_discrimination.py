"""Figure 7 — Intent gate and physics simulation of the decoded gait phase

_gate_discrimination — Fig. 7 gate block: how well the input-side gate separates imagined
walking from resting control, per session, on ONE shared decision boundary

Reads:
  simulation and gate output in 02_decoding/simulation/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

import hashlib
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import Divider, Size
from scipy.stats import gaussian_kde

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import panel_style as PS
import project_palette as PAL
from patient_session_labels import label as _plabel

SCORES = f"{ROOT}/02_decoding/simulation/result/gate_within_session_scores_T3.csv"
ACC = f"{ROOT}/04_source_data/fig7/fig7_gate_accuracy_source_data.csv"
OUT = f"{ROOT}/figures_out/fig7/fig7_gate_discrimination.png"
OUT_LEG = f"{ROOT}/figures_out/fig7/fig7_gate_discrimination_legend.png"
SD = f"{ROOT}/04_source_data/fig7/fig7_gate_discrimination_source_data.csv"
MD = f"{ROOT}/04_source_data/fig7/fig7_gate_discrimination_metadata.csv"
import gate_config as GE
SCORES, ACC, OUT, OUT_LEG, SD, MD = (GE.scoped(p) for p in (SCORES, ACC, OUT, OUT_LEG, SD, MD))

POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
            "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
            "pat02_sess09"]
N_DECISIONS = 2605 if GE.SC else 4012

H_IW, H_REST, FA = PAL.IW_HUE, PAL.REST_HUE, PAL.COND_FILL_ALPHA
F_IW, F_REST = to_rgba(H_IW, FA), to_rgba(H_REST, FA)
INK, MID = PAL.INK, PAL.MID
MM = PS.MM
SEED, B = 0, 1000
XLIM = (-20.0, 30.0)
BW = 1.0
GRID = np.linspace(XLIM[0], XLIM[1], 400)
XR = (0.7, 1.0)

ROW_MM, GAP_MM = 8.0, 1.0
LEFT_MM, DIST_MM, MID_MM, AUC_MM, RIGHT_MM = 15.0, 50.0, 6.0, 20.0, 4.5
BOTTOM_MM, TOP_MM = 9.5, 2.0
LEG_H_MM = 12.0
POOLED_GAP_MM = 2.2

def short(s):
    return _plabel(s, with_patient=True)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def trial_table(df):
    """One row per evaluated trial (session, event): its decision count, how many of its 1 s-hop
    decisions fall on its own side of the symmetric boundary, and that fraction."""
    rows = []
    for (s, ev), g in df.groupby(["session", "event"], sort=False):
        ok = (g.e_sigma.values >= 0) == (g.label.values == 1)
        rows.append((s, ev, int(g.label.iloc[0]), len(g), int(ok.sum()), float(ok.mean())))
    return pd.DataFrame(rows, columns=["session", "event", "label", "n_decisions",
                                       "n_correct", "frac_correct"])


def boot_mean(v):
    """Mean over trials + percentile bootstrap-over-trials 95 % CI (B resamples, own rng(SEED))."""
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, float)
    m = [rng.choice(v, len(v)).mean() for _ in range(B)]
    return float(v.mean()), *np.percentile(m, [2.5, 97.5])


def main():
    d = pd.read_csv(SCORES)
    acc_all = pd.read_csv(ACC)
    acc = acc_all[acc_all.row_type == "session"].set_index("session")
    sess = sorted(d.session.unique())

    assert sess == POST_OPERATIVE, f"session set is not post_operative: {sess}"
    assert len(d) == N_DECISIONS, f"{len(d)} decisions, expected {N_DECISIONS}"
    assert (d.e_sigma == 0).sum() == 0
    print(f"[check 1] sessions = 11 post_operative ({', '.join(sess)}); "
          f"decisions = {len(d)} == {N_DECISIONS}  OK")

    trials = trial_table(d)
    order = ["POOLED"] + sess[::-1]

    dla = {}
    for s in sess + ["POOLED"]:
        ds = d if s == "POOLED" else d[d.session == s]
        dla[s] = float(np.mean((ds.e_sigma.values >= 0) == (ds.label.values == 1)))
        ref = (float(acc_all.query("row_type == 'pooled'").s0_accuracy.iloc[0])
               if s == "POOLED" else float(acc.loc[s, "s0_accuracy"]))
        assert abs(dla[s] - ref) < 1e-6, f"{s}: {dla[s]:.6f} vs s0_accuracy {ref:.6f}"
    print("[check 2] decision-level accuracy recomputed from e_sigma == s0_accuracy of "
          "fig7_gate_accuracy_source_data.csv to 1e-6 for all 11 sessions + POOLED  OK")

    astat = {}
    for s in order:
        ta = (trials if s == "POOLED" else trials[trials.session == s]).frac_correct.to_numpy()
        astat[s] = (len(ta), *boot_mean(ta))
    print(f"[check 3] per-trial accuracy at the symmetric boundary "
          f"(mean over trials, bootstrap-over-trials 95 % CI, B = {B}, seed = {SEED}):")
    for s in order:
        n, m, lo, hi = astat[s]
        print(f"          {('pooled' if s == 'POOLED' else short(s)):11s} "
              f"({s if s != 'POOLED' else 'all 11 sessions':15s}) n_trials {n:4d}  "
              f"{m:.4f} [{lo:.4f}, {hi:.4f}]   decision-level {dla[s]:.4f}")

    PATIENTS = {"pat01_POOLED": "pat01_", "pat02_POOLED": "pat02_"}
    for k, pre in PATIENTS.items():
        ta = trials[trials.session.str.startswith(pre)].frac_correct.to_numpy()
        astat[k] = (len(ta), *boot_mean(ta))
        dp = d[d.session.str.startswith(pre)]
        dla[k] = float(np.mean((dp.e_sigma.values >= 0) == (dp.label.values == 1)))
        n, m, lo, hi = astat[k]
        print(f"          {k:11s} n_trials {n:4d}  {m:.4f} [{lo:.4f}, {hi:.4f}]   "
              f"decision-level {dla[k]:.4f}")
    assert astat["pat01_POOLED"][0] + astat["pat02_POOLED"][0] == astat["POOLED"][0]

    PS.apply(plt)
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    _LW_KDE, _LW_BOX, _LW_MED, _LW_BOUND, _MS_ACC, _LW_CI, _CAPS, _LW_GUIDE = (
        (0.6, 0.55, 0.8, 0.8, 4.0, 0.9, 2.2, 0.6))
    _PT_LAB, _PT_TICK, _PT_TTL = ((PS.NOTE_PT, PS.TICK_PT - 1.5, PS.BASE_PT))
    n_rows = len(sess) + 1
    h_rows = n_rows * ROW_MM + (n_rows - 1) * GAP_MM + POOLED_GAP_MM
    w_mm = LEFT_MM + DIST_MM + MID_MM + AUC_MM + RIGHT_MM
    h_mm = BOTTOM_MM + h_rows + TOP_MM
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM), dpi=PS.DPI)
    hs = [Size.Fixed(LEFT_MM * MM), Size.Fixed(DIST_MM * MM), Size.Fixed(MID_MM * MM),
          Size.Fixed(AUC_MM * MM), Size.Fixed(RIGHT_MM * MM)]
    vs = [Size.Fixed(BOTTOM_MM * MM)]
    ny = {}
    for i, s in enumerate(order):
        if i == 1:
            vs.append(Size.Fixed(POOLED_GAP_MM * MM))
        elif i > 1:
            vs.append(Size.Fixed(GAP_MM * MM))
        ny[s] = len(vs)
        vs.append(Size.Fixed(ROW_MM * MM))
    div = Divider(fig, (0, 0, 1, 1), hs, vs, aspect=False)
    axL = {s: fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=ny[s]))
           for s in order}
    axR = {s: fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=3, ny=ny[s]))
           for s in order}

    box_rows, dens_rows = [], []
    for s in order:
        aL, aR = axL[s], axR[s]
        ds = d if s == "POOLED" else d[d.session == s]
        iw, rest = ds.e_sigma[ds.label == 1].values, ds.e_sigma[ds.label == 0].values
        pi = gaussian_kde(iw, bw_method=BW / np.std(iw))(GRID)
        pr = gaussian_kde(rest, bw_method=BW / np.std(rest))(GRID)
        ymax = max(pi.max(), pr.max()) * 1.05
        aL.fill_between(GRID, 0, pi, facecolor=F_IW, edgecolor="none", zorder=2)
        aL.fill_between(GRID, 0, pr, facecolor=F_REST, edgecolor="none", zorder=2)
        aL.plot(GRID, pi, color=H_IW, lw=_LW_KDE, zorder=3)
        aL.plot(GRID, pr, color=H_REST, lw=_LW_KDE, zorder=4)
        bh = 0.16 * ymax
        for cls, v, yc, face, hue in (("iw", iw, -0.24 * ymax, F_IW, H_IW),
                                      ("rest", rest, -0.46 * ymax, F_REST, H_REST)):
            aL.boxplot([v], vert=False, positions=[yc], widths=[bh], whis=(10, 90),
                       showfliers=False, patch_artist=True, manage_ticks=False, zorder=5,
                       boxprops=dict(facecolor=face, edgecolor=hue, linewidth=_LW_BOX),
                       whiskerprops=dict(color=hue, linewidth=_LW_BOX),
                       capprops=dict(color=hue, linewidth=_LW_BOX),
                       medianprops=dict(color=hue, linewidth=_LW_MED))
            p10, q25, med, q75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
            box_rows.append(dict(row_type="box", session=s, **{"class": cls},
                                 n_decisions=len(v), p10=p10, q25=q25, median=med,
                                 q75=q75, p90=p90))
        for cls, p in (("iw", pi), ("rest", pr)):
            dens_rows.append(pd.DataFrame(dict(row_type="density", session=s, x=GRID,
                                               density=p)).assign(**{"class": cls}))
        aL.axvline(0, color=INK, lw=_LW_BOUND, ls=(0, (1.6, 1.6)), zorder=6)
        aL.set_xlim(*XLIM)
        aL.set_ylim(-0.64 * ymax, ymax)
        for sp in ("top", "right", "left"):
            aL.spines[sp].set_visible(False)
        aL.set_yticks([])
        aL.text(-0.02, 0.45, "pooled" if s == "POOLED" else short(s), transform=aL.transAxes,
                ha="right", va="center", fontsize=_PT_LAB,
                fontweight="bold" if s == "POOLED" else "normal", color=INK)
        if s == "POOLED":
            aL.spines["bottom"].set_visible(True)
            aL.set_xticks([-20, -10, 0, 10, 20, 30])
            aL.tick_params(axis="x", direction="out", labelsize=_PT_TICK)
            aL.set_xlabel("Gate score", fontsize=_PT_TTL, labelpad=1.5)
        else:
            aL.spines["bottom"].set_visible(False)
            aL.set_xticks([])
        _, m, lo, hi = astat[s]
        aR.errorbar([m], [0.5], xerr=[[m - lo], [hi - m]], fmt="o", ms=_MS_ACC, color=INK,
                    ecolor=INK, elinewidth=_LW_CI, capsize=_CAPS, capthick=_LW_CI, zorder=3,
                    clip_on=False)
        aR.set_xlim(*XR)
        aR.set_facecolor("none")
        aR.set_ylim(0, 1)
        aR.set_yticks([])
        for sp in ("top", "right", "left"):
            aR.spines[sp].set_visible(False)
        if s == "POOLED":
            aR.set_xticks(np.round(np.arange(XR[0], 1.0001, 0.1), 1))
            aR.tick_params(axis="x", direction="out", labelsize=_PT_TICK)
            aR.set_xlabel("Accuracy at boundary", fontsize=_PT_TTL, labelpad=1.5)
        else:
            aR.spines["bottom"].set_visible(False)
            aR.set_xticks([])

    fig.canvas.draw()
    x_px = axR["POOLED"].transData.transform((astat["POOLED"][1], 0.0))[0]
    x_fig = x_px / (fig.get_size_inches()[0] * fig.dpi)
    fig.add_artist(Line2D([x_fig, x_fig],
                          [axR["POOLED"].get_position().y0, axR[order[-1]].get_position().y1],
                          transform=fig.transFigure, color=MID, lw=_LW_GUIDE, zorder=0))
    fig.savefig(OUT, dpi=PS.DPI)
    plt.close(fig)
    print(f"wrote {os.path.relpath(OUT, ROOT)}  ({w_mm:.1f} x {h_mm:.1f} mm)")

    h = [Patch(facecolor=F_IW, edgecolor=H_IW, linewidth=0.55),
         Patch(facecolor=F_REST, edgecolor=H_REST, linewidth=0.55),
         Line2D([0], [0], color=INK, lw=0.8, ls=(0, (1.6, 1.6))),
         Line2D([0], [0], color=INK, marker="o", ms=4.2, lw=0.9)]
    lab = ["Imagined walking trial", "Resting-control trial", "Decision boundary",
           "Accuracy at boundary, mean\nover trials (95 % CI)"]
    lf = plt.figure(figsize=(w_mm * MM, LEG_H_MM * MM), dpi=PS.DPI)
    leg = lf.legend(h, lab, loc="center", frameon=False, ncol=2, handlelength=1.6,
                    columnspacing=1.6, labelspacing=0.5, fontsize=PS.BASE_PT)
    lf.canvas.draw()
    bb = leg.get_window_extent()
    assert bb.width <= w_mm * MM * PS.DPI, f"legend {bb.width:.0f} px wider than the panel {w_mm * MM * PS.DPI:.0f} px"
    lf.savefig(OUT_LEG, dpi=PS.DPI, transparent=True)
    plt.close(lf)
    print(f"wrote {os.path.relpath(OUT_LEG, ROOT)}  ({w_mm:.1f} x {LEG_H_MM:.1f} mm, = the panel width)")

    tr = trials.assign(row_type="trial")
    ac = pd.DataFrame([dict(row_type="accuracy", session=s, n_trials=astat[s][0],
                            mean_trial_accuracy=astat[s][1], ci_lo=astat[s][2],
                            ci_hi=astat[s][3], B=B, seed=SEED,
                            decision_level_accuracy=dla[s]) for s in ["POOLED"] + sess + list(PATIENTS)])
    sd = pd.concat([tr, ac, pd.DataFrame(box_rows), pd.concat(dens_rows, ignore_index=True)],
                   ignore_index=True)
    sd = sd[["row_type", "session", "event", "label", "n_decisions", "n_correct", "frac_correct",
             "n_trials", "mean_trial_accuracy", "ci_lo", "ci_hi", "B", "seed",
             "decision_level_accuracy", "class", "p10", "q25", "median", "q75", "p90",
             "x", "density"]]
    sd.to_csv(SD, index=False, float_format="%.6f")
    print(f"wrote {os.path.relpath(SD, ROOT)}  ({len(sd)} rows: "
          + ", ".join(f"{k} {v}" for k, v in sd.row_type.value_counts().items()) + ")")

    meta = [
        ("cohort", "post_operative, n = 2 patients, case-level (11 sessions: " + ", ".join(POST_OPERATIVE)
                   + "). The gate uses 11 of 11 sessions. "
                     
                     "Eleven sessions from two people is not a population estimate."),
        ("score_column", "e_sigma — the fused score (fuse_V1) standardised by the (theta_sym, delta) "
                         "of the fold that emitted it: e_sigma = (fuse_V1 - theta_sym) / delta. The "
                         "decision boundary is therefore 0 for EVERY row of EVERY session and one "
                         "dotted line at 0 serves all 12 rows; in RAW units each of the 10 folds has "
                         "its own theta_sym, so no single raw boundary exists. The pooled row is the "
                         f"union of the standardised decisions of all 11 sessions ({len(d):,}), not a mean "
                         "of session distributions."),
        ("boundary_not_tuned", "theta_sym is the two conditions' natural zero computed from each fold's "
                               "TRAINING folds — ((0 - centre_A)/sd_A + (0 - centre_B)/sd_B)/2 — and "
                               "is never fitted to the scored windows. The accuracy reported here is "
                               "therefore accuracy at a pre-specified operating point, not at a "
                               "threshold chosen on the data being scored."),
        ("kde_bandwidth", "1.0 delta units, FIXED for every row and both classes (gaussian_kde with "
                          "bw_method = 1.0 / std(x)), so the smoothing is identical across sessions "
                          "and rows are comparable. Evaluated on 400 points from -20 to 30 delta "
                          "units (row_type=density is exactly the plotted curve). The KDE is fitted "
                          "to the unclipped data; mass beyond the axis is off-axis, not piled up at "
                          "the edge."),
        ("box_definition", "row_type=box — box 25-75 %, whiskers 10-90 %, median; computed on all "
                           "decisions of that row and class (no outlier markers)."),
        ("right_statistic", "Per-trial accuracy at the symmetric boundary. For each evaluated trial "
                            "(event) the fraction of its 1 s-hop decisions on its own side of "
                            "theta_sym (correct iff (e_sigma >= 0) == (label == 1); no decision has "
                            "e_sigma exactly 0, asserted at run time). The dot is the MEAN OVER "
                            "TRIALS of that fraction, so every trial weighs the same regardless of "
                            "its duration; the capped bar is a percentile (2.5 / 97.5) bootstrap "
                            "OVER TRIALS, B = 1000, seed 0, each row drawn from its own "
                            "np.random.default_rng(0). This is NOT the decision-weighted accuracy: "
                            "row_type=accuracy carries both, decision_level_accuracy being the S0 "
                            "accuracy of 03_figures/fig7_gate_accuracy_table.py (asserted equal to 1e-6 "
                            "at run time)."),
        ("pooled_guide", "The continuous vertical line through the right column is the pooled mean "
                         "trial accuracy (the POOLED row's dot), drawn once in figure coordinates."),
        ("colours", f"Fig. 6d rule — light = fill (hue at COND_FILL_ALPHA = {FA} on white), dark = "
                    f"edge (the hue itself). Imagined walking = IW_HUE {H_IW}, resting control = "
                    f"REST_HUE {H_REST} (03_figures/project_palette.py). Hue encodes the CONDITION only; "
                    f"no hue carries a verdict. Decision boundary and accuracy marker in INK {INK}; "
                    f"pooled guide in MID {MID}."),
        ("producer", "03_figures/fig7_gate_discrimination.py"),
    ]
    meta = GE.meta(meta)
    pd.DataFrame(meta, columns=["key", "value"]).to_csv(MD, index=False)
    print(f"wrote {os.path.relpath(MD, ROOT)}  ({len(meta)} keys)")


if __name__ == "__main__":
    main()
