#!/usr/bin/env python
"""Figure 3 — Gait-phase locking of the imagined-walking representation

panel d — paired per-session gait-phase locking |c| (k1), imagined walking vs rest control

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT / "code"))
import analysis_config as ec
SRC = ec.sd_in("fig3_master_source_data.csv")
RESP = ec.sd_in("fig3_def_resnonres_source_data.csv")
SD  = ROOT / "04_source_data/fig3/fig3d_locking_paired_source_data.csv"
MD  = ROOT / "04_source_data/fig3/fig3d_locking_paired_metadata.csv"

d = pd.read_csv(SRC).merge(pd.read_csv(RESP)[["session", "resp"]], on="session", validate="1:1")
assert len(d) == 80 and d[["mi_k1", "rest_k1"]].notna().all().all(), "expected 80 sessions with finite k1 |c|"
iw, rc = d.mi_k1.values, d.rest_k1.values
out = d[["session"]].copy()
out["subject"] = d.session.str.split("_").str[0]
out["responder"] = d.resp.astype(bool)
out["IW_c_k1"], out["rest_c_k1"] = iw, rc
out["IW_minus_rest"] = iw - rc
out.to_csv(SD, index=False)

w = stats.wilcoxon(iw, rc)
q = lambda v: np.percentile(v, [25, 50, 75])
dif = iw - rc
bs_rng = np.random.default_rng(0)
bs_med = np.array([np.median(dif[bs_rng.integers(0, len(dif), len(dif))]) for _ in range(1000)])
dif_lo, dif_hi = np.percentile(bs_med, [2.5, 97.5])
rows = [
    ("panel", "Fig. 3 panel d - paired per-session k1 |c|, imagined walking vs resting control"),
    ("input", SRC.relative_to(ROOT).as_posix()),
    ("n_sessions", 80), ("n_participants", 42),
    ("IW_median", f"{q(iw)[1]:.4f}"), ("IW_IQR", f"{q(iw)[0]:.4f}-{q(iw)[2]:.4f}"),
    ("rest_median", f"{q(rc)[1]:.4f}"), ("rest_IQR", f"{q(rc)[0]:.4f}-{q(rc)[2]:.4f}"),
    ("median_difference", f"{np.median(dif):.4f}"),
    ("median_difference_CI95", f"{dif_lo:.4f}-{dif_hi:.4f} (percentile bootstrap over sessions, B = 1000, seed 0)"),
    ("wilcoxon_W", w.statistic), ("wilcoxon_p", f"{w.pvalue:.3g}"),
    ("n_IW_greater_than_rest", int((iw > rc).sum())),
]
pd.DataFrame(rows, columns=["key", "value"]).to_csv(MD, index=False)
for k, v in rows: print(f"{k:26s} {v}")


import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
MM = 1/25.4
FS = 1.2
INK, MID, PALE = "#1A1A1A", "#666666", "#B3B3B3"
plt.rcParams.update({"font.family": "Arial", "font.size": 7*FS, "axes.linewidth": 1.4,
                     "xtick.major.width": 1.4, "ytick.major.width": 1.4,
                     "xtick.major.size": 3.0, "ytick.major.size": 3.0, "pdf.fonttype": 42})
PNG = ROOT / "figures_out/fig3/fig3d_locking_paired.png"
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(52*MM, 52*MM), dpi=400)
_WH, _BOX, _MED, _PAIR, _S, _LW_IW, _LW_RC, _BRK = ((1.3, 1.4, 2.4, .3, 5, .35, .4, 1.1))
BW = .30
import project_palette as P
from matplotlib.colors import to_rgba
for x, v, hue in ((0, iw, P.IW_HUE), (1, rc, P.REST_HUE)):
    face = to_rgba(hue, P.COND_FILL_ALPHA)
    q1, md, q3 = np.percentile(v, [25, 50, 75]); w10, w90 = np.percentile(v, [10, 90])
    ax.plot([x, x], [w10, q1], color=hue, lw=_WH, zorder=2)
    ax.plot([x, x], [q3, w90], color=hue, lw=_WH, zorder=2)
    for yy in (w10, w90):
        ax.plot([x-.09, x+.09], [yy, yy], color=hue, lw=_WH, solid_capstyle="butt", zorder=2)
    ax.add_patch(plt.Rectangle((x-BW/2, q1), BW, q3-q1, facecolor=face,
                               edgecolor=hue, lw=_BOX, zorder=3))
    ax.plot([x-BW/2, x+BW/2], [md, md], color=hue, lw=_MED, solid_capstyle="butt", zorder=4)
xs_iw = -.40 + rng.uniform(-.055, .055, len(iw))
xs_rc = 1.40 + rng.uniform(-.055, .055, len(rc))
for a_, b_, p_, q_ in zip(iw, rc, xs_iw, xs_rc):
    ax.plot([p_, q_], [a_, b_], color=PALE, lw=_PAIR, alpha=.40, zorder=1)
ax.scatter(xs_iw, iw, s=_S, facecolor=to_rgba(P.IW_HUE, P.COND_FILL_ALPHA), edgecolor=P.IW_HUE, lw=_LW_IW, zorder=5)
ax.scatter(xs_rc, rc, s=_S, facecolor=to_rgba(P.REST_HUE, P.COND_FILL_ALPHA), edgecolor=P.REST_HUE, lw=_LW_RC, zorder=5)
stars = "***" if w.pvalue < 1e-3 else "**" if w.pvalue < 1e-2 else "*" if w.pvalue < .05 else "n.s."
Y0, Y1 = .02, .262
yb, tick = Y1 + .005*(Y1-Y0), .018*(Y1-Y0)
ax.plot([0, 0, 1, 1], [yb, yb+tick, yb+tick, yb], color=INK, lw=_BRK,
        solid_capstyle="butt", clip_on=False, zorder=6)
ax.text(.5, yb+tick+.012*(Y1-Y0), stars, ha="center", va="bottom",
        fontsize=10.5*FS if stars != "n.s." else 10.5*FS*0.65,
        color=INK, clip_on=False, zorder=6)
ax.set_xlim(-.70, 1.70); ax.set_xticks([0, 1])
ax.set_xticklabels(["Imagined\nwalking", "Rest\ncontrol"], fontsize=6.8*FS)
ax.set_ylim(Y0, Y1); ax.set_yticks([.05, .10, .15, .20, .25])
ax.set_ylabel("Gait-phase locking |c| ($\\mathrm{k}_1$)", fontsize=7*FS, labelpad=2)
ax.spines[["top", "right"]].set_visible(False)
ax.set_box_aspect(1)
fig.savefig(PNG, dpi=400, bbox_inches="tight", facecolor="white", pad_inches=0.04)
plt.close(fig)
print(f"wrote {PNG}")
