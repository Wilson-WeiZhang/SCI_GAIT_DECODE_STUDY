"""Figure 4 — Shared locomotor representation across imagined and real walking

panel g - 3 x 3 electrode RDM grid (k1/k2/k3 x real walking / imagined walking / rest)

Reads:
  04_source_data/fig4/fig4c_electrode_rdm_source_data.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "axes.linewidth": 1.9,
                     "xtick.major.width": 1.9, "ytick.major.width": 1.9,
                     "xtick.major.size": 3.0, "ytick.major.size": 3.0,
                     "font.size": 12.9, "xtick.labelsize": 11.3, "ytick.labelsize": 11.3,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic",
                     "pdf.fonttype": 42})
ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data/fig4"
IN = SD / "fig4c_electrode_rdm_source_data.csv"
OUT = ROOT / "figures_out/fig4/fig4c_rdm_grid.png"

COND = [("RW", "Real walking"), ("IW", "Imagined walking"), ("REST", "Rest control")]
ROW = [(1, r"$\mathrm{k}_1$"), (2, r"$\mathrm{k}_2$"), (3, r"$\mathrm{k}_3$")]
CMAP = LinearSegmentedColormap.from_list("rdm", ["#8c7a64", "#ffffff", "#1f4e79"])

d = pd.read_csv(IN)
chans = sorted(d.channel_i.unique())
pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
order = sorted(chans, key=lambda c: -pos[c][1])
M = {}
for k, _ in ROW:
    for c, _ in COND:
        s = d[(d.harmonic_k == k) & (d.condition == c)].pivot(index="channel_i", columns="channel_j", values="crossnobis_d")
        M[(k, c)] = s.loc[order, order].to_numpy()
vmax = float(np.nanquantile(np.abs(np.concatenate([m.ravel() for m in M.values()])), 0.98))

CELL = 2.0
GAP = 0.06 * CELL
GFS = 1.5
L_IN = 0.50 * GFS
T_IN = 0.30 * GFS
B_IN = 0.34 * GFS
CB_PAD, CB_W = 0.08 * CELL, 0.06 * CELL
CB_LAB_IN = 0.78 * GFS
GRID = 3 * CELL + 2 * GAP
FW = L_IN + GRID + CB_PAD + CB_W + CB_LAB_IN
FH = T_IN + GRID + B_IN

fig, axes = plt.subplots(3, 3, figsize=(FW, FH))
fig.subplots_adjust(left=L_IN / FW, right=(L_IN + GRID) / FW,
                    bottom=B_IN / FH, top=(B_IN + GRID) / FH,
                    wspace=0.06, hspace=0.06)
im = None
for ri, (k, kl) in enumerate(ROW):
    for ci, (c, cl) in enumerate(COND):
        ax = axes[ri, ci]
        im = ax.imshow(M[(k, c)], cmap=CMAP, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(cl, fontsize=11.3 * GFS, pad=4 * GFS)
        if ci == 0:
            ax.set_ylabel(kl, fontsize=14 * GFS, rotation=0, labelpad=14 * GFS, va="center")
cax = fig.add_axes([(L_IN + GRID + CB_PAD) / FW, B_IN / FH, CB_W / FW, GRID / FH])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Cross-validated distance", fontsize=11.3 * GFS, labelpad=4 * GFS)
cb.ax.tick_params(labelsize=11.3 * GFS, length=3.0, width=1.9)
fig.text((L_IN + GRID / 2) / FW, 0.10 * GFS / FH, "EEG channels (anterior → posterior)",
         ha="center", va="bottom", fontsize=11.3 * GFS)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300)
plt.close(fig)

pd.DataFrame([
    ("panel", "Fig. 4g - cross-validated electrode RDM grid, k1/k2/k3 x RW/IW/REST"),
    ("rdm_source", "04_source_data/fig4/fig4c_electrode_rdm_source_data.csv written by 03_figures/fig4b_rsa_per_harmonic_k123.py"),
    ("n_channels", len(order)),
    ("channel_order", "anterior->posterior by standard_1005 y coordinate (ordering only)"),
    ("colour_scale", f"shared symmetric +/-{vmax:.4f} = 98th percentile of |d| over all nine matrices"),
    ("colormap", "tan (#8c7a64) - white - blue (#1f4e79)"),
    ("related", "panel e (03_figures/fig4_rsa_persession_spin.py) quantifies these matrices: 2nd-order Spearman of the IW vs RW upper triangles + spin null"),
], columns=["key", "value"]).to_csv(SD / "fig4c_rdm_grid_metadata.csv", index=False)
print(f"wrote {OUT.relative_to(ROOT)}  channels={len(order)}  vmax={vmax:.4f}")
