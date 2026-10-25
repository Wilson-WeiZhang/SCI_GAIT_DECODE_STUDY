#!/usr/bin/env python3
"""Figure 3 — Gait-phase locking of the imagined-walking representation

panel g - responder | non-responder blocks, side by side

Reads:
  the aligned per-session recordings (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

ROOT = Path(__file__).resolve().parents[1]
D = sio.loadmat(ROOT / "04_source_data/fig3ch_eeglab_maps.mat")
plt.rcParams.update({"font.family": "Arial"})

chans = [str(c[0]) for c in D["channels"].ravel()]
fracs = D["fracs"].ravel(); harms = D["harms"].ravel().astype(int)
clim  = float(D["clim_abs"].ravel()[0])
n_r, n_n = int(D["n_resp"].ravel()[0]), int(D["n_non"].ravel()[0])
R, N, RS, NS = D["resp_signed"], D["non_signed"], D["resp_sig"], D["non_sig"]
cmap = matplotlib.colors.ListedColormap(np.clip(D["cmap"], 0, 1))
MASK = dict(marker="o", markerfacecolor="k", markeredgecolor="white",
            markersize=6.0, markeredgewidth=0.9, linewidth=0)

EEG_SET = ROOT / "data/prep_data/healthy/eeg/SUB_37_sess02_step2.set"
raw = mne.io.read_raw_eeglab(str(EEG_SET), preload=False, verbose="ERROR")
keep = [c for c in raw.ch_names if c in chans]
info = raw.copy().pick(keep).info
idx  = [chans.index(c) for c in keep]

PH  = [0.0, 0.25, 0.50, 0.75, 1.0]
ROW = [(1, r"$\mathregular{k_1}$"), (2, r"$\mathregular{k_2}$"), (3, r"$\mathregular{k_3}$")]
NC  = len(PH)

FW, FH = 11.0, 11.5
fig = plt.figure(figsize=(FW, FH))
MX, MW = 0.070, 0.158
MAP = 0.74
MH  = MW * MAP * FW / FH
RP  = MH * 1.12
BLOCK_Y0 = [0.795, 0.310]
rows = []
for bi, (M, S, gname, ng) in enumerate([(R, RS, "Responder", n_r), (N, NS, "Non-responder", n_n)]):
    y0 = BLOCK_Y0[bi]
    fig.text(MX + NC * MW / 2, y0 + MH + 0.038, f"{gname}  (n={ng})",
             ha="center", va="bottom", fontsize=18)
    for ri, (h, hlab) in enumerate(ROW):
        hi = int(np.where(harms == h)[0][0])
        for ci, f in enumerate(PH):
            pi = int(np.argmin(np.abs(fracs - f)))
            ax = fig.add_axes([MX + ci * MW, y0 - ri * RP, MW * MAP, MH])
            v, msk = M[idx, pi, hi], S[idx, pi, hi].astype(bool)
            mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap=cmap, vlim=(-clim, clim),
                                 mask=msk, mask_params=MASK, contours=6, sensors=True,
                                 sphere="eeglab", res=256, image_interp="cubic", extrapolate="head")
            if ri == 0:
                ax.set_title(f"{100*f:.0f}%", fontsize=16, pad=5)
            if ci == 0:
                ax.text(-0.30, 0.5, hlab, transform=ax.transAxes, ha="center", va="center", fontsize=19)
            rows.append(dict(group=gname.lower(), n=ng, harmonic=f"{h}f0",
                             cycle_fraction=float(fracs[pi]),
                             rms_signed=float(np.sqrt(np.mean(v ** 2))),
                             n_sig_channels=int(msk.sum()), n_channels=len(msk)))

CB_W_IN, CB_H_IN = 0.0085 * FW, 0.46 * FH
_pad_w, _pad_h = 1.90, 0.30
cfig = plt.figure(figsize=(_pad_w, CB_H_IN + _pad_h))
ccax = cfig.add_axes([0.30, _pad_h / 2 / (CB_H_IN + _pad_h),
                      CB_W_IN / _pad_w, CB_H_IN / (CB_H_IN + _pad_h)])
sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-clim, clim), cmap=cmap)
_ticks = [t for t in (-0.10, -0.05, 0.0, 0.05, 0.10) if abs(t) <= clim]
cb = cfig.colorbar(sm, cax=ccax, ticks=_ticks)
cb.ax.set_yticklabels([("0" if t == 0 else f"{t:.2f}".replace("-", "\u2212"))
                       for t in _ticks], fontsize=16)
cb.ax.yaxis.set_label_position("left")
cb.set_label("Phase modulation", fontsize=16, labelpad=10)
out_cb = ROOT / "figures_out/fig3/fig3g_responder_blocks_colorbar.png"
cfig.savefig(out_cb, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
plt.close(cfig)
print("wrote", out_cb,
      f"-- bar {CB_W_IN*25.4:.2f} x {CB_H_IN*25.4:.1f} mm on this canvas; place it at panel g's own "
      f"scale (g is {FW*25.4:.0f} mm wide as rendered)")

out = ROOT / "figures_out/fig3/fig3g_responder_blocks.png"
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("wrote", out)
df = pd.DataFrame(rows)
df.to_csv(ROOT / "fig3g_responder_blocks_source_data.csv (not distributed)", index=False)
print(df.pivot_table(index=["harmonic","cycle_fraction"], columns="group",
                     values="n_sig_channels").to_string())


import csv as _csv
import os as _osw
import re as _rew


def _write_meta(_path, _rows):
    _tmp = _path.parent / (_path.name + ".tmp")
    with _tmp.open("w", newline="", encoding="utf-8") as _f:
        _w = _csv.writer(_f)
        _w.writerow(["key", "value"])
        _w.writerows([(k, v) for k, v in _rows])
    _osw.replace(_tmp, _path)
    print("wrote", _path)


_HPS_SRC = (ROOT / "03_figures/fig3_harmonic_phase_responder_split.py").read_text(encoding="utf-8")
_dumps = sorted(set(_rew.findall(r'"(diag_[A-Za-z0-9_]+)/"', _HPS_SRC)))
if len(_dumps) != 1:
    raise SystemExit(f"INPUT CHECK: expected exactly one dump dir in "
                     f"fig3_harmonic_phase_responder_split.py, found {_dumps}; "
                     "refusing to stamp an build_id.")
_DUMP = _dumps[0]
_ERA_DERIVED = (" " if "" in _DUMP else "") + \
    ("" if _DUMP.endswith("") else "reference")
_n_cohort = int(D["n_cohort"].ravel()[0])
_write_meta(ROOT / "04_source_data/fig3/fig3g_responder_blocks_metadata.csv", [
    ("figure", "Fig. 3 panel g - responder | non-responder blocks side by side, each "
               "3 harmonics x 5 gait-phase points (0/25/50/75/100% of the cycle)"),
    ("producer", "03_figures/fig3g_responder_blocks.py"),
    ("cohort", f"healthy; n_cohort = {_n_cohort} sessions in the .mat"),
    ("input_maps", "04_source_data/fig3ch_eeglab_maps.mat"),
    ("input_maps_exporter", "03_figures/fig3_export_for_eeglab.py"),
    ("input_montage_set", str(EEG_SET.relative_to(ROOT)).replace("\\", "/")),
    ("montage_note", "electrode geometry is taken from the recording's own .set montage"),
    ("n_cohort", _n_cohort),
    ("n_responder", n_r),
    ("n_non_responder", n_n),
    ("n_channels", len(chans)),
    ("harmonics", "; ".join(str(int(h)) for h in harms)),
    ("cycle_fractions", "; ".join(f"{f:g}" for f in fracs)),
    ("clim_abs", float(clim)),
    ("outputs", "figures_out/fig3/fig3g_responder_blocks.png; "
                "figures_out/fig3/fig3g_responder_blocks_colorbar.png; "
                "fig3g_responder_blocks_source_data.csv (not distributed)"),
])
