"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

inter-joint coordination matrix

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from project_goni import polarity_vector

ROOT = Path(__file__).resolve().parents[1]

import analysis_config as _ec
GAIT_BUILD_ID = "main"
_ERA_TAG = "main"

NPZ = _ec.path("h_abd_cohort")
OUT_PNG = _ec.scoped(ROOT / "figures_out/fig2/f2_panelB_cohort_coordmatrix.png")
OUT_CSV = _ec.sd("fig2_coord_matrix_source_data.csv")

plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.labelsize": 12,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5})

z = np.load(NPZ, allow_pickle=True)
labels = [str(x) for x in z["labels"]]
obs_mat, dec_mat, shuf_mat = z["obs_mat"], z["dec_mat"], z["shuf_mat"]
real_sim, null_sim = z["real_sim"], z["null_sim"]

_pol = np.asarray(polarity_vector(labels), dtype=float)
_S = np.outer(_pol, _pol)
obs_mat = obs_mat * _S[None, :, :]
dec_mat = dec_mat * _S[None, :, :]
shuf_mat = shuf_mat * _S[None, :, :]

Co = np.nanmean(obs_mat, axis=0)
Cp = np.nanmean(dec_mat, axis=0)
Cs = np.nanmean(shuf_mat, axis=0)

_iu = np.triu_indices(6, 1)
_rs = []
for _i in range(obs_mat.shape[0]):
    _vo, _vd = obs_mat[_i][_iu], dec_mat[_i][_iu]
    _m = np.isfinite(_vo) & np.isfinite(_vd)
    if _m.sum() >= 6:
        _rs.append(np.corrcoef(_vo[_m], _vd[_m])[0, 1])
real_sim = np.array(_rs, dtype=float)

rs = real_sim[np.isfinite(real_sim)]
ns = null_sim[np.isfinite(null_sim)]
rsmed, rslo, rshi = np.median(rs), np.percentile(rs, 25), np.percentile(rs, 75)
nsmed, nshi = np.median(ns), np.percentile(ns, 95)

fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.7))
fig.subplots_adjust(left=0.06, right=0.92, bottom=0.22, top=0.86, wspace=0.05)
im = None
for k, (C, t) in enumerate([(Co, "Reference"), (Cp, "Decoded"),
                            (Cs, "Decoded (phase-shuffled)")]):
    ax = axes[k]
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels if k == 0 else [], fontsize=9)
    ax.set_title(t, fontsize=12)
cax = None
for _ax in axes:
    _c = make_axes_locatable(_ax).append_axes("right", size="6%", pad=0.08)
    if _ax is axes[2]:
        cax = _c
    else:
        _c.set_axis_off()
cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
cb.set_label("structure similarity", fontsize=10.5, labelpad=3)
cb.ax.tick_params(labelsize=10.5)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=300)
plt.close(fig)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["joint_i", "joint_j", "reference_corr", "decoded_corr", "shuffled_abs_corr"])
    for i in range(6):
        for j in range(6):
            w.writerow([labels[i], labels[j], f"{Co[i,j]:.4f}", f"{Cp[i,j]:.4f}", f"{abs(Cs[i,j]):.4f}"])
    w.writerow([])
    w.writerow(["structure_similarity_real_median", f"{rsmed:.4f}", "IQR", f"{rslo:.4f}", f"{rshi:.4f}"])
    w.writerow(["null_median", f"{nsmed:.4f}", "null_p95", f"{nshi:.4f}", "n_sessions", len(rs)])

print(f"coordination similarity: real median {rsmed:.3f} [{rslo:.3f}-{rshi:.3f}]"
      f"  vs null median {nsmed:.3f} (p95 {nshi:.3f})  n={len(rs)}")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_CSV}")


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


_m = (_rew.fullmatch(r"cohort_reconstruction_(.+)\.npz", NPZ.name)
      or _rew.fullmatch(rf"abd_cohort_({_rew.escape(_ec.TAG)})\.npz", NPZ.name))
if _m is None:
    raise SystemExit(f"INPUT CHECK: no decode tag in {NPZ.name}; refusing to stamp an build_id.")
_ERA_DERIVED = _m.group(1)

_write_meta(_ec.sd("fig2_coord_matrix_metadata.csv"), [
    ("producer", "03_figures/fig2_coord_matrix.py"),
    ("cohort", f"healthy; {obs_mat.shape[0]} sessions present in the input dump"),
    ("input", str(NPZ.relative_to(ROOT)).replace("\\", "/")),
    ("n_sessions_in_dump", obs_mat.shape[0]),
    ("n_sessions_structure_similarity", len(rs)),
    ("n_null_draws_finite", len(ns)),
    ("joint_labels", "; ".join(labels)),
    ("display_polarity", "goniometers are mounted mirror-image on the two legs; the polarity "
                         "vector from 03_figures/project_goni.py is applied as an outer product, "
                         "flipping 9 of the 15 off-diagonal cells"),
    ("outputs", f"{OUT_PNG.relative_to(ROOT).as_posix()}; "
                f"{OUT_CSV.relative_to(ROOT).as_posix()}"),
])
