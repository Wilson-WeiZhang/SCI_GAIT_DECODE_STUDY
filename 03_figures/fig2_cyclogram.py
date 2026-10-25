"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

panel h - inter-joint coordination cyclograms

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection, PolyCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS

import analysis_config as _ec
GAIT_BUILD_ID = _ec.GAIT_BUILD_ID
if GAIT_BUILD_ID not in ("main", "reference", "main") + _ec.DECODER_BUILDS:
    raise SystemExit(f"GAIT_BUILD_ID must be 'main', 'reference' or '', got {GAIT_BUILD_ID!r}")
_ERA_TAG = _ec.TAG

NPZ = _ec.path("h_abd_cohort")
OUT = ROOT / "figures_out/fig2"
SRC = ROOT / "04_source_data/fig2"
NB = 50
OBS = "#5f5e5a"
NM = {0: "LHip", 1: "LKnee", 2: "LAnk"}
plt.rcParams.update({"font.family": "Arial", "font.size": 12.5, "axes.labelsize": 12.5,
                     "axes.linewidth": 1.95,
                     "xtick.major.width": 1.95, "ytick.major.width": 1.95,
                     "xtick.major.size": 3.0, "ytick.major.size": 3.0,
                     "xtick.labelsize": 11.0, "ytick.labelsize": 11.0})

_LW_DEC, _LW_REF, _ARR_LW, _ARR_MS, _ARR_HALO = ((2.4, 1.6, 2.0, 16, 3.2))

z = np.load(NPZ, allow_pickle=True)
obs_curves, dec_curves = z["obs_curves"], z["dec_curves"]


def cl(v):
    return np.append(v, v[0])


def _tube(mx, my, cx, cy, smooth=3):
    """Outer/inner edge curves of a 95% CI tube around a closed loop. Uses a
    PERIODIC tangent (no seam spike), clips egregious single-point CI spikes, and
    lightly smooths the half-width so the band has no sharp outliers."""
    mx = np.asarray(mx, float); my = np.asarray(my, float)
    tx = np.roll(mx, -1) - np.roll(mx, 1)
    ty = np.roll(my, -1) - np.roll(my, 1)
    tn = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tn, tx / tn
    hw = np.hypot(cx * nx, cy * ny)
    hw = np.minimum(hw, 2.5 * np.median(hw))
    if smooth > 1:
        k = np.ones(smooth) / smooth
        hw = np.convolve(np.r_[hw[-smooth:], hw, hw[:smooth]], k, "same")[smooth:-smooth]
    outer = np.column_stack([mx + hw * nx, my + hw * ny])
    inner = np.column_stack([mx - hw * nx, my - hw * ny])
    return np.vstack([outer, outer[0]]), np.vstack([inner, inner[0]])


def ci_outline(ax, mx, my, cx, cy, lw=0.6, color=None, cmap=None, phase_closed=None):
    """95% CI as thin SOLID outline envelopes (outer + inner curves, no fill)."""
    for env in _tube(mx, my, cx, cy):
        if cmap is not None:
            ep = env.reshape(-1, 1, 2)
            es = np.concatenate([ep[:-1], ep[1:]], axis=1)
            ax.add_collection(LineCollection(es, cmap=cmap, array=phase_closed[:-1], lw=lw, zorder=1))
        else:
            ax.plot(env[:, 0], env[:, 1], color=color, lw=lw, zorder=1)


def ci_fill(ax, mx, my, cx, cy, alpha, color=None, cmap=None, phase_closed=None):
    """Per-segment-quad 95% CI fill (shading). Per-quad (not one big polygon) so a
    normal flip at a cusp can't fill a large wedge."""
    outer, inner = _tube(mx, my, cx, cy)
    quads = [np.array([outer[i], outer[i + 1], inner[i + 1], inner[i]])
             for i in range(len(outer) - 1)]
    if cmap is not None:
        pc = PolyCollection(quads, array=phase_closed[:-1], cmap=cmap, alpha=alpha,
                            edgecolors="none", zorder=1)
    else:
        pc = PolyCollection(quads, facecolors=color, alpha=alpha, edgecolors="none", zorder=1)
    ax.add_collection(pc)


phase = np.linspace(0, 100, NB)
ARR_I0, ARR_I1 = 34, 38
pairs = [(0, 1), (0, 2), (1, 2)]
PAIR_RANGE = []
fig, axd = plt.subplots(1, 3, figsize=(7.6, 3.4))
fig.subplots_adjust(left=0.085, right=0.915, bottom=0.185, top=0.965, wspace=0.30)
SINGLES = [plt.subplots(figsize=(2.55, 2.75)) for _ in range(3)]
for _f, _a in SINGLES:
    _f.subplots_adjust(left=0.235, right=0.985, bottom=0.215, top=0.975)
lc_ref = None
D_rows = []
H_rows = []
for _pi, (ax, (a, b)) in enumerate(zip(axd, pairs)):
    v = (np.isfinite(obs_curves[:, a, :]).all(1) & np.isfinite(obs_curves[:, b, :]).all(1)
         & np.isfinite(dec_curves[:, a, :]).all(1) & np.isfinite(dec_curves[:, b, :]).all(1))
    nuse = int(v.sum())
    Oa, Ob, Da, Db = obs_curves[v, a], obs_curves[v, b], dec_curves[v, a], dec_curves[v, b]
    moa, mob = Oa.mean(0), Ob.mean(0)
    sox, soy = moa.std() + 1e-9, mob.std() + 1e-9
    mxo, myo = (moa - moa.mean()) / sox, (mob - mob.mean()) / soy
    cxo = 1.96 * Oa.std(0) / np.sqrt(nuse) / sox
    cyo = 1.96 * Ob.std(0) / np.sqrt(nuse) / soy
    mdx, mdy = Da.mean(0), Db.mean(0)
    sx, sy = mdx.std() + 1e-9, mdy.std() + 1e-9
    mxd, myd = (mdx - mdx.mean()) / sx, (mdy - mdy.mean()) / sy
    cxd = 1.96 * Da.std(0) / np.sqrt(nuse) / sx
    cyd = 1.96 * Db.std(0) / np.sqrt(nuse) / sy
    ph = np.append(phase, 100.0)
    ci_fill(ax, mxo, myo, cxo, cyo, 0.20, color="#6f747a")
    ci_fill(ax, mxd, myd, cxd, cyd, 0.32, cmap="twilight", phase_closed=ph)
    for _ser, _mx, _my, _cx, _cy in (("reference", mxo, myo, cxo, cyo), ("decoded", mxd, myd, cxd, cyd)):
        _out, _inn = _tube(_mx, _my, _cx, _cy)
        H_rows += [(NM[a], NM[b], nuse, _ser, _i, phase[_i], _mx[_i], _my[_i],
                    _out[_i, 0], _out[_i, 1], _inn[_i, 0], _inn[_i, 1]) for _i in range(NB)]
    ax.plot(cl(mxo), cl(myo), color=OBS, lw=1.6, ls="--", zorder=2)
    xx, yy = cl(mxd), cl(myd)
    pts = np.array([xx, yy]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="twilight", array=ph[:-1], lw=2.4, zorder=3)
    ax.add_collection(lc); lc_ref = lc
    ar = ax.annotate("", xy=(mxd[ARR_I1], myd[ARR_I1]), xytext=(mxd[ARR_I0], myd[ARR_I0]),
                     arrowprops=dict(arrowstyle="-|>", mutation_scale=16, color="#111", lw=2.0),
                     zorder=6)
    ar.arrow_patch.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
    allx = np.concatenate([cl(mxo), cl(mxd)]); ally = np.concatenate([cl(myo), cl(myd)])
    LIM_PAD = 0.15
    _lo = min(allx.min(), ally.min()); _hi = max(allx.max(), ally.max())
    _pad = LIM_PAD * (_hi - _lo)
    PAIR_RANGE.append((_lo - _pad, _hi + _pad))
    ax.set_xlabel(PS.jlab(NM[a])); ax.set_ylabel(PS.jlab(NM[b]))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    _sa = SINGLES[_pi][1]
    ci_fill(_sa, mxo, myo, cxo, cyo, 0.20, color="#6f747a")
    ci_fill(_sa, mxd, myd, cxd, cyd, 0.32, cmap="twilight", phase_closed=ph)
    _sa.plot(cl(mxo), cl(myo), color=OBS, lw=_LW_REF, ls="--", zorder=2)
    _lc = LineCollection(segs, cmap="twilight", array=ph[:-1], lw=_LW_DEC, zorder=3)
    _sa.add_collection(_lc)
    _ar = _sa.annotate("", xy=(mxd[ARR_I1], myd[ARR_I1]), xytext=(mxd[ARR_I0], myd[ARR_I0]),
                       arrowprops=dict(arrowstyle="-|>", mutation_scale=_ARR_MS, color="#111", lw=_ARR_LW),
                       zorder=6)
    _ar.arrow_patch.set_path_effects([pe.withStroke(linewidth=_ARR_HALO, foreground="white")])
    _sa.set_xlabel(PS.jlab(NM[a])); _sa.set_ylabel(PS.jlab(NM[b]))
    for s_ in ("top", "right"):
        _sa.spines[s_].set_visible(False)
    D_rows.append((f"{NM[a]}-{NM[b]}", nuse))
_LO = min(r[0] for r in PAIR_RANGE); _HI = max(r[1] for r in PAIR_RANGE)
for ax in axd:
    ax.set_xlim(_LO, _HI); ax.set_ylim(_LO, _HI)
    ax.set_aspect("equal", adjustable="box")
print(f"shared cyclogram window: {_LO:.2f} .. {_HI:.2f} (equal aspect)")

from mpl_toolkits.axes_grid1 import make_axes_locatable
cax = make_axes_locatable(axd[2]).append_axes("right", size="6%", pad=0.08)
cb = fig.colorbar(lc_ref, cax=cax, ticks=[0, 50, 100])
cb.set_label("Gait cycle (%)", fontsize=11.0, labelpad=-1)
cb.ax.tick_params(labelsize=11.0)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(_ec.scoped(OUT / "f2_panelD_cohort_cyclogram.png"), dpi=300)
plt.close(fig)

for _pi, ((_f, _a), (a, b)) in enumerate(zip(SINGLES, pairs)):
    _a.set_xlim(_LO, _HI); _a.set_ylim(_LO, _HI)
    _a.set_aspect("equal", adjustable="box")
    _p = _ec.scoped(OUT / f"f2_panelD_cyclogram_{NM[a]}_{NM[b]}.png")
    _f.savefig(_p, dpi=300)
    plt.close(_f)
    print("wrote", _p.name)

_cf = plt.figure(figsize=(0.95, 2.75))
_cax = _cf.add_axes([0.06, 0.215 * 2.75 / 2.75 * 0.20, 0.16, 0.72])
_cb = _cf.colorbar(lc_ref, cax=_cax, ticks=[0, 50, 100])
_cb.set_label("Gait cycle (%)", fontsize=11.0, labelpad=-1)
_cb.ax.tick_params(labelsize=11.0)
_cp = _ec.scoped(OUT / "f2_panelD_cyclogram_colorbar.png")
_cf.savefig(_cp, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(_cf)
print("wrote", _cp.name)

SRC.mkdir(parents=True, exist_ok=True)
with _ec.sd("fig2_cyclogram_source_data.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["pair", "n_paired_sessions"])
    for pair, nuse in D_rows:
        w.writerow([pair, nuse])

print("cyclogram n per pair:", D_rows)
print("wrote", _ec.scoped(OUT / "f2_panelD_cohort_cyclogram.png"))
print("wrote", _ec.sd("fig2_cyclogram_source_data.csv"))


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

_write_meta(_ec.sd("fig2_cyclogram_metadata.csv"), [
    ("figure", "Fig. 2 panel h - inter-joint coordination cyclograms "
               "(hip-knee / hip-ankle / knee-ankle), phase-coloured"),
    ("producer", "03_figures/fig2_cyclogram.py"),
    ("cohort", f"healthy; {obs_curves.shape[0]} sessions present in the input dump"),
    ("input", str(NPZ.relative_to(ROOT)).replace("\\", "/")),
    ("n_sessions_in_dump", obs_curves.shape[0]),
    ("n_cycle_bins", NB),
    ("pairing_rule", "per-pair finite-curve filter: a session enters a cyclogram only when "
                     "BOTH joints of that pair are present, so each panel carries its own n"),
    ("normalisation", "each loop z-scored separately on x and y (unit scale), so the panel "
                      "compares shape/phase, not amplitude"),
    ("ci_band", "session-level 95% CI = 1.96*SD/sqrt(n), half-width projected onto the local "
                "loop normal, drawn as a translucent band (not discrete ellipses)"),
    ("outputs", "; ".join(_p.relative_to(ROOT).as_posix() for _p in (
        _ec.scoped(OUT / "f2_panelD_cohort_cyclogram.png"),
        _ec.scoped(OUT / "f2_panelD_cyclogram_LHip_LKnee.png"),
        _ec.scoped(OUT / "f2_panelD_cyclogram_LHip_LAnk.png"),
        _ec.scoped(OUT / "f2_panelD_cyclogram_LKnee_LAnk.png"),
        _ec.scoped(OUT / "f2_panelD_cyclogram_colorbar.png"),
        _ec.sd("fig2_cyclogram_source_data.csv")))),
] + [(f"n_paired_sessions_{_pair}", _n) for _pair, _n in D_rows])

from safe_write import safe_write_csv as _swc
_swc(_ec.sd("fig2_cyclogram_trajectories_source_data.csv"),
     ["x_joint", "y_joint", "n_paired_sessions", "series", "cycle_bin", "gait_cycle_pct", "x", "y",
      "ci_outer_x", "ci_outer_y", "ci_inner_x", "ci_inner_y"], H_rows)
_swc(_ec.sd("fig2_cyclogram_trajectories_metadata.csv"), ["key", "value"], [
    ("panel", "Fig. 2h - left-limb triple cyclogram (hip-knee, hip-ankle, knee-ankle phase planes)"),
    ("producer", "03_figures/fig2_cyclogram.py"),
    ("input", str(NPZ.relative_to(ROOT)).replace("\\", "/")),
    ("n", "; ".join(f"{_pair}: {_n} paired sessions" for _pair, _n in D_rows)),
    ("rows", f"{len(H_rows)} = 3 joint pairs x 2 series x {NB} gait-cycle bins"),
    ("x_joint", "joint on the horizontal axis"),
    ("y_joint", "joint on the vertical axis"),
    ("n_paired_sessions", "sessions with both joints of the pair present in the reference and decoded curves"),
    ("series", "reference (grey dashed line, grey CI shading) or decoded (line and CI shading coloured by gait_cycle_pct)"),
    ("cycle_bin", f"gait-cycle bin index 0..{NB - 1}; each loop is closed by joining the last bin to bin 0"),
    ("gait_cycle_pct", "gait-cycle phase (%) of the bin; sets the colour of the decoded segment starting at this bin"),
    ("x", "plotted x: session-mean curve of x_joint, z-scored over the cycle"),
    ("y", "plotted y: session-mean curve of y_joint, z-scored over the cycle"),
    ("ci_outer_x", "outer edge of the drawn 95% CI band, x"),
    ("ci_outer_y", "outer edge of the drawn 95% CI band, y"),
    ("ci_inner_x", "inner edge of the drawn 95% CI band, x"),
    ("ci_inner_y", "inner edge of the drawn 95% CI band, y"),
    ("ci_band", "95% CI across sessions (1.96*SD/sqrt(n) on each axis, in z units), half-width projected onto "
                "the local loop normal, capped at 2.5x its median and smoothed by a 3-bin periodic moving "
                "average; shading = quadrilaterals between consecutive outer and inner edge points"),
])
print("wrote", _ec.sd("fig2_cyclogram_trajectories_source_data.csv"))
