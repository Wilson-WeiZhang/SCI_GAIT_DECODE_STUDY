"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

panel f - reconstruction fidelity: per-joint phase fidelity (3 separate panels)

Writes:
  figures_out/fig2/f2_panelA_cohort_LHip.png
  figures_out/fig2/f2_panelA_cohort_LKnee.png
  figures_out/fig2/f2_panelA_cohort_LAnk.png
  04_source_data/fig2/fig2_recon_fidelity_source_data.csv

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS

import analysis_config as _ec
GAIT_BUILD_ID = "main"
_ERA_TAG = "main"

NPZ = _ec.path("h_abd_cohort")
FIG_OUT = ROOT / "figures_out/fig2"
SD_OUT = ROOT / "04_source_data/fig2"
NB = 50
OBS = "#5f5e5a"
JC = {0: "#F34142", 1: "#1E6FD9", 2: "#3AAF6A"}
NM = {0: "LHip", 1: "LKnee", 2: "LAnk"}
plt.rcParams.update({"font.family": "Arial", "font.size": 16, "axes.labelsize": 16,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})

_LW_REF, _LW_DEC = (1.8, 2.0)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    obs_curves = z["obs_curves"]
    dec_curves = z["dec_curves"]
    percycle_r = z["percycle_r"]
    return obs_curves, dec_curves, percycle_r


def cohort_curve(arr, j):
    """Session-level cohort mean, 95% CI, and valid-session count for joint j."""
    a = arr[:, j, :]
    A = a[np.isfinite(a).all(1)]
    return A.mean(0), 1.96 * A.std(0) / np.sqrt(len(A)), len(A)


obs_curves, dec_curves, percycle_r = load_npz(NPZ)
x = np.linspace(0, 100, NB)

A_rows = []
C_rows, C_n = [], []
for j in range(3):
    mo, cio, no = cohort_curve(obs_curves, j)
    mp, cip, nd = cohort_curve(dec_curves, j)
    pr = percycle_r[:, j]
    pr = pr[np.isfinite(pr)]
    rmed, rlo, rhi = np.median(pr), np.percentile(pr, 25), np.percentile(pr, 75)
    fig, ax = plt.subplots(figsize=(2.9, 3.75))
    fig.subplots_adjust(top=0.84, bottom=0.23, left=0.17, right=0.96)
    ax.fill_between(x, mo - cio, mo + cio, color=OBS, alpha=0.20, lw=0)
    ax.plot(x, mo, color=OBS, lw=_LW_REF, ls="--", label="reference")
    ax.fill_between(x, mp - cip, mp + cip, color=JC[j], alpha=0.20, lw=0)
    ax.plot(x, mp, color=JC[j], lw=_LW_DEC, label="decoded")
    for _ser, _m, _ci, _n in (("reference", mo, cio, no), ("decoded", mp, cip, nd)):
        C_n.append((NM[j], _ser, _n))
        C_rows += [[NM[j], _ser, float(_x), float(_mv), float(_mv - _cv), float(_mv + _cv)]
                   for _x, _mv, _cv in zip(x, _m, _ci)]
    ax.set_ylabel(PS.jlab(NM[j]))
    ax.set_xlabel("Gait cycle (%)")
    _lo, _hi = ax.get_ylim()
    ax.set_ylim(_lo - 0.5, _hi + 0.5)
    print(f"  {NM[j]}: ylim {_lo:.2f}..{_hi:.2f} -> {_lo-0.5:.2f}..{_hi+0.5:.2f}")
    ax.tick_params(width=2.8, length=5.0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_linewidth(2.8)
    fig.savefig(_ec.scoped(FIG_OUT / f"f2_panelA_cohort_{NM[j]}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    A_rows.append((NM[j], no, rmed, rlo, rhi))

SD_OUT.mkdir(parents=True, exist_ok=True)
with _ec.sd("fig2_recon_fidelity_source_data.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["joint", "n_sessions", "percycle_r_median", "p25", "p75"])
    for nm, no, rm, rl, rh in A_rows:
        w.writerow([nm, no, f"{rm:.4f}", f"{rl:.4f}", f"{rh:.4f}"])
from safe_write import safe_write_csv
safe_write_csv(_ec.sd("fig2_recon_fidelity_curves_source_data.csv"),
               ["joint", "series", "gait_cycle_pct", "mean", "ci_lower", "ci_upper"], C_rows)

print("Fig.2 reconstruction-fidelity per-cycle r (cohort, session-level median[IQR]):")
for nm, no, rm, rl, rh in A_rows:
    print(f"  {nm}: {rm:.3f} [{rl:.3f}-{rh:.3f}]  n={no}")
print("wrote figures_out/fig2/f2_panelA_cohort_{LHip,LKnee,LAnk}.png "
      "+ 04_source_data/fig2/fig2_recon_fidelity_source_data.csv")


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

_write_meta(_ec.sd("fig2_recon_fidelity_metadata.csv"), [
    ("figure", "Fig. 2 panel f - per-joint phase fidelity: LHip / LKnee / LAnk "
               "reference-vs-decoded cycle-averaged curves (3 separate panels)"),
    ("producer", "03_figures/fig2_recon_fidelity.py"),
    ("cohort", f"healthy; {obs_curves.shape[0]} sessions present in the input dump"),
    ("input", str(NPZ.relative_to(ROOT)).replace("\\", "/")),
    ("n_sessions_in_dump", obs_curves.shape[0]),
    ("n_cycle_bins", NB),
    ("aggregation", "session-level cohort mean with 95% CI = 1.96*SD/sqrt(n_valid); "
                    "joints are dropped per joint, so each panel carries its own n"),
    ("percycle_r_statistic", "median [p25, p75] over sessions"),
    ("outputs", "; ".join(_p.relative_to(ROOT).as_posix() for _p in (
        _ec.scoped(FIG_OUT / "f2_panelA_cohort_LHip.png"),
        _ec.scoped(FIG_OUT / "f2_panelA_cohort_LKnee.png"),
        _ec.scoped(FIG_OUT / "f2_panelA_cohort_LAnk.png"),
        _ec.sd("fig2_recon_fidelity_source_data.csv")))),
] + [(f"n_sessions_{_nm}", _no) for _nm, _no, _a, _b, _c in A_rows])

_write_meta(_ec.sd("fig2_recon_fidelity_curves_metadata.csv"), [
    ("figure", "Fig. 2 panel f - cohort cycle-averaged reference (dashed) vs decoded z-scored "
               "LHip / LKnee / LAnk angles across the gait cycle, 95% CI shading across sessions"),
    ("producer", "03_figures/fig2_recon_fidelity.py"),
    ("input", str(NPZ.relative_to(ROOT)).replace("\\", "/")),
    ("n_cycle_bins", NB),
    ("col_joint", "left-limb joint (LHip, LKnee, LAnk)"),
    ("col_series", "reference (goniometer, dashed line) or decoded (colour line)"),
    ("col_gait_cycle_pct", "gait-cycle position (%), x of the plotted curve"),
    ("col_mean", "session-level cohort mean of the z-scored cycle-averaged angle"),
    ("col_ci_lower", "mean - 1.96*SD/sqrt(n_sessions), lower edge of the shaded band"),
    ("col_ci_upper", "mean + 1.96*SD/sqrt(n_sessions), upper edge of the shaded band"),
] + [(f"n_sessions_{_nm}_{_ser}", _n) for _nm, _ser, _n in C_n])
