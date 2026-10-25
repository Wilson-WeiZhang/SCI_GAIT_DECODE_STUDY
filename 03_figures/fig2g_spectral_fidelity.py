# -*- coding: utf-8 -*-
"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

panel g — observed-vs-decoded normalized power spectra (healthy cohort)

Renders one panel or table; it performs no
preprocessing and fits no model.
"""


from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_palette import HIP, KNEE, ANKLE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS
import analysis_config as _ec
_ec.require("main", "main")
NPZ = _ec.path("h_spectra")
OUT_PNG = _ec.scoped(ROOT / "figures_out/fig2/fig2g_spectral_fidelity.png")
SD_DIR = ROOT / "04_source_data/fig2"
SOURCE_CSV = _ec.sd("fig2g_spectral_fidelity_source_data.csv")
SESSIONS_CSV = _ec.sd("fig2g_spectral_fidelity_sessions.csv")
META_CSV = _ec.sd("fig2g_spectral_fidelity_metadata.csv")
BUILD_ID = _ec.TAG
COHORT = "healthy cohort (80 sessions / 42 participants)"

OBS = "#5f5e5a"
JC = {0: HIP, 1: KNEE, 2: ANKLE}
NM = {0: "LHip", 1: "LKnee", 2: "LAnk"}

plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})

_LW_DEC, _LW_REF, _LW_HARM, _PT_JOINT = ((2.0, 1.7, 1.0, 12))

z = np.load(NPZ, allow_pickle=True)
S = [str(s) for s in z["sessions"]]
keep = [i for i in range(len(S)) if z["n_trials"][i] >= 5 and np.isfinite(z["f0"][i])]
freq, f0 = z["freq"], z["f0"]
relgrid = np.linspace(0.2, 4.0, 200)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
SD_DIR.mkdir(parents=True, exist_ok=True)

curves = {}
included = {}

SPLIT_PNGS = []
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.92, wspace=0.28)
for j, ax in enumerate(axes):
    Po, Pd, used = [], [], []
    for i in keep:
        if not (np.isfinite(z["psd_obs"][i, j]).all()
                and np.isfinite(z["psd_dec"][i, j]).all()):
            continue
        rel = freq / f0[i]
        o = np.interp(relgrid, rel, z["psd_obs"][i, j])
        d = np.interp(relgrid, rel, z["psd_dec"][i, j])
        Po.append(o / o.max())
        Pd.append(d / d.max())
        used.append(i)
    Po, Pd = np.array(Po), np.array(Pd)
    Lo, Ld = np.log10(Po + 1e-12), np.log10(Pd + 1e-12)
    mo, md = 10 ** Lo.mean(0), 10 ** Ld.mean(0)
    co = 1.96 * Lo.std(0) / np.sqrt(len(Po))
    cd = 1.96 * Ld.std(0) / np.sqrt(len(Pd))
    curves[j] = (mo, co, md, cd, len(Po))
    included[j] = set(used)
    for k in (1, 2, 3):
        ax.axvline(k, color="#c9ccd1", lw=_LW_HARM, ls=":", zorder=0)
    ax.fill_between(relgrid, mo * 10 ** -co, mo * 10 ** co, color=OBS, alpha=0.18, lw=0)
    ax.plot(relgrid, mo, color=OBS, lw=_LW_REF, ls="--")
    ax.fill_between(relgrid, md * 10 ** -cd, md * 10 ** cd, color=JC[j], alpha=0.20, lw=0)
    ax.plot(relgrid, md, color=JC[j], lw=_LW_DEC)
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 2)
    ax.set_xlim(0.2, 4.0)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["$\\mathrm{k}_1$", "$\\mathrm{k}_2$", "$\\mathrm{k}_3$"])
    ax.set_xlabel("Frequency (harmonic)")
    if j == 0:
        ax.set_ylabel("Normalized power")
    ax.text(*((0.97, 0.93)), PS.jlab(NM[j]), transform=ax.transAxes, ha="right",
            va="top", fontsize=_PT_JOINT, color=JC[j])
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_linewidth(1.95)
    ax.tick_params(width=1.95, length=4.5)
    print(f"{NM[j]}: n_sessions={len(Po)}")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

for j in range(3):
    f1, a1 = plt.subplots(figsize=(2.60, 3.0))
    f1.subplots_adjust(left=0.225 if j == 0 else 0.14, right=0.985, bottom=0.19, top=0.92)
    mo, co, md, cd, _n = curves[j]
    for k in (1, 2, 3):
        a1.axvline(k, color="#c9ccd1", lw=1.0, ls=":", zorder=0)
    a1.fill_between(relgrid, mo * 10 ** -co, mo * 10 ** co, color=OBS, alpha=0.18, lw=0)
    a1.plot(relgrid, mo, color=OBS, lw=1.7, ls="--")
    a1.fill_between(relgrid, md * 10 ** -cd, md * 10 ** cd, color=JC[j], alpha=0.20, lw=0)
    a1.plot(relgrid, md, color=JC[j], lw=2.0)
    a1.set_yscale("log"); a1.set_ylim(3e-4, 2); a1.set_xlim(0.2, 4.0)
    a1.set_xticks([1, 2, 3])
    a1.set_xticklabels(["$\\mathrm{k}_1$", "$\\mathrm{k}_2$", "$\\mathrm{k}_3$"])
    a1.set_xlabel("Frequency (harmonic)")
    if j == 0:
        a1.set_ylabel("Normalized power")
    a1.text(0.97, 0.93, PS.jlab(NM[j]), transform=a1.transAxes, ha="right", va="top",
            fontsize=12, color=JC[j])
    for s_ in ("top", "right"):
        a1.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        a1.spines[s_].set_linewidth(1.95)
    a1.tick_params(width=1.95, length=4.5)
    p1 = OUT_PNG.with_name(f"{OUT_PNG.stem}_{NM[j]}.png")
    f1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(f1)
    SPLIT_PNGS.append(p1)
print("wrote split panels: " + ", ".join(p.name for p in SPLIT_PNGS))
print("wrote", OUT_PNG)

with SOURCE_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["joint", "rel_freq", "ref_mean", "ref_ci_lo", "ref_ci_hi",
                "dec_mean", "dec_ci_lo", "dec_ci_hi", "n_sessions"])
    for j in range(3):
        mo, co, md, cd, n = curves[j]
        for gi, rf in enumerate(relgrid):
            w.writerow([NM[j], f"{rf:.6f}",
                        f"{mo[gi]:.8e}", f"{mo[gi] * 10 ** -co[gi]:.8e}", f"{mo[gi] * 10 ** co[gi]:.8e}",
                        f"{md[gi]:.8e}", f"{md[gi] * 10 ** -cd[gi]:.8e}", f"{md[gi] * 10 ** cd[gi]:.8e}",
                        n])

with SESSIONS_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["session", "joint", "f0_hz", "n_trials", "included"])
    for i in range(len(S)):
        for j in range(3):
            w.writerow([S[i], NM[j], f"{float(f0[i]):.6f}", int(z["n_trials"][i]),
                        int(i in included[j])])

with META_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["field", "value"])
    for k, v in [
        ("panel", "Fig. 2g"),
        ("cohort", COHORT),
        ("input_npz", NPZ.relative_to(ROOT).as_posix()),
        ("inclusion_rule", "n_trials >= 5 AND finite f0 AND finite reference+decoded PSD for that joint"),
        ("frequency_axis", "freq / f0 (session cadence), interpolated onto 200 points over 0.2-4.0"),
        ("statistic", "geometric mean across sessions; band = +-1.96 log10-SEM, back-transformed"),
        ("normalization", "each session's PSD divided by its own maximum on the relative grid"),
        ("n_sessions_LHip", curves[0][4]),
        ("n_sessions_LKnee", curves[1][4]),
        ("n_sessions_LAnk", curves[2][4]),
        ("source_data", SOURCE_CSV.relative_to(ROOT).as_posix()),
        ("session_table", SESSIONS_CSV.relative_to(ROOT).as_posix()),
    ]:
        w.writerow([k, v])

print("wrote", SOURCE_CSV)
print("wrote", SESSIONS_CSV)
print("wrote", META_CSV)
