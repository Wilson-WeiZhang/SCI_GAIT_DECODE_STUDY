# -*- coding: utf-8 -*-
"""Figure 5 — Decoding imagined walking after spinal cord injury

panel e - per-joint cycle-average observed vs decoded trajectories (recon fidelity)

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import panel_style as PS
from project_goni import polarity_vector
import project_palette as PAL
import analysis_config as ec

RES = ROOT / "data/decoder_output"
OUT = ROOT / "figures_out/fig5"
SD = ROOT / "04_source_data/fig5"
OUT.mkdir(parents=True, exist_ok=True)
SD.mkdir(parents=True, exist_ok=True)

BUILD_ID = ec.TAG
COHORT = ec.COHORTS["post_operative"]
DECODE_TAG = ec.PAT_TAG

POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07",
            "pat01_sess08", "pat01_sess09",
            "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
            "pat02_sess09"]

NB = 50
OBS = PAL.TARGET_GONI
JC = {0: PAL.HIP, 1: PAL.KNEE, 2: PAL.ANKLE}
NM = {0: "LHip", 1: "LKnee", 2: "LAnk"}
KEY = {0: "LHip", 1: "LKne", 2: "LAnk"}
LAB6 = ["LHip", "LKne", "LAnk", "RHip", "RKne", "RAnk"]

FIGSIZE = (7.6, 2.57)
MARGINS = dict(left=0.085, right=0.915, bottom=0.19, top=0.88, wspace=0.28)
LW_AX = 1.95
plt.rcParams.update({"font.family": "Arial", "font.size": 12.5, "axes.labelsize": 12.5,
                     "xtick.labelsize": 11.0, "ytick.labelsize": 11.0,
                     "axes.linewidth": LW_AX,
                     "xtick.major.width": LW_AX, "ytick.major.width": LW_AX,
                     "xtick.major.size": 4.0, "ytick.major.size": 4.0,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic"})

ROWS = []

_LW_E, _LW_F, _LW_G = (
    (2.0, 1.8), (2.0, 1.7, 1.0), (2.4, 1.6, 2.0, 16, 3.2))


def rec(panel, quantity, key, x, value, n):
    ROWS.append([panel, quantity, key,
                 "" if x is None else f"{float(x):.6g}",
                 "" if value is None or not np.isfinite(value) else f"{float(value):.6g}",
                 int(n)])


def tint(c, f=0.20):
    """The opaque colour that an alpha-`f` fill of `c` produces over white."""
    r, g, b = mcolors.to_rgb(c)
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def strip():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.subplots_adjust(**MARGINS)
    return fig, axes


SOLO_FIGSIZE = (2.432, 2.570)
SOLO_MARGINS = dict(left=0.2467, right=0.9753, bottom=0.19, top=0.88)


def solo():
    fig, ax = plt.subplots(figsize=SOLO_FIGSIZE)
    fig.subplots_adjust(**SOLO_MARGINS)
    return fig, ax


def save_solo(fig, path, dpi=300):
    """White-background PNG + a transparent twin at the same geometry."""
    path = ec.scoped(path)
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_name(f"{path.stem}_transparent.png"), dpi=dpi, transparent=True)
    plt.close(fig)
    print("wrote", path.name, "+", f"{path.stem}_transparent.png")


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


z = np.load(ec.path("p_abd_cohort"), allow_pickle=True)
mask = np.array([str(s) in POST_OPERATIVE for s in z["sessions"]])
assert mask.sum() == len(POST_OPERATIVE), f"post_operative sessions missing from abd npz: {mask.sum()}"
obs_curves, dec_curves = z["obs_curves"][mask], z["dec_curves"][mask]
percycle_r = z["percycle_r"][mask]

x = np.linspace(0, 100, NB)
fig, axes = strip()
for j, ax in enumerate(axes):
    v = (np.isfinite(obs_curves[:, j, :]).all(1)
         & np.isfinite(dec_curves[:, j, :]).all(1))
    A_o, A_d = obs_curves[v, j, :], dec_curves[v, j, :]
    n = int(v.sum())
    mo, cio = A_o.mean(0), 1.96 * A_o.std(0) / np.sqrt(n)
    mp, cip = A_d.mean(0), 1.96 * A_d.std(0) / np.sqrt(n)
    sfig, sax = solo()
    for _a in (ax, sax):
        _a.fill_between(x, mo - cio, mo + cio, color=OBS, alpha=0.20, lw=0, zorder=1)
        _a.fill_between(x, mp - cip, mp + cip, color=tint(JC[j]), lw=0, zorder=1.5)
        _a.plot(x, mo, color=OBS, lw=_LW_E[1], ls="--", zorder=2.5)
        _a.plot(x, mp, color=JC[j], lw=_LW_E[0], zorder=3)
        lo, hi = _a.get_ylim()
        _a.set_ylim(lo - 0.5, hi + 0.5)
        _a.set_xlabel("Gait cycle (%)")
        _a.set_ylabel(PS.jlab(NM[j]))
        despine(_a)
    save_solo(sfig, OUT / f"fig5e_recon_fidelity_{NM[j]}.png")
    for i in range(NB):
        rec("e", "ref_mean", KEY[j], x[i], mo[i], n)
        rec("e", "ref_ci95", KEY[j], x[i], cio[i], n)
        rec("e", "dec_mean", KEY[j], x[i], mp[i], n)
        rec("e", "dec_ci95", KEY[j], x[i], cip[i], n)
    pr = percycle_r[v, j]
    pr = pr[np.isfinite(pr)]
    med = float(np.median(pr))
    rec("e", "percycle_r_median", KEY[j], None, med, n)
    p25, p75 = float(np.percentile(pr, 25)), float(np.percentile(pr, 75))
    rec("e", "percycle_r_p25", KEY[j], None, p25, n)
    rec("e", "percycle_r_p75", KEY[j], None, p75, n)
    print(f"e {KEY[j]}: n={n}, per-cycle r median {med:.4f} [{p25:.4f}, {p75:.4f}]")
fig.savefig(ec.scoped(OUT / "fig5e_recon_fidelity.png"), dpi=300)
plt.close(fig)

zs = np.load(ec.path("p_spectra"), allow_pickle=True)
assert set(map(str, zs["sessions"])) == set(POST_OPERATIVE), "spectra npz is not post_operative"
keepi = [i for i in range(len(zs["sessions"]))
         if zs["n_trials"][i] >= 5 and np.isfinite(zs["f0"][i])]
freq, f0 = zs["freq"], zs["f0"]
relgrid = np.linspace(0.2, 4.0, 200)
fig, axes = strip()
for j, ax in enumerate(axes):
    Po, Pd = [], []
    for i in keepi:
        if not (np.isfinite(zs["psd_obs"][i, j]).all()
                and np.isfinite(zs["psd_dec"][i, j]).all()):
            continue
        rel = freq / f0[i]
        o = np.interp(relgrid, rel, zs["psd_obs"][i, j])
        d = np.interp(relgrid, rel, zs["psd_dec"][i, j])
        Po.append(o / o.max())
        Pd.append(d / d.max())
    Lo, Ld = np.log10(np.array(Po) + 1e-12), np.log10(np.array(Pd) + 1e-12)
    mo, md = 10 ** Lo.mean(0), 10 ** Ld.mean(0)
    co = 1.96 * Lo.std(0) / np.sqrt(len(Lo))
    cd = 1.96 * Ld.std(0) / np.sqrt(len(Ld))
    n = len(Lo)
    sfig, sax = solo()
    for _a in (ax, sax):
        for k in (1, 2, 3):
            _a.axvline(k, color="#c9ccd1", lw=_LW_F[2], ls=":", zorder=0)
        _a.fill_between(relgrid, mo * 10 ** -co, mo * 10 ** co, color=OBS, alpha=0.18, lw=0,
                        zorder=1)
        _a.plot(relgrid, mo, color=OBS, lw=_LW_F[1], ls="--", zorder=2.5)
        _a.fill_between(relgrid, md * 10 ** -cd, md * 10 ** cd, color=tint(JC[j]), lw=0,
                        zorder=1.5)
        _a.plot(relgrid, md, color=JC[j], lw=_LW_F[0], zorder=3)
        _a.set_yscale("log")
        _a.set_ylim(3e-4, 2)
        _a.set_xlim(0.2, 4.0)
        _a.set_xticks([1, 2, 3])
        _a.set_xticklabels(["$\\mathrm{k}_1$", "$\\mathrm{k}_2$", "$\\mathrm{k}_3$"])
        _a.set_xlabel("Frequency (harmonic)")
        if j == 0 or _a is sax:
            _a.set_ylabel("Normalized power")
        _a.text(*((0.95, 0.92)), PS.jlab(NM[j]), transform=_a.transAxes, ha="right",
                va="top", fontsize=12.0, color=JC[j])
        despine(_a)
    save_solo(sfig, OUT / f"fig5f_spectral_fidelity_{NM[j]}.png")
    for i in range(len(relgrid)):
        rec("f", "psd_ref_mean", KEY[j], relgrid[i], mo[i], n)
        rec("f", "psd_ref_ci95_log10", KEY[j], relgrid[i], co[i], n)
        rec("f", "psd_dec_mean", KEY[j], relgrid[i], md[i], n)
        rec("f", "psd_dec_ci95_log10", KEY[j], relgrid[i], cd[i], n)
    print(f"f {KEY[j]}: n={n}")
fig.savefig(ec.scoped(OUT / "fig5f_spectral_fidelity.png"), dpi=300)
plt.close(fig)

def cl(v):
    return np.append(v, v[0])


def _tube(mx, my, cx, cy, smooth=3):
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


def ci_fill(ax, mx, my, cx, cy, alpha, color=None, cmap=None, phase_closed=None):
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
fig, axes = strip()
PAIR_RANGE = []
SOLOS = []
for ax, (a, b) in zip(axes, pairs):
    v = (np.isfinite(obs_curves[:, a, :]).all(1) & np.isfinite(obs_curves[:, b, :]).all(1)
         & np.isfinite(dec_curves[:, a, :]).all(1) & np.isfinite(dec_curves[:, b, :]).all(1))
    nuse = int(v.sum())
    pkey = f"{KEY[a]}-{KEY[b]}"
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
    sfig, sax = solo()
    SOLOS.append((sfig, sax, a, b))
    for _a in (ax, sax):
        ci_fill(_a, mxo, myo, cxo, cyo, 0.20, color="#6f747a")
        ci_fill(_a, mxd, myd, cxd, cyd, 0.32, cmap=PAL.PHASE_CMAP, phase_closed=ph)
        _a.plot(cl(mxo), cl(myo), color=OBS, lw=_LW_G[1], ls="--", zorder=2)
        xx, yy = cl(mxd), cl(myd)
        pts = np.array([xx, yy]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        _a.add_collection(LineCollection(segs, cmap=PAL.PHASE_CMAP, array=ph[:-1], lw=_LW_G[0],
                                         zorder=3))
        ar = _a.annotate("", xy=(mxd[ARR_I1], myd[ARR_I1]), xytext=(mxd[ARR_I0], myd[ARR_I0]),
                         arrowprops=dict(arrowstyle="-|>", mutation_scale=_LW_G[3], color="#111",
                                         lw=_LW_G[2]), zorder=6)
        ar.arrow_patch.set_path_effects([pe.withStroke(linewidth=_LW_G[4], foreground="white")])
        _a.set_xlabel(PS.jlab(NM[a]))
        _a.set_ylabel(PS.jlab(NM[b]))
        despine(_a)
    allx = np.concatenate([cl(mxo), cl(mxd)]); ally = np.concatenate([cl(myo), cl(myd)])
    _lo = min(allx.min(), ally.min()); _hi = max(allx.max(), ally.max())
    _pad = 0.15 * (_hi - _lo)
    PAIR_RANGE.append((_lo - _pad, _hi + _pad))
    for i in range(NB):
        rec("g", "ref_x", pkey, phase[i], mxo[i], nuse)
        rec("g", "ref_y", pkey, phase[i], myo[i], nuse)
        rec("g", "dec_x", pkey, phase[i], mxd[i], nuse)
        rec("g", "dec_y", pkey, phase[i], myd[i], nuse)
        rec("g", "ref_ci95_x", pkey, phase[i], cxo[i], nuse)
        rec("g", "ref_ci95_y", pkey, phase[i], cyo[i], nuse)
        rec("g", "dec_ci95_x", pkey, phase[i], cxd[i], nuse)
        rec("g", "dec_ci95_y", pkey, phase[i], cyd[i], nuse)
    print(f"g {pkey}: n={nuse}")
_LO = min(r[0] for r in PAIR_RANGE); _HI = max(r[1] for r in PAIR_RANGE)
for ax in axes:
    ax.set_xlim(_LO, _HI); ax.set_ylim(_LO, _HI)
    ax.set_aspect("equal", adjustable="box")
fig.savefig(ec.scoped(OUT / "fig5g_cyclogram.png"), dpi=300)
plt.close(fig)
for _sf, _sa, _a, _b in SOLOS:
    _sa.set_xlim(_LO, _HI); _sa.set_ylim(_LO, _HI)
    _sa.set_aspect("equal", adjustable="box")
    save_solo(_sf, OUT / f"fig5g_cyclogram_{KEY[_a]}_{KEY[_b]}.png")

S6 = np.outer(polarity_vector(LAB6), polarity_vector(LAB6))
zx = np.load(ec.path("p_cross_cohort_iw"), allow_pickle=True)
mx = np.array([str(s) in POST_OPERATIVE for s in zx["sessions"]])
assert mx.sum() == len(POST_OPERATIVE), "post_operative sessions missing from cross_cohort IW npz"
compl = np.isfinite(zx["cross_mat"][mx]).all(axis=(1, 2))
Cobs = np.nanmean(zx["obs_mat"][mx][compl], axis=0) * S6
Ciw = np.nanmean(zx["cross_mat"][mx][compl], axis=0) * S6
diag_iw = np.array([np.mean(np.diag(m)) for m in zx["cross_mat"][mx][compl]])

zr = np.load(ec.path("p_cross_cohort_rest"), allow_pickle=True)
mr = np.array([str(s) in POST_OPERATIVE for s in zr["sessions"]])
complr = np.isfinite(zr["cross_mat"][mr]).all(axis=(1, 2))
Cre = np.nanmean(zr["cross_mat"][mr][complr], axis=0) * S6
diag_re = np.array([np.mean(np.diag(m)) for m in zr["cross_mat"][mr][complr]])

n_iw, n_re = int(compl.sum()), int(complr.sum())
med_iw, med_re = float(np.median(diag_iw)), float(np.median(diag_re))
print(f"h: IW n={n_iw} diag-median {med_iw:.4f} | REST(test_only) n={n_re} "
      f"diag-median {med_re:.4f}")

H_BOT_EXTRA = 0.18
H_I = 2.87 + H_BOT_EXTRA
fig, axes = plt.subplots(1, 3, figsize=(7.6, H_I))
fig.subplots_adjust(left=MARGINS["left"], right=MARGINS["right"],
                    bottom=(0.19 * 2.57 + H_BOT_EXTRA) / H_I,
                    top=(0.19 * 2.57 + H_BOT_EXTRA + (0.88 - 0.19) * 2.57) / H_I,
                    wspace=MARGINS["wspace"])
_FS_H, _PAD_H, _LW_H = 11.0, 6, LW_AX
for k, (M, t, q, nn) in enumerate([(Cobs, "Reference", "reference", n_iw),
                                   (Ciw, "Decoded from\nimagined walking", "decoded_IW", n_iw),
                                   (Cre, "Decoded from\nresting EEG", "decoded_REST", n_re)]):
    ax = axes[k]
    ax.imshow(M, vmin=-1, vmax=1, cmap=PAL.DIFF_CMAP)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    _D6 = [PS.jlab(x) for x in LAB6]
    ax.set_xticklabels(_D6, rotation=45, ha="right", fontsize=_FS_H)
    ax.set_yticklabels(_D6 if k == 0 else [], fontsize=_FS_H)
    ax.set_title(t, fontsize=_FS_H, pad=_PAD_H)
    for i in range(6):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="#1a1a1a", lw=_LW_H))
    for i in range(6):
        for jj in range(6):
            rec("h", q, f"{LAB6[i]}|{LAB6[jj]}", None, M[i, jj], nn)
rec("h", "diag_median_IW", "all_joints", None, med_iw, n_iw)
rec("h", "diag_median_REST", "all_joints", None, med_re, n_re)
from scipy import stats as _st
_offabs = lambda m: float(np.mean(np.abs(m[~np.eye(6, dtype=bool)])))
off_iw = np.array([_offabs(m) for m in zx["cross_mat"][mx][compl]])
off_re = np.array([_offabs(m) for m in zr["cross_mat"][mr][complr]])
s_iw = [str(s) for s in zx["sessions"][mx][compl]]
s_re = [str(s) for s in zr["sessions"][mr][complr]]
rec("h", "diag_IW_q25", "all_joints", None, np.percentile(diag_iw, 25), n_iw)
rec("h", "diag_IW_q75", "all_joints", None, np.percentile(diag_iw, 75), n_iw)
rec("h", "offdiag_absmean_median_IW", "all_joints", None, float(np.median(off_iw)), n_iw)
_w_iw = _st.wilcoxon(diag_iw, off_iw)
rec("h", "wilcoxon_W_IW_diag_vs_offdiag", "all_joints", None, _w_iw.statistic, n_iw)
rec("h", "wilcoxon_p_IW_diag_vs_offdiag", "all_joints", None, _w_iw.pvalue, n_iw)
rec("h", "diag_REST_q25", "all_joints", None, np.percentile(diag_re, 25), n_re)
rec("h", "diag_REST_q75", "all_joints", None, np.percentile(diag_re, 75), n_re)
_both = [s for s in s_re if s in s_iw]
_w_pair = _st.wilcoxon(np.array([diag_iw[s_iw.index(s)] for s in _both]),
                       np.array([diag_re[s_re.index(s)] for s in _both]))
rec("h", "wilcoxon_W_diag_IW_vs_REST_paired", "all_joints", None, _w_pair.statistic, len(_both))
rec("h", "wilcoxon_p_diag_IW_vs_REST_paired", "all_joints", None, _w_pair.pvalue, len(_both))
_w_re = _st.wilcoxon(diag_re, off_re)
fig.savefig(ec.scoped(OUT / "fig5h_correspondence.png"), dpi=300)
fig.savefig(ec.scoped(OUT / "fig5h_correspondence_transparent.png"), dpi=300, transparent=True)
plt.close(fig)

sd = ec.sd("fig5_efgh_patient_reconstruction_source_data.csv")
with sd.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["panel", "quantity", "key", "x", "value", "n_sessions_contributing"])
    w.writerows(ROWS)

md = ec.sd("fig5_efgh_patient_reconstruction_metadata.csv")
with md.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["field", "value"])
    for k, v in [
        ("figure", "Fig. 5 e-h"),
        ("panels", "e recon fidelity; f spectral fidelity; g cyclograms; h correspondence"),
        ("cohort", COHORT),
        ("decode_tag", DECODE_TAG),
        ("n_sessions_cohort", len(POST_OPERATIVE)),
        ("n_participants", 2),
        ("sessions", " ".join(POST_OPERATIVE)),
        ("scope_label", "case-level (2 participants, post-operative); descriptive, no population claim"),
        ("renderer", "03_figures/fig5_efgh_patient_reconstruction.py"),
        ("input_e_g", ec.rel("p_abd_cohort")),
        ("input_f", ec.rel("p_spectra")),
        ("input_h_iw", ec.rel("p_cross_cohort_iw")),
        ("input_h_rest", ec.rel("p_cross_cohort_rest")),
        ("n_e_LHip", int([r[5] for r in ROWS if r[0] == "e" and r[1] == "percycle_r_median" and r[2] == "LHip"][0])),
        ("n_e_LKne", int([r[5] for r in ROWS if r[0] == "e" and r[1] == "percycle_r_median" and r[2] == "LKne"][0])),
        ("n_e_LAnk", int([r[5] for r in ROWS if r[0] == "e" and r[1] == "percycle_r_median" and r[2] == "LAnk"][0])),
        ("n_h_IW", n_iw),
        ("n_h_REST", n_re),
        ("display_polarity", "applied at render time from 03_figures/project_goni.py "
                             "(RHipW X, LKneW X, LAnkW X x -1); panel h only, diagonal unaffected"),
        ("colors", "03_figures/project_palette.py: TARGET_GONI reference, HIP/KNEE/ANKLE decoded, "
                   "PHASE_CMAP twilight gait-cycle phase, DIFF_CMAP RdBu_r correspondence"),
        ("units", "e/g normalized (per-cycle z-scored before averaging); f normalized power vs "
                  "cadence-relative frequency; h Pearson r"),
    ]:
        w.writerow([k, v])

print("wrote", sd.name, len(ROWS), "rows;", md.name)
