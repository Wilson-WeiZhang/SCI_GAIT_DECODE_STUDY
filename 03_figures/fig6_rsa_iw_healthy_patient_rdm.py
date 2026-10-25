"""Figure 6 — Representational geometry of patient imagined walking

RSA of the imagined-walking (IW) representation, healthy vs each SCI patient

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_palette import RDM_CMAP_STOPS
FS = 1.30
plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "axes.linewidth": 1.9,
                     "xtick.major.width": 1.9, "ytick.major.width": 1.9,
                     "xtick.major.size": 3.0, "ytick.major.size": 3.0,
                     "font.size": 12.9 * FS, "xtick.labelsize": 11.3 * FS, "ytick.labelsize": 11.3 * FS,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic",
                     "pdf.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/decoder_output"
PATF = R / "diag_patient_harmonic_phase/persession_harmonic_phase_cycle9_complex.csv"
HIWF = R / "diag_harmonic_phase_responder_split/persession_harmonic_phase_cycle9_complex.csv"
HREF = R / "diag_rest_yoked_harmonic_phase/persession_harmonic_phase_cycle9_complex.csv"
OUT_FIG = ROOT / "figures_out/fig6/fig6_rsa_rdm_grid.png"
OUT_SD = ROOT / "04_source_data/fig6/fig6_rsa_iw_healthy_patient_rdm_source_data.csv"
OUT_MAT = ROOT / "04_source_data/fig6/fig6_rsa_iw_healthy_patient_rdm_matrices.csv"
OUT_META = ROOT / "04_source_data/fig6/fig6_rsa_iw_healthy_patient_rdm_metadata.csv"

BUILD_ID = "main"
PAT_TAG = "diag_patient_harmonic_phase"
HIW_TAG = "diag_harmonic_phase_responder_split"
HRE_TAG = "diag_rest_yoked_harmonic_phase"

N_SPIN = 1000
N_SHUF = 5000
RNG = np.random.default_rng(0)

P1_SESSIONS = ["pat01_sess04", "pat01_sess05", "pat01_sess06",
               "pat01_sess07", "pat01_sess08", "pat01_sess09"]
P2_SESSIONS = ["pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
               "pat02_sess09"]
PAT_SESSIONS = {"P1": P1_SESSIONS, "P2": P2_SESSIONS}

_pat_raw = pd.read_csv(PATF)
_hiw_raw = pd.read_csv(HIWF)
_hre_raw = pd.read_csv(HREF)
_missing = [s for s in P1_SESSIONS + P2_SESSIONS if s not in set(_pat_raw.session)]
if _missing:
    raise SystemExit(f"post_operative sessions absent from {PATF.name}: {_missing}")
COMMON = sorted(set(_pat_raw.channel) & set(_hiw_raw.channel) & set(_hre_raw.channel))
E = len(COMMON)
iuE = np.triu_indices(E, 1)

CMAP = LinearSegmentedColormap.from_list("rdm", list(RDM_CMAP_STOPS))
CMAP.set_bad("#dddddd")
_pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
_xyz = np.array([_pos[c] for c in COMMON])
xyz0 = _xyz - _xyz.mean(0)
GORDER = np.argsort(-_xyz[:, 1])


def load(raw, k, sessions=None):
    """Complex harmonic map -> (re, im) arrays [n_sess, n_cycle, E], sessions in the given order."""
    d = raw[raw.harmonic_k == k]
    if sessions is not None:
        d = d[d.session.isin(sessions)]
        sess = list(sessions)
    else:
        sess = sorted(d.session.unique())
    cyc = sorted(d.cycle_fraction.unique())
    re = np.full((len(sess), len(cyc), E), np.nan)
    im = np.full_like(re, np.nan)
    piv = d.set_index(["session", "cycle_fraction", "channel"])
    for si, s in enumerate(sess):
        for ci, c in enumerate(cyc):
            sub = piv.loc[(s, c)]
            re[si, ci] = sub["real"].reindex(COMMON).to_numpy()
            im[si, ci] = sub["imag"].reindex(COMMON).to_numpy()
    return re, im


def crossnobis(P):
    S, C, F = P.shape
    sd = np.nanstd((P - np.nanmean(P, 0)[None]).reshape(-1, F), 0) + 1e-9
    Pw = P / sd
    rdm = np.zeros((C, C))
    for i in range(C):
        for j in range(i + 1, C):
            d = Pw[:, i, :] - Pw[:, j, :]
            rdm[i, j] = rdm[j, i] = float(np.mean([np.dot(d[s], np.delete(d, s, 0).mean(0)) / F
                                                   for s in range(S)]))
    return rdm


def elec_rdm(re, im):
    return crossnobis(np.concatenate([re.transpose(0, 2, 1), im.transpose(0, 2, 1)], 2))


def sp(a, b):
    return stats.spearmanr(a, b)[0]


def rand_rot(rng):
    Q, Rm = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rm)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def spin_null(a_tri, ref_rdm, obs, rng):
    sn = np.empty(N_SPIN)
    for b in range(N_SPIN):
        Dm = np.linalg.norm(xyz0[:, None] - (xyz0 @ rand_rot(rng).T)[None], axis=2)
        _, pm = linear_sum_assignment(Dm)
        sn[b] = sp(a_tri, ref_rdm[np.ix_(pm, pm)][iuE])
    return (np.sum(sn >= obs) + 1) / (N_SPIN + 1), sn


def shuffle_null(a_tri, ref_rdm, obs, rng):
    sn = np.empty(N_SHUF)
    for b in range(N_SHUF):
        pm = rng.permutation(E)
        sn[b] = sp(a_tri, ref_rdm[np.ix_(pm, pm)][iuE])
    return (np.sum(sn >= obs) + 1) / (N_SHUF + 1), sn


CONDS = ["healthy_IW", "P1", "P2", "healthy_REST"]
rdms, rows, mat_rows = {}, [], []
print(f"COMMON electrodes = {E} | spin N={N_SPIN} shuffle N={N_SHUF} | order = anterior->posterior")
print(f"P1 = {P1_SESSIONS}\nP2 = {P2_SESSIONS}")
print(f"{'k':>2} {'pat':>3} | {'rho':>6} {'spin_p':>7} {'shuffle_p':>9} | {'REST rho':>8}")
n_healthy_iw = _hiw_raw.session.nunique()
n_healthy_rest = _hre_raw.session.nunique()
for k in (1, 2, 3):
    rdms[(k, "healthy_IW")] = elec_rdm(*load(_hiw_raw, k))
    rdms[(k, "P1")] = elec_rdm(*load(_pat_raw, k, P1_SESSIONS))
    rdms[(k, "P2")] = elec_rdm(*load(_pat_raw, k, P2_SESSIONS))
    rdms[(k, "healthy_REST")] = elec_rdm(*load(_hre_raw, k))
    ref = rdms[(k, "healthy_IW")][iuE]
    rest = rdms[(k, "healthy_REST")][iuE]
    for pat in ("P1", "P2"):
        a = rdms[(k, pat)][iuE]
        rho = sp(a, ref); rest_rho = sp(a, rest)
        sp_p, sn_spin = spin_null(a, rdms[(k, "healthy_IW")], rho, RNG)
        sf_p, sn_shuf = shuffle_null(a, rdms[(k, "healthy_IW")], rho, RNG)
        rows.append({"harmonic_k": k, "patient": pat,
                     "spatial_rho": round(rho, 4),
                     "spin_p": round(sp_p, 4), "spin_null_mean": round(float(sn_spin.mean()), 4),
                     "spin_null_p95": round(float(np.percentile(sn_spin, 95)), 4),
                     "shuffle_p": round(sf_p, 4), "shuffle_null_mean": round(float(sn_shuf.mean()), 4),
                     "shuffle_null_p95": round(float(np.percentile(sn_shuf, 95)), 4),
                     "rest_rho": round(rest_rho, 4), "n_sessions": len(PAT_SESSIONS[pat])})
        print(f"{k:>2} {pat:>3} | {rho:>6.3f} {sp_p:>7.3f} {sf_p:>9.4f} | {rest_rho:>8.3f}")

chs = [COMMON[i] for i in GORDER]
for k in (1, 2, 3):
    for cond in CONDS:
        M = rdms[(k, cond)][np.ix_(GORDER, GORDER)]
        for i in range(E):
            for j in range(i + 1, E):
                mat_rows.append({"harmonic_k": k, "condition": cond,
                                 "ch_i": chs[i], "ch_j": chs[j],
                                 "dissimilarity": round(float(M[i, j]), 6)})

df = pd.DataFrame(rows)
OUT_SD.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_SD, index=False)
pd.DataFrame(mat_rows).to_csv(OUT_MAT, index=False)

COLS = [("healthy_IW", "Healthy IW"), ("P1", "P1"), ("P2", "P2")]
ROW = [(1, r"$\mathrm{k}_1$"), (2, r"$\mathrm{k}_2$"), (3, r"$\mathrm{k}_3$")]
disp = {(k, c): rdms[(k, c)][np.ix_(GORDER, GORDER)].astype(float) for k, _ in ROW for c, _ in COLS}
VM = float(np.nanquantile(np.abs(np.concatenate([m.ravel() for m in disp.values()])), 0.98))

CELL = 2.0
GAP = 0.06 * CELL
L_IN = 0.50 * FS
T_IN = 0.30 * FS
B_IN = 0.34 * FS
CB_PAD, CB_W = 0.08 * CELL, 0.06 * CELL
CB_LAB_IN = 0.78 * FS
GRID = 3 * CELL + 2 * GAP
FW = L_IN + GRID + CB_PAD + CB_W + CB_LAB_IN
FH = T_IN + GRID + B_IN

fig, axes = plt.subplots(3, 3, figsize=(FW, FH))
fig.subplots_adjust(left=L_IN / FW, right=(L_IN + GRID) / FW,
                    bottom=B_IN / FH, top=(B_IN + GRID) / FH,
                    wspace=0.06, hspace=0.06)
im = None
for ri, (k, kl) in enumerate(ROW):
    for ci, (c, cl) in enumerate(COLS):
        ax = axes[ri, ci]
        im = ax.imshow(disp[(k, c)], cmap=CMAP, vmin=-VM, vmax=VM, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(cl, fontsize=11.3 * FS, pad=4)
        if ci == 0:
            ax.set_ylabel(kl, fontsize=14 * FS, rotation=0, labelpad=14 * FS, va="center")
cax = fig.add_axes([(L_IN + GRID + CB_PAD) / FW, B_IN / FH, CB_W / FW, GRID / FH])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Cross-validated distance", fontsize=11.3 * FS, labelpad=4)
cb.ax.tick_params(labelsize=11.3 * FS, length=3.0, width=1.9)
fig.text((L_IN + GRID / 2) / FW, 0.10 * FS / FH, "EEG channels (anterior → posterior)",
         ha="center", va="bottom", fontsize=11.3 * FS)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)

pd.DataFrame([
    ("panel", "Fig. 6e - patient<->healthy IW representational similarity: cross-validated electrode RDM grid "
              "(rows k1/k2/k3 x cols Healthy IW / P1 / P2) + 2nd-order spatial RSA with spin/shuffle nulls"),
    ("patient_dump", PAT_TAG),
    ("healthy_iw_dump", HIW_TAG),
    ("healthy_rest_dump", HRE_TAG),
    ("readout", "patient-trained 10-fold CV (representation transfer, not zero-shot)"),
    ("scope", "n=2 SCI patients, post-operative, case-level, descriptive; NOT a population claim"),
    ("post_operative_set", "P1 = pat01 sess04-09 (6); P2 = pat02 sess03/05/06/08/09 (5)"),
    ("pre_operative_set", "pat01 sess01/02 + pat02 sess01/02 - present in the dump, NOT used here"),
    ("cohort_selection", "EXPLICIT session lists, not an id-prefix match"),
    ("electrodes", E),
    ("cycle_points", 9),
    ("healthy_IW_sessions", n_healthy_iw),
    ("healthy_REST_sessions", n_healthy_rest),
    ("P1_sessions", len(P1_SESSIONS)),
    ("P2_sessions", len(P2_SESSIONS)),
    ("metric", "electrode-electrode crossnobis RDM (gait-phase re+im features), 2nd-order Spearman RSA "
               "of each patient RDM vs the healthy-IW RDM (upper triangle)"),
    ("spin_N", N_SPIN),
    ("shuffle_N", N_SHUF),
    ("null_note", "SPIN = rotational electrode permutation, preserves spatial autocorrelation, CONSERVATIVE "
                  "and the reportable null; SHUFFLE = plain electrode-label permutation, anti-conservative "
                  "for spatial maps, reported for completeness only"),
    ("rest_control", "healthy yoked-REST RDM; a low or negative rest_rho is the IW-specificity check"),
    ("electrode_order", "anterior->posterior by standard_1005 y coordinate, one order for all matrices; "
                        "ORDERING ONLY - no topomap is drawn, so no electrode position is plotted"),
    ("colour_scale", f"shared symmetric +/-{VM:.6f}"),
    ("claim", "channel-space RSA, not source localization: the step-harmonic k2 IW representation "
              "reproduces the healthy spatial structure in each patient above the conservative spin null; "
              "k1 and k3 do not. Case-level."),
    ("consumed_by", "03_figures/fig6_rsa_quantification_forest.py"),
    ("output_png", "figures_out/fig6/fig6_rsa_rdm_grid.png"),
], columns=["key", "value"]).to_csv(OUT_META, index=False)

print(f"shared vm +/- {VM:.4f}")
print("wrote", OUT_FIG.relative_to(ROOT))
print("wrote", OUT_SD.relative_to(ROOT), "+ matrices + metadata")
print("rows = k1/k2/k3 (top->bottom) ; cols = Healthy IW / P1 / P2 (left->right)")
