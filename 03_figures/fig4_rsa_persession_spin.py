"""Figure 4 — Shared locomotor representation across imagined and real walking

Per-session, spin-calibrated spatial RSA (IW vs RW electrode geometry) per harmonic k1/k2/k3

Reads:
  04_source_data/fig4/fig4d_rsa_persession_spin_source_data.csv
  04_source_data/fig4/fig4d_rsa_persession_spin_metadata.csv
  04_source_data/fig4/fig4d_rsa_persession_spin_summary.csv
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
import mne

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/decoder_output"
IWF = R / "diag_harmonic_phase_responder_split/persession_harmonic_phase_cycle9_complex.csv"
RWF = R / "diag_real_walking_persession/persession_harmonic_phase_cycle9_complex.csv"
SD = ROOT / "04_source_data/fig4"
OUT_PNG = ROOT / "figures_out/fig4/fig4d_rsa_persession_spin.png"
BUILD_ID = "main"
N_SPIN = 1000
SEED = 0

COMMON = sorted(set(pd.read_csv(IWF).channel) & set(pd.read_csv(RWF).channel))
E, C = len(COMMON), 9
iuE = np.triu_indices(E, 1)


def load(path, k):
    """[session, cycle_point, channel] real/imag arrays - identical to fig4b_rsa_per_harmonic_k123.load."""
    d = pd.read_csv(path); d = d[d.harmonic_k == k]
    sess = sorted(d.session.unique()); cyc = sorted(d.cycle_fraction.unique())
    re = np.full((len(sess), len(cyc), len(COMMON)), np.nan); im = np.full_like(re, np.nan)
    piv = d.set_index(["session", "cycle_fraction", "channel"])
    for si, s in enumerate(sess):
        for ci, c in enumerate(cyc):
            sub = piv.loc[(s, c)]
            re[si, ci] = sub["real"].reindex(COMMON).to_numpy(); im[si, ci] = sub["imag"].reindex(COMMON).to_numpy()
    return re, im, sess, cyc


def rand_rot(rng):
    Q, Rm = np.linalg.qr(rng.normal(size=(3, 3))); Q = Q @ np.diag(np.sign(np.diag(Rm)))
    if np.linalg.det(Q) < 0: Q[:, 0] = -Q[:, 0]
    return Q


def session_rdm(re_s, im_s):
    """60 x 60 Euclidean RDM between the 18-value electrode feature vectors of ONE session."""
    F = np.concatenate([re_s, im_s], axis=0).T
    d2 = np.sum((F[:, None, :] - F[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.maximum(d2, 0.0))


def bh_fdr(p):
    p = np.asarray(p, float); n = p.size; o = np.argsort(p)
    q = np.empty(n); running = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        running = min(running, p[i] * n / rank); q[i] = running
    return q


def holm(p):
    p = np.asarray(p, float); n = p.size; o = np.argsort(p)
    q = np.empty(n); running = 0.0
    for j, i in enumerate(o):
        running = max(running, min(1.0, p[i] * (n - j))); q[i] = running
    return q


try:
    from statsmodels.stats.multitest import multipletests
    bh = lambda p: multipletests(p, method="fdr_bh")[1]
    hl = lambda p: multipletests(p, method="holm")[1]
except Exception:
    bh, hl = bh_fdr, holm

montage = mne.channels.make_standard_montage("standard_1005")
pos = montage.get_positions()["ch_pos"]
xyz = np.array([pos[c] for c in COMMON]); xyz0 = xyz - xyz.mean(0)
rng = np.random.default_rng(SEED)
PM = np.empty((N_SPIN, E), int)
for b in range(N_SPIN):
    Dm = np.linalg.norm(xyz0[:, None] - (xyz0 @ rand_rot(rng).T)[None], axis=2)
    _, pm = linear_sum_assignment(Dm)
    PM[b] = pm
FLAT = (PM[:, iuE[0]] * E + PM[:, iuE[1]])

sess_rows, null_rows, stat = [], [], {}
print(f"channels = {E} | cycle points = {C} | N_SPIN = {N_SPIN} | seed = {SEED}")
for K in (1, 2, 3):
    reI, imI, sessI, cyc = load(IWF, K)
    reR, imR, sessR, cycR = load(RWF, K)
    assert sessI == sessR, f"k{K}: IW/RW session lists differ"
    assert cyc == cycR and len(cyc) == C, f"k{K}: cycle grids differ"
    assert np.isfinite(reI).all() and np.isfinite(imI).all() and np.isfinite(reR).all() and np.isfinite(imR).all()
    S = len(sessI)
    rho_obs = np.empty(S); null = np.empty((S, N_SPIN))
    for si in range(S):
        A = session_rdm(reI[si], imI[si]); B = session_rdm(reR[si], imR[si])
        ra = stats.rankdata(A[iuE])
        rb_mat = np.zeros((E, E)); rb_mat[iuE] = stats.rankdata(B[iuE]); rb_mat += rb_mat.T
        rho_obs[si] = stats.spearmanr(A[iuE], B[iuE])[0]
        Prb = rb_mat.ravel()[FLAT]
        a0 = ra - ra.mean(); sa = np.sqrt((a0 ** 2).sum())
        P0 = Prb - Prb.mean(1, keepdims=True); sb = np.sqrt((P0 ** 2).sum(1))
        null[si] = (P0 @ a0) / (sa * sb)
        if si == 0 and K == 1:
            chk = stats.spearmanr(A[iuE], B[np.ix_(PM[0], PM[0])][iuE])[0]
            assert abs(chk - null[si, 0]) < 1e-10, (chk, null[si, 0])
        if (si + 1) % 20 == 0:
            print(f"  k{K}: {si + 1}/{S} sessions")
    nm, ns = null.mean(1), null.std(1, ddof=1)
    p95 = np.percentile(null, 95, axis=1)
    z = (rho_obs - nm) / ns
    T_obs = float(rho_obs.mean()); T_null = null.mean(0)
    p_spin = (np.sum(T_null >= T_obs) + 1) / (N_SPIN + 1)
    w_p = float(stats.wilcoxon(z, alternative="two-sided").pvalue)
    frac95 = float(np.mean(rho_obs > p95))
    part = pd.DataFrame({"p": [s.split("_")[0] for s in sessI], "z": z}).groupby("p")["z"].mean()
    w_pp = float(stats.wilcoxon(part.to_numpy(), alternative="two-sided").pvalue)
    stat[K] = dict(T_obs=T_obs, null_mean=float(nm.mean()), p_spin=float(p_spin), w_p=w_p,
                   frac95=frac95, w_pp=w_pp, n_part=int(part.size), rho=rho_obs, nm=nm, p95=p95)
    for si, s in enumerate(sessI):
        sess_rows.append(dict(session=s, participant=s.split("_")[0], harmonic_k=K,
                              rho_obs=round(float(rho_obs[si]), 6), null_mean=round(float(nm[si]), 6),
                              null_sd=round(float(ns[si]), 6), null_p95=round(float(p95[si]), 6),
                              z=round(float(z[si]), 6)))
    for r in range(N_SPIN):
        null_rows.append((K, r, round(float(T_null[r]), 6)))
    print(f"k{K}: mean rho = {T_obs:+.4f} | mean null = {nm.mean():+.4f} | P_spin = {p_spin:.4f} | "
          f"Wilcoxon(z) P = {w_p:.3e} | frac>null95 = {frac95:.3f} | participant P = {w_pp:.3e}")

ks = (1, 2, 3)
p_spin_v = np.array([stat[k]["p_spin"] for k in ks])
w_p_v = np.array([stat[k]["w_p"] for k in ks])
p_spin_bh, p_spin_holm = bh(p_spin_v), hl(p_spin_v)
w_bh = bh(w_p_v)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS
PS.apply(plt)
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
BLUE, GREY = "#1A1A1A", "#1A1A1A"
KL = {1: r"$\mathrm{k}_1$", 2: r"$\mathrm{k}_2$", 3: r"$\mathrm{k}_3$"}
jr = np.random.default_rng(7)

fig, ax = PS.panel_axes(plt, left_mm=14.5, bottom_mm=10.0, top_mm=11.5)
_LW_BOX, _LW_WHI, _LW_MED, _LW_MEAN, _LW_REF, _LW_ZERO, _S_PT = (
    (1.2, 1.0, 1.6, 1.9, 1.0, 0.8, 4))
stat_lines = []
for i, K in enumerate(ks):
    import project_palette as _P
    v = stat[K]["rho"]; col = _P.HARMONIC_COLORS[i]
    bp = ax.boxplot([v], positions=[i], widths=0.42, whis=(10, 90), showfliers=False,
                    patch_artist=True, zorder=2)
    bp["boxes"][0].set(facecolor=(*__import__("matplotlib.colors").colors.to_rgb(col), 0.32), edgecolor=col, linewidth=_LW_BOX)
    for el in ("whiskers", "caps"):
        for a in bp[el]: a.set(color=col, linewidth=_LW_WHI)
    bp["medians"][0].set(color=col, linewidth=_LW_MED)
    ax.scatter(i + jr.uniform(-0.13, 0.13, v.size), v, s=_S_PT, c=col, alpha=0.55,
               edgecolors="none", zorder=3)
    ax.plot([i - 0.30, i + 0.30], [stat[K]["nm"].mean()] * 2, color="#3A3A3A", lw=_LW_REF,
            ls=(0, (4, 2)), zorder=4)
    ax.plot([i - 0.30, i + 0.30], [stat[K]["T_obs"]] * 2, color="#111", lw=_LW_MEAN, zorder=5)
    mark = ("***" if p_spin_bh[i] < 1e-3 else "**" if p_spin_bh[i] < 1e-2
            else "*" if p_spin_bh[i] < .05 else "ns")
    ax.text(i, 1.02, mark, ha="center", va="bottom",
            fontsize=PS.STAR_PT,
            color="#1A1A1A", transform=ax.get_xaxis_transform())
    stat_lines.append(f"k{K}: spin P = {stat[K]['p_spin']:.3f}, BH {p_spin_bh[i]:.3f}  ({mark})")
ax.axhline(0, color="#555", lw=_LW_ZERO, zorder=1)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels([KL[k] for k in ks], fontsize=PS.TICK_PT)
ax.set_ylabel("Spatial RSA $\\rho$ (RW vs IW)")
ax.set_xlim(-0.55, 2.55)
PS.finish(ax)
fig.savefig(OUT_PNG, dpi=PS.DPI, facecolor="white")
PS.write_stats(OUT_PNG.with_name("fig4d_rsa_persession_spin_stats.txt"), stat_lines)
plt.close(fig)

pd.DataFrame(sess_rows).to_csv(SD / "fig4d_rsa_persession_spin_source_data.csv", index=False)
pd.DataFrame(null_rows, columns=["harmonic_k", "spin_index", "cohort_mean_null_rho"]).to_csv(
    SD / "fig4d_rsa_persession_spin_null.csv", index=False)

coh = pd.read_csv(SD / "fig4b_rsa_per_harmonic_source_data.csv").set_index("harmonic_k")
n_sess = len(sess_rows) // 3
meta = [
    ("panel", "Fig. 4 panel e - per-session spin-calibrated spatial RSA (IW vs RW), k1/k2/k3"),
    ("n_sessions", n_sess),
    ("n_participants", stat[1]["n_part"]),
    ("n_channels", E),
    ("n_cycle_points", C),
    ("n_spin", N_SPIN),
    ("seed", SEED),
    ("input_iw", str(IWF.relative_to(ROOT)).replace("\\", "/")),
    ("input_rw", str(RWF.relative_to(ROOT)).replace("\\", "/")),
    ("session_rdm", "Euclidean distance between 18-value electrode feature vectors [re(phi_1..9), im(phi_1..9)]; upper triangle = 1770 pairs"),
    ("spin_null", "one shared set of 1000 random-rotation electrode permutations (default_rng(0), rand_rot, linear_sum_assignment on standard_1005 xyz), applied to the RW RDM of every session and reused for all k"),
    ("primary_test", "cohort-mean spin test: P = (1 + #{mean_s rho_s^(r) >= mean_s rho_s}) / (N_SPIN + 1)"),
    ("secondary_test", "two-sided Wilcoxon signed-rank of the per-session spin-z across sessions (and across participant means)"),
    ("multiplicity", "BH-FDR and Holm across the three harmonics (statsmodels)"),
]
for i, K in enumerate(ks):
    s = stat[K]
    meta += [
        (f"k{K}_cohort_mean_rho_obs", round(s["T_obs"], 6)),
        (f"k{K}_cohort_mean_null_mean", round(s["null_mean"], 6)),
        (f"k{K}_P_spin", round(s["p_spin"], 6)),
        (f"k{K}_P_spin_BH", round(float(p_spin_bh[i]), 6)),
        (f"k{K}_P_spin_Holm", round(float(p_spin_holm[i]), 6)),
        (f"k{K}_wilcoxon_z_P", f"{s['w_p']:.6e}"),
        (f"k{K}_wilcoxon_z_P_BH", f"{float(w_bh[i]):.6e}"),
        (f"k{K}_frac_sessions_above_null95", round(s["frac95"], 4)),
        (f"k{K}_participant_wilcoxon_P", f"{s['w_pp']:.6e}"),
        (f"k{K}_participant_n", s["n_part"]),
    ]
for K in ks:
    meta += [
        (f"k{K}_cohortlevel_crossnobis_rho", float(coh.loc[K, "spatial_iwrw_rho"])),
        (f"k{K}_cohortlevel_crossnobis_spin_p", float(coh.loc[K, "spatial_iwrw_spin_p"])),
    ]
meta.append(("cohortlevel_source", "fig4b_rsa_per_harmonic_source_data.csv (not distributed) (crossnobis RDM cross-validated ACROSS sessions; ONE statistic per k) - copied here for comparison only"))
pd.DataFrame(meta, columns=["key", "value"]).to_csv(SD / "fig4d_rsa_persession_spin_metadata.csv", index=False)

summary = pd.DataFrame([{
    "harmonic_k": K, "n_sessions": n_sess, "n_participants": stat[K]["n_part"],
    "cohort_mean_rho_obs": round(stat[K]["T_obs"], 6), "cohort_mean_null_mean": round(stat[K]["null_mean"], 6),
    "P_spin_onesided": round(stat[K]["p_spin"], 6), "P_spin_BH": round(float(p_spin_bh[i]), 6),
    "P_spin_Holm": round(float(p_spin_holm[i]), 6), "wilcoxon_z_P_session": f"{stat[K]['w_p']:.3e}",
    "wilcoxon_z_P_BH": f"{float(w_bh[i]):.3e}", "wilcoxon_P_participant": f"{stat[K]['w_pp']:.3e}",
    "frac_sessions_above_own_null95": round(stat[K]["frac95"], 4),
    "median_rho_obs": round(float(np.median([r["rho_obs"] for r in sess_rows if r["harmonic_k"] == K])), 6),
} for i, K in enumerate(ks)])
summary.to_csv(SD / "fig4d_rsa_persession_spin_summary.csv", index=False)
print(summary.to_string(index=False))
wide = pd.DataFrame(sess_rows).pivot(index="session", columns="harmonic_k", values="rho_obs")
wide.columns = [f"k{K}_rho_obs" for K in wide.columns]
wide.to_csv(SD / "fig4d_rsa_persession_spin_wide.csv")
print("\nwrote", OUT_PNG.relative_to(ROOT))
print("wrote fig4d_rsa_persession_spin_{source_data,null,metadata}.csv")

