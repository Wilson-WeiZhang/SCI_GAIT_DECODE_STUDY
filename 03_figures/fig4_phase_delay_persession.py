"""Figure 4 — Shared locomotor representation across imagined and real walking

Per-session IW x RW gait-phase offset: one cosine fit of each session's correspondence matrix

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS
PS.apply(plt)
plt.rcParams.update({
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/decoder_output"
IWF = R / "diag_harmonic_phase_responder_split/persession_harmonic_phase_cycle9_complex.csv"
RWF = R / "diag_real_walking_persession/persession_harmonic_phase_cycle9_complex.csv"
GONI = R / "diag_main_exec_goni_psd_peaks/main_exec_goni_peak_bands.csv"
SD = ROOT / "04_source_data/fig4"
COHORT_META = SD / "fig4_phase_correspondence_metadata.csv"
OUT = ROOT / "figures_out/fig4/fig4_phase_delay_persession.png"
BLUE, GREY = "#1A1A1A", "#1A1A1A"
KL = {1: "$\\mathrm{k}_1$", 2: "$\\mathrm{k}_2$", 3: "$\\mathrm{k}_3$"}
KS = (1, 2, 3)
N_BOOT, SEED_BOOT, SEED_JIT = 1000, 0, 3
N_PERM, SEED_PERM = 1000, 20260925

_LW_BOX, _LW_WHI, _LW_MED, _LW_MEAN, _LW_ZERO, _LW_NULL, _S_PT, _PT_STAR = (
    (1.2, 1.0, 1.6, 1.9, 0.8, 0.7, 4, PS.STAR_PT))

iw, rw = pd.read_csv(IWF), pd.read_csv(RWF)
CH = sorted(set(iw.channel) & set(rw.channel))
SESS = sorted(set(iw.session) & set(rw.session))
FR = sorted(set(iw.cycle_fraction.round(4)) & set(rw.cycle_fraction.round(4)))
NS, NC, NF = len(SESS), len(CH), len(FR)
SUBJ = [s.split("_")[0] for s in SESS]
NSUB = len(set(SUBJ))


def cube(d):
    """(session, k, cycle point, channel) complex cube - identical to fig4_phase_correspondence.py."""
    ix = {"session": SESS, "harmonic_k": list(KS), "cf": FR, "channel": CH}
    d = d.assign(cf=d.cycle_fraction.round(4))
    d = d[d.session.isin(SESS) & d.channel.isin(CH) & d.cf.isin(FR) & d.harmonic_k.isin(KS)]
    idx = [d[c].map({v: i for i, v in enumerate(ix[c])}).to_numpy() for c in ix]
    A = np.full((NS, len(KS), NF, NC), np.nan + 0j)
    A[tuple(idx)] = d["real"].to_numpy() + 1j * d["imag"].to_numpy()
    assert not np.isnan(A).any(), "incomplete session x k x phase x channel cube"
    return A


A_iw, A_rw = cube(iw), cube(rw)
C_rw = A_rw.mean(0)
ref = float(np.angle(C_rw.ravel()[np.nanargmax(np.abs(C_rw.ravel()))]))
S_iw, S_rw = np.real(A_iw * np.exp(-1j * ref)), np.real(A_rw * np.exp(-1j * ref))
dup = max(float(np.abs(S_iw[:, :, 0] - S_iw[:, :, -1]).max()), float(np.abs(S_rw[:, :, 0] - S_rw[:, :, -1]).max()))
assert dup < 1e-9, f"100 % column is not a repeat of the 0 % column (max abs diff {dup:.3g})"
S_iw, S_rw = S_iw[:, :, :-1], S_rw[:, :, :-1]
PHI = np.asarray(FR[:-1], float)
NP_ = len(PHI)
DEL = PHI[:, None] - PHI[None, :]

DESIGN = [np.column_stack([np.cos(2 * np.pi * k * DEL.ravel()), np.sin(2 * np.pi * k * DEL.ravel()),
                           np.ones(DEL.size)]) for k in KS]
PINV = [np.linalg.pinv(X) for X in DESIGN]


def zscore_rows(X):
    return (X - X.mean(-1, keepdims=True)) / X.std(-1, ddof=1, keepdims=True)


def corr_mats(P_iw, P_rw):
    return np.einsum("kic,kjc->kij", zscore_rows(P_iw), zscore_rows(P_rw)) / (NC - 1)


def fit_cos(M):
    """M (k, 8, 8) -> A, Delta (rad of the harmonic), b, R2, each length 3."""
    A = np.empty(len(KS)); D = np.empty(len(KS)); B = np.empty(len(KS)); R2 = np.empty(len(KS))
    for ki in range(len(KS)):
        y = M[ki].ravel()
        beta = PINV[ki] @ y
        A[ki], D[ki], B[ki] = np.hypot(beta[0], beta[1]), np.arctan2(beta[1], beta[0]), beta[2]
        res = y - DESIGN[ki] @ beta
        R2[ki] = 1.0 - res @ res / ((y - y.mean()) @ (y - y.mean()))
    return A, D, B, R2


def wrap_pct(x, k, centre=0.0):
    """radians-derived % of cycle -> wrapped to centre +/- 50/k."""
    per = 100.0 / k
    return (x - centre + per / 2) % per - per / 2 + centre


A_s = np.empty((NS, len(KS))); D_s = np.empty((NS, len(KS)))
B_s = np.empty((NS, len(KS))); R2_s = np.empty((NS, len(KS)))
for si in range(NS):
    A_s[si], D_s[si], B_s[si], R2_s[si] = fit_cos(corr_mats(S_iw[si], S_rw[si]))
DP_s = np.column_stack([wrap_pct(D_s[:, ki] / (2 * np.pi * k) * 100, k) for ki, k in enumerate(KS)])
TH_s = np.column_stack([2 * np.pi * k * DP_s[:, ki] / 100 for ki, k in enumerate(KS)])

rngp = np.random.default_rng(SEED_PERM)
NULL_DP = np.empty((NS, N_PERM, len(KS)))
for si in range(NS):
    M_si = corr_mats(S_iw[si], S_rw[si])
    for b in range(N_PERM):
        D_null = fit_cos(np.stack([M_si[ki][:, rngp.permutation(NP_)] for ki in range(len(KS))]))[1]
        NULL_DP[si, b] = [wrap_pct(D_null[ki] / (2 * np.pi * k) * 100, k) for ki, k in enumerate(KS)]
NULL_BAND = {k: np.percentile(NULL_DP[:, :, ki].ravel(), [2.5, 97.5]) for ki, k in enumerate(KS)}


def circ_mean_pct(theta, k, w=None):
    """circular mean of harmonic angles -> % of cycle in (-50/k, +50/k]."""
    z = np.exp(1j * theta) if w is None else w * np.exp(1j * theta)
    a = float(np.angle(z.sum()))
    return a / (2 * np.pi * k) * 100.0


def rbar(theta, w=None):
    z = np.exp(1j * theta) if w is None else w * np.exp(1j * theta)
    return float(np.abs(z.sum()) / (len(theta) if w is None else w.sum()))


def rayleigh_p(theta):
    """standard approximation: z = n R_bar^2; P ~ exp(sqrt(1 + 4n + 4(n^2 - R^2)) - (1 + 2n)), R = n R_bar."""
    n = len(theta)
    Rb = rbar(theta)
    Rv = n * Rb
    return float(np.exp(np.sqrt(1.0 + 4.0 * n + 4.0 * (n ** 2 - Rv ** 2)) - (1.0 + 2.0 * n))), Rb, n * Rb ** 2


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    q = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1):
        run = min(run, p[o[r]] * m / (r + 1))
        q[o[r]] = run
    return q


gp = pd.read_csv(GONI)
g1 = gp[gp.harmonic_k == 1].set_index("session")
F0 = pd.Series({s: 0.5 * (g1.loc[s, "band_lo"] + g1.loc[s, "band_hi"]) for s in SESS if s in g1.index}, dtype=float)
HAVE_MS = len(F0) == NS
PERIOD = np.array([1.0 / F0[s] for s in SESS]) if HAVE_MS else np.full(NS, np.nan)
MS_s = DP_s / 100.0 * PERIOD[:, None] * 1000.0

cm = pd.read_csv(COHORT_META).set_index("key")["value"]
COH = {k: float(cm[f"k{k}_Delta_pct_cycle"]) for k in KS}

rng = np.random.default_rng(SEED_BOOT)
BOOT_IDX = np.array([rng.integers(0, NS, NS) for _ in range(N_BOOT)])
stat = {}
for ki, k in enumerate(KS):
    th = TH_s[:, ki]
    mu = circ_mean_pct(th, k)
    mu_w = circ_mean_pct(th, k, w=A_s[:, ki])
    P, Rb, z = rayleigh_p(th)
    bm = np.array([circ_mean_pct(th[ix], k) for ix in BOOT_IDX])
    lo, hi = np.percentile(wrap_pct(bm, k, centre=mu), [2.5, 97.5])
    dp = DP_s[:, ki]
    w_st, w_p = stats.wilcoxon(dp)
    pv = pd.DataFrame({"subj": SUBJ, "z": np.exp(1j * th)}).groupby("subj")["z"].mean()
    pth = np.angle(pv.to_numpy())
    pP, pRb, _ = rayleigh_p(pth)
    stat[k] = dict(mu=mu, mu_w=mu_w, lo=lo, hi=hi, Rb=Rb, z=z, P=P, wp=float(w_p), wst=float(w_st),
                   med=float(np.median(dp)), q1=float(np.percentile(dp, 25)), q3=float(np.percentile(dp, 75)),
                   p5=float(np.percentile(dp, 5)), p95=float(np.percentile(dp, 95)),
                   pmu=circ_mean_pct(pth, k), pRb=pRb, pP=pP, npart=len(pth),
                   mA=float(A_s[:, ki].mean()), mdR2=float(np.median(R2_s[:, ki])),
                   ms_med=float(np.median(MS_s[:, ki])) if HAVE_MS else float("nan"))
Q = bh([stat[k]["P"] for k in KS])
for ki, k in enumerate(KS):
    stat[k]["P_bh"] = float(Q[ki])
    stat[k]["sig"] = bool(Q[ki] < 0.05)

fig, ax = PS.panel_axes(plt, left_mm=14.5, bottom_mm=10.0, top_mm=11.5)
rj = np.random.default_rng(SEED_JIT)
stat_lines = []
ax.axhline(0, color="#555555", lw=_LW_ZERO, zorder=1)
for ki, k in enumerate(KS):
    x = ki + 1.0
    import project_palette as _P
    c = _P.HARMONIC_COLORS[ki]
    y = DP_s[:, ki]
    for yb in NULL_BAND[k]:
        ax.plot([x - 0.42, x + 0.42], [yb] * 2, color="#C8C8C8", lw=_LW_NULL,
                ls=(0, (3, 2.5)), zorder=1, clip_on=True)
    ax.scatter(x + rj.uniform(-0.17, 0.17, NS), y, s=_S_PT, c=c, alpha=0.55, edgecolors="none", zorder=3)
    bp = ax.boxplot([y], positions=[x], widths=0.44, whis=(10, 90), showfliers=False,
                    patch_artist=True, zorder=4)
    for b in bp["boxes"]:
        b.set(facecolor=(*__import__("matplotlib.colors").colors.to_rgb(c), 0.32), edgecolor=c, linewidth=_LW_BOX)
    for part in ("whiskers", "caps"):
        for b in bp[part]:
            b.set(color=c, linewidth=_LW_WHI)
    for b in bp["medians"]:
        b.set(color=c, linewidth=_LW_MED)
    ax.plot([x - 0.30, x + 0.30], [stat[k]["mu"]] * 2, color="#111111", lw=_LW_MEAN, solid_capstyle="butt", zorder=6)
    mark = ("***" if stat[k]["P_bh"] < 1e-3 else "**" if stat[k]["P_bh"] < 1e-2
            else "*" if stat[k]["P_bh"] < .05 else "ns")
    ax.text(x, 1.02, mark, ha="center", va="bottom",
            fontsize=_PT_STAR,
            color="#1A1A1A", transform=ax.get_xaxis_transform())
    stat_lines.append(f"k{k}: Delta = {stat[k]['mu']:+.1f} % [{stat[k]['lo']:+.1f}, "
                      f"{stat[k]['hi']:+.1f}], BH P = {stat[k]['P_bh']:.3g}  ({mark})")
ax.set_xlim(0.45, 3.55)
ax.set_ylim(-53, 53)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([KL[k] for k in KS], fontsize=PS.TICK_PT)
ax.set_yticks([-50, -25, 0, 25, 50])
ax.set_ylabel("IW − RW phase offset (% cycle)", labelpad=2)
PS.finish(ax)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=PS.DPI, facecolor="white")
stat_lines += [f"k{k}: phase-label permutation null 95% band [{NULL_BAND[k][0]:+.2f}, {NULL_BAND[k][1]:+.2f}] % "
               f"(pooled {NS} sessions x {N_PERM} shuffles, seed {SEED_PERM})" for k in KS]
stat_lines.append("thick tick = unweighted per-session circular mean of Delta_s; box whiskers = 10th-90th percentiles")
PS.write_stats(OUT.with_name("fig4_phase_delay_persession_stats.txt"), stat_lines)
plt.close(fig)

long = pd.DataFrame([dict(session=SESS[si], participant=SUBJ[si], harmonic_k=k,
                          A=round(float(A_s[si, ki]), 6),
                          Delta_pct_cycle=round(float(DP_s[si, ki]), 4),
                          Delta_deg_harmonic=round(float(DP_s[si, ki] * k * 3.6), 3),
                          Delta_ms=round(float(MS_s[si, ki]), 2) if HAVE_MS else "",
                          b=round(float(B_s[si, ki]), 6), R2=round(float(R2_s[si, ki]), 4))
                     for ki, k in enumerate(KS) for si in range(NS)])
long.to_csv(SD / "fig4_phase_delay_persession_source_data.csv", index=False)

wide = pd.DataFrame({"session": SESS,
                     **{f"k{k}_Delta_pct": np.round(DP_s[:, ki], 4) for ki, k in enumerate(KS)},
                     **{f"k{k}_A": np.round(A_s[:, ki], 6) for ki, k in enumerate(KS)}})
wide.to_csv(SD / "fig4_phase_delay_persession_wide.csv", index=False)

meta = [
    ("panel", "Fig. 4 panel d - per-session IW-vs-RW gait-phase delay (cosine-fit offset, one fit per session)"),
    ("n_sessions", NS), ("n_participants", NSUB), ("n_channels", NC),
    ("input_IW", str(IWF.relative_to(ROOT)).replace("\\", "/")),
    ("input_RW", str(RWF.relative_to(ROOT)).replace("\\", "/")),
    ("input_stride_period", str(GONI.relative_to(ROOT)).replace("\\", "/")),
    ("input_cohort_reference", str(COHORT_META.relative_to(ROOT)).replace("\\", "/")),
    ("cycle_points", ";".join(f"{p:g}" for p in PHI) + " (the 9th, 100 %, column repeats 0 % and is dropped)"),
    ("reference_phase_rad", f"{ref:.6f} (cohort RW map's unified reference, applied to BOTH conditions in every "
                            "session - identical to fig4_phase_correspondence.py; it does NOT cancel, since "
                            "Re(c e^-i ref) is not a rotation of a real map, and is held at the cohort value so "
                            "all sessions share one convention)"),
    ("model", "per session: M_s,k(i,j) = A cos(2 pi k (phi_i - phi_j) - Delta) + b, OLS on [cos, sin, 1] over all 64 cells"),
    ("Delta_convention", "Delta_pct_cycle = Delta_rad / (2 pi k) x 100, wrapped to (-50/k, +50/k]; phi_IW = phi_RW "
                         "+ Delta (mod 1/k), so NEGATIVE Delta = the imagined map arrives EARLIER in the cycle "
                         "than the executed one (imagery leads); Delta_deg_harmonic = Delta_pct x k x 3.6"),
    ("Delta_ms", "Delta_pct/100 x stride period x 1000; period = 1 / f0 with f0 = centre of that SESSION'S OWN "
                 "k=1 EXEC goniometer band, (band_lo + band_hi)/2" if HAVE_MS else
                 "not computed - stride-period table missing"),
    ("stride_period_s_median", round(float(np.median(PERIOD)), 4) if HAVE_MS else "NA"),
    ("circular_statistic", "theta_s = 2 pi k Delta_s / 100; circular mean = angle(sum exp(i theta)); R_bar = "
                           "|sum exp(i theta)| / n; UNWEIGHTED is primary, the amplitude-weighted variant "
                           "(weights = A_s) is reported alongside"),
    ("rayleigh_method", "z = n R_bar^2; P ~ exp(sqrt(1 + 4n + 4(n^2 - R^2)) - (1 + 2n)), R = n R_bar (standard "
                        "approximation; scipy has no Rayleigh test)"),
    ("multiple_comparison", "Benjamini-Hochberg over the 3 harmonics; significance is shown only by the mark "
                            "above each harmonic (*** / ** / * at BH q < 0.001 / 0.01 / 0.05, ns otherwise); "
                            "the points and box of each harmonic are drawn in its identity colour "
                            "(project_palette.HARMONIC_COLORS) whatever its significance"),
    ("wilcoxon_note", "secondary, linear: scipy.stats.wilcoxon of the wrapped Delta_s against 0"),
    ("median_IQR_note", "linear median / quartiles of Delta_s AFTER wrapping to (-50/k, +50/k]; they are linear "
                        "statistics on circular data and are secondary to the circular mean"),
    ("n_boot", N_BOOT), ("seed_bootstrap", SEED_BOOT), ("seed_jitter", SEED_JIT),
    ("bootstrap", "resample the sessions with replacement, recompute the circular mean; 95 % percentile CI taken "
                  "after wrapping the bootstrap means onto the point estimate"),
]
for k in KS:
    s = stat[k]
    meta += [
        (f"k{k}_circ_mean_Delta_pct", round(s["mu"], 4)),
        (f"k{k}_circ_mean_CI95_lo", round(s["lo"], 4)),
        (f"k{k}_circ_mean_CI95_hi", round(s["hi"], 4)),
        (f"k{k}_circ_mean_Delta_pct_weighted", round(s["mu_w"], 4)),
        (f"k{k}_circ_mean_Delta_deg_harmonic", round(s["mu"] * k * 3.6, 3)),
        (f"k{k}_R_bar", round(s["Rb"], 4)),
        (f"k{k}_rayleigh_z", round(s["z"], 4)),
        (f"k{k}_rayleigh_P", f"{s['P']:.4g}"),
        (f"k{k}_rayleigh_P_BH", f"{s['P_bh']:.4g}"),
        (f"k{k}_wilcoxon_P", f"{s['wp']:.4g}"),
        (f"k{k}_median_Delta_pct", round(s["med"], 4)),
        (f"k{k}_IQR_Delta_pct", round(s["q3"] - s["q1"], 4)),
        (f"k{k}_Q1_Delta_pct", round(s["q1"], 4)), (f"k{k}_Q3_Delta_pct", round(s["q3"], 4)),
        (f"k{k}_median_Delta_ms", round(s["ms_med"], 2) if HAVE_MS else "NA"),
        (f"k{k}_participant_circ_mean_Delta_pct", round(s["pmu"], 4)),
        (f"k{k}_participant_R_bar", round(s["pRb"], 4)),
        (f"k{k}_participant_rayleigh_P", f"{s['pP']:.4g}"),
        (f"k{k}_participant_n", s["npart"]),
        (f"k{k}_cohort_Delta_pct_copied", round(COH[k], 4)),
        (f"k{k}_mean_A", round(s["mA"], 4)),
        (f"k{k}_median_R2", round(s["mdR2"], 4)),
    ]
meta.append(("note", "descriptive: the per-session matrix is built from 8 cycle points x 60 channels; the claim is about the CONSISTENCY "
                       "of the offset across sessions (Rayleigh), not about any individual session's Delta"))
meta.append(("thick_tick", "thick black tick in each column = the UNWEIGHTED per-session circular mean of Delta_s "
                           "(k*_circ_mean_Delta_pct), the quantity reported in the legend"))
meta.append(("box_whiskers", "box = median and IQR; whiskers = 10th-90th percentiles of the wrapped Delta_s; no fliers drawn, every session is a jittered point"))
meta.append(("perm_null_method", f"phase-label permutation null: for each session and harmonic the RW cycle-point labels "
                                 f"(columns of that session's 8 x 8 IW x RW correspondence matrix, same convention as the "
                                 f"cohort permutation in fig4_phase_correspondence.py; cohort reference phase unchanged) "
                                 f"are shuffled, the same per-session OLS cosine fit is rerun and the null Delta is wrapped "
                                 f"to (-50/k, +50/k] exactly like the observed Delta; N_PERM = {N_PERM} shuffles per session, "
                                 f"independent permutation per harmonic, numpy default_rng({SEED_PERM}); null Deltas pooled "
                                 f"over the {NS} sessions per harmonic; dashed lines = 2.5th and 97.5th percentiles"))
meta.append(("perm_null_n_per_harmonic", NS * N_PERM))
for k in KS:
    meta += [(f"k{k}_perm_null_Delta_pct_p2.5", round(float(NULL_BAND[k][0]), 4)),
             (f"k{k}_perm_null_Delta_pct_p97.5", round(float(NULL_BAND[k][1]), 4))]
pd.DataFrame(meta, columns=["key", "value"]).to_csv(SD / "fig4_phase_delay_persession_metadata.csv", index=False)

print(f"n_sessions={NS} n_participants={NSUB} n_channels={NC} | ref={ref:.6f} | ms column: {HAVE_MS}")
hdr = f"{'k':>2} | {'circ mean %':>11} [{'CI95':^17}] | {'R_bar':>6} {'rayP':>10} {'BH':>10} {'wilcoxP':>10} | "
print(hdr + f"{'median':>7} {'IQR':>6} | {'meanA':>6} {'medR2':>6} | {'cohort':>7} | {'med ms':>8} | part P")
for k in KS:
    s = stat[k]
    print(f"{k:>2} | {s['mu']:+11.2f} [{s['lo']:+7.2f},{s['hi']:+7.2f}] | {s['Rb']:6.3f} {s['P']:10.3g} "
          f"{s['P_bh']:10.3g} {s['wp']:10.3g} | {s['med']:+7.2f} {s['q3'] - s['q1']:6.2f} | {s['mA']:6.3f} "
          f"{s['mdR2']:6.3f} | {COH[k]:+7.2f} | {s['ms_med']:8.1f} | {s['pP']:.3g}")
for k in KS:
    print(f"k{k}: perm-null 95% band [{NULL_BAND[k][0]:+.2f}, {NULL_BAND[k][1]:+.2f}] %  thick tick {stat[k]['mu']:+.4f} %")
print("wrote", OUT.relative_to(ROOT))

