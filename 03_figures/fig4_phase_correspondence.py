"""Figure 4 — Shared locomotor representation across imagined and real walking

IW x RW gait-phase correspondence matrices (k1 / k2 / k3) and their cyclic lag profile

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import panel_style as PS
from sup_panel_export import PANEL_MODE, PANEL_DIR, export_legend, export_stats

plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.linewidth": 2.2,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
if PANEL_MODE:
    PS.apply(plt)
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic"})
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/decoder_output"
IWF = R / "diag_harmonic_phase_responder_split/persession_harmonic_phase_cycle9_complex.csv"
RWF = R / "diag_real_walking_persession/persession_harmonic_phase_cycle9_complex.csv"
OUT = ROOT / "figures_out/fig4/fig4_phase_correspondence.png"
OUT_MAT = ROOT / "figures_out/fig4/fig4_phase_correspondence_matrices.png"
OUT_LAG = ROOT / "figures_out/supplementary/sup_fig4_phase_lag_profile.png"
SD = ROOT / "04_source_data/fig4"
BLUE, GREY, DARK = "#0072B2", "#9A9A9A", "#3A3A3A"
KL = {1: "$\\mathrm{k}_1$", 2: "$\\mathrm{k}_2$", 3: "$\\mathrm{k}_3$"}
KS = (1, 2, 3)
N_BOOT, N_PERM, SEED_BOOT, SEED_PERM = 1000, 1000, 0, 1

iw, rw = pd.read_csv(IWF), pd.read_csv(RWF)
CH = sorted(set(iw.channel) & set(rw.channel))
SESS = sorted(set(iw.session) & set(rw.session))
FR = sorted(set(iw.cycle_fraction.round(4)) & set(rw.cycle_fraction.round(4)))
NS, NC, NF = len(SESS), len(CH), len(FR)


def cube(d):
    """(session, k, cycle point, channel) complex cube."""
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
    """(k, phi_IW, chan) x (k, phi_RW, chan) -> (k, phi_IW, phi_RW) Pearson r across channels."""
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
    """radians of harmonic k -> % of the gait cycle, wrapped to centre +/- 50/k."""
    per = 100.0 / k
    return (x - centre + per / 2) % per - per / 2 + centre


M_obs = corr_mats(S_iw.mean(0), S_rw.mean(0))
A_obs, D_obs, B_obs, R2_obs = fit_cos(M_obs)
D_pct = np.array([wrap_pct(D_obs[ki] / (2 * np.pi * k) * 100, k) for ki, k in enumerate(KS)])

LAG_BINS = np.arange(-(NP_ // 2), NP_ // 2)
LAG_PCT = LAG_BINS * (100.0 / NP_)
prof = np.array([[np.mean([M_obs[ki, i, (i - L) % NP_] for i in range(NP_)]) for L in LAG_BINS]
                 for ki in range(len(KS))])
prof_fit = np.array([A_obs[ki] * np.cos(2 * np.pi * k * LAG_PCT / 100 - D_obs[ki]) + B_obs[ki]
                     for ki, k in enumerate(KS)])

rng = np.random.default_rng(SEED_BOOT)
bA = np.empty((N_BOOT, len(KS))); bD = np.empty((N_BOOT, len(KS)))
for b in range(N_BOOT):
    s = rng.integers(0, NS, NS)
    a, d, _, _ = fit_cos(corr_mats(S_iw[s].mean(0), S_rw[s].mean(0)))
    bA[b], bD[b] = a, d
A_ci = np.percentile(bA, [2.5, 97.5], axis=0)
D_ci = np.empty((2, len(KS)))
for ki, k in enumerate(KS):
    D_ci[:, ki] = np.percentile(wrap_pct(bD[:, ki] / (2 * np.pi * k) * 100, k, centre=D_pct[ki]), [2.5, 97.5])

rngp = np.random.default_rng(SEED_PERM)
pA = np.empty((N_PERM, len(KS)))
for b in range(N_PERM):
    pA[b] = fit_cos(np.stack([M_obs[ki][:, rngp.permutation(NP_)] for ki in range(len(KS))]))[0]
A_p = (1 + (pA >= A_obs[None]).sum(0)) / (N_PERM + 1)

VMAX = float(np.abs(M_obs).max())
EXT = [-50 / NP_, 100 - 50 / NP_] * 2


STY_H = dict(title=17, tick=11, axlab=13, cblab=11, cbtick=10, axlw=2.2, tickw=2.2,
             ticklen=4, ridge=1.1, diag=2.4, halo=4.4, arrow=2.0, mut=15,
             strk_arr=4.0, cbout=1.2, tpad=5)
STY_V = dict(title=11.0, tick=10.2, axlab=10.5, cblab=9.5, cbtick=9.0, axlw=1.97,
             tickw=1.97, ticklen=3.6, ridge=0.98, diag=2.15, halo=3.95, arrow=1.79, mut=12,
             strk_arr=3.2, cbout=1.07, tpad=4)


def draw_matrices(fig, mx, sty=STY_H, orient="h", cax=None):
    """the three correspondence matrices + shared colorbar; shared by both PNGs."""
    im = None
    for ki, k in enumerate(KS):
        ax = mx[ki]
        im = ax.imshow(M_obs[ki], cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, origin="lower",
                       extent=EXT, interpolation="nearest", aspect="equal", zorder=0)
        offsets = []
        for m in range(k):
            for w in (-1, 0, 1):
                c = D_pct[ki] + m * 100.0 / k + 100.0 * w
                offsets.append(c)
                _xx = np.array([EXT[0], EXT[1]])
                ax.fill_between(_xx, _xx + c + (D_ci[0, ki] - D_pct[ki]),
                                _xx + c + (D_ci[1, ki] - D_pct[ki]),
                                color="white", alpha=0.42, lw=0, zorder=3.5, clip_on=True)
                ax.plot([EXT[0], EXT[1]], [EXT[0] + c, EXT[1] + c], color="#1A1A1A",
                        lw=sty["ridge"], ls=(0, (4, 2.5)), zorder=4, clip_on=True)
        ax.plot([EXT[0], EXT[1]], [EXT[0], EXT[1]], color="white", lw=sty["diag"], ls="-", zorder=5,
                clip_on=True, path_effects=[pe.withStroke(linewidth=sty["halo"], foreground="#1A1A1A")])
        c_near = min(offsets, key=abs)
        xa = 0.5 * (EXT[0] + EXT[1])
        arr = ax.annotate("", xy=(xa, xa + c_near), xytext=(xa, xa), zorder=6,
                          arrowprops=dict(arrowstyle="-|>", color="#1A1A1A", lw=sty["arrow"],
                                          mutation_scale=sty["mut"], shrinkA=0, shrinkB=0))
        arr.arrow_patch.set_path_effects([pe.withStroke(linewidth=sty["strk_arr"], foreground="white")])
        ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
        ax.set_title(KL[k], fontsize=sty["title"], pad=sty["tpad"])
        ax.set_xticks([0, 25, 50, 75]); ax.set_yticks([0, 25, 50, 75])
        if orient == "h":
            ax.set_xlabel("RW gait phase (%)", labelpad=2, fontsize=sty["axlab"])
            if ki == 0:
                ax.set_ylabel("IW gait phase (%)", labelpad=2, fontsize=sty["axlab"])
            else:
                ax.set_yticklabels([])
        else:
            if ki == len(KS) - 1:
                ax.set_xlabel("RW gait phase (%)", labelpad=2, fontsize=sty["axlab"])
            else:
                ax.set_xticklabels([])
        for sp in ax.spines.values():
            sp.set_linewidth(sty["axlw"])
        ax.tick_params(width=sty["tickw"], length=sty["ticklen"], labelsize=sty["tick"])
    if cax is None:
        cb = fig.colorbar(im, ax=mx, fraction=0.03, pad=0.02)
    else:
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("spatial  $r$  (IW map vs RW map)", fontsize=sty["cblab"], labelpad=2)
    cb.ax.tick_params(labelsize=sty["cbtick"], length=3, width=sty["cbout"])
    cb.outline.set_linewidth(sty["cbout"])


fig = plt.figure(figsize=(12.0, 3.4), constrained_layout=True)
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.22])
mx = [fig.add_subplot(gs[0, i]) for i in range(3)]
draw_matrices(fig, mx)

COL = {1: GREY, 2: BLUE, 3: DARK}
LAGGRID = np.linspace(LAG_PCT[0], LAG_PCT[0] + 100, 401)


LAG_LAB_X = LAG_PCT[-1] + 3.0


def draw_lag(axp, labels=True, si=False):
    """cyclic lag profile: observed points + the fitted cosines, each curve labelled at its right-hand
    end."""
    for ki, k in enumerate(KS):
        axp.plot(LAGGRID, A_obs[ki] * np.cos(2 * np.pi * k * LAGGRID / 100 - D_obs[ki]) + B_obs[ki],
                 color=COL[k], lw=1.0 if si else 2.0, ls="-", zorder=2)
        axp.plot(LAG_PCT, prof[ki], ls="none", marker="o", ms=2.4 if si else 5.5, mfc=COL[k],
                 mec="white", mew=0.4 if si else 1.0, zorder=3)
        if labels:
            y_end = A_obs[ki] * np.cos(2 * np.pi * k * LAG_PCT[-1] / 100 - D_obs[ki]) + B_obs[ki]
            axp.text(LAG_LAB_X, y_end, KL[k], color=COL[k], fontsize=13, va="center", ha="left",
                     clip_on=False, zorder=4)
    axp.axhline(0, color="#555555", lw=0.5 if si else 1.0, zorder=1)
    axp.axvline(0, color="#BBBBBB", lw=0.5 if si else 1.0, ls=(0, (3, 2)), zorder=1)
    axp.set_xlabel("IW − RW lag (% gait cycle)", labelpad=2)
    axp.set_ylabel("Spatial correlation ($r$)", labelpad=2)
    axp.set_xlim(LAG_PCT[0] - 3, (LAG_LAB_X + 9) if labels else LAG_PCT[-1] + 3)
    axp.set_ylim(-0.85, 0.85)
    axp.set_yticks([-0.5, 0, 0.5])
    axp.set_xticks([-50, -25, 0, 25])
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    axp.spines["bottom"].set_bounds(LAG_PCT[0], LAG_PCT[-1])
    axp.tick_params(width=2.2, length=5, labelsize=11)


draw_lag(fig.add_subplot(gs[0, 3]))
if not PANEL_MODE:
    fig.savefig(OUT, dpi=300, facecolor="white")
plt.close(fig)

FW_V, DPI_V = 3.2, 300
L_V, R_V = 0.52, 0.06
AXW_V = FW_V - L_V - R_V
TGAP, XLAB_H = 0.27, 0.40
CB_GAP, CB_H, CB_LAB, BOT_V, TOP_V = 0.10, 0.10, 0.33, 0.04, 0.04
CB_Y = BOT_V + CB_LAB
STACK0 = CB_Y + CB_H + CB_GAP + XLAB_H
FH_V = STACK0 + 3 * (AXW_V + TGAP) + TOP_V
figm = plt.figure(figsize=(FW_V, FH_V))
mxv = [figm.add_axes([L_V / FW_V, (STACK0 + (2 - ki) * (AXW_V + TGAP)) / FH_V,
                      AXW_V / FW_V, AXW_V / FH_V]) for ki in range(3)]
caxv = figm.add_axes([L_V / FW_V, CB_Y / FH_V, AXW_V / FW_V, CB_H / FH_V])
draw_matrices(figm, mxv, sty=STY_V, orient="v", cax=caxv)
figm.text(0.115 / FW_V, (STACK0 + 1.5 * AXW_V + TGAP) / FH_V, "IW gait phase (%)", rotation=90,
          ha="center", va="center", fontsize=STY_V["axlab"])
if not PANEL_MODE:
    figm.savefig(OUT_MAT, dpi=DPI_V, facecolor="white")
plt.close(figm)

OUT_LAG.parent.mkdir(parents=True, exist_ok=True)
if PANEL_MODE:
    import si_panel_style as SI
    SI.apply(plt)
    figl, _axl = SI.panel(plt, "third", 48.0)
else:
    figl = plt.figure(figsize=(4.8, 3.6), constrained_layout=True)
    _axl = figl.add_subplot(111)
draw_lag(_axl, labels=not PANEL_MODE, si=PANEL_MODE)


def _pfmt(p):
    """house style: lower-case italic p, leading zero, three decimals, floored at 0.001."""
    return "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"


if PANEL_MODE:
    _axl.tick_params(width=SI.LW_AXIS, length=2.0, labelsize=SI.TICK_PT)
    for _sp in _axl.spines.values():
        _sp.set_linewidth(SI.LW_AXIS)
    _axl.xaxis.label.set_size(SI.TITLE_PT)
    _axl.yaxis.label.set_size(SI.TITLE_PT)
    SI.finish(_axl)
    assert _axl.get_title() == "" and _axl.get_legend() is None, "panel carries a title or a legend"
    assert list(_axl.texts) == [], f"panel carries in-axes text: {[t.get_text() for t in _axl.texts]}"
    assert list(figl.texts) == [], f"panel carries figure text: {[t.get_text() for t in figl.texts]}"
    print("spec-check S4c: title empty, no legend, ax.texts == [], fig.texts == []  -> PASS")
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    _p8d = PANEL_DIR / "S4c_phase_lag_profile.png"
    SI.save(figl, _p8d)
    export_legend([Line2D([0], [0], color=COL[k], lw=1.0, marker="o", ms=3.0, mfc=COL[k],
                          mec="white", mew=0.4, label=KL[k]) for k in KS],
                  "S4", "c", "phase_lag_profile", ncol=3, figsize=(2.4, 0.35),
                  fontsize=SI.TICK_PT)
    _lines = [
        "Cosine fit of the imagined-walking x real-walking correspondence matrix, per harmonic",
        "(95 % CI = percentile of a 1000-resample session bootstrap; "
        f"permutation null = {N_PERM} shuffles of the real-walking gait-phase labels)",
    ]
    for _ki, _k in enumerate(KS):
        _lines.append(
            "k$_%d$:  amplitude $A$ = %.3f [%.3f, %.3f];  lag $\\Delta$ = %+.1f %% gait cycle "
            "[%+.1f, %+.1f];  %s"
            % (_k, A_obs[_ki], A_ci[0, _ki], A_ci[1, _ki], D_pct[_ki], D_ci[0, _ki], D_ci[1, _ki],
               _pfmt(float(A_p[_ki]))))
    export_stats(_lines, "S4", "c", "phase_lag_profile", width=7.4)
    (PANEL_DIR / "S4c_phase_lag_profile_stats.txt").write_text(
        "\n".join(_lines) + "\n", encoding="utf-8")
    print("wrote", PANEL_DIR / "S4c_phase_lag_profile_stats.txt")
else:
    figl.savefig(OUT_LAG, dpi=300, facecolor="white")
plt.close(figl)

def write_sd(df, name):
    """Source Data write, skipped in SI panel-spec runs: those re-render one panel and must not rewrite the Source Data CSV."""
    if PANEL_MODE:
        return
    df.to_csv(SD / name, index=False)


write_sd(pd.DataFrame([dict(harmonic_k=k, iw_phase_frac=round(float(PHI[i]), 4),
                            rw_phase_frac=round(float(PHI[j]), 4),
                            corr=round(float(M_obs[ki, i, j]), 6))
                       for ki, k in enumerate(KS) for i in range(NP_) for j in range(NP_)]),
         "fig4_phase_correspondence_source_data.csv")
write_sd(pd.DataFrame([dict(harmonic_k=k, lag_frac=round(float(LAG_PCT[li] / 100), 4),
                            mean_corr=round(float(prof[ki, li]), 6),
                            fitted_corr=round(float(prof_fit[ki, li]), 6))
                       for ki, k in enumerate(KS) for li in range(len(LAG_BINS))]),
         "fig4_phase_correspondence_lag_profile.csv")

meta = [
    ("n_sessions_IW", NS), ("n_sessions_RW", NS),
    ("n_sessions_note", "same 80 sessions in both conditions; bootstrap resamples them PAIRED"),
    ("n_channels", NC), ("n_sessions_dropped", 0), ("n_channels_dropped", 0),
    ("input_IW", str(IWF.relative_to(ROOT)).replace("\\", "/")),
    ("input_RW", str(RWF.relative_to(ROOT)).replace("\\", "/")),
    ("cycle_points", ";".join(f"{p:g}" for p in PHI) + " (the 9th, 100 %, column is a repeat of 0 % and is dropped)"),
    ("dup_column_max_abs_diff", f"{dup:.3g}"),
    ("reference_phase_rad", f"{ref:.6f} (RW map's own unified reference = panel-b polarity; applied to BOTH conditions, held fixed across bootstrap and permutation)"),
    ("cell", "Pearson r across the 60 channels between Re(c_IW(phi_IW) e^-i ref) and Re(c_RW(phi_RW) e^-i ref)"),
    ("lag_profile", f"mean of cells at cyclic lag delta = (phi_IW - phi_RW) mod 1, over the {NP_} unique phases; bin = {100 / NP_:.1f} % cycle"),
    ("model", "M_k(i,j) = A_k cos(2 pi k (phi_i - phi_j) - Delta_k) + b_k, ordinary least squares on [cos, sin, 1] over all 64 cells"),
    ("fitted_curve", "the plotted line is this closed form sampled at 401 points over one full cycle of lag; reproduce from A, Delta, b below"),
    ("Delta_convention", "Delta_pct_cycle = Delta_rad / (2 pi k) x 100, wrapped to (-50/k, +50/k]; the fitted ridge is phi_IW = phi_RW + Delta_pct/100 (mod 1/k), i.e. positive Delta = the IW map matches the RW map at an IW phase that many % LATER in the cycle than the RW phase"),
    ("colour", "k-series greys/blues (k1 #9A9A9A, k2 #0072B2, k3 #3A3A3A)"),
    ("n_boot", N_BOOT), ("n_perm", N_PERM), ("seed_bootstrap", SEED_BOOT), ("seed_permutation", SEED_PERM),
    ("bootstrap", "resample the 80 sessions with replacement (paired IW/RW), recompute the session-equal maps, matrix and fit; 95 % percentile CI, Delta percentiles taken after wrapping onto the point estimate"),
    ("permutation_null", "random permutation of the 8 RW cycle-point labels (columns of M), refit A; P = (1 + #null >= A_obs) / (n_perm + 1); tests whether the IW-RW correspondence carries harmonic-k structure at all"),
]
for ki, k in enumerate(KS):
    meta += [(f"k{k}_A", round(float(A_obs[ki]), 6)),
             (f"k{k}_A_CI95_lo", round(float(A_ci[0, ki]), 6)), (f"k{k}_A_CI95_hi", round(float(A_ci[1, ki]), 6)),
             (f"k{k}_A_perm_P", round(float(A_p[ki]), 6)),
             (f"k{k}_Delta_pct_cycle", round(float(D_pct[ki]), 4)),
             (f"k{k}_Delta_CI95_lo", round(float(D_ci[0, ki]), 4)), (f"k{k}_Delta_CI95_hi", round(float(D_ci[1, ki]), 4)),
             (f"k{k}_Delta_deg_harmonic", round(float(np.degrees(D_obs[ki])), 3)),
             (f"k{k}_b", round(float(B_obs[ki]), 6)), (f"k{k}_R2", round(float(R2_obs[ki]), 4))]
meta.append(("fit_content", "band amplitude A (spatial similarity), its offset Delta (phase lag) and the permutation test that the banding is harmonic-k structured"))
write_sd(pd.DataFrame(meta, columns=["key", "value"]), "fig4_phase_correspondence_metadata.csv")

print(f"n_sessions={NS} n_channels={NC} | dup 0%-100% max abs diff = {dup:.3g} | ref = {ref:.6f} | vmax = {VMAX:.4f}")
print(f"{'k':>2} | {'A':>6} [{'CI95':^15}] | {'Delta %':>8} [{'CI95':^15}] | {'deg':>8} {'b':>9} {'R2':>6} | permP")
for ki, k in enumerate(KS):
    print(f"{k:>2} | {A_obs[ki]:.4f} [{A_ci[0, ki]:+.4f},{A_ci[1, ki]:+.4f}] | {D_pct[ki]:+8.2f} "
          f"[{D_ci[0, ki]:+7.2f},{D_ci[1, ki]:+7.2f}] | {np.degrees(D_obs[ki]):+8.2f} {B_obs[ki]:+9.2e} "
          f"{R2_obs[ki]:6.3f} | {A_p[ki]:.4f}")
if not PANEL_MODE:
    print("wrote", OUT.relative_to(ROOT), "+", OUT_MAT.relative_to(ROOT), "+", OUT_LAG.relative_to(ROOT))

if not PANEL_MODE:
    import project_palette as _P
    PS.apply(plt)
    _W, _H, _L, _B, _BW, _BH = 107.0, 54.0, 15.0, 12.0, 89.5, 36.0
    _grid = np.linspace(LAG_PCT[0], LAG_PCT[0] + 100, 401)
    _YL = (-0.85, 1.22)
    _KEY = 2
    figh = plt.figure(figsize=(_W * PS.MM, _H * PS.MM), dpi=PS.DPI)
    axh = figh.add_axes([_L / _W, _B / _H, _BW / _W, _BH / _H])
    _HM = {}
    for ki, k in enumerate(KS):
        _c = _P.HARMONIC_COLORS[ki]
        _main = k == _KEY
        _al = 1.0 if _main else 0.55
        _yy = A_obs[ki] * np.cos(2 * np.pi * k * _grid / 100 - D_obs[ki]) + B_obs[ki]
        axh.plot(_grid, _yy, color=_c, lw=1.8 if _main else 0.9, alpha=_al, zorder=3 if _main else 2)
        axh.plot(LAG_PCT, prof[ki], ls="none", marker="o", ms=4.0 if _main else 3.0, mfc=_c, mec="white",
                 mew=0.6, alpha=_al, zorder=3.5 if _main else 2.5)
        _xp, _yp = float(D_pct[ki]), float(A_obs[ki] + B_obs[ki])
        axh.plot([_xp, _xp], [0, _yp], color=_c, lw=0.7, ls=(0, (1.5, 1.5)), alpha=_al, zorder=1.5)
        axh.plot([_xp], [_yp], ls="none", marker="D", ms=5.0 if _main else 4.0, mfc=_c, mec="white", mew=0.7,
                 zorder=5)
        axh.text(_grid[-1] + 2.0, _yy[-1], KL[k], color=_c, fontsize=PS.TICK_PT, va="center", ha="left",
                 zorder=4)
        _HM[k] = (f"Δ = {_xp:+.1f} %".replace("-", "−"), f"A = {A_obs[ki]:.2f}")
    _kk = KS.index(_KEY)
    _xp, _yp = float(D_pct[_kk]), float(A_obs[_kk] + B_obs[_kk])
    _ya = _yp + 0.10
    axh.annotate("", xy=(_xp, _ya), xytext=(0, _ya), zorder=5,
                 arrowprops=dict(arrowstyle="<->", color=PS.INK, lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0))
    _dlab = _HM[_KEY][0] + " (IW leads)" if _xp < 0 else _HM[_KEY][0] + " (RW leads)"
    axh.text(-0.8, _ya + 0.13, _dlab, ha="right", va="bottom", fontsize=PS.NOTE_PT, color=PS.INK, zorder=6)
    axh.text(_xp - 2.5, _ya, "$A$" + _HM[_KEY][1][1:], ha="right", va="center", fontsize=PS.NOTE_PT,
             color=PS.INK, zorder=6)
    axh.axhline(0, color="#555555", lw=0.6, zorder=1)
    axh.axvline(0, color="#BBBBBB", lw=0.8, ls=(0, (3, 2)), zorder=1)
    axh.set_xlim(LAG_PCT[0] - 3, LAG_PCT[0] + 100 + 10)
    axh.set_ylim(*_YL)
    axh.set_xticks([-50, -25, 0, 25, 50])
    axh.set_yticks([-0.5, 0, 0.5, 1.0])
    axh.spines["bottom"].set_bounds(LAG_PCT[0], LAG_PCT[0] + 100)
    axh.set_xlabel("IW − RW lag (% gait cycle)")
    axh.set_ylabel("Spatial correlation ($r$)")
    PS.finish(axh)
    OUT_H = ROOT / "figures_out/fig4/fig4_phase_lag_profile.png"
    figh.savefig(OUT_H, dpi=PS.DPI, facecolor="white")
    plt.close(figh)
    pd.DataFrame([dict(harmonic_k=k, lag_frac=round(float(LAG_PCT[li] / 100), 4),
                       mean_corr=round(float(prof[ki, li]), 6),
                       fitted_corr=round(float(prof_fit[ki, li]), 6))
                  for ki, k in enumerate(KS) for li in range(len(LAG_BINS))]).to_csv(
        SD / "fig4_phase_lag_profile_source_data.csv", index=False)
    _mh = [
        ("peak_marker", "filled diamond in the harmonic colour at the fitted peak nearest zero lag = (Delta_k, "
                        "A_k + b_k) for each harmonic, with a thin dotted vertical guide down to r = 0; Delta_k is "
                        "wrapped to +-50/k % so that peak is the one at Delta_k"),
        ("offset_indicator", "k2 only: double-headed arrow from zero lag to the k2 peak lag, drawn 0.10 r above the "
                             "peak, labelled '%s (IW leads)'; '%s' printed left of the k2 peak; k1 / k3 Delta "
                             "labels not printed" % (_HM[2][0], _HM[2][1])),
        ("annotation_values", "formatted in code from the fit: " + "; ".join(
            f"k{k} {_HM[k][0]}, {_HM[k][1]}" for k in KS) + " (only the k2 values are printed); no p values in "
            "the panel"),
        ("fit_scope", "A and Delta here are the COHORT-level cosine fit of the session-equal correspondence matrix "
                      "(80 sessions pooled); Fig. 4d (phase offset) and Fig. 4f (correspondence amplitude) show "
                      "PER-SESSION fits, a different estimator (per-session values in "
                      "fig4_phase_delay_persession_metadata.csv / fig4_correspondence_amplitude_metadata.csv)"),
        ("producer", "03_figures/fig4_phase_correspondence.py (Fig. 4h)"),
        ("output", "figures_out/fig4/fig4_phase_lag_profile.png"),
        ("canvas", f"{_W:g} x {_H:g} mm at {PS.DPI} dpi; data box {_BW:g} x {_BH:g} mm at left {_L:g} / bottom "
                   f"{_B:g} mm (two single bottom-row slots of 53.5 x 54 mm)"),
        ("source_data_note", "rows identical to fig4_phase_correspondence_lag_profile.csv (same DataFrame); the fit "
                             "parameters, CIs and permutation p are in fig4_phase_correspondence_metadata.csv and "
                             "repeated below"),
        ("n_sessions", NS), ("n_channels", NC),
        ("points", f"mean of the {NP_} x {NP_} correspondence-matrix cells at each cyclic lag (phi_IW - phi_RW) mod 1; "
                   f"bin = {100 / NP_:.1f} % cycle; columns lag_frac / mean_corr"),
        ("line", "fitted cosine A_k cos(2 pi k lag - Delta_k) + b_k sampled at 401 points over one full cycle of "
                 "lag (-50 .. +50 % gait cycle); fitted_corr = the same closed form at the 8 measured lags"),
        ("Delta_convention", "lag = phi_IW - phi_RW; negative Delta = the IW map matches the RW map at an EARLIER IW "
                             "phase, i.e. imagined walking leads"),
        ("colour", "project_palette.HARMONIC_COLORS (k1 %s, k2 %s, k3 %s), as Fig. 4d-f" % _P.HARMONIC_COLORS),
        ("reference_lines", "dashed grey vertical = zero lag; solid grey horizontal = r 0"),
    ]
    for ki, k in enumerate(KS):
        _mh += [(f"k{k}_A", round(float(A_obs[ki]), 6)),
                (f"k{k}_A_CI95", f"[{A_ci[0, ki]:.6f}, {A_ci[1, ki]:.6f}]"),
                (f"k{k}_A_perm_p", round(float(A_p[ki]), 6)),
                (f"k{k}_Delta_pct_cycle", round(float(D_pct[ki]), 4)),
                (f"k{k}_Delta_CI95", f"[{D_ci[0, ki]:.4f}, {D_ci[1, ki]:.4f}]"),
                (f"k{k}_b", round(float(B_obs[ki]), 6))]
    pd.DataFrame(_mh, columns=["key", "value"]).to_csv(SD / "fig4_phase_lag_profile_metadata.csv", index=False)
    print("wrote", OUT_H.relative_to(ROOT), "+ fig4_phase_lag_profile_{source_data,metadata}.csv")

