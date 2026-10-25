"""Figure 3 — Gait-phase locking of the imagined-walking representation

panels c + f — locking->decoding specificity forest + k1 carrier scatter

Renders one panel or table; it performs no
preprocessing and does not fit the decoder. It fits a least-squares regression
for the statistics reported with that panel or table.
"""

from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
import analysis_config as ec

ec.require("main", "main")
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 1.3})
ROOT = Path(__file__).resolve().parents[1]

SD = ROOT / "04_source_data" / "fig3"
OUT = ec.scoped(ROOT / "figures_out" / "fig3" / "fig3_d_dissociation_forest.png")
SIG = "#00568A"; GREY = "#9aa0a6"
import project_palette as _P
RESP, NON, RESP_EDGE = _P.RESPONDER_FACE, _P.NON_RESPONDER_FACE, _P.IW_HUE
K = {1: SIG, 2: SIG, 3: SIG}
def p_mathtext(p):
    """Format a p value for on-figure display as mathtext: 1.1 x 10^-27, or P < 0.001 style."""
    import math
    if p == 0 or not math.isfinite(p):
        return r"$<10^{-300}$"
    e = int(math.floor(math.log10(abs(p))))
    m = p / 10 ** e
    if e >= -2:
        return f"{p:.3f}".rstrip("0")
    return rf"${m:.1f}\times10^{{{e}}}$"


rng = np.random.default_rng(0)

m = pd.read_csv(ec.sd("fig3_master_source_data.csv"))
resp = pd.read_csv(ec.sd("fig3_def_resnonres_source_data.csv"))[["session", "resp"]]
m = m.merge(resp, on="session", how="left"); m["resp"] = m.resp.fillna(False).astype(bool)


def boot_rho(x, y, n=1000):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    r0, p = spearmanr(x, y); i = np.arange(len(x))
    rs = [spearmanr(x[s], y[s])[0] for s in (rng.choice(i, len(i), True) for _ in range(n))]
    rs = np.array(rs); return r0, np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5), p, len(x)


def partial_rho(x, y, covs, n=1000):
    d = pd.DataFrame({"x": x, "y": y}); [d.__setitem__(f"c{i}", c) for i, c in enumerate(covs)]; d = d.dropna()
    R = d.apply(rankdata); Z = np.column_stack([np.ones(len(R))] + [R[f"c{i}"] for i in range(len(covs))])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    rx, ry = res(R.x.values), res(R.y.values); r0 = np.corrcoef(rx, ry)[0, 1]; i = np.arange(len(rx))
    rs = [np.corrcoef(rx[s], ry[s])[0, 1] for s in (rng.choice(i, len(i), True) for _ in range(n))]
    from scipy.stats import t as tdist
    df = len(rx) - 2 - len(covs); tt = r0 * np.sqrt(df / (1 - r0 ** 2)); p = 2 * tdist.sf(abs(tt), df)
    return r0, np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5), p, len(rx)


def bh(ps):
    ps = np.array(ps); o = np.argsort(ps); r = np.empty(len(ps), int); r[o] = np.arange(1, len(ps) + 1)
    return np.minimum(1.0, ps * len(ps) / r)


zr = {k: boot_rho(m[f"excess_k{k}"], m.r_reference) for k in (1, 2, 3)}
zc = {k: boot_rho(m[f"rest_k{k}"], m.r_reference) for k in (1, 2, 3)}
er = boot_rho(m.mubeta_erd_mean, m.r_reference)
fam = [(f"k{k}_taskevoked", f"$k_{k}$ task-evoked", *zr[k]) for k in (1, 2, 3)] + \
      [("mubeta_erd", "mu/$\\beta$ ERD", *er)]
q3 = bh([f[5] for f in fam[:3]])
q4 = list(q3) + [np.nan]
tv = list(q3) + [fam[3][5]]

rows, stat = [], []
for i, (key, lab, r0, lo, hi, p, n) in enumerate(fam):
    sig = tv[i] < 0.05
    rows.append((lab, r0, lo, hi, sig))
    stat.append([key, r0, lo, hi, p, q4[i], n, int(sig), 1])

pr = partial_rho(m.excess_k1, m.r_reference, [m.excess_k2, m.excess_k3])
stat.append(["k1_partial_given_k2k3", *pr[:4], np.nan, pr[4], int(pr[1] > 0 or pr[2] < 0), 0])
for k in (1, 2, 3):
    r0, lo, hi, p, n = zc[k]
    stat.append([f"k{k}_restcontrol", r0, lo, hi, p, np.nan, n, int(lo > 0 or hi < 0), 0])

rows = rows[::-1]

OUT_E = ec.scoped(ROOT / "figures_out" / "fig3" / "fig3_e_carrier_scatter.png")

MM  = 1/25.4
FS  = 1.2
INK, MID, PALE = "#1A1A1A", "#666666", "#B3B3B3"
plt.rcParams.update({"font.family": "Arial", "font.size": 7*FS, "axes.linewidth": 1.4,
                     "xtick.major.width": 1.4, "ytick.major.width": 1.4,
                     "xtick.major.size": 3.0, "ytick.major.size": 3.0, "pdf.fonttype": 42})
E_LAB_FS = 7*FS
def _stars(q): return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"
ROWLAB = ["$\\mathrm{k}_1$", "$\\mathrm{k}_2$", "$\\mathrm{k}_3$", "ERD"][::-1]
QREV   = list(tv)[::-1]

_CI_LW, _CAP, _DOT_MS, _DOT_MEW, _ZERO = ((1.6, .16, 5.0, 1.4, (1.1, (0, (2, 2)))))
figf, axf = plt.subplots(figsize=(52*MM, 52*MM), dpi=400)
axf.axvline(0, color=PALE, lw=_ZERO[0], ls=_ZERO[1], zorder=0)
for i, (lab, r0, lo, hi, sig) in enumerate(rows):
    col = INK if sig else MID
    axf.plot([lo, hi], [i, i], color=col, lw=_CI_LW, solid_capstyle="butt", zorder=2)
    for x in (lo, hi):
        axf.plot([x, x], [i - _CAP, i + _CAP], color=col, lw=_CI_LW, solid_capstyle="butt", zorder=2)
    axf.plot(r0, i, "o", ms=_DOT_MS, mfc=col if sig else "white", mec=col, mew=_DOT_MEW, zorder=3)
    _st = _stars(QREV[i])
    axf.text(1.04, i, _st, transform=axf.get_yaxis_transform(),
             ha="left", va="center",
             fontsize=10.5*FS if _st != "ns" else 10.5*FS*0.65, color=INK)
axf.set_yticks(range(len(rows))); axf.set_yticklabels(ROWLAB, fontsize=8*FS)
axf.set_xlim(-0.22, 1.0); axf.set_ylim(-0.55, len(rows) - 0.45); axf.set_xticks([0, 0.5, 1.0])
axf.spines[["top", "right"]].set_visible(False)
axf.set_xlabel("Spearman $\\rho$ with\ndecoding accuracy", fontsize=7*FS, labelpad=1.5)
axf.set_box_aspect(1)
figf.savefig(OUT, dpi=400, bbox_inches="tight", facecolor="white", pad_inches=0.04)
plt.close(figf)

figs, axs = plt.subplots(figsize=(52*MM, 52*MM), dpi=400)
_FIT_LW, _FIT_LS, _PT_S, _PT_LW = ((1.3, (0, (4, 2)), 14, .5))
x, y, rp = m.excess_k1.values, m.r_reference.values, m.resp.values
ok = np.isfinite(x) & np.isfinite(y); x, y, rp = x[ok], y[ok], rp[ok]
xs = np.linspace(x.min(), x.max(), 60); idx = np.arange(len(x))
lines = np.array([np.polyval(np.polyfit(x[s], y[s], 1), xs)
                  for s in (rng.choice(idx, len(idx), True) for _ in range(1000))])
axs.fill_between(xs, np.percentile(lines, 2.5, 0), np.percentile(lines, 97.5, 0),
                 color="#9A9A9A", alpha=0.20, lw=0, zorder=1)
axs.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=INK, lw=_FIT_LW, ls=_FIT_LS, zorder=2)
axs.scatter(x[~rp], y[~rp], s=_PT_S, facecolor=NON, edgecolor=RESP_EDGE, lw=_PT_LW, zorder=3)
axs.scatter(x[rp],  y[rp],  s=_PT_S, facecolor=RESP, edgecolor=RESP_EDGE, lw=_PT_LW, zorder=4)
r0, lo, hi, p_k1, _ = zr[1]
axs.set_xlabel("Rest-corrected locking |c| ($\\mathrm{k}_1$)",
               fontsize=7*FS, labelpad=1.5)
axs.set_ylabel("Decoding accuracy $r$", fontsize=7*FS, labelpad=2)
axs.set_xticks([0, .10, .20]); axs.set_yticks([.2, .4, .6, .8])
axs.spines[["top", "right"]].set_visible(False)
axs.set_box_aspect(1)
figs.savefig(OUT_E, dpi=400, bbox_inches="tight", facecolor="white", pad_inches=0.04)
plt.close(figs)


def e_legend_strip(fontsize=E_LAB_FS, ncol=1, name="fig3_fh_responder_legend"):
    """Stand-alone legend SHARED by the two responder-coloured scatters, panel f (this file) and
    panel h (fig3_h_selfreport_forest.py): Responder / Non-responder, IW green pair (dark face / light face,
    dark edge on both)."""
    figl, axl = plt.subplots(figsize=(2.2, 0.9))
    axl.axis("off")
    hh = [axl.scatter([], [], s=14, facecolor=c, edgecolor=RESP_EDGE, linewidths=.5)
          for c in (RESP, NON)]
    axl.legend(hh, ["Responder", "Non-responder"], frameon=False, fontsize=fontsize,
               loc="center", ncol=ncol, handletextpad=0.2, borderpad=0, borderaxespad=0,
               labelspacing=0.45)
    out = OUT_E.parent / f"{name}.png"
    figl.savefig(out, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(figl)
    print(f"wrote {out} (shared f + h responder legend, stand-alone)")


e_legend_strip()

pd.DataFrame(stat, columns=["predictor", "rho", "ci_lo", "ci_hi", "p_spearman", "q_bh_fdr",
                            "n", "sig", "in_panel"]).to_csv(
    ec.sd("fig3_d_forest_stats.csv"), index=False)
m[["session", "subj_int", "excess_k1", "excess_k2", "excess_k3",
   "rest_k1", "rest_k2", "rest_k3", "r_reference", "resp"]].rename(
    columns={"subj_int": "subject", "resp": "responder"}).to_csv(ec.sd("fig3_d_scatter_points.csv"), index=False)

pd.DataFrame([
    ("panel", "Fig. 3 panel c - specificity forest rho(decoding r, |c|) for k1/k2/k3 and "
              "mu/beta ERD; the same script also writes the panel f carrier scatter"),
    ("unit_of_analysis", "session"),
    ("n_sessions", len(m)),
    ("n_participants", int(m.subj_int.nunique())),
    ("fdr_family", "k1-k3"),
    ("erd_p", "uncorrected"),
    ("ci_method", "1000-resample session bootstrap, rng seed 0"),
    ("colour_semantic", "blue #00568A = BH-FDR q<0.05 (k1-k3) or uncorrected p<0.05 (ERD); grey = not significant"),
    ("input", f"{ec.sd('fig3_master_source_data.csv').relative_to(ROOT).as_posix()} + fig3_def_resnonres_source_data.csv"),
], columns=["key", "value"]).sort_values("key", kind="stable").to_csv(
    ec.sd("fig3_d_forest_metadata.csv"), index=False)

print(f"wrote {OUT} (panel c forest) + {OUT_E} (panel f scatter)")
print(f"  scatter stats: rho = {r0:.2f} [{lo:.2f}, {hi:.2f}], "
      f"P = {p_mathtext(p_k1)}  (exact rho {r0:.6f}, P {p_k1:.4g})")
print("  PANEL (BH-FDR family k1-k3; ERD uncorrected p):")
for s in stat:
    if s[8]:
        print("    %-18s rho=%+.3f [%+.3f,%+.3f]  p=%.4g  q=%.4g %s" % (s[0], s[1], s[2], s[3], s[4], s[5], "*" if s[7] else ""))
print("  SI ONLY (not drawn, not in the FDR family):")
for s in stat:
    if not s[8]:
        print("    %-18s rho=%+.3f [%+.3f,%+.3f]  p=%.3g" % (s[0], s[1], s[2], s[3], s[4]))
