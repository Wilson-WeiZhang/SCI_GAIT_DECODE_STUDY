"""Figure 4 — Shared locomotor representation across imagined and real walking

per-session IW × RW gait-phase correspondence AMPLITUDE A_k (k1 / k2 / k3)

Reads:
  04_source_data/fig4/fig4_phase_delay_persession_source_data.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS

SRC = ROOT / "04_source_data/fig4/fig4_phase_delay_persession_source_data.csv"
PNG = ROOT / "figures_out/fig4/fig4_correspondence_amplitude.png"
SD = ROOT / "fig4_correspondence_amplitude_source_data.csv (not distributed)"
MD = ROOT / "04_source_data/fig4/fig4_correspondence_amplitude_metadata.csv"
STATS = ROOT / "figures_out/fig4/fig4_correspondence_amplitude_stats.txt"
KS = (1, 2, 3)
KLAB = {1: r"$\mathrm{k}_1$", 2: r"$\mathrm{k}_2$", 3: r"$\mathrm{k}_3$"}

d = pd.read_csv(SRC)
assert set(d.harmonic_k) == set(KS) and (d.groupby("harmonic_k").size() == 80).all(), \
    "expected 80 sessions per harmonic"
V = {k: d.loc[d.harmonic_k == k, "A"].to_numpy(float) for k in KS}

pv = {k: float(stats.wilcoxon(V[k], V[2]).pvalue) for k in (1, 3)}
pv[2] = 1.0
_order = sorted(KS, key=lambda k: pv[k])
q, _run = {}, 1.0
for _i, k in enumerate(reversed(_order)):
    _run = min(_run, pv[k] * 3 / (3 - _i))
    q[k] = _run
mark = lambda v: "***" if v < 1e-3 else "**" if v < 1e-2 else "*" if v < .05 else "ns"

PS.apply(plt)
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic"})
fig, ax = PS.panel_axes(plt, left_mm=14.5, bottom_mm=10.0, top_mm=11.5)
_LW_BOX, _LW_WHI, _LW_MED, _LW_BRK, _S_PT = ((1.2, 1.0, 1.6, 1.1, 4))
rj = np.random.default_rng(7)
for i, k in enumerate(KS):
    v = V[k]
    import project_palette as _P
    _kc = _P.HARMONIC_COLORS[i]
    ax.scatter(i + rj.uniform(-.15, .15, v.size), v, s=_S_PT, c=_kc, alpha=.55,
               edgecolors="none", zorder=3)
    bp = ax.boxplot([v], positions=[i], widths=.44, whis=(10, 90), showfliers=False,
                    patch_artist=True, zorder=4)
    bp["boxes"][0].set(facecolor=(*__import__("matplotlib.colors").colors.to_rgb(_kc), 0.32), edgecolor=_kc, linewidth=_LW_BOX)
    for el in ("whiskers", "caps"):
        for a in bp[el]:
            a.set(color=_kc, linewidth=_LW_WHI)
    bp["medians"][0].set(color=_kc, linewidth=_LW_MED)
for lvl, (i, j, K) in enumerate(((0, 1, 1), (1, 2, 3))):
    yb, tk = 1.015 + 0.105 * lvl, 0.022
    ax.plot([i, i, j, j], [yb, yb + tk, yb + tk, yb], color=PS.INK, lw=_LW_BRK,
            solid_capstyle="butt", clip_on=False, zorder=6, transform=ax.get_xaxis_transform())
    ax.text((i + j) / 2, yb + tk + 0.012, mark(q[K]), ha="center", va="bottom",
            fontsize=(PS.STAR_PT),
            color=PS.INK, clip_on=False, zorder=6,
            transform=ax.get_xaxis_transform())
ax.set_xlim(-.55, 2.55)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([KLAB[k] for k in KS], fontsize=PS.TICK_PT)
ax.set_ylabel("IW × RW correspondence amplitude")
PS.finish(ax)
fig.savefig(PNG, dpi=PS.DPI, facecolor="white")
plt.close(fig)

d[["session", "participant", "harmonic_k", "A"]].rename(columns={"A": "correspondence_amplitude"}
                                                        ).to_csv(SD, index=False,
                                                                 float_format="%.6f")
meta = [
    ("panel", "Fig. 4f - per-session IW x RW correspondence amplitude A_k"),
    ("n_sessions", 80), ("n_participants", 42),
    ("model", "M_k(i,j) = A_k cos(2 pi k (phi_i - phi_j) - Delta_k) + b_k, OLS over 64 cells; "
              "the SAME fit whose Delta_k is the phase-delay panel"),
    ("input", SRC.name),
]
for k in KS:
    meta.append((f"k{k}_median_IQR", f"{np.median(V[k]):.4f} "
                                     f"[{np.percentile(V[k], 25):.4f}, {np.percentile(V[k], 75):.4f}]"))
for k in (1, 3):
    meta.append((f"k{k}_vs_k2_wilcoxon", f"P = {pv[k]:.3g}, BH q = {q[k]:.3g} ({mark(q[k])})"))
meta += [
    ("multiplicity", "BH-FDR, family of 3, matching every other marked panel in Fig. 4"),
]
pd.DataFrame(meta, columns=["key", "value"]).to_csv(MD, index=False)
PS.write_stats(STATS, [f"k{k}: median {np.median(V[k]):.3f} "
                       f"IQR {np.percentile(V[k], 25):.3f}-{np.percentile(V[k], 75):.3f} n=80"
                       for k in KS]
               + [f"k{k} vs k2 (paired Wilcoxon): P = {pv[k]:.2g}, BH q = {q[k]:.2g} ({mark(q[k])})"
                  for k in (1, 3)]
               + ["brackets are paired contrasts against k2"])
for k in KS:
    print(f"k{k}: median {np.median(V[k]):.4f}")
print(f"wrote {PNG.name} + {SD.name} + {MD.name}")
