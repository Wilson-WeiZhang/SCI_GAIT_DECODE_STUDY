"""Figure 3 — Gait-phase locking of the imagined-walking representation

responder / non-responder DEFINITION panel: per-session locking vs its OWN

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
from matplotlib.gridspec import GridSpec
import analysis_config as ec

ec.require("main", "main")
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data" / "fig3"
NULLCSV = ROOT / "data" / "decoder_output" / "persession_rest_null_main.csv"
ec.check_tag(NULLCSV.name, "neural", "resting |c| null")
OUT = ec.scoped(ROOT / "figures_out" / "fig3" / "fig3_def_resnonres.png")
ZTHR = 1.645

C_RESP = "#1e6fd9"; C_NON = "#9aa0a6"; C_IRON = "#f34142"

nu = pd.read_csv(NULLCSV)
nu = nu[nu.harmonic_k == 1].set_index("session")[["rest_null_mean", "rest_null_sd", "rest_null_p95"]]
M = pd.read_csv(ec.sd("fig3_master_source_data.csv")).set_index("session")
m = M.join(nu, how="inner")
m["z"] = (m.mi_k1 - m.rest_null_mean) / m.rest_null_sd
m["resp"] = m.mi_k1 > m.rest_null_p95
m["ironclad"] = (~m.resp) & (m.excess_k1 <= 0)
m["borderline"] = (m.z - ZTHR).abs() < 0.5
ZCAP = 8.0
m["offscale"] = m.z > ZCAP
m["z_disp"] = m.z.clip(upper=ZCAP - 0.15)
m = m.sort_values("z").reset_index()
nresp, nnon, niron, nbord = int(m.resp.sum()), int((~m.resp).sum()), int(m.ironclad.sum()), int(m.borderline.sum())
noff = int(m.offscale.sum())
from scipy.stats import norm
_p = norm.sf(m.z.to_numpy()); _o = np.argsort(_p); _ok = _p[_o] <= 0.05 * np.arange(1, len(_p) + 1) / len(_p)
_fdr = np.zeros(len(_p), bool); _fdr[_o[:(np.nonzero(_ok)[0].max() + 1) if _ok.any() else 0]] = True
nresp_fdr, fdr_same = int(_fdr.sum()), bool((_fdr == m.resp.to_numpy()).all())
y = np.arange(len(m))

fig = plt.figure(figsize=(4.8, 5.6), dpi=300)
fig.patch.set_facecolor("white")
gs = GridSpec(2, 1, height_ratios=[1, 7], hspace=0.06, figure=fig)
xmax = ZCAP
xmin = min(-2.5, float(m.z.min()) * 1.08)

axn = fig.add_subplot(gs[0])
xx = np.linspace(xmin, xmax, 400)
dens = np.exp(-0.5 * xx**2) / np.sqrt(2 * np.pi)
axn.plot(xx, dens, color="0.45", lw=1.1)
axn.fill_between(xx, 0, dens, where=(xx <= ZTHR), color="0.85", zorder=0)
axn.fill_between(xx, 0, dens, where=(xx >= ZTHR), color=C_RESP, alpha=.18, zorder=0)
axn.axvline(ZTHR, color="0.35", lw=1.2, ls="--")
axn.text(ZTHR, dens.max() * 1.05, "p<0.05", ha="center", va="bottom", fontsize=7.5, color="0.3")
axn.set_xlim(xmin, xmax); axn.set_ylim(0, dens.max() * 1.35)
axn.axis("off")
axn.text(0.0, dens.max() * 1.05, "resting null", ha="center", va="bottom", fontsize=7.5, color="0.45")

ax = fig.add_subplot(gs[1], sharex=axn)
ax.axvspan(xmin, ZTHR, color="0.93", zorder=0)
ax.axvline(0, color="0.7", lw=0.8, zorder=1)
ax.axvline(ZTHR, color="0.35", lw=1.2, ls="--", zorder=2)
for yi, row in zip(y, m.itertuples()):
    c = C_RESP if row.resp else C_NON
    ax.plot([0, row.z_disp], [yi, yi], color=c, lw=1.0, alpha=.5, zorder=2)
    mk = ">" if row.offscale else "o"
    ax.scatter(row.z_disp, yi, s=(30 if row.offscale else 22), color=c, marker=mk,
               edgecolors="white", linewidths=.4, zorder=3)
ax.set_xlim(xmin, xmax); ax.set_ylim(-1, len(m)); ax.set_yticks([])
ax.set_xlabel("imagined-walking $|c|_{k_1}$ above own resting chance  (z vs resting null)")
ax.set_ylabel("sessions (sorted)")
ax.text(.97, .05, f"responder  n={nresp}", transform=ax.transAxes, ha="right", va="bottom",
        color=C_RESP, fontsize=10, fontweight="bold")
ax.text(.035, .80, f"non-responder\nn={nnon}", transform=ax.transAxes, ha="left", va="top",
        color="#5f6368", fontsize=10, fontweight="bold")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.06)
plt.close(fig)

out = m[["session", "mi_k1", "rest_null_mean", "rest_null_sd", "rest_null_p95", "z",
         "excess_k1", "r_reference", "resp", "ironclad", "borderline"]]
out.to_csv(ec.sd("fig3_def_resnonres_source_data.csv"), index=False)
pd.DataFrame([
    ("n_sessions", len(m)), ("n_responder", nresp), ("n_nonresponder", nnon),
    ("n_responder_fdr", nresp_fdr), ("fdr_set_equals_percentile_cut", fdr_same),
    ("fdr_definition", "Benjamini-Hochberg at 0.05 over the sessions on the one-sided normal p of z"),
    ("definition", "responder = MI |c|(k1) > own resting null 95th pct (200-resample); within-session one-sided p<0.05"),
    ("z", "(MI |c| - rest_null_mean)/rest_null_sd; null ~N(0,1), cut at 1.645"),
    ("z_range", f"{m.z.min():.2f} .. {m.z.max():.2f}"),
], columns=["key", "value"]).sort_values("key", kind="stable").to_csv(
    ec.sd("fig3_def_resnonres_metadata.csv"), index=False)
print(f"wrote {OUT}")
print(f"responder {nresp} / non {nnon}   z {m.z.min():.2f}..{m.z.max():.2f}")
