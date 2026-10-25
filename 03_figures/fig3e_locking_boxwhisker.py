#!/usr/bin/env python3
"""Figure 3 — Gait-phase locking of the imagined-walking representation

panel e — per-session gait-phase locking |c| at f0, imagined walking vs trial-matched rest

Reads:
  04_source_data/fig3/fig3_def_resnonres_source_data.csv
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
BC   = ROOT / "data/decoder_output/persession_bootstrap_c_main_b1000.csv"
DEF  = ROOT / "04_source_data/fig3/fig3_def_resnonres_source_data.csv"
OUT  = ROOT / "figures_out/fig3/fig3e_locking_boxwhisker.png"
SD   = ROOT / "04_source_data/fig3/fig3e_locking_boxwhisker_source_data.csv"
K = 1
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import project_palette as _P
BLUE, GREY, REST = _P.RESPONDER_FACE, _P.NON_RESPONDER_FACE, _P.REST_LIGHT
EDGE_IW, EDGE_REST = _P.IW_HUE, _P.REST_HUE
plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.linewidth": 2.2})

b = pd.read_csv(BC); b = b[b.harmonic_k == K]
d = pd.read_csv(DEF)[["session", "resp"]]
d["resp"] = d.resp.astype(str).str.strip().isin(["True", "TRUE", "1"])
resp = dict(zip(d.session, d.resp))

P = [10, 25, 50, 75, 90]
rows = []
for s, g in b.groupby("session"):
    r = {"session": s, "responder": bool(resp[s]),
         "excess_median": float(np.median(g.c_mi.to_numpy() - g.c_rest.to_numpy()))}
    for cond, col in (("IW", "c_mi"), ("rest", "c_rest")):
        q = np.percentile(g[col].to_numpy(), P)
        r |= {f"{cond}_p{p}": float(v) for p, v in zip(P, q)}
    rows.append(r)
SORT_BY = "IW_p50"
t = pd.DataFrame(rows).sort_values(SORT_BY, ascending=False).reset_index(drop=True)
t.to_csv(SD, index=False, float_format="%.6f")
print(f"{len(t)} sessions | responders {int(t.responder.sum())} | "
      f"excess_median {t.excess_median.min():.4f}..{t.excess_median.max():.4f}")

fig, ax = plt.subplots(figsize=(14.5, 3.4))
BW, OFF = 0.40, 0.21
EDGE, CAP = (1.15, 0.24)
for i, r in t.iterrows():
    for cond, sign, face in (("IW", -1, BLUE if r.responder else GREY), ("rest", +1, REST)):
        ek = EDGE_IW if cond == "IW" else EDGE_REST
        p10, p25, p50, p75, p90 = (r[f"{cond}_p{p}"] for p in P)
        x = i + sign * OFF
        ax.plot([x, x], [p10, p25], color=ek, lw=EDGE, solid_capstyle="butt", zorder=2)
        ax.plot([x, x], [p75, p90], color=ek, lw=EDGE, solid_capstyle="butt", zorder=2)
        for y in (p10, p90):
            ax.plot([x - CAP / 2, x + CAP / 2], [y, y], color=ek, lw=EDGE, zorder=2)
        ax.add_patch(Rectangle((x - BW / 2, p25), BW, p75 - p25, facecolor=face,
                               edgecolor=ek, linewidth=EDGE, zorder=3))
        ax.plot([x - BW / 2, x + BW / 2], [p50, p50], color=ek, lw=EDGE * 1.4, zorder=4)

ax.set_xlim(-0.8, len(t) - 0.2)
ax.set_xticks([])
ax.tick_params(axis="x", bottom=False)
LAB_FS = 17
YLAB = "Gait-phase locking |c| ($k_1$)"
ax.set_ylabel(YLAB, fontsize=LAB_FS, labelpad=6)
XLAB = f"{len(t)} sessions, ranked by median imagined-walking |c|"
ax.set_xlabel(XLAB, fontsize=LAB_FS, labelpad=4)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", direction="out", length=5, width=2.2, labelsize=14)
fig.subplots_adjust(left=0.082, right=0.995, top=0.985, bottom=0.125)
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)


def legend_strip(n_resp, n_non, fontsize=LAB_FS, ncol=1, name="fig3e_locking_boxwhisker_legend"):
    """Stand-alone legend for panel e, transparent, 600 dpi (the panel itself is 200)."""
    figl, axl = plt.subplots(figsize=(4.2, 1.5))
    axl.axis("off")
    hh = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=e, linewidth=EDGE)
          for c, e in ((BLUE, EDGE_IW), (GREY, EDGE_IW), (REST, EDGE_REST))]
    axl.legend(hh, [f"Imagined walking, responder (n={n_resp})",
                    f"Imagined walking, non-responder (n={n_non})",
                    "Rest control"],
               frameon=False, fontsize=fontsize, loc="center", ncol=ncol,
               handlelength=1.4, borderaxespad=0, borderpad=0, labelspacing=0.35)
    out = OUT.parent / f"{name}.png"
    figl.savefig(out, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(figl)
    print("wrote", out)


legend_strip(int(t.responder.sum()), int((~t.responder).sum()))
print("wrote", SD)


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


_mb = _rew.fullmatch(r"persession_bootstrap_c_(.+)_b(\d+)\.csv", BC.name)
if _mb is None:
    raise SystemExit(f"INPUT CHECK: no decode tag in {BC.name}; refusing to stamp an build_id.")
_write_meta(ROOT / "04_source_data/fig3/fig3e_locking_boxwhisker_metadata.csv", [
    ("figure", "Fig. 3 panel e - per-session gait-phase locking |c| at f0, imagined walking "
               "vs trial-matched rest (box = IQR, centre = median, whiskers = 10th-90th "
               "percentile of the session's bootstrap sampling distribution)"),
    ("producer", "03_figures/fig3e_locking_boxwhisker.py"),
    ("cohort", f"healthy; {t.session.nunique()} sessions plotted"),
    ("input_bootstrap_c", str(BC.relative_to(ROOT)).replace("\\", "/")),
    ("input_responder_split", str(DEF.relative_to(ROOT)).replace("\\", "/")),
    ("harmonic_k", K),
    ("n_boot_in_input", int(_mb.group(2))),
    ("n_sessions", int(t.session.nunique())),
    ("n_responder", int(t.responder.sum())),
    ("n_non_responder", int((~t.responder).sum())),
    ("distribution_plotted", "BOOTSTRAP sampling distribution of each session's inter-trial "
                             "locking c - NOT a distribution over trials; the same trial "
                             "resample is applied to each MI trial's yoked rest window, so IW "
                             "and rest are paired"),
    ("session_order", {"IW_p50": "sorted by the median of the imagined-walking bootstrap distribution "
                                  "(IW_p50), descending",
                       "excess_median": "sorted by the paired bootstrap excess (median c_mi - c_rest), "
                                        "descending"}[SORT_BY]),
    ("outputs", "figures_out/fig3/fig3e_locking_boxwhisker.png; "
                "figures_out/fig3/fig3e_locking_boxwhisker_legend.png; "
                "04_source_data/fig3/fig3e_locking_boxwhisker_source_data.csv"),
])
