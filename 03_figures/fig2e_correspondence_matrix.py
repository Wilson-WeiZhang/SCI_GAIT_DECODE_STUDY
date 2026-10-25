#!/usr/bin/env python
"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

correspondence between decoded and ground-truth joint trajectories

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import wilcoxon
from project_goni import polarity_vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS
import analysis_config as _ec
_ec.require("main", "main")
NPZ_IW = _ec.path("h_cross_cohort_iw")
NPZ_RE = _ec.path("h_cross_cohort_rest")
OUT_PNG = _ec.scoped(ROOT / "figures_out/fig2/fig2e_correspondence.png")
SD = ROOT / "04_source_data/fig2"

plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "axes.linewidth": 1.9,
                     "xtick.major.width": 1.9, "ytick.major.width": 1.9,
                     "xtick.major.size": 3.0, "ytick.major.size": 3.0,
                     "font.size": 12.9, "xtick.labelsize": 11.3, "ytick.labelsize": 11.3,
                     "pdf.fonttype": 42})
NJ = 6
LAB = ["LHip", "LKne", "LAnk", "RHip", "RKne", "RAnk"]
S = np.outer(polarity_vector(LAB), polarity_vector(LAB))


def apa_p(p):
    """APA: no leading zero, three decimals, below .001 -> 'p < .001'."""
    if p < 0.001:
        return "p < .001"
    return "p = " + f"{p:.3f}".lstrip("0")


zi = np.load(NPZ_IW, allow_pickle=True)
zr = np.load(NPZ_RE, allow_pickle=True)
assert [str(s) for s in zi["sessions"]] == [str(s) for s in zr["sessions"]]
Cobs = np.nanmean(zi["obs_mat"], axis=0) * S
Ciw = np.nanmean(zi["cross_mat"], axis=0) * S
Cre = np.nanmean(zr["cross_mat"], axis=0) * S

ps_iw = np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan for r in zi["diag"]])
ps_re = np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan for r in zr["diag"]])
ok = np.isfinite(ps_iw) & np.isfinite(ps_re)
a, b = ps_iw[ok], ps_re[ok]
w = wilcoxon(a, b)
w0 = wilcoxon(b)
med = lambda x: (np.median(x), np.percentile(x, 25), np.percentile(x, 75))
mi, qi1, qi3 = med(a)
mr, qr1, qr3 = med(b)

mats = [(Cobs, "Reference", True),
        (Ciw, "Decoded from imagined walking", True),
        (Cre, "Decoded from resting EEG", True)]

fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.75))
fig.subplots_adjust(left=0.125, right=0.915, bottom=0.28, top=0.85, wspace=0.0937)
for k, (M, t, outline) in enumerate(mats):
    ax = axes[k]
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    _D = [PS.jlab(x) for x in LAB]
    ax.set_xticklabels(_D, rotation=45, ha="right", fontsize=9.7)
    ax.set_yticklabels(_D if k == 0 else [], fontsize=9.7)
    ax.set_title(t, fontsize=9.7)
    if k == 0:
        ax.set_ylabel("Reference", fontsize=11.3, labelpad=2)
    if outline:
        for i in range(6):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#1a1a1a", lw=1.9))
cax = None
for ax in axes:
    c = make_axes_locatable(ax).append_axes("right", size="6%", pad=0.08)
    if ax is axes[2]:
        cax = c
    else:
        c.set_axis_off()
cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
cb.set_label("Correlation ($r$)", fontsize=11.3, labelpad=4)
cb.ax.tick_params(labelsize=11.3)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=300)
plt.close(fig)

SD.mkdir(parents=True, exist_ok=True)
with _ec.sd("fig2e_correspondence_source_data.csv").open("w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["matrix", "row_reference_joint", "col_joint", "r"])
    for name, M, collab in (("reference", Cobs, "reference"),
                            ("iw_decoded", Ciw, "decoded"),
                            ("rest_decoded", Cre, "decoded")):
        for i in range(NJ):
            for j in range(NJ):
                wtr.writerow([name, LAB[i], collab + ":" + LAB[j], f"{M[i, j]:.6f}"])
if _ec.IS_DEFAULT:
    rest_vs_zero = (f"median {np.median(b):+.4f}, Wilcoxon {apa_p(w0.pvalue)} (exact {w0.pvalue:.3g}) - "
                    "'")
else:
    rest_vs_zero = (f"median {np.median(b):+.4f}, Wilcoxon {apa_p(w0.pvalue)} (exact {w0.pvalue:.3g}) - " +
                    (f"statistically {'above' if np.median(b) > 0 else 'below'} zero (two-sided)"
                     if w0.pvalue < 0.05 else
                     "not statistically different from zero (two-sided)"))
if _ec.IS_DEFAULT:
    build_id_note = f"{_ec.TAG} (healthy cohort; BOTH conditions)"
    reporting_format = "median [IQR]; p in APA style (exact values in parentheses)"
    scope_note = ("the 80/80-session accuracy drop ("
                  "W = 0, p < .001, exact 7.9e-15) is the DECODER metric over all 80 sessions (per-joint, no "
                  "all-six-joint requirement) - different scope from the 72-session matrix diagonal")
else:
    def _session_mean(p):
        """session -> mean of the finite rpt_calib over its fold x joint rows (the decoder metric)."""
        acc = {}
        for row in csv.DictReader(p.open(newline="")):
            v = float(row["rpt_calib"]) if row["rpt_calib"] not in ("", "NaN") else np.nan
            if np.isfinite(v):
                acc.setdefault(row["session"], []).append(v)
        return {s: float(np.mean(v)) for s, v in acc.items()}
    sm_iw = _session_mean(_ec.path("h_iw_summary_csv"))
    sm_re = _session_mean(_ec.path("h_rest_summary_csv"))
    S2 = sorted(set(sm_iw) & set(sm_re))
    x_iw, x_re = np.array([sm_iw[s] for s in S2]), np.array([sm_re[s] for s in S2])
    _ec.check_anchor("iw_session_mean", float(x_iw.mean()), "IW session-mean rpt_calib")
    w2 = wilcoxon(x_iw, x_re)
    build_id_note = f"{_ec.TAG} (healthy cohort, causal complex-demodulation decoder; BOTH conditions)"
    reporting_format = "median [IQR]; p in APA style (exact values in parentheses)"
    scope_note = (f"the {int((x_iw > x_re).sum())}/{len(S2)}-session accuracy drop (IW session-mean rpt_calib "
                  f"{x_iw.mean():.4f} -> REST {x_re.mean():.4f}, W = {w2.statistic:.0f}, {apa_p(w2.pvalue)}, "
                  f"exact {w2.pvalue:.3g}) is the DECODER metric over all {len(S2)} sessions (per-joint, no "
                  "all-six-joint requirement; "
                  + "random 10-fold CV folds" + ") - different "
                  f"scope from the {int(ok.sum())}-session matrix diagonal")
with _ec.sd("fig2e_correspondence_metadata.csv").open("w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["field", "value"])
    rows = [
        ("panel", "Fig. 2 panel i - reference x decoded joint correspondence matrices "
                  "(reference / decoded from imagined walking / decoded from resting EEG)"),
        ("iw_source", f"{_ec.rel('h_cross_cohort_iw')} <- {_ec.rel('h_iw_dump_dir').rsplit('/', 1)[-1]}"),
        ("rest_source", f"{_ec.rel('h_cross_cohort_rest')} <- {_ec.rel('h_rest_dump_dir').rsplit('/', 1)[-1]}"),
        ("n_sessions_matrix", str(int(ok.sum())) + " of 80 (sessions with all six joints recorded)"),
        ("reporting_format", reporting_format),
        ("iw_diag_median_iqr", f"{mi:.4f} [{qi1:.4f}, {qi3:.4f}]"),
        ("rest_diag_median_iqr", f"{mr:.4f} [{qr1:.4f}, {qr3:.4f}]"),
        ("paired_drop", f"{int((b < a).sum())}/{int(ok.sum())} sessions, Wilcoxon signed-rank W = {w.statistic:.0f}, "
         f"{apa_p(w.pvalue)} (exact {w.pvalue:.3g})"),
        ("display_polarity", "RHipW X,LKneW X,LAnkW X x(-1) via 03_figures/project_goni.py; display-only"),
        ("rpt_calib_scope_note", scope_note),
    ]
    for row in rows:
        wtr.writerow(row)
print(f"IW  diag median {mi:.3f} [{qi1:.3f}, {qi3:.3f}]   REST {mr:.3f} [{qr1:.3f}, {qr3:.3f}]")
print(f"paired: {int((b < a).sum())}/{int(ok.sum())}  W={w.statistic:.0f}  {apa_p(w.pvalue)}")
print(f"REST vs 0: {apa_p(w0.pvalue)}")
print(f"wrote {OUT_PNG}")
print("wrote source_data + metadata")
