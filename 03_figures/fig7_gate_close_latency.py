"""Figure 7 — Intent gate and physics simulation of the decoded gait phase

_gate_close_latency — Fig. 7 gate block: once rest begins, how fast does the gate close?

Reads:
  04_source_data/fig7/fig7_gate_timeline_metadata.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import project_palette as P

SRC = f"{ROOT}/04_source_data/fig7/fig7_gate_timeline_source_data.csv"
SRC_MD = f"{ROOT}/04_source_data/fig7/fig7_gate_timeline_metadata.csv"
OUT = f"{ROOT}/figures_out/fig7/fig7_gate_close_latency.png"
OUT_LEG = f"{ROOT}/figures_out/fig7/fig7_gate_close_latency_legend.png"
SD = f"{ROOT}/04_source_data/fig7/fig7_gate_close_latency_source_data.csv"
SD_BARS = f"{ROOT}/04_source_data/fig7/fig7_gate_close_latency_bars_source_data.csv"
MD = f"{ROOT}/04_source_data/fig7/fig7_gate_close_latency_metadata.csv"
CAND = f"{ROOT}/figures_out/fig7/fig7_gate_close_latency_reference.png"
import gate_config as GE
SRC, SRC_MD, OUT, OUT_LEG, SD, SD_BARS, MD, CAND = (GE.scoped(p) for p in (SRC, SRC_MD, OUT, OUT_LEG, SD, SD_BARS, MD, CAND))
CAND_MD5 = "d91155379da3c269261913447e19070b"

TMAX = 9.0
HOP = 1.0
H_IN = 1.00
W_IN = 3.45
FS = 1.0


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


plt.rcParams.update({"font.family": "Arial", "font.size": 6.5 * FS, "axes.linewidth": 1.0,
                     "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42})

d = pd.read_csv(SRC)
ev = d[(d.row_type == "event") & (d.instructed == "REST")].copy()
gate = d[d.row_type == "gate_S1"]
rows = []
for _, e in ev.iterrows():
    g = gate[(gate.session == e.session) & (gate.t_end_s > e.t_start_s) & (gate.t_start_s < e.t_end_s)]
    t0 = e.t_start_s
    s_from = round(e.scored_from_s - t0, 3)
    n_dec = int(round(e.scored_to_s - e.scored_from_s))
    lat = np.inf
    for te in s_from + HOP * np.arange(n_dec):
        hit = g[(g.t_start_s <= t0 + te + HOP / 2) & (g.t_end_s > t0 + te + HOP / 2)]
        if len(hit) and hit.iloc[0].state == "closed":
            lat = float(te)
            break
    rows.append(dict(session=e.session, patient=e.session[:5], lat=lat,
                     onset_s=t0, scored_from_s=e.scored_from_s, scored_to_s=e.scored_to_s))
R = pd.DataFrame(rows)
n = len(R)
never = int(np.isinf(R.lat).sum())
first_dec = float(np.round(ev.scored_from_s.iloc[0] - ev.t_start_s.iloc[0], 2))

tg = np.arange(0, TMAX + 1e-9, 0.05)
CHECKPOINTS = (3.0, 4.0, 5.0, 7.0, 9.0)
BAR_W, BAR_GAP = 0.36, 0.04
xs = np.arange(len(CHECKPOINTS), dtype=float)
fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=300)
bar_vals = {}
for j, pat in enumerate(("pat01", "pat02")):
    Rp = R[R.patient == pat]
    face = P.PATIENT_P1 if pat == "pat01" else P.PATIENT_P2
    frac = [float((Rp.lat <= t).mean()) for t in CHECKPOINTS]
    bar_vals[pat] = frac
    xb = xs + (j - 0.5) * (BAR_W + BAR_GAP)
    pct = [100 * f for f in frac]
    ax.bar(xb, pct, width=BAR_W, facecolor=face, edgecolor=P.INK, linewidth=0.9, zorder=3)
    for x_, v in zip(xb, pct):
        ax.text(x_, v + 2.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.5 * FS,
                color=P.INK, zorder=5)
ax.set_xlim(-0.6, len(CHECKPOINTS) - 0.4); ax.set_ylim(0, 112)
ax.set_yticks([0, 50, 100])
ax.set_xticks(xs); ax.set_xticklabels([f"{t:.0f}" for t in CHECKPOINTS])
ax.set_xlabel("Time since rest onset (s)", labelpad=2)
ax.set_ylabel("Gate closed\n(% of rest events)", labelpad=2)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=3, width=1.0)
fig.tight_layout(); fig.savefig(OUT, dpi=300, facecolor="white"); plt.close(fig)

lf = plt.figure(figsize=(2.0, 0.45), dpi=300)
from matplotlib.patches import Patch
lf.legend([Patch(facecolor=P.PATIENT_P1, edgecolor=P.INK, lw=0.8),
           Patch(facecolor=P.PATIENT_P2, edgecolor=P.INK, lw=0.8)],
          ["Patient 1", "Patient 2"], loc="center", frameon=False, ncol=2,
          fontsize=6.5, handlelength=1.6, columnspacing=1.0)
lf.savefig(OUT_LEG, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02); plt.close(lf)

got = md5(OUT)
print(f"wrote {os.path.relpath(OUT, ROOT)}  (md5 {got})")
print(f"wrote {os.path.relpath(OUT_LEG, ROOT)}")

ev_rows = []
for _, r in R.sort_values(["session", "onset_s"]).iterrows():
    fin = np.isfinite(r.lat)
    ev_rows.append(dict(row_type="event", session=r.session, patient=r.patient,
                        onset_s=round(float(r.onset_s), 4),
                        scored_from_s=round(float(r.scored_from_s), 4),
                        scored_to_s=round(float(r.scored_to_s), 4),
                        close_latency_s=(round(float(r.lat), 2) if fin else ""),
                        closed_by_3s=int(fin and r.lat <= 3), closed_by_4s=int(fin and r.lat <= 4),
                        closed_by_5s=int(fin and r.lat <= 5), closed_by_7s=int(fin and r.lat <= 7),
                        t_s="", frac_closed=""))
cv_rows = []
for pat in ("pat01", "pat02"):
    Rp = R[R.patient == pat]
    for t in tg:
        cv_rows.append(dict(row_type="curve", session="", patient=pat, onset_s="",
                            scored_from_s="", scored_to_s="", close_latency_s="",
                            closed_by_3s="", closed_by_4s="", closed_by_5s="", closed_by_7s="",
                            t_s=round(float(t), 2), frac_closed=round(float((Rp.lat <= t).mean()), 6)))
sd = pd.DataFrame(ev_rows + cv_rows)
sd.to_csv(SD, index=False, lineterminator="\n")
print(f"wrote {os.path.relpath(SD, ROOT)}  ({len(ev_rows)} event rows + {len(cv_rows)} curve rows)")

bar_rows = []
for pat in ("pat01", "pat02"):
    Rp = R[R.patient == pat]
    for t, fr in zip(CHECKPOINTS, bar_vals[pat]):
        bar_rows.append(dict(patient=pat, t_s=round(float(t), 1), n_rest_events=int(len(Rp)),
                             n_closed=int((Rp.lat <= t).sum()), fraction_closed=round(fr, 6)))
pd.DataFrame(bar_rows).to_csv(SD_BARS, index=False, lineterminator="\n")
print(f"wrote {os.path.relpath(SD_BARS, ROOT)}  ({len(bar_rows)} bar rows)")

src_md = pd.read_csv(SRC_MD).set_index("key")["value"].to_dict()
per_pat = {}
for pat in ("pat01", "pat02"):
    Rp = R[R.patient == pat]
    per_pat[pat] = dict(n=len(Rp), by3=(Rp.lat <= 3).mean(), by4=(Rp.lat <= 4).mean(),
                        by5=(Rp.lat <= 5).mean(), by7=(Rp.lat <= 7).mean(), by9=(Rp.lat <= 9).mean(),
                        never=float(np.isinf(Rp.lat).mean()))
meta = [
    ("gate_set", src_md["gate_set"]),
    ("window_s", src_md["window_s"]),
    ("karcher_solver", src_md["karcher_solver"]),
    ("not_scored", src_md["not_scored"]),
    ("cohort", src_md["cohort"]),
    ("definition_denominator", "The denominator of every bar is ALL rest events of that patient "
                               "(not only the ones that close), so the never-closed events are the "
                               "visible gap between the 9 s bar and 1."),
    ("n_rest_events_total", f"{n} (both patients pooled; {never} never close = "
                            f"{100 * never / n:.1f} %)"),
    ("pat01_n", per_pat["pat01"]["n"]),
    ("pat01_closed_by_3s", f"{per_pat['pat01']['by3']:.4f}"),
    ("pat01_closed_by_4s", f"{per_pat['pat01']['by4']:.4f}"),
    ("pat01_closed_by_5s", f"{per_pat['pat01']['by5']:.4f}"),
    ("pat01_closed_by_7s", f"{per_pat['pat01']['by7']:.4f}"),
    ("pat01_closed_by_9s", f"{per_pat['pat01']['by9']:.4f}"),
    ("pat01_never_closed", f"{per_pat['pat01']['never']:.4f}"),
    ("pat02_n", per_pat["pat02"]["n"]),
    ("pat02_closed_by_3s", f"{per_pat['pat02']['by3']:.4f}"),
    ("pat02_closed_by_4s", f"{per_pat['pat02']['by4']:.4f}"),
    ("pat02_closed_by_5s", f"{per_pat['pat02']['by5']:.4f}"),
    ("pat02_closed_by_7s", f"{per_pat['pat02']['by7']:.4f}"),
    ("pat02_closed_by_9s", f"{per_pat['pat02']['by9']:.4f}"),
    ("pat02_never_closed", f"{per_pat['pat02']['never']:.4f}"),
    ("bars_source_data", "04_source_data/fig7/fig7_gate_close_latency_bars_source_data.csv — the "
                         "PLOTTED values of this panel, 10 rows (2 patients x 5 checkpoints), "
                         "columns patient,t_s,n_rest_events,n_closed,fraction_closed. The "
                         "event-level fig7_gate_close_latency_source_data.csv is unchanged and "
                         "stays the per-event record the bars are aggregated from."),
    ("producer", "03_figures/fig7_gate_close_latency.py"),
]
meta = GE.meta(meta)
pd.DataFrame(meta, columns=["key", "value"]).to_csv(MD, index=False, lineterminator="\n")
print(f"wrote {os.path.relpath(MD, ROOT)}  ({len(meta)} keys)")
for pat in ("pat01", "pat02"):
    p = per_pat[pat]
    print(f"{pat}: n={p['n']}, closed by 3 s {p['by3']:.3f}, 4 s {p['by4']:.3f}, "
          f"5 s {p['by5']:.3f}, 7 s {p['by7']:.3f}, 9 s {p['by9']:.3f}; never {p['never']:.3f}")
