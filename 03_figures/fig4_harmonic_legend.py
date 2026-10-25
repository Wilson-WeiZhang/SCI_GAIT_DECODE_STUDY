"""Figure 4 — Shared locomotor representation across imagined and real walking

Shared harmonic legend strip for Fig. 4 d/e/f/h/i/j (one row, three circles)

Reads:
  04_source_data/fig2/fig2a_goni_power_summary.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS
import project_palette as P

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "04_source_data/fig2/fig2a_goni_power_summary.csv"
OUT = ROOT / "figures_out/fig4/fig4_harmonic_legend.png"
MD = ROOT / "04_source_data/fig4/fig4_harmonic_legend_metadata.csv"
PEAK_BAND = (0.45, 1.15)

rows = list(csv.DictReader(SRC.open()))
f = np.array([float(r["freq_hz"]) for r in rows])
m = (f >= PEAK_BAND[0]) & (f <= PEAK_BAND[1])
peaks = {}
for jn in ("hip", "knee", "ankle"):
    mean = np.array([float(r[f"{jn}_mean"]) for r in rows])
    peaks[jn] = float(f[m][np.argmax(mean[m])])
k1 = float(np.mean(list(peaks.values())))
freqs = [k1, 2 * k1, 3 * k1]
names = ["stride", "step", "3rd harmonic"]
labels = [f"$\\mathrm{{k}}_{i+1}$  {names[i]}, {freqs[i]:.1f} Hz" for i in range(3)]

PS.apply(plt)
fig, ax = plt.subplots(figsize=(120 * PS.MM, 9 * PS.MM), dpi=PS.DPI)
ax.axis("off")
h = [ax.scatter([], [], s=34, color=c, edgecolors="none", label=l) for c, l in zip(P.HARMONIC_COLORS, labels)]
ax.legend(handles=h, loc="center", ncol=3, frameon=False, fontsize=PS.NOTE_PT,
          handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
fig.savefig(OUT, dpi=PS.DPI, facecolor="white")
with MD.open("w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["key", "value"])
    for k, v in [("panel", "Fig. 4 shared harmonic legend (panels d e f h)"),
                 ("freq_source", SRC.relative_to(ROOT).as_posix()), ("peak_band_hz", f"{PEAK_BAND[0]}-{PEAK_BAND[1]}"),
                 ("peak_hip_hz", f"{peaks['hip']:.2f}"), ("peak_knee_hz", f"{peaks['knee']:.2f}"), ("peak_ankle_hz", f"{peaks['ankle']:.2f}"),
                 ("k1_hz", f"{k1:.3f}"), ("k2_hz", f"{2*k1:.3f}"), ("k3_hz", f"{3*k1:.3f}"),
                 ("note", "cohort-average goniometer PSD peaks (Fig.2b); the decoder uses per-session cadence-calibrated harmonics")]:
        w.writerow([k, v])
print(f"wrote {OUT.name}  k1={k1:.3f} Hz  peaks={ {k: round(v,2) for k,v in peaks.items()} }")
