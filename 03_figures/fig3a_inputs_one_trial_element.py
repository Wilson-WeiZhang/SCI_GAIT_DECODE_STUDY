"""Figure 3 — Gait-phase locking of the imagined-walking representation

element — "Inputs for one trial": real multi-channel EEG + the real gait phase phi(t)

Reads:
  decoder output in data/decoder_output/ (not distributed)
  the aligned per-session recordings (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import h5py
import mne
from scipy.signal import butter, filtfilt, hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS

SET = ROOT / "data/prep_data/healthy/eeg/SUB_37_sess02_step2.set"
DUMP = (ROOT / "data/decoder_output/trial_dump_imagined_walking"
        / "subj37_sess02_fold5_trial_dump.mat")
OUT = ROOT / "figures_out/fig3/fig3a_el_inputs_one_trial"

CHANS = ["Fz", "FC1", "Cz", "C3", "CPz", "Pz"]
FOLD_TRIAL = 0
K1 = (0.6, 1.6)
EEG_DISPLAY = (0.5, 40.0)
ACC = "#1A1A1A"


def bp(x, lo, hi, fs, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


with h5py.File(DUMP, "r") as f:
    fs = float(np.array(f["meta/fs"]).ravel()[0])
    k = int(np.array(f["meta/original_trial_indices"]).ravel()[FOLD_TRIAL])
    y = np.array(f[f["Y_true"][FOLD_TRIAL, 0]])
    tt = np.array(f[f["time_sec"][FOLD_TRIAL, 0]]).ravel()

raw = mne.io.read_raw_eeglab(str(SET), preload=True, verbose="ERROR")
assert raw.info["sfreq"] == fs, "EEG and dump sampling rates differ"
raw.filter(*EEG_DISPLAY, fir_design="firwin", verbose="ERROR")
s1 = [o for o, d in zip(raw.annotations.onset, raw.annotations.description) if d == "S  1"]
on = s1[k - 1]
i0 = int(round((on + tt[0]) * fs))
X = raw.get_data(picks=[raw.ch_names.index(c) for c in CHANS],
                 start=i0, stop=i0 + y.shape[1]) * 1e6
t = tt - tt[0]

phi = np.angle(hilbert(bp(y[0], *K1, fs)))
phi = phi % (2 * np.pi)

PS.apply(plt)
fig, (ax, axp) = plt.subplots(2, 1, figsize=(46 * PS.MM, 40 * PS.MM), dpi=PS.DPI,
                              gridspec_kw=dict(height_ratios=[2.0, 1.0], hspace=0.42))
step = 6.0 * np.median(np.std(X, axis=1))
for i, x in enumerate(X):
    ax.plot(t, x - i * step, color="#1A1A1A", lw=0.45, solid_capstyle="round")
ax.set_xlim(t[0], t[-1]); ax.set_ylim(-(len(CHANS) - 0.4) * step, 0.8 * step)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

d = np.diff(phi)
cut = np.where(d < -np.pi)[0]
segs = np.split(np.arange(phi.size), cut + 1)
for sg in segs:
    if sg.size > 1:
        axp.plot(t[sg], phi[sg], color=ACC, lw=1.0, solid_capstyle="round")
axp.set_xlim(t[0], t[-1]); axp.set_ylim(-0.25, 2 * np.pi + 0.25)
axp.set_yticks([0, 2 * np.pi]); axp.set_yticklabels(["0", "2π"], fontsize=PS.NOTE_PT)
axp.set_xticks([])
axp.set_ylabel(r"$\phi(t)$", fontsize=PS.NOTE_PT, color=ACC, labelpad=-1)
axp.set_xlabel("time", fontsize=PS.NOTE_PT, labelpad=1)
for s_ in ("top", "right"):
    axp.spines[s_].set_visible(False)
axp.tick_params(length=2, width=0.8)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=PS.DPI, facecolor="white", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}_transparent.png", dpi=PS.DPI, transparent=True,
            bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}.svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

cycles = (np.unwrap(phi)[-1] - np.unwrap(phi)[0]) / (2 * np.pi)
print(f"session   : SUB_37_sess02 / subj37_sess02 fold5")
print(f"trial     : dump trial {FOLD_TRIAL} -> MI trial {k} (1-based), S 1 at {on:.2f} s")
print(f"window    : decoder analysis window {tt[0]:.1f}-{tt[-1]:.3f} s after onset "
      f"({y.shape[1]} samples = {y.shape[1]/fs:.2f} s)")
print(f"phase     : LHipW X, k1 band {K1[0]}-{K1[1]} Hz, ~{cycles:.1f} gait cycles")
print(f"eeg       : {', '.join(CHANS)} (6 of 60), display band {EEG_DISPLAY[0]}-{EEG_DISPLAY[1]} Hz")
print(f"wrote {OUT}.png / .svg / _transparent.png")
