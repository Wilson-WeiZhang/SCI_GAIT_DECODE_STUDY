"""Figure 3 — Gait-phase locking of the imagined-walking representation

schematic elements drawn from recorded data, text-free

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
OUT = ROOT / "figures_out/fig3"
INK, MID, PALE = "#1A1A1A", "#666666", "#B3B3B3"
CHANS = ["Fz", "FC1", "Cz", "C3", "CPz", "Pz"]
PHASOR_CH = "Cz"
FOLD_TRIAL = 0
K1 = (0.6, 1.6)
EEG_DISPLAY = (0.5, 40.0)


def bp(x, lo, hi, fs, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=PS.DPI, facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}_transparent.png", dpi=PS.DPI, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {stem}.png / .svg / _transparent.png")


with h5py.File(DUMP, "r") as f:
    fs = float(np.array(f["meta/fs"]).ravel()[0])
    k = int(np.array(f["meta/original_trial_indices"]).ravel()[FOLD_TRIAL])
    y = np.array(f[f["Y_true"][FOLD_TRIAL, 0]])
    tt = np.array(f[f["time_sec"][FOLD_TRIAL, 0]]).ravel()

raw = mne.io.read_raw_eeglab(str(SET), preload=True, verbose="ERROR")
assert raw.info["sfreq"] == fs
s1 = [o for o, d in zip(raw.annotations.onset, raw.annotations.description) if d == "S  1"]
on = s1[k - 1]
i0, n = int(round((on + tt[0]) * fs)), y.shape[1]
raw_nb = raw.copy()
raw.filter(*EEG_DISPLAY, fir_design="firwin", verbose="ERROR")
X = raw.get_data(picks=[raw.ch_names.index(c) for c in CHANS], start=i0, stop=i0 + n) * 1e6
t = tt - tt[0]

phi = np.angle(hilbert(bp(y[0], *K1, fs)))
PAD = int(round(4.0 * fs))


def narrowband_analytic(ch):
    """Band-pass AND Hilbert on the padded segment, cropping only at the end."""
    xp = raw_nb.get_data(picks=[raw_nb.ch_names.index(ch)],
                         start=i0 - PAD, stop=i0 + n + PAD)[0] * 1e6
    ap = hilbert(bp(xp, *K1, fs))
    return ap[PAD:PAD + n]


_cand = {}
for _ch in raw_nb.ch_names:
    _a = narrowband_analytic(_ch)
    _cand[_ch] = np.exp(1j * (np.angle(_a) - phi)).mean()
PHASOR_CH = max(_cand, key=lambda ch: abs(_cand[ch]))
an = narrowband_analytic(PHASOR_CH)
eeg_nb = an.real
theta = np.angle(an)
z = np.exp(1j * (theta - phi))
c = z.mean()
PS.apply(plt)

fig, ax = plt.subplots(figsize=(40 * PS.MM, 22 * PS.MM), dpi=PS.DPI)
step = 6.0 * np.median(np.std(X, axis=1))
for i, x in enumerate(X):
    ax.plot(t, x - i * step, color=INK, lw=0.45, solid_capstyle="round")
ax.set_xlim(t[0], t[-1]); ax.set_ylim(-(len(CHANS) - 0.4) * step, 0.8 * step)
ax.axis("off")
save(fig, "fig3a_el_1a_eeg")

fig, ax = plt.subplots(figsize=(40 * PS.MM, 15 * PS.MM), dpi=PS.DPI)
ph = phi % (2 * np.pi)
for sg in np.split(np.arange(ph.size), np.where(np.diff(ph) < -np.pi)[0] + 1):
    if sg.size > 1:
        ax.plot(t[sg], ph[sg], color=INK, lw=1.0, solid_capstyle="round")
ax.set_xlim(t[0], t[-1]); ax.set_ylim(-0.25, 2 * np.pi + 0.25)
ax.set_yticks([0, 2 * np.pi]); ax.set_yticklabels(["0", "2π"], fontsize=PS.NOTE_PT)
ax.set_xticks([])
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.tick_params(length=2, width=0.8)
save(fig, "fig3a_el_1b_phase")

fig, ax = plt.subplots(figsize=(34 * PS.MM, 20 * PS.MM), dpi=PS.DPI)
env = np.abs(an)
ax.fill_between(t, -env, env, color=PALE, alpha=0.55, lw=0)
ax.plot(t, env, color=MID, lw=0.6)
ax.plot(t, -env, color=MID, lw=0.6)
ax.plot(t, eeg_nb, color=INK, lw=0.9, solid_capstyle="round")
ax.set_xlim(t[0], t[-1]); m = 1.15 * env.max(); ax.set_ylim(-m, m)
ax.axis("off")
save(fig, "fig3a_el_2_analytic")

fig, ax = plt.subplots(figsize=(24 * PS.MM, 24 * PS.MM), dpi=PS.DPI)
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), color=MID, lw=0.8)
sub = z[::max(1, z.size // 24)]
for w in sub:
    ax.annotate("", xy=(w.real, w.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.10,head_length=0.22",
                                color=INK, lw=0.7))
    ax.plot([w.real], [w.imag], marker="o", ms=2.0, color=INK, zorder=3)
ax.set_xlim(-1.22, 1.22); ax.set_ylim(-1.22, 1.22); ax.set_aspect("equal")
ax.axis("off")
save(fig, "fig3a_el_3_phasors")

fig, ax = plt.subplots(figsize=(24 * PS.MM, 24 * PS.MM), dpi=PS.DPI)
ax.plot(np.cos(th), np.sin(th), color=MID, lw=0.8)
for w in z[::max(1, z.size // 90)]:
    ax.plot([0, w.real], [0, w.imag], color=PALE, lw=0.45, zorder=1)
_hl = min(0.30, 0.34 * abs(c))
ax.annotate("", xy=(c.real, c.imag), xytext=(0, 0),
            arrowprops=dict(arrowstyle=f"-|>,head_width={0.55*_hl:.3f},head_length={_hl:.3f}",
                            color=INK, lw=1.6), zorder=4)
ax.set_xlim(-1.22, 1.22); ax.set_ylim(-1.22, 1.22); ax.set_aspect("equal")
ax.axis("off")
save(fig, "fig3a_el_5_locking")

cycles = (np.unwrap(phi)[-1] - np.unwrap(phi)[0]) / (2 * np.pi)
print(f"\nsession : SUB_37_sess02 / subj37_sess02 fold5")
print(f"trial   : dump trial {FOLD_TRIAL} -> MI trial {k}, S 1 at {on:.2f} s; analysis window "
      f"{tt[0]:.1f}-{tt[-1]:.3f} s ({n} samples, ~{cycles:.1f} gait cycles)")
print(f"phasors : channel {PHASOR_CH} (largest |c| of the 60 in this trial), "
      f"band {K1[0]}-{K1[1]} Hz, "
      f"|c| = {abs(c):.3f}")
