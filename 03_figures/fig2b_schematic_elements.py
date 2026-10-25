"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

schematic ELEMENTS extracted from real sample data

Reads:
  the aligned per-session recordings (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, lfilter
import os

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42

import analysis_config as _ec
_ec.require("main", "main")

ROOT = Path(__file__).resolve().parents[1]
SESSION = "subj11_sess01"
FORMAL_CANDIDATES = [
    (__import__("pathlib").Path(__import__("os").environ.get("GAIT_DATA_ROOT", "")) / "healthy_prep"),
    (__import__("pathlib").Path(__import__("os").environ.get("GAIT_DATA_ROOT", "")) / "healthy_prep"),
    Path.home() / "gait" / "data" / "healthy_prep",
]


def _formal_root():
    for r in FORMAL_CANDIDATES:
        if r.exists():
            return r
    return FORMAL_CANDIDATES[0]


EEG_MAT = _formal_root() / SESSION / f"{SESSION}_eeg.mat"
EVENT_MAT = _formal_root() / SESSION / f"{SESSION}_event.mat"
MAIN_SESSION, MAIN_TRIAL_IDX = "subj42_sess01", 1
if _ec.BUILD_ID in _ec.DECODER_BUILDS:
    MAIN_SESSION, MAIN_TRIAL_IDX = "subj06_sess02", 31
if _ec.BUILD_ID in _ec.DECODER_BUILDS:
    MAIN_SESSION, MAIN_TRIAL_IDX = "subj17_sess01", 51
if not _ec.IS_DEFAULT:
    SESSION = MAIN_SESSION
    _r12 = Path(os.environ["GAIT_HEALTHY_PREP_ROOT"]) if os.environ.get("GAIT_HEALTHY_PREP_ROOT") else _formal_root()
    EEG_MAT = _r12 / SESSION / f"{SESSION}_eeg.mat"
    EVENT_MAT = _r12 / SESSION / f"{SESSION}_event.mat"
PCA = _ec.path("h_global_basis")
DUMP = _ec.path("h_iw_dump_dir") / "subj11_sess01_fold6_trial_dump.mat"
EXAMPLE_TRIAL = 4
OUT = ROOT / "figures_out" / "fig2" / "fig2b_elements"

BANDS = {1: (0.65, 1.15), 2: (1.56, 2.06), 3: (2.46, 2.96)}
INK = "#1a1a1a"
JOINT_C = "#f34142"


def bp(x, lo, hi, fs):
    b, a = butter(4, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, x)


def cdm(x, lo, hi, fs):
    """low-pass at 0.18 Hz frequency-shifted to the band centre fc = (lo + hi) / 2, applied forward-only; returns the
    complex band signal (rows = channels, along the last axis)."""
    b, a = butter(1, 0.18 / (fs / 2))
    w = 2 * np.pi * 0.5 * (lo + hi) / fs
    return lfilter(b * np.exp(1j * w * np.arange(b.size)), a * np.exp(1j * w * np.arange(a.size)), x, axis=-1)


def analytic(x, lo, hi, fs):
    """Analytic band feature: causal complex demodulation."""
    return hilbert(bp(x, lo, hi, fs), axis=1) if _ec.IS_DEFAULT else cdm(x, lo, hi, fs)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = _ec.scoped(OUT / name)
    fig.savefig(p, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {p}")


def load_eeg():
    """Continuous EEG of the example session from the release-format prep export."""
    with h5py.File(EEG_MAT, "r") as m:
        lds = m["eeg/c/labels"]
        chs = []
        for j in range(lds.shape[0]):
            r = lds[j, 0]
            g = m[r] if isinstance(r, h5py.Reference) else r
            chs.append("".join(chr(int(c)) for c in np.array(g).ravel()).strip())
        fs = float(np.array(m["eeg/s"]).reshape(-1)[0])
        x = np.array(m["eeg/x"]).T
    return x, chs, fs


def mi_onsets(good_only=False):
    """0-based EEG-sample onsets of the imagined-walking trials, in acquisition order."""
    from scipy.io import loadmat
    ev = loadmat(EVENT_MAT, squeeze_me=True, struct_as_record=False)["event"]
    return [int(e.etstart) - 1 for e in np.atleast_1d(ev)
            if str(e.code).lower() == "gaitimagery" and (not good_only or str(e.qc).lower() == "good")]


def basis_labels():
    with h5py.File(PCA, "r") as f:
        def s(ref):
            g = f[ref] if isinstance(ref, h5py.Reference) else ref
            return "".join(chr(int(c)) for c in np.array(g).ravel())
        lab = f["labels"]
        labels = [s(lab[0, j]) for j in range(lab.shape[1])]
        basis = np.array(f["basis"])
    return labels, basis


def element_k1k2k3(data, ch_names, fs):
    ch = "C3" if "C3" in ch_names else ch_names[0]
    x = data[ch_names.index(ch)]
    disp, margin = 4.0, 2.5
    s0 = int(0.5 * x.size)
    buf = x[s0 - int(margin * fs): s0 + int((disp + margin) * fs)]
    c0, c1 = int(margin * fs), int(margin * fs) + int(disp * fs)
    t = np.arange(c1 - c0) / fs
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for i, k in enumerate([1, 2, 3]):
        y = (bp(buf, *BANDS[k], fs) if _ec.IS_DEFAULT else np.real(cdm(buf, *BANDS[k], fs)))[c0:c1]
        y = y / (np.std(y) + 1e-12)
        ax.plot(t, y - i * 6.4, color=INK, lw=2.6)
    ax.set_xlim(-0.05, t[-1]); ax.axis("off")
    save(fig, "fig2b_el_k1k2k3.png")


def element_pcs(data, ch_names, fs, n_pc=20):
    """Latent features as a 2D embedding scatter (t-SNE, PCA-2D fallback) of the
    PCA-latent time samples, coloured by gait-fundamental (k1) phase."""
    labels, basis = basis_labels()
    n = int(8 * fs); s0 = int(0.40 * data.shape[1]); seg = data[:, s0:s0 + n]
    cache = {}
    for k in (1, 2, 3):
        an = analytic(seg, *BANDS[k], fs)
        cache[(k, "real")] = an.real
        cache[(k, "imag")] = an.imag
    feat = np.zeros((len(labels), n))
    for idx, lab in enumerate(labels):
        ch, kk, ri = lab.rsplit("_", 2)
        k = int(kk[1:])
        if ch in ch_names:
            feat[idx] = cache[(k, ri)][ch_names.index(ch)]
    feat = (feat - feat.mean(1, keepdims=True)) / (feat.std(1, keepdims=True) + 1e-12)
    scores = (basis[:, :n_pc].T @ feat)
    stride = max(1, n // 1600)
    sel = np.arange(0, n, stride)
    emb = scores[:2, sel].T
    tsec = sel / fs
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    ax.plot(emb[:, 0], emb[:, 1], color="#cccccc", lw=0.5, alpha=0.5, zorder=1)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=tsec, cmap="viridis", s=10, alpha=0.85, linewidths=0, zorder=2)
    ax.axis("off"); ax.set_aspect("equal", "datalim")
    ax.text(0.5, -0.01, "PC1", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK)
    ax.text(-0.01, 0.5, "PC2", transform=ax.transAxes, ha="right", va="center", fontsize=8.5, rotation=90, color=INK)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03, shrink=0.78)
    cb.set_ticks([0, tsec[-1]]); cb.set_ticklabels(["0", f"{tsec[-1]:.0f} s"])
    cb.set_label("time", fontsize=8.5, labelpad=2)
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.outline.set_visible(False)
    print(f"  pcs: PC1 vs PC2, {emb.shape[0]} points coloured by time (0-{tsec[-1]:.1f}s) + colorbar")
    save(fig, "fig2b_el_pcs.png")


def element_fir(trial=EXAMPLE_TRIAL, joint_name="LHipW X"):
    """Causal-FIR calibration effect: dashed = before (P_pred), solid = after (C_pred)."""
    if not _ec.IS_DEFAULT:
        import dump_h5
        tr = dump_h5.load_trial(_ec.path("h_iw_dump_dir") / f"{SESSION}_dump.h5", MAIN_TRIAL_IDX, ("A_pred", "C_pred"))
        fs, joint = tr["fs"], tr["joints"].index(joint_name)
        t, before, after = tr["time_sec"], tr["A_pred"][joint], tr["C_pred"][joint]
    else:
        _ec.require_dump_format("matlab_fold_mat")
        with h5py.File(DUMP, "r") as f:
            def deref(name, i):
                d = f[name]
                return np.array(f[d[i, 0]]) if isinstance(d[i, 0], h5py.Reference) else np.array(d[i, 0])
            def labels():
                g = f[f["meta/target_labels"][0, 0]]
                out = []
                for j in range(g.shape[0]):
                    s = f[g[j, 0]] if isinstance(g[j, 0], h5py.Reference) else g[j, 0]
                    out.append("".join(chr(int(c)) for c in np.array(s).ravel()))
                return out
            fs = float(np.array(f["meta/fs"]).squeeze())
            joint = labels().index(joint_name)
            t = deref("time_sec", trial).ravel()
            before = deref("P_pred", trial)[joint]
            after = deref("C_pred", trial)[joint]
    m = np.isfinite(before) & np.isfinite(after)
    w = np.where(m)[0]
    w = w[: int(6 * fs)] if w.size > int(6 * fs) else w
    tt = t[w] - t[w][0]
    zb = (before[w] - np.mean(before[w])) / (np.std(before[w]) + 1e-12)
    za = (after[w] - np.mean(after[w])) / (np.std(after[w]) + 1e-12)
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    ax.plot(tt, zb, color="#9a9a9a", lw=1.5, ls=(0, (4, 3)), label="before")
    ax.plot(tt, za, color=JOINT_C, lw=2.0, label="after")
    ax.set_xlim(tt[0], tt[-1]); ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, frameon=False, handlelength=1.7,
              borderaxespad=0.1, ncol=2, columnspacing=1.2)
    save(fig, "fig2b_el_fir.png")


POOL_SESSIONS = ["subj01_sess01", "subj02_sess01", "subj03_sess01",
                 "subj05_sess01", "subj08_sess01", "subj21_sess01"]


def _read_eeg_window(path, n_samp, frac=0.45):
    with h5py.File(path, "r") as m:
        lds = m["eeg/c/labels"]
        chs = []
        for j in range(lds.shape[0]):
            r = lds[j, 0]
            s = m[r] if isinstance(r, h5py.Reference) else r
            chs.append("".join(chr(int(c)) for c in np.array(s).ravel()).strip())
        fs = float(np.array(m["eeg/s"]).reshape(-1)[0])
        T = m["eeg/x"].shape[0]
        s0 = min(int(frac * T), max(0, T - n_samp))
        x = np.array(m["eeg/x"][s0:s0 + n_samp, :]).T
    return x, chs, fs


def _harmonic_feature(x, ch_names, labels, fs):
    cache = {}
    for k in (1, 2, 3):
        an = analytic(x, *BANDS[k], fs)
        cache[(k, "real")] = an.real
        cache[(k, "imag")] = an.imag
    feat = np.zeros((len(labels), x.shape[1]))
    for idx, lab in enumerate(labels):
        ch, kk, ri = lab.rsplit("_", 2)
        k = int(kk[1:])
        if ch in ch_names:
            feat[idx] = cache[(k, ri)][ch_names.index(ch)]
    return (feat - feat.mean(1, keepdims=True)) / (feat.std(1, keepdims=True) + 1e-12)


PCS3D_MAX_SEC = 3.3


def element_pcs_3d(data, ch_names, fs, fold_trial=EXAMPLE_TRIAL):
    """3-D state-space orbit of one trial in the decoder's own PCA basis."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    labels, basis = basis_labels()
    with h5py.File(PCA, "r") as f:
        expl = np.array(f["explained"]).ravel()
    if not _ec.IS_DEFAULT:
        import dump_h5
        tr = dump_h5.load_trial(_ec.path("h_iw_dump_dir") / f"{SESSION}_dump.h5", MAIN_TRIAL_IDX, ("Y_true",))
        k, y, tt = tr["trial_idx"], tr["Y_true"], tr["time_sec"]
        ons = mi_onsets(good_only=True)
        from scipy.io import loadmat
        _ev = [e for e in np.atleast_1d(loadmat(EVENT_MAT, squeeze_me=True, struct_as_record=False)["event"])
               if str(e.code).lower() == "gaitimagery" and str(e.qc).lower() == "good"]
        _dd = (_ev[k - 1].etstop - _ev[k - 1].etstart) / fs
        if abs(_dd - tt[-1]) > 0.1:
            raise SystemExit(f"trial_idx {k}: event duration {_dd:.3f} s != dump end {tt[-1]:.3f} s")
    else:
        _ec.require_dump_format("matlab_fold_mat")
        with h5py.File(DUMP, "r") as f:
            k = int(np.array(f["meta/original_trial_indices"]).ravel()[fold_trial])
            y = np.array(f[f["Y_true"][fold_trial, 0]])
            tt = np.array(f[f["time_sec"][fold_trial, 0]]).ravel()
        ons = mi_onsets()
    i0 = ons[k - 1]
    on = i0 / fs
    i1 = i0 + int(round(tt[-1] * fs)) + 1
    seg = data[:, i0:i1]
    cache = {}
    for kk in (1, 2, 3):
        an = analytic(seg, *BANDS[kk], fs)
        cache[(kk, "real")] = an.real; cache[(kk, "imag")] = an.imag
    feat = np.zeros((len(labels), seg.shape[1]))
    for idx, lab in enumerate(labels):
        ch, kn, ri = lab.rsplit("_", 2)
        if ch in ch_names:
            feat[idx] = cache[(int(kn[1:]), ri)][ch_names.index(ch)]
    j0 = int(round(tt[0] * fs))
    feat = feat[:, j0:]
    feat = (feat - feat.mean(1, keepdims=True)) / (feat.std(1, keepdims=True) + 1e-12)
    sc = (basis[:, :3].T @ feat).T
    sc = sc / (np.abs(sc).max(0) + 1e-12)
    hip = y[0]
    ph = np.angle(hilbert(bp(hip, *BANDS[1], fs)))
    ph = (ph % (2 * np.pi)) / (2 * np.pi)
    n = min(sc.shape[0], ph.size); sc, ph = sc[:n], ph[:n]
    n = min(n, int(round(PCS3D_MAX_SEC * fs))); sc, ph = sc[:n], ph[:n]
    pts = sc.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    fig = plt.figure(figsize=(3.3, 3.3))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.08, top=0.96)
    lc = Line3DCollection(segs, cmap="twilight", linewidths=2.2)
    lc.set_array(ph[:-1]); lc.set_clim(0, 1)
    ax.add_collection3d(lc)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_zlim(-1.35, 1.35)
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.9))
    A = 1.30
    for dx, dy, dz in ((A, 0, 0), (0, -A, 0), (0, 0, A)):
        ax.quiver(0, 0, 0, dx, dy, dz, color="black", linewidth=1.0,
                  arrow_length_ratio=0.22, capstyle="round", joinstyle="round")
    print(f"  pcs 3d: MI trial {k} (onset at {on:.2f} s), {n} samples = {n/fs:.2f} s from "
          f"{tt[0]:g} s after onset, ~{(np.unwrap(ph*2*np.pi)[-1]-np.unwrap(ph*2*np.pi)[0])/(2*np.pi):.1f} "
          f"gait cycles; basis explained PC1-3 = {expl[0]*100:.2f}/{expl[1]*100:.2f}/{expl[2]*100:.2f}%")
    OUT.mkdir(parents=True, exist_ok=True)
    _p3d = _ec.scoped(OUT / "fig2b_el_pcs_3d.png")
    fig.savefig(_p3d, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    from PIL import Image as _Image
    _im = _Image.open(_p3d).convert("RGBA")
    _bb = _im.split()[-1].getbbox()
    if _bb:
        _pad = 16
        _bb = (max(_bb[0] - _pad, 0), max(_bb[1] - _pad, 0),
               min(_bb[2] + _pad, _im.width), min(_bb[3] + _pad, _im.height))
        _im.crop(_bb).save(_p3d)
        print(f"  pcs 3d: cropped to ink {_im.size} -> {(_bb[2]-_bb[0], _bb[3]-_bb[1])} px")
    plt.close(fig); print(f"wrote {OUT / 'fig2b_el_pcs_3d.png'}")
    fig, cax = plt.subplots(figsize=(1.6, 0.22))
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.62, top=0.98)
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap="twilight", orientation="horizontal")
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", "50", "100"])
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    cb.outline.set_linewidth(0.6)
    cax.set_title("gait cycle (%)", fontsize=7, pad=2, color=INK)
    save(fig, "fig2b_el_pcs_3d_colorbar.png")


def element_pcs_pool(n_samp_sec=7.0):
    """Leave-one-participant-out training pool: PC1 vs PC2 latent samples from
    several OTHER participants (the pool B is estimated from), coloured by
    participant."""
    labels, basis = basis_labels()
    root = _formal_root()
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    used = 0
    for sess in POOL_SESSIONS:
        path = root / sess / f"{sess}_eeg.mat"
        if not path.exists():
            print(f"  pool skip {sess}: missing")
            continue
        x, chs, fs = _read_eeg_window(path, int(n_samp_sec * 250))
        feat = _harmonic_feature(x, chs, labels, fs)
        sc = basis[:, :2].T @ feat
        st = max(1, sc.shape[1] // 320)
        sel = np.arange(0, sc.shape[1], st)
        ax.scatter(sc[0, sel], sc[1, sel], s=7, alpha=0.55, color=cmap(used % 10),
                   linewidths=0, label=f"P{used + 1}")
        used += 1
    ax.axis("off"); ax.set_aspect("equal", "datalim")
    ax.text(0.5, -0.01, "PC1", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK)
    ax.text(-0.01, 0.5, "PC2", transform=ax.transAxes, ha="right", va="center", fontsize=8.5, rotation=90, color=INK)
    ax.legend(title=f"{used} training participants", fontsize=6.5, title_fontsize=7,
              frameon=False, loc="upper right", handletextpad=0.2, labelspacing=0.2,
              ncol=2, columnspacing=0.8)
    print(f"  pcs pool: {used} participants, PC1 vs PC2")
    save(fig, "fig2b_el_pcs_pool.png")


def main():
    data, ch_names, fs = load_eeg()
    print(f"EEG {len(ch_names)} ch @ {fs:g} Hz")
    element_k1k2k3(data, ch_names, fs)
    element_pcs(data, ch_names, fs)
    element_pcs_3d(data, ch_names, fs)
    element_pcs_pool()
    element_fir()


if __name__ == "__main__":
    main()
