"""Figure 7 — physics simulation and walk-intent gate

MATLAB v7.3 / HDF5, one file per (session, fold), 10 folds per session

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os
import csv
import glob
import sys
from pathlib import Path
import numpy as np
import h5py

from biped_model import SIGN

_P = Path(__file__).resolve().parents
sys.path.append(str(_P[2] / "03_figures"))
import analysis_config as ec

FS = 250.0
import os as _os
MIN_DUR_S = float(_os.environ.get("PHYS_MIN_DUR_S", "3.0"))
EXCL_SESSIONS = {"pat01_sess03", "pat02_sess04", "pat02_sess07"}
G0_SESSIONS = ["pat01_sess04", "pat01_sess06", "pat01_sess07", "pat01_sess09",
               "pat02_sess03", "pat02_sess05", "pat02_sess06"]


def build_id_out(out):
    """renderers read."""
    if not ec.SCOPED or os.path.basename(os.path.normpath(out)) == ec.OUT_DIRNAME:
        return out
    os.makedirs(os.path.join(out, ec.OUT_DIRNAME), exist_ok=True)
    return os.path.join(out, ec.OUT_DIRNAME)


def build_id_inputs(**paths):
    """"""
    for what, p in paths.items():
        if p:
            ec.check_tag(p, "decoder", what)


def _mstr(h, ds):
    return "".join(chr(c) for c in np.array(ds).ravel().astype(int))


def loso_setpoint(npz_path, session):
    """[6] per-joint set-point in DATA convention (deg): mean over phase of the same
    patient's leave-this-session-out cycle-average walker template."""
    z = np.load(npz_path, allow_pickle=True)
    S = [str(s) for s in z["sessions"]]
    pat = session.split("_")[0]
    keep = [i for i, s in enumerate(S)
            if s != session and not s.endswith("iw") and s.split("_")[0] == pat
            and s not in EXCL_SESSIONS]
    if not keep:
        raise RuntimeError(f"no LOSO template sessions for {session}")
    T = np.nanmean(z["obs"][keep], axis=0)
    return T.mean(axis=1), [S[i] for i in keep]


def device_stance_ankle(setpoint):
    """Hip and knee set-points stay the LOSO template means.  The ankle set-point is replaced by
    the value that puts the foot plate FLAT on the ground for those hip/knee set-points, i.e.
    what a real exoskeleton does when it calibrates its stance pose at power-on."""
    q = np.asarray(setpoint, float) * SIGN
    q[2] = q[1] - q[0]
    q[5] = q[4] - q[3]
    return q * SIGN


def _longest_finite_run(Y):
    ok = np.isfinite(Y).all(axis=1)
    best, cur, i0, b0 = 0, 0, 0, 0
    for i, v in enumerate(ok):
        if v:
            if cur == 0:
                i0 = i
            cur += 1
            if cur > best:
                best, b0 = cur, i0
        else:
            cur = 0
    return b0, b0 + best


def h5_session_trials(dump_dir, session):
    """Each item: fold, ordinal (0-based position in the fold's test list = the MATLAB row index),
    trial_idx, Y_true / C_pred [T, 6] deg, time_sec [T]."""
    sys.path.insert(0, str(ec.ROOT / "02_decoding/pyport"))
    import dumpio
    S = dumpio.load_session(os.path.join(dump_dir, f"{session}_dump.h5"))
    if S["joints"] != [f"{j} X" for j in ("LHipW", "LKneW", "LAnkW", "RHipW", "RKneW", "RAnkW")]:
        raise RuntimeError(f"{session}: unexpected joint order {S['joints']}")
    out = []
    for fold in sorted(S["folds"]):
        fd = S["folds"][fold]
        for tid, d in fd["trials"].items():
            out.append(dict(fold=int(fold), ordinal=int(fd["test_idx"].index(tid)), trial_idx=int(tid),
                            Y_true=np.asfortranarray(d["Y"]), C_pred=np.asfortranarray(d["C"]),
                            time_sec=d["time_sec"]))
            if F7_WARM or F7_FROM2S:
                out[-1]["A_pred"] = np.asfortranarray(d["A"])
    return sorted(out, key=lambda r: (r["fold"], r["ordinal"]))


F7_WARM = _os.environ.get("GAIT_F7_WARM", "").strip() == "apred"
if F7_WARM and not ec.F7_SIM:
    raise RuntimeError("GAIT_F7_WARM=apred is defined only with GAIT_F7_SIM=causal")
F7_APRED_WARM_N = 500
F7_INIT = _os.environ.get("GAIT_F7_INIT", "").strip()
F7_FROM2S = ec.F7_SIM == "causal_from2s"
if F7_FROM2S and (F7_WARM or F7_INIT):
    raise RuntimeError("GAIT_F7_SIM=causal_from2s already implies the 2-4 s warm-up drive; unset GAIT_F7_WARM and GAIT_F7_INIT")
if F7_INIT and (not F7_WARM or F7_INIT not in ("T1", "T2", "coast", "full", "dec", "decf", "zp")):
    raise RuntimeError(f"GAIT_F7_INIT={F7_INIT!r}: one of T1 / T2 / coast / full / dec / decf / zp, "
                       "only with GAIT_F7_WARM=apred")
F7_INIT_XF_N = 125

_F7_WIN = 1.0 if (ec.F7_SIM and not (F7_WARM or F7_FROM2S)) else 0.0


def f7_warm_input(X, a):
    """GAIT_F7_WARM=apred: [<= F7_APRED_WARM_N, 6] warm-up input = rows [a - 500, a) of X (clipped at 0); per joint,
    every sample before the trailing all-finite run (the one ending at a - 1) is set to NaN, i.e. the warm-up of
    that joint starts at its first finite sample."""
    W = np.array(X[max(a - F7_APRED_WARM_N, 0):a], dtype=float)
    for j in range(W.shape[1]):
        bad = np.flatnonzero(~np.isfinite(W[:, j]))
        if bad.size:
            W[:bad[-1] + 1, j] = np.nan
    return W


def load_session_trials(dump_dir, session, stream="Y_true"):
    """All qualifying trials of one session, de-duplicated across folds."""
    if ec.DUMP_FORMAT == "pyport_session_h5":
        out = []
        for r in h5_session_trials(dump_dir, session):
            Y, t = r[stream], r["time_sec"]
            a, b = _longest_finite_run(Y)
            if (b - a) < int(round(MIN_DUR_S * FS)):
                continue
            out.append(dict(session=session, trial_idx=r["trial_idx"], fold=r["fold"], ordinal=r["ordinal"],
                            t0_sec=float(t[a]), n=int(b - a), dur_s=float((b - a) / FS) - _F7_WIN, Y=Y[a:b]))
            if F7_WARM or F7_FROM2S:
                out[-1]["W"] = f7_warm_input(r["A_pred"] if stream == "C_pred" else Y, a)
        return sorted(out, key=lambda r: r["trial_idx"])
    ec.require_dump_format("matlab_fold_mat")
    files = sorted(glob.glob(os.path.join(dump_dir, f"{session}_fold*_trial_dump.mat")),
                   key=lambda p: int(os.path.basename(p).split("_fold")[1].split("_")[0]))
    seen, out = set(), []
    for f in files:
        fold = int(os.path.basename(f).split("_fold")[1].split("_")[0])
        with h5py.File(f, "r") as h:
            idx = np.array(h["meta/original_trial_indices"]).ravel().astype(int)
            ref, tref = h[stream], h["time_sec"]
            for o in range(ref.shape[0]):
                tid = int(idx[o])
                if tid in seen:
                    continue
                Y = np.array(h[ref[o, 0]]).T
                t = np.array(h[tref[o, 0]]).ravel()
                a, b = _longest_finite_run(Y)
                if (b - a) < int(round(MIN_DUR_S * FS)):
                    continue
                seen.add(tid)
                out.append(dict(session=session, trial_idx=tid, fold=fold, ordinal=o,
                                t0_sec=float(t[a]), n=int(b - a),
                                dur_s=float((b - a) / FS), Y=Y[a:b]))
    return sorted(out, key=lambda r: r["trial_idx"])


def scramble(Y, seed):
    """randomisation.  The amplitude spectrum of every channel is preserved exactly, so each
    channel keeps its power spectrum, its variance and its band content; what is destroyed is
    the phase -- the within-joint timing and, because each joint draws independent phases, the
    between-joint coordination."""
    rng = np.random.default_rng(int(seed))
    out = np.empty_like(Y)
    for j in range(Y.shape[1]):
        mu = Y[:, j].mean()
        F = np.fft.rfft(Y[:, j] - mu)
        ph = rng.uniform(0.0, 2.0 * np.pi, len(F))
        ph[0] = 0.0
        out[:, j] = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(Y)) + mu
    return out


def loso_gain(npz_path, session, Y):
    """Per-joint ratio of the leave-this-session-out template SD to the trial SD; the template is the same
    leave-this-session-out cycle average used for the set-point."""
    z = np.load(npz_path, allow_pickle=True)
    S = [str(s) for s in z["sessions"]]
    pat = session.split("_")[0]
    keep = [i for i, s in enumerate(S)
            if s != session and not s.endswith("iw") and s.split("_")[0] == pat
            and s not in EXCL_SESSIONS]
    tmpl_sd = np.nanmean(z["obs"][keep], axis=0).std(axis=1)
    return tmpl_sd / np.maximum(Y.std(axis=0), 1e-9)


def uscramble(Y, seed):
    """One random phase array is drawn and applied to ALL six channels, so every channel keeps its
    own amplitude spectrum AND the six channels keep their mutual phase relationships: the
    cross-channel coherence (coordination) survives, only the common temporal structure is
    destroyed."""
    rng = np.random.default_rng(int(seed))
    n = len(Y)
    ph = rng.uniform(0.0, 2.0 * np.pi, n // 2 + 1)
    ph[0] = 0.0
    out = np.empty_like(Y)
    for j in range(Y.shape[1]):
        mu = Y[:, j].mean()
        F = np.fft.rfft(Y[:, j] - mu)
        out[:, j] = np.fft.irfft(np.abs(F) * np.exp(1j * ph[:len(F)]), n=n) + mu
    return out


def loso_template(npz_path, session):
    """[6, 50] leave-this-session-out goniometer cycle-average template, deg, DATA convention."""
    z = np.load(npz_path, allow_pickle=True)
    S = [str(s) for s in z["sessions"]]
    pat = session.split("_")[0]
    keep = [i for i, s in enumerate(S)
            if s != session and not s.endswith("iw") and s.split("_")[0] == pat
            and s not in EXCL_SESSIONS]
    if not keep:
        raise RuntimeError(f"no LOSO template sessions for {session}")
    return np.nanmean(z["obs"][keep], axis=0)


def fukuchi_template(csv_path=None):
    """[6, 50] Fukuchi 2018 normative sagittal template (young24, 50 pt), deg, DATA convention."""
    if csv_path is None:
        csv_path = _default_normative_csv()
    with open(csv_path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    hip = np.array([float(r["hip_deg"]) for r in rows])
    kne = np.array([float(r["knee_deg"]) for r in rows])
    ank = np.array([float(r["ankle_deg"]) for r in rows])
    if len(hip) != 50:
        raise RuntimeError(f"expected 50 phase points, got {len(hip)} in {csv_path}")
    L = [hip, kne, ank]
    R = [np.roll(v, 25) for v in L]
    return np.vstack(L + R) * SIGN[:, None]


def _default_normative_csv():
    """Path of the normative template CSV: GAIT_NORMATIVE_CSV, else normative_data/ next to this file."""
    env = os.environ.get("GAIT_NORMATIVE_CSV")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    name = "fukuchi2018_sagittal_young24_50pt.csv"
    for cand in (os.path.join(here, "normative_data", name),
                 os.path.join(here, "..", "reference_data", "normative_data", name)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError(f"{name} not found; set GAIT_NORMATIVE_CSV")


def bandpass(x, lo, hi):
    """2nd-order zero-phase Butterworth band-pass, used only to isolate the cadence band before
    the Hilbert transform (phase extraction), never on anything that is reported."""
    from scipy.signal import butter, filtfilt
    b, a = butter(2, [max(lo, 0.05) / (FS / 2), min(hi, FS / 2 * 0.9) / (FS / 2)], "band")
    return filtfilt(b, a, x, axis=0)


def f0_of(Y):
    """Trial cadence f0 (Hz) = spectral peak of the KNEE channel inside 0.3-1.6 Hz.  The knee
    carries the largest and cleanest stride-rate component of the six."""
    k = Y[:, 1] - Y[:, 1].mean()
    fr = np.fft.rfftfreq(len(k), 1.0 / FS)
    pw = np.abs(np.fft.rfft(k)) ** 2
    band = (fr > 0.3) & (fr < 1.6)
    return float(fr[band][np.argmax(pw[band])])


def global_delta(tmpl_g, tmpl_n):
    """ONE global cycle-origin offset (in cycle fraction) aligning the normative template to the
    goniometer template, plus the mean correlation achieved."""
    best, bshift = -np.inf, 0
    for shift in range(50):
        c = 0.0
        for j in range(6):
            a = tmpl_g[j] - tmpl_g[j].mean()
            b = np.roll(tmpl_n[j], shift) - tmpl_n[j].mean()
            c += float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c > best:
            best, bshift = c, shift
    return bshift / 50.0, best / 6.0


def template_at(phase, tmpl_n):
    """[T, 6] normative template sampled at an arbitrary cycle-position series."""
    grid = np.arange(50) / 50.0
    return np.vstack([np.interp(phase, grid, tmpl_n[j], period=1.0) for j in range(6)]).T


def raw_phases(Y, tmpl_g, f0):
    """[T, 6] per-joint cycle phase in radians (unwrapped), from the k1 (cadence) band."""
    from scipy.signal import hilbert
    Cg = np.fft.rfft(tmpl_g, axis=1)
    out = np.zeros_like(Y)
    for j in range(6):
        x = Y[:, j] - Y[:, j].mean()
        ph = np.unwrap(np.angle(hilbert(bandpass(x, 0.5 * f0, 1.5 * f0))))
        out[:, j] = ph - np.angle(Cg[j, 1])
    return out


def band_rms(Y, f0, lo_mult, hi_mult):
    """[6] per-joint RMS inside a harmonic band — the fusion weight."""
    out = np.zeros(6)
    for j in range(6):
        x = Y[:, j] - Y[:, j].mean()
        out[j] = float(np.sqrt(np.mean(bandpass(x, lo_mult * f0, hi_mult * f0) ** 2)))
    return out


def k2_phases(Y, tmpl_g, f0, PH1):
    """[T, 6] per-joint cycle phase from the k2 (step-frequency) band, plus its [6] band RMS."""
    from scipy.signal import hilbert
    Cg = np.fft.rfft(tmpl_g, axis=1)
    out = np.zeros_like(Y)
    amp = np.zeros(6)
    for j in range(6):
        x = Y[:, j] - Y[:, j].mean()
        b2 = bandpass(x, 1.6 * f0, 2.4 * f0)
        ph2 = np.unwrap(np.angle(hilbert(b2)))
        th2 = (ph2 - np.angle(Cg[j, 2])) / 2.0
        c0 = float(np.mean(np.cos(PH1[:, j] - th2)))
        c1 = float(np.mean(np.cos(PH1[:, j] - th2 - np.pi)))
        out[:, j] = th2 + (0.0 if c0 >= c1 else np.pi)
        amp[j] = float(np.sqrt(np.mean(b2 ** 2)))
    return out, amp


def fuse_phases(PH1, PH2, w1, w2):
    """Amplitude-weighted circular fusion of two per-joint phase estimates (k12)."""
    z = w1[None, :] * np.exp(1j * PH1) + w2[None, :] * np.exp(1j * PH2)
    return np.angle(z)


def cpg_blend(PH, lam):
    """Blend each joint's phase toward the pooled common phase by a fraction lam (a weak
    coordination prior)."""
    if lam <= 0:
        return PH
    z = np.exp(1j * PH)
    common = np.angle(z.mean(axis=1, keepdims=True))
    return np.angle((1.0 - lam) * z + lam * np.exp(1j * common))


def phases_to_template(PH, tmpl_n, delta):
    """[T, 6] deg (DATA convention): the normative template sampled at each joint's own cycle
    position."""
    grid = np.arange(50) / 50.0
    theta = PH / (2.0 * np.pi) - delta
    out = np.zeros_like(theta)
    for j in range(6):
        out[:, j] = np.interp(theta[:, j] % 1.0, grid, tmpl_n[j], period=1.0)
    return out


def phase_source(Y, tmpl_g, f0, source):
    """[T, 6] per-joint phase from the requested source: k1, k2, or their k12 fusion."""
    PH1 = raw_phases(Y, tmpl_g, f0)
    if source == "k1":
        return PH1
    PH2, w2 = k2_phases(Y, tmpl_g, f0, PH1)
    if source == "k2":
        return PH2
    if source == "k12":
        return fuse_phases(PH1, PH2, band_rms(Y, f0, 0.5, 1.5), w2)
    raise ValueError(f"unknown phase source {source!r}")


def f5_aligned(Y, tmpl_g, tmpl_n, delta, f0, source="k1", lam=0.0):
    """Rung 2 (f5n): each joint plays the NORMATIVE waveform at the cycle position read from
    ITS OWN channel of Y.  [T, 6] deg, DATA convention."""
    return phases_to_template(cpg_blend(phase_source(Y, tmpl_g, f0, source), lam),
                              tmpl_n, delta)


F7_WARM_S = 1.0
F7_BW, F7_ORDER, F7_NOTCH, F7_NQ = 0.5, 2, (2, 3), 1.0
F7_F0_PRIOR_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f0_cadence_prior.csv")
_F7_F0 = None


def f7_f0_prior(session):
    global _F7_F0
    if _F7_F0 is None:
        with open(F7_F0_PRIOR_CSV, newline="", encoding="utf-8") as fh:
            _F7_F0 = {r["session"]: float(r["f0_prior_hz"]) for r in csv.DictReader(fh)}
    if session not in _F7_F0:
        raise RuntimeError(f"{session} has no LOSO cadence prior in {F7_F0_PRIOR_CSV}")
    return _F7_F0[session]


def f7_causal_phases(Yin, tmpl_g, f0):
    """[T, 6] causal per-joint cycle phase (radians, the joint's k1 template angle removed, as raw_phases)."""
    from scipy.signal import butter, freqz, iirnotch, lfilter
    ch = [butter(F7_ORDER, [max((1 - F7_BW) * f0, 0.05) / (FS / 2), (1 + F7_BW) * f0 / (FS / 2)], "band")]
    for k in F7_NOTCH:
        ch.append(iirnotch(k * f0, F7_NQ, fs=FS))
    w = 2.0 * np.pi * f0 / FS
    bph = float(sum(np.angle(freqz(b, a, worN=[w])[1][0]) for b, a in ch))
    Cg = np.fft.rfft(tmpl_g, axis=1)
    PH = np.zeros_like(Yin)
    for j in range(6):
        xb = Yin[:, j] - Yin[0, j]
        for b, a in ch:
            xb = lfilter(b, a, xb)
        dx = np.concatenate([[0.0], np.diff(xb)])
        re = np.concatenate([[xb[0]], 0.5 * (xb[1:] + xb[:-1])])
        ph = np.angle(re + 1j * (-dx / (2 * np.sin(w / 2)))) + 0.5 * w - bph
        PH[:, j] = ph - np.angle(Cg[j, 1])
    return PH


def f7_causal_phases_warm(W, Yin, tmpl_g, f0):
    """GAIT_F7_WARM=apred: [len(W) + len(Yin), 6] causal phases of the concatenated stream [W ; Yin] (warm-up, then
    the main stream; raw concatenation, the filter states run on across the switch)."""
    X = np.vstack([W, Yin])
    nw = len(W)
    st = [int(np.flatnonzero(np.isfinite(W[:, j]))[0]) if np.isfinite(W[:, j]).any() else nw for j in range(6)]
    PH = np.full(X.shape, np.nan)
    for s in sorted(set(st)):
        P = f7_causal_phases(X[s:], tmpl_g, f0)
        for j in range(6):
            if st[j] == s:
                PH[s:, j] = P[:, j]
    return PH


def f7_init_phases(warm, Y, tmpl_g, session):
    """GAIT_F7_INIT: (template-clock phase [nw + n], extracted phase of the main stream [n, 6] or None) in the extracted
    phase's convention (phases_to_template(2 pi f0 t) = template_at(f0 t - delta) = data_io.metronome at f0)."""
    f0 = f7_f0_prior(session)
    nw = len(warm)
    if nw != F7_APRED_WARM_N:
        raise RuntimeError(f"GAIT_F7_INIT needs the full {F7_APRED_WARM_N}-sample warm-up (got {nw})")
    ptc = 2.0 * np.pi * f0 * np.arange(nw + len(Y)) / FS
    pex = f7_causal_phases_warm(warm, Y, tmpl_g, f0)[nw:] if F7_INIT in ("T1", "T2") else None
    return ptc, pex


def f7_decwarm_targets(warm, Y, tmpl_g, tmpl_n, delta, setpoint, session):
    """GAIT_F7_INIT=dec / decf / zp : model targets [nw + n, 6] rad driven from the warm-up start by the
    phase of the joined input [warm ; Y] itself (dec / decf: causal f7_causal_phases_warm; zp: zero-phase raw_phases),
    template de-mean, set-point; decf holds the first 0.5 s (F7_INIT_XF_N samples) at the set-point."""
    if len(warm) != F7_APRED_WARM_N or not np.isfinite(warm).all():
        raise RuntimeError(f"GAIT_F7_INIT={F7_INIT} needs the full finite {F7_APRED_WARM_N}-sample warm-up")
    f0 = f7_f0_prior(session)
    if F7_INIT == "zp":
        PH = raw_phases(np.vstack([warm, Y]), tmpl_g, f0)
    else:
        PH = f7_causal_phases_warm(warm, Y, tmpl_g, f0)
    traj = phases_to_template(PH, tmpl_n, delta)
    tg = np.radians((setpoint[None, :] + (traj - tmpl_n.mean(axis=1)[None, :])) * SIGN[None, :])
    if F7_INIT == "decf":
        tg[:F7_INIT_XF_N] = np.radians(setpoint * SIGN)[None, :]
    return tg


F7_FROM2S_COND = {"wphase_matched": "walker", "dphase": "decoded", "dphase_iscr": "iscr", "dphase_uscr": "uscr"}


def f7_from2s_inputs(cond, trial):
    """cond (F7_FROM2S_COND maps the producers' condition names): walker = the window-matched walker trial
    (replay_walker_matched.matched_trials: trial["W"] = its own goniometer over the 2 s before C_pred onset, trial["Y"] =
    the goniometer on the decoded window); decoded = trial of load_session_trials(stream="C_pred"): warm-up = A_pred over
    trial 2-4 s (trial["W"]), main stream = the level-matched C' = C - C[0] + W[-1] per joint; iscr / uscr = scramble /
    uscramble of the WHOLE joined stream [W ; C'] with seed = trial_idx, split back at len(W)."""
    if not F7_FROM2S:
        raise RuntimeError("f7_from2s_inputs is defined only with GAIT_F7_SIM=causal_from2s")
    Y0, W = trial["Y"], trial["W"]
    if len(W) != F7_APRED_WARM_N or not np.isfinite(W).all():
        raise RuntimeError(f"causal_from2s needs the full finite {F7_APRED_WARM_N}-sample warm-up "
                           f"({cond}, trial {trial['trial_idx']}: {len(W)} samples)")
    if cond == "walker":
        return W, Y0
    Cp = Y0 - Y0[0][None, :] + W[-1][None, :]
    if cond == "decoded":
        return W, Cp
    Jn = {"iscr": scramble, "uscr": uscramble}[cond](np.vstack([W, Cp]), seed=trial["trial_idx"])
    return Jn[:len(W)], Jn[len(W):]


def f7_init_targets(warm, Y, tmpl_g, tmpl_n, delta, setpoint, session):
    """GAIT_F7_INIT: model targets [nw + n, 6] rad = template clock over the warm-up, then T1 / T2 / coast / full."""
    if F7_INIT in ("dec", "decf", "zp"):
        return f7_decwarm_targets(warm, Y, tmpl_g, tmpl_n, delta, setpoint, session)
    nw = len(warm)
    ptc, pex = f7_init_phases(warm, Y, tmpl_g, session)
    PH = np.repeat(ptc[:, None], 6, axis=1)
    if F7_INIT == "T1":
        PH[nw:] = pex
    elif F7_INIT == "T2":
        w = np.minimum(np.arange(len(Y)) / F7_INIT_XF_N, 1.0)[:, None]
        PH[nw:] = PH[nw:] + w * np.angle(np.exp(1j * (pex - PH[nw:])))
    traj = phases_to_template(PH, tmpl_n, delta)
    tg = np.radians((setpoint[None, :] + (traj - tmpl_n.mean(axis=1)[None, :])) * SIGN[None, :])
    if F7_INIT == "coast":
        tg[nw:] = np.radians(setpoint * SIGN)[None, :]
    return tg


def f5_targets(Y, tmpl_g, tmpl_n, delta, f0, source, lam, setpoint, session, warm=None):
    """Model joint targets [T, 6] rad of the rung-2 (f5n) mapping. Default: to_targets(f5_aligned(...))."""
    if (warm is not None) != (F7_WARM or F7_FROM2S):
        raise RuntimeError("f5_targets: warm= is required with GAIT_F7_WARM=apred / GAIT_F7_SIM=causal_from2s and refused without it")
    if not ec.F7_SIM:
        return to_targets(f5_aligned(Y, tmpl_g, tmpl_n, delta, f0, source, lam), setpoint)
    if source != "k1" or float(lam) != 0.0:
        raise RuntimeError(f"GAIT_F7_SIM=causal defines only the k1 / lambda 0 mapping (asked {source} / {lam})")
    if F7_FROM2S:
        return f7_decwarm_targets(warm, Y, tmpl_g, tmpl_n, delta, setpoint, session)
    if F7_WARM:
        if len(warm) == 0:
            raise RuntimeError("GAIT_F7_WARM=apred: empty warm-up input (no samples before the stream onset)")
        if F7_INIT:
            return f7_init_targets(warm, Y, tmpl_g, tmpl_n, delta, setpoint, session)
        PH = f7_causal_phases_warm(warm, Y, tmpl_g, f7_f0_prior(session))[len(warm):]
        traj = phases_to_template(PH, tmpl_n, delta)
        return np.radians((setpoint[None, :] + (traj - tmpl_n.mean(axis=1)[None, :])) * SIGN[None, :])
    traj = phases_to_template(f7_causal_phases(Y, tmpl_g, f7_f0_prior(session)), tmpl_n, delta)
    tn_mean = tmpl_n.mean(axis=1)
    tg = np.radians((setpoint[None, :] + (traj - tn_mean[None, :])) * SIGN[None, :])
    tg[:min(int(round(F7_WARM_S * FS)), len(tg))] = np.radians(setpoint * SIGN)[None, :]
    return tg


def metronome(n_samples, tmpl_n, delta, f_hz=0.90):
    """Device baseline: the normative template replayed at a FIXED cadence, no brain signal at
    all."""
    t = np.arange(n_samples) / FS
    return template_at((t * f_hz - delta) % 1.0, tmpl_n)


def fidelity_by_trial(w_trials, d_trials):
    """G3' per-trial decode fidelity: mean over the six joints of the Pearson r between C_pred
    and Y_true, on the window where the two streams OVERLAP in absolute trial time."""
    out = {}
    W = {t["trial_idx"]: t for t in w_trials}
    for d in d_trials:
        w = W.get(d["trial_idx"])
        if w is None:
            continue
        a0 = max(w["t0_sec"], d["t0_sec"])
        a1 = min(w["t0_sec"] + w["n"] / FS, d["t0_sec"] + d["n"] / FS)
        n = int(round((a1 - a0) * FS))
        if n < int(FS):
            continue
        iw = int(round((a0 - w["t0_sec"]) * FS))
        idd = int(round((a0 - d["t0_sec"]) * FS))
        A, B = w["Y"][iw:iw + n], d["Y"][idd:idd + n]
        rs = []
        for j in range(6):
            a, b = A[:, j], B[:, j]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            rs.append(float(np.corrcoef(a, b)[0, 1]))
        if rs:
            out[d["trial_idx"]] = float(np.mean(rs))
    return out


def to_targets(Y, setpoint, gain=1.0):
    """Y [T,6] deg in DATA convention -> [T,6] rad of MODEL joint targets, actuator order
    (hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r)."""
    dev = Y - Y.mean(axis=0, keepdims=True)
    g = np.asarray(gain, float)
    g = g[None, :] if g.ndim == 1 else g
    return np.radians((setpoint[None, :] + g * dev) * SIGN[None, :])
