"""Decoder — per-fold harmonic band tables (02_decoding/fold_tables)

Per-fold gait frequency f0 and harmonic band centres fc_k1..fc_k3, estimated from the training
trials of each outer fold: healthy imagined walking and the patient sessions.

For every session of the first stage (whole-session band table; per-trial tables and feature
exports under data/decoder_output/), the whole-session band table is first recomputed from the
goniometer recordings with the procedure of core/estimate_goniometer_psd_peaks.m and checked
against the stored table and the session's feature export (trial set, goniometer windows, joint
indices, sample rate, band edges to 1e-9, the mean spectrum, the fold map); the run stops before
writing anything if one session fails. Then, for each outer fold of the random 10-fold
cross-validation, f0 and the three band centres are estimated again from all imagined-walking
trials except the trials tested in that fold (and, as a by-product, with each single trial left out).

Writes data/decoder_output/main/fold_tables/fold_centres_source_data.csv (one row per session and
fold: whole-session and held-out values and their difference, Hz) and fold_centres_metadata.csv
(summaries and the gate result); fold_tables/make_band_tables.py reads the first.
Usage, from the repository root: python 02_decoding/fold_tables/fold_centres.py

Reads:
  decoder output in data/decoder_output/ (not distributed)

Part of the decoder's second stage, the per-fold harmonic band tables (README.md,
Decoding, Step 09); it fits nothing.
"""

import csv
import glob
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

R = os.path.join(ROOT, "data", "decoder_output")
EXPORT_DIR = {"healthy": os.path.join(R, "pyport_export", "h_cdm_o1_c0p18_causal_ownlso"),
              "patient": os.path.join(R, "pyport_export", "cdm_o1_c0p18_causal_ownhglobal")}
BAND_HEALTHY = os.path.join(R, "diag_main_goni_psd_peaks", "main_goni_peak_bands.csv")
PSD_HEALTHY = os.path.join(R, "diag_main_goni_psd_peaks", "main_goni_psd_summary.csv")
PAT_BANDDIR = {"pat02_sess08": "diag_patient_goni_psd_peaks_sess08",
               "pat02_sess09": "diag_patient_goni_psd_peaks_sess09"}
PAT_BANDDIR_DEFAULT = "diag_patient_goni_psd_peaks"
COHORT_DUMPS = (("healthy", "h_iw_dump"), ("post_operative", "pat_post_operative_iw"), ("pre_operative", "pat_pre_operative_iw"))
W_X = ("LHipW X", "LKneW X", "LAnkW X", "RHipW X", "RKneW X", "RAnkW X")
FMAX, HALF_WIDTH, F0_BAND, HARM_HALF, NFFT = 4.0, 0.18, (0.45, 1.15), 0.25, 8192
LPF_ORDER, LPF_CUT = 1, 0.18
ZERO_TOL = 1e-9


def matlab_colon(a, d, b):
    """MATLAB a:d:b (colonop: first half a + k*d, second half b - k*d, even-n midpoint (a+b)/2)."""
    n = int(round((b - a) / d))
    v = np.empty(n + 1)
    k = np.arange(n // 2 + 1)
    v[k] = a + k * d
    v[n - k] = b - k * d
    if n % 2 == 0:
        v[n // 2] = (a + b) / 2
    return v


FREQ_GRID = matlab_colon(0.0, 0.01, FMAX)


def h5str(ds):
    return "".join(chr(c) for c in np.asarray(ds[()]).ravel() if c)


def load_export(path):
    with h5py.File(path, "r") as h:
        row = lambda k: np.asarray(h[k][()]).ravel()
        qc = h["goni_qc_flags_env"]
        return dict(label=h5str(h["session_label"]), n_ev=int(row("n_ev")[0]), fa=row("fa_fold").astype(int),
                    Tk=row("Tk").astype(int), ev_idx=row("ev_trial_idx").astype(int),
                    gtstart=row("gtstart").astype(int), gtstop=row("gtstop").astype(int),
                    joint_idx=row("joint_idx"), bands=np.asarray(h["bands"][()]),
                    root=h5str(h["gait_triple_root_env"]), fs_goni=float(row("fs_goni")[0]),
                    qc_env=h5str(qc) if qc.dtype.kind == "u" and qc.dtype.itemsize == 2 else "")


def load_events(base, label):
    p = os.path.join(base, label + "_event.mat")
    try:
        e = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
        ev = np.atleast_1d(e["event"] if "event" in e else e["events"])
        return [dict(code=str(x.code), qc=str(x.qc), gtstart=int(x.gtstart), gtstop=int(x.gtstop)) for x in ev]
    except NotImplementedError:
        with h5py.File(p, "r") as h:
            g = h["event"] if "event" in h else h["events"]
            def col(f):
                return [h[r] for r in np.asarray(g[f][()]).ravel()]
            codes, qcs, g0, g1 = col("code"), col("qc"), col("gtstart"), col("gtstop")
            return [dict(code=h5str(c), qc=h5str(q), gtstart=int(np.ravel(a[()])[0]), gtstop=int(np.ravel(b[()])[0]))
                    for c, q, a, b in zip(codes, qcs, g0, g1)]


def apply_qc_override(events, label, qc_csv):
    """shared/load_triple_mat.m apply_external_goni_qc: ok -> good, problem -> bad, skip -> skip (1-based trial)."""
    if not qc_csv:
        return events
    for r in csv.DictReader(open(qc_csv, newline="")):
        j = int(r["trial"])
        if r["label"] == label and 1 <= j <= len(events):
            events[j - 1]["qc"] = {"ok": "good", "problem": "bad", "skip": "skip"}[r["status"]]
    return events


def load_goni_rows(base, label, want):
    """-> (labels, {label: 1-D row}) for the wanted goniometer labels (find_goni_idx: exact, then case-insensitive)."""
    p = os.path.join(base, label + "_goni.mat")
    try:
        with h5py.File(p, "r") as h:
            g = h["goni"]
            labels = [h5str(h[r]) for r in np.asarray(g["c"][()]).ravel()]
            fs = float(np.ravel(g["s"][()])[0])
            idx = {w: find_idx(labels, w) for w in want}
            rows = {w: np.asarray(g["x"][i, :], dtype=float) for w, i in idx.items()}
    except OSError:
        g = sio.loadmat(p, squeeze_me=True, struct_as_record=False)["goni"]
        labels, fs = [str(c) for c in np.atleast_1d(g.c)], float(g.s)
        idx = {w: find_idx(labels, w) for w in want}
        rows = {w: np.asarray(g.x[:, i], dtype=float) for w, i in idx.items()}
    return labels, idx, rows, fs


def find_idx(labels, target):
    s = [x.strip() for x in labels]
    for cmp in (lambda a: a == target.strip(), lambda a: a.lower() == target.strip().lower()):
        hit = [i for i, a in enumerate(s) if cmp(a)]
        if hit:
            return hit[0]
    raise KeyError(target)


def trace_psds(rows, ev, fs):
    """Per trial x joint normalised PSD on FREQ_GRID (session_walkerx_psd_local). -> (P [n x 401], owner ev position)."""
    P, owner = [], []
    for t, e in enumerate(ev):
        for w in W_X:
            y = rows[w][e["gtstart"] - 1:e["gtstop"]]
            y = y[np.isfinite(y)]
            if y.size < round(2 * fs):
                continue
            y = y - y.mean()
            win = int(min(round(4 * fs), max(round(2 * fs), y.size // 2)))
            f, pxx = signal.welch(y, fs=fs, window=signal.windows.hamming(win, sym=True), nperseg=win,
                                  noverlap=win // 2, nfft=NFFT, detrend=False, return_onesided=True,
                                  scaling="density", average="mean")
            keep = (f >= 0.05) & (f <= FMAX)
            denom = np.trapezoid(pxx[keep], f[keep])
            if not np.isfinite(denom) or denom <= 0:
                continue
            P.append(np.interp(FREQ_GRID, f, pxx / denom))
            owner.append(t)
    return np.array(P), np.array(owner)


def peak_in_band(p, lo, hi):
    m = (FREQ_GRID >= lo) & (FREQ_GRID <= hi) & np.isfinite(p)
    return FREQ_GRID[m][np.argmax(p[m])] if m.any() else np.nan


def calibrate(p):
    """-> f0 and per harmonic (target, peak, band_lo, band_hi, fc)."""
    f0 = peak_in_band(p, *F0_BAND)
    out = []
    for k in (1, 2, 3):
        target = k * f0
        pk = peak_in_band(p, max(0.1, target - HARM_HALF), min(FMAX, target + HARM_HALF))
        lo, hi = max(0.05, pk - HALF_WIDTH), min(FMAX, pk + HALF_WIDTH)
        out.append((target, pk, lo, hi, 0.5 * (lo + hi)))
    return f0, out


def band_table(label):
    if label.startswith("pat"):
        d = os.path.join(R, PAT_BANDDIR.get(label, PAT_BANDDIR_DEFAULT))
        return os.path.join(d, "patient_goni_peak_bands.csv"), os.path.join(d, "patient_goni_psd_summary.csv")
    return BAND_HEALTHY, PSD_HEALTHY


def trial_r_folds(dump_dir, label):
    tr = pd.read_csv(os.path.join(dump_dir, label + "_trial_r.csv"))
    return tr.groupby("trial_idx")["fold"].apply(lambda s: sorted(set(int(x) for x in s)))


def one_session(job):
    cohort, label, dump_dir = job
    E = load_export(os.path.join(EXPORT_DIR["patient" if cohort != "healthy" else "healthy"], label + "_export.mat"))
    assert E["label"] == label, (E["label"], label)
    base = os.path.join(E["root"], label)
    events = apply_qc_override(load_events(base, label), label, E["qc_env"])
    ev_pos = [j for j, e in enumerate(events) if e["code"] == "gaitimagery" and e["qc"] == "good"]
    ev = [events[j] for j in ev_pos]
    gate = []
    gate.append(("ev_trial_idx", len(ev) == E["n_ev"] and np.array_equal(np.array(ev_pos) + 1, E["ev_idx"])
                 and np.array_equal([e["gtstart"] for e in ev], E["gtstart"])
                 and np.array_equal([e["gtstop"] for e in ev], E["gtstop"])))
    labels, idx, rows, fs = load_goni_rows(base, label, W_X)
    gate.append(("joint_idx", np.array_equal(np.array([idx[w] + 1 for w in W_X]), E["joint_idx"])))
    gate.append(("fs", fs == E["fs_goni"]))
    P, owner = trace_psds(rows, ev, fs)
    pm = P.mean(axis=0)
    f0, H = calibrate(pm)
    bcsv, scsv = band_table(label)
    B = pd.read_csv(bcsv)
    B = B[B.session == label].sort_values("harmonic_k")
    ok_tab = len(B) == 3 and all(
        abs(h[0] - b.target_hz) < 1e-9 and abs(h[1] - b.peak_hz) < 1e-9 and abs(h[2] - b.band_lo) < 1e-9
        and abs(h[3] - b.band_hi) < 1e-9 and int(b.n_psd) == len(P) for h, b in zip(H, B.itertuples()))
    gate.append(("band_table", ok_tab))
    S = pd.read_csv(scsv)
    S = S[S.session == label].sort_values("freq_hz")
    psd_dev = float(np.max(np.abs(S.mean_psd.to_numpy() - pm)) / np.max(pm)) if len(S) == len(pm) else np.inf
    gate.append(("psd_summary", psd_dev < 1e-8))
    eb = E["bands"]
    gate.append(("export_bands", np.array_equal(eb[0], [1, 2, 3]) and np.allclose(eb[1], B.band_lo, rtol=0, atol=1e-12)
                 and np.allclose(eb[2], B.band_hi, rtol=0, atol=1e-12)))
    trf = trial_r_folds(dump_dir, label)
    tested = {t for t in range(E["n_ev"]) if E["Tk"][t] > 0}
    gate.append(("fold_map", set(int(i) - 1 for i in trf.index) == tested
                 and all(v == [E["fa"][int(i) - 1]] for i, v in trf.items())))
    whole = (f0, [h[4] for h in H])
    res = dict(cohort=cohort, session=label, gate=gate, psd_dev=psd_dev, n_ev=E["n_ev"], n_traces=len(P),
               n_tk0=E["n_ev"] - len(tested), f0_all=f0, H_all=H, rows=[])

    def variant(kind, fold, dropped, drop_set):
        m = ~np.isin(owner, list(drop_set))
        f0h, Hh = calibrate(P[m].mean(axis=0))
        r = dict(kind=kind, cohort=cohort, session=label, fold=fold, dropped_trial_idx=dropped,
                 n_heldout_trials=len(drop_set), n_pool_trials=int(len(np.unique(owner[m]))), n_pool_traces=int(m.sum()),
                 f0_all=round(whole[0], 10), f0_held=round(f0h, 10))
        r["d_f0"] = round(r["f0_held"] - r["f0_all"], 10)
        for k in range(3):
            a, b = round(whole[1][k], 10), round(Hh[k][4], 10)
            r.update({f"fc_k{k + 1}_all": a, f"fc_k{k + 1}_held": b, f"d_fc_k{k + 1}": round(b - a, 10)})
        res["rows"].append(r)

    for fold in sorted(set(E["fa"].tolist())):
        variant("fold", fold, "", {t for t in tested if E["fa"][t] == fold})
    for t in range(E["n_ev"]):
        variant("single_trial", "", t + 1, {t})
    return res


def butter_response(delta_hz, fs):
    b, a = signal.butter(LPF_ORDER, LPF_CUT / (fs / 2))
    _, h = signal.freqz(b, a, worN=[abs(delta_hz)], fs=fs)
    return float(np.abs(h[0])), float(np.degrees(np.abs(np.angle(h[0]))))


def summarise(df, n_sessions, prefix, md):
    qty = (("f0", "d_f0"), ("fc_k1", "d_fc_k1"), ("fc_k2", "d_fc_k2"), ("fc_k3", "d_fc_k3"))
    for name, col in qty:
        a = df[col].abs().to_numpy()
        chg = df.loc[df[col].abs() > ZERO_TOL, "session"].nunique()
        md += [(f"{prefix} {name}: n_rows", len(a)),
               (f"{prefix} {name}: median |held - all| Hz", round(float(np.median(a)), 6)),
               (f"{prefix} {name}: p95 |held - all| Hz", round(float(np.percentile(a, 95)), 6)),
               (f"{prefix} {name}: max |held - all| Hz", round(float(a.max()), 6)),
               (f"{prefix} {name}: rows exactly 0", f"{int(np.sum(a <= ZERO_TOL))} of {len(a)}"),
               (f"{prefix} {name}: fraction exactly 0", round(float(np.mean(a <= ZERO_TOL)), 6)),
               (f"{prefix} {name}: sessions with any change", f"{chg} of {n_sessions}")]
    anyc = df[[c for _, c in qty]].abs().max(axis=1) > ZERO_TOL
    md.append((f"{prefix} any of f0/fc_k1-3: sessions with any change", f"{df.loc[anyc, 'session'].nunique()} of {n_sessions}"))
    md.append((f"{prefix} max |change| over f0 and fc_k1-3 Hz", round(float(df[[c for _, c in qty]].abs().to_numpy().max()), 6)))


def write(kind, df):
    out = os.path.join(R, "main", "fold_tables")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"fold_centres_{kind}.csv")
    df.to_csv(p + ".tmp", index=False)
    assert len(pd.read_csv(p + ".tmp")) == len(df)
    os.replace(p + ".tmp", p)
    return p


def main():
    jobs = []
    for cohort, key in COHORT_DUMPS:
        d = os.path.join(R, "session_band_table", key)
        for f in sorted(glob.glob(os.path.join(d, "*_trial_r.csv"))):
            jobs.append((cohort, os.path.basename(f)[:-len("_trial_r.csv")], d))
    with Pool(int(os.environ.get("F0_WORKERS", "8"))) as pool:
        res = pool.map(one_session, jobs)
    fails = [(r["session"], [g for g, ok in r["gate"] if not ok]) for r in res if not all(ok for _, ok in r["gate"])]
    for r in res:
        print("GATE %-14s %-8s %s  f0=%.2f n_traces=%d psd_dev=%.2e" % (
            r["session"], r["cohort"], "PASS" if all(ok for _, ok in r["gate"]) else
            "FAIL " + ",".join(g for g, ok in r["gate"] if not ok), r["f0_all"], r["n_traces"], r["psd_dev"]))
    counts = {c: sum(r["cohort"] == c for r in res) for c, _ in COHORT_DUMPS}
    print("GATE summary: %d sessions (%s), %d failing" % (len(res), counts, len(fails)))
    if fails:
        raise SystemExit("reproduction gate FAILED: %s" % fails)
    sd = pd.DataFrame([row for r in res for row in r["rows"]])
    fs = 250.0
    md = [("script", "02_decoding/fold_tables/fold_centres.py"),
          ("definition", "f0 = argmax of the session goniometer PSD (mean over imagined-walking trials x reference-walker "
                         "joints of unit-area Welch PSDs on a 0.01 Hz grid) in 0.45-1.15 Hz; fc_kK = centre of the EEG "
                         "feature band = argmax in K*f0 +/- 0.25 Hz (the demodulator centre of the cdm features); held = "
                         "the same procedure without the held-out trials (fold: the trials tested in that outer fold of "
                         "the random 10-fold cross-validation; single_trial: one imagined-walking trial); "
                         "d = held - all, Hz"),
          ("fold_map", "fa_fold of the first-stage feature exports, equal to the fold column of every tested trial in "
                       "the first-stage per-trial tables (data/decoder_output/session_band_table/)"),
          ("cohort", "healthy cohort (80 sessions / 42 participants); patients post_operative (11 sessions) and pre_operative "
                     "(4 sessions)"),
          ("paper_location", "no number of the paper: input of the second decoder stage (fold_tables/make_band_tables.py)"),
          ("band_table_producer", "02_decoding/core/estimate_goniometer_psd_peaks.m"),
          ("whole_session_tables", "healthy " + os.path.relpath(BAND_HEALTHY, ROOT).replace(os.sep, "/") + "; patients "
                                   "data/decoder_output/diag_patient_goni_psd_peaks{,_sess08,_sess09}/"
                                   "patient_goni_peak_bands.csv"),
          ("trials_pooled", "imagined-walking trials only (code gaitimagery, qc good; patients after the goniometer " "QC override); no real-walking or rest trials"),
          ("gate", "PASS %d of %d sessions: trial set and goniometer windows = export; joint indices = export; "
                   "target_hz/peak_hz/band_lo/band_hi within 1e-9 and n_psd exact vs the stored tables; mean PSD vs the "
                   "stored summary (max relative deviation %.2e); export bands = stored bands; fold map = trial_r.csv"
                   % (len(res), len(res), max(r["psd_dev"] for r in res))),
          ("trials_with_Tk0_kept_in_every_pool", sum(r["n_tk0"] for r in res)),
          ("grid_step_hz", 0.01),
          ("fft_bin_hz", fs / NFFT),
          ("welch_window_max_s", 4.0),
          ("welch_window_rayleigh_resolution_hz", 1 / 4.0),
          ("eeg_feature_filter", "cdm_o1_c0p18_causal: 1st-order Butterworth low-pass, cut-off 0.18 Hz (-3 dB), shifted to fc"),
          ("zero_tolerance_hz", ZERO_TOL)]
    groups = (("healthy", ["healthy"]), ("patients (post_operative+pre_operative)", ["post_operative", "pre_operative"]),
              ("post_operative", ["post_operative"]), ("pre_operative", ["pre_operative"]))
    for gname, coh in groups:
        nsess = sum(counts[c] for c in coh)
        for kind in ("fold", "single_trial"):
            sub = sd[(sd.kind == kind) & sd.cohort.isin(coh)]
            summarise(sub, nsess, f"{gname} {kind}", md)
            dmax = float(sub[["d_f0", "d_fc_k1", "d_fc_k2", "d_fc_k3"]].abs().to_numpy().max())
            g, ph = butter_response(dmax, fs)
            md += [(f"{gname} {kind}: cdm low-pass gain at the largest centre shift", g),
                   (f"{gname} {kind}: cdm low-pass phase at the largest centre shift deg", ph)]
    g, ph = butter_response(0.01, fs)
    md += [("cdm low-pass gain at one grid step (0.01 Hz)", g), ("cdm low-pass phase at one grid step (0.01 Hz) deg", ph)]
    md = pd.DataFrame(md, columns=["key", "value"])
    for k, df in (("source_data", sd), ("metadata", md)):
        print("wrote", os.path.relpath(write(k, df), ROOT))
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 120); pd.set_option("display.max_rows", 500)
    print(md.to_string(index=False))


if __name__ == "__main__":
    main()
