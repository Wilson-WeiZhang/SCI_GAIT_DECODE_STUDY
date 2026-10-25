"""Decoder — per-fold harmonic band tables (02_decoding/fold_tables)

Per-fold band tables of the other runs of the second stage, estimated from the training trials
of each fold: real walking (the participant's own goniometer, random 10-fold) and the
contiguous-fold control of imagined walking (ten temporally contiguous folds).

Same procedure as fold_tables/fold_centres.py: the pool of fold f is every trial of the condition
except the trials tested in fold f. Before anything is written it checks, per session, the trial
set, goniometer windows, joint indices and sample rate against the feature export, the
whole-session table against the stored one, the fold map against the first-stage per-trial table
and, as a writer check, that the whole-session centres reproduce the stored band edges exactly.
Mode iw writes nothing: it must reproduce the imagined-walking fold rows of
fold_centres_source_data.csv exactly, the gate that ties the two estimators together.

Writes data/decoder_output/main/fold_tables/<rw|block>/: fold_centres.csv, bands/<table id>.csv and
tables.csv (block: a table whose centres equal an imagined-walking table names that table's
export in reuse_export, so no new export is needed).
Usage, from the repository root: python 02_decoding/fold_tables/make_fold_tables.py <iw|rw|block> [workers]

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

ROOT = os.getcwd()
assert os.path.isdir(os.path.join(ROOT, "02_decoding")), "run from the repository root"
R = os.path.join(ROOT, "data", "decoder_output")
R13 = os.path.join(R, "session_band_table")
OUT_REL = "result/main/fold_tables"
OUT = os.path.join(R, "zero_phase_prep", "fold_tables")
F0F_SD = os.path.join(OUT, "fold_centres_source_data.csv")
F0F_TABLES = os.path.join(OUT, "imagined_walking", "tables.csv")
EXP_IW = os.path.join(R, "pyport_export", "h_cdm_o1_c0p18_causal_ownlso")
EXP_RW = os.path.join(R, "pyport_export", "h_rw_cdm_o1_c0p18_causal_ownlso_exec")
MODES = {
    "iw": dict(code="gaitimagery", joints=("LHipW X", "LKneW X", "LAnkW X", "RHipW X", "RKneW X", "RAnkW X"),
               export=EXP_IW, dump=os.path.join(R13, "h_iw_dump"), split="random",
               bands=os.path.join(R, "diag_main_goni_psd_peaks", "main_goni_peak_bands.csv"),
               psd=os.path.join(R, "diag_main_goni_psd_peaks", "main_goni_psd_summary.csv")),
    "rw": dict(code="gaitexecution", joints=("LHipS X", "LKneS X", "LAnkS X", "RHipS X", "RKneS X", "RAnkS X"),
               export=EXP_RW, dump=os.path.join(R13, "h_rw_dump"), split="random",
               bands=os.path.join(R, "diag_main_exec_goni_psd_peaks", "main_exec_goni_peak_bands.csv"),
               psd=None),
    "block": dict(code="gaitimagery", joints=("LHipW X", "LKneW X", "LAnkW X", "RHipW X", "RKneW X", "RAnkW X"),
                  export=EXP_IW, dump=os.path.join(R13, "h_blockcv_dump"), split="block",
                  bands=os.path.join(R, "diag_main_goni_psd_peaks", "main_goni_peak_bands.csv"),
                  psd=os.path.join(R, "diag_main_goni_psd_peaks", "main_goni_psd_summary.csv")),
}
TID_TAG = {"rw": "rwt", "block": "bkt"}
FMAX, HALF, F0_BAND, HARM_HALF, NFFT, TOL = 4.0, 0.18, (0.45, 1.15), 0.25, 8192, 1e-9


def matlab_colon(a, d, b):
    n = int(round((b - a) / d))
    v = np.empty(n + 1)
    k = np.arange(n // 2 + 1)
    v[k] = a + k * d
    v[n - k] = b - k * d
    if n % 2 == 0:
        v[n // 2] = (a + b) / 2
    return v


GRID = matlab_colon(0.0, 0.01, FMAX)


def mround(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def block_folds(n, outer_k=10):
    """pyport decoder.block_folds (run_decoder apply_cv_mode_local 'block')."""
    K = min(int(mround(outer_k)), n)
    edges = mround(np.arange(K + 1) * float(n) / K).astype(np.int64)
    edges[0], edges[-1] = 0, n
    fold = np.zeros(n, dtype=np.int64)
    for k in range(1, K + 1):
        fold[edges[k - 1]:edges[k]] = k
    return fold


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


def find_idx(labels, target):
    s = [x.strip() for x in labels]
    for cmp in (lambda a: a == target.strip(), lambda a: a.lower() == target.strip().lower()):
        hit = [i for i, a in enumerate(s) if cmp(a)]
        if hit:
            return hit[0]
    raise KeyError(target)


def load_goni_rows(base, label, want):
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
    return idx, rows, fs


def trace_psds(rows, ev, fs, joints):
    P, owner = [], []
    for t, e in enumerate(ev):
        for w in joints:
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
            P.append(np.interp(GRID, f, pxx / denom))
            owner.append(t)
    return np.array(P), np.array(owner)


def peak_in_band(p, lo, hi):
    m = (GRID >= lo) & (GRID <= hi) & np.isfinite(p)
    return GRID[m][np.argmax(p[m])] if m.any() else np.nan


def calibrate(p):
    f0 = peak_in_band(p, *F0_BAND)
    out = []
    for k in (1, 2, 3):
        target = k * f0
        pk = peak_in_band(p, max(0.1, target - HARM_HALF), min(FMAX, target + HARM_HALF))
        lo, hi = max(0.05, pk - HALF), min(FMAX, pk + HALF)
        out.append((target, pk, lo, hi, 0.5 * (lo + hi)))
    return f0, out


def one_session(job):
    mode, label = job
    M = MODES[mode]
    E = load_export(os.path.join(M["export"], label + "_export.mat"))
    assert E["label"] == label
    base = os.path.join(E["root"], label)
    events = load_events(base, label)
    ev_pos = [j for j, e in enumerate(events) if e["code"] == M["code"] and e["qc"] == "good"]
    ev = [events[j] for j in ev_pos]
    gate = []
    gate.append(("ev_trial_idx", len(ev) == E["n_ev"] and np.array_equal(np.array(ev_pos) + 1, E["ev_idx"])
                 and np.array_equal([e["gtstart"] for e in ev], E["gtstart"])
                 and np.array_equal([e["gtstop"] for e in ev], E["gtstop"])))
    idx, rows, fs = load_goni_rows(base, label, M["joints"])
    gate.append(("joint_idx", np.array_equal(np.array([idx[w] + 1 for w in M["joints"]]), E["joint_idx"])))
    gate.append(("fs", fs == E["fs_goni"]))
    P, owner = trace_psds(rows, ev, fs, M["joints"])
    pm = P.mean(axis=0)
    f0, H = calibrate(pm)
    B = pd.read_csv(M["bands"])
    B = B[B.session == label].sort_values("harmonic_k")
    gate.append(("band_table", len(B) == 3 and all(
        abs(h[0] - b.target_hz) < TOL and abs(h[1] - b.peak_hz) < TOL and abs(h[2] - b.band_lo) < TOL
        and abs(h[3] - b.band_hi) < TOL and int(b.n_psd) == len(P) for h, b in zip(H, B.itertuples()))))
    psd_dev = np.nan
    if M["psd"]:
        S = pd.read_csv(M["psd"])
        S = S[S.session == label].sort_values("freq_hz")
        psd_dev = float(np.max(np.abs(S.mean_psd.to_numpy() - pm)) / np.max(pm)) if len(S) == len(pm) else np.inf
        gate.append(("psd_summary", psd_dev < 1e-8))
    eb = E["bands"]
    gate.append(("export_bands", np.array_equal(eb[0], [1, 2, 3]) and np.allclose(eb[1], B.band_lo, rtol=0, atol=1e-12)
                 and np.allclose(eb[2], B.band_hi, rtol=0, atol=1e-12)))
    fa = E["fa"] if M["split"] == "random" else block_folds(E["n_ev"], 10)
    tr = pd.read_csv(os.path.join(M["dump"], label + "_trial_r.csv"))
    trf = tr.groupby("trial_idx")["fold"].apply(lambda s: sorted(set(int(x) for x in s)))
    tested = {t for t in range(E["n_ev"]) if E["Tk"][t] > 0}
    gate.append(("fold_map", set(int(i) - 1 for i in trf.index) == tested
                 and all(v == [int(fa[int(i) - 1])] for i, v in trf.items())))
    whole = (f0, [h[4] for h in H])
    out = []
    for fold in sorted(set(int(x) for x in fa.tolist())):
        drop = {t for t in tested if fa[t] == fold}
        if not drop:
            continue
        m = ~np.isin(owner, list(drop))
        f0h, Hh = calibrate(P[m].mean(axis=0))
        r = dict(mode=mode, session=label, fold=fold, n_heldout_trials=len(drop),
                 n_pool_trials=int(len(np.unique(owner[m]))), n_pool_traces=int(m.sum()),
                 f0_all=round(whole[0], 10), f0_held=round(f0h, 10))
        r["d_f0"] = round(r["f0_held"] - r["f0_all"], 10)
        for k in range(3):
            a, b = round(whole[1][k], 10), round(Hh[k][4], 10)
            r.update({f"fc_k{k + 1}_all": a, f"fc_k{k + 1}_held": b, f"d_fc_k{k + 1}": round(b - a, 10)})
        out.append(r)
    return dict(session=label, gate=gate, psd_dev=psd_dev, rows=out, n_traces=len(P), f0=f0,
                stored=[(float(b.band_lo), float(b.band_hi)) for b in B.itertuples()])


def fmt(x):
    return "%.15g" % x


def band_rows(session, f0, fcs, n_psd):
    rows = []
    for k, fc in zip((1, 2, 3), fcs):
        pk = round(fc, 2)
        assert abs(pk - fc) < 1e-9, (session, k, fc)
        lo, hi = pk - HALF, pk + HALF
        assert lo >= 0.05 and hi <= 4.0, (session, k, lo, hi)
        rows.append(dict(session=session, harmonic_k=k, target_hz=fmt(round(k * f0, 10)), peak_hz=fmt(pk),
                         band_lo=fmt(lo), band_hi=fmt(hi), n_psd=int(n_psd)))
    return rows


def write_csv(path, rows):
    tmp = path + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False, lineterminator="\n")
    assert len(pd.read_csv(tmp)) == len(rows)
    os.replace(tmp, path)


def main():
    mode = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    M = MODES[mode]
    labels = sorted(os.path.basename(f)[:-len("_trial_r.csv")] for f in glob.glob(os.path.join(M["dump"], "*_trial_r.csv")))
    assert len(labels) == 80, (mode, len(labels))
    with Pool(workers) as pool:
        res = pool.map(one_session, [(mode, s) for s in labels])
    fails = [(r["session"], [g for g, ok in r["gate"] if not ok]) for r in res if not all(ok for _, ok in r["gate"])]
    for r in res:
        print("GATE %-14s %s f0=%.2f n_traces=%d psd_dev=%.2e" % (
            r["session"], "PASS" if all(ok for _, ok in r["gate"]) else
            "FAIL " + ",".join(g for g, ok in r["gate"] if not ok), r["f0"], r["n_traces"], r["psd_dev"]))
    print("GATE summary %s: %d sessions, %d failing" % (mode, len(res), len(fails)))
    if fails:
        raise SystemExit("gate FAILED: %s" % fails)
    sd = pd.DataFrame([row for r in res for row in r["rows"]])
    n_ok = 0
    for r in res:
        g = sd[sd.session == r["session"]].iloc[0]
        for row, (lo, hi) in zip(band_rows(r["session"], g.f0_all, (g.fc_k1_all, g.fc_k2_all, g.fc_k3_all), 0),
                                 r["stored"]):
            assert float(row["band_lo"]) == lo and float(row["band_hi"]) == hi, (r["session"], row, lo, hi)
            n_ok += 1
    print("writer check: %d (session, harmonic) rows reproduce the stored band_lo / band_hi exactly" % n_ok)
    sd["changed"] = sd[["d_fc_k1", "d_fc_k2", "d_fc_k3"]].abs().max(axis=1) > TOL
    if mode == "iw":
        ref = pd.read_csv(F0F_SD)
        ref = ref[(ref.kind == "fold") & (ref.cohort == "healthy")].copy()
        ref["fold"] = ref.fold.astype(int)
        mm = sd.merge(ref, on=["session", "fold"], suffixes=("", "_ref"))
        cols = ["f0_all", "f0_held", "fc_k1_held", "fc_k2_held", "fc_k3_held", "n_pool_traces"]
        dev = max(float(np.max(np.abs(mm[c].to_numpy(float) - mm[c + "_ref"].to_numpy(float)))) for c in cols)
        print("iw gate vs fold_centres_source_data.csv: rows %d (ref %d), max |dev| %.3g, changed folds %d (expected 146)"
              % (len(mm), len(ref), dev, int(sd.changed.sum())))
        if len(mm) != len(ref) or len(mm) != len(sd) or dev > 1e-9 or int(sd.changed.sum()) != 146:
            raise SystemExit("iw gate FAILED")
        print("iw gate PASS (nothing written)")
        return
    od = os.path.join(OUT, mode)
    os.makedirs(os.path.join(od, "bands"), exist_ok=True)
    f0f = pd.read_csv(F0F_TABLES) if mode == "block" else None
    jobs = []
    for s, g in sd[sd.changed].groupby("session", sort=True):
        key = g[["fc_k1_held", "fc_k2_held", "fc_k3_held"]].round(10).apply(tuple, axis=1)
        for t, (fcs, gg) in enumerate(sorted(g.groupby(key), key=lambda kv: kv[1].fold.min()), start=1):
            assert gg.f0_held.nunique() == 1
            tid = "%s__%s%d" % (s, TID_TAG[mode], t)
            reuse = ""
            if f0f is not None:
                hit = f0f[(f0f.session == s) & (f0f.fc_k1.round(10) == fcs[0]) & (f0f.fc_k2.round(10) == fcs[1])
                          & (f0f.fc_k3.round(10) == fcs[2])]
                if len(hit):
                    reuse = hit.export_dir.iloc[0]
            write_csv(os.path.join(od, "bands", tid + ".csv"),
                      band_rows(s, gg.f0_held.iloc[0], fcs, gg.n_pool_traces.iloc[0]))
            jobs.append(dict(table_id=tid, mode=mode, session=s, folds=";".join(str(f) for f in sorted(gg.fold)),
                             n_folds=len(gg), f0_held=gg.f0_held.iloc[0], fc_k1=fcs[0], fc_k2=fcs[1], fc_k3=fcs[2],
                             band_csv="%s/%s/bands/%s.csv" % (OUT_REL, mode, tid),
                             export_dir="%s/%s/export/%s" % (OUT_REL, mode, tid), reuse_export=reuse))
    write_csv(os.path.join(od, "fold_centres.csv"), sd.to_dict("records"))
    write_csv(os.path.join(od, "tables.csv"), jobs)
    J = pd.DataFrame(jobs)
    print("%s: folds %d, changed %d in %d sessions; tables %d (reuse an imagined-walking export: %d)" % (
        mode, len(sd), int(sd.changed.sum()), sd[sd.changed].session.nunique(), len(J),
        int((J.reuse_export != "").sum()) if len(J) else 0))


if __name__ == "__main__":
    sys.exit(main())
