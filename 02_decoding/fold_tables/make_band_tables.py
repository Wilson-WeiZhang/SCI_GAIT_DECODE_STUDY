"""Decoder — per-fold harmonic band tables (02_decoding/fold_tables)

Band tables of the folds whose held-out band centres differ from the whole-session centres
(healthy imagined walking and the patient sessions), read from fold_centres_source_data.csv.

A fold gets its own table when one of its three band centres differs from the whole-session
centre by more than 1e-9 Hz; folds of one session with identical held-out centres share a table.
Each table has the layout of the whole-session band table (band = peak +/- 0.18 Hz, peak = the
held-out centre), so the unchanged MATLAB feature export reads it through bp_band_csv. A writer
check first rebuilds the band edges of every session from its whole-session centres and requires
the stored values exactly. One control table carries the whole-session centres of one session:
its export must reproduce that session's first-stage export.

Writes data/decoder_output/main/fold_tables/imagined_walking/bands/<table id>.csv and tables.csv
(one row per table: session, folds, held-out f0 and centres, band csv, export folder).
Usage, from the repository root: python 02_decoding/fold_tables/make_band_tables.py

Reads:
  decoder output in data/decoder_output/ (not distributed)

Part of the decoder's second stage, the per-fold harmonic band tables (README.md,
Decoding, Step 09); it fits nothing.
"""

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 2))
R = os.path.join(ROOT, "data", "decoder_output")
SD = os.path.join(R, "main", "fold_tables", "fold_centres_source_data.csv")
OUT = os.path.join(R, "main", "fold_tables", "imagined_walking")
OUT_REL = "result/main/fold_tables/imagined_walking"
AS_RUN = {"healthy": os.path.join(R, "diag_main_goni_psd_peaks", "main_goni_peak_bands.csv"),
          "pat": os.path.join(R, "diag_patient_goni_psd_peaks", "patient_goni_peak_bands.csv"),
          "pat02_sess08": os.path.join(R, "diag_patient_goni_psd_peaks_sess08", "patient_goni_peak_bands.csv"),
          "pat02_sess09": os.path.join(R, "diag_patient_goni_psd_peaks_sess09", "patient_goni_peak_bands.csv")}
HALF, TOL, G1_SESSION = 0.18, 1e-9, "subj22_sess02"


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
    sd = pd.read_csv(SD)
    sd = sd[sd.kind == "fold"].copy()
    sd["fold"] = sd.fold.astype(int)
    tabs = {k: pd.read_csv(p, dtype=str).set_index(["session", "harmonic_k"]) for k, p in AS_RUN.items()}
    n_ok = 0
    for s, g in sd.groupby("session"):
        r0 = g.iloc[0]
        src = "healthy" if not s.startswith("pat") else (s if s in ("pat02_sess08", "pat02_sess09") else "pat")
        for row in band_rows(s, r0.f0_all, (r0.fc_k1_all, r0.fc_k2_all, r0.fc_k3_all), 0):
            st = tabs[src].loc[(s, str(row["harmonic_k"]))]
            assert float(row["band_lo"]) == float(st.band_lo) and float(row["band_hi"]) == float(st.band_hi), (s, row, st)
            n_ok += 1
    print("writer check: %d (session, harmonic) rows reproduce the stored band_lo/band_hi exactly" % n_ok)
    sd["changed"] = sd[["d_fc_k1", "d_fc_k2", "d_fc_k3"]].abs().max(axis=1) > TOL
    os.makedirs(os.path.join(OUT, "bands"), exist_ok=True)
    jobs = []
    for (cohort, s), g in sd[sd.changed].groupby(["cohort", "session"], sort=True):
        key = g[["fc_k1_held", "fc_k2_held", "fc_k3_held"]].round(10).apply(tuple, axis=1)
        for t, (fcs, gg) in enumerate(sorted(g.groupby(key), key=lambda kv: kv[1].fold.min()), start=1):
            assert gg.f0_held.nunique() == 1 and gg.n_pool_traces.nunique() >= 1
            tid = "%s__t%d" % (s, t)
            rel = "%s/bands/%s.csv" % (OUT_REL, tid)
            write_csv(os.path.join(OUT, "bands", tid + ".csv"),
                      band_rows(s, gg.f0_held.iloc[0], fcs, gg.n_pool_traces.iloc[0]))
            jobs.append(dict(table_id=tid, cohort=cohort, session=s, folds=";".join(str(f) for f in sorted(gg.fold)),
                             n_folds=len(gg), f0_held=gg.f0_held.iloc[0], fc_k1=fcs[0], fc_k2=fcs[1], fc_k3=fcs[2],
                             band_csv=rel, export_dir="%s/export/%s" % (OUT_REL, tid)))
    g1 = sd[sd.session == G1_SESSION].iloc[0]
    tid = "G1_%s_asrun" % G1_SESSION
    write_csv(os.path.join(OUT, "bands", tid + ".csv"),
              band_rows(G1_SESSION, g1.f0_all, (g1.fc_k1_all, g1.fc_k2_all, g1.fc_k3_all), 0))
    jobs.append(dict(table_id=tid, cohort="healthy", session=G1_SESSION, folds="1;2;3;4;5;6;7;8;9;10", n_folds=10,
                     f0_held=g1.f0_all, fc_k1=g1.fc_k1_all, fc_k2=g1.fc_k2_all, fc_k3=g1.fc_k3_all,
                     band_csv="%s/bands/%s.csv" % (OUT_REL, tid), export_dir="%s/export/%s" % (OUT_REL, tid)))
    jobs = pd.DataFrame(jobs)
    write_csv(os.path.join(OUT, "tables.csv"), jobs.to_dict("records"))
    main_jobs = jobs[~jobs.table_id.str.startswith("G1_")]
    print(main_jobs.groupby("cohort").agg(tables=("table_id", "size"), folds=("n_folds", "sum"),
                                          sessions=("session", "nunique")))
    print("total tables", len(main_jobs), "+ G1")


if __name__ == "__main__":
    sys.exit(main())
