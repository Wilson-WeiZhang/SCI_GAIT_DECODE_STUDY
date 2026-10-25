"""Supplementary item

sup_decoder_trial_tables - the decoder's fold map, per-trial accuracy and example per-trial predictions

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

import argparse
import csv
import os
import sys

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis_config as ec

ec.require("main")

EXAMPLE = "subj27_sess01"
RUNS = [("h_iw_dump_dir", "healthy", "IW"), ("h_rw_dump_dir", "healthy", "RW"),
        ("p_dump_dir", "post_operative", "IW"), ("p_pre_operative_dump_dir", "pre_operative", "IW")]
SHORT = ["LHip", "LKne", "LAnk", "RHip", "RKne", "RAnk"]


def write(df, name):
    p = ec.sd(name)
    df.to_csv(str(p) + ".tmp", index=False, lineterminator="\n", float_format=None)
    os.replace(str(p) + ".tmp", p)
    return p


def write_meta(rows, name):
    p = ec.sd(name)
    with open(str(p) + ".tmp", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerows([("key", "value")] + rows)
    os.replace(str(p) + ".tmp", p)


def trial_r_table():
    parts = []
    for key, cohort, cond in RUNS:
        d = ec.path(key)
        files = sorted(d.glob("*_trial_r.csv"))
        if not files:
            raise SystemExit(f"no *_trial_r.csv in {d}")
        t = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in files], ignore_index=True)
        t.insert(1, "cohort", cohort)
        t.insert(2, "condition", cond)
        t["scored"] = [int(ec.keep_trial(s, cond, i)) for s, i in zip(t.session, t.trial_idx)]
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def predictions(h5, cond):
    """Scored trials of one session dump (layout of 02_decoding/pyport: rows/<field> samples x joints, trials/row_offset
    and n_rows); the scoring mask is applied as in dump_h5.load_session."""
    with h5py.File(h5, "r") as f:
        sess = f.attrs["session"]
        sess = sess.decode() if isinstance(sess, bytes) else str(sess)
        joints = str(f.attrs["joints"]).split(",")
        tr = {k: f["trials/" + k][()] for k in ("fold", "trial_idx", "row_offset", "n_rows")}
        rows = {k: f["rows/" + k][()] for k in ("time_sec", "Y_true", "A_pred", "C_pred")}
    if [j[:4] for j in joints] != SHORT:
        raise SystemExit(f"{h5}: joints {joints}")
    out = []
    for i in np.argsort(tr["trial_idx"], kind="stable"):
        if not ec.keep_trial(sess, cond, tr["trial_idx"][i]):
            continue
        a, n = int(tr["row_offset"][i]), int(tr["n_rows"][i])
        block = {"session": [sess] * n, "condition": [cond] * n, "trial_idx": [int(tr["trial_idx"][i])] * n,
                 "fold": [int(tr["fold"][i])] * n, "time_s": rows["time_sec"][a:a + n]}
        for lab, fld in (("true", "Y_true"), ("stage1", "A_pred"), ("final", "C_pred")):
            for j, s in enumerate(SHORT):
                block[f"{lab}_{s}"] = rows[fld][a:a + n, j]
        out.append(pd.DataFrame(block))
    return pd.concat(out, ignore_index=True), joints


def check_predictions(P, joints, R):
    """r(final, true) over the finite samples of each trial x joint == the trial_r value (1e-6)."""
    n = 0
    for (s, i), g in P.groupby(["session", "trial_idx"]):
        for s6, joint in zip(SHORT, joints):
            y, c = g[f"true_{s6}"].to_numpy(), g[f"final_{s6}"].to_numpy()
            ok = np.isfinite(y) & np.isfinite(c)
            ref = R[(R.session == s) & (R.condition == g.condition.iloc[0]) & (R.trial_idx == i)
                    & (R.joint == joint)].r.to_numpy()
            if len(ref) != 1:
                raise SystemExit(f"trial_r row missing: {s} {i} {joint}")
            if ok.sum() < 3 or np.isnan(ref[0]):
                continue
            r = np.corrcoef(y[ok], c[ok])[0, 1]
            if abs(r - ref[0]) > 1e-6:
                raise SystemExit(f"prediction check failed: {s} trial {i} {joint}: {r} vs {ref[0]}")
            n += 1
    return n


def headline(R):
    """Median of the session medians of the scored trials' joint-mean r, per cohort x condition."""
    out = []
    for (coh, cond), t in R[R.scored == 1].groupby(["cohort", "condition"], sort=False):
        tr = t.groupby(["session", "trial_idx"]).r.mean()
        out.append((f"{coh} {cond}", float(tr.groupby(level=0).median().median())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", help="session name or 'all': write per-trial predictions to --out")
    ap.add_argument("--out", help="output folder for --predictions")
    a = ap.parse_args()
    R = trial_r_table()

    if a.predictions:
        if not a.out:
            raise SystemExit("--predictions needs --out")
        os.makedirs(a.out, exist_ok=True)
        for key, cohort, cond in RUNS:
            run = "imagined_walking" if cohort == "healthy" and cond == "IW" else (
                "real_walking" if cond == "RW" else "patients")
            for h5 in sorted(ec.path(key).glob("*_dump.h5")):
                s = h5.name[:-len("_dump.h5")]
                if a.predictions not in ("all", s):
                    continue
                P, joints = predictions(h5, cond)
                check_predictions(P, joints, R)
                P = P.drop(columns=[c for c in P.columns if c.startswith("true_")])
                out = os.path.join(a.out, f"{run}_{s}_predictions.csv")
                P.to_csv(out + ".tmp", index=False, lineterminator="\n", float_format="%.6g")
                os.replace(out + ".tmp", out)
                print(out, len(P))
        return

    keys = ["session", "cohort", "condition", "trial_idx"]
    g = R.groupby(keys, sort=False)
    if (g.fold.nunique() > 1).any() or (g.test_ordinal.nunique() > 1).any() or (g.scored.nunique() > 1).any():
        raise SystemExit("fold / test_ordinal / scored differ within a trial")
    F = g[["test_ordinal", "fold", "scored"]].first().reset_index()
    cnt = F.groupby(["cohort", "condition"]).agg(sessions=("session", "nunique"), trials=("trial_idx", "size"),
                                                  scored=("scored", "sum"))
    expect = {("healthy", "IW"): (80, 4756, 3172), ("healthy", "RW"): (80, 4739, 3169),
              ("post_operative", "IW"): (11, 449, 282), ("pre_operative", "IW"): (4, 154, 77)}
    for k, v in expect.items():
        if tuple(int(x) for x in cnt.loc[k]) != v:
            raise SystemExit(f"counts {k}: {tuple(cnt.loc[k])} != {v}")
    write(F, "sup_decoder_trial_fold_map_source_data.csv")
    write(R, "sup_decoder_trial_r_source_data.csv")

    P, joints = predictions(ec.path("h_iw_dump_dir") / f"{EXAMPLE}_dump.h5", "IW")
    n_checked = check_predictions(P, joints, R)
    P = P.drop(columns=[c for c in P.columns if c.startswith("true_")])
    p = ec.sd("sup_decoder_trial_predictions_example_source_data.csv")
    P.to_csv(str(p) + ".tmp", index=False, lineterminator="\n", float_format="%.6g")
    os.replace(str(p) + ".tmp", p)

    counts = [(f"count_{c}_{k}", f"{int(v.sessions)} sessions, {int(v.trials)} trials, scored {int(v.scored)}")
              for (k, c), v in cnt.iterrows()]
    common = [("build_id", ec.BUILD_ID), ("script", "03_figures/sup_decoder_trial_tables.py"),
              ("cv", "random 10-fold cross-validation within each session; every trial is decoded once, by the model of "
                     "the fold that holds it out; the same fold map serves the random 10-fold runs of a condition (the "
                     "contiguous-fold control has its own folds)"),
              ("scored", "1 = trial recorded after the session's calibration recording (scoring mask, condition IW / RW, "
                         "column keep); every trial, scored or not, trains the decoder")]
    write_meta([("file", "sup_decoder_trial_fold_map_source_data.csv"),
                ("rows", "one per session x trial"),
                ("columns", "session, cohort (healthy / post_operative / pre_operative), condition (IW = imagined walking, RW = real "
                            "walking), trial_idx (the run's trial index; the scoring-mask key), test_ordinal, fold, "
                            "scored")] + common + counts,
               "sup_decoder_trial_fold_map_metadata.csv")
    write_meta([("file", "sup_decoder_trial_r_source_data.csv"),
                ("rows", "one per session x trial x joint (six joints)"),
                ("columns", "session, cohort, condition, fold, trial_idx, test_ordinal, joint, model (calib = the "
                            "calibrated readout), r (held-out Pearson r of the final decoder output with the goniometer "
                            "trace over the finite samples; empty = not computable, stored as NaN by the readout), rpt_r "
                            "(= r), n_samples, first_finite_time_sec, scored"),
                ("use", "every reported accuracy is computed from the scored rows; e.g. the headline convention (a "
                        "trial's r = mean over its six joints, a session's value = median over its scored trials, then "
                        "the median over sessions) gives " + ", ".join(f"{k} {v:.6f}" for k, v in headline(R))),
                ] + common + counts,
               "sup_decoder_trial_r_metadata.csv")
    write_meta([("file", "sup_decoder_trial_predictions_example_source_data.csv"),
                ("session", f"{EXAMPLE} (healthy imagined walking; the session of Fig. 2d/e)"),
                ("rows", f"one per sample (250 Hz) of the {P.trial_idx.nunique()} scored trials, {len(P)} rows"),
                ("columns", "session, condition, trial_idx, fold, time_s (trial time, s), stage1_<joint> (first-stage "
                            "output after its 6 Hz low-pass), final_<joint> (final output, the one scored); joints LHip, "
                            "LKne, LAnk, RHip, RKne, RAnk; empty = not decoded"),
                ("target", "the goniometer target is not included (goniometer data are not released)"),
                ("check", f"before writing, r(final output, goniometer target) over the finite samples, computed from "
                          f"the undistributed decoder output, equals trial_r to 1e-6 in {n_checked} trial x joint cells"),
                ("all_sessions", "the full set (healthy imagined and real walking, patients; about 11.7 million scored "
                                 "samples) is not distributed; `--predictions all --out <dir>` writes it from the "
                                 "decoder output")] + common,
               "sup_decoder_trial_predictions_example_metadata.csv")
    print("fold map", len(F), "| trial_r", len(R), "| predictions", len(P), "rows,", n_checked, "cells checked")
    print(cnt)


if __name__ == "__main__":
    main()
