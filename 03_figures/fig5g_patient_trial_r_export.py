"""Figure 5 — Decoding imagined walking after spinal cord injury

patient performance CSVs - per-session, per-trial decoding r (Hip/Knee/Ankle/All)

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import analysis_config as ec

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "decoder_output"
if ec.IS_DEFAULT:
    DEFAULT_SRC = ec.path("p_trial_r_long")
    DEFAULT_PARALLEL = ec.path("p_summary_csv")
    SIDECARS = {
        "pat02_sess08": (ec.path("p_trial_r_long_sess08"), ec.path("p_summary_csv_sess08")),
        "pat02_sess09": (ec.path("p_trial_r_long_sess09"), ec.path("p_summary_csv_sess09")),
    }
else:
    DEFAULT_SRC = ec.path("p_trial_r_long")
    DEFAULT_PARALLEL = ec.path("p_summary_csv")
    SIDECARS = {}
DEFAULT_OUTDIR = ROOT / "04_source_data" / "fig5"

BUILD_ID = ec.TAG
DECODE_TAG = ("patientdump (+ patientdump_sess08 for S6, "
              "+ patientdump_sess09 for S7)") if ec.IS_DEFAULT else ec.PAT_TAG
DECODE_RUN = ("run_frozen_patient.sh (not distributed)" if ec.IS_DEFAULT else
              "run_session.py")
if ec.BUILD_ID == "session_band_table":
    DECODE_RUN = ("random 10-fold, causal 6 Hz low-pass on the stage-1 output; reproduced byte-identically by "
                  "02_decoding/pyport/run_session.py with pyport/configs/tenfold_pca30_lpf6.json")
if ec.BUILD_ID == "zero_phase_prep":
    DECODE_RUN = 'random 10-fold, causal 6 Hz low-pass on the stage-1 output, harmonic band table estimated per fold from the training trials, on the zero-phase preprocessing'
if ec.BUILD_ID == "main":
    DECODE_RUN = 'random 10-fold, causal 6 Hz low-pass on the stage-1 output, harmonic band table estimated per fold from the training trials, on the strictly causal preprocessing: every fold decoded from the feature export of its band table (02_decoding/run/export_features.sh, job list 02_decoding/run/make_job_lists.py) and the folds merged per session (02_decoding/run/merge_runs.py); 02_decoding/pyport/run_session.py with pyport/configs/tenfold_pca30_lpf6.json'
BASIS = ("(shared_pca_basis)") if ec.IS_DEFAULT else (
    "frozen all-80-healthy pooled shared PCA30 on causal complex-demodulation features "
    f"({ec.rel('h_global_basis').split('/')[-2]})")
READOUT = "patient-trained 10-fold CV (representation transfer, not zero-shot)"

POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
            "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
            "pat02_sess09"]
PRE_OPERATIVE = ["pat01_sess01", "pat01_sess02", "pat02_sess01", "pat02_sess02"]
CONTROL = "pat01_sess01iw"
COHORT = {s: "pre_operative" for s in PRE_OPERATIVE} | {s: "post_operative" for s in POST_OPERATIVE}
ANCHOR_POST_OPERATIVE = ec.anchor("pat_post_operative_session_mean")[0]
ANCHOR_PRE_OPERATIVE = ec.anchor("pat_pre_operative_session_mean")[0]
ANCHOR_TOL = 5e-6 if ec.IS_DEFAULT else ec.anchor("pat_post_operative_session_mean")[1]

GROUPS = {"Hip": ["LHipW X", "RHipW X"], "Knee": ["LKneW X", "RKneW X"],
          "Ankle": ["LAnkW X", "RAnkW X"],
          "All": ["LHipW X", "RHipW X", "LKneW X", "RKneW X", "LAnkW X", "RAnkW X"]}


def _quantile(vals, q):
    """Linear-interpolation quantile (numpy 'linear' convention), matching fig2g."""
    v = sorted(vals)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def patient_of(session: str) -> str:
    m = re.match(r"^(pat\d+)", str(session))
    return m.group(1) if m else str(session)


def sess_order_key(session: str):
    """Fixed panel order: P1 pre -> P1 post -> P2 pre -> P2 post, by session number within block."""
    m = re.match(r"^(pat\d+)_sess(\d+)", str(session))
    pat, num = (m.group(1), int(m.group(2))) if m else (str(session), 0)
    blk = 0 if COHORT.get(session) == "pre_operative" else 1
    return (pat, blk, num)


def nanmean(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def fold_level_session_means(path: Path) -> dict[str, float]:
    """Session means of `rpt_calib` over the 10 CV folds x 6 joints (NaN ignored)."""
    import math
    vals = defaultdict(list)
    for r in csv.DictReader(path.open(newline="", encoding="utf-8-sig")):
        try:
            v = float(r["rpt_calib"])
        except (ValueError, KeyError):
            continue
        if math.isfinite(v):
            vals[r["session"]].append(v)
    return {s: sum(v) / len(v) for s, v in vals.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fig.5 patient per-trial decoding-r CSVs ()")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--parallel", type=Path, default=DEFAULT_PARALLEL)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src, outdir = args.src, args.outdir
    if not src.exists():
        raise SystemExit(f"Missing {src} - the  patient trial-r merge must exist first.")
    if not args.parallel.exists():
        raise SystemExit(f"Missing {args.parallel} - needed for the fold-level anchor check.")
    for sess, (side_src, side_par) in SIDECARS.items():
        if not side_src.exists() or not side_par.exists():
            raise SystemExit(f"Missing the {sess} sidecar CSVs ({side_src.name} / {side_par.name}).")

    per = defaultdict(dict)
    meta = {}
    seen_sessions = set()
    seen_side: dict[str, set[str]] = {}
    streams = [(src, seen_sessions)]
    for sess, (side_src, _) in SIDECARS.items():
        seen_side[sess] = set()
        streams.append((side_src, seen_side[sess]))
    for path, seen in streams:
        for row in csv.DictReader(path.open(newline="", encoding="utf-8-sig")):
            seen.add(row["session"])
            if row["model"] != "calib":
                continue
            key = (row["session"], int(row["trial_idx"]))
            try:
                v = float(row["rpt_r"])
                if v != v:
                    v = None
            except ValueError:
                v = None
            per[key][row["joint"]] = v
            meta[key] = (int(row["test_ordinal"]), int(row["fold"]))

    if ec.IS_DEFAULT and len(seen_sessions) != 16:
        raise SystemExit(f"{src.name} holds {len(seen_sessions)} sessions, expected the 16 "
                         f"registry sessions: {sorted(seen_sessions)}")
    for sess, seen in seen_side.items():
        if seen != {sess}:
            raise SystemExit(f"{SIDECARS[sess][0].name} must hold exactly {sess}, got: "
                             f"{sorted(seen)}")
        if seen & seen_sessions:
            raise SystemExit("session present in BOTH the patient set merge and a sidecar - would "
                             f"double-weight: {sorted(seen & seen_sessions)}")
        seen_sessions |= seen
    missing = [s for s in POST_OPERATIVE + PRE_OPERATIVE if s not in seen_sessions]
    if missing:
        raise SystemExit(f"cohort sessions absent from the inputs: {missing}")

    trials = []
    for (sess, tidx), jr in per.items():
        if sess not in COHORT:
            continue
        rec = {"patient": patient_of(sess), "cohort": COHORT[sess], "session": sess,
               "trial_idx": tidx, "test_ordinal": meta[(sess, tidx)][0],
               "fold": meta[(sess, tidx)][1]}
        for g, js in GROUPS.items():
            rec[g] = nanmean([jr.get(j) for j in js])
        if rec["All"] is None:
            continue
        trials.append(rec)
    trials.sort(key=lambda r: (sess_order_key(r["session"]), r["trial_idx"]))

    bysess = defaultdict(list)
    for r in trials:
        bysess[r["session"]].append(r)
    sessions = sorted(bysess, key=sess_order_key)
    if len(sessions) != 15:
        raise SystemExit(f"expected the 15 retained sessions, got {len(sessions)}: {sessions}")

    outdir.mkdir(parents=True, exist_ok=True)
    with ec.scoped(outdir / "fig5g_trial_r_long.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "cohort", "session", "trial_idx", "test_ordinal", "fold",
                    "Hip", "Knee", "Ankle", "All"])
        for r in trials:
            w.writerow([r["patient"], r["cohort"], r["session"], r["trial_idx"],
                        r["test_ordinal"], r["fold"]] +
                       [("" if r[g] is None else f"{r[g]:.6f}") for g in ("Hip", "Knee", "Ankle", "All")])

    with ec.scoped(outdir / "fig5g_session_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        head = ["session", "patient", "cohort", "n_trials"]
        for g in ("Hip", "Knee", "Ankle", "All"):
            head += [f"{g}_median", f"{g}_q25", f"{g}_q75", f"{g}_mean", f"{g}_sd"]
        w.writerow(head)
        for s in sessions:
            rs = bysess[s]
            row = [s, patient_of(s), COHORT[s], len(rs)]
            for g in ("Hip", "Knee", "Ankle", "All"):
                vals = [r[g] for r in rs if r[g] is not None]
                row += [f"{median(vals):.6f}", f"{_quantile(vals, 0.25):.6f}",
                        f"{_quantile(vals, 0.75):.6f}", f"{mean(vals):.6f}",
                        f"{pstdev(vals):.6f}" if len(vals) > 1 else "0.000000"]
            w.writerow(row)

    for g in ("Hip", "Knee", "Ankle", "All"):
        cols = {s: [r[g] for r in sorted(bysess[s], key=lambda r: r["trial_idx"])] for s in sessions}
        nmax = max(len(v) for v in cols.values())
        with ec.scoped(outdir / f"fig5g_trial_r_matrix_{g}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(sessions)
            for i in range(nmax):
                w.writerow([("" if i >= len(cols[s]) or cols[s][i] is None else f"{cols[s][i]:.6f}")
                            for s in sessions])

    trial_sess_mean = {s: mean([r["All"] for r in bysess[s]]) for s in sessions}
    fold_sess_mean = fold_level_session_means(args.parallel)
    for _sess, (_, side_par) in SIDECARS.items():
        fold_sess_mean.update(fold_level_session_means(side_par))
    fold_post = mean([fold_sess_mean[s] for s in POST_OPERATIVE])
    fold_pre = mean([fold_sess_mean[s] for s in PRE_OPERATIVE])
    trial_post = mean([trial_sess_mean[s] for s in POST_OPERATIVE])
    trial_pre = mean([trial_sess_mean[s] for s in PRE_OPERATIVE])
    for name, got, want in (("post_operative", fold_post, ANCHOR_POST_OPERATIVE), ("pre_operative", fold_pre, ANCHOR_PRE_OPERATIVE)):
        if want is None:
            print(f"[anchor] {name} fold-level = {got:.6f} - NO recorded anchor to check against "
                  
                  "")
            continue
        if abs(got - want) >= ANCHOR_TOL:
            raise SystemExit(f"ANCHOR MISMATCH {name}: fold-level {got:.6f} vs recorded {ec.BUILD_ID} "
                             f"{want:.6f} (|d|={abs(got - want):.2e} >= {ANCHOR_TOL:g}). "
                             f"Inputs are not the {ec.BUILD_ID} build.")

    if ec.IS_DEFAULT:
        readout = READOUT
        fold_def = "per session: mean of rpt_calib over 10 CV folds x 6 joints (NaN ignored); cohort: session-equal mean. THIS is the project anchor level."
    elif ec.BUILD_ID in ec.DECODER_BUILDS:
        nf = defaultdict(set)
        for r in csv.DictReader(args.parallel.open(newline="")):
            nf[r["session"]].add(int(r["fold"]))
        if {len(v) for v in nf.values()} != {10}:
            raise SystemExit(f"expected 10 CV folds per session, got {sorted({len(v) for v in nf.values()})}")
        readout = "patient-trained random 10-fold CV (representation transfer, not zero-shot)"
        fold_def = ("per session: mean of rpt_calib over 10 CV folds x 6 joints (NaN ignored); cohort: "
                    "session-equal mean. THIS is the anchor level checked against 03_figures/analysis_config.py."
                    "")
    else:
        n_test = {int(r["n_test"]) for r in csv.DictReader(args.parallel.open(newline=""))}
        if n_test != {1}:
            raise SystemExit(f"expected one held-out trial per fold, got n_test {sorted(n_test)}")
        readout = ""
        fold_def = ("per session: mean of rpt_calib over the one-trial folds (one held-out trial "
                    "each) x 6 joints (NaN ignored); cohort: session-equal mean. THIS is the anchor level "
                    "checked against 03_figures/analysis_config.py.")

    with ec.scoped(outdir / "fig5g_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in [
            ("figure", "Fig.5 patient per-session / per-trial decoding r (patient analogue of Fig.2g)"),
            ("decode_tag", DECODE_TAG),
            ("basis", BASIS),
            ("readout", readout),
            ("scope", "n=2 SCI patients, case-level, descriptive; no population claim"),
            ("metric", "rpt_r, model=calib; joint groups Hip/Knee/Ankle = L,R mean; All = mean of the 6 Walker-X joints"
                       " with a finite r (in pat01_sess05 and pat01_sess08 the walker LKneW X channel has no finite r "
                       "in any trial, so there Knee = RKneW X and All is the mean of the other five joints)"),
            ("source", src.relative_to(ROOT).as_posix()),
            ("source_fold_level", args.parallel.relative_to(ROOT).as_posix()),
            ("source_sidecars", "; ".join(f"{sess}: {sc[0].relative_to(ROOT).as_posix()}"
                                          for sess, sc in SIDECARS.items())),
            ("source_sidecars_fold_level", "; ".join(f"{sess}: {sc[1].relative_to(ROOT).as_posix()}"
                                                     for sess, sc in SIDECARS.items())),
            ("n_trials", len(trials)),
            ("n_sessions", len(sessions)),
            ("n_patients", len({r["patient"] for r in trials})),
            ("pre_operative_set", "pat01 sess01/02 + pat02 sess01/02 (4 sessions)"),
            ("column_order", "P1 pre -> P1 post -> P2 pre -> P2 post, by session number within each block (fixed; NOT ranked by performance, unlike Fig.2g)"),
            ("fold_level_post_operative_mean_rpt_calib", round(fold_post, 6)),
            ("fold_level_pre_operative_mean_rpt_calib", round(fold_pre, 6)),
            ("trial_level_post_operative_mean_All", round(trial_post, 6)),
            ("trial_level_pre_operative_mean_All", round(trial_pre, 6)),
            ("summary_statistic", "median with IQR (q25/q75) as in Fig.2g; mean+sd retained for reference"),
            ("outputs", "fig5g_trial_r_long.csv; fig5g_session_summary.csv; fig5g_trial_r_matrix_{Hip,Knee,Ankle,All}.csv; fig5g_metadata.csv"),
        ]:
            w.writerow([k, v])

    print(f"trials={len(trials)}  sessions={len(sessions)}  patients={len({r['patient'] for r in trials})}")
    for s in sessions:
        print(f"  {s:<14} {COHORT[s]:<8} n={len(bysess[s]):>3}  trial-All={trial_sess_mean[s]:.4f}  "
              f"fold-rpt_calib={fold_sess_mean[s]:.4f}")
    print(f"FOLD-level (anchor level)  post_operative = {fold_post:.6f}   pre_operative = {fold_pre:.6f}")
    print(f"TRIAL-level (this file)    post_operative = {trial_post:.6f}   pre_operative = {trial_pre:.6f}")
    print("wrote fig5g_trial_r_long.csv, fig5g_session_summary.csv, "
          "fig5g_trial_r_matrix_{Hip,Knee,Ankle,All}.csv, fig5g_metadata.csv -> 04_source_data/")


if __name__ == "__main__":
    main()
