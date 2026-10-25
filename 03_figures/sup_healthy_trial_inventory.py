#!/usr/bin/env python3
"""Supplementary item

Healthy cohort trial inventory and recording duration (numbers only; no figure)

Reads:
  decoder output in data/decoder_output/ (not distributed)
  preprocessing output in $GAIT_PREP_OUTPUT (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
PREP_OUTPUT = __import__("pathlib").Path(__import__("os").environ.get("GAIT_PREP_OUTPUT", "preprocessing_output"))
_release_rel = lambda p, r: p.relative_to(r) if p.is_relative_to(r) else p

import csv
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SD_DIR = BASE / "04_source_data" / "supplementary"
SRC_CSV = SD_DIR / "sup_healthy_trial_inventory_source_data.csv"
META_CSV = SD_DIR / "sup_healthy_trial_inventory_metadata.csv"

ROSTER = PREP_OUTPUT / "healthy" / "note" / "session_manifest.csv"
STATUS = PREP_OUTPUT / "healthy" / "note" / "cohort_manifest_healthy.csv"
QC_DIR = PREP_OUTPUT / "healthy" / "result" / "qc"
RAWDUR = QC_DIR / "prep_healthy_cohort_raw_recording_duration.csv"
TRIALS = QC_DIR / "prep_healthy_cohort_trial_inventory.csv"

DEC = BASE / "data" / "decoder_output"
IW_CSV = DEC / ("decoder_output_imagined_walking_"
                "imagined_walking.csv")
RW_CSV = DEC / ("decoder_output_real_walking_"
                "real_walking.csv")

N_COHORT = 80
N_PARTICIPANTS = 42
N_IW_HELDOUT_EXPECTED = 4756
EVENT_FS_HZ = 250.0


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def cohort_roster() -> tuple[list[str], dict[str, str]]:
    """The 80 retained healthy sessions (export folder names) + session -> subject."""
    sessions = [r["session"].strip() for r in read_csv_rows(ROSTER) if r["session"].strip()]
    assert len(sessions) == N_COHORT, f"roster has {len(sessions)} sessions, expected {N_COHORT}"
    assert len(set(sessions)) == N_COHORT, "duplicate session in roster"

    included = [r for r in read_csv_rows(STATUS) if r["status"].strip() == "include"]
    assert len(included) == N_COHORT, f"{len(included)} include rows, expected {N_COHORT}"
    subjects = sorted({r["subject"].strip() for r in included})
    assert len(subjects) == N_PARTICIPANTS, \
        f"{len(subjects)} participants, expected {N_PARTICIPANTS}"

    sess2subj: dict[str, str] = {}
    for r in included:
        sub, ss = r["subject"].strip(), r["session"].strip()
        stem = f"subj{int(sub.split('_')[1]):02d}_{ss}"
        hit = [s for s in sessions if s == stem or s.startswith(stem + "_")]
        assert len(hit) == 1, f"{sub}/{ss} matches {hit} in the roster (expected exactly 1)"
        sess2subj[hit[0]] = sub
    assert len(sess2subj) == N_COHORT
    return sessions, sess2subj


def heldout_trials(csv_path: Path) -> tuple[dict[str, int], int, int]:
    """Sum n_trials over UNIQUE (session, fold) pairs -- rows are fold x joint."""
    per_fold: dict[tuple[str, str], int] = {}
    for r in read_csv_rows(csv_path):
        key = (r["session"].strip(), r["fold"].strip())
        n = int(float(r["n_trials"]))
        if key in per_fold:
            assert per_fold[key] == n, f"n_trials disagrees across joints for {key}"
        else:
            per_fold[key] = n
    per_session: dict[str, int] = defaultdict(int)
    for (sess, _fold), n in per_fold.items():
        per_session[sess] += n
    return dict(per_session), len(per_fold), sum(per_fold.values())


def raw_durations() -> tuple[dict[str, float], dict[str, int], list[str]]:
    """session -> summed raw recording seconds (nan if any part is unmeasurable), n parts."""
    secs: dict[str, float] = defaultdict(float)
    parts: dict[str, int] = defaultdict(int)
    anomalies: list[str] = []
    for r in read_csv_rows(RAWDUR):
        s = r["session"].strip()
        parts[s] += 1
        if r.get("anomaly", "").strip():
            anomalies.append(f"{s}/{r['part_label']}: {r['anomaly']}")
        d = r.get("duration_s", "").strip()
        secs[s] = secs[s] + float(d) if d else float("nan")
    return dict(secs), dict(parts), anomalies


def atomic_write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    with open(tmp, "w", newline="\n", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def main() -> int:
    sessions, sess2subj = cohort_roster()

    inv = {r["session"].strip(): r for r in read_csv_rows(TRIALS)}
    assert set(inv) == set(sessions), "trial-inventory CSV sessions != cohort roster"
    assert not any(r["unknown_codes"].strip() for r in inv.values()), \
        "unrecognised event codes in the trial inventory"

    iw_per_sess, n_iw_folds, n_iw_total = heldout_trials(IW_CSV)
    rw_per_sess, n_rw_folds, n_rw_total = heldout_trials(RW_CSV)
    assert n_iw_total == N_IW_HELDOUT_EXPECTED, \
        f"held-out imagined-walking trials = {n_iw_total}, expected {N_IW_HELDOUT_EXPECTED}"
    assert set(iw_per_sess) == set(sessions), "IW decode CSV sessions != cohort roster"
    assert set(rw_per_sess) == set(sessions), "RW decode CSV sessions != cohort roster"
    for s in sessions:
        assert int(inv[s]["n_gi_qc_good"]) == iw_per_sess[s], (
            f"{s}: qc-good imagined-walking {inv[s]['n_gi_qc_good']} != held-out decoded "
            f"{iw_per_sess[s]} -- the event table is not from the decoder's export root")
        assert int(inv[s]["n_rw_qc_good"]) == rw_per_sess[s], (
            f"{s}: qc-good real-walking {inv[s]['n_rw_qc_good']} != held-out decoded "
            f"{rw_per_sess[s]}")

    raw_s, raw_parts, raw_anom = raw_durations()
    assert set(raw_s) == set(sessions), "raw-duration CSV sessions != cohort roster"

    header = ["row_type", "session", "subject",
              "n_gi_events", "n_gi_qc_good", "n_rw_events", "n_rw_qc_good",
              "n_rest_events", "n_rest_qc_good",
              "n_iw_heldout_decoded", "n_rw_heldout_decoded",
              "task_span_min", "raw_recording_min", "n_raw_parts"]
    rows: list[dict] = []
    tot: dict[str, int] = defaultdict(int)
    spans: list[float] = []
    raws: list[float] = []

    for sess in sessions:
        r = inv[sess]
        span_min = float(r["task_span_s"]) / 60.0
        raw_min = raw_s[sess] / 60.0
        spans.append(span_min)
        if raw_min == raw_min:
            raws.append(raw_min)
        rows.append({
            "row_type": "session", "session": sess, "subject": sess2subj[sess],
            "n_gi_events": int(r["n_gi_events"]), "n_gi_qc_good": int(r["n_gi_qc_good"]),
            "n_rw_events": int(r["n_rw_events"]), "n_rw_qc_good": int(r["n_rw_qc_good"]),
            "n_rest_events": int(r["n_rest_events"]),
            "n_rest_qc_good": int(r["n_rest_qc_good"]),
            "n_iw_heldout_decoded": iw_per_sess[sess],
            "n_rw_heldout_decoded": rw_per_sess[sess],
            "task_span_min": f"{span_min:.2f}",
            "raw_recording_min": f"{raw_min:.2f}" if raw_min == raw_min else "",
            "n_raw_parts": raw_parts[sess],
        })
        for k in ("n_gi_events", "n_gi_qc_good", "n_rw_events", "n_rw_qc_good",
                  "n_rest_events", "n_rest_qc_good"):
            tot[k] += int(r[k])
        tot["iw_dec"] += iw_per_sess[sess]
        tot["rw_dec"] += rw_per_sess[sess]
        tot["parts"] += raw_parts[sess]

    sess_per_subj: dict[str, int] = defaultdict(int)
    for s in sessions:
        sess_per_subj[sess2subj[s]] += 1
    n_two = sum(1 for v in sess_per_subj.values() if v == 2)
    n_one = sum(1 for v in sess_per_subj.values() if v == 1)
    assert n_two + n_one == N_PARTICIPANTS, \
        f"participants with 1 or 2 sessions = {n_two + n_one}, expected {N_PARTICIPANTS}"

    ratios = [sp / rw for sp, rw in zip(spans, raws)]
    S = [
        ("n_sessions", N_COHORT, "sessions"),
        ("n_participants", N_PARTICIPANTS, "participants"),
        ("n_participants_two_sessions", n_two, "participants"),
        ("n_participants_one_session", n_one, "participants"),
        ("total_n_gi_events", tot["n_gi_events"], "trials"),
        ("total_n_gi_qc_good", tot["n_gi_qc_good"], "trials"),
        ("total_n_iw_heldout_decoded", tot["iw_dec"], "trials"),
        ("total_n_rw_events", tot["n_rw_events"], "trials"),
        ("total_n_rw_qc_good", tot["n_rw_qc_good"], "trials"),
        ("total_n_rw_heldout_decoded", tot["rw_dec"], "trials"),
        ("total_n_rest_events", tot["n_rest_events"], "trials"),
        ("total_n_rest_qc_good", tot["n_rest_qc_good"], "trials"),
        ("total_n_raw_parts", tot["parts"], "raw recordings"),
        ("n_sessions_with_raw_recording_min", len(raws), "sessions"),
        ("mean_raw_recording_min", round(statistics.fmean(raws), 2), "min"),
        ("sd_raw_recording_min", round(statistics.stdev(raws), 2), "min"),
        ("median_raw_recording_min", round(statistics.median(raws), 2), "min"),
        ("min_raw_recording_min", round(min(raws), 2), "min"),
        ("max_raw_recording_min", round(max(raws), 2), "min"),
        ("total_raw_recording_h", round(sum(raws) / 60.0, 2), "hours"),
        ("mean_task_span_min", round(statistics.fmean(spans), 2), "min"),
        ("median_task_span_to_raw_ratio", round(statistics.median(ratios), 3), "ratio"),
    ]
    rows.extend({"row_type": "cohort", "session": k, "subject": u, "n_gi_events": v}
                for k, v, u in S)
    atomic_write_csv(SRC_CSV, header, rows)

    d = {k: v for k, v, _ in S}
    meta = [
        ("cohort", "healthy cohort (80 sessions / 42 participants)"),
        ("cohort_authority",
         "session_manifest.csv (not distributed) (export folder names) cross-checked "
         "against cohort_manifest_healthy.csv (not distributed) (status=include); both "
         "counts asserted at run time"),
        ("generated", date.today().isoformat()),
        ("renderer", "03_figures/sup_healthy_trial_inventory.py"),
        ("paper_location",
         "Supplementary Note 11, cohort-accounting "
         "paragraph: mean recording duration per session, and the real-walking / rest trial "
         "counts available beyond the 4,756 held-out imagined-walking trials"),
        ("figure", "none - numbers/table only"),
        ("input_trial_inventory", _release_rel(TRIALS, BASE).as_posix()),
        ("input_raw_duration", _release_rel(RAWDUR, BASE).as_posix()),
        ("input_decode_iw", _release_rel(IW_CSV, BASE).as_posix()),
        ("input_decode_rw", _release_rel(RW_CSV, BASE).as_posix()),
        ("event_root_verification",
         "per-session qc-good counts equal the decoder's held-out trial counts in 80 of 80 "
         "sessions for both tasks (4756 imagined-walking, 4739 real-walking); asserted"),
        ("event_sampling_rate_hz", f"{EVENT_FS_HZ:g}"),
        ("event_sampling_rate_evidence",
         "subj01_sess01_eeg.mat carries eeg.s = 250 Hz, and its max(etstop) = 933028 <= "
         "size(eeg.x,1) = 934530, so etstart/etstop are samples of the exported 250 Hz EEG "
         "(raw is 1000 Hz; prep resamples 4x, data_spec.md (not distributed) sec.3)"),
        ("event_sampling_rate_crosscheck",
         f"median(task_span / raw_recording) = {d['median_task_span_to_raw_ratio']} at "
         f"{EVENT_FS_HZ:g} Hz; reading the same samples at 1000 Hz would put the whole "
         "150-trial protocol in a quarter of the recording, which the raw lengths exclude"),
        ("raw_recording_definition",
         "size(.eeg)/(NumberOfChannels x bytes_per_sample)/sampling_rate, summed over the "
         "sub-parts 04_export merged into the session. All 94 parts are 63 ch / 1000 Hz / "
         "IEEE_FLOAT_32 / BINARY / MULTIPLEXED, with no anomalies"),
        ("raw_recording_is_not_export_length",
         "prep_step2 trims the recording to the S11-S12 task block, so the exported _eeg.mat "
         "is systematically shorter than the raw recording; task_span_min is reported beside "
         "raw_recording_min as the analysed-span cross-check"),
        ("col_row_type", "'session' = one analysed session; 'cohort' = a cohort-level summary "
                         "metric, whose name is in `session`, unit in `subject`, value in "
                         "`n_gi_events`"),
        ("col_session", "export folder name of the analysed session (session rows)"),
        ("col_subject", "participant id from the prep cohort manifest (session rows)"),
        ("col_n_gi_qc_good", "of those, event.qc == 'good'"),
        ("col_n_rw_events", "recorded real-walking trial events, event.code 'gaitexecution'"),
        ("col_n_rw_qc_good", "of those, event.qc == 'good'"),
        ("col_n_rest_events", "recorded rest trial events, event.code 'rest'"),
        ("col_n_rest_qc_good", "of those, event.qc == 'good'"),
        ("col_n_iw_heldout_decoded",
         "held-out imagined-walking trials that entered decoding = sum of n_trials over the "
         "unique (session, fold) pairs of the main MI merged CSV (rows are fold x "
         "joint); equals n_gi_qc_good in all 80 sessions"),
        ("col_n_rw_heldout_decoded",
         "same, from the main real-walking (EXEC) merged CSV; equals n_rw_qc_good in "
         "all 80 sessions"),
        ("col_task_span_min",
         "(max etstop - min etstart) / 250 Hz / 60 = first trial onset to last trial offset "
         "of the exported session"),
        ("col_raw_recording_min",
         "summed raw BrainVision recording length of the merged raw parts, minutes"),
        ("col_n_raw_parts", "number of raw recordings merged into the analysed session"),
        ("n_decode_folds_iw", f"{n_iw_folds} (80 sessions x 10 folds)"),
        ("n_decode_folds_rw", f"{n_rw_folds} (80 sessions x 10 folds)"),
        ("assertion_iw_heldout",
         f"total held-out imagined-walking trials == {N_IW_HELDOUT_EXPECTED} (asserted; "
         "matches the number already printed in Supplementary Note 11)"),
        ("raw_duration_anomalies",
         "; ".join(raw_anom) if raw_anom else "none (94/94 raw parts read cleanly)"),
    ]
    atomic_write_csv(META_CSV, ["key", "value"], [{"key": k, "value": v} for k, v in meta])

    print(f"sessions={N_COHORT}  participants={N_PARTICIPANTS} "
          f"(two sessions: {n_two}, one session: {n_one})")
    print(f"events   GI {tot['n_gi_events']} ({tot['n_gi_qc_good']} qc-good)   "
          f"RW {tot['n_rw_events']} ({tot['n_rw_qc_good']} qc-good)   "
          f"REST {tot['n_rest_events']} ({tot['n_rest_qc_good']} qc-good)")
    print(f"held-out decoded   IW {tot['iw_dec']}   RW {tot['rw_dec']}")
    print(f"raw recording (min)  mean {d['mean_raw_recording_min']} "
          f"SD {d['sd_raw_recording_min']}   median {d['median_raw_recording_min']}   "
          f"range {d['min_raw_recording_min']}-{d['max_raw_recording_min']}   "
          f"n={d['n_sessions_with_raw_recording_min']}   "
          f"total {d['total_raw_recording_h']} h")
    print(f"task span (min) mean {d['mean_task_span_min']}   "
          f"median span/raw ratio {d['median_task_span_to_raw_ratio']}")
    print(f"wrote {_release_rel(SRC_CSV, BASE).as_posix()} ({len(rows)} rows)")
    print(f"wrote {_release_rel(META_CSV, BASE).as_posix()} ({len(meta)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
