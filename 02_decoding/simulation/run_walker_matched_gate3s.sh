#!/usr/bin/env bash
# Walker goniometer phase replayed on exactly the decoded window (after the decoder's 2 s causal start-up) of every decoded trial, so Walker vs Decoded is
# paired on the same trials and samples. Same inputs and physics as run_gate3s_replays.sh; PHYS_MIN_DUR_S=3.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
ROOT=${GAIT_PROJECT_ROOT}
DUMP=$ROOT/data/decoder_output/trial_dump_patient
NPZ=$ROOT/data/decoder_output/patient_cycle_average.npz
SESSIONS=pat01_sess04,pat01_sess06,pat01_sess07,pat01_sess09,pat02_sess03,pat02_sess05,pat02_sess06,pat02_sess08,pat02_sess09
OUTDIR=simulation_rollouts_gate3s
export PHYS_MIN_DUR_S=3
echo "== walker-matched replay start $(date -Is)"
taskset -c 14-31 "$PY" replay_walker_matched.py --dump "$DUMP" --npz "$NPZ" --sessions "$SESSIONS" \
  --out "$OUTDIR" --workers 10 2>&1 | tee "$OUTDIR/walker_matched.log"
echo "== walker-matched replay done $(date -Is)"
md5sum "$OUTDIR/walker_matched_rhythm.csv" "$OUTDIR/walker_matched_traces_x.csv.gz"
