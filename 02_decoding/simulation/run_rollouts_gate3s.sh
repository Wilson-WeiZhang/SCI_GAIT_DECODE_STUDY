#!/usr/bin/env bash
# 3 s run of the post_operative rollouts: trials with at least 3 s of decoder output after the decoder's
# start-up (PHYS_MIN_DUR_S=3). Inputs: patient set dump / patient set cycavg / 9-session post_operative kinematic set;
# output simulation_rollouts_gate3s/. The duration rule applies to walker
# AND decoded trials, so the blind calibration set is drawn from the same trials.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
ROOT=${GAIT_PROJECT_ROOT}
DUMP=$ROOT/data/decoder_output/trial_dump_patient
REST=$ROOT/data/decoder_output/trial_dump_patient_REST
NPZ=$ROOT/data/decoder_output/patient_cycle_average.npz
OUTDIR=simulation_rollouts_gate3s
WORKERS=${WORKERS:-10}
SESSIONS=pat01_sess04,pat01_sess06,pat01_sess07,pat01_sess09,pat02_sess03,pat02_sess05,pat02_sess06,pat02_sess08,pat02_sess09
export PHYS_MIN_DUR_S=3
mkdir -p "$OUTDIR"
echo "== gate3s start $(date -Is)  workers=$WORKERS  out=$OUTDIR  PHYS_MIN_DUR_S=$PHYS_MIN_DUR_S"
echo "sessions: $SESSIONS"
taskset -c 14-31 "$PY" run_rollouts.py --dump "$DUMP" --rest-dump "$REST" --npz "$NPZ" \
  --out "$OUTDIR" --workers "$WORKERS" --sessions "$SESSIONS"
echo "== gate3s done $(date -Is)"
wc -l "$OUTDIR/g_v5_rollouts.csv"
