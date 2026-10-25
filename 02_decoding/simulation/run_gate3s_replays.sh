#!/usr/bin/env bash
# Rhythm metrics (R2-R6) + hip-x traces replayed on the simulation_rollouts_gate3s rollout set (trials with at
# least 3 s of decoder output after the decoder's start-up; PHYS_MIN_DUR_S=3). Output directory simulation_rollouts_gate3s/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
ROOT=${GAIT_PROJECT_ROOT}
DUMP=$ROOT/data/decoder_output/trial_dump_patient
NPZ=$ROOT/data/decoder_output/patient_cycle_average.npz
SESSIONS=pat01_sess04,pat01_sess06,pat01_sess07,pat01_sess09,pat02_sess03,pat02_sess05,pat02_sess06,pat02_sess08,pat02_sess09
OUTDIR=simulation_rollouts_gate3s
WORKERS=${WORKERS:-10}
export PHYS_MIN_DUR_S=3
echo "PHYS_MIN_DUR_S=$PHYS_MIN_DUR_S"
echo "== gate3s rhythm replay start $(date -Is) workers=$WORKERS"
taskset -c 14-31 "$PY" rhythm_metrics.py replay --dump "$DUMP" --npz "$NPZ" \
  --out "$OUTDIR" --workers "$WORKERS" --sessions "$SESSIONS" 2>&1 | tee "$OUTDIR/rhythm_metrics.log"
echo "== gate3s traces dump start $(date -Is)"
taskset -c 14-31 "$PY" dump_traces_patient.py --dump "$DUMP" --npz "$NPZ" --sessions "$SESSIONS" \
  --out "$OUTDIR/traces_x.csv.gz" --workers "$WORKERS" 2>&1 | tee "$OUTDIR/traces_x.log"
echo "== gate3s replays done $(date -Is)"
wc -l "$OUTDIR/rhythm_metrics.csv"
md5sum "$OUTDIR/rhythm_metrics.csv" "$OUTDIR/traces_x.csv.gz"
