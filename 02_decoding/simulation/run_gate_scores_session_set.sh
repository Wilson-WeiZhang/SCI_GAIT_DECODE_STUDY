#!/usr/bin/env bash
# Cores 20-31 ; 4 BLAS threads.
# Read-only on every patient prep root; writes only result/gate_event_set_*.
set -euo pipefail
cd ${GAIT_PROJECT_ROOT}/02_decoding/simulation
mkdir -p result
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nohup taskset -c 20-31 ${PYTHON:-python} -u gate_scores_session_set.py \
  > result/gate_event_set.log 2>&1 &
echo $! > result/gate_event_set.pid
echo "started pid $(cat result/gate_event_set.pid)"
