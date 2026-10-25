#!/usr/bin/env bash
# Read-only on every patient prep root; writes only result/gate_within_session_*.
set -euo pipefail
cd ${GAIT_PROJECT_ROOT}/02_decoding/simulation
mkdir -p result
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nohup taskset -c 0-19 ${PYTHON:-python} -u gate_scores_within_session.py \
  > result/gate_within_session.log 2>&1 &
echo $! > result/gate_within_session.pid
echo "started pid $(cat result/gate_within_session.pid)"
