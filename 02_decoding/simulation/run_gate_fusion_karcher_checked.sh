#!/usr/bin/env bash
# Cores 20-31 ; 4 BLAS threads.
# Read-only on the patient prep root; writes only result/gate_fused_checked_*.
set -euo pipefail
cd ${GAIT_PROJECT_ROOT}/02_decoding/simulation
mkdir -p result
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nohup taskset -c 20-31 ${PYTHON:-python} -u gate_scores_karcher_checked.py \
  > result/gate_fused_checked.log 2>&1 &
echo $! > result/gate_fused_checked.pid
echo "started pid $(cat result/gate_fused_checked.pid)"
