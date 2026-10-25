#!/usr/bin/env bash
# Runs gate_fusion.py and gate_fusion_working_point.py.
set -euo pipefail
cd ${GAIT_PROJECT_ROOT}/02_decoding/simulation
mkdir -p result
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
taskset -c 0-15 ${PYTHON:-python} gate_fusion.py 2>&1 \
  | tee result/gate_fusion.log
taskset -c 0-15 ${PYTHON:-python} gate_fusion_working_point.py 2>&1 \
  | tee result/gate_fusion_working_point.log
