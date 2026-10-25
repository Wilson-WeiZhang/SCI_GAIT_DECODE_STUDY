#!/usr/bin/env bash
#
# The walk-intent gate on the causal preprocessing: builds the gate session set of the causal patient
# triples (build_gate_session_set_causal.py, once), then scores it (gate_scores_causal.py), two cores.
# Usage: bash run_gate_scores_causal.sh [cpu list for Linux taskset]
set -euo pipefail
PY=${PYTHON:-python}
D=${GAIT_PROJECT_ROOT}/02_decoding/simulation
L=$D/../logs
CORES=${1:-26-27}
mkdir -p $L
echo "host $(hostname); load $(cut -d' ' -f1-3 /proc/loadavg); cores $CORES; start $(date -Is)"
cd ${GAIT_PROJECT_ROOT}/02_decoding/simulation
md5sum $D/build_gate_session_set_causal.py $D/gate_scores_causal.py gate_scores_within_session.py gate_window3_fold_reference.py \
  gate_window3_within_session.py gate_fusion.py gate_features.py karcher_checked.py \
  result/gate_within_session_scores_T3.csv
R=${GAIT_PROJECT_ROOT}/data/decoder_output/main/triples/pat_gate
if [ ! -e $R/READY ]; then
  taskset -c $CORES nice -n 19 $PY $D/build_gate_session_set_causal.py > $L/gate_root.log 2>&1
  echo "gate root exit 0; $(date -Is)"; tail -1 $L/gate_root.log
fi
taskset -c $CORES nice -n 19 $PY $D/gate_scores_causal.py > $L/gate_within10.log 2>&1
echo "within10 exit 0; $(date -Is)"; tail -4 $L/gate_within10.log
