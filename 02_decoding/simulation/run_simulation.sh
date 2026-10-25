#!/usr/bin/env bash
# The Figure 7 simulation as run for the published build (GAIT_F7_SIM=causal_from2s, 3 s usable-duration gate),
# on the decoder output data/decoder_output/main/: run_rollouts.py (rollouts), dump_traces_patient.py (hip-x traces),
# rhythm_metrics.py replay (rhythm metrics), replay_walker_matched.py (window-matched Walker: rollouts, rhythm,
# traces). Usage: bash run_simulation.sh [workers, default 8] [cpu list for Linux taskset, default 72-79]; BLAS
# threads 1; each stage starts only with at least 40 GB of memory available. Output
# 02_decoding/simulation/simulation_rollouts_gate3s/.
set -euo pipefail
PY=${PYTHON:-python}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RD=main
R=$ROOT/data/decoder_output/$RD
DUMP=$R/pat_post_operative_iw
NPZ=$R/cycavg_pat.npz
SESSIONS=pat01_sess04,pat01_sess06,pat01_sess07,pat01_sess09,pat02_sess03,pat02_sess05,pat02_sess06,pat02_sess08,pat02_sess09
W=${1:-8}
CORES=${2:-72-79}
unset GAIT_F7_WARM GAIT_F7_INIT
export GAIT_F7_SIM=causal_from2s PHYS_MIN_DUR_S=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GAIT_NORMATIVE_CSV=${GAIT_NORMATIVE_CSV:?"set GAIT_NORMATIVE_CSV to the normative sagittal gait template fukuchi2018_sagittal_young24_50pt.csv (not distributed)"}
gate() { local a; a=$(free -g | awk '/^Mem:/{print $7}'); echo "free avail ${a} GB; load $(cut -d' ' -f1-3 /proc/loadavg)"; [ "$a" -ge 40 ] || { echo "MEMORY GATE FAIL"; exit 3; }; }
echo "host $(hostname); workers $W; cores $CORES"
cd $ROOT/02_decoding/simulation
md5sum data_io.py replay_rig.py biped_model.py run_rollouts.py dump_traces_patient.py rhythm_metrics.py replay_walker_matched.py ../../03_figures/analysis_config.py $NPZ
OUT=simulation_rollouts_gate3s
O=$OUT
mkdir -p $O
if [ -e $O/g_v5_rollouts.csv ]; then echo "$O/g_v5_rollouts.csv exists; refusing to overwrite"; exit 4; fi
gate; echo "== run_rollouts start $(date -Is)"
taskset -c $CORES $PY run_rollouts.py --dump $DUMP --npz $NPZ --out $OUT --workers $W --sessions $SESSIONS > $O/run_rollouts.log 2>&1
echo "== run_rollouts done $(date -Is)"; wc -l $O/g_v5_rollouts.csv
gate; echo "== traces start $(date -Is)"
taskset -c $CORES $PY dump_traces_patient.py --dump $DUMP --npz $NPZ --sessions $SESSIONS --out $OUT/traces_x.csv.gz --workers $W > $O/traces_x.log 2>&1
gate; echo "== rhythm replay start $(date -Is)"
taskset -c $CORES $PY rhythm_metrics.py replay --dump $DUMP --npz $NPZ --out $OUT --workers $W --sessions $SESSIONS > $O/rhythm_metrics.log 2>&1
tail -1 $O/rhythm_metrics.log
gate; echo "== walker-matched start $(date -Is)"
taskset -c $CORES $PY replay_walker_matched.py --dump $DUMP --npz $NPZ --sessions $SESSIONS --out $OUT --workers $W > $O/walker_matched.log 2>&1
tail -3 $O/walker_matched.log
echo "== done $(date -Is)"; ls -la $O; md5sum $O/*.csv $O/*.csv.gz  $O/*.json
