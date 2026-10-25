#!/usr/bin/env bash
#
# One shared PCA basis build on the causal triples (Step 07).
#
# Usage (from 02_decoding/): bash run/build_basis.sh <iw|exec|hglobal> <cpu list> <parallel workers>
#   iw       imagined walking, leave-one-participant-out: one <label>_shared_delta_pca_lso.mat per session
#   exec     the same for real walking (condition EXEC, the real-walking manifest and band table)
#   hglobal  imagined walking, one basis pooled over the 80 healthy sessions (the patient decoder's basis)
#
# The MATLAB call is that of core/compute_shared_delta_pca_lso_basis.m with the decoder's causal
# features (cdm_o1_c0p18_causal), GAIT_TRIPLE_ROOT = the causal healthy triples. One build at a time
# (a lock file), started only with at least MAIN_BASIS_MIN_GB (default 120) GB of memory available
# and stopped if it drops below 40 GB; refuses an existing output folder and any session without a
# passing causal-preprocessing check. Ends with run/check_basis.py, which writes BASES_READY_<kind>.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/run_paths.sh"
KIND="$1"; CORES="$2"; W="$3"; FEAT=cdm_o1_c0p18_causal; TAG="main"
cd "$MAIN_DMI" || exit 2
export GAIT_TRIPLE_ROOT="$MAIN_H_TRIPLE"
unset GAIT_GONI_QC_TRIAL_FLAGS
case "$KIND" in
  iw)      cond=MI;   manifest="result/session_manifest_${TAG}.csv"
           band_csv="result/diag_${TAG}_goni_psd_peaks/${TAG}_goni_peak_bands.csv"; basis_dir="$MAIN_BASIS_IW"
           extra="'leave_subject_out',true" ;;
  exec)    cond=EXEC; manifest="result/session_manifest_${TAG}_rw.csv"
           band_csv="result/diag_${TAG}_exec_goni_psd_peaks/${TAG}_exec_goni_peak_bands.csv"; basis_dir="$MAIN_BASIS_EXEC"
           extra="'leave_subject_out',true" ;;
  hglobal) cond=MI;   manifest="result/session_manifest_${TAG}.csv"
           band_csv="result/diag_${TAG}_goni_psd_peaks/${TAG}_goni_peak_bands.csv"; basis_dir="$MAIN_BASIS_HG"
           extra="'global_only',true,'write_global_basis',true,'global_label','healthy_global'" ;;
  *) echo "KIND must be iw|exec|hglobal" >&2; exit 2 ;;
esac
B=$MAIN_R/bases; mkdir -p "$B/logs"
log="$B/logs/basis_${KIND}.log"; mem="$B/logs/basis_${KIND}_mem.log"
job_storage="/tmp/matlab_job_storage_${USER}_basis_${KIND}"
exec 9> "$B/.lock"
echo "[basis] $(date +%F_%T) waiting for the basis lock (one build at a time) kind=$KIND" | tee -a "$log"
flock 9
[ -e "$basis_dir" ] && { echo "[basis] REFUSE: $basis_dir exists" | tee -a "$log"; exit 2; }
for f in "$manifest" "$band_csv"; do [ -s "$f" ] || { echo "[basis] MISSING $f" | tee -a "$log"; exit 2; }; done
# prerequisite: every manifest session has a causal triple with a passing prep check
nbad=0
for s in $(tail -n +2 "$manifest" | cut -d, -f1); do
  c="$MAIN_H_PREP/check/$s.csv"
  if [ ! -s "$GAIT_TRIPLE_ROOT/$s/${s}_eeg.mat" ] || [ ! -s "$c" ] || [ "$(tail -1 "$c" | awk -F, '{print $NF}')" != True ]; then
    echo "[basis] triple not ready: $s" | tee -a "$log"; nbad=$((nbad + 1)); fi
done
[ "$nbad" -eq 0 ] || { echo "[basis] REFUSE: $nbad sessions without a checked causal triple" | tee -a "$log"; exit 2; }
MEMAV="$(dirname "$0")/mem_available.sh"      # MemAvailable + reclaimable ZFS ARC (MB); see mem_available.sh
while [ "$(( $(bash "$MEMAV") / 1024 ))" -lt "${MAIN_BASIS_MIN_GB:-120}" ]; do sleep 30; done
echo "[basis] $(date +%F_%T) START kind=$KIND cores=$CORES workers=$W root=$GAIT_TRIPLE_ROOT avail=$(( $(bash "$MEMAV") / 1024 ))G -> $basis_dir" | tee -a "$log"
rm -rf "$job_storage"
setsid taskset -c "$CORES" "$MAIN_MATLAB" -nodisplay -batch \
  "addpath('../core'); compute_shared_delta_pca_lso_basis('manifest_path','$manifest','condition','$cond','bp_lo',0.1,'bp_hi',2.0,'bp_band_csv','$band_csv','bp_band_harmonic_k_list',[1 2 3],'bp_band_feature_mode','$FEAT','sample_stride',1,'aggregation_mode','correlation','parallel_workers',${W},'job_storage_dir','$job_storage','basis_dir','$basis_dir',$extra)" \
  >> "$log" 2>&1 &
mpid=$!
sleep 2
sid=$(ps -o sid= -p "$mpid" | tr -d ' ')
echo "[basis] matlab pid=$mpid sid=$sid" | tee -a "$log"
peak_rss=0; min_av=999999
echo "time avail_mb session_rss_mb nproc" > "$mem"
while kill -0 "$mpid" 2>/dev/null; do
  av=$(bash "$MEMAV")
  rss=$(ps -s "$sid" -o rss= 2>/dev/null | awk '{s+=$1} END {printf "%d", s/1024}')
  np=$(ps -s "$sid" -o pid= 2>/dev/null | wc -l)
  echo "$(date +%T) $av $rss $np" >> "$mem"
  [ "$rss" -gt "$peak_rss" ] && peak_rss=$rss
  [ "$av" -lt "$min_av" ] && min_av=$av
  if [ "$av" -lt 40960 ]; then
    echo "[basis] $(date +%F_%T) MEMORY GUARD: available ${av} MB < 40 GB -> killing session $sid" | tee -a "$log"
    kill -TERM -- "-$sid" 2>/dev/null; sleep 10; kill -KILL -- "-$sid" 2>/dev/null
  fi
  sleep 5
done
wait "$mpid"; rc=$?
bn=$(find "$basis_dir" -maxdepth 1 -name '*_shared_delta_pca*.mat' 2>/dev/null | wc -l)
{ echo "build_id $MAIN_TAG"; echo "kind $KIND"; echo "GAIT_TRIPLE_ROOT $GAIT_TRIPLE_ROOT"; echo "manifest $manifest"; echo "band_csv $band_csv"
  echo "cores $CORES workers $W"; echo "rc $rc files $bn peak_session_rss_mb $peak_rss min_avail_mb $min_av"
  echo "wrapper_md5 $(md5sum "$HERE/build_basis.sh" | cut -c1-32)"
  echo "generator_md5 $(md5sum 02_decoding/core/compute_shared_delta_pca_lso_basis.m | cut -c1-32)"
  echo "log $log"; echo "date $(date +%F_%T)"; } > "$basis_dir/MAIN_BUILD.txt" 2>/dev/null
echo "[basis] $(date +%F_%T) DONE kind=$KIND rc=$rc files=$bn peak_session_rss_mb=$peak_rss min_avail_mb=$min_av" | tee -a "$log"
rm -rf "$job_storage"
[ "$rc" -eq 0 ] || exit "$rc"
"$MAIN_PY" "$HERE/check_basis.py" "$KIND" 2>&1 | tee -a "$log"
exit "${PIPESTATUS[0]}"
