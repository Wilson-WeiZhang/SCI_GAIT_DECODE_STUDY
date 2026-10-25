#!/usr/bin/env bash
#
# One MATLAB feature export (Step 08) = one row of exports.csv (run/make_job_lists.py).
#
# Usage (from 02_decoding/): bash run/export_features.sh <cpu list> <export_id>
#
# Runs run/run_decoder_sessions.sh -> core/run_decoder.m with CP_EXPORT_ONLY=1 (CP_EXPORT_REST_ONLY=1 for
# a rest-window export) and the published decoder configuration, with GAIT_TRIPLE_ROOT = the causal
# triples, BP_BAND_CSV = the band table of the row (the whole-session table or a per-fold table of
# fold_tables/), the basis of the same kind (run/build_basis.sh) and the export folder
# <result>/exports/<export_id>/. The env column of the row selects the configuration:
#   h_iw        healthy imagined walking, the session's own leave-one-participant-out basis
#   h_rw        healthy real walking (condition EXEC, real-walking manifest and basis)
#   p_iw        patients: the pooled healthy basis, the session group's manifest, the goniometer
#               QC override (GAIT_GONI_QC_TRIAL_FLAGS)
#   h_rest      healthy rest windows of the resting-input control (seed 20260820, outer fold 1; band
#               table = the paired imagined-walking table)
#   p_rest      patient rest windows (rest-window root of run/patient_rest_root.py, seed 20260829)
#   xt_mi2exec  real-walking trials with the imagined-walking bands (cross-task, train IW -> test RW)
#   xt_exec2mi  imagined-walking trials with the real-walking bands (cross-task, train RW -> test IW)
# Exit codes: 0 export written; 3 a prerequisite is not ready (basis check, preprocessing check, triple,
# rest root), nothing started; 2 bad row or missing input; else the MATLAB exit code. Then run
# python run/check_export.py <export_id>, which writes EXPORT_READY. MAIN_PREREQ_ONLY=1 stops after the
# prerequisite checks, MAIN_SKIP_PREREQ=1 skips them, MAIN_LIST_INPUTS=1 prints the input paths.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/run_paths.sh"
CORES="$1"; EID="$2"; FEAT=cdm_o1_c0p18_causal; THREADS=2; CTAG="main"; R=result
cd "$MAIN_DMI" || exit 2
XCSV=${MAIN_EXPORTS_CSV:-$HERE/exports.csv}; RUNNER=${MAIN_RUNNER:-run/run_decoder_sessions.sh}
QCOVR="$MAIN_GAIT/01_preprocessing/patient/result/qc/goni_qc_trial_override.csv"
row=$(awk -F, -v id="$EID" 'NR == 1 {for (i = 1; i <= NF; i++) c[$i] = i; next}
  $c["export_id"] == id {print $c["env"], $c["session"], $c["band_csv"], $c["out_file"]; exit}' "$XCSV")
[ -n "$row" ] || { echo "no row $EID in exports.csv" >&2; exit 2; }
read -r ENV SESS band_csv OUTF <<< "$row"
XD="$MAIN_EXPORTS/$EID"; LG="$MAIN_EXPORTS/_logs"
LOG="$LG/${EID}.log"
cv_grid="[1e-3 1e-2 0.1 1 10 100]"
lag_grid="-2000:50:0"
unset GAIT_GONI_QC_TRIAL_FLAGS STAGE2_DUMP_DIR CP_EXPORT_DIR CP_EXPORT_ONLY CP_EXPORT_REST_DIR CP_EXPORT_REST_ONLY
case "$ENV" in
  h_iw|h_rest|xt_mi2exec) BK=iw; basis_file="$MAIN_BASIS_IW/${SESS}_shared_delta_pca_lso.mat"; basis_pat="$MAIN_BASIS_IW/%s_shared_delta_pca_lso.mat" ;;
  h_rw|xt_exec2mi)        BK=exec; basis_file="$MAIN_BASIS_EXEC/${SESS}_shared_delta_pca_lso.mat"; basis_pat="$MAIN_BASIS_EXEC/%s_shared_delta_pca_lso.mat" ;;
  p_iw|p_rest)            BK=hglobal; basis_file="$MAIN_BASIS_HG/shared_pca_basis.mat"; basis_pat="$basis_file" ;;
  *) echo "unknown env $ENV" >&2; exit 2 ;;
esac
case "$ENV" in
  p_iw)   TROOT="$MAIN_P_TRIPLE"; CHK="$MAIN_P_PREP/check/$SESS.csv"
          case "$SESS" in
            pat02_sess08) manifest=$R/patient_session_manifest_sess08.csv ;;
            pat02_sess09) manifest=$R/patient_session_manifest_sess09.csv ;;
            *) manifest=$R/patient_session_manifest.csv ;;
          esac ;;
  p_rest) TROOT="$MAIN_P_REST_TRIPLE"; CHK="$MAIN_P_PREP/check/$SESS.csv"
          case "$SESS" in
            pat01_sess04|pat01_sess06|pat01_sess07|pat01_sess09|pat02_sess03|pat02_sess05|pat02_sess06)
              manifest=$R/patient_session_manifest_rest7.csv ;;
            pat01_sess05|pat01_sess08|pat02_sess08|pat02_sess09)
              manifest=$R/patient_session_manifest_rest_add4.csv ;;
            *) echo "no derived rest root for $SESS" >&2; exit 2 ;;
          esac ;;
  h_iw|h_rest) TROOT="$MAIN_H_TRIPLE"; CHK="$MAIN_H_PREP/check/$SESS.csv"; manifest="result/session_manifest_${CTAG}.csv" ;;
  *)      TROOT="$MAIN_H_TRIPLE"; CHK="$MAIN_H_PREP/check/$SESS.csv"; manifest="result/session_manifest_${CTAG}_rw.csv" ;;
esac
if [ "${MAIN_LIST_INPUTS:-0}" = 1 ]; then
  echo "$TROOT/$SESS"; echo "$MAIN_DMI/$band_csv"; echo "$MAIN_DMI/$basis_file"; echo "$MAIN_DMI/$manifest"
  case "$ENV" in p_iw|p_rest) echo "$QCOVR" ;; esac
  echo "$XCSV"; exit 0
fi
# prerequisites (exit 3 = not ready yet; the queue retries later)
if [ "${MAIN_SKIP_PREREQ:-0}" != 1 ]; then
[ -s "$MAIN_R/bases/BASES_READY_$BK" ] || { echo "[export] $EID: basis $BK not ready" >&2; exit 3; }
[ -s "$CHK" ] && [ "$(tail -1 "$CHK" | awk -F, '{print $NF}')" = True ] || { echo "[export] $EID: prep check of $SESS not passed" >&2; exit 3; }
[ -s "$TROOT/$SESS/${SESS}_eeg.mat" ] || { echo "[export] $EID: no triple $TROOT/$SESS" >&2; exit 3; }
[ "$ENV" != p_rest ] || [ -s "$MAIN_P_REST_TRIPLE/READY" ] || { echo "[export] $EID: derived rest root not built" >&2; exit 3; }
fi
[ "${MAIN_PREREQ_ONLY:-0}" = 1 ] && exit 0
mkdir -p "$XD" "$LG"
[ -s "$band_csv" ] || { echo "MISSING band csv $band_csv" >&2; exit 2; }
[ -s "$basis_file" ] || { echo "MISSING basis $basis_file" >&2; exit 2; }
export GAIT_TRIPLE_ROOT="$TROOT"
while [ "$(( $(bash "$HERE/mem_available.sh") / 1024 ))" -lt "${MAIN_MIN_AVAIL_GB:-40}" ]; do sleep 30; done   # ARC-aware (mem_available.sh)
echo "[export] $(date +%F_%T) START eid=$EID env=$ENV sess=$SESS cores=$CORES root=$TROOT band=$band_csv -> $XD"
case "$ENV" in
h_iw)
  export CP_EXPORT_DIR="$XD"; export CP_EXPORT_ONLY=1; COND=MI; TS=_i; ST="$EID"
  SESSION_LIST="$SESS" SUBSET_TAG="$ST" TAG_SUFFIX="$TS" \
  SESSION_MANIFEST="$manifest" OUTER_K="10" OUTER_FOLDS="[1]" INNER_K="10" POST_INNER_K="10" CV_MODE="random" \
  FEATURE_MODE="delta_fold_diagwhiten_shared_pca" \
  SHARED_PCA_FILE="$basis_pat" SHARED_PCA_COMPONENTS="20" SHARED_PCA_SCALER_SCOPE="fold" \
  BP_LO="0.1" BP_HI="2.0" BP_BAND_CSV="$band_csv" BP_BAND_HARMONIC_K="1" \
  BP_BAND_HARMONIC_K_LIST="[1 2 3]" BP_BAND_FEATURE_MODE="$FEAT" \
  LAG_MS="$lag_grid" EEG_BASELINE_MODE="mean" ALPHA_GRID="0" \
  PHASE_RECON="delay_template" PHASE_TARGET_MODE="phase_pj" LAMBDA_MODE="scalar" \
  LAMBDA_GRID="$cv_grid" STAGE1_LAMBDA_MODE="innercv" \
  POST_MODE="prediction_fir_cvlambda" POST_LAMBDA="1.0" \
  POST_LAMBDA_GRID="$cv_grid" POST_GUARD_MARGIN="0.0" POST_FIR_LAG_MS="-2000:50:0" \
  POST_FIR_LAG_UNIT="ms" POST_FIR_CAUSAL_ONLY="true" DIAG_PLOT_PCA="false" DIAG_MAX_TRIALS="0" \
  TRIAL_DUMP_STAGE1_MODELS="false" TRIAL_DUMP_DIR="" VALID_FROM_SEC="0" MIN_TRIAL_SEC="0" POST_TRAIN_FROM_SEC="0" \
  GAIT_TRIPLE_ROOT="$TROOT" MAX_PARALLEL="1" MAX_THREADS="$THREADS" SESSION_CHUNK_SIZE="1" \
  taskset -c "$CORES" bash "$RUNNER" > "$LOG" 2>&1 ;;
h_rw)
  export CP_EXPORT_DIR="$XD"; export CP_EXPORT_ONLY=1; COND=EXEC; TS=_w; ST="$EID"
  CONDITION="EXEC" SESSION_LIST="$SESS" SUBSET_TAG="$ST" TAG_SUFFIX="$TS" \
  SESSION_MANIFEST="$manifest" OUTER_K="10" OUTER_FOLDS="[1]" INNER_K="10" POST_INNER_K="10" CV_MODE="random" \
  FEATURE_MODE="delta_fold_diagwhiten_shared_pca" \
  SHARED_PCA_FILE="$basis_pat" SHARED_PCA_COMPONENTS="20" SHARED_PCA_SCALER_SCOPE="fold" \
  BP_LO="0.1" BP_HI="2.0" BP_BAND_CSV="$band_csv" BP_BAND_HARMONIC_K="1" \
  BP_BAND_HARMONIC_K_LIST="[1 2 3]" BP_BAND_FEATURE_MODE="$FEAT" \
  LAG_MS="$lag_grid" EEG_BASELINE_MODE="mean" ALPHA_GRID="0" \
  PHASE_RECON="delay_template" PHASE_TARGET_MODE="phase_pj" LAMBDA_MODE="scalar" \
  LAMBDA_GRID="$cv_grid" STAGE1_LAMBDA_MODE="innercv" \
  POST_MODE="prediction_fir_cvlambda" POST_LAMBDA="1.0" \
  POST_LAMBDA_GRID="$cv_grid" POST_GUARD_MARGIN="0.0" POST_FIR_LAG_MS="-2000:50:0" \
  POST_FIR_LAG_UNIT="ms" POST_FIR_CAUSAL_ONLY="true" DIAG_PLOT_PCA="false" DIAG_MAX_TRIALS="0" \
  TRIAL_DUMP_STAGE1_MODELS="false" TRIAL_DUMP_DIR="" VALID_FROM_SEC="0" MIN_TRIAL_SEC="0" POST_TRAIN_FROM_SEC="0" \
  GAIT_TRIPLE_ROOT="$TROOT" MAX_PARALLEL="1" MAX_THREADS="$THREADS" SESSION_CHUNK_SIZE="1" \
  taskset -c "$CORES" bash "$RUNNER" > "$LOG" 2>&1 ;;
p_iw)
  export GAIT_GONI_QC_TRIAL_FLAGS="$QCOVR"
  [ -s "$GAIT_GONI_QC_TRIAL_FLAGS" ] || { echo "MISSING qc override $GAIT_GONI_QC_TRIAL_FLAGS" >&2; exit 2; }
  export CP_EXPORT_DIR="$XD"; export CP_EXPORT_ONLY=1; COND=MI; TS=_p; ST="$EID"
  tdump_dir=""; s1="false"
  SESSION_LIST="$SESS" SUBSET_TAG="$ST" TAG_SUFFIX="$TS" \
  SESSION_MANIFEST="$manifest" CONDITION="MI" \
  OUTER_K="10" OUTER_FOLDS="[1 2 3 4 5 6 7 8 9 10]" INNER_K="10" POST_INNER_K="10" CV_MODE="random" \
  FEATURE_MODE="delta_fold_diagwhiten_shared_pca" \
  SHARED_PCA_FILE="$basis_file" SHARED_PCA_COMPONENTS="20" SHARED_PCA_SCALER_SCOPE="fold" \
  BP_LO="0.1" BP_HI="2.0" BP_BAND_CSV="$band_csv" BP_BAND_HARMONIC_K="1" \
  BP_BAND_HARMONIC_K_LIST="[1 2 3]" BP_BAND_FEATURE_MODE="$FEAT" \
  LAG_MS="$lag_grid" EEG_BASELINE_MODE="mean" ALPHA_GRID="0" \
  PHASE_RECON="delay_template" PHASE_TARGET_MODE="phase_pj" LAMBDA_MODE="scalar" \
  LAMBDA_GRID="$cv_grid" STAGE1_LAMBDA_MODE="innercv" \
  POST_MODE="prediction_fir_cvlambda" POST_LAMBDA="1.0" \
  POST_LAMBDA_GRID="$cv_grid" POST_GUARD_MARGIN="0.0" POST_FIR_LAG_MS="$lag_grid" \
  POST_FIR_LAG_UNIT="ms" POST_FIR_CAUSAL_ONLY="true" DIAG_PLOT_PCA="false" DIAG_MAX_TRIALS="0" \
  TRIAL_DUMP_DIR="$tdump_dir" TRIAL_DUMP_STAGE1_MODELS="$s1" VALID_FROM_SEC="0" MIN_TRIAL_SEC="0" \
  MAX_PARALLEL="1" MAX_THREADS="$THREADS" SESSION_CHUNK_SIZE="1" \
  taskset -c "$CORES" bash "$RUNNER" > "$LOG" 2>&1 ;;
h_rest)
  FOLD=1; ICM=resting_random_window_test_only; ICS=20260820; CVM=random; PCA=20
  export CP_EXPORT_REST_DIR="$XD"; export CP_EXPORT_REST_ONLY=1; COND=MI; TS=_r; ST="${EID}_f${FOLD}"
  SESSION_LIST="$SESS" SUBSET_TAG="$ST" TAG_SUFFIX="$TS" \
  SESSION_MANIFEST="$manifest" OUTER_K="10" OUTER_FOLDS="[${FOLD}]" INNER_K="10" POST_INNER_K="10" CV_MODE="$CVM" \
  INPUT_CONTROL_MODE="$ICM" INPUT_CONTROL_SEED="$ICS" \
  FEATURE_MODE="delta_fold_diagwhiten_shared_pca" \
  SHARED_PCA_FILE="$basis_pat" SHARED_PCA_COMPONENTS="$PCA" SHARED_PCA_SCALER_SCOPE="fold" \
  BP_LO="0.1" BP_HI="2.0" BP_BAND_CSV="$band_csv" BP_BAND_HARMONIC_K="1" \
  BP_BAND_HARMONIC_K_LIST="[1 2 3]" BP_BAND_FEATURE_MODE="$FEAT" \
  LAG_MS="$lag_grid" EEG_BASELINE_MODE="mean" ALPHA_GRID="0" \
  PHASE_RECON="delay_template" PHASE_TARGET_MODE="phase_pj" LAMBDA_MODE="scalar" \
  LAMBDA_GRID="$cv_grid" STAGE1_LAMBDA_MODE="innercv" \
  POST_MODE="prediction_fir_cvlambda" POST_LAMBDA="1.0" \
  POST_LAMBDA_GRID="$cv_grid" POST_GUARD_MARGIN="0.0" POST_FIR_LAG_MS="-2000:50:0" \
  POST_FIR_LAG_UNIT="ms" POST_FIR_CAUSAL_ONLY="true" DIAG_PLOT_PCA="false" DIAG_MAX_TRIALS="0" \
  TRIAL_DUMP_STAGE1_MODELS="false" TRIAL_DUMP_DIR="" VALID_FROM_SEC="0" MIN_TRIAL_SEC="0" POST_TRAIN_FROM_SEC="0" \
  GAIT_TRIPLE_ROOT="$TROOT" MAX_PARALLEL="1" MAX_THREADS="$THREADS" SESSION_CHUNK_SIZE="1" \
  taskset -c "$CORES" bash "$RUNNER" > "$LOG" 2>&1 ;;
p_rest)
  [ -s "$manifest" ] || { echo "MISSING $manifest" >&2; exit 2; }
  export GAIT_GONI_QC_TRIAL_FLAGS="$QCOVR"
  [ -s "$GAIT_GONI_QC_TRIAL_FLAGS" ] || { echo "MISSING qc override $GAIT_GONI_QC_TRIAL_FLAGS" >&2; exit 2; }
  export CP_EXPORT_REST_DIR="$XD"; export CP_EXPORT_REST_ONLY=1; COND=MI; TS=_q; PHASE=export; ST="${EID}_${PHASE}"
  tdump_dir=""; s1="false"
  SESSION_LIST="$SESS" SUBSET_TAG="$ST" TAG_SUFFIX="$TS" \
  SESSION_MANIFEST="$manifest" CONDITION="MI" \
  INPUT_CONTROL_MODE="resting_random_window_test_only" INPUT_CONTROL_SEED="20260829" \
  OUTER_K="10" OUTER_FOLDS="[1 2 3 4 5 6 7 8 9 10]" INNER_K="10" POST_INNER_K="10" CV_MODE="random" \
  FEATURE_MODE="delta_fold_diagwhiten_shared_pca" \
  SHARED_PCA_FILE="$basis_file" SHARED_PCA_COMPONENTS="20" SHARED_PCA_SCALER_SCOPE="fold" \
  BP_LO="0.1" BP_HI="2.0" BP_BAND_CSV="$band_csv" BP_BAND_HARMONIC_K="1" \
  BP_BAND_HARMONIC_K_LIST="[1 2 3]" BP_BAND_FEATURE_MODE="$FEAT" \
  LAG_MS="$lag_grid" EEG_BASELINE_MODE="mean" ALPHA_GRID="0" \
  PHASE_RECON="delay_template" PHASE_TARGET_MODE="phase_pj" LAMBDA_MODE="scalar" \
  LAMBDA_GRID="$cv_grid" STAGE1_LAMBDA_MODE="innercv" \
  POST_MODE="prediction_fir_cvlambda" POST_LAMBDA="1.0" \
  POST_LAMBDA_GRID="$cv_grid" POST_GUARD_MARGIN="0.0" POST_FIR_LAG_MS="$lag_grid" \
  POST_FIR_LAG_UNIT="ms" POST_FIR_CAUSAL_ONLY="true" DIAG_PLOT_PCA="false" DIAG_MAX_TRIALS="0" \
  TRIAL_DUMP_DIR="$tdump_dir" TRIAL_DUMP_STAGE1_MODELS="$s1" VALID_FROM_SEC="0" MIN_TRIAL_SEC="0" \
  MAX_PARALLEL="1" MAX_THREADS="$THREADS" SESSION_CHUNK_SIZE="1" \
  taskset -c "$CORES" bash "$RUNNER" > "$LOG" 2>&1 ;;
xt_mi2exec|xt_exec2mi)
  [ -s "$manifest" ] || { echo "MISSING $manifest" >&2; exit 2; }
  if [ "$ENV" = xt_mi2exec ]; then COND=EXEC; TS=_x; else COND=MI; TS=_y; fi
  export CP_EXPORT_DIR="$XD"; export CP_EXPORT_ONLY=1; ST="$EID"
  CONDITION="$COND" SESSION_LIST="$SESS" SUBSET_TAG="$ST" TAG_SUFFIX="$TS" \
  SESSION_MANIFEST="$manifest" OUTER_K="10" OUTER_FOLDS="[1]" INNER_K="10" POST_INNER_K="10" CV_MODE="random" \
  FEATURE_MODE="delta_fold_diagwhiten_shared_pca" \
  SHARED_PCA_FILE="$basis_pat" SHARED_PCA_COMPONENTS="20" SHARED_PCA_SCALER_SCOPE="fold" \
  BP_LO="0.1" BP_HI="2.0" BP_BAND_CSV="$band_csv" BP_BAND_HARMONIC_K="1" \
  BP_BAND_HARMONIC_K_LIST="[1 2 3]" BP_BAND_FEATURE_MODE="$FEAT" \
  LAG_MS="$lag_grid" EEG_BASELINE_MODE="mean" ALPHA_GRID="0" \
  PHASE_RECON="delay_template" PHASE_TARGET_MODE="phase_pj" LAMBDA_MODE="scalar" \
  LAMBDA_GRID="$cv_grid" STAGE1_LAMBDA_MODE="innercv" \
  POST_MODE="prediction_fir_cvlambda" POST_LAMBDA="1.0" \
  POST_LAMBDA_GRID="$cv_grid" POST_GUARD_MARGIN="0.0" POST_FIR_LAG_MS="-2000:50:0" \
  POST_FIR_LAG_UNIT="ms" POST_FIR_CAUSAL_ONLY="true" DIAG_PLOT_PCA="false" DIAG_MAX_TRIALS="0" \
  TRIAL_DUMP_STAGE1_MODELS="false" TRIAL_DUMP_DIR="" VALID_FROM_SEC="0" MIN_TRIAL_SEC="0" POST_TRAIN_FROM_SEC="0" \
  GAIT_TRIPLE_ROOT="$TROOT" MAX_PARALLEL="1" MAX_THREADS="$THREADS" SESSION_CHUNK_SIZE="1" \
  taskset -c "$CORES" bash "$RUNNER" > "$LOG" 2>&1 ;;
esac
rc=$?
mlog="result/decoder_${COND}${TS}_${ST}_${SESS}.log"
sc=""; grep -q "H1 self-check PASS" "$mlog" 2>/dev/null && sc=H1_selfcheck_PASS
[ -s "$XD/$OUTF" ] && ex=export_OK || ex=export_MISSING
echo "[export] $(date +%F_%T) DONE eid=$EID rc=$rc $ex $sc mlog=$mlog"
[ "$ex" = export_OK ] || exit $(( rc == 0 ? 1 : rc ))
exit 0
