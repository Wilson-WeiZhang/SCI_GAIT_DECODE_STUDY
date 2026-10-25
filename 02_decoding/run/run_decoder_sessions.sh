#!/usr/bin/env bash
set -euo pipefail

cd ${GAIT_PROJECT_ROOT}/02_decoding

condition="${CONDITION:-MI}"
session_manifest="${SESSION_MANIFEST:-}"
input_control_mode="${INPUT_CONTROL_MODE:-full}"
input_control_seed="${INPUT_CONTROL_SEED:-20260514}"
cross_condition="${CROSS_CONDITION:-}"
cv_mode="${CV_MODE:-random}"
outer_k="${OUTER_K:-5}"
outer_folds="${OUTER_FOLDS:-[]}"
inner_k="${INNER_K:-3}"
tag_suffix="${TAG_SUFFIX:-}"
aux_condition="${AUX_CONDITION:-none}"
aux_mode="${AUX_MODE:-phase_decoder_only}"
aux_inner_mode="${AUX_INNER_MODE:-off}"
post_mode="${POST_MODE:-none}"
post_components="${POST_COMPONENTS:-6}"
post_basis_components="${POST_BASIS_COMPONENTS:-8}"
post_lambda="${POST_LAMBDA:-0.01}"
post_lambda_grid="${POST_LAMBDA_GRID:-[0.001 0.003 0.01 0.03 0.1 0.3 1]}"
post_inner_k="${POST_INNER_K:-3}"
post_lambda_grid="${post_lambda_grid},'post_inner_k',${post_inner_k}"
post_guard_margin="${POST_GUARD_MARGIN:-0.0}"
post_fir_lag_ms="${POST_FIR_LAG_MS:--400:40:0}"
post_fir_lag_unit="${POST_FIR_LAG_UNIT:-ms}"
post_fir_causal_only="${POST_FIR_CAUSAL_ONLY:-true}"
rpt_pred_smooth_sec="${RPT_PRED_SMOOTH_SEC:-0}"
bp_lo="${BP_LO:-0.1}"
bp_hi="${BP_HI:-3.0}"
bp_band_csv="${BP_BAND_CSV:-}"
bp_band_harmonic_k="${BP_BAND_HARMONIC_K:-0}"
bp_band_harmonic_k_list="${BP_BAND_HARMONIC_K_LIST:-[]}"
bp_band_feature_mode="${BP_BAND_FEATURE_MODE:-real}"
bp_channel_labels="${BP_CHANNEL_LABELS:-}"
phase_lo="${PHASE_LO:-0.3}"
phase_hi="${PHASE_HI:-2.0}"
lag_ms="${LAG_MS:--2000:100:0}"
phase_recon="${PHASE_RECON:-template}"
phase_target_mode="${PHASE_TARGET_MODE:-phase_pj}"
harmonic_k="${HARMONIC_K:-3}"
phase_delay_gain="${PHASE_DELAY_GAIN:-1.0}"
phase_select_margin="${PHASE_SELECT_MARGIN:-0.0}"
phase_jitter_ms="${PHASE_JITTER_MS:-0}"
lambda_mode="${LAMBDA_MODE:-scalar}"
lambda_grid="${LAMBDA_GRID:-[1e-3 1e-2 0.1 1 10 100]}"
stage1_lambda_mode="${STAGE1_LAMBDA_MODE:-innercv}"
alpha_grid="${ALPHA_GRID:--0.5:0.05:0.8}"
feature_mode="${FEATURE_MODE:-delta}"
eeg_baseline_mode="${EEG_BASELINE_MODE:-none}"
shared_pca_file="${SHARED_PCA_FILE:-}"
shared_pca_components="${SHARED_PCA_COMPONENTS:-20}"
shared_pca_scaler_scope="${SHARED_PCA_SCALER_SCOPE:-fold}"
harmonic_ssd_append_filters="${HARMONIC_SSD_APPEND_FILTERS:-3}"
pca_score_gate="${PCA_SCORE_GATE:-none}"
pca_score_gate_agg="${PCA_SCORE_GATE_AGG:-p95}"
pca_score_gate_mad_k="${PCA_SCORE_GATE_MAD_K:-3.0}"
pca_score_gate_quantile="${PCA_SCORE_GATE_QUANTILE:-0.95}"
pca_score_gate_max_drop_frac="${PCA_SCORE_GATE_MAX_DROP_FRAC:-0.25}"
pca_score_gate_min_trials="${PCA_SCORE_GATE_MIN_TRIALS:-5}"
pca_score_gate_shrinkage="${PCA_SCORE_GATE_SHRINKAGE:-0.10}"
pca_score_gate_apply_test="${PCA_SCORE_GATE_APPLY_TEST:-true}"
pca_score_gate_scope="${PCA_SCORE_GATE_SCOPE:-fold}"
smooth="${SMOOTH:-lp}"
diag_dir="${DIAG_DIR:-}"
trial_dump_dir="${TRIAL_DUMP_DIR:-}"
trial_dump_stage1_models="${TRIAL_DUMP_STAGE1_MODELS:-false}"
export TRIAL_DUMP_STAGE1_MODELS="${trial_dump_stage1_models}"
stage1_readout_mode="${STAGE1_READOUT_MODE:-plain}"
group_prior_file="${GROUP_PRIOR_FILE:-}"
group_shrink_grid="${GROUP_SHRINK_GRID:-[0]}"
group_shrink_force="${GROUP_SHRINK_FORCE:--1}"
diag_max_trials="${DIAG_MAX_TRIALS:-2}"
diag_plot_pca="${DIAG_PLOT_PCA:-true}"
diag_low_trial_threshold="${DIAG_LOW_TRIAL_THRESHOLD:-NaN}"
diag_low_max_per_fold="${DIAG_LOW_MAX_PER_FOLD:-6}"
gate_csv="${GATE_CSV:-}"
gate_metric="${GATE_METRIC:-mu_beta_erd_pct}"
gate_threshold="${GATE_THRESHOLD:-Inf}"
gate_keep_frac="${GATE_KEEP_FRAC:-1.0}"
min_gate_trials="${MIN_GATE_TRIALS:-5}"
trim_start_sec="${TRIM_START_SEC:-0}"
trim_end_sec="${TRIM_END_SEC:-0}"
valid_from_sec="${VALID_FROM_SEC:-0}"
min_trial_sec="${MIN_TRIAL_SEC:-0}"
post_train_from_sec="${POST_TRAIN_FROM_SEC:-0}"
max_parallel="${MAX_PARALLEL:-2}"
threads="${MAX_THREADS:-4}"
session_chunk_size="${SESSION_CHUNK_SIZE:-1}"
subset_tag="${SUBSET_TAG:-n9}"
session_list="${SESSION_LIST:-subj01_sess01 subj02_sess01 subj04_sess01 subj10_sess02 subj13_sess01 subj16_sess01 subj18_sess01 subj24_sess02 subj07_sess01}"

read -r -a sessions <<< "${session_list}"
if [[ "${session_chunk_size}" -lt 1 ]]; then
  echo "SESSION_CHUNK_SIZE must be >= 1" >&2
  exit 2
fi

export OMP_NUM_THREADS="${threads}"
export MKL_NUM_THREADS="${threads}"
export OPENBLAS_NUM_THREADS="${threads}"

pids=()
status=0
csvs=()
chunk_idx=0
for ((start=0; start<${#sessions[@]}; start+=session_chunk_size)); do
  chunk=("${sessions[@]:start:session_chunk_size}")
  if [[ "${#chunk[@]}" -eq 1 ]]; then
    tag="decoder_${condition}${tag_suffix}_${subset_tag}_${chunk[0]}"
    label="${chunk[0]}"
  else
    chunk_idx=$((chunk_idx + 1))
    tag="decoder_${condition}${tag_suffix}_${subset_tag}_chunk${chunk_idx}"
    label="$(IFS=,; echo "${chunk[*]}")"
  fi
  csv="result/performance_imagined_walking_${tag}.csv"
  csvs+=("${csv}")
  log="result/${tag}.log"
  if [[ -s "${csv}" ]]; then
    echo "reuse existing ${csv}"
    continue
  fi
  matlab_sessions=""
  for sess in "${chunk[@]}"; do
    if [[ -n "${matlab_sessions}" ]]; then
      matlab_sessions="${matlab_sessions},"
    fi
    matlab_sessions="${matlab_sessions}'${sess}'"
  done
  ${MATLAB:-matlab} -nodisplay -nosplash \
    -batch "addpath('../core'); run_decoder('session_manifest','${session_manifest}','sessions',{${matlab_sessions}},'condition','${condition}','result_tag','${tag}','input_control_mode','${input_control_mode}','input_control_seed',${input_control_seed},'cross_condition','${cross_condition}','cv_mode','${cv_mode}','outer_k',${outer_k},'outer_folds',${outer_folds},'inner_k',${inner_k},'aux_condition','${aux_condition}','aux_mode','${aux_mode}','aux_inner_mode','${aux_inner_mode}','post_mode','${post_mode}','post_components',${post_components},'post_basis_components',${post_basis_components},'post_lambda',${post_lambda},'post_lambda_grid',${post_lambda_grid},'post_guard_margin',${post_guard_margin},'post_fir_lag_ms',${post_fir_lag_ms},'post_fir_lag_unit','${post_fir_lag_unit}','post_fir_causal_only',${post_fir_causal_only},'rpt_pred_smooth_sec',${rpt_pred_smooth_sec},'bp_lo',${bp_lo},'bp_hi',${bp_hi},'bp_band_csv','${bp_band_csv}','bp_band_harmonic_k',${bp_band_harmonic_k},'bp_band_harmonic_k_list',${bp_band_harmonic_k_list},'bp_band_feature_mode','${bp_band_feature_mode}','bp_channel_labels','${bp_channel_labels}','phase_lo',${phase_lo},'phase_hi',${phase_hi},'lag_ms',${lag_ms},'phase_recon','${phase_recon}','phase_target_mode','${phase_target_mode}','harmonic_k',${harmonic_k},'phase_delay_gain',${phase_delay_gain},'phase_select_margin',${phase_select_margin},'phase_jitter_ms',${phase_jitter_ms},'lambda_grid',${lambda_grid},'lambda_mode','${lambda_mode}','stage1_lambda_mode','${stage1_lambda_mode}','alpha_grid',${alpha_grid},'feature_mode','${feature_mode}','eeg_baseline_mode','${eeg_baseline_mode}','shared_pca_file','${shared_pca_file}','shared_pca_components',${shared_pca_components},'shared_pca_scaler_scope','${shared_pca_scaler_scope}','harmonic_ssd_append_filters',${harmonic_ssd_append_filters},'pca_score_gate','${pca_score_gate}','pca_score_gate_agg','${pca_score_gate_agg}','pca_score_gate_mad_k',${pca_score_gate_mad_k},'pca_score_gate_quantile',${pca_score_gate_quantile},'pca_score_gate_max_drop_frac',${pca_score_gate_max_drop_frac},'pca_score_gate_min_trials',${pca_score_gate_min_trials},'pca_score_gate_shrinkage',${pca_score_gate_shrinkage},'pca_score_gate_apply_test',${pca_score_gate_apply_test},'pca_score_gate_scope','${pca_score_gate_scope}','smooth','${smooth}','diag_dir','${diag_dir}','trial_dump_dir','${trial_dump_dir}','diag_max_trials',${diag_max_trials},'diag_plot_pca',${diag_plot_pca},'diag_low_trial_threshold',${diag_low_trial_threshold},'diag_low_max_per_fold',${diag_low_max_per_fold},'gate_csv','${gate_csv}','gate_metric','${gate_metric}','gate_threshold',${gate_threshold},'gate_keep_frac',${gate_keep_frac},'min_gate_trials',${min_gate_trials},'trim_start_sec',${trim_start_sec},'trim_end_sec',${trim_end_sec},'stage1_readout_mode','${stage1_readout_mode}','group_prior_file','${group_prior_file}','group_shrink_grid',${group_shrink_grid},'group_shrink_force',${group_shrink_force},'max_threads',${threads},'valid_from_sec',${valid_from_sec},'min_trial_sec',${min_trial_sec},'post_train_from_sec',${post_train_from_sec})" \
    > "${log}" 2>&1 &
  pids+=("$!")
  echo "started ${condition} ${input_control_mode} ${label} pid=${pids[-1]}"

  while [[ "${#pids[@]}" -ge "${max_parallel}" ]]; do
    live=()
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        live+=("${pid}")
      else
        wait "${pid}" || status=1
      fi
    done
    pids=("${live[@]}")
    [[ "${#pids[@]}" -lt "${max_parallel}" ]] || sleep 5
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

if [[ "${status}" -ne 0 ]]; then
  echo "At least one ${condition} ${input_control_mode} session failed" >&2
  exit "${status}"
fi

out="result/performance_imagined_walking_decoder_${condition}${tag_suffix}_${subset_tag}_parallel.csv"
rm -f "${out}"
first=1
for csv in "${csvs[@]}"; do
  if [[ ! -s "${csv}" ]]; then
    echo "missing output ${csv}" >&2
    exit 2
  fi
  if [[ "${first}" -eq 1 ]]; then
    cp "${csv}" "${out}"
    first=0
  else
    tail -n +2 "${csv}" >> "${out}"
  fi
done

echo "merged ${out}"
