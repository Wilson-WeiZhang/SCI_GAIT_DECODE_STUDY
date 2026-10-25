function run_decoder(varargin)
%RUN_DECODER  Session-level EEG-to-joint-angle decoder and feature export.
%
% In the published configuration this function computes the decoder features of
% each session and writes them to a per-session export; the readout is fitted and
% tested in Python on that export (02_decoding/pyport, random 10-fold cross-validation).
%
% Published configuration, set by run/export_features.sh (it calls
% run/run_decoder_sessions.sh, the generic launcher):
%   features   for each goniometer harmonic band k = 1, 2, 3 of the band table bp_band_csv,
%              bp_band_feature_mode = 'cdm_o1_c0p18_causal': a
%              fixed-frequency complex demodulation, i.e. a first-order Butterworth
%              low-pass at 0.18 Hz shifted to the band centre and applied forward
%              only, so every feature sample uses only EEG up to that sample; the
%              real and imaginary parts give two rows per channel and band
%   basis      shared_pca_file = the shared PCA basis learned on the same
%              features (run/build_basis.sh); eeg_baseline_mode mean
%   export     environment CP_EXPORT_DIR: one <label>_export.mat per session with
%              the un-lagged feature window of every trial, the pre-trial feature
%              mean that is subtracted, the joint-angle targets, the trial indices,
%              the outer fold assignment and the basis file with its md5; a bitwise
%              self-check against the decoder's own lagged design runs first.
%              CP_EXPORT_ONLY=1 stops after the export (no decoding in MATLAB);
%              CP_EXPORT_REST_DIR writes the rest-window export that the
%              rest-input control reads (CP_EXPORT_REST_ONLY=1 skips decoding)
%   readout    in Python (pyport/run_session.py, configs/tenfold_pca30_lpf6.json):
%              the leading 30 basis components with a fold-scoped scaler, ridge
%              regression from lagged features (lag_ms -2000:50:0, lambda_grid
%              [1e-3 1e-2 0.1 1 10 100] chosen by inner 10-fold CV), a causal
%              second-order Butterworth low-pass at 6 Hz on that first-stage output
%              (stage1_lpf_hz), a causal FIR on the low-passed predictions fitted on
%              the low-passed signal (post_fir_lag_ms -2000:50:0, lambda chosen by
%              inner CV), outer random 10-fold cross-validation
%   band table estimated per fold from the training trials only (README, Step 09):
%              each session is exported once with its whole-session table and once
%              with every per-fold table that differs from it (bp_band_csv, one row of
%              exports.csv each); every fold is decoded from the export of its table
%
% With CP_EXPORT_DIR unset the function also decodes in MATLAB (outer_k-fold CV,
% the same readout without the first-stage low-pass); the Python readout with
% stage1_lpf_hz 0 reproduces that path fold by fold (pyport/gate.py, per-trial r
% within 1e-6).
%
% Any other value of feature_mode, bp_band_feature_mode, phase_target_mode,
% phase_recon or post_mode is an alternative that no published number rests on.
% input_control_mode and cross_condition select the control conditions that are
% reported next to the main result (rest-window substitution, within-trial
% circular shift, zero input, shuffled test labels, cross-task test).

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'shared'));
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'readout'));

trial_dump_stage1_models_default = false;
trial_dump_stage1_models_env = getenv('TRIAL_DUMP_STAGE1_MODELS');
if ~isempty(trial_dump_stage1_models_env)
    trial_dump_stage1_models_default = any(strcmpi(trial_dump_stage1_models_env, {'1','true','yes','on'}));
end

p = inputParser;
p.addParameter('out_dir', fullfile(fileparts(fileparts(mfilename('fullpath'))), 'result'));
p.addParameter('session_manifest', '');
p.addParameter('sessions', {});
p.addParameter('condition', 'MI'); % MI | EXEC
p.addParameter('lag_ms', -2000:100:0);
p.addParameter('bp_lo', 0.1);
p.addParameter('bp_hi', 3.0);
p.addParameter('bp_band_csv', '');
p.addParameter('bp_band_harmonic_k', 0);
p.addParameter('bp_band_harmonic_k_list', []);
p.addParameter('bp_band_feature_mode', 'real');
p.addParameter('bp_channel_labels', {});
p.addParameter('phase_lo', 0.3);
p.addParameter('phase_hi', 2.0);
p.addParameter('feature_mode', 'delta');
p.addParameter('eeg_baseline_mode', 'none'); % none | mean | zscore
p.addParameter('central_labels', {'Cz','C3','C4','FC1','FC2','CP1','CP2','C1','C2','FC3','FC4','CP3','CP4'});
p.addParameter('shared_pca_file', '');
p.addParameter('shared_pca_components', 20);
p.addParameter('shared_pca_scaler_scope', 'fold'); % fold | session
p.addParameter('harmonic_ssd_append_filters', 3);
p.addParameter('pca_score_gate', 'none'); % none | t2_mad | t2_quantile
p.addParameter('pca_score_gate_agg', 'p95'); % mean | median | p95 | max
p.addParameter('pca_score_gate_mad_k', 3.0);
p.addParameter('pca_score_gate_quantile', 0.95);
p.addParameter('pca_score_gate_max_drop_frac', 0.25);
p.addParameter('pca_score_gate_min_trials', 5);
p.addParameter('pca_score_gate_shrinkage', 0.10);
p.addParameter('pca_score_gate_apply_test', true);
p.addParameter('pca_score_gate_scope', 'fold'); % fold | precv
p.addParameter('smooth', 'lp');
p.addParameter('trim_start_sec', 0.0);
p.addParameter('trim_end_sec', 0.0);
p.addParameter('start_window_sec', 0.2);
p.addParameter('n_bins', 36);
p.addParameter('phase_recon', 'template');
p.addParameter('phase_target_mode', 'phase_pj');
p.addParameter('harmonic_k', 3);
p.addParameter('phase_delay_gain', 1.0);
p.addParameter('phase_select_margin', 0.0);
p.addParameter('phase_jitter_ms', 0);
p.addParameter('inner_k', 3);
p.addParameter('lambda_grid', [1e-3 1e-2 0.1 1 10 100]);
p.addParameter('lambda_mode', 'scalar');
p.addParameter('stage1_lambda_mode', 'innercv'); % innercv | oof
p.addParameter('alpha_grid', -0.5:0.05:0.8);
p.addParameter('aux_condition', 'none'); % none | EXEC
p.addParameter('aux_mode', 'phase_decoder_only'); % phase_decoder_only
p.addParameter('aux_inner_mode', 'off'); % off | on
p.addParameter('post_mode', 'none');
p.addParameter('post_components', 6);
p.addParameter('post_basis_components', 8);
p.addParameter('post_lambda', 0.01);
p.addParameter('post_lambda_grid', [0.001 0.003 0.01 0.03 0.1 0.3 1]);
p.addParameter('post_inner_k', 3);
p.addParameter('post_guard_margin', 0.0);
p.addParameter('post_fir_lag_ms', -400:40:0);
p.addParameter('post_fir_lag_unit', 'ms'); % ms | samples
p.addParameter('post_fir_causal_only', true);
p.addParameter('rpt_pred_smooth_sec', 0.0);
p.addParameter('valid_from_sec', 0);
p.addParameter('min_trial_sec', 0);
p.addParameter('post_train_from_sec', 0);
p.addParameter('input_control_mode', 'full'); % full | resting_random_window_within_trial | resting_random_window_test_only | within_trial_circular_shift | zero_eeg | test_label_shuffle
% resting_random_window_test_only: the same substitution as
% resting_random_window_within_trial (one contiguous rest window per trial, covering the
% whole trial, goniometer target untouched) but applied to the TEST fold only. The fold
% transform and the Stage-1 and Stage-2 fits still see the real imagined-walking training
% trials, so the model is identical to the one of the uncontrolled run and only the input
% at prediction time changes. Requires alpha_grid == 0.
p.addParameter('cross_condition', '');
p.addParameter('input_control_seed', 20260514);
p.addParameter('cv_mode', 'random'); % random | block
p.addParameter('outer_k', 5); % outer trial-CV fold count
p.addParameter('outer_folds', []); % [] = all folds; 
p.addParameter('gate_csv', '');
p.addParameter('gate_metric', 'mu_beta_erd_pct');
p.addParameter('gate_threshold', Inf);
p.addParameter('gate_keep_frac', 1.0);
p.addParameter('min_gate_trials', 5);
p.addParameter('result_tag', 'decoder');
p.addParameter('diag_dir', '');
p.addParameter('diag_sessions', {});
p.addParameter('diag_max_trials', 2);
p.addParameter('diag_plot_pca', true);
p.addParameter('diag_low_trial_threshold', NaN);
p.addParameter('diag_low_max_per_fold', 6);
p.addParameter('trial_dump_dir', '');
p.addParameter('trial_dump_stage1_models', trial_dump_stage1_models_default);
p.addParameter('stage1_readout_mode', 'plain'); % plain | group_shrink_ridge (empirical-Bayes partial pooling of the readout)
p.addParameter('group_prior_file', ''); % printf %s -> session; *_group_prior_readout.mat with Wgrp_raw [n_feat x n_lag x J]
p.addParameter('group_shrink_grid', [0 0.1 0.2 0.35 0.5 0.7 1.0]); % convex James-Stein blend weights (0 = baseline)
p.addParameter('group_shrink_force', -1); % >=0 forces this borrow weight (diagnostic), bypassing inner-CV selection
p.addParameter('max_threads', 4);
p.addParameter('verbose_progress', false);
p.parse(varargin{:});

shared_pca_scaler_scope = lower(char(p.Results.shared_pca_scaler_scope));
assert(any(strcmp(shared_pca_scaler_scope, {'fold', 'session'})), ...
    'shared_pca_scaler_scope must be fold or session');

stage1_readout_mode = lower(char(p.Results.stage1_readout_mode));
assert(any(strcmp(stage1_readout_mode, {'plain', 'group_shrink_ridge'})), ...
    'stage1_readout_mode must be plain or group_shrink_ridge');
use_group_shrink = strcmp(stage1_readout_mode, 'group_shrink_ridge');
if use_group_shrink
    assert(~isempty(char(p.Results.group_prior_file)), ...
        'stage1_readout_mode=group_shrink_ridge requires group_prior_file');
end

try
    maxNumCompThreads(p.Results.max_threads);
catch
end

out_dir = char(p.Results.out_dir);
if isempty(p.Results.session_manifest)
    manifest_path = fullfile(out_dir, 'session_manifest.csv');
else
    manifest_path = char(p.Results.session_manifest);
end
T = readtable(manifest_path, 'TextType', 'string', 'Delimiter', ',', ...
    'ReadVariableNames', true);
sessions = cellstr(T.session);
if ~isempty(p.Results.sessions)
    requested_sessions = cellstr(p.Results.sessions);
    if any(contains(string(requested_sessions), "+"))
        sessions = requested_sessions;
    else
        sessions = intersect(sessions, requested_sessions, 'stable');
    end
end
gate_tbl = table();
if ~isempty(p.Results.gate_csv)
    gate_tbl = readtable(p.Results.gate_csv, 'TextType', 'string');
    assert(ismember(p.Results.gate_metric, gate_tbl.Properties.VariableNames), ...
        'gate_metric %s not found in %s', p.Results.gate_metric, p.Results.gate_csv);
end
bp_band_tbl = table();
if ~isempty(p.Results.bp_band_csv)
    bp_band_tbl = readtable(p.Results.bp_band_csv, 'TextType', 'string');
    required_bp_cols = {'session', 'harmonic_k', 'band_lo', 'band_hi'};
    missing_bp_cols = setdiff(required_bp_cols, bp_band_tbl.Properties.VariableNames);
    assert(isempty(missing_bp_cols), 'bp_band_csv missing columns: %s', strjoin(missing_bp_cols, ','));
end
bp_channel_labels = parse_channel_labels_local(p.Results.bp_channel_labels);

W_X = {'LHipW X','LKneW X','LAnkW X','RHipW X','RKneW X','RAnkW X'};
S_X = {'LHipS X','LKneS X','LAnkS X','RHipS X','RKneS X','RAnkS X'};
condition = upper(char(p.Results.condition));
if strcmp(condition, 'MI')
    target_labels = W_X;
elseif strcmp(condition, 'EXEC')
    target_labels = S_X;
else
    error('condition must be MI or EXEC');
end
tag = char(p.Results.result_tag);
if ~isempty(tag), tag = ['_' tag]; end
out_csv = fullfile(out_dir, sprintf('performance_imagined_walking%s.csv', tag));
fid = fopen(out_csv, 'w');
fprintf(fid, ['session,fold,joint,r_angle,r_phase,r_fused,rpt_angle,rpt_phase,' ...
              'rpt_fused,r_calib,rpt_calib,alpha,n_trials,n_gate_kept,n_gate_dropped,' ...
              'lambda_angle,lambda_phase,post_mode,post_components,post_basis_components,post_lambda,' ...
              'post_fir_lag_unit,post_fir_causal_only,pca_gate_train_kept,pca_gate_train_dropped,' ...
              'pca_gate_test_kept,pca_gate_test_dropped,pca_gate_threshold,pca_gate_train_center,' ...
              'pca_gate_train_scale,pca_gate_scope,pca_gate_pre_kept,pca_gate_pre_dropped,' ...
              'pca_gate_pre_total\n']);

opt_delta = struct('bp_lo', p.Results.bp_lo, 'bp_hi', p.Results.bp_hi, ...
    'lag_ms', p.Results.lag_ms, ...
    'trim_start_sec', p.Results.trim_start_sec, ...
    'trim_end_sec', p.Results.trim_end_sec, ...
    'start_window_sec', p.Results.start_window_sec, ...
    'feature_mode', p.Results.feature_mode, ...
    'bp_band_feature_mode', p.Results.bp_band_feature_mode, ...
    'bp_channel_labels', {bp_channel_labels}, ...
    'eeg_baseline_mode', p.Results.eeg_baseline_mode, ...
    'central_labels', {p.Results.central_labels}, ...
    'shared_pca_file', p.Results.shared_pca_file, ...
    'shared_pca_components', p.Results.shared_pca_components, ...
    'input_control_mode', p.Results.input_control_mode, ...
    'input_control_seed', p.Results.input_control_seed, ...
    'rest_events', []);
opt_phase = struct('bp_lo', p.Results.bp_lo, 'bp_hi', p.Results.bp_hi, ...
    'phase_lo', p.Results.phase_lo, 'phase_hi', p.Results.phase_hi, ...
    'lag_ms', p.Results.lag_ms, 'trim_sec', 0, ...
    'trim_start_sec', p.Results.trim_start_sec, ...
    'trim_end_sec', p.Results.trim_end_sec, ...
    'target_mode', 'phase_pj');

cross_task = strcmpi(p.Results.input_control_mode, 'cross_task_test');
% cross_task_test: the training folds are this condition's trials, with the same CV split,
% fold transform and Stage-1 and Stage-2 fits as an ordinary run; test fold f is made of the
% cross_condition trials assigned to fold f by make_cv_split(label, cross_condition,
% outer_k). Cross trials are appended after the real trials and never enter tr_idx.
% Requires alpha_grid == 0.
if cross_task
    assert(numel(p.Results.alpha_grid) == 1 && p.Results.alpha_grid(1) == 0, 'cross_task_test requires ALPHA_GRID=0');
    assert(any(strcmpi(p.Results.cross_condition, {'MI', 'EXEC'})), 'cross_task_test requires cross_condition MI|EXEC');
end
rest_test_only = strcmpi(p.Results.input_control_mode, 'resting_random_window_test_only');
if rest_test_only && ~(numel(p.Results.alpha_grid) == 1 && p.Results.alpha_grid(1) == 0)
    % The substitution is applied to the delta/angle features only. With a live phase branch the
    % fused score would mix rest-driven and IW-driven inputs and the number would be meaningless.
    error('resting_random_window_test_only requires ALPHA_GRID=0 (angle-only fused score)');
end

t_start = tic;
for si = 1:numel(sessions)
    label = sessions{si};
    session_start = tic;
    fprintf('[%d/%d] %s ', si, numel(sessions), label);
    group_prior_raw = [];
    if use_group_shrink
        gp_file = char(p.Results.group_prior_file);
        if contains(gp_file, '%s'); gp_file = sprintf(gp_file, label); end
        GP = load(gp_file);
        group_prior_raw = GP.Wgrp_raw; % [n_feat x n_lag x J] in common raw band-feature x lag space
    end
    try
        D = build_session_or_pool_dataset_local(label, condition, target_labels, S_X, ...
            opt_delta, opt_phase, p.Results, gate_tbl, bp_band_tbl);
        S = D.S;
        ev = D.ev;
        gate_mask = D.gate_mask;
        n_gate_kept = D.n_gate_kept;
        n_gate_dropped = D.n_gate_dropped;
        X_delta = D.X_delta;
        if isfield(D, 'X_delta_rest'), X_delta_rest = D.X_delta_rest; else, X_delta_rest = {}; end
        n_real_trials = numel(D.X_delta); cross_fold = [];
        if cross_task, n_real_trials = D.n_real_trials; cross_fold = D.cross_fold; end
        Y_delta = D.Y_delta;
        Yd_delta = D.Yd_delta;
        Y_phase = D.Y_phase;
        X_phase = D.X_phase;
        X_aux_phase = D.X_aux_phase;
        Y_aux_phase = D.Y_aux_phase;
        assert_xy_sample_counts_local(X_delta, Y_delta, 1:numel(X_delta), label, 0, 'raw-build');
        if isfield(S, 'pool_parts') && ~isempty(S.pool_parts)
            fa = make_cv_split_indices_local(label, condition, p.Results.outer_k, ...
                1:numel(X_delta), p.Results.cv_mode);
        else
            fa = make_cv_split(label, condition, p.Results.outer_k);
            fa = apply_cv_mode_local(fa, p.Results.cv_mode, p.Results.outer_k);
        end
    catch ME
        fprintf('LOAD/BUILD FAIL: %s\n', ME.message);
        continue;
    end
    if cp_export_h2_local(label, target_labels, ev, fa, X_delta, Y_delta, Y_phase, p.Results), continue; end
    if cp_export_h3_skip_local(label), continue; end

    pca_gate_scope = lower(char(p.Results.pca_score_gate_scope));
    pca_pre_gate_info = empty_pca_score_gate_info(0, 0);
    pca_pre_total = 0;
    if any(strcmp(pca_gate_scope, {'precv', 'pre_cv', 'pre-cv'}))
        if any(strcmpi(char(p.Results.feature_mode), {'delta_fold_diagwhiten_csp_phase', 'delta_fold_diagwhiten_harmonic_csp_phase'})) && ...
                ~strcmpi(char(p.Results.pca_score_gate), 'none')
            error('preCV PCA gate is not allowed with target-supervised CSP feature_mode');
        end
        try
            all_trial_idx = 1:numel(ev);
            valid_all = valid_trial_mask(X_delta, Y_delta, Y_phase, all_trial_idx);
            pre_base_idx = all_trial_idx(valid_all & gate_mask);
            pca_pre_total = numel(pre_base_idx);
            [pca_pre_keep_mask, pca_pre_gate_info] = make_pre_cv_pca_gate_mask_local( ...
                X_delta, X_phase, Y_delta, S, pre_base_idx, p.Results.feature_mode, ...
                p.Results.shared_pca_file, p.Results.shared_pca_components, ...
                p.Results.phase_lo, p.Results.phase_hi, numel(p.Results.lag_ms), ...
                p.Results.pca_score_gate, p.Results.pca_score_gate_agg, ...
                p.Results.pca_score_gate_mad_k, p.Results.pca_score_gate_quantile, ...
                p.Results.pca_score_gate_max_drop_frac, p.Results.pca_score_gate_min_trials, ...
                p.Results.pca_score_gate_shrinkage, shared_pca_scaler_scope);
            pre_clean_idx = find(pca_pre_keep_mask & gate_mask);
            if numel(pre_clean_idx) < p.Results.min_gate_trials
                fprintf('SKIP (preCV PCA gate kept %d/%d)\n', numel(pre_clean_idx), pca_pre_total);
                continue;
            end
            fa = make_cv_split_indices_local(label, condition, p.Results.outer_k, pre_clean_idx, p.Results.cv_mode);
            fprintf('preCV PCA gate kept %d/%d trials before split\n', numel(pre_clean_idx), pca_pre_total);
            pca_gate_scope = 'precv';
        catch ME
            fprintf('PRECV PCA GATE FAIL: %s\n', ME.message);
            continue;
        end
    elseif ~strcmp(pca_gate_scope, 'fold')
        error('pca_score_gate_scope must be fold or precv');
    end

    K = max(fa.fold);
    folds_to_run = 1:K;
    if ~isempty(p.Results.outer_folds)
        folds_to_run = unique(p.Results.outer_folds(:)');
        folds_to_run = folds_to_run(folds_to_run >= 1 & folds_to_run <= K);
        if isempty(folds_to_run)
            error('outer_folds has no valid folds for K=%d', K);
        end
    end
    for fold = folds_to_run
        if isfield(fa, 'trial_idx')
            fa_trial_idx = fa.trial_idx(:)';
            fa_fold = fa.fold(:)';
            tr_idx = fa_trial_idx(fa_fold ~= fold);
            te_idx = fa_trial_idx(fa_fold == fold);
        else
            tr_idx = find(fa.fold ~= fold);
            te_idx = find(fa.fold == fold);
        end
        if cross_task
            % test only on the cross-condition trials make_cv_split assigned to this fold.
            tr_idx = tr_idx(tr_idx <= n_real_trials);
            te_idx = n_real_trials + find(cross_fold(:)' == fold);
        end
        keep_tr = valid_trial_mask(X_delta, Y_delta, Y_phase, tr_idx);
        if rest_test_only
            % Test trials must be valid under their REST substitute (a trial with no usable rest
            % window comes back with 0 columns and has to be dropped before it reaches the fit).
            keep_te = valid_trial_mask(X_delta_rest, Y_delta, Y_phase, te_idx);
        else
            keep_te = valid_trial_mask(X_delta, Y_delta, Y_phase, te_idx);
        end
        tr_idx = tr_idx(keep_tr);
        te_idx = te_idx(keep_te);
        tr_idx = tr_idx(gate_mask(tr_idx));
        te_idx = te_idx(gate_mask(te_idx));
        if isempty(tr_idx) || isempty(te_idx), continue; end

        X_delta_eff = X_delta;
        if rest_test_only
            % below fits the fold transform (shared PCA + scaler) on genuine imagined-walking data
            % and Stage-1/Stage-2 are trained exactly as in the real run; only the prediction-time
            % input differs. Downstream code reads X_delta_eff wherever the raw cells are needed.
            X_delta_eff(te_idx) = X_delta_rest(te_idx);
        end

        try
            fold_tic = tic;
            if logical(p.Results.verbose_progress)
                fprintf('\n[%s] %s outer fold %d/%d start | train=%d test=%d\n', ...
                    datestr(now, 'HH:MM:SS'), label, fold, K, numel(tr_idx), numel(te_idx));
                drawnow;
            end
            [X_delta_fit, X_phase_fit, feature_info] = prepare_fold_feature_cells_local(X_delta_eff, X_phase, ...
                Y_delta, S, tr_idx, p.Results.feature_mode, p.Results.shared_pca_file, ...
                p.Results.shared_pca_components, p.Results.phase_lo, p.Results.phase_hi, ...
                numel(p.Results.lag_ms), shared_pca_scaler_scope, p.Results.harmonic_ssd_append_filters);
            if strcmp(pca_gate_scope, 'fold')
                [tr_idx, te_idx, pca_gate_info] = apply_pca_score_gate_local(X_delta_fit, tr_idx, te_idx, ...
                    p.Results.pca_score_gate, round(p.Results.shared_pca_components), ...
                    p.Results.pca_score_gate_agg, p.Results.pca_score_gate_mad_k, ...
                    p.Results.pca_score_gate_quantile, p.Results.pca_score_gate_max_drop_frac, ...
                    p.Results.pca_score_gate_min_trials, p.Results.pca_score_gate_shrinkage, ...
                    logical(p.Results.pca_score_gate_apply_test));
            else
                pca_gate_info = empty_pca_score_gate_info(numel(tr_idx), numel(te_idx));
                pca_gate_info.threshold = pca_pre_gate_info.threshold;
                pca_gate_info.train_center = pca_pre_gate_info.train_center;
                pca_gate_info.train_scale = pca_pre_gate_info.train_scale;
            end
            if isempty(tr_idx) || isempty(te_idx)
                fprintf('fold %d SKIP after PCA score gate (train=%d test=%d)\n', ...
                    fold, pca_gate_info.train_kept, pca_gate_info.test_kept);
                continue;
            end
            assert_prepared_sample_counts_preserved_local(X_delta_fit, X_delta_eff, tr_idx, label, fold, 'stage1-train');
            assert_prepared_sample_counts_preserved_local(X_delta_fit, X_delta_eff, te_idx, label, fold, 'stage1-test');
            assert_xy_sample_counts_local(X_delta_fit, Y_delta, tr_idx, label, fold, 'stage1-train');
            assert_xy_sample_counts_local(X_delta_fit, Y_delta, te_idx, label, fold, 'stage1-test');
            if logical(p.Results.verbose_progress)
                fprintf('[%s] %s outer fold %d prepared | train=%d test=%d elapsed=%.1fs\n', ...
                    datestr(now, 'HH:MM:SS'), label, fold, numel(tr_idx), numel(te_idx), toc(fold_tic));
                drawnow;
            end
            A_oof_precomputed = {};
            fixed_alpha0 = numel(p.Results.alpha_grid) == 1 && p.Results.alpha_grid(1) == 0;
            stage1_info = struct();
            shrink_lambda = NaN;
            if use_group_shrink
                Wgrp_fold = project_group_prior_to_fold_local(group_prior_raw, feature_info, numel(p.Results.lag_ms));
                [A_te, lambda_a, angle_model, angle_target_mu, angle_target_sd, shrink_lambda] = fit_predict_angle_group_shrink( ...
                    X_delta_fit, Y_delta, tr_idx, te_idx, p.Results.lambda_grid, ...
                    p.Results.lambda_mode, p.Results.inner_k, Wgrp_fold, p.Results.group_shrink_grid, p.Results.group_shrink_force);
            elseif strcmpi(char(p.Results.stage1_lambda_mode), 'oof')
                [A_te, lambda_a, A_oof_precomputed, angle_model, angle_target_mu, angle_target_sd] = fit_predict_angle_oof_lambda( ...
                    X_delta_fit, Y_delta, tr_idx, te_idx, p.Results.lambda_grid, ...
                    p.Results.inner_k, fold, p.Results.lambda_mode, p.Results.verbose_progress);
            else
                [A_te, lambda_a, angle_model, angle_target_mu, angle_target_sd] = fit_predict_angle(X_delta_fit, Y_delta, tr_idx, te_idx, ...
                    p.Results.lambda_grid, p.Results.lambda_mode, p.Results.inner_k);
            end
            if logical(p.Results.trial_dump_stage1_models)
                stage1_info.outer_fold = fold;
                stage1_info.train_trial_indices = tr_idx;
                stage1_info.test_trial_indices = te_idx;
                stage1_info.target_labels = {target_labels};
                stage1_info.angle_model = angle_model;
                stage1_info.angle_target_mu = angle_target_mu;
                stage1_info.angle_target_sd = angle_target_sd;
                stage1_info.feature_mode = char(p.Results.feature_mode);
                stage1_info.shared_pca_file = char(p.Results.shared_pca_file);
                stage1_info.shared_pca_components = round(p.Results.shared_pca_components);
                stage1_info.shared_pca_scaler_scope = shared_pca_scaler_scope;
                stage1_info.lag_ms = p.Results.lag_ms;
                stage1_info.lambda_angle = lambda_a;
                stage1_info.lambda_grid = p.Results.lambda_grid;
                stage1_info.lambda_mode = char(p.Results.lambda_mode);
                stage1_info.stage1_lambda_mode = char(p.Results.stage1_lambda_mode);
                stage1_info.inner_k = p.Results.inner_k;
                stage1_info.fixed_alpha0 = fixed_alpha0;
                stage1_info.stage1_readout_mode = stage1_readout_mode;
                stage1_info.shrink_lambda = shrink_lambda;
                stage1_info.feature_info = feature_info;
            end
            ydy_parallel_mode = strcmpi(char(p.Results.post_mode), 'prediction_ydy_parallel_fusion');
            D_te = {};
            lambda_d = NaN;
            if ydy_parallel_mode
                if ~fixed_alpha0
                    error('prediction_ydy_parallel_fusion currently requires ALPHA_GRID=0');
                end
                [D_te, lambda_d] = fit_predict_angle(X_delta_fit, Yd_delta, tr_idx, te_idx, ...
                    p.Results.lambda_grid, p.Results.lambda_mode, p.Results.inner_k);
            end
            if logical(p.Results.verbose_progress)
                fprintf('[%s] %s outer fold %d stage1 test prediction done | elapsed=%.1fs\n', ...
                    datestr(now, 'HH:MM:SS'), label, fold, toc(fold_tic));
                drawnow;
            end
            S_te = {};
            if fixed_alpha0
                P_te = A_te;
                lambda_p = NaN;
            else
                [P_te, lambda_p, S_te] = fit_predict_phase_recon(X_phase_fit, Y_phase, Y_delta, ...
                    tr_idx, te_idx, p.Results.lambda_grid, p.Results.smooth, S.eeg.fs, ...
                    p.Results.n_bins, p.Results.phase_recon, p.Results.harmonic_k, ...
                    p.Results.lambda_mode, p.Results.inner_k, p.Results.phase_delay_gain, ...
                    p.Results.phase_lo, p.Results.phase_hi, p.Results.phase_select_margin, ...
                    p.Results.phase_jitter_ms, X_aux_phase, Y_aux_phase);
            end
            if strcmpi(char(p.Results.aux_inner_mode), 'on')
                X_inner_aux_phase = X_aux_phase;
                Y_inner_aux_phase = Y_aux_phase;
            elseif strcmpi(char(p.Results.aux_inner_mode), 'off')
                X_inner_aux_phase = {};
                Y_inner_aux_phase = {};
            else
                error('aux_inner_mode must be off or on');
            end
            A_oof = {};
            P_oof = {};
            F_oof = {};
            S_oof = {};
            D_oof = {};
            if numel(p.Results.alpha_grid) == 1 && strcmpi(char(p.Results.post_mode), 'none')
                alpha = repmat(p.Results.alpha_grid(1), 1, numel(target_labels));
            elseif fixed_alpha0
                alpha = repmat(0, 1, numel(target_labels));
                if ~isempty(A_oof_precomputed)
                    A_oof = A_oof_precomputed;
                else
                    A_oof = angle_oof_predictions(X_delta_fit, Y_delta, tr_idx, ...
                        p.Results.lambda_grid, p.Results.inner_k, fold, p.Results.lambda_mode, ...
                        p.Results.verbose_progress);
                end
                if ydy_parallel_mode
                    D_oof = angle_oof_predictions(X_delta_fit, Yd_delta, tr_idx, ...
                        p.Results.lambda_grid, p.Results.inner_k, fold, p.Results.lambda_mode, ...
                        p.Results.verbose_progress);
                end
                P_oof = A_oof;
                F_oof = A_oof;
            else
                [A_oof, P_oof, S_oof] = inner_oof_predictions(X_delta_fit, X_phase_fit, Y_delta, Y_phase, ...
                    tr_idx, p.Results.lambda_grid, p.Results.smooth, S.eeg.fs, ...
                    p.Results.n_bins, p.Results.phase_recon, p.Results.harmonic_k, ...
                    p.Results.inner_k, fold, p.Results.lambda_mode, p.Results.phase_delay_gain, ...
                    p.Results.phase_lo, p.Results.phase_hi, p.Results.phase_select_margin, ...
                    p.Results.phase_jitter_ms, X_inner_aux_phase, Y_inner_aux_phase, ...
                    p.Results.verbose_progress);
                alpha = choose_alpha(Y_delta(tr_idx), A_oof, P_oof, p.Results.alpha_grid, numel(target_labels));
                F_oof = fuse_cells(A_oof, P_oof, alpha);
            end
            if logical(p.Results.verbose_progress)
                fprintf('[%s] %s outer fold %d train OOF done | elapsed=%.1fs\n', ...
                    datestr(now, 'HH:MM:SS'), label, fold, toc(fold_tic));
                drawnow;
            end
            F_te = fuse_cells(A_te, P_te, alpha);
            if ydy_parallel_mode
                [C_te, post_info] = fit_predict_ydy_parallel_fusion( ...
                    F_oof, Y_delta(tr_idx), F_te, D_oof, Yd_delta(tr_idx), D_te, ...
                    p.Results.post_lambda, p.Results.post_fir_lag_ms, S.eeg.fs, ...
                    p.Results.post_fir_lag_unit, p.Results.post_fir_causal_only);
                post_info.stage1_lambda_y = lambda_a;
                post_info.stage1_lambda_dy = lambda_d;
            else
                % With post_train_from_sec > 0 the Stage-2 targets before that time
                % (same time axis as the trial dumps' time_sec) are NaN, so fir_lag_design drops those rows.
                Y_post_tr = Y_delta(tr_idx);
                if p.Results.post_train_from_sec > 0
                    t_post = make_diag_time_cells(Y_post_tr, p.Results.lag_ms, S.eeg.fs);
                    for kk = 1:numel(Y_post_tr)
                        Y_post_tr{kk}(t_post{kk} < p.Results.post_train_from_sec, :) = NaN;
                    end
                end
                [C_te, post_info] = post_calibrate_predictions(p.Results.post_mode, A_oof, P_oof, F_oof, S_oof, ...
                    Y_post_tr, A_te, P_te, F_te, S_te, p.Results.post_components, ...
                    p.Results.post_basis_components, p.Results.post_lambda, p.Results.post_lambda_grid, p.Results.post_guard_margin, ...
                    p.Results.post_fir_lag_ms, S.eeg.fs, p.Results.post_fir_lag_unit, ...
                    p.Results.post_fir_causal_only, p.Results.post_inner_k);
            end
            % explicit valid-from / min-trial-length gating: this NaNs the
            % first valid_from_sec of every kept trial's predictions, so safe_corr drops them,
            % and it drops trials shorter than min_trial_sec. With the default 0 / 0 neither
            % gate is active.
            vfs = p.Results.valid_from_sec;
            mts = p.Results.min_trial_sec;
            if vfs > 0 || mts > 0
                fs_gate = S.goni.fs;
                if mts > 0
                    keep = true(1, numel(te_idx));
                    min_n = round(mts * fs_gate);
                    for ti = 1:numel(te_idx)
                        if size(Y_delta{te_idx(ti)}, 1) < min_n
                            keep(ti) = false;
                        end
                    end
                    te_idx = te_idx(keep);
                    if ~isempty(A_te), A_te = A_te(keep); end
                    if ~isempty(P_te), P_te = P_te(keep); end
                    if ~isempty(F_te), F_te = F_te(keep); end
                    if ~isempty(C_te), C_te = C_te(keep); end
                end
                if vfs > 0
                    warm_n = round(vfs * fs_gate);
                    for ti = 1:numel(te_idx)
                        if ~isempty(A_te) && ~isempty(A_te{ti}), A_te{ti}(1:min(warm_n, size(A_te{ti}, 1)), :) = NaN; end
                        if ~isempty(P_te) && ~isempty(P_te{ti}), P_te{ti}(1:min(warm_n, size(P_te{ti}, 1)), :) = NaN; end
                        if ~isempty(F_te) && ~isempty(F_te{ti}), F_te{ti}(1:min(warm_n, size(F_te{ti}, 1)), :) = NaN; end
                        if ~isempty(C_te) && ~isempty(C_te{ti}), C_te{ti}(1:min(warm_n, size(C_te{ti}, 1)), :) = NaN; end
                    end
                end
            end
            s2dir = getenv('STAGE2_DUMP_DIR');
            if ~isempty(s2dir)
                try
                    if ~exist(s2dir, 'dir'); mkdir(s2dir); end
                    s2 = struct();
                    s2.F_oof = F_oof; s2.F_te = F_te;
                    s2.Y_train = Y_delta(tr_idx); s2.Y_te = Y_delta(te_idx);
                    s2.eeg_fs = S.eeg.fs; s2.goni_fs = S.goni.fs;
                    s2.post_fir_lags = post_fir_lags_to_samples(p.Results.post_fir_lag_ms, S.eeg.fs, p.Results.post_fir_lag_unit, p.Results.post_fir_causal_only);
                    s2.rpt_pred_smooth_sec = p.Results.rpt_pred_smooth_sec;
                    s2.label = char(label); s2.fold = fold;
                    save(fullfile(s2dir, sprintf('%s_fold%d_stage2.mat', char(label), fold)), '-struct', 's2', '-v7');
                catch s2err
                    fprintf('STAGE2_DUMP fail %s fold %d: %s\n', char(label), fold, s2err.message);
                end
            end
            post_info.outer_fold = fold;
            post_info.train_trial_indices = tr_idx;
            post_info.test_trial_indices = te_idx;
            post_info.target_labels = {target_labels};
            post_info.alpha = alpha;
            post_info.lambda_angle = lambda_a;
            post_info.lambda_phase = lambda_p;
            post_info.lambda_grid = p.Results.lambda_grid;
            post_info.stage1_lambda_mode = char(p.Results.stage1_lambda_mode);
            post_info.post_mode = char(p.Results.post_mode);
            if logical(p.Results.verbose_progress)
                fprintf('[%s] %s outer fold %d dynamic FIR done | elapsed=%.1fs\n', ...
                    datestr(now, 'HH:MM:SS'), label, fold, toc(fold_tic));
                drawnow;
            end
            Y_eval = Y_delta(te_idx);
            if strcmpi(p.Results.input_control_mode, 'test_label_shuffle')
                Y_eval = shuffle_cell_order_local(Y_eval, p.Results.input_control_seed + 10007 * fold);
            end
            diag_time_sec = make_diag_time_cells(Y_eval, p.Results.lag_ms, S.eeg.fs);
            rows = eval_three(Y_eval, A_te, P_te, F_te, target_labels, alpha, ...
                S.goni.fs, p.Results.rpt_pred_smooth_sec);
            maybe_write_diagnostics(p.Results.diag_dir, p.Results.diag_sessions, ...
                p.Results.diag_max_trials, label, fold, S, Y_eval, A_te, P_te, F_te, ...
                C_te, diag_time_sec, rows, target_labels, p.Results.feature_mode, p.Results.shared_pca_file, ...
                p.Results.shared_pca_components, p.Results.diag_plot_pca, ...
                p.Results.diag_low_trial_threshold, p.Results.diag_low_max_per_fold);
            maybe_write_trial_dump(p.Results.trial_dump_dir, label, fold, te_idx, ...
                target_labels, Y_eval, A_te, P_te, F_te, C_te, post_info, diag_time_sec, ...
                S.goni.fs, p.Results.rpt_pred_smooth_sec, stage1_info);
        catch ME
            fprintf('fold %d FAIL: %s\n', fold, ME.message);
            continue;
        end

        for j = 1:numel(rows)
            post_lambda_report = p.Results.post_lambda;
            if isfield(post_info, 'selected_lambda') && isfinite(post_info.selected_lambda)
                post_lambda_report = post_info.selected_lambda;
            end
            if isempty(C_te)
                r_calib = NaN;
                rpt_calib = NaN;
            else
                [r_calib, rpt_calib] = eval_one(Y_eval, C_te, j, S.goni.fs, ...
                    p.Results.rpt_pred_smooth_sec);
            end
            fprintf(fid, '%s,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%d,%d,%d,%g,%g,%s,%d,%d,%g,%s,%d,%d,%d,%d,%d,%.6g,%.6g,%.6g,%s,%d,%d,%d\n', ...
                label, fold, rows(j).joint, rows(j).r_angle, rows(j).r_phase, ...
                rows(j).r_fused, rows(j).rpt_angle, rows(j).rpt_phase, ...
                rows(j).rpt_fused, r_calib, rpt_calib, rows(j).alpha, numel(te_idx), ...
                n_gate_kept, n_gate_dropped, lambda_a, lambda_p, char(p.Results.post_mode), ...
                round(p.Results.post_components), round(p.Results.post_basis_components), post_lambda_report, ...
                char(p.Results.post_fir_lag_unit), logical(p.Results.post_fir_causal_only), ...
                pca_gate_info.train_kept, pca_gate_info.train_dropped, ...
                pca_gate_info.test_kept, pca_gate_info.test_dropped, ...
                pca_gate_info.threshold, pca_gate_info.train_center, pca_gate_info.train_scale, ...
                pca_gate_scope, pca_pre_gate_info.train_kept, ...
                pca_pre_gate_info.train_dropped, pca_pre_total);
        end
    end
    fprintf('done. session_elapsed=%.1f min total_elapsed=%.1f min\n', ...
        toc(session_start) / 60, toc(t_start) / 60);
end

function D = build_session_or_pool_dataset_local(label, condition, target_labels, S_X, opt_delta, opt_phase, params, gate_tbl, bp_band_tbl)
parts = split_session_pool_label_local(label);
if numel(parts) == 1
    D = build_single_session_dataset_local(parts{1}, condition, target_labels, S_X, ...
        opt_delta, opt_phase, params, gate_tbl, bp_band_tbl);
    return;
end

D_parts = cell(1, numel(parts));
X_delta = {};
X_delta_rest = {};
Y_delta = {};
Yd_delta = {};
Y_phase = {};
X_phase = {};
X_aux_phase = {};
Y_aux_phase = {};
gate_mask = [];
n_gate_kept = 0;
n_gate_dropped = 0;
pool_parts = struct('label', {}, 'S', {}, 'global_idx', {});
offset = 0;
for pi = 1:numel(parts)
    Dp = build_single_session_dataset_local(parts{pi}, condition, target_labels, S_X, ...
        opt_delta, opt_phase, params, gate_tbl, bp_band_tbl);
    D_parts{pi} = Dp;
    n_trials = numel(Dp.X_delta);
    global_idx = offset + (1:n_trials);
    pool_parts(pi).label = parts{pi};
    pool_parts(pi).S = Dp.S;
    pool_parts(pi).global_idx = global_idx;
    X_delta = [X_delta, Dp.X_delta(:)']; %#ok<AGROW>
    if isfield(Dp, 'X_delta_rest') && ~isempty(Dp.X_delta_rest)
        X_delta_rest = [X_delta_rest, Dp.X_delta_rest(:)']; %#ok<AGROW>
    end
    Y_delta = [Y_delta, Dp.Y_delta(:)']; %#ok<AGROW>
    Yd_delta = [Yd_delta, Dp.Yd_delta(:)']; %#ok<AGROW>
    Y_phase = [Y_phase, Dp.Y_phase(:)']; %#ok<AGROW>
    X_phase = [X_phase, Dp.X_phase(:)']; %#ok<AGROW>
    X_aux_phase = [X_aux_phase, Dp.X_aux_phase(:)']; %#ok<AGROW>
    Y_aux_phase = [Y_aux_phase, Dp.Y_aux_phase(:)']; %#ok<AGROW>
    gate_mask = [gate_mask, Dp.gate_mask(:)']; %#ok<AGROW>
    n_gate_kept = n_gate_kept + Dp.n_gate_kept;
    n_gate_dropped = n_gate_dropped + Dp.n_gate_dropped;
    offset = offset + n_trials;
end

D = D_parts{1};
D.label = char(label);
D.S.label = char(label);
D.S.pool_parts = pool_parts;
D.ev = 1:numel(X_delta);
D.gate_mask = gate_mask;
D.n_gate_kept = n_gate_kept;
D.n_gate_dropped = n_gate_dropped;
D.X_delta = X_delta;
D.X_delta_rest = X_delta_rest;
D.Y_delta = Y_delta;
D.Yd_delta = Yd_delta;
D.Y_phase = Y_phase;
D.X_phase = X_phase;
D.X_aux_phase = X_aux_phase;
D.Y_aux_phase = Y_aux_phase;
end

function D = build_single_session_dataset_local(label, condition, target_labels, S_X, opt_delta, opt_phase, params, gate_tbl, bp_band_tbl)
S = load_triple_mat(label);
ev = trial_filter(S.events, condition);
rest_ev = rest_trial_filter_local(S.events);
if numel(ev) < 5
    error('SKIP (%d trials)', numel(ev));
end
if any(strcmpi(params.input_control_mode, ...
        {'resting_random_window_within_trial', 'resting_random_window_test_only'})) && isempty(rest_ev)
    error('SKIP (no good rest trials)');
end
opt_delta_s = opt_delta;
opt_phase_s = opt_phase;
opt_delta_s.rest_events = rest_ev;
if ~isempty(bp_band_tbl)
    harmonic_list = double(params.bp_band_harmonic_k_list(:)');
    if isempty(harmonic_list) && params.bp_band_harmonic_k > 0
        harmonic_list = double(params.bp_band_harmonic_k);
    end
    if numel(harmonic_list) > 1
        opt_delta_s.bp_band_tbl = bp_band_tbl;
        opt_delta_s.bp_band_harmonic_k_list = harmonic_list;
        opt_delta_s.session_label = label;
        [bp_lo_s, bp_hi_s] = lookup_session_bp_band_local(bp_band_tbl, label, harmonic_list(1));
        opt_phase_s.bp_lo = bp_lo_s;
        opt_phase_s.bp_hi = bp_hi_s;
    else
        [bp_lo_s, bp_hi_s] = lookup_session_bp_band_local( ...
            bp_band_tbl, label, params.bp_band_harmonic_k);
        opt_delta_s.bp_lo = bp_lo_s;
        opt_delta_s.bp_hi = bp_hi_s;
        opt_phase_s.bp_lo = bp_lo_s;
        opt_phase_s.bp_hi = bp_hi_s;
    end
end
[gate_mask, n_gate_kept, n_gate_dropped] = make_gate_mask(label, numel(ev), ...
    gate_tbl, params.gate_metric, params.gate_threshold, ...
    params.gate_keep_frac, params.min_gate_trials);
if n_gate_kept < params.min_gate_trials
    error('SKIP (gate kept %d/%d)', n_gate_kept, numel(ev));
end
joint_idx = nan(1, numel(target_labels));
for j = 1:numel(target_labels)
    joint_idx(j) = find_goni_idx(S.goni.labels, target_labels{j});
end
[X_delta, Y_delta] = build_xy_delta0_local(S, ev, joint_idx, opt_delta_s);
X_delta_rest = {};
if strcmpi(params.input_control_mode, 'resting_random_window_test_only')
    % real imagined-walking features (build_xy_delta0_local does not recognise the test_only
    % string, so it takes the untouched 'full' path); the fold loop swaps in these cells for the
    % test trials only. Y is identical in both builds by construction.
    opt_delta_rest = opt_delta_s;
    opt_delta_rest.input_control_mode = 'resting_random_window_within_trial';
    X_delta_rest = build_xy_delta0_local(S, ev, joint_idx, opt_delta_rest);
end
n_real_trials = numel(X_delta); cross_fold = [];
if strcmpi(params.input_control_mode, 'cross_task_test')
    % session-calibrated pipeline/bands; goni targets = that condition's own joints) after the real ones.
    cross_condition = upper(char(params.cross_condition));
    ev_x = trial_filter(S.events, cross_condition);
    if numel(ev_x) < 5, error('SKIP (cross %s %d trials)', cross_condition, numel(ev_x)); end
    if strcmp(cross_condition, 'MI'), x_labels = regexprep(target_labels, 'S X$', 'W X'); else, x_labels = regexprep(target_labels, 'W X$', 'S X'); end
    joint_idx_x = nan(1, numel(x_labels));
    for j = 1:numel(x_labels), joint_idx_x(j) = find_goni_idx(S.goni.labels, x_labels{j}); end
    [X_x, Y_x] = build_xy_delta0_local(S, ev_x, joint_idx_x, opt_delta_s);
    fa_x = make_cv_split(label, cross_condition, params.outer_k);
    fa_x = apply_cv_mode_local(fa_x, params.cv_mode, params.outer_k);
    assert(numel(fa_x.fold) == numel(X_x), 'cross split/trial count mismatch (%d vs %d)', numel(fa_x.fold), numel(X_x));
    cross_fold = fa_x.fold(:)';
    X_delta = [X_delta(:)', X_x(:)'];
    Y_delta = [Y_delta(:)', Y_x(:)'];
    gate_mask = [gate_mask(:)', true(1, numel(X_x))];
end
Yd_delta = {};
if strcmpi(char(params.post_mode), 'prediction_ydy_parallel_fusion')
    Yd_delta = derivative_prediction_cells_local(Y_delta);
end
skip_phase_branch = numel(params.alpha_grid) == 1 && params.alpha_grid(1) == 0;
if skip_phase_branch
    Y_phase = Y_delta;
else
    [~, Y_phase] = build_xy_v2_boundary(S, ev, joint_idx, opt_phase_s);
    Y_phase = apply_phase_target_mode(Y_phase, Y_delta, params.phase_target_mode, S.eeg.fs, ...
        params.phase_lo, params.phase_hi);
end
X_phase = X_delta;
[X_aux_phase, Y_aux_phase] = build_aux_phase_training_local(S, params.aux_condition, ...
    params.aux_mode, condition, S_X, opt_delta_s, opt_phase_s, params.phase_target_mode, ...
    params.phase_lo, params.phase_hi);

D = struct();
D.label = char(label);
D.S = S;
D.ev = ev;
D.gate_mask = gate_mask;
D.n_gate_kept = n_gate_kept;
D.n_gate_dropped = n_gate_dropped;
D.X_delta = X_delta;
D.X_delta_rest = X_delta_rest;
D.n_real_trials = n_real_trials; D.cross_fold = cross_fold;
D.Y_delta = Y_delta;
D.Yd_delta = Yd_delta;
D.Y_phase = Y_phase;
D.X_phase = X_phase;
D.X_aux_phase = X_aux_phase;
D.Y_aux_phase = Y_aux_phase;
end

function parts = split_session_pool_label_local(label)
parts = strsplit(char(label), '+');
parts = parts(~cellfun('isempty', parts));
end

function maybe_write_trial_dump(trial_dump_dir, label, fold, te_idx, target_labels, Y_true, A_pred, P_pred, F_pred, C_pred, post_info, time_sec, fs, rpt_pred_smooth_sec, stage1_info)
if isempty(trial_dump_dir)
    return;
end
if nargin < 15
    stage1_info = struct();
end
if ~exist(trial_dump_dir, 'dir')
    mkdir(trial_dump_dir);
end
safe_label = regexprep(label, '[^A-Za-z0-9_]+', '_');
dump_path = fullfile(trial_dump_dir, sprintf('%s_fold%d_trial_dump.mat', safe_label, fold));
csv_path = fullfile(trial_dump_dir, sprintf('%s_fold%d_trial_r.csv', safe_label, fold));

models = {'angle', 'phase', 'fused', 'calib'};
preds = {A_pred, P_pred, F_pred, C_pred};
trial_rows = struct('session', {}, 'fold', {}, 'trial_idx', {}, 'test_ordinal', {}, ...
    'joint', {}, 'model', {}, 'r', {}, 'rpt_r', {}, 'n_samples', {}, 'first_finite_time_sec', {});
for ti = 1:numel(Y_true)
    for j = 1:numel(target_labels)
        for mi = 1:numel(models)
            [r, rpt_r, n_samples, first_t] = single_trial_model_r(Y_true, preds{mi}, time_sec, ti, j, fs, rpt_pred_smooth_sec);
            trial_rows(end+1) = struct( ... %#ok<AGROW>
                'session', label, ...
                'fold', fold, ...
                'trial_idx', te_idx(ti), ...
                'test_ordinal', ti, ...
                'joint', target_labels{j}, ...
                'model', models{mi}, ...
                'r', r, ...
                'rpt_r', rpt_r, ...
                'n_samples', n_samples, ...
                'first_finite_time_sec', first_t);
        end
    end
end

meta = struct();
meta.session = label;
meta.fold = fold;
meta.original_trial_indices = te_idx;
meta.target_labels = {target_labels};
meta.models = {models};
meta.fs = fs;
meta.rpt_pred_smooth_sec = rpt_pred_smooth_sec;
meta.post_info_schema = 'post_fir_model_v1';
% per-trial review without using any extra target information.
if isstruct(stage1_info) && ~isempty(fieldnames(stage1_info))
    save(dump_path, 'meta', 'trial_rows', 'Y_true', 'A_pred', 'P_pred', 'F_pred', 'C_pred', 'post_info', 'stage1_info', 'time_sec', '-v7.3');
else
    save(dump_path, 'meta', 'trial_rows', 'Y_true', 'A_pred', 'P_pred', 'F_pred', 'C_pred', 'post_info', 'time_sec', '-v7.3');
end

dump_csv_fid = fopen(csv_path, 'w');
if dump_csv_fid < 0
    error('Could not open trial-r CSV: %s', csv_path);
end
fprintf(dump_csv_fid, 'session,fold,trial_idx,test_ordinal,joint,model,r,rpt_r,n_samples,first_finite_time_sec\n');
for ri = 1:numel(trial_rows)
    fprintf(dump_csv_fid, '%s,%d,%d,%d,%s,%s,%.10g,%.10g,%d,%.10g\n', ...
        trial_rows(ri).session, trial_rows(ri).fold, trial_rows(ri).trial_idx, trial_rows(ri).test_ordinal, ...
        trial_rows(ri).joint, trial_rows(ri).model, trial_rows(ri).r, trial_rows(ri).rpt_r, ...
        trial_rows(ri).n_samples, trial_rows(ri).first_finite_time_sec);
end
fclose(dump_csv_fid);
end

function [r, rpt_r, n_samples, first_t] = single_trial_model_r(Y_true, Y_pred, time_sec, ti, j, fs, rpt_pred_smooth_sec)
r = NaN;
rpt_r = NaN;
n_samples = 0;
first_t = NaN;
if isempty(Y_pred) || ti > numel(Y_true) || ti > numel(Y_pred) || isempty(Y_true{ti}) || isempty(Y_pred{ti})
    return;
end
if size(Y_true{ti}, 2) < j || size(Y_pred{ti}, 2) < j
    return;
end
n = min(size(Y_true{ti}, 1), size(Y_pred{ti}, 1));
if n < 5
    return;
end
yt = Y_true{ti}(1:n, j);
yh = Y_pred{ti}(1:n, j);
ok = isfinite(yt) & isfinite(yh);
n_samples = sum(ok);
if n_samples >= 5
    r = safe_corr_local(yt(ok), yh(ok));
    if rpt_pred_smooth_sec > 0
        win = max(1, round(rpt_pred_smooth_sec * fs));
        yh_rpt = movmean(yh, [win - 1, 0], 'omitnan');
        ok_rpt = isfinite(yt) & isfinite(yh_rpt);
        if sum(ok_rpt) >= 5
            rpt_r = safe_corr_local(yt(ok_rpt), yh_rpt(ok_rpt));
        end
    else
        rpt_r = r;
    end
end
if ti <= numel(time_sec) && ~isempty(time_sec{ti})
    t = time_sec{ti};
    nn = min(numel(t), numel(ok));
    ii = find(ok(1:nn), 1, 'first');
    if ~isempty(ii)
        first_t = t(ii);
    end
end
end

function [gate_mask, n_kept, n_dropped] = make_gate_mask(label, n_trials, gate_tbl, metric, threshold, keep_frac, min_trials)
gate_mask = true(1, n_trials);
if isempty(gate_tbl)
    n_kept = n_trials;
    n_dropped = 0;
    return;
end
Gs = gate_tbl(gate_tbl.session == string(label), :);
if isempty(Gs)
    gate_mask = false(1, n_trials);
    n_kept = 0;
    n_dropped = n_trials;
    return;
end
score = nan(1, n_trials);
for gi = 1:height(Gs)
    ti = Gs.trial_idx(gi);
    if ti >= 1 && ti <= n_trials
        score(ti) = Gs.(metric)(gi);
    end
end
gate_mask = isfinite(score);
if isfinite(threshold)
    gate_mask = gate_mask & score < threshold;
end
if keep_frac < 1.0
    valid = find(isfinite(score));
    n_rank_keep = max(min_trials, ceil(keep_frac * numel(valid)));
    n_rank_keep = min(n_rank_keep, numel(valid));
    [~, ord] = sort(score(valid), 'ascend');
    rank_mask = false(1, n_trials);
    rank_mask(valid(ord(1:n_rank_keep))) = true;
    gate_mask = gate_mask & rank_mask;
end
n_kept = sum(gate_mask);
n_dropped = n_trials - n_kept;
end

function [bp_lo, bp_hi] = lookup_session_bp_band_local(bp_band_tbl, label, harmonic_k)
if isempty(bp_band_tbl)
    error('lookup_session_bp_band_local called with empty table');
end
mask = string(bp_band_tbl.session) == string(label);
if harmonic_k > 0
    hk = double(bp_band_tbl.harmonic_k);
    mask = mask & hk == double(harmonic_k);
end
idx = find(mask);
if isempty(idx)
    error('No bp band found for session=%s harmonic_k=%g', label, harmonic_k);
end
if numel(idx) > 1
    error('Multiple bp bands found for session=%s harmonic_k=%g', label, harmonic_k);
end
bp_lo = double(bp_band_tbl.band_lo(idx));
bp_hi = double(bp_band_tbl.band_hi(idx));
if ~isfinite(bp_lo) || ~isfinite(bp_hi) || bp_lo <= 0 || bp_hi <= bp_lo
    error('Invalid bp band for session=%s harmonic_k=%g: [%g %g]', label, harmonic_k, bp_lo, bp_hi);
end
end

function [tr_idx_out, te_idx_out, info] = apply_pca_score_gate_local(X_cell, tr_idx, te_idx, mode, K, agg, mad_k, q_keep, max_drop_frac, min_train_trials, shrinkage, apply_test)
mode = lower(char(mode));
info = empty_pca_score_gate_info(numel(tr_idx), numel(te_idx));
tr_idx_out = tr_idx;
te_idx_out = te_idx;
if strcmp(mode, 'none')
    return;
end
if ~any(strcmp(mode, {'t2_mad', 't2_quantile'}))
    error('pca_score_gate must be none, t2_mad, or t2_quantile');
end
K = round(K);
if K < 1
    error('pca_score_gate requires positive PCA component count');
end
[mu, inv_cov] = fit_pca_score_t2_model_local(X_cell, tr_idx, K, shrinkage);
scores = score_pca_t2_trials_local(X_cell, 1:numel(X_cell), K, mu, inv_cov, agg);
train_score = scores(tr_idx);
finite_train = train_score(isfinite(train_score));
if isempty(finite_train)
    return;
end
switch mode
    case 't2_mad'
        center = median(finite_train);
        raw_mad = median(abs(finite_train - center));
        scale = 1.4826 * raw_mad;
        if ~isfinite(scale) || scale <= eps
            scale = max(std(finite_train, 0, 'omitnan'), eps);
        end
        threshold = center + double(mad_k) * scale;
    case 't2_quantile'
        center = median(finite_train);
        scale = 1.4826 * median(abs(finite_train - center));
        threshold = local_quantile(finite_train, double(q_keep));
end
max_drop_frac = min(max(double(max_drop_frac), 0), 0.95);
if max_drop_frac < 1
    floor_thr = local_quantile(finite_train, 1 - max_drop_frac);
    threshold = max(threshold, floor_thr);
end
keep_tr = isfinite(train_score) & train_score <= threshold;
if sum(keep_tr) < min_train_trials
    keep_tr = keep_lowest_scores_local(train_score, min_train_trials);
end
test_score = scores(te_idx);
if apply_test
    keep_te = isfinite(test_score) & test_score <= threshold;
else
    keep_te = true(size(test_score));
end
tr_idx_out = tr_idx(keep_tr);
te_idx_out = te_idx(keep_te);
info.train_kept = numel(tr_idx_out);
info.train_dropped = numel(tr_idx) - info.train_kept;
info.test_kept = numel(te_idx_out);
info.test_dropped = numel(te_idx) - info.test_kept;
info.threshold = threshold;
info.train_center = center;
info.train_scale = scale;
end

function [keep_mask, info] = make_pre_cv_pca_gate_mask_local(X_delta, X_phase, Y_delta, S, base_idx, feature_mode, shared_pca_file, shared_pca_components, phase_lo, phase_hi, lag_count, mode, agg, mad_k, q_keep, max_drop_frac, min_trials, shrinkage, shared_pca_scaler_scope)
if nargin < 19 || isempty(shared_pca_scaler_scope)
    shared_pca_scaler_scope = 'fold';
end
keep_mask = false(1, numel(X_delta));
info = empty_pca_score_gate_info(numel(base_idx), 0);
if isempty(base_idx)
    return;
end
if strcmpi(char(mode), 'none')
    keep_mask(base_idx) = true;
    return;
end
[X_delta_pre, ~] = prepare_fold_feature_cells_local(X_delta, X_phase, Y_delta, S, base_idx, ...
    feature_mode, shared_pca_file, shared_pca_components, phase_lo, phase_hi, lag_count, shared_pca_scaler_scope, 3);
[keep_idx, ~, info] = apply_pca_score_gate_local(X_delta_pre, base_idx, [], mode, ...
    round(shared_pca_components), agg, mad_k, q_keep, max_drop_frac, min_trials, shrinkage, false);
keep_mask(keep_idx) = true;
end

function info = empty_pca_score_gate_info(n_train, n_test)
info = struct('train_kept', n_train, 'train_dropped', 0, ...
    'test_kept', n_test, 'test_dropped', 0, ...
    'threshold', NaN, 'train_center', NaN, 'train_scale', NaN);
end

function [mu, inv_cov] = fit_pca_score_t2_model_local(X_cell, tr_idx, K, shrinkage)
sum_x = zeros(K, 1);
n_obs = 0;
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    assert(mod(size(X, 1), K) == 0, 'PCA score rows must be divisible by K');
    n_lag = size(X, 1) / K;
    for li = 1:n_lag
        blk = X((li-1)*K+1:li*K, :);
        ok = all(isfinite(blk), 1);
        if any(ok)
            sum_x = sum_x + sum(blk(:, ok), 2);
            n_obs = n_obs + sum(ok);
        end
    end
end
if n_obs < K + 5
    mu = zeros(K, 1);
    inv_cov = eye(K);
    return;
end
mu = sum_x ./ n_obs;
C = zeros(K, K);
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    n_lag = size(X, 1) / K;
    for li = 1:n_lag
        blk = X((li-1)*K+1:li*K, :);
        ok = all(isfinite(blk), 1);
        if any(ok)
            Z = blk(:, ok) - mu;
            C = C + Z * Z';
        end
    end
end
C = C ./ max(n_obs - 1, 1);
avg_var = trace(C) / max(K, 1);
shrinkage = min(max(double(shrinkage), 0), 1);
C = (1 - shrinkage) * C + shrinkage * avg_var * eye(K);
C = (C + C') / 2;
[V, D] = eig(C);
d = real(diag(D));
floor_var = max(avg_var * 1e-6, eps);
d = max(d, floor_var);
inv_cov = V * diag(1 ./ d) * V';
inv_cov = (real(inv_cov) + real(inv_cov)') / 2;
mu(~isfinite(mu)) = 0;
inv_cov(~isfinite(inv_cov)) = 0;
end

function scores = score_pca_t2_trials_local(X_cell, idx, K, mu, inv_cov, agg)
scores = nan(1, numel(X_cell));
agg = lower(char(agg));
for ii = 1:numel(idx)
    ti = idx(ii);
    X = double(X_cell{ti});
    if isempty(X) || mod(size(X, 1), K) ~= 0
        continue;
    end
    n_lag = size(X, 1) / K;
    d2_all = [];
    for li = 1:n_lag
        blk = X((li-1)*K+1:li*K, :);
        ok = all(isfinite(blk), 1);
        if any(ok)
            Z = blk(:, ok) - mu;
            d2 = sum(Z .* (inv_cov * Z), 1);
            d2_all = [d2_all, d2(isfinite(d2) & d2 >= 0)]; %#ok<AGROW>
        end
    end
    if isempty(d2_all), continue; end
    switch agg
        case 'mean'
            scores(ti) = mean(d2_all);
        case 'median'
            scores(ti) = median(d2_all);
        case 'p95'
            scores(ti) = local_quantile(d2_all, 0.95);
        case 'max'
            scores(ti) = max(d2_all);
        otherwise
            error('pca_score_gate_agg must be mean, median, p95, or max');
    end
end
end

function keep = keep_lowest_scores_local(score, min_keep)
keep = false(size(score));
valid = find(isfinite(score));
if isempty(valid), return; end
n_keep = min(numel(valid), max(1, round(min_keep)));
[~, ord] = sort(score(valid), 'ascend');
keep(valid(ord(1:n_keep))) = true;
end

function q = local_quantile(x, p)
x = sort(x(isfinite(x)));
if isempty(x)
    q = NaN;
    return;
end
p = min(max(p, 0), 1);
if numel(x) == 1
    q = x(1);
    return;
end
pos = 1 + (numel(x) - 1) * p;
lo = floor(pos);
hi = ceil(pos);
if lo == hi
    q = x(lo);
else
    q = x(lo) + (pos - lo) * (x(hi) - x(lo));
end
end

function fa = apply_cv_mode_local(fa, cv_mode, outer_k)
mode = lower(char(cv_mode));
switch mode
    case 'random'
        return;
    case 'block'
        n = numel(fa.fold);
        if n < 1
            return;
        end
        K = min(round(double(outer_k)), n);
        assert(isfinite(K) && K >= 1, 'outer_k must be a positive scalar');
        edges = round(linspace(0, n, K + 1));
        fold = zeros(n, 1);
        for k = 1:K
            idx = (edges(k) + 1):edges(k + 1);
            fold(idx) = k;
        end
        fa.fold = fold;
        fa.cv_mode = 'block';
    otherwise
        error('cv_mode must be random or block');
end
end

function fa = make_cv_split_indices_local(label, condition, K, trial_idx, cv_mode)
label = char(label);
condition = char(condition);
trial_idx = trial_idx(:)';
n = numel(trial_idx);
K = min(round(double(K)), n);
assert(isfinite(K) && K >= 1, 'outer_k must be a positive scalar');
mode = lower(char(cv_mode));
fold = zeros(1, n);
switch mode
    case 'random'
        cond_tag = double(condition(1));
        seed_base = 20260505;
        seed = seed_base + sum(double(label)) + cond_tag;
        old_state = rng;
        rng(seed);
        perm = randperm(n);
        rng(old_state);
        for k = 1:K
            fold(perm(k:K:end)) = k;
        end
    case 'block'
        seed = NaN;
        edges = round(linspace(0, n, K + 1));
        for k = 1:K
            idx = (edges(k) + 1):edges(k + 1);
            fold(idx) = k;
        end
    otherwise
        error('cv_mode must be random or block');
end
fa = struct('session', label, 'condition', condition, 'seed', seed, ...
    'n_trials', n, 'fold', fold(:), 'trial_idx', trial_idx(:), 'cv_mode', mode);
end

function Y_shuf = shuffle_cell_order_local(Y_cell, seed)
Y_shuf = Y_cell;
n = numel(Y_cell);
if n < 2
    return;
end
old_state = rng;
cleanup = onCleanup(@() rng(old_state));
rng(seed);
perm = randperm(n);
if all(perm == 1:n)
    perm = [2:n 1];
end
Y_shuf = Y_cell(perm);
end
fclose(fid);
fprintf('Wrote %s\n', out_csv);
end

function maybe_write_diagnostics(diag_dir, diag_sessions, diag_max_trials, label, fold, S, Y_true, A_pred, P_pred, F_pred, C_pred, time_sec, rows, target_labels, feature_mode, shared_pca_file, shared_pca_components, diag_plot_pca, diag_low_trial_threshold, diag_low_max_per_fold)
if isempty(diag_dir)
    return;
end
try
if isstring(diag_sessions) || ischar(diag_sessions)
    diag_sessions = cellstr(diag_sessions);
end
if ~isempty(diag_sessions) && ~any(strcmp(label, diag_sessions))
    return;
end
if ~exist(diag_dir, 'dir')
    mkdir(diag_dir);
end

rvals = [rows.r_fused];
finite = find(isfinite(rvals));
if isempty(finite)
    return;
end
[~, best_pos] = max(rvals(finite));
best_j = finite(best_pos);
safe_label = regexprep(label, '[^A-Za-z0-9_]+', '_');
safe_joint = regexprep(target_labels{best_j}, '[^A-Za-z0-9_]+', '_');

pred_png = fullfile(diag_dir, sprintf('%s_fold%d_%s_real_pred.png', safe_label, fold, safe_joint));
plot_real_pred_diag(pred_png, label, fold, target_labels{best_j}, Y_true, A_pred, P_pred, F_pred, C_pred, time_sec, best_j, rows(best_j), diag_max_trials);

if diag_plot_pca && any(strcmpi(char(feature_mode), {'delta_shared_pca', 'delta_shared_pca_tf_whiten', 'delta_fold_diagwhiten_shared_pca'}))
    topo_png = fullfile(diag_dir, sprintf('%s_fold%d_sharedPCA_topo.png', safe_label, fold));
    plot_shared_pca_topo_diag(topo_png, label, S, shared_pca_file, shared_pca_components);
end

diag_mat = fullfile(diag_dir, sprintf('%s_fold%d_%s_diag.mat', safe_label, fold, safe_joint));
Y_true_plot = Y_true; %#ok<NASGU>
A_pred_plot = A_pred; %#ok<NASGU>
P_pred_plot = P_pred; %#ok<NASGU>
F_pred_plot = F_pred; %#ok<NASGU>
C_pred_plot = C_pred; %#ok<NASGU>
time_sec_plot = time_sec; %#ok<NASGU>
row_plot = rows(best_j); %#ok<NASGU>
save(diag_mat, 'label', 'fold', 'target_labels', 'best_j', 'Y_true_plot', ...
    'A_pred_plot', 'P_pred_plot', 'F_pred_plot', 'C_pred_plot', 'time_sec_plot', 'row_plot', '-v7');
if isfinite(diag_low_trial_threshold)
    plot_low_trial_diagnostics(diag_dir, label, fold, target_labels, Y_true, A_pred, P_pred, F_pred, ...
        time_sec, diag_low_trial_threshold, diag_low_max_per_fold);
end
catch ME
    warning('Diagnostics skipped for %s fold %d: %s', label, fold, ME.message);
end
end

function plot_low_trial_diagnostics(diag_dir, label, fold, target_labels, Y_true, A_pred, P_pred, F_pred, time_sec, threshold, max_per_fold)
cand = [];
for ti = 1:numel(Y_true)
    for j = 1:numel(target_labels)
        [r_f, ~] = eval_one(Y_true(ti), F_pred(ti), j);
        if isfinite(r_f) && r_f < threshold
            [r_a, ~] = eval_one(Y_true(ti), A_pred(ti), j);
            [r_p, ~] = eval_one(Y_true(ti), P_pred(ti), j);
            cand(end+1).trial = ti; %#ok<AGROW>
            cand(end).joint = j;
            cand(end).r_fused = r_f;
            cand(end).r_angle = r_a;
            cand(end).r_phase = r_p;
        end
    end
end
if isempty(cand), return; end
[~, ord] = sort([cand.r_fused], 'ascend');
cand = cand(ord);
n_plot = min(max_per_fold, numel(cand));
safe_label = regexprep(label, '[^A-Za-z0-9_]+', '_');
for ci = 1:n_plot
    ti = cand(ci).trial;
    j = cand(ci).joint;
    safe_joint = regexprep(target_labels{j}, '[^A-Za-z0-9_]+', '_');
    out_png = fullfile(diag_dir, sprintf('%s_fold%d_trial%02d_%s_lowTrial_r%.3f_real_pred.png', ...
        safe_label, fold, ti, safe_joint, cand(ci).r_fused));
    plot_single_trial_diag(out_png, label, fold, ti, target_labels{j}, Y_true{ti}, ...
        A_pred{ti}, P_pred{ti}, F_pred{ti}, get_time_cell(time_sec, ti, size(Y_true{ti}, 1)), j, cand(ci));
end
end

function plot_single_trial_diag(out_png, label, fold, trial_idx, joint_label, Y_true, A_pred, P_pred, F_pred, t_sec, j, row)
fig = figure('Visible', 'off', 'Position', [100 100 1200 760]);
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
preds = {A_pred, P_pred, F_pred};
names = {'angle', 'phase', 'fused'};
colors = [0.10 0.35 0.80; 0.85 0.35 0.10; 0.10 0.55 0.25];
rs = [row.r_angle row.r_phase row.r_fused];
n = min([size(Y_true, 1), size(A_pred, 1), size(P_pred, 1), size(F_pred, 1)]);
t = t_sec(1:min(numel(t_sec), n));
n = min(n, numel(t));
for pi = 1:3
    nexttile;
    hold on;
    plot(t, Y_true(1:n, j), 'Color', [0.10 0.10 0.10], 'LineWidth', 1.2);
    plot(t, preds{pi}(1:n, j), 'Color', colors(pi, :), 'LineWidth', 1.2);
    hold off;
    grid on;
    ylabel('delta angle');
    title(sprintf('%s | fold %d trial %d | %s | %s model r=%.3f', ...
        label, fold, trial_idx, joint_label, names{pi}, rs(pi)), 'Interpreter', 'none');
    if pi == 1
        legend({'real','pred'}, 'Location', 'best');
    end
end
xlabel('time from raw trial onset (s)');
export_figure_local(fig, out_png);
close(fig);
end

function plot_real_pred_diag(out_png, label, fold, joint_label, Y_true, A_pred, P_pred, F_pred, C_pred, time_sec, j, row, max_trials)
fig = figure('Visible', 'off', 'Position', [100 100 1200 920]);
preds = {A_pred, P_pred, F_pred};
names = {'angle model', 'phase model', 'fused model'};
colors = [0.10 0.35 0.80; 0.85 0.35 0.10; 0.10 0.55 0.25; 0.45 0.20 0.65];
if ~isempty(C_pred)
    preds{end+1} = C_pred;
    names{end+1} = 'calibrated model';
end
tiledlayout(numel(preds), 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for pi = 1:numel(preds)
    nexttile;
    hold on;
    n_plot = min(max_trials, numel(Y_true));
    offset = 0;
    for ti = 1:n_plot
        t0 = get_time_cell(time_sec, ti, size(Y_true{ti}, 1));
        n = min([size(Y_true{ti}, 1), size(preds{pi}{ti}, 1), numel(t0)]);
        if n == 0, continue; end
        y = Y_true{ti}(1:n, j);
        yp = preds{pi}{ti}(1:n, j);
        if all(~isfinite(y)) || all(~isfinite(yp)), continue; end
        tt = t0(1:n) + offset;
        plot(tt, y, 'Color', [0.15 0.15 0.15], 'LineWidth', 1.1);
        plot(tt, yp, 'Color', colors(pi, :), 'LineWidth', 1.1);
        first_pred = find(isfinite(yp), 1, 'first');
        if ~isempty(first_pred) && first_pred > 1
            xline(tt(first_pred), '--', 'Color', colors(pi, :));
        end
        xline(tt(end), ':', 'Color', [0.70 0.70 0.70]);
        offset = tt(end) + 0.5 - t0(1);
    end
    hold off;
    grid on;
    ylabel('delta angle');
    [r_model, rpt_model] = eval_one(Y_true, preds{pi}, j);
    first_t = first_finite_prediction_time(preds{pi}, time_sec, j, n_plot);
    title(sprintf('%s | %s | fold %d | %s | r=%.3f rpt=%.3f alpha=%.2f | first pred %.2fs', ...
        label, joint_label, fold, names{pi}, r_model, rpt_model, row.alpha, first_t), ...
        'Interpreter', 'none');
    if pi == 1
        legend({'real', 'pred'}, 'Location', 'best');
    end
end
xlabel('concatenated seconds; each retained segment is aligned to raw trial onset');
export_figure_local(fig, out_png);
close(fig);
end

function time_sec = make_diag_time_cells(Y_cell, lag_ms, fs)
if nargin < 3 || isempty(fs), fs = 250; end
lag_samples = unique(round(double(lag_ms) * double(fs) / 1000));
if isempty(lag_samples), lag_samples = 0; end
start_sample = max(1, 1 - min(lag_samples)) - 1;
time_sec = cell(size(Y_cell));
for k = 1:numel(Y_cell)
    n = size(Y_cell{k}, 1);
    time_sec{k} = (double(start_sample) + (0:n-1)') ./ double(fs);
end
end

function t = get_time_cell(time_sec, idx, n)
if nargin < 3, n = 0; end
if ~isempty(time_sec) && idx <= numel(time_sec) && ~isempty(time_sec{idx})
    t = time_sec{idx}(:);
else
    t = (0:max(n-1, 0))';
end
end

function first_t = first_finite_prediction_time(Y_pred, time_sec, j, n_plot)
first_t = NaN;
n_plot = min(n_plot, numel(Y_pred));
for ti = 1:n_plot
    if isempty(Y_pred{ti}) || size(Y_pred{ti}, 2) < j, continue; end
    t0 = get_time_cell(time_sec, ti, size(Y_pred{ti}, 1));
    n = min(size(Y_pred{ti}, 1), numel(t0));
    if n == 0, continue; end
    ii = find(isfinite(Y_pred{ti}(1:n, j)), 1, 'first');
    if ~isempty(ii)
        first_t = t0(ii);
        return;
    end
end
end

function plot_shared_pca_topo_diag(out_png, label, S, shared_pca_file, shared_pca_components)
shared_file = char(shared_pca_file);
if contains(shared_file, '%s')
    shared_file = sprintf(shared_file, label);
end
if isempty(shared_file) || exist(shared_file, 'file') ~= 2
    return;
end
P = load(shared_file);
if ~isfield(P, 'basis')
    return;
end
K = min([round(shared_pca_components), size(P.basis, 2), 6]);
if K < 1
    return;
end
fig = figure('Visible', 'off', 'Position', [100 100 1200 650]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:K
    nexttile;
    w = double(P.basis(:, k));
    if isfield(S.eeg, 'chanlocs') && ~isempty(S.eeg.chanlocs)
        chanlocs = S.eeg.chanlocs;
    elseif isfield(S.eeg, 'c') && ~isempty(S.eeg.c)
        chanlocs = S.eeg.c;
    else
        return;
    end
    plot_scalp_values_local(w, chanlocs);
    title(sprintf('%s shared PCA %d', label, k), 'Interpreter', 'none');
end
export_figure_local(fig, out_png);
close(fig);
end

function plot_scalp_values_local(w, chanlocs)
if exist('topoplot', 'file') == 2
    lim = max(abs(w), [], 'omitnan');
    if ~isfinite(lim) || lim == 0, lim = 1; end
    topoplot(w, chanlocs, 'electrodes', 'on', 'maplimits', [-lim lim]);
    colorbar;
    return;
end
[x, y] = chanloc_xy_local(chanlocs);
scatter(x, y, 90, w, 'filled');
axis equal off;
colormap(gca, parula);
colorbar;
end

function [x, y] = chanloc_xy_local(chanlocs)
n = numel(chanlocs);
x = nan(n, 1);
y = nan(n, 1);
for ci = 1:n
    if isfield(chanlocs, 'X') && isfinite(chanlocs(ci).X) && isfield(chanlocs, 'Y') && isfinite(chanlocs(ci).Y)
        x(ci) = chanlocs(ci).X;
        y(ci) = chanlocs(ci).Y;
    elseif isfield(chanlocs, 'theta') && isfield(chanlocs, 'radius') && isfinite(chanlocs(ci).theta) && isfinite(chanlocs(ci).radius)
        th = deg2rad(chanlocs(ci).theta);
        x(ci) = chanlocs(ci).radius * cos(th);
        y(ci) = chanlocs(ci).radius * sin(th);
    else
        x(ci) = cos(2 * pi * ci / n);
        y(ci) = sin(2 * pi * ci / n);
    end
end
end

function export_figure_local(fig, out_png)
try
    exportgraphics(fig, out_png, 'Resolution', 180);
catch
    print(fig, out_png, '-dpng', '-r180');
end
end

function [X_aux_phase, Y_aux_phase] = build_aux_phase_training_local(S, aux_condition, aux_mode, main_condition, S_X, opt_delta, opt_phase, phase_target_mode, phase_lo, phase_hi)
X_aux_phase = {};
Y_aux_phase = {};
aux_condition = upper(char(aux_condition));
if strcmp(aux_condition, 'NONE')
    return;
end
if ~strcmpi(char(aux_mode), 'phase_decoder_only')
    error('aux_mode must be phase_decoder_only');
end
if ~strcmpi(main_condition, 'MI') || ~strcmp(aux_condition, 'EXEC')
    error('aux_condition=EXEC is only implemented for MI target decoding');
end

ev_aux = trial_filter(S.events, 'EXEC');
if isempty(ev_aux)
    return;
end
joint_idx = nan(1, numel(S_X));
for j = 1:numel(S_X)
    joint_idx(j) = find_goni_idx(S.goni.labels, S_X{j});
end

opt_aux_delta = opt_delta;
opt_aux_delta.input_control_mode = 'full';
opt_aux_delta.rest_events = [];
[X_aux, Y_aux_delta] = build_xy_delta0_local(S, ev_aux, joint_idx, opt_aux_delta);
[~, Y_aux_phase0] = build_xy_v2_boundary(S, ev_aux, joint_idx, opt_phase);
Y_aux_phase0 = apply_phase_target_mode(Y_aux_phase0, Y_aux_delta, phase_target_mode, S.eeg.fs, phase_lo, phase_hi);

keep = valid_aux_phase_mask(X_aux, Y_aux_phase0);
X_aux_phase = X_aux(keep);
Y_aux_phase = Y_aux_phase0(keep);
end

function keep = valid_aux_phase_mask(X_aux, Y_aux_phase)
keep = false(size(X_aux));
for ii = 1:numel(X_aux)
    keep(ii) = size(X_aux{ii}, 2) > 0 && size(Y_aux_phase{ii}, 1) > 0;
end
end

function keep = valid_trial_mask(X_delta, Y_delta, Y_phase, idx)
keep = false(size(idx));
for ii = 1:numel(idx)
    k = idx(ii);
    keep(ii) = size(X_delta{k}, 2) > 0 && size(Y_delta{k}, 1) > 0 && ...
        size(Y_phase{k}, 1) > 0;
end
end

function assert_xy_sample_counts_local(X_cell, Y_cell, idx, label, fold, stage)
for ii = 1:numel(idx)
    k = idx(ii);
    nx = size(X_cell{k}, 2);
    ny = size(Y_cell{k}, 1);
    if nx ~= ny
        error('%s %s fold %d trial %d X_cols=%d Y_rows=%d', ...
            char(stage), char(label), fold, k, nx, ny);
    end
end
end

function assert_prepared_sample_counts_preserved_local(X_fit, X_raw, idx, label, fold, stage)
for ii = 1:numel(idx)
    k = idx(ii);
    n_fit = size(X_fit{k}, 2);
    n_raw = size(X_raw{k}, 2);
    if n_fit ~= n_raw
        error('%s %s fold %d trial %d prepared_cols=%d raw_cols=%d', ...
            char(stage), char(label), fold, k, n_fit, n_raw);
    end
end
end

function [X_delta_fit, X_phase_fit, feature_info] = prepare_fold_feature_cells_local(X_delta, X_phase, Y_delta, S, tr_idx, feature_mode, shared_pca_file, shared_pca_components, phase_lo, phase_hi, lag_count, shared_pca_scaler_scope, harmonic_ssd_append_filters)
if nargin < 12 || isempty(shared_pca_scaler_scope)
    shared_pca_scaler_scope = 'fold';
end
if nargin < 13 || isempty(harmonic_ssd_append_filters)
    harmonic_ssd_append_filters = 3;
end
shared_pca_scaler_scope = lower(char(shared_pca_scaler_scope));
mode = lower(char(feature_mode));
feature_info = struct();
feature_info.mode = mode;
if ~any(strcmp(mode, {'delta_fold_diagwhiten', 'delta_fold_diagwhiten_feature', 'delta_fold_diagwhiten_csp_phase', 'delta_fold_diagwhiten_harmonic_csp_phase', 'delta_fold_diagwhiten_harmonic_ssd', 'delta_fold_diagwhiten_harmonic_block_pca', 'delta_fold_diagwhiten_joint_complex_pca', 'delta_fold_diagwhiten_harmonic_block_free_ri_pca', 'delta_fold_diagwhiten_shared_pca_plus_harmonic_ssd', 'delta_fold_diagwhiten_shared_pca_plus_harmonic_pls', 'delta_fold_diagwhiten_shared_pca_plus_harmonic_pls_resid', 'delta_fold_diagwhiten_shared_pca_plus_harmonic_cca_resid', 'delta_fold_diagwhiten_shared_pca_plus_joint_harmonic_response', 'delta_fold_diagwhiten_shared_pca_harmonic_response_basis', 'delta_fold_diagwhiten_shared_pca_orthcomp_response_basis', 'delta_fold_diagwhiten_shared_pca_plus_harmonic_spoc', 'delta_fold_diagwhiten_harmonic_response_residual_pca', 'delta_fold_diagwhiten_shared_pca', 'delta_fold_diagwhiten_shared_pca_ea', 'delta_fold_diagwhiten_shared_pca_plus_feature', 'delta_fold_diagwhiten_lagspace_shared_pca', 'delta_shared_pca_tf_whiten', 'delta_lagspace_pca_tf_whiten'}))
    X_delta_fit = X_delta;
    X_phase_fit = X_phase;
    feature_info.transform = 'none';
    return;
end
if numel(X_delta) ~= numel(X_phase)
    error('%s currently requires matching X_delta/X_phase cells', mode);
end
if isfield(S, 'pool_parts') && ~isempty(S.pool_parts)
    [X_delta_fit, X_phase_fit] = prepare_pooled_fold_feature_cells_local( ...
        X_delta, X_phase, Y_delta, S.pool_parts, tr_idx, feature_mode, shared_pca_file, ...
        shared_pca_components, phase_lo, phase_hi, lag_count, shared_pca_scaler_scope, harmonic_ssd_append_filters);
    feature_info.transform = 'pooled_session_specific';
    return;
end
if strcmp(mode, 'delta_shared_pca_tf_whiten')
    K = min(round(shared_pca_components), first_nonempty_feature_rows_local(X_delta));
    [mu, W] = fold_pca_whitener_from_lag_cells_local(X_delta, tr_idx, K);
    X_delta_fit = apply_fold_pca_whitener_local(X_delta, K, mu, W);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'pca_tf_whiten';
    feature_info.K = K;
    feature_info.mu = mu;
    feature_info.W = W;
    return;
end
if strcmp(mode, 'delta_lagspace_pca_tf_whiten')
    K = min(round(shared_pca_components), first_nonempty_feature_rows_local(X_delta));
    [mu, W] = fold_lagspace_pca_whitener_local(X_delta, tr_idx, K);
    X_delta_fit = apply_lagspace_pca_whitener_local(X_delta, mu, W);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'lagspace_pca_tf_whiten';
    feature_info.K = K;
    feature_info.mu = mu;
    feature_info.W = W;
    return;
end
n_ch = numel(S.eeg.labels);
if strcmp(mode, 'delta_fold_diagwhiten')
    [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_delta, tr_idx, n_ch);
    X_delta_fit = apply_fold_diagwhiten_local(X_delta, n_ch, mu, sd);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten';
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = n_ch;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_feature')
    n_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_delta, tr_idx, n_feat);
    X_delta_fit = apply_fold_diagwhiten_local(X_delta, n_feat, mu, sd);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_feature';
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = n_feat;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_csp_phase')
    n_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    K = min(round(shared_pca_components), n_feat);
    [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_delta, tr_idx, n_feat);
    X_scaled = apply_fold_diagwhiten_local(X_delta, n_feat, mu, sd);
    [basis, csp_info] = fit_phase_csp_basis_from_lag_cells_local( ...
        X_scaled, Y_delta, tr_idx, n_feat, K, S.goni.fs, phase_lo, phase_hi);
    X_delta_fit = apply_fold_csp_local(X_scaled, n_feat, basis);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_csp_phase';
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = n_feat;
    feature_info.K = size(basis, 2);
    feature_info.basis = basis;
    feature_info.csp = csp_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_harmonic_csp_phase')
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    pair_counts = allocate_harmonic_csp_pair_counts_local(shared_pca_components, layout.n_harm, layout.n_ch);
    [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_delta, tr_idx, layout.n_feat);
    X_scaled = apply_fold_diagwhiten_local(X_delta, layout.n_feat, mu, sd);
    [basis_cell, csp_info] = fit_harmonic_phase_csp_basis_from_lag_cells_local( ...
        X_scaled, Y_delta, tr_idx, layout, pair_counts, S.goni.fs, phase_lo, phase_hi);
    X_delta_fit = apply_harmonic_csp_local(X_scaled, layout, basis_cell);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_harmonic_csp_phase';
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = layout.n_feat;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.pair_counts = pair_counts;
    feature_info.K = first_nonempty_feature_rows_local(X_delta_fit) / lag_count;
    feature_info.basis_cell = basis_cell;
    feature_info.csp = csp_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_harmonic_ssd')
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    filter_counts = allocate_harmonic_ssd_filter_counts_local(shared_pca_components, layout.n_harm, layout.n_ch);
    [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_delta, tr_idx, layout.n_feat);
    X_scaled = apply_fold_diagwhiten_local(X_delta, layout.n_feat, mu, sd);
    [basis_cell, ssd_info] = fit_harmonic_ssd_basis_from_lag_cells_local( ...
        X_scaled, tr_idx, layout, filter_counts);
    X_delta_fit = apply_harmonic_csp_local(X_scaled, layout, basis_cell);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_harmonic_ssd';
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = layout.n_feat;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.filter_counts = filter_counts;
    feature_info.K = first_nonempty_feature_rows_local(X_delta_fit) / lag_count;
    feature_info.basis_cell = basis_cell;
    feature_info.ssd = ssd_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_harmonic_block_pca')
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    filter_counts = allocate_harmonic_ssd_filter_counts_local(shared_pca_components, layout.n_harm, layout.n_ch);
    [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_delta, tr_idx, layout.n_feat);
    X_scaled = apply_fold_diagwhiten_local(X_delta, layout.n_feat, mu, sd);
    [basis_cell, block_pca_info] = fit_harmonic_block_pca_basis_from_lag_cells_local( ...
        X_scaled, tr_idx, layout, filter_counts);
    X_delta_fit = apply_harmonic_csp_local(X_scaled, layout, basis_cell);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_harmonic_block_pca';
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = layout.n_feat;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.filter_counts = filter_counts;
    feature_info.K = first_nonempty_feature_rows_local(X_delta_fit) / lag_count;
    feature_info.basis_cell = basis_cell;
    feature_info.block_pca = block_pca_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_joint_complex_pca')
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    K_complex = max(1, floor(double(shared_pca_components) / 2));
    K_complex = min(K_complex, layout.n_harm * layout.n_ch);
    [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_delta, tr_idx, layout.n_feat);
    X_scaled = apply_fold_diagwhiten_local(X_delta, layout.n_feat, mu, sd);
    [basis, complex_pca_info] = fit_joint_complex_pca_basis_from_lag_cells_local( ...
        X_scaled, tr_idx, layout, K_complex);
    X_delta_fit = apply_joint_complex_pca_local(X_scaled, layout, basis);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_joint_complex_pca';
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = layout.n_feat;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.K_complex = K_complex;
    feature_info.K = first_nonempty_feature_rows_local(X_delta_fit) / lag_count;
    feature_info.basis = basis;
    feature_info.complex_pca = complex_pca_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_harmonic_block_free_ri_pca')
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    component_counts = allocate_harmonic_real_component_counts_local(shared_pca_components, layout.n_harm, 2 * layout.n_ch);
    [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_delta, tr_idx, layout.n_feat);
    X_scaled = apply_fold_diagwhiten_local(X_delta, layout.n_feat, mu, sd);
    [basis_cell, block_free_ri_info] = fit_harmonic_block_free_ri_pca_basis_from_lag_cells_local( ...
        X_scaled, tr_idx, layout, component_counts);
    X_delta_fit = apply_harmonic_block_free_ri_pca_local(X_scaled, layout, basis_cell);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_harmonic_block_free_ri_pca';
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = layout.n_feat;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.component_counts = component_counts;
    feature_info.K = first_nonempty_feature_rows_local(X_delta_fit) / lag_count;
    feature_info.basis_cell = basis_cell;
    feature_info.block_free_ri_pca = block_free_ri_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_harmonic_response_residual_pca')
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    K_total = min(round(shared_pca_components), layout.n_feat);
    [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_delta, tr_idx, layout.n_feat);
    X_scaled = apply_fold_diagwhiten_local(X_delta, layout.n_feat, mu, sd);
    filter_counts = allocate_harmonic_append_ssd_filter_counts_local( ...
        harmonic_ssd_append_filters, layout.n_harm, layout.n_ch);
    n_response_rows = 2 * sum(filter_counts);
    if n_response_rows >= K_total
        error('%s requires response rows %d to be smaller than total feature budget %d for session %s', ...
            mode, n_response_rows, K_total, S.label);
    end
    pca_keep = K_total - n_response_rows;
    [basis_cell, pls_info] = fit_harmonic_pls_basis_from_lag_cells_local( ...
        X_scaled, Y_delta, tr_idx, layout, filter_counts);
    X_response_fit = apply_harmonic_csp_local(X_scaled, layout, basis_cell);
    X_resid = remove_harmonic_response_basis_from_lag_cells_local(X_scaled, layout, basis_cell);
    [resid_basis, resid_pca_info] = fit_feature_pca_basis_from_lag_cells_local( ...
        X_resid, tr_idx, layout.n_feat, pca_keep);
    X_pca_fit = apply_fold_diagwhiten_shared_pca_local( ...
        X_resid, layout.n_feat, resid_basis, zeros(layout.n_feat, 1), ones(layout.n_feat, 1));
    X_delta_fit = concat_feature_cells_local(X_pca_fit, X_response_fit);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_harmonic_response_residual_pca';
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = layout.n_feat;
    feature_info.n_pca_rows = pca_keep;
    feature_info.n_response_rows = first_nonempty_feature_rows_local(X_response_fit) / lag_count;
    feature_info.n_total_rows = first_nonempty_feature_rows_local(X_delta_fit) / lag_count;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.filter_counts = filter_counts;
    feature_info.resid_basis = resid_basis;
    feature_info.basis_cell = basis_cell;
    feature_info.pls = pls_info;
    feature_info.residual_pca = resid_pca_info;
    return;
end
shared_file = char(shared_pca_file);
if isempty(shared_file)
    error('%s requires shared_pca_file', mode);
end
if contains(shared_file, '%s')
    shared_file = sprintf(shared_file, S.label);
end
P = load(shared_file);
if ~isfield(P, 'basis')
    error('shared_pca_file %s has no basis variable', shared_file);
end
if isfield(P, 'labels')
    assert(numel(P.labels) == size(P.basis, 1), ...
        'shared PCA label count does not match basis rows for session %s', S.label);
end
K = min(round(shared_pca_components), size(P.basis, 2));
basis = double(P.basis(:, 1:K));
feature_info.shared_pca_file = shared_file;
feature_info.K = K;
feature_info.basis = basis;
if isfield(P, 'labels')
    feature_info.labels = P.labels;
end
if strcmp(mode, 'delta_fold_diagwhiten_lagspace_shared_pca')
    [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_delta, tr_idx, n_ch);
    X_delta_fit = apply_fold_diagwhiten_local(X_delta, n_ch, mu, sd);
    if size(basis, 1) ~= first_nonempty_feature_rows_local(X_delta_fit)
        error('lag-space PCA rows %d do not match lagged feature rows %d', ...
            size(basis, 1), first_nonempty_feature_rows_local(X_delta_fit));
    end
    X_delta_fit = apply_lagspace_shared_pca_local(X_delta_fit, basis);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_lagspace_shared_pca';
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = n_ch;
    return;
end
n_feat = size(basis, 1);
if strcmp(mode, 'delta_fold_diagwhiten_shared_pca_plus_feature')
    total_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    n_extra = total_feat - n_feat;
    if n_extra <= 0
        error('%s requires appended feature rows after %d shared-PCA rows; got total feature rows %d for session %s', ...
            mode, n_feat, total_feat, S.label);
    end
    [X_pca_raw, X_extra_raw] = split_lag_feature_cells_local(X_delta, n_feat, n_extra);
    [mu_pca, sd_pca, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local( ...
        X_pca_raw, tr_idx, n_feat, shared_pca_scaler_scope);
    X_pca_fit = apply_fold_diagwhiten_shared_pca_local(X_pca_raw, n_feat, basis, mu_pca, sd_pca);
    [mu_extra, sd_extra] = fold_channel_scaler_from_lag_cells_local(X_extra_raw, tr_idx, n_extra);
    X_extra_fit = apply_fold_diagwhiten_local(X_extra_raw, n_extra, mu_extra, sd_extra);
    X_delta_fit = concat_feature_cells_local(X_pca_fit, X_extra_fit);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_shared_pca_plus_feature';
    feature_info.scaler_scope = scaler_scope_used;
    feature_info.feature_mu = mu_pca;
    feature_info.feature_sd = sd_pca;
    feature_info.extra_mu = mu_extra;
    feature_info.extra_sd = sd_extra;
    feature_info.n_feature_rows = n_feat;
    feature_info.n_extra_rows = n_extra;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_shared_pca_plus_harmonic_ssd')
    total_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    if total_feat ~= n_feat
        error('%s requires the same raw quadrature rows as the shared PCA basis; got total feature rows %d and PCA rows %d for session %s', ...
            mode, total_feat, n_feat, S.label);
    end
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    if layout.n_feat ~= n_feat
        error('%s requires a three-harmonic quadrature layout matching PCA rows; got layout rows %d and PCA rows %d for session %s', ...
            mode, layout.n_feat, n_feat, S.label);
    end
    [mu_pca, sd_pca, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local( ...
        X_delta, tr_idx, n_feat, shared_pca_scaler_scope);
    X_scaled = apply_fold_diagwhiten_local(X_delta, n_feat, mu_pca, sd_pca);
    X_pca_fit = apply_fold_diagwhiten_shared_pca_local(X_delta, n_feat, basis, mu_pca, sd_pca);
    filter_counts = allocate_harmonic_append_ssd_filter_counts_local( ...
        harmonic_ssd_append_filters, layout.n_harm, layout.n_ch);
    [basis_cell, ssd_info] = fit_harmonic_ssd_basis_from_lag_cells_local( ...
        X_scaled, tr_idx, layout, filter_counts);
    X_ssd_fit = apply_harmonic_csp_local(X_scaled, layout, basis_cell);
    X_delta_fit = concat_feature_cells_local(X_pca_fit, X_ssd_fit);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_shared_pca_plus_harmonic_ssd';
    feature_info.scaler_scope = scaler_scope_used;
    feature_info.feature_mu = mu_pca;
    feature_info.feature_sd = sd_pca;
    feature_info.n_feature_rows = n_feat;
    feature_info.n_pca_rows = K;
    feature_info.n_ssd_rows = first_nonempty_feature_rows_local(X_ssd_fit) / lag_count;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.filter_counts = filter_counts;
    feature_info.basis_cell = basis_cell;
    feature_info.ssd = ssd_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_shared_pca_plus_joint_harmonic_response')
    total_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    if total_feat ~= n_feat
        error('%s requires the same raw quadrature rows as the shared PCA basis; got total feature rows %d and PCA rows %d for session %s', ...
            mode, total_feat, n_feat, S.label);
    end
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    if layout.n_feat ~= n_feat
        error('%s requires a three-harmonic quadrature layout matching PCA rows; got layout rows %d and PCA rows %d for session %s', ...
            mode, layout.n_feat, n_feat, S.label);
    end
    [mu_pca, sd_pca, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local( ...
        X_delta, tr_idx, n_feat, shared_pca_scaler_scope);
    X_scaled = apply_fold_diagwhiten_local(X_delta, n_feat, mu_pca, sd_pca);
    X_pca_fit = apply_fold_diagwhiten_shared_pca_local(X_delta, n_feat, basis, mu_pca, sd_pca);
    K_response = min(max(1, round(double(harmonic_ssd_append_filters))), layout.n_feat);
    [response_basis, response_info] = fit_joint_harmonic_response_basis_from_lag_cells_local( ...
        X_scaled, Y_delta, tr_idx, layout, K_response);
    X_response_fit = apply_joint_harmonic_response_basis_local(X_scaled, layout, response_basis);
    X_delta_fit = concat_feature_cells_local(X_pca_fit, X_response_fit);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_shared_pca_plus_joint_harmonic_response';
    feature_info.scaler_scope = scaler_scope_used;
    feature_info.feature_mu = mu_pca;
    feature_info.feature_sd = sd_pca;
    feature_info.n_feature_rows = n_feat;
    feature_info.n_pca_rows = K;
    feature_info.n_response_rows = first_nonempty_feature_rows_local(X_response_fit) / lag_count;
    feature_info.n_total_rows = first_nonempty_feature_rows_local(X_delta_fit) / lag_count;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.K_response = K_response;
    feature_info.response_basis = response_basis;
    feature_info.response = response_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_shared_pca_harmonic_response_basis')
    total_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    if total_feat ~= n_feat
        error('%s requires the same raw quadrature rows as the shared PCA basis; got total feature rows %d and PCA rows %d for session %s', ...
            mode, total_feat, n_feat, S.label);
    end
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    if layout.n_feat ~= n_feat
        error('%s requires a three-harmonic quadrature layout matching PCA rows; got layout rows %d and PCA rows %d for session %s', ...
            mode, layout.n_feat, n_feat, S.label);
    end
    [mu_pca, sd_pca, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local( ...
        X_delta, tr_idx, n_feat, shared_pca_scaler_scope);
    X_scaled = apply_fold_diagwhiten_local(X_delta, n_feat, mu_pca, sd_pca);
    filter_counts = allocate_harmonic_append_ssd_filter_counts_local( ...
        harmonic_ssd_append_filters, layout.n_harm, layout.n_ch);
    n_response_rows = 2 * sum(filter_counts);
    if n_response_rows >= K
        error('%s requires response rows %d to be smaller than total feature budget %d for session %s', ...
            mode, n_response_rows, K, S.label);
    end
    pca_keep = K - n_response_rows;
    pca_basis = basis(:, 1:pca_keep);
    X_pca_fit = apply_fold_diagwhiten_shared_pca_local(X_delta, n_feat, pca_basis, mu_pca, sd_pca);
    [basis_cell, pls_info] = fit_harmonic_pls_basis_from_lag_cells_local( ...
        X_scaled, Y_delta, tr_idx, layout, filter_counts);
    X_response_fit = apply_harmonic_csp_local(X_scaled, layout, basis_cell);
    X_delta_fit = concat_feature_cells_local(X_pca_fit, X_response_fit);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_shared_pca_harmonic_response_basis';
    feature_info.scaler_scope = scaler_scope_used;
    feature_info.feature_mu = mu_pca;
    feature_info.feature_sd = sd_pca;
    feature_info.n_feature_rows = n_feat;
    feature_info.n_pca_rows = pca_keep;
    feature_info.n_response_rows = first_nonempty_feature_rows_local(X_response_fit) / lag_count;
    feature_info.n_total_rows = first_nonempty_feature_rows_local(X_delta_fit) / lag_count;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.filter_counts = filter_counts;
    feature_info.basis = pca_basis;
    feature_info.basis_cell = basis_cell;
    feature_info.pls = pls_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_shared_pca_orthcomp_response_basis')
    total_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    if total_feat ~= n_feat
        error('%s requires the same raw quadrature rows as the shared PCA basis; got total feature rows %d and PCA rows %d for session %s', ...
            mode, total_feat, n_feat, S.label);
    end
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    if layout.n_feat ~= n_feat
        error('%s requires a three-harmonic quadrature layout matching PCA rows; got layout rows %d and PCA rows %d for session %s', ...
            mode, layout.n_feat, n_feat, S.label);
    end
    [mu_pca, sd_pca, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local( ...
        X_delta, tr_idx, n_feat, shared_pca_scaler_scope);
    X_scaled = apply_fold_diagwhiten_local(X_delta, n_feat, mu_pca, sd_pca);
    filter_counts = allocate_harmonic_append_ssd_filter_counts_local( ...
        harmonic_ssd_append_filters, layout.n_harm, layout.n_ch);
    n_response_rows = 2 * sum(filter_counts);
    if n_response_rows >= K
        error('%s requires response rows %d to be smaller than total feature budget %d for session %s', ...
            mode, n_response_rows, K, S.label);
    end
    pca_keep = K - n_response_rows;
    pca_basis = basis(:, 1:pca_keep);
    X_pca_fit = apply_fold_diagwhiten_shared_pca_local(X_delta, n_feat, pca_basis, mu_pca, sd_pca);
    X_resid = remove_shared_pca_basis_from_lag_cells_local(X_scaled, n_feat, pca_basis);
    [basis_cell, pls_info] = fit_harmonic_pls_basis_from_lag_cells_local( ...
        X_resid, Y_delta, tr_idx, layout, filter_counts);
    X_response_fit = apply_harmonic_csp_local(X_resid, layout, basis_cell);
    X_delta_fit = concat_feature_cells_local(X_pca_fit, X_response_fit);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_shared_pca_orthcomp_response_basis';
    feature_info.scaler_scope = scaler_scope_used;
    feature_info.feature_mu = mu_pca;
    feature_info.feature_sd = sd_pca;
    feature_info.n_feature_rows = n_feat;
    feature_info.n_pca_rows = pca_keep;
    feature_info.n_response_rows = first_nonempty_feature_rows_local(X_response_fit) / lag_count;
    feature_info.n_total_rows = first_nonempty_feature_rows_local(X_delta_fit) / lag_count;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.filter_counts = filter_counts;
    feature_info.basis = pca_basis;
    feature_info.basis_cell = basis_cell;
    feature_info.pls = pls_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_shared_pca_plus_harmonic_spoc')
    total_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    if total_feat ~= n_feat
        error('%s requires the same raw quadrature rows as the shared PCA basis; got total feature rows %d and PCA rows %d for session %s', ...
            mode, total_feat, n_feat, S.label);
    end
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    if layout.n_feat ~= n_feat
        error('%s requires a three-harmonic quadrature layout matching PCA rows; got layout rows %d and PCA rows %d for session %s', ...
            mode, layout.n_feat, n_feat, S.label);
    end
    [mu_pca, sd_pca, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local( ...
        X_delta, tr_idx, n_feat, shared_pca_scaler_scope);
    X_scaled = apply_fold_diagwhiten_local(X_delta, n_feat, mu_pca, sd_pca);
    filter_counts = allocate_harmonic_append_ssd_filter_counts_local( ...
        harmonic_ssd_append_filters, layout.n_harm, layout.n_ch);
    n_response_rows = 2 * sum(filter_counts);
    if n_response_rows >= K
        error('%s requires response rows %d to be smaller than total feature budget %d for session %s', ...
            mode, n_response_rows, K, S.label);
    end
    pca_keep = K - n_response_rows;
    pca_basis = basis(:, 1:pca_keep);
    X_pca_fit = apply_fold_diagwhiten_shared_pca_local(X_delta, n_feat, pca_basis, mu_pca, sd_pca);
    X_resid = remove_shared_pca_basis_from_lag_cells_local(X_scaled, n_feat, pca_basis);
    [basis_cell, spoc_info] = fit_harmonic_spoc_basis_from_lag_cells_local( ...
        X_resid, Y_delta, tr_idx, layout, filter_counts);
    X_response_fit = apply_harmonic_csp_local(X_resid, layout, basis_cell);
    X_delta_fit = concat_feature_cells_local(X_pca_fit, X_response_fit);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_shared_pca_orthcomp_spoc';
    feature_info.scaler_scope = scaler_scope_used;
    feature_info.feature_mu = mu_pca;
    feature_info.feature_sd = sd_pca;
    feature_info.n_feature_rows = n_feat;
    feature_info.n_pca_rows = pca_keep;
    feature_info.n_response_rows = first_nonempty_feature_rows_local(X_response_fit) / lag_count;
    feature_info.n_total_rows = first_nonempty_feature_rows_local(X_delta_fit) / lag_count;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.filter_counts = filter_counts;
    feature_info.basis = pca_basis;
    feature_info.basis_cell = basis_cell;
    feature_info.spoc = spoc_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_shared_pca_plus_harmonic_pls')
    total_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    if total_feat ~= n_feat
        error('%s requires the same raw quadrature rows as the shared PCA basis; got total feature rows %d and PCA rows %d for session %s', ...
            mode, total_feat, n_feat, S.label);
    end
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    if layout.n_feat ~= n_feat
        error('%s requires a three-harmonic quadrature layout matching PCA rows; got layout rows %d and PCA rows %d for session %s', ...
            mode, layout.n_feat, n_feat, S.label);
    end
    [mu_pca, sd_pca, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local( ...
        X_delta, tr_idx, n_feat, shared_pca_scaler_scope);
    X_scaled = apply_fold_diagwhiten_local(X_delta, n_feat, mu_pca, sd_pca);
    X_pca_fit = apply_fold_diagwhiten_shared_pca_local(X_delta, n_feat, basis, mu_pca, sd_pca);
    filter_counts = allocate_harmonic_append_ssd_filter_counts_local( ...
        harmonic_ssd_append_filters, layout.n_harm, layout.n_ch);
    [basis_cell, pls_info] = fit_harmonic_pls_basis_from_lag_cells_local( ...
        X_scaled, Y_delta, tr_idx, layout, filter_counts);
    X_pls_fit = apply_harmonic_csp_local(X_scaled, layout, basis_cell);
    X_delta_fit = concat_feature_cells_local(X_pca_fit, X_pls_fit);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_shared_pca_plus_harmonic_pls';
    feature_info.scaler_scope = scaler_scope_used;
    feature_info.feature_mu = mu_pca;
    feature_info.feature_sd = sd_pca;
    feature_info.n_feature_rows = n_feat;
    feature_info.n_pca_rows = K;
    feature_info.n_pls_rows = first_nonempty_feature_rows_local(X_pls_fit) / lag_count;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.filter_counts = filter_counts;
    feature_info.basis_cell = basis_cell;
    feature_info.pls = pls_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_shared_pca_plus_harmonic_pls_resid')
    total_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    if total_feat ~= n_feat
        error('%s requires the same raw quadrature rows as the shared PCA basis; got total feature rows %d and PCA rows %d for session %s', ...
            mode, total_feat, n_feat, S.label);
    end
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    if layout.n_feat ~= n_feat
        error('%s requires a three-harmonic quadrature layout matching PCA rows; got layout rows %d and PCA rows %d for session %s', ...
            mode, layout.n_feat, n_feat, S.label);
    end
    [mu_pca, sd_pca, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local( ...
        X_delta, tr_idx, n_feat, shared_pca_scaler_scope);
    X_scaled = apply_fold_diagwhiten_local(X_delta, n_feat, mu_pca, sd_pca);
    X_pca_fit = apply_fold_diagwhiten_shared_pca_local(X_delta, n_feat, basis, mu_pca, sd_pca);
    X_resid = remove_shared_pca_basis_from_lag_cells_local(X_scaled, n_feat, basis);
    filter_counts = allocate_harmonic_append_ssd_filter_counts_local( ...
        harmonic_ssd_append_filters, layout.n_harm, layout.n_ch);
    [basis_cell, pls_info] = fit_harmonic_pls_basis_from_lag_cells_local( ...
        X_resid, Y_delta, tr_idx, layout, filter_counts);
    X_pls_fit = apply_harmonic_csp_local(X_resid, layout, basis_cell);
    X_delta_fit = concat_feature_cells_local(X_pca_fit, X_pls_fit);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_shared_pca_plus_harmonic_pls_resid';
    feature_info.scaler_scope = scaler_scope_used;
    feature_info.feature_mu = mu_pca;
    feature_info.feature_sd = sd_pca;
    feature_info.n_feature_rows = n_feat;
    feature_info.n_pca_rows = K;
    feature_info.n_pls_rows = first_nonempty_feature_rows_local(X_pls_fit) / lag_count;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.filter_counts = filter_counts;
    feature_info.basis_cell = basis_cell;
    feature_info.pls = pls_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_shared_pca_plus_harmonic_cca_resid')
    total_feat = infer_feature_rows_from_lag_count_local(X_delta, lag_count, S.label);
    if total_feat ~= n_feat
        error('%s requires the same raw quadrature rows as the shared PCA basis; got total feature rows %d and PCA rows %d for session %s', ...
            mode, total_feat, n_feat, S.label);
    end
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    if layout.n_feat ~= n_feat
        error('%s requires a three-harmonic quadrature layout matching PCA rows; got layout rows %d and PCA rows %d for session %s', ...
            mode, layout.n_feat, n_feat, S.label);
    end
    [mu_pca, sd_pca, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local( ...
        X_delta, tr_idx, n_feat, shared_pca_scaler_scope);
    X_scaled = apply_fold_diagwhiten_local(X_delta, n_feat, mu_pca, sd_pca);
    X_pca_fit = apply_fold_diagwhiten_shared_pca_local(X_delta, n_feat, basis, mu_pca, sd_pca);
    X_resid = remove_shared_pca_basis_from_lag_cells_local(X_scaled, n_feat, basis);
    filter_counts = allocate_harmonic_append_ssd_filter_counts_local( ...
        harmonic_ssd_append_filters, layout.n_harm, layout.n_ch);
    [basis_cell, cca_info] = fit_harmonic_cca_basis_from_lag_cells_local( ...
        X_resid, Y_delta, tr_idx, layout, filter_counts);
    X_cca_fit = apply_harmonic_csp_local(X_resid, layout, basis_cell);
    X_delta_fit = concat_feature_cells_local(X_pca_fit, X_cca_fit);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_shared_pca_plus_harmonic_cca_resid';
    feature_info.scaler_scope = scaler_scope_used;
    feature_info.feature_mu = mu_pca;
    feature_info.feature_sd = sd_pca;
    feature_info.n_feature_rows = n_feat;
    feature_info.n_pca_rows = K;
    feature_info.n_response_rows = first_nonempty_feature_rows_local(X_cca_fit) / lag_count;
    feature_info.n_harmonic = layout.n_harm;
    feature_info.n_channel = layout.n_ch;
    feature_info.filter_counts = filter_counts;
    feature_info.basis_cell = basis_cell;
    feature_info.cca = cca_info;
    return;
end
if strcmp(mode, 'delta_fold_diagwhiten_shared_pca_ea')
    layout = infer_three_harmonic_quadrature_layout_local(X_delta, lag_count, S.label);
    if layout.n_feat ~= n_feat
        error('%s requires a three-harmonic quadrature layout matching the EA PCA basis; got layout rows %d and PCA rows %d for session %s', ...
            mode, layout.n_feat, n_feat, S.label);
    end
    ea_gamma = 0.1;   % matches the EA basis build shrinkage
    W_ea = fit_ea_channel_whiten_from_lag_cells_local(X_delta, tr_idx, n_feat, layout.n_ch, ea_gamma);
    X_ea = apply_ea_channel_whiten_lag_cells_local(X_delta, n_feat, layout.n_ch, W_ea);
    [mu, sd, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local(X_ea, tr_idx, n_feat, shared_pca_scaler_scope);
    X_delta_fit = apply_fold_diagwhiten_shared_pca_local(X_ea, n_feat, basis, mu, sd);
    X_phase_fit = X_delta_fit;
    feature_info.transform = 'fold_diagwhiten_shared_pca_ea';
    feature_info.scaler_scope = scaler_scope_used;
    feature_info.feature_mu = mu;
    feature_info.feature_sd = sd;
    feature_info.n_feature_rows = n_feat;
    feature_info.n_channel = layout.n_ch;
    feature_info.ea_gamma = ea_gamma;
    return;
end
if mod(first_nonempty_feature_rows_local(X_delta), n_feat) ~= 0
    error('shared PCA rows %d do not divide lagged decoder feature rows %d for session %s', ...
        n_feat, first_nonempty_feature_rows_local(X_delta), S.label);
end
[mu, sd, scaler_scope_used] = shared_pca_scaler_from_lag_cells_local(X_delta, tr_idx, n_feat, shared_pca_scaler_scope);
X_delta_fit = apply_fold_diagwhiten_shared_pca_local(X_delta, n_feat, basis, mu, sd);
X_phase_fit = X_delta_fit;
feature_info.transform = 'fold_diagwhiten_shared_pca';
feature_info.scaler_scope = scaler_scope_used;
feature_info.feature_mu = mu;
feature_info.feature_sd = sd;
feature_info.n_feature_rows = n_feat;
end

function [X_delta_fit, X_phase_fit] = prepare_pooled_fold_feature_cells_local(X_delta, X_phase, Y_delta, pool_parts, tr_idx, feature_mode, shared_pca_file, shared_pca_components, phase_lo, phase_hi, lag_count, shared_pca_scaler_scope, harmonic_ssd_append_filters)
X_delta_fit = cell(size(X_delta));
X_phase_fit = cell(size(X_phase));
for pi = 1:numel(pool_parts)
    global_idx = pool_parts(pi).global_idx(:)';
    local_tr = find(ismember(global_idx, tr_idx));
    if isempty(local_tr)
        error('pooled fold has no train trials for session %s', pool_parts(pi).label);
    end
    part_X_delta = X_delta(global_idx);
    part_X_phase = X_phase(global_idx);
    part_Y_delta = Y_delta(global_idx);
    [Xd_part, Xp_part] = prepare_fold_feature_cells_local(part_X_delta, part_X_phase, ...
        part_Y_delta, pool_parts(pi).S, local_tr, feature_mode, shared_pca_file, ...
        shared_pca_components, phase_lo, phase_hi, lag_count, shared_pca_scaler_scope, harmonic_ssd_append_filters);
    for ii = 1:numel(global_idx)
        raw_cols = size(part_X_delta{ii}, 2);
        prep_cols = size(Xd_part{ii}, 2);
        if raw_cols ~= prep_cols
            error('pooled prepare changed sample count for %s local_trial=%d raw_cols=%d prepared_cols=%d', ...
                pool_parts(pi).label, ii, raw_cols, prep_cols);
        end
        X_delta_fit{global_idx(ii)} = Xd_part{ii};
        X_phase_fit{global_idx(ii)} = Xp_part{ii};
    end
end
end

function n_rows = first_nonempty_feature_rows_local(X_cell)
n_rows = 0;
for ii = 1:numel(X_cell)
    if ~isempty(X_cell{ii})
        n_rows = size(X_cell{ii}, 1);
        return;
    end
end
error('cannot infer feature rows from empty X_cell');
end

function n_feat = infer_feature_rows_from_lag_count_local(X_cell, lag_count, label)
total_rows = first_nonempty_feature_rows_local(X_cell);
lag_count = max(1, round(double(lag_count)));
if mod(total_rows, lag_count) ~= 0
    error('feature rows %d are not divisible by lag_count=%d for session %s', ...
        total_rows, lag_count, char(label));
end
n_feat = total_rows / lag_count;
end

function [mu, sd, scope_used] = shared_pca_scaler_from_lag_cells_local(X_cell, tr_idx, n_ch, scaler_scope)
scope_used = lower(char(scaler_scope));
switch scope_used
    case 'fold'
        scaler_idx = tr_idx;
    case 'session'
        scaler_idx = 1:numel(X_cell);
    otherwise
        error('shared_pca_scaler_scope must be fold or session, got %s', char(scaler_scope));
end
[mu, sd] = fold_channel_scaler_from_lag_cells_local(X_cell, scaler_idx, n_ch);
end

function layout = infer_three_harmonic_quadrature_layout_local(X_cell, lag_count, label)
n_feat = infer_feature_rows_from_lag_count_local(X_cell, lag_count, label);
n_harm = 3;
if mod(n_feat, 2 * n_harm) ~= 0
    error('three-harmonic quadrature CSP expects feature rows divisible by 2*3, got n_feat=%d for %s', ...
        n_feat, char(label));
end
n_ch = n_feat / (2 * n_harm);
layout = struct('n_feat', n_feat, 'n_harm', n_harm, 'n_ch', n_ch);
end

function pair_counts = allocate_harmonic_csp_pair_counts_local(shared_pca_components, n_harm, n_ch)
max_pairs_per_harm = max(1, floor(double(n_ch) / 2));
total_pairs = max(1, floor(double(shared_pca_components) / 4));
total_pairs = min(total_pairs, n_harm * max_pairs_per_harm);
pair_counts = zeros(1, n_harm);
for ii = 1:total_pairs
    h = mod(ii - 1, n_harm) + 1;
    pair_counts(h) = pair_counts(h) + 1;
end
pair_counts = min(pair_counts, max_pairs_per_harm);
end

function filter_counts = allocate_harmonic_ssd_filter_counts_local(shared_pca_components, n_harm, n_ch)
total_filters = max(1, floor(double(shared_pca_components) / 2));
total_filters = min(total_filters, n_harm * n_ch);
filter_counts = zeros(1, n_harm);
for ii = 1:total_filters
    h = mod(ii - 1, n_harm) + 1;
    if filter_counts(h) < n_ch
        filter_counts(h) = filter_counts(h) + 1;
    end
end
end

function filter_counts = allocate_harmonic_append_ssd_filter_counts_local(total_filters, n_harm, n_ch)
total_filters = max(1, round(double(total_filters)));
total_filters = min(total_filters, n_harm * n_ch);
filter_counts = zeros(1, n_harm);
for ii = 1:total_filters
    h = mod(ii - 1, n_harm) + 1;
    if filter_counts(h) < n_ch
        filter_counts(h) = filter_counts(h) + 1;
    end
end
end

function [basis_cell, info] = fit_harmonic_ssd_basis_from_lag_cells_local(X_cell, tr_idx, layout, filter_counts)
n_harm = layout.n_harm;
n_ch = layout.n_ch;
n_feat = layout.n_feat;
C_sig = cell(1, n_harm);
C_noise = cell(1, n_harm);
for h = 1:n_harm
    C_sig{h} = zeros(n_ch, n_ch);
    C_noise{h} = zeros(n_ch, n_ch);
end
n_sig = zeros(1, n_harm);
n_noise = zeros(1, n_harm);
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    if mod(size(X, 1), n_feat) ~= 0
        error('harmonic SSD feature rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    n_lag = size(X, 1) / n_feat;
    for li = 1:n_lag
        lag_offset = (li - 1) * n_feat;
        harm_samples = cell(1, n_harm);
        for h = 1:n_harm
            harm_offset = lag_offset + (h - 1) * 2 * n_ch;
            real_rows = harm_offset + (1:n_ch);
            imag_rows = harm_offset + n_ch + (1:n_ch);
            harm_samples{h} = [X(real_rows, :), X(imag_rows, :)];
        end
        for h = 1:n_harm
            [C_sig{h}, n_sig(h)] = add_trace_normalized_cov_samples_local( ...
                C_sig{h}, n_sig(h), harm_samples{h});
            noise = [];
            for hh = 1:n_harm
                if hh ~= h
                    noise = [noise, harm_samples{hh}]; %#ok<AGROW>
                end
            end
            [C_noise{h}, n_noise(h)] = add_trace_normalized_cov_samples_local( ...
                C_noise{h}, n_noise(h), noise);
        end
    end
end
basis_cell = cell(1, n_harm);
fallback = false(1, n_harm);
eigenvalues = cell(1, n_harm);
for h = 1:n_harm
    [basis_cell{h}, eigenvalues{h}, fallback(h)] = solve_ssd_basis_local( ...
        C_sig{h}, C_noise{h}, n_sig(h), n_noise(h), max(1, filter_counts(h)));
end
info = struct('noise_reference', 'other_harmonics', ...
    'filter_counts', filter_counts, 'n_signal_blocks', n_sig, ...
    'n_noise_blocks', n_noise, 'fallback', fallback, ...
    'eigenvalues_by_harmonic', {eigenvalues});
end

function [basis, eigenvalues_sel, fallback] = solve_ssd_basis_local(C_sig, C_noise, n_sig, n_noise, n_filters)
n_ch = size(C_sig, 1);
K = min(max(1, round(double(n_filters))), n_ch);
if n_sig < 2 || n_noise < 2
    basis = zeros(n_ch, K);
    basis(1:K, 1:K) = eye(K);
    eigenvalues_sel = nan(K, 1);
    fallback = true;
    return;
end
C_sig = C_sig / n_sig;
C_noise = C_noise / n_noise;
C_sig = (C_sig + C_sig') / 2;
C_noise = (C_noise + C_noise') / 2;
avg_noise = trace(C_noise) / max(n_ch, 1);
reg = max(avg_noise * 1e-4, eps);
try
    [V, D] = eig(C_sig, C_noise + reg * eye(n_ch));
    d = real(diag(D));
catch
    [V, D] = eig(pinv(C_noise + reg * eye(n_ch)) * C_sig);
    d = real(diag(D));
end
[~, ord] = sort(d, 'descend');
sel = ord(1:min(K, numel(ord)));
basis = real(V(:, sel));
for ci = 1:size(basis, 2)
    s = norm(basis(:, ci));
    if isfinite(s) && s > eps
        basis(:, ci) = basis(:, ci) ./ s;
    end
end
basis(~isfinite(basis)) = 0;
eigenvalues_sel = d(sel);
fallback = false;
end

function [basis_cell, info] = fit_harmonic_spoc_basis_from_lag_cells_local(X_cell, Y_cell, tr_idx, layout, filter_counts)
% Per-harmonic SPoC (Dahne et al. 2014): spatial filters maximizing
% gait-target-covarying band power relative to total band power. Supervised by
% the OUTER-TRAIN target only; the scalar target is the first principal
% component of the z-scored multi-joint gait target. Fit on the column-stacked
% real/imag quadrature samples, so each per-harmonic basis is n_ch x K and is
% consumed verbatim by apply_harmonic_csp_local (same contract as SSD/PLS).
n_harm = layout.n_harm;
n_ch = layout.n_ch;
n_feat = layout.n_feat;
C_pow = cell(1, n_harm);   % plain total band-power covariance (GEVD denominator)
C_z = cell(1, n_harm);     % target-weighted covariance (GEVD numerator)
for h = 1:n_harm
    C_pow{h} = zeros(n_ch, n_ch);
    C_z{h} = zeros(n_ch, n_ch);
end
n_blocks = zeros(1, n_harm);
for ii = 1:numel(tr_idx)
    k = tr_idx(ii);
    X = double(X_cell{k});
    Y = double(Y_cell{k});
    if isempty(X) || isempty(Y), continue; end
    if mod(size(X, 1), n_feat) ~= 0
        error('harmonic SPoC feature rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    T = min(size(X, 2), size(Y, 1));
    if T < 10, continue; end
    Yc = prepare_target_matrix_for_pls_local(Y(1:T, :));
    if isempty(Yc), continue; end
    z = spoc_target_pc1_local(Yc);          % T x 1 scalar target
    if isempty(z), continue; end
    zdup = [z; z];                          % 2T x 1, aligns to [real, imag] columns
    n_lag = size(X, 1) / n_feat;
    for li = 1:n_lag
        lag_offset = (li - 1) * n_feat;
        for h = 1:n_harm
            harm_offset = lag_offset + (h - 1) * 2 * n_ch;
            real_rows = harm_offset + (1:n_ch);
            imag_rows = harm_offset + n_ch + (1:n_ch);
            Xh = [X(real_rows, 1:T), X(imag_rows, 1:T)];
            Xh = center_feature_matrix_for_pls_local(Xh);   % n_ch x 2T, row-centered
            if isempty(Xh), continue; end
            denom = max(1, size(Xh, 2) - 1);
            Cb = (Xh * Xh') / denom;                        % plain block covariance
            tr_b = trace(Cb) / n_ch;
            if ~isfinite(tr_b) || tr_b <= eps, continue; end
            Czb = (Xh .* (zdup')) * Xh' / denom;            % target-weighted block covariance
            C_pow{h} = C_pow{h} + Cb / tr_b;                % per-block trace normalized
            C_z{h} = C_z{h} + Czb / tr_b;                   % same scale as C_pow
            n_blocks(h) = n_blocks(h) + 1;
        end
    end
end
basis_cell = cell(1, n_harm);
eigenvalues = cell(1, n_harm);
fallback = false(1, n_harm);
for h = 1:n_harm
    [basis_cell{h}, eigenvalues{h}, fallback(h)] = solve_spoc_basis_local( ...
        C_z{h}, C_pow{h}, n_blocks(h), max(1, filter_counts(h)));
end
info = struct('fit_rule', 'train_fold_spoc_target_power_gevd', ...
    'target_mode', 'targetpc1', 'filter_counts', filter_counts, ...
    'n_blocks', n_blocks, 'fallback', fallback, ...
    'eigenvalues_by_harmonic', {eigenvalues});
end

function z = spoc_target_pc1_local(Yc)
% First principal component score of the z-scored multi-joint target (T x J -> T x 1).
if isempty(Yc) || size(Yc, 1) < 2
    z = [];
    return;
end
Yc(~isfinite(Yc)) = 0;
try
    [~, ~, V] = svd(Yc, 'econ');
    z = Yc * V(:, 1);
catch
    z = Yc(:, 1);
end
z = z - mean(z);
z(~isfinite(z)) = 0;
end

function [basis, eigenvalues_sel, fallback] = solve_spoc_basis_local(C_z, C_pow, n_blocks, n_filters)
% Generalized eigenproblem eig(C_z, C_pow): directions whose band power covaries
% most (positively or negatively) with the gait target. Keep abs-largest
% eigenvalues so both target-power-up and target-power-down filters are retained.
n_ch = size(C_pow, 1);
K = min(max(1, round(double(n_filters))), n_ch);
if n_blocks < 2
    basis = zeros(n_ch, K);
    basis(1:K, 1:K) = eye(K);
    eigenvalues_sel = nan(K, 1);
    fallback = true;
    return;
end
C_z = C_z / n_blocks;
C_pow = C_pow / n_blocks;
C_z = (C_z + C_z') / 2;
C_pow = (C_pow + C_pow') / 2;
avg_pow = trace(C_pow) / max(n_ch, 1);
reg = max(avg_pow * 1e-4, eps);
try
    [V, D] = eig(C_z, C_pow + reg * eye(n_ch));
    d = real(diag(D));
catch
    [V, D] = eig(pinv(C_pow + reg * eye(n_ch)) * C_z);
    d = real(diag(D));
end
[~, ord] = sort(abs(d), 'descend');
sel = ord(1:min(K, numel(ord)));
basis = real(V(:, sel));
for ci = 1:size(basis, 2)
    s = norm(basis(:, ci));
    if isfinite(s) && s > eps
        basis(:, ci) = basis(:, ci) ./ s;
    end
end
basis(~isfinite(basis)) = 0;
eigenvalues_sel = d(sel);
fallback = false;
end

function [basis_cell, info] = fit_harmonic_block_pca_basis_from_lag_cells_local(X_cell, tr_idx, layout, filter_counts)
n_harm = layout.n_harm;
n_ch = layout.n_ch;
n_feat = layout.n_feat;
C_sig = cell(1, n_harm);
for h = 1:n_harm
    C_sig{h} = zeros(n_ch, n_ch);
end
n_blocks = zeros(1, n_harm);
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    if mod(size(X, 1), n_feat) ~= 0
        error('harmonic block PCA feature rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    n_lag = size(X, 1) / n_feat;
    for li = 1:n_lag
        lag_offset = (li - 1) * n_feat;
        for h = 1:n_harm
            harm_offset = lag_offset + (h - 1) * 2 * n_ch;
            real_rows = harm_offset + (1:n_ch);
            imag_rows = harm_offset + n_ch + (1:n_ch);
            B = [X(real_rows, :), X(imag_rows, :)];
            [C_sig{h}, n_blocks(h)] = add_trace_normalized_cov_samples_local( ...
                C_sig{h}, n_blocks(h), B);
        end
    end
end
basis_cell = cell(1, n_harm);
eigenvalues = cell(1, n_harm);
fallback = false(1, n_harm);
for h = 1:n_harm
    [basis_cell{h}, eigenvalues{h}, fallback(h)] = solve_harmonic_block_pca_basis_local( ...
        C_sig{h}, n_blocks(h), max(1, filter_counts(h)));
end
info = struct('fit_rule', 'train_fold_harmonic_block_pca', ...
    'filter_counts', filter_counts, 'n_blocks', n_blocks, ...
    'fallback', fallback, 'eigenvalues_by_harmonic', {eigenvalues});
end

function [basis, eigenvalues_sel, fallback] = solve_harmonic_block_pca_basis_local(C_sig, n_sig, n_filters)
n_ch = size(C_sig, 1);
K = min(max(1, round(double(n_filters))), n_ch);
if n_sig < 2 || ~any(isfinite(C_sig(:)))
    basis = zeros(n_ch, K);
    basis(1:K, 1:K) = eye(K);
    eigenvalues_sel = nan(K, 1);
    fallback = true;
    return;
end
C_sig = C_sig / n_sig;
C_sig = (C_sig + C_sig') / 2;
avg_var = trace(C_sig) / max(n_ch, 1);
reg = max(avg_var * 1e-8, eps);
try
    [V, D] = eig(C_sig + reg * eye(n_ch));
    d = real(diag(D));
catch
    [V, D] = eig((C_sig + reg * eye(n_ch) + C_sig') / 2);
    d = real(diag(D));
end
[~, ord] = sort(d, 'descend');
sel = ord(1:min(K, numel(ord)));
basis = real(V(:, sel));
for ci = 1:size(basis, 2)
    s = norm(basis(:, ci));
    if isfinite(s) && s > eps
        basis(:, ci) = basis(:, ci) ./ s;
    end
end
basis(~isfinite(basis)) = 0;
eigenvalues_sel = d(sel);
fallback = false;
end

function component_counts = allocate_harmonic_real_component_counts_local(total_components, n_harm, max_per_harm)
total_components = max(1, round(double(total_components)));
total_components = min(total_components, n_harm * max_per_harm);
component_counts = zeros(1, n_harm);
for ii = 1:total_components
    h = mod(ii - 1, n_harm) + 1;
    if component_counts(h) < max_per_harm
        component_counts(h) = component_counts(h) + 1;
    end
end
end

function [basis, info] = fit_joint_complex_pca_basis_from_lag_cells_local(X_cell, tr_idx, layout, K_complex)
n_complex = layout.n_harm * layout.n_ch;
C_sig = complex(zeros(n_complex, n_complex), zeros(n_complex, n_complex));
n_blocks = 0;
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    if mod(size(X, 1), layout.n_feat) ~= 0
        error('joint complex PCA feature rows %d are not divisible by n_feat=%d', size(X, 1), layout.n_feat);
    end
    n_lag = size(X, 1) / layout.n_feat;
    for li = 1:n_lag
        lag_offset = (li - 1) * layout.n_feat;
        Z = harmonic_complex_lag_matrix_local(X, layout, lag_offset);
        [C_sig, n_blocks] = add_trace_normalized_complex_cov_samples_local(C_sig, n_blocks, Z);
    end
end
[basis, eigenvalues, fallback] = solve_complex_pca_basis_local(C_sig, n_blocks, K_complex);
info = struct('fit_rule', 'train_fold_joint_complex_harmonic_pca', ...
    'K_complex', K_complex, 'n_blocks', n_blocks, ...
    'fallback', fallback, 'eigenvalues', eigenvalues);
end

function Z = harmonic_complex_lag_matrix_local(X, layout, lag_offset)
Z = complex(zeros(layout.n_harm * layout.n_ch, size(X, 2)), ...
    zeros(layout.n_harm * layout.n_ch, size(X, 2)));
out = 0;
for h = 1:layout.n_harm
    harm_offset = lag_offset + (h - 1) * 2 * layout.n_ch;
    real_rows = harm_offset + (1:layout.n_ch);
    imag_rows = harm_offset + layout.n_ch + (1:layout.n_ch);
    rows_out = out + (1:layout.n_ch);
    R = X(real_rows, :);
    I = X(imag_rows, :);
    R(~isfinite(R)) = 0;
    I(~isfinite(I)) = 0;
    Z(rows_out, :) = complex(R, I);
    out = out + layout.n_ch;
end
end

function [C_acc, n_acc] = add_trace_normalized_complex_cov_samples_local(C_acc, n_acc, Z)
if isempty(Z) || size(Z, 2) < 5
    return;
end
ok = all(isfinite(real(Z)), 1) & all(isfinite(imag(Z)), 1);
B = Z(:, ok);
if size(B, 2) < 5
    return;
end
B = B - mean(B, 2);
B(~isfinite(real(B)) | ~isfinite(imag(B))) = 0;
Ci = (B * B') / max(size(B, 2) - 1, 1);
tr = real(trace(Ci));
if ~(isfinite(tr) && tr > eps)
    return;
end
C_acc = C_acc + Ci / tr;
n_acc = n_acc + 1;
end

function [basis, eigenvalues_sel, fallback] = solve_complex_pca_basis_local(C_sig, n_sig, K_complex)
n_feat = size(C_sig, 1);
K = min(max(1, round(double(K_complex))), n_feat);
if n_sig < 2 || ~any(isfinite(real(C_sig(:)))) || ~any(isfinite(imag(C_sig(:))))
    basis = complex(zeros(n_feat, K), zeros(n_feat, K));
    basis(1:K, 1:K) = eye(K);
    eigenvalues_sel = nan(K, 1);
    fallback = true;
    return;
end
C_sig = C_sig / n_sig;
C_sig = (C_sig + C_sig') / 2;
avg_var = real(trace(C_sig)) / max(n_feat, 1);
reg = max(avg_var * 1e-8, eps);
try
    [V, D] = eig(C_sig + reg * eye(n_feat));
    d = real(diag(D));
catch
    [V, D] = eig(full(C_sig + reg * eye(n_feat)));
    d = real(diag(D));
end
[~, ord] = sort(d, 'descend');
sel = ord(1:min(K, numel(ord)));
basis = V(:, sel);
for ci = 1:size(basis, 2)
    s = norm(basis(:, ci));
    if isfinite(s) && s > eps
        basis(:, ci) = basis(:, ci) ./ s;
    end
end
basis(~isfinite(real(basis)) | ~isfinite(imag(basis))) = 0;
eigenvalues_sel = d(sel);
fallback = false;
end

function X_out = apply_joint_complex_pca_local(X_cell, layout, basis)
K = size(basis, 2);
rows_per_lag = 2 * K;
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if mod(size(X, 1), layout.n_feat) ~= 0
        error('joint complex PCA apply rows %d are not divisible by n_feat=%d', size(X, 1), layout.n_feat);
    end
    n_lag = size(X, 1) / layout.n_feat;
    Z_out = zeros(rows_per_lag * n_lag, size(X, 2));
    for li = 1:n_lag
        lag_offset = (li - 1) * layout.n_feat;
        Z = harmonic_complex_lag_matrix_local(X, layout, lag_offset);
        S = basis' * Z;
        out_rows = (li - 1) * rows_per_lag + (1:rows_per_lag);
        Z_out(out_rows, :) = [real(S); imag(S)];
    end
    X_out{ii} = Z_out;
end
end

function [basis_cell, info] = fit_harmonic_block_free_ri_pca_basis_from_lag_cells_local(X_cell, tr_idx, layout, component_counts)
n_harm = layout.n_harm;
n_block = 2 * layout.n_ch;
C_sig = cell(1, n_harm);
for h = 1:n_harm
    C_sig{h} = zeros(n_block, n_block);
end
n_blocks = zeros(1, n_harm);
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    if mod(size(X, 1), layout.n_feat) ~= 0
        error('harmonic block free-RI PCA feature rows %d are not divisible by n_feat=%d', size(X, 1), layout.n_feat);
    end
    n_lag = size(X, 1) / layout.n_feat;
    for li = 1:n_lag
        lag_offset = (li - 1) * layout.n_feat;
        for h = 1:n_harm
            harm_offset = lag_offset + (h - 1) * 2 * layout.n_ch;
            real_rows = harm_offset + (1:layout.n_ch);
            imag_rows = harm_offset + layout.n_ch + (1:layout.n_ch);
            B = [X(real_rows, :); X(imag_rows, :)];
            [C_sig{h}, n_blocks(h)] = add_trace_normalized_cov_samples_local( ...
                C_sig{h}, n_blocks(h), B);
        end
    end
end
basis_cell = cell(1, n_harm);
eigenvalues = cell(1, n_harm);
fallback = false(1, n_harm);
for h = 1:n_harm
    [basis_cell{h}, eigenvalues{h}, fallback(h)] = solve_harmonic_block_pca_basis_local( ...
        C_sig{h}, n_blocks(h), max(1, component_counts(h)));
end
info = struct('fit_rule', 'train_fold_harmonic_block_free_ri_pca', ...
    'component_counts', component_counts, 'n_blocks', n_blocks, ...
    'fallback', fallback, 'eigenvalues_by_harmonic', {eigenvalues});
end

function X_out = apply_harmonic_block_free_ri_pca_local(X_cell, layout, basis_cell)
rows_per_lag = 0;
for h = 1:layout.n_harm
    rows_per_lag = rows_per_lag + size(basis_cell{h}, 2);
end
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if mod(size(X, 1), layout.n_feat) ~= 0
        error('harmonic block free-RI PCA apply rows %d are not divisible by n_feat=%d', size(X, 1), layout.n_feat);
    end
    n_lag = size(X, 1) / layout.n_feat;
    Z = zeros(rows_per_lag * n_lag, size(X, 2));
    for li = 1:n_lag
        lag_in = (li - 1) * layout.n_feat;
        lag_out = (li - 1) * rows_per_lag;
        out_cursor = 0;
        for h = 1:layout.n_harm
            W = basis_cell{h};
            K = size(W, 2);
            harm_offset = lag_in + (h - 1) * 2 * layout.n_ch;
            real_rows = harm_offset + (1:layout.n_ch);
            imag_rows = harm_offset + layout.n_ch + (1:layout.n_ch);
            B = [X(real_rows, :); X(imag_rows, :)];
            B(~isfinite(B)) = 0;
            out_rows = lag_out + out_cursor + (1:K);
            Z(out_rows, :) = W' * B;
            out_cursor = out_cursor + K;
        end
    end
    X_out{ii} = Z;
end
end

function [basis_cell, info] = fit_harmonic_pls_basis_from_lag_cells_local(X_cell, Y_cell, tr_idx, layout, filter_counts)
n_harm = layout.n_harm;
n_ch = layout.n_ch;
n_feat = layout.n_feat;
Cxy = cell(1, n_harm);
for h = 1:n_harm
    Cxy{h} = [];
end
n_blocks = zeros(1, n_harm);
n_samples = zeros(1, n_harm);
for ii = 1:numel(tr_idx)
    k = tr_idx(ii);
    X = double(X_cell{k});
    Y = double(Y_cell{k});
    if isempty(X) || isempty(Y), continue; end
    if mod(size(X, 1), n_feat) ~= 0
        error('harmonic PLS feature rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    T = min(size(X, 2), size(Y, 1));
    if T < 10, continue; end
    Yc = prepare_target_matrix_for_pls_local(Y(1:T, :));
    if isempty(Yc), continue; end
    n_lag = size(X, 1) / n_feat;
    for li = 1:n_lag
        lag_offset = (li - 1) * n_feat;
        for h = 1:n_harm
            harm_offset = lag_offset + (h - 1) * 2 * n_ch;
            real_rows = harm_offset + (1:n_ch);
            imag_rows = harm_offset + n_ch + (1:n_ch);
            Xh = [X(real_rows, 1:T), X(imag_rows, 1:T)];
            Xh = center_feature_matrix_for_pls_local(Xh);
            if isempty(Xh), continue; end
            Ydup = [Yc; Yc];
            denom = max(1, size(Ydup, 1) - 1);
            block = Xh * Ydup / denom;
            if isempty(Cxy{h})
                Cxy{h} = zeros(size(block));
            end
            Cxy{h} = Cxy{h} + block;
            n_blocks(h) = n_blocks(h) + 1;
            n_samples(h) = n_samples(h) + size(Ydup, 1);
        end
    end
end
basis_cell = cell(1, n_harm);
singular_values = cell(1, n_harm);
Cxy_norm = cell(1, n_harm);
fallback = false(1, n_harm);
for h = 1:n_harm
    K = min(max(1, round(double(filter_counts(h)))), n_ch);
    if isempty(Cxy{h}) || n_blocks(h) < 1 || ~any(isfinite(Cxy{h}(:)))
        basis_cell{h} = zeros(n_ch, K);
        basis_cell{h}(1:K, 1:K) = eye(K);
        singular_values{h} = nan(K, 1);
        Cxy_norm{h} = [];
        fallback(h) = true;
        continue;
    end
    M = Cxy{h} / max(1, n_blocks(h));
    M(~isfinite(M)) = 0;
    Cxy_norm{h} = M; % normalized channel-by-target cross-cov, for cross-session pooling
    try
        [U, Ssv, ~] = svd(M, 'econ');
        s = diag(Ssv);
    catch
        [U, Ssv, ~] = svd(M);
        s = diag(Ssv);
    end
    K = min(K, size(U, 2));
    basis = real(U(:, 1:K));
    for ci = 1:size(basis, 2)
        bnorm = norm(basis(:, ci));
        if isfinite(bnorm) && bnorm > eps
            basis(:, ci) = basis(:, ci) ./ bnorm;
        end
    end
    basis(~isfinite(basis)) = 0;
    basis_cell{h} = basis;
    singular_values{h} = s(1:min(K, numel(s)));
end
info = struct('fit_rule', 'train_fold_channel_target_crosscov_svd', ...
    'filter_counts', filter_counts, 'n_blocks', n_blocks, ...
    'n_samples', n_samples, 'fallback', fallback, ...
    'singular_values_by_harmonic', {singular_values}, ...
    'Cxy_by_harmonic', {Cxy_norm});
end

function W = ea_inv_sqrt_local(R)
% Symmetric inverse square root of a (regularized) covariance, via eigendecomp.
R = (R + R') / 2;
R(~isfinite(R)) = 0;
n = size(R, 1);
[V, D] = eig(R, 'vector');
d = real(D);
floor_d = max(eps, 1e-10 * max(d));
d(~isfinite(d) | d < floor_d) = floor_d;
W = V * diag(d .^ (-0.5)) * V';
W = real((W + W') / 2);
W(~isfinite(W)) = 0;
if isempty(W) || ~all(isfinite(W(:))), W = eye(n); end
end

function [basis_cell, info] = fit_harmonic_cca_basis_from_lag_cells_local(X_cell, Y_cell, tr_idx, layout, filter_counts)
% Per-harmonic CCA spatial filters: directions of maximal CORRELATION (not raw
% covariance) between the harmonic quadrature EEG and the z-scored multi-joint
% target. Whitens EEG (shrinkage) and target before the cross-covariance SVD, so
% low-variance-but-high-correlation residual directions are recovered (vs raw
% cross-cov SVD which is variance-biased). Train-fold only; frozen for held-out.
n_harm = layout.n_harm; n_ch = layout.n_ch; n_feat = layout.n_feat;
gamma = 0.1;
Cxx = cell(1, n_harm); Cxy = cell(1, n_harm);
for h = 1:n_harm, Cxx{h} = zeros(n_ch, n_ch); Cxy{h} = []; end
Cyy = []; nb = zeros(1, n_harm); nyy = 0;
for ii = 1:numel(tr_idx)
    k = tr_idx(ii); X = double(X_cell{k}); Y = double(Y_cell{k});
    if isempty(X) || isempty(Y), continue; end
    if mod(size(X, 1), n_feat) ~= 0
        error('harmonic CCA feature rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    T = min(size(X, 2), size(Y, 1)); if T < 10, continue; end
    Yc = prepare_target_matrix_for_pls_local(Y(1:T, :)); if isempty(Yc), continue; end
    Yd = [Yc; Yc]; d = max(1, size(Yd, 1) - 1);
    if isempty(Cyy), Cyy = zeros(size(Yd, 2)); end
    Cyy = Cyy + (Yd' * Yd) / d; nyy = nyy + 1;
    n_lag = size(X, 1) / n_feat;
    for li = 1:n_lag
        lo = (li - 1) * n_feat;
        for h = 1:n_harm
            ho = lo + (h - 1) * 2 * n_ch;
            Xh = [X(ho + (1:n_ch), 1:T), X(ho + n_ch + (1:n_ch), 1:T)];
            Xh = center_feature_matrix_for_pls_local(Xh);
            if isempty(Xh), continue; end
            Cxx{h} = Cxx{h} + (Xh * Xh') / d;
            blk = Xh * Yd / d;
            if isempty(Cxy{h}), Cxy{h} = zeros(size(blk)); end
            Cxy{h} = Cxy{h} + blk; nb(h) = nb(h) + 1;
        end
    end
end
basis_cell = cell(1, n_harm); fallback = false(1, n_harm); sv = cell(1, n_harm);
if isempty(Cyy) || nyy < 1
    for h = 1:n_harm
        K = min(max(1, round(double(filter_counts(h)))), n_ch);
        basis_cell{h} = zeros(n_ch, K); basis_cell{h}(1:K, 1:K) = eye(K); sv{h} = nan(K, 1); fallback(h) = true;
    end
    info = struct('fit_rule', 'train_fold_harmonic_cca_whitened_crosscov_svd', 'filter_counts', filter_counts, 'n_blocks', nb, 'fallback', fallback, 'singular_values_by_harmonic', {sv}, 'gamma', gamma);
    return;
end
Cyy = Cyy / max(1, nyy); Cyy = (Cyy + Cyy') / 2;
Wy = ea_inv_sqrt_local((1 - gamma) * Cyy + gamma * (trace(Cyy) / size(Cyy, 1)) * eye(size(Cyy, 1)));
for h = 1:n_harm
    K = min(max(1, round(double(filter_counts(h)))), n_ch);
    if nb(h) < 1 || isempty(Cxy{h}) || ~any(isfinite(Cxy{h}(:)))
        basis_cell{h} = zeros(n_ch, K); basis_cell{h}(1:K, 1:K) = eye(K); sv{h} = nan(K, 1); fallback(h) = true; continue;
    end
    Cx = Cxx{h} / max(1, nb(h)); Cx = (Cx + Cx') / 2;
    Cx = (1 - gamma) * Cx + gamma * (trace(Cx) / n_ch) * eye(n_ch);
    Wx = ea_inv_sqrt_local(Cx);
    M = Wx * (Cxy{h} / max(1, nb(h))) * Wy; M(~isfinite(M)) = 0;
    [U, S, ~] = svd(M, 'econ'); s = diag(S);
    Kk = min(K, size(U, 2));
    B = Wx * U(:, 1:Kk);
    for ci = 1:size(B, 2)
        bn = norm(B(:, ci)); if isfinite(bn) && bn > eps, B(:, ci) = B(:, ci) / bn; end
    end
    B(~isfinite(B)) = 0; basis_cell{h} = B; sv{h} = s(1:min(Kk, numel(s)));
end
info = struct('fit_rule', 'train_fold_harmonic_cca_whitened_crosscov_svd', 'filter_counts', filter_counts, 'n_blocks', nb, 'fallback', fallback, 'singular_values_by_harmonic', {sv}, 'gamma', gamma);
end

function [basis, info] = fit_joint_harmonic_response_basis_from_lag_cells_local(X_cell, Y_cell, tr_idx, layout, K)
n_feat = layout.n_feat;
K = min(max(1, round(double(K))), n_feat);
Cxy = [];
n_blocks = 0;
n_samples = 0;
for ii = 1:numel(tr_idx)
    k = tr_idx(ii);
    X = double(X_cell{k});
    Y = double(Y_cell{k});
    if isempty(X) || isempty(Y), continue; end
    if mod(size(X, 1), n_feat) ~= 0
        error('joint harmonic response feature rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    T = min(size(X, 2), size(Y, 1));
    if T < 10, continue; end
    Yc = prepare_target_matrix_for_pls_local(Y(1:T, :));
    if isempty(Yc), continue; end
    n_lag = size(X, 1) / n_feat;
    for li = 1:n_lag
        rows = (li - 1) * n_feat + (1:n_feat);
        Xh = center_feature_matrix_for_pls_local(X(rows, 1:T));
        if isempty(Xh), continue; end
        denom = max(1, size(Yc, 1) - 1);
        block = Xh * Yc / denom;
        if isempty(Cxy)
            Cxy = zeros(size(block));
        end
        Cxy = Cxy + block;
        n_blocks = n_blocks + 1;
        n_samples = n_samples + size(Yc, 1);
    end
end
fallback = false;
if isempty(Cxy) || n_blocks < 1 || ~any(isfinite(Cxy(:)))
    basis = zeros(n_feat, K);
    basis(1:K, 1:K) = eye(K);
    singular_values = nan(K, 1);
    fallback = true;
else
    M = Cxy / max(1, n_blocks);
    M(~isfinite(M)) = 0;
    try
        [U, Ssv, ~] = svd(M, 'econ');
        s = diag(Ssv);
    catch
        [U, Ssv, ~] = svd(M);
        s = diag(Ssv);
    end
    K = min(K, size(U, 2));
    basis = real(U(:, 1:K));
    for ci = 1:size(basis, 2)
        bnorm = norm(basis(:, ci));
        if isfinite(bnorm) && bnorm > eps
            basis(:, ci) = basis(:, ci) ./ bnorm;
        end
    end
    basis(~isfinite(basis)) = 0;
    singular_values = s(1:min(K, numel(s)));
end
info = struct('fit_rule', 'train_fold_joint_harmonic_target_crosscov_svd', ...
    'K_response', K, 'n_blocks', n_blocks, 'n_samples', n_samples, ...
    'fallback', fallback, 'singular_values', singular_values);
end

function X_out = apply_joint_harmonic_response_basis_local(X_cell, layout, basis)
n_feat = layout.n_feat;
K = size(basis, 2);
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if mod(size(X, 1), n_feat) ~= 0
        error('joint harmonic response apply rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    n_lag = size(X, 1) / n_feat;
    Z = zeros(K * n_lag, size(X, 2));
    for li = 1:n_lag
        in_rows = (li - 1) * n_feat + (1:n_feat);
        out_rows = (li - 1) * K + (1:K);
        B = X(in_rows, :);
        B(~isfinite(B)) = 0;
        Z(out_rows, :) = basis' * B;
    end
    X_out{ii} = Z;
end
end

function X_out = remove_harmonic_response_basis_from_lag_cells_local(X_cell, layout, basis_cell)
n_harm = layout.n_harm;
n_ch = layout.n_ch;
n_feat = layout.n_feat;
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if mod(size(X, 1), n_feat) ~= 0
        error('harmonic response residual rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    n_lag = size(X, 1) / n_feat;
    Z = X;
    for li = 1:n_lag
        lag_offset = (li - 1) * n_feat;
        for h = 1:n_harm
            W = basis_cell{h};
            if isempty(W), continue; end
            harm_offset = lag_offset + (h - 1) * 2 * n_ch;
            real_rows = harm_offset + (1:n_ch);
            imag_rows = harm_offset + n_ch + (1:n_ch);
            R = X(real_rows, :);
            I = X(imag_rows, :);
            R(~isfinite(R)) = 0;
            I(~isfinite(I)) = 0;
            Z(real_rows, :) = R - W * (W' * R);
            Z(imag_rows, :) = I - W * (W' * I);
        end
    end
    Z(~isfinite(Z)) = 0;
    X_out{ii} = Z;
end
end

function [basis, info] = fit_feature_pca_basis_from_lag_cells_local(X_cell, tr_idx, n_feat, K)
K = min(max(1, round(double(K))), n_feat);
sum_x = zeros(n_feat, 1);
n_obs = 0;
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    if mod(size(X, 1), n_feat) ~= 0
        error('feature PCA rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    n_lag = size(X, 1) / n_feat;
    for li = 1:n_lag
        blk = X((li-1)*n_feat+1:li*n_feat, :);
        ok = all(isfinite(blk), 1);
        if any(ok)
            sum_x = sum_x + sum(blk(:, ok), 2);
            n_obs = n_obs + sum(ok);
        end
    end
end
if n_obs < K + 5
    basis = zeros(n_feat, K);
    basis(1:K, 1:K) = eye(K);
    info = struct('fit_rule', 'train_fold_residual_feature_pca', ...
        'n_obs', n_obs, 'fallback', true, 'eigenvalues', nan(K, 1));
    return;
end
mu = sum_x ./ max(1, n_obs);
C = zeros(n_feat, n_feat);
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    n_lag = size(X, 1) / n_feat;
    for li = 1:n_lag
        blk = X((li-1)*n_feat+1:li*n_feat, :);
        ok = all(isfinite(blk), 1);
        if any(ok)
            Z = blk(:, ok) - mu;
            C = C + Z * Z';
        end
    end
end
C = C ./ max(n_obs - 1, 1);
C(~isfinite(C)) = 0;
C = (C + C') / 2;
try
    [V, D] = eig(C);
catch
    C = C + eye(n_feat) * eps;
    [V, D] = eig(C);
end
[d, ord] = sort(real(diag(D)), 'descend');
V = real(V(:, ord));
K = min(K, size(V, 2));
basis = V(:, 1:K);
for ci = 1:size(basis, 2)
    bnorm = norm(basis(:, ci));
    if isfinite(bnorm) && bnorm > eps
        basis(:, ci) = basis(:, ci) ./ bnorm;
    end
end
basis(~isfinite(basis)) = 0;
info = struct('fit_rule', 'train_fold_residual_feature_pca', ...
    'n_obs', n_obs, 'fallback', false, 'eigenvalues', d(1:min(K, numel(d))));
end

function Yc = prepare_target_matrix_for_pls_local(Y)
Y = double(Y);
if isempty(Y)
    Yc = [];
    return;
end
col_ok = sum(isfinite(Y), 1) >= 5;
Y = Y(:, col_ok);
if isempty(Y)
    Yc = [];
    return;
end
mu = nanmean_local(Y, 1);
for jj = 1:size(Y, 2)
    bad = ~isfinite(Y(:, jj));
    Y(bad, jj) = mu(jj);
end
Yc = bsxfun(@minus, Y, mu);
sd = sqrt(sum(Yc.^2, 1) ./ max(1, size(Yc, 1) - 1));
sd(~isfinite(sd) | sd < eps) = 1;
Yc = bsxfun(@rdivide, Yc, sd);
Yc(~isfinite(Yc)) = 0;
end

function Xc = center_feature_matrix_for_pls_local(X)
X = double(X);
if isempty(X)
    Xc = [];
    return;
end
for rr = 1:size(X, 1)
    row = X(rr, :);
    ok = isfinite(row);
    if any(ok)
        m = mean(row(ok));
        row(~ok) = m;
        X(rr, :) = row - m;
    else
        X(rr, :) = 0;
    end
end
Xc = X;
Xc(~isfinite(Xc)) = 0;
end

function m = nanmean_local(X, dim)
if nargin < 2
    dim = 1;
end
mask = isfinite(X);
X2 = X;
X2(~mask) = 0;
den = sum(mask, dim);
den(den == 0) = 1;
m = sum(X2, dim) ./ den;
end

function [basis, info] = fit_phase_csp_basis_from_lag_cells_local(X_cell, Y_cell, tr_idx, n_feat, K, fs, phase_lo, phase_hi)
K = max(2, min(round(double(K)), n_feat));
n_pair = max(1, floor(K / 2));
C_pos = zeros(n_feat, n_feat);
C_neg = zeros(n_feat, n_feat);
n_pos = 0;
n_neg = 0;
for ii = 1:numel(tr_idx)
    k = tr_idx(ii);
    X = double(X_cell{k});
    Y = double(Y_cell{k});
    if isempty(X) || isempty(Y), continue; end
    if mod(size(X, 1), n_feat) ~= 0
        error('CSP feature rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    T = min(size(X, 2), size(Y, 1));
    if T < 10, continue; end
    try
        phi = local_global_phase_from_joints(Y(1:T, :), fs, phase_lo, phase_hi);
    catch
        phi = nan(T, 1);
    end
    if numel(phi) < T
        phi(end+1:T, 1) = NaN;
    end
    pos_mask = isfinite(phi(1:T)) & cos(phi(1:T)) >= 0;
    neg_mask = isfinite(phi(1:T)) & cos(phi(1:T)) < 0;
    n_lag = size(X, 1) / n_feat;
    for li = 1:n_lag
        rows = (li-1)*n_feat+1:li*n_feat;
        [C_pos, n_pos] = add_trace_normalized_cov_local(C_pos, n_pos, X(rows, 1:T), pos_mask);
        [C_neg, n_neg] = add_trace_normalized_cov_local(C_neg, n_neg, X(rows, 1:T), neg_mask);
    end
end
if n_pos < 2 || n_neg < 2
    basis = zeros(n_feat, min(K, n_feat));
    basis(1:size(basis, 2), 1:size(basis, 2)) = eye(size(basis, 2));
    info = struct('class_rule', 'cos(global_phase)>=0', 'n_pos_blocks', n_pos, ...
        'n_neg_blocks', n_neg, 'fallback', true);
    return;
end
C_pos = C_pos / n_pos;
C_neg = C_neg / n_neg;
C_sum = C_pos + C_neg;
avg_var = trace(C_sum) / max(n_feat, 1);
reg = max(avg_var * 1e-6, eps);
C_pos = (C_pos + C_pos') / 2;
C_sum = (C_sum + C_sum') / 2 + reg * eye(n_feat);
try
    [V, D] = eig(C_pos, C_sum);
    d = real(diag(D));
catch
    [V, D] = eig(pinv(C_sum) * C_pos);
    d = real(diag(D));
end
[~, ord] = sort(d, 'descend');
n_pair = min(n_pair, floor(numel(ord) / 2));
sel = [ord(1:n_pair); ord(end-n_pair+1:end)];
basis = real(V(:, sel));
for ci = 1:size(basis, 2)
    s = norm(basis(:, ci));
    if isfinite(s) && s > eps
        basis(:, ci) = basis(:, ci) ./ s;
    end
end
basis(~isfinite(basis)) = 0;
info = struct('class_rule', 'cos(global_phase)>=0', 'n_pos_blocks', n_pos, ...
    'n_neg_blocks', n_neg, 'fallback', false, 'eigenvalues', d(sel));
end

function [C_acc, n_acc] = add_trace_normalized_cov_local(C_acc, n_acc, X, mask)
mask = mask(:)';
if isempty(X) || numel(mask) ~= size(X, 2) || nnz(mask) < 5
    return;
end
B = double(X(:, mask));
ok = all(isfinite(B), 1);
B = B(:, ok);
if size(B, 2) < 5
    return;
end
B = B - mean(B, 2, 'omitnan');
B(~isfinite(B)) = 0;
Ci = (B * B') / max(size(B, 2) - 1, 1);
tr = trace(Ci);
if ~(isfinite(tr) && tr > eps)
    return;
end
C_acc = C_acc + Ci / tr;
n_acc = n_acc + 1;
end

function [basis_cell, info] = fit_harmonic_phase_csp_basis_from_lag_cells_local(X_cell, Y_cell, tr_idx, layout, pair_counts, fs, phase_lo, phase_hi)
n_harm = layout.n_harm;
n_ch = layout.n_ch;
n_feat = layout.n_feat;
C_pos = cell(1, n_harm);
C_neg = cell(1, n_harm);
for h = 1:n_harm
    C_pos{h} = zeros(n_ch, n_ch);
    C_neg{h} = zeros(n_ch, n_ch);
end
n_pos = zeros(1, n_harm);
n_neg = zeros(1, n_harm);
for ii = 1:numel(tr_idx)
    k = tr_idx(ii);
    X = double(X_cell{k});
    Y = double(Y_cell{k});
    if isempty(X) || isempty(Y), continue; end
    if mod(size(X, 1), n_feat) ~= 0
        error('harmonic CSP feature rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    T = min(size(X, 2), size(Y, 1));
    if T < 10, continue; end
    try
        phi = local_global_phase_from_joints(Y(1:T, :), fs, phase_lo, phase_hi);
    catch
        phi = nan(T, 1);
    end
    if numel(phi) < T
        phi(end+1:T, 1) = NaN;
    end
    pos_mask = isfinite(phi(1:T)) & cos(phi(1:T)) >= 0;
    neg_mask = isfinite(phi(1:T)) & cos(phi(1:T)) < 0;
    n_lag = size(X, 1) / n_feat;
    for li = 1:n_lag
        lag_offset = (li - 1) * n_feat;
        for h = 1:n_harm
            harm_offset = lag_offset + (h - 1) * 2 * n_ch;
            real_rows = harm_offset + (1:n_ch);
            imag_rows = harm_offset + n_ch + (1:n_ch);
            Z_pos = [X(real_rows, pos_mask), X(imag_rows, pos_mask)];
            Z_neg = [X(real_rows, neg_mask), X(imag_rows, neg_mask)];
            [C_pos{h}, n_pos(h)] = add_trace_normalized_cov_samples_local(C_pos{h}, n_pos(h), Z_pos);
            [C_neg{h}, n_neg(h)] = add_trace_normalized_cov_samples_local(C_neg{h}, n_neg(h), Z_neg);
        end
    end
end
basis_cell = cell(1, n_harm);
fallback = false(1, n_harm);
eigenvalues = cell(1, n_harm);
for h = 1:n_harm
    [basis_cell{h}, eigenvalues{h}, fallback(h)] = solve_csp_basis_local( ...
        C_pos{h}, C_neg{h}, n_pos(h), n_neg(h), max(1, pair_counts(h)));
end
info = struct('class_rule', 'cos(global_phase)>=0', ...
    'pair_counts', pair_counts, 'n_pos_blocks', n_pos, ...
    'n_neg_blocks', n_neg, 'fallback', fallback, ...
    'eigenvalues_by_harmonic', {eigenvalues});
end

function [basis, eigenvalues_sel, fallback] = solve_csp_basis_local(C_pos, C_neg, n_pos, n_neg, n_pair)
n_ch = size(C_pos, 1);
K = min(2 * n_pair, n_ch);
if n_pos < 2 || n_neg < 2
    basis = zeros(n_ch, K);
    basis(1:K, 1:K) = eye(K);
    eigenvalues_sel = nan(K, 1);
    fallback = true;
    return;
end
C_pos = C_pos / n_pos;
C_neg = C_neg / n_neg;
C_sum = C_pos + C_neg;
avg_var = trace(C_sum) / max(n_ch, 1);
reg = max(avg_var * 1e-6, eps);
C_pos = (C_pos + C_pos') / 2;
C_sum = (C_sum + C_sum') / 2 + reg * eye(n_ch);
try
    [V, D] = eig(C_pos, C_sum);
    d = real(diag(D));
catch
    [V, D] = eig(pinv(C_sum) * C_pos);
    d = real(diag(D));
end
[~, ord] = sort(d, 'descend');
n_pair = min(n_pair, floor(numel(ord) / 2));
sel = [ord(1:n_pair); ord(end-n_pair+1:end)];
basis = real(V(:, sel));
for ci = 1:size(basis, 2)
    s = norm(basis(:, ci));
    if isfinite(s) && s > eps
        basis(:, ci) = basis(:, ci) ./ s;
    end
end
basis(~isfinite(basis)) = 0;
eigenvalues_sel = d(sel);
fallback = false;
end

function [C_acc, n_acc] = add_trace_normalized_cov_samples_local(C_acc, n_acc, X)
if isempty(X) || size(X, 2) < 5
    return;
end
B = double(X);
ok = all(isfinite(B), 1);
B = B(:, ok);
if size(B, 2) < 5
    return;
end
B = B - mean(B, 2, 'omitnan');
B(~isfinite(B)) = 0;
Ci = (B * B') / max(size(B, 2) - 1, 1);
tr = trace(Ci);
if ~(isfinite(tr) && tr > eps)
    return;
end
C_acc = C_acc + Ci / tr;
n_acc = n_acc + 1;
end

function X_out = apply_fold_csp_local(X_cell, n_feat, basis)
K = size(basis, 2);
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if mod(size(X, 1), n_feat) ~= 0
        error('CSP apply rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    n_lag = size(X, 1) / n_feat;
    Z = zeros(K * n_lag, size(X, 2));
    for li = 1:n_lag
        rows = (li-1)*n_feat+1:li*n_feat;
        blk = X(rows, :);
        blk(~isfinite(blk)) = 0;
        Z((li-1)*K+1:li*K, :) = basis' * blk;
    end
    X_out{ii} = Z;
end
end

function [mu, W] = fold_pca_whitener_from_lag_cells_local(X_cell, tr_idx, K)
sum_x = zeros(K, 1);
n_obs = 0;
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    if mod(size(X, 1), K) ~= 0
        error('lagged feature rows %d are not divisible by %d PCA components', size(X, 1), K);
    end
    n_lag = size(X, 1) / K;
    for li = 1:n_lag
        blk = X((li-1)*K+1:li*K, :);
        ok = all(isfinite(blk), 1);
        if any(ok)
            sum_x = sum_x + sum(blk(:, ok), 2);
            n_obs = n_obs + sum(ok);
        end
    end
end
if n_obs < K + 5
    mu = zeros(K, 1);
    W = eye(K);
    return;
end
mu = sum_x ./ n_obs;
C = zeros(K, K);
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    n_lag = size(X, 1) / K;
    for li = 1:n_lag
        blk = X((li-1)*K+1:li*K, :);
        ok = all(isfinite(blk), 1);
        if any(ok)
            Z = blk(:, ok) - mu;
            C = C + Z * Z';
        end
    end
end
C = C ./ max(n_obs - 1, 1);
avg_var = trace(C) / max(K, 1);
shrink = 0.10;
C = (1 - shrink) * C + shrink * avg_var * eye(K);
C = (C + C') / 2;
[V, D] = eig(C);
[d, ord] = sort(real(diag(D)), 'descend');
V = real(V(:, ord));
floor_var = max(avg_var * 1e-6, eps);
d = max(d, floor_var);
W = diag(1 ./ sqrt(d)) * V';
mu(~isfinite(mu)) = 0;
W(~isfinite(W)) = 0;
end

function X_out = apply_fold_pca_whitener_local(X_cell, K, mu, W)
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if mod(size(X, 1), K) ~= 0
        error('lagged feature rows %d are not divisible by %d PCA components', size(X, 1), K);
    end
    n_lag = size(X, 1) / K;
    Z = zeros(size(X));
    for li = 1:n_lag
        rows = (li-1)*K+1:li*K;
        blk = X(rows, :) - mu;
        blk(~isfinite(blk)) = 0;
        Z(rows, :) = W * blk;
    end
    X_out{ii} = Z;
end
end

function [mu, W] = fold_lagspace_pca_whitener_local(X_cell, tr_idx, K)
P = first_nonempty_feature_rows_local(X_cell);
sum_x = zeros(P, 1);
n_obs = 0;
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    ok = all(isfinite(X), 1);
    if any(ok)
        sum_x = sum_x + sum(X(:, ok), 2);
        n_obs = n_obs + sum(ok);
    end
end
if n_obs < K + 5
    mu = zeros(P, 1);
    W = zeros(K, P);
    W(:, 1:K) = eye(K);
    return;
end
mu = sum_x ./ n_obs;
C = zeros(P, P);
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    ok = all(isfinite(X), 1);
    if any(ok)
        Z = X(:, ok) - mu;
        C = C + Z * Z';
    end
end
C = C ./ max(n_obs - 1, 1);
avg_var = trace(C) / max(P, 1);
shrink = 0.05;
C = (1 - shrink) * C + shrink * avg_var * eye(P);
C = (C + C') / 2;
opts = struct('disp', 0);
try
    [V, D] = eigs(C, K, 'largestreal', opts);
catch
    [Vfull, Dfull] = eig(C);
    [~, ord] = sort(real(diag(Dfull)), 'descend');
    V = Vfull(:, ord(1:K));
    D = Dfull(ord(1:K), ord(1:K));
end
d = real(diag(D));
[d, ord] = sort(d, 'descend');
V = real(V(:, ord));
floor_var = max(avg_var * 1e-6, eps);
d = max(d, floor_var);
W = diag(1 ./ sqrt(d)) * V';
mu(~isfinite(mu)) = 0;
W(~isfinite(W)) = 0;
end

function X_out = apply_lagspace_pca_whitener_local(X_cell, mu, W)
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    Z = X - mu;
    Z(~isfinite(Z)) = 0;
    X_out{ii} = W * Z;
end
end

function [mu, sd] = fold_channel_scaler_from_lag_cells_local(X_cell, tr_idx, n_ch)
sum_x = zeros(n_ch, 1);
sum_x2 = zeros(n_ch, 1);
count_x = zeros(n_ch, 1);
for ii = 1:numel(tr_idx)
    X = double(X_cell{tr_idx(ii)});
    if isempty(X), continue; end
    if mod(size(X, 1), n_ch) ~= 0
        error('lagged feature rows %d are not divisible by %d EEG channels', size(X, 1), n_ch);
    end
    n_lag = size(X, 1) / n_ch;
    for li = 1:n_lag
        blk = X((li-1)*n_ch+1:li*n_ch, :);
        ok = isfinite(blk);
        blk(~ok) = 0;
        sum_x = sum_x + sum(blk, 2);
        sum_x2 = sum_x2 + sum(blk.^2, 2);
        count_x = count_x + sum(ok, 2);
    end
end
count_safe = max(count_x, 1);
mu = sum_x ./ count_safe;
var_x = (sum_x2 ./ count_safe) - mu.^2;
sd = sqrt(max(var_x, 0));
mu(~isfinite(mu) | count_x == 0) = 0;
sd(~isfinite(sd) | sd <= eps | count_x == 0) = 1;
end

function X_out = apply_fold_diagwhiten_local(X_cell, n_ch, mu, sd)
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if mod(size(X, 1), n_ch) ~= 0
        error('lagged feature rows %d are not divisible by %d EEG channels', size(X, 1), n_ch);
    end
    n_lag = size(X, 1) / n_ch;
    Z = zeros(size(X));
    for li = 1:n_lag
        rows = (li-1)*n_ch+1:li*n_ch;
        blk = (X(rows, :) - mu) ./ sd;
        blk(~isfinite(blk)) = 0;
        Z(rows, :) = blk;
    end
    X_out{ii} = Z;
end
end

function X_out = apply_fold_diagwhiten_shared_pca_local(X_cell, n_ch, basis, mu, sd)
K = size(basis, 2);
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if mod(size(X, 1), n_ch) ~= 0
        error('lagged feature rows %d are not divisible by %d EEG channels', size(X, 1), n_ch);
    end
    n_lag = size(X, 1) / n_ch;
    Z = zeros(K * n_lag, size(X, 2));
    for li = 1:n_lag
        blk = X((li-1)*n_ch+1:li*n_ch, :);
        blk = (blk - mu) ./ sd;
        blk(~isfinite(blk)) = 0;
        Z((li-1)*K+1:li*K, :) = basis' * blk;
    end
    X_out{ii} = Z;
end
end

function X_out = apply_lagspace_shared_pca_local(X_cell, basis)
K = size(basis, 2);
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if size(X, 1) ~= size(basis, 1)
        error('lag-space PCA rows %d do not match X rows %d', size(basis, 1), size(X, 1));
    end
    Z = basis' * X;
    Z(~isfinite(Z)) = 0;
    X_out{ii} = reshape(Z, K, size(X, 2));
end
end

function [X_first, X_extra] = split_lag_feature_cells_local(X_cell, n_first, n_extra)
block_rows = n_first + n_extra;
X_first = cell(size(X_cell));
X_extra = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_first{ii} = X;
        X_extra{ii} = X;
        continue;
    end
    if mod(size(X, 1), block_rows) ~= 0
        error('lagged feature rows %d are not divisible by hybrid block rows %d', size(X, 1), block_rows);
    end
    n_lag = size(X, 1) / block_rows;
    A = zeros(n_first * n_lag, size(X, 2));
    B = zeros(n_extra * n_lag, size(X, 2));
    for li = 1:n_lag
        src = (li-1)*block_rows + (1:block_rows);
        dst_a = (li-1)*n_first + (1:n_first);
        dst_b = (li-1)*n_extra + (1:n_extra);
        A(dst_a, :) = X(src(1:n_first), :);
        B(dst_b, :) = X(src(n_first+1:end), :);
    end
    X_first{ii} = A;
    X_extra{ii} = B;
end
end

function X_out = concat_feature_cells_local(A_cell, B_cell)
if numel(A_cell) ~= numel(B_cell)
    error('Cannot concatenate feature cells with different lengths');
end
X_out = cell(size(A_cell));
for ii = 1:numel(A_cell)
    A = double(A_cell{ii});
    B = double(B_cell{ii});
    if isempty(A)
        X_out{ii} = B;
    elseif isempty(B)
        X_out{ii} = A;
    else
        if size(A, 2) ~= size(B, 2)
            error('Cannot concatenate feature cells with different sample counts at trial %d', ii);
        end
        X_out{ii} = [A; B];
    end
end
end

function X_out = remove_shared_pca_basis_from_lag_cells_local(X_cell, n_feat, basis)
if size(basis, 1) ~= n_feat
    error('shared PCA residualization basis rows %d do not match n_feat=%d', size(basis, 1), n_feat);
end
[Q, ~] = qr(double(basis), 0);
Q(~isfinite(Q)) = 0;
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if mod(size(X, 1), n_feat) ~= 0
        error('shared PCA residualization rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    n_lag = size(X, 1) / n_feat;
    Z = zeros(size(X));
    for li = 1:n_lag
        rows = (li - 1) * n_feat + (1:n_feat);
        B = X(rows, :);
        B(~isfinite(B)) = 0;
        Z(rows, :) = B - Q * (Q' * B);
    end
    Z(~isfinite(Z)) = 0;
    X_out{ii} = Z;
end
end

function W = fit_ea_channel_whiten_from_lag_cells_local(X_cell, tr_idx, n_feat, n_ch, gamma)
% Euclidean Alignment whitener from OUTER-TRAIN trials only: pool the n_ch x n_ch
% channel covariance over all harmonic/quad blocks and lags, shrink, return R^-1/2.
if n_ch <= 0 || mod(n_feat, n_ch) ~= 0
    W = eye(max(n_ch, 1));
    return;
end
nb = n_feat / n_ch;
R = zeros(n_ch, n_ch);
cnt = 0;
for ii = 1:numel(tr_idx)
    Xc = double(X_cell{tr_idx(ii)});
    if isempty(Xc), continue; end
    if mod(size(Xc, 1), n_feat) ~= 0, continue; end
    nlag = size(Xc, 1) / n_feat;
    for li = 1:nlag
        base = (li - 1) * n_feat;
        for b = 1:nb
            rows = base + (b - 1) * n_ch + (1:n_ch);
            Xb = Xc(rows, :);
            Xb(~isfinite(Xb)) = 0;
            R = R + Xb * Xb';
            cnt = cnt + size(Xb, 2);
        end
    end
end
if cnt < 2 || ~all(isfinite(R(:)))
    W = eye(n_ch);
    return;
end
R = R / max(cnt - 1, 1);
R = (R + R') / 2;
g = min(max(gamma, 0), 1);
R = (1 - g) * R + g * (trace(R) / n_ch) * eye(n_ch);
[V, D] = eig((R + R') / 2, 'vector');
d = real(D);
d(~isfinite(d) | d < eps) = eps;
W = real(V * diag(d .^ (-0.5)) * V');
W = (W + W') / 2;
end

function X_out = apply_ea_channel_whiten_lag_cells_local(X_cell, n_feat, n_ch, W)
% Apply the n_ch x n_ch EA whitener to every n_ch-channel block within every
% n_feat-row lag block, preserving harmonic/quadrature/lag structure.
X_out = cell(size(X_cell));
if n_ch <= 0 || mod(n_feat, n_ch) ~= 0
    X_out = X_cell;
    return;
end
nb = n_feat / n_ch;
for ii = 1:numel(X_cell)
    Xc = double(X_cell{ii});
    if isempty(Xc) || mod(size(Xc, 1), n_feat) ~= 0
        X_out{ii} = Xc;
        continue;
    end
    nlag = size(Xc, 1) / n_feat;
    Z = Xc;
    for li = 1:nlag
        base = (li - 1) * n_feat;
        for b = 1:nb
            rows = base + (b - 1) * n_ch + (1:n_ch);
            Z(rows, :) = W * Xc(rows, :);
        end
    end
    Z(~isfinite(Z)) = 0;
    X_out{ii} = Z;
end
end

function X_out = apply_harmonic_csp_local(X_cell, layout, basis_cell)
n_harm = layout.n_harm;
n_ch = layout.n_ch;
n_feat = layout.n_feat;
rows_per_lag = 0;
for h = 1:n_harm
    rows_per_lag = rows_per_lag + 2 * size(basis_cell{h}, 2);
end
X_out = cell(size(X_cell));
for ii = 1:numel(X_cell)
    X = double(X_cell{ii});
    if isempty(X)
        X_out{ii} = X;
        continue;
    end
    if mod(size(X, 1), n_feat) ~= 0
        error('harmonic CSP apply rows %d are not divisible by n_feat=%d', size(X, 1), n_feat);
    end
    n_lag = size(X, 1) / n_feat;
    Z = zeros(rows_per_lag * n_lag, size(X, 2));
    for li = 1:n_lag
        lag_in = (li - 1) * n_feat;
        lag_out = (li - 1) * rows_per_lag;
        out_cursor = 0;
        for h = 1:n_harm
            W = basis_cell{h};
            K = size(W, 2);
            harm_offset = lag_in + (h - 1) * 2 * n_ch;
            real_rows = harm_offset + (1:n_ch);
            imag_rows = harm_offset + n_ch + (1:n_ch);
            R = X(real_rows, :);
            I = X(imag_rows, :);
            R(~isfinite(R)) = 0;
            I(~isfinite(I)) = 0;
            out_rows = lag_out + out_cursor + (1:2*K);
            Z(out_rows, :) = [W' * R; W' * I];
            out_cursor = out_cursor + 2 * K;
        end
    end
    X_out{ii} = Z;
end
end

function Y_phase = apply_phase_target_mode(Y_phase, Y_delta, mode, fs, phase_lo, phase_hi)
mode = lower(char(mode));
switch mode
    case 'phase_pj'
        return;
    case 'phase_plane'
        J = size(Y_delta{1}, 2);
        opt = struct('phase_lo', phase_lo, 'phase_hi', phase_hi);
        [Y_plane, ~] = build_phase_plane_target(Y_delta, opt, fs);
        for k = 1:numel(Y_delta)
            T = size(Y_delta{k}, 1);
            Yk = nan(T, 2 * J);
            n = min(T, size(Y_plane{k}, 1));
            if n > 0
                for j = 1:J
                    Yk(1:n, 2*j - 1) = Y_plane{k}(1:n, 1);
                    Yk(1:n, 2*j) = Y_plane{k}(1:n, 2);
                end
            end
            Y_phase{k} = Yk;
        end
        return;
    case 'phase_global6'
        src_j = NaN;
    case {'phase_lhip', 'phase_lhip_portrait'}
        src_j = 1;
    case {'phase_rhip', 'phase_rhip_portrait'}
        src_j = 4;
    otherwise
        error('phase_target_mode: unsupported value');
end
J = size(Y_delta{1}, 2);
for k = 1:numel(Y_delta)
    if size(Y_delta{k}, 1) == 0 || (~isnan(src_j) && src_j > size(Y_delta{k}, 2))
        Y_phase{k} = zeros(0, 2 * J);
        continue;
    end
    if strcmp(mode, 'phase_global6')
        phi = local_global_phase_from_joints(Y_delta{k}, fs, phase_lo, phase_hi);
    else
        y = Y_delta{k}(:, src_j);
        if all(~isfinite(y))
            Y_phase{k} = nan(size(Y_delta{k}, 1), 2 * J);
            continue;
        end
        if contains(mode, 'portrait')
            phi = local_phase_portrait(y, fs, phase_lo, phase_hi);
        else
            phi = local_phase_from_angle(y, fs, phase_lo, phase_hi);
        end
    end
    if all(~isfinite(phi))
        Y_phase{k} = nan(size(Y_delta{k}, 1), 2 * J);
        continue;
    end
    Yk = nan(numel(phi), 2 * J);
    for j = 1:J
        Yk(:, 2*j - 1) = cos(phi);
        Yk(:, 2*j) = sin(phi);
    end
    Y_phase{k} = Yk;
end
end

function phi = local_global_phase_from_joints(Y, fs, phase_lo, phase_hi)
Y = double(Y);
T = size(Y, 1);
keep = any(isfinite(Y), 1);
if nnz(keep) == 0
    phi = nan(T, 1);
    return;
end
[z, p, k] = butter(2, [phase_lo phase_hi] / (fs / 2), 'bandpass');
sos = zp2sos(z, p, k);
Z = Y(:, keep);
for c = 1:size(Z, 2)
    y = fill_finite_local(Z(:, c));
    y = filtfilt(sos, 1, y);
    y = y - mean(y, 'omitnan');
    sd = std(y, 0, 'omitnan');
    if sd > eps
        y = y ./ sd;
    end
    Z(:, c) = y;
end
valid = std(Z, 0, 1, 'omitnan') > eps;
if nnz(valid) == 0
    phi = nan(T, 1);
    return;
end
Z = Z(:, valid);
if size(Z, 2) == 1
    pc1 = Z(:, 1);
else
    [~, S, V] = svd(Z, 'econ');
    pc1 = Z * V(:, 1);
    if ~isempty(S) && S(1, 1) <= eps
        pc1 = Z(:, 1);
    end
end
anchor = choose_global_phase_anchor(Y, fs, phase_lo, phase_hi);
if any(isfinite(anchor))
    cc = corr(pc1, anchor, 'rows', 'complete');
    if isfinite(cc) && cc < 0
        pc1 = -pc1;
    end
end
phi = angle(hilbert(pc1));
end

function anchor = choose_global_phase_anchor(Y, fs, phase_lo, phase_hi)
J = size(Y, 2);
candidate = min(4, J); % RHip when the standard 6-joint Walker layout is present.
if candidate >= 1 && any(isfinite(Y(:, candidate)))
    anchor = Y(:, candidate);
else
    idx = find(any(isfinite(Y), 1), 1);
    if isempty(idx)
        anchor = nan(size(Y, 1), 1);
        return;
    end
    anchor = Y(:, idx);
end
anchor = fill_finite_local(anchor);
[z, p, k] = butter(2, [phase_lo phase_hi] / (fs / 2), 'bandpass');
sos = zp2sos(z, p, k);
anchor = filtfilt(sos, 1, anchor);
anchor = anchor - mean(anchor, 'omitnan');
sd = std(anchor, 0, 'omitnan');
if sd > eps
    anchor = anchor ./ sd;
end
end

function phi = local_phase_portrait(y, fs, phase_lo, phase_hi)
y = fill_finite_local(y);
[z, p, k] = butter(2, [phase_lo phase_hi] / (fs / 2), 'bandpass');
sos = zp2sos(z, p, k);
q = filtfilt(sos, 1, y);
q = q - mean(q, 'omitnan');
dq = gradient(q) * fs;
q_sd = std(q, 0, 'omitnan');
dq_sd = std(dq, 0, 'omitnan');
if q_sd <= eps || dq_sd <= eps
    phi = angle(hilbert(q));
else
    phi = unwrap(atan2(dq ./ dq_sd, q ./ q_sd));
end
end

function phi = local_phase_from_angle(y, fs, phase_lo, phase_hi)
 y = fill_finite_local(y);
[z, p, k] = butter(2, [phase_lo phase_hi] / (fs / 2), 'bandpass');
sos = zp2sos(z, p, k);
y_lp = filtfilt(sos, 1, y);
phi = angle(hilbert(y_lp));
end

function y = fill_finite_local(y)
y = double(y(:));
if any(~isfinite(y))
    good = isfinite(y);
    if sum(good) == 0
        y(:) = 0;
    elseif sum(good) == 1
        y(~good) = y(find(good, 1));
    else
        y(~good) = interp1(find(good), y(good), find(~good), 'linear', 'extrap');
    end
end
end

function [A_pred, lambda, model, mu, sd] = fit_predict_angle(X, Y_delta, tr_idx, te_idx, lambda_grid, lambda_mode, inner_k)
if nargin < 7 || isempty(inner_k)
    inner_k = 5;
end
[Y_tr_z, mu, sd] = standardize_cells_local(Y_delta(tr_idx), []);
model = train_decoder_local(X(tr_idx), Y_tr_z, lambda_grid, lambda_mode, inner_k);
Yp_z = predict_wiener(model, X(te_idx));
A_pred = inverse_standardize_cells(Yp_z, mu, sd);
lambda = summarize_lambda(model.lambda);
end

function Wgrp_fold = project_group_prior_to_fold_local(Wgrp_raw, feature_info, n_lag)
% Project the leave-session-out group-prior readout (common raw band-feature x
% lag space) into THIS fold's (basis, sd) shared-PCA frame. Inverse of the
% forward map used to build the prior (rawW = (basis*Wpc)./sd): since basis is
% orthonormal, Wpc = basis' * (sd .* rawW). Cross-session => projection of the
% group readout into the held-out session's own coordinate frame.
assert(isfield(feature_info, 'basis') && isfield(feature_info, 'feature_sd'), ...
    'stage1_readout_mode=group_shrink_ridge requires a shared-PCA feature_mode (basis + feature_sd)');
basis = double(feature_info.basis);          % [n_feat x K]
sd = double(feature_info.feature_sd(:));      % [n_feat x 1]
K = size(basis, 2);
J = size(Wgrp_raw, 3);
Wgrp_fold = nan(K * n_lag + 1, J);            % [P+1 x J], last row = intercept slot
for j = 1:J
    raw_j = Wgrp_raw(:, :, j);                % [n_feat x n_lag]
    if all(isfinite(raw_j(:)))
        Wpc = basis' * (sd .* raw_j);         % [K x n_lag]
        Wgrp_fold(1:K * n_lag, j) = reshape(Wpc, K * n_lag, 1); % row = (lag-1)*K + pc
        Wgrp_fold(K * n_lag + 1, j) = 0;      % intercept is never shrunk
    end
end
end

function Wb = blend_group_prior_local(W, Wgrp_fold, lam_s)
% Convex James-Stein / Efron-Morris blend toward a non-zero (group-mean) target.
% lam_s=0 reproduces W exactly (built-in null). Slopes only; the intercept and
% any joint/column absent from the prior keep the baseline readout.
Wb = W;
if isempty(Wgrp_fold) || ~(lam_s > 0); return; end
P1 = size(W, 1);
slope = 1:(P1 - 1);
for j = 1:size(W, 2)
    if ~all(isnan(W(slope, j))) && all(isfinite(Wgrp_fold(slope, j)))
        Wb(slope, j) = (1 - lam_s) * W(slope, j) + lam_s * Wgrp_fold(slope, j);
    end
end
end

function [A_pred, lambda, model, mu, sd, shrink_lambda] = fit_predict_angle_group_shrink( ...
        X, Y_delta, tr_idx, te_idx, lambda_grid, lambda_mode, inner_k, Wgrp_fold, shrink_grid, force_lambda)
% Empirical-Bayes / James-Stein partial pooling of the Stage-1 readout (Efron &
% Morris 1975; Gelman & Hill partial pooling; Alamgir 2010 shared-mean BCI prior):
% fit the per-session ridge readout, then shrink it toward the leave-session-out
% group-mean readout by a convex weight chosen by nested inner-CV. Weight 0
% reproduces the per-session baseline exactly (built-in negative control).
% force_lambda >= 0 bypasses inner-CV and forces that borrow weight (diagnostic).
if nargin < 10 || isempty(force_lambda); force_lambda = -1; end
[Y_tr_z, mu, sd] = standardize_cells_local(Y_delta(tr_idx), []);
model = train_decoder_local(X(tr_idx), Y_tr_z, lambda_grid, lambda_mode, inner_k);
lambda = summarize_lambda(model.lambda);
shrink_grid = unique([0, double(shrink_grid(:)')]);     % always include the null
shrink_grid = shrink_grid(isfinite(shrink_grid) & shrink_grid >= 0 & shrink_grid <= 1);
shrink_lambda = 0;
if ~isempty(Wgrp_fold) && force_lambda >= 0
    shrink_lambda = max(0, min(1, double(force_lambda)));    % forced borrow weight
elseif ~isempty(Wgrp_fold) && any(shrink_grid > 0) && numel(tr_idx) >= 2
    ridge_lambda = double(model.lambda);
    if ~isscalar(ridge_lambda); ridge_lambda = mean(ridge_lambda(isfinite(ridge_lambda))); end
    shrink_lambda = inner_select_shrink_local(X, Y_delta, Y_tr_z, tr_idx, mu, sd, ...
        ridge_lambda, inner_k, Wgrp_fold, shrink_grid);
end
model.W = blend_group_prior_local(model.W, Wgrp_fold, shrink_lambda);
model.shrink_lambda = shrink_lambda;
Yp_z = predict_wiener(model, X(te_idx));
A_pred = inverse_standardize_cells(Yp_z, mu, sd);
end

function lam_best = inner_select_shrink_local(X, Y_delta, Y_tr_z, tr_idx, mu, sd, ...
        ridge_lambda, inner_k, Wgrp_fold, shrink_grid)
% Fold-contained inner-CV selection of the convex shrink weight, with a
% conservative 1-SE rule that DEFAULTS to no shrinkage. Per inner fold, fit the
% baseline readout at the selected ridge lambda, blend toward the group prior at
% each candidate weight, and score held-out rpt PER FOLD. A non-zero shrink is
% chosen only if it beats lambda=0 robustly (its mean inner score exceeds the
% best-minus-1SE threshold while a smaller lambda does not); otherwise lambda=0
% is kept.
% Training trials only.
shrink_grid = unique([0, double(shrink_grid(:)')]);   % ascending, lambda=0 first
n = numel(tr_idx);
K = min(max(2, round(double(inner_k))), n);
rng(20260610, 'twister');
perm = randperm(n);
fold_id = zeros(1, n);
for k = 1:K
    fold_id(perm(k:K:end)) = k;
end
ng = numel(shrink_grid);
fold_score = nan(K, ng);                              % per-fold rpt for each candidate
for k = 1:K
    inner_tr_pos = find(fold_id ~= k);
    inner_te_pos = find(fold_id == k);
    if numel(inner_tr_pos) < 2 || isempty(inner_te_pos); continue; end
    m = train_wiener_fixed_scalar_local(X(tr_idx(inner_tr_pos)), Y_tr_z(inner_tr_pos), ridge_lambda);
    for gi = 1:ng
        mb = m;
        mb.W = blend_group_prior_local(m.W, Wgrp_fold, shrink_grid(gi));
        A_val = inverse_standardize_cells(predict_wiener(mb, X(tr_idx(inner_te_pos))), mu, sd);
        fold_score(k, gi) = mean_rpt_all_joints(Y_delta(tr_idx(inner_te_pos)), A_val);
    end
end
keep = ~all(isnan(fold_score), 2);
fs = fold_score(keep, :);
if isempty(fs)
    lam_best = 0;
    return;
end
mean_s = mean(fs, 1, 'omitnan');
nrow = max(1, sum(~isnan(fs), 1));
se_s = std(fs, 0, 1, 'omitnan') ./ sqrt(nrow);
[best_mu, bi] = max(mean_s);
within = find(mean_s >= best_mu - se_s(bi));          % candidates within 1 SE of best
lam_best = shrink_grid(within(1));                    % smallest (closest to baseline) such lambda
end

function [A_pred, lambda, A_oof, model, mu, sd] = fit_predict_angle_oof_lambda(X, Y_delta, tr_idx, te_idx, lambda_grid, inner_k, outer_fold, lambda_mode, verbose_progress)
if nargin < 9
    verbose_progress = false;
end
if ~strcmpi(char(lambda_mode), 'scalar')
    error('stage1_lambda_mode=oof currently requires lambda_mode=scalar');
end
lambda_grid = unique(double(lambda_grid(:)'));
lambda_grid = lambda_grid(isfinite(lambda_grid) & lambda_grid >= 0);
if isempty(lambda_grid)
    lambda_grid = 0.001;
end
[Y_tr_z, mu, sd] = standardize_cells_local(Y_delta(tr_idx), []);
n = numel(tr_idx);
if n < 2
    error('Stage1 OOF lambda selection requires at least two training trials');
end
K = min(max(2, round(double(inner_k))), n);
score = nan(size(lambda_grid));
oof_by_lambda = cell(size(lambda_grid));
if K >= 2
    rng(20260531 + outer_fold, 'twister');
    perm = randperm(n);
    fold_id = zeros(1, n);
    for k = 1:K
        fold_id(perm(k:K:end)) = k;
    end
    for li = 1:numel(lambda_grid)
        A_oof_try_z = cell(size(tr_idx));
        for k = 1:K
            inner_tr_pos = fold_id ~= k;
            inner_te_pos = find(fold_id == k);
            if sum(inner_tr_pos) < 2 || isempty(inner_te_pos)
                continue;
            end
            model = train_decoder_fixed_lambda_local( ...
                X(tr_idx(inner_tr_pos)), Y_tr_z(inner_tr_pos), lambda_grid(li), lambda_mode);
            A_val_z = predict_wiener(model, X(tr_idx(inner_te_pos)));
            for ii = 1:numel(inner_te_pos)
                A_oof_try_z{inner_te_pos(ii)} = A_val_z{ii};
            end
        end
        A_oof_try = inverse_standardize_cells(A_oof_try_z, mu, sd);
        oof_by_lambda{li} = A_oof_try;
        score(li) = mean_rpt_all_joints(Y_delta(tr_idx), A_oof_try);
    end
end
if all(~isfinite(score))
    [~, best_li] = min(abs(lambda_grid - 0.001));
else
    [~, best_li] = max(score);
end
lambda = lambda_grid(best_li);
A_oof = oof_by_lambda{best_li};
if isempty(A_oof)
    A_oof = angle_oof_predictions(X, Y_delta, tr_idx, lambda, inner_k, outer_fold, lambda_mode, verbose_progress);
end
model = train_decoder_fixed_lambda_local(X(tr_idx), Y_tr_z, lambda, lambda_mode);
Yp_z = predict_wiener(model, X(te_idx));
A_pred = inverse_standardize_cells(Yp_z, mu, sd);
if logical(verbose_progress)
    fprintf('  Stage1 OOF lambda selected %.6g from %d candidates, score=%.4f\n', ...
        lambda, numel(lambda_grid), score(best_li));
    drawnow;
end
end

function [P_pred, lambda, S_pred] = fit_predict_phase_recon(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, phase_recon, harmonic_k, lambda_mode, inner_k, phase_delay_gain, phase_lo, phase_hi, phase_select_margin, phase_jitter_ms, X_aux_phase, Y_aux_phase)
if nargin < 13 || isempty(inner_k)
    inner_k = 5;
end
if nargin < 19
    X_aux_phase = {};
    Y_aux_phase = {};
end
S_pred = {};
mode = lower(char(phase_recon));
switch mode
    case 'template'
        [X_phase_train, Y_phase_train] = append_aux_phase_cells(X(tr_idx), Y_phase(tr_idx), X_aux_phase, Y_aux_phase);
        model = train_decoder_local(X_phase_train, Y_phase_train, lambda_grid, lambda_mode, inner_k);
        Yp_phase = predict_wiener(model, X(te_idx));
        Yp_phase = smooth_phase_cells(Yp_phase, smooth_type, fs);
        templates = fit_delta_templates(Y_phase(tr_idx), Y_delta(tr_idx), n_bins);
        P_pred = phase_template_predict(Yp_phase, templates);
        S_pred = Yp_phase;
    case 'select_template_tempo'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda] = fit_predict_select_template_tempo(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, n_bins, harmonic_k, lambda_mode, ...
            phase_lo, phase_hi, phase_select_margin);
        return;
    case 'blend_template_tempo'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda] = fit_predict_blend_template_tempo(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, n_bins, harmonic_k, lambda_mode, ...
            phase_lo, phase_hi);
        return;
    case 'delay_template'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda, S_pred] = fit_predict_delay_template(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain);
        return;
    case 'delay_template_jitter'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda, S_pred] = fit_predict_delay_template_jitter(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain, phase_jitter_ms);
        return;
    case 'delay_template_orient'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda, S_pred] = fit_predict_delay_template_orient(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain);
        return;
    case 'delay_template_osc'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda, S_pred] = fit_predict_delay_template_osc(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain);
        return;
    case 'delay_template_kalman'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda, S_pred] = fit_predict_delay_template_kalman(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain);
        return;
    case 'template_residual'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda, S_pred] = fit_predict_template_residual(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, n_bins, lambda_mode);
        return;
    case 'template_residual_pca'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda, S_pred] = fit_predict_template_residual_pca(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, n_bins, harmonic_k, lambda_mode);
        return;
    case 'bias_template'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda] = fit_predict_bias_template(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain);
        return;
    case 'harmonic'
        [X_phase_train, Y_phase_train] = append_aux_phase_cells(X(tr_idx), Y_phase(tr_idx), X_aux_phase, Y_aux_phase);
        model = train_decoder_local(X_phase_train, Y_phase_train, lambda_grid, lambda_mode, inner_k);
        Yp_phase = predict_wiener(model, X(te_idx));
        Yp_phase = smooth_phase_cells(Yp_phase, smooth_type, fs);
        H = fit_harmonic_models(Y_phase(tr_idx), Y_delta(tr_idx), harmonic_k);
        P_pred = harmonic_predict(Yp_phase, H, harmonic_k);
        S_pred = Yp_phase;
    case 'delay_harmonic'
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [P_pred, lambda, S_pred] = fit_predict_delay_harmonic(X, Y_phase, Y_delta, tr_idx, te_idx, ...
            lambda_grid, smooth_type, fs, harmonic_k, lambda_mode, phase_delay_gain);
        return;
    case {'amp_phase', 'amp_harmonic', 'tempo_harmonic'}
        if ~isempty(X_aux_phase)
            error('aux_condition currently supports template/harmonic phase_recon only');
        end
        [Y_state, layout] = build_state_cells(Y_phase, Y_delta, fs, phase_lo, phase_hi, mode, harmonic_k);
        [Y_state_tr_z, mu, sd] = standardize_cells_local(Y_state(tr_idx), []);
        model = train_decoder_local(X(tr_idx), Y_state_tr_z, lambda_grid, lambda_mode, inner_k);
        Yp_state_z = predict_wiener(model, X(te_idx));
        Yp_state = inverse_standardize_cells(Yp_state_z, mu, sd);
        R = fit_state_recon_models(Y_state(tr_idx), Y_delta(tr_idx), layout);
        P_pred = state_recon_predict(Yp_state, R, layout);
        S_pred = Yp_state;
    otherwise
        error('phase_recon: unsupported value');
end
lambda = summarize_lambda(model.lambda);
end

function [P_pred, lambda] = fit_predict_select_template_tempo(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, harmonic_k, lambda_mode, phase_lo, phase_hi, phase_select_margin)
[P_template_te, P_template_oof, lambda_template] = predict_template_with_oof(X, Y_phase, Y_delta, tr_idx, te_idx, ...
    lambda_grid, smooth_type, fs, n_bins, lambda_mode);
[P_tempo_te, P_tempo_oof, lambda_tempo] = predict_state_recon_with_oof(X, Y_phase, Y_delta, tr_idx, te_idx, ...
    lambda_grid, fs, phase_lo, phase_hi, 'tempo_harmonic', harmonic_k, lambda_mode);
use_tempo = choose_prediction_family(Y_delta(tr_idx), P_template_oof, P_tempo_oof, phase_select_margin);
P_pred = guarded_prediction_cells(P_tempo_te, P_template_te, use_tempo);
lambda = mean([lambda_template, lambda_tempo], 'omitnan');
end

function [P_pred, lambda] = fit_predict_blend_template_tempo(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, harmonic_k, lambda_mode, phase_lo, phase_hi)
[P_template_te, P_template_oof, lambda_template] = predict_template_with_oof(X, Y_phase, Y_delta, tr_idx, te_idx, ...
    lambda_grid, smooth_type, fs, n_bins, lambda_mode);
[P_tempo_te, P_tempo_oof, lambda_tempo] = predict_state_recon_with_oof(X, Y_phase, Y_delta, tr_idx, te_idx, ...
    lambda_grid, fs, phase_lo, phase_hi, 'tempo_harmonic', harmonic_k, lambda_mode);
beta = choose_blend_beta(Y_delta(tr_idx), P_template_oof, P_tempo_oof, 0:0.1:1);
P_pred = blend_prediction_cells(P_template_te, P_tempo_te, beta);
lambda = mean([lambda_template, lambda_tempo], 'omitnan');
end

function [P_te, P_oof, lambda] = predict_template_with_oof(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, lambda_mode)
model = train_decoder_local(X(tr_idx), Y_phase(tr_idx), lambda_grid, lambda_mode);
Yp_phase_te = predict_wiener(model, X(te_idx));
Yp_phase_te = smooth_phase_cells(Yp_phase_te, smooth_type, fs);
Yp_phase_oof = phase_oof_predictions(X, Y_phase, tr_idx, lambda_grid, smooth_type, fs, lambda_mode);
templates = fit_delta_templates(Y_phase(tr_idx), Y_delta(tr_idx), n_bins);
P_te = phase_template_predict(Yp_phase_te, templates);
P_oof = phase_template_predict(Yp_phase_oof, templates);
lambda = summarize_lambda(model.lambda);
end

function [P_te, P_oof, lambda] = predict_state_recon_with_oof(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, fs, phase_lo, phase_hi, mode, harmonic_k, lambda_mode)
[Y_state, layout] = build_state_cells(Y_phase, Y_delta, fs, phase_lo, phase_hi, mode, harmonic_k);
[Y_state_tr_z, mu, sd] = standardize_cells_local(Y_state(tr_idx), []);
model = train_decoder_local(X(tr_idx), Y_state_tr_z, lambda_grid, lambda_mode);
Yp_state_te_z = predict_wiener(model, X(te_idx));
Yp_state_te = inverse_standardize_cells(Yp_state_te_z, mu, sd);
Yp_state_oof_z = target_oof_predictions(X(tr_idx), Y_state_tr_z, lambda_grid, 'none', fs, lambda_mode);
Yp_state_oof = inverse_standardize_cells(Yp_state_oof_z, mu, sd);
R = fit_state_recon_models(Y_state(tr_idx), Y_delta(tr_idx), layout);
P_te = state_recon_predict(Yp_state_te, R, layout);
P_oof = state_recon_predict(Yp_state_oof, R, layout);
lambda = summarize_lambda(model.lambda);
end

function use_alt = choose_prediction_family(Y_true, P_base, P_alt, margin)
J = size(Y_true{find_first_nonempty_cell(Y_true)}, 2);
use_alt = false(1, J);
for j = 1:J
    [r_base, ~] = eval_one(Y_true, P_base, j);
    [r_alt, ~] = eval_one(Y_true, P_alt, j);
    use_alt(j) = isfinite(r_alt) && (~isfinite(r_base) || r_alt >= r_base + margin);
end
end

function [P_pred, lambda] = fit_predict_bias_template(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain)
model_phase = train_decoder_local(X(tr_idx), Y_phase(tr_idx), lambda_grid, lambda_mode);
Yp_phase_te = predict_wiener(model_phase, X(te_idx));
Yp_phase_te = smooth_phase_cells(Yp_phase_te, smooth_type, fs);
Yp_phase_oof = phase_oof_predictions(X, Y_phase, tr_idx, lambda_grid, smooth_type, fs, lambda_mode);

bias = fit_phase_bias_cells(Y_phase(tr_idx), Yp_phase_oof);
Yp_bias_te = apply_phase_bias_cells(Yp_phase_te, bias, phase_delay_gain);
Yp_bias_oof = apply_phase_bias_cells(Yp_phase_oof, bias, phase_delay_gain);

templates = fit_delta_templates(Y_phase(tr_idx), Y_delta(tr_idx), n_bins);
P_raw_te = phase_template_predict(Yp_phase_te, templates);
P_bias_te = phase_template_predict(Yp_bias_te, templates);
P_raw_oof = phase_template_predict(Yp_phase_oof, templates);
P_bias_oof = phase_template_predict(Yp_bias_oof, templates);
beta = choose_blend_beta(Y_delta(tr_idx), P_raw_oof, P_bias_oof, 0:0.1:1);
P_pred = blend_prediction_cells(P_raw_te, P_bias_te, beta);
lambda = summarize_lambda(model_phase.lambda);
end

function [P_pred, lambda, S_pred] = fit_predict_delay_template(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain)
model_phase = train_decoder_local(X(tr_idx), Y_phase(tr_idx), lambda_grid, lambda_mode);
Yp_phase_te = predict_wiener(model_phase, X(te_idx));
Yp_phase_te = smooth_phase_cells(Yp_phase_te, smooth_type, fs);

Yp_phase_oof = phase_oof_predictions(X, Y_phase, tr_idx, lambda_grid, smooth_type, fs, lambda_mode);
Y_delay = build_phase_delay_cells(Y_phase(tr_idx), Yp_phase_oof);
model_delay = train_decoder_local(X(tr_idx), Y_delay, lambda_grid, lambda_mode);
Yp_delay_te = predict_wiener(model_delay, X(te_idx));
Yp_delay_te = smooth_phase_cells(Yp_delay_te, smooth_type, fs);
Yp_delay_oof = target_oof_predictions(X(tr_idx), Y_delay, lambda_grid, smooth_type, fs, lambda_mode);

templates = fit_delta_templates(Y_phase(tr_idx), Y_delta(tr_idx), n_bins);
Yp_corr_te = apply_phase_delay_cells(Yp_phase_te, Yp_delay_te, phase_delay_gain);
Yp_corr_oof = apply_phase_delay_cells(Yp_phase_oof, Yp_delay_oof, phase_delay_gain);
P_raw_te = phase_template_predict(Yp_phase_te, templates);
P_corr_te = phase_template_predict(Yp_corr_te, templates);
P_raw_oof = phase_template_predict(Yp_phase_oof, templates);
P_corr_oof = phase_template_predict(Yp_corr_oof, templates);
beta = choose_blend_beta(Y_delta(tr_idx), P_raw_oof, P_corr_oof, 0:0.1:1);
P_pred = blend_prediction_cells(P_raw_te, P_corr_te, beta);
S_pred = blend_phase_state_by_joint(Yp_phase_te, Yp_corr_te, beta);
lambda = mean([summarize_lambda(model_phase.lambda), summarize_lambda(model_delay.lambda)], 'omitnan');
end

function [P_pred, lambda, S_pred] = fit_predict_delay_template_jitter(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain, phase_jitter_ms)
Y_phase_fit = jitter_phase_target_cells(Y_phase, fs, phase_jitter_ms);
model_phase = train_decoder_local(X(tr_idx), Y_phase_fit(tr_idx), lambda_grid, lambda_mode);
Yp_phase_te = predict_wiener(model_phase, X(te_idx));
Yp_phase_te = smooth_phase_cells(Yp_phase_te, smooth_type, fs);

Yp_phase_oof = phase_oof_predictions(X, Y_phase_fit, tr_idx, lambda_grid, smooth_type, fs, lambda_mode);
Y_delay = build_phase_delay_cells(Y_phase_fit(tr_idx), Yp_phase_oof);
model_delay = train_decoder_local(X(tr_idx), Y_delay, lambda_grid, lambda_mode);
Yp_delay_te = predict_wiener(model_delay, X(te_idx));
Yp_delay_te = smooth_phase_cells(Yp_delay_te, smooth_type, fs);
Yp_delay_oof = target_oof_predictions(X(tr_idx), Y_delay, lambda_grid, smooth_type, fs, lambda_mode);

templates = fit_delta_templates(Y_phase_fit(tr_idx), Y_delta(tr_idx), n_bins);
Yp_corr_te = apply_phase_delay_cells(Yp_phase_te, Yp_delay_te, phase_delay_gain);
Yp_corr_oof = apply_phase_delay_cells(Yp_phase_oof, Yp_delay_oof, phase_delay_gain);
P_raw_te = phase_template_predict(Yp_phase_te, templates);
P_corr_te = phase_template_predict(Yp_corr_te, templates);
P_raw_oof = phase_template_predict(Yp_phase_oof, templates);
P_corr_oof = phase_template_predict(Yp_corr_oof, templates);
beta = choose_blend_beta(Y_delta(tr_idx), P_raw_oof, P_corr_oof, 0:0.1:1);
P_pred = blend_prediction_cells(P_raw_te, P_corr_te, beta);
S_pred = blend_phase_state_by_joint(Yp_phase_te, Yp_corr_te, beta);
lambda = mean([summarize_lambda(model_phase.lambda), summarize_lambda(model_delay.lambda)], 'omitnan');
end

function [P_pred, lambda, S_pred] = fit_predict_delay_template_orient(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain)
model_phase = train_decoder_local(X(tr_idx), Y_phase(tr_idx), lambda_grid, lambda_mode);
Yp_phase_te = predict_wiener(model_phase, X(te_idx));
Yp_phase_te = smooth_phase_cells(Yp_phase_te, smooth_type, fs);

Yp_phase_oof = phase_oof_predictions(X, Y_phase, tr_idx, lambda_grid, smooth_type, fs, lambda_mode);
orient = fit_phase_orientation_cells(Y_phase(tr_idx), Yp_phase_oof);
Yp_phase_te = apply_phase_orientation_cells(Yp_phase_te, orient);
Yp_phase_oof = apply_phase_orientation_cells(Yp_phase_oof, orient);

Y_delay = build_phase_delay_cells(Y_phase(tr_idx), Yp_phase_oof);
model_delay = train_decoder_local(X(tr_idx), Y_delay, lambda_grid, lambda_mode);
Yp_delay_te = predict_wiener(model_delay, X(te_idx));
Yp_delay_te = smooth_phase_cells(Yp_delay_te, smooth_type, fs);
Yp_delay_oof = target_oof_predictions(X(tr_idx), Y_delay, lambda_grid, smooth_type, fs, lambda_mode);

templates = fit_delta_templates(Y_phase(tr_idx), Y_delta(tr_idx), n_bins);
Yp_corr_te = apply_phase_delay_cells(Yp_phase_te, Yp_delay_te, phase_delay_gain);
Yp_corr_oof = apply_phase_delay_cells(Yp_phase_oof, Yp_delay_oof, phase_delay_gain);
P_raw_te = phase_template_predict(Yp_phase_te, templates);
P_corr_te = phase_template_predict(Yp_corr_te, templates);
P_raw_oof = phase_template_predict(Yp_phase_oof, templates);
P_corr_oof = phase_template_predict(Yp_corr_oof, templates);
beta = choose_blend_beta(Y_delta(tr_idx), P_raw_oof, P_corr_oof, 0:0.1:1);
P_pred = blend_prediction_cells(P_raw_te, P_corr_te, beta);
S_pred = blend_phase_state_by_joint(Yp_phase_te, Yp_corr_te, beta);
lambda = mean([summarize_lambda(model_phase.lambda), summarize_lambda(model_delay.lambda)], 'omitnan');
end

function [P_pred, lambda, S_pred] = fit_predict_delay_template_osc(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain)
model_phase = train_decoder_local(X(tr_idx), Y_phase(tr_idx), lambda_grid, lambda_mode);
Yp_phase_te = predict_wiener(model_phase, X(te_idx));
Yp_phase_te = smooth_phase_cells(Yp_phase_te, smooth_type, fs);

Yp_phase_oof = phase_oof_predictions(X, Y_phase, tr_idx, lambda_grid, smooth_type, fs, lambda_mode);
omega = fit_phase_omega_cells(Y_phase(tr_idx));
templates = fit_delta_templates(Y_phase(tr_idx), Y_delta(tr_idx), n_bins);
beta = choose_phase_oscillator_beta(Y_delta(tr_idx), Yp_phase_oof, omega, templates, 0:0.1:0.9);
Yp_osc_oof = apply_phase_oscillator_cells(Yp_phase_oof, omega, beta);

Y_delay = build_phase_delay_cells(Y_phase(tr_idx), Yp_osc_oof);
model_delay = train_decoder_local(X(tr_idx), Y_delay, lambda_grid, lambda_mode);
Yp_delay_te = predict_wiener(model_delay, X(te_idx));
Yp_delay_te = smooth_phase_cells(Yp_delay_te, smooth_type, fs);
Yp_delay_oof = target_oof_predictions(X(tr_idx), Y_delay, lambda_grid, smooth_type, fs, lambda_mode);

Yp_osc_te = apply_phase_oscillator_cells(Yp_phase_te, omega, beta);
Yp_corr_te = apply_phase_delay_cells(Yp_osc_te, Yp_delay_te, phase_delay_gain);
Yp_corr_oof = apply_phase_delay_cells(Yp_osc_oof, Yp_delay_oof, phase_delay_gain);
P_osc_te = phase_template_predict(Yp_osc_te, templates);
P_corr_te = phase_template_predict(Yp_corr_te, templates);
P_osc_oof = phase_template_predict(Yp_osc_oof, templates);
P_corr_oof = phase_template_predict(Yp_corr_oof, templates);
beta_delay = choose_blend_beta(Y_delta(tr_idx), P_osc_oof, P_corr_oof, 0:0.1:1);
P_pred = blend_prediction_cells(P_osc_te, P_corr_te, beta_delay);
S_pred = blend_phase_state_by_joint(Yp_osc_te, Yp_corr_te, beta_delay);
lambda = mean([summarize_lambda(model_phase.lambda), summarize_lambda(model_delay.lambda)], 'omitnan');
end

function [P_pred, lambda, S_pred] = fit_predict_delay_template_kalman(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, lambda_mode, phase_delay_gain)
model_phase = train_decoder_local(X(tr_idx), Y_phase(tr_idx), lambda_grid, lambda_mode);
Yp_phase_te = predict_wiener(model_phase, X(te_idx));
Yp_phase_te = smooth_phase_cells(Yp_phase_te, smooth_type, fs);

Yp_phase_oof = phase_oof_predictions(X, Y_phase, tr_idx, lambda_grid, smooth_type, fs, lambda_mode);
kf_base = fit_phase_kalman_params_cells(Y_phase(tr_idx), Yp_phase_oof);
templates = fit_delta_templates(Y_phase(tr_idx), Y_delta(tr_idx), n_bins);
kf = choose_phase_kalman_scale(Y_delta(tr_idx), Yp_phase_oof, kf_base, templates, [0.25 0.5 1 2 4]);
Yp_kf_oof = apply_phase_kalman_cells(Yp_phase_oof, kf);

Y_delay = build_phase_delay_cells(Y_phase(tr_idx), Yp_kf_oof);
model_delay = train_decoder_local(X(tr_idx), Y_delay, lambda_grid, lambda_mode);
Yp_delay_te = predict_wiener(model_delay, X(te_idx));
Yp_delay_te = smooth_phase_cells(Yp_delay_te, smooth_type, fs);
Yp_delay_oof = target_oof_predictions(X(tr_idx), Y_delay, lambda_grid, smooth_type, fs, lambda_mode);

Yp_kf_te = apply_phase_kalman_cells(Yp_phase_te, kf);
Yp_corr_te = apply_phase_delay_cells(Yp_kf_te, Yp_delay_te, phase_delay_gain);
Yp_corr_oof = apply_phase_delay_cells(Yp_kf_oof, Yp_delay_oof, phase_delay_gain);
P_kf_te = phase_template_predict(Yp_kf_te, templates);
P_corr_te = phase_template_predict(Yp_corr_te, templates);
P_kf_oof = phase_template_predict(Yp_kf_oof, templates);
P_corr_oof = phase_template_predict(Yp_corr_oof, templates);
beta_delay = choose_blend_beta(Y_delta(tr_idx), P_kf_oof, P_corr_oof, 0:0.1:1);
P_pred = blend_prediction_cells(P_kf_te, P_corr_te, beta_delay);
S_pred = blend_phase_state_by_joint(Yp_kf_te, Yp_corr_te, beta_delay);
lambda = mean([summarize_lambda(model_phase.lambda), summarize_lambda(model_delay.lambda)], 'omitnan');
end

function [P_pred, lambda, S_pred] = fit_predict_template_residual(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, lambda_mode)
model_phase = train_decoder_local(X(tr_idx), Y_phase(tr_idx), lambda_grid, lambda_mode);
Yp_phase_te = predict_wiener(model_phase, X(te_idx));
Yp_phase_te = smooth_phase_cells(Yp_phase_te, smooth_type, fs);
Yp_phase_oof = phase_oof_predictions(X, Y_phase, tr_idx, lambda_grid, smooth_type, fs, lambda_mode);

templates = fit_delta_templates(Y_phase(tr_idx), Y_delta(tr_idx), n_bins);
P_true_train = phase_template_predict(Y_phase(tr_idx), templates);
Y_resid = subtract_prediction_cells(Y_delta(tr_idx), P_true_train);
model_resid = train_decoder_local(X(tr_idx), Y_resid, lambda_grid, lambda_mode);
R_te = predict_wiener(model_resid, X(te_idx));
R_te = smooth_phase_cells(R_te, smooth_type, fs);
R_oof = target_oof_predictions(X(tr_idx), Y_resid, lambda_grid, smooth_type, fs, lambda_mode);

P_raw_te = phase_template_predict(Yp_phase_te, templates);
P_raw_oof = phase_template_predict(Yp_phase_oof, templates);
P_resid_te = add_prediction_cells(P_raw_te, R_te);
P_resid_oof = add_prediction_cells(P_raw_oof, R_oof);
beta = choose_blend_beta(Y_delta(tr_idx), P_raw_oof, P_resid_oof, 0:0.1:1);
P_pred = blend_prediction_cells(P_raw_te, P_resid_te, beta);
S_pred = Yp_phase_te;
lambda = mean([summarize_lambda(model_phase.lambda), summarize_lambda(model_resid.lambda)], 'omitnan');
end

function [P_pred, lambda, S_pred] = fit_predict_template_residual_pca(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, n_bins, n_components, lambda_mode)
model_phase = train_decoder_local(X(tr_idx), Y_phase(tr_idx), lambda_grid, lambda_mode);
Yp_phase_te = predict_wiener(model_phase, X(te_idx));
Yp_phase_te = smooth_phase_cells(Yp_phase_te, smooth_type, fs);
Yp_phase_oof = phase_oof_predictions(X, Y_phase, tr_idx, lambda_grid, smooth_type, fs, lambda_mode);

templates = fit_delta_templates(Y_phase(tr_idx), Y_delta(tr_idx), n_bins);
P_true_train = phase_template_predict(Y_phase(tr_idx), templates);
Y_resid = subtract_prediction_cells(Y_delta(tr_idx), P_true_train);
[Y_resid_z, r_mu, r_sd] = standardize_cells_local(Y_resid, []);
[Score_train, basis] = residual_pca_scores_cells(Y_resid_z, n_components);
model_score = train_decoder_local(X(tr_idx), Score_train, lambda_grid, lambda_mode);
Score_te = predict_wiener(model_score, X(te_idx));
Score_oof = target_oof_predictions(X(tr_idx), Score_train, lambda_grid, 'none', fs, lambda_mode);

R_te = residual_pca_reconstruct_cells(Score_te, basis, r_mu, r_sd);
R_oof = residual_pca_reconstruct_cells(Score_oof, basis, r_mu, r_sd);
P_raw_te = phase_template_predict(Yp_phase_te, templates);
P_raw_oof = phase_template_predict(Yp_phase_oof, templates);
P_resid_te = add_prediction_cells(P_raw_te, R_te);
P_resid_oof = add_prediction_cells(P_raw_oof, R_oof);
beta = choose_blend_beta(Y_delta(tr_idx), P_raw_oof, P_resid_oof, 0:0.1:1);
P_pred = blend_prediction_cells(P_raw_te, P_resid_te, beta);
S_pred = Yp_phase_te;
lambda = mean([summarize_lambda(model_phase.lambda), summarize_lambda(model_score.lambda)], 'omitnan');
end

function [P_pred, lambda, S_pred] = fit_predict_delay_harmonic(X, Y_phase, Y_delta, tr_idx, te_idx, lambda_grid, smooth_type, fs, harmonic_k, lambda_mode, phase_delay_gain)
model_phase = train_decoder_local(X(tr_idx), Y_phase(tr_idx), lambda_grid, lambda_mode);
Yp_phase_te = predict_wiener(model_phase, X(te_idx));
Yp_phase_te = smooth_phase_cells(Yp_phase_te, smooth_type, fs);

Yp_phase_oof = phase_oof_predictions(X, Y_phase, tr_idx, lambda_grid, smooth_type, fs, lambda_mode);
Y_delay = build_phase_delay_cells(Y_phase(tr_idx), Yp_phase_oof);
model_delay = train_decoder_local(X(tr_idx), Y_delay, lambda_grid, lambda_mode);
Yp_delay = predict_wiener(model_delay, X(te_idx));
Yp_delay = smooth_phase_cells(Yp_delay, smooth_type, fs);

Yp_phase_corr = apply_phase_delay_cells(Yp_phase_te, Yp_delay, phase_delay_gain);
H = fit_harmonic_models(Y_phase(tr_idx), Y_delta(tr_idx), harmonic_k);
P_pred = harmonic_predict(Yp_phase_corr, H, harmonic_k);
S_pred = Yp_phase_corr;
lambda = mean([summarize_lambda(model_phase.lambda), summarize_lambda(model_delay.lambda)], 'omitnan');
end

function Yp_oof = phase_oof_predictions(X, Y_phase, tr_idx, lambda_grid, smooth_type, fs, lambda_mode, inner_k)
if nargin < 8 || isempty(inner_k)
    inner_k = 3;
end
Yp_oof = target_oof_predictions(X(tr_idx), Y_phase(tr_idx), lambda_grid, smooth_type, fs, lambda_mode, inner_k);
end

function Yp_oof = target_oof_predictions(X_sub, Y_sub, lambda_grid, smooth_type, fs, lambda_mode, inner_k)
if nargin < 7 || isempty(inner_k)
    inner_k = 3;
end
n = numel(X_sub);
if n < 2
    error('target_oof_predictions requires at least two training trials');
end
K = min(max(2, round(double(inner_k))), n);
rng(20260516, 'twister');
perm = randperm(n);
fold_id = zeros(1, n);
for k = 1:K
    fold_id(perm(k:K:end)) = k;
end
Yp_oof = cell(size(X_sub));
for k = 1:K
    inner_tr = find(fold_id ~= k);
    inner_te_pos = find(fold_id == k);
    model = train_decoder_local(X_sub(inner_tr), Y_sub(inner_tr), lambda_grid, lambda_mode, inner_k);
    Yp = predict_wiener(model, X_sub(inner_te_pos));
    Yp = smooth_phase_cells(Yp, smooth_type, fs);
    for ii = 1:numel(inner_te_pos)
        Yp_oof{inner_te_pos(ii)} = Yp{ii};
    end
end
end

function Y_delay = build_phase_delay_cells(Y_true, Y_pred)
Y_delay = cell(size(Y_true));
for k = 1:numel(Y_true)
    n = min(size(Y_true{k}, 1), size(Y_pred{k}, 1));
    D = nan(n, size(Y_true{k}, 2));
    J = floor(size(Y_true{k}, 2) / 2);
    for j = 1:J
        c = 2*j - 1;
        s = 2*j;
        phi_true = atan2(Y_true{k}(1:n, s), Y_true{k}(1:n, c));
        phi_pred = atan2(Y_pred{k}(1:n, s), Y_pred{k}(1:n, c));
        dphi = angle(exp(1i * (phi_true - phi_pred)));
        D(:, c) = cos(dphi);
        D(:, s) = sin(dphi);
    end
    Y_delay{k} = D;
end
end

function Y_jit = jitter_phase_target_cells(Y_phase, fs, jitter_ms)
Y_jit = Y_phase;
win = max(0, round((double(jitter_ms) / 1000) * double(fs)));
if win <= 0
    return;
end
for k = 1:numel(Y_phase)
    Y = Y_phase{k};
    if isempty(Y)
        continue;
    end
    Z = nan(size(Y));
    J = floor(size(Y, 2) / 2);
    for j = 1:J
        c = 2*j - 1;
        s = 2*j;
        C = Y(:, c);
        S = Y(:, s);
        if all(~isfinite(C)) || all(~isfinite(S))
            continue;
        end
        Cj = movmean(C, [win win], 'omitnan', 'Endpoints', 'shrink');
        Sj = movmean(S, [win win], 'omitnan', 'Endpoints', 'shrink');
        rr = sqrt(Cj.^2 + Sj.^2);
        good = isfinite(rr) & rr > eps;
        Z(good, c) = Cj(good) ./ rr(good);
        Z(good, s) = Sj(good) ./ rr(good);
        bad = ~good & isfinite(C) & isfinite(S);
        Z(bad, c) = C(bad);
        Z(bad, s) = S(bad);
    end
    Y_jit{k} = Z;
end
end

function omega = fit_phase_omega_cells(Y_phase)
J = floor(size(Y_phase{find_first_nonempty_cell(Y_phase)}, 2) / 2);
omega = zeros(1, J);
for j = 1:J
    c = 2*j - 1;
    s = 2*j;
    dphi_all = [];
    for k = 1:numel(Y_phase)
        if isempty(Y_phase{k}) || size(Y_phase{k}, 2) < s
            continue;
        end
        phi = atan2(Y_phase{k}(:, s), Y_phase{k}(:, c));
        good = isfinite(phi);
        if sum(good) < 3
            continue;
        end
        phi = unwrap(phi(good));
        dphi = diff(phi);
        dphi = dphi(isfinite(dphi) & abs(dphi) < pi / 2);
        dphi_all = [dphi_all; dphi(:)]; %#ok<AGROW>
    end
    if ~isempty(dphi_all)
        omega(j) = median(dphi_all, 'omitnan');
    end
end
omega(~isfinite(omega)) = 0;
end

function beta = choose_phase_oscillator_beta(Y_delta, Yp_phase, omega, templates, beta_grid)
J = size(templates, 2);
beta = zeros(1, J);
for j = 1:J
    best_r = -Inf;
    best_b = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        Yp_osc = apply_phase_oscillator_cells(Yp_phase, omega, repmat(b, 1, J));
        P_osc = phase_template_predict(Yp_osc, templates);
        [r, ~] = eval_one(Y_delta, P_osc, j);
        if isfinite(r) && r > best_r
            best_r = r;
            best_b = b;
        end
    end
    beta(j) = best_b;
end
end

function Y_osc = apply_phase_oscillator_cells(Y_phase_pred, omega, beta)
Y_osc = cell(size(Y_phase_pred));
for k = 1:numel(Y_phase_pred)
    Y = Y_phase_pred{k};
    if isempty(Y)
        Y_osc{k} = Y;
        continue;
    end
    C = nan(size(Y));
    J = min([floor(size(Y, 2) / 2), numel(omega), numel(beta)]);
    for j = 1:J
        c = 2*j - 1;
        s = 2*j;
        phi_m = atan2(Y(:, s), Y(:, c));
        n = numel(phi_m);
        if n == 0
            continue;
        end
        phi_c = nan(n, 1);
        first = find(isfinite(phi_m), 1);
        if isempty(first)
            continue;
        end
        phi_c(1:first-1) = phi_m(first);
        phi_c(first) = phi_m(first);
        for t = (first + 1):n
            phi_pred = phi_c(t - 1) + omega(j);
            if isfinite(phi_m(t))
                z = (1 - beta(j)) * exp(1i * phi_m(t)) + beta(j) * exp(1i * phi_pred);
                if abs(z) > eps
                    phi_c(t) = angle(z);
                else
                    phi_c(t) = phi_pred;
                end
            else
                phi_c(t) = phi_pred;
            end
        end
        C(:, c) = cos(phi_c);
        C(:, s) = sin(phi_c);
    end
    Y_osc{k} = C;
end
end

function kf = fit_phase_kalman_params_cells(Y_phase, Y_pred)
J = floor(size(Y_phase{find_first_nonempty_cell(Y_phase)}, 2) / 2);
omega = zeros(1, J);
q = 0.01 * ones(1, J);
r = 0.5 * ones(1, J);
for j = 1:J
    c = 2*j - 1;
    s = 2*j;
    dphi_all = [];
    err_all = [];
    for k = 1:numel(Y_phase)
        if isempty(Y_phase{k}) || isempty(Y_pred{k}) || size(Y_phase{k}, 2) < s || size(Y_pred{k}, 2) < s
            continue;
        end
        n = min(size(Y_phase{k}, 1), size(Y_pred{k}, 1));
        if n < 3
            continue;
        end
        phi_true = atan2(Y_phase{k}(1:n, s), Y_phase{k}(1:n, c));
        phi_pred = atan2(Y_pred{k}(1:n, s), Y_pred{k}(1:n, c));
        good = isfinite(phi_true) & isfinite(phi_pred);
        if sum(good) >= 3
            pt = unwrap(phi_true(good));
            dphi = diff(pt);
            dphi = dphi(isfinite(dphi) & abs(dphi) < pi / 2);
            dphi_all = [dphi_all; dphi(:)]; %#ok<AGROW>
            err = angle(exp(1i * (phi_true(good) - phi_pred(good))));
            err_all = [err_all; err(:)]; %#ok<AGROW>
        end
    end
    if ~isempty(dphi_all)
        omega(j) = median(dphi_all, 'omitnan');
        dq = dphi_all - omega(j);
        q(j) = circular_variance_proxy(dq, 0.005, 0.5);
    end
    if ~isempty(err_all)
        r(j) = circular_variance_proxy(err_all, 0.02, 2.0);
    end
end
kf = struct('omega', omega, 'q', q, 'r', r);
end

function v = circular_variance_proxy(err, lo, hi)
err = err(isfinite(err));
if isempty(err)
    v = hi;
    return;
end
v = 1 - abs(mean(exp(1i * err)));
if ~isfinite(v)
    v = hi;
end
v = min(max(v, lo), hi);
end

function kf = choose_phase_kalman_scale(Y_delta, Yp_phase, kf_base, templates, scale_grid)
J = size(templates, 2);
kf = kf_base;
scale = ones(1, J);
for j = 1:J
    best_r = -Inf;
    best_s = 1;
    for si = 1:numel(scale_grid)
        tmp = kf_base;
        tmp.q(j) = kf_base.q(j) * scale_grid(si);
        Yp_kf = apply_phase_kalman_cells(Yp_phase, tmp);
        P_kf = phase_template_predict(Yp_kf, templates);
        [rj, ~] = eval_one(Y_delta, P_kf, j);
        if isfinite(rj) && rj > best_r
            best_r = rj;
            best_s = scale_grid(si);
        end
    end
    scale(j) = best_s;
end
kf.q = kf_base.q .* scale;
end

function Y_kf = apply_phase_kalman_cells(Y_phase_pred, kf)
Y_kf = cell(size(Y_phase_pred));
for k = 1:numel(Y_phase_pred)
    Y = Y_phase_pred{k};
    if isempty(Y)
        Y_kf{k} = Y;
        continue;
    end
    C = nan(size(Y));
    J = min([floor(size(Y, 2) / 2), numel(kf.omega), numel(kf.q), numel(kf.r)]);
    for j = 1:J
        c = 2*j - 1;
        s = 2*j;
        z = atan2(Y(:, s), Y(:, c));
        n = numel(z);
        if n == 0
            continue;
        end
        phi = nan(n, 1);
        first = find(isfinite(z), 1);
        if isempty(first)
            continue;
        end
        phi(1:first-1) = z(first);
        phi(first) = z(first);
        P = max(kf.r(j), 0.02);
        Q = max(kf.q(j), 0.005);
        R = max(kf.r(j), 0.02);
        for t = (first + 1):n
            phi_pred = phi(t - 1) + kf.omega(j);
            P_pred = P + Q;
            if isfinite(z(t))
                K = P_pred / (P_pred + R);
                innov = angle(exp(1i * (z(t) - phi_pred)));
                phi(t) = phi_pred + K * innov;
                P = max((1 - K) * P_pred, 1e-4);
            else
                phi(t) = phi_pred;
                P = P_pred;
            end
        end
        C(:, c) = cos(phi);
        C(:, s) = sin(phi);
    end
    Y_kf{k} = C;
end
end

function bias = fit_phase_bias_cells(Y_true, Y_pred)
J = floor(size(Y_true{find_first_nonempty_cell(Y_true)}, 2) / 2);
bias = zeros(1, J);
for j = 1:J
    c = 2*j - 1;
    s = 2*j;
    acc = 0;
    for k = 1:numel(Y_true)
        n = min(size(Y_true{k}, 1), size(Y_pred{k}, 1));
        if n < 1
            continue;
        end
        phi_true = atan2(Y_true{k}(1:n, s), Y_true{k}(1:n, c));
        phi_pred = atan2(Y_pred{k}(1:n, s), Y_pred{k}(1:n, c));
        d = exp(1i * angle(exp(1i * (phi_true - phi_pred))));
        good = isfinite(real(d)) & isfinite(imag(d));
        if any(good)
            acc = acc + sum(d(good));
        end
    end
    if abs(acc) > 0
        bias(j) = angle(acc);
    end
end
end

function idx = find_first_nonempty_cell(C)
idx = 1;
for k = 1:numel(C)
    if ~isempty(C{k})
        idx = k;
        return;
    end
end
end

function Y_corr = apply_phase_bias_cells(Y_phase_pred, bias, phase_delay_gain)
Y_corr = cell(size(Y_phase_pred));
for k = 1:numel(Y_phase_pred)
    Y = Y_phase_pred{k};
    C = nan(size(Y));
    J = min(floor(size(Y, 2) / 2), numel(bias));
    for j = 1:J
        c = 2*j - 1;
        s = 2*j;
        phi = atan2(Y(:, s), Y(:, c));
        pc = phi + phase_delay_gain * bias(j);
        C(:, c) = cos(pc);
        C(:, s) = sin(pc);
    end
    Y_corr{k} = C;
end
end

function orient = fit_phase_orientation_cells(Y_true, Y_pred)
J = floor(size(Y_true{find_first_nonempty_cell(Y_true)}, 2) / 2);
orient = repmat(eye(2), 1, 1, J);
for j = 1:J
    c = 2*j - 1;
    s = 2*j;
    X = [];
    Y = [];
    for k = 1:numel(Y_true)
        n = min(size(Y_true{k}, 1), size(Y_pred{k}, 1));
        if n < 5
            continue;
        end
        xp = Y_pred{k}(1:n, [c s]);
        yt = Y_true{k}(1:n, [c s]);
        good = all(isfinite(xp), 2) & all(isfinite(yt), 2);
        X = [X; xp(good, :)]; %#ok<AGROW>
        Y = [Y; yt(good, :)]; %#ok<AGROW>
    end
    if size(X, 1) < 10
        continue;
    end
    [U, ~, V] = svd(X' * Y, 'econ');
    orient(:, :, j) = U * V';
end
end

function Y_corr = apply_phase_orientation_cells(Y_phase_pred, orient)
Y_corr = cell(size(Y_phase_pred));
for k = 1:numel(Y_phase_pred)
    Y = Y_phase_pred{k};
    C = nan(size(Y));
    J = min(floor(size(Y, 2) / 2), size(orient, 3));
    for j = 1:J
        cols = 2*j - 1:2*j;
        C(:, cols) = Y(:, cols) * orient(:, :, j);
        rr = sqrt(sum(C(:, cols).^2, 2));
        good = isfinite(rr) & rr > eps;
        C(good, cols) = C(good, cols) ./ rr(good);
    end
    Y_corr{k} = C;
end
end

function Y_corr = apply_phase_delay_cells(Y_phase_pred, Y_delay_pred, phase_delay_gain)
Y_corr = cell(size(Y_phase_pred));
for k = 1:numel(Y_phase_pred)
    n = min(size(Y_phase_pred{k}, 1), size(Y_delay_pred{k}, 1));
    C = nan(n, size(Y_phase_pred{k}, 2));
    J = floor(size(Y_phase_pred{k}, 2) / 2);
    for j = 1:J
        c = 2*j - 1;
        s = 2*j;
        phi = atan2(Y_phase_pred{k}(1:n, s), Y_phase_pred{k}(1:n, c));
        dphi = atan2(Y_delay_pred{k}(1:n, s), Y_delay_pred{k}(1:n, c));
        pc = phi + phase_delay_gain * dphi;
        C(:, c) = cos(pc);
        C(:, s) = sin(pc);
    end
    Y_corr{k} = C;
end
end

function beta = choose_blend_beta(Y_true, P_raw, P_corr, beta_grid)
J = size(Y_true{1}, 2);
beta = zeros(1, J);
for j = 1:J
    best_r = -Inf;
    best_b = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        P_mix = blend_prediction_cells(P_raw, P_corr, repmat(b, 1, J));
        [r, ~] = eval_one(Y_true, P_mix, j);
        if isfinite(r) && r > best_r
            best_r = r;
            best_b = b;
        end
    end
    beta(j) = best_b;
end
end

function P_mix = blend_prediction_cells(P_raw, P_corr, beta)
P_mix = cell(size(P_raw));
for k = 1:numel(P_raw)
    n = min(size(P_raw{k}, 1), size(P_corr{k}, 1));
    J = min(size(P_raw{k}, 2), size(P_corr{k}, 2));
    M = nan(n, J);
    for j = 1:J
        b = beta(min(j, numel(beta)));
        M(:, j) = (1 - b) * P_raw{k}(1:n, j) + b * P_corr{k}(1:n, j);
    end
    P_mix{k} = M;
end
end

function P_out = add_prediction_cells(P_base, P_add)
P_out = cell(size(P_base));
for k = 1:numel(P_base)
    n = min(size(P_base{k}, 1), size(P_add{k}, 1));
    J = min(size(P_base{k}, 2), size(P_add{k}, 2));
    P_out{k} = P_base{k}(1:n, 1:J) + P_add{k}(1:n, 1:J);
end
end

function P_out = subtract_prediction_cells(P_base, P_sub)
P_out = cell(size(P_base));
for k = 1:numel(P_base)
    n = min(size(P_base{k}, 1), size(P_sub{k}, 1));
    J = min(size(P_base{k}, 2), size(P_sub{k}, 2));
    P_out{k} = P_base{k}(1:n, 1:J) - P_sub{k}(1:n, 1:J);
end
end

function [Score_cell, basis] = residual_pca_scores_cells(Y_resid_z, n_components)
Y_all = vertcat(Y_resid_z{:});
Y_all(~isfinite(Y_all)) = 0;
if isempty(Y_all)
    basis = [];
    Score_cell = cell(size(Y_resid_z));
    return;
end
[~, ~, V] = svd(Y_all, 'econ');
K = min([max(1, round(n_components)), size(V, 2)]);
basis = V(:, 1:K);
Score_cell = cell(size(Y_resid_z));
for k = 1:numel(Y_resid_z)
    Yk = Y_resid_z{k};
    Yk(~isfinite(Yk)) = 0;
    Score_cell{k} = Yk * basis;
end
end

function Y_resid = residual_pca_reconstruct_cells(Score_cell, basis, r_mu, r_sd)
Y_resid = cell(size(Score_cell));
if isempty(basis)
    for k = 1:numel(Score_cell)
        Y_resid{k} = nan(size(Score_cell{k}, 1), numel(r_mu));
    end
    return;
end
for k = 1:numel(Score_cell)
    Rz = Score_cell{k} * basis';
    Y_resid{k} = Rz .* r_sd + r_mu;
end
end

function S_mix = blend_phase_state_by_joint(S_raw, S_corr, beta)
S_mix = cell(size(S_raw));
for k = 1:numel(S_raw)
    n = min(size(S_raw{k}, 1), size(S_corr{k}, 1));
    D = min(size(S_raw{k}, 2), size(S_corr{k}, 2));
    M = nan(n, D);
    J = floor(D / 2);
    for j = 1:J
        b = beta(min(j, numel(beta)));
        cols = 2*j-1:2*j;
        M(:, cols) = (1 - b) * S_raw{k}(1:n, cols) + b * S_corr{k}(1:n, cols);
    end
    if 2 * J < D
        M(:, 2*J+1:D) = S_raw{k}(1:n, 2*J+1:D);
    end
    S_mix{k} = M;
end
end

function [X_phase_train, Y_phase_train] = append_aux_phase_cells(X_main, Y_main, X_aux_phase, Y_aux_phase)
if isempty(X_aux_phase)
    X_phase_train = X_main;
    Y_phase_train = Y_main;
else
    X_phase_train = [X_main, X_aux_phase];
    Y_phase_train = [Y_main, Y_aux_phase];
end
end

function [A_oof, P_oof, S_oof] = inner_oof_predictions(X_delta, X_phase, Y_delta, Y_phase, tr_idx, lambda_grid, smooth_type, fs, n_bins, phase_recon, harmonic_k, inner_k, outer_fold, lambda_mode, phase_delay_gain, phase_lo, phase_hi, phase_select_margin, phase_jitter_ms, X_aux_phase, Y_aux_phase, verbose_progress)
if nargin < 20
    X_aux_phase = {};
    Y_aux_phase = {};
end
if nargin < 22
    verbose_progress = false;
end
A_oof = cell(size(tr_idx));
P_oof = cell(size(tr_idx));
S_oof = cell(size(tr_idx));
n = numel(tr_idx);
if n < 2
    error('inner_oof_predictions requires at least two training trials');
end
K = min(max(2, round(double(inner_k))), n);
if logical(verbose_progress)
    fprintf('  train OOF angle+phase: K=%d, train trials=%d\n', K, n);
    drawnow;
end
rng(20260512 + outer_fold, 'twister');
perm = randperm(n);
fold_id = zeros(1, n);
for k = 1:K
    fold_id(perm(k:K:end)) = k;
end
for k = 1:K
    inner_tic = tic;
    inner_tr = tr_idx(fold_id ~= k);
    inner_te_pos = find(fold_id == k);
    inner_te = tr_idx(inner_te_pos);
    if logical(verbose_progress)
        fprintf('    [%s] inner OOF %d/%d start | train=%d test=%d\n', ...
            datestr(now, 'HH:MM:SS'), k, K, numel(inner_tr), numel(inner_te));
        drawnow;
    end
    A_val = fit_predict_angle(X_delta, Y_delta, inner_tr, inner_te, lambda_grid, lambda_mode, inner_k);
    [P_val, ~, S_val] = fit_predict_phase_recon(X_phase, Y_phase, Y_delta, inner_tr, inner_te, ...
        lambda_grid, smooth_type, fs, n_bins, phase_recon, harmonic_k, lambda_mode, ...
        inner_k, phase_delay_gain, phase_lo, phase_hi, phase_select_margin, phase_jitter_ms, ...
        X_aux_phase, Y_aux_phase);
    for ii = 1:numel(inner_te_pos)
        A_oof{inner_te_pos(ii)} = A_val{ii};
        P_oof{inner_te_pos(ii)} = P_val{ii};
        if ~isempty(S_val)
            S_oof{inner_te_pos(ii)} = S_val{ii};
        end
    end
    if logical(verbose_progress)
        fprintf('    [%s] inner OOF %d/%d done | elapsed=%.1fs\n', ...
            datestr(now, 'HH:MM:SS'), k, K, toc(inner_tic));
        drawnow;
    end
end
end

function A_oof = angle_oof_predictions(X_delta, Y_delta, tr_idx, lambda_grid, inner_k, outer_fold, lambda_mode, verbose_progress)
if nargin < 8
    verbose_progress = false;
end
A_oof = cell(size(tr_idx));
n = numel(tr_idx);
if n < 2
    error('angle_oof_predictions requires at least two training trials');
end
K = min(max(2, round(double(inner_k))), n);
if logical(verbose_progress)
    fprintf('  train OOF angle only: K=%d, train trials=%d\n', K, n);
    drawnow;
end
rng(20260518 + outer_fold, 'twister');
perm = randperm(n);
fold_id = zeros(1, n);
for k = 1:K
    fold_id(perm(k:K:end)) = k;
end
for k = 1:K
    inner_tic = tic;
    inner_tr = tr_idx(fold_id ~= k);
    inner_te_pos = find(fold_id == k);
    inner_te = tr_idx(inner_te_pos);
    if logical(verbose_progress)
        fprintf('    [%s] inner OOF %d/%d start | train=%d test=%d\n', ...
            datestr(now, 'HH:MM:SS'), k, K, numel(inner_tr), numel(inner_te));
        drawnow;
    end
    A_val = fit_predict_angle(X_delta, Y_delta, inner_tr, inner_te, lambda_grid, lambda_mode, inner_k);
    for ii = 1:numel(inner_te_pos)
        A_oof{inner_te_pos(ii)} = A_val{ii};
    end
    if logical(verbose_progress)
        fprintf('    [%s] inner OOF %d/%d done | elapsed=%.1fs\n', ...
            datestr(now, 'HH:MM:SS'), k, K, toc(inner_tic));
        drawnow;
    end
end
end

function model = train_decoder_local(X_cell, Y_cell, lambda_grid, lambda_mode, inner_k)
if nargin < 5 || isempty(inner_k)
    inner_k = 5;
end
if strcmpi(char(lambda_mode), 'scalar')
    model = train_wiener(X_cell, Y_cell, lambda_grid, inner_k);
elseif strcmpi(char(lambda_mode), 'per_output')
    model = train_wiener_per_output_local(X_cell, Y_cell, lambda_grid, inner_k);
elseif strcmpi(char(lambda_mode), 'robust')
    model = train_wiener_robust_local(X_cell, Y_cell, lambda_grid, inner_k);
elseif strcmpi(char(lambda_mode), 'bagged')
    model = train_wiener_bagged_local(X_cell, Y_cell, lambda_grid, inner_k);
elseif strcmpi(char(lambda_mode), 'rrr')
    model = train_wiener_rrr(X_cell, Y_cell, lambda_grid, inner_k);
else
    error('lambda_mode: unsupported value');
end
end

function model = train_decoder_fixed_lambda_local(X_cell, Y_cell, lambda, lambda_mode)
if ~strcmpi(char(lambda_mode), 'scalar')
    error('fixed-lambda training currently requires lambda_mode=scalar');
end
model = train_wiener_fixed_scalar_local(X_cell, Y_cell, lambda);
end

function model = train_wiener_fixed_scalar_local(X_cell, Y_cell, lambda)
P = size(X_cell{1}, 1);
J = size(Y_cell{1}, 2);
X_all = horzcat(X_cell{:});
Y_all = vertcat(Y_cell{:});
N_total = size(X_all, 2);
assert(size(Y_all, 1) == N_total, 'X and Y trial-sample counts mismatch');
joint_active = false(1, J);
for j = 1:J
    joint_active(j) = ~all(isnan(Y_all(:, j)));
end
X_aug = [X_all; ones(1, N_total)];
P1 = P + 1;
A = X_aug * X_aug';
A(1:P1+1:end) = A(1:P1+1:end) + double(lambda);
B = X_aug * Y_all;
W = A \ B;
model = struct();
model.W = W;
model.lambda = double(lambda);
model.cv_mse = NaN;
model.lambda_grid = double(lambda);
model.P = P;
model.J = J;
model.joint_active = joint_active;
end

function model = train_wiener_bagged_local(X_cell, Y_cell, lambda_grid, inner_k)
base = train_wiener(X_cell, Y_cell, lambda_grid, inner_k);
P = size(X_cell{1}, 1);
J = size(Y_cell{1}, 2);
X_all = horzcat(X_cell{:});
Y_all = vertcat(Y_cell{:});
N_total = size(X_all, 2);
assert(size(Y_all, 1) == N_total, 'X and Y trial-sample counts mismatch');
X_aug = [X_all; ones(1, N_total)];
P1 = P + 1;
ridge = max(double(base.lambda), 0);
rng(20260520);
n_bags = 11;
n_boot = max(P1 + 5, round(0.75 * N_total));
W_acc = base.W;
n_acc = 1;
for b = 1:n_bags
    idx = randi(N_total, [1, n_boot]);
    Xb = X_aug(:, idx);
    Yb = Y_all(idx, :);
    A = Xb * Xb';
    A(1:P1+1:end) = A(1:P1+1:end) + ridge;
    B = Xb * Yb;
    Wb = A \ B;
    if all(isfinite(Wb(:)))
        W_acc = W_acc + Wb;
        n_acc = n_acc + 1;
    end
end
model = base;
model.W = W_acc ./ max(1, n_acc);
model.lambda_mode = 'bagged';
end

function model = train_wiener_robust_local(X_cell, Y_cell, lambda_grid, inner_k)
base = train_wiener(X_cell, Y_cell, lambda_grid, inner_k);
P = size(X_cell{1}, 1);
J = size(Y_cell{1}, 2);
X_all = horzcat(X_cell{:});
Y_all = vertcat(Y_cell{:});
N_total = size(X_all, 2);
assert(size(Y_all, 1) == N_total, 'X and Y trial-sample counts mismatch');
X_aug = [X_all; ones(1, N_total)];
P1 = P + 1;
W = base.W;
ridge = max(double(base.lambda), 0);
for j = 1:J
    y = Y_all(:, j);
    valid = isfinite(y);
    if sum(valid) < P1 || all(~isfinite(W(:, j)))
        continue;
    end
    Xj = X_aug(:, valid);
    yj = y(valid);
    w = W(:, j);
    for iter = 1:5
        resid = Xj' * w - yj;
        sig = 1.4826 * median(abs(resid - median(resid, 'omitnan')), 'omitnan');
        if ~isfinite(sig) || sig <= eps
            sig = std(resid, 0, 'omitnan');
        end
        if ~isfinite(sig) || sig <= eps
            break;
        end
        c = 1.5 * sig;
        wt = ones(size(resid));
        big = abs(resid) > c;
        wt(big) = c ./ max(abs(resid(big)), eps);
        wt(~isfinite(wt) | wt < 0.05) = 0.05;
        sw = sqrt(wt(:)');
        Xw = Xj .* sw;
        yw = yj(:)' .* sw;
        A = Xw * Xw';
        reg = ridge * eye(size(A));
        reg(end, end) = 0;
        w_new = (A + reg) \ (Xw * yw');
        if norm(w_new - w) <= 1e-5 * max(1, norm(w))
            w = w_new;
            break;
        end
        w = w_new;
    end
    W(:, j) = w;
end
model = base;
model.W = W;
model.lambda_mode = 'robust';
end

function model = train_wiener_per_output_local(X_cell, Y_cell, lambda_grid, inner_k)
if nargin < 4 || isempty(inner_k)
    inner_k = 5;
end
P = size(X_cell{1}, 1);
J = size(Y_cell{1}, 2);
X_all = horzcat(X_cell{:});
Y_all = vertcat(Y_cell{:});
N_total = size(X_all, 2);
assert(size(Y_all, 1) == N_total, 'X and Y trial-sample counts mismatch');

joint_active = false(1, J);
for j = 1:J
    joint_active(j) = any(isfinite(Y_all(:, j)));
end

rng(20260505);
inner_K = min(max(1, round(double(inner_k))), N_total);
perm = randperm(N_total);
inner_fold = zeros(N_total, 1);
for k = 1:inner_K
    inner_fold(perm(k:inner_K:end)) = k;
end

X_aug = [X_all; ones(1, N_total)];
P1 = P + 1;
L_lam = numel(lambda_grid);
cv_mse = nan(L_lam, J);
lambda = nan(1, J);

for j = find(joint_active)
    sse = zeros(L_lam, 1);
    nobs = zeros(L_lam, 1);
    for k = 1:inner_K
        tr = inner_fold ~= k;
        te = inner_fold == k;
        ytr = Y_all(tr, j);
        yte = Y_all(te, j);
        tr_valid = isfinite(ytr);
        te_valid = isfinite(yte);
        if sum(tr_valid) < P1 || sum(te_valid) < 10, continue; end
        Xt = X_aug(:, tr);
        Xe = X_aug(:, te);
        Xt = Xt(:, tr_valid);
        ytr = ytr(tr_valid);
        Xe = Xe(:, te_valid);
        yte = yte(te_valid);
        A0 = Xt * Xt';
        XY = Xt * ytr;
        diag_idx = 1:P1+1:numel(A0);
        for li = 1:L_lam
            A = A0;
            A(diag_idx) = A(diag_idx) + lambda_grid(li);
            w = A \ XY;
            r = Xe' * w - yte;
            sse(li) = sse(li) + sum(r .^ 2);
            nobs(li) = nobs(li) + numel(r);
        end
    end
    cv_mse(:, j) = sse ./ nobs;
    [~, best_li] = min(cv_mse(:, j));
    lambda(j) = lambda_grid(best_li);
end

W = nan(P1, J);
for j = find(joint_active)
    valid = isfinite(Y_all(:, j));
    if sum(valid) < P1, continue; end
    Xj = X_aug(:, valid);
    yj = Y_all(valid, j);
    A = Xj * Xj';
    A(1:P1+1:end) = A(1:P1+1:end) + lambda(j);
    W(:, j) = A \ (Xj * yj);
end

model = struct();
model.W = W;
model.lambda = lambda;
model.cv_mse = cv_mse;
model.lambda_grid = lambda_grid;
model.P = P;
model.J = J;
model.joint_active = joint_active;
end

function lambda = summarize_lambda(lambda_in)
if isscalar(lambda_in)
    lambda = lambda_in;
else
    lambda = median(lambda_in, 'omitnan');
end
end

function alpha = choose_alpha(Y_true, A_pred, P_pred, alpha_grid, J)
alpha = zeros(1, J);
for j = 1:J
    best_r = -Inf;
    best_a = 0;
    for ai = 1:numel(alpha_grid)
        a = alpha_grid(ai);
        y = [];
        yp = [];
        for k = 1:numel(Y_true)
            if isempty(A_pred{k}) || isempty(P_pred{k}), continue; end
            n = min([size(Y_true{k}, 1), size(A_pred{k}, 1), size(P_pred{k}, 1)]);
            y = [y; Y_true{k}(1:n, j)]; %#ok<AGROW>
            ypa = (1 - a) * A_pred{k}(1:n, j) + a * P_pred{k}(1:n, j);
            yp = [yp; ypa]; %#ok<AGROW>
        end
        r = safe_corr_local(y, yp);
        if isfinite(r) && r > best_r
            best_r = r;
            best_a = a;
        end
    end
    alpha(j) = best_a;
end
end

function Y_fused = fuse_cells(A_pred, P_pred, alpha)
Y_fused = cell(size(A_pred));
for k = 1:numel(A_pred)
    n = min(size(A_pred{k}, 1), size(P_pred{k}, 1));
    J = size(A_pred{k}, 2);
    Yk = nan(n, J);
    for j = 1:J
        Yk(:, j) = (1 - alpha(j)) * A_pred{k}(1:n, j) + alpha(j) * P_pred{k}(1:n, j);
    end
    Y_fused{k} = Yk;
end
end

function [C_pred, post_info] = post_calibrate_predictions(post_mode, A_oof, P_oof, F_oof, S_oof, Y_train, A_te, P_te, F_te, S_te, n_components, post_basis_components, lambda, post_lambda_grid, guard_margin, post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only, post_inner_k)
if nargin < 20 || isempty(post_inner_k)
    post_inner_k = 3;
end
mode = lower(char(post_mode));
post_info = struct('mode', mode);
if strcmp(mode, 'none')
    C_pred = {};
    return;
elseif strcmp(mode, 'pls_calib_guardcv')
    C_pred = post_calibrate_guardcv(A_oof, P_oof, F_oof, Y_train, A_te, P_te, F_te, ...
        n_components, lambda, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir(F_oof, Y_train, F_te, lambda, post_fir_lags);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_cvlambda')
    % Plain full-history FIR on the predictions (post_prediction_fir: the filter warm-up is
    % returned as NaN, no prefix is fabricated) with lambda chosen by an inner
    % cross-validation over post_lambda_grid. This is the post stage of the published
    % configuration.
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_cv_lambda(F_oof, Y_train, F_te, post_lambda_grid, post_fir_lags, post_inner_k);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_kalman_y')
    C_pred = post_prediction_kalman_y(F_oof, Y_train, F_te, lambda);
    return;
elseif strcmp(mode, 'prediction_kalman_pv')
    [C_pred, post_info] = post_prediction_kalman_pv(F_oof, Y_train, F_te, lambda);
    post_info.mode = mode;
    return;
elseif strcmp(mode, 'prediction_fir_mean_padded')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_mean_padded(F_oof, Y_train, F_te, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_fir_warmup_bypass')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_warmup_bypass(F_oof, Y_train, F_te, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_fir_prefix_basis')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_prefix_basis(F_oof, Y_train, F_te, lambda, post_fir_lags, post_basis_components);
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, post_fir_lags);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_dense')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, post_fir_lags, 'dense');
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_halfsec')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, post_fir_lags, 'halfsec', fs);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_imputed_prefix')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    imp_opts = struct('prefix_stride', max(1, round(0.05 * fs)), 'query_max_sec', 0.5);
    [C_pred, post_info] = post_prediction_fir_imputed_persample(F_oof, Y_train, F_te, lambda, post_fir_lags, fs, imp_opts);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_halfsec_derivative_fusion')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix_derivative_fusion( ...
        F_oof, Y_train, F_te, lambda, post_fir_lags, 'halfsec', fs);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_onesec')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, post_fir_lags, 'onesec', fs);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_halfsec_robust')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix_robust(F_oof, Y_train, F_te, lambda, post_fir_lags, 'halfsec', fs);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_halfsec_target_z')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    Y_train_z = zscore_target_cells_per_trial(Y_train);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train_z, F_te, lambda, post_fir_lags, 'halfsec', fs);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    post_info.target_transform = 'per_trial_joint_zscore';
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_halfsec_lag_aug')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix_lag_aug( ...
        F_oof, Y_train, F_te, lambda, post_fir_lags, 'halfsec', fs);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_halfsec_lag_head_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix_lag_head_guard( ...
        F_oof, Y_train, F_te, lambda, post_fir_lags, 'halfsec', fs);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_halfsec_cvlambda')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix_cv_lambda( ...
        F_oof, Y_train, F_te, post_lambda_grid, post_fir_lags, 'halfsec', fs, post_inner_k);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif startsWith(mode, 'prediction_fir_dynamic_prefix_halfsec_rrr')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    rtok = regexp(mode, 'rrr(\d+)$', 'tokens', 'once');
    if isempty(rtok), rrr_rank = 3; else, rrr_rank = str2double(rtok{1}); end
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix_rrr( ...
        F_oof, Y_train, F_te, lambda, post_fir_lags, rrr_rank, 'halfsec', fs);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_halfsec_reliability_residual')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix_reliability_residual( ...
        F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin, 'halfsec', fs);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_halfsec_spatial_residual_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    [C_pred, post_info] = post_prediction_fir_dynamic_prefix_spatial_residual_guard( ...
        F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin, 'halfsec', fs);
    post_info.mode = mode;
    post_info.post_fir_lag_ms = post_fir_lag_ms;
    post_info.post_fir_lag_unit = post_fir_lag_unit;
    post_info.post_fir_causal_only = post_fir_causal_only;
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_spatial_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_dynamic_prefix_spatial_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_spatial_orth_mean_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_dynamic_prefix_spatial_orth_mean_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_dynamic_prefix_spatial_parallel_mean_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_dynamic_prefix_spatial_parallel_mean_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_masked')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_masked(F_oof, Y_train, F_te, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_fir_mask_only')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_mask_only(F_oof, Y_train, F_te, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_fir_multi_horizon_weighted')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_multi_horizon_weighted(F_oof, Y_train, F_te, lambda, post_fir_lags, fs, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_grid_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_grid_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, fs, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_on_fir_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_on_fir_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_on_fir_reliability_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_on_fir_reliability_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_on_fir_session_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_on_fir_session_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_on_fir_highbase_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_on_fir_highbase_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_on_fir_quality_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_on_fir_quality_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_on_fir_robust_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_on_fir_robust_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_on_fir_stability_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_on_fir_stability_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_on_fir_smooth_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_on_fir_smooth_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_multi_on_fir_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_multi_on_fir_guard(F_oof, Y_train, F_te, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_residual_multi_lowrank')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_multi_lowrank(F_oof, Y_train, F_te, n_components, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_fir_residual_multi_lowrank_on_fir_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_residual_multi_lowrank_on_fir_guard(F_oof, Y_train, F_te, n_components, lambda, post_fir_lags, guard_margin);
    return;
elseif strcmp(mode, 'prediction_fir_basis')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_basis(F_oof, Y_train, F_te, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_fir_apf')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_apf(A_oof, P_oof, F_oof, Y_train, A_te, P_te, F_te, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_apf_router')
    C_pred = post_prediction_apf_router(A_oof, P_oof, F_oof, Y_train, A_te, P_te, F_te, lambda);
    return;
elseif strcmp(mode, 'prediction_fir_multi')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_multi(F_oof, Y_train, F_te, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_fir_multi_basis')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_multi_basis(F_oof, Y_train, F_te, lambda, post_fir_lags, post_basis_components);
    return;
elseif strcmp(mode, 'prediction_fir_multi_basis_lowrank')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_multi_basis_lowrank(F_oof, Y_train, F_te, n_components, lambda, post_fir_lags, post_basis_components);
    return;
elseif strcmp(mode, 'prediction_fir_multi_basis_lowrank_on_fir_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_multi_basis_lowrank_on_fir_guard(F_oof, Y_train, F_te, n_components, lambda, post_fir_lags, guard_margin, post_basis_components);
    return;
elseif strcmp(mode, 'prediction_fir_multi_pls')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_multi_pls(F_oof, Y_train, F_te, n_components, lambda, post_fir_lags, false);
    return;
elseif strcmp(mode, 'prediction_fir_multi_basis_pls')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_multi_pls(F_oof, Y_train, F_te, n_components, lambda, post_fir_lags, true, post_basis_components);
    return;
elseif strcmp(mode, 'prediction_fir_multi_basis_pls_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_multi_pls_guard(F_oof, Y_train, F_te, n_components, lambda, post_fir_lags, true, guard_margin, post_basis_components);
    return;
elseif strcmp(mode, 'prediction_fir_lowrank')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_lowrank(F_oof, Y_train, F_te, n_components, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_fir_multi_robust')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_multi_robust(F_oof, Y_train, F_te, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_fir_phase_multi')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_phase_multi(F_oof, S_oof, Y_train, F_te, S_te, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_fir_apf_multi')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_fir_apf_multi(A_oof, P_oof, F_oof, Y_train, A_te, P_te, F_te, lambda, post_fir_lags);
    return;
elseif strcmp(mode, 'prediction_state_multi')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_state_multi(F_oof, Y_train, F_te, lambda, post_fir_lags, fs, false, guard_margin);
    return;
elseif strcmp(mode, 'prediction_state_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_state_multi(F_oof, Y_train, F_te, lambda, post_fir_lags, fs, true, guard_margin);
    return;
elseif strcmp(mode, 'prediction_arx_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_arx_multi(F_oof, Y_train, F_te, lambda, post_fir_lags, fs, true, guard_margin);
    return;
elseif strcmp(mode, 'prediction_rhythm_residual_guard')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_rhythm_residual(F_oof, Y_train, F_te, lambda, post_fir_lags, fs, guard_margin);
    return;
elseif strcmp(mode, 'prediction_lag_select')
    post_fir_lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
    C_pred = post_prediction_lag_select(F_oof, Y_train, F_te, post_fir_lags);
    return;
elseif ~ismember(mode, {'pls_calib', 'pls_calib_guard'})
    error('post_mode: unsupported value');
end

[Xtr, lens_tr] = post_feature_matrix(A_oof, P_oof, F_oof);
[Xte, lens_te] = post_feature_matrix(A_te, P_te, F_te);
Ytr = post_target_matrix(Y_train, lens_tr);
if isempty(Xtr) || isempty(Xte) || isempty(Ytr)
    C_pred = cell(size(A_te));
    return;
end
n_train = min(size(Xtr, 1), size(Ytr, 1));
Xtr = Xtr(1:n_train, :);
Ytr = Ytr(1:n_train, :);
Ftr = post_target_matrix(F_oof, lens_tr);
Ftr = Ftr(1:min(size(Ftr, 1), n_train), :);

x_mu = mean(Xtr, 1, 'omitnan');
x_sd = std(Xtr, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xtr_z = (Xtr - x_mu) ./ x_sd;
Xte_z = (Xte - x_mu) ./ x_sd;
Xtr_z(~isfinite(Xtr_z)) = 0;
Xte_z(~isfinite(Xte_z)) = 0;

[~, ~, V] = svd(Xtr_z, 'econ');
K = min([round(n_components), size(V, 2), size(Xtr_z, 1) - 1]);
if K < 1
    C_pred = cell(size(A_te));
    return;
end
Ztr = Xtr_z * V(:, 1:K);
Zte = Xte_z * V(:, 1:K);
Yhat = nan(size(Xte, 1), size(Ytr, 2));
Yhat_tr = nan(size(Xtr, 1), size(Ytr, 2));
ridge = max(double(lambda), 0);
for j = 1:size(Ytr, 2)
    y = Ytr(:, j);
    valid = isfinite(y);
    if sum(valid) < K + 5 || std(y(valid)) <= eps
        continue;
    end
    y_mu = mean(y(valid), 'omitnan');
    y_sd = std(y(valid), 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps
        y_sd = 1;
    end
    y_z = (y(valid) - y_mu) ./ y_sd;
    Xj = [Ztr(valid, :), ones(sum(valid), 1)];
    A = Xj' * Xj;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xj' * y_z);
    Yhat(:, j) = ([Zte, ones(size(Zte, 1), 1)] * w) * y_sd + y_mu;
    Yhat_tr(:, j) = ([Ztr, ones(size(Ztr, 1), 1)] * w) * y_sd + y_mu;
end
if strcmp(mode, 'pls_calib_guard')
    Fte = post_target_matrix(F_te, lens_te);
    n_te = min(size(Fte, 1), size(Yhat, 1));
    n_tr_guard = min(size(Ftr, 1), size(Yhat_tr, 1));
    for j = 1:size(Ytr, 2)
        r_base = safe_corr_local(Ytr(1:n_tr_guard, j), Ftr(1:n_tr_guard, j));
        r_cal = safe_corr_local(Ytr(1:n_tr_guard, j), Yhat_tr(1:n_tr_guard, j));
        if ~isfinite(r_cal) || ~isfinite(r_base) || r_cal < r_base + guard_margin
            Yhat(1:n_te, j) = Fte(1:n_te, j);
        end
    end
end
C_pred = split_prediction_matrix(Yhat, lens_te);
end

function lags = post_fir_lags_to_samples(lag_values, fs, lag_unit, causal_only)
if nargin < 2 || isempty(fs), fs = 250; end
if nargin < 3 || isempty(lag_unit), lag_unit = 'ms'; end
if nargin < 4 || isempty(causal_only), causal_only = true; end
lags = double(lag_values);
lags = lags(:)';
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:0;
end
switch lower(char(lag_unit))
    case 'ms'
        lags = round(lags * double(fs) / 1000);
    case {'sample', 'samples'}
        lags = round(lags);
    otherwise
        error('post_fir_lag_unit must be ms or samples');
end
lags = unique(lags(:)');
if causal_only
    lags = lags(lags <= 0);
end
if isempty(lags)
    error('post FIR has no causal lags; use zero or negative post_fir_lag_ms values');
end
end

function C_pred = post_prediction_kalman_y(F_oof, Y_train, F_te, lambda)
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = F_te;
    return;
end
J = size(Y_train{first_y}, 2);
models = cell(1, J);
for j = 1:J
    [yp, yc] = collect_adjacent_y_pairs(Y_train, j);
    [yt, fp] = collect_same_time_y_pred_pairs(Y_train, F_oof, j);
    yt = yt(:);
    fp = fp(:);
    valid_obs = isfinite(yt) & isfinite(fp);
    yt = yt(valid_obs);
    fp = fp(valid_obs);
    valid_dyn = isfinite(yp) & isfinite(yc);
    yp = yp(valid_dyn);
    yc = yc(valid_dyn);
    if numel(yt) < 30 || numel(yp) < 30 || std(yt) <= eps || std(fp) <= eps
        models{j} = [];
        continue;
    end
    y_mu = mean(yt, 'omitnan');
    y_var = var(yt, 0, 'omitnan');
    if ~isfinite(y_var) || y_var <= eps, y_var = 1; end

    Xdyn = [yp, ones(numel(yp), 1)];
    reg_dyn = ridge * eye(2);
    reg_dyn(2, 2) = 0;
    wb_dyn = (Xdyn' * Xdyn + reg_dyn) \ (Xdyn' * yc);
    a = wb_dyn(1);
    b = wb_dyn(2);
    dyn_res = yc - Xdyn * wb_dyn;
    q = var(dyn_res, 0, 'omitnan');
    if ~isfinite(q) || q <= eps, q = 0.01 * y_var; end

    Xobs = [fp, ones(numel(fp), 1)];
    reg_obs = ridge * eye(2);
    reg_obs(2, 2) = 0;
    wb_obs = (Xobs' * Xobs + reg_obs) \ (Xobs' * yt);
    obs = Xobs * wb_obs;
    obs_res = yt - obs;
    r = var(obs_res, 0, 'omitnan');
    if ~isfinite(r) || r <= eps, r = 0.01 * y_var; end

    noise_floor = 1e-4 * y_var;
    q = max(q, noise_floor);
    r = max(r, noise_floor);
    models{j} = struct('a', a, 'b', b, 'q', q, 'r', r, ...
        'obs_w', wb_obs(1), 'obs_b', wb_obs(2), 'y_mu', y_mu, 'y_var', y_var);
end

C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}) || size(F_te{k}, 2) < j
            continue;
        end
        Ck(:, j) = apply_scalar_kalman_y(F_te{k}(:, j), models{j});
    end
    C_pred{k} = Ck;
end
end

function yhat = apply_scalar_kalman_y(f, model)
f = double(f(:));
yhat = nan(size(f));
x = model.y_mu;
P = max(model.y_var, model.r);
for t = 1:numel(f)
    if t > 1
        x_pred = model.a * x + model.b;
        P_pred = model.a * model.a * P + model.q;
    else
        x_pred = x;
        P_pred = P + model.q;
    end
    if isfinite(f(t))
        z = model.obs_w * f(t) + model.obs_b;
        K = P_pred / (P_pred + model.r);
        x = x_pred + K * (z - x_pred);
        P = max((1 - K) * P_pred, 0);
        yhat(t) = x;
    else
        x = x_pred;
        P = P_pred;
    end
end
end

function [yp, yc] = collect_adjacent_y_pairs(Y_cell, j)
yp = [];
yc = [];
for k = 1:numel(Y_cell)
    if isempty(Y_cell{k}) || size(Y_cell{k}, 2) < j || size(Y_cell{k}, 1) < 2
        continue;
    end
    y = Y_cell{k}(:, j);
    ok = isfinite(y(1:end-1)) & isfinite(y(2:end));
    yp = [yp; y(find(ok))]; %#ok<FNDSB,AGROW>
    yc = [yc; y(find(ok) + 1)]; %#ok<FNDSB,AGROW>
end
end

function [yt, fp] = collect_same_time_y_pred_pairs(Y_cell, F_cell, j)
yt = [];
fp = [];
for k = 1:min(numel(Y_cell), numel(F_cell))
    if isempty(Y_cell{k}) || isempty(F_cell{k}) || size(Y_cell{k}, 2) < j || size(F_cell{k}, 2) < j
        continue;
    end
    n = min(size(Y_cell{k}, 1), size(F_cell{k}, 1));
    if n < 1
        continue;
    end
    y = Y_cell{k}(1:n, j);
    f = F_cell{k}(1:n, j);
    ok = isfinite(y) & isfinite(f);
    yt = [yt; y(ok)]; %#ok<AGROW>
    fp = [fp; f(ok)]; %#ok<AGROW>
end
end

function [C_pred, info] = post_prediction_kalman_pv(F_oof, Y_train, F_te, lambda)
% Position-velocity causal Kalman post-smoother: oscillator-capable upgrade of
% the scalar prediction_kalman_y. Per joint, state s=[y; dy]; a 2x2 VAR(1)
% transition A + drift c is learned from TRAIN-TARGET dynamics (A can carry
% complex eigenvalues, i.e. model the gait limit cycle, which a scalar AR(1)
% cannot); process noise Q from train residuals. The Stage-1 prediction is a
% scalar measurement z = Hw*y + Hb (fit on train-OOF predictions vs train
% targets), measurement noise R from train-OOF residuals. Filtered causally on
% test predictions; the held-out target never enters any fit.
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = F_te; info = struct('mode', 'kalman_pv', 'fallback', true);
    return;
end
J = size(Y_train{first_y}, 2);
models = cell(1, J);
nfb = 0;
for j = 1:J
    Sxx = zeros(3, 3); Syx = zeros(2, 3); Syy = zeros(2, 2); ndyn = 0;
    for k = 1:numel(Y_train)
        Y = Y_train{k};
        if isempty(Y) || size(Y, 2) < j, continue; end
        y = double(Y(:, j)); y = y(:);
        if numel(y) < 3, continue; end
        dy = [NaN; diff(y)];
        s = [y'; dy'];                              % 2 x T, s(:,t)=[y(t); y(t)-y(t-1)]
        good = all(isfinite(s), 1);
        t = 2:size(s, 2);
        sel = t(good(t) & good(t - 1));
        if isempty(sel), continue; end
        Xk = [s(:, sel - 1); ones(1, numel(sel))];  % 3 x m  ([s_prev; 1])
        Yk = s(:, sel);                              % 2 x m  (s_cur)
        Sxx = Sxx + Xk * Xk';
        Syx = Syx + Yk * Xk';
        Syy = Syy + Yk * Yk';
        ndyn = ndyn + numel(sel);
    end
    [yt, fp] = collect_same_time_y_pred_pairs(Y_train, F_oof, j);
    yt = yt(:); fp = fp(:);
    vobs = isfinite(yt) & isfinite(fp); yt = yt(vobs); fp = fp(vobs);
    if ndyn < 30 || numel(yt) < 30 || std(yt) <= eps || std(fp) <= eps
        models{j} = []; nfb = nfb + 1; continue;
    end
    y_var = var(yt, 0, 'omitnan'); if ~isfinite(y_var) || y_var <= eps, y_var = 1; end
    y_mu = mean(yt, 'omitnan');
    Rg = ridge * eye(3); Rg(3, 3) = 0;
    AB = Syx / (Sxx + Rg);                           % 2 x 3 = [A(2x2) c(2x1)]
    if any(~isfinite(AB(:))), models{j} = []; nfb = nfb + 1; continue; end
    A = AB(:, 1:2); c = AB(:, 3);
    ev = max(abs(eig(A)));
    if isfinite(ev) && ev > 0.999, A = A * (0.999 / ev); end   % stabilize transition
    Q = (Syy - AB * Syx') / max(1, ndyn - 1);
    Q = (Q + Q') / 2 + 1e-6 * y_var * eye(2);
    Xo = [fp, ones(numel(fp), 1)];
    Ro = ridge * eye(2); Ro(2, 2) = 0;
    wb = (Xo' * Xo + Ro) \ (Xo' * yt);
    Rm = var(yt - Xo * wb, 0, 'omitnan');
    if ~isfinite(Rm) || Rm <= eps, Rm = 0.01 * y_var; end
    models{j} = struct('A', A, 'c', c, 'Q', Q, 'Hw', wb(1), 'Hb', wb(2), ...
        'R', max(Rm, 1e-4 * y_var), 'y_mu', y_mu, 'y_var', y_var);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}) || size(F_te{k}, 2) < j, continue; end
        Ck(:, j) = apply_kalman_pv(F_te{k}(:, j), models{j});
    end
    C_pred{k} = Ck;
end
info = struct('mode', 'kalman_pv', 'fallback', false, 'n_fallback_joints', nfb);
end

function yhat = apply_kalman_pv(f, m)
f = double(f(:));
n = numel(f);
yhat = nan(n, 1);
x = [m.y_mu; 0];
P = diag([max(m.y_var, m.R), m.y_var]);
Hw = m.Hw; if abs(Hw) < 1e-8, Hw = sign(Hw + eps) * 1e-8; end
for t = 1:n
    if t > 1
        x = m.A * x + m.c;
        P = m.A * P * m.A' + m.Q;
        P = (P + P') / 2;
    end
    if isfinite(f(t))
        innov = f(t) - (Hw * x(1) + m.Hb);
        S = Hw * Hw * P(1, 1) + m.R;
        if ~isfinite(S) || S <= 0, S = m.R + eps; end
        K = (P * [Hw; 0]) / S;                       % 2 x 1
        x = x + K * innov;
        P = (eye(2) - K * [Hw, 0]) * P;
        P = (P + P') / 2;
    end
    yhat(t) = x(1);
end
end

function [C_pred, info] = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir requires at least one lag sample');
end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
models = cell(1, J);
n_valid = zeros(1, J);
fit_ok = false(1, J);
for j = 1:J
    [X, y] = fir_lag_design(F_oof, Y_train, j, lags);
    valid = isfinite(y) & all(isfinite(X), 2);
    X = X(valid, :);
    y = y(valid);
    n_valid(j) = numel(y);
    if size(X, 1) < numel(lags) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
    fit_ok(j) = true;
end
info = struct('mode', 'prediction_fir', 'lambda', lambda, 'ridge', ridge, ...
    'lags', lags, 'n_valid', n_valid, 'fit_ok', fit_ok, 'models', {models});
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}), continue; end
        [X, idx] = fir_lag_matrix(F_te{k}(:, j), lags);
        if isempty(X), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function [C_pred, info] = post_prediction_fir_cv_lambda(F_oof, Y_train, F_te, lambda_grid, lags, inner_k)
% CV structure of post_prediction_fir_dynamic_prefix_cv_lambda (same rng seed / folds),
% but the base learner is plain post_prediction_fir (NaN warmup, no dynamic-prefix).
if nargin < 6 || isempty(inner_k)
    inner_k = 3;
end
lambda_grid = unique(double(lambda_grid(:)'));
lambda_grid = lambda_grid(isfinite(lambda_grid) & lambda_grid >= 0);
if isempty(lambda_grid)
    lambda_grid = 0.01;
end
n = numel(Y_train);
K = min(max(1, round(double(inner_k))), n);
cv_score = nan(size(lambda_grid));
if K >= 2
    rng(20260528, 'twister');
    perm = randperm(n);
    fold_id = zeros(1, n);
    for k = 1:K
        fold_id(perm(k:K:end)) = k;
    end
    for li = 1:numel(lambda_grid)
        fold_scores = nan(1, K);
        for k = 1:K
            tr = fold_id ~= k;
            va = fold_id == k;
            if sum(tr) < 3 || sum(va) < 1
                continue;
            end
            C_val = post_prediction_fir(F_oof(tr), Y_train(tr), F_oof(va), lambda_grid(li), lags);
            fold_scores(k) = mean_rpt_all_joints(Y_train(va), C_val);
        end
        cv_score(li) = mean(fold_scores, 'omitnan');
    end
end
if all(~isfinite(cv_score))
    [~, best_li] = min(abs(lambda_grid - 0.01));
else
    [~, best_li] = max(cv_score);
end
selected_lambda = lambda_grid(best_li);
[C_pred, base_info] = post_prediction_fir(F_oof, Y_train, F_te, selected_lambda, lags);
info = base_info;
info.mode = 'prediction_fir_cvlambda';
info.lambda_grid = lambda_grid;
info.cv_score = cv_score;
info.selected_lambda = selected_lambda;
info.cv_inner_k = K;
end

function C_pred = post_prediction_fir_mean_padded(F_oof, Y_train, F_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_mean_padded requires at least one lag sample');
end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
models = cell(1, J);
for j = 1:J
    [X, y] = fir_lag_design(F_oof, Y_train, j, lags);
    valid = isfinite(y) & all(isfinite(X), 2);
    X = X(valid, :);
    y = y(valid);
    if size(X, 1) < numel(lags) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}) || size(F_te{k}, 2) < j
            continue;
        end
        [X, idx] = fir_lag_matrix_mean_padded(F_te{k}(:, j), lags, models{j}.x_mu);
        if isempty(X), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_warmup_bypass(F_oof, Y_train, F_te, lambda, lags)
C_pred = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
C_pred = fill_missing_prediction_cells(C_pred, F_te);
end

function C_pred = post_prediction_fir_prefix_basis(F_oof, Y_train, F_te, lambda, lags, n_basis)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_prefix_basis requires at least one lag sample');
end
ridge = max(double(lambda), 0);
LagB = fir_lag_smooth_basis_n(lags, n_basis);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
models = cell(1, J);
for j = 1:J
    [X, y] = fir_prefix_basis_design(F_oof, Y_train, j, lags, LagB);
    valid = isfinite(y) & all(isfinite(X), 2);
    X = X(valid, :);
    y = y(valid);
    if size(X, 1) < size(X, 2) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    Xz(~isfinite(Xz)) = 0;
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}), continue; end
        [X, idx] = fir_prefix_basis_matrix(F_te{k}(:, j), lags, LagB);
        if isempty(X), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function [C_pred, info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, lags, prefix_mode, fs)
if nargin < 6 || isempty(prefix_mode)
    prefix_mode = 'coarse';
end
if nargin < 7 || isempty(fs)
    fs = 250;
end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_dynamic_prefix requires at least one lag sample');
end
max_age = max(0, -min(lags));
if max_age == 0
    [C_pred, base_info] = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
    info = struct('mode', 'prediction_fir_dynamic_prefix', 'prefix_mode', char(prefix_mode), ...
        'fs', fs, 'lags', lags, 'candidate_ages', max_age, 'candidates', base_info);
    return;
end
if strcmpi(char(prefix_mode), 'dense')
    candidate_ages = unique(max(0, -lags));
elseif strcmpi(char(prefix_mode), 'halfsec')
    candidate_ages = unique(round([0 0.5 1.0 1.5 2.0 2.5 3.0] * double(fs)));
elseif strcmpi(char(prefix_mode), 'onesec')
    candidate_ages = unique(round([0 1.0 2.0 3.0] * double(fs)));
else
    candidate_ages = unique(round(linspace(0, max_age, 6)));
end
candidate_ages = candidate_ages(candidate_ages >= 0 & candidate_ages <= max_age);
if candidate_ages(end) ~= max_age
    candidate_ages(end + 1) = max_age;
end
info = struct();
info.mode = 'prediction_fir_dynamic_prefix';
info.prefix_mode = char(prefix_mode);
info.fs = fs;
info.lags = lags;
info.candidate_ages = candidate_ages;
info.candidate_age_sec = double(candidate_ages) ./ double(fs);
info.candidates = repmat(struct('age_samples', [], 'age_sec', [], 'lags', [], 'fir_info', []), 1, numel(candidate_ages));
C_pred = nan_like_prediction_cells(F_te);
for ai = 1:numel(candidate_ages)
    age = candidate_ages(ai);
    cand_lags = lags(lags >= -age & lags <= 0);
    if isempty(cand_lags)
        cand_lags = 0;
    end
    [C_tmp, fir_info] = post_prediction_fir(F_oof, Y_train, F_te, lambda, cand_lags);
    info.candidates(ai).age_samples = age;
    info.candidates(ai).age_sec = double(age) / double(fs);
    info.candidates(ai).lags = cand_lags;
    info.candidates(ai).fir_info = fir_info;
    C_pred = overwrite_finite_prediction_cells(C_pred, C_tmp);
end
end

function [C_pred, info] = post_prediction_fir_imputed_persample(F_oof, Y_train, F_te, lambda, lags, fs, opts)
% For t<=age the full-FIR window needs (age) pre-trial samples; they are imputed from the
% TRAINING-OOF library, matched ONLY on the (capped) causal prefix F_te(1:t,:) (<= t, no
% future, no test target), then the full FIR is applied. t>age uses the real window
if nargin < 6 || isempty(fs); fs = 250; end
if nargin < 7 || isempty(opts); opts = struct(); end
stride = 1;
if isfield(opts,'prefix_stride') && ~isempty(opts.prefix_stride); stride = max(1, round(opts.prefix_stride)); end
qmax = max(1, round(0.5 * fs));
if isfield(opts,'query_max_sec') && ~isempty(opts.query_max_sec); qmax = max(1, round(opts.query_max_sec * fs)); end
lags = unique(round(double(lags(:)'))); lags = lags(isfinite(lags));
if isempty(lags); error('post_prediction_fir_imputed_persample requires at least one lag'); end
age = max(0, -min(lags));
[models, J] = fit_fir_models_imp(F_oof, Y_train, lambda, lags);
lib = F_oof(~cellfun(@isempty, F_oof));
C_pred = cell(size(F_te)); n_imp = 0;
for k = 1:numel(F_te)
    f = F_te{k};
    if isempty(f); C_pred{k} = f; continue; end
    T = size(f, 1); Ck = nan(T, J);
    for j = 1:J
        if isempty(models{j}); continue; end
        [X, idx] = fir_lag_matrix(f(:, j), lags);
        if ~isempty(X); Ck(idx, j) = apply_fir_row_imp(models{j}, X); end
    end
    if age >= 1
        last_h = []; last_t = 0; last_ok = false;
        for t = 1:min(age, T)
            if mod(t-1, stride) == 0 || ~last_ok
                q = min(t, qmax);
                [h, ok] = nn_impute_prehistory_imp(lib, f(t-q+1:t, :), t, age);
                last_h = h; last_t = t; last_ok = ok;
            else
                ok = last_ok; h = last_h(1 + (t - last_t):end, :);
            end
            if ~ok; continue; end
            for j = 1:J
                if isempty(models{j}); continue; end
                w = [h(:, j); f(1:t, j)];
                xrow = reshape(w(end + lags), 1, []);
                Ck(t, j) = apply_fir_row_imp(models{j}, xrow);
            end
            n_imp = n_imp + 1;
        end
    end
    C_pred{k} = Ck;
end
info = struct('mode','prediction_fir_imputed_persample','lags',lags,'age',age,'fs',fs, ...
    'prefix_stride',stride,'query_max_samples',qmax,'n_imputed_samples',n_imp,'J',J);
end

function [models, J] = fit_fir_models_imp(F_oof, Y_train, lambda, lags)
ridge = max(double(lambda), 0);
nz = find(~cellfun(@isempty, Y_train), 1); J = size(Y_train{nz}, 2);
models = cell(1, J);
for j = 1:J
    [X, y] = fir_lag_design(F_oof, Y_train, j, lags);
    valid = isfinite(y) & all(isfinite(X), 2); X = X(valid, :); y = y(valid);
    if size(X, 1) < numel(lags) + 10 || std(y) <= eps; models{j} = []; continue; end
    x_mu = mean(X, 1, 'omitnan'); x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0; x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    y_mu = mean(y, 'omitnan'); y_sd = std(y, 0, 'omitnan'); if ~isfinite(y_sd) || y_sd <= eps; y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)]; A = Xa' * Xa; reg = ridge * eye(size(A)); reg(end, end) = 0;
    models{j} = struct('w', (A + reg) \ (Xa' * yz), 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
end

function yhat = apply_fir_row_imp(model, X)
Xz = (X - model.x_mu) ./ model.x_sd; Xz(~isfinite(Xz)) = 0;
yhat = ([Xz, ones(size(Xz, 1), 1)] * model.w) * model.y_sd + model.y_mu;
end

function [h, ok] = nn_impute_prehistory_imp(lib, query, t_full, age)
% query: q x J = last q samples of the causal prefix. Returns h: (age-t_full+1) x J = the
% missing pre-trial history, taken from the library window whose length-q tail best matches
% query.
q = size(query, 1); best = inf; h = []; ok = false;
for g = 1:numel(lib)
    G = lib{g}; Tg = size(G, 1);
    for s = (age + 1):Tg
        if s - q + 1 < 1; continue; end
        seg = G(s - q + 1:s, :);
        d = sum(sum((seg - query) .^ 2));
        if d < best; best = d; h = G(s - age:s - t_full, :); ok = true; end
    end
end
end

function [C_pred, info] = post_prediction_fir_rrr(F_oof, Y_train, F_te, lambda, lags, rank_r)
% Multi-output REDUCED-RANK causal FIR: calibrate all J joints jointly from the
% lagged Stage-1 predictions of ALL J joints, through a rank-r (<=J) operator.
% Borrows cross-joint coupling the per-joint FIR cannot use. Ridge full fit ->
% project coefficient columns onto the top-r subspace of the fitted response.
% Causal (lags<=0); fit on train-OOF predictions -> train targets only.
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = F_te; info = struct('mode', 'fir_rrr', 'fallback', true); return;
end
J = size(Y_train{first_y}, 2);
[Xall, Yall] = fir_lag_design_multi(F_oof, Y_train, lags, J);
if ~isempty(Xall)
    valid = all(isfinite(Xall), 2) & all(isfinite(Yall), 2);
    Xall = Xall(valid, :); Yall = Yall(valid, :);
end
if isempty(Xall) || size(Xall, 1) < size(Xall, 2) + 10
    [C_pred, info] = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
    info.mode = 'fir_rrr_fallback_perjoint';
    return;
end
x_mu = mean(Xall, 1); x_sd = std(Xall, 0, 1);
x_mu(~isfinite(x_mu)) = 0; x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
y_mu = mean(Yall, 1); y_sd = std(Yall, 0, 1);
y_mu(~isfinite(y_mu)) = 0; y_sd(~isfinite(y_sd) | y_sd <= eps) = 1;
Xz = (Xall - x_mu) ./ x_sd;
Yz = (Yall - y_mu) ./ y_sd;
p = size(Xz, 2);
Bfull = (Xz' * Xz + ridge * eye(p)) \ (Xz' * Yz);   % p x J ridge map
Yhat = Xz * Bfull;                                   % n x J fitted response
r = max(1, min(round(double(rank_r)), size(Yhat, 2)));
[~, ~, V] = svd(Yhat, 'econ');
Vr = V(:, 1:min(r, size(V, 2)));
Brrr = Bfull * (Vr * Vr');                           % rank-r restricted map
Brrr(~isfinite(Brrr)) = 0;
C_pred = nan_like_prediction_cells(F_te);
for k = 1:numel(F_te)
    if isempty(F_te{k}), continue; end
    [Xt, idx] = fir_lag_matrix_multi(F_te{k}, lags, J);
    if isempty(Xt), continue; end
    Xtz = (Xt - x_mu) ./ x_sd; Xtz(~isfinite(Xtz)) = 0;
    Yt = Xtz * Brrr;                                  % m x J
    Ck = nan(size(F_te{k}));
    for j = 1:J
        Ck(idx, j) = Yt(:, j) * y_sd(j) + y_mu(j);
    end
    C_pred{k} = Ck;
end
info = struct('mode', 'fir_rrr', 'rank', r, 'p', p, 'J', J, 'fallback', false);
end

function [C_pred, info] = post_prediction_fir_dynamic_prefix_rrr(F_oof, Y_train, F_te, lambda, lags, rank_r, prefix_mode, fs)
if nargin < 7 || isempty(prefix_mode), prefix_mode = 'halfsec'; end
if nargin < 8 || isempty(fs), fs = 250; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_dynamic_prefix_rrr requires at least one lag sample');
end
max_age = max(0, -min(lags));
if max_age == 0
    [C_pred, base_info] = post_prediction_fir_rrr(F_oof, Y_train, F_te, lambda, lags, rank_r);
    info = struct('mode', 'prediction_fir_dynamic_prefix_rrr', 'rank', rank_r, 'fs', fs, 'lags', lags, 'candidates', base_info);
    return;
end
if strcmpi(char(prefix_mode), 'dense')
    candidate_ages = unique(max(0, -lags));
elseif strcmpi(char(prefix_mode), 'halfsec')
    candidate_ages = unique(round([0 0.5 1.0 1.5 2.0 2.5 3.0] * double(fs)));
elseif strcmpi(char(prefix_mode), 'onesec')
    candidate_ages = unique(round([0 1.0 2.0 3.0] * double(fs)));
else
    candidate_ages = unique(round(linspace(0, max_age, 6)));
end
candidate_ages = candidate_ages(candidate_ages >= 0 & candidate_ages <= max_age);
if candidate_ages(end) ~= max_age
    candidate_ages(end + 1) = max_age;
end
info = struct('mode', 'prediction_fir_dynamic_prefix_rrr', 'prefix_mode', char(prefix_mode), ...
    'fs', fs, 'lags', lags, 'rank', rank_r, 'candidate_ages', candidate_ages);
C_pred = nan_like_prediction_cells(F_te);
for ai = 1:numel(candidate_ages)
    age = candidate_ages(ai);
    cand_lags = lags(lags >= -age & lags <= 0);
    if isempty(cand_lags), cand_lags = 0; end
    C_tmp = post_prediction_fir_rrr(F_oof, Y_train, F_te, lambda, cand_lags, rank_r);
    C_pred = overwrite_finite_prediction_cells(C_pred, C_tmp);
end
end

function [C_pred, info] = fit_predict_ydy_parallel_fusion(Y_oof, Y_train, Y_te, D_oof, D_train, D_te, lambda, post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only)
if isempty(D_oof) || isempty(D_te)
    error('prediction_ydy_parallel_fusion requires direct Stage 1 EEG->dy OOF and test predictions');
end
lags = post_fir_lags_to_samples(post_fir_lag_ms, fs, post_fir_lag_unit, post_fir_causal_only);
[Y_cal_te, y_te_info] = post_prediction_fir_dynamic_prefix(Y_oof, Y_train, Y_te, lambda, lags, 'halfsec', fs);
[Y_cal_oof, y_oof_info] = post_prediction_fir_dynamic_prefix(Y_oof, Y_train, Y_oof, lambda, lags, 'halfsec', fs);
[D_cal_te, d_te_info] = post_prediction_fir_dynamic_prefix(D_oof, D_train, D_te, lambda, lags, 'halfsec', fs);
[D_cal_oof, d_oof_info] = post_prediction_fir_dynamic_prefix(D_oof, D_train, D_oof, lambda, lags, 'halfsec', fs);

Y_from_d_te = integrate_derivative_prediction_cells_local(Y_cal_te, D_cal_te, Y_te);
Y_from_d_oof = integrate_derivative_prediction_cells_local(Y_cal_oof, D_cal_oof, Y_oof);

J = prediction_cell_joint_count_local(Y_train);
beta_grid = [0 0.125 0.25 0.5 0.75 1.0];
beta = zeros(1, J);
base_rpt = nan(1, J);
dy_rpt = nan(1, J);
fusion_rpt = nan(1, J);
for j = 1:J
    base_rpt(j) = safe_rpt_cells_local(Y_train, Y_cal_oof, j);
    dy_rpt(j) = safe_rpt_cells_local(Y_train, Y_from_d_oof, j);
    best_score = base_rpt(j);
    if ~isfinite(best_score)
        best_score = -Inf;
    end
    best_beta = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        C_mix_oof = blend_prediction_cells_local(Y_cal_oof, Y_from_d_oof, b);
        score = safe_rpt_cells_local(Y_train, C_mix_oof, j);
        if isfinite(score) && score > best_score
            best_score = score;
            best_beta = b;
        end
    end
    beta(j) = best_beta;
    if isfinite(best_score)
        fusion_rpt(j) = best_score;
    end
end

C_pred = blend_prediction_cells_local(Y_cal_te, Y_from_d_te, beta);
info = struct();
info.mode = 'prediction_ydy_parallel_fusion';
info.lags = lags;
info.beta_grid = beta_grid;
info.beta = beta;
info.base_oof_rpt = base_rpt;
info.dy_oof_rpt = dy_rpt;
info.fusion_oof_rpt = fusion_rpt;
info.mean_base_oof_rpt = mean(base_rpt, 'omitnan');
info.mean_dy_oof_rpt = mean(dy_rpt, 'omitnan');
info.mean_fusion_oof_rpt = mean(fusion_rpt, 'omitnan');
info.y_te_info = y_te_info;
info.y_oof_info = y_oof_info;
info.dy_te_info = d_te_info;
info.dy_oof_info = d_oof_info;
end

function [C_pred, info] = post_prediction_fir_dynamic_prefix_derivative_fusion(F_oof, Y_train, F_te, lambda, lags, prefix_mode, fs)
if nargin < 6 || isempty(prefix_mode)
    prefix_mode = 'halfsec';
end
if nargin < 7 || isempty(fs)
    fs = 250;
end
[base_te, base_te_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, lags, prefix_mode, fs);
[base_oof, base_oof_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_oof, lambda, lags, prefix_mode, fs);

Yd_train = derivative_prediction_cells_local(Y_train);
Fd_oof = derivative_prediction_cells_local(F_oof);
Fd_te = derivative_prediction_cells_local(F_te);
[D_te, deriv_te_info] = post_prediction_fir_dynamic_prefix(Fd_oof, Yd_train, Fd_te, lambda, lags, prefix_mode, fs);
[D_oof, deriv_oof_info] = post_prediction_fir_dynamic_prefix(Fd_oof, Yd_train, Fd_oof, lambda, lags, prefix_mode, fs);

vel_te = integrate_derivative_prediction_cells_local(base_te, D_te, F_te);
vel_oof = integrate_derivative_prediction_cells_local(base_oof, D_oof, F_oof);

J = prediction_cell_joint_count_local(Y_train);
if J == 0
    J = prediction_cell_joint_count_local(base_te);
end
beta_grid = [0 0.125 0.25 0.5 0.75 1.0];
beta = zeros(1, J);
base_rpt = nan(1, J);
deriv_rpt = nan(1, J);
fusion_rpt = nan(1, J);
for j = 1:J
    base_rpt(j) = safe_rpt_cells_local(Y_train, base_oof, j);
    deriv_rpt(j) = safe_rpt_cells_local(Y_train, vel_oof, j);
    best_score = base_rpt(j);
    if ~isfinite(best_score)
        best_score = -Inf;
    end
    best_beta = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        C_mix_oof = blend_prediction_cells_local(base_oof, vel_oof, b);
        score = safe_rpt_cells_local(Y_train, C_mix_oof, j);
        if isfinite(score) && score > best_score
            best_score = score;
            best_beta = b;
        end
    end
    beta(j) = best_beta;
    if isfinite(best_score)
        fusion_rpt(j) = best_score;
    end
end

C_pred = blend_prediction_cells_local(base_te, vel_te, beta);
info = struct();
info.mode = 'prediction_fir_dynamic_prefix_derivative_fusion';
info.prefix_mode = char(prefix_mode);
info.fs = fs;
info.lags = lags;
info.beta_grid = beta_grid;
info.beta = beta;
info.base_oof_rpt = base_rpt;
info.derivative_oof_rpt = deriv_rpt;
info.fusion_oof_rpt = fusion_rpt;
info.mean_base_oof_rpt = mean(base_rpt, 'omitnan');
info.mean_derivative_oof_rpt = mean(deriv_rpt, 'omitnan');
info.mean_fusion_oof_rpt = mean(fusion_rpt, 'omitnan');
info.base_te_info = base_te_info;
info.base_oof_info = base_oof_info;
info.deriv_te_info = deriv_te_info;
info.deriv_oof_info = deriv_oof_info;
end

function D = derivative_prediction_cells_local(C)
D = cell(size(C));
for k = 1:numel(C)
    if isempty(C{k})
        D{k} = C{k};
        continue;
    end
    X = double(C{k});
    Dk = nan(size(X));
    if size(X, 1) >= 2
        Dk(2:end, :) = diff(X, 1, 1);
    end
    D{k} = Dk;
end
end

function C_int = integrate_derivative_prediction_cells_local(C_base, D_pred, F_anchor)
C_int = C_base;
for k = 1:numel(D_pred)
    if isempty(D_pred{k})
        continue;
    end
    Dk = double(D_pred{k});
    n = size(Dk, 1);
    J = size(Dk, 2);
    Bk = nan(n, J);
    if k <= numel(C_base) && ~isempty(C_base{k})
        nb = min(n, size(C_base{k}, 1));
        jb = min(J, size(C_base{k}, 2));
        Bk(1:nb, 1:jb) = double(C_base{k}(1:nb, 1:jb));
    end
    Ak = Bk;
    if nargin >= 3 && k <= numel(F_anchor) && ~isempty(F_anchor{k})
        na = min(n, size(F_anchor{k}, 1));
        ja = min(J, size(F_anchor{k}, 2));
        Ak(1:na, 1:ja) = double(F_anchor{k}(1:na, 1:ja));
    end
    Yk = Bk;
    for j = 1:J
        d = Dk(:, j);
        anchor = Bk(:, j);
        if ~any(isfinite(anchor))
            anchor = Ak(:, j);
        end
        y = anchor;
        idx = find(isfinite(d) & isfinite(anchor), 1, 'first');
        if isempty(idx)
            continue;
        end
        y(idx) = anchor(idx);
        for ii = (idx + 1):n
            if isfinite(d(ii)) && isfinite(y(ii - 1))
                y(ii) = y(ii - 1) + d(ii);
            elseif isfinite(anchor(ii))
                y(ii) = anchor(ii);
            else
                y(ii) = NaN;
            end
        end
        Yk(:, j) = y;
    end
    C_int{k} = Yk;
end
end

function C_mix = blend_prediction_cells_local(C_base, C_alt, beta)
C_mix = C_base;
for k = 1:numel(C_base)
    if isempty(C_base{k})
        continue;
    end
    B = double(C_base{k});
    A = nan(size(B));
    if k <= numel(C_alt) && ~isempty(C_alt{k})
        n = min(size(B, 1), size(C_alt{k}, 1));
        J = min(size(B, 2), size(C_alt{k}, 2));
        A(1:n, 1:J) = double(C_alt{k}(1:n, 1:J));
    end
    if isscalar(beta)
        beta_vec = repmat(double(beta), 1, size(B, 2));
    else
        beta_vec = double(beta(:)');
    end
    M = B;
    for j = 1:min(size(B, 2), numel(beta_vec))
        b = beta_vec(j);
        if ~isfinite(b)
            b = 0;
        end
        both = isfinite(B(:, j)) & isfinite(A(:, j));
        M(both, j) = (1 - b) .* B(both, j) + b .* A(both, j);
        only_alt = ~isfinite(B(:, j)) & isfinite(A(:, j));
        M(only_alt, j) = A(only_alt, j);
    end
    C_mix{k} = M;
end
end

function J = prediction_cell_joint_count_local(C)
J = 0;
for k = 1:numel(C)
    if ~isempty(C{k}) && size(C{k}, 2) > 0
        J = size(C{k}, 2);
        return;
    end
end
end

function [C_pred, info] = post_prediction_fir_dynamic_prefix_robust(F_oof, Y_train, F_te, lambda, lags, prefix_mode, fs)
if nargin < 6 || isempty(prefix_mode)
    prefix_mode = 'coarse';
end
if nargin < 7 || isempty(fs)
    fs = 250;
end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_dynamic_prefix_robust requires at least one lag sample');
end
max_age = max(0, -min(lags));
if max_age == 0
    [C_pred, base_info] = post_prediction_fir_robust(F_oof, Y_train, F_te, lambda, lags);
    info = struct('mode', 'prediction_fir_dynamic_prefix_robust', 'prefix_mode', char(prefix_mode), ...
        'fs', fs, 'lags', lags, 'candidate_ages', max_age, 'candidates', base_info);
    return;
end
if strcmpi(char(prefix_mode), 'dense')
    candidate_ages = unique(max(0, -lags));
elseif strcmpi(char(prefix_mode), 'halfsec')
    candidate_ages = unique(round([0 0.5 1.0 1.5 2.0 2.5 3.0] * double(fs)));
else
    candidate_ages = unique(round(linspace(0, max_age, 6)));
end
candidate_ages = candidate_ages(candidate_ages >= 0 & candidate_ages <= max_age);
if candidate_ages(end) ~= max_age
    candidate_ages(end + 1) = max_age;
end
info = struct();
info.mode = 'prediction_fir_dynamic_prefix_robust';
info.prefix_mode = char(prefix_mode);
info.fs = fs;
info.lags = lags;
info.candidate_ages = candidate_ages;
info.candidate_age_sec = double(candidate_ages) ./ double(fs);
info.candidates = repmat(struct('age_samples', [], 'age_sec', [], 'lags', [], 'fir_info', []), 1, numel(candidate_ages));
C_pred = nan_like_prediction_cells(F_te);
for ai = 1:numel(candidate_ages)
    age = candidate_ages(ai);
    cand_lags = lags(lags >= -age & lags <= 0);
    if isempty(cand_lags)
        cand_lags = 0;
    end
    [C_tmp, fir_info] = post_prediction_fir_robust(F_oof, Y_train, F_te, lambda, cand_lags);
    info.candidates(ai).age_samples = age;
    info.candidates(ai).age_sec = double(age) / double(fs);
    info.candidates(ai).lags = cand_lags;
    info.candidates(ai).fir_info = fir_info;
    C_pred = overwrite_finite_prediction_cells(C_pred, C_tmp);
end
end

function [C_pred, info] = post_prediction_fir_robust(F_oof, Y_train, F_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_robust requires at least one lag sample');
end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
models = cell(1, J);
n_valid = zeros(1, J);
mean_weight = nan(1, J);
fit_ok = false(1, J);
for j = 1:J
    [X, y] = fir_lag_design(F_oof, Y_train, j, lags);
    valid = isfinite(y) & all(isfinite(X), 2);
    X = X(valid, :);
    y = y(valid);
    n_valid(j) = numel(y);
    if size(X, 1) < numel(lags) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    Xz(~isfinite(Xz)) = 0;
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    w0 = ridge_solve_local(Xa, yz, ridge);
    resid = yz - Xa * w0;
    sig = 1.4826 * median(abs(resid - median(resid, 'omitnan')), 'omitnan');
    if ~isfinite(sig) || sig <= eps
        sig = std(resid, 0, 'omitnan');
    end
    if ~isfinite(sig) || sig <= eps
        sig = 1;
    end
    c = 2.5 * sig;
    wt = ones(size(resid));
    big = abs(resid) > c;
    wt(big) = c ./ max(abs(resid(big)), eps);
    wt(~isfinite(wt) | wt < 0.05) = 0.05;
    sw = sqrt(wt);
    Xw = Xa .* sw;
    yw = yz .* sw;
    w = ridge_solve_local(Xw, yw, ridge);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
    mean_weight(j) = mean(wt, 'omitnan');
    fit_ok(j) = true;
end
info = struct('mode', 'prediction_fir_robust', 'lambda', lambda, 'ridge', ridge, ...
    'lags', lags, 'n_valid', n_valid, 'mean_weight', mean_weight, ...
    'fit_ok', fit_ok, 'models', {models});
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}), continue; end
        [X, idx] = fir_lag_matrix(F_te{k}(:, j), lags);
        if isempty(X), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function Yz = zscore_target_cells_per_trial(Y)
Yz = cell(size(Y));
for k = 1:numel(Y)
    if isempty(Y{k})
        Yz{k} = Y{k};
        continue;
    end
    Yk = double(Y{k});
    for j = 1:size(Yk, 2)
        y = Yk(:, j);
        ok = isfinite(y);
        if sum(ok) < 5
            continue;
        end
        mu = mean(y(ok), 'omitnan');
        sd = std(y(ok), 0, 'omitnan');
        if isfinite(sd) && sd > eps
            Yk(ok, j) = (y(ok) - mu) ./ sd;
        end
    end
    Yz{k} = Yk;
end
end

function [C_pred, info] = post_prediction_fir_dynamic_prefix_cv_lambda(F_oof, Y_train, F_te, lambda_grid, lags, prefix_mode, fs, inner_k)
if nargin < 8 || isempty(inner_k)
    inner_k = 3;
end
lambda_grid = unique(double(lambda_grid(:)'));
lambda_grid = lambda_grid(isfinite(lambda_grid) & lambda_grid >= 0);
if isempty(lambda_grid)
    lambda_grid = 0.01;
end
n = numel(Y_train);
K = min(max(1, round(double(inner_k))), n);
cv_score = nan(size(lambda_grid));
if K >= 2
    rng(20260528, 'twister');
    perm = randperm(n);
    fold_id = zeros(1, n);
    for k = 1:K
        fold_id(perm(k:K:end)) = k;
    end
    for li = 1:numel(lambda_grid)
        fold_scores = nan(1, K);
        for k = 1:K
            tr = fold_id ~= k;
            va = fold_id == k;
            if sum(tr) < 3 || sum(va) < 1
                continue;
            end
            C_val = post_prediction_fir_dynamic_prefix( ...
                F_oof(tr), Y_train(tr), F_oof(va), lambda_grid(li), lags, prefix_mode, fs);
            fold_scores(k) = mean_rpt_all_joints(Y_train(va), C_val);
        end
        cv_score(li) = mean(fold_scores, 'omitnan');
    end
end
if all(~isfinite(cv_score))
    [~, best_li] = min(abs(lambda_grid - 0.01));
else
    [~, best_li] = max(cv_score);
end
selected_lambda = lambda_grid(best_li);
[C_pred, base_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, selected_lambda, lags, prefix_mode, fs);
info = base_info;
info.lambda_grid = lambda_grid;
info.cv_score = cv_score;
info.selected_lambda = selected_lambda;
info.cv_inner_k = K;
end

function [C_pred, info] = post_prediction_fir_dynamic_prefix_lag_aug(F_oof, Y_train, F_te, lambda, lags, prefix_mode, fs)
if nargin < 6 || isempty(prefix_mode)
    prefix_mode = 'coarse';
end
if nargin < 7 || isempty(fs)
    fs = 250;
end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_dynamic_prefix_lag_aug requires at least one lag sample');
end
target_shifts = unique(round([-0.5 -0.25 0 0.25 0.5] * double(fs)));
sigma_samples = max(1, round(0.25 * double(fs)));
shift_weights = exp(-0.5 * (double(target_shifts) ./ double(sigma_samples)).^2);
shift_weights = shift_weights ./ max(shift_weights);
max_age = max(0, -min(lags));
if strcmpi(char(prefix_mode), 'halfsec')
    candidate_ages = unique(round([0 0.5 1.0 1.5 2.0 2.5 3.0] * double(fs)));
elseif strcmpi(char(prefix_mode), 'dense')
    candidate_ages = unique(max(0, -lags));
else
    candidate_ages = unique(round(linspace(0, max_age, 6)));
end
candidate_ages = candidate_ages(candidate_ages >= 0 & candidate_ages <= max_age);
if isempty(candidate_ages)
    candidate_ages = max_age;
end
if candidate_ages(end) ~= max_age
    candidate_ages(end + 1) = max_age;
end
info = struct();
info.mode = 'prediction_fir_dynamic_prefix_lag_aug';
info.prefix_mode = char(prefix_mode);
info.fs = fs;
info.lags = lags;
info.target_shifts = target_shifts;
info.target_shift_sec = double(target_shifts) ./ double(fs);
info.shift_weights = shift_weights;
info.candidate_ages = candidate_ages;
info.candidate_age_sec = double(candidate_ages) ./ double(fs);
info.candidates = repmat(struct('age_samples', [], 'age_sec', [], 'lags', [], 'fir_info', []), 1, numel(candidate_ages));
C_pred = nan_like_prediction_cells(F_te);
for ai = 1:numel(candidate_ages)
    age = candidate_ages(ai);
    cand_lags = lags(lags >= -age & lags <= 0);
    if isempty(cand_lags)
        cand_lags = 0;
    end
    [C_tmp, fir_info] = post_prediction_fir_lag_aug(F_oof, Y_train, F_te, lambda, cand_lags, target_shifts, shift_weights);
    info.candidates(ai).age_samples = age;
    info.candidates(ai).age_sec = double(age) / double(fs);
    info.candidates(ai).lags = cand_lags;
    info.candidates(ai).fir_info = fir_info;
    C_pred = overwrite_finite_prediction_cells(C_pred, C_tmp);
end
end

function [C_pred, info] = post_prediction_fir_lag_aug(F_oof, Y_train, F_te, lambda, lags, target_shifts, shift_weights)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_lag_aug requires at least one lag sample');
end
target_shifts = round(double(target_shifts(:)'));
shift_weights = double(shift_weights(:)');
if numel(shift_weights) ~= numel(target_shifts)
    error('target_shifts and shift_weights must have the same length');
end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
models = cell(1, J);
n_valid = zeros(1, J);
fit_ok = false(1, J);
for j = 1:J
    [X, y, w] = fir_lag_design_lag_aug(F_oof, Y_train, j, lags, target_shifts, shift_weights);
    valid = isfinite(y) & isfinite(w) & w > 0 & all(isfinite(X), 2);
    X = X(valid, :);
    y = y(valid);
    w = w(valid);
    n_valid(j) = numel(y);
    if size(X, 1) < numel(lags) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    sw = sqrt(w(:) ./ mean(w, 'omitnan'));
    Xa = [Xz, ones(size(Xz, 1), 1)];
    Xw = Xa .* sw;
    yw = yz .* sw;
    A = Xw' * Xw;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    model_w = (A + reg) \ (Xw' * yw);
    models{j} = struct('w', model_w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
    fit_ok(j) = true;
end
info = struct('mode', 'prediction_fir_lag_aug', 'lambda', lambda, 'ridge', ridge, ...
    'lags', lags, 'target_shifts', target_shifts, 'shift_weights', shift_weights, ...
    'n_valid', n_valid, 'fit_ok', fit_ok, 'models', {models});
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}), continue; end
        [X, idx] = fir_lag_matrix(F_te{k}(:, j), lags);
        if isempty(X), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function [C_pred, info] = post_prediction_fir_dynamic_prefix_lag_head_guard(F_oof, Y_train, F_te, lambda, lags, prefix_mode, fs)
if nargin < 6 || isempty(prefix_mode), prefix_mode = 'halfsec'; end
if nargin < 7 || isempty(fs), fs = 250; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_dynamic_prefix_lag_head_guard requires at least one lag sample');
end
[base_oof, base_oof_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_oof, lambda, lags, prefix_mode, fs);
[base_te, base_te_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, lags, prefix_mode, fs);
[models, J] = fit_prediction_lag_head_models(F_oof, base_oof, Y_train, fs, lambda);
shift_oof = apply_prediction_lag_head_models(F_oof, base_oof, models, J, fs);
shift_te = apply_prediction_lag_head_models(F_te, base_te, models, J, fs);
beta = select_lag_head_betas_by_rpt(Y_train, base_oof, shift_oof, J);
C_pred = blend_lag_head_cells(base_te, shift_te, beta);
info = struct('mode', 'prediction_fir_dynamic_prefix_lag_head_guard', ...
    'base_oof_info', base_oof_info, 'base_te_info', base_te_info, ...
    'lag_models', {models}, 'beta', beta, 'fs', fs);
end

function [models, J] = fit_prediction_lag_head_models(F_cell, C_cell, Y_cell, fs, lambda)
first_y = find(~cellfun(@isempty, Y_cell), 1);
J = size(Y_cell{first_y}, 2);
models = cell(1, J);
max_lag = round(0.5 * double(fs));
step = max(1, round(0.10 * double(fs)));
win = max(5, round(1.0 * double(fs)));
half_win = floor(win / 2);
ridge = max(double(lambda), 0.01);
for j = 1:J
    Xall = [];
    yall = [];
    wall = [];
    for k = 1:min([numel(F_cell), numel(C_cell), numel(Y_cell)])
        if isempty(F_cell{k}) || isempty(C_cell{k}) || isempty(Y_cell{k}), continue; end
        if size(F_cell{k}, 2) < j || size(C_cell{k}, 2) < j || size(Y_cell{k}, 2) < j, continue; end
        n = min([size(F_cell{k}, 1), size(C_cell{k}, 1), size(Y_cell{k}, 1)]);
        if n < win + 5, continue; end
        f = F_cell{k}(1:n, j);
        c = C_cell{k}(1:n, j);
        y = Y_cell{k}(1:n, j);
        centers = (1 + half_win):step:(n - half_win);
        for ci = 1:numel(centers)
            center = centers(ci);
            idx = (center - half_win):(center + half_win);
            [lag, best_r, base_r] = local_best_lag_label(y(idx), c(idx), max_lag);
            if ~isfinite(lag), continue; end
            feat = lag_head_feature_at(f, c, center, half_win);
            if any(~isfinite(feat)), continue; end
            Xall = [Xall; feat]; %#ok<AGROW>
            yall = [yall; lag]; %#ok<AGROW>
            wall = [wall; max(0.10, min(1.0, best_r - base_r + 0.10))]; %#ok<AGROW>
        end
    end
    if size(Xall, 1) < size(Xall, 2) + 10 || std(yall) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(Xall, 1, 'omitnan');
    x_sd = std(Xall, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (Xall - x_mu) ./ x_sd;
    y_mu = mean(yall, 'omitnan');
    y_sd = std(yall, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (yall - y_mu) ./ y_sd;
    sw = sqrt(wall(:) ./ mean(wall, 'omitnan'));
    Xa = [Xz, ones(size(Xz, 1), 1)];
    Xw = Xa .* sw;
    yw = yz .* sw;
    A = Xw' * Xw;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xw' * yw);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, ...
        'y_mu', y_mu, 'y_sd', y_sd, 'max_lag', max_lag, ...
        'smooth_win', max(1, round(0.5 * double(fs))), 'half_win', half_win);
end
end

function C_shift = apply_prediction_lag_head_models(F_cell, C_cell, models, J, fs)
C_shift = cell(size(C_cell));
for k = 1:numel(C_cell)
    Ck = C_cell{k};
    if isempty(Ck)
        C_shift{k} = Ck;
        continue;
    end
    Sk = Ck;
    for j = 1:min(J, size(Ck, 2))
        if j > numel(models) || isempty(models{j}) || k > numel(F_cell) || isempty(F_cell{k}) || size(F_cell{k}, 2) < j
            continue;
        end
        n = min(size(F_cell{k}, 1), size(Ck, 1));
        f = F_cell{k}(1:n, j);
        c = Ck(1:n, j);
        lag_hat = predict_lag_head_series(f, c, models{j});
        Sk(1:n, j) = shift_series_by_lag(c, lag_hat);
    end
    C_shift{k} = Sk;
end
end

function lag_hat = predict_lag_head_series(f, c, model)
n = min(numel(f), numel(c));
lag_hat = nan(n, 1);
for t = 1:n
    feat = lag_head_feature_at(f, c, t, model.half_win);
    if any(~isfinite(feat)), continue; end
    xz = (feat - model.x_mu) ./ model.x_sd;
    xz(~isfinite(xz)) = 0;
    lag_hat(t) = ([xz, 1] * model.w) * model.y_sd + model.y_mu;
end
lag_hat = max(-model.max_lag, min(model.max_lag, lag_hat));
finite = isfinite(lag_hat);
if any(finite)
    lag_hat = fillmissing(lag_hat, 'linear', 'EndValues', 'nearest');
    lag_hat = movmean(lag_hat, [model.smooth_win - 1, 0], 'omitnan');
else
    lag_hat(:) = 0;
end
end

function feat = lag_head_feature_at(f, c, t, half_win)
n = min(numel(f), numel(c));
if t < 1 || t > n
    feat = nan(1, 8);
    return;
end
c0 = c(t);
f0 = f(t);
dc = local_diff(c, t, 1);
df = local_diff(f, t, 1);
ddc = local_diff(c, t, 2);
idx = max(1, t - half_win):min(n, t + half_win);
feat = [c0, dc, ddc, f0, df, c0 - f0, std(c(idx), 0, 'omitnan'), std(f(idx), 0, 'omitnan')];
end

function d = local_diff(x, t, order)
if order == 1
    if t <= 1 || ~isfinite(x(t)) || ~isfinite(x(t - 1))
        d = NaN;
    else
        d = x(t) - x(t - 1);
    end
else
    if t <= 2 || ~isfinite(x(t)) || ~isfinite(x(t - 1)) || ~isfinite(x(t - 2))
        d = NaN;
    else
        d = x(t) - 2 * x(t - 1) + x(t - 2);
    end
end
end

function [lag_best, best_r, base_r] = local_best_lag_label(y, c, max_lag)
base_r = safe_corr_local(y(:), c(:));
best_r = base_r;
lag_best = 0;
for lag = -max_lag:max_lag
    if lag >= 0
        yy = y((1 + lag):end);
        cc = c(1:(end - lag));
    else
        yy = y(1:(end + lag));
        cc = c((1 - lag):end);
    end
    r = safe_corr_local(yy(:), cc(:));
    if isfinite(r) && (~isfinite(best_r) || r > best_r)
        best_r = r;
        lag_best = lag;
    end
end
if ~isfinite(best_r)
    lag_best = NaN;
end
end

function y = shift_series_by_lag(c, lag_hat)
n = numel(c);
x = (1:n)';
src = x - lag_hat(:);
y = interp1(x, c(:), src, 'linear', NaN);
end

function beta = select_lag_head_betas_by_rpt(Y_true, C_base, C_shift, J)
beta_grid = 0:0.25:1;
beta = zeros(1, J);
for j = 1:J
    base_r = safe_rpt_cells_local(Y_true, C_base, j);
    best_r = base_r;
    best_b = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        C_try = blend_lag_head_cells(C_base, C_shift, scalar_beta_for_joint(J, j, b));
        r_try = safe_rpt_cells_local(Y_true, C_try, j);
        if isfinite(r_try) && (~isfinite(best_r) || r_try > best_r)
            best_r = r_try;
            best_b = b;
        end
    end
    if isfinite(best_r) && isfinite(base_r) && best_r > base_r
        beta(j) = best_b;
    end
end
end

function beta = scalar_beta_for_joint(J, j, b)
beta = zeros(1, J);
beta(j) = b;
end

function C_blend = blend_lag_head_cells(C_base, C_shift, beta)
C_blend = cell(size(C_base));
for k = 1:numel(C_base)
    B = C_base{k};
    if isempty(B) || k > numel(C_shift) || isempty(C_shift{k})
        C_blend{k} = B;
        continue;
    end
    S = C_shift{k};
    n = min(size(B, 1), size(S, 1));
    J = min(size(B, 2), size(S, 2));
    Ck = B;
    for j = 1:J
        if j <= numel(beta)
            b = beta(j);
        else
            b = 0;
        end
        ok = isfinite(B(1:n, j)) & isfinite(S(1:n, j));
        idx = find(ok);
        Ck(idx, j) = (1 - b) * B(idx, j) + b * S(idx, j);
    end
    C_blend{k} = Ck;
end
end

function [C_pred, info] = post_prediction_fir_dynamic_prefix_reliability_residual(F_oof, Y_train, F_te, lambda, lags, guard_margin, prefix_mode, fs)
if nargin < 6 || isempty(guard_margin), guard_margin = 0; end
if nargin < 7 || isempty(prefix_mode), prefix_mode = 'halfsec'; end
if nargin < 8 || isempty(fs), fs = 250; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_dynamic_prefix_reliability_residual requires at least one lag sample');
end
[base_oof, base_oof_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_oof, lambda, lags, prefix_mode, fs);
[base_te, base_te_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, lags, prefix_mode, fs);
[models, J] = fit_prediction_fir_residual_models(base_oof, Y_train, lambda, lags);
R_oof = apply_prediction_fir_residual_models(base_oof, models, lags, J);
R_te = apply_prediction_fir_residual_models(base_te, models, lags, J);
beta = select_residual_betas_by_reliability(Y_train, base_oof, R_oof, J, guard_margin);
C_pred = add_scaled_residual_cells(base_te, R_te, [], beta);
info = struct();
info.mode = 'prediction_fir_dynamic_prefix_reliability_residual';
info.lambda = lambda;
info.guard_margin = guard_margin;
info.prefix_mode = char(prefix_mode);
info.fs = fs;
info.lags = lags;
info.beta = beta;
info.residual_fit_ok = ~cellfun(@isempty, models);
info.base_oof = base_oof_info;
info.base_te = base_te_info;
end

function [C_pred, info] = post_prediction_fir_dynamic_prefix_spatial_residual_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin, prefix_mode, fs)
if nargin < 6 || isempty(guard_margin), guard_margin = 0; end
if nargin < 7 || isempty(prefix_mode), prefix_mode = 'halfsec'; end
if nargin < 8 || isempty(fs), fs = 250; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_dynamic_prefix_spatial_residual_guard requires at least one lag sample');
end
[base_oof, base_oof_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_oof, lambda, lags, prefix_mode, fs);
[base_te, base_te_info] = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, lags, prefix_mode, fs);
[models, J] = fit_prediction_spatial_offdiag_residual_models(base_oof, Y_train, lambda);
R_oof = apply_prediction_spatial_offdiag_residual_models(base_oof, models, J);
R_te = apply_prediction_spatial_offdiag_residual_models(base_te, models, J);
alpha = select_global_residual_alpha_by_rpt(Y_train, base_oof, R_oof, guard_margin);
C_pred = add_scaled_residual_cells(base_te, R_te, [], alpha);
info = struct();
info.mode = 'prediction_fir_dynamic_prefix_spatial_residual_guard';
info.lambda = lambda;
info.guard_margin = guard_margin;
info.prefix_mode = char(prefix_mode);
info.fs = fs;
info.lags = lags;
info.alpha = alpha;
info.spatial_fit_ok = ~cellfun(@isempty, models);
info.base_oof = base_oof_info;
info.base_te = base_te_info;
end

function r = mean_rpt_all_joints(Y_true, Y_pred)
first_y = find(~cellfun(@isempty, Y_true), 1);
if isempty(first_y)
    r = NaN;
    return;
end
J = size(Y_true{first_y}, 2);
vals = nan(1, J);
for j = 1:J
    vals(j) = safe_rpt_cells_local(Y_true, Y_pred, j);
end
r = mean(vals, 'omitnan');
end

function C_pred = post_prediction_fir_dynamic_prefix_spatial_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
base_oof = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_spatial_models(base_oof, Y_train, lambda);
spatial_oof = apply_prediction_spatial_models(base_oof, models, J);
spatial_te = apply_prediction_spatial_models(base_te, models, J);
use_spatial = false(1, J);
for j = 1:J
    r_base = safe_rpt_cells_local(Y_train, base_oof, j);
    r_spatial = safe_rpt_cells_local(Y_train, spatial_oof, j);
    use_spatial(j) = isfinite(r_spatial) && (~isfinite(r_base) || r_spatial >= r_base + guard_margin);
end
C_pred = guarded_spatial_prediction_cells(base_te, spatial_te, use_spatial);
end

function C_pred = post_prediction_fir_dynamic_prefix_spatial_orth_mean_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
base_oof = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_spatial_orth_residual_models(base_oof, Y_train, lambda);
R_oof = apply_prediction_spatial_orth_residual_models(base_oof, models, J);
R_te = apply_prediction_spatial_orth_residual_models(base_te, models, J);
C_oof = add_scaled_residual_cells(base_oof, R_oof, [], 0.5);
C_te = add_scaled_residual_cells(base_te, R_te, [], 0.5);
use_spatial = false(1, J);
for j = 1:J
    r_base = safe_rpt_cells_local(Y_train, base_oof, j);
    r_orth = safe_rpt_cells_local(Y_train, C_oof, j);
    use_spatial(j) = isfinite(r_orth) && (~isfinite(r_base) || r_orth >= r_base + guard_margin);
end
C_pred = guarded_spatial_prediction_cells(base_te, C_te, use_spatial);
end

function C_pred = post_prediction_fir_dynamic_prefix_spatial_parallel_mean_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
D_oof = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_oof, lambda, lags);
D_te = post_prediction_fir_dynamic_prefix(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_spatial_models(F_oof, Y_train, lambda);
S_oof = apply_prediction_spatial_models(F_oof, models, J);
S_te = apply_prediction_spatial_models(F_te, models, J);
M_oof = mean_prediction_cells(D_oof, S_oof, 0.5);
M_te = mean_prediction_cells(D_te, S_te, 0.5);
use_mean = false(1, J);
for j = 1:J
    r_dynamic = safe_rpt_cells_local(Y_train, D_oof, j);
    r_mean = safe_rpt_cells_local(Y_train, M_oof, j);
    use_mean(j) = isfinite(r_mean) && (~isfinite(r_dynamic) || r_mean >= r_dynamic + guard_margin);
end
C_pred = guarded_spatial_prediction_cells(D_te, M_te, use_mean);
end

function Y_cell = mean_prediction_cells(A_cell, B_cell, alpha)
if nargin < 3, alpha = 0.5; end
Y_cell = cell(size(A_cell));
for k = 1:numel(A_cell)
    if isempty(A_cell{k})
        Y_cell{k} = A_cell{k};
        continue;
    end
    Yk = A_cell{k};
    if k > numel(B_cell) || isempty(B_cell{k})
        Y_cell{k} = Yk;
        continue;
    end
    n = min(size(A_cell{k}, 1), size(B_cell{k}, 1));
    J = min(size(A_cell{k}, 2), size(B_cell{k}, 2));
    Ak = A_cell{k}(1:n, 1:J);
    Bk = B_cell{k}(1:n, 1:J);
    ok = isfinite(Ak) & isfinite(Bk);
    Ysub = Ak;
    Ysub(ok) = (1 - alpha) * Ak(ok) + alpha * Bk(ok);
    Yk(1:n, 1:J) = Ysub;
    Y_cell{k} = Yk;
end
end

function [models, J] = fit_prediction_spatial_orth_residual_models(F_cell, Y_cell, lambda)
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_cell), 1);
if isempty(first_y)
    models = {};
    J = 0;
    return;
end
J = size(Y_cell{first_y}, 2);
X_all = [];
Y_all = [];
for k = 1:min(numel(F_cell), numel(Y_cell))
    if isempty(F_cell{k}) || isempty(Y_cell{k}) || size(F_cell{k}, 2) < J || size(Y_cell{k}, 2) < J
        continue;
    end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if n < J + 10
        continue;
    end
    X_all = [X_all; F_cell{k}(1:n, 1:J)]; %#ok<AGROW>
    Y_all = [Y_all; Y_cell{k}(1:n, 1:J)]; %#ok<AGROW>
end
models = cell(1, J);
if isempty(X_all) || isempty(Y_all)
    return;
end
for j = 1:J
    if size(X_all, 2) < j
        continue;
    end
    y = Y_all(:, j);
    b = X_all(:, j);
    valid = isfinite(y) & isfinite(b) & any(isfinite(X_all), 2);
    X = X_all(valid, :);
    y = y(valid);
    b = b(valid);
    if size(X, 1) < J + 10 || std(y) <= eps || std(b) <= eps
        models{j} = [];
        continue;
    end
    r = y - b;
    if std(r) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    Ximp = X;
    for c = 1:size(Ximp, 2)
        bad = ~isfinite(Ximp(:, c));
        Ximp(bad, c) = x_mu(c);
    end
    b_mu = mean(b, 'omitnan');
    b0 = b - b_mu;
    X0 = Ximp - x_mu;
    denom = sum(b0 .^ 2);
    if ~isfinite(denom) || denom <= eps
        models{j} = [];
        continue;
    end
    gamma = (b0' * X0) ./ denom;
    Xorth = X0 - b0 * gamma;
    x_sd = std(Xorth, 0, 1, 'omitnan');
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = Xorth ./ x_sd;
    Xz(~isfinite(Xz)) = 0;
    r_mu = mean(r, 'omitnan');
    r_sd = std(r, 0, 'omitnan');
    if ~isfinite(r_sd) || r_sd <= eps, r_sd = 1; end
    rz = (r - r_mu) ./ r_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    w = ridge_solve_local(Xa, rz, ridge);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, ...
        'b_mu', b_mu, 'gamma', gamma, 'r_mu', r_mu, 'r_sd', r_sd);
end
end

function R_pred = apply_prediction_spatial_orth_residual_models(F_cell, models, J)
R_pred = cell(size(F_cell));
for k = 1:numel(F_cell)
    Rk = nan(size(F_cell{k}));
    if isempty(F_cell{k}) || J < 1
        R_pred{k} = Rk;
        continue;
    end
    Jk = min(J, size(F_cell{k}, 2));
    X = F_cell{k}(:, 1:Jk);
    if Jk < J
        X(:, end + 1:J) = NaN;
    end
    for j = 1:Jk
        if j > numel(models) || isempty(models{j})
            continue;
        end
        model = models{j};
        b = X(:, j);
        valid = isfinite(b) & any(isfinite(X), 2);
        Ximp = X;
        for c = 1:size(Ximp, 2)
            bad = ~isfinite(Ximp(:, c));
            Ximp(bad, c) = model.x_mu(c);
        end
        X0 = Ximp - model.x_mu;
        b0 = b - model.b_mu;
        Xorth = X0 - b0 * model.gamma;
        Xz = Xorth ./ model.x_sd;
        Xz(~isfinite(Xz)) = 0;
        rhat = ([Xz, ones(size(Xz, 1), 1)] * model.w) * model.r_sd + model.r_mu;
        Rk(valid, j) = rhat(valid);
    end
    R_pred{k} = Rk;
end
end

function [models, J] = fit_prediction_spatial_models(F_cell, Y_cell, lambda)
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_cell), 1);
if isempty(first_y)
    models = {};
    J = 0;
    return;
end
J = size(Y_cell{first_y}, 2);
X_all = [];
Y_all = [];
for k = 1:min(numel(F_cell), numel(Y_cell))
    if isempty(F_cell{k}) || isempty(Y_cell{k}) || size(F_cell{k}, 2) < J || size(Y_cell{k}, 2) < J
        continue;
    end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if n < J + 10
        continue;
    end
    X_all = [X_all; F_cell{k}(1:n, 1:J)]; %#ok<AGROW>
    Y_all = [Y_all; Y_cell{k}(1:n, 1:J)]; %#ok<AGROW>
end
models = cell(1, J);
if isempty(X_all) || isempty(Y_all)
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
for j = 1:J
    y = Y_all(:, j);
    valid = isfinite(y) & any(isfinite(X_all), 2);
    Xz = Xz_all(valid, :);
    y = y(valid);
    if size(Xz, 1) < J + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    w = ridge_solve_local(Xa, yz, ridge);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
end

function C_pred = apply_prediction_spatial_models(F_cell, models, J)
C_pred = cell(size(F_cell));
for k = 1:numel(F_cell)
    Ck = nan(size(F_cell{k}));
    if isempty(F_cell{k}) || J < 1
        C_pred{k} = Ck;
        continue;
    end
    Jk = min(J, size(F_cell{k}, 2));
    X = F_cell{k}(:, 1:Jk);
    if Jk < J
        X(:, end + 1:J) = NaN;
    end
    row_ok = any(isfinite(X), 2);
    for j = 1:Jk
        if j > numel(models) || isempty(models{j})
            continue;
        end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(row_ok, j) = yhat(row_ok);
    end
    C_pred{k} = Ck;
end
end

function [models, J] = fit_prediction_spatial_offdiag_residual_models(F_cell, Y_cell, lambda)
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_cell), 1);
if isempty(first_y)
    models = {};
    J = 0;
    return;
end
J = size(Y_cell{first_y}, 2);
X_all = [];
Y_all = [];
for k = 1:min(numel(F_cell), numel(Y_cell))
    if isempty(F_cell{k}) || isempty(Y_cell{k}) || size(F_cell{k}, 2) < J || size(Y_cell{k}, 2) < J
        continue;
    end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if n < J + 10
        continue;
    end
    X_all = [X_all; F_cell{k}(1:n, 1:J)]; %#ok<AGROW>
    Y_all = [Y_all; Y_cell{k}(1:n, 1:J)]; %#ok<AGROW>
end
models = cell(1, J);
if isempty(X_all) || isempty(Y_all)
    return;
end
for j = 1:J
    cols = setdiff(1:J, j);
    y = Y_all(:, j);
    b = X_all(:, j);
    X = X_all(:, cols);
    valid = isfinite(y) & isfinite(b) & any(isfinite(X), 2);
    X = X(valid, :);
    r = y(valid) - b(valid);
    if size(X, 1) < numel(cols) + 10 || std(r) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    Xz(~isfinite(Xz)) = 0;
    r_mu = mean(r, 'omitnan');
    r_sd = std(r, 0, 'omitnan');
    if ~isfinite(r_sd) || r_sd <= eps, r_sd = 1; end
    rz = (r - r_mu) ./ r_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    w = ridge_solve_local(Xa, rz, ridge);
    models{j} = struct('w', w, 'cols', cols, 'x_mu', x_mu, 'x_sd', x_sd, 'r_mu', r_mu, 'r_sd', r_sd);
end
end

function R_pred = apply_prediction_spatial_offdiag_residual_models(F_cell, models, J)
R_pred = cell(size(F_cell));
for k = 1:numel(F_cell)
    Rk = nan(size(F_cell{k}));
    if isempty(F_cell{k}) || J < 1
        R_pred{k} = Rk;
        continue;
    end
    Jk = min(J, size(F_cell{k}, 2));
    X_all = F_cell{k}(:, 1:Jk);
    if Jk < J
        X_all(:, end + 1:J) = NaN;
    end
    for j = 1:Jk
        if j > numel(models) || isempty(models{j})
            continue;
        end
        model = models{j};
        X = X_all(:, model.cols);
        valid = isfinite(X_all(:, j)) & any(isfinite(X), 2);
        Xz = (X - model.x_mu) ./ model.x_sd;
        Xz(~isfinite(Xz)) = 0;
        rhat = ([Xz, ones(size(Xz, 1), 1)] * model.w) * model.r_sd + model.r_mu;
        Rk(valid, j) = rhat(valid);
    end
    R_pred{k} = Rk;
end
end

function alpha = select_global_residual_alpha_by_rpt(Y_true, F_base, R_pred, guard_margin)
if nargin < 4, guard_margin = 0; end
alpha_grid = 0:0.25:1.0;
base_score = mean_rpt_all_joints(Y_true, F_base);
best_score = base_score;
best_alpha = 0;
for ai = 1:numel(alpha_grid)
    a = alpha_grid(ai);
    C_try = add_scaled_residual_cells(F_base, R_pred, [], a);
    score = mean_rpt_all_joints(Y_true, C_try);
    if isfinite(score) && (~isfinite(best_score) || score > best_score)
        best_score = score;
        best_alpha = a;
    end
end
if isfinite(best_score) && (~isfinite(base_score) || best_score >= base_score + guard_margin)
    alpha = best_alpha;
else
    alpha = 0;
end
end

function C_pred = guarded_spatial_prediction_cells(base_cell, spatial_cell, use_spatial)
C_pred = cell(size(base_cell));
for k = 1:numel(base_cell)
    Ck = base_cell{k};
    if isempty(Ck) || k > numel(spatial_cell) || isempty(spatial_cell{k})
        C_pred{k} = Ck;
        continue;
    end
    n = min(size(Ck, 1), size(spatial_cell{k}, 1));
    J = min([size(Ck, 2), size(spatial_cell{k}, 2), numel(use_spatial)]);
    for j = 1:J
        if ~use_spatial(j)
            continue;
        end
        y = spatial_cell{k}(1:n, j);
        ok = isfinite(y);
        Ck(ok, j) = y(ok);
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_masked(F_oof, Y_train, F_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_masked requires at least one lag sample');
end
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = F_te;
    return;
end
J = size(Y_train{first_y}, 2);
models = cell(1, J);
for j = 1:J
    [X, M, y] = fir_lag_design_masked(F_oof, Y_train, j, lags);
    valid = isfinite(y);
    X = X(valid, :);
    M = M(valid, :);
    y = y(valid);
    y_std = std(y);
    if size(X, 1) < 2 * numel(lags) + 10 || ~isfinite(y_std) || y_std <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    [Ximp, M] = impute_masked_fir_values(X, M, x_mu);
    Xfeat = [Ximp, M];
    f_mu = mean(Xfeat, 1, 'omitnan');
    f_sd = std(Xfeat, 0, 1, 'omitnan');
    f_mu(~isfinite(f_mu)) = 0;
    f_sd(~isfinite(f_sd) | f_sd <= eps) = 1;
    Xz = (Xfeat - f_mu) ./ f_sd;
    Xz(~isfinite(Xz)) = 0;
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'x_mu', x_mu, 'f_mu', f_mu, 'f_sd', f_sd, ...
        'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}) || size(F_te{k}, 2) < j
            continue;
        end
        [X, M, idx] = fir_lag_matrix_masked(F_te{k}(:, j), lags);
        if isempty(X)
            continue;
        end
        [Ximp, M] = impute_masked_fir_values(X, M, models{j}.x_mu);
        Xfeat = [Ximp, M];
        Xz = (Xfeat - models{j}.f_mu) ./ models{j}.f_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_mask_only(F_oof, Y_train, F_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_mask_only requires at least one lag sample');
end
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = F_te;
    return;
end
J = size(Y_train{first_y}, 2);
models = cell(1, J);
for j = 1:J
    [~, M, y] = fir_lag_design_masked(F_oof, Y_train, j, lags);
    valid = isfinite(y);
    M = M(valid, :);
    y = y(valid);
    y_std = std(y);
    if size(M, 1) < numel(lags) + 10 || ~isfinite(y_std) || y_std <= eps
        models{j} = [];
        continue;
    end
    Xfeat = double(M > 0);
    f_mu = mean(Xfeat, 1, 'omitnan');
    f_sd = std(Xfeat, 0, 1, 'omitnan');
    f_mu(~isfinite(f_mu)) = 0;
    f_sd(~isfinite(f_sd) | f_sd <= eps) = 1;
    Xz = (Xfeat - f_mu) ./ f_sd;
    Xz(~isfinite(Xz)) = 0;
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'f_mu', f_mu, 'f_sd', f_sd, ...
        'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}) || size(F_te{k}, 2) < j
            continue;
        end
        [~, M, idx] = fir_lag_matrix_masked(F_te{k}(:, j), lags);
        if isempty(M)
            continue;
        end
        Xfeat = double(M > 0);
        Xz = (Xfeat - models{j}.f_mu) ./ models{j}.f_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_multi_horizon_weighted(F_oof, Y_train, F_te, lambda, lags, fs, guard_margin)
if nargin < 6 || isempty(fs), fs = 250; end
if nargin < 7 || isempty(guard_margin), guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_multi_horizon_weighted requires at least one lag sample');
end
max_age = max(0, -min(lags));
if max_age == 0
    C_pred = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
    return;
end
candidate_ages = unique(round([0 0.5 1.0 1.5 2.0 2.5 3.0] * double(fs)));
candidate_ages = candidate_ages(candidate_ages >= 0 & candidate_ages <= max_age);
if isempty(candidate_ages) || candidate_ages(1) ~= 0
    candidate_ages = [0 candidate_ages];
end
if candidate_ages(end) ~= max_age
    candidate_ages(end + 1) = max_age;
end
candidate_ages = unique(candidate_ages);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = F_te;
    return;
end
J = size(Y_train{first_y}, 2);
H = numel(candidate_ages);
C_te_all = cell(1, H);
weights = zeros(J, H);
base_r = nan(1, J);
for j = 1:J
    base_r(j) = safe_rpt_cells_local(Y_train, F_oof, j);
end
for hi = 1:H
    age = candidate_ages(hi);
    cand_lags = lags(lags >= -age & lags <= 0);
    if isempty(cand_lags)
        cand_lags = 0;
    end
    C_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, cand_lags);
    C_te_all{hi} = post_prediction_fir(F_oof, Y_train, F_te, lambda, cand_lags);
    for j = 1:J
        r_cal = safe_rpt_cells_local(Y_train, C_oof, j);
        gain = r_cal - base_r(j) - guard_margin;
        if isfinite(gain) && gain > 0
            weights(j, hi) = gain;
        end
    end
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    if isempty(F_te{k})
        C_pred{k} = [];
        continue;
    end
    Ck = F_te{k};
    n = size(Ck, 1);
    Jk = min(J, size(Ck, 2));
    for j = 1:Jk
        if sum(weights(j, :)) <= 0
            continue;
        end
        num = zeros(n, 1);
        den = zeros(n, 1);
        for hi = 1:H
            w = weights(j, hi);
            if w <= 0 || k > numel(C_te_all{hi}) || isempty(C_te_all{hi}{k})
                continue;
            end
            nk = min(n, size(C_te_all{hi}{k}, 1));
            if size(C_te_all{hi}{k}, 2) < j || nk < 1
                continue;
            end
            yhat = C_te_all{hi}{k}(1:nk, j);
            mask = isfinite(yhat);
            if any(mask)
                tmp = zeros(nk, 1);
                tmp(mask) = yhat(mask);
                num(1:nk) = num(1:nk) + w * tmp;
                den(1:nk) = den(1:nk) + w * double(mask);
            end
        end
        mask = den > 0;
        Ck(mask, j) = num(mask) ./ den(mask);
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
C_train = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
C_pred = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
use_calib = false(1, J);
for j = 1:J
    r_base = safe_corr_cells_local(Y_train, F_oof, j);
    r_cal = safe_corr_cells_local(Y_train, C_train, j);
    use_calib(j) = isfinite(r_cal) && (~isfinite(r_base) || r_cal >= r_base + guard_margin);
end
for k = 1:numel(C_pred)
    n = min(size(C_pred{k}, 1), size(F_te{k}, 1));
    Jk = min([size(C_pred{k}, 2), size(F_te{k}, 2), J]);
    for j = 1:Jk
        if ~use_calib(j)
            C_pred{k}(1:n, j) = F_te{k}(1:n, j);
        end
    end
end
end

function C_pred = post_prediction_fir_grid_guard(F_oof, Y_train, F_te, lambda, lags, fs, guard_margin)
if nargin < 7, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_grid_guard requires at least one lag sample');
end
if nargin < 6 || isempty(fs), fs = 250; end
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
C_train_base = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
C_pred = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
best_r = nan(1, J);
for j = 1:J
    best_r(j) = safe_rpt_cells_local(Y_train, C_train_base, j);
end
lag_ms_starts = [-3000, -2500, -2000, -1500, -1000, -750];
lambda_grid = unique([double(lambda), 1, 3, 10, 30, 100, 300, 1000]);
for li = 1:numel(lag_ms_starts)
    cand_lags = post_fir_lags_to_samples(lag_ms_starts(li):40:0, fs, 'ms', true);
    if isempty(cand_lags), continue; end
    for ri = 1:numel(lambda_grid)
        cand_lambda = lambda_grid(ri);
        C_train = post_prediction_fir(F_oof, Y_train, F_oof, cand_lambda, cand_lags);
        improve = false(1, J);
        cand_r = nan(1, J);
        for j = 1:J
            cand_r(j) = safe_rpt_cells_local(Y_train, C_train, j);
            improve(j) = isfinite(cand_r(j)) && ...
                (~isfinite(best_r(j)) || cand_r(j) > best_r(j) + guard_margin);
        end
        if ~any(improve), continue; end
        C_te = post_prediction_fir(F_oof, Y_train, F_te, cand_lambda, cand_lags);
        for k = 1:numel(C_pred)
            if isempty(C_pred{k}) || k > numel(C_te) || isempty(C_te{k}), continue; end
            n = min(size(C_pred{k}, 1), size(C_te{k}, 1));
            Jk = min([size(C_pred{k}, 2), size(C_te{k}, 2), J]);
            for j = 1:Jk
                if improve(j)
                    C_pred{k}(1:n, j) = C_te{k}(1:n, j);
                end
            end
        end
        best_r(improve) = cand_r(improve);
    end
end
end

function C_pred = post_prediction_fir_residual_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_guard requires at least one lag sample');
end
[models, J] = fit_prediction_fir_residual_models(F_oof, Y_train, lambda, lags);
R_oof = apply_prediction_fir_residual_models(F_oof, models, lags, J);
R_te = apply_prediction_fir_residual_models(F_te, models, lags, J);
beta = select_residual_betas_by_rpt(Y_train, F_oof, R_oof, J, guard_margin);
C_pred = add_scaled_residual_cells(F_te, R_te, [], beta);
end

function C_pred = post_prediction_fir_residual_on_fir_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_on_fir_guard requires at least one lag sample');
end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_fir_residual_models(base_oof, Y_train, lambda, lags);
R_oof = apply_prediction_fir_residual_models(base_oof, models, lags, J);
R_te = apply_prediction_fir_residual_models(base_te, models, lags, J);
beta = select_residual_betas_by_rpt(Y_train, base_oof, R_oof, J, guard_margin);
C_pred = add_scaled_residual_cells(base_te, R_te, [], beta);
end

function C_pred = post_prediction_fir_residual_on_fir_reliability_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_on_fir_reliability_guard requires at least one lag sample');
end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_fir_residual_models(base_oof, Y_train, lambda, lags);
R_oof = apply_prediction_fir_residual_models(base_oof, models, lags, J);
R_te = apply_prediction_fir_residual_models(base_te, models, lags, J);
beta = select_residual_betas_by_reliability(Y_train, base_oof, R_oof, J, guard_margin);
C_pred = add_scaled_residual_cells(base_te, R_te, [], beta);
end

function C_pred = post_prediction_fir_residual_on_fir_session_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_on_fir_session_guard requires at least one lag sample');
end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_fir_residual_models(base_oof, Y_train, lambda, lags);
R_oof = apply_prediction_fir_residual_models(base_oof, models, lags, J);
R_te = apply_prediction_fir_residual_models(base_te, models, lags, J);
beta = select_residual_betas_by_reliability(Y_train, base_oof, R_oof, J, guard_margin);
C_oof = add_scaled_residual_cells(base_oof, R_oof, [], beta);
if mean_rpt_across_joints_local(Y_train, C_oof, J) < mean_rpt_across_joints_local(Y_train, base_oof, J) + guard_margin
    beta(:) = 0;
end
C_pred = add_scaled_residual_cells(base_te, R_te, [], beta);
end

function C_pred = post_prediction_fir_residual_on_fir_highbase_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_on_fir_highbase_guard requires at least one lag sample');
end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_fir_residual_models(base_oof, Y_train, lambda, lags);
R_oof = apply_prediction_fir_residual_models(base_oof, models, lags, J);
R_te = apply_prediction_fir_residual_models(base_te, models, lags, J);
beta = select_residual_betas_by_highbase(Y_train, base_oof, R_oof, J, guard_margin);
C_pred = add_scaled_residual_cells(base_te, R_te, [], beta);
end

function C_pred = post_prediction_fir_residual_on_fir_quality_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_on_fir_quality_guard requires at least one lag sample');
end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_fir_residual_models(base_oof, Y_train, lambda, lags);
R_oof = apply_prediction_fir_residual_models(base_oof, models, lags, J);
R_te = apply_prediction_fir_residual_models(base_te, models, lags, J);
beta = select_residual_betas_by_quality(Y_train, base_oof, R_oof, J, guard_margin);
C_pred = add_scaled_residual_cells(base_te, R_te, [], beta);
end

function C_pred = post_prediction_fir_residual_on_fir_robust_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_on_fir_robust_guard requires at least one lag sample');
end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_fir_residual_models(base_oof, Y_train, lambda, lags);
R_oof = apply_prediction_fir_residual_models(base_oof, models, lags, J);
R_te = apply_prediction_fir_residual_models(base_te, models, lags, J);
beta = select_residual_betas_by_robust_rpt(Y_train, base_oof, R_oof, J, guard_margin);
C_pred = add_scaled_residual_cells(base_te, R_te, [], beta);
end

function C_pred = post_prediction_fir_residual_on_fir_stability_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_on_fir_stability_guard requires at least one lag sample');
end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_fir_residual_models(base_oof, Y_train, lambda, lags);
R_oof = apply_prediction_fir_residual_models(base_oof, models, lags, J);
R_te = apply_prediction_fir_residual_models(base_te, models, lags, J);
beta = select_residual_betas_by_stability(Y_train, base_oof, R_oof, J, guard_margin);
C_pred = add_scaled_residual_cells(base_te, R_te, [], beta);
end

function C_pred = post_prediction_fir_residual_on_fir_smooth_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_on_fir_smooth_guard requires at least one lag sample');
end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_fir_residual_models(base_oof, Y_train, lambda, lags, 'smooth');
R_oof = apply_prediction_fir_residual_models(base_oof, models, lags, J);
R_te = apply_prediction_fir_residual_models(base_te, models, lags, J);
beta = select_residual_betas_by_rpt(Y_train, base_oof, R_oof, J, guard_margin);
C_pred = add_scaled_residual_cells(base_te, R_te, [], beta);
end

function C_pred = post_prediction_fir_residual_multi_on_fir_guard(F_oof, Y_train, F_te, lambda, lags, guard_margin)
if nargin < 6, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_multi_on_fir_guard requires at least one lag sample');
end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
[models, J] = fit_prediction_fir_residual_multi_models(base_oof, Y_train, lambda, lags);
R_oof = apply_prediction_fir_residual_multi_models(base_oof, models, lags, J);
R_te = apply_prediction_fir_residual_multi_models(base_te, models, lags, J);
beta = select_residual_betas_by_rpt(Y_train, base_oof, R_oof, J, guard_margin);
C_pred = add_scaled_residual_cells(base_te, R_te, [], beta);
end

function [models, J] = fit_prediction_fir_residual_multi_models(F_oof, Y_train, lambda, lags)
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    J = 0;
    models = {};
    return;
end
J = size(Y_train{first_y}, 2);
[X_all, R_all] = fir_lag_design_residual_multi(F_oof, Y_train, lags, J);
valid_x = all(isfinite(X_all), 2);
X_all = X_all(valid_x, :);
R_all = R_all(valid_x, :);
models = cell(1, J);
if isempty(X_all) || isempty(R_all)
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
for j = 1:J
    y = R_all(:, j);
    valid = isfinite(y);
    Xz = Xz_all(valid, :);
    y = y(valid);
    if size(Xz, 1) < size(Xz, 2) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    w = ridge_solve_local(Xa, yz, ridge);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
end

function R_pred = apply_prediction_fir_residual_multi_models(F_cell, models, lags, J)
R_pred = cell(size(F_cell));
for k = 1:numel(F_cell)
    Rk = nan(size(F_cell{k}));
    [X, idx] = fir_lag_matrix_multi(F_cell{k}, lags, J);
    if isempty(X)
        R_pred{k} = Rk;
        continue;
    end
    for j = 1:J
        if j > numel(models) || isempty(models{j}), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Rk(idx, j) = yhat;
    end
    R_pred{k} = Rk;
end
end

function [models, J] = fit_prediction_fir_residual_models(F_oof, Y_train, lambda, lags, penalty_mode)
if nargin < 5, penalty_mode = 'ridge'; end
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    J = 0;
    models = {};
    return;
end
J = size(Y_train{first_y}, 2);
models = cell(1, J);
for j = 1:J
    [X, y] = fir_lag_design_residual(F_oof, Y_train, j, lags);
    valid = isfinite(y) & all(isfinite(X), 2);
    X = X(valid, :);
    y = y(valid);
    if size(X, 1) < numel(lags) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    if strcmpi(char(penalty_mode), 'smooth')
        w = smooth_fir_solve_local(Xa, yz, ridge, numel(lags));
    else
        w = ridge_solve_local(Xa, yz, ridge);
    end
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
end

function w = smooth_fir_solve_local(Xa, y, lambda, n_lag)
ridge = max(double(lambda), 0);
P = zeros(size(Xa, 2));
if n_lag >= 3
    D = diff(eye(n_lag), 2, 1);
    P(1:n_lag, 1:n_lag) = ridge * (D' * D) + max(ridge, 1) * 1e-6 * eye(n_lag);
else
    P(1:n_lag, 1:n_lag) = ridge * eye(n_lag);
end
w = (Xa' * Xa + P) \ (Xa' * y);
if any(~isfinite(w))
    w = ridge_solve_local(Xa, y, ridge);
end
end

function beta = select_residual_betas_by_rpt(Y_true, F_base, R_pred, J, guard_margin)
beta_grid = 0:0.25:1.25;
beta = zeros(1, J);
for j = 1:J
    r_base = safe_rpt_cells_local(Y_true, F_base, j);
    best_r = r_base;
    best_beta = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        C_try = add_scaled_residual_cells(F_base, R_pred, j, b);
        r_try = safe_rpt_cells_local(Y_true, C_try, j);
        if isfinite(r_try) && (~isfinite(best_r) || r_try > best_r)
            best_r = r_try;
            best_beta = b;
        end
    end
    if isfinite(best_r) && (~isfinite(r_base) || best_r >= r_base + guard_margin)
        beta(j) = best_beta;
    end
end
end

function beta = select_residual_betas_by_reliability(Y_true, F_base, R_pred, J, guard_margin)
beta_grid = 0:0.25:1.25;
tau = 0.01;
beta = zeros(1, J);
for j = 1:J
    r_base = safe_rpt_cells_local(Y_true, F_base, j);
    best_r = r_base;
    best_beta = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        C_try = add_scaled_residual_cells(F_base, R_pred, j, b);
        r_try = safe_rpt_cells_local(Y_true, C_try, j);
        if isfinite(r_try) && (~isfinite(best_r) || r_try > best_r)
            best_r = r_try;
            best_beta = b;
        end
    end
    gain = best_r - r_base;
    if isfinite(gain) && gain >= guard_margin && best_beta > 0
        beta(j) = best_beta * gain / (gain + tau);
    end
end
end

function r = mean_rpt_across_joints_local(Y_true, Y_pred, J)
vals = nan(1, J);
for j = 1:J
    vals(j) = safe_rpt_cells_local(Y_true, Y_pred, j);
end
vals = vals(isfinite(vals));
if isempty(vals)
    r = NaN;
else
    r = mean(vals, 'omitnan');
end
end

function beta = select_residual_betas_by_highbase(Y_true, F_base, R_pred, J, guard_margin)
beta_grid = 0:0.25:1.25;
tau = 0.01;
high_base_cut = 0.55;
high_gain_cut = 0.02;
beta = zeros(1, J);
for j = 1:J
    r_base = safe_rpt_cells_local(Y_true, F_base, j);
    best_r = r_base;
    best_beta = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        C_try = add_scaled_residual_cells(F_base, R_pred, j, b);
        r_try = safe_rpt_cells_local(Y_true, C_try, j);
        if isfinite(r_try) && (~isfinite(best_r) || r_try > best_r)
            best_r = r_try;
            best_beta = b;
        end
    end
    gain = best_r - r_base;
    if ~(isfinite(gain) && gain >= guard_margin && best_beta > 0)
        continue;
    end
    if isfinite(r_base) && r_base >= high_base_cut && gain < high_gain_cut
        continue;
    end
    beta(j) = best_beta * gain / (gain + tau);
end
end

function beta = select_residual_betas_by_quality(Y_true, F_base, R_pred, J, guard_margin)
beta_grid = 0:0.25:1.25;
beta = zeros(1, J);
for j = 1:J
    r_base = safe_rpt_cells_local(Y_true, F_base, j);
    best_r = r_base;
    best_beta = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        C_try = add_scaled_residual_cells(F_base, R_pred, j, b);
        r_try = safe_rpt_cells_local(Y_true, C_try, j);
        if isfinite(r_try) && (~isfinite(best_r) || r_try > best_r)
            best_r = r_try;
            best_beta = b;
        end
    end
    gain = best_r - r_base;
    if isfinite(gain) && gain >= guard_margin && best_beta > 0
        quality = min(max(r_base, 0), 0.70) / 0.70;
        tau = 0.005 + 0.025 * (1 - quality)^2;
        beta(j) = best_beta * gain / (gain + tau);
    end
end
end

function beta = select_residual_betas_by_robust_rpt(Y_true, F_base, R_pred, J, guard_margin)
beta_grid = 0:0.25:1.25;
beta = zeros(1, J);
for j = 1:J
    base_trials = trial_rpt_cells_local(Y_true, F_base, j);
    base_score = robust_trial_score_local(base_trials);
    best_score = base_score;
    best_beta = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        C_try = add_scaled_residual_cells(F_base, R_pred, j, b);
        trial_r = trial_rpt_cells_local(Y_true, C_try, j);
        score = robust_trial_score_local(trial_r);
        if isfinite(score) && (~isfinite(best_score) || score > best_score)
            best_score = score;
            best_beta = b;
        end
    end
    if isfinite(best_score) && (~isfinite(base_score) || best_score >= base_score + guard_margin)
        beta(j) = best_beta;
    end
end
end

function score = robust_trial_score_local(r_trials)
r = r_trials(isfinite(r_trials));
if isempty(r)
    score = NaN;
    return;
end
score = mean(r, 'omitnan') + 0.25 * quantile(r, 0.25);
end

function beta = select_residual_betas_by_stability(Y_true, F_base, R_pred, J, guard_margin)
beta_grid = 0:0.25:1.25;
tau = 0.01;
beta = zeros(1, J);
for j = 1:J
    r_base = safe_rpt_cells_local(Y_true, F_base, j);
    best_r = r_base;
    best_beta = 0;
    for bi = 1:numel(beta_grid)
        b = beta_grid(bi);
        C_try = add_scaled_residual_cells(F_base, R_pred, j, b);
        r_try = safe_rpt_cells_local(Y_true, C_try, j);
        if isfinite(r_try) && (~isfinite(best_r) || r_try > best_r)
            best_r = r_try;
            best_beta = b;
        end
    end
    gain = best_r - r_base;
    if ~(isfinite(gain) && gain >= guard_margin && best_beta > 0)
        continue;
    end
    C_best = add_scaled_residual_cells(F_base, R_pred, j, best_beta);
    trial_gain = trial_rpt_gain_cells_local(Y_true, F_base, C_best, j);
    trial_gain = trial_gain(isfinite(trial_gain));
    if numel(trial_gain) >= 4
        pos_frac = mean(trial_gain > 0);
        med_gain = median(trial_gain, 'omitnan');
    else
        pos_frac = 0;
        med_gain = -inf;
    end
    if pos_frac >= 0.60 && med_gain >= 0
        beta(j) = best_beta;
    else
        beta(j) = best_beta * gain / (gain + tau);
    end
end
end

function gain = trial_rpt_gain_cells_local(Y_true, Y_base, Y_try, j)
gain = nan(1, numel(Y_true));
for k = 1:numel(Y_true)
    if k > numel(Y_base) || k > numel(Y_try), continue; end
    if isempty(Y_true{k}) || isempty(Y_base{k}) || isempty(Y_try{k}), continue; end
    if size(Y_true{k}, 2) < j || size(Y_base{k}, 2) < j || size(Y_try{k}, 2) < j, continue; end
    n = min([size(Y_true{k}, 1), size(Y_base{k}, 1), size(Y_try{k}, 1)]);
    if n < 2, continue; end
    r_base = safe_corr_local(Y_true{k}(1:n, j), Y_base{k}(1:n, j));
    r_try = safe_corr_local(Y_true{k}(1:n, j), Y_try{k}(1:n, j));
    gain(k) = r_try - r_base;
end
end

function r_trials = trial_rpt_cells_local(Y_true, Y_pred, j)
r_trials = nan(1, numel(Y_true));
for k = 1:numel(Y_true)
    if k > numel(Y_pred) || isempty(Y_true{k}) || isempty(Y_pred{k}), continue; end
    if size(Y_true{k}, 2) < j || size(Y_pred{k}, 2) < j, continue; end
    n = min(size(Y_true{k}, 1), size(Y_pred{k}, 1));
    if n < 2, continue; end
    r_trials(k) = safe_corr_local(Y_true{k}(1:n, j), Y_pred{k}(1:n, j));
end
end

function R_pred = apply_prediction_fir_residual_models(F_cell, models, lags, J)
R_pred = cell(size(F_cell));
for k = 1:numel(F_cell)
    Rk = nan(size(F_cell{k}));
    for j = 1:J
        if j > numel(models) || isempty(models{j}), continue; end
        [X, idx] = fir_lag_matrix(F_cell{k}(:, j), lags);
        if isempty(X), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Rk(idx, j) = yhat;
    end
    R_pred{k} = Rk;
end
end

function C_cell = add_scaled_residual_cells(F_cell, R_cell, joint_idx, beta)
C_cell = cell(size(F_cell));
for k = 1:numel(F_cell)
    Ck = F_cell{k};
    if isempty(Ck) || k > numel(R_cell) || isempty(R_cell{k})
        C_cell{k} = Ck;
        continue;
    end
    n = min(size(Ck, 1), size(R_cell{k}, 1));
    J = min(size(Ck, 2), size(R_cell{k}, 2));
    if isempty(joint_idx)
        joints = 1:J;
    else
        joints = joint_idx;
    end
    for ji = 1:numel(joints)
        j = joints(ji);
        if j > J, continue; end
        if numel(beta) == 1
            b = beta;
        elseif j <= numel(beta)
            b = beta(j);
        else
            b = 0;
        end
        r = R_cell{k}(1:n, j);
        ok = isfinite(r);
        idx = find(ok);
        Ck(idx, j) = Ck(idx, j) + b * r(idx);
    end
    C_cell{k} = Ck;
end
end

function C_pred = post_prediction_fir_basis(F_oof, Y_train, F_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_basis requires at least one lag sample');
end
B = fir_lag_smooth_basis(lags);
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
models = cell(1, J);
for j = 1:J
    [X, y] = fir_lag_design(F_oof, Y_train, j, lags);
    if ~isempty(X)
        X = X * B;
    end
    valid = isfinite(y) & all(isfinite(X), 2);
    X = X(valid, :);
    y = y(valid);
    if size(X, 1) < size(X, 2) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, ...
        'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}), continue; end
        [X, idx] = fir_lag_matrix(F_te{k}(:, j), lags);
        if isempty(X), continue; end
        X = X * B;
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_apf(A_oof, P_oof, F_oof, Y_train, A_te, P_te, F_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
models = cell(1, J);
for j = 1:J
    [X, y] = fir_lag_design_apf(A_oof, P_oof, F_oof, Y_train, j, lags);
    valid = isfinite(y) & all(isfinite(X), 2);
    X = X(valid, :);
    y = y(valid);
    if size(X, 1) < size(X, 2) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}), continue; end
        [X, idx] = fir_lag_matrix_apf(A_te{k}(:, j), P_te{k}(:, j), F_te{k}(:, j), lags);
        if isempty(X), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_apf_router(A_oof, P_oof, F_oof, Y_train, A_te, P_te, F_te, lambda)
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
models = cell(1, J);
for j = 1:J
    [X, H, y] = apf_router_design(A_oof, P_oof, F_oof, Y_train, j);
    valid = isfinite(y) & all(isfinite(X), 2) & all(isfinite(H), 2);
    X = X(valid, :);
    H = H(valid, :);
    y = y(valid);
    if size(X, 1) < size(X, 2) + 30 || std(y, 0, 'omitnan') <= eps
        models{j} = [];
        continue;
    end
    x_mu = mean(X, 1, 'omitnan');
    x_sd = std(X, 0, 1, 'omitnan');
    x_mu(~isfinite(x_mu)) = 0;
    x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
    Xz = (X - x_mu) ./ x_sd;
    Xz(~isfinite(Xz)) = 0;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    W = zeros(size(Xa, 2), 3);
    for h = 1:3
        q = -abs(H(:, h) - y);
        q_mu = mean(q, 'omitnan');
        q_sd = std(q, 0, 'omitnan');
        if ~isfinite(q_sd) || q_sd <= eps, q_sd = 1; end
        qz = (q - q_mu) ./ q_sd;
        W(:, h) = ridge_solve_local(Xa, qz, ridge);
    end
    models{j} = struct('W', W, 'x_mu', x_mu, 'x_sd', x_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        if isempty(models{j}) || isempty(A_te{k}) || isempty(P_te{k}) || isempty(F_te{k})
            continue;
        end
        n = min([size(A_te{k}, 1), size(P_te{k}, 1), size(F_te{k}, 1)]);
        H = [A_te{k}(1:n, j), P_te{k}(1:n, j), F_te{k}(1:n, j)];
        X = apf_router_features(H);
        valid = all(isfinite(X), 2) & all(isfinite(H), 2);
        if ~any(valid), continue; end
        Xz = (X(valid, :) - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        scores = [Xz, ones(size(Xz, 1), 1)] * models{j}.W;
        scores = scores - max(scores, [], 2);
        wt = exp(scores);
        wt = wt ./ max(sum(wt, 2), eps);
        yhat = sum(wt .* H(valid, :), 2);
        idx = find(valid);
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_multi(F_oof, Y_train, F_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
[X_all, Y_all] = fir_lag_design_multi(F_oof, Y_train, lags, J);
models = cell(1, J);
if isempty(X_all) || isempty(Y_all)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
for j = 1:J
    y = Y_all(:, j);
    valid = isfinite(y);
    Xz = Xz_all(valid, :);
    y = y(valid);
    if size(Xz, 1) < size(Xz, 2) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    [X, idx] = fir_lag_matrix_multi(F_te{k}, lags, J);
    if isempty(X)
        C_pred{k} = Ck;
        continue;
    end
    for j = 1:J
        if isempty(models{j}), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_multi_basis(F_oof, Y_train, F_te, lambda, lags, n_basis)
if nargin < 6, n_basis = 8; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
J = size(Y_train{first_y}, 2);
LagB = fir_lag_smooth_basis_n(lags, n_basis);
[X_all, Y_all] = fir_lag_design_multi_basis(F_oof, Y_train, lags, J, LagB);
models = cell(1, J);
if isempty(X_all) || isempty(Y_all)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
for j = 1:J
    y = Y_all(:, j);
    valid = isfinite(y);
    Xz = Xz_all(valid, :);
    y = y(valid);
    if size(Xz, 1) < size(Xz, 2) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    w = ridge_solve_local(Xa, yz, ridge);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    [X, idx] = fir_lag_matrix_multi_basis(F_te{k}, lags, J, LagB);
    if isempty(X)
        C_pred{k} = Ck;
        continue;
    end
    for j = 1:J
        if isempty(models{j}), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_multi_basis_lowrank(F_oof, Y_train, F_te, n_components, lambda, lags, n_basis)
if nargin < 7, n_basis = 8; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
J = size(Y_train{first_y}, 2);
K = max(1, min(round(double(n_components)), J));
LagB = fir_lag_smooth_basis_n(lags, n_basis);
[X_all, Y_all] = fir_lag_design_multi_basis(F_oof, Y_train, lags, J, LagB);
if isempty(X_all) || isempty(Y_all)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
valid = any(isfinite(Y_all(:, 1:J)), 2);
X_all = X_all(valid, :);
Y_all = Y_all(valid, 1:J);
if size(X_all, 1) < max(20, K + 10)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
y_mu = mean(Y_all, 1, 'omitnan');
y_sd = std(Y_all, 0, 1, 'omitnan');
y_mu(~isfinite(y_mu)) = 0;
y_sd(~isfinite(y_sd) | y_sd <= eps) = 1;
Yz_all = (Y_all - y_mu) ./ y_sd;
Yz_all(~isfinite(Yz_all)) = 0;
[~, ~, V] = svd(Yz_all, 'econ');
if isempty(V)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
Ybasis = V(:, 1:min(K, size(V, 2)));
Z_all = Yz_all * Ybasis;
models = cell(1, size(Ybasis, 2));
Xa = [Xz_all, ones(size(Xz_all, 1), 1)];
for c = 1:size(Ybasis, 2)
    z = Z_all(:, c);
    if std(z, 0, 'omitnan') <= eps
        models{c} = [];
        continue;
    end
    models{c} = ridge_solve_local(Xa, z, ridge);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    [X, idx] = fir_lag_matrix_multi_basis(F_te{k}, lags, J, LagB);
    if isempty(X)
        C_pred{k} = Ck;
        continue;
    end
    Xz = (X - x_mu) ./ x_sd;
    Xz(~isfinite(Xz)) = 0;
    Xza = [Xz, ones(size(Xz, 1), 1)];
    Zh = zeros(size(Xza, 1), size(Ybasis, 2));
    for c = 1:size(Ybasis, 2)
        if isempty(models{c}), continue; end
        Zh(:, c) = Xza * models{c};
    end
    Yh = (Zh * Ybasis') .* y_sd + y_mu;
    Ck(idx, 1:J) = Yh(:, 1:J);
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_multi_basis_lowrank_on_fir_guard(F_oof, Y_train, F_te, n_components, lambda, lags, guard_margin, n_basis)
if nargin < 7, guard_margin = 0; end
if nargin < 8, n_basis = 8; end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
C_oof = post_prediction_fir_multi_basis_lowrank(base_oof, Y_train, base_oof, n_components, lambda, lags, n_basis);
C_pred = post_prediction_fir_multi_basis_lowrank(base_oof, Y_train, base_te, n_components, lambda, lags, n_basis);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = base_te;
    return;
end
J = size(Y_train{first_y}, 2);
use_lowrank = false(1, J);
for j = 1:J
    r_base = safe_rpt_cells_local(Y_train, base_oof, j);
    r_cal = safe_rpt_cells_local(Y_train, C_oof, j);
    use_lowrank(j) = isfinite(r_cal) && (~isfinite(r_base) || r_cal >= r_base + guard_margin);
end
for k = 1:numel(C_pred)
    if isempty(C_pred{k}) || k > numel(base_te) || isempty(base_te{k}), continue; end
    n = min(size(C_pred{k}, 1), size(base_te{k}, 1));
    Jk = min([J, size(C_pred{k}, 2), size(base_te{k}, 2)]);
    for j = 1:Jk
        if ~use_lowrank(j)
            C_pred{k}(1:n, j) = base_te{k}(1:n, j);
        end
    end
end
end

function C_pred = post_prediction_fir_multi_pls(F_oof, Y_train, F_te, n_components, lambda, lags, use_basis, n_basis)
if nargin < 8, n_basis = 8; end
model = fit_prediction_fir_multi_pls_model(F_oof, Y_train, n_components, lambda, lags, use_basis, n_basis);
C_pred = apply_prediction_fir_multi_pls_model(F_te, model);
end

function C_pred = post_prediction_fir_multi_pls_guard(F_oof, Y_train, F_te, n_components, lambda, lags, use_basis, guard_margin, n_basis)
if nargin < 8, guard_margin = 0; end
if nargin < 9, n_basis = 8; end
model = fit_prediction_fir_multi_pls_model(F_oof, Y_train, n_components, lambda, lags, use_basis, n_basis);
C_oof = apply_prediction_fir_multi_pls_model(F_oof, model);
C_pred = apply_prediction_fir_multi_pls_model(F_te, model);
if isempty(model) || ~isfield(model, 'J') || model.J < 1
    C_pred = F_te;
    return;
end
use_pls = true(1, model.J);
for j = 1:model.J
    r_base = safe_rpt_cells_local(Y_train, F_oof, j);
    r_pls = safe_rpt_cells_local(Y_train, C_oof, j);
    use_pls(j) = isfinite(r_pls) && (~isfinite(r_base) || r_pls >= r_base + guard_margin);
end
for k = 1:numel(C_pred)
    if isempty(C_pred{k}) || k > numel(F_te) || isempty(F_te{k}), continue; end
    n = min(size(C_pred{k}, 1), size(F_te{k}, 1));
    Jk = min([model.J, size(C_pred{k}, 2), size(F_te{k}, 2)]);
    for j = 1:Jk
        if ~use_pls(j)
            C_pred{k}(1:n, j) = F_te{k}(1:n, j);
        end
    end
end
end

function model = fit_prediction_fir_multi_pls_model(F_oof, Y_train, n_components, lambda, lags, use_basis, n_basis)
if nargin < 7, n_basis = 8; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    model = struct('ok', false, 'J', 0);
    return;
end
J = size(Y_train{first_y}, 2);
LagB = [];
if use_basis
    LagB = fir_lag_smooth_basis_n(lags, n_basis);
    [X_all, Y_all] = fir_lag_design_multi_basis(F_oof, Y_train, lags, J, LagB);
else
    [X_all, Y_all] = fir_lag_design_multi(F_oof, Y_train, lags, J);
end
model = struct('ok', false, 'J', J, 'lags', lags, 'use_basis', use_basis, 'LagB', LagB);
if isempty(X_all) || isempty(Y_all)
    return;
end
valid = any(isfinite(Y_all(:, 1:J)), 2);
X_all = X_all(valid, :);
Y_all = Y_all(valid, 1:J);
if size(X_all, 1) < 20 || size(Y_all, 1) < 20
    return;
end
[Xz_all, x_mu, x_sd] = zscore_feature_matrix_local(X_all);
y_mu = mean(Y_all, 1, 'omitnan');
y_sd = std(Y_all, 0, 1, 'omitnan');
y_mu(~isfinite(y_mu)) = 0;
y_sd(~isfinite(y_sd) | y_sd <= eps) = 1;
Yz_all = (Y_all - y_mu) ./ y_sd;
Yz_all(~isfinite(Yz_all)) = 0;
Cxy = Xz_all' * Yz_all;
[U, ~, ~] = svd(Cxy, 'econ');
K = min([max(1, round(double(n_components))), size(U, 2), size(Xz_all, 1) - 2]);
if K < 1
    return;
end
Proj = U(:, 1:K);
T = Xz_all * Proj;
W = nan(K + 1, J);
use_joint = false(1, J);
for j = 1:J
    y = Yz_all(:, j);
    valid_j = isfinite(y) & all(isfinite(T), 2);
    if sum(valid_j) < K + 10 || std(y(valid_j), 0, 'omitnan') <= eps
        continue;
    end
    W(:, j) = ridge_solve_local([T(valid_j, :), ones(sum(valid_j), 1)], y(valid_j), ridge);
    use_joint(j) = true;
end
model.ok = any(use_joint);
model.x_mu = x_mu;
model.x_sd = x_sd;
model.y_mu = y_mu;
model.y_sd = y_sd;
model.Proj = Proj;
model.W = W;
model.use_joint = use_joint;
end

function C_pred = apply_prediction_fir_multi_pls_model(F_cell, model)
if isempty(model) || ~isfield(model, 'ok') || ~model.ok
    C_pred = nan_like_prediction_cells(F_cell);
    return;
end
C_pred = cell(size(F_cell));
for k = 1:numel(F_cell)
    Ck = nan(size(F_cell{k}));
    if model.use_basis
        [X, idx] = fir_lag_matrix_multi_basis(F_cell{k}, model.lags, model.J, model.LagB);
    else
        [X, idx] = fir_lag_matrix_multi(F_cell{k}, model.lags, model.J);
    end
    if isempty(X)
        C_pred{k} = Ck;
        continue;
    end
    Xz = (X - model.x_mu) ./ model.x_sd;
    Xz(~isfinite(Xz)) = 0;
    T = Xz * model.Proj;
    Yz = [T, ones(size(T, 1), 1)] * model.W;
    Yh = Yz .* model.y_sd + model.y_mu;
    Jk = min([model.J, size(Ck, 2), size(Yh, 2)]);
    for j = 1:Jk
        if model.use_joint(j)
            Ck(idx, j) = Yh(:, j);
        end
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_lowrank(F_oof, Y_train, F_te, n_components, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
K = max(1, min(round(double(n_components)), J));
[X_all, Y_all] = fir_lag_design_multi(F_oof, Y_train, lags, J);
if isempty(X_all) || isempty(Y_all)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
valid = any(isfinite(Y_all(:, 1:J)), 2);
X_all = X_all(valid, :);
Y_all = Y_all(valid, 1:J);
if size(X_all, 1) < max(20, K + 10)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
y_mu = mean(Y_all, 1, 'omitnan');
y_sd = std(Y_all, 0, 1, 'omitnan');
y_mu(~isfinite(y_mu)) = 0;
y_sd(~isfinite(y_sd) | y_sd <= eps) = 1;
Yz_all = (Y_all - y_mu) ./ y_sd;
Yz_all(~isfinite(Yz_all)) = 0;
[~, ~, V] = svd(Yz_all, 'econ');
if isempty(V)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
B = V(:, 1:min(K, size(V, 2)));
Z_all = Yz_all * B;
models = cell(1, size(B, 2));
Xa = [Xz_all, ones(size(Xz_all, 1), 1)];
for c = 1:size(B, 2)
    z = Z_all(:, c);
    if std(z, 0, 'omitnan') <= eps
        models{c} = [];
        continue;
    end
    models{c} = ridge_solve_local(Xa, z, ridge);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    [X, idx] = fir_lag_matrix_multi(F_te{k}, lags, J);
    if isempty(X)
        C_pred{k} = Ck;
        continue;
    end
    Xz = (X - x_mu) ./ x_sd;
    Xz(~isfinite(Xz)) = 0;
    Xza = [Xz, ones(size(Xz, 1), 1)];
    Zh = zeros(size(Xza, 1), size(B, 2));
    for c = 1:size(B, 2)
        if isempty(models{c}), continue; end
        Zh(:, c) = Xza * models{c};
    end
    Yh = (Zh * B') .* y_sd + y_mu;
    Ck(idx, 1:J) = Yh(:, 1:J);
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_residual_multi_lowrank(F_oof, Y_train, F_te, n_components, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
ridge = max(double(lambda), 0);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = F_te;
    return;
end
J = size(Y_train{first_y}, 2);
K = max(1, min(round(double(n_components)), J));
[X_all, R_all] = fir_lag_design_residual_multi(F_oof, Y_train, lags, J);
if isempty(X_all) || isempty(R_all)
    C_pred = F_te;
    return;
end
valid = any(isfinite(R_all(:, 1:J)), 2);
X_all = X_all(valid, :);
R_all = R_all(valid, 1:J);
if size(X_all, 1) < max(20, K + 10)
    C_pred = F_te;
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
r_mu = mean(R_all, 1, 'omitnan');
r_sd = std(R_all, 0, 1, 'omitnan');
r_mu(~isfinite(r_mu)) = 0;
r_sd(~isfinite(r_sd) | r_sd <= eps) = 1;
Rz_all = (R_all - r_mu) ./ r_sd;
Rz_all(~isfinite(Rz_all)) = 0;
[~, ~, V] = svd(Rz_all, 'econ');
if isempty(V)
    C_pred = F_te;
    return;
end
B = V(:, 1:min(K, size(V, 2)));
Z_all = Rz_all * B;
models = cell(1, size(B, 2));
Xa = [Xz_all, ones(size(Xz_all, 1), 1)];
for c = 1:size(B, 2)
    z = Z_all(:, c);
    if std(z, 0, 'omitnan') <= eps
        models{c} = [];
        continue;
    end
    models{c} = ridge_solve_local(Xa, z, ridge);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Fk = F_te{k};
    Ck = Fk;
    [X, idx] = fir_lag_matrix_multi(Fk, lags, J);
    if isempty(X)
        C_pred{k} = Ck;
        continue;
    end
    Xz = (X - x_mu) ./ x_sd;
    Xz(~isfinite(Xz)) = 0;
    Xza = [Xz, ones(size(Xz, 1), 1)];
    Zh = zeros(size(Xza, 1), size(B, 2));
    for c = 1:size(B, 2)
        if isempty(models{c}), continue; end
        Zh(:, c) = Xza * models{c};
    end
    Rh = (Zh * B') .* r_sd + r_mu;
    Jk = min([J, size(Ck, 2), size(Rh, 2)]);
    Ck(idx, 1:Jk) = Ck(idx, 1:Jk) + Rh(:, 1:Jk);
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_residual_multi_lowrank_on_fir_guard(F_oof, Y_train, F_te, n_components, lambda, lags, guard_margin)
if nargin < 7, guard_margin = 0; end
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    error('post_prediction_fir_residual_multi_lowrank_on_fir_guard requires at least one lag sample');
end
base_oof = post_prediction_fir(F_oof, Y_train, F_oof, lambda, lags);
base_te = post_prediction_fir(F_oof, Y_train, F_te, lambda, lags);
C_oof = post_prediction_fir_residual_multi_lowrank(base_oof, Y_train, base_oof, n_components, lambda, lags);
C_pred = post_prediction_fir_residual_multi_lowrank(base_oof, Y_train, base_te, n_components, lambda, lags);
first_y = find(~cellfun(@isempty, Y_train), 1);
if isempty(first_y)
    C_pred = base_te;
    return;
end
J = size(Y_train{first_y}, 2);
use_lowrank = false(1, J);
for j = 1:J
    r_base = safe_rpt_cells_local(Y_train, base_oof, j);
    r_cal = safe_rpt_cells_local(Y_train, C_oof, j);
    use_lowrank(j) = isfinite(r_cal) && (~isfinite(r_base) || r_cal >= r_base + guard_margin);
end
for k = 1:numel(C_pred)
    if k > numel(base_te) || isempty(C_pred{k}) || isempty(base_te{k})
        continue;
    end
    n = min(size(C_pred{k}, 1), size(base_te{k}, 1));
    Jk = min([size(C_pred{k}, 2), size(base_te{k}, 2), J]);
    for j = 1:Jk
        if ~use_lowrank(j)
            C_pred{k}(1:n, j) = base_te{k}(1:n, j);
        end
    end
end
end

function C_pred = post_prediction_fir_multi_robust(F_oof, Y_train, F_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
[X_all, Y_all] = fir_lag_design_multi(F_oof, Y_train, lags, J);
models = cell(1, J);
if isempty(X_all) || isempty(Y_all)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
for j = 1:J
    y = Y_all(:, j);
    valid = isfinite(y);
    Xz = Xz_all(valid, :);
    y = y(valid);
    if size(Xz, 1) < size(Xz, 2) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    w0 = ridge_solve_local(Xa, yz, ridge);
    resid = yz - Xa * w0;
    sig = 1.4826 * median(abs(resid - median(resid, 'omitnan')), 'omitnan');
    if ~isfinite(sig) || sig <= eps
        sig = std(resid, 0, 'omitnan');
    end
    if ~isfinite(sig) || sig <= eps
        sig = 1;
    end
    c = 2.5 * sig;
    wt = ones(size(resid));
    big = abs(resid) > c;
    wt(big) = c ./ max(abs(resid(big)), eps);
    wt(~isfinite(wt) | wt < 0.05) = 0.05;
    sw = sqrt(wt);
    Xw = Xa .* sw;
    yw = yz .* sw;
    w = ridge_solve_local(Xw, yw, ridge);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    [X, idx] = fir_lag_matrix_multi(F_te{k}, lags, J);
    if isempty(X)
        C_pred{k} = Ck;
        continue;
    end
    for j = 1:J
        if isempty(models{j}), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function C_pred = post_prediction_fir_phase_multi(F_oof, S_oof, Y_train, F_te, S_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
Z_oof = combine_prediction_phase_cells(F_oof, S_oof);
Z_te = combine_prediction_phase_cells(F_te, S_te);
[X_all, Y_all, D] = fir_lag_design_state_multi(Z_oof, Y_train, lags, J);
models = cell(1, J);
if isempty(X_all) || isempty(Y_all)
    C_pred = post_prediction_fir_multi(F_oof, Y_train, F_te, lambda, lags);
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
for j = 1:J
    y = Y_all(:, j);
    valid = isfinite(y);
    Xz = Xz_all(valid, :);
    y = y(valid);
    if size(Xz, 1) < size(Xz, 2) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    w = ridge_solve_local(Xa, yz, ridge);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    [X, idx] = fir_lag_matrix_state_multi(Z_te{k}, lags, D);
    if isempty(X)
        C_pred{k} = Ck;
        continue;
    end
    for j = 1:J
        if isempty(models{j}), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function w = ridge_solve_local(Xa, y, ridge)
A = Xa' * Xa;
reg = ridge * eye(size(A));
reg(end, end) = 0;
w = (A + reg) \ (Xa' * y);
end

function C_pred = post_prediction_fir_apf_multi(A_oof, P_oof, F_oof, Y_train, A_te, P_te, F_te, lambda, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
[X_all, Y_all] = fir_lag_design_apf_multi(A_oof, P_oof, F_oof, Y_train, lags, J);
models = cell(1, J);
if isempty(X_all) || isempty(Y_all)
    C_pred = nan_like_prediction_cells(F_te);
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
for j = 1:J
    y = Y_all(:, j);
    valid = isfinite(y);
    Xz = Xz_all(valid, :);
    y = y(valid);
    if size(Xz, 1) < size(Xz, 2) + 10 || std(y) <= eps
        models{j} = [];
        continue;
    end
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    [X, idx] = fir_lag_matrix_apf_multi(A_te{k}, P_te{k}, F_te{k}, lags, J);
    if isempty(X)
        C_pred{k} = Ck;
        continue;
    end
    for j = 1:J
        if isempty(models{j}), continue; end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function C_pred = nan_like_prediction_cells(F_te)
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    C_pred{k} = nan(size(F_te{k}));
end
end

function C_base = overwrite_finite_prediction_cells(C_base, C_new)
for k = 1:min(numel(C_base), numel(C_new))
    if isempty(C_base{k}) || isempty(C_new{k})
        continue;
    end
    n = min(size(C_base{k}, 1), size(C_new{k}, 1));
    jn = min(size(C_base{k}, 2), size(C_new{k}, 2));
    if n < 1 || jn < 1
        continue;
    end
    src = C_new{k}(1:n, 1:jn);
    mask = isfinite(src);
    dst = C_base{k}(1:n, 1:jn);
    dst(mask) = src(mask);
    C_base{k}(1:n, 1:jn) = dst;
end
end

function C_pred = fill_missing_prediction_cells(C_pred, F_base)
for k = 1:min(numel(C_pred), numel(F_base))
    if isempty(C_pred{k}) || isempty(F_base{k})
        continue;
    end
    n = min(size(C_pred{k}, 1), size(F_base{k}, 1));
    jn = min(size(C_pred{k}, 2), size(F_base{k}, 2));
    if n < 1 || jn < 1
        continue;
    end
    dst = C_pred{k}(1:n, 1:jn);
    src = F_base{k}(1:n, 1:jn);
    mask = ~isfinite(dst) & isfinite(src);
    dst(mask) = src(mask);
    C_pred{k}(1:n, 1:jn) = dst;
end
end

function C_pred = post_prediction_state_multi(F_oof, Y_train, F_te, lambda, lags, fs, guard_mode, guard_margin)
if nargin < 7, guard_mode = false; end
if nargin < 8, guard_margin = 0; end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
wins = state_windows_from_lags(lags, fs);
[X_all, Y_all, F_base] = state_design_multi(F_oof, Y_train, wins, J);
models = cell(1, J);
use_state = true(1, J);
if isempty(X_all) || isempty(Y_all)
    C_pred = F_te;
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
Yhat_tr = nan(size(Y_all));
for j = 1:J
    y = Y_all(:, j);
    valid = isfinite(y) & all(isfinite(Xz_all), 2);
    Xz = Xz_all(valid, :);
    y = y(valid);
    if size(Xz, 1) < size(Xz, 2) + 10 || std(y) <= eps
        models{j} = [];
        use_state(j) = false;
        continue;
    end
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
    yhat_j = ([Xz_all, ones(size(Xz_all, 1), 1)] * w) * y_sd + y_mu;
    Yhat_tr(:, j) = yhat_j;
end
if guard_mode
    for j = 1:J
        r_base = safe_corr_local(Y_all(:, j), F_base(:, j));
        r_state = safe_corr_local(Y_all(:, j), Yhat_tr(:, j));
        use_state(j) = ~isempty(models{j}) && isfinite(r_state) && ...
            (~isfinite(r_base) || r_state >= r_base + guard_margin);
    end
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    [X, idx] = state_matrix_multi(F_te{k}, wins, J);
    if isempty(X)
        C_pred{k} = F_te{k};
        continue;
    end
    for j = 1:J
        if ~use_state(j) || isempty(models{j})
            Ck(idx, j) = F_te{k}(idx, j);
            continue;
        end
        Xz = (X - models{j}.x_mu) ./ models{j}.x_sd;
        Xz(~isfinite(Xz)) = 0;
        yhat = ([Xz, ones(size(Xz, 1), 1)] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        Ck(idx, j) = yhat;
    end
    C_pred{k} = Ck;
end
end

function wins = state_windows_from_lags(lags, fs)
lags = round(double(lags));
age = abs(lags(lags <= 0));
if isempty(age)
    age = round([0.08 0.2 0.4 0.6] * double(fs));
end
wins = unique([1, round(0.08 * double(fs)), round(0.2 * double(fs)), ...
    round(0.4 * double(fs)), round(0.6 * double(fs)), max(age)]);
wins = wins(isfinite(wins) & wins >= 1);
wins = unique(min(wins, max(1, max(age))));
end

function [Xall, Yall, Fall] = state_design_multi(F_cell, Y_cell, wins, J)
Xall = [];
Yall = [];
Fall = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if n < 20, continue; end
    [X, idx] = state_matrix_multi(F_cell{k}(1:n, :), wins, J);
    if isempty(X), continue; end
    Xall = [Xall; X]; %#ok<AGROW>
    Yall = [Yall; Y_cell{k}(idx, 1:J)]; %#ok<AGROW>
    Fall = [Fall; F_cell{k}(idx, 1:J)]; %#ok<AGROW>
end
end

function [X, idx] = state_matrix_multi(F, wins, J)
if isempty(F)
    X = [];
    idx = [];
    return;
end
J = min(J, size(F, 2));
F = fill_finite_matrix_local(double(F(:, 1:J)));
n = size(F, 1);
idx = (1:n)';
parts = {F};
for wi = 1:numel(wins)
    w = max(1, round(wins(wi)));
    ma = movmean(F, [w - 1, 0], 1, 'omitnan');
    lagged = causal_lag_matrix_local(F, w);
    parts{end+1} = ma; %#ok<AGROW>
    parts{end+1} = F - lagged; %#ok<AGROW>
end
X = cat(2, parts{:});
X(~isfinite(X)) = 0;
end

function C_pred = post_prediction_arx_multi(F_oof, Y_train, F_te, lambda, lags, fs, guard_mode, guard_margin)
if nargin < 7, guard_mode = true; end
if nargin < 8, guard_margin = 0; end
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
wins = state_windows_from_lags(lags, fs);
[X_all, Y_all] = arx_design_multi(F_oof, Y_train, wins, J);
models = cell(1, J);
use_arx = true(1, J);
if isempty(X_all) || isempty(Y_all)
    C_pred = F_te;
    return;
end
x_mu = mean(X_all, 1, 'omitnan');
x_sd = std(X_all, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xz_all = (X_all - x_mu) ./ x_sd;
Xz_all(~isfinite(Xz_all)) = 0;
for j = 1:J
    y = Y_all(:, j);
    valid = isfinite(y) & all(isfinite(Xz_all), 2);
    Xz = Xz_all(valid, :);
    y = y(valid);
    if size(Xz, 1) < size(Xz, 2) + 10 || std(y) <= eps
        models{j} = [];
        use_arx(j) = false;
        continue;
    end
    y_mu = mean(y, 'omitnan');
    y_sd = std(y, 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps, y_sd = 1; end
    yz = (y - y_mu) ./ y_sd;
    Xa = [Xz, ones(size(Xz, 1), 1)];
    A = Xa' * Xa;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    w = (A + reg) \ (Xa' * yz);
    models{j} = struct('w', w, 'x_mu', x_mu, 'x_sd', x_sd, 'y_mu', y_mu, 'y_sd', y_sd);
end
if guard_mode
    C_oof = apply_arx_models(F_oof, models, wins, J, use_arx);
    for j = 1:J
        r_base = safe_corr_cells_local(Y_train, F_oof, j);
        r_arx = safe_corr_cells_local(Y_train, C_oof, j);
        use_arx(j) = ~isempty(models{j}) && isfinite(r_arx) && ...
            (~isfinite(r_base) || r_arx >= r_base + guard_margin);
    end
end
C_pred = apply_arx_models(F_te, models, wins, J, use_arx);
end

function C_pred = post_prediction_rhythm_residual(F_oof, Y_train, F_te, lambda, lags, fs, guard_margin)
ridge = max(double(lambda), 0);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
wins = state_windows_from_lags(lags, fs);
[R_all, E_all, Y_all, F_base] = rhythm_residual_design_multi(F_oof, Y_train, wins, J);
if isempty(R_all) || isempty(E_all) || isempty(Y_all)
    C_pred = F_te;
    return;
end
[Rz_all, r_mu, r_sd] = zscore_feature_matrix_local(R_all);
[Ez_all, e_mu, e_sd] = zscore_feature_matrix_local(E_all);
rhythm_models = cell(1, J);
resid_models = cell(1, J);
Y_rhythm_tr = nan(size(Y_all));
Y_hat_tr = nan(size(Y_all));
use_dual = false(1, J);
for j = 1:J
    y = Y_all(:, j);
    valid_r = isfinite(y) & all(isfinite(Rz_all), 2);
    if sum(valid_r) < size(Rz_all, 2) + 10 || std(y(valid_r), 0, 'omitnan') <= eps
        continue;
    end
    [w_r, y_mu, y_sd] = fit_standardized_ridge(Rz_all(valid_r, :), y(valid_r), ridge);
    rhythm_models{j} = struct('w', w_r, 'y_mu', y_mu, 'y_sd', y_sd);
    yr = ([Rz_all, ones(size(Rz_all, 1), 1)] * w_r) * y_sd + y_mu;
    Y_rhythm_tr(:, j) = yr;

    resid = y - yr;
    valid_e = isfinite(resid) & all(isfinite(Ez_all), 2);
    if sum(valid_e) >= size(Ez_all, 2) + 10 && std(resid(valid_e), 0, 'omitnan') > eps
        [w_e, e_y_mu, e_y_sd] = fit_standardized_ridge(Ez_all(valid_e, :), resid(valid_e), ridge);
        resid_models{j} = struct('w', w_e, 'y_mu', e_y_mu, 'y_sd', e_y_sd);
        er = ([Ez_all, ones(size(Ez_all, 1), 1)] * w_e) * e_y_sd + e_y_mu;
        Y_hat_tr(:, j) = yr + er;
    else
        Y_hat_tr(:, j) = yr;
    end
    r_base = safe_corr_local(Y_all(:, j), F_base(:, j));
    r_dual = safe_corr_local(Y_all(:, j), Y_hat_tr(:, j));
    use_dual(j) = isfinite(r_dual) && (~isfinite(r_base) || r_dual >= r_base + guard_margin);
end

C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Fk = F_te{k};
    Ck = Fk;
    [R, E, idx] = rhythm_residual_feature_matrix(Fk, wins, J);
    if isempty(R)
        C_pred{k} = Ck;
        continue;
    end
    Rz = (R - r_mu) ./ r_sd;
    Ez = (E - e_mu) ./ e_sd;
    Rz(~isfinite(Rz)) = 0;
    Ez(~isfinite(Ez)) = 0;
    for j = 1:J
        if ~use_dual(j) || isempty(rhythm_models{j})
            continue;
        end
        yr = ([Rz, ones(size(Rz, 1), 1)] * rhythm_models{j}.w) * ...
            rhythm_models{j}.y_sd + rhythm_models{j}.y_mu;
        if ~isempty(resid_models{j})
            er = ([Ez, ones(size(Ez, 1), 1)] * resid_models{j}.w) * ...
                resid_models{j}.y_sd + resid_models{j}.y_mu;
            yr = yr + er;
        end
        Ck(idx, j) = yr;
    end
    C_pred{k} = Ck;
end
end

function [Rall, Eall, Yall, Fall] = rhythm_residual_design_multi(F_cell, Y_cell, wins, J)
Rall = [];
Eall = [];
Yall = [];
Fall = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if n < 20, continue; end
    [R, E, idx] = rhythm_residual_feature_matrix(F_cell{k}(1:n, :), wins, J);
    if isempty(R), continue; end
    Rall = [Rall; R]; %#ok<AGROW>
    Eall = [Eall; E]; %#ok<AGROW>
    Yall = [Yall; Y_cell{k}(idx, 1:J)]; %#ok<AGROW>
    Fall = [Fall; F_cell{k}(idx, 1:J)]; %#ok<AGROW>
end
end

function [R, E, idx] = rhythm_residual_feature_matrix(F, wins, J)
if isempty(F)
    R = [];
    E = [];
    idx = [];
    return;
end
J = min(J, size(F, 2));
F = fill_finite_matrix_local(double(F(:, 1:J)));
n = size(F, 1);
idx = (1:n)';
Rparts = {F};
Eparts = {};
for wi = 1:numel(wins)
    w = max(1, round(wins(wi)));
    ma = movmean(F, [w - 1, 0], 1, 'omitnan');
    lagged = causal_lag_matrix_local(F, w);
    Rparts{end+1} = ma; %#ok<AGROW>
    Eparts{end+1} = F - ma; %#ok<AGROW>
    Eparts{end+1} = F - lagged; %#ok<AGROW>
end
R = cat(2, Rparts{:});
E = cat(2, Eparts{:});
R(~isfinite(R)) = 0;
E(~isfinite(E)) = 0;
end

function [Xz, mu, sd] = zscore_feature_matrix_local(X)
mu = mean(X, 1, 'omitnan');
sd = std(X, 0, 1, 'omitnan');
mu(~isfinite(mu)) = 0;
sd(~isfinite(sd) | sd <= eps) = 1;
Xz = (X - mu) ./ sd;
Xz(~isfinite(Xz)) = 0;
end

function [w, y_mu, y_sd] = fit_standardized_ridge(Xz, y, ridge)
y_mu = mean(y, 'omitnan');
y_sd = std(y, 0, 'omitnan');
if ~isfinite(y_sd) || y_sd <= eps
    y_sd = 1;
end
yz = (y - y_mu) ./ y_sd;
Xa = [Xz, ones(size(Xz, 1), 1)];
w = ridge_solve_local(Xa, yz, ridge);
end

function [Xall, Yall] = arx_design_multi(F_cell, Y_cell, wins, J)
Xall = [];
Yall = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if n < 20, continue; end
    [S, idx] = state_matrix_multi(F_cell{k}(1:n, :), wins, J);
    if isempty(S), continue; end
    Yk = fill_finite_matrix_local(double(Y_cell{k}(1:n, 1:J)));
    keep = idx > 1;
    if ~any(keep), continue; end
    Xall = [Xall; S(keep, :), Yk(idx(keep) - 1, :)]; %#ok<AGROW>
    Yall = [Yall; Y_cell{k}(idx(keep), 1:J)]; %#ok<AGROW>
end
end

function C_pred = apply_arx_models(F_cell, models, wins, J, use_arx)
C_pred = cell(size(F_cell));
for k = 1:numel(F_cell)
    Fk = F_cell{k};
    Ck = nan(size(Fk));
    [S, idx] = state_matrix_multi(Fk, wins, J);
    if isempty(S)
        C_pred{k} = Fk;
        continue;
    end
    prev = fill_finite_matrix_local(double(Fk(:, 1:min(J, size(Fk, 2)))));
    if isempty(prev)
        C_pred{k} = Fk;
        continue;
    end
    y_prev = prev(1, :);
    for ii = 1:numel(idx)
        row = idx(ii);
        xrow = [S(ii, :), y_prev];
        yhat = nan(1, J);
        for j = 1:J
            if j > numel(models) || j > numel(use_arx) || ~use_arx(j) || isempty(models{j})
                yhat(j) = Fk(row, j);
                continue;
            end
            Xz = (xrow - models{j}.x_mu) ./ models{j}.x_sd;
            Xz(~isfinite(Xz)) = 0;
            yhat(j) = ([Xz, 1] * models{j}.w) * models{j}.y_sd + models{j}.y_mu;
        end
        yhat(~isfinite(yhat)) = prev(row, ~isfinite(yhat));
        Ck(row, 1:J) = yhat;
        y_prev = yhat;
    end
    C_pred{k} = Ck;
end
end

function r = safe_corr_cells_local(Y_true, Y_pred, j)
y = [];
yp = [];
for k = 1:numel(Y_true)
    if isempty(Y_true{k}) || isempty(Y_pred{k}), continue; end
    if size(Y_true{k}, 2) < j || size(Y_pred{k}, 2) < j, continue; end
    n = min(size(Y_true{k}, 1), size(Y_pred{k}, 1));
    if n <= 0, continue; end
    y = [y; Y_true{k}(1:n, j)]; %#ok<AGROW>
    yp = [yp; Y_pred{k}(1:n, j)]; %#ok<AGROW>
end
r = safe_corr_local(y, yp);
end

function r = safe_rpt_cells_local(Y_true, Y_pred, j)
r_trials = nan(1, numel(Y_true));
for k = 1:numel(Y_true)
    if k > numel(Y_pred) || isempty(Y_true{k}) || isempty(Y_pred{k}), continue; end
    if size(Y_true{k}, 2) < j || size(Y_pred{k}, 2) < j, continue; end
    n = min(size(Y_true{k}, 1), size(Y_pred{k}, 1));
    if n < 2, continue; end
    r_trials(k) = safe_corr_local(Y_true{k}(1:n, j), Y_pred{k}(1:n, j));
end
r = mean(r_trials, 'omitnan');
end

function Y = causal_lag_matrix_local(X, lag)
lag = max(1, round(lag));
Y = X;
if size(X, 1) <= 1
    return;
end
if lag >= size(X, 1)
    Y(:,:) = repmat(X(1, :), size(X, 1), 1);
else
    Y(1:lag, :) = repmat(X(1, :), lag, 1);
    Y(lag+1:end, :) = X(1:end-lag, :);
end
end

function X = fill_finite_matrix_local(X)
for j = 1:size(X, 2)
    X(:, j) = fill_finite_local(X(:, j));
end
end

function C_pred = post_prediction_lag_select(F_oof, Y_train, F_te, lags)
lags = round(double(lags));
lags = unique(lags(:)');
lags = lags(isfinite(lags));
if isempty(lags)
    lags = -400:40:400;
end
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
best_lags = zeros(1, J);
for j = 1:J
    best_r = -Inf;
    for lag = lags
        [pred, target] = lag_select_design(F_oof, Y_train, j, lag);
        r = safe_corr_local(target, pred);
        if isfinite(r) && r > best_r
            best_r = r;
            best_lags(j) = lag;
        end
    end
end
C_pred = cell(size(F_te));
for k = 1:numel(F_te)
    Ck = nan(size(F_te{k}));
    for j = 1:J
        [pred, idx] = apply_prediction_lag(F_te{k}(:, j), best_lags(j));
        if isempty(idx), continue; end
        Ck(idx, j) = pred;
    end
    C_pred{k} = Ck;
end
end

function [pred_all, target_all] = lag_select_design(F_cell, Y_cell, j, lag)
pred_all = [];
target_all = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    [pred, idx] = apply_prediction_lag(F_cell{k}(1:n, j), lag);
    if isempty(idx), continue; end
    pred_all = [pred_all; pred]; %#ok<AGROW>
    target_all = [target_all; Y_cell{k}(idx, j)]; %#ok<AGROW>
end
valid = isfinite(pred_all) & isfinite(target_all);
pred_all = pred_all(valid);
target_all = target_all(valid);
end

function [pred, idx] = apply_prediction_lag(x, lag)
n = numel(x);
lo = max(1, 1 - lag);
hi = min(n, n - lag);
if hi < lo
    pred = [];
    idx = [];
    return;
end
idx = (lo:hi)';
pred = x(idx + lag);
end

function [Xall, yall] = fir_lag_design(F_cell, Y_cell, j, lags)
Xall = [];
yall = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if ~fir_lag_has_rows(n, lags), continue; end
    [X, idx] = fir_lag_matrix(F_cell{k}(1:n, j), lags);
    if isempty(X), continue; end
    y = Y_cell{k}(idx, j);
    Xall = [Xall; X]; %#ok<AGROW>
    yall = [yall; y]; %#ok<AGROW>
end
end

function [Xall, yall, wall] = fir_lag_design_lag_aug(F_cell, Y_cell, j, lags, target_shifts, shift_weights)
Xall = [];
yall = [];
wall = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    if size(F_cell{k}, 2) < j || size(Y_cell{k}, 2) < j, continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if ~fir_lag_has_rows(n, lags), continue; end
    [X, idx] = fir_lag_matrix(F_cell{k}(1:n, j), lags);
    if isempty(X), continue; end
    for si = 1:numel(target_shifts)
        src = idx + target_shifts(si);
        in_range = src >= 1 & src <= n;
        if ~any(in_range), continue; end
        y = nan(size(idx));
        y(in_range) = Y_cell{k}(src(in_range), j);
        Xall = [Xall; X]; %#ok<AGROW>
        yall = [yall; y]; %#ok<AGROW>
        wall = [wall; repmat(shift_weights(si), numel(y), 1)]; %#ok<AGROW>
    end
end
end

function [Xall, Mall, yall] = fir_lag_design_masked(F_cell, Y_cell, j, lags)
Xall = [];
Mall = [];
yall = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    if size(F_cell{k}, 2) < j || size(Y_cell{k}, 2) < j, continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if n < 5, continue; end
    [X, M, idx] = fir_lag_matrix_masked(F_cell{k}(1:n, j), lags);
    if isempty(X), continue; end
    y = Y_cell{k}(idx, j);
    Xall = [Xall; X]; %#ok<AGROW>
    Mall = [Mall; M]; %#ok<AGROW>
    yall = [yall; y]; %#ok<AGROW>
end
end

function [Xall, yall] = fir_lag_design_residual(F_cell, Y_cell, j, lags)
Xall = [];
yall = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if ~fir_lag_has_rows(n, lags), continue; end
    [X, idx] = fir_lag_matrix(F_cell{k}(1:n, j), lags);
    if isempty(X), continue; end
    y = Y_cell{k}(idx, j) - F_cell{k}(idx, j);
    Xall = [Xall; X]; %#ok<AGROW>
    yall = [yall; y]; %#ok<AGROW>
end
end

function [Xall, Rall] = fir_lag_design_residual_multi(F_cell, Y_cell, lags, J)
Xall = [];
Rall = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if ~fir_lag_has_rows(n, lags), continue; end
    [X, idx] = fir_lag_matrix_multi(F_cell{k}(1:n, :), lags, J);
    if isempty(X), continue; end
    Jk = min([J, size(Y_cell{k}, 2), size(F_cell{k}, 2)]);
    R = nan(numel(idx), J);
    R(:, 1:Jk) = Y_cell{k}(idx, 1:Jk) - F_cell{k}(idx, 1:Jk);
    Xall = [Xall; X]; %#ok<AGROW>
    Rall = [Rall; R]; %#ok<AGROW>
end
end

function [Xall, yall] = fir_lag_design_apf(A_cell, P_cell, F_cell, Y_cell, j, lags)
Xall = [];
yall = [];
for k = 1:numel(F_cell)
    if isempty(A_cell{k}) || isempty(P_cell{k}) || isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min([size(A_cell{k}, 1), size(P_cell{k}, 1), size(F_cell{k}, 1), size(Y_cell{k}, 1)]);
    if ~fir_lag_has_rows(n, lags), continue; end
    [X, idx] = fir_lag_matrix_apf(A_cell{k}(1:n, j), P_cell{k}(1:n, j), F_cell{k}(1:n, j), lags);
    if isempty(X), continue; end
    y = Y_cell{k}(idx, j);
    Xall = [Xall; X]; %#ok<AGROW>
    yall = [yall; y]; %#ok<AGROW>
end
end

function tf = fir_lag_has_rows(n, lags)
if isempty(lags) || ~isfinite(n) || n < 1
    tf = false;
    return;
end
lo = max(1, 1 - min(lags));
hi = min(n, n - max(lags));
tf = hi >= lo;
end

function [X, idx] = fir_lag_matrix_apf(a, p, f, lags)
[Xf, idx] = fir_lag_matrix(f, lags);
if isempty(Xf)
    X = [];
    return;
end
Xa = nan(size(Xf));
Xp = nan(size(Xf));
for li = 1:numel(lags)
    Xa(:, li) = a(idx + lags(li));
    Xp(:, li) = p(idx + lags(li));
end
X = [Xa, Xp, Xf];
end

function [X, idx] = fir_lag_matrix(x, lags)
n = numel(x);
lo = max(1, 1 - min(lags));
hi = min(n, n - max(lags));
if hi < lo
    X = [];
    idx = [];
    return;
end
idx = (lo:hi)';
X = nan(numel(idx), numel(lags));
for li = 1:numel(lags)
    X(:, li) = x(idx + lags(li));
end
end

function [X, idx] = fir_lag_matrix_mean_padded(x, lags, x_mu)
n = numel(x);
if n < 1
    X = [];
    idx = [];
    return;
end
idx = (1:n)';
L = numel(lags);
x_mu = double(x_mu(:)');
if numel(x_mu) < L
    x_mu(end+1:L) = 0;
end
X = repmat(x_mu(1:L), n, 1);
for li = 1:L
    src = idx + lags(li);
    in_range = src >= 1 & src <= n;
    vals = nan(n, 1);
    vals(in_range) = x(src(in_range));
    finite = isfinite(vals);
    X(finite, li) = vals(finite);
end
end

function [X, M, idx] = fir_lag_matrix_masked(x, lags)
n = numel(x);
if n < 1
    X = [];
    M = [];
    idx = [];
    return;
end
idx = (1:n)';
L = numel(lags);
X = nan(n, L);
M = zeros(n, L);
for li = 1:L
    src = idx + lags(li);
    in_range = src >= 1 & src <= n;
    vals = nan(n, 1);
    vals(in_range) = x(src(in_range));
    finite = in_range & isfinite(vals);
    X(finite, li) = vals(finite);
    M(finite, li) = 1;
end
end

function [Ximp, M] = impute_masked_fir_values(X, M, x_mu)
Ximp = X;
M = double(M > 0 & isfinite(Ximp));
for c = 1:size(Ximp, 2)
    missing = ~isfinite(Ximp(:, c)) | M(:, c) <= 0;
    Ximp(missing, c) = x_mu(c);
    M(missing, c) = 0;
end
end

function [Xall, yall] = fir_prefix_basis_design(F_cell, Y_cell, j, lags, LagB)
Xall = [];
yall = [];
for k = 1:numel(Y_cell)
    if isempty(Y_cell{k}) || isempty(F_cell{k})
        continue;
    end
    if size(Y_cell{k}, 2) < j || size(F_cell{k}, 2) < j
        continue;
    end
    n = min(size(Y_cell{k}, 1), size(F_cell{k}, 1));
    if n < 5
        continue;
    end
    [X, idx] = fir_prefix_basis_matrix(F_cell{k}(1:n, j), lags, LagB);
    if isempty(X)
        continue;
    end
    Xall = [Xall; X]; %#ok<AGROW>
    yall = [yall; Y_cell{k}(idx, j)]; %#ok<AGROW>
end
end

function [X, idx] = fir_prefix_basis_matrix(x, lags, LagB)
n = numel(x);
if n < 1
    X = [];
    idx = [];
    return;
end
lags = round(double(lags(:)'));
L = numel(lags);
if size(LagB, 1) ~= L
    error('lag basis row count must match lag count');
end
K = size(LagB, 2);
idx = (1:n)';
X = nan(n, K);
for ii = 1:n
    src = ii + lags;
    avail = src >= 1 & src <= n;
    if ~any(avail)
        continue;
    end
    vals = x(src(avail));
    B = LagB(avail, :);
    denom = sum(B, 1);
    denom(~isfinite(denom) | abs(denom) <= eps) = NaN;
    X(ii, :) = (vals(:)' * B) ./ denom;
end
end

function [Xall, Hall, yall] = apf_router_design(A_cell, P_cell, F_cell, Y_cell, j)
Xall = [];
Hall = [];
yall = [];
for k = 1:numel(Y_cell)
    if isempty(Y_cell{k}) || isempty(A_cell{k}) || isempty(P_cell{k}) || isempty(F_cell{k})
        continue;
    end
    n = min([size(Y_cell{k}, 1), size(A_cell{k}, 1), size(P_cell{k}, 1), size(F_cell{k}, 1)]);
    if n < 20, continue; end
    H = [A_cell{k}(1:n, j), P_cell{k}(1:n, j), F_cell{k}(1:n, j)];
    X = apf_router_features(H);
    Xall = [Xall; X]; %#ok<AGROW>
    Hall = [Hall; H]; %#ok<AGROW>
    yall = [yall; Y_cell{k}(1:n, j)]; %#ok<AGROW>
end
end

function X = apf_router_features(H)
a = H(:, 1);
p = H(:, 2);
f = H(:, 3);
X = [a, p, f, abs(a - p), abs(a - f), abs(p - f), ...
     0.5 * (a + p), 0.5 * (a + f), 0.5 * (p + f), ...
     mean(H, 2, 'omitnan'), std(H, 0, 2, 'omitnan')];
X(~isfinite(X)) = NaN;
end

function B = fir_lag_smooth_basis(lags)
B = fir_lag_smooth_basis_n(lags, 5);
end

function B = fir_lag_smooth_basis_n(lags, n_basis)
lags = unique(round(double(lags(:)')));
L = numel(lags);
if L <= 1
    B = ones(L, 1);
    return;
end
age = -lags(:);
age = age - min(age);
span = max(age);
if ~isfinite(span) || span <= eps
    B = ones(L, 1) / L;
    return;
end
K = min(max(1, round(double(n_basis))), L);
centers = linspace(0, span, K);
if K == 1
    width = span;
else
    width = max(1, (centers(2) - centers(1)) * 1.25);
end
B = zeros(L, K);
for k = 1:K
    B(:, k) = exp(-0.5 * ((age - centers(k)) ./ width) .^ 2);
    s = sum(abs(B(:, k)));
    if isfinite(s) && s > eps
        B(:, k) = B(:, k) ./ s;
    end
end
end

function [Xall, Yall] = fir_lag_design_multi(F_cell, Y_cell, lags, J)
Xall = [];
Yall = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if ~fir_lag_has_rows(n, lags), continue; end
    [X, idx] = fir_lag_matrix_multi(F_cell{k}(1:n, :), lags, J);
    if isempty(X), continue; end
    Xall = [Xall; X]; %#ok<AGROW>
    Yall = [Yall; Y_cell{k}(idx, 1:J)]; %#ok<AGROW>
end
end

function [X, idx] = fir_lag_matrix_multi(F, lags, J)
n = size(F, 1);
lo = max(1, 1 - min(lags));
hi = min(n, n - max(lags));
if hi < lo
    X = [];
    idx = [];
    return;
end
idx = (lo:hi)';
X = nan(numel(idx), numel(lags) * J);
col = 1;
for j = 1:J
    x = F(:, j);
    for li = 1:numel(lags)
        X(:, col) = x(idx + lags(li));
        col = col + 1;
    end
end
end

function [Xall, Yall] = fir_lag_design_multi_basis(F_cell, Y_cell, lags, J, LagB)
Xall = [];
Yall = [];
for k = 1:numel(F_cell)
    if isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min(size(F_cell{k}, 1), size(Y_cell{k}, 1));
    if ~fir_lag_has_rows(n, lags), continue; end
    [X, idx] = fir_lag_matrix_multi_basis(F_cell{k}(1:n, :), lags, J, LagB);
    if isempty(X), continue; end
    Xall = [Xall; X]; %#ok<AGROW>
    Yall = [Yall; Y_cell{k}(idx, 1:J)]; %#ok<AGROW>
end
end

function [X, idx] = fir_lag_matrix_multi_basis(F, lags, J, LagB)
n = size(F, 1);
J = min(J, size(F, 2));
L = numel(lags);
if size(LagB, 1) ~= L
    error('lag basis row count must match lag count');
end
K = size(LagB, 2);
lo = max(1, 1 - min(lags));
hi = min(n, n - max(lags));
if hi < lo || J < 1 || K < 1
    X = [];
    idx = [];
    return;
end
idx = (lo:hi)';
X = nan(numel(idx), J * K);
for j = 1:J
    lagged = nan(numel(idx), L);
    x = F(:, j);
    for li = 1:L
        lagged(:, li) = x(idx + lags(li));
    end
    cols = (j - 1) * K + (1:K);
    X(:, cols) = lagged * LagB;
end
end

function Z = combine_prediction_phase_cells(F_cell, S_cell)
Z = cell(size(F_cell));
for k = 1:numel(F_cell)
    F = F_cell{k};
    if isempty(F)
        Z{k} = [];
        continue;
    end
    if k <= numel(S_cell) && ~isempty(S_cell{k})
        S = S_cell{k};
        n = min(size(F, 1), size(S, 1));
        Z{k} = [F(1:n, :), S(1:n, :)];
    else
        Z{k} = F;
    end
end
end

function [Xall, Yall, D] = fir_lag_design_state_multi(Z_cell, Y_cell, lags, J)
Xall = [];
Yall = [];
D = 0;
for k = 1:numel(Z_cell)
    if isempty(Z_cell{k}) || isempty(Y_cell{k}), continue; end
    if D == 0
        D = size(Z_cell{k}, 2);
    end
    n = min(size(Z_cell{k}, 1), size(Y_cell{k}, 1));
    if ~fir_lag_has_rows(n, lags), continue; end
    [X, idx] = fir_lag_matrix_state_multi(Z_cell{k}(1:n, :), lags, D);
    if isempty(X), continue; end
    Xall = [Xall; X]; %#ok<AGROW>
    Yall = [Yall; Y_cell{k}(idx, 1:J)]; %#ok<AGROW>
end
end

function [X, idx] = fir_lag_matrix_state_multi(Z, lags, D)
n = size(Z, 1);
D = min(D, size(Z, 2));
lo = max(1, 1 - min(lags));
hi = min(n, n - max(lags));
if hi < lo || D < 1
    X = [];
    idx = [];
    return;
end
idx = (lo:hi)';
X = nan(numel(idx), numel(lags) * D);
col = 1;
for d = 1:D
    z = Z(:, d);
    for li = 1:numel(lags)
        X(:, col) = z(idx + lags(li));
        col = col + 1;
    end
end
end

function [Xall, Yall] = fir_lag_design_apf_multi(A_cell, P_cell, F_cell, Y_cell, lags, J)
Xall = [];
Yall = [];
for k = 1:numel(F_cell)
    if isempty(A_cell{k}) || isempty(P_cell{k}) || isempty(F_cell{k}) || isempty(Y_cell{k}), continue; end
    n = min([size(A_cell{k}, 1), size(P_cell{k}, 1), size(F_cell{k}, 1), size(Y_cell{k}, 1)]);
    if ~fir_lag_has_rows(n, lags), continue; end
    [X, idx] = fir_lag_matrix_apf_multi(A_cell{k}(1:n, :), P_cell{k}(1:n, :), F_cell{k}(1:n, :), lags, J);
    if isempty(X), continue; end
    Xall = [Xall; X]; %#ok<AGROW>
    Yall = [Yall; Y_cell{k}(idx, 1:J)]; %#ok<AGROW>
end
end

function [X, idx] = fir_lag_matrix_apf_multi(A, P, F, lags, J)
n = min([size(A, 1), size(P, 1), size(F, 1)]);
J = min([J, size(A, 2), size(P, 2), size(F, 2)]);
lo = max(1, 1 - min(lags));
hi = min(n, n - max(lags));
if hi < lo || J < 1
    X = [];
    idx = [];
    return;
end
idx = (lo:hi)';
X = nan(numel(idx), numel(lags) * J * 3);
col = 1;
for block = 1:3
    if block == 1
        M = A;
    elseif block == 2
        M = P;
    else
        M = F;
    end
    for j = 1:J
        x = M(:, j);
        for li = 1:numel(lags)
            X(:, col) = x(idx + lags(li));
            col = col + 1;
        end
    end
end
end

function C_pred = post_calibrate_guardcv(A_oof, P_oof, F_oof, Y_train, A_te, P_te, F_te, n_components, lambda, guard_margin)
n = numel(Y_train);
if n < 6
    C_pred = F_te;
    return;
end
rng(20260516, 'twister');
perm = randperm(n);
is_val = false(1, n);
is_val(perm(1:3:end)) = true;
if sum(is_val) < 2 || sum(~is_val) < 4
    C_pred = F_te;
    return;
end
model_val = fit_post_calib_model(A_oof(~is_val), P_oof(~is_val), F_oof(~is_val), ...
    Y_train(~is_val), n_components, lambda);
if isempty(model_val)
    C_pred = F_te;
    return;
end
C_val = apply_post_calib_model(model_val, A_oof(is_val), P_oof(is_val), F_oof(is_val));
Y_val = Y_train(is_val);
F_val = F_oof(is_val);
J = size(Y_train{find(~cellfun(@isempty, Y_train), 1)}, 2);
use_calib = false(1, J);
for j = 1:J
    [r_base, ~] = eval_one(Y_val, F_val, j);
    [r_cal, ~] = eval_one(Y_val, C_val, j);
    use_calib(j) = isfinite(r_cal) && isfinite(r_base) && r_cal >= r_base + guard_margin;
end
model_all = fit_post_calib_model(A_oof, P_oof, F_oof, Y_train, n_components, lambda);
if isempty(model_all)
    C_pred = F_te;
    return;
end
C_all = apply_post_calib_model(model_all, A_te, P_te, F_te);
C_pred = guarded_prediction_cells(C_all, F_te, use_calib);
end

function model = fit_post_calib_model(A_cell, P_cell, F_cell, Y_cell, n_components, lambda)
[Xtr, lens_tr] = post_feature_matrix(A_cell, P_cell, F_cell);
Ytr = post_target_matrix(Y_cell, lens_tr);
if isempty(Xtr) || isempty(Ytr)
    model = [];
    return;
end
n_train = min(size(Xtr, 1), size(Ytr, 1));
Xtr = Xtr(1:n_train, :);
Ytr = Ytr(1:n_train, :);
x_mu = mean(Xtr, 1, 'omitnan');
x_sd = std(Xtr, 0, 1, 'omitnan');
x_mu(~isfinite(x_mu)) = 0;
x_sd(~isfinite(x_sd) | x_sd <= eps) = 1;
Xtr_z = (Xtr - x_mu) ./ x_sd;
Xtr_z(~isfinite(Xtr_z)) = 0;
[~, ~, V] = svd(Xtr_z, 'econ');
K = min([round(n_components), size(V, 2), size(Xtr_z, 1) - 1]);
if K < 1
    model = [];
    return;
end
Ztr = Xtr_z * V(:, 1:K);
ridge = max(double(lambda), 0);
J = size(Ytr, 2);
W = nan(K + 1, J);
y_mu_all = nan(1, J);
y_sd_all = ones(1, J);
for j = 1:J
    y = Ytr(:, j);
    valid = isfinite(y);
    if sum(valid) < K + 5 || std(y(valid)) <= eps
        continue;
    end
    y_mu = mean(y(valid), 'omitnan');
    y_sd = std(y(valid), 0, 'omitnan');
    if ~isfinite(y_sd) || y_sd <= eps
        y_sd = 1;
    end
    y_z = (y(valid) - y_mu) ./ y_sd;
    Xj = [Ztr(valid, :), ones(sum(valid), 1)];
    A = Xj' * Xj;
    reg = ridge * eye(size(A));
    reg(end, end) = 0;
    W(:, j) = (A + reg) \ (Xj' * y_z);
    y_mu_all(j) = y_mu;
    y_sd_all(j) = y_sd;
end
model = struct('x_mu', x_mu, 'x_sd', x_sd, 'V', V(:, 1:K), ...
    'W', W, 'y_mu', y_mu_all, 'y_sd', y_sd_all);
end

function C_pred = apply_post_calib_model(model, A_cell, P_cell, F_cell)
[Xte, lens_te] = post_feature_matrix(A_cell, P_cell, F_cell);
if isempty(Xte)
    C_pred = cell(size(A_cell));
    return;
end
Xte_z = (Xte - model.x_mu) ./ model.x_sd;
Xte_z(~isfinite(Xte_z)) = 0;
Zte = Xte_z * model.V;
Yhat = nan(size(Xte, 1), size(model.W, 2));
Xaug = [Zte, ones(size(Zte, 1), 1)];
for j = 1:size(model.W, 2)
    if all(~isfinite(model.W(:, j)))
        continue;
    end
    Yhat(:, j) = (Xaug * model.W(:, j)) * model.y_sd(j) + model.y_mu(j);
end
C_pred = split_prediction_matrix(Yhat, lens_te);
end

function Y_out = guarded_prediction_cells(C_cell, F_cell, use_calib)
Y_out = cell(size(F_cell));
for k = 1:numel(F_cell)
    n = min(size(F_cell{k}, 1), size(C_cell{k}, 1));
    J = size(F_cell{k}, 2);
    Y = F_cell{k}(1:n, :);
    for j = 1:J
        if j <= numel(use_calib) && use_calib(j)
            Y(:, j) = C_cell{k}(1:n, j);
        end
    end
    Y_out{k} = Y;
end
end

function [X, lens] = post_feature_matrix(A_pred, P_pred, F_pred)
n_cells = numel(A_pred);
lens = zeros(1, n_cells);
X = [];
for k = 1:n_cells
    if isempty(A_pred{k}) || isempty(P_pred{k}) || isempty(F_pred{k})
        continue;
    end
    n = min([size(A_pred{k}, 1), size(P_pred{k}, 1), size(F_pred{k}, 1)]);
    if n <= 0
        continue;
    end
    Ak = A_pred{k}(1:n, :);
    Pk = P_pred{k}(1:n, :);
    Fk = F_pred{k}(1:n, :);
    X = [X; Ak, Pk, Fk]; %#ok<AGROW>
    lens(k) = n;
end
end

function Y = post_target_matrix(Y_cell, lens)
Y = [];
for k = 1:numel(lens)
    n = lens(k);
    if n <= 0 || isempty(Y_cell{k})
        continue;
    end
    n = min(n, size(Y_cell{k}, 1));
    Y = [Y; Y_cell{k}(1:n, :)]; %#ok<AGROW>
end
end

function Y_cell = split_prediction_matrix(Y, lens)
Y_cell = cell(size(lens));
pos = 1;
J = size(Y, 2);
for k = 1:numel(lens)
    n = lens(k);
    if n <= 0
        Y_cell{k} = zeros(0, J);
        continue;
    end
    Y_cell{k} = Y(pos:pos + n - 1, :);
    pos = pos + n;
end
end

function templates = fit_delta_templates(Y_phase, Y_delta, n_bins)
J = size(Y_delta{1}, 2);
bin_edges = linspace(-pi, pi, n_bins + 1);
templates = nan(n_bins, J);
for j = 1:J
    c = 2 * j - 1;
    s = 2 * j;
    phi = [];
    theta = [];
    for k = 1:numel(Y_phase)
        n = min(size(Y_phase{k}, 1), size(Y_delta{k}, 1));
        if n == 0 || all(isnan(Y_phase{k}(:, c))), continue; end
        phi = [phi; atan2(Y_phase{k}(1:n, s), Y_phase{k}(1:n, c))]; %#ok<AGROW>
        theta = [theta; Y_delta{k}(1:n, j)]; %#ok<AGROW>
    end
    [~, ~, bin_idx] = histcounts(phi, bin_edges);
    M = nan(n_bins, 1);
    for b = 1:n_bins
        rows = bin_idx == b;
        if any(rows)
            M(b) = mean(theta(rows), 'omitnan');
        end
    end
    templates(:, j) = fill_template_local(M);
end
end

function P_pred = phase_template_predict(Yp_phase, templates)
n_bins = size(templates, 1);
J = size(templates, 2);
bin_edges = linspace(-pi, pi, n_bins + 1);
P_pred = cell(size(Yp_phase));
for k = 1:numel(Yp_phase)
    Yp = Yp_phase{k};
    Pk = nan(size(Yp, 1), J);
    for j = 1:J
        c = 2 * j - 1;
        s = 2 * j;
        phi = atan2(Yp(:, s), Yp(:, c));
        [~, ~, b_idx] = histcounts(phi, bin_edges);
        b_idx(b_idx == 0) = 1;
        Pk(:, j) = templates(b_idx, j);
    end
    P_pred{k} = Pk;
end
end

function [Y_state, layout] = build_state_cells(Y_phase, Y_delta, fs, phase_lo, phase_hi, mode, harmonic_k)
J = size(Y_delta{1}, 2);
switch lower(char(mode))
    case 'amp_phase'
        D = 3; % [A, cos(phi), sin(phi)] per joint.
    case 'amp_harmonic'
        D = 2 * harmonic_k; % [A cos(k phi), A sin(k phi)] per joint.
    case 'tempo_harmonic'
        D = 4 * harmonic_k; % [cos/sin(k phi), omega*cos/sin(k phi)] per joint.
    otherwise
        error('unsupported phase state mode: %s', mode);
end
layout = struct();
layout.D = D;
layout.J = J;
layout.cols = cell(1, J);
for j = 1:J
    layout.cols{j} = (j - 1) * D + (1:D);
end

Y_state = cell(size(Y_delta));
for k = 1:numel(Y_delta)
    T = size(Y_delta{k}, 1);
    Yk = nan(T, J * D);
    for j = 1:J
        c = 2 * j - 1;
        s = 2 * j;
        n = min(size(Y_phase{k}, 1), T);
        if n == 0 || all(isnan(Y_phase{k}(:, c))), continue; end
        y = Y_delta{k}(1:n, j);
        phi = atan2(Y_phase{k}(1:n, s), Y_phase{k}(1:n, c));
        amp = local_amp_from_angle(y, fs, phase_lo, phase_hi);
        cols = layout.cols{j};
        switch lower(char(mode))
            case 'amp_phase'
                Yk(1:n, cols) = [amp, cos(phi), sin(phi)];
            case 'amp_harmonic'
                Z = nan(n, D);
                col = 1;
                for h = 1:harmonic_k
                    Z(:, col) = amp .* cos(h * phi);
                    Z(:, col + 1) = amp .* sin(h * phi);
                    col = col + 2;
                end
                Yk(1:n, cols) = Z;
            case 'tempo_harmonic'
                phi_u = unwrap(phi);
                omega = local_phase_velocity_hz(phi_u, fs);
                Z = nan(n, D);
                col = 1;
                for h = 1:harmonic_k
                    ch = cos(h * phi_u);
                    sh = sin(h * phi_u);
                    Z(:, col) = ch;
                    Z(:, col + 1) = sh;
                    Z(:, col + 2) = omega .* ch;
                    Z(:, col + 3) = omega .* sh;
                    col = col + 4;
                end
                Yk(1:n, cols) = Z;
        end
    end
    Y_state{k} = Yk;
end
end

function omega = local_phase_velocity_hz(phi_u, fs)
omega = gradient(double(phi_u(:))) * fs / (2 * pi);
omega = fill_finite_local(omega);
if numel(omega) >= 9
    omega = movmedian(omega, 9, 'omitnan');
end
med_omega = median(omega, 'omitnan');
mad_omega = median(abs(omega - med_omega), 'omitnan');
if isfinite(mad_omega) && mad_omega > eps
    lo = med_omega - 6 * 1.4826 * mad_omega;
    hi = med_omega + 6 * 1.4826 * mad_omega;
    omega = min(max(omega, lo), hi);
end
end

function amp = local_amp_from_angle(y, fs, phase_lo, phase_hi)
y = fill_finite_local(y);
[z, p, k] = butter(2, [phase_lo phase_hi] / (fs / 2), 'bandpass');
sos = zp2sos(z, p, k);
y_lp = filtfilt(sos, 1, y);
amp = abs(hilbert(y_lp));
end

function R = fit_state_recon_models(Y_state, Y_delta, layout)
J = layout.J;
R = nan(layout.D + 1, J);
ridge = 1e-6;
for j = 1:J
    cols = layout.cols{j};
    X = [];
    y = [];
    for k = 1:numel(Y_state)
        n = min(size(Y_state{k}, 1), size(Y_delta{k}, 1));
        if n == 0, continue; end
        Xk = [ones(n, 1), Y_state{k}(1:n, cols)];
        yk = Y_delta{k}(1:n, j);
        valid = all(isfinite(Xk), 2) & isfinite(yk);
        X = [X; Xk(valid, :)]; %#ok<AGROW>
        y = [y; yk(valid)]; %#ok<AGROW>
    end
    if size(X, 1) < size(X, 2) + 5, continue; end
    A = X' * X;
    A(1:size(A, 1)+1:end) = A(1:size(A, 1)+1:end) + ridge;
    R(:, j) = A \ (X' * y);
end
end

function P_pred = state_recon_predict(Yp_state, R, layout)
J = layout.J;
P_pred = cell(size(Yp_state));
for k = 1:numel(Yp_state)
    Yp = Yp_state{k};
    Pk = nan(size(Yp, 1), J);
    for j = 1:J
        if all(~isfinite(R(:, j))), continue; end
        cols = layout.cols{j};
        X = [ones(size(Yp, 1), 1), Yp(:, cols)];
        Pk(:, j) = X * R(:, j);
    end
    P_pred{k} = Pk;
end
end

function H = fit_harmonic_models(Y_phase, Y_delta, harmonic_k)
J = size(Y_delta{1}, 2);
B = 1 + 2 * harmonic_k;
H = nan(B, J);
ridge = 1e-6;
for j = 1:J
    c = 2 * j - 1;
    s = 2 * j;
    X = [];
    y = [];
    for k = 1:numel(Y_phase)
        n = min(size(Y_phase{k}, 1), size(Y_delta{k}, 1));
        if n == 0 || all(isnan(Y_phase{k}(:, c))), continue; end
        phi = atan2(Y_phase{k}(1:n, s), Y_phase{k}(1:n, c));
        Xk = harmonic_basis(phi, harmonic_k);
        yk = Y_delta{k}(1:n, j);
        valid = all(isfinite(Xk), 2) & isfinite(yk);
        X = [X; Xk(valid, :)]; %#ok<AGROW>
        y = [y; yk(valid)]; %#ok<AGROW>
    end
    if size(X, 1) < B + 5, continue; end
    A = X' * X;
    A(1:B+1:end) = A(1:B+1:end) + ridge;
    H(:, j) = A \ (X' * y);
end
end

function P_pred = harmonic_predict(Yp_phase, H, harmonic_k)
J = size(H, 2);
P_pred = cell(size(Yp_phase));
for k = 1:numel(Yp_phase)
    Yp = Yp_phase{k};
    Pk = nan(size(Yp, 1), J);
    for j = 1:J
        if all(~isfinite(H(:, j))), continue; end
        c = 2 * j - 1;
        s = 2 * j;
        phi = atan2(Yp(:, s), Yp(:, c));
        X = harmonic_basis(phi, harmonic_k);
        Pk(:, j) = X * H(:, j);
    end
    P_pred{k} = Pk;
end
end

function X = harmonic_basis(phi, harmonic_k)
phi = phi(:);
X = ones(numel(phi), 1 + 2 * harmonic_k);
col = 2;
for h = 1:harmonic_k
    X(:, col) = cos(h * phi);
    X(:, col + 1) = sin(h * phi);
    col = col + 2;
end
end

function Yp = smooth_phase_cells(Yp, smooth_type, fs)
if strcmpi(smooth_type, 'none'), return; end
params = struct('lp_lo', 0.3, 'lp_hi', 3.0, 'omega_hz', 1.0, ...
                'q_noise', 0.01, 'r_noise', 0.5);
Yp = cellfun(@(x) smooth_cs(x, fs, smooth_type, params), Yp, 'UniformOutput', false);
end

function rows = eval_three(Y_true, A_pred, P_pred, F_pred, labels, alpha, fs, rpt_pred_smooth_sec)
J = numel(labels);
rows = repmat(struct('joint', '', 'r_angle', NaN, 'r_phase', NaN, ...
    'r_fused', NaN, 'rpt_angle', NaN, 'rpt_phase', NaN, 'rpt_fused', NaN, ...
    'alpha', NaN), 1, J);
for j = 1:J
    rows(j).joint = labels{j};
    rows(j).alpha = alpha(j);
    [rows(j).r_angle, rows(j).rpt_angle] = eval_one(Y_true, A_pred, j, fs, rpt_pred_smooth_sec);
    [rows(j).r_phase, rows(j).rpt_phase] = eval_one(Y_true, P_pred, j, fs, rpt_pred_smooth_sec);
    [rows(j).r_fused, rows(j).rpt_fused] = eval_one(Y_true, F_pred, j, fs, rpt_pred_smooth_sec);
end
end

function [r_concat, r_pt] = eval_one(Y_true, Y_pred, j, fs, rpt_pred_smooth_sec)
if nargin < 4 || isempty(fs), fs = 250; end
if nargin < 5 || isempty(rpt_pred_smooth_sec), rpt_pred_smooth_sec = 0; end
y = [];
yp = [];
r_trials = nan(1, numel(Y_true));
for k = 1:numel(Y_true)
    if k > numel(Y_pred) || isempty(Y_pred{k}) || size(Y_pred{k}, 2) < j
        continue;
    end
    n = min(size(Y_true{k}, 1), size(Y_pred{k}, 1));
    if n < 2
        continue;
    end
    yt = Y_true{k}(1:n, j);
    yh = Y_pred{k}(1:n, j);
    yh_rpt = smooth_prediction_for_rpt(yh, fs, rpt_pred_smooth_sec);
    r_trials(k) = safe_corr_local(yt, yh_rpt);
    y = [y; yt]; %#ok<AGROW>
    yp = [yp; yh]; %#ok<AGROW>
end
r_concat = safe_corr_local(y, yp);
r_pt = mean(r_trials, 'omitnan');
end

function y = smooth_prediction_for_rpt(y, fs, smooth_sec)
if smooth_sec <= 0 || numel(y) < 3
    return;
end
w = max(3, round(double(smooth_sec) * double(fs)));
if mod(w, 2) == 0
    w = w + 1;
end
w = min(w, numel(y));
if mod(w, 2) == 0
    w = max(1, w - 1);
end
if w >= 3
    y = movmean(y, [w - 1, 0], 'omitnan');
end
end

function [X_cell, Y_cell] = build_xy_delta0_local(S, ev, joint_idx, opt)
fs = S.eeg.fs;
lag_samples = unique(round(opt.lag_ms * fs / 1000));
L = numel(lag_samples);
max_lag_pos = max(0, max(lag_samples));
x_delta = build_eeg_band_feature_signal_local(S, opt, fs);
x_feat = x_delta;
feature_mode = 'delta';
if isfield(opt, 'feature_mode')
    feature_mode = lower(char(opt.feature_mode));
end
if any(strcmp(feature_mode, {'delta_ea', 'delta_ea_shared_pca'}))
    x_delta = euclidean_align_eeg_local(x_delta);
    x_feat = x_delta;
end
if any(strcmp(feature_mode, {'delta_shared_pca', 'delta_shared_pca_tf_whiten', 'delta_ea_shared_pca'}))
    if ~isfield(opt, 'shared_pca_file') || isempty(opt.shared_pca_file)
        error('feature_mode=%s requires shared_pca_file', feature_mode);
    end
    shared_file = char(opt.shared_pca_file);
    if contains(shared_file, '%s')
        shared_file = sprintf(shared_file, S.label);
    end
    P = load(shared_file);
    if ~isfield(P, 'basis')
        error('shared_pca_file %s has no basis variable', shared_file);
    end
    if isfield(P, 'labels')
        assert(numel(P.labels) == size(P.basis, 1), ...
            'shared PCA label count does not match basis rows for session %s', S.label);
    end
    if size(P.basis, 1) ~= size(x_delta, 1)
        error('shared PCA rows %d do not match decoder feature rows %d for session %s', ...
            size(P.basis, 1), size(x_delta, 1), S.label);
    end
    K = min(round(opt.shared_pca_components), size(P.basis, 2));
    x_feat = double(P.basis(:, 1:K))' * double(x_delta);
elseif any(strcmp(feature_mode, {'delta_fold_diagwhiten', 'delta_fold_diagwhiten_shared_pca'}))
    x_feat = x_delta;
elseif strcmp(feature_mode, 'delta_mubeta_roi')
    cc = [];
    for ci = 1:numel(opt.central_labels)
        hit = find(strcmpi(S.eeg.labels, opt.central_labels{ci}), 1);
        if ~isempty(hit), cc(end+1) = hit; end %#ok<AGROW>
    end
    if ~isempty(cc)
        bands = [8 13; 13 30];
        env = [];
        for bi = 1:size(bands, 1)
            [zb, pb, kb] = butter(4, bands(bi, :) / (fs / 2), 'bandpass');
            xb = filtfilt(zp2sos(zb, pb, kb), 1, double(S.eeg.x(cc, :))')';
            env_b = log(abs(hilbert(xb'))' + eps);
            env = [env; env_b]; %#ok<AGROW>
        end
        env = (env - mean(env, 2, 'omitnan')) ./ std(env, 0, 2, 'omitnan');
        env(~isfinite(env)) = 0;
        x_feat = [x_delta; env];
    end
end
nfeat = size(x_feat, 1);
T_total = size(x_feat, 2);
trim_start = round(opt.trim_start_sec * fs);
trim_end = round(opt.trim_end_sec * fs);
start_win_n = max(1, round(opt.start_window_sec * S.goni.fs));
X_cell = cell(1, numel(ev));
Y_cell = cell(1, numel(ev));
cp_export_dir = getenv('CP_EXPORT_DIR');
if ~isempty(cp_export_dir)
    cpx = cp_export_h1_init_local(S, ev, joint_idx, opt, T_total, lag_samples, start_win_n, trim_start, trim_end);
end
cp_rest_dir = cp_export_h3_dir_local(opt);
if ~isempty(cp_rest_dir)
    crx = cp_export_h3_init_local(S, ev, opt, T_total, lag_samples);
end
for t = 1:numel(ev)
    et0_orig = ev(t).etstart;
    gt0_orig = ev(t).gtstart;
    et0 = et0_orig + trim_start;
    et1 = ev(t).etstop - trim_end;
    gt0 = gt0_orig + trim_start;
    gt1 = ev(t).gtstop - trim_end;
    if et1 < et0 || gt1 < gt0
        X_cell{t} = zeros(nfeat * L, 0);
        Y_cell{t} = zeros(0, numel(joint_idx));
        continue;
    end
    Lk = min(et1 - et0 + 1, gt1 - gt0 + 1);
    valid_start = max(1, 1 - min(lag_samples)) + et0 - 1;
    valid_end = min(T_total - max(lag_samples), et0 + Lk - 1 - max_lag_pos);
    if valid_end < valid_start
        if ~isempty(cp_export_dir), cpx.Lk(t) = Lk; end
        X_cell{t} = zeros(nfeat * L, 0);
        Y_cell{t} = zeros(0, numel(joint_idx));
        continue;
    end
    out_t = valid_start:valid_end;
    Tk = numel(out_t);
    Xk = zeros(nfeat * L, Tk);
    [base_mu, base_sd] = feature_pretrial_baseline(x_feat, S.events, ev(t), fs);
    if ~isempty(cp_export_dir)
        cpx = cp_export_h1_trial_local(cpx, t, x_feat, S, ev, joint_idx, et0, gt0, gt0_orig, Lk, Tk, base_mu, fs);
    end
    sample_t = out_t;
    if isfield(opt, 'input_control_mode') && strcmpi(opt.input_control_mode, 'resting_random_window_within_trial')
        sample_t = resting_random_anchor_local(opt.rest_events, Tk, lag_samples, T_total, ...
            double(opt.input_control_seed) + 1009 * t);
        if ~isempty(cp_rest_dir)
            crx = cp_export_h3_trial_local(crx, t, x_feat, sample_t, lag_samples, et0, Lk, Tk, base_mu);
        end
        if isempty(sample_t)
            X_cell{t} = zeros(nfeat * L, 0);
            Y_cell{t} = zeros(0, numel(joint_idx));
            continue;
        end
    elseif isfield(opt, 'input_control_mode') && strcmpi(opt.input_control_mode, 'within_trial_circular_shift')
        sample_t = circular_shift_anchor_local(out_t, lag_samples, T_total, ...
            double(opt.input_control_seed) + 1009 * t);
    end
    for li = 1:L
        Xseg = x_feat(:, sample_t + lag_samples(li));
        switch lower(char(opt.eeg_baseline_mode))
            case 'none'
            case 'mean'
                Xseg = Xseg - base_mu;
            case {'zscore', 'mean_std'}
                Xseg = (Xseg - base_mu) ./ base_sd;
            otherwise
                error('Unknown eeg_baseline_mode: %s', opt.eeg_baseline_mode);
        end
        if isfield(opt, 'input_control_mode') && strcmpi(opt.input_control_mode, 'zero_eeg')
            Xseg(:) = 0;
        end
        Xk((li-1)*nfeat+1:li*nfeat, :) = Xseg;
    end
    X_cell{t} = Xk;
    local_idx = out_t - et0 + 1;
    goni_t = gt0 + local_idx - 1;
    goni_t = min(goni_t, size(S.goni.x, 1));
    Yk = nan(Tk, numel(joint_idx));
    base_end = min(ev(t).gtstop, gt0_orig + start_win_n - 1);
    for j = 1:numel(joint_idx)
        if isnan(joint_idx(j)), continue; end
        base = mean(double(S.goni.x(gt0_orig:base_end, joint_idx(j))), 'omitnan');
        Yk(:, j) = double(S.goni.x(goni_t, joint_idx(j))) - base;
    end
    Y_cell{t} = Yk;
end
if ~isempty(cp_export_dir)
    cp_export_h1_write_local(cp_export_dir, cpx, X_cell, Y_cell, lag_samples, nfeat, opt);
end
if ~isempty(cp_rest_dir)
    cp_export_h3_write_local(cp_rest_dir, crx, X_cell, lag_samples, nfeat);
end
end

% ---------------------------------------------------------------------------------------------------------------
% Feature export for the Python readout (02_decoding/pyport), hooks H1 and H2.
% Active only when env CP_EXPORT_DIR is non-empty; then one <label>_export.mat (-v7.3) per session is written to it:
% H1 (from build_xy_delta0_local) = un-lagged fold-independent feature window U_raw{t} = x_feat(:, et0:et0+Lk-1),
% the pre-trial feature mean base_mu{t} actually subtracted, the full-window target Y_full{t} (Lk x 6, same
% expressions as the decoder's Y), per-trial indices and per-session provenance; H2 (from the main function, after the
% CV split) appends the outer fold assignment, the valid-trial mask and the resolved basis path with its checksum.
% H1 self-check (error, nothing written, on any mismatch): every valid trial Y_cell{t} == Y_full{t}(off+1:off+Tk,:)
% and its middle lag block == U_raw - base_mu, all lag blocks of the first valid trial, base_mu re-derived from the
% recorded window; all bitwise (isequaln). env CP_EXPORT_ONLY=1 skips the session's decoding after H2.
function cpx = cp_export_h1_init_local(S, ev, joint_idx, opt, T_total, lag_samples, start_win_n, trim_start, trim_end)
icm = 'full';
if isfield(opt, 'input_control_mode') && ~isempty(opt.input_control_mode), icm = lower(char(opt.input_control_mode)); end
assert(strcmp(icm, 'full'), 'CP_EXPORT: input_control_mode must be full (got %s)', icm);
bmode = lower(char(opt.eeg_baseline_mode));
assert(any(strcmp(bmode, {'mean', 'none'})), 'CP_EXPORT: eeg_baseline_mode must be mean or none (got %s)', bmode);
assert(isequal(double(S.eeg.fs), double(S.goni.fs)), 'CP_EXPORT: fs_eeg %g ~= fs_goni %g', S.eeg.fs, S.goni.fs);
assert(all(lag_samples <= 0), 'CP_EXPORT: needs all stage-1 lags <= 0');
n_ev = numel(ev);
cpx = struct();
cpx.session_label = char(S.label);
cpx.n_ev = n_ev;
cpx.U_raw = cell(1, n_ev);
cpx.base_mu = cell(1, n_ev);
cpx.Y_full = cell(1, n_ev);
cpx.et0 = nan(n_ev, 1); cpx.gt0 = nan(n_ev, 1); cpx.Lk = nan(n_ev, 1); cpx.Tk = zeros(n_ev, 1);
cpx.base_win = nan(n_ev, 2);
cpx.y_nan_count = nan(n_ev, numel(joint_idx));
cpx.ev_trial_idx = nan(n_ev, 1);
cpx.etstart = nan(n_ev, 1); cpx.etstop = nan(n_ev, 1); cpx.gtstart = nan(n_ev, 1); cpx.gtstop = nan(n_ev, 1);
for t = 1:n_ev
    if isfield(ev, 'trial_idx'), cpx.ev_trial_idx(t) = double(ev(t).trial_idx); end
    cpx.etstart(t) = double(ev(t).etstart); cpx.etstop(t) = double(ev(t).etstop);
    cpx.gtstart(t) = double(ev(t).gtstart); cpx.gtstop(t) = double(ev(t).gtstop);
end
cpx.T_total = T_total;
cpx.fs_eeg = double(S.eeg.fs);
cpx.fs_goni = double(S.goni.fs);
cpx.goni_nrows = size(S.goni.x, 1);
cpx.goni_class = class(S.goni.x);
cpx.eeg_class = class(S.eeg.x);
cpx.eeg_labels = S.eeg.labels(:)';
cpx.goni_labels = S.goni.labels(:)';
cpx.joint_idx = double(joint_idx(:)');
cpx.lag_samples = double(lag_samples(:)');
cpx.crop_offset = max(1, 1 - min(lag_samples)) - 1;
cpx.start_win_n = start_win_n;
cpx.trim_start = trim_start; cpx.trim_end = trim_end;
cpx.eeg_baseline_mode = bmode;
cpx.bp_band_feature_mode = lower(char(opt.bp_band_feature_mode));
cpx.bp_channel_labels = opt.bp_channel_labels;
if isfield(opt, 'bp_band_tbl') && isfield(opt, 'session_label') && isfield(opt, 'bp_band_harmonic_k_list')
    hk = double(opt.bp_band_harmonic_k_list(:)');
    cpx.bands = nan(numel(hk), 3);
    for bi = 1:numel(hk)
        idx = find(string(opt.bp_band_tbl.session) == string(opt.session_label) & ...
            double(opt.bp_band_tbl.harmonic_k) == hk(bi));
        assert(numel(idx) == 1, 'CP_EXPORT: band row k=%g for %s', hk(bi), opt.session_label);
        cpx.bands(bi, :) = [hk(bi), double(opt.bp_band_tbl.band_lo(idx)), double(opt.bp_band_tbl.band_hi(idx))];
    end
else
    cpx.bands = [NaN, double(opt.bp_lo), double(opt.bp_hi)];
end
cpx.goni_qc_flags_env = getenv('GAIT_GONI_QC_TRIAL_FLAGS');
cpx.gait_triple_root_env = getenv('GAIT_TRIPLE_ROOT');
cpx.matlab_version = version;
cpx.cp_path = [mfilename('fullpath') '.m'];
cpx.cp_md5 = cp_export_md5_local(cpx.cp_path);
end

function cpx = cp_export_h1_trial_local(cpx, t, x_feat, S, ev, joint_idx, et0, gt0, gt0_orig, Lk, Tk, base_mu, fs)
assert(et0 >= 1 && et0 + Lk - 1 <= size(x_feat, 2), ...
    'CP_EXPORT: trial %d window [%d %d] outside 1..%d', t, et0, et0 + Lk - 1, size(x_feat, 2));
cpx.et0(t) = et0; cpx.gt0(t) = gt0; cpx.Lk(t) = Lk; cpx.Tk(t) = Tk;
cpx.U_raw{t} = x_feat(:, et0:et0 + Lk - 1);
cpx.base_mu{t} = base_mu;
% pre-trial window: same expressions as feature_pretrial_baseline, re-derived and checked against base_mu
n_base = round(fs);
etstart = round(double(ev(t).etstart));
if isfield(ev(t), 'eeg_pre1s_window') && all(isfinite(ev(t).eeg_pre1s_window))
    win = round(double(ev(t).eeg_pre1s_window));
    start_idx = max(1, win(1));
    stop_idx = min(size(x_feat, 2), win(2));
else
    prev_stop = 0;
    if ~isempty(S.events) && isfield(S.events, 'etstop')
        starts = [S.events.etstart];
        stops = [S.events.etstop];
        prev = find(starts < etstart & stops < etstart);
        if ~isempty(prev)
            prev_stop = max(stops(prev));
        end
    end
    start_idx = max([1, etstart - n_base, prev_stop + 1]);
    stop_idx = min(size(x_feat, 2), etstart - 1);
end
cpx.base_win(t, :) = [start_idx, stop_idx];
if stop_idx >= start_idx
    mu = mean(x_feat(:, start_idx:stop_idx), 2, 'omitnan');
else
    mu = zeros(size(x_feat, 1), 1);
end
mu(~isfinite(mu)) = 0;
assert(isequaln(mu, base_mu), 'CP_EXPORT: base_mu re-derivation mismatch, trial %d', t);
% Y over the full trial window, identical expressions to the decoder's Y (build_xy_delta0_local)
goni_t = gt0 + (1:Lk) - 1;
goni_t = min(goni_t, size(S.goni.x, 1));
Yf = nan(Lk, numel(joint_idx));
base_end = min(ev(t).gtstop, gt0_orig + cpx.start_win_n - 1);
for j = 1:numel(joint_idx)
    if isnan(joint_idx(j)), continue; end
    base = mean(double(S.goni.x(gt0_orig:base_end, joint_idx(j))), 'omitnan');
    Yf(:, j) = double(S.goni.x(goni_t, joint_idx(j))) - base;
end
cpx.Y_full{t} = Yf;
cpx.y_nan_count(t, :) = sum(isnan(Yf), 1);
end

function cp_export_h1_write_local(xdir, cpx, X_cell, Y_cell, lag_samples, nfeat, opt)
off = cpx.crop_offset;
L = numel(lag_samples);
valid = find(cpx.Tk > 0)';
assert(~isempty(valid), 'CP_EXPORT: no valid trial in %s', cpx.session_label);
n_xblk = 0;
for t = valid
    Tk = cpx.Tk(t);
    assert(size(X_cell{t}, 2) == Tk && size(Y_cell{t}, 1) == Tk, 'CP_EXPORT: Tk mismatch trial %d', t);
    assert(isequaln(Y_cell{t}, cpx.Y_full{t}(off + (1:Tk), :)), 'CP_EXPORT: Y self-check FAIL trial %d', t);
    if t == valid(1), lis = 1:L; else, lis = ceil(L / 2); end
    for li = lis
        U = cpx.U_raw{t}(:, off + lag_samples(li) + (1:Tk));
        if strcmp(cpx.eeg_baseline_mode, 'mean'), U = U - cpx.base_mu{t}; end
        assert(isequaln(X_cell{t}((li-1)*nfeat+1:li*nfeat, :), U), ...
            'CP_EXPORT: X self-check FAIL trial %d lag %d', t, lag_samples(li));
        n_xblk = n_xblk + 1;
    end
end
assert(size(cpx.U_raw{valid(1)}, 1) == nfeat, 'CP_EXPORT: feature rows %d ~= %d', size(cpx.U_raw{valid(1)}, 1), nfeat);
for t = find(cpx.Tk == 0)'
    % empty trial: keep numbering aligned with ev
    cpx.U_raw{t} = zeros(nfeat, 0);
    cpx.base_mu{t} = zeros(nfeat, 0);
    cpx.Y_full{t} = zeros(0, numel(cpx.joint_idx));
end
cpx.self_check = sprintf('PASS: Y %d trials, X %d lag blocks (all %d lags of trial %d + middle lag of every valid trial), base_mu %d trials', ...
    numel(valid), n_xblk, L, valid(1), numel(valid));
cpx.lag_ms = double(opt.lag_ms(:)');
cpx.export_created = datestr(now, 'yyyy-mm-dd HH:MM:SS');
if ~exist(xdir, 'dir'), mkdir(xdir); end
f = fullfile(xdir, sprintf('%s_export.mat', cpx.session_label));
if isfile(f)
    fprintf('[CP_EXPORT] replacing existing %s\n', f);
    delete(f);
end
save(f, '-struct', 'cpx', '-v7.3');
fprintf('[CP_EXPORT] H1 self-check PASS label=%s n_ev=%d n_valid=%d x_blocks=%d file=%s\n', ...
    cpx.session_label, cpx.n_ev, numel(valid), n_xblk, f);
end

function skip = cp_export_h2_local(label, target_labels, ev, fa, X_delta, Y_delta, Y_phase, params)
skip = false;
xdir = getenv('CP_EXPORT_DIR');
xonly = any(strcmpi(getenv('CP_EXPORT_ONLY'), {'1', 'true', 'yes', 'on'}));
if isempty(xdir)
    assert(~xonly, 'CP_EXPORT_ONLY=1 requires CP_EXPORT_DIR');
    return;
end
f = fullfile(xdir, sprintf('%s_export.mat', char(label)));
assert(isfile(f), 'CP_EXPORT H2: H1 file missing %s', f);
n_ev = numel(ev);
assert(~isfield(fa, 'trial_idx'), 'CP_EXPORT H2: pooled-session split not supported');
assert(numel(fa.fold) == n_ev && numel(X_delta) == n_ev, 'CP_EXPORT H2: fold/trial count mismatch');
h2 = struct();
h2.fa_fold = double(fa.fold(:));
h2.fa_seed = double(fa.seed);
h2.valid_trial = logical(valid_trial_mask(X_delta, Y_delta, Y_phase, 1:n_ev)');
h2.target_labels = target_labels(:)';
h2.condition = char(params.condition);
basis_path = char(params.shared_pca_file);
if contains(basis_path, '%s'), basis_path = sprintf(basis_path, char(label)); end
h2.basis_path = basis_path;
h2.basis_path_abs = basis_path;
if ~startsWith(basis_path, '/'), h2.basis_path_abs = fullfile(pwd, basis_path); end
h2.basis_md5 = cp_export_md5_local(basis_path);
P = load(basis_path, 'basis');
h2.basis_size = size(P.basis);
h2.K = min(round(params.shared_pca_components), size(P.basis, 2));
h2.outer_k = double(params.outer_k);
h2.cv_mode = char(params.cv_mode);
h2.inner_k = double(params.inner_k);
h2.feature_mode = char(params.feature_mode);
h2.shared_pca_scaler_scope = char(params.shared_pca_scaler_scope);
h2.lambda_grid = double(params.lambda_grid(:)');
h2.post_mode = char(params.post_mode);
h2.post_lambda_grid = double(params.post_lambda_grid(:)');
h2.post_fir_lag_ms = double(params.post_fir_lag_ms(:)');
h2.post_train_from_sec = double(params.post_train_from_sec);
h2.alpha_grid = double(params.alpha_grid(:)');
save(f, '-struct', 'h2', '-append');
fprintf('[CP_EXPORT] H2 written label=%s n_ev=%d n_valid=%d K=%d basis_md5=%s basis=%s\n', ...
    char(label), n_ev, sum(h2.valid_trial), h2.K, h2.basis_md5, basis_path);
skip = xonly;
if skip
    fprintf('[CP_EXPORT] CP_EXPORT_ONLY=1: decoding of %s skipped\n', char(label));
end
end

function h = cp_export_md5_local(path)
fid = fopen(path, 'r');
assert(fid > 0, 'CP_EXPORT: cannot open %s', path);
b = fread(fid, Inf, '*uint8');
fclose(fid);
md = java.security.MessageDigest.getInstance('MD5');
md.update(b);
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2)', 1, []));
end
% ---------------------------------------------------------------------------------------------------------------

function x_delta = build_eeg_band_feature_signal_local(S, opt, fs)
[ch_idx, eeg_labels] = resolve_channel_indices_local(S.eeg.labels, opt.bp_channel_labels);
eeg_x = double(S.eeg.x(ch_idx, :));
harmonic_list = [];
if isfield(opt, 'bp_band_harmonic_k_list') && ~isempty(opt.bp_band_harmonic_k_list)
    harmonic_list = double(opt.bp_band_harmonic_k_list(:)');
end
if ~isempty(harmonic_list) && isfield(opt, 'bp_band_tbl') && isfield(opt, 'session_label')
    x_parts = cell(1, numel(harmonic_list));
    xa_parts = cell(1, numel(harmonic_list));
    spatial_parts = cell(1, numel(harmonic_list));
    extra_parts = cell(1, numel(harmonic_list));
    bp_feature_mode = lower(char(opt.bp_band_feature_mode));
    causal_mode = is_causal_feature_mode_local(bp_feature_mode);
    relphase_mode = is_relphase_feature_mode_local(bp_feature_mode);
    append_spatial_mode = is_spatial_append_feature_mode_local(bp_feature_mode);
    append_phaseextra_mode = is_phaseextra_append_feature_mode_local(bp_feature_mode);
    for bi = 1:numel(harmonic_list)
        mask = string(opt.bp_band_tbl.session) == string(opt.session_label) & ...
            double(opt.bp_band_tbl.harmonic_k) == double(harmonic_list(bi));
        idx = find(mask);
        if numel(idx) ~= 1
            error('Expected one bp band for session=%s harmonic_k=%g, found %d', ...
                opt.session_label, harmonic_list(bi), numel(idx));
        end
        bp_lo = double(opt.bp_band_tbl.band_lo(idx));
        bp_hi = double(opt.bp_band_tbl.band_hi(idx));
        if ~isfinite(bp_lo) || ~isfinite(bp_hi) || bp_lo <= 0 || bp_hi <= bp_lo
            error('Invalid bp band for session=%s harmonic_k=%g: [%g %g]', ...
                opt.session_label, harmonic_list(bi), bp_lo, bp_hi);
        end
        [z, p, k] = butter(4, [bp_lo bp_hi] / (fs / 2), 'bandpass');
        if causal_mode && ~strcmp(bp_feature_mode, 'quadrature_causal')
            x_band = [];
        elseif causal_mode
            % Same butter(4) design as the zero-phase path, zero-state, no filtfilt.
            x_band = causal_sos_filter_local(zp2sos(z, p, k), eeg_x);
        else
            x_band = filtfilt(zp2sos(z, p, k), 1, eeg_x')';
        end
        if causal_mode && ~strcmp(bp_feature_mode, 'quadrature_causal')
            x_parts{bi} = causal_variant_feature_local(bp_feature_mode, eeg_x, fs, bp_lo, bp_hi);
        elseif causal_mode
            x_parts{bi} = make_band_feature_causal_local(x_band, fs, 0.5 * (bp_lo + bp_hi));
        elseif append_spatial_mode
            x_parts{bi} = make_band_feature_local(x_band, 'quadrature');
            spatial_parts{bi} = make_spatial_prior_quadrature_feature_local( ...
                x_band, eeg_labels, harmonic_list(bi), spatial_append_prior_mode_local(bp_feature_mode));
        elseif append_phaseextra_mode
            x_parts{bi} = make_band_feature_local(x_band, 'quadrature');
            extra_parts{bi} = make_phaseextra_append_feature_local(x_band, bp_feature_mode);
        elseif relphase_mode
            xa_parts{bi} = hilbert(x_band')';
            x_parts{bi} = [real(xa_parts{bi}); imag(xa_parts{bi})];
        elseif is_spatial_prior_feature_mode_local(bp_feature_mode)
            x_parts{bi} = make_spatial_prior_quadrature_feature_local( ...
                x_band, eeg_labels, harmonic_list(bi), bp_feature_mode);
        else
            x_parts{bi} = make_band_feature_local(x_band, bp_feature_mode);
        end
    end
    x_delta = vertcat(x_parts{:});
    if append_spatial_mode
        x_delta = [x_delta; vertcat(spatial_parts{:})];
    end
    if append_phaseextra_mode
        x_delta = [x_delta; vertcat(extra_parts{:})];
    end
    if relphase_mode
        x_delta = [x_delta; make_relative_phase_feature_local(xa_parts, harmonic_list)];
    end
    return;
end
if is_harmonic_only_feature_mode_local(opt.bp_band_feature_mode)
    error('%s requires bp_band_harmonic_k_list [1 2 3]', char(opt.bp_band_feature_mode));
end
[z, p, k] = butter(4, [opt.bp_lo opt.bp_hi] / (fs / 2), 'bandpass');
x_band = filtfilt(zp2sos(z, p, k), 1, eeg_x')';
x_delta = make_band_feature_local(x_band, lower(char(opt.bp_band_feature_mode)));
end

function x_feature = causal_variant_feature_local(mode, eeg_x, fs, bp_lo, bp_hi)
% Strictly causal analytic-pair variants; every output sample uses
% only past and present EEG. Row layout [real; imag], as make_band_feature_local('quadrature').
%   quadrature_o2_causal   : butter(2) band-pass on the same band (lower group delay), forward-only,
%                            quarter-period quadrature
%   quadrature_wide_causal : butter(2) band-pass on fc +/- 0.30 Hz (wider -> shorter delay), forward-only,
%                            quarter-period quadrature
%   echt_causal            : endpoint-corrected Hilbert transform (Schreglmann et al. 2021) over a 4 s
%                            window with a 2nd-order band-pass on the same band; implemented as its
%                            exact equivalent complex FIR kernel h = ifft(A .* H)
%   gabor_causal           : one-sided Hann-windowed complex exponential at fc, length 2 cycles
%                            (zero phase lag at fc, envelope delay = 1 cycle)
fc = 0.5 * (bp_lo + bp_hi);
x = eeg_x';                                   % [T x nch]
% iir_o<order>_bw<nat|width Hz with p for the point>_causal, e.g. iir_o2_bw0p5_causal:
% butter(order) band-pass, forward-only; 'nat' keeps the session band, a number sets a band of that total width
% centred on fc; then the quarter-period quadrature used by quadrature_causal.
% fir<N>_causal = linear-phase FIR band-pass, N taps (fir1, Hamming) on the session band,
% forward-only (constant delay (N-1)/2 samples), then quarter-period quadrature.
tok = regexp(mode, '^fir(\d+)_causal$', 'tokens', 'once');
if ~isempty(tok)
    ntap = str2double(tok{1});
    h = fir1(ntap - 1, [bp_lo bp_hi] / (fs / 2), 'bandpass');
    xb = filter(h, 1, x, [], 1)';             % [nch x T]
    x_feature = make_band_feature_causal_local(xb, fs, fc);
    return;
end
% kf_q<a>em<b>_causal = steady-state Kalman filter of a damped rotating oscillator at fc
% (state-space phase estimation, Wodeyar et al. 2021); damping from the session band width, process/observation
% noise ratio q = a*10^-b; state [re; im] is the analytic estimate.
tok = regexp(mode, '^kf_q(\d+)em(\d+)_causal$', 'tokens', 'once');
if ~isempty(tok)
    q = str2double(tok{1}) * 10^(-str2double(tok{2}));
    a = exp(-pi * (bp_hi - bp_lo) / fs); w = 2 * pi * fc / fs;
    A = a * [cos(w) -sin(w); sin(w) cos(w)]; Hm = [1 0]; P = eye(2);
    for it = 1:20000
        Pp = A * P * A' + q * eye(2); K = Pp * Hm' / (Hm * Pp * Hm' + 1); P = (eye(2) - K * Hm) * Pp;
    end
    M = (eye(2) - K * Hm) * A;
    T = size(eeg_x, 2); s = zeros(2, size(eeg_x, 1)); out = zeros(2 * size(eeg_x, 1), T);
    nch = size(eeg_x, 1);
    for tt = 1:T
        s = M * s + K * eeg_x(:, tt)';
        out(1:nch, tt) = s(1, :)'; out(nch+1:end, tt) = s(2, :)';
    end
    x_feature = out;
    return;
end
tok = regexp(mode, '^iir_o(\d)_bw(nat|\d+p\d+)_causal$', 'tokens', 'once');
if ~isempty(tok)
    ordr = str2double(tok{1});
    if strcmp(tok{2}, 'nat')
        lo = bp_lo; hi = bp_hi;
    else
        bw = str2double(strrep(tok{2}, 'p', '.'));
        lo = max(0.05, fc - bw / 2); hi = fc + bw / 2;
    end
    [z, p, k] = butter(ordr, [lo hi] / (fs / 2), 'bandpass');
    xb = causal_sos_filter_local(zp2sos(z, p, k), eeg_x);
    x_feature = make_band_feature_causal_local(xb, fs, fc);
    return;
end
% cdm_o<order>_c<cutoff Hz>_causal = fixed-frequency complex demodulation: butter(order) low-pass at the cutoff,
%   frequency-shifted to fc (coefficients b_k e^{i w k}, a_k e^{i w k}), i.e. a one-sided complex resonator; output
%   is the complex band signal itself (no quadrature approximation).
tok = regexp(mode, '^cdm_o(\d)_c(\d+p\d+)_causal$', 'tokens', 'once');
if ~isempty(tok)
    [b, a] = butter(str2double(tok{1}), str2double(strrep(tok{2}, 'p', '.')) / (fs / 2));
    w = 2 * pi * fc / fs;
    y = filter(b .* exp(1i * w * (0:numel(b)-1)), a .* exp(1i * w * (0:numel(a)-1)), x, [], 1);
    x_feature = [real(y)'; imag(y)'];
    return;
end
% cfir_d<delay s>_causal = least-squares complex FIR (Smetanin et al. 2020 style), 3 s long, approximating the ideal
%   one-sided band-pass [lo, hi] delayed by D; transition bands +/-0.1 Hz around the edges unweighted; dense grid
%   (0.005 Hz) on |f| <= 5 Hz, 0.1 Hz elsewhere.
tok = regexp(mode, '^cfir_d(\d+p\d+)_causal$', 'tokens', 'once');
if ~isempty(tok)
    D = str2double(strrep(tok{1}, 'p', '.'));
    N = round(3 * fs); n = 0:N-1;
    f = unique([-5:0.005:5, -fs/2:0.1:fs/2])';
    dsr = double(f >= bp_lo & f <= bp_hi) .* exp(-1i * 2 * pi * f * D);
    wt = ones(size(f));
    wt((f < bp_lo & f > bp_lo - 0.1) | (f > bp_hi & f < bp_hi + 0.1)) = 0;
    A = exp(-1i * 2 * pi * f * n / fs);
    h = ((wt .* A) \ (wt .* dsr)).';
    y = filter(h, 1, x, [], 1);
    x_feature = [real(y)'; imag(y)'];
    return;
end
% oracle_d<delay s>_causal = not causal: the zero-phase feature (filtfilt butter(4) +
%   whole-record hilbert, as make_band_feature_local('quadrature')) delayed by D s.
tok = regexp(mode, '^oracle_d(\d+p\d+)_causal$', 'tokens', 'once');
if ~isempty(tok)
    [z, p, k] = butter(4, [bp_lo bp_hi] / (fs / 2), 'bandpass');
    xa = hilbert(filtfilt(zp2sos(z, p, k), 1, eeg_x'))';   % [nch x T], = hilbert(x_band')' of the zero-phase path
    dly = round(str2double(strrep(tok{1}, 'p', '.')) * fs);
    xa = [zeros(size(xa, 1), dly), xa(:, 1:end-dly)];
    x_feature = [real(xa); imag(xa)];
    return;
end
% arpad_p<order>_w<window s>_h<horizon s>_causal = AR forward-prediction padding: every 80 ms,
%   fit AR(order) (arburg, per channel) on the past <window> s of the causally low-passed, 25 Hz-decimated EEG, forecast
%   <horizon> s, then apply the butter(4) zero-phase band-pass + hilbert to [past window, forecast] and read
%   the analytic value at the current sample (linearly interpolated to 250 Hz from positions computed at that update,
%   advanced by the anti-alias low-pass group delay so the 25 Hz path is not phase-lagged).
%   Every step outside the window is forward-only; inside it only past samples + forecast are used -> strictly causal.
% arpadoracle_p<..>_w<..>_h<..>_causal = not causal: same machinery with the true future samples in
%   place of the forecast.
tok = regexp(mode, '^arpad(oracle)?_p(\d+)_w(\d+)_h(\d+)_causal$', 'tokens', 'once');
if ~isempty(tok)
    x_feature = arpad_feature_local(eeg_x, fs, bp_lo, bp_hi, str2double(tok{2}), str2double(tok{3}), ...
        str2double(tok{4}), ~isempty(tok{1}));
    return;
end
switch mode
    case 'quadrature_o2_causal'
        [z, p, k] = butter(2, [bp_lo bp_hi] / (fs / 2), 'bandpass');
        xb = causal_sos_filter_local(zp2sos(z, p, k), eeg_x);
        x_feature = make_band_feature_causal_local(xb, fs, fc);
        return;
    case 'quadrature_wide_causal'
        lo = max(0.05, fc - 0.30); hi = fc + 0.30;
        [z, p, k] = butter(2, [lo hi] / (fs / 2), 'bandpass');
        xb = causal_sos_filter_local(zp2sos(z, p, k), eeg_x);
        x_feature = make_band_feature_causal_local(xb, fs, fc);
        return;
    case 'echt_causal'
        N = round(4 * fs);
        q = 0:N-1;
        A = zeros(1, N); A(1) = 1; A(2:ceil(N/2)) = 2;
        if mod(N, 2) == 0, A(N/2 + 1) = 1; end
        [z, p, k] = butter(2, [bp_lo bp_hi] / (fs / 2), 'bandpass');
        zz = exp(1i * 2 * pi * q / N);
        H = k * ones(1, N);
        for ii = 1:numel(z), H = H .* (zz - z(ii)); end
        for ii = 1:numel(p), H = H ./ (zz - p(ii)); end
        h = ifft(A .* H);                     % h(m+1) weights x[n-m], m = 0..N-1
    case 'gabor_causal'
        L = max(2, round(2 * fs / fc));
        m = 0:L-1;
        w = 0.5 - 0.5 * cos(2 * pi * (m + 0.5) / L);
        w = w / sum(w);
        h = 2 * w .* exp(1i * 2 * pi * fc * m / fs);
    otherwise
        error('unknown causal variant %s', mode);
end
y = filter(h, 1, x, [], 1);                   % causal complex FIR, zero initial state
x_feature = [real(y)'; imag(y)'];
end

function tf = is_causal_feature_mode_local(feature_mode)
% Gate for the strictly causal feature path: true only when the bp_band_feature_mode
% string ends in '_causal', so every other mode name ('real', 'quadrature',
% 'quadrature_phaseperm', ...) keeps the zero-phase filtering path. The published
% decoder uses the causal mode 'cdm_o1_c0p18_causal'.
m = lower(char(feature_mode));
tf = numel(m) >= 7 && strcmp(m(end-6:end), '_causal');
end

function x_out = causal_sos_filter_local(sos, x)
% Forward-only cascaded second-order-section filtering from a zero initial state, the
% causal counterpart of filtfilt(sos, 1, x')'.
% zp2sos called with ONE output embeds the overall gain in the first section, so the
% cascade below reproduces the zero-phase call's transfer function with no extra gain.
% The (b, a) transfer-function form is deliberately NOT used: at fs = 250 Hz these
% bands are ~0.005-0.021 normalised and a direct-form butter(4) is ill-conditioned.
x_out = x';                                  % [T x nch]
for si = 1:size(sos, 1)
    x_out = filter(sos(si, 1:3), sos(si, 4:6), x_out, [], 1);
end
x_out = x_out';                              % [nch x T]
end

function x_feature = make_band_feature_causal_local(x_band, fs, fc)
% Strictly causal analytic pair for a narrowband signal: the
% quadrature component of a band-limited oscillation at centre frequency fc is that
% same signal delayed by a quarter period. The real part is undelayed, so the pair
% carries no net group delay and needs no compensation; train and test see the
% identical construction.
%
%
% Row layout matches make_band_feature_local('quadrature'): [real; imag].
d = max(1, round(double(fs) / (4 * double(fc))));
T = size(x_band, 2);
if d >= T
    x_q = zeros(size(x_band));
else
    x_q = [zeros(size(x_band, 1), d), x_band(:, 1:T-d)];
end
x_feature = [x_band; x_q];
end

function x_feature = arpad_feature_local(eeg_x, fs, bp_lo, bp_hi, p, W, H, is_oracle)
D = 10; fsd = fs / D; U = 2;
[zl, pl, kl] = butter(4, 5 / (fs / 2), 'low');
xd = causal_sos_filter_local(zp2sos(zl, pl, kl), eeg_x);   % forward-only anti-alias at fs
xd = xd(:, 1:D:end);                                       % xd(:, n) = low-passed EEG at raw sample 1 + D*(n-1)
[nch, Td] = size(xd); T = size(eeg_x, 2);
Wd = round(W * fsd); Hd = round(H * fsd);
[zb, pb, kb] = butter(4, [bp_lo bp_hi] / (fsd / 2), 'bandpass'); sosb = zp2sos(zb, pb, kb);
if ~is_oracle
    A = arpad_coeffs_local(xd, p, Wd, U);
end
% read position advanced by the anti-alias low-pass group delay at the band centre (tau, in decimated samples): the
% value for the current sample sits tau later in [window, forecast], which is still only past data + forecast
tau = grpdelay(zp2sos(zl, pl, kl), [0.5 * (bp_lo + bp_hi), 0.5 * (bp_lo + bp_hi) + 1e-3], fs);
tau = tau(1) / D;
nW = ceil(tau + U) + 1;
Wt = zeros(nW, D * U);                                     % linear interpolation weights, positions n+tau .. n+tau+U
for j = 0:D * U - 1
    f = tau + j / D; i0 = floor(f); a = f - i0;
    Wt(1 + i0, j + 1) = 1 - a; Wt(2 + i0, j + 1) = a;
end
out = complex(zeros(nch, T));
k = 0;
for n = Wd:U:Td
    k = k + 1;
    seg = xd(:, n - Wd + 1:n);
    if is_oracle
        fut = xd(:, n + 1:min(Td, n + Hd)); fut(:, end + 1:Hd) = 0;
    else
        fut = arpad_forecast_local(seg, A(:, :, k), Hd);
    end
    xa = hilbert(filtfilt(sosb, 1, [seg, fut]'))';
    r0 = 1 + D * (n - 1); r1 = min(T, r0 + D * U - 1);
    blk = xa(:, Wd:Wd + nW - 1) * Wt;
    out(:, r0:r1) = blk(:, 1:r1 - r0 + 1);
end
x_feature = [real(out); imag(out)];
end

function A = arpad_coeffs_local(xd, p, Wd, U)
% arburg AR(p) per channel on each past window; cached across the three harmonic-band calls of
% one session (key = size, p, window, update step, checksum of the data).
persistent key Ac
k_now = [size(xd), p, Wd, U, sum(xd(:, 1:min(end, 5000)), 'all'), sum(xd(:, max(1, end - 4999):end), 'all')];
if isequal(key, k_now)
    A = Ac; return;
end
[nch, Td] = size(xd); ns = numel(Wd:U:Td);
A = zeros(nch, p, ns);
k = 0;
for n = Wd:U:Td
    k = k + 1;
    seg = xd(:, n - Wd + 1:n);
    seg = seg - mean(seg, 2);
    for c = 1:nch
        a = arburg(seg(c, :), p);
        A(c, :, k) = a(2:end);
    end
end
key = k_now; Ac = A;
end

function fut = arpad_forecast_local(seg, a, Hd)
% AR recursion y[t] = -sum_i a_i y[t-i] per channel; a channel whose forecast exceeds 10x the
% window std (unstable fit) is padded with its window mean instead.
p = size(a, 2); mu = mean(seg, 2); y = seg - mu;
buf = y(:, end - p + 1:end);
fut = zeros(size(seg, 1), Hd);
ar = a(:, end:-1:1);                                       % ar(:, j) multiplies buf(:, j) (oldest first)
for h = 1:Hd
    nx = -sum(ar .* buf, 2);
    fut(:, h) = nx;
    buf = [buf(:, 2:end), nx];
end
bad = any(abs(fut) > 10 * std(y, 0, 2), 2) | any(~isfinite(fut), 2);
fut(bad, :) = 0;
fut = fut + mu;
end

function tf = is_relphase_feature_mode_local(feature_mode)
tf = any(strcmp(lower(char(feature_mode)), ...
    {'quadrature_relphase', 'quadrature_relative_phase', 'analytic_relphase'}));
end

function tf = is_harmonic_only_feature_mode_local(feature_mode)
tf = any(strcmp(lower(char(feature_mode)), ...
    {'quadrature_causal', 'quadrature_o2_causal', 'quadrature_wide_causal', 'echt_causal', 'gabor_causal', ...
     'quadrature_relphase', 'quadrature_relative_phase', 'analytic_relphase', ...
     'quadrature_unitphase', 'unitphase', 'analytic_unitphase', ...
     'quadrature_envelope', 'envelope', 'analytic_envelope', ...
     'quadrature_append_unitphase', 'analytic_append_unitphase', ...
     'quadrature_append_unitphase_envelope', 'analytic_append_unitphase_envelope', ...
     'quadrature_spatial_main', 'analytic_spatial_main', ...
     'quadrature_spatial_prior', 'analytic_spatial_prior', ...
     'quadrature_spatial_append_main', 'analytic_spatial_append_main', ...
     'quadrature_spatial_append_prior', 'analytic_spatial_append_prior'}));
end

function tf = is_spatial_prior_feature_mode_local(feature_mode)
tf = any(strcmp(lower(char(feature_mode)), ...
    {'quadrature_spatial_main', 'analytic_spatial_main', ...
     'quadrature_spatial_prior', 'analytic_spatial_prior'}));
end

function tf = is_spatial_append_feature_mode_local(feature_mode)
tf = any(strcmp(lower(char(feature_mode)), ...
    {'quadrature_spatial_append_main', 'analytic_spatial_append_main', ...
     'quadrature_spatial_append_prior', 'analytic_spatial_append_prior'}));
end

function tf = is_phaseextra_append_feature_mode_local(feature_mode)
tf = any(strcmp(lower(char(feature_mode)), ...
    {'quadrature_append_unitphase', 'analytic_append_unitphase', ...
     'quadrature_append_unitphase_envelope', 'analytic_append_unitphase_envelope'}));
end

function prior_mode = spatial_append_prior_mode_local(feature_mode)
mode = lower(char(feature_mode));
if any(strcmp(mode, {'quadrature_spatial_append_main', 'analytic_spatial_append_main'}))
    prior_mode = 'quadrature_spatial_main';
elseif any(strcmp(mode, {'quadrature_spatial_append_prior', 'analytic_spatial_append_prior'}))
    prior_mode = 'quadrature_spatial_prior';
else
    error('Not a spatial append feature mode: %s', char(feature_mode));
end
end

function labels = parse_channel_labels_local(value)
if isempty(value)
    labels = {};
    return;
end
if ischar(value) || isstring(value)
    parts = regexp(char(strjoin(string(value), ',')), '[,;\s]+', 'split');
elseif iscell(value)
    parts = cellfun(@char, value(:)', 'UniformOutput', false);
else
    error('bp_channel_labels must be a string or cellstr');
end
trimmed = cellfun(@strtrim, parts, 'UniformOutput', false);
keep = ~cellfun(@isempty, trimmed);
labels = trimmed(keep);
end

function [idx, labels] = resolve_channel_indices_local(raw_labels, requested_labels)
raw = cellstr(string(raw_labels(:)));
if isempty(requested_labels)
    idx = 1:numel(raw);
    labels = raw(:);
    return;
end
idx = nan(1, numel(requested_labels));
for ii = 1:numel(requested_labels)
    hit = find(strcmpi(raw, requested_labels{ii}), 1);
    if isempty(hit)
        error('Requested EEG channel not found: %s', requested_labels{ii});
    end
    idx(ii) = hit;
end
labels = raw(idx);
labels = labels(:);
end

function x_feature = make_band_feature_local(x_band, feature_mode)
switch lower(char(feature_mode))
    case {'real', 'bandpass'}
        x_feature = x_band;
    case {'quadrature', 'analytic'}
        xa = hilbert(x_band')';
        x_feature = [real(xa); imag(xa)];
    case {'quadrature_phaseperm', 'phase_randomized', 'phaseperm'}
        xa = hilbert(x_band')';
        x_feature = phase_permute_analytic_local(xa);
    case {'quadrature_unitphase', 'unitphase', 'analytic_unitphase'}
        xa = hilbert(x_band')';
        xa = xa ./ max(abs(xa), eps);
        x_feature = [real(xa); imag(xa)];
    case {'quadrature_envelope', 'envelope', 'analytic_envelope'}
        xa = hilbert(x_band')';
        x_feature = abs(xa);
    case {'quadrature_relphase', 'quadrature_relative_phase', 'analytic_relphase'}
        error('quadrature_relphase requires bp_band_harmonic_k_list [1 2 3]');
    case {'quadrature_spatial_main', 'analytic_spatial_main', ...
          'quadrature_spatial_prior', 'analytic_spatial_prior', ...
          'quadrature_spatial_append_main', 'analytic_spatial_append_main', ...
          'quadrature_spatial_append_prior', 'analytic_spatial_append_prior'}
        error('%s requires bp_band_harmonic_k_list so harmonic-specific spatial priors can be selected', char(feature_mode));
    case {'quadrature_append_unitphase', 'analytic_append_unitphase', ...
          'quadrature_append_unitphase_envelope', 'analytic_append_unitphase_envelope'}
        error('%s requires bp_band_harmonic_k_list so  quadrature rows can be followed by appended rows', char(feature_mode));
    otherwise
        error('bp_band_feature_mode: unsupported value %s', char(feature_mode));
end
end

function x_feature = make_phaseextra_append_feature_local(x_band, feature_mode)
xa = hilbert(x_band')';
unit = xa ./ max(abs(xa), eps);
switch lower(char(feature_mode))
    case {'quadrature_append_unitphase', 'analytic_append_unitphase'}
        x_feature = [real(unit); imag(unit)];
    case {'quadrature_append_unitphase_envelope', 'analytic_append_unitphase_envelope'}
        env = log(abs(xa) + eps);
        x_feature = [real(unit); imag(unit); env];
    otherwise
        error('Not a phase-extra append feature mode: %s', char(feature_mode));
end
end

function x_feature = make_spatial_prior_quadrature_feature_local(x_band, eeg_labels, harmonic_k, feature_mode)
xa = hilbert(x_band')';
[W, ~] = spatial_prior_weights_local(eeg_labels, harmonic_k, feature_mode);
z = W' * xa;
x_feature = [real(z); imag(z)];
end

function [W, mode_names] = spatial_prior_weights_local(eeg_labels, harmonic_k, feature_mode)
mode = lower(char(feature_mode));
if any(strcmp(mode, {'quadrature_spatial_main', 'analytic_spatial_main'}))
    if harmonic_k == 2
        mode_names = {'ap'};
    else
        mode_names = {'lr'};
    end
else
    if harmonic_k == 2
        mode_names = {'ap', 'lr'};
    else
        mode_names = {'lr', 'ap'};
    end
end
W = zeros(numel(eeg_labels), numel(mode_names));
for mi = 1:numel(mode_names)
    W(:, mi) = make_one_spatial_prior_weight_local(eeg_labels, mode_names{mi});
end
end

function w = make_one_spatial_prior_weight_local(eeg_labels, mode_name)
switch lower(char(mode_name))
    case 'lr'
        pos = {'FC2','FC4','C2','C4','C6','CP2','CP4','CP6','P4','P6','PO4','P8'};
        neg = {'FC1','FC3','C1','C3','C5','CP1','CP3','CP5','P3','P5','PO3','PO7','P7'};
    case 'ap'
        pos = {'TP7','TP8','P3','P4','P5','P6','P7','P8','PO3','PO4','PO7','PO8','O1','Oz','O2'};
        neg = {'AF3','AF4','F1','F2','F3','F4','Fz','FC1','FC2','FC3','FC4','FCz','C1','C2'};
    otherwise
        error('Unknown spatial prior mode: %s', char(mode_name));
end
w = signed_group_weight_local(eeg_labels, pos, neg);
end

function w = signed_group_weight_local(eeg_labels, pos_labels, neg_labels)
labels = string(eeg_labels(:));
w = zeros(numel(labels), 1);
pos_idx = label_hits_local(labels, pos_labels);
neg_idx = label_hits_local(labels, neg_labels);
if isempty(pos_idx) || isempty(neg_idx)
    error('Spatial prior needs both positive and negative channel groups; found pos=%d neg=%d', numel(pos_idx), numel(neg_idx));
end
w(pos_idx) = 1 / numel(pos_idx);
w(neg_idx) = -1 / numel(neg_idx);
scale = sum(abs(w));
if scale > 0
    w = w / scale;
end
end

function idx = label_hits_local(labels, requested)
idx = [];
for ii = 1:numel(requested)
    hit = find(strcmpi(labels, requested{ii}), 1);
    if ~isempty(hit)
        idx(end+1) = hit; %#ok<AGROW>
    end
end
idx = unique(idx, 'stable');
end

function x_rel = make_relative_phase_feature_local(xa_parts, harmonic_list)
i1 = find(double(harmonic_list) == 1, 1);
i2 = find(double(harmonic_list) == 2, 1);
i3 = find(double(harmonic_list) == 3, 1);
if isempty(i1) || isempty(i2) || isempty(i3)
    error('quadrature_relphase requires harmonic list [1 2 3]');
end
phi1 = angle(xa_parts{i1});
phi2 = angle(xa_parts{i2});
phi3 = angle(xa_parts{i3});
r21 = exp(1i * (phi2 - 2 * phi1));
r31 = exp(1i * (phi3 - 3 * phi1));
x_rel = [real(r21); imag(r21); real(r31); imag(r31)];
end

function x_feature = phase_permute_analytic_local(xa)
amp = abs(xa);
phi = angle(xa);
[n_ch, n_t] = size(xa);
x_feature = zeros(2 * n_ch, n_t);
old_state = rng;
cleanup = onCleanup(@() rng(old_state)); %#ok<NASGU>
for ch = 1:n_ch
    rng(20260531 + 1009 * ch + 37 * n_t, 'twister');
    perm = randperm(n_t);
    z = amp(ch, :) .* exp(1i * phi(ch, perm));
    x_feature(ch, :) = real(z);
    x_feature(n_ch + ch, :) = imag(z);
end
end

function x_aligned = euclidean_align_eeg_local(x)
x = double(x);
mu = mean(x, 2, 'omitnan');
mu(~isfinite(mu)) = 0;
xc = x - mu;
xc(~isfinite(xc)) = 0;
n = max(1, size(xc, 2) - 1);
C = (xc * xc') ./ n;
C = (C + C') ./ 2;
tr = trace(C) / max(1, size(C, 1));
if ~isfinite(tr) || tr <= eps
    x_aligned = xc;
    return;
end
shrink = 0.1;
C = (1 - shrink) * C + shrink * tr * eye(size(C));
[V, D] = eig(C);
d = real(diag(D));
d(~isfinite(d) | d < tr * 1e-6) = tr * 1e-6;
W = V * diag(1 ./ sqrt(d)) * V';
x_aligned = W * xc;
end

function ev = rest_trial_filter_local(events)
if isempty(events)
    ev = events;
    return;
end
mask = strcmp({events.code}, 'rest') & strcmp({events.qc}, 'good');
ev = events(mask);
end

% ---------------------------------------------------------------------------------------------------------------
% Rest-window export H3, read by the Python readout for the rest-input control.
% Active only when env CP_EXPORT_REST_DIR is non-empty AND the build is the rest-substituted one
% (input_control_mode = resting_random_window_within_trial, i.e. the second build of a resting_random_window_test_only
% run, or a within_trial run). Writes <label>_rest.mat (-v7.3): per ev trial the rest window the decoder actually uses,
% U_rest{t} = x_feat(:, a+min(lag) : a+Tk-1+max(lag)) with a = rest_start(t) (drawn by resting_random_anchor_local,
% seed input_control_seed + 1009*t), the IW trial's base_mu{t} (the decoder subtracts the IW trial's pre-trial mean
% from the rest window too), rest_ok(t) (false = no rest window long enough -> the decoder drops that test trial),
% et0 / Lk / Tk for cross-checks against the H1 export. Read-only: nothing the decoder computes changes.
% Self-check (error, nothing written, on any mismatch): every rest_ok trial's middle lag block (all lags of the first)
% isequaln X_cell{t} block == U_rest{t}(:, off+lag+(1:Tk)) - base_mu. env CP_EXPORT_REST_ONLY=1 skips decoding.
function d = cp_export_h3_dir_local(opt)
d = getenv('CP_EXPORT_REST_DIR');
if isempty(d) || ~isfield(opt, 'input_control_mode') || ...
        ~strcmpi(char(opt.input_control_mode), 'resting_random_window_within_trial')
    d = '';
end
end

function crx = cp_export_h3_init_local(S, ev, opt, T_total, lag_samples)
bmode = lower(char(opt.eeg_baseline_mode));
assert(any(strcmp(bmode, {'mean', 'none'})), 'CP_EXPORT_REST: eeg_baseline_mode must be mean or none (got %s)', bmode);
assert(all(lag_samples <= 0), 'CP_EXPORT_REST: needs all stage-1 lags <= 0');
n_ev = numel(ev);
crx = struct();
crx.session_label = char(S.label);
crx.n_ev = n_ev;
crx.U_rest = cell(1, n_ev);
crx.base_mu = cell(1, n_ev);
crx.rest_start = nan(n_ev, 1);
crx.rest_ok = false(n_ev, 1);
crx.et0 = nan(n_ev, 1); crx.Lk = nan(n_ev, 1); crx.Tk = zeros(n_ev, 1);
crx.input_control_mode = char(opt.input_control_mode);
crx.input_control_seed = double(opt.input_control_seed);
crx.seed_rule = 'rng(input_control_seed + 1009*t, twister) inside resting_random_anchor_local';
re = opt.rest_events;
if isempty(re)
    crx.rest_events_etstart = zeros(0, 1); crx.rest_events_etstop = zeros(0, 1);
else
    crx.rest_events_etstart = double([re.etstart])'; crx.rest_events_etstop = double([re.etstop])';
end
crx.T_total = T_total;
crx.lag_samples = double(lag_samples(:)');
crx.crop_offset = max(1, 1 - min(lag_samples)) - 1;
crx.eeg_baseline_mode = bmode;
crx.bp_band_feature_mode = lower(char(opt.bp_band_feature_mode));
crx.lag_ms = double(opt.lag_ms(:)');
crx.goni_qc_flags_env = getenv('GAIT_GONI_QC_TRIAL_FLAGS');
crx.gait_triple_root_env = getenv('GAIT_TRIPLE_ROOT');
crx.matlab_version = version;
crx.cp_path = [mfilename('fullpath') '.m'];
crx.cp_md5 = cp_export_md5_local(crx.cp_path);
end

function crx = cp_export_h3_trial_local(crx, t, x_feat, sample_t, lag_samples, et0, Lk, Tk, base_mu)
crx.et0(t) = et0; crx.Lk(t) = Lk; crx.Tk(t) = Tk;
crx.base_mu{t} = base_mu;
if isempty(sample_t)
    return;
end
a = sample_t(1);
assert(isequal(sample_t, a:(a + Tk - 1)), 'CP_EXPORT_REST: trial %d rest window not contiguous', t);
lo = a + min(lag_samples); hi = sample_t(end) + max(lag_samples);
assert(lo >= 1 && hi <= size(x_feat, 2), 'CP_EXPORT_REST: trial %d window [%d %d] outside 1..%d', t, lo, hi, size(x_feat, 2));
crx.U_rest{t} = x_feat(:, lo:hi);
crx.rest_start(t) = a;
crx.rest_ok(t) = true;
end

function cp_export_h3_write_local(xdir, crx, X_cell, lag_samples, nfeat)
off = -min(lag_samples);
L = numel(lag_samples);
ok = find(crx.rest_ok)';
n_xblk = 0;
for t = ok
    Tk = crx.Tk(t);
    assert(size(X_cell{t}, 2) == Tk, 'CP_EXPORT_REST: Tk mismatch trial %d', t);
    assert(size(crx.U_rest{t}, 2) == Tk + max(lag_samples) - min(lag_samples), 'CP_EXPORT_REST: window length trial %d', t);
    if t == ok(1), lis = 1:L; else, lis = ceil(L / 2); end
    for li = lis
        U = crx.U_rest{t}(:, off + lag_samples(li) + (1:Tk));
        if strcmp(crx.eeg_baseline_mode, 'mean'), U = U - crx.base_mu{t}; end
        assert(isequaln(X_cell{t}((li-1)*nfeat+1:li*nfeat, :), U), ...
            'CP_EXPORT_REST: X self-check FAIL trial %d lag %d', t, lag_samples(li));
        n_xblk = n_xblk + 1;
    end
end
for t = 1:crx.n_ev
    if ~crx.rest_ok(t)
        assert(size(X_cell{t}, 2) == 0, 'CP_EXPORT_REST: trial %d has X but no recorded rest window', t);
        crx.U_rest{t} = zeros(nfeat, 0);
    end
    if isempty(crx.base_mu{t}), crx.base_mu{t} = zeros(nfeat, 0); end
end
first_ok = NaN;
if ~isempty(ok), first_ok = ok(1); end
crx.self_check = sprintf('PASS: %d rest trials, X %d lag blocks (all %d lags of trial %g + middle lag of the others)', ...
    numel(ok), n_xblk, L, first_ok);
crx.export_created = datestr(now, 'yyyy-mm-dd HH:MM:SS');
if ~exist(xdir, 'dir'), mkdir(xdir); end
f = fullfile(xdir, sprintf('%s_rest.mat', crx.session_label));
if isfile(f)
    fprintf('[CP_EXPORT_REST] replacing existing %s\n', f);
    delete(f);
end
save(f, '-struct', 'crx', '-v7.3');
fprintf('[CP_EXPORT_REST] H3 self-check PASS label=%s n_ev=%d n_rest_ok=%d x_blocks=%d seed=%d file=%s\n', ...
    crx.session_label, crx.n_ev, numel(ok), n_xblk, crx.input_control_seed, f);
end

function skip = cp_export_h3_skip_local(label)
skip = any(strcmpi(getenv('CP_EXPORT_REST_ONLY'), {'1', 'true', 'yes', 'on'}));
if ~skip, return; end
xdir = getenv('CP_EXPORT_REST_DIR');
assert(~isempty(xdir), 'CP_EXPORT_REST_ONLY=1 requires CP_EXPORT_REST_DIR');
f = fullfile(xdir, sprintf('%s_rest.mat', char(label)));
assert(isfile(f), 'CP_EXPORT_REST_ONLY: %s not written (run must use INPUT_CONTROL_MODE=resting_random_window_test_only)', f);
fprintf('[CP_EXPORT_REST] CP_EXPORT_REST_ONLY=1: decoding of %s skipped\n', char(label));
end

function sample_t = resting_random_anchor_local(rest_events, n_cols, lag_samples, total_samples, seed)
sample_t = [];
if isempty(rest_events) || n_cols <= 0
    return;
end
min_lag = min(lag_samples);
max_lag = max(lag_samples);
valid = [];
for ri = 1:numel(rest_events)
    r0 = round(double(rest_events(ri).etstart));
    r1 = round(double(rest_events(ri).etstop));
    lo = max(1 - min_lag, r0 - min_lag);
    hi = min(total_samples - max_lag - n_cols + 1, r1 - max_lag - n_cols + 1);
    if hi >= lo
        valid(end+1).lo = lo; %#ok<AGROW>
        valid(end).hi = hi;
    end
end
if isempty(valid)
    return;
end
rng(seed, 'twister');
ri = randi(numel(valid));
start_t = valid(ri).lo + randi(valid(ri).hi - valid(ri).lo + 1) - 1;
sample_t = start_t:(start_t + n_cols - 1);
end

function sample_t = circular_shift_anchor_local(out_t, lag_samples, total_samples, seed)
sample_t = out_t;
n = numel(out_t);
if n <= 1
    return;
end
min_lag = min(lag_samples);
max_lag = max(lag_samples);
valid_lo = max(1 - min_lag, out_t(1));
valid_hi = min(total_samples - max_lag, out_t(end));
valid_t = valid_lo:valid_hi;
if numel(valid_t) ~= n
    return;
end
min_shift = min(max(abs(lag_samples)) + 1, max(1, n - 1));
if min_shift >= n
    min_shift = max(1, floor(n / 3));
end
rng(seed, 'twister');
choices = [min_shift:(n-1), 1:max(1, n-min_shift)];
choices = unique(mod(choices - 1, n) + 1);
shift = choices(randi(numel(choices)));
sample_t = valid_t(mod((0:n-1) + shift, n) + 1);
end

function [mu, sd] = feature_pretrial_baseline(x_feat, all_events, ev, fs)
nfeat = size(x_feat, 1);
n_base = round(fs);
etstart = round(double(ev.etstart));
if isfield(ev, 'eeg_pre1s_window') && all(isfinite(ev.eeg_pre1s_window))
    win = round(double(ev.eeg_pre1s_window));
    start_idx = max(1, win(1));
    stop_idx = min(size(x_feat, 2), win(2));
else
    prev_stop = 0;
    if ~isempty(all_events) && isfield(all_events, 'etstop')
        starts = [all_events.etstart];
        stops = [all_events.etstop];
        prev = find(starts < etstart & stops < etstart);
        if ~isempty(prev)
            prev_stop = max(stops(prev));
        end
    end
    start_idx = max([1, etstart - n_base, prev_stop + 1]);
    stop_idx = min(size(x_feat, 2), etstart - 1);
end
if stop_idx >= start_idx
    seg = x_feat(:, start_idx:stop_idx);
    mu = mean(seg, 2, 'omitnan');
    sd = std(seg, 0, 2, 'omitnan');
else
    mu = zeros(nfeat, 1);
    sd = ones(nfeat, 1);
end
mu(~isfinite(mu)) = 0;
sd(~isfinite(sd) | sd <= eps) = 1;
end

function [Y_z, mu, sd] = standardize_cells_local(Y_cell, scaler)
if isempty(scaler)
    Y_all = vertcat(Y_cell{:});
    mu = mean(Y_all, 1, 'omitnan');
    sd = std(Y_all, 0, 1, 'omitnan');
    sd(~isfinite(sd) | sd <= 0) = 1;
else
    mu = scaler.mu;
    sd = scaler.sd;
end
Y_z = cell(size(Y_cell));
for k = 1:numel(Y_cell)
    Y_z{k} = (Y_cell{k} - mu) ./ sd;
end
end

function Y_cell = inverse_standardize_cells(Y_z_cell, mu, sd)
Y_cell = cell(size(Y_z_cell));
for k = 1:numel(Y_z_cell)
    Y_cell{k} = Y_z_cell{k} .* sd + mu;
end
end

function M = fill_template_local(M)
if all(isnan(M)), return; end
x = (1:numel(M))';
good = isfinite(M);
if sum(good) == 1
    M(~good) = M(good);
else
    M(~good) = interp1(x(good), M(good), x(~good), 'linear', 'extrap');
end
end

function r = safe_corr_local(y, yp)
valid = isfinite(y) & isfinite(yp);
if sum(valid) < 10 || std(y(valid)) == 0 || std(yp(valid)) == 0
    r = NaN;
else
    r = corr(y(valid), yp(valid));
end
end
