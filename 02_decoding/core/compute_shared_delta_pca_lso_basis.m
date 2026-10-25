function compute_shared_delta_pca_lso_basis(varargin)
%COMPUTE_SHARED_DELTA_PCA_LSO_BASIS  Shared EEG-only PCA bases, one per target session.
%
% Computes one low-frequency EEG covariance per session, then, for each target
% session, saves the PCA basis of the summed covariances of the other sessions.
% No goniometer signal and no target value enters a basis.
%
% leave_subject_out decides which sessions are excluded from a target's source
% pool. With leave_subject_out = false only the target session itself is left out.
% With leave_subject_out = true every session of the same participant is left out,
% so a session's basis is estimated from OTHER PARTICIPANTS only
% (leave-one-participant-out). The published healthy bases were built with
% leave_subject_out = true (run/build_basis.sh iw and exec);
% running with the default false does not reproduce them. The patient basis is one
% basis pooled over all healthy sessions (global_only, write_global_basis;
% run/build_basis.sh hglobal). The participant identifier is the part of the
% session label before `_sess`.
%
% Parameters (name/value):
%   manifest_path            session list, column `session`
%   condition                'MI' (imagined walking) | 'EXEC' (real walking)
%   bp_lo, bp_hi             band-pass edges in Hz, used when no band csv is given
%   bp_band_csv              per-session harmonic bands written by
%                            estimate_goniometer_psd_peaks
%   bp_band_harmonic_k_list  harmonics read from that csv
%   bp_band_feature_mode     'real' | 'quadrature' | ... | a mode ending in '_causal',
%                            which builds each band with run_decoder's causal feature
%                            code (the published bases: 'cdm_o1_c0p18_causal')
%   bp_channel_labels        restrict the basis to these channels
%   eeg_baseline_mode        'none' | 'mean'
%   sample_stride            sample decimation while accumulating covariances
%   ea_align, ea_gamma       optional per-session channel whitening before pooling
%   aggregation_mode         how source covariances are pooled: sample_weighted |
%                            session_equal | trace_normalized | correlation
%   leave_subject_out        see above
%   parallel_workers         0 = serial
%   basis_dir                output folder
%   write_global_basis,      additionally write one basis pooled over all sessions
%   global_only, global_label
%
% Output: {label}_shared_delta_pca_lso.mat per session, holding basis, eigenvalues,
% explained, labels, used_sessions and a metadata struct recording every parameter
% above. run_decoder keeps the leading components of that basis.

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'shared'));

p = inputParser;
p.addParameter('out_dir', fullfile(fileparts(fileparts(mfilename('fullpath'))), 'result'));
p.addParameter('manifest_path', '');
p.addParameter('condition', 'MI');
p.addParameter('bp_lo', 0.1);
p.addParameter('bp_hi', 2.0);
p.addParameter('bp_band_csv', '');
p.addParameter('bp_band_harmonic_k', 0);
p.addParameter('bp_band_harmonic_k_list', []);
p.addParameter('bp_band_feature_mode', 'real');
p.addParameter('bp_channel_labels', {});
p.addParameter('eeg_baseline_mode', 'none'); % none | mean
p.addParameter('sample_stride', 5);
p.addParameter('ea_align', false);   % Euclidean Alignment: whiten per-session 60-ch covariance to identity before pooling
p.addParameter('ea_gamma', 0.1);     % shrinkage on the per-session channel covariance for stable R^-1/2
p.addParameter('aggregation_mode', 'sample_weighted'); % sample_weighted | session_equal | trace_normalized | correlation
p.addParameter('parallel_workers', 0);
p.addParameter('job_storage_dir', '');
p.addParameter('basis_dir', '');
p.addParameter('write_global_basis', false);
p.addParameter('global_only', false);
p.addParameter('global_label', 'healthy_global');
p.addParameter('leave_subject_out', false); % true => exclude same-subject sessions from each target's source pool
p.parse(varargin{:});

out_dir = char(p.Results.out_dir);
if isempty(p.Results.manifest_path)
    manifest_path = fullfile(out_dir, 'session_manifest.csv');
else
    manifest_path = char(p.Results.manifest_path);
end
if isempty(p.Results.basis_dir)
    basis_dir = fullfile(out_dir, 'shared_delta_pca_lso_bp01_2_stride5');
else
    basis_dir = char(p.Results.basis_dir);
end
if ~exist(basis_dir, 'dir'), mkdir(basis_dir); end

T = readtable(manifest_path, 'TextType', 'string', 'Delimiter', ',', ...
    'ReadVariableNames', true);
sessions = cellstr(T.session);
n_sess = numel(sessions);
leave_subject_out = logical(p.Results.leave_subject_out);
subj_id = regexprep(sessions, '_sess.*$', '');   % subject id per session, for leave-subject-out
aggregation_mode = lower(char(p.Results.aggregation_mode));
valid_modes = {'sample_weighted', 'session_equal', 'trace_normalized', 'correlation'};
assert(any(strcmp(aggregation_mode, valid_modes)), ...
    'aggregation_mode must be sample_weighted, session_equal, trace_normalized, or correlation');
harmonic_list = double(p.Results.bp_band_harmonic_k_list(:)');
if isempty(harmonic_list) && p.Results.bp_band_harmonic_k > 0
    harmonic_list = double(p.Results.bp_band_harmonic_k);
end
bp_band_tbl = table();
if ~isempty(p.Results.bp_band_csv)
    bp_band_tbl = readtable(p.Results.bp_band_csv, 'TextType', 'string');
    required_bp_cols = {'session', 'harmonic_k', 'band_lo', 'band_hi'};
    missing_bp_cols = setdiff(required_bp_cols, bp_band_tbl.Properties.VariableNames);
    assert(isempty(missing_bp_cols), 'bp_band_csv missing columns: %s', strjoin(missing_bp_cols, ','));
end
bp_channel_labels = parse_channel_labels_local(p.Results.bp_channel_labels);

C_sess = [];
n_samples = zeros(n_sess, 1);
labels = {};
raw_labels = {};
n_bands = max(1, numel(harmonic_list));
bp_used = nan(n_sess, n_bands, 2);
parallel_workers = max(0, floor(double(p.Results.parallel_workers)));
job_storage_dir = char(p.Results.job_storage_dir);
condition_name = upper(char(p.Results.condition));
default_bp = [p.Results.bp_lo p.Results.bp_hi];
feature_mode = char(p.Results.bp_band_feature_mode);
eeg_baseline_mode = lower(char(p.Results.eeg_baseline_mode));
assert(any(strcmp(eeg_baseline_mode, {'none', 'mean'})), ...
    'eeg_baseline_mode must be none or mean');
sample_stride = p.Results.sample_stride;
ea_align = logical(p.Results.ea_align);
ea_gamma = double(p.Results.ea_gamma);
global_only = logical(p.Results.global_only);

cov_results = cell(n_sess, 1);
if parallel_workers > 1
    ensure_parallel_pool_local(parallel_workers, job_storage_dir);
    q = parallel.pool.DataQueue;
    afterEach(q, @(msg) fprintf('%s\n', msg));
    parfor si = 1:n_sess
        [cov_results{si}, msg] = compute_session_covariance_local( ...
            si, n_sess, sessions, condition_name, bp_band_tbl, harmonic_list, ...
            default_bp, feature_mode, bp_channel_labels, sample_stride, eeg_baseline_mode, ea_align, ea_gamma);
        send(q, msg);
    end
else
    for si = 1:n_sess
        [cov_results{si}, msg] = compute_session_covariance_local( ...
            si, n_sess, sessions, condition_name, bp_band_tbl, harmonic_list, ...
            default_bp, feature_mode, bp_channel_labels, sample_stride, eeg_baseline_mode, ea_align, ea_gamma);
        fprintf('%s\n', msg);
    end
end

first_ok = find(cellfun(@(r) isstruct(r) && isfield(r, 'ok') && r.ok, cov_results), 1);
if isempty(first_ok)
    error('No usable sessions for PCA covariance');
end
labels = cov_results{first_ok}.labels;
raw_labels = cov_results{first_ok}.raw_labels;
nfeat = numel(labels);
C_sess = zeros(nfeat, nfeat, n_sess);
for si = 1:n_sess
    r = cov_results{si};
    if ~isstruct(r) || ~isfield(r, 'ok') || ~r.ok
        continue;
    end
    assert(isequal(string(raw_labels(:)), string(r.raw_labels(:))), ...
        'EEG labels mismatch for %s', r.label);
    assert(numel(labels) == size(r.C, 1), 'feature row mismatch for %s', r.label);
    C_sess(:, :, si) = r.C;
    n_samples(si) = r.n_samples;
    bp_used(si, 1:size(r.bp_session, 1), :) = r.bp_session;
end

C_total = sum(C_sess, 3);
n_total = sum(n_samples);
used_sessions = sessions(n_samples > 0);

if ~global_only
    for si = 1:n_sess
        label = sessions{si};
        source_mask = n_samples > 0;
        source_mask(si) = false;
        if leave_subject_out
            source_mask(strcmp(subj_id, subj_id{si})) = false;   % drop all same-subject sessions
        end
        source_idx = find(source_mask);
        C = aggregate_source_covariances_local(C_sess, n_samples, source_idx, aggregation_mode);
        n = sum(n_samples(source_idx));
        C = (C + C') / 2;
        [basis, D] = eig(C, 'vector');
        [eigenvalues, ord] = sort(D, 'descend');
        basis = basis(:, ord);
        explained = eigenvalues / sum(eigenvalues);

        metadata = struct();
        metadata.condition = upper(char(p.Results.condition));
        metadata.bp = [p.Results.bp_lo p.Results.bp_hi];
        metadata.bp_band_csv = char(p.Results.bp_band_csv);
        metadata.bp_band_harmonic_k = p.Results.bp_band_harmonic_k;
        metadata.bp_band_harmonic_k_list = harmonic_list;
        metadata.bp_band_feature_mode = char(p.Results.bp_band_feature_mode);
        metadata.bp_channel_labels = bp_channel_labels;
        metadata.eeg_baseline_mode = eeg_baseline_mode;
        metadata.bp_used_by_session = bp_used;
        metadata.sample_stride = p.Results.sample_stride;
        metadata.ea_align = ea_align;
        metadata.ea_gamma = ea_gamma;
        metadata.aggregation_mode = aggregation_mode;
        metadata.parallel_workers = parallel_workers;
        metadata.target_session_excluded = label;
        metadata.n_total_samples = n;
        metadata.n_sessions_used = numel(source_idx);
        metadata.note = 'Shared EEG-only PCA basis from low-frequency imagined-walking samples. The target session is excluded; with leave_subject_out every session of the same participant is excluded. No goniometer target is used.';

        out_file = fullfile(basis_dir, sprintf('%s_shared_delta_pca_lso.mat', label));
        save(out_file, 'basis', 'eigenvalues', 'explained', 'labels', 'used_sessions', 'metadata', '-v7.3');
        fprintf('[basis %d/%d] wrote %s (n_sessions_used=%d)\n', ...
            si, n_sess, out_file, metadata.n_sessions_used);
    end
else
    fprintf('[basis] global_only=true; skipping leave-session-out basis writes\n');
end

if p.Results.write_global_basis
    source_idx = find(n_samples > 0);
    C = aggregate_source_covariances_local(C_sess, n_samples, source_idx, aggregation_mode);
    n = sum(n_samples(source_idx));
    C = (C + C') / 2;
    [basis, D] = eig(C, 'vector');
    [eigenvalues, ord] = sort(D, 'descend');
    basis = basis(:, ord);
    explained = eigenvalues / sum(eigenvalues);

    metadata = struct();
    metadata.condition = upper(char(p.Results.condition));
    metadata.bp = [p.Results.bp_lo p.Results.bp_hi];
    metadata.bp_band_csv = char(p.Results.bp_band_csv);
    metadata.bp_band_harmonic_k = p.Results.bp_band_harmonic_k;
    metadata.bp_band_harmonic_k_list = harmonic_list;
    metadata.bp_band_feature_mode = char(p.Results.bp_band_feature_mode);
    metadata.bp_channel_labels = bp_channel_labels;
    metadata.eeg_baseline_mode = eeg_baseline_mode;
    metadata.bp_used_by_session = bp_used;
    metadata.sample_stride = p.Results.sample_stride;
    metadata.aggregation_mode = aggregation_mode;
    metadata.parallel_workers = parallel_workers;
    metadata.target_session_excluded = '';
    metadata.n_total_samples = n;
    metadata.n_sessions_used = numel(source_idx);
    metadata.note = 'Global EEG-only shared PCA basis from all listed healthy MI low-delta samples; no patient data or goni target used.';

    used_sessions = sessions(source_idx);
    global_label = char(p.Results.global_label);
    out_file = fullfile(basis_dir, sprintf('%s_shared_delta_pca.mat', global_label));
    save(out_file, 'basis', 'eigenvalues', 'explained', 'labels', 'used_sessions', 'metadata', '-v7.3');
    fprintf('[global basis] wrote %s (n_sessions_used=%d)\n', out_file, metadata.n_sessions_used);
end

summary_file = fullfile(basis_dir, 'summary.mat');
save(summary_file, 'sessions', 'n_samples', 'used_sessions', 'labels', 'bp_used', 'aggregation_mode', 'eeg_baseline_mode', 'parallel_workers', '-v7.3');
if global_only
    fprintf('Done: wrote global basis only to %s\n', basis_dir);
else
    fprintf('Done: wrote %d leave-session-out bases to %s\n', n_sess, basis_dir);
end
end

function ensure_parallel_pool_local(parallel_workers, job_storage_dir)
pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= parallel_workers
    if ~isempty(pool)
        delete(pool);
    end
    cluster = parcluster('local');
    if ~isempty(job_storage_dir)
        if ~exist(job_storage_dir, 'dir'), mkdir(job_storage_dir); end
        cluster.JobStorageLocation = job_storage_dir;
    end
    parpool(cluster, parallel_workers);
end
end

function [result, msg] = compute_session_covariance_local(si, n_sess, sessions, condition_name, bp_band_tbl, harmonic_list, default_bp, feature_mode, bp_channel_labels, sample_stride, eeg_baseline_mode, ea_align, ea_gamma)
if nargin < 12 || isempty(ea_align), ea_align = false; end
if nargin < 13 || isempty(ea_gamma), ea_gamma = 0.1; end
label = sessions{si};
result = struct('ok', false, 'label', label, 'C', [], 'n_samples', 0, ...
    'labels', {{}}, 'raw_labels', {{}}, 'bp_session', []);
try
    S = load_triple_mat(label);
    ev = trial_filter(S.events, condition_name);
    if numel(ev) < 5
        msg = sprintf('[cov %d/%d] %s skip (%d trials)', si, n_sess, label, numel(ev));
        return;
    end
    [ch_idx, session_labels] = resolve_channel_indices_local(S.eeg.labels, bp_channel_labels);
    fs = S.eeg.fs;
    [x, bp_session, feature_labels] = build_session_band_feature_local( ...
        S, bp_band_tbl, label, harmonic_list, default_bp, ...
        fs, feature_mode, ch_idx, session_labels);
    X = collect_event_feature_samples_local(x, S.events, ev, fs, sample_stride, eeg_baseline_mode);
    X = X - mean(X, 2, 'omitnan');
    X(~isfinite(X)) = 0;
    if ea_align
        X = ea_whiten_channel_blocks_local(X, numel(session_labels), ea_gamma);
    end
    result.ok = true;
    result.C = X * X';
    result.n_samples = size(X, 2);
    result.labels = feature_labels;
    result.raw_labels = session_labels;
    result.bp_session = bp_session;
    msg = sprintf('[cov %d/%d] %s bands=%d samples=%d', ...
        si, n_sess, label, size(bp_session, 1), result.n_samples);
catch ME
    msg = sprintf('[cov %d/%d] %s FAIL: %s', si, n_sess, label, ME.message);
end
end

function X = collect_event_feature_samples_local(x, all_events, ev, fs, sample_stride, eeg_baseline_mode)
parts = cell(1, numel(ev));
for ei = 1:numel(ev)
    idx = ev(ei).etstart:sample_stride:ev(ei).etstop;
    idx = idx(idx >= 1 & idx <= size(x, 2));
    Xseg = x(:, idx);
    if strcmp(eeg_baseline_mode, 'mean')
        base_mu = feature_pretrial_baseline_mean_local(x, all_events, ev(ei), fs);
        Xseg = Xseg - base_mu;
    end
    parts{ei} = Xseg;
end
if isempty(parts)
    X = zeros(size(x, 1), 0);
else
    X = [parts{:}];
end
end

function mu = feature_pretrial_baseline_mean_local(x_feat, all_events, ev, fs)
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
    mu = mean(x_feat(:, start_idx:stop_idx), 2, 'omitnan');
else
    mu = zeros(nfeat, 1);
end
mu(~isfinite(mu)) = 0;
end

function C = aggregate_source_covariances_local(C_sess, n_samples, source_idx, aggregation_mode)
if isempty(source_idx)
    error('No source sessions available for shared PCA basis');
end
switch aggregation_mode
    case 'sample_weighted'
        C = sum(C_sess(:, :, source_idx), 3) / max(sum(n_samples(source_idx)) - 1, 1);
    case 'session_equal'
        C = zeros(size(C_sess, 1), size(C_sess, 2));
        for ii = 1:numel(source_idx)
            si = source_idx(ii);
            C = C + session_covariance_from_sum_local(C_sess(:, :, si), n_samples(si));
        end
        C = C / numel(source_idx);
    case 'trace_normalized'
        C = zeros(size(C_sess, 1), size(C_sess, 2));
        for ii = 1:numel(source_idx)
            si = source_idx(ii);
            Ci = session_covariance_from_sum_local(C_sess(:, :, si), n_samples(si));
            tr = trace(Ci);
            if isfinite(tr) && tr > eps
                Ci = Ci / tr;
            end
            C = C + Ci;
        end
        C = C / numel(source_idx);
    case 'correlation'
        C = zeros(size(C_sess, 1), size(C_sess, 2));
        for ii = 1:numel(source_idx)
            si = source_idx(ii);
            Ci = session_covariance_from_sum_local(C_sess(:, :, si), n_samples(si));
            sd = sqrt(max(diag(Ci), eps));
            Ci = Ci ./ (sd * sd');
            Ci(~isfinite(Ci)) = 0;
            C = C + Ci;
        end
        C = C / numel(source_idx);
    otherwise
        error('Unknown aggregation_mode: %s', aggregation_mode);
end
end

function C = session_covariance_from_sum_local(C_sum, n)
C = C_sum / max(double(n) - 1, 1);
C = (C + C') / 2;
C(~isfinite(C)) = 0;
end

function Xout = ea_whiten_channel_blocks_local(X, n_ch, gamma)
% Euclidean Alignment: whiten the per-session n_ch x n_ch channel covariance
% (pooled across the harmonic/quadrature blocks) to identity, applying the same
% R^-1/2 to every n_ch-channel block so the harmonic/quad block structure is
% preserved. Unsupervised (no target). gamma shrinks the channel covariance.
Xout = X;
nfeat = size(X, 1);
if n_ch <= 0 || mod(nfeat, n_ch) ~= 0
    return;
end
nb = nfeat / n_ch;
R = zeros(n_ch, n_ch);
ncols = max(size(X, 2) - 1, 1);
for b = 1:nb
    Xb = X((b - 1) * n_ch + (1:n_ch), :);
    R = R + (Xb * Xb');
end
R = R / (nb * ncols);
R = (R + R') / 2;
if ~all(isfinite(R(:)))
    return;
end
g = min(max(gamma, 0), 1);
R = (1 - g) * R + g * (trace(R) / n_ch) * eye(n_ch);
W = ea_inv_sqrt_local(R);
for b = 1:nb
    rows = (b - 1) * n_ch + (1:n_ch);
    Xout(rows, :) = W * X(rows, :);
end
Xout(~isfinite(Xout)) = 0;
end

function W = ea_inv_sqrt_local(R)
[V, D] = eig((R + R') / 2, 'vector');
d = real(D);
d(~isfinite(d) | d < eps) = eps;
W = V * diag(d .^ (-0.5)) * V';
W = real((W + W') / 2);
end

function [x, bp_session, feature_labels] = build_session_band_feature_local(S, bp_band_tbl, label, harmonic_list, default_bp, fs, feature_mode, ch_idx, eeg_labels)
eeg_x = double(S.eeg.x(ch_idx, :));
if isempty(harmonic_list) || isempty(bp_band_tbl)
    if is_harmonic_only_feature_mode_local(feature_mode)
        error('%s requires bp_band_harmonic_k_list [1 2 3]', char(feature_mode));
    end
    bp_session = default_bp;
    [z, pp, k] = butter(4, default_bp / (fs / 2), 'bandpass');
    x_band = filtfilt(zp2sos(z, pp, k), 1, eeg_x')';
    x = make_band_feature_local(x_band, feature_mode);
    feature_labels = make_band_feature_labels_local(eeg_labels, 0, feature_mode);
    return;
end
x_parts = cell(1, numel(harmonic_list));
xa_parts = cell(1, numel(harmonic_list));
feature_labels = {};
bp_session = nan(numel(harmonic_list), 2);
relphase_mode = is_relphase_feature_mode_local(feature_mode);
for bi = 1:numel(harmonic_list)
    [bp_lo, bp_hi] = lookup_session_bp_band_local(bp_band_tbl, label, harmonic_list(bi));
    bp_session(bi, :) = [bp_lo bp_hi];
    if is_causal_feature_mode_local(feature_mode)
        if strcmp(lower(char(feature_mode)), 'quadrature_causal')
            [z, pp, k] = butter(4, [bp_lo bp_hi] / (fs / 2), 'bandpass');
            x_parts{bi} = make_band_feature_causal_local(causal_sos_filter_local(zp2sos(z, pp, k), eeg_x), fs, 0.5 * (bp_lo + bp_hi));
        else
            x_parts{bi} = causal_variant_feature_local(lower(char(feature_mode)), eeg_x, fs, bp_lo, bp_hi);
        end
        feature_labels = [feature_labels; make_band_feature_labels_local(eeg_labels, harmonic_list(bi), 'quadrature')]; %#ok<AGROW>
        continue;
    end
    [z, pp, k] = butter(4, [bp_lo bp_hi] / (fs / 2), 'bandpass');
    x_band = filtfilt(zp2sos(z, pp, k), 1, eeg_x')';
    if relphase_mode
        xa_parts{bi} = hilbert(x_band')';
        x_parts{bi} = [real(xa_parts{bi}); imag(xa_parts{bi})];
        feature_labels = [feature_labels; make_band_feature_labels_local(eeg_labels, harmonic_list(bi), 'quadrature')]; %#ok<AGROW>
    else
        x_parts{bi} = make_band_feature_local(x_band, feature_mode);
        feature_labels = [feature_labels; make_band_feature_labels_local(eeg_labels, harmonic_list(bi), feature_mode)]; %#ok<AGROW>
    end
end
x = vertcat(x_parts{:});
if relphase_mode
    x = [x; make_relative_phase_feature_local(xa_parts, harmonic_list)];
    feature_labels = [feature_labels; make_relative_phase_feature_labels_local(eeg_labels)];
end
end

function tf = is_relphase_feature_mode_local(feature_mode)
tf = any(strcmp(lower(char(feature_mode)), ...
    {'quadrature_relphase', 'quadrature_relative_phase', 'analytic_relphase'}));
end

function tf = is_harmonic_only_feature_mode_local(feature_mode)
tf = any(strcmp(lower(char(feature_mode)), ...
    {'quadrature_relphase', 'quadrature_relative_phase', 'analytic_relphase', ...
     'quadrature_unitphase', 'unitphase', 'analytic_unitphase', ...
     'quadrature_envelope', 'envelope', 'analytic_envelope'}));
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

function labels = make_relative_phase_feature_labels_local(raw_labels)
base = string(raw_labels(:));
labels = [cellstr(base + "_rel_k2_2k1_cos"); ...
          cellstr(base + "_rel_k2_2k1_sin"); ...
          cellstr(base + "_rel_k3_3k1_cos"); ...
          cellstr(base + "_rel_k3_3k1_sin")];
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
    otherwise
        error('bp_band_feature_mode must be real, quadrature, quadrature_phaseperm, quadrature_unitphase, quadrature_envelope, or quadrature_relphase, got %s', char(feature_mode));
end
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

function feature_labels = make_band_feature_labels_local(raw_labels, harmonic_k, feature_mode)
if harmonic_k > 0
    base = string(raw_labels(:)) + sprintf('_k%d', harmonic_k);
else
    base = string(raw_labels(:));
end
switch lower(char(feature_mode))
    case {'real', 'bandpass'}
        feature_labels = cellstr(base);
    case {'quadrature', 'analytic'}
        feature_labels = [cellstr(base + "_real"); cellstr(base + "_imag")];
    case {'quadrature_phaseperm', 'phase_randomized', 'phaseperm'}
        feature_labels = [cellstr(base + "_phaseperm_real"); cellstr(base + "_phaseperm_imag")];
    case {'quadrature_unitphase', 'unitphase', 'analytic_unitphase'}
        feature_labels = [cellstr(base + "_unitphase_real"); cellstr(base + "_unitphase_imag")];
    case {'quadrature_envelope', 'envelope', 'analytic_envelope'}
        feature_labels = cellstr(base + "_envelope");
    case {'quadrature_relphase', 'quadrature_relative_phase', 'analytic_relphase'}
        error('quadrature_relphase labels require bp_band_harmonic_k_list [1 2 3]');
    otherwise
        error('bp_band_feature_mode must be real, quadrature, quadrature_phaseperm, quadrature_unitphase, quadrature_envelope, or quadrature_relphase, got %s', char(feature_mode));
end
end

function [bp_lo, bp_hi] = lookup_session_bp_band_local(bp_band_tbl, label, harmonic_k)
mask = string(bp_band_tbl.session) == string(label);
if harmonic_k > 0
    mask = mask & double(bp_band_tbl.harmonic_k) == double(harmonic_k);
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

