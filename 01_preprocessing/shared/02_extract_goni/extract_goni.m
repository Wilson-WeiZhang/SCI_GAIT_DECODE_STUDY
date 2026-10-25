function out = extract_goni(step1_set_path, out_dir, opts)
%EXTRACT_GONI  Cut per-trial goniometer epochs at 1000 Hz and give each trial an initial QC verdict.
%
%   Reads the alignment stored in step1.set (EEG.etc.align_log), reloads the raw
%   goniometer and slices every trial at its native sample rate.
%
%   Output {out_dir}/{label}_goni.mat:
%     align_info   {1x1 cell of struct}  method/verdict/n_matched/match_frac/n_segs/...
%     goni_labels  {1 x nCh cell}
%     goni_srate   1000
%     label        e.g. 'SUB_07_sess01'
%     n_imagine, n_walk, n_rest
%     n_problems   number of trials with status 'problem'
%     n_segments   number of clock segments in the alignment
%     qc           [1 x n_trials] struct: trial_idx, type, status, reason, active_std, baseline
%     trials       [1 x n_trials] struct: type, segment, start_eeg_sec, end_eeg_sec,
%                  dur_sec, goni_start_idx, goni_end_idx, joint_angles ([T x nCh]), goni_labels
%
%   Initial QC, per session and condition (imagine, walk; rest is 'skip'):
%     K = 3 x 1.4826; effective MAD = max(MAD, 5% of the median)
%     'problem' if trial length is outside median +/- K*MAD, or if any active channel's
%     standard deviation is below max(median - K*MAD, 1 degree).
%   The verdict used for analysis is recomputed by 04_qc/recompute_goni_qc.m.

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'dur_min_sec'), opts.dur_min_sec = 3; end
    if ~isfield(opts, 'dur_max_sec'), opts.dur_max_sec = 60; end

    K = 3 * 1.4826;

    EEG = pop_loadset('filename', step1_set_path);
    [~, label, ~] = fileparts(step1_set_path);
    label = strrep(label, '_step1', '');
    if ~isfield(EEG.etc, 'align_log')
        error('extract_goni: %s has no EEG.etc.align_log', step1_set_path);
    end
    AL = EEG.etc.align_log;

    is_multi_goni = isfield(AL, 'special_case_type') && strcmp(AL.special_case_type, 'multi_goni_file');

    % ---- Locate raw goni ----
    if is_multi_goni
        if isfield(opts, 'raw_goni_paths') && ~isempty(opts.raw_goni_paths)
            raw_goni_paths = opts.raw_goni_paths;
        elseif isfield(EEG.etc, 'raw_goni_files') && ~isempty(EEG.etc.raw_goni_files)
            raw_goni_paths = EEG.etc.raw_goni_files;
        else
            error('extract_goni: multi_goni_file requires opts.raw_goni_paths or EEG.etc.raw_goni_files');
        end
        [goni_d, goni_labels, goni_srate, edges_samples] = load_multi_goni_files(raw_goni_paths, AL);
    elseif isfield(opts, 'raw_goni_path') && ~isempty(opts.raw_goni_path)
        raw_goni_path = opts.raw_goni_path;
        G = load_goniometer(raw_goni_path);
        goni_d      = G.data;             % [N x nCh]
        goni_labels = G.labels;
        goni_srate  = G.srate;
        edges_samples = AL.goni_edges_sec_1kHz * 1000;     % goni edge sample @ 1000 Hz
        if goni_srate ~= 1000
            warning('extract_goni: goni srate=%d, expected 1000', goni_srate);
        end
    elseif isfield(EEG.etc, 'raw_goni_file') && ~isempty(EEG.etc.raw_goni_file)
        raw_goni_path = EEG.etc.raw_goni_file;
        G = load_goniometer(raw_goni_path);
        goni_d      = G.data;             % [N x nCh]
        goni_labels = G.labels;
        goni_srate  = G.srate;
        edges_samples = AL.goni_edges_sec_1kHz * 1000;     % goni edge sample @ 1000 Hz
        if goni_srate ~= 1000
            warning('extract_goni: goni srate=%d, expected 1000', goni_srate);
        end
    elseif isfield(EEG.etc, 'goni') && isfield(EEG.etc.goni, 'goni_file')
        raw_goni_path = EEG.etc.goni.goni_file;
        G = load_goniometer(raw_goni_path);
        goni_d      = G.data;             % [N x nCh]
        goni_labels = G.labels;
        goni_srate  = G.srate;
        edges_samples = AL.goni_edges_sec_1kHz * 1000;     % goni edge sample @ 1000 Hz
        if goni_srate ~= 1000
            warning('extract_goni: goni srate=%d, expected 1000', goni_srate);
        end
    else
        error('extract_goni: cannot determine raw goni path (pass opts.raw_goni_path or set EEG.etc.raw_goni_file)');
    end

    % ---- Channel index resolution ----
    W_X = {'LHipW X','LKneW X','LAnkW X','RHipW X','RKneW X','RAnkW X'};
    S_X = {'LHipS X','LKneS X','LAnkS X','RHipS X','RKneS X','RAnkS X'};
    [w_idx, w_names] = resolve_channels(goni_labels, W_X);
    [s_idx, s_names] = resolve_channels(goni_labels, S_X);

    % ---- Alignment & event-level goni_idx ----
    offset_samp   = AL.offset_samples_1kHz;            % goni_t = eeg_t + offset (samples @ 1000 Hz)
    eeg_srate     = EEG.srate;                         % e.g. 250 after step1D resample
    n_events      = length(EEG.event);
    goni_idx_arr  = nan(1, n_events);
    for k = 1:n_events
        if isfield(EEG.event(k), 'goni_idx') && ~isempty(EEG.event(k).goni_idx)
            goni_idx_arr(k) = EEG.event(k).goni_idx;
        end
    end
    types = {EEG.event.type};
    evt_lats = [EEG.event.latency];

    % ---- Pair trials chronologically ----
    cond_specs = extract_condition_specs(label);

    trials = struct('type',{},'segment',{},'start_eeg_sec',{},'end_eeg_sec',{}, ...
                    'dur_sec',{},'goni_start_idx',{},'goni_end_idx',{}, ...
                    'joint_angles',{},'goni_labels',{});
    qc_arr = struct('trial_idx',{},'type',{},'status',{},'reason',{}, ...
                    'active_std',{},'baseline',{});
    cond_buf = struct( ...
        'imagine', struct('per_ch_std', [], 'n_pts', [], 'idx', [], 'names', {w_names}), ...
        'walk',    struct('per_ch_std', [], 'n_pts', [], 'idx', [], 'names', {s_names}));
    trial_eeg_event_idx = [];   % parallel to trials[]: trial j's onset event index in EEG.event

    if isfield(AL, 'drift_segments')
        drift_segments = AL.drift_segments;
    else
        drift_segments = struct('start_eeg_sec', -Inf, 'end_eeg_sec', Inf, 'offset_sec', NaN);
    end
    n_seg = length(drift_segments);

    for ci = 1:size(cond_specs, 1)
        on_t = cond_specs{ci, 1}; off_t = cond_specs{ci, 2};
        cond = cond_specs{ci, 3}; act = cond_specs{ci, 4};
        on_e  = find(strcmp(types, on_t));
        off_e = find(strcmp(types, off_t));
        pairs = pair_chrono(on_e, off_e);
        r1_mode = cond_specs{ci, 5};
        if ~isempty(r1_mode)
            pairs = filter_pairs_by_r1(pairs, r1_mode, types, evt_lats, EEG.srate);
        end

        switch act
            case 'W', ach = w_idx;
            case 'S', ach = s_idx;
            otherwise, ach = [];
        end

        for p = 1:size(pairs, 1)
            k_on = pairs(p, 1); k_off = pairs(p, 2);
            t_on  = (EEG.event(k_on).latency  - 1) / eeg_srate;
            t_off = (EEG.event(k_off).latency - 1) / eeg_srate;
            dur   = t_off - t_on;
            if dur < opts.dur_min_sec || dur > opts.dur_max_sec, continue; end

            % Convert eeg time to goni-clock samples @ 1000 Hz for fallback
            eeg_samp_1k_on  = t_on  * 1000 + 1;
            eeg_samp_1k_off = t_off * 1000 + 1;
            g_start = goni_sample(k_on,  goni_idx_arr, edges_samples, eeg_samp_1k_on,  offset_samp);
            g_end   = goni_sample(k_off, goni_idx_arr, edges_samples, eeg_samp_1k_off, offset_samp);
            if isnan(g_start) || isnan(g_end), continue; end
            g_start = max(1, round(g_start));
            g_end   = min(size(goni_d, 1), round(g_end));
            if g_end <= g_start, continue; end

            joint = goni_d(g_start:g_end, :);   % [T x nCh] @ 1000 Hz

            seg_id = which_segment(t_on, drift_segments);   % primary segment by trial onset

            tr = struct('type', cond, 'segment', seg_id, ...
                'start_eeg_sec', t_on, 'end_eeg_sec', t_off, 'dur_sec', dur, ...
                'goni_start_idx', g_start, 'goni_end_idx', g_end, ...
                'joint_angles',   joint, 'goni_labels', {goni_labels});
            trials(end+1) = tr; %#ok<AGROW>
            tidx = length(trials);

            if isempty(ach)
                per_ch_std = [];
                active_std = NaN;
            else
                per_ch_std = std(double(joint(:, ach)), 0, 1, 'omitnan');
                active_std = mean(per_ch_std, 'omitnan');
            end
            qc_arr(end+1) = struct('trial_idx', tidx, 'type', cond, ...
                'status', '', 'reason', '', 'active_std', active_std, 'baseline', NaN); %#ok<AGROW>

            if ismember(cond, {'imagine', 'walk'})
                cond_buf.(cond).per_ch_std(end+1, :) = per_ch_std;
                cond_buf.(cond).n_pts(end+1) = size(joint, 1);
                cond_buf.(cond).idx(end+1)   = tidx;
            end

            % Record EEG event index for trial-onset → use later for event_to_trial
            trial_eeg_event_idx(tidx) = k_on; %#ok<AGROW>
        end
    end

    % ---- Apply the initial MAD verdict per condition ----
    n_problems = 0;
    for cond_c = {'imagine', 'walk'}
        cond = cond_c{1};
        bd = cond_buf.(cond);
        if isempty(bd.idx), continue; end
        n_pts_thr = mad_thr(bd.n_pts, K);
        nCh = size(bd.per_ch_std, 2);
        std_lo = nan(1, nCh);
        for ji = 1:nCh
            v = bd.per_ch_std(:, ji); v = v(~isnan(v));
            if ~isempty(v), std_lo(ji) = mad_thr(v, K).lo; end
        end
        baseline_active = median(arrayfun(@(i) qc_arr(i).active_std, bd.idx), 'omitnan');
        for ii = 1:length(bd.idx)
            ti = bd.idx(ii);
            qc_arr(ti).baseline = baseline_active;
            reasons = {};
            if bd.n_pts(ii) < n_pts_thr.lo || bd.n_pts(ii) > n_pts_thr.hi
                reasons{end+1} = sprintf('length:n=%d not in [%.0f,%.0f]', ...
                    bd.n_pts(ii), n_pts_thr.lo, n_pts_thr.hi); %#ok<AGROW>
            end
            for ji = 1:nCh
                if ~isnan(bd.per_ch_std(ii, ji)) && ~isnan(std_lo(ji)) && ...
                   bd.per_ch_std(ii, ji) < std_lo(ji)
                    reasons{end+1} = sprintf('%s std:%.2f<%.2f', ...
                        bd.names{ji}, bd.per_ch_std(ii, ji), std_lo(ji)); %#ok<AGROW>
                end
            end
            if isempty(reasons)
                qc_arr(ti).status = 'ok';
            else
                qc_arr(ti).status = 'problem';
                qc_arr(ti).reason = strjoin(reasons, '; ');
                n_problems = n_problems + 1;
            end
        end
    end
    for i = 1:length(qc_arr)
        if strcmp(qc_arr(i).type, 'rest')
            qc_arr(i).status = 'skip';
        end
    end

    if isfield(AL.residual_ms, 'median'), mean_res = AL.residual_ms.median; else, mean_res = NaN; end
    if isfield(AL.residual_ms, 'max'),    max_res  = AL.residual_ms.max;    else, max_res  = NaN; end
    align_info = {struct( ...
        'method',           AL.match_method, ...
        'verdict',          AL.auto_verdict, ...
        'n_matched',        AL.n_matched, ...
        'match_frac',       AL.n_matched / max(AL.n_eeg_events_ttl, 1), ...
        'n_segs',           n_seg, ...
        'mean_residual_ms', mean_res, ...
        'max_residual_ms',  max_res, ...
        'n_bad',            AL.n_eeg_events_ttl - AL.n_matched)};

    n_imagine = sum(strcmp({trials.type}, 'imagine'));
    n_walk    = sum(strcmp({trials.type}, 'walk'));
    n_rest    = sum(strcmp({trials.type}, 'rest'));

    % ---- Build event_to_trial[1×n_eeg_events] ----
    %   For each EEG.event(k):
    %     event_to_trial(k) = j ∈ {1..n_trials}  if event k is a trial-onset and trial j was extracted
    %     event_to_trial(k) = NaN                 otherwise
    event_to_trial = nan(1, n_events);
    for j = 1:length(trials)
        k = trial_eeg_event_idx(j);
        event_to_trial(k) = j;
    end

    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    out_path = fullfile(out_dir, [label '_goni.mat']);

    out_pkg = struct( ...
        'align_info',     {align_info}, ...
        'goni_labels',    {goni_labels}, ...
        'goni_srate',     goni_srate, ...
        'label',          label, ...
        'n_imagine',      n_imagine, ...
        'n_walk',         n_walk, ...
        'n_rest',         n_rest, ...
        'n_problems',     n_problems, ...
        'n_segments',     n_seg, ...
        'qc',             qc_arr, ...
        'qc_method',      'mad_initial', ...
        'trials',         trials, ...
        'event_to_trial', event_to_trial);
    save(out_path, '-struct', 'out_pkg', '-v7.3');
    fprintf('  saved %s | n=%d (I=%d W=%d R=%d) problems=%d\n', ...
        out_path, length(trials), n_imagine, n_walk, n_rest, n_problems);

    out = out_pkg;
    out.out_path = out_path;
    out.skipped  = false;
end


function pairs = pair_chrono(on_e, off_e)
    pairs = pair_events_chrono(on_e, off_e);
end


function [goni_d, goni_labels, goni_srate, edges_samples] = load_multi_goni_files(raw_goni_paths, AL)
    if ischar(raw_goni_paths), raw_goni_paths = {raw_goni_paths}; end
    assert(iscell(raw_goni_paths) && ~isempty(raw_goni_paths), ...
        'raw_goni_paths must be a non-empty cell array');

    if isfield(AL, 'IEI_threshold_ms') && ~isempty(AL.IEI_threshold_ms)
        iei_thr_ms = AL.IEI_threshold_ms;
    else
        iei_thr_ms = 200;
    end
    iei_thr_samples = round(iei_thr_ms);

    data_cells = cell(1, length(raw_goni_paths));
    n_samples = zeros(1, length(raw_goni_paths));
    n_edges = zeros(1, length(raw_goni_paths));
    edges_samples = [];
    goni_labels = {};
    goni_srate = 1000;
    sample_offset = 0;

    for f = 1:length(raw_goni_paths)
        G = load_goniometer(raw_goni_paths{f});
        if G.srate ~= 1000
            warning('extract_goni:multiSrate', ...
                'goni file %d srate=%d, expected 1000', f, G.srate);
        end
        if f == 1
            goni_labels = G.labels;
            goni_srate = G.srate;
        elseif ~isequal(G.labels, goni_labels)
            warning('extract_goni:multiLabelMismatch', ...
                'goni file %d labels differ from file 1; using file 1 layout', f);
        end

        raw_edges = detect_stim_edges(G.stim, 0.5);
        [dedup_edges, ~] = iei_dedup(raw_edges, iei_thr_samples);
        n_edges(f) = length(dedup_edges);
        edges_samples = [edges_samples, dedup_edges(:)' + sample_offset]; %#ok<AGROW>

        data_cells{f} = G.data;
        n_samples(f) = size(G.data, 1);
        sample_offset = sample_offset + n_samples(f);
    end

    if isfield(AL, 'n_goni_edges_per_file') && ~isequal(n_edges, AL.n_goni_edges_per_file)
        warning('extract_goni:multiEdgeCountMismatch', ...
            'recomputed goni edge counts [%s] differ from align log [%s]', ...
            num2str(n_edges), num2str(AL.n_goni_edges_per_file));
    end

    goni_d = vertcat(data_cells{:});
end


function specs = extract_condition_specs(label)
    if strcmp(label, 'SUB_01_sess01')
        specs = {
            'S  1', 'S  2', 'imagine', 'W', '';
            'S  4', 'S  5', 'walk',    'S', 'without_r1';
            'S  4', 'S  5', 'rest',    '',  'with_r1'
        };
    else
        specs = {
            'S  1', 'S  2', 'imagine', 'W', '';
            'S  4', 'S  5', 'walk',    'S', '';
            'S  7', 'S  8', 'rest',    '',  ''
        };
    end
end


function pairs_out = filter_pairs_by_r1(pairs, mode, types, evt_lats, srate)
    keep = false(size(pairs, 1), 1);
    for i = 1:size(pairs, 1)
        sl  = evt_lats(pairs(i, 1));
        nxt = evt_lats(pairs(i, 2));
        has_r1 = has_coincident_r1(types, evt_lats, sl, nxt, srate);
        switch mode
            case 'with_r1'
                keep(i) = has_r1;
            case 'without_r1'
                keep(i) = ~has_r1;
            otherwise
                error('extract_goni:badR1Mode', 'Unknown R1 mode: %s', mode);
        end
    end
    pairs_out = pairs(keep, :);
end


function tf = has_coincident_r1(types, evt_lats, sl, nxt, srate)
    is_r1 = cellfun(@(t) ~isempty(regexp(strtrim(t), '^R\s*1$', 'once')), types);
    r1_lats = evt_lats(is_r1);
    tol = max(2, round(0.02 * srate));
    tf = any(abs(r1_lats - sl) <= tol) || any(abs(r1_lats - nxt) <= tol);
end


function [g, via] = goni_sample(k, goni_idx_arr, edges_samples, eeg_lat, offset_samp)
    gi = goni_idx_arr(k);
    if ~isnan(gi) && gi >= 1 && gi <= length(edges_samples)
        g   = edges_samples(gi);
        via = 'goni_idx';
    else
        g   = eeg_lat + offset_samp;
        via = 'offset_fallback';
    end
end


function r = which_segment(t_sec, drift_segs)
    r = 1;
    for k = 1:length(drift_segs)
        if t_sec >= drift_segs(k).start_eeg_sec && t_sec <= drift_segs(k).end_eeg_sec
            r = k; return;
        end
    end
end


function s = mad_thr(v, K)
    med = median(v);
    eff = max(mad(v, 1), 0.05 * med);
    lo  = max(med - K * eff, 1.0);
    hi  = med + K * eff;
    s = struct('med', med, 'lo', lo, 'hi', hi);
end


function [idx, names] = resolve_channels(goni_labels, target_labels)
    idx = []; names = {};
    for i = 1:length(target_labels)
        j = find_goni_idx(goni_labels, target_labels{i});
        if j > 0
            idx(end+1) = j; names{end+1} = target_labels{i}; %#ok<AGROW>
        end
    end
end
