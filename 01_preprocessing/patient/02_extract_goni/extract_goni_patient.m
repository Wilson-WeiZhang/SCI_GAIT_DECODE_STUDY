function out = extract_goni_patient(step1_set_path, out_dir, opts)
%EXTRACT_GONI_PATIENT  Patient goniometer extraction with an initial per-trial QC verdict.
%
%   Same procedure as 01_preprocessing/shared/02_extract_goni/extract_goni.m, with:
%     * four task conditions: imagine (S1->S2), rest (S4r->S5r), sit-to-stand (S4->S5),
%       walk (S6->S7 or S7->S8; S7 plus a fixed 120 s when no end marker was recorded)
%     * active channels by task: imagine -> reference walker, rest/s2s/walk -> participant
%     * the relax break after each imagined-walking trial is sent as marker 20/21, which the
%       4-bit trigger port records as 4/5; split_rest_events relabels those breaks as
%       S4r/S5r so they are not mistaken for sit-to-stand trials
%     * rest trials are not scored (movement-amplitude rules do not apply at rest)
%   This verdict (length, and per-channel standard deviation with a 1 degree floor; K = 3)
%   is the one used for analysis of the patient sessions.

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'dur_min_sec'), opts.dur_min_sec = 3; end
    if ~isfield(opts, 'dur_max_sec'), opts.dur_max_sec = 180; end   % 2-Min Walk ~120s

    K = 3 * 1.4826;

    EEG = pop_loadset('filename', step1_set_path);
    [~, label, ~] = fileparts(step1_set_path);
    label = strrep(label, '_step1', '');
    if ~isfield(EEG.etc, 'align_log')
        error('extract_goni_patient: %s has no EEG.etc.align_log', step1_set_path);
    end
    AL = EEG.etc.align_log;

    is_multi_goni = isfield(AL, 'special_case_type') && strcmp(AL.special_case_type, 'multi_goni_file');
    is_legacy = isfield(AL, 'special_case_type') && strcmp(AL.special_case_type, 'legacy_per_pair');

    % ---- Locate raw goni (legacy: concat from raw_goni_files; multi_goni: same) ----
    if is_legacy
        if ~isfield(EEG.etc, 'raw_goni_files') || isempty(EEG.etc.raw_goni_files)
            error('extract_goni_patient: legacy requires EEG.etc.raw_goni_files');
        end
        [goni_d, goni_labels, goni_srate, ~] = load_multi_goni_files(EEG.etc.raw_goni_files, AL);
        edges_samples = [];   % unused in legacy path; we use event.goni_sec directly
    elseif is_multi_goni
        if isfield(opts, 'raw_goni_paths') && ~isempty(opts.raw_goni_paths)
            raw_goni_paths = opts.raw_goni_paths;
        elseif isfield(EEG.etc, 'raw_goni_files') && ~isempty(EEG.etc.raw_goni_files)
            raw_goni_paths = EEG.etc.raw_goni_files;
        else
            error('extract_goni_patient: multi_goni_file requires opts.raw_goni_paths or EEG.etc.raw_goni_files');
        end
        [goni_d, goni_labels, goni_srate, edges_samples] = load_multi_goni_files(raw_goni_paths, AL);
    elseif isfield(opts, 'raw_goni_path') && ~isempty(opts.raw_goni_path)
        raw_goni_path = opts.raw_goni_path;
        G = load_goniometer(raw_goni_path);
        goni_d = G.data; goni_labels = G.labels; goni_srate = G.srate;
        edges_samples = AL.goni_edges_sec_1kHz * 1000;
    elseif isfield(EEG.etc, 'raw_goni_file') && ~isempty(EEG.etc.raw_goni_file)
        raw_goni_path = EEG.etc.raw_goni_file;
        G = load_goniometer(raw_goni_path);
        goni_d = G.data; goni_labels = G.labels; goni_srate = G.srate;
        edges_samples = AL.goni_edges_sec_1kHz * 1000;
    else
        error('extract_goni_patient: cannot determine raw goni path');
    end
    if goni_srate ~= 1000
        warning('extract_goni_patient: goni srate=%d, expected 1000', goni_srate);
    end

    % ---- Channel index resolution (Walker-X and Subject-X) ----
    W_X = {'LHipW X','LKneW X','LAnkW X','RHipW X','RKneW X','RAnkW X'};
    S_X = {'LHipS X','LKneS X','LAnkS X','RHipS X','RKneS X','RAnkS X'};
    [w_idx, w_names] = resolve_channels(goni_labels, W_X);
    [s_idx, s_names] = resolve_channels(goni_labels, S_X);

    % ---- Alignment & event-level goni_idx ----
    offset_samp  = AL.offset_samples_1kHz;
    eeg_srate    = EEG.srate;
    % Split BEFORE goni_idx_arr is built. goni_idx_arr is indexed by event position and is later
    % dereferenced with indices derived from `types`, so both must come from the SAME event array;
    % split_rest_events ends in eeg_checkset(..., 'eventconsistency'), which is entitled to reorder
    EEG = split_rest_events(EEG);
    n_events     = length(EEG.event);
    goni_idx_arr = nan(1, n_events);
    for k = 1:n_events
        if isfield(EEG.event(k), 'goni_idx') && ~isempty(EEG.event(k).goni_idx)
            goni_idx_arr(k) = EEG.event(k).goni_idx;
        end
    end
    types    = {EEG.event.type};
    evt_lats = [EEG.event.latency];

    % multi_goni_file: offset_samp is NaN globally (per-file offsets vary).
    % Estimate effective offset_samp from matched S10 events for fallback use.
    if isnan(offset_samp)
        s10_mask = ~isnan(goni_idx_arr);
        if any(s10_mask)
            anchor_eeg_1k_samp  = (evt_lats(s10_mask) - 1) / eeg_srate * 1000 + 1;
            anchor_goni_samp_1k = edges_samples(goni_idx_arr(s10_mask));
            offset_samp = median(anchor_goni_samp_1k - anchor_eeg_1k_samp);
            fprintf('  multi_goni: estimated effective offset_samp=%.1f from %d S10 anchors\n', ...
                offset_samp, sum(s10_mask));
        else
            warning('extract_goni_patient: NaN offset_samp and no matched S10 anchors — fallback will fail');
            offset_samp = 0;
        end
    end

    % ---- Patient condition specs ----
    %   on_marker, off_marker_or_dursec (char|numeric), cond_name, active_channel ('W'|'S'), dur_max_sec
    %   Walk2min: prefer S7→S8 pair when S8 exists in this session
    %   (a walk may be split into several S7/S8 pairs in one round); fall back to
    %   S7 + 120s fixed duration only when no S8 markers are present.
    %   dur_max_sec: per-condition cap so the next-S2 search cannot bridge
    %   across a missed pair. Imagine/S2S are typically 5–15 s; Walk2min ~120 s.
    if is_legacy
        % two-marker protocol: only S1/S2 markers exist in the EEG
        cond_specs = {
            'S  1', 'S  2', 'imagine', 'W', 30
        };
    else
        [walk_on, walk_off] = choose_walk_markers(types);
        % 'S  4r'/'S  5r' = the automatic relax break, relabelled by split_rest_events above.
        % rest has NO active channel ('') and is never MAD-judged -- it is motionless by definition,
        % so a movement-amplitude verdict would flag every segment. This matches healthy prep
        % extract_goni.m exactly: its rest rows carry '' in the active-channel column, rest is
        % absent from cond_buf and from the MAD loop, and every rest trial is finally set to
        % status='skip'.
        cond_specs = {
            'S  1',  'S  2',   'imagine', 'W', 30;
            'S  4r', 'S  5r',  'rest',    '',  30;
            'S  4',  'S  5',   's2s',     'S', 30;
            walk_on, walk_off, 'walk',    'S', 180
        };
    end

    trials = struct('type',{},'segment',{},'start_eeg_sec',{},'end_eeg_sec',{}, ...
                    'dur_sec',{},'goni_start_idx',{},'goni_end_idx',{}, ...
                    'joint_angles',{},'goni_labels',{}, ...
                    'start_marker',{},'end_marker',{}, ...
                    'start_via',{},'end_via',{}, ...
                    'crosses_seg_boundary',{},'align_status',{});
    qc_arr = struct('trial_idx',{},'type',{},'status',{},'reason',{}, ...
                    'active_std',{},'baseline',{});
    cond_buf = struct( ...
        'imagine', struct('per_ch_std', [], 'n_pts', [], 'idx', [], 'names', {w_names}), ...
        's2s',     struct('per_ch_std', [], 'n_pts', [], 'idx', [], 'names', {s_names}), ...
        'walk',    struct('per_ch_std', [], 'n_pts', [], 'idx', [], 'names', {s_names}));
    trial_eeg_event_idx = [];

    if isfield(AL, 'drift_segments')
        drift_segments = AL.drift_segments;
    else
        drift_segments = struct('start_eeg_sec', -Inf, 'end_eeg_sec', Inf, 'offset_sec', NaN);
    end
    n_seg = length(drift_segments);

    for ci = 1:size(cond_specs, 1)
        on_t  = cond_specs{ci, 1}; off_t = cond_specs{ci, 2};
        cond  = cond_specs{ci, 3}; act   = cond_specs{ci, 4};
        cond_dur_max = cond_specs{ci, 5};
        on_e  = find(strcmp(types, on_t));

        if ischar(off_t)
            off_e = find(strcmp(types, off_t));
            pairs = pair_chrono(on_e, off_e);
            fixed_dur_sec = NaN;
        else
            % Fixed-duration end (e.g. patient Walk2min = S7 + 120s)
            % Synthesize end-event indices = on_e + dur_samples.
            pairs = [on_e(:), on_e(:)];   % col 2 reused as on_e for fixed-duration path
            fixed_dur_sec = off_t;
        end

        switch act
            case 'W', ach = w_idx;
            case 'S', ach = s_idx;
            otherwise, ach = [];
        end

        for p = 1:size(pairs, 1)
            k_on = pairs(p, 1); k_off = pairs(p, 2);
            t_on  = (EEG.event(k_on).latency  - 1) / eeg_srate;
            if isnan(fixed_dur_sec)
                t_off = (EEG.event(k_off).latency - 1) / eeg_srate;
            else
                t_off = t_on + fixed_dur_sec;
                if t_off * eeg_srate >= EEG.pnts
                    fprintf('  Walk2min %d: fixed end %.1fs exceeds recording end, clipping\n', p, t_off);
                    t_off = (EEG.pnts - 1) / eeg_srate;
                end
            end
            dur   = t_off - t_on;
            % Per-condition cap: prevents next-S2 search from bridging
            % across a missed pair (e.g. legacy session pairing trial 24
            % to a 175-sec span when imagine trials are typically 5–15s).
            if dur < opts.dur_min_sec || dur > cond_dur_max, continue; end

            eeg_samp_1k_on  = t_on  * 1000 + 1;
            eeg_samp_1k_off = t_off * 1000 + 1;
            if is_legacy
                gs_on  = NaN; gs_off = NaN;
                if isfield(EEG.event(k_on),  'goni_sec') && ~isempty(EEG.event(k_on).goni_sec)
                    gs_on = EEG.event(k_on).goni_sec;
                end
                if isfield(EEG.event(k_off), 'goni_sec') && ~isempty(EEG.event(k_off).goni_sec)
                    gs_off = EEG.event(k_off).goni_sec;
                end
                if isnan(gs_on) || isnan(gs_off), continue; end
                g_start = gs_on  * 1000 + 1; start_via = 'legacy_2anchor';
                g_end   = gs_off * 1000 + 1; end_via   = 'legacy_2anchor';
            else
                [g_start, start_via] = goni_sample(k_on,  goni_idx_arr, edges_samples, eeg_samp_1k_on,  offset_samp);
                [g_end,   end_via]   = goni_sample(k_off, goni_idx_arr, edges_samples, eeg_samp_1k_off, offset_samp);
            end
            if isnan(g_start) || isnan(g_end), continue; end
            g_start = max(1, round(g_start));
            g_end   = min(size(goni_d, 1), round(g_end));
            if g_end <= g_start, continue; end

            joint = goni_d(g_start:g_end, :);

            seg_id_start = which_segment(t_on,  drift_segments);
            seg_id_end   = which_segment(t_off, drift_segments);
            crosses_seg  = (n_seg >= 2) && (seg_id_start ~= seg_id_end);
            seg_id       = seg_id_start;

            if strcmp(start_via, 'offset_fallback') || strcmp(end_via, 'offset_fallback')
                align_st = 'fallback';
            elseif crosses_seg
                align_st = 'cross_segment';
            else
                align_st = 'ok';
            end

            tr = struct('type', cond, 'segment', seg_id, ...
                'start_eeg_sec', t_on, 'end_eeg_sec', t_off, 'dur_sec', dur, ...
                'goni_start_idx', g_start, 'goni_end_idx', g_end, ...
                'joint_angles',   joint, 'goni_labels', {goni_labels}, ...
                'start_marker',   on_t,  'end_marker', marker_label(off_t), ...
                'start_via',      start_via, 'end_via',  end_via, ...
                'crosses_seg_boundary', crosses_seg, 'align_status', align_st);
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

            % rest is excluded: it has no active channel, so per_ch_std is empty and it is never
            % MAD-judged (see cond_specs). Same guard as the healthy extract_goni.m.
            if ismember(cond, {'imagine', 's2s', 'walk'})
                cond_buf.(cond).per_ch_std(end+1, :) = per_ch_std;
                cond_buf.(cond).n_pts(end+1) = size(joint, 1);
                cond_buf.(cond).idx(end+1)   = tidx;
            end

            trial_eeg_event_idx(tidx) = k_on; %#ok<AGROW>
        end
    end

    % ---- initial MAD verdict per condition ----
    n_problems = 0;
    for cond_c = {'imagine', 's2s', 'walk'}
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
    % rest is motionless by definition -- a movement-amplitude verdict does not apply, so it is
    % marked 'skip' and never counted as a problem. Same as the healthy extract_goni.m.
    for i = 1:length(qc_arr)
        if strcmp(qc_arr(i).type, 'rest')
            qc_arr(i).status = 'skip';
        end
    end

    % ---- align_info ----
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
    n_rest    = sum(strcmp({trials.type}, 'rest'));
    n_s2s     = sum(strcmp({trials.type}, 's2s'));
    n_walk    = sum(strcmp({trials.type}, 'walk'));

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
        'cohort',         'patient', ...
        'n_imagine',      n_imagine, ...
        'n_rest',         n_rest, ...
        's2s_count',      n_s2s, ...
        'n_walk',         n_walk, ...
        'n_problems',     n_problems, ...
        'n_segments',     n_seg, ...
        'ws_swapped',     false, ...
        'qc',             qc_arr, ...
        'qc_method',      'mad_initial_patient', ...
        'trials',         trials, ...
        'event_to_trial', event_to_trial);
    save(out_path, '-struct', 'out_pkg', '-v7.3');
    fprintf('  saved %s | n=%d (I=%d Rest=%d S2S=%d W=%d) problems=%d\n', ...
        out_path, length(trials), n_imagine, n_rest, n_s2s, n_walk, n_problems);

    out = out_pkg;
    out.out_path = out_path;
    out.skipped  = false;
end


% ===== helpers (same as the healthy extraction) =================================

function [walk_on, walk_off] = choose_walk_markers(types)
    if any(strcmp(types, 'S  6')) && any(strcmp(types, 'S  7'))
        walk_on = 'S  6';
        walk_off = 'S  7';
    elseif any(strcmp(types, 'S  8'))
        walk_on = 'S  7';
        walk_off = 'S  8';
    else
        walk_on = 'S  7';
        walk_off = 120;
    end
end

function s = marker_label(marker)
    if ischar(marker)
        s = marker;
    else
        s = sprintf('fixed_%gs', marker);
    end
end

function pairs = pair_chrono(on_e, off_e)
    pairs = zeros(0, 2);
    used  = false(size(off_e));
    for i = 1:length(on_e)
        cand = find(off_e > on_e(i) & ~used, 1);
        if ~isempty(cand)
            pairs(end+1, :) = [on_e(i), off_e(cand)]; %#ok<AGROW>
            used(cand) = true;
        end
    end
end

function [goni_d, goni_labels, goni_srate, edges_samples] = load_multi_goni_files(raw_goni_paths, AL)
    if ischar(raw_goni_paths), raw_goni_paths = {raw_goni_paths}; end
    if isfield(AL, 'IEI_threshold_ms') && ~isempty(AL.IEI_threshold_ms)
        iei_thr_samples = round(AL.IEI_threshold_ms);
    else
        iei_thr_samples = 200;
    end
    data_cells = cell(1, length(raw_goni_paths));
    n_samples = zeros(1, length(raw_goni_paths));
    n_edges   = zeros(1, length(raw_goni_paths));
    edges_samples = []; goni_labels = {}; goni_srate = 1000;
    sample_offset = 0;
    for f = 1:length(raw_goni_paths)
        G = load_goniometer(raw_goni_paths{f});
        if f == 1
            goni_labels = G.labels; goni_srate = G.srate;
        end
        raw_edges = detect_stim_edges(G.stim, 0.5);
        [dedup_edges, ~] = iei_dedup(raw_edges, iei_thr_samples);
        n_edges(f) = length(dedup_edges);
        edges_samples = [edges_samples, dedup_edges(:)' + sample_offset]; %#ok<AGROW>
        data_cells{f} = G.data;
        n_samples(f)  = size(G.data, 1);
        sample_offset = sample_offset + n_samples(f);
    end
    goni_d = vertcat(data_cells{:});
end

function [g, via] = goni_sample(k, goni_idx_arr, edges_samples, eeg_lat, offset_samp)
    gi = goni_idx_arr(k);
    if ~isnan(gi) && gi >= 1 && gi <= length(edges_samples)
        g = edges_samples(gi); via = 'goni_idx';
    else
        g = eeg_lat + offset_samp; via = 'offset_fallback';
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

function EEG = split_rest_events(EEG)
%SPLIT_REST_EVENTS  Rename the auto Relax-break S4/S5 events to 'S  4r'/'S  5r'.
% Kept in sync with the copies in 03_prep_eeg/prep_step2.m and prep_step3.m (prep likewise carries
% its analogous passes_r1_mode in both step2 and step3).
%
% S4/S5 is a marker collision. The task software sends marker 20/21 for the automatic relax
% break after every imagined-walking trial, but the trigger port is 4 bit, so 20 & 0b1111 = 4
% and 21 -> 5: the relax break arrives in the EEG under the sit-to-stand codes. No raw marker
% file contains S20 or S21, the task-software-to-EEG timestamps agree to a median of 0.002 s,
% and the per-session counts reconcile.
%
% They separate on the gap to the preceding MI end ('S  2'): the Relax break opens 3.7-11.2 s after
% the imagined-walk trial it follows, true sit-to-stand opens a new block 270-1370 s later -- a
% 27x-235x separation with nothing in between, in every patient recording.
%
    if isempty(EEG.event), return; end
    gap_rest_max_sec = 30;   % sits inside the empty band between 11.2 s and 270 s
    evt_types = {EEG.event.type};
    evt_lats  = [EEG.event.latency];
    mi_end = sort(evt_lats(strcmp(evt_types, 'S  2')));
    n_rest = 0; n_s2s = 0;
    for ei = 1:numel(EEG.event)
        t = EEG.event(ei).type;
        if ~ischar(t) || ~ismember(t, {'S  4', 'S  5'}), continue; end
        prev = mi_end(mi_end < EEG.event(ei).latency);
        if isempty(prev)
            n_s2s = n_s2s + 1;   % no preceding imagined-walk trial -> cannot be a Relax break
            continue;
        end
        if (EEG.event(ei).latency - prev(end)) / EEG.srate <= gap_rest_max_sec
            EEG.event(ei).type = [t 'r'];
            n_rest = n_rest + 1;
        else
            n_s2s = n_s2s + 1;
        end
    end
    EEG = eeg_checkset(EEG, 'eventconsistency');
    fprintf('  [rest/S2S split] %d Relax-break events -> S4r/S5r; %d kept as true sit-to-stand S4/S5\n', ...
        n_rest, n_s2s);
end
