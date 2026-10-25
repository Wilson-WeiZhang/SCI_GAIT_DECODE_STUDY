function [EEG, align_log] = align_goni_to_eeg(EEG, goni_stim_1kHz, srate_1kHz, opts)
%ALIGN_GONI_TO_EEG  Pair goniometer trigger edges with EEG markers; write EEG.event.goni_idx.
%
%   Call with EEG still at 1000 Hz (before pop_resample).
%
%   Steps
%     1. detect goniometer trigger edges; edges closer than iei_thr_ms form one burst
%     2. global offset from inter-onset-interval (IOI) pattern matching (align_ioi)
%     3. marker pairing by segmented IOI matching, which allows piecewise-constant
%        clock offsets; if fewer than 95% of markers pair, a drift-segment
%        nearest-neighbour pairing is tried and the better result kept. Recordings
%        from the second acquisition system (unstable clock) take the best of three
%        matchers: segmented IOI at 300 ms, segmented IOI at ioi_match_tol_ms, and
%        offset tracking
%     4. hand-verified pairings (manual_align_corrections) replace the automatic
%        result for the segments that have one
%     5. isolated unmatched markers are recovered from the local offset of their
%        three nearest matched neighbours, within ioi_tol_ms
%     6. verdict: PASS if at least 95% of EEG markers are paired, else FAIL;
%        sessions with several goniometer files: SPECIAL_CASE if at least 80%, else FAIL
%
%   Input:
%     EEG             EEGLAB struct @ 1000 Hz with .event populated
%     goni_stim_1kHz  [1 x N] binary goniometer trigger channel @ 1000 Hz
%     srate_1kHz      scalar = 1000
%     opts            struct:
%                       iei_thr_ms              (default 200)
%                       ioi_tol_ms              (default 150)  per-marker assignment tolerance
%                       ioi_match_tol_ms        (default 300)  IOI matching tolerance
%                       special_case            '' | 'multi_goni_file'
%                       ttl_type_re             regex for marker events (default '^S\s*\d+$')
%                       spurious_goni_override  1-based indices of edges to remove
%
%   Output:
%     EEG        same struct with .event(k).goni_idx filled (marker events only)
%     align_log  alignment statistics (written to align_log/{label}.json)

    if nargin < 4, opts = struct(); end
    opts = set_default(opts, 'iei_thr_ms',   200);
    opts = set_default(opts, 'ioi_tol_ms',   150);
    opts = set_default(opts, 'ioi_match_tol_ms', 300);  % IOI matching tolerance; 1200 for the second acquisition system
    opts = set_default(opts, 'special_case', '');
    opts = set_default(opts, 'ttl_type_re',  '^S\s*\d+$');
    opts = set_default(opts, 'spurious_goni_override', []);  % 1-based indices into dedup goni edges to force-remove

    iei_thr_samples = round(opts.iei_thr_ms * srate_1kHz / 1000);
    ioi_tol_samples = round(opts.ioi_tol_ms * srate_1kHz / 1000);

    % Initialize .goni_idx = NaN for all events
    for k = 1:length(EEG.event)
        EEG.event(k).goni_idx = NaN;
    end

    if strcmp(opts.special_case, 'multi_goni_file')
        [EEG, align_log] = align_multi_goni_file(EEG, goni_stim_1kHz, srate_1kHz, opts, ...
            iei_thr_samples, ioi_tol_samples);
        return;
    end

    % TTL filter
    is_ttl = cellfun(@(t) ~isempty(regexp(t, opts.ttl_type_re, 'once')), {EEG.event.type});
    ttl_event_idx = find(is_ttl);
    eeg_ttl_lat   = [EEG.event(ttl_event_idx).latency].';   % samples @ 1000 Hz
    n_eeg_ttl     = length(eeg_ttl_lat);

    % Detect + dedup goni edges
    edges_raw = detect_stim_edges(goni_stim_1kHz, 0.5);
    [edges_dedup, burst_dropped_idx] = iei_dedup(edges_raw, iei_thr_samples);

    % Spurious-edge override: remove edges marked on visual review.
    % (Used for sessions where IEI dedup can't catch the spurious edge —
    % e.g. SUB_16_sess02 where a lone spurious edge at 271.223 s has valid
    % IEI spacing but no corresponding EEG TTL, causing mid-session greedy drift.)
    manual_spurious = [];
    if ~isempty(opts.spurious_goni_override)
        keep_mask = true(1, length(edges_dedup));
        ov = opts.spurious_goni_override(:);
        ov = ov(ov >= 1 & ov <= length(edges_dedup));
        keep_mask(ov) = false;
        manual_spurious = edges_dedup(~keep_mask);
        edges_dedup = edges_dedup(keep_mask);
        fprintf('     manual spurious override: removed %d edge(s) at [%s] s\n', ...
            length(manual_spurious), num2str(manual_spurious/srate_1kHz, '%.3f '));
    end
    n_goni = length(edges_dedup);

    % Phase 1: initial global offset via align_ioi
    eeg_times_sec  = eeg_ttl_lat / srate_1kHz;
    goni_times_sec = edges_dedup / srate_1kHz;
    [offset_sec, info] = align_ioi(eeg_times_sec, goni_times_sec, '_internal');
    offset = offset_sec * srate_1kHz;   % samples @ 1000 Hz

    % Phase 2: try segmented IOI greedy first (primary); if low match, try
    % drift detection (secondary). Segmented IOI handles piecewise-constant
    % offset sessions (e.g. SUB_16_sess02, two goni-clock regions).
    % Goal: maximize matched goni-EEG pairs. Drift offset = QC metric.
    if opts.ioi_match_tol_ms >= 1000
        % second acquisition system: take the best of three matchers:
        %   (a) segmented IOI at tol 300 -> the standard result
        %   (b) segmented IOI at the wide tol         -> bridges sub-second steps
        %   (c) offset-tracking sequential matcher    -> follows arbitrary steps
        % A wide tol mis-matches on clean sessions, so (a) guards against that.
        [pair_a, ~, info_a, round_a] = match_via_ioi_segmented(eeg_times_sec, goni_times_sec, offset_sec, struct('tol_ms', 300));
        [pair_b, ~, info_b, round_b] = match_via_ioi_segmented(eeg_times_sec, goni_times_sec, offset_sec, struct('tol_ms', opts.ioi_match_tol_ms));
        [pair_c, ~, info_c] = match_offset_tracking(eeg_times_sec, goni_times_sec, offset_sec, opts.ioi_match_tol_ms/1000);
        sa = sum(~isnan(pair_a)); sb = sum(~isnan(pair_b)); sc = sum(~isnan(pair_c));
        if sc >= sa && sc >= sb
            pair_ioi = pair_c; per_ttl_round_ioi = double(~isnan(pair_c(:)).');
            ioi_info = struct('match_score', info_c.match_score, 'match_fraction', info_c.match_fraction, ...
                'n_rounds', 1, 'method', 'offset_tracking', 'seed_offset_sec', info_c.seed_offset_sec);
        elseif sb > sa
            pair_ioi = pair_b; per_ttl_round_ioi = round_b; ioi_info = info_b;
        else
            pair_ioi = pair_a; per_ttl_round_ioi = round_a; ioi_info = info_a;
        end
    else
        [pair_ioi, ~, ioi_info, per_ttl_round_ioi] = match_via_ioi_segmented(eeg_times_sec, goni_times_sec, offset_sec, struct('tol_ms', opts.ioi_match_tol_ms));
    end
    score_ioi = sum(~isnan(pair_ioi));
    score_drift = 0; pair_drift = nan(n_eeg_ttl, 1);
    [drift_segments, per_ttl_seg] = detect_drift_segments( ...
        eeg_times_sec, goni_times_sec, offset_sec);
    if score_ioi / n_eeg_ttl < 0.95
        % IOI failed — suspect clock drift. Compute drift-segment nearest-neighbor.
        for k = 1:n_eeg_ttl
            seg_off = drift_segments(per_ttl_seg(k)).offset_sec * srate_1kHz;
            target = eeg_ttl_lat(k) + seg_off;
            [dmin, g] = min(abs(edges_dedup - target));
            if dmin <= ioi_tol_samples
                pair_drift(k) = g;
            end
        end
        % Dedup nearest-neighbor duplicates
        seen = false(n_goni, 1);
        for ii = find(~isnan(pair_drift))'
            g = pair_drift(ii);
            if seen(g), pair_drift(ii) = NaN; else, seen(g) = true; end
        end
        score_drift = sum(~isnan(pair_drift));
    end

    if score_drift > score_ioi
        goni_idx_per_ttl = pair_drift(:);
        chosen_method = 'drift_segment';
    else
        goni_idx_per_ttl = pair_ioi(:);
        if isfield(ioi_info, 'method') && strcmp(ioi_info.method, 'offset_tracking')
            chosen_method = 'offset_tracking';
        elseif ioi_info.n_rounds > 1
            chosen_method = 'ioi_segmented';
        else
            chosen_method = 'ioi_greedy';
        end
        % Build per-round drift_segments so residual is computed against each
        % round's own offset (otherwise seg-A matches appear as 8s residuals
        % vs seg-B offset in sessions like SUB_16_sess02).
        drift_segments = repmat(struct( ...
            'start_ttl', 0, 'end_ttl', 0, ...
            'start_eeg_sec', 0, 'end_eeg_sec', 0, ...
            'offset_sec', NaN, 'n_ttl', 0), 1, ioi_info.n_rounds);
        for r = 1:ioi_info.n_rounds
            idx_r = find(per_ttl_round_ioi == r);
            if isempty(idx_r)
                drift_segments(r).offset_sec = ioi_info.rounds(r).offset_sec;
                continue;
            end
            % Recompute offset as median actual (goni - eeg) of round-r matched pairs
            pair_offs = nan(1, length(idx_r));
            for jj = 1:length(idx_r)
                k = idx_r(jj);
                pair_offs(jj) = edges_dedup(pair_ioi(k))/srate_1kHz - eeg_times_sec(k);
            end
            drift_segments(r).start_ttl     = idx_r(1);
            drift_segments(r).end_ttl       = idx_r(end);
            drift_segments(r).start_eeg_sec = eeg_times_sec(idx_r(1));
            drift_segments(r).end_eeg_sec   = eeg_times_sec(idx_r(end));
            drift_segments(r).offset_sec    = median(pair_offs);
            drift_segments(r).n_ttl         = length(idx_r);
        end
        per_ttl_seg = per_ttl_round_ioi;
        % Unmatched TTLs: assign to nearest round segment by time
        for k = 1:n_eeg_ttl
            if per_ttl_seg(k) == 0
                [~, rr] = min(abs(arrayfun(@(s) ...
                    (s.start_eeg_sec + s.end_eeg_sec)/2, drift_segments) - eeg_times_sec(k)));
                per_ttl_seg(k) = rr;
            end
        end
    end
    fprintf('     method=%s | ioi=%d drift=%d → %d/%d matched\n', ...
        chosen_method, score_ioi, score_drift, sum(~isnan(goni_idx_per_ttl)), n_eeg_ttl);

    % --- MANUAL OVERRIDE: curated EEG<->goni pairing for the few hand-verified
    % sessions. apply_manual_override forces the map in (hard assert = never a
    % silent fall-through); a complete manual map also suppresses the transition-
    [manual_applied, gio] = apply_manual_override(opts, n_eeg_ttl, n_goni);
    if manual_applied
        goni_idx_per_ttl = gio;
        chosen_method    = 'manual_override';
    end

    % Per-TTL effective offset (samples). Initialized from per-segment offset;
    % overwritten for augmented TTLs below.
    per_ttl_offset_samp = nan(n_eeg_ttl, 1);
    for k = 1:n_eeg_ttl
        s = per_ttl_seg(k);
        if s >= 1 && s <= length(drift_segments) && ~isnan(drift_segments(s).offset_sec)
            per_ttl_offset_samp(k) = drift_segments(s).offset_sec * srate_1kHz;
        end
    end

    % --- Transition-zone augmentation pass ---
    % Isolated unmatched TTLs in offset-transition zones may not trigger
    % another segmented IOI round (which requires a contiguous unmatched
    % block of length ≥ min_block_len). Recover them via per-TTL local
    % offset (median of nearest matched neighbors) + nearest-neighbor goni
    % within ioi_tol_samples. Safe: only adds matches; bounded by tol.
    per_ttl_augmented = false(n_eeg_ttl, 1);
    n_aug = 0;
    if ~manual_applied && any(isnan(goni_idx_per_ttl)) && sum(~isnan(goni_idx_per_ttl)) >= 2
        used_goni            = unique(goni_idx_per_ttl(~isnan(goni_idx_per_ttl)));
        available_goni       = setdiff(1:n_goni, used_goni);
        matched_ttl          = find(~isnan(goni_idx_per_ttl));
        matched_offsets_samp = arrayfun(@(kk) ...
            edges_dedup(goni_idx_per_ttl(kk)) - eeg_ttl_lat(kk), matched_ttl);

        unmatched_idx = find(isnan(goni_idx_per_ttl));
        for k = unmatched_idx(:)'
            if isempty(available_goni), break; end
            [~, sorted_idx] = sort(abs(matched_ttl - k));
            n_use = min(3, length(matched_ttl));
            local_off_samp = median(matched_offsets_samp(sorted_idx(1:n_use)));
            target = eeg_ttl_lat(k) + local_off_samp;
            [dmin, ai] = min(abs(edges_dedup(available_goni) - target));
            if dmin <= ioi_tol_samples
                goni_idx_per_ttl(k)    = available_goni(ai);
                per_ttl_offset_samp(k) = local_off_samp;
                per_ttl_augmented(k)   = true;
                available_goni(ai)     = [];
                n_aug                  = n_aug + 1;
            end
        end
        if n_aug > 0
            fprintf('     transition-zone augmentation: +%d (total %d/%d)\n', ...
                n_aug, sum(~isnan(goni_idx_per_ttl)), n_eeg_ttl);
        end
    end

    % QC metric: drift range (max offset difference across segments)
    seg_offsets_sec = arrayfun(@(s) s.offset_sec, drift_segments);
    drift_range_ms = (max(seg_offsets_sec) - min(seg_offsets_sec)) * 1000;

    % Compute per-TTL residual using per-TTL effective offset (segment
    % offset for normal matches; local interpolated offset for augmented).
    residuals = nan(n_eeg_ttl, 1);
    for k = 1:n_eeg_ttl
        if ~isnan(goni_idx_per_ttl(k)) && ~isnan(per_ttl_offset_samp(k))
            target = eeg_ttl_lat(k) + per_ttl_offset_samp(k);
            residuals(k) = abs(edges_dedup(goni_idx_per_ttl(k)) - target);
        end
    end

    % Spurious goni = edges not matched
    matched_goni_set = unique(goni_idx_per_ttl(~isnan(goni_idx_per_ttl)));
    spurious_goni_idx = setdiff(1:n_goni, matched_goni_set);

    % Write back
    for k = 1:n_eeg_ttl
        EEG.event(ttl_event_idx(k)).goni_idx = goni_idx_per_ttl(k);
    end

    % Compute residual stats in ms (only over matched)
    n_matched = sum(~isnan(goni_idx_per_ttl));
    per_ttl_residual_ms = residuals * 1000 / srate_1kHz;   % NaN for unmatched
    per_ttl_eeg_time_sec = eeg_ttl_lat / srate_1kHz;
    per_ttl_matched = ~isnan(goni_idx_per_ttl);

    % IOI validation: for consecutive matched pairs, EEG IOI and goni IOI should match
    ioi_diff_ms = nan(n_eeg_ttl, 1);
    ioi_inconsistent = false(n_eeg_ttl, 1);
    ioi_tol_ms = 100;  % consecutive pair IOI tolerance (above TTL jitter ~50ms, below single-TTL sep)
    matched_list = find(per_ttl_matched);
    for ii = 1:length(matched_list)-1
        k1 = matched_list(ii); k2 = matched_list(ii+1);
        ioi_eeg_ms  = (eeg_ttl_lat(k2) - eeg_ttl_lat(k1)) * 1000 / srate_1kHz;
        ioi_goni_ms = (edges_dedup(goni_idx_per_ttl(k2)) - edges_dedup(goni_idx_per_ttl(k1))) * 1000 / srate_1kHz;
        d = ioi_eeg_ms - ioi_goni_ms;
        ioi_diff_ms(k1) = d;
        if abs(d) > ioi_tol_ms
            ioi_inconsistent(k1) = true;
            ioi_inconsistent(k2) = true;
        end
    end
    n_ioi_inconsistent = sum(ioi_inconsistent);

    matched_res_ms = per_ttl_residual_ms(per_ttl_matched);
    if ~isempty(matched_res_ms)
        res_ms = struct('median', median(matched_res_ms), ...
                        'max',    max(matched_res_ms), ...
                        'p95',    prctile(matched_res_ms, 95));
    else
        res_ms = struct('median', NaN, 'max', NaN, 'p95', NaN);
    end

    verdict = 'PASS';
    if n_matched / n_eeg_ttl < 0.95, verdict = 'FAIL'; end

    unmatched_eeg_idx_in_ttl = find(isnan(goni_idx_per_ttl));
    align_log = struct( ...
        'n_eeg_events',             length(EEG.event), ...
        'n_eeg_events_ttl',         n_eeg_ttl, ...
        'n_goni_edges_raw',         length(edges_raw), ...
        'n_goni_edges_after_dedup', n_goni, ...
        'n_bursts_removed',         length(burst_dropped_idx), ...
        'burst_edge_samples_1kHz',  edges_raw(burst_dropped_idx).', ...
        'n_matched',                n_matched, ...
        'n_unmatched_eeg',          n_eeg_ttl - n_matched, ...
        'unmatched_eeg_idx',        ttl_event_idx(unmatched_eeg_idx_in_ttl).', ...
        'residual_ms',              res_ms, ...
        'per_ttl_eeg_time_sec',     per_ttl_eeg_time_sec(:).', ...
        'per_ttl_residual_ms',      per_ttl_residual_ms(:).', ...
        'per_ttl_matched',          per_ttl_matched(:).', ...
        'per_ttl_segment',          per_ttl_seg(:).', ...
        'per_ttl_goni_time_sec',    store_matched_goni_sec(goni_idx_per_ttl, edges_dedup, srate_1kHz), ...
        'per_ttl_ioi_diff_ms',      ioi_diff_ms(:).', ...
        'per_ttl_ioi_inconsistent', ioi_inconsistent(:).', ...
        'n_ioi_inconsistent',       n_ioi_inconsistent, ...
        'per_ttl_augmented',        per_ttl_augmented(:).', ...
        'n_augmented',              n_aug, ...
        'per_ttl_local_offset_samp_1kHz', per_ttl_offset_samp(:).', ...
        'goni_edges_sec_1kHz',      (edges_dedup(:).') / srate_1kHz, ...
        'spurious_goni_idx',        spurious_goni_idx(:).', ...
        'n_spurious_goni',          length(spurious_goni_idx), ...
        'ioi_match_info',           ioi_info, ...
        'match_method',             chosen_method, ...
        'score_ioi',                score_ioi, ...
        'score_drift',              score_drift, ...
        'drift_range_ms',           drift_range_ms, ...
        'drift_segments',           drift_segments, ...
        'IEI_threshold_ms',         opts.iei_thr_ms, ...
        'IOI_match_tolerance_ms',   opts.ioi_tol_ms, ...
        'offset_samples_1kHz',      offset, ...
        'auto_verdict',             verdict, ...
        'special_case_type',        opts.special_case, ...
        'alignment_algorithm',      info.method, ...
        'align_ioi_info',           info, ...
        'align_timestamp',          datestr(now, 31));
end


function s = set_default(s, f, default)
    if ~isfield(s, f), s.(f) = default; end
end


function gs = store_matched_goni_sec(goni_idx_per_ttl, edges_dedup, srate_1kHz)
    n = length(goni_idx_per_ttl);
    gs = nan(1, n);
    for k = 1:n
        if ~isnan(goni_idx_per_ttl(k))
            gs(k) = edges_dedup(goni_idx_per_ttl(k)) / srate_1kHz;
        end
    end
end


function [applied, goni_idx_per_ttl] = apply_manual_override(opts, n_eeg_ttl, n_goni)
%APPLY_MANUAL_OVERRIDE  Replace the auto pairing with the hand-verified curated map
%   (manual_align_corrections.m) for the few corrected sessions. Returns
%   applied=false, [] when the session has no manual correction. A listed session whose map comes out empty raises an error. Shared by the single-goni and multi_goni paths.
    applied = false; goni_idx_per_ttl = [];
    man_lbl = '';
    if isfield(opts, 'label') && ~isempty(opts.label), man_lbl = char(opts.label); end
    [man_pairs, man_known] = manual_align_corrections(man_lbl, n_eeg_ttl, n_goni);
    if ~man_known, return; end
    assert(~isempty(man_pairs), ...
        ['align_goni_to_eeg: manual-correction session "%s" produced an EMPTY map ' ...
         '(n_eeg_ttl=%d n_goni=%d) -- refusing to fall through to the auto matcher'], ...
        man_lbl, n_eeg_ttl, n_goni);
    goni_idx_per_ttl = nan(n_eeg_ttl, 1);
    goni_idx_per_ttl(man_pairs(:, 1) + 1) = man_pairs(:, 2) + 1;   % 0-based -> 1-based
    applied = true;
    fprintf('     MANUAL OVERRIDE %s: %d curated pairs applied (auto matcher discarded)\n', ...
        man_lbl, size(man_pairs, 1));
end


function [EEG, align_log] = align_multi_goni_file(EEG, goni_stim_cells, srate_1kHz, opts, ...
        iei_thr_samples, ioi_tol_samples) %#ok<INUSD>
%ALIGN_MULTI_GONI_FILE  Per-file IOI alignment for 1 EEG + N goni files.
%
%   Each goni file has its own zero-based clock; we align each independently
%   to the shared EEG clock via align_ioi (IOI-fingerprint search is robust to
%   covering only a sub-range of EEG TTLs). Outputs use a global 1..N_total
%   goni edge numbering across files, with edges stored in EEG-clock seconds.

    assert(iscell(goni_stim_cells), ...
        'multi_goni_file expects goni_stim_1kHz as a cell array of stim signals');
    nFiles = length(goni_stim_cells);
    assert(nFiles >= 1, 'multi_goni_file needs at least 1 goni file');

    % TTL filter (same regex as standard path)
    is_ttl = cellfun(@(t) ~isempty(regexp(t, opts.ttl_type_re, 'once')), {EEG.event.type});
    ttl_event_idx = find(is_ttl);
    eeg_ttl_lat   = [EEG.event(ttl_event_idx).latency].';
    n_eeg_ttl     = length(eeg_ttl_lat);
    eeg_times_sec = eeg_ttl_lat / srate_1kHz;

    % --- Detect + dedup edges per file (each in its own file-local clock)
    edges_per_file   = cell(1, nFiles);
    n_edges_per_file = zeros(1, nFiles);
    n_raw_per_file   = zeros(1, nFiles);
    n_burst_per_file = zeros(1, nFiles);
    for f = 1:nFiles
        raw_f = detect_stim_edges(goni_stim_cells{f}, 0.5);
        [dd_f, burst_f] = iei_dedup(raw_f, iei_thr_samples);
        edges_per_file{f}   = dd_f;
        n_edges_per_file(f) = length(dd_f);
        n_raw_per_file(f)   = length(raw_f);
        n_burst_per_file(f) = length(burst_f);
    end

    % --- Per-file IOI alignment against shared EEG TTL list
    offsets_sec    = nan(1, nFiles);
    pair_per_file  = cell(1, nFiles);   % local goni idx per EEG TTL (NaN if that file doesn't match)
    ioi_info_per_f = cell(1, nFiles);
    for f = 1:nFiles
        if n_edges_per_file(f) == 0
            pair_per_file{f}  = nan(n_eeg_ttl, 1);
            ioi_info_per_f{f} = struct('method', 'empty', 'n_eeg', n_eeg_ttl, 'n_goni', 0);
            fprintf('     multi_goni_file[%d/%d]: no edges, skipped\n', f, nFiles);
            continue;
        end
        goni_sec_f = edges_per_file{f} / srate_1kHz;
        try
            [off_f, info_f] = align_ioi(eeg_times_sec, goni_sec_f, ...
                sprintf('multi_goni_f%d', f));
        catch ME
            warning('align_multi_goni_file:align_failed', ...
                'File %d align_ioi failed: %s', f, ME.message);
            off_f = NaN;
            info_f = struct('method', 'failed', 'error', ME.message);
        end
        offsets_sec(f)    = off_f;
        ioi_info_per_f{f} = info_f;

        if isnan(off_f)
            pair_per_file{f} = nan(n_eeg_ttl, 1);
            continue;
        end

        [pair_f, ~, ~, ~] = match_via_ioi_segmented(eeg_times_sec, goni_sec_f, off_f);
        pair_per_file{f} = pair_f(:);
        fprintf('     multi_goni_file[%d/%d]: offset=%.3fs, matched=%d/%d\n', ...
            f, nFiles, off_f, sum(~isnan(pair_f)), n_eeg_ttl);
    end

    % --- Combine per-TTL: first file that claims a TTL wins (rare overlap)
    cumulative_edges = cumsum([0, n_edges_per_file]);   % global base per file
    goni_idx_per_ttl  = nan(n_eeg_ttl, 1);
    goni_file_per_ttl = nan(n_eeg_ttl, 1);
    for k = 1:n_eeg_ttl
        for f = 1:nFiles
            pf = pair_per_file{f};
            if ~isempty(pf) && ~isnan(pf(k)) && isnan(goni_idx_per_ttl(k))
                goni_idx_per_ttl(k)  = cumulative_edges(f) + pf(k);
                goni_file_per_ttl(k) = f;
                break;
            end
        end
    end

    % --- MANUAL OVERRIDE (multi_goni): same curated map as the single-goni path,
    % but in the global 1..N_total (cumulative_edges) numbering that
    % load_multi_goni_files rebuilds. Recovers sessions where align_ioi finds the
    % right offset but the per-file segmented matcher under-matches (e.g.
    % SUB_22_sess01b: 143 curated vs 10 auto).
    [manual_applied, gio] = apply_manual_override(opts, n_eeg_ttl, cumulative_edges(end));
    if manual_applied
        goni_idx_per_ttl  = gio;
        goni_file_per_ttl = nan(n_eeg_ttl, 1);
        mk = find(~isnan(goni_idx_per_ttl));
        % file of each global edge index (== find(g > cumulative_edges,1,'last'))
        goni_file_per_ttl(mk) = sum(goni_idx_per_ttl(mk) > cumulative_edges(:).', 2);
    end

    % --- Concatenate edges in EEG-clock seconds (global 1..N_total)
    % edges_dedup_eeg_samples(g) = file_edge_sample + offset_f*srate
    n_goni_total = cumulative_edges(end);
    edges_dedup_eeg = nan(1, n_goni_total);
    for f = 1:nFiles
        if n_edges_per_file(f) == 0, continue; end
        if isnan(offsets_sec(f))
            % No offset -> leave as file-local time (won't be referenced since
            % no TTL matched this file, but preserve NaN-free for storage).
            base = 0;
        else
            base = offsets_sec(f) * srate_1kHz;
        end
        rng = (cumulative_edges(f)+1):cumulative_edges(f+1);
        edges_dedup_eeg(rng) = edges_per_file{f} + base;
    end

    % --- Residuals: each TTL uses its own file's offset
    residuals = nan(n_eeg_ttl, 1);
    for k = 1:n_eeg_ttl
        if ~isnan(goni_idx_per_ttl(k))
            f = goni_file_per_ttl(k);
            seg_off_samp = offsets_sec(f) * srate_1kHz;
            target = eeg_ttl_lat(k) + seg_off_samp;
            residuals(k) = abs(edges_dedup_eeg(goni_idx_per_ttl(k)) - target);
        end
    end

    % --- Write back to EEG.event.goni_idx
    for k = 1:n_eeg_ttl
        EEG.event(ttl_event_idx(k)).goni_idx = goni_idx_per_ttl(k);
    end

    % --- Stats
    n_matched            = sum(~isnan(goni_idx_per_ttl));
    per_ttl_matched      = ~isnan(goni_idx_per_ttl);
    per_ttl_residual_ms  = residuals * 1000 / srate_1kHz;
    per_ttl_eeg_time_sec = eeg_ttl_lat / srate_1kHz;

    matched_res_ms = per_ttl_residual_ms(per_ttl_matched);
    if ~isempty(matched_res_ms)
        res_ms = struct('median', median(matched_res_ms), ...
                        'max',    max(matched_res_ms), ...
                        'p95',    prctile(matched_res_ms, 95));
    else
        res_ms = struct('median', NaN, 'max', NaN, 'p95', NaN);
    end

    matched_goni_set  = unique(goni_idx_per_ttl(~isnan(goni_idx_per_ttl)));
    spurious_goni_idx = setdiff(1:n_goni_total, matched_goni_set);

    % --- Verdict
    if n_eeg_ttl > 0 && (n_matched / n_eeg_ttl) >= 0.8
        verdict = 'SPECIAL_CASE';
    else
        verdict = 'FAIL';
    end

    unmatched_eeg_idx_in_ttl = find(isnan(goni_idx_per_ttl));

    align_log = struct( ...
        'n_eeg_events',              length(EEG.event), ...
        'n_eeg_events_ttl',          n_eeg_ttl, ...
        'n_goni_edges_raw',          sum(n_raw_per_file), ...
        'n_goni_edges_after_dedup',  n_goni_total, ...
        'n_bursts_removed',          sum(n_burst_per_file), ...
        'burst_edge_samples_1kHz',   [], ...
        'n_matched',                 n_matched, ...
        'n_unmatched_eeg',           n_eeg_ttl - n_matched, ...
        'unmatched_eeg_idx',         ttl_event_idx(unmatched_eeg_idx_in_ttl).', ...
        'residual_ms',               res_ms, ...
        'per_ttl_eeg_time_sec',      per_ttl_eeg_time_sec(:).', ...
        'per_ttl_residual_ms',       per_ttl_residual_ms(:).', ...
        'per_ttl_matched',           per_ttl_matched(:).', ...
        'per_ttl_goni_time_sec',     store_matched_goni_sec(goni_idx_per_ttl, edges_dedup_eeg, srate_1kHz), ...
        'goni_edges_sec_1kHz',       edges_dedup_eeg / srate_1kHz, ...
        'spurious_goni_idx',         spurious_goni_idx(:).', ...
        'n_spurious_goni',           length(spurious_goni_idx), ...
        'n_files',                   nFiles, ...
        'offsets_sec_per_file',      offsets_sec, ...
        'n_goni_edges_per_file',     n_edges_per_file, ...
        'n_goni_edges_raw_per_file', n_raw_per_file, ...
        'n_bursts_per_file',         n_burst_per_file, ...
        'goni_file_per_ttl',         goni_file_per_ttl(:).', ...
        'match_method',              'multi_goni_file', ...
        'IEI_threshold_ms',          opts.iei_thr_ms, ...
        'IOI_match_tolerance_ms',    opts.ioi_tol_ms, ...
        'offset_samples_1kHz',       NaN, ...
        'auto_verdict',              verdict, ...
        'special_case_type',         'multi_goni_file', ...
        'alignment_algorithm',       'multi_goni_file_IOI', ...
        'align_ioi_info_per_file',   {ioi_info_per_f}, ...
        'align_timestamp',           datestr(now, 31));
end


