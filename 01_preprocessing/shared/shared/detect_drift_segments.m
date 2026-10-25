function [segments, per_ttl_seg, per_ttl_raw_offset_sec] = detect_drift_segments( ...
        eeg_ttl_sec, goni_edges_sec, global_offset_sec, opts)
%DETECT_DRIFT_SEGMENTS  Detect piecewise-constant clock-offset segments.
%
%   For each EEG TTL, find the nearest goni edge (wide search window around
%   global_offset_sec). The resulting per-TTL raw offset = goni_edge - eeg_ttl
%   should be piecewise-constant when clock drift is stepwise. Detects jumps
%   via running-median + threshold, returns segment boundaries and per-segment
%   median offsets.
%
%   opts:
%     search_window_sec  wide window around global_offset (default 5.0)
%     jump_thr_sec       change-point threshold on running median (default 0.030)
%     min_seg_len        min TTLs per segment (default 5)
%     smooth_win         running median window (default 11)
%
%   Returns:
%     segments(s).start_ttl, .end_ttl, .start_eeg_sec, .end_eeg_sec,
%                 .offset_sec (median raw offset within segment)
%     per_ttl_seg          [1xN] segment index for each EEG TTL
%     per_ttl_raw_offset_sec [1xN] raw goni-eeg offset for each TTL (NaN if no goni within window)

    if nargin < 4, opts = struct(); end
    if ~isfield(opts, 'search_window_sec'), opts.search_window_sec = 5.0; end
    if ~isfield(opts, 'jump_thr_sec'),      opts.jump_thr_sec = 0.030; end
    if ~isfield(opts, 'min_seg_len'),       opts.min_seg_len = 5; end
    if ~isfield(opts, 'smooth_win'),        opts.smooth_win = 11; end

    eeg_ttl_sec    = eeg_ttl_sec(:).';
    goni_edges_sec = goni_edges_sec(:).';
    N = length(eeg_ttl_sec);

    % --- Step 1: wide-window nearest search to get per-TTL raw offset ---
    per_ttl_raw_offset_sec = nan(1, N);
    for k = 1:N
        target = eeg_ttl_sec(k) + global_offset_sec;
        d = goni_edges_sec - target;
        [dmin, g] = min(abs(d));
        if dmin <= opts.search_window_sec
            per_ttl_raw_offset_sec(k) = goni_edges_sec(g) - eeg_ttl_sec(k);
        end
    end

    % --- Step 2: running median + jump detection ---
    half_w = floor(opts.smooth_win / 2);
    sm = nan(1, N);
    for k = 1:N
        lo = max(1, k - half_w);
        hi = min(N, k + half_w);
        vals = per_ttl_raw_offset_sec(lo:hi);
        vals = vals(~isnan(vals));
        if ~isempty(vals), sm(k) = median(vals); end
    end

    diff_sm = diff(sm);
    jumps = find(abs(diff_sm) > opts.jump_thr_sec);

    % Merge jumps closer than min_seg_len apart
    merged = [];
    for j = 1:length(jumps)
        if isempty(merged) || (jumps(j) - merged(end) > opts.min_seg_len)
            merged(end+1) = jumps(j); %#ok<AGROW>
        end
    end
    jumps = merged;

    % --- Step 3: build segments, compute per-segment median offset ---
    seg_starts = [1, jumps + 1];
    seg_ends   = [jumps, N];
    n_segs = length(seg_starts);
    segments = repmat(struct( ...
        'start_ttl', 0, 'end_ttl', 0, ...
        'start_eeg_sec', 0, 'end_eeg_sec', 0, ...
        'offset_sec', NaN, 'n_ttl', 0), 1, n_segs);
    per_ttl_seg = zeros(1, N);
    for s = 1:n_segs
        idx = seg_starts(s):seg_ends(s);
        per_ttl_seg(idx) = s;
        offs = per_ttl_raw_offset_sec(idx);
        offs = offs(~isnan(offs));
        if isempty(offs)
            seg_off = global_offset_sec;  % fallback
        else
            seg_off = median(offs);
        end
        segments(s).start_ttl = seg_starts(s);
        segments(s).end_ttl   = seg_ends(s);
        segments(s).start_eeg_sec = eeg_ttl_sec(seg_starts(s));
        segments(s).end_eeg_sec   = eeg_ttl_sec(seg_ends(s));
        segments(s).offset_sec    = seg_off;
        segments(s).n_ttl         = length(idx);
    end
end
