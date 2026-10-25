function [offset, info] = align_ioi(eeg_times, goni_times, label)
%% align_ioi — Align EEG and goni using IOI pattern matching
%
% Both systems receive the same TTL trigger sequence. We find the time
% offset by matching inter-onset-interval (IOI) patterns.
%
% Strategy:
%   1. If counts match, try direct 1:1 alignment (fast path)
%   2. If direct fails or counts differ, use IOI substring sliding match
%
% Input:
%   eeg_times  - [1 x N] EEG marker onset times in seconds
%   goni_times - [1 x M] goni stim onset times in seconds
%   label      - string, session label for logging
%
% Output:
%   offset - scalar, add to eeg_times to get goni time frame
%   info   - struct with alignment details

n_eeg  = length(eeg_times);
n_goni = length(goni_times);
eeg_times  = eeg_times(:)';
goni_times = goni_times(:)';

fprintf('  Alignment: EEG=%d events, Goni=%d events\n', n_eeg, n_goni);

if n_eeg == 0 || n_goni == 0
    error('align_ioi:noEvents', 'No events found (EEG=%d, Goni=%d)', n_eeg, n_goni);
end

%% Try direct 1:1 alignment if counts match
if n_eeg == n_goni
    offsets_all = goni_times - eeg_times;
    offset_try = median(offsets_all);
    residuals_try = abs(offsets_all - offset_try) * 1000;

    if max(residuals_try) < 100  % all within 100ms
        offset = offset_try;
        info.method = 'direct';
        info.n_eeg  = n_eeg;
        info.n_goni = n_goni;
        info.n_matched = n_eeg;
        info.match_frac = 1.0;
        info.mean_residual_ms = mean(residuals_try);
        info.max_residual_ms  = max(residuals_try);
        fprintf('  Direct match: offset=%.3fs, residual mean=%.1fms max=%.1fms\n', ...
            offset, info.mean_residual_ms, info.max_residual_ms);
        return;
    end
    fprintf('  Direct match failed (max_residual=%.0fms). Using IOI search...\n', ...
        max(residuals_try));
end

%% IOI substring sliding match — multi-candidate with validation
eeg_ioi  = diff(eeg_times);
goni_ioi = diff(goni_times);

sub_len = min(50, min(length(eeg_ioi), length(goni_ioi)) - 5);
if sub_len < 5
    error('align_ioi:tooFewEvents', 'Not enough events for IOI matching (EEG=%d, Goni=%d)', n_eeg, n_goni);
end

% Collect best candidate offset from each starting position
starts = unique(max(1, round([0.05, 0.2, 0.4, 0.6, 0.8] * (length(eeg_ioi) - sub_len))));
candidates = struct('offset',{},'cost',{},'eeg_start',{},'goni_start',{},'match_count',{},'match_frac',{});
TOL = 0.15;  % 150ms tolerance for counting matches

for si = 1:length(starts)
    sub_start = starts(si);
    sub_ioi = eeg_ioi(sub_start : sub_start + sub_len - 1);

    n_slides = length(goni_ioi) - sub_len + 1;
    best_cost = Inf;
    best_gs = 1;
    for s = 1:n_slides
        seg = goni_ioi(s : s + sub_len - 1);
        cost = sum(abs(sub_ioi - seg));
        if cost < best_cost
            best_cost = cost;
            best_gs = s;
        end
    end

    cand_offset = goni_times(best_gs) - eeg_times(sub_start);

    % Quick validation: count matches
    eeg_shifted = eeg_times + cand_offset;
    mc = 0;
    for i = 1:n_eeg
        if min(abs(goni_times - eeg_shifted(i))) < TOL
            mc = mc + 1;
        end
    end

    c_idx = length(candidates) + 1;
    candidates(c_idx).offset = cand_offset;
    candidates(c_idx).cost = best_cost;
    candidates(c_idx).eeg_start = sub_start;
    candidates(c_idx).goni_start = best_gs;
    candidates(c_idx).match_count = mc;
    candidates(c_idx).match_frac = mc / n_eeg;
end

% Pick candidate with highest match rate (break ties by lowest cost)
[~, sort_idx] = sortrows([[candidates.match_frac]' -[candidates.cost]'], [-1 2]);
best = candidates(sort_idx(1));

fprintf('  IOI candidates: %d tried, best from eeg_pos=%d->goni_pos=%d (cost=%.3f)\n', ...
    length(candidates), best.eeg_start, best.goni_start, best.cost);

% Final validation with detailed errors
offset_from_ioi = best.offset;
eeg_shifted = eeg_times + offset_from_ioi;
match_count = 0;
match_errors = [];
for i = 1:n_eeg
    min_err = min(abs(goni_times - eeg_shifted(i)));
    if min_err < TOL
        match_count = match_count + 1;
        match_errors(end+1) = min_err * 1000; %#ok<AGROW>
    end
end

offset = offset_from_ioi;
info.method = 'ioi_multi_candidate';
info.n_eeg  = n_eeg;
info.n_goni = n_goni;
info.n_matched = match_count;
info.match_frac = match_count / n_eeg;
info.mean_residual_ms = mean(match_errors);
info.max_residual_ms  = max(match_errors);
info.n_candidates = length(candidates);

fprintf('  IOI search: offset=%.3fs, matched=%d/%d (%.0f%%), residual mean=%.1fms max=%.1fms\n', ...
    offset, match_count, n_eeg, info.match_frac*100, ...
    info.mean_residual_ms, info.max_residual_ms);

%% Fallback: streaming Hough offset estimation for low-match cases
if info.match_frac < 0.9 && n_eeg >= 20 && n_goni >= 20
    fprintf('  Trying Hough-style fallback...\n');
    % Streaming histogram: for each EEG event, compute offset to each goni event
    % and bin directly (no huge array allocation)
    bin_width = 0.05;
    off_range = goni_times(end) - eeg_times(1) + 100;  % generous range
    off_min = goni_times(1) - eeg_times(end) - 50;
    n_bins = ceil(off_range / bin_width) + 1;
    hist_counts = zeros(1, n_bins);

    for i = 1:n_eeg
        offs = goni_times - eeg_times(i);
        bins = floor((offs - off_min) / bin_width) + 1;
        valid = bins >= 1 & bins <= n_bins;
        for b = find(valid)
            hist_counts(bins(b)) = hist_counts(bins(b)) + 1;
        end
    end

    [peak_count, peak_idx] = max(hist_counts);
    hough_offset = off_min + (peak_idx - 0.5) * bin_width;

    % Refine: median of nearby matching offsets
    eeg_shifted_h = eeg_times + hough_offset;
    near_offs = [];
    for i = 1:n_eeg
        [min_err, mi] = min(abs(goni_times - eeg_shifted_h(i)));
        if min_err < 0.2  % 200ms for collecting refinement data
            near_offs(end+1) = goni_times(mi) - eeg_times(i); %#ok<AGROW>
        end
    end
    if length(near_offs) > 10
        hough_offset = median(near_offs);
    end

    % Validate Hough offset
    eeg_shifted_h = eeg_times + hough_offset;
    mc_h = 0; errs_h = [];
    for i = 1:n_eeg
        min_err = min(abs(goni_times - eeg_shifted_h(i)));
        if min_err < TOL
            mc_h = mc_h + 1;
            errs_h(end+1) = min_err * 1000; %#ok<AGROW>
        end
    end
    hough_frac = mc_h / n_eeg;

    fprintf('  Hough: offset=%.3fs, matched=%d/%d (%.0f%%)\n', ...
        hough_offset, mc_h, n_eeg, hough_frac*100);

    if hough_frac > info.match_frac
        offset = hough_offset;
        info.method = 'hough';
        info.n_matched = mc_h;
        info.match_frac = hough_frac;
        info.mean_residual_ms = mean(errs_h);
        info.max_residual_ms = max(errs_h);
        fprintf('  Hough improved: %.0f%% -> %.0f%%\n', ...
            match_count/n_eeg*100, hough_frac*100);
    end
end

if info.match_frac < 0.9
    warning('align_ioi:lowMatch', 'Low match rate (%.0f%%) for %s.', info.match_frac*100, label);
end
end
