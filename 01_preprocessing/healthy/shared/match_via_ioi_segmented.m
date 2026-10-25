function [goni_idx_per_ttl, spurious_goni_idx, info, per_ttl_round] = match_via_ioi_segmented( ...
        eeg_sec, goni_sec, initial_offset_sec, opts)
%MATCH_VIA_IOI_SEGMENTED  Multi-round IOI greedy for sessions with piecewise-constant offset.
%
%   Round 1 runs match_via_ioi at the global initial offset. For each residual
%   unmatched EEG block (len >= min_block_len), we estimate a local offset from
%   the nearest-neighbor distances against *remaining* goni edges (those not
%   consumed by previous rounds), then rerun match_via_ioi restricted to that
%   block + remaining goni. Matches are merged back into the global pair.
%
%   Handles sessions like SUB_16_sess02 where goniometer clock jumps mid-session
%   (two or more stable offsets, IOI pattern clean within each).
%
%   opts (forwarded to match_via_ioi except):
%     min_block_len   min unmatched EEG TTLs to trigger a new round (default 5)
%
%   info.rounds(r).offset_sec / .block_range / .matched
%   info.match_score / .n_rounds

    if nargin < 4, opts = struct(); end
    if ~isfield(opts, 'min_block_len'), opts.min_block_len = 5; end
    if ~isfield(opts, 'max_rounds'),    opts.max_rounds = 4; end

    eeg_sec  = eeg_sec(:).';
    goni_sec = goni_sec(:).';
    N = length(eeg_sec);
    M = length(goni_sec);

    per_ttl_round = zeros(1, N);

    % --- Round 1: global offset
    [pair, ~, info1] = match_via_ioi(eeg_sec, goni_sec, initial_offset_sec, opts);
    per_ttl_round(~isnan(pair)) = 1;
    rounds = struct( ...
        'round',       1, ...
        'offset_sec',  initial_offset_sec, ...
        'block_start', 1, ...
        'block_end',   N, ...
        'matched',     sum(~isnan(pair)), ...
        'ioi_info',    info1);

    for r = 2:opts.max_rounds
        unmatched_mask = isnan(pair);
        if sum(unmatched_mask) < opts.min_block_len, break; end

        blocks = find_unmatched_blocks(unmatched_mask);
        blocks = blocks([blocks.len] >= opts.min_block_len);
        if isempty(blocks), break; end

        [~, bi] = max([blocks.len]);
        bstart  = blocks(bi).start;
        bend    = blocks(bi).end;

        used_goni = unique(pair(~isnan(pair)));
        rem_idx   = setdiff(1:M, used_goni);
        if length(rem_idx) < opts.min_block_len, break; end
        rem_sec   = goni_sec(rem_idx);

        % Local offset via IOI pattern match on (block, remaining goni) —
        % nearest-neighbor median is unreliable when remaining goni spans
        % the whole session (each TTL finds a time-close goni that's actually
        % matched under a different-but-nonzero offset). align_ioi uses IOI
        % fingerprints which are offset-invariant.
        eeg_block_early = eeg_sec(bstart:bend);
        try
            [local_offset, ~] = align_ioi(eeg_block_early, rem_sec, 'round2_local');
        catch
            local_offset = initial_offset_sec;  % fallback if align_ioi fails
        end
        if isnan(local_offset) || isempty(local_offset)
            local_offset = initial_offset_sec;
        end

        % Re-run IOI greedy on block, restricted to remaining goni
        eeg_block = eeg_sec(bstart:bend);
        [bp, ~, binfo] = match_via_ioi(eeg_block, rem_sec, local_offset, opts);

        added = 0;
        for kk = 1:length(bp)
            if ~isnan(bp(kk)) && isnan(pair(bstart + kk - 1))
                pair(bstart + kk - 1) = rem_idx(bp(kk));
                per_ttl_round(bstart + kk - 1) = r;
                added = added + 1;
            end
        end

        rounds(end+1) = struct( ...
            'round',       r, ...
            'offset_sec',  local_offset, ...
            'block_start', bstart, ...
            'block_end',   bend, ...
            'matched',     added, ...
            'ioi_info',    binfo); %#ok<AGROW>

        if added == 0, break; end
    end

    % Ensure no goni is double-assigned (earlier rounds take precedence)
    seen = false(1, M);
    for k = 1:N
        if ~isnan(pair(k))
            g = pair(k);
            if seen(g)
                pair(k) = NaN; per_ttl_round(k) = 0;
            else
                seen(g) = true;
            end
        end
    end

    goni_idx_per_ttl  = pair;
    spurious_goni_idx = setdiff(1:M, unique(pair(~isnan(pair))));
    info = struct( ...
        'match_score',    sum(~isnan(pair)), ...
        'match_fraction', sum(~isnan(pair)) / N, ...
        'n_rounds',       length(rounds), ...
        'rounds',         rounds, ...
        'n_eeg_unmatched',sum(isnan(pair)), ...
        'n_goni_spurious',length(spurious_goni_idx));
end


function blocks = find_unmatched_blocks(mask)
    mask = mask(:).';
    d = diff([false, mask, false]);
    starts = find(d == 1);
    ends   = find(d == -1) - 1;
    blocks = repmat(struct('start', 0, 'end', 0, 'len', 0), 1, length(starts));
    for i = 1:length(starts)
        blocks(i).start = starts(i);
        blocks(i).end   = ends(i);
        blocks(i).len   = ends(i) - starts(i) + 1;
    end
end
