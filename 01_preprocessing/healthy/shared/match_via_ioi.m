function [goni_idx_per_ttl, spurious_goni_idx, info] = match_via_ioi( ...
        eeg_sec, goni_sec, initial_offset_sec, opts)
%MATCH_VIA_IOI  IOI-pattern-based EEG→Goni pairing; identifies spurious goni edges.
%
%   Algorithm (greedy forward, IOI pattern match):
%     1. j_start = goni edge nearest to (eeg_sec(1) + initial_offset_sec)
%     2. pair(1) = j_start
%     3. For i = 2..N:
%          target_ioi = eeg_sec(i) - eeg_sec(i-1)
%          Scan cumulative goni IOI from j_prev forward (up to max_skip steps):
%          cum = 0, k = 0
%          while k < max_skip and (j_prev+k+1) <= M:
%              k = k + 1
%              cum = cum + (goni_sec(j_prev+k) - goni_sec(j_prev+k-1))
%              if |cum - target_ioi| < tol: pair(i) = j_prev+k; break
%              if cum > target_ioi + tol: break   # overshoot -> EEG-side spurious
%     4. Spurious goni = all edges not in pair
%
%   opts:
%     tol_ms       IOI match tolerance in ms (default 300). The wide tolerance rejects
%                  false matches at a clock transition while still covering real drift,
%                  which is of the order of milliseconds per IOI.
%     max_skip     max goni edges to skip per EEG IOI (default 5)
%     max_i_skip   max leading EEG markers that may be skipped when choosing the
%                  starting pair (default 3)
%     j_start_try  optional explicit candidates for the starting goni index; when it is
%                  absent, the five edges nearest the initial offset are tried
%
%   Returns:
%     goni_idx_per_ttl   [1xN] goni edge index for each EEG TTL (NaN if unmatched)
%     spurious_goni_idx  indices of goni edges not matched to any EEG
%     info               struct with match_score, j_start_chosen, n_eeg_spurious

    if nargin < 4, opts = struct(); end
    if ~isfield(opts, 'tol_ms'),       opts.tol_ms = 300; end  % best-fit scan; 300ms rejects false transition matches (SUB_16 TTL30→goni29 = 556ms) while still covering real clock drift (per-IOI drift is ms-scale)
    if ~isfield(opts, 'max_skip'),     opts.max_skip = 5; end
    if ~isfield(opts, 'max_i_skip'),   opts.max_i_skip = 3; end  % allow skipping first EEG TTLs
    tol_sec = opts.tol_ms / 1000;

    eeg_sec  = eeg_sec(:).';
    goni_sec = goni_sec(:).';
    N = length(eeg_sec);
    M = length(goni_sec);

    best = struct('pair', nan(1, N), 'score', -inf, 'j_start', NaN, 'i_start', NaN);
    for i_start = 1:opts.max_i_skip+1
        target = eeg_sec(i_start) + initial_offset_sec;
        [~, j_nearest] = min(abs(goni_sec - target));
        if isfield(opts, 'j_start_try')
            js = opts.j_start_try;
        else
            js = unique([max(1, j_nearest-2), max(1, j_nearest-1), j_nearest, ...
                         min(M, j_nearest+1), min(M, j_nearest+2)]);
        end
        for jj = 1:length(js)
            j_start = js(jj);
            pair = greedy_forward(eeg_sec, goni_sec, i_start, j_start, tol_sec, opts.max_skip);
            score = sum(~isnan(pair));
            if score > best.score
                best.pair = pair; best.score = score;
                best.j_start = j_start; best.i_start = i_start;
            end
        end
    end

    goni_idx_per_ttl = best.pair;
    matched_goni = best.pair(~isnan(best.pair));
    spurious_goni_idx = setdiff(1:M, matched_goni);

    info = struct( ...
        'match_score',       best.score, ...
        'match_fraction',    best.score / N, ...
        'i_start_chosen',    best.i_start, ...
        'j_start_chosen',    best.j_start, ...
        'n_eeg_unmatched',   N - best.score, ...
        'n_goni_spurious',   length(spurious_goni_idx));
end


function pair = greedy_forward(eeg_sec, goni_sec, i_start, j_start, tol_sec, max_skip)
    N = length(eeg_sec); M = length(goni_sec);
    pair = nan(1, N);
    pair(i_start) = j_start;
    j_prev = j_start; i_prev = i_start;
    for i = (i_start+1):N
        target_ioi = eeg_sec(i) - eeg_sec(i_prev);
        % Best-fit scan: find k that minimizes |cum - target_ioi|
        cum = 0; best_k = 0; best_diff = inf;
        for k = 1:max_skip
            if j_prev + k > M, break; end
            cum = cum + (goni_sec(j_prev + k) - goni_sec(j_prev + k - 1));
            d = abs(cum - target_ioi);
            if d < best_diff
                best_diff = d;
                best_k = k;
            end
            if cum > target_ioi + 2*tol_sec
                break;  % no point scanning further, already way overshot
            end
        end
        if best_k > 0 && best_diff < tol_sec
            pair(i) = j_prev + best_k;
            j_prev = j_prev + best_k;
            i_prev = i;
        end
    end
end
