function [goni_idx_per_ttl, spurious_goni_idx, info] = match_offset_tracking( ...
        eeg_sec, goni_sec, init_offset_sec, tol_sec)
%MATCH_OFFSET_TRACKING  Sequential EEG->goni pairing with a running offset that
%   follows arbitrary clock drift/steps. The EEG/goni clock offset is NEVER
%   assumed constant (or piecewise-constant): after each matched pair the running
%   offset is updated to the median of the last few matches, so a clock step is
%   absorbed within a marker or two and matching does not lose sync.
%
%   Designed for recordings from the second acquisition system, whose clock steps
%   frequently (no single global offset fits more than a fraction of the run).
%   IOI residual is irrelevant for the analysis (events are extracted between
%   matched marker pairs); the only goal is to pair every real EEG marker with
%   its real goni edge in order.
%
%   eeg_sec          [1xN] sorted EEG TTL times (s)
%   goni_sec         [1xM] sorted goni edge times (s)
%   tol_sec          max |goni - predicted| to accept a pair. Pick > the largest
%                    expected clock step but < the smallest inter-marker gap
%                    (e.g. 1.2 s: covers sub-second steps, well below ~2.5 s gap).
%
%   goni_idx_per_ttl [1xN] goni edge index per EEG TTL (NaN if unmatched)
%   spurious_goni_idx goni edges matched to no EEG TTL
%   info             struct(match_score, match_fraction, seed_offset_sec)

    eeg_sec  = eeg_sec(:).';
    goni_sec = goni_sec(:).';
    N = numel(eeg_sec);
    M = numel(goni_sec);

    % Try a few seed offsets around the align_ioi estimate; that estimate can be
    % off by a step or two because no single offset fits these sessions well.
    seeds = init_offset_sec + [-0.6 -0.3 -0.1 0 0.1 0.3 0.6];
    best_pair = nan(1, N);
    best_score = -1;
    best_seed = init_offset_sec;
    for s = 1:numel(seeds)
        [pr, sc] = track_once(eeg_sec, goni_sec, seeds(s), tol_sec, N, M);
        if sc > best_score
            best_score = sc; best_pair = pr; best_seed = seeds(s);
        end
    end

    goni_idx_per_ttl  = best_pair;
    matched_goni      = best_pair(~isnan(best_pair));
    spurious_goni_idx = setdiff(1:M, matched_goni);
    info = struct('match_score', best_score, ...
                  'match_fraction', best_score / max(N,1), ...
                  'seed_offset_sec', best_seed);
end


function [pair, score] = track_once(eeg_sec, goni_sec, O0, tol_sec, N, M)
    pair   = nan(1, N);
    used   = false(1, M);
    O      = O0;
    recent = [];
    for i = 1:N
        cand = find(~used);
        if isempty(cand), break; end
        target = eeg_sec(i) + O;
        [dmin, ci] = min(abs(goni_sec(cand) - target));
        j = cand(ci);
        if dmin <= tol_sec
            pair(i) = j;
            used(j) = true;
            recent(end+1) = goni_sec(j) - eeg_sec(i); %#ok<AGROW>
            if numel(recent) > 3, recent(1) = []; end
            O = median(recent);     % running offset follows drift/steps
        end
        % else: EEG marker i has no goni within tol (missing edge); keep O,
        % leave unmatched. The next inter-marker gap (> tol) prevents grabbing
        % a neighbour's edge.
    end
    score = sum(~isnan(pair));
end
