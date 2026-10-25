function pairs = pair_events_chrono(on_idx, off_idx)
%PAIR_EVENTS_CHRONO  Chronological onset-offset pairing.
%   pairs = pair_events_chrono(on_idx, off_idx) returns an Nx2 array
%   [onset_event_idx, offset_event_idx]. For each onset, takes the earliest
%   offset that falls strictly before the next onset. Unmatched onsets are
%   dropped. Robust to missing offsets (Buffer Overflow, dropped triggers)
%   that would otherwise cascade-mispair a position-based zip.

    pairs = zeros(0, 2);
    if isempty(on_idx) || isempty(off_idx), return; end
    on_idx  = on_idx(:);
    off_idx = off_idx(:);
    next_on = [on_idx(2:end); Inf];
    for i = 1:length(on_idx)
        cand = off_idx(off_idx > on_idx(i) & off_idx < next_on(i));
        if ~isempty(cand)
            pairs(end+1, :) = [on_idx(i), cand(1)]; %#ok<AGROW>
        end
    end
end
