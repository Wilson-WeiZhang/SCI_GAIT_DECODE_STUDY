function [kept_samples, dropped_idx] = iei_dedup(edges_samples, min_iei_samples)
%IEI_DEDUP  Drop spurious burst edges with inter-event-interval below threshold.
%
%   Keeps first edge, drops subsequent edges within min_iei_samples of previous kept.
%
%   Input:
%     edges_samples    [N x 1] rising edge sample indices (sorted ascending)
%     min_iei_samples  scalar; edges closer than this to previous kept are dropped
%
%   Output:
%     kept_samples     [M x 1] kept edges
%     dropped_idx      [K x 1] indices into original edges_samples that were dropped

    edges_samples = edges_samples(:);
    n = length(edges_samples);
    if n == 0
        kept_samples = []; dropped_idx = []; return;
    end
    keep = true(n, 1);
    last_kept = edges_samples(1);
    for i = 2:n
        if edges_samples(i) - last_kept < min_iei_samples
            keep(i) = false;
        else
            last_kept = edges_samples(i);
        end
    end
    kept_samples = edges_samples(keep);
    dropped_idx  = find(~keep);
end
