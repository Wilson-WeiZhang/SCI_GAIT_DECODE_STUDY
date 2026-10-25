function edges_samples = detect_stim_edges(stim_channel, threshold)
%DETECT_STIM_EDGES  Find rising edges in a binary/stim channel.
%
%   edges_samples = DETECT_STIM_EDGES(stim, threshold)
%
%   Input:
%     stim_channel  1 x N numeric vector (0/1 or continuous analog)
%     threshold     scalar; samples where signal crosses from <=thr to >thr
%                   default 0.5
%
%   Output:
%     edges_samples column vector of rising-edge sample indices (1-based).

    if nargin < 2, threshold = 0.5; end
    x = stim_channel(:)';
    above = x > threshold;
    edges_samples = find(above(2:end) & ~above(1:end-1)) + 1;
    edges_samples = edges_samples(:);
end
