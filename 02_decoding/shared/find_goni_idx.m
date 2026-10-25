function idx = find_goni_idx(goni_labels, target_label)
%FIND_GONI_IDX Find goni channel index by exact label match.
%   idx = find_goni_idx(goni_labels, 'LKneW X')
%   Returns NaN (not 0) if the label is not present, so callers can
%   distinguish "channel missing in this session"  from "channel exists at index 0" (which never happens).
%   Tries exact match first (after strtrim), then case-insensitive.

idx_found = find(strcmp(strtrim(goni_labels), strtrim(target_label)));
if ~isempty(idx_found)
    idx = idx_found(1);
    return;
end

idx_found = find(strcmpi(strtrim(goni_labels), strtrim(target_label)));
if ~isempty(idx_found)
    idx = idx_found(1);
    return;
end

idx = NaN;
end
