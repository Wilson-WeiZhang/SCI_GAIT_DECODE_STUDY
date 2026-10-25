function idx = find_goni_idx(goni_labels, target_label)
%FIND_GONI_IDX Find goni channel index by exact label match
%   idx = find_goni_idx(goni_labels, 'LKneW X') returns the index.
%   Tries exact match first, then trimmed match. Returns 0 if not found.

% Exact match
idx_found = find(strcmp(strtrim(goni_labels), strtrim(target_label)));
if ~isempty(idx_found)
    idx = idx_found(1);
    return;
end

% Fallback: case-insensitive
idx_found = find(strcmpi(strtrim(goni_labels), strtrim(target_label)));
if ~isempty(idx_found)
    idx = idx_found(1);
    return;
end

idx = 0;
end
