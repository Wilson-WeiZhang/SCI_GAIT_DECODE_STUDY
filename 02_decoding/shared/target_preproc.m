function Y = target_preproc(joint_traj)
%TARGET_PREPROC Per-trial z-score on joint angle, NaN columns pass through.
%   Y = target_preproc(joint_traj)
%   joint_traj : [T x J]; columns that are entirely NaN (missing joint) stay NaN. Columns with zero variance
%                map to 0 (no orientation to z-score).

[T, J] = size(joint_traj);
Y = nan(T, J);
for j = 1:J
    col = joint_traj(:, j);
    if all(isnan(col))
        continue;
    end
    mu = mean(col, 'omitnan');
    sd = std(col,  'omitnan');
    if sd > 0
        Y(:, j) = (col - mu) ./ sd;
    else
        Y(:, j) = 0;
    end
end
end
