function model = train_wiener_rrr(X_cell, Y_cell, lambda_grid, inner_K, rank_grid)
%TRAIN_WIENER_RRR Reduced-rank ridge readout with inner-CV model selection.
%   model = train_wiener_rrr(X_cell, Y_cell, lambda_grid, inner_K, rank_grid)
%   keeps the same X/Y cell interface and W output contract as train_wiener.

P = size(X_cell{1}, 1);
J = size(Y_cell{1}, 2);
if nargin < 4 || isempty(inner_K)
    inner_K = 5;
end
inner_K = max(1, round(double(inner_K)));

lambda_grid = unique(double(lambda_grid(:)'));
lambda_grid = lambda_grid(isfinite(lambda_grid) & lambda_grid >= 0);
if isempty(lambda_grid)
    lambda_grid = 0.001;
end

X_all = horzcat(X_cell{:});
Y_all = vertcat(Y_cell{:});
N_total = size(X_all, 2);
assert(size(Y_all, 1) == N_total, 'X and Y trial-sample counts mismatch');

joint_active = false(1, J);
for j = 1:J
    joint_active(j) = ~all(isnan(Y_all(:, j)));
end
active_cols = find(joint_active);
if isempty(active_cols)
    model = empty_rrr_model_local(P, J, lambda_grid(1), 0, joint_active);
    return;
end

Y_active = Y_all(:, active_cols);
valid = all(isfinite(Y_active), 2) & all(isfinite(X_all), 1)';
X_fit = X_all(:, valid);
Y_fit = Y_active(valid, :);
N_fit = size(X_fit, 2);
assert(size(Y_fit, 1) == N_fit, 'valid X and Y sample counts mismatch');
if N_fit < 2
    model = empty_rrr_model_local(P, J, lambda_grid(1), 0, joint_active);
    return;
end

J_active = size(Y_fit, 2);
if nargin < 5 || isempty(rank_grid)
    env_rank_grid = getenv('RRR_RANK_GRID');
    if ~isempty(env_rank_grid)
        rank_grid = str2num(env_rank_grid); %#ok<ST2NM>
    else
        rank_grid = 1:J_active;
    end
end
rank_grid = unique(round(double(rank_grid(:)')));
rank_grid = rank_grid(isfinite(rank_grid) & rank_grid >= 1);
rank_grid = rank_grid(rank_grid <= J_active);
if isempty(rank_grid)
    rank_grid = 1:J_active;
end

X_aug = [X_fit; ones(1, N_fit)];
P1 = P + 1;
ridge_eye = eye(P1);
ridge_eye(end, end) = 0;

L_lam = numel(lambda_grid);
L_rank = numel(rank_grid);
cv_mse = NaN(L_lam, L_rank);

if inner_K <= 1 || N_fit < 4
    for li = 1:L_lam
        for ri = 1:L_rank
            W_active = fit_rrr_weights_local(X_aug, Y_fit, lambda_grid(li), rank_grid(ri), ridge_eye);
            resid = X_aug' * W_active - Y_fit;
            cv_mse(li, ri) = mean(resid(:).^2);
        end
    end
else
    rng(20260604);
    inner_K = min(inner_K, N_fit);
    perm = randperm(N_fit);
    inner_fold = zeros(N_fit, 1);
    for k = 1:inner_K
        inner_fold(perm(k:inner_K:end)) = k;
    end

    sse = zeros(L_lam, L_rank);
    n = zeros(L_lam, L_rank);
    for k = 1:inner_K
        tr = inner_fold ~= k;
        te = inner_fold == k;
        Xt = X_aug(:, tr);
        Yt = Y_fit(tr, :);
        Xe = X_aug(:, te);
        Ye = Y_fit(te, :);
        for li = 1:L_lam
            for ri = 1:L_rank
                W_active = fit_rrr_weights_local(Xt, Yt, lambda_grid(li), rank_grid(ri), ridge_eye);
                resid = Xe' * W_active - Ye;
                sse(li, ri) = sse(li, ri) + sum(resid(:).^2);
                n(li, ri) = n(li, ri) + numel(resid);
            end
        end
    end
    cv_mse = sse ./ n;
end

[~, best_idx] = min(cv_mse(:));
[best_li, best_ri] = ind2sub(size(cv_mse), best_idx);
lambda = lambda_grid(best_li);
rank = rank_grid(best_ri);
W_active = fit_rrr_weights_local(X_aug, Y_fit, lambda, rank, ridge_eye);

W = NaN(P1, J);
W(:, active_cols) = W_active;

model = struct();
model.W = W;
model.lambda = lambda;
model.rank = rank;
model.cv_mse = cv_mse;
model.lambda_grid = lambda_grid;
model.rank_grid = rank_grid;
model.P = P;
model.J = J;
model.joint_active = joint_active;
model.cv_inner_k = inner_K;
model.lambda_mode = 'rrr';
end

function W = fit_rrr_weights_local(X_aug, Y, lambda, rank, ridge_eye)
A = X_aug * X_aug' + double(lambda) * ridge_eye;
XY = X_aug * Y;
W_full = A \ XY;
Y_hat = X_aug' * W_full;
Y_hat = Y_hat - mean(Y_hat, 1, 'omitnan');
[~, ~, V] = svd(Y_hat, 'econ');
if isempty(V)
    W = W_full;
    return;
end
r = min([round(double(rank)), size(V, 2), size(Y, 2)]);
V_r = V(:, 1:r);
Y_score = Y * V_r;
W_score = A \ (X_aug * Y_score);
W = W_score * V_r';
end

function model = empty_rrr_model_local(P, J, lambda, rank, joint_active)
model = struct();
model.W = NaN(P + 1, J);
model.lambda = double(lambda);
model.rank = double(rank);
model.cv_mse = NaN;
model.lambda_grid = double(lambda);
model.rank_grid = double(rank);
model.P = P;
model.J = J;
model.joint_active = joint_active;
model.cv_inner_k = 0;
model.lambda_mode = 'rrr';
end
