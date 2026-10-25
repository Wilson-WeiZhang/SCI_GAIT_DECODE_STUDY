function model = train_wiener(X_cell, Y_cell, lambda_grid, inner_K)
%TRAIN_WIENER Ridge regression with delay embedding, λ chosen by inner CV.
%   model = train_wiener(X_cell, Y_cell, lambda_grid, inner_K)
%   X_cell {1 x N} of [P x T_k]  delay-embedded EEG (output of build_xy)
%   Y_cell {1 x N} of [T_k x J]  z-scored joint targets
%   lambda_grid 1 x L_λ vector, e.g. [1e-3 1e-2 1e-1 1 10 100]
%   inner_K scalar inner-CV fold count
%
%   model.W       [P+1 x J]  (ridge weights, last row = intercept)
%   model.lambda  scalar
%   model.cv_mse  L_λ x 1  validation MSE per λ across inner folds
%   model.P       feature dim
%   model.J       joint count

% Concat all train trials → [P x N_total], [N_total x J]
P = size(X_cell{1}, 1);
J = size(Y_cell{1}, 2);
N = numel(X_cell);
if nargin < 4 || isempty(inner_K)
    inner_K = 5;
end
inner_K = max(1, round(double(inner_K)));

X_all = horzcat(X_cell{:});       % [P x N_total]
Y_all = vertcat(Y_cell{:});       % [N_total x J]
N_total = size(X_all, 2);
assert(size(Y_all, 1) == N_total, 'X and Y trial-sample counts mismatch');

% Drop columns of Y that are all-NaN (missing joints) before fitting; keep
% NaN sentinel in W output to preserve column index alignment.
joint_active = false(1, J);
for j = 1:J
    joint_active(j) = ~all(isnan(Y_all(:, j)));
end

% Inner CV over λ — random K-fold on samples
rng(20260505);
inner_K = min(inner_K, N_total);
perm = randperm(N_total);
inner_fold = zeros(N_total, 1);
for k = 1:inner_K
    inner_fold(perm(k:inner_K:end)) = k;
end

L_lam = numel(lambda_grid);
cv_mse = zeros(L_lam, 1);

% Add intercept row
X_aug = [X_all; ones(1, N_total)];     % [P+1 x N_total]
P1 = P + 1;
ridge_eye = eye(P1);
ridge_eye(end, end) = 0;

% Outer over inner folds, inner over λ — the per-fold gram matrices are
% independent of λ, so compute them once and reuse for all λ.
sse_per_lam = zeros(L_lam, 1);
n_per_lam   = zeros(L_lam, 1);
for k = 1:inner_K
    tr = inner_fold ~= k;
    te = inner_fold == k;
    Xt = X_aug(:, tr);
    Yt = Y_all(tr, :);
    Xe = X_aug(:, te);
    Ye = Y_all(te, :);
    A0 = Xt * Xt';      % [P1 x P1]   independent of λ
    XY = Xt * Yt;       % [P1 x J]    independent of λ
    for li = 1:L_lam
        A = A0 + lambda_grid(li) * ridge_eye;
        W = A \ XY;
        Ypred = Xe' * W;
        for j = find(joint_active)
            r = Ypred(:, j) - Ye(:, j);
            sse_per_lam(li) = sse_per_lam(li) + sum(r.^2);
            n_per_lam(li)   = n_per_lam(li)   + numel(r);
        end
    end
end
cv_mse = sse_per_lam ./ n_per_lam;

% Pick best λ, refit on full train
[~, best_li] = min(cv_mse);
lambda = lambda_grid(best_li);
A = X_aug * X_aug' + lambda * ridge_eye;
B = X_aug * Y_all;
W = A \ B;

model = struct();
model.W = W;
model.lambda = lambda;
model.cv_mse = cv_mse;
model.lambda_grid = lambda_grid;
model.P = P;
model.J = J;
model.joint_active = joint_active;
model.cv_inner_k = inner_K;
end
