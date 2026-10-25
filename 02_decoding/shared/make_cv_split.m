function fa = make_cv_split(label, condition, K)
%MAKE_CV_SPLIT Deterministic trial-level CV per session per condition.
%   fa = make_cv_split('SUB_07_sess01', 'MI')
%   fa = make_cv_split('SUB_07_sess01', 'MI', 10)
%   Returns struct fa with:
%     .session   echoed label
%     .condition 'MI' | 'EXEC'
%     .seed      seed used (a fixed base constant plus a hash of the label plus a
%                condition tag; the seed is a fixed constant, its value carries no
%                meaning, and changing it would change every fold assignment)
%     .n_trials  number of qc=good trials of this condition in this session
%     .fold      [n_trials x 1] int in 1..K (or 1..n if n < K)
%
%   Same input -> same fold assignment. 

if nargin < 3 || isempty(K)
    K = 5;
end
K = round(double(K));
assert(isfinite(K) && isscalar(K) && K >= 1, 'K must be a positive scalar');

% Coerce string scalars to char so condition(1) gives a single char (not the
% whole string scalar — MATLAB string indexing differs from char indexing).
label     = char(label);
condition = char(condition);

S = load_triple_mat(label);
ev = trial_filter(S.events, condition);
n  = numel(ev);

cond_tag = double(condition(1));        % 'M' or 'E'
seed_base = 20260505;
label_hash = sum(double(label));
rng(seed_base + label_hash + cond_tag);

perm = randperm(n);
fold = zeros(n, 1);
Keff = min(K, n);                       % handle tiny edge case
for k = 1:Keff
    fold(perm(k:Keff:end)) = k;
end

fa = struct( ...
    'session',  label, ...
    'condition', condition, ...
    'seed',     seed_base + label_hash + cond_tag, ...
    'n_trials', n, ...
    'fold',     fold);
end
