function mref_units(out_file)
%MREF_UNITS  MATLAB ground truth for the unit tests of the Python readout.
%
% Writes randperm and rand draws for fixed seeds, the rounded lag grids, MATLAB
% round() on halves, and train_wiener fits on two synthetic problems (one target all
% NaN, one partly NaN) to out_file (-v7); pyport/tests/test_units.py compares the
% Python helpers with these values. All inputs are synthetic. Run:
%   matlab -batch "addpath('<repository>/02_decoding/pyport/tests'); mref_units('mref_units.mat')"
% readout/train_wiener.m is used unchanged.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..', 'readout'));
R = struct();
seeds = [20260505 20261753 20260518 20260528 1 12345];
ns = [12 59 100 100000 7 3];
for i = 1:numel(seeds)
    rng(seeds(i)); R.(sprintf('perm_%d_%d', seeds(i), ns(i))) = randperm(ns(i));
end
rng(20260505); R.rand5 = rand(1, 5);
R.lag_round = unique(round((-2000:50:0) * 250 / 1000));
R.lag_round_25 = unique(round((-2000:25:0) * 250 / 1000));
R.round_halves = round([-2.5 -1.5 -0.5 0.5 1.5 2.5 -462.5 -12.5]);
% train_wiener: two synthetic problems (case A: joint 3 all NaN; case B: joint 2 partially NaN)
for c = 1:2
    rng(100 + c);
    P = 30; J = 3; N = 9;
    B = randn(P, J);
    X_cell = cell(1, N); Y_cell = cell(1, N);
    for k = 1:N
        T = 50 + randi(40);
        X = cumsum(randn(P, T), 2) * 0.1 + randn(P, T);
        Y = X' * B + 0.5 * randn(T, J);
        X_cell{k} = X; Y_cell{k} = Y;
    end
    if c == 1
        for k = 1:N, Y_cell{k}(:, 3) = NaN; end
    else
        Y_cell{4}(5:7, 2) = NaN;
    end
    grid = [1e-3 1e-2 0.1 1 10 100];
    m = train_wiener(X_cell, Y_cell, grid, 10);
    R.(sprintf('tw%d_X', c)) = X_cell;
    R.(sprintf('tw%d_Y', c)) = Y_cell;
    R.(sprintf('tw%d_W', c)) = m.W;
    R.(sprintf('tw%d_cv_mse', c)) = m.cv_mse;
    R.(sprintf('tw%d_lambda', c)) = m.lambda;
end
save(out_file, '-struct', 'R', '-v7');
fprintf('wrote %s\n', out_file);
end
