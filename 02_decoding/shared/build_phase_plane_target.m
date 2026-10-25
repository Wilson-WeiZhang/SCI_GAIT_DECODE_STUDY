function [Y_cell, info] = build_phase_plane_target(Y_raw_cells, opt, fs)
%BUILD_PHASE_PLANE_TARGET Convert raw joint angles to Borghese-plane phasors.
%
%   [Y_cell, info] = build_phase_plane_target(Y_raw_cells, opt, fs)
%
%   Input Y_raw_cells is the raw target output from build_xy(... target_mode='raw'),
%   one [T x 6] matrix per trial. The function fits one session-pooled
%   elevation-angle SVD plane, applies a stable anatomical sign convention, and
%   returns [cos(phi), sin(phi)] per trial.
%
%   Selected by run_decoder with phase_target_mode = 'phase_plane'. It is one of the
%   alternative target representations the decoder can be run against; the published
%   results use the per-joint phase target, phase_target_mode = 'phase_pj'.

if nargin < 2 || isempty(opt), opt = struct(); end
if nargin < 3 || isempty(fs), fs = 250; end
if ~isfield(opt, 'phase_lo'), opt.phase_lo = 0.3; end
if ~isfield(opt, 'phase_hi'), opt.phase_hi = 2.0; end

[basis, n_link, valid, explained] = fit_session_plane(Y_raw_cells, opt, fs);

Y_cell = cell(size(Y_raw_cells));
for k = 1:numel(Y_raw_cells)
    y = Y_raw_cells{k};
    if isempty(y)
        Y_cell{k} = zeros(0, 2);
    elseif ~valid
        Y_cell{k} = nan(size(y, 1), 2);
    else
        Y_cell{k} = apply_plane_phasor(y, basis, n_link, opt, fs);
    end
end

info = struct('method', 'M1_phase_plane', ...
              'n_link', n_link, ...
              'valid', valid, ...
              'explained12', explained, ...
              'basis', basis);
end


function [basis, n_link, valid, explained] = fit_session_plane(Y_raw_cells, opt, fs)
% Session-pooled SVD plane basis on elevation angles.
basis = [];
n_link = 0;
valid = false;
explained = NaN;

first_y = [];
for k = 1:numel(Y_raw_cells)
    if ~isempty(Y_raw_cells{k}) && size(Y_raw_cells{k}, 1) > 0
        first_y = Y_raw_cells{k};
        break;
    end
end
if isempty(first_y), return; end

type_a = size(first_y, 2) >= 3 && all(isnan(first_y(:, 3)));
if type_a
    n_link = 5;
else
    n_link = 6;
end

all_E = zeros(0, n_link);
for k = 1:numel(Y_raw_cells)
    y = Y_raw_cells{k};
    if isempty(y), continue; end
    e = compute_elevation(y);
    if size(e, 2) ~= n_link, continue; end
    all_E = [all_E; e]; %#ok<AGROW>
end
if size(all_E, 1) < 100, return; end

[zf, pf, kf] = butter(2, [opt.phase_lo opt.phase_hi] / (fs/2), 'bandpass');
sos = zp2sos(zf, pf, kf);
for c = 1:n_link
    col = all_E(:, c);
    if any(isnan(col)) || std(col, 'omitnan') < 1e-9, continue; end
    all_E(:, c) = filtfilt(sos, 1, col);
end
all_E = all_E - mean(all_E, 1, 'omitnan');
all_E(isnan(all_E)) = 0;

[~, S, V] = svd(all_E, 'econ');
basis = V(:, 1:2);
sing = diag(S);
if numel(sing) >= 2
    explained = sum(sing(1:2).^2) / sum(sing.^2);
end

% elevation, making the plane frame comparable across sessions.
if n_link == 6
    rhip_col = 4;
else
    rhip_col = 3;
end
score_pc1 = all_E * basis(:, 1);
rhip_signal = all_E(:, rhip_col);
if std(rhip_signal) > 1e-9
    sign_pc1 = sign(corr(score_pc1, rhip_signal));
    if sign_pc1 == 0 || isnan(sign_pc1), sign_pc1 = 1; end
    basis(:, 1) = basis(:, 1) * sign_pc1;
end

scores = all_E * basis;
phi = unwrap(atan2(scores(:, 2), scores(:, 1)));
slope = (phi(end) - phi(1)) / numel(phi);
if slope < 0
    basis(:, 2) = -basis(:, 2);
end

valid = true;
end


function E = compute_elevation(Y_raw)
J = size(Y_raw, 2);
type_a = J >= 3 && all(isnan(Y_raw(:, 3)));
T = size(Y_raw, 1);
if type_a
    E = nan(T, 5);
    E(:, 1) = Y_raw(:, 1);
    E(:, 2) = Y_raw(:, 1) - Y_raw(:, 2);
    E(:, 3) = Y_raw(:, 4);
    E(:, 4) = Y_raw(:, 4) - Y_raw(:, 5);
    E(:, 5) = Y_raw(:, 4) - Y_raw(:, 5) - Y_raw(:, 6);
else
    E = nan(T, 6);
    E(:, 1) = Y_raw(:, 1);
    E(:, 2) = Y_raw(:, 1) - Y_raw(:, 2);
    E(:, 3) = Y_raw(:, 1) - Y_raw(:, 2) - Y_raw(:, 3);
    E(:, 4) = Y_raw(:, 4);
    E(:, 5) = Y_raw(:, 4) - Y_raw(:, 5);
    E(:, 6) = Y_raw(:, 4) - Y_raw(:, 5) - Y_raw(:, 6);
end
end


function Y_phasor = apply_plane_phasor(Y_raw, basis, n_link, opt, fs)
if isempty(Y_raw) || isempty(basis)
    Y_phasor = nan(size(Y_raw, 1), 2);
    return;
end

E = compute_elevation(Y_raw);
if size(E, 2) ~= n_link
    Y_phasor = nan(size(Y_raw, 1), 2);
    return;
end

[zf, pf, kf] = butter(2, [opt.phase_lo opt.phase_hi] / (fs/2), 'bandpass');
sos = zp2sos(zf, pf, kf);
for c = 1:size(E, 2)
    col = E(:, c);
    if any(isnan(col)) || std(col, 'omitnan') < 1e-9, continue; end
    E(:, c) = filtfilt(sos, 1, col);
end
E = E - mean(E, 1, 'omitnan');
E(isnan(E)) = 0;

scores = E * basis;
amp = sqrt(scores(:, 1).^2 + scores(:, 2).^2) + 1e-9;
Y_phasor = [scores(:, 1) ./ amp, scores(:, 2) ./ amp];
end
