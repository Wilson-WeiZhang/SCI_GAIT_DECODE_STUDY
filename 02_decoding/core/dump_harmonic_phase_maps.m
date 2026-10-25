function dump_harmonic_phase_maps(varargin)
%DUMP_HARMONIC_PHASE_MAPS  Scalp maps of gait-cycle harmonic phase, per session and per group.
%
% For each session in the manifest the EEG is band-passed in that session's band
% for each requested harmonic (read from band_csv), the analytic signal is formed,
% and the complex values are averaged within n_cycle_points equally spaced bins of
% the goniometer cycle phase of the reference joint given by goni_label. The result
% is one complex scalp map per session, harmonic and cycle point.
%
% Sessions are then split into two groups at a threshold on their decoding score,
% read from session_r_csv (NaN = split at the median of the available scores), and
% each group's maps are rendered on ONE shared reference phase and ONE shared
% colour limit, so a group with weaker phase locking visibly washes out.
%
% Parameters (name/value): manifest_path, band_csv, session_r_csv, out_dir,
% condition, harmonic_list, goni_label, lag_ms, n_cycle_points, eeglab_root,
% electrodes, split_threshold, max_threads, max_sessions.
%
% Output in out_dir:
%   persession_harmonic_phase_cycle9_complex.csv   per session, harmonic, cycle
%                                                  point and channel
%   persession_harmonic_contrast_vs_r.csv          per-session contrast magnitude
%                                                  joined to the decoding score
%   shared_reference.csv                           the shared reference phase and
%                                                  colour limit
%   group_contrast_magnitude_summary.csv           per group and harmonic
%   harmonic_phase_cycle9_*_sharedscale.png        the rendered group maps
%
% The per-session csv is the input of the Figure 3 and Figure 4 renderers.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'shared'));

p = inputParser;
p.addParameter('manifest_path', fullfile(root, 'result', 'session_manifest.csv'));
p.addParameter('band_csv', fullfile(root, 'result', 'diag_goniometer_psd_peaks', 'goniometer_peak_bands.csv'));
p.addParameter('session_r_csv', fullfile(root, 'result', 'session_r.csv'));
p.addParameter('out_dir', fullfile(root, 'result', 'diag_harmonic_phase_responder_split'));
p.addParameter('condition', 'MI');
p.addParameter('harmonic_list', [1 2 3]);
p.addParameter('goni_label', 'LHipW X');
p.addParameter('lag_ms', 0);
p.addParameter('n_cycle_points', 9);
p.addParameter('eeglab_root', '${GAIT_DATA_ROOT}/eeglab2024');
p.addParameter('electrodes', 'on');
p.addParameter('split_threshold', NaN);   % NaN -> median of available r
p.addParameter('max_threads', 8);
p.addParameter('max_sessions', Inf);
p.parse(varargin{:});
opt = p.Results;

try
    maxNumCompThreads(opt.max_threads);
catch
end

assert(opt.n_cycle_points >= 3, 'n_cycle_points must be >= 3');
if ~exist(opt.out_dir, 'dir')
    mkdir(opt.out_dir);
end
ensure_eeglab_topoplot_local(opt.eeglab_root);

assert(isfile(opt.band_csv), 'Missing band CSV: %s', opt.band_csv);
band_tbl = readtable(opt.band_csv, 'TextType', 'string', 'Delimiter', ',');

sessions = read_session_list_local(opt, band_tbl);
if isfinite(opt.max_sessions)
    sessions = sessions(1:min(numel(sessions), opt.max_sessions));
end
assert(~isempty(sessions), 'No sessions to process');

harmonics = double(opt.harmonic_list(:)');
cycle_fraction = linspace(0, 1, opt.n_cycle_points);
cycle_phase = cycle_fraction * 2 * pi;
half_width = pi / (opt.n_cycle_points - 1);
lag_samples = [];
chan_labels = strings(0, 1);
chanlocs = [];
phase_stack = [];
kept_sessions = strings(0, 1);
session_rows = table('Size', [0 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'session', 'harmonic_k', 'n_trials', 'n_valid_samples', 'min_cycle_point_samples'});

for si = 1:numel(sessions)
    session = sessions(si);
    fprintf('[%d/%d] %s\n', si, numel(sessions), session);
    S = load_triple_mat(char(session));
    ev = trial_filter(S.events, opt.condition);
    if isempty(ev)
        warning('No %s events in %s', opt.condition, session);
        continue;
    end
    if isempty(chan_labels)
        chan_labels = string(S.eeg.labels(:));
        chanlocs = get_chanlocs_local(S);
        lag_samples = round(opt.lag_ms * S.eeg.fs / 1000);
    else
        assert(isequal(chan_labels, string(S.eeg.labels(:))), 'Channel labels mismatch in %s', session);
    end

    [phase_map, count_rows] = session_cycle_maps_local( ...
        S, ev, band_tbl, harmonics, cycle_phase, cycle_fraction, half_width, lag_samples, opt);
    if isempty(phase_stack)
        phase_stack = phase_map;
    else
        phase_stack = cat(4, phase_stack, phase_map);
    end
    kept_sessions(end + 1, 1) = string(S.label); %#ok<AGROW>
    session_rows = [session_rows; count_rows]; %#ok<AGROW>
end
assert(~isempty(phase_stack), 'No session produced maps');
n_kept = numel(kept_sessions);

% ---- per-session contrast magnitude + r join ----
r_map = read_session_r_local(opt.session_r_csv);
sess_r = nan(n_kept, 1);
for si = 1:n_kept
    if isKey(r_map, char(kept_sessions(si)))
        sess_r(si) = r_map(char(kept_sessions(si)));
    end
end

thr = opt.split_threshold;
if ~isfinite(thr)
    thr = median(sess_r(isfinite(sess_r)));
end
high_mask = isfinite(sess_r) & sess_r >= thr;
low_mask  = isfinite(sess_r) & sess_r <  thr;
fprintf('split_threshold=%.4f  high=%d  low=%d  no_r=%d\n', ...
    thr, nnz(high_mask), nnz(low_mask), nnz(~isfinite(sess_r)));

% per-session mean |c| over channels x cycle points, per harmonic
n_h = numel(harmonics);
persess = table('Size', [0 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'string'}, ...
    'VariableNames', {'session', 'harmonic_k', 'mean_abs', 'r', 'group'});
for si = 1:n_kept
    if isfinite(sess_r(si))
        if sess_r(si) >= thr, gtag = "responder"; else, gtag = "nonresponder"; end
    else
        gtag = "no_r";
    end
    for hi = 1:n_h
        slab = phase_stack(:, :, hi, si);
        ma = mean(abs(slab(:)), 'omitnan');
        persess = [persess; {kept_sessions(si), harmonics(hi), ma, sess_r(si), gtag}]; %#ok<AGROW>
    end
end
writetable(persess, fullfile(opt.out_dir, 'persession_harmonic_contrast_vs_r.csv'));

% per-session complex dump (for local re-splitting)
write_persession_complex_dump_local( ...
    fullfile(opt.out_dir, 'persession_harmonic_phase_cycle9_complex.csv'), ...
    phase_stack, kept_sessions, chan_labels, cycle_fraction, harmonics);

% ---- shared reference + clim from full cohort (sessions with r) ----
use_mask = isfinite(sess_r);
full_complex = group_mean_complex_local(phase_stack, use_mask);
[~, ref_phase, ref_row] = unified_reference_projection_local(full_complex, chan_labels, cycle_fraction, harmonics);
full_signed = real(full_complex * exp(-1i * ref_phase));
map_lim = common_limit_local(full_signed);
writetable(ref_row, fullfile(opt.out_dir, 'shared_reference.csv'));

% ---- group maps + render + magnitude summary ----
group_defs = { 'responder', high_mask, 'highr'; 'nonresponder', low_mask, 'lowr' };
gsum = table('Size', [0 6], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'group', 'harmonic_k', 'n_sessions', 'n_trials', 'group_map_mean_abs', 'mean_r'});
for gi = 1:size(group_defs, 1)
    gname = group_defs{gi, 1};
    gmask = group_defs{gi, 2};
    gshort = group_defs{gi, 3};
    gcomplex = group_mean_complex_local(phase_stack, gmask);
    gsigned = real(gcomplex * exp(-1i * ref_phase));

    row_titles = strings(n_h, 1);
    for hi = 1:n_h
        rows_h = session_rows(session_rows.harmonic_k == harmonics(hi), :);
        % restrict trial counts to this group's sessions
        gs = kept_sessions(gmask);
        in_g = ismember(string(rows_h.session), gs);
        nT = sum(rows_h.n_trials(in_g));
        gmap_abs = mean(abs(reshape(gcomplex(:, :, hi), [], 1)), 'omitnan');
        row_titles(hi) = sprintf('k%d nS=%d nT=%d |c|=%.4f', harmonics(hi), nnz(gmask), nT, gmap_abs);
        gsum = [gsum; {string(gname), harmonics(hi), nnz(gmask), nT, gmap_abs, mean(sess_r(gmask), 'omitnan')}]; %#ok<AGROW>
    end

    out_png = fullfile(opt.out_dir, sprintf('harmonic_phase_cycle9_%s_%s_sharedscale.png', gshort, gname));
    plot_cycle_grid_local(out_png, gsigned, chanlocs, cycle_fraction, row_titles, map_lim, opt, gname, thr);
    write_harmonic_complex_csv_local( ...
        fullfile(opt.out_dir, sprintf('harmonic_phase_cycle9_%s_%s_complex.csv', gshort, gname)), ...
        gcomplex, chan_labels, cycle_fraction, harmonics);
    fprintf('Wrote %s\n', out_png);
end
writetable(gsum, fullfile(opt.out_dir, 'group_contrast_magnitude_summary.csv'));

fprintf('shared color limit = [%.6g %.6g]\n', map_lim(1), map_lim(2));
fprintf('DONE group split. out_dir=%s\n', opt.out_dir);
end

% ===================== helpers  =====================

function r_map = read_session_r_local(path)
r_map = containers.Map('KeyType', 'char', 'ValueType', 'double');
assert(isfile(path), 'Missing session_r_csv: %s', path);
T = readtable(path, 'TextType', 'string', 'Delimiter', ',');
for ri = 1:height(T)
    r_map(char(string(T.session(ri)))) = double(T.r(ri));
end
end

function M = group_mean_complex_local(phase_stack, mask)
sub = phase_stack(:, :, :, mask);
M = mean(real(sub), 4, 'omitnan') + 1i * mean(imag(sub), 4, 'omitnan');
end

function write_persession_complex_dump_local(path, phase_stack, kept_sessions, labels, cycle_fraction, harmonics)
fid = fopen(path, 'w');
assert(fid > 0, 'Cannot write %s', path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'session,harmonic_k,cycle_fraction,cycle_percent,channel,real,imag\n');
n_kept = numel(kept_sessions);
for si = 1:n_kept
    for hi = 1:numel(harmonics)
        for pi = 1:numel(cycle_fraction)
            for ci = 1:numel(labels)
                v = phase_stack(ci, pi, hi, si);
                fprintf(fid, '%s,%d,%.10g,%.6g,%s,%.10g,%.10g\n', ...
                    char(kept_sessions(si)), harmonics(hi), cycle_fraction(pi), ...
                    100 * cycle_fraction(pi), char(labels(ci)), real(v), imag(v));
            end
        end
    end
end
end

function sessions = read_session_list_local(opt, band_tbl)
if isfile(opt.manifest_path)
    T = readtable(opt.manifest_path, 'TextType', 'string', 'Delimiter', ',');
    if ismember('session', T.Properties.VariableNames)
        sessions = string(T.session);
    else
        sessions = string(T{:, 1});
    end
else
    rows = double(band_tbl.harmonic_k) == 1;
    sessions = unique(string(band_tbl.session(rows)), 'stable');
end
sessions = sessions(:);
sessions = sessions(strlength(sessions) > 0);
end

function [phase_map, count_rows] = session_cycle_maps_local( ...
    S, ev, band_tbl, harmonics, cycle_phase, cycle_fraction, half_width, lag_samples, opt)
n_ch = numel(S.eeg.labels);
n_point = numel(cycle_phase);
n_h = numel(harmonics);
phase_map = nan(n_ch, n_point, n_h);
count_rows = table('Size', [0 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'session', 'harmonic_k', 'n_trials', 'n_valid_samples', 'min_cycle_point_samples'});

goni_idx = find_goni_idx_local(S.goni.labels, opt.goni_label);
[gait_lo, gait_hi] = lookup_harmonic_band_local(band_tbl, S.label, 1);
gait_phase = gait_phase_from_goni_local(S, goni_idx, gait_lo, gait_hi);

for hi = 1:n_h
    harmonic_k = harmonics(hi);
    [bp_lo, bp_hi] = lookup_harmonic_band_local(band_tbl, S.label, harmonic_k);
    [phase_mod, n_trials, n_valid, point_counts] = accumulate_session_cycle_local( ...
        S, ev, gait_phase, bp_lo, bp_hi, cycle_phase, half_width, lag_samples);
    phase_map(:, :, hi) = phase_mod;
    count_rows = [count_rows; {string(S.label), harmonic_k, n_trials, n_valid, min(point_counts)}]; %#ok<AGROW>
end
end

function gait_phase = gait_phase_from_goni_local(S, goni_idx, bp_lo, bp_hi)
fs = S.goni.fs;
g = double(S.goni.x(:, goni_idx));
valid = isfinite(g);
if nnz(valid) < 10
    gait_phase = nan(size(g));
    return;
end
if any(~valid)
    xi = (1:numel(g))';
    g(~valid) = interp1(xi(valid), g(valid), xi(~valid), 'linear', 'extrap');
end
[z, pp, k] = butter(2, [bp_lo bp_hi] / (fs / 2), 'bandpass');
g_band = filtfilt(zp2sos(z, pp, k), 1, g);
gait_phase = angle(hilbert(g_band));
end

function [phase_mod, n_trials, n_valid_samples, point_counts] = accumulate_session_cycle_local( ...
    S, ev, gait_phase, bp_lo, bp_hi, cycle_phase, half_width, lag_samples)
fs = S.eeg.fs;
[z, pp, k] = butter(4, [bp_lo bp_hi] / (fs / 2), 'bandpass');
x_band = filtfilt(zp2sos(z, pp, k), 1, double(S.eeg.x)')';
xa = hilbert(x_band')';
phase_unit = xa ./ max(abs(xa), eps);

n_ch = size(xa, 1);
n_point = numel(cycle_phase);
T_total = size(xa, 2);
sum_point = complex(zeros(n_ch, n_point), zeros(n_ch, n_point));
count_point_ch = zeros(n_ch, n_point);
sum_all = complex(zeros(n_ch, 1), zeros(n_ch, 1));
count_all_ch = zeros(n_ch, 1);
point_counts = zeros(1, n_point);
n_trials = 0;
n_valid_samples = 0;

for ti = 1:numel(ev)
    [out_t, phi] = aligned_time_and_phase_local(S, ev(ti), gait_phase, lag_samples, T_total);
    if isempty(out_t)
        continue;
    end
    good = isfinite(phi);
    if nnz(good) < 5
        continue;
    end
    eeg_cols = out_t + lag_samples;
    good_cols = find(good);
    V_all = phase_unit(:, eeg_cols(good_cols));
    ok_all = isfinite(real(V_all)) & isfinite(imag(V_all));
    V_all(~ok_all) = 0;
    sum_all = sum_all + sum(V_all, 2);
    count_all_ch = count_all_ch + sum(ok_all, 2);
    n_trials = n_trials + 1;
    n_valid_samples = n_valid_samples + numel(good_cols);

    for pi = 1:n_point
        d = angle(exp(1i * (phi - cycle_phase(pi))));
        cols = find(good & abs(d) <= half_width);
        if isempty(cols)
            continue;
        end
        V = phase_unit(:, eeg_cols(cols));
        ok = isfinite(real(V)) & isfinite(imag(V));
        V(~ok) = 0;
        sum_point(:, pi) = sum_point(:, pi) + sum(V, 2);
        count_point_ch(:, pi) = count_point_ch(:, pi) + sum(ok, 2);
        point_counts(pi) = point_counts(pi) + numel(cols);
    end
end

mean_point = safe_divide_local(sum_point, count_point_ch);
mean_all = safe_divide_local(sum_all, count_all_ch);
phase_mod = mean_point - mean_all;
phase_mod(~isfinite(real(phase_mod)) | ~isfinite(imag(phase_mod))) = NaN;
end

function [out_t, phi] = aligned_time_and_phase_local(S, ev, gait_phase, lag_samples, T_total)
et0 = ev.etstart;
et1 = ev.etstop;
gt0 = ev.gtstart;
gt1 = ev.gtstop;
if et1 < et0 || gt1 < gt0
    out_t = zeros(1, 0); phi = zeros(1, 0); return;
end
Lk = min(et1 - et0 + 1, gt1 - gt0 + 1);
valid_start = max(1, 1 - min(lag_samples)) + et0 - 1;
valid_end = min(T_total - max(lag_samples), et0 + Lk - 1);
if valid_end < valid_start
    out_t = zeros(1, 0); phi = zeros(1, 0); return;
end
out_t = valid_start:valid_end;
local_idx = out_t - et0 + 1;
goni_t = gt0 + local_idx - 1;
goni_t = min(max(goni_t, 1), numel(gait_phase));
phi = double(gait_phase(goni_t));
phi = phi(:)';
end

function [bp_lo, bp_hi] = lookup_harmonic_band_local(tbl, session, harmonic_k)
mask = string(tbl.session) == string(session) & double(tbl.harmonic_k) == double(harmonic_k);
idx = find(mask);
assert(numel(idx) == 1, 'Expected one band row for %s k%d, found %d', session, harmonic_k, numel(idx));
bp_lo = double(tbl.band_lo(idx));
bp_hi = double(tbl.band_hi(idx));
assert(isfinite(bp_lo) && isfinite(bp_hi) && bp_lo > 0 && bp_hi > bp_lo, ...
    'Invalid band for %s k%d: [%g %g]', session, harmonic_k, bp_lo, bp_hi);
end

function idx = find_goni_idx_local(labels, target)
labels = string(labels(:));
idx = find(strcmp(strtrim(labels), strtrim(string(target))), 1);
if isempty(idx)
    idx = find(strcmpi(strtrim(labels), strtrim(string(target))), 1);
end
assert(~isempty(idx), 'Cannot find goni label %s', target);
end

function M = safe_divide_local(Sn, C)
M = Sn ./ C;
M(C == 0 | ~isfinite(real(M)) | ~isfinite(imag(M))) = NaN;
end

function [signed_maps, ref_phase, ref_row] = unified_reference_projection_local(complex_maps, labels, cycle_fraction, harmonics)
mag = abs(complex_maps);
mag(~isfinite(mag)) = -Inf;
[best_abs, idx] = max(mag(:));
if ~isfinite(best_abs) || best_abs <= 0
    ref_phase = 0; ci = 1; pi = 1; hi = 1;
else
    [ci, pi, hi] = ind2sub(size(complex_maps), idx);
    ref_phase = angle(complex_maps(ci, pi, hi));
end
signed_maps = real(complex_maps * exp(-1i * ref_phase));
ref_row = table(harmonics(hi), cycle_fraction(pi), 100 * cycle_fraction(pi), labels(ci), ref_phase, best_abs, ...
    'VariableNames', {'harmonic_k', 'reference_cycle_fraction', 'reference_cycle_percent', ...
    'reference_channel', 'reference_phase_rad', 'reference_abs'});
end

function lim = common_limit_local(maps)
vals = abs(maps(:));
vals = vals(isfinite(vals));
if isempty(vals)
    lim = [-1 1]; return;
end
q = quantile(vals, 0.98);
if ~isfinite(q) || q <= eps, q = max(vals); end
if ~isfinite(q) || q <= eps, q = 1; end
lim = [-q q];
end

function plot_cycle_grid_local(out_png, maps, chanlocs, cycle_fraction, row_titles, maplimits, opt, gname, thr)
n_rows = size(maps, 3);
n_point = size(maps, 2);
n_cols = n_point + 1;
fig_w = max(1750, 150 * n_point + 240);
fig_h = max(820, 200 * n_rows + 70);
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 fig_w fig_h]);
tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
for ri = 1:n_rows
    nexttile;
    axis off;
    text(0.98, 0.5, char(row_titles(ri)), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
        'Interpreter', 'none', 'FontSize', 8, 'FontWeight', 'bold');
    for pi = 1:n_point
        tile_idx = (ri - 1) * n_cols + 1 + pi;
        nexttile(tile_idx);
        topoplot(maps(:, pi, ri), chanlocs, 'style', 'map', 'electrodes', char(opt.electrodes), ...
            'maplimits', maplimits, 'numcontour', 6);
        set(gca, 'CLim', maplimits);
        if ri == 1
            title(sprintf('%.1f%%', 100 * cycle_fraction(pi)), 'FontSize', 8);
        end
    end
end
colormap(fig, redblue_cmap_local(256));
cb = colorbar;
cb.Layout.Tile = 'east';
if strcmp(gname, 'responder')
    cmp = sprintf('r >= %.2f', thr);
else
    cmp = sprintf('r < %.2f', thr);
end
cb.Label.String = sprintf('signed phase modulation  |  %s (%s)  |  shared scale', gname, cmp);
exportgraphics(fig, out_png, 'Resolution', 240, 'BackgroundColor', 'white');
close(fig);
end

function chanlocs = get_chanlocs_local(S)
if isfield(S.eeg, 'chanlocs') && ~isempty(S.eeg.chanlocs)
    chanlocs = S.eeg.chanlocs;
elseif isfield(S.eeg, 'c') && ~isempty(S.eeg.c)
    chanlocs = S.eeg.c;
else
    error('No chanlocs in %s', S.label);
end
end

function cmap = redblue_cmap_local(n)
if nargin < 1, n = 256; end
x = linspace(-1, 1, n)';
r = ones(n, 1); g = ones(n, 1); b = ones(n, 1);
neg = x < 0; pos = x > 0;
r(neg) = 1 + x(neg) * 0.95; g(neg) = 1 + x(neg) * 0.80; b(neg) = 1 + x(neg) * 0.45;
r(pos) = 1 - x(pos) * 0.30; g(pos) = 1 - x(pos) * 0.95; b(pos) = 1 - x(pos) * 0.95;
cmap = min(max([r g b], 0), 1);
end

function write_harmonic_complex_csv_local(path, maps, labels, cycle_fraction, harmonics)
fid = fopen(path, 'w');
assert(fid > 0, 'Cannot write %s', path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'harmonic_k,cycle_fraction,cycle_percent,channel,real,imag,abs,angle_rad\n');
for hi = 1:numel(harmonics)
    for pi = 1:numel(cycle_fraction)
        for ci = 1:numel(labels)
            v = maps(ci, pi, hi);
            fprintf(fid, '%d,%.10g,%.6g,%s,%.10g,%.10g,%.10g,%.10g\n', ...
                harmonics(hi), cycle_fraction(pi), 100 * cycle_fraction(pi), ...
                char(labels(ci)), real(v), imag(v), abs(v), angle(v));
        end
    end
end
end

function ensure_eeglab_topoplot_local(eeglab_root)
if exist('topoplot', 'file') == 2
    return;
end
if exist(eeglab_root, 'dir')
    addpath(genpath(eeglab_root));
end
assert(exist('topoplot', 'file') == 2, 'topoplot not found');
end
