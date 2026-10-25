function estimate_goniometer_psd_peaks(varargin)
%ESTIMATE_GONIOMETER_PSD_PEAKS  Per-session goniometer harmonic bands f0, 2*f0, 3*f0.
%
% For every session in the manifest, estimates the step-frequency fundamental f0 of
% the leg angles and the two harmonics above it, and writes the EEG feature band
% placed around each peak.
%
% Method: the per-trial power spectra of the six leg-angle channels of the chosen
% condition are normalised and averaged over trials and joints; f0 is the largest
% peak in 0.45-1.15 Hz; harmonic k is the largest peak within 0.25 Hz of k*f0; the
% band of harmonic k is that peak plus and minus band_half_width.
%
% Parameters (name/value):
%   manifest         session list, column `session`
%                    (default result/session_manifest.csv)
%   n_sessions       process at most this many sessions from the manifest
%   condition        'MI' (imagined walking) | 'EXEC' (real walking)
%   out_dir          output folder (default result/diag_main_goni_psd_peaks)
%   output_prefix    prefix of the files written in out_dir (default 'main')
%   fmax             upper edge of the searched spectrum, in Hz
%   band_half_width  half-width of the EEG feature band around a peak, in Hz
%   target_labels    channel labels to use instead of the six defaults
%
% Output in out_dir:
%   {output_prefix}_goni_peak_bands.csv    one row per session and harmonic
%                                          (session, harmonic_k, target_hz,
%                                          peak_hz, band_lo, band_hi, n_psd)
%   {output_prefix}_goni_psd_summary.csv   the averaged spectrum per session
%   README.md                              the band table in readable form
%
% The band csv is the bp_band_csv input of compute_shared_delta_pca_lso_basis and
% of run_decoder, and is read by run/build_basis.sh and run/export_features.sh.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'shared'));

p = inputParser;
p.addParameter('manifest', fullfile(root, 'result', 'session_manifest.csv'));
p.addParameter('n_sessions', 999);
p.addParameter('condition', 'MI');
p.addParameter('out_dir', fullfile(root, 'result', 'diag_main_goni_psd_peaks'));
p.addParameter('output_prefix', 'main');
p.addParameter('fmax', 4.0);
p.addParameter('band_half_width', 0.18);
p.addParameter('target_labels', {});
p.parse(varargin{:});
opt = p.Results;

if ~exist(opt.out_dir, 'dir')
    mkdir(opt.out_dir);
end

T = readtable(opt.manifest, 'TextType', 'string', 'Delimiter', ',', ...
    'ReadVariableNames', true);
if ismember('session', T.Properties.VariableNames)
    sessions_all = T.session;
else
    raw = readcell(opt.manifest, 'Delimiter', ',');
    sessions_all = string(raw(2:end, 1));
end
sessions = cellstr(sessions_all(1:min(opt.n_sessions, numel(sessions_all))));

if isempty(opt.target_labels)
    if strcmpi(char(opt.condition), 'EXEC')
        labels = {'LHipS X','LKneS X','LAnkS X','RHipS X','RKneS X','RAnkS X'};
    else
        labels = {'LHipW X','LKneW X','LAnkW X','RHipW X','RKneW X','RAnkW X'};
    end
else
    labels = cellstr(opt.target_labels);
end
freq_grid = (0:0.01:opt.fmax)';
session_col = strings(0, 1);
harmonic_k = zeros(0, 1);
target_hz = zeros(0, 1);
peak_hz = zeros(0, 1);
band_lo = zeros(0, 1);
band_hi = zeros(0, 1);
n_psd = zeros(0, 1);

output_prefix = char(opt.output_prefix);
summary_csv = fullfile(opt.out_dir, sprintf('%s_goni_psd_summary.csv', output_prefix));
summary_fid = fopen(summary_csv, 'w');
fprintf(summary_fid, 'session,freq_hz,mean_psd\n');

for si = 1:numel(sessions)
    label = sessions{si};
    try
        [P_mean, n_valid] = session_walkerx_psd_local(label, opt.condition, labels, freq_grid, opt.fmax);
        f0 = local_peak_in_band(freq_grid, P_mean, [0.45 1.15]);
        for k = 1:3
            target = double(k) * f0;
            band = [max(0.1, target - 0.25), min(opt.fmax, target + 0.25)];
            pk = local_peak_in_band(freq_grid, P_mean, band);
            session_col(end+1, 1) = string(label); %#ok<AGROW>
            harmonic_k(end+1, 1) = k; %#ok<AGROW>
            target_hz(end+1, 1) = target; %#ok<AGROW>
            peak_hz(end+1, 1) = pk; %#ok<AGROW>
            band_lo(end+1, 1) = max(0.05, pk - opt.band_half_width); %#ok<AGROW>
            band_hi(end+1, 1) = min(opt.fmax, pk + opt.band_half_width); %#ok<AGROW>
            n_psd(end+1, 1) = n_valid; %#ok<AGROW>
        end
        for fi = 1:numel(freq_grid)
            fprintf(summary_fid, '%s,%.5f,%.10g\n', label, freq_grid(fi), P_mean(fi));
        end
        fprintf('[%d/%d] %s f0=%.2fHz n_psd=%d\n', si, numel(sessions), label, f0, n_valid);
    catch ME
        fprintf('[%d/%d] %s FAIL: %s\n', si, numel(sessions), label, ME.message);
    end
end
fclose(summary_fid);

bands = table(session_col, harmonic_k, target_hz, peak_hz, band_lo, band_hi, n_psd, ...
    'VariableNames', {'session', 'harmonic_k', 'target_hz', 'peak_hz', 'band_lo', 'band_hi', 'n_psd'});
bands_csv = fullfile(opt.out_dir, sprintf('%s_goni_peak_bands.csv', output_prefix));
writetable(bands, bands_csv);

readme = fullfile(opt.out_dir, 'README.md');
fid = fopen(readme, 'w');
fprintf(fid, '# Goniometer PSD peak bands\n\n');
fprintf(fid, '- Source: %s %s joints, per-trial normalized PSD averaged across trials and joints.\n', ...
    char(opt.condition), strjoin(labels, '; '));
fprintf(fid, '- Fundamental search band: `0.45-1.15 Hz`.\n');
fprintf(fid, '- Harmonics are searched near `k*f0 +/- 0.25 Hz`.\n');
fprintf(fid, '- Suggested EEG feature bands use `peak +/- %.2f Hz`.\n\n', opt.band_half_width);
fprintf(fid, '| session | k | target Hz | peak Hz | band lo | band hi | n PSD traces |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|\n');
for ii = 1:height(bands)
    fprintf(fid, '| %s | %d | %.4f | %.4f | %.4f | %.4f | %d |\n', ...
        char(bands.session(ii)), bands.harmonic_k(ii), bands.target_hz(ii), bands.peak_hz(ii), ...
        bands.band_lo(ii), bands.band_hi(ii), bands.n_psd(ii));
end
fprintf(fid, '\nOne row per session and harmonic: the f0, 2*f0 and 3*f0 peak EEG bands.\n');
fclose(fid);

fprintf('Wrote %s\n', bands_csv);
fprintf('Wrote %s\n', summary_csv);
end

function [P_mean, n_valid] = session_walkerx_psd_local(label, condition, labels, freq_grid, fmax)
S = load_triple_mat(label);
ev = trial_filter(S.events, condition);
fs = S.goni.fs;
joint_idx = nan(1, numel(labels));
for j = 1:numel(labels)
    joint_idx(j) = find_goni_idx(S.goni.labels, labels{j});
end

P_trials = zeros(numel(freq_grid), 0);
for ti = 1:numel(ev)
    gt0 = ev(ti).gtstart;
    gt1 = ev(ti).gtstop;
    for j = 1:numel(labels)
        y = double(S.goni.x(gt0:gt1, joint_idx(j)));
        y = y(isfinite(y));
        if numel(y) < round(2 * fs)
            continue;
        end
        y = detrend(y(:), 0);
        win = min(round(4 * fs), max(round(2 * fs), floor(numel(y) / 2)));
        noverlap = floor(0.5 * win);
        nfft = 8192;
        [pxx, f] = pwelch(y, hamming(win), noverlap, nfft, fs);
        keep = f >= 0.05 & f <= fmax;
        denom = trapz(f(keep), pxx(keep));
        if ~isfinite(denom) || denom <= 0
            continue;
        end
        P_trials(:, end+1) = interp1(f, pxx / denom, freq_grid, 'linear', 0); %#ok<AGROW>
    end
end
if isempty(P_trials)
    error('no valid Walker-X PSD traces');
end
P_mean = mean(P_trials, 2, 'omitnan');
n_valid = size(P_trials, 2);
end

function peak_hz = local_peak_in_band(f, p, band)
mask = f >= band(1) & f <= band(2) & isfinite(p);
ff = f(mask);
pp = p(mask);
if isempty(ff)
    peak_hz = NaN;
    return;
end
[~, ix] = max(pp);
peak_hz = ff(ix);
end
