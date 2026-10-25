function [X_cell, Y_cell, lag_samples_used] = build_xy_v2_boundary(S, ev, joint_idx, opt)
%BUILD_XY_V2_BOUNDARY  Delay-embedded EEG features and matching joint targets, per trial.
%
%   [X_cell, Y_cell, lag_samples_used] = build_xy_v2_boundary(S, ev, joint_idx, opt)
%
%   S           session struct from load_triple_mat
%   ev          trials to build (output of trial_filter)
%   joint_idx   column index in S.goni.x of each target joint (NaN = not available)
%   opt         see below
%
%   The continuous EEG is band-passed once for the whole recording (fourth-order
%   Butterworth in second-order-section form, applied forward and backward). For
%   each trial a feature matrix [nch*L x Tk] is then assembled, whose rows are the
%   channels at each of the L lags of opt.lag_ms, together with the joint angles
%   for the same output samples.
%
%   Boundary handling: an output sample is kept only when every lag falls inside
%   the recording AND inside the trial. The start of the window is clamped inward
%   by the largest backward reach and the end by the largest forward reach, so an
%   EEG slice can never be drawn from the period between trials. The usable window
%   is shorter than the trial by those two clamps; with a lag grid that contains no
%   positive lag, as in the published configuration, the end clamp is zero.
%
%   opt fields:
%     lag_ms          lag grid in ms, rounded to samples at the EEG rate
%     bp_lo, bp_hi    band-pass edges in Hz
%     trim_sec        symmetric trim of each trial, in s
%     trim_start_sec  trim at the start of a trial, in s; overrides trim_sec
%     trim_end_sec    trim at the end of a trial, in s; overrides trim_sec
%     target_mode     'raw' (default) passes the angles through target_preproc;
%                     any other value passes them to build_target
%
%   A trial with no usable output sample returns empty matrices of the right shape.

fs    = S.eeg.fs;
nch   = size(S.eeg.x, 1);
lag_samples = round(opt.lag_ms * fs / 1000);
lag_samples = unique(lag_samples);
L = numel(lag_samples);

% Band-pass on full continuous EEG, once. SOS form for low-frequency stability
% (0.1 Hz at 250 Hz fs = normalized cutoff ~8e-4, ill-conditioned in tf form).
[z, p, k] = butter(4, [opt.bp_lo opt.bp_hi] / (fs/2), 'bandpass');
sos = zp2sos(z, p, k);
x_filt = filtfilt(sos, 1, S.eeg.x')';     % [nch x T_total]

T_total = size(x_filt, 2);
N = numel(ev);
X_cell = cell(1, N);
Y_cell = cell(1, N);

J = numel(joint_idx);

trim_start = 0; trim_end = 0;
if isfield(opt, 'trim_sec') && opt.trim_sec > 0
    % symmetric trim
    trim_start = round(opt.trim_sec * fs);
    trim_end   = round(opt.trim_sec * fs);
end
if isfield(opt, 'trim_start_sec') && opt.trim_start_sec > 0
    trim_start = round(opt.trim_start_sec * fs);
end
if isfield(opt, 'trim_end_sec') && opt.trim_end_sec > 0
    trim_end = round(opt.trim_end_sec * fs);
end

% Precompute symmetric lag clamps (done once, outside trial loop)
max_lag_pos = max(0, max(lag_samples));   % positive lags only; used for end clamp

for k = 1:N
    et0 = ev(k).etstart; et1 = ev(k).etstop;
    gt0 = ev(k).gtstart; gt1 = ev(k).gtstop;
    if trim_start > 0
        et0 = et0 + trim_start; gt0 = gt0 + trim_start;
    end
    if trim_end > 0
        et1 = et1 - trim_end; gt1 = gt1 - trim_end;
    end
    if et1 < et0 || gt1 < gt0
        X_cell{k} = zeros(nch * L, 0);
        Y_cell{k} = zeros(0, J);
        continue;
    end
    L_eeg = et1 - et0 + 1;
    L_goni = gt1 - gt0 + 1;
    Lk = min(L_eeg, L_goni);

    % Output time grid: positions inside [et0, et0+Lk-1] where ALL lags fall
    % inside [1, T_total] AND inside the trial window.
    %
    % valid_start: clamp inward by |min_lag| on the negative-lag side
    %   (prevents a backward reach before the start of the recording)
    % valid_end:   clamp inward by max_lag_pos on the positive-lag side
    %   ensures t_out + max_lag_pos <= et0+Lk-1 (trial end), so EEG
    %   positive-lag slices never reach into the inter-trial period.
    valid_start = max(1, 1 - min(lag_samples)) + et0 - 1;
    valid_end   = min(T_total - max(lag_samples), et0 + Lk - 1 - max_lag_pos);
    if valid_end < valid_start
        X_cell{k} = zeros(nch * L, 0);
        Y_cell{k} = zeros(0, J);
        continue;
    end
    out_t = valid_start:valid_end;       % absolute EEG sample indices
    Tk = numel(out_t);

    % Build delay-embedded feature matrix [nch*L x Tk]
    Xk = zeros(nch * L, Tk);
    for li = 1:L
        rng_idx = out_t + lag_samples(li);
        Xk((li-1)*nch+1 : li*nch, :) = x_filt(:, rng_idx);
    end
    X_cell{k} = Xk;

    % Build joint target for the same out_t window, mapped to goni indices.
    % Map: out_t - et0 + 1 = local position inside trial, then offset to gt0.
    local_idx = out_t - et0 + 1;
    goni_t = gt0 + local_idx - 1;
    goni_t = min(goni_t, size(S.goni.x, 1));
    Yk_raw = nan(Tk, J);
    for j = 1:J
        if isnan(joint_idx(j)), continue; end
        Yk_raw(:, j) = S.goni.x(goni_t, joint_idx(j));
    end
    if isfield(opt, 'target_mode') && ~strcmpi(opt.target_mode, 'raw')
        topt = opt; topt.fs = fs;
        Y_cell{k} = build_target(Yk_raw, topt);
    else
        Y_cell{k} = target_preproc(Yk_raw);
    end
end

lag_samples_used = lag_samples;
end
