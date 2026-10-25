function yp_s = smooth_cs(yp, fs, smooth_type, params)
%SMOOTH_CS  Post-prediction smoothing for (cos, sin) phase predictions.
%
%   yp_s = smooth_cs(yp, fs, smooth_type, params)
%   yp           : [T x K] predictions, K=2 (phase_g) or 2J (phase_pj)
%                  columns interleaved [c1 s1 c2 s2 ...] for phase_pj
%   fs           : sampling rate (Hz)
%   smooth_type  : 'none' | 'lp' | 'kalman'
%   params       : struct with smoothing-specific fields:
%                    .lp_lo, .lp_hi  (Hz) for type='lp'
%                    .omega_hz, .q_noise, .r_noise for type='kalman'
%
%   For each (c, s) pair: smooth columns separately then renormalize so
%   resulting (c', s') has unit norm at each t (so atan2 maps back to a
%   valid phase angle without amplitude distortion).

if nargin < 3 || isempty(smooth_type), smooth_type = 'none'; end
if strcmpi(smooth_type, 'none'), yp_s = yp; return; end

K = size(yp, 2);
if mod(K, 2) ~= 0, error('smooth_cs: ncols must be even, got %d', K); end
yp_s = zeros(size(yp));

for k = 1:2:K
    c = yp(:, k);
    s = yp(:, k+1);
    if all(~isfinite(c)) || all(~isfinite(s))
        yp_s(:, k:k+1) = NaN;
        continue;
    end
    if any(~isfinite(c)) || any(~isfinite(s))
        c = fillmissing(c, 'linear', 'EndValues', 'nearest');
        s = fillmissing(s, 'linear', 'EndValues', 'nearest');
    end
    switch lower(smooth_type)
        case 'lp'
            if ~isfield(params, 'lp_lo'), params.lp_lo = 0.3; end
            if ~isfield(params, 'lp_hi'), params.lp_hi = 3.0; end
            [z, p, kk] = butter(2, [params.lp_lo params.lp_hi] / (fs/2), 'bandpass');
            sos = zp2sos(z, p, kk);
            c_s = filtfilt(sos, 1, c);
            s_s = filtfilt(sos, 1, s);
        case 'kalman'
            if ~isfield(params, 'omega_hz'), params.omega_hz = 1.0; end
            if ~isfield(params, 'q_noise'),  params.q_noise = 0.01; end
            if ~isfield(params, 'r_noise'),  params.r_noise = 0.5;  end
            xs = kalman_rts_phase([c s], fs, params.omega_hz, params.q_noise, params.r_noise);
            c_s = xs(:, 1);
            s_s = xs(:, 2);
        otherwise
            error('smooth_cs: unknown type %s', smooth_type);
    end
    % Renormalize so (c_s, s_s) has unit norm — atan2 invariant but
    % keeps phase amplitude consistent for downstream PLV computation.
    r = sqrt(c_s.^2 + s_s.^2);
    r(r < 1e-9) = 1;
    yp_s(:, k)   = c_s ./ r;
    yp_s(:, k+1) = s_s ./ r;
end
end

function xs = kalman_rts_phase(yp, fs, omega_hz, q_noise, r_noise)
%KALMAN_RTS_PHASE  RTS smoother on (cos, sin) with rotational state dynamics.
%   xs = kalman_rts_phase(yp, fs, omega_hz, q_noise, r_noise)
%   yp: [T x 2] noisy (cos, sin) ; State: x = [c; s] evolving by R(omega*dt)

T = size(yp, 1);
dt = 1 / fs;
omega = 2 * pi * omega_hz;
R = [cos(omega*dt) -sin(omega*dt); sin(omega*dt) cos(omega*dt)];
Q = q_noise * eye(2);
Rk = r_noise * eye(2);
H = eye(2);

x  = zeros(2, T);
Pf = zeros(2, 2, T);
xp_pred  = zeros(2, T);
Pp_pred  = zeros(2, 2, T);

x(:, 1) = yp(1, :)';
Pf(:, :, 1) = eye(2);
xp_pred(:, 1) = x(:, 1);
Pp_pred(:, :, 1) = Pf(:, :, 1);

% Forward (filter)
for t = 2:T
    xp = R * x(:, t-1);
    Pp = R * Pf(:, :, t-1) * R' + Q;
    xp_pred(:, t)    = xp;
    Pp_pred(:, :, t) = Pp;
    K = Pp * H' / (H * Pp * H' + Rk);
    x(:, t) = xp + K * (yp(t, :)' - H * xp);
    Pf(:, :, t) = (eye(2) - K * H) * Pp;
end

% Backward (RTS)
xs_arr = x;
for t = T-1:-1:1
    Gn = Pf(:, :, t) * R' / Pp_pred(:, :, t+1);
    xs_arr(:, t) = x(:, t) + Gn * (xs_arr(:, t+1) - xp_pred(:, t+1));
end

xs = xs_arr';
end
