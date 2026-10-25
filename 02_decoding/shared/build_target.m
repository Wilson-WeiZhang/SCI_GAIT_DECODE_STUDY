function Y = build_target(joint_traj, opt)
%BUILD_TARGET Mode-dependent regression target from per-trial goni.
%
%   Y = build_target(joint_traj, opt)
%   joint_traj : [T x J]  (NaN cols allowed for missing joints)
%   opt.target_mode : 'raw' | 'phase_pj' | 'phase_g'
%   opt.fs          : sampling rate (Hz)
%   opt.phase_lo    : LP low cutoff for phase extraction (default 0.3 Hz)
%   opt.phase_hi    : LP high cutoff for phase extraction (default 2.0 Hz)
%
%   Outputs:
%     'raw'      : [T x J]   per-trial z-scored joint angle 
%     'phase_pj' : [T x 2J]  per-joint Hilbert phase, columns = [c1 s1 c2 s2 ...]
%                            cⱼ=cos φⱼ, sⱼ=sin φⱼ ; missing joint → NaN cols
%     'phase_g'  : [T x 2]   global phase from PC1 of LP'd active joints,
%                            columns = [cos φ_g, sin φ_g]
%
%   Phase extraction: band-pass Butterworth (zero-phase) on each joint, then
%   analytic-signal angle. Edge effects ~0.5 s; predictor handles via continuous
%   delay embedding.

    if ~isfield(opt, 'phase_lo'), opt.phase_lo = 0.3; end
    if ~isfield(opt, 'phase_hi'), opt.phase_hi = 2.0; end

    mode = lower(string(opt.target_mode));
    [T, J] = size(joint_traj);

    switch mode
        case "raw"
            Y = target_preproc(joint_traj);

        case "phase_pj"
            Y = nan(T, 2*J);
            for j = 1:J
                col = joint_traj(:, j);
                if all(isnan(col)), continue; end
                phi = local_hilbert_phase(col, opt);
                Y(:, 2*j - 1) = cos(phi);
                Y(:, 2*j)     = sin(phi);
            end

        case "phase_g"
            % Global phase from PC1 of LP'd active joints
            active = find(any(~isnan(joint_traj), 1));
            if isempty(active)
                Y = nan(T, 2); return;
            end
            Z = joint_traj(:, active);
            % Replace NaN with column mean (then center will zero them out)
            for c = 1:size(Z, 2)
                m = mean(Z(:, c), 'omitnan');
                Z(isnan(Z(:, c)), c) = m;
            end
            Z = Z - mean(Z, 1);
            % Per-joint LP
            for c = 1:size(Z, 2)
                Z(:, c) = local_lp(Z(:, c), opt);
            end
            % SVD-based PC1
            [U, ~, ~] = svd(Z, 'econ');
            pc1 = U(:, 1);
            phi = angle(hilbert(pc1));
            Y = [cos(phi), sin(phi)];

        otherwise
            error('build_target: target_mode must be raw | phase_pj | phase_g');
    end
end

function phi = local_hilbert_phase(x, opt)
    fs = opt.fs;
    [z, p, k] = butter(2, [opt.phase_lo opt.phase_hi] / (fs/2), 'bandpass');
    sos = zp2sos(z, p, k);
    x_lp = filtfilt(sos, 1, x);
    phi  = angle(hilbert(x_lp));
end

function x_lp = local_lp(x, opt)
    fs = opt.fs;
    [z, p, k] = butter(2, [opt.phase_lo opt.phase_hi] / (fs/2), 'bandpass');
    sos = zp2sos(z, p, k);
    x_lp = filtfilt(sos, 1, x);
end
