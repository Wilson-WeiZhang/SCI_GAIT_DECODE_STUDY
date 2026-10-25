function mref_cshift(out_file)
%MREF_CSHIFT  MATLAB ground truth for the within-trial circular-shift control.
%
% Calls a verbatim copy of run_decoder's circular-shift anchor function on synthetic
% trials and saves, per case, n, et0, T_total, seed and the returned sample index
% minus out_t(1) (the 0-based column read by each output column) to out_file (-v7).
lag_samples = unique(round((-2000:50:0) * 250 / 1000));
crop = max(1, 1 - min(lag_samples)) - 1;
ns = [1 2 3 50 400 500 501 502 1000 1871 2406 3000];
C = struct('n', {}, 'et0', {}, 'T_total', {}, 'seed', {}, 'col', {});
k = 0;
for n = ns
    for t = [1 7 59]
        for et0 = [1 1000]
            k = k + 1;
            seed = 20260905 + 1009 * t + n;
            out_t = (et0 + crop):(et0 + crop + n - 1);
            T_total = et0 + crop + n + 100;
            s = circular_shift_anchor_local(out_t, lag_samples, T_total, seed);
            C(k).n = n; C(k).et0 = et0; C(k).T_total = T_total; C(k).seed = seed; C(k).col = s - out_t(1);
        end
    end
end
save(out_file, 'C', '-v7');
fprintf('wrote %s (%d cases)\n', out_file, k);
end

function sample_t = circular_shift_anchor_local(out_t, lag_samples, total_samples, seed)
sample_t = out_t;
n = numel(out_t);
if n <= 1
    return;
end
min_lag = min(lag_samples);
max_lag = max(lag_samples);
valid_lo = max(1 - min_lag, out_t(1));
valid_hi = min(total_samples - max_lag, out_t(end));
valid_t = valid_lo:valid_hi;
if numel(valid_t) ~= n
    return;
end
min_shift = min(max(abs(lag_samples)) + 1, max(1, n - 1));
if min_shift >= n
    min_shift = max(1, floor(n / 3));
end
rng(seed, 'twister');
choices = [min_shift:(n-1), 1:max(1, n-min_shift)];
choices = unique(mod(choices - 1, n) + 1);
shift = choices(randi(numel(choices)));
sample_t = valid_t(mod((0:n-1) + shift, n) + 1);
end
