function cpc_step2_cal(sid, out_root, min_amica_min)
%CPC_STEP2_CAL  Step 2 of one healthy session on its calibration copy only.
%
%   cpc_prep_step2 (prep_step2 with runica's clock reseed switched off) on <out_root>/cal/<SID>_step1.set:
%   1-40 Hz copy, S11 trim, task and rest segments, bad-channel detection, interpolation, FCz added, common
%   average reference, ASR on the Walk segments, extended infomax ICA, ICLabel. Every sample it sees
%   precedes the 1 s baseline window of the first trial after the calibration. rng('default') and four
%   computational threads (run under taskset with at least four cores): the decomposition is reproducible
%   bit for bit at that thread count on one host. Fails if prep_step2 skipped the ICA (task time below
%   min_amica_min minutes). Writes <out_root>/cal/<SID>_step2.set and <out_root>/info/<SID>_step2_cal_info.txt.
%   usage: cpc_step2_cal(sid, out_root, min_amica_min)
if nargin < 3, min_amica_min = 5; end
if ischar(min_amica_min) || isstring(min_amica_min), min_amica_min = str2double(min_amica_min); end
ICA_THREADS = 4;
maxNumCompThreads(ICA_THREADS);
E = cpc_env();
addpath(E.eeglab);
addpath(genpath(fullfile(E.root, '01_preprocessing', 'healthy')));
eeglab nogui;
rng('default');
cal = fullfile(out_root, 'cal');
info = {sprintf('host %s; MATLAB %s; EEGLAB %s; maxNumCompThreads %d; root %s', E.host, version, eeg_getversion, maxNumCompThreads, E.root), sprintf('cpc_prep_step2: %s; min_amica_min %g; maxNumCompThreads %d (%d)', which('cpc_prep_step2'), min_amica_min, ICA_THREADS, maxNumCompThreads)};
info{end+1} = sprintf('ICA seed: production (runica rand(''state'', 0)); runica = %s', which('runica'));
o = cpc_prep_step2(fullfile(cal, [sid '_step1.set']), cal, struct('overwrite', true, 'min_amica_min', min_amica_min));
S = pop_loadset('filename', [sid '_step2.set'], 'filepath', cal);
m = S.etc.step2_meta;
info{end+1} = sprintf('calibration step2: %d segments, %.1f s, bad ch %d {%s}, rank %d, ASR segments %d, ICs %d', ...
    m.n_trial_segs, m.trial_dur_sec, m.n_ch_interpolated, strjoin(m.bad_ch_labels, ','), m.data_rank, ...
    m.n_asr_segments, size(S.icaweights, 1));
if isempty(S.icaweights)
    error('cpc_step2_cal:noICA', ['%s: the calibration step2 has NO ICA: prep_step2 skipped it because the task time it ' ...
          'kept after its S11-S12 trim is %.1f s (%.1f min) < min_amica_min %g min. Compare plan.csv est_cal_task_s and ' ...
          'the trim line of the step2 log.'], sid, m.trial_dur_sec, m.trial_dur_sec / 60, min_amica_min);
end
info{end+1} = sprintf('icaweights md5 %s; icasphere md5 %s', md5hex(S.icaweights), md5hex(S.icasphere));
A = pop_loadset('filename', [sid '_step2.set'], 'filepath', fullfile(E.root, 'data', 'prep_data', 'healthy', 'eeg'));
a = A.etc.step2_meta;
info{end+1} = sprintf('as-run step2:      %d segments, %.1f s, bad ch %d {%s}, rank %d, ASR segments %d, ICs %d', ...
    a.n_trial_segs, a.trial_dur_sec, a.n_ch_interpolated, strjoin(a.bad_ch_labels, ','), a.data_rank, ...
    a.n_asr_segments, size(A.icaweights, 1));
info_dir = fullfile(out_root, 'info');
if ~exist(info_dir, 'dir'), mkdir(info_dir); end
fid = fopen(fullfile(info_dir, [sid '_step2_cal_info.txt']), 'w');
fprintf(fid, '%s\n', info{:});
fclose(fid);
fprintf('%s\n', info{:});
disp(o);
assert(~isempty(S.icaweights), 'cpc_step2_cal: %s calibration step2 has no ICA (task time %.1f s < %g min?)', ...
       sid, m.trial_dur_sec, min_amica_min);
end

function h = md5hex(x)
md = java.security.MessageDigest.getInstance('MD5');
md.update(typecast(double(x(:)), 'uint8'));
h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
end
