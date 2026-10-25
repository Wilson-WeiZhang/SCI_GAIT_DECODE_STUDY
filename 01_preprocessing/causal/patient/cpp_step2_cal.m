function cpp_step2_cal(label, out_root, min_amica_min, step2_dir)
%CPP_STEP2_CAL  Step 2 of one patient session on its calibration copy only.
%
%   cpp_prep_step2 (the patient prep_step2 with runica's clock reseed switched off) on
%   <out_root>/cal/<label>_step1.set: rest-break split, 1-40 Hz copy, S11 trim, imagined-walking and rest
%   segments, bad-channel detection, interpolation, FCz added, common average reference, ASR on its target
%   segments (none inside the calibration copy), extended infomax ICA, ICLabel. rng('default') and four
%   computational threads, as cpc_step2_cal.m. min_amica_min (patients 3): below it prep_step2 skips the
%   ICA and this runner fails. step2_dir (default <out_root>/cal): a second folder gives the determinism
%   check (equal icaweights). Writes <step2_dir>/<label>_step2.set and the info file.
%   usage: cpp_step2_cal(label, out_root, min_amica_min, step2_dir)
if nargin < 3 || isempty(min_amica_min), min_amica_min = 3; end
if ischar(min_amica_min) || isstring(min_amica_min), min_amica_min = str2double(min_amica_min); end
if nargin < 4 || isempty(step2_dir), step2_dir = fullfile(out_root, 'cal'); end
ICA_THREADS = 4;
maxNumCompThreads(ICA_THREADS);
addpath(getenv("CPC_EEGLAB"));
eeglab nogui;
prep_patient = fullfile(getenv('CPC_ROOT'), '01_preprocessing', 'patient');
addpath(fullfile(prep_patient, 'shared'));
addpath(fullfile(prep_patient, '01_align'));
addpath(fullfile(prep_patient, '03_prep_eeg'));
prep_patient_paths();
here = fileparts(mfilename('fullpath'));
assert(strcmp(which('cpp_prep_step2'), fullfile(here, 'cpp_prep_step2.m')), 'cpp_prep_step2 resolves to %s', which('cpp_prep_step2'));
assert(strcmp(which('connector_fix_sessions'), fullfile(prep_patient, 'shared', 'connector_fix_sessions.m')), 'connector_fix_sessions path');
rng('default');
cal = fullfile(out_root, 'cal');
tag = 'step2_cal';
if ~strcmp(step2_dir, cal), tag = 'step2_det'; end
info = {sprintf('cpp_prep_step2: %s; min_amica_min %g; maxNumCompThreads %d (%d); step2 dir %s', which('cpp_prep_step2'), ...
                min_amica_min, ICA_THREADS, maxNumCompThreads, step2_dir)};
o = cpp_prep_step2(fullfile(cal, [label '_step1.set']), step2_dir, struct('overwrite', true, 'min_amica_min', min_amica_min));
S = pop_loadset('filename', [label '_step2.set'], 'filepath', step2_dir);
m = S.etc.step2_meta;
info{end+1} = sprintf('calibration step2: %d segments, %.1f s, bad ch %d {%s}, rank %d, ASR segments %d, ICs %d, max_step %d', ...
    m.n_trial_segs, m.trial_dur_sec, m.n_ch_interpolated, strjoin(m.bad_ch_labels, ','), m.data_rank, ...
    m.n_asr_segments, size(S.icaweights, 1), m.phase_b_max_step);
info{end+1} = sprintf('icaweights md5 %s; icasphere md5 %s', md5hex(S.icaweights), md5hex(S.icasphere));
info_dir = fullfile(out_root, 'info');
if ~exist(info_dir, 'dir'), mkdir(info_dir); end
fid = fopen(fullfile(info_dir, [label '_' tag '_info.txt']), 'w');
fprintf(fid, '%s\n', info{:});
fclose(fid);
fprintf('%s\n', info{:});
disp(o);
assert(~isempty(S.icaweights) && ~S.etc.skipped_amica, ...
       'cpp_step2_cal: %s calibration step2 has no ICA (task time %.1f s < %g min, or max_step %d)', ...
       label, m.trial_dur_sec, min_amica_min, m.phase_b_max_step);
end

function h = md5hex(x)
md = java.security.MessageDigest.getInstance('MD5');
md.update(typecast(double(x(:)), 'uint8'));
h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
end
