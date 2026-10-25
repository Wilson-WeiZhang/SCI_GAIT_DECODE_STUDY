function cpp_step3(label, plan_csv, out_root, mode, use_asr)
%CPP_STEP3  Step 3 of one patient session through cpp_prep_step3.m.
%
%   mode 'causal' (default): the causal step 1 (cpp_step1) with the ICA weights, ICLabel input, bad channels
%       and trial definitions of the calibration step 2 (cpp_step2_cal), the 0.1-40 Hz band-pass converted to
%       minimum phase and applied forward only, ASR as in the frozen chain -> <out_root>/causal/eeg/<label>_step3.set.
%       Components are rejected by ICLabel only.
%   mode 'g2': the frozen step 1 and step 2 of the session's lane with the zero-phase band-pass must
%       reproduce its frozen step 3 bit for bit, with the same rejected components (run at the thread count
%       of the frozen run, plan.asrun_step23_threads: ASR results depend on it).
%   Writes <out_root>/info/<label>_step3_<mode>_info.txt.
%   usage: cpp_step3(label, plan_csv, out_root, mode, use_asr)
if nargin < 4 || isempty(mode), mode = 'causal'; end
if nargin < 5 || isempty(use_asr), use_asr = true; end
if ischar(use_asr) || isstring(use_asr), use_asr = any(strcmpi(char(use_asr), {'1', 'true'})); end
addpath(getenv("CPC_EEGLAB"));
eeglab nogui;
prep_patient = fullfile(getenv('CPC_ROOT'), '01_preprocessing', 'patient');
addpath(fullfile(prep_patient, 'shared'));
addpath(fullfile(prep_patient, '01_align'));
addpath(fullfile(prep_patient, '03_prep_eeg'));
prep_patient_paths();
here = fileparts(mfilename('fullpath'));
assert(strcmp(which('cpp_prep_step3'), fullfile(here, 'cpp_prep_step3.m')), 'cpp_prep_step3 resolves to %s', which('cpp_prep_step3'));
assert(~isempty(which('iclabel')), 'iclabel not on the path');
r = plan_row(plan_csv, label);
if strcmp(mode, 'g2'), nthr = r.asrun_step23_threads; else, nthr = 8; end
maxNumCompThreads(nthr);
A = pop_loadset('filename', [label '_step3.set'], 'filepath', r.step23_dir);
switch mode
    case 'causal'
        od = fullfile(out_root, 'causal', 'eeg');
        o = cpp_prep_step3(fullfile(od, [label '_step1.set']), od, struct('overwrite', true, ...
                 'step2_path', fullfile(out_root, 'cal', [label '_step2.set']), 'bp_minphase', 1, 'use_asr', use_asr));
    case 'g2'
        od = fullfile(out_root, 'g2');
        o = cpp_prep_step3(fullfile(r.step1_dir, [label '_step1.set']), od, struct('overwrite', true, ...
                 'step2_path', fullfile(r.step23_dir, [label '_step2.set']), 'bp_minphase', 0, 'use_asr', true));
    otherwise
        error('cpp_step3: mode %s', mode);
end
disp(o);
assert(strcmp(o.status, 'ok'), 'cpp_step3: %s step3 status %s', label, o.status);
C = pop_loadset('filename', [label '_step3.set'], 'filepath', od);
d = NaN;
if isequal(size(C.data), size(A.data)), d = max(abs(double(C.data(:)) - double(A.data(:)))); end
evs = @(E) [{E.event.type}; num2cell([E.event.latency])];
same_ev = isequal(evs(C), evs(A));
info = {sprintf('%s step3 (%s, use_asr %d, %d threads) %d x %d, as-run %s %d x %d; events %d / %d (type+latency identical %d); max|this - as-run| = %.3g (max|as-run| %.3g); ASR segments %d / %d', ...
                label, mode, use_asr, maxNumCompThreads, C.nbchan, C.pnts, r.step23_dir, A.nbchan, A.pnts, numel(C.event), numel(A.event), same_ev, ...
                d, max(abs(double(A.data(:)))), C.etc.step3_meta.n_asr_segments, A.etc.step3_meta.n_asr_segments), ...
        sprintf('rejected ICs: this %s of %d (%s); as-run %s of %d (%s)', mat2str(C.etc.step3_meta.rejected_ics(:)'), ...
                size(C.etc.step3_meta.step2_iclabel_classifications, 1), strjoin(C.etc.step3_meta.ic_labels, ','), ...
                mat2str(A.etc.step3_meta.rejected_ics(:)'), size(A.etc.step3_meta.step2_iclabel_classifications, 1), ...
                strjoin(A.etc.step3_meta.ic_labels, ','))};
info_dir = fullfile(out_root, 'info');
if ~exist(info_dir, 'dir'), mkdir(info_dir); end
fid = fopen(fullfile(info_dir, [label '_step3_' mode '_info.txt']), 'w');
fprintf(fid, '%s\n', info{:});
fclose(fid);
fprintf('%s\n', info{:});
assert(isequal(size(C.data), size(A.data)) && same_ev, 'cpp_step3: %s step3 size / events differ from the as-run step3', label);
if strcmp(mode, 'g2')
    assert(d == 0 && isequal(C.etc.step3_meta.rejected_ics, A.etc.step3_meta.rejected_ics), 'G2 failed for %s', label);
end
end

function r = plan_row(plan_csv, label)
o = detectImportOptions(plan_csv, 'Delimiter', ',');
o = setvartype(o, {'sid', 'session', 'lane', 'step1_dir', 'step23_dir', 'goni_dir', 'export_fn', 'relabel_override'}, 'char');
P = readtable(plan_csv, o);
k = find(strcmp(P.sid, label));
assert(numel(k) == 1, 'cpp_step3: %s not (uniquely) in %s', label, plan_csv);
r = table2struct(P(k, :));
end
