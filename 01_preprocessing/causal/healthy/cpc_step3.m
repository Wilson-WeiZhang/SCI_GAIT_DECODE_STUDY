function cpc_step3(sid, out_root, mode, use_asr)
%CPC_STEP3  Step 3 of one healthy session through cp_step3.m.
%
%   mode 'causal' (default): the causal step 1 (cpc_step1) with the ICA weights, ICLabel classification and
%       bad channels of the calibration step 2 (cpc_step2_cal), the 0.1-40 Hz band-pass of prep_step3
%       converted to minimum phase and applied forward only, and ASR on the Walk segments as in prep_step3
%       (use_asr true; imagined-walking samples never pass through ASR) -> <out_root>/causal/eeg/<SID>_step3.set.
%       Components are rejected by ICLabel only (asserted: no manual list next to cp_step3.m).
%   mode 'g2': the reproduction check: the frozen step 1 and step 2 of the session with the zero-phase
%       band-pass must reproduce the frozen step 3 bit for bit -> <out_root>/g2/<SID>_step3.set.
%   Five computational threads. Writes <out_root>/info/<SID>_step3_<mode>_info.txt.
%   usage: cpc_step3(sid, out_root, mode, use_asr)
if nargin < 3, mode = 'causal'; end
if nargin < 4, use_asr = true; end
if ischar(use_asr) || isstring(use_asr), use_asr = any(strcmpi(char(use_asr), {'1', 'true'})); end
maxNumCompThreads(5);
E = cpc_env();
addpath(E.eeglab);
addpath(genpath(fullfile(E.root, '01_preprocessing', 'healthy')));
addpath(E.probe);
eeglab nogui;
cp = which('cp_step3');
assert(strcmp(cp, fullfile(E.probe, 'cp_step3.m')), 'cp_step3 resolves to %s', cp);
assert(~exist(fullfile(fileparts(cp), 'manual_eye_ic_overrides.csv'), 'file'), 'a manual eye-IC override file sits next to cp_step3');
qe = fullfile(E.root, 'data', 'prep_data_prep', 'healthy', 'eeg');
A = pop_loadset('filename', [sid '_step3.set'], 'filepath', qe);
switch mode
    case 'causal'
        od = fullfile(out_root, 'causal', 'eeg');
        cp_step3(fullfile(od, [sid '_step1.set']), od, struct('overwrite', true, ...
                 'step2_path', fullfile(out_root, 'cal', [sid '_step2.set']), 'bp_minphase', 1, 'use_asr', logical(use_asr)));
    case 'g2'
        assert(~strcmp(sid, 'SUB_33_sess01'), 'cpc_step3: g2 is not defined for %s', sid);
        od = fullfile(out_root, 'g2');
        cp_step3(fullfile(qe, [sid '_step1.set']), od, struct('overwrite', true, ...
                 'step2_path', fullfile(qe, [sid '_step2.set']), 'bp_minphase', 0, 'use_asr', true));
    otherwise
        error('cpc_step3: mode %s', mode);
end
C = pop_loadset('filename', [sid '_step3.set'], 'filepath', od);
d = NaN;
if isequal(size(C.data), size(A.data)), d = max(abs(double(C.data(:)) - double(A.data(:)))); end
info = {sprintf('host %s; MATLAB %s; EEGLAB %s; maxNumCompThreads %d; root %s', E.host, version, eeg_getversion, maxNumCompThreads, E.root), sprintf('%s step3 (%s) %d x %d, as-run prep %d x %d; events %d / %d; max|this - as-run| = %.3g (max|as-run| %.3g)', ...
                sid, [mode sprintf(', use_asr %d', use_asr)], C.nbchan, C.pnts, A.nbchan, A.pnts, numel(C.event), numel(A.event), d, max(abs(double(A.data(:))))), ...
        sprintf('rejected ICs: this %s of %d (%s); as-run %s of %d (%s)', mat2str(C.etc.step3_meta.rejected_ics(:)'), ...
                size(C.etc.step3_meta.step2_iclabel_classifications, 1), strjoin(C.etc.step3_meta.ic_labels, ','), ...
                mat2str(A.etc.step3_meta.rejected_ics(:)'), size(A.etc.step3_meta.step2_iclabel_classifications, 1), ...
                strjoin(A.etc.step3_meta.ic_labels, ','))};
info_dir = fullfile(out_root, 'info');
if ~exist(info_dir, 'dir'), mkdir(info_dir); end
fid = fopen(fullfile(info_dir, [sid '_step3_' mode '_info.txt']), 'w');
fprintf(fid, '%s\n', info{:});
fclose(fid);
fprintf('%s\n', info{:});
assert(isequal(size(C.data), size(A.data)) && numel(C.event) == numel(A.event), ...
       'cpc_step3: %s step3 size / events differ from the as-run step3', sid);
if strcmp(mode, 'g2')
    assert(d <= 1e-6 * max(abs(double(A.data(:)))) && isequal(C.etc.step3_meta.rejected_ics, A.etc.step3_meta.rejected_ics), ...
           'G2 failed for %s', sid);
end
end
