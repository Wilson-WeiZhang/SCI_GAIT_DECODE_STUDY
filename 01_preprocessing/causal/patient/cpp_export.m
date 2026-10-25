function cpp_export(label, plan_csv, out_root)
%CPP_EXPORT  Export of one patient session from its causal step 3: the triple <out_root>/causal_triple/<label>/.
%
%   Calls the frozen patient export of the session's lane unchanged (01_preprocessing/patient/05_export/
%   export_session.m; for the session whose goniometer channels are relabelled, the relabelling export of
%   the frozen chain). The goniometer side is that of the frozen chain; the causal EEG folder holds no
%   step 2, so a missing step 3 makes the export fail instead of falling back to step-2 data. The
%   goniometer QC override is applied at decode time (GAIT_GONI_QC_TRIAL_FLAGS), not here.
%   usage: cpp_export(label, plan_csv, out_root)
maxNumCompThreads(8);
addpath(getenv("CPC_EEGLAB"));
eeglab nogui;
prep_patient = fullfile(getenv('CPC_ROOT'), '01_preprocessing', 'patient');
addpath(fullfile(prep_patient, 'shared'));
addpath(fullfile(prep_patient, '01_align'));
addpath(fullfile(prep_patient, '02_extract_goni'));
addpath(fullfile(prep_patient, '05_export'));
prep_patient_paths();
r = plan_row(plan_csv, label);
ce = fullfile(out_root, 'causal', 'eeg');
assert(exist(fullfile(ce, [label '_step3.set']), 'file') == 2, 'no causal step3 for %s', label);
assert(~exist(fullfile(ce, [label '_step2.set']), 'file'), 'a step2 sits in the causal eeg dir');
args = {'eeg_dir', ce, 'goni_dir', r.goni_dir, 'out_root', fullfile(out_root, 'causal_triple'), 'overwrite', true};
switch r.export_fn
    case 'export_session'
        assert(isempty(r.relabel_override), 'relabel override without export_session_relabel');
        fprintf('export_session: %s; goni_dir %s\n', which('export_session'), r.goni_dir);
        o = export_session(label, args{:});
    case 'export_session_relabel'
        ovr = strsplit(r.relabel_override, ';');
        assert(numel(ovr) == 3, 'relabel override %s', r.relabel_override);
        fprintf('export_session_relabel: %s; goni_dir %s; override %s\n', which('export_session_relabel'), r.goni_dir, r.relabel_override);
        o = export_session_relabel(label, args{:}, 'goni_label_override', ovr);
    otherwise
        error('cpp_export: export_fn %s', r.export_fn);
end
disp(o);
assert(strcmp(o.status, 'ok') && strcmp(o.eeg_source, 'step3'), 'cpp_export: %s export status %s', label, o.status);
end

function r = plan_row(plan_csv, label)
o = detectImportOptions(plan_csv, 'Delimiter', ',');
o = setvartype(o, {'sid', 'session', 'lane', 'step1_dir', 'step23_dir', 'goni_dir', 'export_fn', 'relabel_override'}, 'char');
P = readtable(plan_csv, o);
k = find(strcmp(P.sid, label));
assert(numel(k) == 1, 'cpp_export: %s not (uniquely) in %s', label, plan_csv);
r = table2struct(P(k, :));
if isnumeric(r.relabel_override) || (ischar(r.relabel_override) && strcmpi(r.relabel_override, 'nan')), r.relabel_override = ''; end
end
