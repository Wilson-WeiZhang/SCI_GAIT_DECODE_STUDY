function cpc_export(sid, out_root)
%CPC_EXPORT  Export of one healthy session from its causal step 3: the triple <out_root>/causal_triple/<label>/.
%
%   Calls 01_preprocessing/healthy/04_export/export_session.m unchanged. The goniometer side (trial windows,
%   goniometer QC) is that of the frozen chain: the causal EEG folder links the per-part step-1 files the
%   export reads for the part headers, and <out_root>/causal/goni links the frozen goniometer folder.
%   usage: cpc_export(sid, out_root)
maxNumCompThreads(5);
E = cpc_env();
addpath(E.eeglab);
addpath(genpath(fullfile(E.root, '01_preprocessing', 'healthy')));
eeglab nogui;
qe = fullfile(E.root, 'data', 'prep_data_prep', 'healthy', 'eeg');
qg = fullfile(E.root, 'data', 'prep_data_prep', 'healthy', 'goni');
ce = fullfile(out_root, 'causal', 'eeg');
cg = fullfile(out_root, 'causal', 'goni');
if ~exist(cg, 'dir')
    [st, msg] = system(sprintf('ln -sfn %s %s', qg, cg));
    assert(st == 0, 'symlink %s: %s', cg, msg);
end
f = dir(fullfile(qe, [sid '*_step1.*']));
f = f(~cellfun(@isempty, regexp({f.name}, ['^' sid '[a-z]_step1\.(set|fdt)$'], 'once')));
for i = 1:numel(f)
    [st, msg] = system(sprintf('ln -sfn %s %s', fullfile(qe, f(i).name), fullfile(ce, f(i).name)));
    assert(st == 0, 'symlink %s: %s', f(i).name, msg);
end
fprintf('%s\nexport_session: %s; part step1 links: %s\n', sprintf('host %s; MATLAB %s; EEGLAB %s; maxNumCompThreads %d; root %s', E.host, version, eeg_getversion, maxNumCompThreads, E.root), which('export_session'), strjoin({f.name}, ', '));
o = export_session(sid, 'eeg_dir', ce, 'goni_dir', qg, 'out_root', fullfile(out_root, 'causal_triple'), ...
                   'raw_root', fullfile(E.root, 'data', 'raw_data'), 'eeg_stage', 'step3', 'overwrite', true);
disp(o);
assert(strcmp(o.status, 'ok'), 'cpc_export: %s export status %s', sid, o.status);
end
