function cpp_session(label, plan_csv, out_root, min_amica_min, use_asr)
%CPP_SESSION  The causal preprocessing of one patient session, every MATLAB stage in one process.
%
%   cpp_session(label, plan_csv, out_root, min_amica_min, use_asr)
%     cpp_step1 -> cpp_step2_cal -> cpp_step3 ('causal') -> cpp_export, with per-stage logs and done markers
%     as cpc_session.m. min_amica_min 3 for the patients (the task segments of the first natural segment of a
%     patient session last 111-238 s); use_asr as in the frozen patient chain (ASR only on the sit-to-stand
%     and two-minute walking segments, never on imagined walking or rest). Writes the stage timings, the
%     MATLAB / EEGLAB versions, the plan row and the thread counts to <out_root>/info/<label>_session_info.txt.
%     cpp_check.py is the last stage.
%   usage: matlab -batch "addpath('01_preprocessing/causal/patient'); cpp_session('pat02_sess03', 'plan.csv', '<out_root>', 3, 1)"
if ischar(min_amica_min) || isstring(min_amica_min), min_amica_min = str2double(min_amica_min); end
if ischar(use_asr) || isstring(use_asr), use_asr = str2double(use_asr); end
stages = {'step1', 'step2_cal', 'step3', 'export'};
calls = {@() cpp_step1(label, plan_csv, out_root, false), @() cpp_step2_cal(label, out_root, min_amica_min), ...
         @() cpp_step3(label, plan_csv, out_root, 'causal', use_asr), @() cpp_export(label, plan_csv, out_root)};
for d = {'logs', 'done', 'info'}
    if ~exist(fullfile(out_root, d{1}), 'dir'), mkdir(fullfile(out_root, d{1})); end
end
fid = fopen(fullfile(out_root, 'info', [label '_session_info.txt']), 'a');
fprintf(fid, '%s cpp_session %s: min_amica_min %g, use_asr %d, plan %s\n', char(datetime('now')), label, min_amica_min, use_asr, plan_csv);
write_provenance(fid, label, plan_csv, out_root);
t_all = tic;
for i = 1:numel(stages)
    marker = fullfile(out_root, 'done', [label '.' stages{i}]);
    if exist(marker, 'file'), fprintf(fid, '  %-9s skipped (done marker)\n', stages{i}); continue; end
    lg = fullfile(out_root, 'logs', [label '_' stages{i} '.log']);
    if exist(lg, 'file'), delete(lg); end
    diary(lg);
    t = tic;
    try
        calls{i}();
    catch ME
        diary off;
        fprintf(fid, '  %-9s FAILED after %.1f s: %s\n', stages{i}, toc(t), ME.message);
        fclose(fid);
        rethrow(ME);
    end
    diary off;
    fclose(fopen(marker, 'w'));
    fprintf(fid, '  %-9s %.1f s (maxNumCompThreads %d)\n', stages{i}, toc(t), maxNumCompThreads);
    fprintf('[cpp_session] %s %s done in %.1f s\n', label, stages{i}, toc(t));
end
fprintf(fid, '  total MATLAB %.1f s\n', toc(t_all));
fclose(fid);
end

function write_provenance(fid, label, plan_csv, out_root)
[~, host] = system('hostname');
fprintf(fid, '  host %s; MATLAB %s\n', strtrim(host), version);
[st, lst] = system(['bash ' getenv('CPC_EEGLAB_MD5_SH') ' ' getenv('CPC_EEGLAB') ' 2>/dev/null']);
assert(st == 0 || ~isempty(lst), 'eeglab_md5.sh failed');
f2 = fopen(fullfile(out_root, 'info', [label '_eeglab_md5.txt']), 'w');
fwrite(f2, lst);
fclose(f2);
fprintf(fid, '  EEGLAB listing (CPC_EEGLAB_MD5_SH CPC_EEGLAB, %d lines): combined md5 %s\n', ...
        numel(regexp(lst, '\n')), md5bytes(uint8(lst)));
o = detectImportOptions(plan_csv, 'Delimiter', ',');
o = setvartype(o, {'sid', 'lane', 'relabel_override', 'marker_protocol', 'export_fn', 'n_cal_rule', 'step1_dir', ...
                   'step23_dir', 'goni_dir'}, 'char');
P = readtable(plan_csv, o);
r = table2struct(P(strcmp(P.sid, label), :));
fprintf(fid, '  plan: lane %s; connector_fix %d; relabel ''%s''; marker_protocol %s; rest_split %d; dur_tol_s %g; export %s\n', ...
        r.lane, r.connector_fix, r.relabel_override, r.marker_protocol, r.rest_split, r.dur_tol_s, r.export_fn);
fprintf(fid, '  plan: step1 %s; as-run step2/3 %s; goni %s\n', r.step1_dir, r.step23_dir, r.goni_dir);
fprintf(fid, '  plan: n_cal %d (%s; calib.csv %d); cut_step1 %d (%.2f s; export %.2f s); est task %.1f s; approved %d\n', ...
        r.n_cal, r.n_cal_rule, r.n_cal_calib, r.cut_step1, r.cut_s, r.cut_export_s, r.est_cal_task_s, r.approved);
here = fileparts(mfilename('fullpath'));
f = [dir(fullfile(here, 'cpp_*.m')); dir(fullfile(here, 'cpp_*.py')); dir(fullfile(here, 'cpp_*.sh'))];
for i = 1:numel(f)
    g = fopen(fullfile(here, f(i).name), 'r');
    b = fread(g, inf, '*uint8');
    fclose(g);
    fprintf(fid, '  md5 %s %s\n', md5bytes(b), f(i).name);
end
end

function h = md5bytes(b)
md = java.security.MessageDigest.getInstance('MD5');
md.update(b(:));
h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
end
