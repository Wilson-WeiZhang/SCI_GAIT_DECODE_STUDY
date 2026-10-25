function cpc_session(sid, plan_csv, out_root, min_amica_min, use_asr)
%CPC_SESSION  The causal preprocessing of one healthy session, every MATLAB stage in one process.
%
%   cpc_session(sid, plan_csv, out_root, min_amica_min, use_asr)
%     cpc_step1 -> cpc_step2_cal -> cpc_step3 ('causal') -> cpc_export, each with its own log
%     (<out_root>/logs/<SID>_<stage>.log) and done marker (<out_root>/done/<SID>.<stage>; a stage with a
%     marker is skipped, so a session resumes). min_amica_min (default 5) is the prep_step2 option below
%     which ICA is skipped (the run then fails); use_asr (default true) keeps the ASR of step3 on the Walk
%     segments, as in the frozen chain. Appends the MATLAB / EEGLAB versions, the plan row (n_cal, cut), the
%     settings and the ICA seed (production, runica rand('state', 0)) to
%     <out_root>/info/<SID>_provenance.txt. cpc_check.py is the last stage.
%   usage: matlab -batch "addpath('01_preprocessing/causal/healthy'); cpc_session('SUB_01_sess01', 'plan.csv', '<out_root>', 5)"
if nargin < 4, min_amica_min = 5; end
if nargin < 5, use_asr = true; end      % step3 ASR as run (Walk segments only); false = no ASR
stages = {'step1', 'step2_cal', 'step3', 'export'};
calls = {@() cpc_step1(sid, plan_csv, out_root, false), @() cpc_step2_cal(sid, out_root, min_amica_min), ...
         @() cpc_step3(sid, out_root, 'causal', use_asr), @() cpc_export(sid, out_root)};
for d = {'logs', 'done'}
    if ~exist(fullfile(out_root, d{1}), 'dir'), mkdir(fullfile(out_root, d{1})); end
end
t_all = tic;
for k = 1:numel(stages)
    marker = fullfile(out_root, 'done', [sid '.' stages{k}]);
    if exist(marker, 'file'), fprintf('[cpc_session] %s %s: done marker, skipped\n', sid, stages{k}); continue; end
    lg = fullfile(out_root, 'logs', [sid '_' stages{k} '.log']);
    if exist(lg, 'file'), delete(lg); end
    diary(lg);
    t = tic;
    try
        calls{k}();
    catch err
        fprintf(2, '%s\n', getReport(err));
        diary off
        error('cpc_session:stageFailed', '%s failed at %s: %s', sid, stages{k}, err.message);
    end
    fprintf('[cpc_session] %s %s done in %.1f s\n', sid, stages{k}, toc(t));
    diary off
    fclose(fopen(marker, 'w'));
end
fprintf('[cpc_session] %s all stages done in %.1f s (this process)\n', sid, toc(t_all));
o = detectImportOptions(plan_csv, 'Delimiter', ',');
o = setvartype(o, {'sid', 'session', 'step1_parts', 'raw_vhdr', 'notes'}, 'char');
P = readtable(plan_csv, o);
r = P(strcmp(P.sid, sid), :);
E = cpc_env();
lines = {sprintf('host %s; MATLAB %s; EEGLAB %s; root %s', E.host, version, eeg_getversion, E.root), ...
         sprintf('maxNumCompThreads: ICA (cpc_step2_cal) 4; step3 / export 5; step1 MATLAB default under taskset'), ...
         sprintf('plan %s: n_cal %d (calib.csv %d, rule %s), cut_step1 %d (cut_rule_step1 %d, margin_s %g, cut_export_s %.2f), notes: %s', ...
                 plan_csv, r.n_cal, r.n_cal_calib, char(r.rule), r.cut_step1, r.cut_rule_step1, r.margin_s, r.cut_export_s, char(r.notes)), ...
         sprintf('step3 use_asr %d; min_amica_min %g; wall time this process %.1f s', use_asr, min_amica_min, toc(t_all)), ...
         sprintf('ICA seed: %s', ternary_seed(getenv('CPC_ICA_SEED')))};
fid = fopen(fullfile(out_root, 'info', [sid '_provenance.txt']), 'a');
fprintf(fid, '%s\n', lines{:});
fclose(fid);
end

function t = ternary_seed(v)
if isempty(v), t = 'production (runica rand(''state'', 0))'; else, t = ['CPC_ICA_SEED=' v]; end
end
