function run_prep_one(sid)
%RUN_PREP_ONE  01_preprocessing/healthy per-session: prep_step3 (clean) + export.
%
maxNumCompThreads(5);
addpath('${GAIT_DATA_ROOT}/eeglab2024');
addpath(genpath('${GAIT_PROJECT_ROOT}/01_preprocessing'));
eeglab nogui;

qe    = '${GAIT_PROJECT_ROOT}/data/prep_data_prep/healthy/eeg';
qg    = '${GAIT_PROJECT_ROOT}/data/prep_data_prep/healthy/goni';
troot = '${GAIT_PROJECT_ROOT}/data/healthy_prep';

step1 = fullfile(qe, [sid '_step1.set']);
assert(exist(step1, 'file') == 2, 'run_prep_one:noStep1', 'no step1 for %s in %s', sid, qe);

t0 = tic;
o3 = prep_step3(step1, qe, struct('overwrite', false));
oe = export_session(sid, 'eeg_dir', qe, 'goni_dir', qg, ...
        'out_root', troot, 'eeg_stage', 'step3', 'overwrite', false);
st = 'NA'; if isfield(oe, 'status'), st = oe.status; end
fprintf('=== PREP %s step3=%s n_rej=%d export=%s in %.1f min ===\n', ...
    sid, o3.status, o3.n_rej, st, toc(t0)/60);
end
