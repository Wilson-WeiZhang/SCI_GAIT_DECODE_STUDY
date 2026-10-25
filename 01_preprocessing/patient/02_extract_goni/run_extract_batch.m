function results = run_extract_batch(varargin)
%RUN_EXTRACT_BATCH (patient, prep_patient)  Batch goni extraction -> _goni.mat per session.
%
%   For each patient step1.set in in_dir, look up registry entry, slice goni at
%   1000 Hz with task-aware active channels, save {label}_goni.mat in goni_dir.
%
%   Usage:
%     results = run_extract_batch();                              % all sessions in registry
%     results = run_extract_batch('only', {'pat02_sess03'});

    p = inputParser;
    addParameter(p, 'in_dir',   '${GAIT_PROJECT_ROOT}/data/prep_data/patient/eeg',  @ischar);
    addParameter(p, 'goni_dir', '${GAIT_PROJECT_ROOT}/data/prep_data/patient/goni', @ischar);
    addParameter(p, 'raw_root', '${GAIT_PROJECT_ROOT}/data/raw_data/patient', @ischar);
    addParameter(p, 'qc_png_dir', '${GAIT_PROJECT_ROOT}/01_preprocessing/patient/result/qc/goni_qc_png', @ischar);
    addParameter(p, 'only',     {}, @(x) iscell(x) || ischar(x));
    parse(p, varargin{:});
    opt = p.Results;
    if ischar(opt.only), opt.only = {opt.only}; end

    prep_patient_paths();   % add the shared and patient preprocessing folders

    sets = dir(fullfile(opt.in_dir, '*_step1.set'));
    if isempty(sets)
        error('run_extract_batch (prep_patient): no step1.set found in %s', opt.in_dir);
    end
    if ~exist(opt.goni_dir, 'dir'), mkdir(opt.goni_dir); end

    registry = build_patient_registry(opt.raw_root);
    reg_map = containers.Map('KeyType','char','ValueType','any');
    for k = 1:length(registry)
        s = registry(k);
        if isempty(s.label) || isempty(s.goni_paths), continue; end
        if strcmp(s.special_case, 'multi_goni_file')
            % Double-wrap: struct() treats single-cell value as struct-array dims.
            reg_map(s.label) = struct('raw_goni_paths', {{s.goni_paths{:}}});
        else
            reg_map(s.label) = struct('raw_goni_path', s.goni_paths{1});
        end
    end

    fprintf('\n==== extract (patient): %d step1.set | %d in registry ====\n', length(sets), reg_map.Count);
    fprintf('  in_dir   = %s\n  goni_dir = %s\n  raw_root = %s\n\n', opt.in_dir, opt.goni_dir, opt.raw_root);

    results = struct('label',{},'verdict',{},'n_trials',{},'n_problems',{},'err',{});
    t0 = tic;
    for i = 1:length(sets)
        path = fullfile(sets(i).folder, sets(i).name);
        [~, fname, ~] = fileparts(sets(i).name);
        lbl = strrep(fname, '_step1', '');
        if ~isempty(opt.only) && ~ismember(lbl, opt.only), continue; end

        fprintf('\n##### [%d/%d] %s #####\n', i, length(sets), lbl);
        if ~isKey(reg_map, lbl)
            results(end+1) = struct('label', lbl, 'verdict', 'SKIP', ...
                'n_trials', NaN, 'n_problems', NaN, 'err', 'not in registry'); %#ok<AGROW>
            fprintf('  SKIP - %s not in patient registry (or no goni)\n', lbl);
            continue;
        end
        try
            out = extract_goni_patient(path, opt.goni_dir, reg_map(lbl));
            results(end+1) = struct('label', lbl, 'verdict', 'OK', ...
                'n_trials', length(out.trials), 'n_problems', out.n_problems, 'err', ''); %#ok<AGROW>
            try
                if ~exist(opt.qc_png_dir, 'dir'), mkdir(opt.qc_png_dir); end
                plot_goni_qc(out.out_path, fullfile(opt.qc_png_dir, [lbl '_goni_qc.png']));
            catch ME_plot
                fprintf('  WARN plot_goni_qc failed: %s\n', ME_plot.message);
            end
        catch ME
            fprintf('  ERROR %s: %s\n', lbl, ME.message);
            results(end+1) = struct('label', lbl, 'verdict', 'ERR', ...
                'n_trials', NaN, 'n_problems', NaN, 'err', ME.message); %#ok<AGROW>
        end
    end
    fprintf('\n==== done in %.1f min ====\n', toc(t0) / 60);

    fprintf('\n%-22s %-8s %-9s %-11s %s\n', 'label', 'verdict', 'n_trials', 'n_problems', 'err');
    fprintf('%s\n', repmat('-', 1, 80));
    for i = 1:length(results)
        r = results(i);
        fprintf('%-22s %-8s %-9s %-11s %s\n', r.label, r.verdict, ...
            num2csv(r.n_trials), num2csv(r.n_problems), r.err);
    end
end

function s = num2csv(v)
    if isempty(v) || (isnumeric(v) && isnan(v)), s = '';
    elseif ischar(v), s = v;
    else, s = sprintf('%g', v);
    end
end
