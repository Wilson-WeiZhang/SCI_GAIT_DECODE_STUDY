function results = run_extract_batch(varargin)
%RUN_EXTRACT_BATCH  Batch goniometer extraction: one {label}_goni.mat per session.
%
%   For each step1.set in in_dir, locate the raw goniometer file through the
%   session registry, run extract_goni and save {label}_goni.mat in goni_dir.
%
%   Usage:
%     results = run_extract_batch();                                % all sessions
%     results = run_extract_batch('only', {'SUB_07_sess01'});
%

    p = inputParser;
    addParameter(p, 'in_dir',   '${GAIT_PROJECT_ROOT}/data/prep_data/healthy/eeg',  @ischar);
    addParameter(p, 'goni_dir', '${GAIT_PROJECT_ROOT}/data/prep_data/healthy/goni', @ischar);
    addParameter(p, 'raw_root', '${GAIT_PROJECT_ROOT}/data/raw_data', @ischar);
    addParameter(p, 'only',     {}, @(x) iscell(x) || ischar(x));
    parse(p, varargin{:});
    opt = p.Results;
    if ischar(opt.only), opt.only = {opt.only}; end

    sets = dir(fullfile(opt.in_dir, '*_step1.set'));
    if isempty(sets)
        error('run_extract_batch: no step1.set found in %s', opt.in_dir);
    end
    if ~exist(opt.goni_dir, 'dir'), mkdir(opt.goni_dir); end

    % Build registry: session label → raw goni path
    sessions = build_session_registry(opt.raw_root);
    reg_map = containers.Map('KeyType','char','ValueType','any');
    for k = 1:length(sessions)
        s = sessions(k);
        if isfield(s, 'goni_paths_multi') && ~isempty(s.goni_paths_multi)
            % Double-wrap the cell so struct() doesn't treat it as struct-array dims.
            reg_map(s.label) = struct('raw_goni_paths', {{s.goni_paths_multi{:}}});
        elseif isfield(s, 'goni_path_override') && ~isempty(s.goni_path_override)
            reg_map(s.label) = struct('raw_goni_path', s.goni_path_override);
        else
            try
                [eeg_dir, eeg_name, ~] = fileparts(s.vhdr);
                reg_map(s.label) = struct('raw_goni_path', find_goni_file(eeg_dir, eeg_name));
            catch
                % no goni found; skip
            end
        end
    end

    fprintf('\n==== extract: %d step1.set | %d in registry ====\n', length(sets), reg_map.Count);
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
            fprintf('  SKIP — %s not in session registry\n', lbl);
            continue;
        end
        try
            out = extract_goni(path, opt.goni_dir, reg_map(lbl));
            % Recompute the final goniometer QC verdict for this session
            % (04_qc/recompute_goni_qc.m).
            try
                recompute_goni_qc('goni_dir', opt.goni_dir, ...
                    'pattern', [lbl '_goni.mat']);
            catch ME_qc
                fprintf('  WARN goniometer QC recompute failed for %s: %s\n', lbl, ME_qc.message);
            end
            results(end+1) = struct('label', lbl, 'verdict', 'OK', ...
                'n_trials', length(out.trials), 'n_problems', out.n_problems, 'err', ''); %#ok<AGROW>
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
