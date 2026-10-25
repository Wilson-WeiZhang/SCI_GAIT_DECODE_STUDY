function results = run_align_batch(varargin)
%RUN_ALIGN_BATCH  Alignment batch runner for the healthy sessions.
%
%   Usage:
%     results = run_align_batch();                              % all sessions
%     results = run_align_batch('only', {'SUB_01_sess01'});     % subset
%     results = run_align_batch('skip_existing', true);         % skip if _step1.set exists
%     results = run_align_batch('out_dir', '/path', 'qc_dir', '/path');
%
%   Writes:
%     {out_dir}/{label}_step1.set
%     {qc_dir}/align_log/{label}.json
%     {qc_dir}/align_png/{label}.png
%     {qc_dir}/phase_a_summary.csv
%
%   Sessions recorded as several files are listed and skipped at this stage.
%
%   The per-session settings (raw file locations, special cases, goniometer edge
%   overrides) are read from build_session_registry(raw_root). That registry is not
%   distributed with this repository, because it holds the file layout of the raw
%   recordings.

    p = inputParser;
    addParameter(p, 'only', {}, @(x) iscell(x) || ischar(x));
    addParameter(p, 'skip_existing', false, @islogical);
    addParameter(p, 'out_dir', '${GAIT_PROJECT_ROOT}/data/prep_data/healthy/eeg', @ischar);
    addParameter(p, 'qc_dir',  '${GAIT_PROJECT_ROOT}/01_preprocessing/shared/result/qc', @ischar);
    addParameter(p, 'raw_root','${GAIT_PROJECT_ROOT}/data/raw_data', @ischar);
    parse(p, varargin{:});
    opt = p.Results;
    if ischar(opt.only), opt.only = {opt.only}; end

    % ---- Session registry (single-file healthy only) ----
    sessions = build_session_registry(opt.raw_root);

    % Apply 'only' filter
    if ~isempty(opt.only)
        keep = ismember({sessions.label}, opt.only);
        sessions = sessions(keep);
    end

    fprintf('\n==== Phase A: align %d sessions ====\n', length(sessions));

    if ~exist(opt.out_dir, 'dir'), mkdir(opt.out_dir); end
    if ~exist(opt.qc_dir,  'dir'), mkdir(opt.qc_dir);  end

    results = repmat(struct('label','','verdict','','matched',NaN,'ttl',NaN, ...
        'max_ms',NaN,'err',''), 1, length(sessions));

    t0 = tic;
    for i = 1:length(sessions)
        s = sessions(i);
        fprintf('\n\n######## [%d/%d] %s ########\n', i, length(sessions), s.label);
        set_out = fullfile(opt.out_dir, [s.label '_step1.set']);
        if opt.skip_existing && exist(set_out, 'file')
            fprintf('     skip_existing: %s\n', set_out);
            results(i).label = s.label; results(i).verdict = 'SKIP_EXIST';
            continue;
        end
        try
            prep_opts = struct('label', s.label, 'special_case', s.special_case);
            if isfield(s, 'spurious_goni_override') && ~isempty(s.spurious_goni_override)
                prep_opts.spurious_goni_override = s.spurious_goni_override;
            end
            if isfield(s, 'goni_path_override') && ~isempty(s.goni_path_override)
                prep_opts.goni_path_override = s.goni_path_override;
            end
            if isfield(s, 'goni_paths_multi') && ~isempty(s.goni_paths_multi)
                prep_opts.goni_paths_multi = s.goni_paths_multi;
            end
            out = prep_align(s.vhdr, opt.out_dir, opt.qc_dir, prep_opts);
            results(i) = struct( ...
                'label',   s.label, ...
                'verdict', out.verdict, ...
                'matched', out.align_log.n_matched, ...
                'ttl',     out.align_log.n_eeg_events_ttl, ...
                'max_ms',  out.align_log.residual_ms.max, ...
                'err',     '');
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            results(i).label   = s.label;
            results(i).verdict = 'ERROR';
            results(i).err     = ME.message;
        end
    end
    fprintf('\n==== done in %.1f min ====\n', toc(t0)/60);

    % ---- Write CSV summary ----
    csv_path = fullfile(opt.qc_dir, 'phase_a_summary.csv');
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'label,verdict,matched,ttl,max_ms,err\n');
    for i = 1:length(results)
        r = results(i);
        fprintf(fid, '%s,%s,%s,%s,%s,"%s"\n', ...
            r.label, r.verdict, ...
            num2csv(r.matched), num2csv(r.ttl), num2csv(r.max_ms), ...
            strrep(r.err, '"', ''''));
    end
    fclose(fid);
    fprintf('Wrote %s\n', csv_path);

    % Print summary table
    fprintf('\n%-18s %-10s %8s %6s %6s\n', ...
        'session','verdict','matched','ttl','maxMs');
    for i = 1:length(results)
        r = results(i);
        fprintf('%-18s %-10s %8s %6s %6s\n', ...
            r.label, r.verdict, ...
            num2str(r.matched), num2str(r.ttl), num2str(r.max_ms));
    end
end


% ======================== helpers ========================

function s = num2csv(v)
    if isnan(v), s = ''; else, s = num2str(v); end
end
