function results = run_align_batch(varargin)
%RUN_ALIGN_BATCH  Alignment batch runner for the patient sessions.
%
%   Usage:
%     results = run_align_batch();
%     results = run_align_batch('only', {'pat02_sess03'});           % subset
%     results = run_align_batch('skip_existing', true);              % skip if step1 exists
%     results = run_align_batch('out_dir', '/path', 'qc_dir', '/path');
%
%   Writes:
%     {out_dir}/{label}_step1.set
%     {qc_dir}/align_log/{label}.json
%     {qc_dir}/align_png/{label}.png
%     {qc_dir}/phase_a_summary.csv
%
%   Sessions with special_case='multi_block' are listed and skipped.

    prep_patient_paths();

    p = inputParser;
    addParameter(p, 'only', {}, @(x) iscell(x) || ischar(x));
    addParameter(p, 'skip_existing', false, @islogical);
    addParameter(p, 'raw_root', '', @ischar);
    addParameter(p, 'out_dir',  '${GAIT_PROJECT_ROOT}/data/prep_data/patient/eeg', @ischar);
    addParameter(p, 'qc_dir',   '${GAIT_PROJECT_ROOT}/01_preprocessing/patient/result/qc',     @ischar);
    parse(p, varargin{:});
    opt = p.Results;
    if ischar(opt.only), opt.only = {opt.only}; end

    if isempty(opt.raw_root)
        sessions = build_patient_registry();
    else
        sessions = build_patient_registry(opt.raw_root);
    end

    if ~isempty(opt.only)
        keep = ismember({sessions.label}, opt.only);
        if ~any(keep)
            error('run_align_batch:noMatch', 'No sessions match: %s', strjoin(opt.only, ', '));
        end
        sessions = sessions(keep);
    end

    fprintf('\n==== Phase A (patient): align %d session%s ====\n', ...
        length(sessions), plural(length(sessions)));

    if ~exist(opt.out_dir, 'dir'), mkdir(opt.out_dir); end
    if ~exist(opt.qc_dir,  'dir'), mkdir(opt.qc_dir);  end

    results = repmat(struct( ...
        'label', '', 'verdict', '', 'matched', NaN, 'n_s10', NaN, ...
        'max_ms', NaN, 'err', ''), 1, length(sessions));

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
            out = prep_align_patient(s, opt.out_dir, opt.qc_dir);
            results(i).label   = s.label;
            results(i).verdict = out.verdict;
            if isfield(out.align_log, 'n_matched')
                results(i).matched = out.align_log.n_matched;
                results(i).n_s10   = out.align_log.n_eeg_events_ttl;
                if isfield(out.align_log.residual_ms, 'max')
                    results(i).max_ms = out.align_log.residual_ms.max;
                end
            end
        catch ME
            fprintf(2, '  ERROR %s: %s\n', s.label, ME.message);
            results(i).label   = s.label;
            results(i).verdict = 'ERROR';
            results(i).err     = ME.message;
        end
    end

    fprintf('\n==== Phase A done in %.1f min ====\n', toc(t0)/60);

    % Write CSV summary
    csv_path = fullfile(opt.qc_dir, 'phase_a_summary.csv');
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'label,verdict,matched,n_s10,max_ms,err\n');
    for i = 1:length(results)
        r = results(i);
        fprintf(fid, '%s,%s,%s,%s,%s,%s\n', r.label, r.verdict, ...
            num2str_or_empty(r.matched), num2str_or_empty(r.n_s10), ...
            num2str_or_empty(r.max_ms, '%.1f'), strrep(r.err, ',', ';'));
    end
    fclose(fid);
    fprintf('     summary: %s\n', csv_path);

    % Per-verdict tally
    verdicts = {results.verdict};
    [u, ~, ix] = unique(verdicts);
    counts = accumarray(ix, 1);
    fprintf('\n     verdict tally:\n');
    for k = 1:length(u)
        fprintf('       %-15s %d\n', u{k}, counts(k));
    end
end


function p = plural(n)
    if n == 1, p = ''; else, p = 's'; end
end

function s = num2str_or_empty(v, fmt)
    if nargin < 2, fmt = '%g'; end
    if isempty(v) || (isnumeric(v) && isnan(v))
        s = '';
    else
        s = sprintf(fmt, v);
    end
end
