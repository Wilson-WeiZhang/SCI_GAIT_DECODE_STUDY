function out = make_qc_summary(varargin)
%MAKE_QC_SUMMARY (patient, prep_patient)  Aggregate QC across all processed sessions.
%
%   Outputs:
%     phase_a_summary.csv   - one row per session: align verdict + match stats
%                             (existing file from run_align_batch is preserved)
%     phase_b_summary.csv   - extract_goni n_trials, n_problems per condition
%     qc_summary.md         - markdown rollup with all registry sessions + status

    p = inputParser;
    addParameter(p, 'eeg_dir',  '${GAIT_PROJECT_ROOT}/data/prep_data/patient/eeg', @ischar);
    addParameter(p, 'goni_dir', '${GAIT_PROJECT_ROOT}/data/prep_data/patient/goni', @ischar);
    addParameter(p, 'qc_dir',   '${GAIT_PROJECT_ROOT}/01_preprocessing/patient/result/qc', @ischar);
    addParameter(p, 'out_root', '${GAIT_PROJECT_ROOT}/patient_prep', @ischar);
    parse(p, varargin{:});
    opt = p.Results;

    prep_patient_paths();

    % --- Session list from registry ---
    reg = build_patient_registry();
    n = length(reg);

    phase_a_csv  = fullfile(opt.qc_dir, 'phase_a_summary.csv');
    phase_b_csv  = fullfile(opt.qc_dir, 'phase_b_summary.csv');
    iclabel_csv  = fullfile(opt.qc_dir, 'iclabel_brain_count.csv');
    md_path      = fullfile(opt.qc_dir, 'qc_summary.md');

    if ~exist(opt.qc_dir, 'dir'), mkdir(opt.qc_dir); end

    fa = fopen(phase_a_csv, 'w');
    fprintf(fa, 'label,verdict,matched,n_s10,median_ms,p95_ms,max_ms,method,err\n');
    fb = fopen(phase_b_csv, 'w');
    fprintf(fb, 'label,extract_status,n_trials,n_imagine,n_s2s,n_walk,n_problems,goni_qc_png\n');
    fi = fopen(iclabel_csv, 'w');
    fprintf(fi, 'label,step3_status,total_ic,brain_gt_07,brain_gt_08,brain_gt_09,n_rejected,top1,top2,top3\n');
    fm = fopen(md_path, 'w');
    fprintf(fm, '# 01_preprocessing/patient — Patient QC Summary\n\n');
    fprintf(fm, 'Generated: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fm, 'Pipeline: patient preprocessing. %d registry sessions in scope (resting excluded).\n\n', n);
    fprintf(fm, '## Per-session status\n\n');
    fprintf(fm, '| Label | Stage 01 align | Stage 02 goni | Stage 03 EEG | Brain ICs (>.7/.8/.9) | Triple | Status |\n');
    fprintf(fm, '|---|---|---|---|---|---|---|\n');

    rows_b = {};
    rows_i = {};

    for k = 1:n
        r = reg(k);
        label = r.label;
        if isempty(label), continue; end

        align_ok  = false; align_str = '-';
        extract_ok = false; extract_str = '-';
        prep_ok   = false; prep_str = '-';
        triple_ok = false;
        brain_str = '-';
        notes = '';
        med_ms = NaN; max_ms = NaN;       % needed by status rollup
        n70 = NaN; n80 = NaN; n90 = NaN;

        % --- Stage 01 align (read align_log JSON if exists) ---
        json_path = fullfile(opt.qc_dir, 'align_log', [label '.json']);
        if exist(json_path, 'file')
            try
                txt = fileread(json_path); J = jsondecode(txt);
                v = '?'; if isfield(J, 'auto_verdict'), v = char(J.auto_verdict); end
                m = NaN; t = NaN;
                if isfield(J, 'n_matched'), m = J.n_matched; end
                if isfield(J, 'n_eeg_events_ttl'), t = J.n_eeg_events_ttl; end
                med_ms = NaN; p95_ms = NaN; max_ms = NaN;
                if isfield(J, 'residual_ms')
                    if isfield(J.residual_ms, 'median'), med_ms = scalar_or_nan(J.residual_ms.median); end
                    if isfield(J.residual_ms, 'p95'),    p95_ms = scalar_or_nan(J.residual_ms.p95); end
                    if isfield(J.residual_ms, 'max'),    max_ms = scalar_or_nan(J.residual_ms.max); end
                end
                method = ''; if isfield(J, 'match_method'), method = char(J.match_method); end
                align_str = sprintf('%s %d/%d', v, m, t);
                align_ok = any(strcmp(v, {'PASS', 'SPECIAL_CASE', 'PASS_LEGACY'}));
                fprintf(fa, '%s,%s,%d,%d,%.2f,%.2f,%.2f,%s,\n', label, v, m, t, med_ms, p95_ms, max_ms, method);
            catch ME
                align_str = '(parse-failed)';
                fprintf(fa, '%s,parse-fail,,,,,,,%s\n', label, ME.message);
            end
        else
            fprintf(fa, '%s,no-json,,,,,,,\n', label);
        end

        % --- Stage 02 goni ---
        goni_path = fullfile(opt.goni_dir, [label '_goni.mat']);
        if exist(goni_path, 'file')
            S = load(goni_path);
            n_trials = length(S.trials);
            n_i = sum(strcmp({S.trials.type}, 'imagine'));
            n_s = sum(strcmp({S.trials.type}, 's2s'));
            n_w = sum(strcmp({S.trials.type}, 'walk'));
            n_p = S.n_problems;
            extract_str = sprintf('%dT (I=%d S=%d W=%d, prob=%d)', n_trials, n_i, n_s, n_w, n_p);
            extract_ok = true;
            png_rel = fullfile('goni_qc_png', [label '_goni_qc.png']);
            png_abs = fullfile(opt.qc_dir, png_rel);
            png_str = ''; if exist(png_abs, 'file'), png_str = png_rel; end
            fprintf(fb, '%s,ok,%d,%d,%d,%d,%d,%s\n', label, n_trials, n_i, n_s, n_w, n_p, png_str);
        else
            fprintf(fb, '%s,missing,,,,,,\n', label);
        end

        % --- Stage 03 EEG (step3.set or step2 fallback) ---
        step3_path = fullfile(opt.eeg_dir, [label '_step3.set']);
        step2_path = fullfile(opt.eeg_dir, [label '_step2.set']);
        if exist(step3_path, 'file')
            try
                EEG = pop_loadset('filename', [label '_step3.set'], 'filepath', opt.eeg_dir, 'loadmode', 'info');
                ic = EEG.etc.ic_classification.ICLabel.classifications;
                brain = ic(:, 1);
                n_total = size(ic, 1);
                n70 = sum(brain > 0.7); n80 = sum(brain > 0.8); n90 = sum(brain > 0.9);
                top = sort(brain, 'descend');
                if length(top) >= 3
                    top1 = top(1); top2 = top(2); top3 = top(3);
                else
                    top1 = top(1); top2 = NaN; top3 = NaN;
                end
                n_rej = 0;
                if isfield(EEG.etc, 'step3_meta') && isfield(EEG.etc.step3_meta, 'n_ics_rejected')
                    n_rej = EEG.etc.step3_meta.n_ics_rejected;
                end
                prep_str = sprintf('step3 %dIC rej=%d', n_total, n_rej);
                brain_str = sprintf('%d/%d/%d', n70, n80, n90);
                prep_ok = true;
                fprintf(fi, '%s,step3,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f\n', ...
                    label, n_total, n70, n80, n90, n_rej, top1, top2, top3);
            catch ME
                prep_str = sprintf('step3 LOAD-FAIL: %s', ME.message);
                fprintf(fi, '%s,step3-load-fail,,,,,,,,\n', label);
            end
        elseif exist(step2_path, 'file')
            prep_str = 'step2 only';
            brain_str = 'n/a';
            prep_ok = true;
            fprintf(fi, '%s,step2_only,,,,,,,,\n', label);
        else
            fprintf(fi, '%s,missing,,,,,,,,\n', label);
        end

        % --- Triple check ---
        triple_dir = fullfile(opt.out_root, label);
        if exist(fullfile(triple_dir, [label '_eeg.mat']), 'file') && ...
           exist(fullfile(triple_dir, [label '_goni.mat']), 'file') && ...
           exist(fullfile(triple_dir, [label '_event.mat']), 'file')
            triple_ok = true;
        end
        triple_str = 'no'; if triple_ok, triple_str = 'yes'; end

        % --- Status rollup ---
        % `pipeline_complete` is loose (all artifacts produced).
        % Sessions that complete the pipeline but fail a quality gate get
        % `needs_review` so downstream analysts can spot them at a glance.
        ALIGN_RESID_MS_GATE = 5000;   % >5s median residual = degraded align
        BRAIN_70_GATE = 2;
        BRAIN_80_GATE = 2;
        BRAIN_90_GATE = 1;

        align_resid_ok = ~isnan(med_ms) && med_ms <= ALIGN_RESID_MS_GATE;
        brain_count_ok = ~isnan(n70) && ~isnan(n80) && ~isnan(n90) && ...
            n70 >= BRAIN_70_GATE && n80 >= BRAIN_80_GATE && n90 >= BRAIN_90_GATE;

        pipeline_complete = align_ok && extract_ok && prep_ok && triple_ok;

        if pipeline_complete && align_resid_ok && brain_count_ok
            status = 'OK';
        elseif pipeline_complete
            % All artifacts exist but a quality gate failed.
            reasons = {};
            if ~align_resid_ok
                reasons{end+1} = sprintf('align median %.0fms > %dms', ...
                    med_ms, ALIGN_RESID_MS_GATE); %#ok<AGROW>
            end
            if ~brain_count_ok
                reasons{end+1} = sprintf('brain %d/%d/%d < %d/%d/%d', ...
                    n70, n80, n90, BRAIN_70_GATE, BRAIN_80_GATE, BRAIN_90_GATE); %#ok<AGROW>
            end
            status = ['needs_review (' strjoin(reasons, '; ') ')'];
        elseif align_ok && ~prep_ok
            status = 'partial-align-only';
        elseif r.max_step < 3
            status = 'skipped (recording hardware failure)';
        elseif isempty(r.vhdr_paths) || isempty(r.goni_paths)
            status = 'deferred (raw_missing)';
        elseif strcmp(r.marker_protocol, 'legacy')
            status = 'deferred (legacy)';
        else
            status = '?';
        end

        fprintf(fm, '| %s | %s | %s | %s | %s | %s | %s |\n', ...
            label, align_str, extract_str, prep_str, brain_str, triple_str, status);
    end
    fclose(fa); fclose(fb); fclose(fi);

    % --- Summary section ---
    fprintf(fm, '\n## Summary\n\n');
    fprintf(fm, '- Outputs: `data/prep_data/patient/{eeg,goni}/`\n');
    
    
    fprintf(fm, '\nCSV files:\n- `phase_a_summary.csv` (from align stage, 1 row per attempted session)\n');
    fprintf(fm, '- `phase_b_summary.csv` (this run, extract_goni stats)\n');
    fprintf(fm, '- `iclabel_brain_count.csv` (this run, ICLabel brain-IC counts)\n');

    fclose(fm);
    fprintf('\n=== qc_summary done ===\n');
    fprintf('  %s\n  %s\n  %s\n', phase_b_csv, iclabel_csv, md_path);

    out = struct('phase_b_csv', phase_b_csv, 'iclabel_csv', iclabel_csv, 'md_path', md_path);
end

function v = scalar_or_nan(x)
    % JSON null → jsondecode returns []; coerce non-scalar/non-numeric to NaN.
    if isnumeric(x) && isscalar(x)
        v = x;
    else
        v = NaN;
    end
end
