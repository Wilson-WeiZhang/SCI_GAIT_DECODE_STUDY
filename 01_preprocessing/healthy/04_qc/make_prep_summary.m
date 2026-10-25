function make_prep_summary(varargin)
%MAKE_PREP_SUMMARY  EEG quality-control summary of the healthy cohort (stage 04_qc).
%
%   Reads etc.step3_meta from each *_step3.set (partial load of the `etc` variable) and
%   writes one row per session: interpolated channels, data rank, ICLabel brain-component
%   counts at 0.70 / 0.80 / 0.90 from the step-2 classification that prep_step3 gated on,
%   rejected components, and the effect of the manual ocular-component list
%   (03_prep_eeg/manual_eye_ic_overrides.csv). Nothing is re-scored and no .set is written.
%
%   Usage:
%     make_prep_summary();
%     make_prep_summary('only', {'SUB_33_sess01'});
%
%   Outputs (out_dir):
%     prep_healthy_cohort_iclabel.csv    one row per session
%     prep_healthy_cohort_qc_summary.md  readable summary
%
%   eeg_weak_badch (n_bad_ch > 10) is a descriptive reporting tier.
%   eeg_qc_ok_legacy_iclabel (brain80 >= 2 and brain90 >= 1) is a descriptive ICLabel
%   cut reported for reference only; it is not an inclusion criterion.
%   Session inclusion rests on alignment and goniometer QC alone.

    p = inputParser;
    addParameter(p, 'eeg_dir',      '${GAIT_PROJECT_ROOT}/data/prep_data_prep/healthy/eeg', @ischar);
    addParameter(p, 'cohort_file',  '${GAIT_PROJECT_ROOT}/01_preprocessing/healthy/note/healthy_cohort.txt', @ischar);
    addParameter(p, 'override_csv', '${GAIT_PROJECT_ROOT}/01_preprocessing/healthy/03_prep_eeg/manual_eye_ic_overrides.csv', @ischar);
    addParameter(p, 'out_dir',      '${GAIT_PROJECT_ROOT}/01_preprocessing/healthy/result/qc', @ischar);
    addParameter(p, 'only',         {}, @(x) iscell(x) || ischar(x));
    parse(p, varargin{:});
    opt = p.Results;
    if ischar(opt.only), opt.only = {opt.only}; end

    if ~exist(opt.out_dir, 'dir'), mkdir(opt.out_dir); end

    labels = read_cohort(opt.cohort_file);
    if ~isempty(opt.only), labels = labels(ismember(labels, opt.only)); end
    n = numel(labels);
    fprintf('make_prep_summary: %d sessions from %s\n', n, opt.cohort_file);

    decl = read_override_csv(opt.override_csv);   % struct: .session .ics

    R = repmat(blank_row(), n, 1);
    for i = 1:n
        R(i) = scan_session(labels{i}, opt.eeg_dir, decl);
        fprintf('  [%2d/%2d] %-16s nrej=%s manual_forced=%s\n', i, n, ...
            R(i).label, R(i).n_rej_s, R(i).manual_forced_ics);
    end

    csv_path = fullfile(opt.out_dir, 'prep_healthy_cohort_iclabel.csv');
    write_csv(csv_path, R);
    fprintf('\nWrote %s\n', csv_path);

    md_path = fullfile(opt.out_dir, 'prep_healthy_cohort_qc_summary.md');
    write_md(md_path, R, decl, opt);
    fprintf('Wrote %s\n', md_path);
end


% ======================== per-session scan ========================

function r = scan_session(label, eeg_dir, decl)
    r = blank_row();
    r.label = label;
    tok = regexp(label, '^(SUB_\d+)_(sess\d+\w*)$', 'tokens', 'once');
    if ~isempty(tok), r.subject = tok{1}; r.session = tok{2}; end

    f = fullfile(eeg_dir, [label '_step3.set']);
    if ~exist(f, 'file')
        r.status = 'MISSING_STEP3';
        return;
    end

    S = load('-mat', f, 'etc');
    if ~isfield(S, 'etc') || ~isfield(S.etc, 'step3_meta')
        r.status = 'NO_STEP3_META';
        return;
    end
    m = S.etc.step3_meta;

    r.variant              = get_str(m, 'variant', 'NA');
    r.iclabel_score_source = get_str(m, 'iclabel_score_source', 'NA');
    r.artifact_thresh      = get_num(m, 'artifact_thresh', NaN);
    r.n_asr_segments       = get_num(m, 'n_asr_segments', NaN);

    if isfield(m, 'step2_meta')
        sm = m.step2_meta;
        r.n_bad_ch       = get_num(sm, 'n_ch_interpolated', NaN);
        r.data_rank      = get_num(sm, 'data_rank', NaN);
        if isfield(sm, 'bad_ch_labels') && ~isempty(sm.bad_ch_labels)
            r.bad_ch_labels = strjoin(cellstr(sm.bad_ch_labels(:)'), '|');
        end
        r.n_trial_segs   = get_num(sm, 'n_trial_segs', NaN);
    end

    % ---- ICLabel counts, taken from the step-2 classification that prep_step3 gated on
    C = m.step2_iclabel_classifications;
    bp = C(:, 1);
    r.n_ics   = size(C, 1);
    r.brain70 = sum(bp > 0.70);
    r.brain80 = sum(bp > 0.80);
    r.brain90 = sum(bp > 0.90);
    % brain80>=2 & brain90>=1. b70>=2 is implied because {b80} is a subset of {b70}.
    r.eeg_qc_ok_legacy_iclabel = double(r.brain80 >= 2 && r.brain90 >= 1);
    if isnan(r.n_bad_ch)
        r.eeg_weak_badch = NaN;   % unknown, not "not weak"
    else
        r.eeg_weak_badch = double(r.n_bad_ch > badch_weak_thresh());
    end

    % ---- what the 0.80 ICLabel artifact gate alone would reject
    thr = r.artifact_thresh; if isnan(thr), thr = 0.8; end
    artifact_max = max(C(:, [2 3 4 5 6]), [], 2);   % Muscle Eye Heart LineNoise ChanNoise
    ic_rej = find(artifact_max > thr)';
    r.n_iclabel_rej      = numel(ic_rej);
    r.iclabel_rejected_ics = join_ints(ic_rej);

    % ---- manual ocular override
    if isfield(m, 'manual_eye_ics')
        me = m.manual_eye_ics(:)';
        r.has_manual_field = 1;
    else
        me = [];
        r.has_manual_field = 0;
    end
    r.n_manual_eye   = numel(me);
    r.manual_eye_ics = join_ints(me);
    forced = setdiff(me, ic_rej);
    r.n_manual_forced      = numel(forced);
    r.manual_forced_ics    = join_ints(forced);
    r.manual_override_applied = double(~isempty(forced));

    di = strcmp({decl.session}, label);
    if any(di)
        dics = decl(find(di,1)).ics;
        r.override_declared_ics = join_ints(dics);
        % Provenance for the no-op/forced verdict: the max artifact-class
        % probability (Muscle Eye Heart LineNoise ChanNoise) ICLabel gave each
        % declared IC. > artifact_thresh => the gate already had it => no-op.
        parts = arrayfun(@(k) sprintf('%d:%.4f', k, artifact_max(k)), ...
                         dics(:)', 'Uni', 0);
        r.override_declared_max_artifact = strjoin(parts, ' ');
    end

    % ---- as-applied rejection set
    rej = m.rejected_ics(:)';
    r.n_rej         = numel(rej);
    r.rejected_ics  = join_ints(rej);
    if isfield(m, 'ic_labels') && ~isempty(m.ic_labels)
        r.rejected_labels = strjoin(cellstr(m.ic_labels(:)'), '|');
    end
    if isfield(m, 'n_ics_rejected')
        r.n_rej_meta = m.n_ics_rejected;
    end

    expect = union(ic_rej, me);
    r.consistent = double(isequal(sort(rej(:)'), sort(expect(:)')) && ...
                          r.n_rej == r.n_rej_meta);
    if r.consistent, r.status = 'ok'; else, r.status = 'INCONSISTENT'; end
    r.n_rej_s = sprintf('%d', r.n_rej);
end


% ======================== output ========================

function write_csv(path, R)
    cols = {'label','subject','session','status','variant','iclabel_score_source', ...
            'n_bad_ch','bad_ch_labels','data_rank','n_trial_segs','n_asr_segments', ...
            'n_ics','brain70','brain80','brain90', ...
            'eeg_weak_badch','eeg_qc_ok_legacy_iclabel', ...
            'artifact_thresh','n_iclabel_rej','iclabel_rejected_ics', ...
            'n_manual_eye','manual_eye_ics','n_manual_forced','manual_forced_ics', ...
            'manual_override_applied','has_manual_eye_field','override_declared_ics', ...
            'override_declared_max_artifact', ...
            'n_rej','rejected_ics','rejected_labels','consistent'};
    fid = fopen(path, 'w');
    fprintf(fid, '%s\n', strjoin(cols, ','));
    for i = 1:numel(R)
        r = R(i);
        fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
            r.label, r.subject, r.session, r.status, r.variant, r.iclabel_score_source, ...
            nstr(r.n_bad_ch), r.bad_ch_labels, nstr(r.data_rank), nstr(r.n_trial_segs), ...
            nstr(r.n_asr_segments), nstr(r.n_ics), nstr(r.brain70), nstr(r.brain80), ...
            nstr(r.brain90), nstr(r.eeg_weak_badch), nstr(r.eeg_qc_ok_legacy_iclabel), ...
            fstr(r.artifact_thresh), ...
            nstr(r.n_iclabel_rej), r.iclabel_rejected_ics, ...
            nstr(r.n_manual_eye), r.manual_eye_ics, nstr(r.n_manual_forced), ...
            r.manual_forced_ics, nstr(r.manual_override_applied), ...
            nstr(r.has_manual_field), r.override_declared_ics, ...
            r.override_declared_max_artifact, ...
            nstr(r.n_rej), r.rejected_ics, r.rejected_labels, nstr(r.consistent));
    end
    fclose(fid);
end


function write_md(path, R, decl, opt)
    fid = fopen(path, 'w');
    fprintf(fid, '# prep healthy cohort QC summary (auto-generated %s)\n\n', ...
        datestr(now, 'yyyy-mm-dd HH:MM'));
    fprintf(fid, 'Generated by `make_prep_summary.m`.\n');
    fprintf(fid, 'Source: `etc.step3_meta` of each `*_step3.set` in `%s`.\n', opt.eeg_dir);
    fprintf(fid, 'Cohort list: `%s` (%d sessions).\n', opt.cohort_file, numel(R));
    fprintf(fid, 'Machine-readable twin: `prep_healthy_cohort_iclabel.csv`.\n\n');
    fprintf(fid, 'ICLabel counts are read from the `step2_iclabel_classifications`\n');
    fprintf(fid, 'matrix stored in each step3 artifact -- the exact matrix `prep_step3.m`\n');
    fprintf(fid, 'gated on. Nothing is re-scored and nothing upstream is re-run.\n\n');

    % ---------- SECTION 1: the manual ocular override ----------
    fprintf(fid, '## 1. Manual ocular-IC override\n\n');
    fprintf(fid, 'Declared in `03_prep_eeg/manual_eye_ic_overrides.csv`: %d IC entries over %d sessions.\n\n', ...
        sum(arrayfun(@(d) numel(d.ics), decl)), numel(decl));
    fprintf(fid, '`manual_forced` = ICs rejected ONLY because of the override, i.e. the\n');
    fprintf(fid, 'override minus what the ICLabel artifact>%.2f gate already caught.\n\n', 0.8);
    fprintf(fid, '`declared IC max_artifact` is the max artifact-class probability\n');
    fprintf(fid, '(Muscle/Eye/Heart/LineNoise/ChanNoise) ICLabel assigned each declared IC.\n');
    fprintf(fid, 'Above the gate => the gate already had it => the override is a no-op.\n\n');
    fprintf(fid, '| Session | declared override ICs | declared IC max_artifact | ICLabel-gate ICs | as-applied rejected_ics | n_rej | manual_forced | effect |\n');
    fprintf(fid, '|---|---|---|---|---|---|---|---|\n');
    for i = 1:numel(decl)
        lbl = decl(i).session;
        k = find(strcmp({R.label}, lbl), 1);
        if isempty(k)
            fprintf(fid, '| %s | %s | | (not in healthy cohort) | | | | |\n', lbl, join_ints(decl(i).ics));
            continue;
        end
        r = R(k);
        if r.manual_override_applied
            eff = '';
        else
            eff = 'no-op (already over the gate, not below it)';
        end
        fprintf(fid, '| %s | %s | %s | %s | **%s** | **%s** | %s | %s |\n', ...
            r.label, r.override_declared_ics, r.override_declared_max_artifact, ...
            r.iclabel_rejected_ics, ...
            r.rejected_ics, nstr(r.n_rej), ...
            ternary(isempty(r.manual_forced_ics), '(none)', r.manual_forced_ics), eff);
    end
    nch = sum([R.manual_override_applied]);
    fprintf(fid, '\n**%d of %d healthy cohort sessions had their IC rejection set changed by the manual override.**\n\n', ...
        nch, numel(R));

    % ---------- SECTION 3: the EEG-QC-weak tier ----------
    thr_bc  = badch_weak_thresh();
    bc      = [R.n_bad_ch];
    weak    = [R.eeg_weak_badch] == 1;
    
    fprintf(fid, '## 3. EEG-QC-weak tier: `n_bad_ch > %d`\n\n', thr_bc);
    fprintf(fid, 'Weak = interpolated-channel count above %d. \n\n', thr_bc);
    fprintf(fid, '**Weak set: %d of %d.**\n\n', sum(weak), numel(R));
    fprintf(fid, '| Session | n_bad_ch | data_rank | b70 | b80 | b90 | bad channels |\n');
    fprintf(fid, '|---|---|---|---|---|---|---|\n');
    [~, ord] = sort(bc(weak), 'descend');
    wi = find(weak); wi = wi(ord);
    for i = wi
        r = R(i);
        % bad_ch_labels is '|'-joined in the CSV; '|' would break this md table.
        fprintf(fid, '| %s | **%s** | %s | %s | %s | %s | %s |\n', r.label, ...
            nstr(r.n_bad_ch), nstr(r.data_rank), nstr(r.brain70), ...
            nstr(r.brain80), nstr(r.brain90), strrep(r.bad_ch_labels, '|', ' '));
    end
    fprintf(fid, '\nHealthy cohort `n_bad_ch`: median %.1f, min %d, max %d.\n\n', ...
        median(bc, 'omitnan'), min(bc), max(bc));
    
    
        
    
    fprintf(fid, 'A descriptive ICLabel cut (`eeg_qc_ok_legacy_iclabel` = brain80>=2 AND\n');
    fprintf(fid, 'brain90>=1) is kept in the CSV for reference.\n');
    fprintf(fid, 'It is not an inclusion criterion.\n');
    
    
    
        
    
    
        
    
    
        
    fprintf(fid, '\n');
    
    
    
    
    
    
    
    
    
    
    fprintf(fid, '> \n');
    fprintf(fid, '> These are reporting tiers only. Cohort membership is decided by align +\n');
    fprintf(fid, '> goni QC alone (no EEG-IC gate): 80 included / 3 excluded / 42\n');
    fprintf(fid, '> participants.\n\n');

    % ---------- SECTION 4: per-session table ----------
    fprintf(fid, '## 4. Per-session QC\n\n');
    fprintf(fid, '`weak?` = reporting tier (`n_bad_ch > %d`). `legacy IC` = a descriptive\n', thr_bc);
    fprintf(fid, 'ICLabel cut (`brain80>=2 & brain90>=1`), reported for reference only.\n');
    fprintf(fid, 'Neither is an inclusion criterion.\n\n');
    fprintf(fid, '| Session | bad ch | data_rank | n_ICs | b70 | b80 | b90 | weak? | legacy IC | n_rej | rejected_ics | manual_forced |\n');
    fprintf(fid, '|---|---|---|---|---|---|---|---|---|---|---|---|\n');
    for i = 1:numel(R)
        r = R(i);
        fprintf(fid, '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n', ...
            r.label, nstr(r.n_bad_ch), nstr(r.data_rank), nstr(r.n_ics), ...
            nstr(r.brain70), nstr(r.brain80), nstr(r.brain90), ...
            ternary(r.eeg_weak_badch == 1, '**WEAK**', ''), ...
            ternary(r.eeg_qc_ok_legacy_iclabel == 1, 'ok', 'fail'), ...
            nstr(r.n_rej), r.rejected_ics, ...
            ternary(isempty(r.manual_forced_ics), '', ['**' r.manual_forced_ics '**']));
    end

    % ---------- SECTION 5: aggregates ----------
    fprintf(fid, '\n## 5. Aggregates\n\n');
    fprintf(fid, '- Sessions scanned: %d\n', numel(R));
    fprintf(fid, '- Participants: %d\n', numel(unique({R.subject})));
    fprintf(fid, '- status ok: %d / INCONSISTENT: %d / other: %d\n', ...
        sum(strcmp({R.status}, 'ok')), sum(strcmp({R.status}, 'INCONSISTENT')), ...
        sum(~ismember({R.status}, {'ok','INCONSISTENT'})));
    fprintf(fid, '- **EEG-QC-weak (n_bad_ch > %d): %d / %d%s**\n', ...
        thr_bc, sum(weak), numel(R), ...
        ternary(any(weak), [' -- ' strjoin({R(weak).label}, ', ')], ' -- none'));
    
    fprintf(fid, '- Total ICs rejected across cohort: %d\n', sum([R.n_rej], 'omitnan'));
    fprintf(fid, '- Sessions with manual ocular override applied: %d (%s)\n', nch, ...
        ternary(nch == 0, 'none', strjoin({R([R.manual_override_applied] == 1).label}, ', ')));
    fprintf(fid, '\n> prep cohort membership is decided by align + goni QC only (no EEG-IC gate);\n');
    fprintf(fid, '> a `WEAK` row here is a reporting tier, not an exclusion.\n');
    
    
    
    fclose(fid);
end


% ======================== helpers ========================

function r = blank_row()
    r = struct('label','', 'subject','', 'session','', 'status','', ...
        'variant','', 'iclabel_score_source','', ...
        'n_bad_ch',NaN, 'bad_ch_labels','', 'data_rank',NaN, 'n_trial_segs',NaN, ...
        'n_asr_segments',NaN, 'n_ics',NaN, 'brain70',NaN, 'brain80',NaN, ...
        'brain90',NaN, 'eeg_weak_badch',NaN, 'eeg_qc_ok_legacy_iclabel',NaN, ...
        'artifact_thresh',NaN, ...
        'n_iclabel_rej',NaN, 'iclabel_rejected_ics','', ...
        'n_manual_eye',NaN, 'manual_eye_ics','', 'n_manual_forced',NaN, ...
        'manual_forced_ics','', 'manual_override_applied',NaN, ...
        'has_manual_field',NaN, 'override_declared_ics','', ...
        'override_declared_max_artifact','', ...
        'n_rej',NaN, 'n_rej_meta',NaN, 'n_rej_s','', 'rejected_ics','', ...
        'rejected_labels','', 'consistent',NaN);
end

function t = badch_weak_thresh()
% BADCH_WEAK_THRESH  Current EEG-QC-weak cut: weak iff n_bad_ch > t.
    t = 10;
end

function labels = read_cohort(f)
    fid = fopen(f, 'r');
    c = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    labels = strtrim(c{1});
    labels = labels(~cellfun(@isempty, labels));
end

function d = read_override_csv(f)
    d = struct('session', {}, 'ics', {});
    if ~exist(f, 'file'), return; end
    fid = fopen(f, 'r');
    hdr = fgetl(fid); %#ok<NASGU>
    while true
        ln = fgetl(fid);
        if ~ischar(ln), break; end
        if isempty(strtrim(ln)), continue; end
        parts = strsplit(ln, ',');
        sess = strtrim(parts{1});
        ic   = str2double(parts{2});
        k = find(strcmp({d.session}, sess), 1);
        if isempty(k)
            d(end+1).session = sess; %#ok<AGROW>
            d(end).ics = ic;
        else
            d(k).ics = [d(k).ics ic];
        end
    end
    fclose(fid);
end

function s = join_ints(v)
    v = v(:)';
    if isempty(v), s = ''; return; end
    s = strjoin(arrayfun(@(x) sprintf('%d', x), v, 'Uni', 0), ' ');
end

function s = nstr(v)
    if isempty(v) || (isnumeric(v) && isnan(v)), s = ''; else, s = sprintf('%d', v); end
end

function s = fstr(v)
    if isempty(v) || isnan(v), s = ''; else, s = sprintf('%.2f', v); end
end

function v = get_str(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = char(s.(f)); else, v = d; end
end

function v = get_num(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = double(s.(f)); v = v(1); else, v = d; end
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end


    
