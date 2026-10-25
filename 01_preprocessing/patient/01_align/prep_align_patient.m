function out = prep_align_patient(reg_entry, out_dir, qc_dir, opts)
%PREP_ALIGN_PATIENT  Load (and merge) one patient EEG recording, align the goniometer, resample.
%
%   Same alignment as 01_preprocessing/shared/shared/align_goni_to_eeg.m, with:
%     * several vhdr files of one session are merged with pop_mergeset
%     * only S10 markers reach the goniometer trigger channel, so only S10 is matched
%     * sessions on the two-marker protocol use prep_align_patient_legacy;
%       'multi_block' sessions are listed and skipped
%
%   Input:
%     reg_entry  one struct from the patient session registry
%     out_dir    where to write {label}_step1.set
%     qc_dir     where to write align_log/ and align_png/
%     opts       optional overrides (srate_out, iei_thr_ms, ioi_tol_ms)
%
%   Output struct:
%     .set_path     path to _step1.set (or '' if skipped)
%     .json_path    path to align_log json (or '' if skipped)
%     .verdict      'PASS' | 'FAIL' | 'SPECIAL_CASE' | 'SKIP_DEFERRED'
%     .align_log    alignment statistics (empty when skipped)

    if nargin < 4, opts = struct(); end
    opts = set_default(opts, 'srate_out',  250);
    opts = set_default(opts, 'iei_thr_ms', 200);
    opts = set_default(opts, 'ioi_tol_ms', 150);

    prep_patient_paths();   % add the shared and patient preprocessing folders

    label = reg_entry.label;
    fprintf('\n================================================================\n');
    fprintf('=== prep_align_patient: %s (%s/%s)\n', label, ...
        reg_entry.marker_protocol, reg_entry.special_case);
    fprintf('================================================================\n');

    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    if ~exist(fullfile(qc_dir, 'align_log'), 'dir'), mkdir(fullfile(qc_dir, 'align_log')); end
    if ~exist(fullfile(qc_dir, 'align_png'), 'dir'), mkdir(fullfile(qc_dir, 'align_png')); end

    out = struct('set_path', '', 'json_path', '', 'png_path', '', ...
                 'verdict', '', 'align_log', struct());

    % --- Deferred branches: skip cleanly with clear status ---
    % multi_block: relies on match_via_ioi_segmented in align_goni_to_eeg, which already
    % re-anchors over several rounds. The standard path is tried; if the match fraction
    % is too low, the downstream QC flags it.
    if strcmp(reg_entry.marker_protocol, 'legacy')
        % Two-marker protocol (the first two patient recordings): a per-pair two-anchor
        % linear alignment on the goniometer trigger start and end edges, delegated to
        % prep_align_patient_legacy.
        out = prep_align_patient_legacy(reg_entry, out_dir, qc_dir, opts);
        return;
    end

    % --- Step 1A: load EEG (merge if multi-vhdr) ---
    vhdrs = reg_entry.vhdr_paths;
    if isempty(vhdrs)
        error('prep_align_patient:noVhdr', '%s: registry has no vhdr_paths', label);
    end
    % length(vhdrs) > 1 case: pop_mergeset (boundary events between files).
    % Continuous master goni in dir 1 is assumed to cover all merged EEG segments.
    fprintf('\n[1A] Load EEG (%d file%s)\n', length(vhdrs), plural(length(vhdrs)));
    EEG = load_and_merge_vhdrs(vhdrs);
    assert(EEG.srate == 1000, 'Expected 1000 Hz EEG, got %d', EEG.srate);
    fprintf('     %d ch x %d pts @ %d Hz (%.1f s, %d events)\n', ...
        EEG.nbchan, EEG.pnts, EEG.srate, EEG.xmax, length(EEG.event));

    % --- Step 1A2: chanlocs + 59-ch brain whitelist ---
    fprintf('[1A] Apply standard_1005 chanlocs + 59-ch whitelist\n');
    dipfit_dir = fileparts(which('pop_dipfit_settings'));
    eloc_file  = fullfile(dipfit_dir, 'standard_BEM', 'elec', 'standard_1005.elc');
    EEG = pop_chanedit(EEG, 'lookup', eloc_file);
    EEG = eeg_checkset(EEG);

    % whitelist excludes FCz (recorded data does not contain it). Restored as
    % zero-fill in step2/3.
    brain_whitelist = prep_patient_brain_whitelist();
    all_labels = {EEG.chanlocs.labels};
    brain_keep = find(ismember(all_labels, brain_whitelist));
    non_brain  = setdiff(1:EEG.nbchan, brain_keep);
    if ~isempty(non_brain)
        fprintf('     Removing %d non-brain: %s\n', length(non_brain), ...
            strjoin(all_labels(non_brain), ', '));
        EEG = pop_select(EEG, 'channel', brain_keep);
        EEG = eeg_checkset(EEG);
    end
    fprintf('     Brain channels: %d\n', EEG.nbchan);

    % --- Step 1B: load goni + S10 IOI align ---
    fprintf('\n[1B] Load goni + S10 IOI align\n');
    if strcmp(reg_entry.special_case, 'multi_goni_file')
        gonis = reg_entry.goni_paths;
        nF = length(gonis);
        fprintf('     multi_goni: %d files\n', nF);
        goni_stim_cells = cell(1, nF);
        for f = 1:nF
            fprintf('       [%d/%d] %s\n', f, nF, gonis{f});
            g_f = load_goniometer(gonis{f});
            assert(g_f.srate == 1000, 'Expected 1000 Hz goni, got %d', g_f.srate);
            goni_stim_cells{f} = g_f.stim(:)';
        end
        align_special = 'multi_goni_file';
        goni_stim_for_align = goni_stim_cells;
        goni_files_record = gonis;
    else
        if isempty(reg_entry.goni_paths)
            error('prep_align_patient:noGoni', '%s: no goni file in registry', label);
        end
        goni_file = reg_entry.goni_paths{1};
        fprintf('     goni: %s\n', goni_file);
        goni = load_goniometer(goni_file);
        assert(goni.srate == 1000, 'Expected 1000 Hz goni, got %d', goni.srate);
        fprintf('     goni: %d ch x %d pts, stim col=%d\n', ...
            size(goni.data, 2), size(goni.data, 1), goni.stim_col);
        align_special = '';
        goni_stim_for_align = goni.stim(:)';
        goni_files_record = {goni_file};
    end

    align_opts = struct( ...
        'iei_thr_ms',             opts.iei_thr_ms, ...
        'ioi_tol_ms',             opts.ioi_tol_ms, ...
        'special_case',           align_special, ...
        'spurious_goni_override', reg_entry.spurious_goni_override, ...
        'ttl_type_re',            '^S\s*10$');   % patient: only S10 reaches goni Stim

    [EEG, align_log] = align_goni_to_eeg(EEG, goni_stim_for_align, 1000, align_opts);
    fprintf('     algorithm=%s\n', align_log.alignment_algorithm);
    fprintf('     verdict=%s  matched=%d/%d S10\n', ...
        align_log.auto_verdict, align_log.n_matched, align_log.n_eeg_events_ttl);
    if isfield(align_log.residual_ms, 'median')
        fprintf('     residual ms: median=%.2f  p95=%.2f  max=%.2f\n', ...
            align_log.residual_ms.median, align_log.residual_ms.p95, align_log.residual_ms.max);
    end

    % --- Step 1C: resample EEG to target srate ---
    fprintf('\n[1C] Resample EEG -> %d Hz\n', opts.srate_out);
    EEG = pop_resample(EEG, opts.srate_out);
    EEG = eeg_checkset(EEG);
    fprintf('     EEG: %d ch x %d pts @ %d Hz\n', EEG.nbchan, EEG.pnts, EEG.srate);

    % --- Save outputs ---
    EEG.etc.align_log        = align_log;
    EEG.etc.raw_goni_file    = goni_files_record{1};
    if length(goni_files_record) > 1
        EEG.etc.raw_goni_files = goni_files_record;
    end
    EEG.etc.cohort           = 'patient';
    EEG.etc.marker_protocol  = reg_entry.marker_protocol;
    EEG.etc.special_case     = reg_entry.special_case;
    EEG.etc.pipeline_version = '-patient';

    out.set_path = fullfile(out_dir, [label '_step1.set']);
    pop_saveset(EEG, 'filename', [label '_step1.set'], 'filepath', out_dir, ...
        'savemode', 'onefile');

    out.json_path = fullfile(qc_dir, 'align_log', [label '.json']);
    write_align_log_json(align_log, out.json_path);

    out.png_path = fullfile(qc_dir, 'align_png', [label '.png']);
    try
        plot_session_align(EEG, out.png_path, label);
    catch ME
        warning('prep_align_patient:plotFailed', 'plot_session_align: %s', ME.message);
    end

    out.verdict   = align_log.auto_verdict;
    out.align_log = align_log;
    fprintf('\n[done] %s | verdict=%s | %s\n', label, out.verdict, out.set_path);
end


% =====================================================================
function s = set_default(s, f, default)
    if ~isfield(s, f), s.(f) = default; end
end

function p = plural(n)
    if n == 1, p = ''; else, p = 's'; end
end

function EEG = load_and_merge_vhdrs(vhdrs)
    n = length(vhdrs);
    EEGs = cell(1, n);
    for k = 1:n
        [d, nm, ~] = fileparts(vhdrs{k});
        fprintf('       (%d/%d) %s\n', k, n, vhdrs{k});
        EEGs{k} = pop_loadbv(d, [nm '.vhdr']);
        EEGs{k} = eeg_checkset(EEGs{k});
    end
    if n == 1
        EEG = EEGs{1};
    else
        EEG = EEGs{1};
        for k = 2:n
            EEG = pop_mergeset(EEG, EEGs{k}, 0);   % pairwise merge; insert boundary events
        end
        for k = 1:n, EEGs{k} = []; end
        EEG = eeg_checkset(EEG);
        fprintf('     pop_mergeset: %d files -> %d pts (%d events incl. boundaries)\n', ...
            n, EEG.pnts, length(EEG.event));
    end
end
