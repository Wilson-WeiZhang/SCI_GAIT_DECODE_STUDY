function out = prep_align(input_vhdr, out_dir, qc_dir, opts)
%PREP_ALIGN  Load one healthy EEG recording, align the goniometer to it, resample to 250 Hz.
%
%   Output per session:
%     {out_dir}/{label}_step1.set    EEG + EEG.etc.align_log + EEG.event.goni_idx
%                                    (goniometer epochs are cut later by extract_goni)
%     {qc_dir}/align_log/{label}.json
%     {qc_dir}/align_png/{label}.png  alignment figure used for visual review
%
%   Usage:
%     out = prep_align('/path/to/sub.vhdr', out_dir, qc_dir);
%     out = prep_align(..., struct('iei_thr_ms', 200));
%
%   opts fields:
%     label                   session label (default: from the vhdr file name)
%     srate_out               target sample rate (default 250)
%     iei_thr_ms              goniometer edges closer than this are one burst (default 200)
%     ioi_tol_ms              per-marker assignment tolerance (default 150)
%     ioi_match_tol_ms        IOI matching tolerance (default 300; 1200 for recordings
%                             from the second acquisition system, see below)
%     special_case            '' | 'multi_goni_file' (default '')
%     spurious_goni_override  1-based indices of goniometer edges to remove before matching
%     goni_path_override      explicit goniometer file for a recording part
%     goni_paths_multi        goniometer files for one EEG file ('multi_goni_file')
%
%   Output struct:
%     .set_path     path to _step1.set
%     .json_path    path to align_log json
%     .png_path     path to align png
%     .verdict      'PASS' | 'FAIL' | 'SPECIAL_CASE'
%     .align_log    alignment statistics

    if nargin < 4, opts = struct(); end
    opts = set_default(opts, 'srate_out',    250);
    opts = set_default(opts, 'iei_thr_ms',   200);
    opts = set_default(opts, 'ioi_tol_ms',   150);
    opts = set_default(opts, 'special_case', '');
    opts = set_default(opts, 'spurious_goni_override', []);
    opts = set_default(opts, 'goni_path_override',     '');  % explicit goni .txt path; when set, bypasses find_goni_file (for sub-session a/b/c)
    opts = set_default(opts, 'goni_paths_multi',       {});  % cell array of goni .txt paths for special_case='multi_goni_file' (1 EEG + N goni)

    input_vhdr = char(input_vhdr);
    out_dir    = char(out_dir);
    qc_dir     = char(qc_dir);
    [in_dir, in_name, ~] = fileparts(input_vhdr);
    in_dir  = char(in_dir);
    in_name = char(in_name);
    if isfield(opts, 'label') && ~isempty(opts.label)
        label = char(opts.label);
    else
        label = in_name;
    end

    % recordings from the second acquisition system have an unstable clock with frequent
    % sub-second steps; widen the IOI tolerances.
    if contains(in_name, 'gaiter')
        opts.ioi_match_tol_ms = 1200;
        opts.ioi_tol_ms       = max(opts.ioi_tol_ms, 1200);
    else
        opts = set_default(opts, 'ioi_match_tol_ms', 300);
    end

    fprintf('\n================================================================\n');
    fprintf('=== prep_align: %s ===\n', label);
    fprintf('================================================================\n');

    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    if ~exist(fullfile(qc_dir, 'align_log'), 'dir'), mkdir(fullfile(qc_dir, 'align_log')); end
    if ~exist(fullfile(qc_dir, 'align_png'), 'dir'), mkdir(fullfile(qc_dir, 'align_png')); end

    % ---- Step 1A: Load EEG (1000 Hz) ----
    fprintf('\n[1A] Load EEG from %s\n', input_vhdr);
    EEG = pop_loadbv(in_dir, [in_name '.vhdr']);
    EEG = eeg_checkset(EEG);
    assert(EEG.srate == 1000, 'Expected 1000 Hz EEG, got %d', EEG.srate);
    fprintf('     %d ch x %d pts @ %d Hz (%.1f s, %d events)\n', ...
        EEG.nbchan, EEG.pnts, EEG.srate, EEG.xmax, length(EEG.event));

    % Channel locations + 59-ch brain whitelist
    fprintf('[1A] Apply standard_1005 chanlocs + 59-ch whitelist\n');
    dipfit_dir = fileparts(which('pop_dipfit_settings'));
    eloc_file  = fullfile(dipfit_dir, 'standard_BEM', 'elec', 'standard_1005.elc');
    EEG = pop_chanedit(EEG, 'lookup', eloc_file);
    EEG = eeg_checkset(EEG);

    brain_whitelist = {'Fp1','Fz','F3','F7','FC5','FC1','C3','T7','CP5','CP1', ...
        'Pz','P3','P7','O1','Oz','O2','P4','P8','CP6','CP2', ...
        'Cz','C4','T8','FC6','FC2','F4','F8','Fp2', ...
        'AF7','AF3','AFz','F1','F5','FT7','FC3','C1','C5','TP7', ...
        'CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2', ...
        'CPz','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4','F2'};
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

    % ---- Step 1B: Load goni + align at 1000 Hz ----
    fprintf('\n[1B] Load goni + align\n');
    if strcmp(opts.special_case, 'multi_goni_file')
        % Multi-goni: 1 EEG + N goni files (e.g. wireless dropout)
        if isempty(opts.goni_paths_multi)
            error('prep_align: special_case=multi_goni_file requires opts.goni_paths_multi (cell array)');
        end
        goni_files = cellfun(@char, opts.goni_paths_multi, 'UniformOutput', false);
        nF = length(goni_files);
        fprintf('     multi_goni: %d goni files\n', nF);
        goni_stim_cells = cell(1, nF);
        for f = 1:nF
            fprintf('       [%d/%d] %s\n', f, nF, goni_files{f});
            g_f = load_goniometer(goni_files{f});
            assert(g_f.srate == 1000, 'Expected 1000 Hz goni, got %d', g_f.srate);
            goni_stim_cells{f} = g_f.stim(:)';
        end
        % Keep the first file as the primary path; extract_goni reads the
        % full EEG.etc.raw_goni_files set for multi-goni sessions.
        goni_file = goni_files{1};
        goni_stim_for_align = goni_stim_cells;
    else
        if ~isempty(opts.goni_path_override)
            goni_file = char(opts.goni_path_override);
        else
            goni_file = find_goni_file(in_dir, in_name);
        end
        fprintf('     goni file: %s\n', goni_file);
        goni = load_goniometer(goni_file);
        assert(goni.srate == 1000, 'Expected 1000 Hz goni, got %d', goni.srate);
        fprintf('     goni: %d ch x %d pts, stim col=%d\n', ...
            size(goni.data, 2), size(goni.data, 1), goni.stim_col);
        goni_stim_for_align = goni.stim(:)';
    end

    align_opts = struct( ...
        'iei_thr_ms',           opts.iei_thr_ms, ...
        'ioi_tol_ms',           opts.ioi_tol_ms, ...
        'ioi_match_tol_ms',     opts.ioi_match_tol_ms, ...
        'special_case',         opts.special_case, ...
        'label',                label, ...
        'spurious_goni_override', opts.spurious_goni_override);
    [EEG, align_log] = align_goni_to_eeg(EEG, goni_stim_for_align, 1000, align_opts);
    fprintf('     algorithm=%s\n', align_log.alignment_algorithm);
    fprintf('     verdict=%s  matched=%d/%d (TTL) / %d total events\n', ...
        align_log.auto_verdict, align_log.n_matched, align_log.n_eeg_events_ttl, ...
        align_log.n_eeg_events);
    fprintf('     residuals ms: median=%.2f  p95=%.2f  max=%.2f\n', ...
        align_log.residual_ms.median, align_log.residual_ms.p95, align_log.residual_ms.max);

    % ---- Step 1C: Resample EEG to target srate ----
    % Goni extraction happens separately via extract_goni.m, which writes
    % {label}_goni.mat in data/prep_data/healthy/goni/ using the alignment
    % metadata stored in EEG.etc.align_log.
    fprintf('\n[1C] Resample EEG -> %d Hz\n', opts.srate_out);
    EEG = pop_resample(EEG, opts.srate_out);
    EEG = eeg_checkset(EEG);
    fprintf('     EEG: %d ch x %d pts @ %d Hz\n', EEG.nbchan, EEG.pnts, EEG.srate);

    % ---- Save outputs ----
    EEG.etc.align_log        = align_log;
    EEG.etc.raw_goni_file    = goni_file;       % path-only metadata for downstream extraction
    if strcmp(opts.special_case, 'multi_goni_file')
        EEG.etc.raw_goni_files = goni_files;
    end
    EEG.etc.pipeline_version = '';
    set_path = fullfile(out_dir, [label '_step1.set']);
    pop_saveset(EEG, 'filename', [label '_step1.set'], 'filepath', out_dir, 'savemode', 'onefile');

    json_path = fullfile(qc_dir, 'align_log', [label '.json']);
    write_align_log_json(align_log, json_path);

    png_path = fullfile(qc_dir, 'align_png', [label '.png']);
    try
        plot_session_align(EEG, png_path, label);
    catch ME
        warning('prep_align:plotSessionAlignFailed', ...
            'plot_session_align failed: %s', ME.message);
    end


    fprintf('\n[done] %s | verdict=%s | %s\n', label, align_log.auto_verdict, set_path);

    out = struct( ...
        'set_path',  set_path, ...
        'json_path', json_path, ...
        'png_path',  png_path, ...
        'verdict',   align_log.auto_verdict, ...
        'align_log', align_log);
end


% ======================== helpers ========================

function s = set_default(s, f, default)
    if ~isfield(s, f), s.(f) = default; end
end
