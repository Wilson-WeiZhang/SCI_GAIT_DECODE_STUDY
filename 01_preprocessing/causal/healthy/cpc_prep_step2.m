%CPC_PREP_STEP2  Step 2 of the healthy preprocessing with a reproducible ICA.
%
%   A copy of 01_preprocessing/healthy/03_prep_eeg/prep_step2.m with two changes only: the function name,
%   and 'rndreset', 'off' in the pop_runica call, so runica keeps the random state set by the caller
%   (cpc_step2_cal.m sets rng('default')) instead of reseeding from the clock. Every other line, options
%   included, is prep_step2.
function out = cpc_prep_step2(step1_set_path, out_dir, opts)
%PREP_STEP2  Phase B step 2: bad-ch + ASR + AMICA on task/rest segments.
%
%   Input  : {label}_step1.set  (from stage 01_align; 250 Hz, 59ch healthy)
%   Output : {label}_step2.set  (segments + AMICA weights). EEG only — goni is
%            in {label}_goni.mat from stage 02_extract_goni and is not touched here.
%
%   Pipeline:
%     1. BP 1-40 Hz (ICA copy)
%     2. Trim S11-S12 (drop pre/post-task data)
%     3. Extract task+rest segments (no baseline)
%     4. clean_rawdata bad-ch detection -> interpolate
%     5. Restore FCz zero-fill (healthy only) -> CAR
%     6. ASR (k=20) on Walk segments only
%     7. ICA: runica (extended Infomax, default) or AMICA via opts.ica_method.
%
%   Cohort-aware:
%     - healthy/patient CACS-64 data: 59 brain ch -> +FCz -> 60ch
%       (FCz restored as the online reference; cohort is still recorded in metadata.)
%
%   Skip rules:
%     - Data duration < 5 min -> skip AMICA, only bad-ch + CAR; mark
%       EEG.etc.skipped_amica = true.
%
%   Usage:
%     out = prep_step2('.../SUB_01_sess01_step1.set', out_dir);
%     out = prep_step2(..., struct('amica_iter', 1000, 'asr_k', 20));
%
%   Opts:
%     ica_method      (default 'runica')  -- 'runica' | 'amica'
%     amica_num_mod   (default 2)         -- only used if ica_method='amica'
%     amica_iter      (default 1000)      -- only used if ica_method='amica'
%     asr_k           (default 20)
%     bp_ica_lo/hi    (default 1 / 40 Hz)
%     min_amica_min   (default 5)         -- ICA lower bound (min duration in min)
%     overwrite       (default false)
%
%   Output struct:
%     .set_path    path to _step2.set
%     .n_bad_ch    number of interpolated bad channels
%     .n_asr       number of ASR-cleaned Walk segments
%     .n_ic        number of ICA components
%     .skipped_amica  bool

    if nargin < 3, opts = struct(); end
    opts = set_default(opts, 'ica_method',    'runica');  % 'runica' | 'amica'
    opts = set_default(opts, 'amica_num_mod', 2);
    opts = set_default(opts, 'amica_iter',    1000);
    opts = set_default(opts, 'asr_k',         20);
    opts = set_default(opts, 'bp_ica_lo',     1);
    opts = set_default(opts, 'bp_ica_hi',     40);
    opts = set_default(opts, 'min_amica_min', 5);
    opts = set_default(opts, 'overwrite',     false);
    opts = set_default(opts, 'ica_exclude_walk', false);  % true: ICA trained without the real-walking segments

    step1_set_path = char(step1_set_path);
    out_dir        = char(out_dir);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    [~, step1_name, ~] = fileparts(step1_set_path);
    label = regexprep(step1_name, '_step1$', '');

    file_step2 = fullfile(out_dir, [label '_step2.set']);

    fprintf('\n================================================================\n');
    fprintf('=== prep_step2: %s ===\n', label);
    fprintf('================================================================\n');

    if ~opts.overwrite && exist(file_step2, 'file')
        fprintf('  already exists: %s\n', file_step2);
        EEG_seg = pop_loadset('filename', [label '_step2.set'], 'filepath', out_dir);
        out = step2_out(file_step2, EEG_seg);
        return;
    end

    % ---- Load step1 ----
    [in_dir, in_name, ~] = fileparts(step1_set_path);
    EEG = pop_loadset('filename', [in_name '.set'], 'filepath', in_dir);
    EEG = eeg_checkset(EEG);
    fprintf('  Loaded step1: %d ch x %d pts @ %d Hz (%.1f s)\n', ...
        EEG.nbchan, EEG.pnts, EEG.srate, EEG.xmax);

    cohort = get_cohort(EEG);
    fprintf('  Cohort: %s\n', cohort);

    [brain_whitelist, restore_fcz] = get_cohort_whitelist(cohort, EEG);
    brain_chanlocs = EEG.chanlocs;
    brain_labels   = {EEG.chanlocs.labels};

    % Trial definitions. SUB_01_sess01 uses the legacy marker scheme where
    % rest is encoded as R1+S4 -> R1+S5 instead of S7 -> S8.
    trial_defs = get_trial_defs(label);
    if opts.ica_exclude_walk
        % ICA variant: exclude Walk (real-walking) segments from the ICA training
        % set so the decomposition is not shaped by gait motion. bad-ch detection,
        % interpolation and CAR are unchanged; ASR becomes a no-op because it only targets Walk trials.
        trial_defs = trial_defs(~strcmp(trial_defs(:, 3), 'Walk'), :);
        fprintf('  [ica_exclude_walk] Walk excluded from ICA training -> %d trial types\n', size(trial_defs, 1));
    end

    % ---- 1. BP 1-40 Hz (ICA copy) ----
    fprintf('\n[1] BP %g-%g Hz (ICA copy)\n', opts.bp_ica_lo, opts.bp_ica_hi);
    EEG = pop_eegfiltnew(EEG, 'locutoff', opts.bp_ica_lo, 'hicutoff', opts.bp_ica_hi);
    EEG = eeg_checkset(EEG);

    % ---- 2. Trim S11-S12 ----
    EEG = trim_s11_s12(EEG);

    % ---- 3. Extract task+rest segments (no baseline) ----
    fprintf('\n[3] Extract task+rest segments (no BL)\n');
    [EEG_seg, trial_type_id, n_seg, total_dur] = extract_segments_no_bl(EEG, trial_defs, EEG.srate);
    clear EEG;

    % ---- 4. Bad-channel detection on segments ----
    fprintf('\n[4] Bad-channel detection (clean_rawdata corr=0.8)\n');
    EEG_probe = clean_rawdata(EEG_seg, 5, 'off', 0.8, 4, 'off', 'off');
    clean_labels = {EEG_probe.chanlocs.labels};
    bad_mask     = ~ismember(brain_labels, clean_labels);
    bad_labels   = brain_labels(bad_mask);
    n_bad        = sum(bad_mask);
    clear EEG_probe;
    if n_bad > 15
        warning('prep_step2:manyBadCh', '%s: many bad ch (%d): %s', ...
            label, n_bad, strjoin(bad_labels, ', '));
    elseif n_bad > 0
        fprintf('  Bad channels (%d): %s\n', n_bad, strjoin(bad_labels, ', '));
    else
        fprintf('  No bad channels detected.\n');
    end

    if n_bad > 0
        keep_idx = find(~bad_mask);
        EEG_seg = pop_select(EEG_seg, 'channel', keep_idx);
        EEG_seg = eeg_checkset(EEG_seg);
        interp_mask = ismember(brain_labels, bad_labels);
        target_locs = [EEG_seg.chanlocs, brain_chanlocs(interp_mask)];
        EEG_seg = pop_interp(EEG_seg, target_locs, 'spherical');
        EEG_seg = eeg_checkset(EEG_seg);
        fprintf('  Interpolated %d bad ch\n', n_bad);
    end
    EEG_seg = reorder_channels(EEG_seg, brain_whitelist);

    % ---- 5. Restore FCz (healthy) -> CAR ----
    if restore_fcz
        fprintf('\n[5] Restore FCz zero-fill + CAR (60ch)\n');
        EEG_seg = add_fcz_zerofill(EEG_seg);
        EEG_seg = pop_reref(EEG_seg, []);
        EEG_seg = reorder_channels(EEG_seg, [brain_whitelist, {'FCz'}]);
    else
        fprintf('\n[5] CAR only (restore_fcz=false)\n');
        EEG_seg = pop_reref(EEG_seg, []);
        EEG_seg = reorder_channels(EEG_seg, brain_whitelist);
    end
    EEG_seg = eeg_checkset(EEG_seg);

    n_brain  = EEG_seg.nbchan;
    % Data rank:
    %   healthy: 60 - n_interp (FCz zero-fill absorbs avgref -1)
    %   patient: n_brain - n_interp - 1 (CAR drops one rank)
    if restore_fcz
        formula_rank = n_brain - n_bad;
    else
        formula_rank = n_brain - n_bad - 1;
    end
    numerical_rank = rank(double(EEG_seg.data'));
    data_rank = min(formula_rank, numerical_rank);
    fprintf('  Data rank: %d (formula=%d, numerical=%d, nch=%d, interp=%d)\n', ...
        data_rank, formula_rank, numerical_rank, n_brain, n_bad);

    % ---- 6. ASR on Walk segments only ----
    fprintf('\n[6] ASR (k=%g) on Walk segments\n', opts.asr_k);
    [EEG_seg, n_asr, asr_log] = apply_asr(EEG_seg, trial_defs, EEG_seg.srate, opts.asr_k);

    % ---- 7. AMICA (skip if duration < min_amica_min) ----
    dur_min = total_dur / 60;
    skipped_amica = false;
    if dur_min < opts.min_amica_min
        warning('prep_step2:skipAMICA', ...
            '%s: duration %.1f min < %.0f min threshold -> skipping AMICA.', ...
            label, dur_min, opts.min_amica_min);
        skipped_amica = true;
    else
        switch lower(opts.ica_method)
        case 'amica'
            fprintf('\n[7] AMICA: %d ch x %d pts, %d model, %d iter, rank=%d\n', ...
                EEG_seg.nbchan, EEG_seg.pnts, opts.amica_num_mod, opts.amica_iter, data_rank);
            amica_outdir = fullfile(out_dir, [label '_amicatmp']);
            if ~exist(amica_outdir, 'dir'), mkdir(amica_outdir); end
            EEG_seg = pop_runamica(EEG_seg, ...
                'num_mod', opts.amica_num_mod, ...
                'maxiter', opts.amica_iter, ...
                'pcakeep', data_rank, ...
                'outdir',  amica_outdir);
            EEG_seg = eeg_checkset(EEG_seg);
            fprintf('  AMICA done: %d components\n', size(EEG_seg.icaweights, 1));
            if exist(amica_outdir, 'dir'), rmdir(amica_outdir, 's'); end
        case 'runica'
            fprintf('\n[7] runica (extended infomax): %d ch x %d pts, pca=%d\n', ...
                EEG_seg.nbchan, EEG_seg.pnts, data_rank);
            EEG_seg = pop_runica(EEG_seg, 'icatype', 'runica', ...
                'extended', 1, 'pca', data_rank, 'rndreset', 'off');
            EEG_seg = eeg_checkset(EEG_seg);
            fprintf('  runica done: %d components\n', size(EEG_seg.icaweights, 1));
        otherwise
            error('prep_step2:badIcaMethod', ...
                'opts.ica_method must be ''amica'' or ''runica'', got ''%s''', opts.ica_method);
        end
    end

    % ---- Metadata ----
    EEG_seg.etc.step2_meta = struct( ...
        'bad_ch_labels',     {bad_labels}, ...
        'brain_chanlocs',    brain_chanlocs, ...
        'brain_whitelist',   {brain_whitelist}, ...
        'restore_fcz',       restore_fcz, ...
        'n_trial_segs',      n_seg, ...
        'trial_type_id',     trial_type_id, ...
        'trial_defs',        {trial_defs}, ...
        'trial_dur_sec',     total_dur, ...
        'n_ch_interpolated', n_bad, ...
        'data_rank',         data_rank, ...
        'n_asr_segments',    n_asr, ...
        'asr_log',           asr_log, ...
        'ica_method',        opts.ica_method, ...
        'amica_num_mod',     opts.amica_num_mod, ...
        'amica_max_iter',    opts.amica_iter, ...
        'bp_ica_lo',         opts.bp_ica_lo, ...
        'bp_ica_hi',         opts.bp_ica_hi, ...
        'ica_exclude_walk',  opts.ica_exclude_walk, ...
        'cohort',            cohort);
    EEG_seg.etc.skipped_amica = skipped_amica;

    % stage 02_extract_goni produces standalone {label}_goni.mat.
    pop_saveset(EEG_seg, 'filename', [label '_step2.set'], ...
        'filepath', out_dir, 'savemode', 'onefile');
    fprintf('\n[done] %s step2 -> %s\n', label, file_step2);

    out = step2_out(file_step2, EEG_seg);
end


% ======================== helpers ========================

function out = step2_out(fp, EEG_seg)
    n_ic = 0;
    if isfield(EEG_seg, 'icaweights'), n_ic = size(EEG_seg.icaweights, 1); end
    meta = EEG_seg.etc.step2_meta;
    out = struct( ...
        'set_path',      fp, ...
        'n_bad_ch',      meta.n_ch_interpolated, ...
        'n_asr',         meta.n_asr_segments, ...
        'n_ic',          n_ic, ...
        'skipped_amica', isfield(EEG_seg.etc, 'skipped_amica') && EEG_seg.etc.skipped_amica);
end

function s = set_default(s, f, default)
    if ~isfield(s, f), s.(f) = default; end
end

function cohort = get_cohort(EEG)
    if isfield(EEG, 'etc') && isfield(EEG.etc, 'cohort') && ~isempty(EEG.etc.cohort)
        cohort = char(EEG.etc.cohort);
    else
        cohort = 'healthy';
    end
end

function [wl, restore_fcz] = get_cohort_whitelist(cohort, EEG) %#ok<INUSD>
    % Healthy and patient share the SAME CACS-64 cap with FCz as online reference
    wl = {'Fp1','Fz','F3','F7','FC5','FC1','C3','T7','CP5','CP1', ...
          'Pz','P3','P7','O1','Oz','O2','P4','P8','CP6','CP2', ...
          'Cz','C4','T8','FC6','FC2','F4','F8','Fp2', ...
          'AF7','AF3','AFz','F1','F5','FT7','FC3','C1','C5','TP7', ...
          'CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2', ...
          'CPz','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4','F2'};
    restore_fcz = true;
    if ~strcmp(cohort, 'healthy') && ~strcmp(cohort, 'patient')
        error('prep_step2:unknownCohort', 'Unknown cohort: %s', cohort);
    end
end

function trial_defs = get_trial_defs(label)
    if strcmp(label, 'SUB_01_sess01')
        trial_defs = {
            'S  1', 'S  2', 'MI',   false, ''
            'S  4', 'S  5', 'Walk', true,  'without_r1'
            'S  4', 'S  5', 'Rest', false, 'with_r1'
        };
    else
        trial_defs = {
            'S  1', 'S  2', 'MI',   false, ''
            'S  4', 'S  5', 'Walk', true,  ''
            'S  7', 'S  8', 'Rest', false, ''
        };
    end
end

function ok = passes_r1_mode(mode, evt_types, evt_lats, sl, nxt, srate)
    if isempty(mode)
        ok = true;
        return;
    end
    has_r1 = has_coincident_r1(evt_types, evt_lats, sl, nxt, srate);
    switch mode
        case 'with_r1'
            ok = has_r1;
        case 'without_r1'
            ok = ~has_r1;
        otherwise
            error('prep_step2:badR1Mode', 'Unknown R1 mode: %s', mode);
    end
end

function tf = has_coincident_r1(evt_types, evt_lats, sl, nxt, srate)
    is_r1 = cellfun(@(t) ~isempty(regexp(strtrim(t), '^R\s*1$', 'once')), evt_types);
    r1_lats = evt_lats(is_r1);
    tol = max(2, round(0.02 * srate));
    tf = any(abs(r1_lats - sl) <= tol) || any(abs(r1_lats - nxt) <= tol);
end

function EEG = trim_s11_s12(EEG)
    if isempty(EEG.event), return; end
    evt_types = {EEG.event.type};
    evt_lats  = [EEG.event.latency];
    s11 = find(strcmp(evt_types, 'S 11'), 1, 'first');
    s12 = find(strcmp(evt_types, 'S 12'), 1, 'last');
    if ~isempty(s11) && ~isempty(s12)
        i1 = max(1, round(evt_lats(s11)));
        i2 = min(EEG.pnts, round(evt_lats(s12)));
        fprintf('  Trim S11-S12: %.1f-%.1f s (%.1f s)\n', ...
            (i1-1)/EEG.srate, (i2-1)/EEG.srate, (i2-i1)/EEG.srate);
        kept = i1:i2;
        EEG = pop_select(EEG, 'point', [i1 i2]);
        EEG = eeg_checkset(EEG);
        EEG = sync_goni_if_any(EEG, kept);
    elseif ~isempty(s11)
        i1 = max(1, round(evt_lats(s11)));
        kept = i1:EEG.pnts;
        EEG = pop_select(EEG, 'point', [i1 EEG.pnts]);
        EEG = eeg_checkset(EEG);
        EEG = sync_goni_if_any(EEG, kept);
    else
        fprintf('  No S11/S12, keeping full recording.\n');
    end
end

function EEG = sync_goni_if_any(EEG, kept_samples) %#ok<INUSD>
    % EEG sample rejection. Nothing to sync.
end

function [EEG_seg, trial_type_id, n_seg, total_dur] = extract_segments_no_bl(EEG, trial_defs, srate)
    n_trial_types = size(trial_defs, 1);
    trial_ranges  = zeros(0, 2);
    trial_type_id = [];
    n_seg = 0; total_dur = 0;

    evt_types = {EEG.event.type};
    evt_lats  = [EEG.event.latency];
    boundary_lats = evt_lats(strcmp(evt_types, 'boundary'));

    for td = 1:n_trial_types
        smk = trial_defs{td, 1};
        emk = trial_defs{td, 2};
        r1_mode = '';
        if size(trial_defs, 2) >= 5, r1_mode = trial_defs{td, 5}; end
        s_lats = evt_lats(strcmp(evt_types, smk));
        if ischar(emk)
            e_lats = evt_lats(strcmp(evt_types, emk));
        else
            e_lats = s_lats + emk * srate;
        end

        n_this = 0;
        for i = 1:length(s_lats)
            sl = s_lats(i);
            if ischar(emk)
                nxt = e_lats(e_lats > sl);
                if isempty(nxt), continue; end
                nxt = nxt(1);
                if any(s_lats > sl & s_lats < nxt), continue; end
            else
                nxt = e_lats(i);
                if round(nxt) > EEG.pnts, continue; end
            end
            dur = (nxt - sl) / srate;
            if dur < 3 || dur > 300, continue; end
            if any(boundary_lats > sl & boundary_lats < nxt), continue; end
            if ~passes_r1_mode(r1_mode, evt_types, evt_lats, sl, nxt, srate), continue; end

            i1 = max(1, round(sl));
            i2 = min(EEG.pnts, round(nxt));
            t_start = (i1 - 1) / srate;
            t_end   = (i2 - 1) / srate;
            trial_ranges(end+1, :) = [t_start, t_end];    %#ok<AGROW>
            trial_type_id(end+1)   = td;                   %#ok<AGROW>
            n_seg = n_seg + 1;
            total_dur = total_dur + dur;
            n_this = n_this + 1;
        end
        fprintf('  %s: %d trials\n', trial_defs{td, 3}, n_this);
    end
    if n_seg == 0
        error('prep_step2:noSegments', 'No valid trial segments found.');
    end
    fprintf('  Total: %d segments, %.1f s (%.1f min)\n', n_seg, total_dur, total_dur/60);

    [trial_ranges, sort_ord] = sortrows(trial_ranges, 1);
    trial_type_id = trial_type_id(sort_ord);

    % Build kept-sample index BEFORE pop_select so we can sync goni.
    kept_samples = [];
    for s = 1:size(trial_ranges, 1)
        i1 = max(1, round(trial_ranges(s, 1) * srate) + 1);
        i2 = min(EEG.pnts, round(trial_ranges(s, 2) * srate) + 1);
        kept_samples = [kept_samples, i1:i2]; %#ok<AGROW>
    end

    EEG_seg = pop_select(EEG, 'time', trial_ranges);
    if isempty(EEG_seg.chanlocs) || ...
            (isstruct(EEG_seg.chanlocs) && isfield(EEG_seg.chanlocs, 'labels') && ...
             all(cellfun(@isempty, {EEG_seg.chanlocs.labels})))
        EEG_seg.chanlocs = EEG.chanlocs;
        EEG_seg.chaninfo = EEG.chaninfo;
    end
    EEG_seg = eeg_checkset(EEG_seg, 'eventconsistency');

    % EEG.etc.goni preserved as-is from step1.
    fprintf('  Segmented: %d ch x %d pts\n', EEG_seg.nbchan, EEG_seg.pnts);
end

function EEG = reorder_channels(EEG, whitelist)
    cur_labels = {EEG.chanlocs.labels};
    [~, reorder_idx] = ismember(whitelist, cur_labels);
    reorder_idx = reorder_idx(reorder_idx > 0);
    if length(reorder_idx) == EEG.nbchan
        EEG.data     = EEG.data(reorder_idx, :);
        EEG.chanlocs = EEG.chanlocs(reorder_idx);
        EEG = eeg_checkset(EEG);
    else
        missing = whitelist(~ismember(whitelist, cur_labels));
        extra   = cur_labels(~ismember(cur_labels, whitelist));
        error('prep_step2:chanMismatch', ...
            'Channel mismatch (%d matched vs %d expected). Missing: {%s}. Extra: {%s}', ...
            length(reorder_idx), EEG.nbchan, strjoin(missing, ', '), strjoin(extra, ', '));
    end
end

function EEG = add_fcz_zerofill(EEG)
    % FCz coords must go through pop_chanedit lookup so they end up in
    % EEGLAB's nose=+X frame (matches the 59 channels written by step1).
    % Raw readlocs(.elc) returns nose=+Y coords -> 90deg rotation against
    % the rest, putting FCz at left lateral instead of fronto-central
    EEG.nbchan = EEG.nbchan + 1;
    EEG.data(end+1, :) = zeros(1, EEG.pnts);
    EEG.chanlocs(end+1).labels = 'FCz';
    dipfit_dir = fileparts(which('pop_dipfit_settings'));
    eloc_file = fullfile(dipfit_dir, 'standard_BEM', 'elec', 'standard_1005.elc');
    EEG = pop_chanedit(EEG, 'lookup', eloc_file);
    EEG = eeg_checkset(EEG);
end

function [EEG_seg, n_asr, asr_log] = apply_asr(EEG_seg, trial_defs, srate, asr_k)
    evt_types = {EEG_seg.event.type};
    evt_lats  = [EEG_seg.event.latency];
    boundary_lats = evt_lats(strcmp(evt_types, 'boundary'));
    n_asr = 0;
    asr_log = struct('type', {}, 'index', {}, 'dur_sec', {}, 'status', {}, 'error_msg', {});

    for td = 1:size(trial_defs, 1)
        if ~trial_defs{td, 4}, continue; end  % only ASR-target trial types
        smk = trial_defs{td, 1};
        emk = trial_defs{td, 2};
        r1_mode = '';
        if size(trial_defs, 2) >= 5, r1_mode = trial_defs{td, 5}; end
        s_lats = evt_lats(strcmp(evt_types, smk));
        if ischar(emk)
            e_lats = evt_lats(strcmp(evt_types, emk));
        else
            e_lats = s_lats + emk * srate;
        end

        for i = 1:length(s_lats)
            sl = s_lats(i);
            if ischar(emk)
                nxt = e_lats(e_lats > sl);
                if isempty(nxt), continue; end
                nxt = nxt(1);
                if any(s_lats > sl & s_lats < nxt), continue; end
            else
                nxt = e_lats(i);
                if round(nxt) > EEG_seg.pnts, continue; end
            end
            i1 = max(1, round(sl));
            i2 = min(EEG_seg.pnts, round(nxt));
            dur = (i2 - i1) / srate;
            if dur < 3 || dur > 300, continue; end
            if any(boundary_lats > sl & boundary_lats < nxt), continue; end
            if ~passes_r1_mode(r1_mode, evt_types, evt_lats, sl, nxt, srate), continue; end

            seg_data = EEG_seg.data(:, i1:i2);
            EEG_tmp = eeg_emptyset();
            EEG_tmp.data     = seg_data;
            EEG_tmp.nbchan   = size(seg_data, 1);
            EEG_tmp.pnts     = size(seg_data, 2);
            EEG_tmp.srate    = srate;
            EEG_tmp.xmax     = (EEG_tmp.pnts - 1) / srate;
            EEG_tmp.times    = (0:EEG_tmp.pnts-1) / srate * 1000;
            EEG_tmp.trials   = 1;
            EEG_tmp.chanlocs = EEG_seg.chanlocs;
            EEG_tmp = eeg_checkset(EEG_tmp);

            try
                EEG_tmp = clean_rawdata(EEG_tmp, 'off', 'off', 'off', 'off', asr_k, 'off');
                EEG_seg.data(:, i1:i2) = EEG_tmp.data;
                n_asr = n_asr + 1;
                fprintf('  ASR %s #%d: %.1f s\n', trial_defs{td, 3}, i, dur);
                asr_log(end+1) = struct('type', trial_defs{td, 3}, 'index', i, ...
                    'dur_sec', dur, 'status', 'ok', 'error_msg', '');    %#ok<AGROW>
            catch ME_asr
                fprintf('  ASR SKIP %s #%d (%.1f s): %s\n', trial_defs{td, 3}, i, dur, ME_asr.message);
                asr_log(end+1) = struct('type', trial_defs{td, 3}, 'index', i, ...
                    'dur_sec', dur, 'status', 'skip', 'error_msg', ME_asr.message);    %#ok<AGROW>
            end
        end
    end
    n_skip = sum(strcmp({asr_log.status}, 'skip'));
    fprintf('  ASR total: %d cleaned, %d skipped (k=%g)\n', n_asr, n_skip, asr_k);
end
