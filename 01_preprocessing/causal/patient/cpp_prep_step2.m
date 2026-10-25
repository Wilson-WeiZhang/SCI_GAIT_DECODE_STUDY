%CPP_PREP_STEP2  Step 2 of the patient preprocessing with a reproducible ICA.
%
%   A copy of the frozen patient step 2 (01_preprocessing/patient/03_prep_eeg/prep_step2.m, in the variant
%   that also serves the three sessions with the connector fix) with two changes only: the function name,
%   and 'rndreset', 'off' in the pop_runica call, so runica keeps the random state set by the caller.
function out = cpp_prep_step2(step1_set_path, out_dir, opts)
%
%
%
%   Same as 01_preprocessing/healthy/03_prep_eeg/prep_step2.m, with these patient differences:
%     * Trial defs driven by EEG.etc.marker_protocol (set by 01_align):
%       'standard'  -> MI (S1->S2), S2S (S4->S5, ASR), Walk2min (S6->S7 or legacy fallback, ASR)
%       'legacy'    -> MI only (S1->S2), no S4/S5/S7
%     * Reads max_step from patient registry; if <3 the EEG.etc.phase_b_max_step
%       flag is propagated so prep_step3 skips itself.
%
%   Cohort = 'patient' enforced; FCz restored same as healthy (60 ch out).

    if nargin < 3, opts = struct(); end
    opts = set_default(opts, 'ica_method',    'runica');
    opts = set_default(opts, 'asr_k',         20);
    opts = set_default(opts, 'bp_ica_lo',     1);
    opts = set_default(opts, 'bp_ica_hi',     40);
    opts = set_default(opts, 'min_amica_min', 5);
    opts = set_default(opts, 'overwrite',     false);
    % Diagnostic knobs; defaults reproduce the frozen behaviour exactly.
    % badch_corr: clean_rawdata correlation criterion (0.8 = frozen, matches healthy prep).
    % diagnostic_only: stop after bad-channel detection and return the counts, skipping CAR/ASR/ICA.
    opts = set_default(opts, 'badch_corr',    0.8);
    opts = set_default(opts, 'diagnostic_only',    false);

    prep_patient_paths();

    step1_set_path = char(step1_set_path);
    out_dir        = char(out_dir);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    [~, step1_name, ~] = fileparts(step1_set_path);
    label = regexprep(step1_name, '_step1$', '');

    file_step2 = fullfile(out_dir, [label '_step2.set']);

    fprintf('\n================================================================\n');
    fprintf('=== prep_step2 (patient): %s ===\n', label);
    fprintf('================================================================\n');

    if ~opts.overwrite && exist(file_step2, 'file')
        fprintf('  already exists: %s\n', file_step2);
        EEG_seg = pop_loadset('filename', [label '_step2.set'], 'filepath', out_dir);
        out = step2_out(file_step2, EEG_seg);
        return;
    end

    [in_dir, in_name, ~] = fileparts(step1_set_path);
    EEG = pop_loadset('filename', [in_name '.set'], 'filepath', in_dir);
    EEG = eeg_checkset(EEG);
    fprintf('  Loaded step1: %d ch x %d pts @ %d Hz (%.1f s)\n', ...
        EEG.nbchan, EEG.pnts, EEG.srate, EEG.xmax);

    % --- patient-specific config from EEG.etc + registry ---
    cohort = 'patient';
    if isfield(EEG.etc, 'cohort'), cohort = char(EEG.etc.cohort); end
    if ~strcmp(cohort, 'patient')
        warning('prep_step2 (prep_patient): EEG.etc.cohort=%s, expected patient', cohort);
    end
    marker_protocol = 'standard';
    if isfield(EEG.etc, 'marker_protocol') && ~isempty(EEG.etc.marker_protocol)
        marker_protocol = char(EEG.etc.marker_protocol);
    end
    fprintf('  marker_protocol: %s\n', marker_protocol);

    [reg_max_step, reg_special] = lookup_registry_max_step(label);
    fprintf('  registry max_step: %d (special=%s)\n', reg_max_step, reg_special);

    % --- module connector sessions ---
    CONNECTOR_FIX_LABELS = connector_fix_sessions();
    if ismember(label, CONNECTOR_FIX_LABELS) && reg_max_step < 3
        fprintf('  [connector_fix] %s: registry max_step %d -> 3 (channel order restored at alignment)\n', ...
            label, reg_max_step);
        reg_max_step = 3;
    end

    brain_whitelist = prep_patient_brain_whitelist();
    brain_chanlocs = EEG.chanlocs;
    brain_labels   = {EEG.chanlocs.labels};

    EEG = split_rest_events(EEG);
    trial_defs = get_patient_trial_defs(marker_protocol, {EEG.event.type});

    % ---- 1. BP 1-40 Hz ----
    fprintf('\n[1] BP %g-%g Hz (ICA copy)\n', opts.bp_ica_lo, opts.bp_ica_hi);
    EEG = pop_eegfiltnew(EEG, 'locutoff', opts.bp_ica_lo, 'hicutoff', opts.bp_ica_hi);
    EEG = eeg_checkset(EEG);

    % ---- 2. Trim S11-S12 (S12 may be missing for some patient sessions) ----
    EEG = trim_s11_s12(EEG);

    % ---- 3. Extract task segments (no baseline) ----
    fprintf('\n[3] Extract task segments (no BL)\n');
    [EEG_seg, trial_type_id, n_seg, total_dur] = extract_segments_no_bl(EEG, trial_defs, EEG.srate);
    clear EEG;

    % ---- 4. Bad-channel detection ----
    fprintf('\n[4] Bad-channel detection (clean_rawdata corr=0.8)\n');
    EEG_probe = clean_rawdata(EEG_seg, 5, 'off', opts.badch_corr, 4, 'off', 'off');
    clean_labels = {EEG_probe.chanlocs.labels};
    bad_mask     = ~ismember(brain_labels, clean_labels);
    bad_labels   = brain_labels(bad_mask);
    n_bad        = sum(bad_mask);
    clear EEG_probe;
    if n_bad > 15
        warning('prep_step2:manyBadCh', '%s: many bad ch (%d): %s', label, n_bad, strjoin(bad_labels, ', '));
    elseif n_bad > 0
        fprintf('  Bad channels (%d): %s\n', n_bad, strjoin(bad_labels, ', '));
    else
        fprintf('  No bad channels detected.\n');
    end
    if opts.diagnostic_only
        out = struct('diagnostic_only', true, 'label', label, 'badch_corr', opts.badch_corr, ...
                     'n_bad', n_bad, 'bad_labels', {bad_labels}, 'n_trial_segs', n_seg, ...
                     'trial_dur_sec', total_dur);
        return;
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

    % ---- 5. Restore FCz + CAR ----
    fprintf('\n[5] Restore FCz zero-fill + CAR (60ch)\n');
    EEG_seg = add_fcz_zerofill(EEG_seg);
    EEG_seg = pop_reref(EEG_seg, []);
    EEG_seg = reorder_channels(EEG_seg, [brain_whitelist, {'FCz'}]);
    EEG_seg = eeg_checkset(EEG_seg);

    n_brain = EEG_seg.nbchan;
    formula_rank = n_brain - n_bad;       % FCz zero-fill absorbs the avgref-1 rank loss
    numerical_rank = rank(double(EEG_seg.data'));
    data_rank = min(formula_rank, numerical_rank);
    fprintf('  Data rank: %d (formula=%d, numerical=%d, nch=%d, interp=%d)\n', ...
        data_rank, formula_rank, numerical_rank, n_brain, n_bad);

    % ---- 6. ASR on Walk segments only ----
    fprintf('\n[6] ASR (k=%g) on ASR-target trials\n', opts.asr_k);
    [EEG_seg, n_asr, asr_log] = apply_asr(EEG_seg, trial_defs, EEG_seg.srate, opts.asr_k);

    % ---- 7. ICA ----
    dur_min = total_dur / 60;
    skipped_ica = false;
    if dur_min < opts.min_amica_min
        warning('prep_step2:skipICA', '%s: duration %.1f min < %.0f min -> skipping ICA.', ...
            label, dur_min, opts.min_amica_min);
        skipped_ica = true;
    elseif reg_max_step < 3
        fprintf('  registry max_step=%d -> skipping ICA\n', reg_max_step);
        skipped_ica = true;
    else
        fprintf('\n[7] runica (extended infomax): %d ch x %d pts, pca=%d\n', ...
            EEG_seg.nbchan, EEG_seg.pnts, data_rank);
        EEG_seg = pop_runica(EEG_seg, 'icatype', 'runica', 'extended', 1, 'pca', data_rank, 'rndreset', 'off');
        EEG_seg = eeg_checkset(EEG_seg);
        fprintf('  runica done: %d components\n', size(EEG_seg.icaweights, 1));
    end

    EEG_seg.etc.step2_meta = struct( ...
        'bad_ch_labels',     {bad_labels}, ...
        'brain_chanlocs',    brain_chanlocs, ...
        'brain_whitelist',   {brain_whitelist}, ...
        'restore_fcz',       true, ...
        'n_trial_segs',      n_seg, ...
        'trial_type_id',     trial_type_id, ...
        'trial_defs',        {trial_defs}, ...
        'trial_dur_sec',     total_dur, ...
        'n_ch_interpolated', n_bad, ...
        'data_rank',         data_rank, ...
        'n_asr_segments',    n_asr, ...
        'asr_log',           asr_log, ...
        'ica_method',        opts.ica_method, ...
        'cohort',            cohort, ...
        'marker_protocol',   marker_protocol, ...
        'phase_b_max_step',  reg_max_step, ...
        'special_case',      reg_special);
    EEG_seg.etc.skipped_amica   = skipped_ica;   % keep field name for prep_step3 compat
    EEG_seg.etc.phase_b_max_step = reg_max_step;

    pop_saveset(EEG_seg, 'filename', [label '_step2.set'], 'filepath', out_dir, 'savemode', 'onefile');
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
        'skipped_amica', isfield(EEG_seg.etc, 'skipped_amica') && EEG_seg.etc.skipped_amica, ...
        'max_step',      meta.phase_b_max_step);
end

function s = set_default(s, f, default)
    if ~isfield(s, f), s.(f) = default; end
end

function trial_defs = get_patient_trial_defs(marker_protocol, eeg_event_types)
% trial_defs columns: {start_marker, end_marker_or_dursec, name, asr_target}
% Walk2min: prefer S7→S8 paired if S8 exists in this recording (handles
% multi-pair Walk2min); else fall back to S7 + 120s.
%
% Rest vs S2S: split_rest_events has already renamed the auto Relax break to 'S  4r'/'S  5r', so
% the two ex-colliding classes are now addressed by distinct markers. Rest carries asr_target=false
% to match the healthy prep Rest row (prep get_trial_defs: MI false / Walk true / Rest false).
    switch lower(marker_protocol)
        case 'standard'
            if nargin >= 2
                [walk_on, walk_off] = choose_walk_markers(eeg_event_types);
            else
                walk_on = 'S  7'; walk_off = 120;
            end
            trial_defs = {
                'S  1',  'S  2',   'MI',       false
                'S  4r', 'S  5r',  'Rest',     false
                'S  4',  'S  5',   'S2S',      true
                walk_on, walk_off, 'Walk2min', true
            };
        case 'legacy'
            trial_defs = {
                'S  1', 'S  2', 'MI', false
            };
        otherwise
            error('prep_step2 (prep_patient): unknown marker_protocol=%s', marker_protocol);
    end
end

function EEG = split_rest_events(EEG)
%SPLIT_REST_EVENTS  Rename the auto Relax-break S4/S5 events to 'S  4r'/'S  5r'.
%
% S4/S5 is a marker collision. PyLog sends marker 20/21 for the automatic Relax break after every
% imagined-walk trial, but the Cedrus C-POD trigger port is 4 bit, so 20 & 0b1111 = 4 and 21 -> 5.
%
% The two separate on the gap to the preceding MI end ('S  2'): the Relax break opens 3.7-11.2 s
% after the imagined-walk trial it follows, whereas true sit-to-stand opens a new block 270-1370 s later. 
%
    if isempty(EEG.event), return; end
    gap_rest_max_sec = 30;   % sits inside the empty band between 11.2 s and 270 s
    evt_types = {EEG.event.type};
    evt_lats  = [EEG.event.latency];
    mi_end = sort(evt_lats(strcmp(evt_types, 'S  2')));
    n_rest = 0; n_s2s = 0;
    for ei = 1:numel(EEG.event)
        t = EEG.event(ei).type;
        if ~ischar(t) || ~ismember(t, {'S  4', 'S  5'}), continue; end
        prev = mi_end(mi_end < EEG.event(ei).latency);
        if isempty(prev)
            n_s2s = n_s2s + 1;   % no preceding imagined-walk trial -> cannot be a Relax break
            continue;
        end
        if (EEG.event(ei).latency - prev(end)) / EEG.srate <= gap_rest_max_sec
            EEG.event(ei).type = [t 'r'];
            n_rest = n_rest + 1;
        else
            n_s2s = n_s2s + 1;
        end
    end
    EEG = eeg_checkset(EEG, 'eventconsistency');
    fprintf('  [rest/S2S split] %d Relax-break events -> S4r/S5r; %d kept as true sit-to-stand S4/S5\n', ...
        n_rest, n_s2s);
end

function [walk_on, walk_off] = choose_walk_markers(types)
    if any(strcmp(types, 'S  6')) && any(strcmp(types, 'S  7'))
        walk_on = 'S  6';
        walk_off = 'S  7';
    elseif any(strcmp(types, 'S  8'))
        walk_on = 'S  7';
        walk_off = 'S  8';
    else
        walk_on = 'S  7';
        walk_off = 120;
    end
end

function [max_step, special] = lookup_registry_max_step(label)
    max_step = 3; special = '';
    try
        reg = build_patient_registry();
        for k = 1:length(reg)
            if strcmp(reg(k).label, label)
                max_step = reg(k).max_step;
                special  = reg(k).special_case;
                return;
            end
        end
        warning('prep_step2 (prep_patient): %s not in registry, defaulting max_step=3', label);
    catch ME
        warning('prep_step2 (prep_patient): build_patient_registry failed (%s); defaulting max_step=3', ME.message);
    end
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
        EEG = pop_select(EEG, 'point', [i1 i2]);
        EEG = eeg_checkset(EEG);
    elseif ~isempty(s11)
        i1 = max(1, round(evt_lats(s11)));
        fprintf('  Trim S11-end (no S12): %.1f-%.1f s\n', (i1-1)/EEG.srate, (EEG.pnts-1)/EEG.srate);
        EEG = pop_select(EEG, 'point', [i1 EEG.pnts]);
        EEG = eeg_checkset(EEG);
    else
        fprintf('  No S11/S12, keeping full recording.\n');
    end
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
            i1 = max(1, round(sl));
            i2 = min(EEG.pnts, round(nxt));
            t_start = (i1 - 1) / srate;
            t_end   = (i2 - 1) / srate;
            trial_ranges(end+1, :) = [t_start, t_end]; %#ok<AGROW>
            trial_type_id(end+1)   = td; %#ok<AGROW>
            n_seg = n_seg + 1;
            total_dur = total_dur + dur;
            n_this = n_this + 1;
        end
        fprintf('  %s: %d trials\n', trial_defs{td, 3}, n_this);
    end
    if n_seg == 0
        error('prep_step2 (prep_patient):noSegments', 'No valid trial segments found.');
    end
    fprintf('  Total: %d segments, %.1f s (%.1f min)\n', n_seg, total_dur, total_dur/60);
    [trial_ranges, sort_ord] = sortrows(trial_ranges, 1);
    trial_type_id = trial_type_id(sort_ord);
    EEG_seg = pop_select(EEG, 'time', trial_ranges);
    if isempty(EEG_seg.chanlocs) || ...
            (isstruct(EEG_seg.chanlocs) && isfield(EEG_seg.chanlocs, 'labels') && ...
             all(cellfun(@isempty, {EEG_seg.chanlocs.labels})))
        EEG_seg.chanlocs = EEG.chanlocs;
        EEG_seg.chaninfo = EEG.chaninfo;
    end
    EEG_seg = eeg_checkset(EEG_seg, 'eventconsistency');
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
        error('prep_step2 (prep_patient):chanMismatch', ...
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
        if ~trial_defs{td, 4}, continue; end
        smk = trial_defs{td, 1};
        emk = trial_defs{td, 2};
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
                    'dur_sec', dur, 'status', 'ok', 'error_msg', ''); %#ok<AGROW>
            catch ME_asr
                fprintf('  ASR SKIP %s #%d (%.1f s): %s\n', trial_defs{td, 3}, i, dur, ME_asr.message);
                asr_log(end+1) = struct('type', trial_defs{td, 3}, 'index', i, ...
                    'dur_sec', dur, 'status', 'skip', 'error_msg', ME_asr.message); %#ok<AGROW>
            end
        end
    end
    fprintf('  ASR total: %d cleaned, %d skipped (k=%g)\n', n_asr, ...
        sum(strcmp({asr_log.status}, 'skip')), asr_k);
end
