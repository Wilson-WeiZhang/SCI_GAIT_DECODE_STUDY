%CPP_PREP_STEP3  Step 3 of the patient preprocessing with the causal switches.
%
%   A copy of 01_preprocessing/patient/03_prep_eeg/prep_step3.m with three additions, as cp_step3.m adds to
%   the healthy step 3; every other line is that file's code:
%     opts.step2_path   the step2 .set whose ICA weights, ICLabel input, bad channels and trial definitions
%                       are applied (default <out_dir>/<label>_step2.set)
%     opts.bp_minphase  1 = the 0.1-40 Hz band-pass converted to minimum phase and applied forward only
%     opts.use_asr      false = no ASR on the ASR target segments (patients: sit-to-stand and two-minute
%                       walking only)
%   With the session's own step2, bp_minphase 0 and use_asr true it reproduces prep_step3 bit for bit.
function out = cpp_prep_step3(step1_set_path, out_dir, opts)
%PREP_STEP3 (patient, prep_patient)  Phase B step 3: 0.1 Hz copy + ICA transfer + ICLabel.
%
%
%   Same as 01_preprocessing/healthy/03_prep_eeg/prep_step3.m, with these patient skip rules:
%     * If step2.etc.skipped_amica == true -> skipped_no_ica
%     * If step2.etc.phase_b_max_step < 3 -> skipped_max_step

    if nargin < 3, opts = struct(); end
    opts = set_default(opts, 'bp_out_lo',       0.1);
    opts = set_default(opts, 'bp_out_hi',       40);
    opts = set_default(opts, 'asr_k',           20);
    opts = set_default(opts, 'artifact_thresh', 0.8);
    opts = set_default(opts, 'overwrite',       false);
    opts = set_default(opts, 'bp_minphase',     0);
    opts = set_default(opts, 'use_asr',         true);

    prep_patient_paths();

    step1_set_path = char(step1_set_path);
    out_dir        = char(out_dir);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    [~, step1_name, ~] = fileparts(step1_set_path);
    label = regexprep(step1_name, '_step1$', '');

    file_step2 = fullfile(out_dir, [label '_step2.set']);
    if isfield(opts, 'step2_path'), file_step2 = char(opts.step2_path); end
    file_step3 = fullfile(out_dir, [label '_step3.set']);

    fprintf('\n================================================================\n');
    fprintf('=== prep_step3 (patient): %s ===\n', label);
    fprintf('================================================================\n');

    if ~opts.overwrite && exist(file_step3, 'file')
        fprintf('  already exists: %s\n', file_step3);
        out = struct('set_path', file_step3, 'status', 'exists', 'n_rej', NaN);
        return;
    end
    if ~exist(file_step2, 'file')
        error('prep_step3 (prep_patient):noStep2', 'step2 .set not found: %s', file_step2);
    end

    [s2_dir, s2_name, s2_ext] = fileparts(file_step2);
    EEG_seg = pop_loadset('filename', [s2_name s2_ext], 'filepath', s2_dir);
    EEG_seg = eeg_checkset(EEG_seg);

    if isfield(EEG_seg.etc, 'skipped_amica') && EEG_seg.etc.skipped_amica
        fprintf('  step2 skipped ICA -> step3 is skipped as well.\n');
        out = struct('set_path', '', 'status', 'skipped_no_ica', 'n_rej', NaN);
        return;
    end
    if isfield(EEG_seg.etc, 'phase_b_max_step') && EEG_seg.etc.phase_b_max_step < 3
        fprintf('  step2 max_step=%d -> step3 skipped.\n', EEG_seg.etc.phase_b_max_step);
        out = struct('set_path', '', 'status', 'skipped_max_step', 'n_rej', NaN);
        return;
    end

    meta = EEG_seg.etc.step2_meta;
    bad_labels      = meta.bad_ch_labels;
    brain_chanlocs  = meta.brain_chanlocs;
    brain_whitelist = meta.brain_whitelist;
    trial_defs      = meta.trial_defs;

    [in_dir, in_name, ~] = fileparts(step1_set_path);
    EEG_out = pop_loadset('filename', [in_name '.set'], 'filepath', in_dir);
    EEG_out = eeg_checkset(EEG_out);
    fprintf('  Loaded step1: %d ch x %d pts @ %d Hz\n', EEG_out.nbchan, EEG_out.pnts, EEG_out.srate);

    fprintf('\n[1] BP %g-%g Hz (output copy)\n', opts.bp_out_lo, opts.bp_out_hi);
    EEG_out = pop_eegfiltnew(EEG_out, 'locutoff', opts.bp_out_lo, 'hicutoff', opts.bp_out_hi, 'minphase', opts.bp_minphase);
    EEG_out = eeg_checkset(EEG_out);

    EEG_out = trim_s11_s12(EEG_out);
    % step3 reloads step1, so its S4/S5 are still un-split, while trial_defs (read from step2's
    % meta) addresses rest as 'S  4r'/'S  5r'. Without this the 'S2S' row (asr_target=true) would
    % match EVERY S4 -- including the ~45 Relax breaks -- and ASR-clean the rest again.
    EEG_out = split_rest_events(EEG_out);

    if ~isempty(bad_labels)
        keep_mask = ~ismember({EEG_out.chanlocs.labels}, bad_labels);
        EEG_out = pop_select(EEG_out, 'channel', find(keep_mask));
        EEG_out = eeg_checkset(EEG_out);
        interp_mask = ismember({brain_chanlocs.labels}, bad_labels);
        target_locs = [EEG_out.chanlocs, brain_chanlocs(interp_mask)];
        EEG_out = pop_interp(EEG_out, target_locs, 'spherical');
        EEG_out = eeg_checkset(EEG_out);
        fprintf('  Interpolated %d bad ch on 0.1Hz copy\n', length(bad_labels));
    end
    EEG_out = reorder_channels(EEG_out, brain_whitelist);

    fprintf('\n[4] Restore FCz zero-fill + CAR (60ch)\n');
    EEG_out = add_fcz_zerofill(EEG_out);
    EEG_out = pop_reref(EEG_out, []);
    EEG_out = reorder_channels(EEG_out, [brain_whitelist, {'FCz'}]);
    EEG_out = eeg_checkset(EEG_out);

    fprintf('\n[5] ASR (k=%g) on ASR-target trials (continuous)\n', opts.asr_k);
    if opts.use_asr
        [EEG_out, n_asr_out, asr_log_out] = apply_asr(EEG_out, trial_defs, EEG_out.srate, opts.asr_k);
    else
        n_asr_out = 0; asr_log_out = struct('type', {}, 'index', {}, 'dur_sec', {}, 'status', {}, 'error_msg', {});
        fprintf('  [cpp_prep_step3] ASR skipped (use_asr false)\n');
    end

    fprintf('\n[6] Transfer ICA weights from step2\n');
    if size(EEG_seg.icasphere, 2) ~= EEG_out.nbchan
        error('prep_step3 (prep_patient):channelCountMismatch', ...
            'step2 ICA expects %d ch but step3 has %d ch.', size(EEG_seg.icasphere, 2), EEG_out.nbchan);
    end
    EEG_out.icaweights  = EEG_seg.icaweights;
    EEG_out.icasphere   = EEG_seg.icasphere;
    EEG_out.icawinv     = EEG_seg.icawinv;
    EEG_out.icachansind = EEG_seg.icachansind;
    EEG_out = eeg_checkset(EEG_out);
    fprintf('  %d components transferred\n', size(EEG_out.icaweights, 1));

    fprintf('\n[7] ICLabel (step2-scored; artifact > %.2f rejected)\n', opts.artifact_thresh);
    EEG_score  = iclabel(EEG_seg);
    ic_classes = EEG_score.etc.ic_classification.ICLabel.classifications;
    % Carry the step2-scored ICLabel onto the step3 output so QC readers
    % (make_qc_summary / plot_iclabel read EEG.etc.ic_classification) see it.
    EEG_out.etc.ic_classification = EEG_score.etc.ic_classification;
    artifact_cols = [2 3 4 5 6];
    artifact_max  = max(ic_classes(:, artifact_cols), [], 2);
    rej_idx = find(artifact_max > opts.artifact_thresh);
    % Diagnostic option, off by default and not used for the published results: also reject any component whose ARGMAX
    % class is ChanNoise, regardless of whether it clears artifact_thresh.
    if isfield(opts, 'reject_channoise_argmax') && opts.reject_channoise_argmax
        [~, argmax_cls] = max(ic_classes, [], 2);
        cn_idx = find(argmax_cls == 6);          % 6 = ChanNoise
        n_extra = numel(setdiff(cn_idx, rej_idx));
        rej_idx = union(rej_idx, cn_idx);
        fprintf(' [reject_channoise_argmax] +%d ChanNoise IC (%d)\n', n_extra, numel(rej_idx));
    end
    n_rej = length(rej_idx);
    label_names = {'Brain','Muscle','Eye','Heart','LineNoise','ChanNoise','Other'};
    rej_labels = cell(n_rej, 1);
    for k = 1:n_rej
        [~, mc] = max(ic_classes(rej_idx(k), :));
        rej_labels{k} = label_names{mc};
    end
    fprintf('  Rejecting %d / %d ICs\n', n_rej, size(ic_classes, 1));
    if n_rej > 0
        fprintf('    Categories: %s\n', strjoin(rej_labels, ', '));
        EEG_out.etc.step3_meta.icaweights_pre_reject = EEG_seg.icaweights;
        EEG_out.etc.step3_meta.icasphere_pre_reject  = EEG_seg.icasphere;
        EEG_out = pop_subcomp(EEG_out, rej_idx, 0);
        EEG_out = eeg_checkset(EEG_out);
    end

    EEG_out.etc.step3_meta.iclabel_score_source          = 'step2';
    EEG_out.etc.step3_meta.step2_iclabel_classifications = ic_classes;
    EEG_out.etc.step3_meta.rejected_ics      = rej_idx(:)';
    EEG_out.etc.step3_meta.ic_labels         = rej_labels;
    EEG_out.etc.step3_meta.n_ics_rejected    = n_rej;
    EEG_out.etc.step3_meta.artifact_thresh   = opts.artifact_thresh;
    EEG_out.etc.step3_meta.n_asr_segments    = n_asr_out;
    EEG_out.etc.step3_meta.asr_log           = asr_log_out;
    EEG_out.etc.step3_meta.step2_meta        = meta;
    EEG_out.etc.step3_meta.bp_out_lo         = opts.bp_out_lo;
    EEG_out.etc.step3_meta.bp_out_hi         = opts.bp_out_hi;
    EEG_out.etc.step3_meta.pipeline = sprintf( ...
        'REFERENCE-patient prep-aligned: step1(250Hz)->BP(%g-%g)->trim->badch interp->+FCz->CAR->ASR(k=%g)->ICA transfer(runica)->ICLabel[step2-scored](>%.1f)', ...
        opts.bp_out_lo, opts.bp_out_hi, opts.asr_k, opts.artifact_thresh);

    pop_saveset(EEG_out, 'filename', [label '_step3.set'], 'filepath', out_dir, 'savemode', 'onefile');
    fprintf('\n[done] %s step3 -> %s\n', label, file_step3);

    out = struct('set_path', file_step3, 'status', 'ok', 'n_rej', n_rej);
end


% ======================== helpers ========================

function s = set_default(s, f, default)
    if ~isfield(s, f), s.(f) = default; end
end

function EEG = split_rest_events(EEG)
%SPLIT_REST_EVENTS  Rename the auto Relax-break S4/S5 events to 'S  4r'/'S  5r'.
% Kept in sync with the copy in prep_step2.m, mirroring how prep carries passes_r1_mode in both
% prep_step2.m and prep_step3.m. See prep_step2.m for the full rationale (4-bit Cedrus C-POD
% overflow of PyLog marker 20/21; rest 3.7-11.2 s vs true sit-to-stand 270-1370 s after the
% preceding 'S  2').
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

function EEG = trim_s11_s12(EEG)
    if isempty(EEG.event), return; end
    evt_types = {EEG.event.type};
    evt_lats  = [EEG.event.latency];
    s11 = find(strcmp(evt_types, 'S 11'), 1, 'first');
    s12 = find(strcmp(evt_types, 'S 12'), 1, 'last');
    if ~isempty(s11) && ~isempty(s12)
        i1 = max(1, round(evt_lats(s11)));
        i2 = min(EEG.pnts, round(evt_lats(s12)));
        fprintf('  Trim S11-S12: %.1f-%.1f s\n', (i1-1)/EEG.srate, (i2-1)/EEG.srate);
        EEG = pop_select(EEG, 'point', [i1 i2]);
        EEG = eeg_checkset(EEG);
    elseif ~isempty(s11)
        i1 = max(1, round(evt_lats(s11)));
        fprintf('  Trim S11-end (no S12): %.1f-%.1f s\n', (i1-1)/EEG.srate, (EEG.pnts-1)/EEG.srate);
        EEG = pop_select(EEG, 'point', [i1 EEG.pnts]);
        EEG = eeg_checkset(EEG);
    end
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
        error('prep_step3 (prep_patient):chanMismatch', ...
            'Channel mismatch (%d matched vs %d expected). Missing: {%s}. Extra: {%s}', ...
            length(reorder_idx), EEG.nbchan, strjoin(missing, ', '), strjoin(extra, ', '));
    end
end

function EEG = add_fcz_zerofill(EEG)
    % See prep_step2.m (add_fcz_zerofill) for the rationale.
    EEG.nbchan = EEG.nbchan + 1;
    EEG.data(end+1, :) = zeros(1, EEG.pnts);
    EEG.chanlocs(end+1).labels = 'FCz';
    dipfit_dir = fileparts(which('pop_dipfit_settings'));
    eloc_file = fullfile(dipfit_dir, 'standard_BEM', 'elec', 'standard_1005.elc');
    EEG = pop_chanedit(EEG, 'lookup', eloc_file);
    EEG = eeg_checkset(EEG);
end

function [EEG, n_asr, asr_log] = apply_asr(EEG, trial_defs, srate, asr_k)
    evt_types = {EEG.event.type};
    evt_lats  = [EEG.event.latency];
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
                if round(nxt) > EEG.pnts, continue; end
            end
            i1 = max(1, round(sl));
            i2 = min(EEG.pnts, round(nxt));
            dur = (i2 - i1) / srate;
            if dur < 3 || dur > 300, continue; end
            if any(boundary_lats > sl & boundary_lats < nxt), continue; end
            seg_data = EEG.data(:, i1:i2);
            EEG_tmp = eeg_emptyset();
            EEG_tmp.data     = seg_data;
            EEG_tmp.nbchan   = size(seg_data, 1);
            EEG_tmp.pnts     = size(seg_data, 2);
            EEG_tmp.srate    = srate;
            EEG_tmp.xmax     = (EEG_tmp.pnts - 1) / srate;
            EEG_tmp.times    = (0:EEG_tmp.pnts-1) / srate * 1000;
            EEG_tmp.trials   = 1;
            EEG_tmp.chanlocs = EEG.chanlocs;
            EEG_tmp = eeg_checkset(EEG_tmp);
            try
                EEG_tmp = clean_rawdata(EEG_tmp, 'off', 'off', 'off', 'off', asr_k, 'off');
                EEG.data(:, i1:i2) = EEG_tmp.data;
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
