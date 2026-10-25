function out = prep_step3(step1_set_path, out_dir, opts)
%PREP_STEP3  Healthy step 3: transfer the step-2 ICA solution and remove artefact components.
%
%   One fixed method, no variant switches. Produces the continuous cleaned EEG that
%   stage 04_export writes out for decoding.
%
%   Method:
%     1. Load the step-1 recording (unfiltered, 250 Hz) and band-pass 0.1-40 Hz
%     2. Trim the pre- and post-task data (S11-S12)
%     3. Interpolate the same bad channels as step 2
%     4. Restore FCz, then common average reference
%     5. ASR (k = 20) on the real-walking segments, in place
%     6. Transfer the runica weights from step 2, where the ICA was trained on the
%        imagined-walking, real-walking and rest segments together
%     7. Score ICLabel on that step-2 decomposition and reject every component whose
%        largest artefact-class probability exceeds 0.8
%        8. Also reject the components listed for the session in manual_eye_ic_overrides.csv, next to this file.
%
%   Input  : {label}_step1.set (stage 01_align, unfiltered)
%            {label}_step2.set (03_prep_eeg/prep_step2; ICA weights + bad-channel list)
%   Output : {label}_step3.set (0.1-40 Hz continuous, artefact components removed)
%
%   Opts (operational only, not method variants):
%     overwrite  (default false)

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'overwrite'), opts.overwrite = false; end

    % --- locked method constants (single method, no variants) ---
    BP_LO = 0.1;
    BP_HI = 40;
    ASR_K = 20;
    ARTIFACT_THRESH = 0.8;   % reject IC where max artifact-class prob exceeds this
    OUTPUT_STAGE = 'step3';

    step1_set_path = char(step1_set_path);
    out_dir        = char(out_dir);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    [~, step1_name, ~] = fileparts(step1_set_path);
    label = regexprep(step1_name, '_step1$', '');

    file_step2 = fullfile(out_dir, [label '_step2.set']);
    file_step3 = fullfile(out_dir, [label '_' OUTPUT_STAGE '.set']);

    fprintf('\n================================================================\n');
    fprintf('=== prep_step3 (prep): %s ===\n', label);
    fprintf('================================================================\n');

    if ~opts.overwrite && exist(file_step3, 'file')
        fprintf('  already exists: %s\n', file_step3);
        out = struct('set_path', file_step3, 'status', 'exists', 'n_rej', NaN);
        return;
    end
    if ~exist(file_step2, 'file')
        error('prep_step3:noStep2', 'step2 .set not found: %s', file_step2);
    end

    % ---- Load step2 for ICA weights + bad-ch meta ----
    EEG_seg = pop_loadset('filename', [label '_step2.set'], 'filepath', out_dir);
    EEG_seg = eeg_checkset(EEG_seg);
    if isfield(EEG_seg.etc, 'skipped_amica') && EEG_seg.etc.skipped_amica
        fprintf('  step2 skipped ICA -> step3 is skipped as well.\n');
        out = struct('set_path', '', 'status', 'skipped_no_ica', 'n_rej', NaN);
        return;
    end
    meta = EEG_seg.etc.step2_meta;
    bad_labels      = meta.bad_ch_labels;
    brain_chanlocs  = meta.brain_chanlocs;
    brain_whitelist = meta.brain_whitelist;
    restore_fcz     = meta.restore_fcz;
    trial_defs      = meta.trial_defs;

    % ---- Load step1 (unfiltered continuous) ----
    [in_dir, in_name, ~] = fileparts(step1_set_path);
    EEG_out = pop_loadset('filename', [in_name '.set'], 'filepath', in_dir);
    EEG_out = eeg_checkset(EEG_out);
    fprintf('  Loaded step1: %d ch x %d pts @ %d Hz\n', EEG_out.nbchan, EEG_out.pnts, EEG_out.srate);

    if isfield(EEG_out.etc, 'phase_b_max_step') && EEG_out.etc.phase_b_max_step < 3
        fprintf('  Phase A flagged max_step=%d -> skip step3.\n', EEG_out.etc.phase_b_max_step);
        out = struct('set_path', '', 'status', 'skipped_max_step', 'n_rej', NaN);
        return;
    end

    % ---- 1. BP 0.1-40 Hz ----
    fprintf('\n[1] BP %g-%g Hz (output copy)\n', BP_LO, BP_HI);
    EEG_out = pop_eegfiltnew(EEG_out, 'locutoff', BP_LO, 'hicutoff', BP_HI);
    EEG_out = eeg_checkset(EEG_out);

    % ---- 2. Trim S11-S12 ----
    EEG_out = trim_s11_s12(EEG_out);

    % ---- 3. Bad-ch interp (same channels as step2) ----
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

    % ---- 4. Restore FCz (healthy) -> CAR ----
    if restore_fcz
        fprintf('\n[4] Restore FCz zero-fill + CAR (60ch)\n');
        EEG_out = add_fcz_zerofill(EEG_out);
        EEG_out = pop_reref(EEG_out, []);
        EEG_out = reorder_channels(EEG_out, [brain_whitelist, {'FCz'}]);
    else
        fprintf('\n[4] CAR only (restore_fcz=false)\n');
        EEG_out = pop_reref(EEG_out, []);
        EEG_out = reorder_channels(EEG_out, brain_whitelist);
    end
    EEG_out = eeg_checkset(EEG_out);

    % ---- 5. ASR (k=20) on Walk segments (continuous, in-place) ----
    fprintf('\n[5] ASR (k=%g) on Walk segments (continuous)\n', ASR_K);
    [EEG_out, n_asr_out, asr_log_out] = apply_asr(EEG_out, trial_defs, EEG_out.srate, ASR_K);

    % ---- 6. Transfer runica weights from step2 (MI+Walk+Rest ICA) ----
    fprintf('\n[6] Transfer runica weights from step2 (MI+Walk+Rest)\n');
    if size(EEG_seg.icasphere, 2) ~= EEG_out.nbchan
        error('prep_step3:channelCountMismatch', ...
            'step2 ICA expects %d ch but step3 has %d ch.', ...
            size(EEG_seg.icasphere, 2), EEG_out.nbchan);
    end
    EEG_out.icaweights  = EEG_seg.icaweights;
    EEG_out.icasphere   = EEG_seg.icasphere;
    EEG_out.icawinv     = EEG_seg.icawinv;
    EEG_out.icachansind = EEG_seg.icachansind;
    EEG_out = eeg_checkset(EEG_out);
    fprintf('  %d components transferred\n', size(EEG_out.icaweights, 1));

    % ---- 7. ICLabel (scored on step2 MI+Walk+Rest decomposition) -> reject ----
    fprintf('\n[7] ICLabel (score on step2; artifact > %.2f rejected)\n', ARTIFACT_THRESH);
    EEG_score  = iclabel(EEG_seg);
    ic_classes = EEG_score.etc.ic_classification.ICLabel.classifications;
    artifact_cols = [2 3 4 5 6];   % Muscle Eye Heart LineNoise ChanNoise
    artifact_max  = max(ic_classes(:, artifact_cols), [], 2);
    rej_idx = find(artifact_max > ARTIFACT_THRESH);

    % ---- 8. Manual ocular-component override ----
    % Visually-confirmed ocular ICs that ICLabel scored below ARTIFACT_THRESH are
    % force-added to the rejection set here, read from manual_eye_ic_overrides.csv
    % (co-located with this script). Recorded in step3_meta.manual_eye_ics.
    %
    % This is the single documented exception to the rule that step 3 has one fixed
    % method. ICLabel's Eye class is trained mostly on blinks and systematically
    % misses horizontal saccades, so a lateral-saccade component can stay below the
    % 0.80 gate and survive into step 3 while carrying a large share of the session's
    % component variance. The components listed in the csv were identified by visual
    % inspection of topography, time course and spectrum. The list is additive: a
    % listed component that ICLabel already rejects changes nothing.
    manual_eye_ics = read_manual_eye_overrides(label, size(ic_classes,1));
    if ~isempty(manual_eye_ics)
        new_ics = setdiff(manual_eye_ics, rej_idx(:)');
        rej_idx = union(rej_idx(:), manual_eye_ics(:));
        fprintf('  [manual eye override] %s: force-reject IC %s (new beyond ICLabel: %s)\n', ...
            label, mat2str(manual_eye_ics(:)'), mat2str(new_ics(:)'));
    end
    EEG_out.etc.step3_meta.manual_eye_ics = manual_eye_ics(:)';

    n_rej = length(rej_idx);

    label_names = {'Brain','Muscle','Eye','Heart','LineNoise','ChanNoise','Other'};
    rej_labels = cell(n_rej, 1);
    for k = 1:n_rej
        [~, mc] = max(ic_classes(rej_idx(k), :));
        rej_labels{k} = label_names{mc};
    end
    fprintf('  Candidate artifact ICs: %d / %d\n', n_rej, size(ic_classes, 1));
    if n_rej > 0
        fprintf('    Categories: %s\n', strjoin(rej_labels, ', '));
        EEG_out.etc.step3_meta.icaweights_pre_reject = EEG_seg.icaweights;
        EEG_out.etc.step3_meta.icasphere_pre_reject  = EEG_seg.icasphere;
        EEG_out = pop_subcomp(EEG_out, rej_idx, 0);
        EEG_out = eeg_checkset(EEG_out);
    end

    % ---- Metadata ----
    EEG_out.etc.step3_meta.variant              = 'prep';
    EEG_out.etc.step3_meta.iclabel_score_source = 'step2';
    EEG_out.etc.step3_meta.step2_iclabel_classifications = ic_classes;
    EEG_out.etc.step3_meta.rejected_ics         = rej_idx(:)';
    EEG_out.etc.step3_meta.ic_rejection_applied = true;
    EEG_out.etc.step3_meta.ic_labels            = rej_labels;
    EEG_out.etc.step3_meta.n_ics_rejected       = n_rej;
    EEG_out.etc.step3_meta.artifact_thresh      = ARTIFACT_THRESH;
    EEG_out.etc.step3_meta.n_asr_segments       = n_asr_out;
    EEG_out.etc.step3_meta.asr_log              = asr_log_out;
    EEG_out.etc.step3_meta.step2_meta           = meta;
    EEG_out.etc.step3_meta.bp_out_lo            = BP_LO;
    EEG_out.etc.step3_meta.bp_out_hi            = BP_HI;
    EEG_out.etc.step3_meta.pipeline = sprintf( ...
        'PREP: step1(01_align,250Hz)->BP(%g-%g)->trim->badch interp->+FCz[%d]->CAR->ASR(k=%g)->runica transfer(MI+Walk+Rest)->ICLabel[step2](artifact>%.1f, apply=1)+manual_ocular_override', ...
        BP_LO, BP_HI, double(restore_fcz), ASR_K, ARTIFACT_THRESH);

    pop_saveset(EEG_out, 'filename', [label '_' OUTPUT_STAGE '.set'], ...
        'filepath', out_dir, 'savemode', 'onefile');
    fprintf('\n[done] %s step3 -> %s\n', label, file_step3);

    out = struct('set_path', file_step3, 'status', 'ok', 'n_rej', n_rej);
end


% ======================== helpers ========================

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
        error('prep_step3:chanMismatch', ...
            'Channel mismatch (%d matched vs %d expected). Missing: {%s}. Extra: {%s}', ...
            length(reorder_idx), EEG.nbchan, strjoin(missing, ', '), strjoin(extra, ', '));
    end
end

function EEG = add_fcz_zerofill(EEG)
    EEG.nbchan = EEG.nbchan + 1;
    EEG.data(end+1, :) = zeros(1, EEG.pnts);
    EEG.chanlocs(end+1).labels = 'FCz';
    dipfit_dir = fileparts(which('pop_dipfit_settings'));
    eloc_file = fullfile(dipfit_dir, 'standard_BEM', 'elec', 'standard_1005.elc');
    EEG = pop_chanedit(EEG, 'lookup', eloc_file);
    EEG = eeg_checkset(EEG);
end

function ok = passes_r1_mode(mode, evt_types, evt_lats, sl, nxt, srate)
    if isempty(mode), ok = true; return; end
    has_r1 = has_coincident_r1(evt_types, evt_lats, sl, nxt, srate);
    switch mode
        case 'with_r1',    ok = has_r1;
        case 'without_r1', ok = ~has_r1;
        otherwise, error('prep_step3:badR1Mode', 'Unknown R1 mode: %s', mode);
    end
end

function tf = has_coincident_r1(evt_types, evt_lats, sl, nxt, srate)
    is_r1 = cellfun(@(t) ~isempty(regexp(strtrim(t), '^R\s*1$', 'once')), evt_types);
    r1_lats = evt_lats(is_r1);
    tol = max(2, round(0.02 * srate));
    tf = any(abs(r1_lats - sl) <= tol) || any(abs(r1_lats - nxt) <= tol);
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
                if round(nxt) > EEG.pnts, continue; end
            end
            i1 = max(1, round(sl));
            i2 = min(EEG.pnts, round(nxt));
            dur = (i2 - i1) / srate;
            if dur < 3 || dur > 300, continue; end
            if any(boundary_lats > sl & boundary_lats < nxt), continue; end
            if ~passes_r1_mode(r1_mode, evt_types, evt_lats, sl, nxt, srate), continue; end

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
                    'dur_sec', dur, 'status', 'ok', 'error_msg', '');   %#ok<AGROW>
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

function ics = read_manual_eye_overrides(label, n_ic)
% READ_MANUAL_EYE_OVERRIDES  IC indices to force-reject as ocular for a session.
%   Reads manual_eye_ic_overrides.csv (co-located with prep_step3.m); returns the
%   ic_index list where session==label, clamped to integer 1..n_ic. Empty if the
%   file or session is absent. These are visually-confirmed ocular ICs (lateral /
%   horizontal saccades) that ICLabel scored below ARTIFACT_THRESH.
    ics = [];
    csv = fullfile(fileparts(mfilename('fullpath')), 'manual_eye_ic_overrides.csv');
    if ~exist(csv, 'file')
        error('prep_step3:missingOverrides', ...
              ['manual_eye_ic_overrides.csv is missing from %s. It is part of the ' ...
               'preprocessing and must be present: without it the manually confirmed ' ...
               'ocular components are not rejected.'], fileparts(csv));
    end
    T = readtable(csv, 'TextType', 'string');
    if isempty(T) || ~all(ismember({'session','ic_index'}, T.Properties.VariableNames))
        return;
    end
    v = double(T.ic_index(strcmp(string(T.session), string(label))));
    v = v(:)';
    ics = unique(v(v >= 1 & v <= n_ic & v == round(v)));
end
