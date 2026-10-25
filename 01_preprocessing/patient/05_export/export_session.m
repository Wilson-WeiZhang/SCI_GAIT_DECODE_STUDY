function out = export_session(label, varargin)
%EXPORT_SESSION (patient, prep_patient)  Pack one patient session into 3 user-facing .mat.
%
%   Output naming: pat0X_sessYY_{eeg,goni,event}.mat
%   Schemas:
%     eeg.x   [60 × T_eeg]      double   continuous step3 (or step2) EEG @ 250 Hz
%     eeg.s   250
%     eeg.c   chanlocs struct (60 entries)
%     eeg.source_set            'step3' | 'step2'
%     eeg.pipeline_status       'ok' | 'step2_only'
%
%     goni.x  [T_goni × 24]     double   continuous goni @ 250 Hz, canonical 24-ch order
%     goni.s  250
%     goni.c  {24×1} cell of canonical labels
%
%     event(j) struct
%       .label    int 0=MI 1=S2S 2=Walk2min
%       .code     'gaitimagery' | 'sit2stand' | 'gaitexecution'
%       .etstart  1-based @ 250 Hz, into eeg.x (column)
%       .etstop
%       .gtstart  1-based @ 250 Hz, into goni.x (row)
%       .gtstop
%       .qc       'good' | 'bad' | 'skip'

    p = inputParser;
    addParameter(p, 'eeg_dir',  '${GAIT_PROJECT_ROOT}/data/prep_data/patient/eeg', @ischar);
    addParameter(p, 'goni_dir', '${GAIT_PROJECT_ROOT}/data/prep_data/patient/goni', @ischar);
    addParameter(p, 'out_root', '${GAIT_PROJECT_ROOT}/patient_prep', @ischar);
    addParameter(p, 'overwrite', false, @islogical);
    parse(p, varargin{:});
    opt = p.Results;

    prep_patient_paths();

    out_dir    = fullfile(opt.out_root, label);
    eeg_path   = fullfile(out_dir, [label '_eeg.mat']);
    goni_path  = fullfile(out_dir, [label '_goni.mat']);
    event_path = fullfile(out_dir, [label '_event.mat']);

    if exist(event_path, 'file') && ~opt.overwrite
        fprintf('[%s] already exported, skipping\n', label);
        out = struct('label', label, 'status', 'skipped_existing');
        return;
    end

    fprintf('\n================================================================\n');
    fprintf('=== export_session (patient): %s\n', label);
    fprintf('================================================================\n');

    % --- Load step3 (or step2 fallback) ---
    step3_path = fullfile(opt.eeg_dir, [label '_step3.set']);
    step2_path = fullfile(opt.eeg_dir, [label '_step2.set']);
    pipeline_status = 'ok';
    if exist(step3_path, 'file')
        EEG = pop_loadset('filename', [label '_step3.set'], 'filepath', opt.eeg_dir);
        eeg_source = 'step3';
    elseif exist(step2_path, 'file')
        EEG = pop_loadset('filename', [label '_step2.set'], 'filepath', opt.eeg_dir);
        eeg_source = 'step2';
        pipeline_status = 'step2_only';
        fprintf('  no step3; using step2 (pipeline_status=step2_only)\n');
    else
        warning('export_session (prep_patient):noEeg', '%s: no step2/step3', label);
        out = struct('label', label, 'status', 'no_eeg');
        return;
    end
    fprintf('  EEG (%s): %d ch x %d pts @ %d Hz (%.1f min)\n', ...
        eeg_source, EEG.nbchan, EEG.pnts, EEG.srate, EEG.xmax/60);

    % --- Load _goni.mat + reload raw goni for continuous track ---
    goni_mat_path = fullfile(opt.goni_dir, [label '_goni.mat']);
    if ~exist(goni_mat_path, 'file')
        warning('export_session (prep_patient):noGoniMat', '%s: %s missing', label, goni_mat_path);
        out = struct('label', label, 'status', 'no_goni');
        return;
    end
    G = load(goni_mat_path);

    % Locate raw goni via registry (or _goni.mat hint)
    raw_goni_paths = locate_raw_goni(label);
    if isempty(raw_goni_paths)
        warning('export_session (prep_patient):rawGoniMissing', '%s: raw goni not found', label);
        out = struct('label', label, 'status', 'raw_goni_missing');
        return;
    end

    [goni_1k, goni_labels_canon, sub_part_pnts_1k] = load_concat_goni_canonical(raw_goni_paths);
    [goni_250, ~] = downsample_goni(goni_1k, sub_part_pnts_1k, 1000, 250);
    fprintf('  Goni: 1000 Hz [%dx%d] -> 250 Hz [%dx%d]\n', ...
        size(goni_1k,1), size(goni_1k,2), size(goni_250,1), size(goni_250,2));
    clear goni_1k;

    % --- Build event struct (3 conditions) ---
    event = build_events_patient(EEG, G);
    n_total = length(event);
    n_mi   = sum([event.label] == 0);
    n_s2s  = sum([event.label] == 1);
    n_walk = sum([event.label] == 2);
    n_bad  = sum(strcmp({event.qc}, 'bad'));
    n_skip = sum(strcmp({event.qc}, 'skip'));
    fprintf('  Event: %d trials (MI=%d S2S=%d Walk=%d, bad=%d skip=%d)\n', ...
        n_total, n_mi, n_s2s, n_walk, n_bad, n_skip);

    % --- Save ---
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    save_eeg(EEG, eeg_source, pipeline_status, eeg_path);
    save_goni(goni_250, goni_labels_canon, goni_path);
    save(event_path, 'event', '-v7');
    fprintf('  -> %s\n', out_dir);

    out = struct('label', label, 'status', 'ok', ...
                 'n_trials', n_total, 'n_bad', n_bad, ...
                 'eeg_source', eeg_source, 'pipeline_status', pipeline_status);
end


% ============================================================
% Helpers
% ============================================================

function paths = locate_raw_goni(label)
    paths = {};
    try
        reg = build_patient_registry();
        for k = 1:length(reg)
            if strcmp(reg(k).label, label)
                paths = reg(k).goni_paths;
                return;
            end
        end
    catch ME
        warning('export_session (prep_patient): registry lookup failed: %s', ME.message);
    end
end

function labels = canonical_goni_labels()
    labels = { ...
        'LHipS X', 'LHipS Y', ...
        'LKneS X', 'LKneS Y', ...
        'LAnkS X', 'LAnkS Y', ...
        'RHipS X', 'RHipS Y', ...
        'RKneS X', 'RKneS Y', ...
        'RAnkS X', 'RAnkS Y', ...
        'LHipW X', 'LHipW Y', ...
        'LKneW X', 'LKneW Y', ...
        'LAnkW X', 'LAnkW Y', ...
        'RHipW X', 'RHipW Y', ...
        'RKneW X', 'RKneW Y', ...
        'RAnkW X', 'RAnkW Y'};
end

function [goni_1k, goni_labels_canon, sub_part_pnts] = load_concat_goni_canonical(raw_goni_paths)
    if ischar(raw_goni_paths), raw_goni_paths = {raw_goni_paths}; end
    canon = canonical_goni_labels();
    n_ch = length(canon);
    raw_cells = cell(1, length(raw_goni_paths));
    T_files = zeros(1, length(raw_goni_paths));
    for f = 1:length(raw_goni_paths)
        raw_cells{f} = load_goniometer(raw_goni_paths{f});
        T_files(f) = size(raw_cells{f}.data, 1);
    end
    T_total = sum(T_files);
    goni_1k = nan(T_total, n_ch, 'single');
    row0 = 0;
    for f = 1:length(raw_cells)
        Graw = raw_cells{f};
        rows = row0 + (1:T_files(f));
        for c = 1:n_ch
            idx = find(strcmp(Graw.labels, canon{c}), 1);
            if ~isempty(idx)
                goni_1k(rows, c) = single(Graw.data(:, idx));
            end
        end
        row0 = row0 + T_files(f);
    end
    goni_labels_canon = canon;
    sub_part_pnts = T_files;
end

function [goni_250, sub_part_pnts_250] = downsample_goni(goni_1k, sub_part_pnts_1k, fs_in, fs_out)
    [p, q] = rat(fs_out / fs_in);
    [T_in, n_ch] = size(goni_1k);
    T_out = ceil(T_in * p / q);
    goni_250 = nan(T_out, n_ch, 'single');
    for c = 1:n_ch
        col = goni_1k(:, c);
        if all(isnan(col)), continue; end
        if any(isnan(col)), col(isnan(col)) = 0; end
        y = resample(double(col), p, q);
        if length(y) < T_out
            y(end+1:T_out) = y(end);
        end
        goni_250(:, c) = single(y(1:T_out));
    end
    sub_part_pnts_250 = round(sub_part_pnts_1k * p / q);
end

function event = build_events_patient(EEG, G)
% Drive entirely from G.trials in `_goni.mat` so EEG and goni windows
% always describe the same physical trial. Both come from extract_goni_patient,
% which is the single source of truth for trial timing.
%
% G.trials(j).start_eeg_sec / end_eeg_sec — seconds in the **step1/raw EEG
%   timeline** (extract_goni runs on step1.set @ 1000 Hz native). step3 EEG
%   has had ASR drop samples, so its clock is shifted earlier; we calibrate
%   the offset using step3 EEG markers and subtract it before indexing.
% G.trials(j).goni_start_idx / goni_end_idx — 1-based indices at the goni's
%   native 1000 Hz; downsample by 4 for 250 Hz indexing into goni.x.

    % 'rest' (label 3) is the automatic relax break: the task software sends
    % marker 20/21 for it, and the 4-bit Cedrus C-POD port folds those onto 4/5 -- the sit-to-stand
    % codes. See 02_extract_goni/extract_goni_patient.m split_rest_events. Existing codes/labels are
    % unchanged; rest is its own class.
    type_to_meta = struct( ...
        'imagine', struct('code','gaitimagery',   'label', 0), ...
        's2s',     struct('code','sit2stand',     'label', 1), ...
        'walk',    struct('code','gaitexecution', 'label', 2), ...
        'rest',    struct('code','rest',          'label', 3));

    n_trials = length(G.trials);
    if n_trials == 0
        error('export_session (prep_patient):noTrials', 'G.trials is empty');
    end
    if length(G.qc) ~= n_trials
        error('export_session (prep_patient):qcLengthMismatch', ...
            'numel(G.qc)=%d but numel(G.trials)=%d', length(G.qc), n_trials);
    end

    EEG_T_250 = EEG.pnts;   % EEG is at 250 Hz post-step3
    event = struct('label', {}, 'code', {}, 'etstart', {}, 'etstop', {}, ...
                   'gtstart', {}, 'gtstop', {}, 'qc', {});

    % Per-trial step1->step3 alignment. extract_goni_patient wrote
    % start_eeg_sec in step1 clock; ASR drops can be scattered through the
    % session, so a single global offset is insufficient when offsets vary
    % across trials (spread > 20 s in some sessions). Instead, pair
    % each trial with the corresponding step3 EEG.event marker by trial
    % type + sequence position. trial_step3_starts(j) gives the per-trial
    % step3 sample index, or NaN if no matching marker survives in step3.
    [trial_step3_starts, fallback_offset] = build_per_trial_step3_starts(EEG, G);

    for j = 1:n_trials
        T = G.trials(j);
        ttype = T.type;
        if ~isfield(type_to_meta, ttype)
            error('export_session (prep_patient):unknownType', ...
                'trial %d has unknown type "%s"', j, ttype);
        end
        meta = type_to_meta.(ttype);

        if isnan(T.start_eeg_sec) || isnan(T.end_eeg_sec) || ...
           T.start_eeg_sec >= T.end_eeg_sec
            error('export_session (prep_patient):badEegTime', ...
                'trial %d (%s) has bad start/end_eeg_sec [%g, %g]', ...
                j, ttype, T.start_eeg_sec, T.end_eeg_sec);
        end

        % Per-trial step3 start index. Fall back to median offset if marker
        % missing for this trial (e.g. ASR-dropped marker for a short trial).
        dur_samples = round((T.end_eeg_sec - T.start_eeg_sec) * 250);
        if ~isnan(trial_step3_starts(j))
            et_start_raw = trial_step3_starts(j);
        else
            et_start_raw = round(T.start_eeg_sec * 250) + 1 - fallback_offset;
        end
        et_stop_raw = et_start_raw + dur_samples;
        et_start_250 = max(1, et_start_raw);
        et_stop_250  = min(EEG_T_250, et_stop_raw);
        if et_start_raw > EEG_T_250 || et_stop_250 <= et_start_250
            % Trial lies fully past the (post-step3) EEG length — happens
            % when stage 03 trims trailing data while stage 02 wrote the
            % goni trial in the longer step1 timeline. Skip with warning.
            warning('export_session (prep_patient):trialPastEegEnd', ...
                'trial %d (%s) start=%.2fs > EEG length %.2fs; skipping', ...
                j, ttype, T.start_eeg_sec, EEG_T_250 / 250);
            continue;
        end
        if et_stop_raw > EEG_T_250
            warning('export_session (prep_patient):trialClippedAtEegEnd', ...
                'trial %d (%s) end=%.2fs clipped to EEG end %.2fs', ...
                j, ttype, T.end_eeg_sec, EEG_T_250 / 250);
        end

        gt_start_250 = max(1, round((T.goni_start_idx - 1) / 4) + 1);
        gt_stop_250  = max(gt_start_250 + 1, round((T.goni_end_idx - 1) / 4) + 1);

        switch G.qc(j).status
            case 'ok',      qc_str = 'good';
            case 'problem', qc_str = 'bad';
            case 'skip',    qc_str = 'skip';
            otherwise,      qc_str = 'skip';
        end

        % Auto-downgrade qc to 'bad' when EEG and goni windows for the
        % same trial disagree on duration. Affects two-marker-protocol and multi-vhdr
        % sessions where stage-02 marker pairing can yield over-long
        % EEG spans (e.g. pat01_sess01 trial 24: et 175s vs gt 60s).
        DUR_TOL_SEC = 0.050;
        et_sec = (et_stop_250 - et_start_250) / 250;
        gt_sec = (gt_stop_250 - gt_start_250) / 250;
        if abs(et_sec - gt_sec) > DUR_TOL_SEC && strcmp(qc_str, 'good')
            warning('export_session (prep_patient):durationMismatch', ...
                'trial %d (%s) et=%.2fs gt=%.2fs diff=%+.2fs > %.1fs; qc->bad', ...
                j, ttype, et_sec, gt_sec, gt_sec-et_sec, DUR_TOL_SEC);
            qc_str = 'bad';
        end

        event(end+1).label = double(meta.label); %#ok<AGROW>
        event(end).code    = meta.code;
        event(end).etstart = double(et_start_250);
        event(end).etstop  = double(et_stop_250);
        event(end).gtstart = double(gt_start_250);
        event(end).gtstop  = double(gt_stop_250);
        event(end).qc      = qc_str;
    end
end

function [trial_starts, fallback_offset] = build_per_trial_step3_starts(EEG, G)
% Build per-trial step3 EEG sample index by pairing each G.trial with the
% k-th step3 EEG.event marker of its trial type. ASR can drop samples
% scattered through the recording (boundary events), so the step1->step3
% offset varies trial-by-trial; using per-trial markers avoids the
% spread-out median-offset failure mode.
%
% Default marker map per trial type:
%   imagine -> 'S  1' (S1 = imagine start; both standard + legacy)
%   s2s     -> 'S  4' (standard only; legacy has no S2S trials)
%   walk    -> 'S  7' unless G.trials(j).start_marker records the actual start
%              marker (later sessions use S6->S7 for the 2-min walk)
%
% Returns:
%   trial_starts(j)  = step3 sample index for G.trials(j) start, or NaN if
%                      no surviving step3 marker for this trial position.
%   fallback_offset  = median(step1_predicted - step3_marker) across all
%                      successfully paired trials, used by build_events as a
%                      fallback for trials with NaN trial_starts.

    fs = EEG.srate;
    if abs(fs - 250) > 1e-6
        error('export_session (prep_patient):badSrate', 'Expected 250 Hz step3 EEG, got %.6g', fs);
    end

    type_marker_map = struct( ...
        'imagine', 'S  1', ...
        's2s',     'S  4', ...
        'walk',    'S  7');

    evt_types = cell(1, length(EEG.event));
    evt_lats  = nan(1, length(EEG.event));
    for k = 1:length(EEG.event)
        evt_types{k} = strtrim(char(EEG.event(k).type));
        evt_lats(k)  = EEG.event(k).latency;
    end

    n_trials = numel(G.trials);
    trial_starts = nan(n_trials, 1);
    all_offsets = [];   % for fallback median + diagnostic
    summary = struct();

    trial_markers = cell(n_trials, 1);
    for j = 1:n_trials
        ttype = G.trials(j).type;
        if isfield(G.trials, 'start_marker') && ~isempty(G.trials(j).start_marker)
            trial_markers{j} = strtrim(char(G.trials(j).start_marker));
        elseif isfield(type_marker_map, ttype)
            trial_markers{j} = strtrim(type_marker_map.(ttype));
        else
            trial_markers{j} = '';
        end
    end

    marker_names = unique(trial_markers(~cellfun(@isempty, trial_markers)));
    for ti = 1:numel(marker_names)
        marker_code = marker_names{ti};
        tr_idx = find(strcmp(trial_markers, marker_code));
        marker_lats = evt_lats(strcmp(evt_types, marker_code));
        n_pair = min(numel(tr_idx), numel(marker_lats));
        for k = 1:n_pair
            mlat = round(marker_lats(k));
            trial_starts(tr_idx(k)) = mlat;
            raw_step3 = round(G.trials(tr_idx(k)).start_eeg_sec * fs) + 1;
            all_offsets(end+1) = raw_step3 - mlat; %#ok<AGROW>
        end
        key = regexprep(marker_code, '\s+', '');
        summary.(key) = struct('marker', marker_code, ...
                               'n_trials', numel(tr_idx), ...
                               'n_markers', numel(marker_lats), ...
                               'n_paired', n_pair);
    end

    n_unpaired = sum(isnan(trial_starts));
    if isempty(all_offsets)
        warning('export_session (prep_patient):noTimebaseOffset', ...
            'no EEG marker pairs found for any trial type; using offset=0 fallback');
        fallback_offset = 0;
    else
        fallback_offset = round(median(all_offsets));
    end

    if isempty(all_offsets)
        spread = 0;
    else
        spread = max(all_offsets) - min(all_offsets);
    end
    fprintf('  Timebase: per-trial pairing — paired %d/%d trials, fallback offset=%d (%.3fs), per-trial offset spread=%d samples (%.3fs)\n', ...
        n_trials - n_unpaired, n_trials, fallback_offset, fallback_offset / fs, spread, spread / fs);
    for ti = 1:numel(marker_names)
        key = regexprep(marker_names{ti}, '\s+', '');
        s = summary.(key);
        if s.n_trials > 0
            unpaired_str = '';
            if s.n_paired < s.n_trials
                unpaired_str = sprintf(' (%d unpaired -> fallback)', s.n_trials - s.n_paired);
            end
            fprintf('    %s: %d trials, %d markers, %d paired%s\n', ...
                s.marker, s.n_trials, s.n_markers, s.n_paired, unpaired_str);
        end
    end
end


function save_eeg(EEG, eeg_source, pipeline_status, out_path)
    eeg = struct();
    eeg.x = double(EEG.data);
    eeg.s = double(EEG.srate);
    eeg.c = EEG.chanlocs;
    eeg.source_set = eeg_source;
    eeg.pipeline_status = pipeline_status;
    save(out_path, 'eeg', '-v7.3');
end

function save_goni(goni_250, goni_labels, out_path)
    goni = struct();
    goni.x = double(goni_250);
    goni.s = double(250);
    goni.c = goni_labels(:);
    save(out_path, 'goni', '-v7.3');
end
