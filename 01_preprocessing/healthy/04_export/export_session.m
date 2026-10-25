function out = export_session(session_id, varargin)
%EXPORT_SESSION  Pack one healthy session into 3 user-facing .mat files.
%
%   Output naming: subjNN_sessMM[_abc]_{eeg,goni,event}.mat
%     - "subjNN_sessMM"        single-part session (e.g. subj01_sess01)
%     - "subjNN_sessMM_abc"    multi-part merged session (suffix = sub-parts joined)
%       (e.g. SUB_28_sess01 with sub-parts a/b/c → subj28_sess01)
%
%   Each .mat is a single struct (eeg / goni / event) so users load it as
%       S = load('subj28_sess01_abc_eeg.mat'); data = S.eeg.x; ...
%
%   Schemas:
%     eeg.x   [60 × T_eeg]   double   continuous post-IClabel EEG @ 250 Hz
%     eeg.s   250
%     eeg.c   chanlocs struct (60 entries, with .labels and 1005 coords)
%
%     goni.x  [T_goni × 24]  double   continuous goni RESAMPLED to 250 Hz,
%                                     channels in fixed canonical order
%                                     (missing channels = NaN columns)
%     goni.s  250
%     goni.c  {24×1} cell of canonical labels (column cell from labels(:))
%
%     event(j) struct
%       .label    int 0=imag, 1=real, 2=rest
%       .code     'gaitimagery' | 'gaitexecution' | 'rest'
%       .etstart  1-based @ 250 Hz, into eeg.x (column)
%       .etstop
%       .gtstart  1-based @ 250 Hz, into goni.x (row)
%       .gtstop
%       .qc       'good' | 'bad' | 'skip'   (rest defaults to 'good')
%       .eeg_pre1s_mean [60×1] channel-wise mean in the 1 s before etstart
%       .eeg_pre1s_std  [60×1] channel-wise std in the 1 s before etstart
%       .eeg_pre1s_n    number of baseline samples actually used

    p = inputParser;
    addParameter(p, 'eeg_dir',  '${GAIT_PROJECT_ROOT}/data/prep_data/healthy/eeg', @ischar);
    addParameter(p, 'goni_dir', '${GAIT_PROJECT_ROOT}/data/prep_data/healthy/goni', @ischar);
    addParameter(p, 'raw_root', '${GAIT_PROJECT_ROOT}/data/raw_data', @ischar);
    addParameter(p, 'out_root', '${GAIT_PROJECT_ROOT}/data/healthy_prep', @ischar);
    addParameter(p, 'eeg_stage', 'auto', @(s) ischar(s) || isstring(s));
    addParameter(p, 'overwrite', false, @islogical);
    addParameter(p, 'refresh_stale', false, @islogical);
    parse(p, varargin{:});
    opt = p.Results;
    opt.eeg_stage = lower(char(opt.eeg_stage));
    assert(any(strcmp(opt.eeg_stage, {'auto','step2','step3','step3_noic','step3_ic09','step3_ic208','step3_ic308','step3_norw'})), ...
        'eeg_stage must be auto, step2, step3, step3_noic, step3_ic09, step3_ic208, step3_ic308, or step3_norw');

    % --- Detect sub-parts and compute output prefix ---
    parts = list_sub_parts(session_id, opt.eeg_dir);
    if isempty(parts)
        part_labels = {session_id};
        suffix = '';
    else
        part_labels = parts;
        letters = cellfun(@(s) s(end), parts);   % 'a', 'b', 'c'
        suffix = ['_' letters(:)'];
    end
    prefix = [to_subj_prefix(session_id) suffix];      % e.g. subj01_sess01 or subj28_sess01
    out_dir = fullfile(opt.out_root, prefix);
    eeg_path   = fullfile(out_dir, [prefix '_eeg.mat']);
    goni_path  = fullfile(out_dir, [prefix '_goni.mat']);
    event_path = fullfile(out_dir, [prefix '_event.mat']);

    if exist(event_path, 'file') && ~opt.overwrite
        if is_stale_event_export(event_path)
            if opt.refresh_stale
                fprintf('[%s] existing event export has old schema; regenerating\n', prefix);
            else
                fprintf('[%s] existing event export has old schema; skipping (set refresh_stale=true or overwrite=true to rewrite)\n', prefix);
                out = struct('session_id', session_id, 'prefix', prefix, 'status', 'stale_existing');
                return;
            end
        else
            fprintf('[%s] already exported, skipping\n', prefix);
            out = struct('session_id', session_id, 'prefix', prefix, 'status', 'skipped_existing');
            return;
        end
    end

    fprintf('\n================================================================\n');
    fprintf('=== export_session: %s → %s ===\n', session_id, prefix);
    fprintf('================================================================\n');
    fprintf('  sub-parts: %d (%s)\n', length(part_labels), strjoin(part_labels, ', '));

    % --- Load merged step3.set (or step2 fallback) ---
    step3_path = fullfile(opt.eeg_dir, [session_id '_step3.set']);
    step3_noic_path = fullfile(opt.eeg_dir, [session_id '_step3_noic.set']);
    step3_ic09_path = fullfile(opt.eeg_dir, [session_id '_step3_ic09.set']);
    step2_path = fullfile(opt.eeg_dir, [session_id '_step2.set']);
    if strcmp(opt.eeg_stage, 'step3')
        if ~exist(step3_path, 'file')
            warning('export_session:noStep3', '%s: no step3 -> skip', session_id);
            out = struct('session_id', session_id, 'prefix', prefix, 'status', 'no_step3');
            return;
        end
        EEG = pop_loadset('filename', [session_id '_step3.set'], 'filepath', opt.eeg_dir);
        eeg_source = 'step3';
    elseif strcmp(opt.eeg_stage, 'step3_noic')
        if ~exist(step3_noic_path, 'file')
            warning('export_session:noStep3Noic', '%s: no step3_noic -> skip', session_id);
            out = struct('session_id', session_id, 'prefix', prefix, 'status', 'no_step3_noic');
            return;
        end
        EEG = pop_loadset('filename', [session_id '_step3_noic.set'], 'filepath', opt.eeg_dir);
        eeg_source = 'step3_no_ic_rejection';
    elseif strcmp(opt.eeg_stage, 'step3_ic09')
        if ~exist(step3_ic09_path, 'file')
            warning('export_session:noStep3Ic09', '%s: no step3_ic09 -> skip', session_id);
            out = struct('session_id', session_id, 'prefix', prefix, 'status', 'no_step3_ic09');
            return;
        end
        EEG = pop_loadset('filename', [session_id '_step3_ic09.set'], 'filepath', opt.eeg_dir);
        eeg_source = 'step3_iclabel_0p9';
    elseif strcmp(opt.eeg_stage, 'step2')
        if ~exist(step2_path, 'file')
            warning('export_session:noStep2', '%s: no step2 -> skip', session_id);
            out = struct('session_id', session_id, 'prefix', prefix, 'status', 'no_step2');
            return;
        end
        EEG = pop_loadset('filename', [session_id '_step2.set'], 'filepath', opt.eeg_dir);
        eeg_source = 'step2_no_ic_rejection';
    elseif any(strcmp(opt.eeg_stage, {'step3_ic208','step3_ic308','step3_norw'}))
        variant_path = fullfile(opt.eeg_dir, [session_id '_' opt.eeg_stage '.set']);
        if ~exist(variant_path, 'file')
            warning('export_session:noStageVariant', '%s: no %s -> skip', session_id, opt.eeg_stage);
            out = struct('session_id', session_id, 'prefix', prefix, 'status', ['no_' opt.eeg_stage]);
            return;
        end
        EEG = pop_loadset('filename', [session_id '_' opt.eeg_stage '.set'], 'filepath', opt.eeg_dir);
        eeg_source = opt.eeg_stage;
    elseif exist(step3_path, 'file')
        EEG = pop_loadset('filename', [session_id '_step3.set'], 'filepath', opt.eeg_dir);
        eeg_source = 'step3';
    elseif exist(step2_path, 'file')
        warning('export_session:noStep3', '%s: no step3, falling back to step2', session_id);
        EEG = pop_loadset('filename', [session_id '_step2.set'], 'filepath', opt.eeg_dir);
        eeg_source = 'step2';
    else
        warning('export_session:noEeg', '%s: no step2/step3 → skip', session_id);
        out = struct('session_id', session_id, 'prefix', prefix, 'status', 'no_eeg');
        return;
    end
    fprintf('  EEG (%s): %d ch x %d pts @ %d Hz (%.1f min)\n', ...
        eeg_source, EEG.nbchan, EEG.pnts, EEG.srate, EEG.xmax/60);

    % --- Load + concat goni at 1000 Hz, then resample to 250 Hz ---
    [goni_1k, goni_labels, sub_part_pnts_1k, all_trials_1k, all_qc, source_part_arr] = ...
        load_concat_goni(session_id, part_labels, opt);
    if isempty(goni_1k)
        warning('export_session:noGoni', '%s: no goni → skip goni+event', session_id);
        if ~exist(out_dir, 'dir'), mkdir(out_dir); end
        save_eeg(EEG, eeg_source, eeg_path);
        out = struct('session_id', session_id, 'prefix', prefix, 'status', 'no_goni');
        return;
    end
    [goni_250, sub_part_pnts_250] = downsample_goni(goni_1k, sub_part_pnts_1k, 1000, 250);
    fprintf('  Goni: 1000 Hz [%d×%d] → 250 Hz [%d×%d]; sub_part_pnts_250=[%s]\n', ...
        size(goni_1k,1), size(goni_1k,2), size(goni_250,1), size(goni_250,2), num2str(sub_part_pnts_250));
    clear goni_1k;

    % --- Build event struct (250 Hz coordinates for both EEG and goni) ---
    event = build_events(EEG, all_trials_1k, all_qc, source_part_arr, ...
        length(part_labels), session_id);
    event = add_pretrial_eeg_baseline(event, EEG, 1.0);
    n_total = length(event);
    n_imag = sum([event.label] == 0);   % gaitimagery
    n_real = sum([event.label] == 1);   % gaitexecution
    n_rest = sum([event.label] == 2);   % rest
    n_bad  = sum(strcmp({event.qc}, 'bad'));
    n_skip = sum(strcmp({event.qc}, 'skip'));
    fprintf('  Event: %d trials (imag=%d real=%d rest=%d, bad=%d skip=%d)\n', ...
        n_total, n_imag, n_real, n_rest, n_bad, n_skip);

    % --- Save ---
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    save_eeg(EEG, eeg_source, eeg_path);
    save_goni(goni_250, goni_labels, goni_path);
    save(event_path, 'event', '-v7');
    fprintf('  → %s\n', out_dir);

    out = struct('session_id', session_id, 'prefix', prefix, ...
                 'status', 'ok', 'n_trials', n_total, ...
                 'n_bad', n_bad, 'eeg_source', eeg_source);
end


% ============================================================
% Helpers
% ============================================================

function tf = is_stale_event_export(event_path)
    tf = true;
    try
        S = load(event_path, 'event');
        if ~isfield(S, 'event') || isempty(S.event), return; end
        ev = S.event;
        required_fields = {'eeg_pre1s_mean','eeg_pre1s_std','eeg_pre1s_n'};
        if any(~isfield(ev, required_fields))
            return;
        end
        for k = 1:length(ev)
            code = char(ev(k).code);
            lbl  = double(ev(k).label);
            if (strcmp(code, 'gaitimagery')   && lbl ~= 0) || ...
               (strcmp(code, 'gaitexecution') && lbl ~= 1) || ...
               (strcmp(code, 'rest')          && lbl ~= 2)
                return;
            end
        end
        tf = false;
    catch
        tf = true;
    end
end


function event = add_pretrial_eeg_baseline(event, EEG, baseline_sec)
% Store channel-wise EEG mean/std from the 1 s immediately before each trial.
% The window is clamped to the previous exported event boundary and to the
% recording start; when a full pre-trial second is unavailable, n records the
% shorter clean window and mean/std are NaN if no samples exist.

    if isempty(event)
        return;
    end
    fs = double(EEG.srate);
    n_base = round(baseline_sec * fs);
    n_ch = double(EEG.nbchan);
    x = double(EEG.data);
    [~, order] = sort([event.etstart]);
    prev_stop = 0;

    for oi = 1:numel(order)
        j = order(oi);
        etstart = round(double(event(j).etstart));
        stop_idx = etstart - 1;
        start_idx = max(etstart - n_base, prev_stop + 1);
        start_idx = max(start_idx, 1);
        stop_idx = min(stop_idx, size(x, 2));

        if stop_idx >= start_idx
            seg = x(:, start_idx:stop_idx);
            event(j).eeg_pre1s_mean = mean(seg, 2, 'omitnan');
            if size(seg, 2) >= 2
                event(j).eeg_pre1s_std = std(seg, 0, 2, 'omitnan');
            else
                event(j).eeg_pre1s_std = nan(n_ch, 1);
            end
            event(j).eeg_pre1s_n = double(size(seg, 2));
            event(j).eeg_pre1s_window = double([start_idx, stop_idx]);
        else
            event(j).eeg_pre1s_mean = nan(n_ch, 1);
            event(j).eeg_pre1s_std = nan(n_ch, 1);
            event(j).eeg_pre1s_n = double(0);
            event(j).eeg_pre1s_window = double([nan, nan]);
        end

        prev_stop = max(prev_stop, round(double(event(j).etstop)));
    end
end


function p = to_subj_prefix(session_id)
% 'SUB_01_sess02' → 'subj01_sess02'
    tok = regexp(session_id, '^SUB_(\d+)_sess(\d+)$', 'tokens', 'once');
    if isempty(tok)
        p = lower(session_id);
    else
        p = sprintf('subj%s_sess%s', tok{1}, tok{2});
    end
end


function parts = list_sub_parts(sid, eeg_dir)
% Return sub-part labels {'SUB_28_sess01a', 'b', 'c'} or {} if single-part.
% Sub-parts whose `_goni.mat` carries `align_info{1}.verdict = 'FAIL'` are skipped.
    re = sprintf('^%s([a-z])_step1\\.set$', regexptranslate('escape', sid));
    files = dir(fullfile(eeg_dir, [sid '*_step1.set']));
    parts = {};
    for i = 1:length(files)
        tok = regexp(files(i).name, re, 'tokens', 'once');
        if ~isempty(tok)
            parts{end+1} = [sid tok{1}]; %#ok<AGROW>
        end
    end
    parts = sort(parts);

    % Filter out align-FAIL parts
    goni_dir = strrep(eeg_dir, '/eeg', '/goni');
    keep = true(size(parts));
    for i = 1:length(parts)
        gp = fullfile(goni_dir, [parts{i} '_goni.mat']);
        if ~exist(gp, 'file'), continue; end
        try
            tmp = load(gp, 'align_info');
            if isfield(tmp,'align_info') && ~isempty(tmp.align_info) && ...
               isfield(tmp.align_info{1},'verdict') && ...
               strcmp(tmp.align_info{1}.verdict, 'FAIL')
                keep(i) = false;
                fprintf('  list_sub_parts: skip %s (align verdict=FAIL)\n', parts{i});
            end
        catch
        end
    end
    parts = parts(keep);
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


function [goni_data, goni_labels, sub_part_pnts, all_trials, all_qc, source_part_arr] = ...
        load_concat_goni(sid, part_labels, opt) %#ok<INUSL>
% Concat each sub-part's raw goni @ 1000 Hz into a single matrix, mapped to
% canonical 24-ch order (NaN for missing). Trial metadata kept in sub-part-
% local coordinates plus a cumulative offset so build_events can place them
% in the merged 1000 Hz timeline (later resampled to 250 Hz).

    canonical_labels = canonical_goni_labels();
    n_ch = length(canonical_labels);

    goni_data = [];
    goni_labels = canonical_labels;
    sub_part_pnts = zeros(1, length(part_labels));
    all_trials = struct([]);
    all_qc = struct([]);
    source_part_arr = [];

    sessions = build_session_registry(opt.raw_root);
    reg_map = containers.Map({sessions.label}, num2cell(1:length(sessions)));

    cum_off = 0;
    cum_eeg_sec = 0;   % cumulative EEG length (seconds @ 1000 Hz native) for merged-time conversion
    for p = 1:length(part_labels)
        label = part_labels{p};
        goni_mat = fullfile(opt.goni_dir, [label '_goni.mat']);
        if ~exist(goni_mat, 'file')
            error('export_session:noGoniMat', ...
                '%s: missing %s; refusing partial multi-part export', label, goni_mat);
        end
        G = load(goni_mat);

        % Per-sub-part EEG length in seconds, used to project per-sub-part
        % start/end_eeg_sec into the merged step3 timeline. step1.set is at
        % 1000 Hz native (extract_goni computes start_eeg_sec at this rate);
        % we divide by EEG_p_info.srate, not a hardcoded constant.
        EEG_p_info = pop_loadset('filename', [label '_step1.set'], ...
            'filepath', opt.eeg_dir, 'loadmode', 'info');
        eeg_part_dur_sec = EEG_p_info.pnts / EEG_p_info.srate;

        if ~isKey(reg_map, label)
            error('export_session:notInRegistry', '%s: not in active registry', label);
        end
        s = sessions(reg_map(label));
        if ~isempty(s.goni_paths_multi)
            raw_goni_paths = s.goni_paths_multi;
        elseif ~isempty(s.goni_path_override)
            raw_goni_paths = {s.goni_path_override};
        else
            [eeg_dir, eeg_name, ~] = fileparts(s.vhdr);
            raw_goni_paths = {find_goni_file(eeg_dir, eeg_name)};
        end
        if isempty(raw_goni_paths) || any(cellfun(@(p) isempty(p) || ~exist(p, 'file'), raw_goni_paths))
            error('export_session:rawGoniMissing', '%s: raw goni not found (vhdr=%s)', label, s.vhdr);
        end

        raw_cells = cell(1, length(raw_goni_paths));
        T_files = zeros(1, length(raw_goni_paths));
        for f = 1:length(raw_goni_paths)
            raw_cells{f} = load_goniometer(raw_goni_paths{f});
            T_files(f) = size(raw_cells{f}.data, 1);
        end
        T_part = sum(T_files);
        sub_part_pnts(p) = T_part;
        part_data = nan(T_part, n_ch, 'single');
        row0 = 0;
        for f = 1:length(raw_cells)
            Graw = raw_cells{f};
            rows = row0 + (1:T_files(f));
            for c = 1:n_ch
                idx = find(strcmp(Graw.labels, canonical_labels{c}), 1);
                if ~isempty(idx)
                    part_data(rows, c) = single(Graw.data(:, idx));
                end
            end
            row0 = row0 + T_files(f);
        end
        goni_data = [goni_data; part_data]; %#ok<AGROW>
        clear raw_cells Graw part_data;

        n_trials_old = length(all_trials);
        for j = 1:length(G.trials)
            t = G.trials(j);
            t.goni_start_idx_1k = t.goni_start_idx + cum_off;
            t.goni_end_idx_1k   = t.goni_end_idx   + cum_off;
            % Project per-sub-part start/end_eeg_sec into the merged step3 timeline
            % (extract_goni recorded these at the per-sub-part 1000 Hz EEG; merge
            % adds cum_eeg_sec of all preceding sub-parts).
            t.start_eeg_sec_merged = t.start_eeg_sec + cum_eeg_sec;
            t.end_eeg_sec_merged   = t.end_eeg_sec   + cum_eeg_sec;
            t.source_part       = p;
            t = rmfield(t, {'joint_angles', 'goni_labels', 'goni_start_idx', 'goni_end_idx'});
            if isempty(all_trials), all_trials = t;
            else
                [all_trials, t] = normalize_struct_fields(all_trials, t);
                all_trials(end+1) = t; %#ok<AGROW>
            end
        end
        for j = 1:length(G.qc)
            q = G.qc(j);
            q.trial_idx = j + n_trials_old;
            if isempty(all_qc)
                all_qc = q;
            else
                % Across sub-parts (esp. multi_goni_file vs standard) qc
                % struct fieldnames can differ; normalize before assignment.
                [all_qc, q] = normalize_struct_fields(all_qc, q);
                all_qc(end+1) = q; %#ok<AGROW>
            end
        end
        source_part_arr = [source_part_arr, repmat(p, 1, length(G.trials))]; %#ok<AGROW>
        cum_off = cum_off + T_part;
        cum_eeg_sec = cum_eeg_sec + eeg_part_dur_sec;
    end
end


function [goni_250, sub_part_pnts_250] = downsample_goni(goni_1k, sub_part_pnts_1k, fs_in, fs_out)
% Per-column resample, preserving fully-NaN columns as NaN.
    [p, q] = rat(fs_out / fs_in);
    [T_in, n_ch] = size(goni_1k);
    T_out = ceil(T_in * p / q);
    goni_250 = nan(T_out, n_ch, 'single');
    for c = 1:n_ch
        col = goni_1k(:, c);
        if all(isnan(col)), continue; end   % keep all-NaN column
        % If a column has scattered NaN (rare), replace with 0 before resample
        if any(isnan(col)), col(isnan(col)) = 0; end
        y = resample(double(col), p, q);
        if length(y) < T_out
            y(end+1:T_out) = y(end);
        end
        goni_250(:, c) = single(y(1:T_out));
    end
    sub_part_pnts_250 = round(sub_part_pnts_1k * p / q);
end


function event = build_events(EEG, all_trials_1k, all_qc, source_part_arr, ...
        n_parts, session_id) %#ok<INUSD>
% Drive event(j) entirely from G.trials (concatenated as all_trials_1k by
% load_concat_goni). Stage 02 extract_goni is the single source of truth for
% trial timing — both EEG and goni windows for a given trial come from the
% same trial record, so they always describe the same physical trial.
%
% all_trials_1k(j).start_eeg_sec_merged / end_eeg_sec_merged — already in
%   merged step1/raw EEG timeline (load_concat_goni adds cumulative sub-part
%   EEG length to each trial's per-sub-part start_eeg_sec). Step3 can be
%   shorter than step1 after ASR boundary rejection, so the step1/raw clock
%   must be shifted into step3's marker clock before indexing EEG.data.
% all_trials_1k(j).goni_start_idx_1k / end_idx_1k — concatenated goni 1000 Hz
%   indices.
% Public label contract: 0=imagine, 1=walk (gaitexecution), 2=rest.

    type_to_meta = struct( ...
        'imagine', struct('code','gaitimagery',   'label', 0), ...
        'walk',    struct('code','gaitexecution', 'label', 1), ...
        'rest',    struct('code','rest',          'label', 2));

    n_trials = length(all_trials_1k);
    event = struct('label',{},'code',{},'etstart',{},'etstop',{}, ...
                   'gtstart',{},'gtstop',{},'qc',{});
    EEG_T_250 = EEG.pnts;
    DUR_TOL_SEC = 1.0;
    if isempty(source_part_arr)
        source_part_arr = ones(1, n_trials);
    end
    step3_offsets = estimate_step3_timebase_offsets(EEG, all_trials_1k, ...
        source_part_arr, n_parts, session_id);

    for j = 1:n_trials
        t = all_trials_1k(j);
        if ~isfield(type_to_meta, t.type)
            error('export_session:unknownType', ...
                'trial %d has unknown type "%s"', j, t.type);
        end
        meta = type_to_meta.(t.type);

        if ~isfield(t, 'start_eeg_sec_merged') || ...
           isnan(t.start_eeg_sec_merged) || isnan(t.end_eeg_sec_merged) || ...
           t.start_eeg_sec_merged >= t.end_eeg_sec_merged
            error('export_session:badEegTime', ...
                'trial %d (%s) bad start/end_eeg_sec_merged', j, t.type);
        end

        src_part = source_part_arr(min(j, numel(source_part_arr)));
        if isnan(src_part) || src_part < 1 || src_part > numel(step3_offsets)
            src_part = 1;
        end
        step3_offset = step3_offsets(src_part);

        et_start_raw = round(t.start_eeg_sec_merged * 250) + 1 - step3_offset;
        et_stop_raw  = round(t.end_eeg_sec_merged   * 250) + 1 - step3_offset;
        et_start_250 = max(1, et_start_raw);
        et_stop_250  = min(EEG_T_250, et_stop_raw);
        if et_start_raw > EEG_T_250 || et_stop_250 <= et_start_250
            warning('export_session:trialPastEegEnd', ...
                'trial %d (%s) start=%.2fs > EEG length %.2fs; skipping', ...
                j, t.type, t.start_eeg_sec_merged, EEG_T_250 / 250);
            continue;
        end

        % goni 1000 Hz idx → 250 Hz idx (factor 4)
        gt_start_250 = max(1, round((t.goni_start_idx_1k - 1) / 4) + 1);
        gt_stop_250  = max(gt_start_250 + 1, round((t.goni_end_idx_1k - 1) / 4) + 1);

        % qc mapping. rest defaults to 'good' (no active goni QC done).
        if strcmp(t.type, 'rest')
            qc_str = 'good';
        else
            switch all_qc(j).status
                case 'ok',      qc_str = 'good';
                case 'problem', qc_str = 'bad';
                case 'skip',    qc_str = 'skip';
                otherwise,      qc_str = 'skip';
            end
        end

        % Auto-downgrade qc to 'bad' when EEG and goni windows for the same
        % trial disagree on duration (e.g. multi-part boundary edge cases).
        et_sec = (et_stop_250 - et_start_250) / 250;
        gt_sec = (gt_stop_250 - gt_start_250) / 250;
        if abs(et_sec - gt_sec) > DUR_TOL_SEC && strcmp(qc_str, 'good')
            warning('export_session:durationMismatch', ...
                'trial %d (%s) et=%.2fs gt=%.2fs diff=%+.2fs > %.1fs; qc->bad', ...
                j, t.type, et_sec, gt_sec, gt_sec-et_sec, DUR_TOL_SEC);
            qc_str = 'bad';
        end

        % All numeric fields stored as double to match example schema
        event(end+1).label = double(meta.label); %#ok<AGROW>
        event(end).code    = meta.code;
        event(end).etstart = double(et_start_250);
        event(end).etstop  = double(et_stop_250);
        event(end).gtstart = double(gt_start_250);
        event(end).gtstop  = double(gt_stop_250);
        event(end).qc      = qc_str;
    end
end


function offsets = estimate_step3_timebase_offsets(EEG, all_trials_1k, ...
        source_part_arr, n_parts, session_id)
% Estimate, per merged sub-part, how many samples step3's event clock is
% ahead of the step1/raw clock used by extract_goni. This is usually the
% amount of setup/bad-data removed by ASR before or within the part.
%
% The trial list and goni indices still come from G.trials. We use EEG marker
% starts only to calibrate the clock transform:
%     et_step3 = round(start_eeg_sec_merged * 250) + 1 - offset(part)

    offsets = zeros(1, max(1, n_parts));
    fs = EEG.srate;
    if abs(fs - 250) > 1e-6
        error('export_session:badSrate', 'Expected 250 Hz step3 EEG, got %.6g', fs);
    end

    marker_pairs = collect_marker_pairs_by_part(EEG, n_parts, session_id, ...
        all_trials_1k, source_part_arr);
    trial_types = {all_trials_1k.type};
    type_names = {'imagine', 'walk', 'rest'};

    for p = 1:numel(offsets)
        part_offsets = [];
        for ti = 1:numel(type_names)
            ttype = type_names{ti};
            tr_idx = find(source_part_arr == p & strcmp(trial_types, ttype));
            starts = marker_pairs(p).(ttype).start;
            n = min(numel(tr_idx), numel(starts));
            if n == 0
                continue;
            end
            raw_start = nan(n, 1);
            marker_start = nan(n, 1);
            for k = 1:n
                raw_start(k) = round(all_trials_1k(tr_idx(k)).start_eeg_sec_merged * fs) + 1;
                marker_start(k) = round(starts(k));
            end
            d = raw_start - marker_start;
            med_d = median(d);
            keep = abs(d - med_d) <= round(2 * fs);
            part_offsets = [part_offsets; d(keep)]; %#ok<AGROW>
            if any(~keep)
                warning('export_session:timebaseOffsetOutliers', ...
                    '%s part %d %s: dropping %d/%d offset outlier(s)', ...
                    session_id, p, ttype, sum(~keep), numel(keep));
            end
        end

        if isempty(part_offsets)
            warning('export_session:noTimebaseOffset', ...
                '%s part %d: no EEG marker pairs for offset calibration; using 0', ...
                session_id, p);
            offsets(p) = 0;
            continue;
        end

        offsets(p) = round(median(part_offsets));
        spread = max(abs(part_offsets - offsets(p)));
        fprintf('  Timebase: part %d step1→step3 offset = %d samples (%.3fs), spread=%d, n=%d\n', ...
            p, offsets(p), offsets(p) / fs, spread, numel(part_offsets));
        if spread > round(2 * fs)
            warning('export_session:timebaseOffsetSpread', ...
                '%s part %d: offset spread %d samples > 2 seconds', ...
                session_id, p, spread);
        end
    end
end


function marker_pairs = collect_marker_pairs_by_part(EEG, n_parts, session_id, ...
        all_trials_1k, source_part_arr)
    empty = struct( ...
        'imagine', struct('start', [], 'stop', []), ...
        'walk',    struct('start', [], 'stop', []), ...
        'rest',    struct('start', [], 'stop', []));
    marker_pairs = repmat(empty, 1, max(1, n_parts));
    all_pairs = empty;

    evt_types = cell(1, length(EEG.event));
    evt_lats = nan(1, length(EEG.event));
    for k = 1:length(EEG.event)
        evt_types{k} = strtrim(char(EEG.event(k).type));
        evt_lats(k) = EEG.event(k).latency;
    end

    specs = event_condition_specs(session_id);
    for si = 1:size(specs, 1)
        start_code = strtrim(specs{si, 1});
        stop_code  = strtrim(specs{si, 2});
        code       = specs{si, 3};
        r1_mode    = specs{si, 5};
        trial_type = specs{si, 6};

        starts = find(strcmp(evt_types, start_code));
        stops  = find(strcmp(evt_types, stop_code));
        stop_lats = evt_lats(stops);
        for ii = 1:numel(starts)
            sl = evt_lats(starts(ii));
            nxt = stop_lats(find(stop_lats > sl, 1, 'first'));
            if isempty(nxt)
                continue;
            end
            if ~passes_r1_mode(r1_mode, evt_types, evt_lats, sl, nxt, EEG.srate)
                continue;
            end
            switch code
                case 'gaitimagery'
                    key = 'imagine';
                case 'gaitexecution'
                    key = 'walk';
                case 'rest'
                    key = 'rest';
                otherwise
                    key = trial_type;
            end
            all_pairs.(key).start(end+1) = sl; %#ok<AGROW>
            all_pairs.(key).stop(end+1) = nxt; %#ok<AGROW>
        end
    end

    % Split the merged step3 marker stream into sub-parts by the G.trials
    % source_part counts. Boundary events are not reliable here because ASR
    % also writes boundary events; the trial extraction order is reliable.
    trial_types = {all_trials_1k.type};
    type_names = {'imagine', 'walk', 'rest'};
    cursor = struct('imagine', 1, 'walk', 1, 'rest', 1);
    for p = 1:max(1, n_parts)
        for ti = 1:numel(type_names)
            ttype = type_names{ti};
            want = sum(source_part_arr == p & strcmp(trial_types, ttype));
            if want == 0
                continue;
            end
            available = numel(all_pairs.(ttype).start) - cursor.(ttype) + 1;
            take = min(want, max(0, available));
            if take == 0
                warning('export_session:noMarkersForPart', ...
                    '%s part %d %s: wanted %d marker pairs, found none remaining', ...
                    session_id, p, ttype, want);
                continue;
            end
            idx = cursor.(ttype):(cursor.(ttype) + take - 1);
            marker_pairs(p).(ttype).start = all_pairs.(ttype).start(idx);
            marker_pairs(p).(ttype).stop  = all_pairs.(ttype).stop(idx);
            cursor.(ttype) = cursor.(ttype) + take;
            if take < want
                warning('export_session:fewerMarkersThanTrials', ...
                    '%s part %d %s: wanted %d marker pairs, only %d available', ...
                    session_id, p, ttype, want, take);
            end
        end
    end
end

function specs = event_condition_specs(session_id) %#ok<DEFNU>
% Retained for backward-compat / external callers; build_events no longer
% uses this — trial type/label is driven entirely by extract_goni's
% G.trials(j).type via type_to_meta in build_events.
    if strcmp(session_id, 'SUB_01_sess01')
        specs = {
            'S  1', 'S  2', 'gaitimagery',   0, '',           'imagine';
            'S  4', 'S  5', 'gaitexecution', 1, 'without_r1', 'walk';
            'S  4', 'S  5', 'rest',          2, 'with_r1',    'rest'
        };
    else
        specs = {
            'S  1', 'S  2', 'gaitimagery',   0, '', 'imagine';
            'S  4', 'S  5', 'gaitexecution', 1, '', 'walk';
            'S  7', 'S  8', 'rest',          2, '', 'rest'
        };
    end
end


function part_ranges = export_part_ranges(EEG, n_parts)
    if nargin < 2 || isempty(n_parts), n_parts = 1; end
    if n_parts <= 1
        part_ranges = [1, EEG.pnts];
        return;
    end

    evt_types = {EEG.event.type};
    evt_lats  = [EEG.event.latency];
    is_boundary = strcmp(evt_types, 'boundary');
    if isfield(EEG.event, 'duration')
        durations = nan(1, length(EEG.event));
        for k = 1:length(EEG.event)
            d = EEG.event(k).duration;
            if isnumeric(d) && ~isempty(d), durations(k) = d; end
        end
        merge_lats = sort(evt_lats(is_boundary & isnan(durations)));
    else
        merge_lats = sort(evt_lats(is_boundary));
    end

    need = n_parts - 1;
    if length(merge_lats) < need
        error('export_session:missingMergeBoundaries', ...
            'Expected %d merge boundary event(s), found %d', need, length(merge_lats));
    end
    merge_lats = merge_lats(1:need);
    starts = [1, floor(merge_lats) + 1];
    ends   = [floor(merge_lats), EEG.pnts];
    part_ranges = [starts(:), ends(:)];
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
            error('export_session:badR1Mode', 'Unknown R1 mode: %s', mode);
    end
end


function tf = has_coincident_r1(evt_types, evt_lats, sl, nxt, srate)
    is_r1 = cellfun(@(t) ~isempty(regexp(strtrim(t), '^R\s*1$', 'once')), evt_types);
    r1_lats = evt_lats(is_r1);
    tol = max(2, round(0.02 * srate));
    tf = any(abs(r1_lats - sl) <= tol) || any(abs(r1_lats - nxt) <= tol);
end


function save_eeg(EEG, eeg_source, out_path)
    eeg = struct();
    eeg.x = double(EEG.data);          % match example schema (double)
    eeg.s = double(EEG.srate);
    eeg.c = EEG.chanlocs;
    eeg.source_set = eeg_source;       % step3 or step2
    save(out_path, 'eeg', '-v7.3');
end


function save_goni(goni_250, goni_labels, out_path)
    goni = struct();
    goni.x = double(goni_250);          % match example schema (double)
    goni.s = double(250);
    goni.c = goni_labels(:);            % cell {24×1} of short canonical labels (e.g. 'LHipS X')
    save(out_path, 'goni', '-v7.3');
end


function [a_norm, b_norm] = normalize_struct_fields(a, b)
% Make scalar struct b have the same fieldnames as struct array a, and
% conversely, so that a(end+1) = b assignment is legal. Missing fields
% on either side are filled with [] (numeric/empty equivalent).
    fa = fieldnames(a);
    fb = fieldnames(b);
    extra_in_b = setdiff(fb, fa);
    for k = 1:length(extra_in_b)
        % add to all elements of a
        for ai = 1:length(a)
            a(ai).(extra_in_b{k}) = [];
        end
    end
    extra_in_a = setdiff(fa, fb);
    for k = 1:length(extra_in_a)
        b.(extra_in_a{k}) = [];
    end
    % Reorder b's fields to match a
    b = orderfields(b, fieldnames(a));
    a_norm = a;
    b_norm = b;
end
