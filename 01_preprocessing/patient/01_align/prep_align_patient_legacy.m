function out = prep_align_patient_legacy(reg_entry, out_dir, qc_dir, opts)
%PREP_ALIGN_PATIENT_LEGACY  Alignment for the two-marker protocol (first two P1 sessions).
%
%   Each (vhdr, goniometer) pair is one experimental block. The goniometer trigger
%   channel carries only the block start and end edges; the EEG carries all S1/S2
%   markers. Marker times are mapped linearly between the two anchors:
%     g_sec(k) = g_start + (g_end - g_start) * (t_eeg(k) - t_eeg_first) / (t_eeg_last - t_eeg_first)
%
%   All EEG blocks are then merged and the goniometer data concatenated with
%   cumulative sample offsets. Each merged event carries .goni_sec on the global
%   goniometer clock (1000 Hz). The verdict is 'PASS_LEGACY'.

    if nargin < 4, opts = struct(); end
    opts = set_default(opts, 'srate_out',  250);
    opts = set_default(opts, 'iei_thr_ms', 200);

    prep_patient_paths();

    label = reg_entry.label;
    fprintf('\n================================================================\n');
    fprintf('=== prep_align_patient_legacy: %s\n', label);
    fprintf('================================================================\n');

    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    if ~exist(fullfile(qc_dir, 'align_log'), 'dir'), mkdir(fullfile(qc_dir, 'align_log')); end
    if ~exist(fullfile(qc_dir, 'align_png'), 'dir'), mkdir(fullfile(qc_dir, 'align_png')); end

    out = struct('set_path', '', 'json_path', '', 'png_path', '', ...
                 'verdict', '', 'align_log', struct());

    vhdrs = reg_entry.vhdr_paths;
    gonis = reg_entry.goni_paths_per_dir;   % cell of cells (1 goni per dir)
    nP = length(vhdrs);
    if nP ~= length(gonis)
        error('prep_align_patient_legacy:%d vhdrs != %d goni groups', nP, length(gonis));
    end
    if nP == 0
        error('prep_align_patient_legacy: no vhdr/goni pairs');
    end

    pair_eegs   = cell(1, nP);
    pair_gonis  = cell(1, nP);   % goni data 1000 Hz [T x ch]
    pair_labels = {};
    pair_meta   = repmat(struct('vhdr','','goni','','g_start_sec',NaN, ...
        'g_end_sec',NaN,'t_first_sec',NaN,'t_last_sec',NaN, ...
        'slope',NaN,'intercept',NaN,'n_events',0,'n_stim_edges',0,'anchor_mode',''), 1, nP);
    cum_goni_samples = 0;
    goni_sample_offsets = zeros(1, nP);   % start offset of pair p in global goni timeline

    for p = 1:nP
        fprintf('\n--- pair %d/%d ---\n', p, nP);
        if isempty(vhdrs{p})
            error('prep_align_patient_legacy: pair %d has no vhdr', p);
        end
        gonigrp = gonis{p};
        if isempty(gonigrp)
            error('prep_align_patient_legacy: pair %d has no goni', p);
        end
        gpath = gonigrp{1};   % use first goni file in this dir
        fprintf('   EEG  : %s\n   GONI : %s\n', vhdrs{p}, gpath);

        [d, nm, ~] = fileparts(vhdrs{p});
        EEG_p = pop_loadbv(d, [nm '.vhdr']);
        EEG_p = eeg_checkset(EEG_p);
        if EEG_p.srate ~= 1000
            warning('prep_align_patient_legacy: EEG srate=%d, expected 1000', EEG_p.srate);
        end

        G = load_goniometer(gpath);
        if G.srate ~= 1000
            warning('prep_align_patient_legacy: goni srate=%d, expected 1000', G.srate);
        end

        % Detect stim edges on goni
        edges_samp = detect_stim_edges(G.stim, 0.5);
        [edges_dedup, ~] = iei_dedup(edges_samp, round(opts.iei_thr_ms));
        n_edges = length(edges_dedup);
        fprintf('   goni stim edges (dedup): %d\n', n_edges);

        % The two-marker protocol uses only S1/S2 (no S4/S5/S7).
        types_p = {EEG_p.event.type};
        is_task = ismember(types_p, {'S  1', 'S  2'});
        if ~any(is_task)
            error('prep_align_patient_legacy: pair %d has no S1/S2 events', p);
        end
        task_lats = [EEG_p.event(is_task).latency];
        t_first_sec = (min(task_lats) - 1) / EEG_p.srate;
        t_last_sec  = (max(task_lats) - 1) / EEG_p.srate;
        if t_last_sec - t_first_sec < 1
            error('prep_align_patient_legacy: pair %d task span < 1s', p);
        end

        if n_edges >= 2
            g_start_sec = (edges_dedup(1)   - 1) / G.srate;
            g_end_sec   = (edges_dedup(end) - 1) / G.srate;
            slope_p     = (g_end_sec - g_start_sec) / (t_last_sec - t_first_sec);
            intercept_p = g_start_sec - slope_p * t_first_sec;
            anchor_mode = '2anchor';
        elseif n_edges == 1
            g_start_sec = (edges_dedup(1) - 1) / G.srate;
            g_end_sec   = NaN;
            slope_p     = 1;
            intercept_p = g_start_sec - t_first_sec;
            anchor_mode = '1anchor';
            fprintf('   single-anchor mode (slope=1, offset=%.3fs)\n', intercept_p);
        else
            error('prep_align_patient_legacy: pair %d has 0 goni edges', p);
        end

        % Compute per-event goni_sec (LOCAL goni clock for this pair)
        for k = 1:length(EEG_p.event)
            t_eeg_k = (EEG_p.event(k).latency - 1) / EEG_p.srate;
            EEG_p.event(k).goni_sec_local = intercept_p + slope_p * t_eeg_k;
            EEG_p.event(k).goni_sec       = EEG_p.event(k).goni_sec_local + cum_goni_samples / G.srate;
            EEG_p.event(k).source_pair    = p;
        end

        fprintf('   %s fit: g_start=%.2fs g_end=%.2fs | t_first=%.2fs t_last=%.2fs | slope=%.6g\n', ...
            anchor_mode, g_start_sec, g_end_sec, t_first_sec, t_last_sec, slope_p);

        pair_meta(p).vhdr        = vhdrs{p};
        pair_meta(p).goni        = gpath;
        pair_meta(p).g_start_sec = g_start_sec;
        pair_meta(p).g_end_sec   = g_end_sec;
        pair_meta(p).t_first_sec = t_first_sec;
        pair_meta(p).t_last_sec  = t_last_sec;
        pair_meta(p).slope       = slope_p;
        pair_meta(p).intercept   = intercept_p;
        pair_meta(p).n_events    = length(EEG_p.event);
        pair_meta(p).n_stim_edges = n_edges;
        pair_meta(p).anchor_mode = anchor_mode;

        pair_eegs{p}  = EEG_p;
        pair_gonis{p} = G.data;     % [T x nCh]
        if p == 1, goni_labels = G.labels; goni_srate = G.srate; end
        goni_sample_offsets(p) = cum_goni_samples;
        cum_goni_samples = cum_goni_samples + size(G.data, 1);
        pair_labels{p} = sprintf('p%d', p); %#ok<AGROW>
    end

    % --- Mergeset all EEGs ---
    fprintf('\n[MERGE] %d EEG files\n', nP);
    EEG = pair_eegs{1};
    for p = 2:nP
        EEG = pop_mergeset(EEG, pair_eegs{p}, 0);
    end
    EEG = eeg_checkset(EEG);
    fprintf('   merged EEG: %d ch x %d pts, %d events\n', EEG.nbchan, EEG.pnts, length(EEG.event));

    % --- chanlocs + 59-ch whitelist ---
    fprintf('\n[CHAN] standard_1005 lookup + 59-ch whitelist\n');
    dipfit_dir = fileparts(which('pop_dipfit_settings'));
    eloc_file  = fullfile(dipfit_dir, 'standard_BEM', 'elec', 'standard_1005.elc');
    EEG = pop_chanedit(EEG, 'lookup', eloc_file);
    EEG = eeg_checkset(EEG);

    brain_whitelist = prep_patient_brain_whitelist();
    all_labels = {EEG.chanlocs.labels};
    brain_keep = find(ismember(all_labels, brain_whitelist));
    non_brain  = setdiff(1:EEG.nbchan, brain_keep);
    if ~isempty(non_brain)
        fprintf('   removing %d non-brain: %s\n', length(non_brain), strjoin(all_labels(non_brain), ', '));
        EEG = pop_select(EEG, 'channel', brain_keep);
        EEG = eeg_checkset(EEG);
    end
    fprintf('   brain channels: %d\n', EEG.nbchan);

    % --- Resample EEG to 250 Hz (preserves goni_sec via re-mapped latency) ---
    fprintf('\n[RESAMPLE] EEG -> %d Hz\n', opts.srate_out);
    EEG = pop_resample(EEG, opts.srate_out);
    EEG = eeg_checkset(EEG);

    % --- Build align_log ---
    align_log = struct();
    align_log.special_case_type    = 'legacy_per_pair';
    align_log.alignment_algorithm  = 'per_pair_2anchor_linear';
    align_log.match_method         = 'legacy_2anchor';
    align_log.auto_verdict         = 'PASS_LEGACY';
    align_log.n_pairs              = nP;
    align_log.pair_meta            = pair_meta;
    align_log.goni_sample_offsets  = goni_sample_offsets;
    align_log.goni_total_samples   = cum_goni_samples;
    align_log.goni_srate           = goni_srate;
    align_log.n_eeg_events_total   = length(EEG.event);
    align_log.n_eeg_events_ttl     = length(EEG.event);   % alias for compatibility
    align_log.n_matched            = length(EEG.event);
    align_log.residual_ms          = struct('median', NaN, 'p95', NaN, 'max', NaN);
    align_log.offset_samples_1kHz  = NaN;   % per-pair; use event.goni_sec directly
    align_log.align_timestamp      = datestr(now, 'yyyy-mm-ddTHH:MM:SS');

    EEG.etc.align_log        = align_log;
    EEG.etc.cohort           = 'patient';
    EEG.etc.marker_protocol  = 'legacy';
    EEG.etc.special_case     = reg_entry.special_case;
    EEG.etc.pipeline_version = '-patient-legacy';
    EEG.etc.raw_goni_files   = cellfun(@(c) c{1}, gonis, 'UniformOutput', false);
    EEG.etc.raw_goni_file    = EEG.etc.raw_goni_files{1};

    % Save outputs
    out.set_path = fullfile(out_dir, [label '_step1.set']);
    pop_saveset(EEG, 'filename', [label '_step1.set'], 'filepath', out_dir, 'savemode', 'onefile');

    out.json_path = fullfile(qc_dir, 'align_log', [label '.json']);
    write_align_log_json(align_log, out.json_path);

    out.verdict   = 'PASS_LEGACY';
    out.align_log = align_log;
    fprintf('\n[done] %s | verdict=%s | %s\n', label, out.verdict, out.set_path);
end


function s = set_default(s, f, default)
    if ~isfield(s, f), s.(f) = default; end
end
