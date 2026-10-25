function plot_session_align(EEG, out_png_path, session_label)
%PLOT_SESSION_ALIGN  Two-panel alignment figure used for visual review of every session:
%                    EEG, goniometer and event lines (top) and per-marker alignment
%                    residual (bottom).
%

    if nargin < 3, session_label = EEG.setname; end

    t_eeg  = ((1:EEG.pnts) - 1) / EEG.srate;
    has_goni_etc = isfield(EEG.etc, 'goni') && ~isempty(EEG.etc.goni);
    if has_goni_etc && isfield(EEG.etc.goni, 'labels')
        goni_l = EEG.etc.goni.labels;
    else
        goni_l = {};
    end

    has_continuous = has_goni_etc && isfield(EEG.etc.goni, 'data') && ~isempty(EEG.etc.goni.data);
    has_epochs     = has_goni_etc && isfield(EEG.etc.goni, 'epochs') && isfield(EEG.etc.goni, 'meta');
    if has_continuous
        assert(size(EEG.etc.goni.data, 2) == EEG.pnts, 'goni/EEG length mismatch');
        goni_d = EEG.etc.goni.data;
    else
        goni_d = [];
    end

    eeg_ch_idx = find(strcmpi({EEG.chanlocs.labels}, 'Cz'), 1);
    if isempty(eeg_ch_idx), eeg_ch_idx = 1; end

    idx_LKneW = find(strcmp(goni_l, 'LKneW X'), 1);
    idx_LKneS = find(strcmp(goni_l, 'LKneS X'), 1);
    idx_RKneW = find(strcmp(goni_l, 'RKneW X'), 1);
    idx_RKneS = find(strcmp(goni_l, 'RKneS X'), 1);

    fig = figure('Visible', 'off', 'Position', [100 100 3200 900]);

    % ========== TOP PANEL: EEG + goni overlay + event lines ==========
    ax1 = subplot(5, 1, [1 2 3]);
    hold(ax1, 'on');

    % First: plot EEG + goni on yyaxis
    yyaxis(ax1, 'left');
    plot(ax1, t_eeg, EEG.data(eeg_ch_idx, :), 'Color', [0.1 0.1 0.1 0.5], 'LineWidth', 0.5);
    ylabel(ax1, sprintf('EEG %s (\\muV)', EEG.chanlocs(eeg_ch_idx).labels));
    ax1.YColor = [0.1 0.1 0.1];

    yyaxis(ax1, 'right');
    legend_entries = {sprintf('EEG %s', EEG.chanlocs(eeg_ch_idx).labels)};

    % Pick channels to overlay (prefer Left knee, fallback to Right)
    overlay_specs = {};
    if ~isempty(idx_LKneW), overlay_specs(end+1, :) = {idx_LKneW, [0.1 0.4 0.85], 'LKneW X (Walker)'}; end
    if ~isempty(idx_LKneS), overlay_specs(end+1, :) = {idx_LKneS, [0.8 0.25 0.65], 'LKneS X (Subject)'}; end
    if isempty(overlay_specs)
        if ~isempty(idx_RKneW), overlay_specs(end+1, :) = {idx_RKneW, [0.1 0.4 0.85], 'RKneW X'}; end
        if ~isempty(idx_RKneS), overlay_specs(end+1, :) = {idx_RKneS, [0.8 0.25 0.65], 'RKneS X'}; end
    end

    if has_continuous
        % Continuous goniometer overlay for the full session
        for jj = 1:size(overlay_specs, 1)
            plot(ax1, t_eeg, goni_d(overlay_specs{jj, 1}, :), 'Color', overlay_specs{jj, 2}, 'LineWidth', 0.6);
            legend_entries{end+1} = overlay_specs{jj, 3}; %#ok<AGROW>
        end
    elseif has_epochs
        ep   = EEG.etc.goni.epochs;
        meta = EEG.etc.goni.meta;
        srate_g = EEG.etc.goni.srate;
        first_legend = true(1, size(overlay_specs, 1));
        for ti = 1:length(ep)
            seg = ep{ti};
            if isempty(seg), continue; end
            t0 = meta(ti).eeg_start_sec;
            t_seg = t0 + ((1:size(seg, 2)) - 1) / srate_g;
            for jj = 1:size(overlay_specs, 1)
                ch_idx = overlay_specs{jj, 1};
                if ch_idx > size(seg, 1), continue; end
                if first_legend(jj)
                    plot(ax1, t_seg, seg(ch_idx, :), 'Color', overlay_specs{jj, 2}, 'LineWidth', 0.6);
                    legend_entries{end+1} = overlay_specs{jj, 3}; %#ok<AGROW>
                    first_legend(jj) = false;
                else
                    plot(ax1, t_seg, seg(ch_idx, :), 'Color', overlay_specs{jj, 2}, 'LineWidth', 0.6, 'HandleVisibility', 'off');
                end
            end
        end
    end
    ylabel(ax1, 'Goni knee angle (deg)');
    ax1.YColor = [0.3 0.3 0.3];
    xlim(ax1, [0 t_eeg(end)]);

    % NOW draw event lines so they render on top of the plots
    ttl_re = '^S\s*\d+$';
    is_ttl = cellfun(@(t) ~isempty(regexp(t, ttl_re, 'once')), {EEG.event.type});
    n_matched = 0; n_unmatched = 0; n_nonttl = 0;
    ev_t_list = []; ev_label_list = {}; ev_col_list = {};
    yyaxis(ax1, 'left'); yl = ylim(ax1);
    for k = 1:length(EEG.event)
        t_ev = (EEG.event(k).latency - 1) / EEG.srate;
        if t_ev < 0 || t_ev > t_eeg(end), continue; end
        if is_ttl(k)
            if isfield(EEG.event(k), 'goni_idx') && ~isnan(EEG.event(k).goni_idx)
                col = [0.15 0.65 0.25]; lw = 1.2; alph = 0.8;
                n_matched = n_matched + 1;
            else
                col = [0.85 0.15 0.15]; lw = 1.8; alph = 0.95;
                n_unmatched = n_unmatched + 1;
            end
        else
            col = [0.55 0.55 0.55]; lw = 0.6; alph = 0.5;
            n_nonttl = n_nonttl + 1;
        end
        % Use line() for reliable top-layer vertical lines
        line(ax1, [t_ev t_ev], yl, 'Color', [col alph], 'LineWidth', lw, 'HandleVisibility', 'off');
        ev_label = char(EEG.event(k).type);
        ev_label = regexprep(ev_label, '^[A-Za-z]\s+(?=\d)', '');  % "S  1"->"1", "R 1"->"1"
        ev_t_list(end+1) = t_ev;          %#ok<AGROW>
        ev_label_list{end+1} = ev_label;  %#ok<AGROW>
        ev_col_list{end+1} = col;         %#ok<AGROW>
    end
    ylim(ax1, yl);

    title(ax1, sprintf('%s | TTL matched: %d (green) / unmatched: %d (red) | non-TTL: %d (gray)', ...
                  session_label, n_matched, n_unmatched, n_nonttl), 'Interpreter', 'none');
    legend(ax1, legend_entries, 'Location', 'northeast', 'Box', 'off', 'FontSize', 8);
    box(ax1, 'on'); grid(ax1, 'on');

    % Event labels below the top panel (separate text, not inline with line)
    y_text = yl(1);
    for k = 1:length(ev_t_list)
        text(ax1, ev_t_list(k), y_text, ev_label_list{k}, ...
             'Color', ev_col_list{k}, 'FontSize', 5, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
             'Clipping', 'off');
    end

    % ========== MIDDLE PANEL: IOI strip (two rows, segments proportional to IOI) ==========
    %  Top row    = EEG valid (matched) TTLs. Horizontal segments between
    %               consecutive matched ticks — length = EEG IOI.
    %  Bottom row = Goni matched edges projected to EEG clock via global offset.
    %               Horizontal segments = goni IOI (same clock as EEG row).
    %  Unmatched EEG TTLs + spurious goni edges drawn as red ticks only (no segment).
    %  Segment color: green if |goni IOI − EEG IOI| ≤ 100 ms, orange otherwise.
    %  When alignment is correct the two rows' segments overlap along x.
    %  When offset shifts (e.g. SUB_16_sess02) the goni row shows a horizontal jump.
    ax_strip = subplot(5, 1, 4);
    hold(ax_strip, 'on');

    if isfield(EEG.etc, 'align_log') && isfield(EEG.etc.align_log, 'goni_edges_sec_1kHz')
        AL = EEG.etc.align_log;
        t_ttl  = AL.per_ttl_eeg_time_sec;
        mtc    = logical(AL.per_ttl_matched);
        t_goni = AL.goni_edges_sec_1kHz;
        per_ttl_goni      = AL.per_ttl_goni_time_sec;
        % multi_goni_file sessions log no single global offset
        % (offset_samples_1kHz empty); the per-file offsets put goni edges
        % tens of seconds off the EEG clock so the goni row would render off
        % screen. Derive a robust offset from the matched pairs to project the
        % goni row onto the EEG row. Single-goni sessions keep the logged
        % offset, so their output is unchanged.
        if ~isfield(AL, 'offset_samples_1kHz') || isempty(AL.offset_samples_1kHz) ...
                || ~all(isfinite(AL.offset_samples_1kHz(:)))
            global_offset_sec = median(per_ttl_goni(mtc) - t_ttl(mtc));
        else
            global_offset_sec = AL.offset_samples_1kHz / 1000;
        end

        y_eeg = 1.0; y_goni = 0.0;
        tick_h = 0.30; seg_lw = 2.2;
        col_ok  = [0.15 0.65 0.25];   % green
        col_bad = [0.90 0.45 0.10];   % orange (IOI mismatch)
        col_err = [0.85 0.15 0.15];   % red   (unmatched / spurious)
        ioi_tol_sec = 0.100;

        % Resolve which goni edges got matched (build projected array)
        goni_matched = false(size(t_goni));
        goni_proj    = t_goni - global_offset_sec;
        gi_of_ttl    = nan(size(t_ttl));   % matched goni index per TTL (reused by the pairing lines below)
        for k = 1:length(t_ttl)
            if mtc(k)
                [~, gi] = min(abs(t_goni - per_ttl_goni(k)));
                goni_matched(gi) = true;
                gi_of_ttl(k)     = gi;
            end
        end

        % --- Top row: EEG matched sequence ---
        matched_k = find(mtc);
        for jj = 1:length(matched_k)-1
            k1 = matched_k(jj); k2 = matched_k(jj+1);
            ioi_e = t_ttl(k2) - t_ttl(k1);
            ioi_g = per_ttl_goni(k2) - per_ttl_goni(k1);
            sc = col_ok; if abs(ioi_e - ioi_g) > ioi_tol_sec, sc = col_bad; end
            line(ax_strip, [t_ttl(k1) t_ttl(k2)], [y_eeg y_eeg], ...
                 'Color', [sc 0.75], 'LineWidth', seg_lw);
        end
        for k = 1:length(t_ttl)
            if mtc(k), c = col_ok; lw = 1.1; else, c = col_err; lw = 1.6; end
            line(ax_strip, [t_ttl(k) t_ttl(k)], [y_eeg-tick_h y_eeg+tick_h], ...
                 'Color', c, 'LineWidth', lw);
            % EEG marker number above the tick (small, rotated to save x-space)
            text(ax_strip, t_ttl(k), y_eeg+tick_h+0.03, sprintf('%d', k), ...
                 'FontSize', 3.2, 'Rotation', 90, 'HorizontalAlignment', 'left', ...
                 'VerticalAlignment', 'middle', 'Color', c*0.55, 'Clipping', 'on');
        end

        % --- Bottom row: Goni matched sequence, same IOI tolerance colormap ---
        matched_g = find(goni_matched);
        for jj = 1:length(matched_g)-1
            g1 = matched_g(jj); g2 = matched_g(jj+1);
            % Find corresponding EEG IOI for this goni pair
            k1 = find(mtc & (abs(t_goni(g1) - per_ttl_goni) < 1e-6), 1);
            k2 = find(mtc & (abs(t_goni(g2) - per_ttl_goni) < 1e-6), 1);
            sc = col_ok;
            if ~isempty(k1) && ~isempty(k2)
                ioi_e = t_ttl(k2) - t_ttl(k1);
                ioi_g = t_goni(g2) - t_goni(g1);
                if abs(ioi_e - ioi_g) > ioi_tol_sec, sc = col_bad; end
            end
            line(ax_strip, [goni_proj(g1) goni_proj(g2)], [y_goni y_goni], ...
                 'Color', [sc 0.75], 'LineWidth', seg_lw);
        end
        for g = 1:length(t_goni)
            if goni_matched(g), c = col_ok; lw = 1.1; else, c = col_err; lw = 1.6; end
            line(ax_strip, [goni_proj(g) goni_proj(g)], [y_goni-tick_h y_goni+tick_h], ...
                 'Color', c, 'LineWidth', lw);
            % Goni marker number below the tick (small, rotated to save x-space)
            text(ax_strip, goni_proj(g), y_goni-tick_h-0.03, sprintf('%d', g), ...
                 'FontSize', 3.2, 'Rotation', 90, 'HorizontalAlignment', 'right', ...
                 'VerticalAlignment', 'middle', 'Color', c*0.55, 'Clipping', 'on');
        end

        % --- Pairing lines: each matched EEG TTL -> its paired goni edge ---
        % Driven by the SAME per-TTL goni pairing the pipeline writes to
        % EEG.event.goni_idx (goni_idx_per_ttl, INCLUDING manual overrides), so
        % what is drawn here IS the EEG<->goni correspondence used downstream for
        % the goni extraction / ICA epoching / trial export. Slanted or index-
        % skipping lines expose offset drift and spurious-edge skips at a glance.
        for k = 1:length(t_ttl)
            if ~mtc(k), continue; end
            gi = gi_of_ttl(k);
            line(ax_strip, [t_ttl(k) goni_proj(gi)], [y_eeg-tick_h y_goni+tick_h], ...
                 'Color', [0 0 0 0.5], 'LineWidth', 1.6);
        end

        n_spurious = sum(~goni_matched);
        n_unmatched_eeg = sum(~mtc);
        title(ax_strip, sprintf(['IOI strip (EEG top, Goni bottom projected by global offset %.3fs)  |  ' ...
            'thin lines = EEG<->goni pairing used by extract/ICA/export  |  ' ...
            'EEG unmatched: %d  |  goni spurious: %d  |  green=IOI consistent, orange=IOI mismatch'], ...
            global_offset_sec, n_unmatched_eeg, n_spurious));
    end
    set(ax_strip, 'YTick', [0 1], 'YTickLabel', {'Goni','EEG'}, 'TickDir', 'out');
    ylim(ax_strip, [-0.95 1.95]);
    box(ax_strip, 'on');
    xlim(ax_strip, [0 t_eeg(end)]);

    % ========== BOTTOM PANEL: per-trigger alignment error ==========
    ax2 = subplot(5, 1, 5);
    hold(ax2, 'on');

    if isfield(EEG.etc, 'align_log') && isfield(EEG.etc.align_log, 'per_ttl_eeg_time_sec')
        AL = EEG.etc.align_log;
        t_ttl = AL.per_ttl_eeg_time_sec;
        mtc   = logical(AL.per_ttl_matched);
        % multi_goni_file: per_ttl_residual_ms carries a constant per-file
        % offset (tens of seconds) because no global offset is logged.
        % Recompute the residual as jitter about the robust matched-pair offset
        % so the scatter shows the true per-trigger error (~tens of ms), not
        % the constant offset. Single-goni sessions use the logged residual.
        mg_offset_sec = NaN;
        if ~isfield(AL, 'offset_samples_1kHz') || isempty(AL.offset_samples_1kHz) ...
                || ~all(isfinite(AL.offset_samples_1kHz(:)))
            mg_offset_sec = median(AL.per_ttl_goni_time_sec(mtc) - t_ttl(mtc));
            res = (AL.per_ttl_goni_time_sec - t_ttl - mg_offset_sec) * 1000;
        else
            res = AL.per_ttl_residual_ms;
        end
        has_seg = isfield(AL, 'per_ttl_segment') && isfield(AL, 'drift_segments');
        if has_seg
            per_ttl_seg = AL.per_ttl_segment;
            segs = AL.drift_segments;
            n_segs = length(segs);
        else
            per_ttl_seg = ones(size(t_ttl));
            n_segs = 1;
        end
        seg_palette = [0.15 0.65 0.25; 0.9 0.4 0.1; 0.2 0.4 0.8; 0.7 0.2 0.7; 0.5 0.5 0.1; 0.0 0.6 0.6];

        leg_h = []; leg_s = {};
        for s = 1:n_segs
            sc = seg_palette(mod(s-1, size(seg_palette, 1)) + 1, :);
            seg_mask = (per_ttl_seg == s);
            m = seg_mask & mtc;
            u = seg_mask & ~mtc;
            if any(m)
                h = scatter(ax2, t_ttl(m), res(m), 22, sc, 'filled', ...
                            'MarkerFaceAlpha', 0.8);
                leg_h(end+1) = h;  %#ok<AGROW>
                if n_segs > 1
                    off_ms = (segs(s).offset_sec - AL.offset_samples_1kHz/1000) * 1000;
                    leg_s{end+1} = sprintf('Seg%d %+.1fms (%d ok)', s, off_ms, sum(m));  %#ok<AGROW>
                else
                    leg_s{end+1} = sprintf('Matched (%d)', sum(m));  %#ok<AGROW>
                end
            end
            if any(u)
                h = scatter(ax2, t_ttl(u), zeros(1, sum(u)), 36, sc, 'x', 'LineWidth', 1.4);
                leg_h(end+1) = h;  %#ok<AGROW>
                leg_s{end+1} = sprintf('Seg%d unmatched (%d)', s, sum(u));  %#ok<AGROW>
            end
        end
        yline(ax2, 0, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, 'HandleVisibility', 'off');

        % Segment boundary vertical lines
        for s = 2:n_segs
            b = (segs(s-1).end_eeg_sec + segs(s).start_eeg_sec) / 2;
            xline(ax2, b, '--', sprintf('seg%d|%d', s-1, s), ...
                  'Color', [0.8 0.1 0.1], 'LineWidth', 1.5, 'FontSize', 7, ...
                  'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
        end

        matched_res = res(mtc);
        if ~isempty(matched_res)
            mean_r = mean(matched_res); max_r = max(matched_res); med_r = median(matched_res);
        else
            mean_r = NaN; max_r = NaN; med_r = NaN;
        end
        seg_str = '';
        if n_segs > 1
            off_list = arrayfun(@(s) sprintf('%+.1f', (s.offset_sec - AL.offset_samples_1kHz/1000) * 1000), segs, 'UniformOutput', false);
            seg_str = sprintf('  |  %d drift segs [%s ms]', n_segs, strjoin(off_list, '/'));
        end
        mg_str = '';
        if ~isnan(mg_offset_sec)
            mg_str = sprintf('  |  multi_goni: jitter about per-file offset %.1f s', mg_offset_sec);
        end
        title(ax2, sprintf('Trigger alignment error: %d/%d matched  |  median=%.1f ms  max=%.1f ms%s%s', ...
            sum(mtc), length(mtc), med_r, max_r, seg_str, mg_str));
        if ~isempty(leg_h)
            legend(ax2, leg_h, leg_s, 'Location', 'northeast', 'FontSize', 8);
        end
    else
        text(ax2, 0.5, 0.5, 'no per-TTL residuals available', ...
             'Units', 'normalized', 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
    end
    xlabel(ax2, 'EEG time (s)');
    ylabel(ax2, 'Error (ms)');
    box(ax2, 'on'); grid(ax2, 'on');
    xlim(ax2, [0 t_eeg(end)]);
    linkaxes([ax1 ax_strip ax2], 'x');

    % Force all 3 panels to same x position + width
    drawnow;
    p1 = ax1.Position; ps = ax_strip.Position; p2 = ax2.Position;
    ax_strip.Position = [p1(1) ps(2) p1(3) ps(4)];
    ax2.Position      = [p1(1) p2(2) p1(3) p2(4)];

    out_dir = fileparts(out_png_path);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    exportgraphics(fig, out_png_path, 'Resolution', 240);
    close(fig);
end
