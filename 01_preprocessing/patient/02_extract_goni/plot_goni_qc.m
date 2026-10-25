function plot_goni_qc(goni_mat_path, out_png_path)
%PLOT_GONI_QC (patient)  6-joint x 3-condition QC plot.
%
%   Cols: MI/Walker-X | S2S/Subject-X | 2-Min Walk/Subject-X
%   Rows: LHip, LKnee, LAnkle, RHip, RKnee, RAnkle (X axis)

    S = load(goni_mat_path);
    if isfield(S, 'goni_labels'), goni_labels = S.goni_labels;
    else, goni_labels = S.trials(1).goni_labels; end
    if isfield(S, 'goni_srate'), srate = S.goni_srate; else, srate = 1000; end

    AL = []; if isfield(S,'align_info') && ~isempty(S.align_info), AL = S.align_info{1}; end

    qc_status = {S.qc.status};
    qc_reason = {S.qc.reason};
    types     = {S.trials.type};

    mi_idx   = find(strcmpi(types, 'imagine'));
    s2s_idx  = find(strcmpi(types, 's2s'));
    walk_idx = find(strcmpi(types, 'walk'));
    prob_mi   = mi_idx(strcmp(qc_status(mi_idx),'problem'));
    prob_s2s  = s2s_idx(strcmp(qc_status(s2s_idx),'problem'));
    prob_walk = walk_idx(strcmp(qc_status(walk_idx),'problem'));
    n_prob = length(prob_mi) + length(prob_s2s) + length(prob_walk);

    joint_w = {'LHipW X','LKneW X','LAnkW X','RHipW X','RKneW X','RAnkW X'};
    joint_s = {'LHipS X','LKneS X','LAnkS X','RHipS X','RKneS X','RAnkS X'};
    joint_names = {'LHip','LKnee','LAnkle','RHip','RKnee','RAnkle'};

    fig = figure('Position',[0 0 1500 1400],'Color','w','Visible','off');

    cols = { ...
        struct('idx', mi_idx,   'prob', prob_mi,   'ch_lbl', {joint_w}, 'title', 'MI / Walker X'), ...
        struct('idx', s2s_idx,  'prob', prob_s2s,  'ch_lbl', {joint_s}, 'title', 'S2S / Subject X'), ...
        struct('idx', walk_idx, 'prob', prob_walk, 'ch_lbl', {joint_s}, 'title', '2-Min Walk / Subject X')};

    for j = 1:6
        for c = 1:3
            ax = subplot(6, 3, (j-1)*3 + c); hold(ax, 'on');
            ch_idx = find_goni_idx(goni_labels, cols{c}.ch_lbl{j});
            if ch_idx > 0
                plot_panel(ax, S, cols{c}.idx, cols{c}.prob, ch_idx, qc_status, qc_reason, srate);
            else
                text(ax, 0.5, 0.5, 'missing', 'Units','normalized', 'HorizontalAlignment','center');
            end
            if c == 1, ylabel(ax, sprintf('%s (deg)', joint_names{j})); end
            if j == 1
                title(ax, sprintf('%s  (%d trials, %d prob)', ...
                    cols{c}.title, length(cols{c}.idx), length(cols{c}.prob)), 'FontSize', 9);
            end
            if j == 6, xlabel(ax, 'Time (s)'); end
            box(ax, 'off');
        end
    end

    if ~isempty(AL)
        sgtitle(fig, sprintf('%s | align=%s verdict=%s match=%.1f%%  |  %d total prob', ...
            S.label, AL.method, AL.verdict, AL.match_frac*100, n_prob), ...
            'Interpreter','none', 'FontSize', 11, 'FontWeight','bold');
    else
        sgtitle(fig, sprintf('%s  |  %d total prob', S.label, n_prob), ...
            'Interpreter','none', 'FontSize', 11, 'FontWeight','bold');
    end

    if ~exist(fileparts(out_png_path), 'dir'), mkdir(fileparts(out_png_path)); end
    exportgraphics(fig, out_png_path, 'Resolution', 120);
    close(fig);
end


function plot_panel(ax, S, all_idx, prob_idx, ch_idx, qc_status, qc_reason, srate)
    ok_idx = all_idx(strcmp(qc_status(all_idx),'ok'));
    for k = ok_idx
        data = S.trials(k).joint_angles;
        if size(data,2) < ch_idx, continue; end
        t = (0:size(data,1)-1) / srate;
        plot(ax, t, data(:,ch_idx), 'Color', [0.78 0.78 0.78], 'LineWidth', 0.4);
    end
    for k = prob_idx
        data = S.trials(k).joint_angles;
        if size(data,2) < ch_idx, continue; end
        t = (0:size(data,1)-1) / srate;
        reason = qc_reason{k};
        if contains(reason, 'flat')
            col = [0.90 0.10 0.10];
        elseif contains(reason, 'length')
            col = [0.10 0.50 0.90];
        elseif contains(reason, 'shape')
            col = [0.10 0.65 0.20];
        else
            col = [0.50 0.50 0.50];
        end
        plot(ax, t, data(:,ch_idx), 'Color', col, 'LineWidth', 1.3);
        text(ax, t(end), data(end,ch_idx), sprintf(' #%d', k), ...
            'Color', col, 'FontSize', 7);
    end
end
