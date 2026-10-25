function recompute_goni_qc(varargin)
%RECOMPUTE_GONI_QC  Final per-trial goniometer QC verdict for the healthy cohort.
%   Rewrites the qc field of every *_goni.mat in place.
%
%   Features per trial:
%     active_std  mean over active channels of the channel standard deviation
%                 (reference-walker channels for imagined walking, participant
%                 channels for real walking)
%     n_pts       trial length in samples
%     shape_corr  mean over active channels of the peak cross-correlation with the
%                 mean of all other trials of the condition, within +/- 0.5 s lag
%
%   Rules, per session and condition, K = 3 x 1.4826, effective MAD = max(MAD, 5% of median):
%     flat    active_std < median - K*MAD                      (one-sided)
%     length  n_pts outside median +/- K*MAD                   (two-sided)
%     shape   shape_corr < median - K*MAD  AND  shape_corr < 0.5
%   A trial meeting any rule is 'problem'. Rest trials are 'skip'.
%   The number of signal peaks is stored for inspection and does not enter the verdict.

    p = inputParser;
    addParameter(p, 'goni_dir', '${GAIT_PROJECT_ROOT}/data/prep_data/healthy/goni', @ischar);
    addParameter(p, 'pattern',  'SUB_*_goni.mat', @ischar);
    parse(p, varargin{:}); opt = p.Results;

    K_MAD = 3 * 1.4826;

    files = dir(fullfile(opt.goni_dir, opt.pattern));
    fprintf('Found %d goni files (pattern=%s)\n', numel(files), opt.pattern);

    for fi = 1:numel(files)
        fpath = fullfile(opt.goni_dir, files(fi).name);
        label = regexprep(files(fi).name, '_goni\.mat$', '');
        G = load(fpath);
        if ~isfield(G,'trials') || isempty(G.trials)
            fprintf('  [%2d] %s skip (no trials)\n', fi, label); continue;
        end
        if isfield(G,'goni_labels'), goni_labels = G.goni_labels;
        else, goni_labels = G.trials(1).goni_labels; end

        % --- find Walker X (imagine active) and Spectator X (walk active) ---
        w_x = []; s_x = [];
        for j = 1:numel(goni_labels)
            lbl = goni_labels{j};
            if contains(lbl,'W') && endsWith(lbl,'X'), w_x(end+1)=j; end %#ok<AGROW>
            if contains(lbl,'S') && endsWith(lbl,'X'), s_x(end+1)=j; end %#ok<AGROW>
        end

        if isfield(G,'goni_srate'), srate = G.goni_srate; else, srate = 1000; end
        maxlag = round(0.5 * srate);

        n = numel(G.trials);
        act    = nan(1,n);
        npts   = nan(1,n);
        npeaks = nan(1,n);
        shape_corr = nan(1,n);
        for k = 1:n
            seg = double(G.trials(k).joint_angles);
            npts(k) = size(seg,1);
            switch G.trials(k).type
                case 'imagine', ch = w_x;
                case 'walk',    ch = s_x;
                otherwise,      ch = [];
            end
            if ~isempty(ch)
                stds = arrayfun(@(c) std(seg(:,c),'omitnan'), ch);
                act(k) = mean(stds, 'omitnan');
                pkc = zeros(1, numel(ch));
                for ci = 1:numel(ch)
                    sig = seg(:, ch(ci)); sig = sig - median(sig, 'omitnan');
                    [pks, ~] = findpeaks(sig, 'MinPeakProminence', 5);
                    pkc(ci) = sum(pks > 0);
                end
                npeaks(k) = mean(pkc);
            end
        end

        % --- per-condition MAD on active_std + length ---
        types = {G.trials.type};
        [thr_mi, dlo_mi, dhi_mi, med_mi, dmed_mi] = mad_thr(act(strcmp(types,'imagine')), ...
            npts(strcmp(types,'imagine')), K_MAD);
        [thr_w,  dlo_w,  dhi_w,  med_w,  dmed_w]  = mad_thr(act(strcmp(types,'walk')), ...
            npts(strcmp(types,'walk')), K_MAD);

        % --- shape (xcorr template) per condition ---
        shape_corr(strcmp(types,'imagine')) = compute_shape_corr( ...
            G.trials, find(strcmp(types,'imagine')), w_x, maxlag);
        shape_corr(strcmp(types,'walk'))    = compute_shape_corr( ...
            G.trials, find(strcmp(types,'walk')),    s_x, maxlag);
        [shape_thr_mi, shape_med_mi] = mad_lo_thr(shape_corr(strcmp(types,'imagine')), K_MAD);
        [shape_thr_w,  shape_med_w]  = mad_lo_thr(shape_corr(strcmp(types,'walk')),    K_MAD);

        qc = struct('trial_idx',{},'type',{},'status',{},'reason',{}, ...
                    'active_std',{},'baseline',{},'n_pts',{},'n_peaks',{},'dur_median',{}, ...
                    'shape_corr',{},'shape_thr',{});

        n_flat=[0 0]; n_len=[0 0]; n_shape=[0 0];   % [imagine walk]
        for k = 1:n
            qc(k).trial_idx  = k;
            qc(k).type       = G.trials(k).type;
            qc(k).active_std = act(k);
            qc(k).n_pts      = npts(k);
            qc(k).n_peaks    = npeaks(k);
            qc(k).shape_corr = shape_corr(k);
            switch G.trials(k).type
                case 'imagine'
                    [qc(k).status, qc(k).reason, fl] = verdict(act(k), thr_mi, ...
                        npts(k), dlo_mi, dhi_mi, shape_corr(k), shape_thr_mi);
                    qc(k).baseline   = med_mi;
                    qc(k).dur_median = dmed_mi;
                    qc(k).shape_thr  = shape_thr_mi;
                    n_flat(1)=n_flat(1)+fl(1); n_len(1)=n_len(1)+fl(2); n_shape(1)=n_shape(1)+fl(3);
                case 'walk'
                    [qc(k).status, qc(k).reason, fl] = verdict(act(k), thr_w, ...
                        npts(k), dlo_w, dhi_w, shape_corr(k), shape_thr_w);
                    qc(k).baseline   = med_w;
                    qc(k).dur_median = dmed_w;
                    qc(k).shape_thr  = shape_thr_w;
                    n_flat(2)=n_flat(2)+fl(1); n_len(2)=n_len(2)+fl(2); n_shape(2)=n_shape(2)+fl(3);
                case 'rest'
                    qc(k).baseline=0; qc(k).dur_median=0; qc(k).shape_thr=NaN;
                    qc(k).status='skip'; qc(k).reason='rest';
                otherwise
                    qc(k).status='skip'; qc(k).reason=['unknown type:' G.trials(k).type];
                    qc(k).baseline=NaN; qc(k).dur_median=NaN; qc(k).shape_thr=NaN;
            end
        end

        G.qc = qc;
        G.qc_method = 'mad_flat_length_shape';
        n_problem = sum(strcmp({qc.status},'problem'));
        G.n_problems = n_problem;
        save(fpath, '-struct', 'G');
        fprintf('  [%2d/%d] %s I:flat=%d len=%d shape=%d | W:flat=%d len=%d shape=%d | total=%d\n', ...
            fi, numel(files), label, n_flat(1), n_len(1), n_shape(1), ...
            n_flat(2), n_len(2), n_shape(2), n_problem);
    end
    fprintf('\n=== recompute_goni_qc done ===\n');
end


function [thr, med] = mad_lo_thr(v, K)
    v = v(~isnan(v)); thr=NaN; med=NaN;
    if isempty(v), return; end
    med = median(v);
    emad = max(mad(v,1), 0.05*max(med, 0.01));
    thr = med - K*emad;
end


function corrs = compute_shape_corr(trials, idx, chs, maxlag)
    n = length(idx); corrs = nan(1, n);
    if n < 5 || isempty(chs), return; end
    L = min(arrayfun(@(t) size(t.joint_angles,1), trials(idx)));
    if L < 1000, return; end
    nch = length(chs);
    X = nan(n, L, nch);
    for ii = 1:n
        seg = double(trials(idx(ii)).joint_angles(1:L, chs));
        X(ii,:,:) = detrend(seg, 'constant');
    end
    sc = nan(n, nch);
    for ii = 1:n
        others = setdiff(1:n, ii);
        for ch = 1:nch
            tmpl = squeeze(mean(X(others,:,ch), 1, 'omitnan'));
            x = squeeze(X(ii,:,ch));
            if std(x)==0 || std(tmpl)==0, continue; end
            c = xcorr(x - mean(x), tmpl - mean(tmpl), maxlag, 'coeff');
            sc(ii, ch) = max(c);
        end
    end
    corrs = mean(sc, 2, 'omitnan')';
end


function [thr, dlo, dhi, med, dmed] = mad_thr(act, npts, K)
    act = act(~isnan(act)); npts = npts(~isnan(npts));
    thr=NaN; dlo=NaN; dhi=NaN; med=NaN; dmed=NaN;
    if ~isempty(act)
        med = median(act);
        emad = max(mad(act,1), 0.05*med);
        thr  = med - K*emad;
    end
    if ~isempty(npts)
        dmed = median(npts);
        edmad = max(mad(npts,1), 0.05*dmed);
        dlo = dmed - K*edmad;
        dhi = dmed + K*edmad;
    end
end


function [status, reason, fl] = verdict(act, thr, npts, dlo, dhi, shape, shape_thr)
    SHAPE_ABS_FLOOR = 0.5;   % shape outlier must satisfy BOTH MAD and absolute floor
    is_flat  = ~isnan(thr) && act < thr;
    is_len   = ~isnan(dlo) && (npts < dlo || npts > dhi);
    is_shape = nargin >= 7 && ~isnan(shape_thr) && ~isnan(shape) ...
               && shape < shape_thr && shape < SHAPE_ABS_FLOOR;
    fl = [is_flat is_len is_shape];
    reasons = {};
    if is_flat,  reasons{end+1}=sprintf('flat: std=%.2f<%.2f', act, thr); end
    if is_len,   reasons{end+1}=sprintf('length: n=%d vs [%.0f %.0f]', npts, dlo, dhi); end
    if is_shape, reasons{end+1}=sprintf('shape: corr=%.2f<min(%.2f,%.2f)', shape, shape_thr, SHAPE_ABS_FLOOR); end
    if ~isempty(reasons)
        status='problem'; reason=strjoin(reasons,'; ');
    else
        status='ok'; reason='';
    end
end
