function info = discover_patient_paths(raw_root, subject, session_num, opts)
%DISCOVER_PATIENT_PATHS  Locate the raw .vhdr and goniometer .txt files of a patient session.
%
%   Default layout of the recording archive:
%     {raw_root}/SUBJECT-{NN}/P2_{NNN}_Sess{NN}/sess{NN}_{date}-{time}/EEG/*.vhdr
%     {raw_root}/SUBJECT-{NN}/P2_{NNN}_Sess{NN}/sess{NN}_{date}-{time}/Goniometer/*eng*unit*.txt
%
%   Sessions stored under a different folder name are resolved through `opts`:
%     opts.subj_subdir   relative to raw_root      default 'SUBJECT-{NN}'
%     opts.sess_subdir   relative to subj_subdir.  default 'P2_{NNN}_Sess{NN}'.
%                        '' means the timestamp folders sit directly under subj_subdir
%     opts.ts_glob       glob for timestamp folders.  default 'sess{NN}_*'
%
%   Output struct:
%     .vhdr_paths    cell array of .vhdr full paths (sorted by name)
%     .goni_paths    cell array of CELL ARRAYS (one per timestamp folder; each
%                    inner cell lists 0+ goniometer .txt files in its
%                    Goniometer/ subfolder)
%     .rec_dirs      cell array of timestamped recording folder paths

    if nargin < 4, opts = struct(); end

    raw_root = char(raw_root);
    subject  = char(subject);

    % Defaults. Patient subj dirs are 2-digit (`SUBJECT-02`); P2 dirs are 3-digit (`P2_*`).
    subj_num            = str2double(subject(2:end));
    subj_subdir_default = sprintf('SUBJECT-%02d', subj_num);
    sess_subdir_default = sprintf('P2_%03d_Sess%02d', subj_num, session_num);
    ts_glob_default     = sprintf('sess%02d_*', session_num);

    opts = set_default(opts, 'subj_subdir', subj_subdir_default);
    opts = set_default(opts, 'sess_subdir', sess_subdir_default);
    opts = set_default(opts, 'ts_glob',     ts_glob_default);

    if isempty(opts.sess_subdir)
        ts_root = fullfile(raw_root, opts.subj_subdir);
    else
        ts_root = fullfile(raw_root, opts.subj_subdir, opts.sess_subdir);
    end
    if ~exist(ts_root, 'dir')
        error('discover_patient_paths:noTsRoot', ...
            'Timestamp root not found: %s', ts_root);
    end

    rec_dirs = dir(fullfile(ts_root, opts.ts_glob));
    rec_dirs = rec_dirs([rec_dirs.isdir]);
    if isempty(rec_dirs)
        error('discover_patient_paths:noRecDirs', ...
            'No timestamp dirs matching %s in %s', opts.ts_glob, ts_root);
    end

    % Sort by timestamp HHMMSS (extracted as \d{6} after last '-'), not by
    % alphabetical name -- because folder names do not always follow the layout (a session can have
    % sess03_*113045 chronologically first, before sess03_*121640) would
    % otherwise mis-order multi-vhdr merges.
    ts_keys = nan(1, length(rec_dirs));
    for r = 1:length(rec_dirs)
        tok = regexp(rec_dirs(r).name, '-(\d{6})', 'tokens', 'once');
        if ~isempty(tok), ts_keys(r) = str2double(tok{1}); end
    end
    if any(isnan(ts_keys))
        [~, ord] = sort({rec_dirs.name});
    else
        [~, ord] = sort(ts_keys);
    end
    rec_dirs = rec_dirs(ord);

    vhdr_paths = {};
    goni_paths = {};
    rec_paths  = {};
    for r = 1:length(rec_dirs)
        rec_path = fullfile(rec_dirs(r).folder, rec_dirs(r).name);
        rec_paths{end+1} = rec_path; %#ok<AGROW>

        vh = dir(fullfile(rec_path, 'EEG', '*.vhdr'));
        if isempty(vh)
            vhdr_paths{end+1} = ''; %#ok<AGROW>
        else
            vhdr_paths{end+1} = fullfile(vh(1).folder, vh(1).name); %#ok<AGROW>
        end

        % Goni files per recording: usually 1, sometimes 0 (recorder dropped)
        % or multiple (recorder restarted mid-recording).
        % Filter out -CBCR-C1.txt backup files (duplicate of the main file
        % seen in pat01_sess07).
        gn = dir(fullfile(rec_path, 'Goniometer', '*eng*unit*.txt'));
        gn = gn(arrayfun(@(g) isempty(regexp(g.name, '-CBCR-C1\.txt$', 'once')), gn));
        if isempty(gn)
            goni_paths{end+1} = {}; %#ok<AGROW>
        else
            [~, gord] = sort({gn.name});
            gn = gn(gord);
            paths_here = arrayfun(@(g) fullfile(g.folder, g.name), gn, 'UniformOutput', false);
            goni_paths{end+1} = paths_here(:)'; %#ok<AGROW>
        end
    end

    info = struct( ...
        'vhdr_paths', {vhdr_paths}, ...
        'goni_paths', {goni_paths}, ...
        'rec_dirs',   {rec_paths});
end


function s = set_default(s, f, default)
    % Only fall back when the field is missing. An explicit empty string is a
    % valid override (e.g. pat02_sess03 sets sess_subdir='' for the orphan dir
    % living directly under SUBJECT-02).
    if ~isfield(s, f)
        s.(f) = default;
    end
end
