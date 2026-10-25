function gfile = find_goni_file(eeg_in_dir, eeg_in_name) %#ok<INUSD>
%FIND_GONI_FILE  Locate raw goni .txt file for a session given its EEG dir.
%
%   Layout varies: EEG may be in SUB_XX/sessNN/EEG/ and Goni in
%   SUB_XX/sessNN_TIMESTAMP/Goniometer/. Strategy: go up to subject dir,
%   find all sibling folders starting with same sess prefix, scan each for
%   Goniometer/*enggunit*.txt (fallback *.txt).
    sess_root = fileparts(eeg_in_dir);
    subj_dir  = fileparts(sess_root);
    sess_name = get_basename(sess_root);
    tok = regexp(sess_name, '^(sess\d+)', 'tokens', 'once');
    if isempty(tok)
        error('find_goni_file: cannot parse sess prefix from %s', sess_root);
    end
    sess_prefix = tok{1};

    cand_dirs = dir(fullfile(subj_dir, [sess_prefix '*']));
    cand_dirs = cand_dirs([cand_dirs.isdir]);

    for i = 1:length(cand_dirs)
        goni_dir = fullfile(cand_dirs(i).folder, cand_dirs(i).name, 'Goniometer');
        if ~exist(goni_dir, 'dir')
            alt = dir(fullfile(cand_dirs(i).folder, cand_dirs(i).name, '[Gg]oni*'));
            if isempty(alt), continue; end
            goni_dir = fullfile(alt(1).folder, alt(1).name);
        end
        hits = dir(fullfile(goni_dir, '*enggunit*.txt'));
        if isempty(hits)
            hits = dir(fullfile(goni_dir, '*.txt'));
            hits = hits(~endsWith({hits.name}, '.log') & ~endsWith({hits.name}, '.cnt'));
        end
        if ~isempty(hits)
            gfile = fullfile(hits(1).folder, hits(1).name);
            return;
        end
    end
    error('find_goni_file: no goni .txt found under %s (sess_prefix=%s)', subj_dir, sess_prefix);
end


function n = get_basename(p)
    parts = strsplit(p, filesep);
    n = parts{end};
end
