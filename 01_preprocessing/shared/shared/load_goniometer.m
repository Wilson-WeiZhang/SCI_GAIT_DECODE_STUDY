function goni = load_goniometer(goni_file)
%% load_goniometer - Load Biometrics goniometer data from enggunit.txt
%
% Usage:
%   goni = load_goniometer('path/to/file_enggunit.txt')
%
% Output struct:
%   goni.data     - [N x nCh] double, joint angles in degrees
%   goni.labels   - {1 x nCh} cell, channel names (e.g. 'LHipW X', 'Stim')
%   goni.srate    - sampling rate (1000 Hz)
%   goni.stim     - [N x 1] stim channel (extracted separately)
%   goni.stim_col - column index of Stim channel
%   goni.file     - source filename
%
% Channel naming convention:
%   L/R = Left/Right
%   Hip/Kne/Ank = Hip/Knee/Ankle
%   W/S = Walker/Subject
%   X/Y axis = two measurement axes

%% Read file (UTF-16LE encoded)
fid = fopen(goni_file, 'r', 'n', 'UTF-16LE');
if fid == -1
    error('Cannot open file: %s', goni_file);
end

% Parse header lines to get channel names
labels = {};
header_lines = 0;
while ~feof(fid)
    line = fgetl(fid);
    if isempty(line), continue; end

    % Check if this is a data row (starts with number or minus sign)
    stripped = strtrim(line);
    if ~isempty(stripped) && (stripped(1) >= '0' && stripped(1) <= '9') || stripped(1) == '-'
        break;
    end

    % Parse channel header: "Channel N: 'Name', ..."
    tok = regexp(line, '''([^'']+)''', 'tokens');
    if ~isempty(tok)
        labels{end+1} = tok{1}{1};
    end
    header_lines = header_lines + 1;
end
fclose(fid);

n_ch = length(labels);
fprintf('  Goniometer: %d channels, header=%d lines\n', n_ch, header_lines);

%% Read numeric data
% Use readmatrix with UTF-16 handling
raw = readmatrix(goni_file, 'FileType', 'text', 'Encoding', 'UTF-16LE', ...
    'NumHeaderLines', header_lines, 'Delimiter', '\t');

% Remove rows with all NaN
raw = raw(~all(isnan(raw), 2), :);

% Ensure column count matches
if size(raw, 2) > n_ch
    raw = raw(:, 1:n_ch);
elseif size(raw, 2) < n_ch
    warning('Data has %d columns but header has %d channels', size(raw, 2), n_ch);
    labels = labels(1:size(raw, 2));
    n_ch = size(raw, 2);
end

%% Clean up labels: shorten for readability
short_labels = cell(1, n_ch);
for i = 1:n_ch
    lbl = labels{i};
    % Extract: sensor name + axis (e.g. 'LHipW [sensor id] X axis' -> 'LHipW X')
    tok = regexp(lbl, '^(\w+)\s+K\d+\s+(X|Y)\s+axis', 'tokens');
    if ~isempty(tok)
        short_labels{i} = [tok{1}{1} ' ' tok{1}{2}];
    elseif contains(lbl, 'Stim')
        short_labels{i} = 'Stim';
    else
        short_labels{i} = lbl;
    end
end

%% Identify Stim channel
stim_col = find(strcmp(short_labels, 'Stim'));
if isempty(stim_col)
    stim_col = find(contains(labels, 'Stim'), 1);
end

if ~isempty(stim_col)
    stim = raw(:, stim_col);
    % Remove Stim from data columns
    data_cols = setdiff(1:n_ch, stim_col);
else
    stim = [];
    data_cols = 1:n_ch;
    warning('No Stim channel found');
end

%% Build output struct
goni.data       = raw(:, data_cols);
goni.labels     = short_labels(data_cols);
goni.labels_raw = labels(data_cols);
goni.srate      = 1000;  % Biometrics standard
goni.stim       = stim;
goni.stim_col   = stim_col;
goni.n_samples  = size(raw, 1);
goni.n_channels = length(data_cols);
goni.file       = goni_file;

fprintf('  Goniometer loaded: %d samples (%.1f sec), %d joint channels + stim\n', ...
    goni.n_samples, goni.n_samples / goni.srate, goni.n_channels);
end
