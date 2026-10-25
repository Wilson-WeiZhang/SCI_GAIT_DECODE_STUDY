function S = load_triple_mat(label)
%LOAD_TRIPLE_MAT  Load one session's EEG, goniometer and event files into one struct.
%   S = load_triple_mat('subj07_sess01')
%
%   Translates the on-disk schema (eeg.s, eeg.c struct, goni.c cell, event) into
%   the standardised struct that every decoding entry point expects:
%     S.eeg.x       [60 x T] @ 250 Hz, 59 recorded channels + FCz
%     S.eeg.fs      sample rate (Hz)
%     S.eeg.labels  {1 x 60} cell of channel labels
%     S.goni.x      [T x 24] @ 250 Hz
%     S.goni.fs     sample rate (Hz)
%     S.goni.labels {1 x 24} cell
%     S.events      struct array (renamed from the on-disk `event`)
%                   fields: label, code, etstart, etstop, gtstart, gtstop, qc
%     S.label       echoed input
%
%   Data root: the environment variable GAIT_TRIPLE_ROOT; a built-in path is used
%   when it is unset. Inside that root each session is at
%   <root>/<label>/<label>_{eeg,goni,event}.mat, the layout written by the export
%   stage of 01_preprocessing.

data_root = getenv('GAIT_TRIPLE_ROOT');
if isempty(data_root)
    data_root = fullfile(getenv('HOME'), 'gait', 'data', 'healthy_prep');
end
base = fullfile(data_root, label);

eeg_file   = fullfile(base, [label '_eeg.mat']);
goni_file  = fullfile(base, [label '_goni.mat']);
event_file = fullfile(base, [label '_event.mat']);

assert(isfile(eeg_file),   'missing %s', eeg_file);
assert(isfile(goni_file),  'missing %s', goni_file);
assert(isfile(event_file), 'missing %s', event_file);

e = load(eeg_file);
g = load(goni_file);
v = load(event_file);

S = struct();

% EEG: pull labels out of chanlocs-style struct array
eeg_in = e.eeg;
S.eeg = struct();
S.eeg.x      = eeg_in.x;
S.eeg.fs     = eeg_in.s;
S.eeg.labels = {eeg_in.c.labels};
S.eeg.chanlocs = eeg_in.c;

% Goni: labels are already a cell array
goni_in = g.goni;
S.goni = struct();
S.goni.x      = goni_in.x;
S.goni.fs     = goni_in.s;
S.goni.labels = goni_in.c(:)';   % normalize to row cell

% Events
if isfield(v, 'event')
    S.events = v.event;
elseif isfield(v, 'events')
    S.events = v.events;
else
    error('event_file %s has no event/events field', event_file);
end
for j = 1:numel(S.events)
    S.events(j).session_label = label;
    S.events(j).trial_idx = j;
end

% Optional: preped goni layer (LP, phase_pj, phase_g_W/S, cycle_template)
% Auto-load if {label}_goni_preped.mat exists alongside triple-mat.
preped_file = fullfile(base, [label '_goni_preped.mat']);
if isfile(preped_file)
    pp = load(preped_file);
    S.preped = pp.preped;
end

S.label = label;
S = apply_external_goni_qc(S);
end

function S = apply_external_goni_qc(S)
% Overlay optional per-trial goni QC labels without rewriting export .mat files.
% Set GAIT_GONI_QC_TRIAL_FLAGS to result/qc/goni_qc_trial_flags.csv.
qc_csv = getenv('GAIT_GONI_QC_TRIAL_FLAGS');
if isempty(qc_csv)
    return;
end
if ~isfile(qc_csv)
    error('GAIT_GONI_QC_TRIAL_FLAGS does not exist: %s', qc_csv);
end

persistent cache_path cache_table
if isempty(cache_path) || ~strcmp(cache_path, qc_csv)
    cache_table = readtable(qc_csv, 'TextType', 'string');
    cache_path = qc_csv;
end

rows = cache_table(cache_table.label == string(S.label), :);
if isempty(rows)
    return;
end
for i = 1:height(rows)
    j = rows.trial(i);
    if j < 1 || j > numel(S.events)
        continue;
    end
    status = char(rows.status(i));
    switch status
        case 'ok'
            S.events(j).qc = 'good';
        case 'problem'
            S.events(j).qc = 'bad';
        case 'skip'
            S.events(j).qc = 'skip';
    end
end
end
