function cpc_step1(sid, plan_csv, out_root, g1_only)
%CPC_STEP1  Causal step 1 of one healthy session, the reproduction check G1, and the calibration copy.
%
%   The step-1 alignment (01_preprocessing/healthy/01_align/prep_align.m) changes the EEG samples only by
%   pop_resample(EEG, 250) of the 59 analysed channels, per raw part; sessions recorded in several parts
%   are then concatenated in a fixed order. For each raw part of the plan (plan.csv step1_parts):
%     G1      pop_resample of the raw channels must equal the matching block of the merged step-1
%             recording bit for bit, and the blocks must tile it exactly;
%     causal  pop_eegfiltnew low-pass at 100 Hz, minimum phase, at 1000 Hz, then input sample 4(k-1)+1 ->
%             output sample k within each continuous stretch (the pop_resample grid, so every event
%             latency is unchanged; a recording with a data break is resampled stretch by stretch).
%   The causal step 1 is the merged step-1 recording (events, channel locations, boundaries, alignment
%   log) with its data replaced by the causal blocks. The calibration copy is the causal step 1 cut at
%   plan.cut_step1; markers of a later segment that fall inside it are removed.
%   Writes <out_root>/causal/eeg/<SID>_step1.set, <out_root>/cal/<SID>_step1.set and
%   <out_root>/info/<SID>_step1_info.txt. g1_only true: G1 only, nothing else written.
%   usage: cpc_step1(sid, plan_csv, out_root, g1_only)
if nargin < 4, g1_only = false; end
if ischar(g1_only) || isstring(g1_only), g1_only = any(strcmpi(char(g1_only), {'1', 'true'})); end
E = cpc_env();
addpath(E.eeglab);
eeglab('nogui');
r = plan_row(plan_csv, sid);
parts = strsplit(r.step1_parts, ';');
if isempty(parts{1}), parts = {sid}; end
raw_root = fullfile(E.root, 'data', 'raw_data');
addpath(fullfile(E.root, '01_preprocessing', 'healthy', 'shared'));      % build_session_registry (frozen, read only)
reg = build_session_registry(raw_root);
rmpath(fullfile(E.root, '01_preprocessing', 'healthy', 'shared'));
plan_vhdr = strsplit(r.raw_vhdr, ';');
asrun_dir = fullfile(E.root, 'data', 'prep_data', 'healthy', 'eeg');
A = pop_loadset('filename', [sid '_step1.set'], 'filepath', asrun_dir);
A = eeg_checkset(A);
assert(A.pnts == r.step1_pnts, 'cpc_step1: as-run step1 has %d points, plan %d', A.pnts, r.step1_pnts);
info = {sprintf('host %s; MATLAB %s; EEGLAB %s; maxNumCompThreads %d; root %s', E.host, version, eeg_getversion, maxNumCompThreads, E.root), sprintf('%s: as-run merged step1 %d ch x %d pts @ %g Hz, class %s; parts %s', sid, A.nbchan, A.pnts, A.srate, ...
                class(A.data), strjoin(parts, ','))};
Cdata = zeros(size(A.data), 'like', A.data);
col = 0;
baa = [];
for p = 1:numel(parts)
    k = find(strcmp({reg.label}, parts{p}));
    assert(numel(k) == 1, 'cpc_step1: %s not (uniquely) in the registry', parts{p});
    [vd, vn, ve] = fileparts(reg(k).vhdr);
    assert(endsWith(reg(k).vhdr, plan_vhdr{p}), 'cpc_step1: registry vhdr %s differs from the plan %s', reg(k).vhdr, plan_vhdr{p});
    R = pop_loadbv(vd, [vn ve]);
    R = eeg_checkset(R);
    assert(R.srate == 1000, 'raw %s is not 1000 Hz', parts{p});
    dipfit_dir = fileparts(which('pop_dipfit_settings'));
    R = pop_chanedit(R, 'lookup', fullfile(dipfit_dir, 'standard_BEM', 'elec', 'standard_1005.elc'));
    R = eeg_checkset(R);
    R = pop_select(R, 'channel', find(ismember({R.chanlocs.labels}, {A.chanlocs.labels})));
    R = eeg_checkset(R);
    assert(isequal({R.chanlocs.labels}, {A.chanlocs.labels}), 'raw channel order of %s differs from the as-run step1', parts{p});
    % G1
    G = pop_resample(R, 250);
    n = G.pnts;
    assert(col + n <= A.pnts, 'G1: part %s overruns the as-run step1 (%d + %d > %d)', parts{p}, col, n, A.pnts);
    blk = A.data(:, col + (1:n));
    g1 = max(abs(double(G.data(:)) - double(blk(:))));
    info{end+1} = sprintf('G1 part %s: raw %d pts @ 1000 Hz -> %d pts, block %d-%d of the merged step1; max|pop_resample(raw) - as-run| = %.3g (max|step1| %.3g); bit-identical %d', ...
        parts{p}, R.pnts, n, col + 1, col + n, g1, max(abs(double(blk(:)))), isequal(cast(G.data, class(blk)), blk)); %#ok<AGROW>
    assert(g1 <= 1e-6 * max(abs(double(blk(:)))), 'G1 failed for %s', parts{p});
    clear G blk
    % causal anti-aliasing + decimation
    [Rf, ~, b] = pop_eegfiltnew(R, 'hicutoff', 100, 'minphase', 1);
    if isempty(baa), baa = b; else, assert(isequal(b, baa), 'anti-alias filter differs between parts'); end
    % decimation grid = pop_resample's: it resamples every continuous stretch between data breaks ('boundary' events)
    % on its own (pop_resample.m:129-138, 167-169), output k of stretch j <- input b_j + 4(k-1), ceil(len_j / 4)
    % samples; pop_eegfiltnew filters the stretches separately too. One stretch (no interior break) = 1:4:end.
    bi = eeg_findboundaries(R.event);
    bnd = [];
    if ~isempty(bi)
        bnd = [R.event(bi).latency];
        bnd(bnd <= 0 | bnd > size(R.data, 2)) = [];
        bnd(mod(bnd, 1) ~= 0) = round(bnd(mod(bnd, 1) ~= 0) + 0.5);
    end
    bnd = unique([1 bnd size(R.data, 2) + 1]);
    idx = cell(1, numel(bnd) - 1);
    for j = 1:numel(bnd) - 1
        idx{j} = bnd(j) + 4 * (0:ceil((bnd(j + 1) - bnd(j)) / 4) - 1);
    end
    idx = [idx{:}];
    assert(numel(idx) == n && idx(end) <= Rf.pnts, 'raw part %s: decimation grid %d samples, step1 block %d', ...
           parts{p}, numel(idx), n);
    info{end+1} = sprintf('part %s: %d continuous stretch(es) between data breaks (raw starts %s)', parts{p}, ...
                          numel(bnd) - 1, mat2str(bnd(1:end-1))); %#ok<AGROW>
    Cdata(:, col + (1:n)) = cast(Rf.data(:, idx), class(A.data));
    col = col + n;
    clear R Rf
end
assert(col == A.pnts, 'G1: the parts give %d points, the as-run merged step1 has %d', col, A.pnts);
info{end+1} = sprintf('G1 passed: %d part(s) tile the as-run merged step1 (%d pts)', numel(parts), col);
w = 2 * pi * [0.9 1.8 2.7 10] / 1000;
gd = arrayfun(@(x) grpdelay(baa, 1, [x x]), w, 'UniformOutput', false);
info{end+1} = sprintf('anti-alias taps %d; group delay (ms) at 0.9/1.8/2.7/10 Hz: %s', numel(baa), mat2str(cellfun(@(g) g(1), gd), 4));
info_dir = fullfile(out_root, 'info');
if ~exist(info_dir, 'dir'), mkdir(info_dir); end
if g1_only
    write_info(fullfile(info_dir, [sid '_step1_g1only_info.txt']), info);
    return
end
assert(r.approved == 1, 'cpc_step1: %s is not approved in %s (review the plan first)', sid, plan_csv);

C = A;
C.data = Cdata;
clear Cdata
C.etc.causal_step1 = struct('anti_alias', 'pop_eegfiltnew hicutoff 100 Hz, minphase 1, at 1000 Hz (causal)', ...
    'taps', numel(baa), 'decimation', 'output k <- input 4(k-1)+1 (pop_resample alignment)', 'parts', {parts});
C = eeg_checkset(C);
ce = fullfile(out_root, 'causal', 'eeg');
if ~exist(ce, 'dir'), mkdir(ce); end
pop_saveset(C, 'filename', [sid '_step1.set'], 'filepath', ce, 'savemode', 'twofiles');

% calibration copy
cut = r.cut_step1;
types = {C.event.type};
lats = [C.event.latency];
s11 = find(strcmp(types, 'S 11'), 1, 'first');
if isempty(s11), s11 = 1; else, s11 = round(lats(s11)); end
assert(cut > s11 && cut < C.pnts, 'cut %d outside (S11 %d, pnts %d)', cut, s11, C.pnts);
starts = {'S  1', 'S  4', 'S  7'};
ends = {'S  2', 'S  5', 'S  8'};
for k = 1:3
    s = lats(strcmp(types, starts{k}) & lats >= s11 & lats < cut);
    e = lats(strcmp(types, ends{k}));
    for i = 1:numel(s)
        nxt = e(e > s(i));
        assert(~isempty(nxt) && nxt(1) < cut, 'a %s segment straddles the calibration cut', starts{k});
    end
end
n_mi_cal = sum(strcmp(types, 'S  1') & lats >= s11 & lats < cut);
Ccal = pop_select(C, 'point', [1 cut]);
% The frozen prep_step2 trims from the first 'S 11' to the LAST 'S 12' (trim_s11_s12, prep_step2.m:332-336). In a
% multi-part session an earlier part's end marker 'S 12' can lie inside the calibration copy (SUB_09_sess02: part a's S12
% at 455.7 s, cut at 1304.6 s), which would trim the calibration recording back to part a. Every 'S 12' inside the copy
% is such an internal end marker (the session's own S12 lies after the cut), so they are removed: prep_step2 then trims
% from the first S11 to the end of the copy, i.e. it uses exactly the planned calibration recording.
is12 = strcmp({Ccal.event.type}, 'S 12');
n12 = sum(is12);
Ccal.event(is12) = [];
Ccal = eeg_checkset(Ccal);
cd_ = fullfile(out_root, 'cal');
if ~exist(cd_, 'dir'), mkdir(cd_); end
pop_saveset(Ccal, 'filename', [sid '_step1.set'], 'filepath', cd_, 'savemode', 'twofiles');
info{end+1} = sprintf('calibration copy: %d internal ''S 12'' marker(s) removed', n12);
info{end+1} = sprintf('calibration cut at step1 point %d (%.2f s; plan cut_export_s %.2f, n_cal %d); S1 starts inside (any qc): %d; calibration copy %d pts', ...
                      cut, (cut - 1) / C.srate, r.cut_export_s, r.n_cal, n_mi_cal, Ccal.pnts);
write_info(fullfile(info_dir, [sid '_step1_info.txt']), info);
end

function r = plan_row(plan_csv, sid)
o = detectImportOptions(plan_csv, 'Delimiter', ',');
o = setvartype(o, {'sid', 'session', 'step1_parts', 'raw_vhdr'}, 'char');
P = readtable(plan_csv, o);
k = find(strcmp(P.sid, sid));
assert(numel(k) == 1, 'cpc_step1: %s not (uniquely) in %s', sid, plan_csv);
r = table2struct(P(k, :));
end

function write_info(path, info)
fid = fopen(path, 'w');
fprintf(fid, '%s\n', info{:});
fclose(fid);
fprintf('%s\n', info{:});
end
