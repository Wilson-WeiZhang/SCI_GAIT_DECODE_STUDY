function cpp_step1(label, plan_csv, out_root, g1_only)
%CPP_STEP1  Causal step 1 of one patient session, the reproduction check G1, and the calibration copy.
%
%   The patient step-1 alignment (01_preprocessing/patient/01_align/) changes the EEG samples only by
%   loading and merging the raw recordings in registry order at 1000 Hz (for the sessions with the
%   connector fix, the fix right after the merge), the standard_1005 channel lookup, the 59-channel
%   selection and pop_resample(EEG, 250), each boundary-delimited segment separately. The same load and
%   merge is rebuilt here from the raw files and
%     G1      its pop_resample must equal the frozen step-1 data bit for bit;
%     causal  pop_eegfiltnew low-pass at 100 Hz, minimum phase, on the merged 1000 Hz data (restarted at every
%             boundary), then every fourth sample on the pop_resample grid of each segment, so every event
%             latency is unchanged.
%   The causal step 1 is the frozen step 1 of the session's lane with its data replaced; the calibration
%   copy is it cut at plan.cut_step1. Writes <out_root>/causal/eeg/<label>_step1.set,
%   <out_root>/cal/<label>_step1.set and <out_root>/info/<label>_step1_info.txt.
%   usage: cpp_step1(label, plan_csv, out_root, g1_only)
if nargin < 4, g1_only = false; end
if ischar(g1_only) || isstring(g1_only), g1_only = any(strcmpi(char(g1_only), {'1', 'true'})); end
addpath(getenv("CPC_EEGLAB"));
eeglab('nogui');
prep_patient = fullfile(getenv('CPC_ROOT'), '01_preprocessing', 'patient');
addpath(fullfile(prep_patient, 'shared'));
addpath(fullfile(prep_patient, '01_align'));
prep_patient_paths();
assert(strcmp(which('apply_module_connector_fix'), fullfile(prep_patient, '01_align', 'apply_module_connector_fix.m')), 'apply_module_connector_fix path');
assert(strcmp(which('build_patient_registry'), fullfile(prep_patient, 'shared', 'build_patient_registry.m')), 'registry path');
r = plan_row(plan_csv, label);
reg = build_patient_registry(fullfile(getenv('GAIT_RAW_ROOT'), 'patient'));
k = find(strcmp({reg.label}, label));
assert(numel(k) == 1, 'cpp_step1: %s not (uniquely) in the registry', label);
vhdrs = reg(k).vhdr_paths;
assert(~isempty(vhdrs), 'cpp_step1: registry has no vhdr for %s', label);
A = pop_loadset('filename', [label '_step1.set'], 'filepath', r.step1_dir);
A = eeg_checkset(A);
assert(A.pnts == r.step1_pnts, 'cpp_step1: as-run step1 has %d points, plan %d', A.pnts, r.step1_pnts);
info = {sprintf('%s: lane %s, as-run step1 %s/%s_step1.set %d ch x %d pts @ %g Hz, class %s; connector_fix %d; %d vhdr(s):', ...
                label, r.lane, r.step1_dir, label, A.nbchan, A.pnts, A.srate, class(A.data), r.connector_fix, numel(vhdrs))};
info = [info, cellfun(@(v) ['    ' v], vhdrs(:)', 'UniformOutput', false)];

% load + merge exactly as load_and_merge_vhdrs (prep_align_patient.m:182-203) / the legacy aligners
R = [];
for i = 1:numel(vhdrs)
    [d, nm] = fileparts(vhdrs{i});
    Ri = pop_loadbv(d, [nm '.vhdr']);
    Ri = eeg_checkset(Ri);
    assert(Ri.srate == 1000, 'raw %s is not 1000 Hz', vhdrs{i});
    if isempty(R), R = Ri; else, R = pop_mergeset(R, Ri, 0); end
    clear Ri
end
R = eeg_checkset(R);
want_swap = logical(r.connector_fix);
assert(want_swap == connector_fix_sessions(label), 'plan connector_fix %d disagrees with connector_fix_sessions(%s)', want_swap, label);
assert(want_swap == (isfield(A.etc, 'connector_fix_applied') && ~isempty(A.etc.connector_fix_applied)), ...
       'as-run step1 etc.connector_fix_applied disagrees with plan connector_fix %d', want_swap);
if want_swap
    R = apply_module_connector_fix(R, label);
end
dipfit_dir = fileparts(which('pop_dipfit_settings'));
R = pop_chanedit(R, 'lookup', fullfile(dipfit_dir, 'standard_BEM', 'elec', 'standard_1005.elc'));
R = eeg_checkset(R);
R = pop_select(R, 'channel', find(ismember({R.chanlocs.labels}, prep_patient_brain_whitelist())));
R = eeg_checkset(R);
assert(isequal({R.chanlocs.labels}, {A.chanlocs.labels}), 'raw channel order differs from the as-run step1');

% G1
G = pop_resample(R, 250);
assert(G.pnts == A.pnts, 'G1: pop_resample gives %d points, as-run step1 has %d', G.pnts, A.pnts);
g1 = max(abs(double(G.data(:)) - double(A.data(:))));
info{end+1} = sprintf('G1 max|pop_resample(raw merge) - as-run step1| = %.3g (max|step1| %.3g); bit-identical %d', ...
                      g1, max(abs(double(A.data(:)))), isequal(cast(G.data, class(A.data)), A.data));
assert(g1 <= 1e-6 * max(abs(double(A.data(:)))), 'G1 failed for %s', label);
clear G

% segment starts exactly as pop_resample computes them (eeglab2024 pop_resample.m:130-140)
tmpevent = R.event;
bi = eeg_findboundaries(tmpevent);
if isempty(bi)
    bounds = [1 R.pnts + 1];
else
    bounds = [tmpevent(bi).latency];
    bounds(bounds <= 0 | bounds > R.pnts) = [];
    nonint = mod(bounds, 1) ~= 0;
    bounds(nonint) = round(bounds(nonint) + 0.5);
    bounds = unique([1 bounds R.pnts + 1]);
end
dc = findboundaries(R.event);            % what firfiltsplit uses
dc = unique(dc(dc >= 1 & dc <= R.pnts));
assert(isequal(dc(:)', bounds(1:end-1)), 'segment starts: pop_resample %s vs firfiltsplit %s', mat2str(bounds), mat2str(dc));

% causal anti-aliasing + decimation
[Rf, ~, baa] = pop_eegfiltnew(R, 'hicutoff', 100, 'minphase', 1);
clear R
idx = [];
for s = 1:numel(bounds) - 1
    n_in = bounds(s + 1) - bounds(s);
    n_out = ceil(n_in / 4);              % resample(): ceil(n * p / q) points per segment
    idx = [idx, bounds(s) + 4 * (0:n_out - 1)]; %#ok<AGROW>
end
assert(numel(idx) == A.pnts && idx(end) <= Rf.pnts, 'decimation gives %d points, as-run step1 %d', numel(idx), A.pnts);
Cdata = cast(Rf.data(:, idx), class(A.data));
clear Rf
info{end+1} = sprintf('G1 passed; %d pop_resample segment(s) at 1000 Hz starting %s', numel(bounds) - 1, mat2str(bounds(1:end-1)));
w = 2 * pi * [0.9 1.8 2.7 10] / 1000;
gd = arrayfun(@(x) grpdelay(baa, 1, [x x]), w, 'UniformOutput', false);
info{end+1} = sprintf('anti-alias taps %d; group delay (ms) at 0.9/1.8/2.7/10 Hz: %s', numel(baa), mat2str(cellfun(@(g) g(1), gd), 4));
info_dir = fullfile(out_root, 'info');
if ~exist(info_dir, 'dir'), mkdir(info_dir); end
if g1_only
    write_info(fullfile(info_dir, [label '_step1_g1only_info.txt']), info);
    return
end
assert(r.approved == 1, 'cpp_step1: %s is not approved in %s (review the plan first)', label, plan_csv);

C = A;
C.data = Cdata;
clear Cdata
C.etc.causal_step1 = struct('anti_alias', 'pop_eegfiltnew hicutoff 100 Hz, minphase 1, at 1000 Hz (causal, per boundary segment)', ...
    'taps', numel(baa), 'decimation', 'per pop_resample segment: output k <- input b + 4(k-1)', 'vhdrs', {vhdrs}, ...
    'connector_fix', want_swap, 'lane', r.lane);
C = eeg_checkset(C);
ce = fullfile(out_root, 'causal', 'eeg');
if ~exist(ce, 'dir'), mkdir(ce); end
pop_saveset(C, 'filename', [label '_step1.set'], 'filepath', ce, 'savemode', 'twofiles');

% calibration copy
cut = r.cut_step1;
types = {C.event.type};
lats = [C.event.latency];
s11 = find(strcmp(types, 'S 11'), 1, 'first');
if isempty(s11), s11 = 1; else, s11 = round(lats(s11)); end
assert(cut > s11 && cut < C.pnts, 'cut %d outside (S11 %d, pnts %d)', cut, s11, C.pnts);
starts = {'S  1', 'S  4'};
ends = {'S  2', 'S  5'};
for q = 1:2
    s = lats(strcmp(types, starts{q}) & lats >= s11 & lats < cut);
    e = lats(strcmp(types, ends{q}));
    for i = 1:numel(s)
        nxt = e(e > s(i));
        assert(~isempty(nxt) && nxt(1) < cut, 'a %s segment straddles the calibration cut', starts{q});
    end
end
% walking segments (S6->S7 / S7->S8 / S7 + 120 s) are trial_defs rows of the standard protocol only; the legacy protocol
% uses S1->S2 alone (prep_patient prep_step2 get_patient_trial_defs), and pat01_sess01 carries a stray unpaired 'S  6' at 201 s
if ~strcmp(C.etc.marker_protocol, 'legacy')
    assert(~any(ismember(types, {'S  6', 'S  7'}) & lats < cut), 'a walking start marker lies before the calibration cut');
end
n_mi_cal = sum(strcmp(types, 'S  1') & lats >= s11 & lats < cut);
Ccal = pop_select(C, 'point', [1 cut]);
Ccal = eeg_checkset(Ccal);
cd_ = fullfile(out_root, 'cal');
if ~exist(cd_, 'dir'), mkdir(cd_); end
pop_saveset(Ccal, 'filename', [label '_step1.set'], 'filepath', cd_, 'savemode', 'twofiles');
info{end+1} = sprintf('calibration cut at step1 point %d (%.2f s; plan cut_export_s %.2f, n_cal %d, est task %.1f s); S1 starts inside (any qc): %d; calibration copy %d pts', ...
                      cut, (cut - 1) / C.srate, r.cut_export_s, r.n_cal, r.est_cal_task_s, n_mi_cal, Ccal.pnts);
write_info(fullfile(info_dir, [label '_step1_info.txt']), info);
end

function r = plan_row(plan_csv, label)
o = detectImportOptions(plan_csv, 'Delimiter', ',');
o = setvartype(o, {'sid', 'session', 'lane', 'step1_dir', 'step23_dir', 'goni_dir', 'export_fn', 'relabel_override'}, 'char');
P = readtable(plan_csv, o);
k = find(strcmp(P.sid, label));
assert(numel(k) == 1, 'cpp_step1: %s not (uniquely) in %s', label, plan_csv);
r = table2struct(P(k, :));
end

function write_info(path, info)
fid = fopen(path, 'w');
fprintf(fid, '%s\n', info{:});
fclose(fid);
fprintf('%s\n', info{:});
end
