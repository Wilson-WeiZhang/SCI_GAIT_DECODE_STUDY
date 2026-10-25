# Preprocessing quality control

Two quality-control stages sit between the raw recordings and EEG preprocessing:
EEG-goniometer alignment, and goniometer trial quality control. Their outputs, together
with a review of every recording, decide which sessions and which trials reach the
decoder. This page describes both stages as implemented in `01_preprocessing/`, the
decisions that were taken by hand, and their effect on the analysed data.

## 1. EEG-goniometer alignment

The EEG amplifier and the goniometer logger receive the same trigger pulses. Alignment
pairs every EEG marker with its goniometer trigger edge, so that each trial can be cut
from the goniometer at its own sample positions.

Code: `shared/shared/align_goni_to_eeg.m`, called by `healthy/01_align/prep_align.m` and
`patient/01_align/prep_align_patient.m`; figure `shared/shared/plot_session_align.m`.

1. Trigger edges are detected on the goniometer trigger channel at 1000 Hz. Edges closer
   than 200 ms are one burst and are reduced to the first edge.
2. A global offset is found by matching the pattern of inter-onset intervals (IOI) between
   the two marker trains (`align_ioi.m`).
3. Markers are paired by segmented IOI matching (tolerance 300 ms), which allows the
   offset to change in steps. If fewer than 95% of the markers pair, a drift-segment
   nearest-neighbour pairing is also tried and the better result is kept. Recordings from
   the second acquisition system, whose clock steps by fractions of a second, take the best
   of three matchers: segmented IOI at 300 ms, segmented IOI at 1200 ms, and an
   offset-tracking matcher (`match_offset_tracking.m`).
4. Isolated unmatched markers are recovered from the local offset of their three nearest
   matched neighbours, within 150 ms (1200 ms on the second acquisition system).
5. Verdict: `PASS` when at least 95% of the EEG markers are paired, otherwise `FAIL`. A
   session whose goniometer data are split over several files is aligned file by file
   against the merged EEG and receives `SPECIAL_CASE` when at least 80% of the markers
   are paired, otherwise `FAIL`.

Every recording segment produces an alignment log (`align_log/<label>.json`: matched
markers, per-marker residuals, clock segments) and an alignment figure, and every figure
was inspected. Two corrections were applied where the inspection showed an error the
matcher could not resolve:

- **Spurious goniometer edges.** An edge with a plausible interval but no EEG counterpart
  can shift the pairing. Its index is passed as `spurious_goni_override` and the edge is
  removed before matching.
- **Hand-verified pairing.** For a few segments the EEG-goniometer pairing was read off
  the figure and entered in `shared/01_align/manual_align_corrections.m`.
  `align_goni_to_eeg.m` applies the map and asserts that it did.

Patient recordings differ in three ways. Only the S10 marker reaches the goniometer
trigger channel, so only S10 is paired. The first two sessions of participant P1 used a
two-marker protocol with one start and one end edge per block; their markers are placed by
linear interpolation between those two anchors (`prep_align_patient_legacy.m`, verdict
`PASS_LEGACY`). Several vhdr files of one session are merged before alignment.

## 2. Goniometer trial quality control

Each trial is scored against the other trials of the same session and condition, so the
thresholds adapt to the participant, the sensors and the task.

Code: `healthy/02_goni_qc/extract_goni.m` (epochs and an initial verdict),
`shared/04_qc/recompute_goni_qc.m` (the verdict used for analysis of the healthy cohort),
`patient/02_extract_goni/extract_goni_patient.m` (patients).

Active channels are the reference walker's goniometers for imagined walking and the
participant's goniometers for real walking. With K = 3 x 1.4826 and an effective median
absolute deviation (MAD) of at least 5% of the median, a healthy trial is marked `problem` when

- **flat**: the mean standard deviation of its active channels is below median - K x MAD;
- **length**: its length in samples lies outside median +/- K x MAD;
- **shape**: its mean peak cross-correlation with the average of the other trials
  (within +/- 0.5 s lag) is below median - K x MAD and below 0.5.

Rest trials are not scored. One healthy session whose goniometer file was extracted on its
own keeps the verdict of `extract_goni.m`, which applies the length rule and a per-channel
standard-deviation rule with a floor of 1 degree, without the shape rule. Patient trials are
screened with the same two rules and K = 3 (`extract_goni_patient.m`); the patient decoder
uses imagined-walking trials only.

At export (`healthy/04_export/export_session.m`, `patient/05_export/export_session.m`)
`ok` trials become `good` and `problem` trials become `bad`. A trial is also set to
`bad` when its EEG window and goniometer window differ in duration by more than 1.0 s.
The decoder loads `good` trials only.

In one patient session the reference walker's left-knee goniometer carries no signal,
which marks every imagined-walking trial of that session as `problem`. The decoder does
not score a joint without a reference, so `patient/04_qc/make_goni_qc_trial_override.py`
re-admits the trials that fail on that channel alone; any trial failing on another
channel or on length stays excluded.

## 3. Session inclusion

A session was retained if its EEG and goniometer recordings were intact and aligned. The
alignment verdict and figure, the goniometer QC and the recording itself were checked
against this criterion for every session. EEG quality measures (interpolated channels,
ICLabel brain components) were reported for every healthy session and were not used to
include or exclude a healthy session. Three patient sessions were excluded because of a
recording hardware failure.

The batch runners read per-session settings (raw file locations, `special_case`,
`spurious_goni_override`, goniometer file overrides) from a session registry,
`build_session_registry(raw_root)` for the healthy cohort and a patient registry of the
same form. The registries are not distributed because they hold the file layout of the
raw recordings; every correction they contain is listed in Section 5.

## 4. Summary of automatic and manual decisions

Counts are recording segments, sessions or trials as stated. Trial counts refer to the
retained sessions.

### Healthy participants

| Stage | Measure | Decided by | n | of | % |
|---|---|---|---:|---:|---:|
| alignment | recorded sessions | input | 83 | 83 | 100.00 |
| alignment | recording segments | input | 98 | 98 | 100.00 |
| alignment | segments passed with no manual input | coded rule | 88 | 98 | 89.80 |
| alignment | of which on the wider-tolerance rule for the second acquisition system | coded rule | 6 | 98 | 6.12 |
| alignment | segments not passed with no manual input | coded rule | 1 | 98 | 1.02 |
| alignment | segments with a hand-verified marker pairing | manual review | 6 | 98 | 6.12 |
| alignment | segments with spurious goniometer edges removed | manual review | 6 | 98 | 6.12 |
| alignment | segments with any manual input | manual review | 9 | 98 | 9.18 |
| alignment | sessions with any manual input | manual review | 7 | 83 | 8.43 |
| alignment | final segment verdict PASS | outcome | 95 | 98 | 96.94 |
| alignment | final segment verdict SPECIAL_CASE | outcome | 1 | 98 | 1.02 |
| alignment | final segment verdict FAIL | outcome | 2 | 98 | 2.04 |
| session inclusion | excluded: recording too short for analysis | manual review | 1 | 83 | 1.20 |
| session inclusion | excluded: goniometer recording failure | manual review | 1 | 83 | 1.20 |
| session inclusion | excluded: EEG-goniometer alignment failure: markers could not be paired | manual review | 1 | 83 | 1.20 |
| session inclusion | retained sessions | outcome | 80 | 83 | 96.39 |
| session inclusion | retained participants | outcome | 42 | 42 | 100.00 |
| goniometer trial QC | imagined + real walking trials exported | input | 9543 | 9543 | 100.00 |
| goniometer trial QC | trials excluded by the goniometer MAD rules | coded rule | 44 | 9543 | 0.46 |
| goniometer trial QC | trials excluded by the EEG-goniometer duration check | coded rule | 4 | 9543 | 0.04 |
| goniometer trial QC | imagined walking trials excluded | outcome | 19 | 4775 | 0.40 |
| goniometer trial QC | real walking trials excluded | outcome | 29 | 4768 | 0.61 |
| goniometer trial QC | imagined walking trials reaching the decoder | outcome | 4756 | 4775 | 99.60 |
| goniometer trial QC | real walking trials reaching the decoder | outcome | 4739 | 4768 | 99.39 |
| goniometer trial QC | sessions with no excluded walking trial | outcome | 56 | 80 | 70.00 |

### Participants with spinal cord injury

| Stage | Measure | Decided by | n | of | % |
|---|---|---|---:|---:|---:|
| alignment | recorded sessions | input | 18 | 18 | 100.00 |
| alignment | sessions passed on the standard marker protocol | coded rule | 11 | 18 | 61.11 |
| alignment | sessions passed on the two-marker protocol of the first two recordings | coded rule | 2 | 18 | 11.11 |
| alignment | sessions passed on the multi-file rule (at least 80% of markers matched) | coded rule | 3 | 18 | 16.67 |
| alignment | sessions not passed | coded rule | 2 | 18 | 11.11 |
| alignment | not passed but retained after review | manual review | 1 | 18 | 5.56 |
| session inclusion | excluded: recording hardware failure | manual review | 3 | 18 | 16.67 |
| EEG channel mapping | sessions with connector order restored | manual review | 3 | 18 | 16.67 |
| goniometer channel labels | sessions with a sensor label corrected | manual review | 1 | 18 | 5.56 |
| session inclusion | retained sessions | outcome | 15 | 18 | 83.33 |
| goniometer trial QC | imagined walking trials exported | input | 629 | 629 | 100.00 |
| goniometer trial QC | trials re-admitted: failing only on a reference-walker channel without signal, which the decoder does not score | manual review | 43 | 629 | 6.84 |
| goniometer trial QC | imagined walking trials excluded | outcome | 26 | 629 | 4.13 |
| goniometer trial QC | imagined walking trials reaching the decoder | outcome | 603 | 629 | 95.87 |
| goniometer trial QC | pre-operative imagined walking trials reaching the decoder (4 sessions) | outcome | 154 | 172 | 89.53 |
| goniometer trial QC | post-operative imagined walking trials reaching the decoder (11 sessions) | outcome | 449 | 457 | 98.25 |

## 5. Manual interventions per session

| Cohort | Session | Segment | Stage | Intervention | Result |
|---|---|---|---|---|---|
| healthy | subj16_sess02 |  | alignment | removed goniometer edge(s) 29 with no EEG counterpart | PASS (301/301 markers matched) |
| healthy | subj18_sess01 |  | alignment | removed goniometer edge(s) 302 with no EEG counterpart | PASS (301/302 markers matched) |
| healthy | subj22_sess01 | b | alignment | hand-verified marker pairing | SPECIAL_CASE (143/152 markers matched) |
| healthy | subj27_sess01 |  | alignment | hand-verified marker pairing; removed goniometer edge(s) 2 with no EEG counterpart | PASS (302/302 markers matched) |
| healthy | subj31_sess01 | b | alignment | hand-verified marker pairing | PASS (202/202 markers matched) |
| healthy | subj40_sess02 | b | alignment | hand-verified marker pairing; removed goniometer edge(s) 1 with no EEG counterpart | PASS (100/101 markers matched) |
| healthy | subj40_sess02 | c | alignment | removed goniometer edge(s) 1 with no EEG counterpart | PASS (71/72 markers matched) |
| healthy | subj40_sess02 | e | alignment | hand-verified marker pairing; removed goniometer edge(s) 20 21 22 23 with no EEG counterpart | FAIL (16/19 markers matched); segment not exported or analysed |
| healthy | subj42_sess01 |  | alignment | hand-verified marker pairing | PASS (296/301 markers matched) |
| healthy | subj10_sess01 |  | session inclusion | whole session excluded | recording too short for analysis |
| healthy | subj14_sess01 |  | session inclusion | whole session excluded | goniometer recording failure |
| healthy | subj33_sess02 |  | session inclusion | whole session excluded | EEG-goniometer alignment failure: markers could not be paired |
| patient | pat01_sess04 |  | alignment | retained after review: the goniometer recording ends before the EEG, so the unmatched markers are the final trials only | FAIL (82 markers matched) |
| patient | pat01_sess03 |  | session inclusion | whole session excluded | recording hardware failure |
| patient | pat02_sess04 |  | session inclusion | whole session excluded | recording hardware failure |
| patient | pat02_sess07 |  | session inclusion | whole session excluded | recording hardware failure |
| patient | pat01_sess04 |  | EEG channel mapping | amplifier module connectors swapped at recording; channel order restored before bad-channel detection | retained |
| patient | pat02_sess01 |  | EEG channel mapping | amplifier module connectors swapped at recording; channel order restored before bad-channel detection | retained |
| patient | pat02_sess02 |  | EEG channel mapping | amplifier module connectors swapped at recording; channel order restored before bad-channel detection | retained |
| patient | pat02_sess09 |  | goniometer channel labels | one reference-walker ankle sensor was labelled as a participant sensor at recording; label corrected by sensor serial number | retained |
| patient | pat01_sess05 |  | goniometer trial QC | trials failing goniometer QC only on the reference-walker left-knee channel without signal re-admitted; that joint is not scored | 43 trials re-admitted |
