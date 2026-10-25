# Imagined-Walking Gait Decoding

This repository contains the analysis code for the research paper:

**Non-invasive decoding of continuous lower-limb kinematics from imagined walking in healthy adults and spinal cord injury patients**

Wei Zhang<sup>1</sup>, Aung Aung Phyo Wai<sup>1</sup>, Neethu Robinson<sup>1</sup>, Rui Liu<sup>1</sup>, Han Wei Ng<sup>1</sup>, Seng Kwee Wee<sup>2,3,4,5</sup>, Min Wee Phua<sup>2,3</sup>, Wen Li Lui<sup>2,3</sup>, Ananda Sidarta<sup>6</sup>, Lek Syn Lim<sup>6</sup>, Karen Sui Geok Chua<sup>2,3,5</sup>, Kai Rui Wan<sup>7,8</sup>, Zhi Yan Valerie Ng<sup>2,3</sup>, Cuntai Guan<sup>1,9,\*</sup>

<sup>1</sup> College of Computing and Data Science, Nanyang Technological University, Singapore
<sup>2</sup> Department of Rehabilitation Medicine, Tan Tock Seng Hospital, Singapore
<sup>3</sup> Institute of Rehabilitation Excellence, Tan Tock Seng Hospital Rehabilitation Centre, NHG Health, Singapore
<sup>4</sup> Singapore Institute of Technology, Singapore
<sup>5</sup> Lee Kong Chian School of Medicine, Nanyang Technological University, Singapore
<sup>6</sup> R3 Institute Singapore, Nanyang Technological University, Singapore
<sup>7</sup> Department of Neurosurgery, National Neuroscience Institute, Tan Tock Seng Hospital, Singapore
<sup>8</sup> Department of Neurosurgery, National Neuroscience Institute, Singapore General Hospital, Singapore
<sup>9</sup> Center of AI in Medicine, Nanyang Technological University, Singapore

## Overview

This repository contains the analysis code and source data for a study of
continuous locomotor decoding from scalp EEG recorded during imagined walking.
Participants imagined walking in time with an observed reference walker while
remaining still, so the reconstruction cannot draw on the participant's own limb
movement. The code covers the full analysis path: EEG-goniometer alignment and
preprocessing, the decoder, the transfer analysis in two individuals with spinal
cord injury, and the figure and table code for the reported results.

## Citation

If you use this code in your research, please cite our paper:

```
Zhang, W., Phyo Wai, A. A., Robinson, N., Liu, R., Ng, H. W., Wee, S. K., Phua, M. W.,
Lui, W. L., Sidarta, A., Lim, L. S., Chua, K. S. G., Wan, K. R., Ng, Z. Y. V., & Guan, C. (2026).
Non-invasive decoding of continuous lower-limb kinematics from imagined walking
in healthy adults and spinal cord injury patients.
```

## Data Availability

Access to the raw EEG and goniometer recordings is governed by data-use agreements
or sponsor restrictions, and they are therefore not publicly available. Requests for
access should be directed to the corresponding author and will be reviewed against
the governing agreements and the ethics approval.

`04_source_data/` contains the processed values behind every figure and table in
the paper, so all reported numbers can be checked directly. Each item is a pair:
`<name>_source_data.csv` holds the values and `<name>_metadata.csv` records what
they are, how many sessions contributed, and which analysis produced them. Dates
in the patient records are given as day or month offsets from each patient's first
record rather than as calendar dates.

The repository reproduces the published analysis given the intermediate artefacts,
which are not distributed: the aligned per-session EEG/goniometer/event triples, the
preprocessing QC tables, the decoder's per-sample output (one example session is
released, see below), and the simulation rollout and gate score tables. The header of
every figure script lists what it reads (`Reads:`):
each of these intermediates it needs, and each released `04_source_data/` table written
by another script. Some of the
decoder-derived tables the figure scripts read (per-trial correlation tables, cohort
cycle-average arrays, harmonic phase-map dumps) were assembled from the decoder output
by scripts that are not part of this repository; the headers and error messages of the
scripts that read them give their file names.

The decoder writes its output to `02_decoding/result/`, and the figure, simulation and
per-fold band-table scripts read it from `data/decoder_output/`; point one at the other
before running them (`mkdir -p data && ln -s ../02_decoding/result data/decoder_output`).
The published decoder output is `result/main/`. The job lists and the merge checks of the
decode chain (Step 09) take the trial layout of every run (trial sets, folds, sample counts)
from the same decoder run on the zero-phase preprocessing, `result/zero_phase_prep/` (not
distributed); the two differ only in the EEG preprocessing, so that layout is identical. The
per-trial tables the figure scripts read are the per-session `<session>_trial_r.csv` files of
the Python readout concatenated over sessions.

`03_figures/sup_decoder_trial_tables.py` writes, from the decoder output, the fold map of the
random 10-fold cross-validation with the scored flag, the held-out correlation of every trial
and joint, and (`--predictions all --out <dir>`) the per-sample decoder output, session by
session.

Supplementary tables are released as their `*_source_data.csv` values; the workbooks
themselves are not committed and are written into `figures_out/` by the 5
scripts that build them (`fig7_gate_accuracy_table.py`, `sup_decoding_controls.py`, `sup_goni_qc_exclusion.py`, `sup_participant_demographics.py`, `sup_table_sim_outcomes.py`).

## Repository Structure

```
01_preprocessing/    healthy/, patient/, shared/   EEG and goniometer preparation
                     causal/                          the causal preprocessing of the published results
02_decoding/         core/, shared/, readout/, run/   the decoder: features and export (MATLAB)
                     pyport/                          the decoder: readout (Python)
                     fold_tables/                     the decoder: per-fold harmonic band tables (Python)
                     run/                             the decoder: bases, exports, job lists and merge (shell, Python)
                     simulation/                      Figure 7 physics simulation and walk-intent gate (Python)
03_figures/          fig1_* ... fig7_*, sup_*      one script per panel or table
04_source_data/      fig1/ ... fig7/, supplementary/   values behind every number
```

## Analysis Pipeline

Preprocessing (`01_preprocessing/`, MATLAB)

**Step 01**: Align EEG markers to goniometer trigger edges by inter-onset interval
matching; a segment passes when at least 95% of its EEG markers are paired, and every
alignment figure is reviewed  
**Step 02**: Extract goniometer joint angles and flag trials whose length, amplitude or
waveform shape deviates from the other trials of the session (median absolute deviation
rules); only trials that pass reach the decoder  
**Step 03**: Preprocess EEG - band-pass filter, bad-channel interpolation, common  
average reference, artefact subspace reconstruction, ICA, ICLabel rejection  
**Step 04**: Session-level quality-control summary  
**Step 05**: Export the aligned EEG, goniometer and event triple per session

The published results use the strictly causal variant of Steps 03-05
(`01_preprocessing/causal/`; one MATLAB process per session, `healthy/cpc_session.m` and
`patient/cpp_session.m`). It runs the stages above with three changes: the
1000 to 250 Hz anti-aliasing low-pass is the same filter converted to minimum phase
(`cpc_step1.m`, `cpp_step1.m`); bad channels, ICA and ICLabel are fitted on a calibration
copy of the beginning of the recording (the first run of a healthy session, else its first
third; for a patient session the first natural segment if its task segments reach 180 s,
else the shortest beginning whose task segments do) and applied forward to the whole recording (`cpc_step2_cal.m`, `cpp_step2_cal.m`,
with a copy of `prep_step2` whose ICA keeps the caller's random state); and the 0.1-40 Hz band-pass is the same FIR
converted to minimum phase and applied forward only (`cp_step3.m`, `cpp_prep_step3.m`, copies
of `prep_step3` with these switches). Components are rejected by ICLabel only. The
calibration cut of every session is in `healthy/plan.csv` and `patient/plan.csv` (written by
`cpc_plan.py` / `cpp_plan.py` from the run boundaries in `calibration_lengths.py`; the reviewed decisions for the healthy sessions in
`healthy/plan_overrides.csv`). Each runner first checks that it reproduces the zero-phase step 1 bit for bit, and its last stage (`cpc_check.py`, `cpp_check.py`) requires the causal triple to equal the zero-phase triple in every goniometer array and event field and to differ in the EEG.
The trials inside a session's calibration copy are filtered with spatial filters fitted on
data that include them; every later trial is preprocessed strictly causally. The calibration
trials train the decoder but are not scored (Step 09). The neural
analyses of Figures 3, 4 and 6 read the zero-phase triples of Steps 03-05.

The quality-control criteria, the decisions taken by hand and their effect on the
analysed sessions and trials are documented in
[`01_preprocessing/QUALITY_CONTROL.md`](01_preprocessing/QUALITY_CONTROL.md).

Decoding (`02_decoding/`, MATLAB and Python)

Preprocessing and decoding are causal: each estimate uses only EEG up to the predicted
sample, and the spatial filters of the preprocessing come from the calibration copy at the
start of the session.

**Step 06**: Estimate the step-frequency harmonic bands from the goniometer spectra
(`core/estimate_goniometer_psd_peaks.m`; condition `EXEC` with `band_half_width` 0.18
for real walking, `output_prefix` `patient` for the patient sessions)  
**Step 07**: Fit the shared principal-component bases on the causal decoder features of the
causal triples - leave-one-participant-out for every healthy session (imagined and real
walking), and one basis pooled over the 80 healthy sessions for the patients
(`run/build_bases.sh`, one build at a time with `run/build_basis.sh`; `run/check_basis.py`
checks each build)  
**Step 08**: Compute the causal features and write the feature exports
(`run/export_features.sh <cpu list> <export_id>` → `core/run_decoder.m` with
`CP_EXPORT_ONLY=1`; one row of `exports.csv`, written by `run/make_job_lists.py`;
`run/check_export.py` checks each export): at each gait harmonic, a fixed-frequency complex
demodulation (first-order low-pass at 0.18 Hz, applied forward only)  
**Step 09**: Fit and test the readout in Python (`pyport/run_session.py` with
`pyport/configs/tenfold_pca30_lpf6.json`): projection onto the leading 30 components of
the shared basis, lagged ridge readout, causal second-order Butterworth low-pass at 6 Hz
on that first-stage output, causal finite impulse response correction fitted on the
low-passed predictions, random 10-fold cross-validation; the control conditions use the
other files in `pyport/configs/`. The harmonic band table of the features (the gait
frequency f0 and the band centres of the three harmonics) is estimated for each fold from
that fold's training trials only. The tables come from the goniometer recordings and the
trial sets, which the causal preprocessing does not change:
`fold_tables/fold_centres.py` estimates f0 and the band centres of every fold from its
training trials (healthy imagined walking and the patient sessions) after checking that it
reproduces each session's stored whole-session table exactly, and
`fold_tables/make_band_tables.py` writes one band table per distinct set of held-out centres
that differs from the whole-session one; `fold_tables/make_fold_tables.py` does the same for
real walking (`rw`) and for the contiguous-fold control (`block`), and in mode `iw` must
reproduce the imagined-walking fold centres exactly. `run/make_job_lists.py` turns them into
`exports.csv` (one feature export per session and table: the whole-session table and every
per-fold table that differs from it) and `decodes.csv` (one row per decoded fold). Every fold
is decoded from the export of its own table, with the package and configuration file of its
run (`python -m pyport.run_session <export> <basis> <config> <out> --mode fast --folds <k>
--dump`, plus `--rest-export <rest.mat>` for the resting-input control; the cross-task runs
fit one model per session on every trial of the training condition, with `--cross-export`
and the whole-session table, and no `--folds`). `run/merge_runs.py <run>` assembles each run from its fold jobs
with the functions of `fold_tables/merge_dumps.py`; it requires every fold of the trial layout
to be present with the same trial set, the fold of every tested trial to equal the export's
fold map, one decoder configuration and basis per session, and every merged per-trial r to
equal the correlation recomputed from the merged dump to 1e-6. Runs: imagined and real
walking, the patients (post- and pre-operative), and the resting-input, cadence-shuffle,
contiguous-fold, cross-task, patient circular-shift and patient zero-input controls (the
target-label shuffle is computed afterwards from the merged dumps).  
Scoring: every trial trains the decoder, but accuracy is reported only on the trials recorded
after the session's calibration recording (onset later than the calibration cut of `plan.csv`),
whose spatial filters were fitted on earlier data only. The mask ships as
`03_figures/scoring_mask.csv` (one row per session, condition and trial, column `keep`;
definitions in `scoring_mask_metadata.csv`) and is applied wherever the figure scripts read
decoder output (`03_figures/analysis_config.py`, `keep_trial` / `filter_trials`, and
`03_figures/dump_h5.py`): 3,172 of 4,756 healthy imagined-walking trials, 282 of 449
post-operative and 77 of 154 pre-operative patient trials, and 3,169 of 4,739 real-walking
trials are scored. The fold map, the training and the decoder outputs do not depend on the
mask; the Figure 7 rollouts and the walk-intent gate are reported on the same scored trials.
**Step 10**: Dump gait-cycle harmonic phase maps for the representational analyses
(zero-phase filtering; these maps do not depend on the decoder)

The Python readout reproduces the MATLAB decoder of `core/run_decoder.m` fold by fold:
`pyport/gate.py` compares the two on the same session and configuration (per-trial
correlation within 1e-6), and `pyport/tests/test_units.py` checks the MATLAB-semantics
helpers against MATLAB output stored in `pyport/tests/mref_*.mat` (synthetic inputs,
regenerated by the `mref_*.m` scripts beside them); both run with the first-stage
low-pass off (`stage1_lpf_hz` 0). The shipped package is the code that produced the published decoding results; in the released copy `run_session.py` resolves relative basis paths against
`02_decoding/`.

Simulation (`02_decoding/simulation/`, Python)

The Figure 7 rollouts drive a MuJoCo biped from 2 s into each trial with the gait phase
that a causal extractor reads from the condition's own input: for the decoded condition,
the decoder's first-stage output over 2-4 s followed by its final output; for the
reference walker, its goniometer. Outcomes are measured from the onset of the final
decoder output, the forward displacement 3 s after it (`run_simulation.sh`; the per-session cadence prior it reads is `f0_cadence_prior.csv`). Every Figure 7 contrast is tested
on the per-trial difference with a linear mixed model d ~ 1 + (1 | session) (session random
intercept, REML, statsmodels MixedLM, Wald p), pooled and within each patient. The walk-intent gate reads the
preprocessed patient EEG: `run_gate_scores_causal.sh` builds the gate session set from the
causal patient triples (`build_gate_session_set_causal.py`) and scores it with the gate of
`gate_scores_within_session.py` (`gate_scores_causal.py`).  The gate is
trained on every event; `gate_scored_events.py` keeps the events whose onset is at or after the
session's calibration cut (`result/main/scored/`, the stream the figure scripts read).

Figures and tables (`03_figures/`, Python)

**Step 11**: Render each panel and table from a fixed preprocessing or decoder
product, writing the figure together with its source data

| Prefix | Content |
|---|---|
| `fig1_*` | Study design and reference-walker kinematics |
| `fig2_*` | Reconstruction of continuous gait kinematics from imagined walking |
| `fig3_*` | Gait-phase locking of the imagined-walking representation |
| `fig4_*` | Shared locomotor representation across imagined and real walking |
| `fig5_*` | Decoding imagined walking after spinal cord injury |
| `fig6_*` | Representational geometry of patient imagined walking |
| `fig7_*` | Intent gate and physics simulation of the decoded gait phase |
| `sup_*`  | Supplementary analyses whose values the text quotes, and shared helpers |

### Script, figure and Source Data

| Script | Paper location | Source Data |
|---|---|---|
| `03_figures/pooled_trial_medians.py` | Supplementary Note 1 | `04_source_data/supplementary/pooled_trial_medians_metadata.csv`<br>`04_source_data/supplementary/pooled_trial_medians_source_data.csv` |
| `03_figures/dump_h5.py` | - | - |
| `03_figures/analysis_config.py` | - | - |
| `03_figures/f7_scored_trials.py` | - | - |
| `03_figures/fig1c_goni_eeg_draw_example.py` | Figure 1 | `04_source_data/fig1/fig1c_goni_eeg_draw_example_metadata.csv`<br>`04_source_data/fig1/fig1c_goni_eeg_draw_example_source_data.csv` |
| `03_figures/fig2_coord_matrix.py` | Supplementary Figure 2b | `04_source_data/fig2/fig2_coord_matrix_metadata.csv`<br>`04_source_data/fig2/fig2_coord_matrix_source_data.csv` |
| `03_figures/fig2_cyclogram.py` | Figure 2 | `04_source_data/fig2/fig2_cyclogram_metadata.csv`<br>`04_source_data/fig2/fig2_cyclogram_trajectories_metadata.csv`<br>`04_source_data/fig2/fig2_cyclogram_trajectories_source_data.csv` |
| `03_figures/fig2_recon_fidelity.py` | Figure 2 | `04_source_data/fig2/fig2_recon_fidelity_curves_metadata.csv`<br>`04_source_data/fig2/fig2_recon_fidelity_curves_source_data.csv`<br>`04_source_data/fig2/fig2_recon_fidelity_metadata.csv`<br>`04_source_data/fig2/fig2_recon_fidelity_source_data.csv` |
| `03_figures/fig2a_goni_power.py` | Figure 2 | `04_source_data/fig2/fig2a_goni_power_summary.csv` |
| `03_figures/fig2b_causal_fir_node.py` | Figure 2 | - |
| `03_figures/fig2b_glyph_fir.py` | Figure 2 | - |
| `03_figures/fig2b_glyph_output.py` | Figure 2 | - |
| `03_figures/fig2b_schematic_elements.py` | Figure 2 | - |
| `03_figures/fig2c_null_paired_test.py` | Supplementary Note 1, Figure 2 | `04_source_data/fig2/fig2c_null_paired_test_metadata.csv` |
| `03_figures/fig2c_plot_individual_ypred_yreal.py` | Supplementary Figure 2f, Figure 2d | - |
| `03_figures/fig2e_correspondence_matrix.py` | Figure 2 | `04_source_data/fig2/fig2e_correspondence_metadata.csv`<br>`04_source_data/fig2/fig2e_correspondence_source_data.csv` |
| `03_figures/fig2e_right_limb_traces.py` | Figure 2e | `04_source_data/fig2/fig2e_right_limb_traces_metadata.csv`<br>`04_source_data/fig2/fig2e_right_limb_traces_source_data.csv` |
| `03_figures/fig2g_spectral_fidelity.py` | Figure 2g | `04_source_data/fig2/fig2g_spectral_fidelity_metadata.csv`<br>`04_source_data/fig2/fig2g_spectral_fidelity_source_data.csv` |
| `03_figures/fig2g_trial_r_export.py` | Figure 2 | `04_source_data/fig2/fig2g_trial_r_long.csv` |
| `03_figures/fig3_build_master.py` | Figure 3 | - |
| `03_figures/fig3_d_dissociation_forest.py` | Figure 3 | - |
| `03_figures/fig3_d_forest_bootstrap_draws.py` | Figure 3 | `04_source_data/fig3/fig3_d_forest_bootstrap_draws_metadata.csv` |
| `03_figures/fig3_def_resnonres.py` | Figure 3 | `04_source_data/fig3/fig3_def_resnonres_metadata.csv`<br>`04_source_data/fig3/fig3_def_resnonres_source_data.csv` |
| `03_figures/fig3_export_for_eeglab.py` | Figure 3 | - |
| `03_figures/fig3_h_forest_bootstrap_draws.py` | Figure 3 | `04_source_data/fig3/fig3_h_forest_bootstrap_draws_metadata.csv` |
| `03_figures/fig3_h_selfreport_forest.py` | Figure 3 | - |
| `03_figures/fig3_harmonic_phase_responder_split.py` | Figure 3 | `04_source_data/fig3/fig3_harmonic_phase_responder_split_metadata.csv` |
| `03_figures/fig3_mediation.py` | Figure 3 | `04_source_data/fig3/fig3_mediation_metadata.csv`<br>`04_source_data/fig3/fig3_mediation_source_data.csv` |
| `03_figures/fig3a_elements_real.py` | Figure 3a | - |
| `03_figures/fig3a_inputs_one_trial_element.py` | Figure 3a | - |
| `03_figures/fig3d_locking_paired_stats.py` | Figure 3d | `04_source_data/fig3/fig3d_locking_paired_metadata.csv`<br>`04_source_data/fig3/fig3d_locking_paired_source_data.csv` |
| `03_figures/fig3e_locking_boxwhisker.py` | Figure 3 | `04_source_data/fig3/fig3e_locking_boxwhisker_metadata.csv`<br>`04_source_data/fig3/fig3e_locking_boxwhisker_source_data.csv` |
| `03_figures/fig3g_responder_blocks.py` | Figure 3 | `04_source_data/fig3/fig3g_responder_blocks_metadata.csv` |
| `03_figures/fig4_correspondence_amplitude.py` | Figure 4 | `04_source_data/fig4/fig4_correspondence_amplitude_metadata.csv` |
| `03_figures/fig4_harmonic_legend.py` | Figure 4 | `04_source_data/fig4/fig4_harmonic_legend_metadata.csv` |
| `03_figures/fig4_decoding_gain.py` | Figure 4 | `04_source_data/fig4/fig4_decoding_gain_metadata.csv`<br>`04_source_data/fig4/fig4_decoding_gain_sessions_metadata.csv` |
| `03_figures/fig4_phase_correspondence.py` | Supplementary Figure 4c | `04_source_data/fig4/fig4_phase_correspondence_metadata.csv` |
| `03_figures/fig4_phase_delay_persession.py` | Figure 4 | `04_source_data/fig4/fig4_phase_delay_persession_metadata.csv`<br>`04_source_data/fig4/fig4_phase_delay_persession_source_data.csv` |
| `03_figures/fig4_rsa_persession_spin.py` | Figure 4 | - |
| `03_figures/fig4_trialmedian_export.py` | Figure 4 | `04_source_data/fig4/fig4_trialmedian_metadata.csv`<br>`04_source_data/fig4/fig4_trialmedian_source_data.csv` |
| `03_figures/fig4a_iw_rw_decodability_export.py` | Figure 4 | - |
| `03_figures/fig4b_rsa_per_harmonic_k123.py` | Supplementary Note 5, Figure 4g | `04_source_data/fig4/fig4b_rsa_per_harmonic_metadata.csv` |
| `03_figures/fig4c_rdm_grid.py` | Figure 4 | `04_source_data/fig4/fig4c_rdm_grid_metadata.csv` |
| `03_figures/fig4d_locking_indexes_decoding.py` | Figure 4 | `04_source_data/fig4/fig4d_locking_indexes_decoding_fit_by_state.csv`<br>`04_source_data/fig4/fig4d_locking_indexes_decoding_metadata.csv`<br>`04_source_data/fig4/fig4d_locking_indexes_decoding_source_data.csv` |
| `03_figures/fig4a_rw_harmonic_strip.py` | Figure 4 | `04_source_data/fig4/fig4a_rw_harmonic_strip_metadata.csv`<br>`04_source_data/fig4/fig4a_rw_harmonic_strip_source_data.csv` |
| `03_figures/fig5_cd_patient_trial_example.py` | Figure 5d | `04_source_data/fig5/fig5_cd_patient_trial_example_metadata.csv`<br>`04_source_data/fig5/fig5_cd_patient_trial_example_source_data.csv` |
| `03_figures/fig5_efgh_patient_reconstruction.py` | Figure 5 | `04_source_data/fig5/fig5_efgh_patient_reconstruction_metadata.csv`<br>`04_source_data/fig5/fig5_efgh_patient_reconstruction_source_data.csv` |
| `03_figures/fig5a_patient_decode_margin.py` | Figure 5 | `04_source_data/fig5/fig5a_patient_decode_margin_metadata.csv` |
| `03_figures/fig5b_session_box.py` | Figure 5 | `04_source_data/fig5/fig5b_session_box_metadata.csv`<br>`04_source_data/fig5/fig5b_session_box_source_data.csv` |
| `03_figures/fig5b_session_box_export.py` | Figure 5 | `04_source_data/fig5/fig5b_session_box_metadata.csv`<br>`04_source_data/fig5/fig5b_session_box_source_data.csv` |
| `03_figures/fig5g_patient_trial_r_export.py` | Figure 5 | - |
| `03_figures/fig5g_patient_trialwise_lme.py` | Figure 6g | `04_source_data/fig5/fig5g_patient_trialwise_lme_metadata.csv` |
| `03_figures/fig6_cross_harmonic_specificity.py` | Figure 6 | `04_source_data/fig6/fig6_cross_harmonic_specificity_metadata.csv`<br>`04_source_data/fig6/fig6_cross_harmonic_specificity_source_data.csv` |
| `03_figures/fig6_locking_vs_transfer.py` | Supplementary Figure 1h, Figure 6g | `04_source_data/fig6/fig6_locking_vs_transfer_metadata.csv`<br>`04_source_data/fig6/fig6_locking_vs_transfer_source_data.csv` |
| `03_figures/fig6_movement_free_dissociation.py` | Supplementary Note 6, Figure 6h | `04_source_data/fig6/fig6_movement_free_dissociation_metadata.csv`<br>`04_source_data/fig6/fig6_movement_free_dissociation_source_data.csv` |
| `03_figures/fig6_patient_harmonic_phase.py` | Supplementary Figure 6a, Figure 6a | `04_source_data/fig6/fig6_patient_harmonic_phase_metadata.csv`<br>`04_source_data/fig6/fig6_patient_harmonic_phase_source_data.csv` |
| `03_figures/fig6_patient_healthy_harmonic_similarity.py` | Figure 6 | `04_source_data/fig6/fig6_patient_healthy_harmonic_similarity_metadata.csv` |
| `03_figures/fig6_patient_in_healthy_similarity.py` | Supplementary Note 6, Figure 6c | `04_source_data/fig6/fig6_patient_in_healthy_similarity_metadata.csv`<br>`04_source_data/fig6/fig6_patient_in_healthy_similarity_source_data.csv` |
| `03_figures/fig6_rsa_iw_healthy_patient_rdm.py` | Figure 6e | `04_source_data/fig6/fig6_rsa_iw_healthy_patient_rdm_matrices.csv`<br>`04_source_data/fig6/fig6_rsa_iw_healthy_patient_rdm_metadata.csv`<br>`04_source_data/fig6/fig6_rsa_iw_healthy_patient_rdm_source_data.csv` |
| `03_figures/fig6_rsa_quantification_forest.py` | Figure 6 | `04_source_data/fig6/fig6_rsa_quantification_forest_metadata.csv`<br>`04_source_data/fig6/fig6_rsa_quantification_forest_null_box_stats.csv`<br>`04_source_data/fig6/fig6_rsa_quantification_forest_spin_nulls.csv` |
| `03_figures/fig6_similarity_vs_transfer.py` | Supplementary Figure 1g | `04_source_data/fig6/fig6_similarity_vs_transfer_metadata.csv`<br>`04_source_data/fig6/fig6_similarity_vs_transfer_source_data.csv` |
| `03_figures/fig6_trial_excess.py` | Figure 6 | `04_source_data/fig6/fig6_trial_excess_metadata.csv`<br>`04_source_data/fig6/fig6_trial_excess_source_data.csv` |
| `03_figures/fig7_b_physics_filmstrip.py` | Figure 7 | - |
| `03_figures/fig7_gate_accuracy_table.py` | Supplementary Note 7 | `04_source_data/fig7/fig7_gate_accuracy_metadata.csv` |
| `03_figures/fig7_gate_close_latency.py` | Figure 7 | `04_source_data/fig7/fig7_gate_close_latency_bars_source_data.csv`<br>`04_source_data/fig7/fig7_gate_close_latency_metadata.csv`<br>`04_source_data/fig7/fig7_gate_close_latency_source_data.csv` |
| `03_figures/fig7_gate_discrimination.py` | Figure 7 | `04_source_data/fig7/fig7_gate_discrimination_metadata.csv`<br>`04_source_data/fig7/fig7_gate_discrimination_source_data.csv` |
| `03_figures/fig7_gate_method_schematic.py` | Supplementary Figure 1e | `04_source_data/fig7/fig7_gate_method_schematic_metadata.csv` |
| `03_figures/fig7_gate_roc.py` | Supplementary Figure 1d | `04_source_data/fig7/fig7_gate_roc_metadata.csv`<br>`04_source_data/fig7/fig7_gate_roc_source_data.csv` |
| `03_figures/fig7_gate_timeline.py` | no panel of its own; its table is an input of Figure 7f | `04_source_data/fig7/fig7_gate_timeline_metadata.csv` |
| `03_figures/fig7_gate_window_verdict.py` | Supplementary Figure 1f | `04_source_data/fig7/fig7_gate_window_verdict_metadata.csv` |
| `03_figures/fig7_sim_cadence_symmetry.py` | Figure 7 | `04_source_data/fig7/fig7_sim_cadence_metadata.csv`<br>`04_source_data/fig7/fig7_sim_cadence_source_data.csv`<br>`04_source_data/fig7/fig7_sim_cadence_stats.csv` |
| `03_figures/fig7_sim_trajectories.py` | Figure 7 | `04_source_data/fig7/fig7_sim_trajectories_metadata.csv`<br>`04_source_data/fig7/fig7_sim_trajectories_source_data.csv`<br>`04_source_data/fig7/fig7_sim_trajectories_stats.csv` |
| `03_figures/gate_config.py` | - | - |
| `03_figures/headline_session_medians.py` | Supplementary Note 1 | `04_source_data/supplementary/headline_session_medians_metadata.csv`<br>`04_source_data/supplementary/headline_session_medians_source_data.csv` |
| `03_figures/results_statistics.py` | - | - |
| `03_figures/safe_write.py` | - | - |
| `03_figures/sup_alignment_qc_patient.py` | Supplementary Note 11 | `04_source_data/supplementary/sup_alignment_qc_patient_metadata.csv` |
| `03_figures/sup_ao_locking.py` | Results; Supplementary Note 8 | `04_source_data/supplementary/sup_ao_locking_metadata.csv`<br>`04_source_data/supplementary/sup_ao_locking_source_data.csv` |
| `03_figures/sup_ao_target_activity.py` | Supplementary Note 2 | `04_source_data/supplementary/sup_ao_target_activity_metadata.csv` |
| `03_figures/sup_ao_transfer.py` | Supplementary Note 2 | `04_source_data/supplementary/sup_ao_transfer_metadata.csv` |
| `03_figures/sup_cadence_sensitivity.py` | Supplementary Note 1 | `04_source_data/supplementary/sup_cadence_sensitivity_metadata.csv` |
| `03_figures/sup_decoder_trial_tables.py` | - | `04_source_data/supplementary/sup_decoder_trial_fold_map_metadata.csv`<br>`04_source_data/supplementary/sup_decoder_trial_predictions_example_metadata.csv`<br>`04_source_data/supplementary/sup_decoder_trial_r_metadata.csv` |
| `03_figures/sup_decoding_controls.py` | Supplementary Figure 1a | `04_source_data/supplementary/sup_decoding_controls_metadata.csv`<br>`04_source_data/supplementary/sup_decoding_controls_source_data.csv`<br>`04_source_data/supplementary/sup_decoding_controls_table_metadata.csv`<br>`04_source_data/supplementary/sup_decoding_controls_table_source_data.csv` |
| `03_figures/sup_fig3_crosssession_predictor.py` | Supplementary Figure 3d | `04_source_data/supplementary/sup_fig3_crosssession_predictor_metadata.csv`<br>`04_source_data/supplementary/sup_fig3_crosssession_predictor_source_data.csv` |
| `03_figures/sup_fig4_participant_iw_rw.py` | Figure 4i | `04_source_data/supplementary/sup_fig4_participant_iw_rw_metadata.csv` |
| `03_figures/sup_fig7_rest_input_timing.py` | Supplementary Note 7 | `04_source_data/supplementary/sup_fig7_rest_input_timing_metadata.csv` |
| `03_figures/sup_fig7_sim_per_patient_gate3s.py` | Supplementary Note 7 | `04_source_data/supplementary/sup_fig7_sim_per_patient_gate3s_metadata.csv` |
| `03_figures/sup_goni_qc_exclusion.py` | Supplementary Table 5 | `04_source_data/supplementary/sup_goni_qc_exclusion_metadata.csv`<br>`04_source_data/supplementary/sup_goni_qc_exclusion_source_data.csv` |
| `03_figures/sup_healthy_trial_inventory.py` | Supplementary Note 11 | `04_source_data/supplementary/sup_healthy_trial_inventory_metadata.csv` |
| `03_figures/sup_interp_channel_sensitivity.py` | Supplementary Note 9 | `04_source_data/supplementary/sup_interp_channel_sensitivity_metadata.csv` |
| `03_figures/sup_participant_demographics.py` | Supplementary Note 11 | `04_source_data/supplementary/participant_demographics_metadata.csv`<br>`04_source_data/supplementary/participant_demographics_source_data.csv` |
| `03_figures/sup_patient_harmonic_locking_vs_decoding.py` | Supplementary Note 6 | `04_source_data/supplementary/sup_patient_harmonic_locking_vs_decoding_metadata.csv` |
| `03_figures/sup_preprocessing_qc_summary.py` | - | `04_source_data/supplementary/sup_preprocessing_qc_summary_metadata.csv` |
| `03_figures/sup_responder_participant_consistency.py` | Figure 3 | `04_source_data/supplementary/sup_responder_participant_consistency_metadata.csv` |
| `03_figures/sup_restinput_trial_lme.py` | Supplementary Note 1 | `04_source_data/supplementary/sup_restinput_trial_lme_metadata.csv` |
| `03_figures/sup_session_order_lme.py` | Figure 2 | `04_source_data/supplementary/sup_session_order_lme_metadata.csv` |
| `03_figures/sup_imagery_observation.py` | Supplementary Figure 11 | `04_source_data/supplementary/sup_imagery_observation_metadata.csv`<br>`04_source_data/supplementary/sup_imagery_observation_source_data.csv` |
| `03_figures/sup_table_sim_outcomes.py` | Supplementary Table 4 | `04_source_data/supplementary/sup_table_sim_outcomes_metadata.csv`<br>`04_source_data/supplementary/sup_table_sim_outcomes_source_data.csv` |

Shared settings imported by the scripts above, which write no output of their own: `fig6_panel_style.py`, `panel_style.py`, `patient_session_labels.py`, `project_goni.py`, `project_palette.py`, `si_panel_style.py`, `sup_panel_export.py`, `svg_panel_tools.py`.

Preprocessing and decoding entry points:

| Script | What it runs |
|---|---|
| `01_preprocessing/healthy/run_prep_one.m` | one healthy session through step 3 and the export |
| `01_preprocessing/patient/...` | the same stages for the patient sessions |
| `01_preprocessing/causal/healthy/cpc_session.m` | the causal preprocessing of one healthy session (step 1, calibration step 2, step 3, export; `cpc_check.py` last) |
| `01_preprocessing/causal/patient/cpp_session.m` | the same for one patient session (`cpp_check.py` last) |
| `02_decoding/run/build_bases.sh` | the three shared bases on the causal triples (`build_basis.sh iw / hglobal / exec`, each checked by `check_basis.py`) |
| `02_decoding/run/make_job_lists.py` | the job lists `exports.csv` and `decodes.csv` |
| `02_decoding/run/export_features.sh` | one feature export (one row of `exports.csv`; `check_export.py` checks it) |
| `02_decoding/run/patient_rest_root.py` | the rest-window patient root of the resting-input control |
| `02_decoding/pyport/run_session.py` | the readout of one exported session or fold (`python -m pyport.run_session`, run from `02_decoding/`) |
| `02_decoding/run/merge_runs.py` | the decoded folds of one run merged per session |
| `02_decoding/fold_tables/fold_centres.py` | per-fold gait frequency and band centres from the training trials (healthy imagined walking, patients) |
| `02_decoding/fold_tables/make_band_tables.py` | the band tables of the folds whose centres differ from the whole-session table |
| `02_decoding/fold_tables/make_fold_tables.py` | the same for real walking and the contiguous-fold control (`iw`: the reproduction check) |
| `02_decoding/run/run_decoder_sessions.sh` | the generic MATLAB launcher the export script calls |
| `02_decoding/simulation/run_simulation.sh` | the Figure 7 simulation: rollouts, traces, rhythm metrics and the window-matched walker |
| `02_decoding/simulation/run_gate_scores_causal.sh` | the Figure 7 walk-intent gate on the causal patient EEG |
| `02_decoding/simulation/gate_scored_events.py` | the gate stream restricted to the events after the calibration cut |
| `03_figures/scoring_mask.csv` | the scored trials (after each session's calibration recording) that every figure script uses |
| `03_figures/sup_decoder_trial_tables.py` | the decoder's fold map, per-trial accuracy and per-trial predictions, written from the decoder output |

## Requirements

The code in this repository requires the following major dependencies:
- MATLAB R2025b (https://www.mathworks.com)
- EEGLAB 2024.2 (https://eeglab.org/), with `clean_rawdata`, `firfilt` and `ICLabel`
- Python 3.13.7, with the packages pinned in `requirements.txt`

Paths are read from environment variables, so that nothing in the analysis code
depends on a particular directory layout: `GAIT_PROJECT_ROOT` (this repository),
`GAIT_DATA_ROOT` and `GAIT_RAW_ROOT` (the recordings), `GAIT_TRIPLE_ROOT` (the
aligned per-session triples the decoder reads) and `GAIT_PREP_OUTPUT`
(preprocessing intermediates: alignment logs and QC tables). The causal preprocessing reads
`CPC_ROOT` and `CPC_EEGLAB` (`01_preprocessing/causal/healthy/cpc_env.m`), and the decode chain
takes its roots from `02_decoding/run/run_paths.sh` / `run_paths.py`, each overridable from the
environment (`MAIN_*`). A few scripts take
further optional variables, named where they are read:
`GAIT_SIM_WALKER`, `GAIT_SIM_TEST`, `GAIT_SIM_DE`, `GAIT_HEALTHY_EXPORT_ROOT`,
`GAIT_GONI_QC_TRIAL_FLAGS`, `GAIT_R_CONV` and `GAIT_F7_FDR_BOOTSTRAP` (needed only when
the Figure 7 statistics are rebuilt from nothing: `fig7_sim_cadence_symmetry.py` then runs
once with it, before `results_statistics.py`, and once again without it). The MATLAB and EEGLAB installations
are located through the MATLAB path in the usual way; the shell runners call `matlab`
and `python` from the `PATH`, or the executables named by `MATLAB` and `PYTHON`.
Rendered panels and workbooks are written to `figures_out/` (one folder per figure,
plus `supplementary/`); create those folders before a first run, as below.

The cohort: 80 healthy sessions from 42 participants, and two participants with
spinal cord injury contributing 15 retained sessions (4 pre-operative and 11
post-operative). The quality-control accounting is in
[`01_preprocessing/QUALITY_CONTROL.md`](01_preprocessing/QUALITY_CONTROL.md).

The example below needs only the released Source Data (it redraws the Figure 3c,f
locking-to-decoding forest and scatter from `04_source_data/fig3/`); run from the
repository root.

```bash
pip install -r requirements.txt
mkdir -p figures_out/fig1 figures_out/fig2 figures_out/fig3 figures_out/fig4          figures_out/fig5 figures_out/fig6 figures_out/fig7 figures_out/supplementary
python 03_figures/fig3_d_dissociation_forest.py
```

## Acknowledgements

We thank the healthy volunteers and the participants with spinal cord injury for
their time and engagement across the recording sessions.

## License

MIT License

Copyright (c) 2026 Wei Zhang, Cuntai Guan and co-authors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files, to deal in the Software without
restriction, subject to the conditions in `LICENSE`.

The contents of `04_source_data/` are released under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/

## Contact

For any questions regarding this code repository, please contact:

PI: Prof. Cuntai Guan  
College of Computing and Data Science,  
Nanyang Technological University, Singapore  
Email: ctguan@ntu.edu.sg


or the author: Dr. Wei Zhang  
College of Computing and Data Science,  
Nanyang Technological University, Singapore  
Email: wilson.zhangwei@ntu.edu.sg
