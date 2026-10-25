"""Shared figure settings

Decoder build read by the figure scripts: the published causal decoder

Imported by the figure scripts; it renders no panel of its own.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SD = ROOT / "04_source_data"
BUILD_ID_ENV = "GAIT_BUILD_ID"

ALIASES = {"main": "main"}

BUILD_IDS = {
    "main": dict(
        name="main",
        gait_build_id="main",
        label="strictly causal EEG preprocessing (minimum-phase anti-aliasing and 0.1-40 Hz band-pass; bad channels, ICA and ICLabel fitted on a calibration recording and applied forward), causal complex-demodulation features cdm_o1_c0p18_causal, harmonic band table estimated per fold from the training trials, shared PCA 30, causal 6 Hz low-pass on the first-stage output, random 10-fold cross-validation, Python readout 02_decoding/pyport; scored on the trials recorded after each session's calibration recording (03_figures/scoring_mask.csv)",
        healthy_tag="main",
        patient_tag="main",
        neural_build_id="main",
        cohorts={'healthy': 'main', 'post_operative': 'post_operative', 'pre_operative': 'pre_operative', 'healthy_n_sessions': 80, 'healthy_n_participants': 42},
        dump_format='pyport_session_h5',
        scoped=False,
        anchors={'iw_session_mean': 0.403732, 'iw_session_median': 0.422981, 'rw_session_mean': 0.831798, 'pat_post_operative_session_mean': 0.459923, 'pat_pre_operative_session_mean': 0.304632, 'iw_10fold_session_mean': 0.403732, 'iw_pooled_trial_median': 0.460905, 'rw_pooled_trial_median': 0.877259, 'pat_post_operative_pooled_trial_median': 0.532959, 'pat_pre_operative_pooled_trial_median': 0.367698, 'iw_median_of_session_medians': 0.465586, 'fig4c_state_rho_iw': 0.70579, 'fig4c_state_rho_rw': 0.72909},
        anchor_tol=5e-05,
        artifacts={
            'h_iw_dump_dir': 'data/decoder_output/main/h_iw_dump',
            'h_iw_summary_csv': 'data/decoder_output/main/h_iw_session_summary.csv',
            'h_iw_trial_r_long': 'data/decoder_output/main/h_iw_trial_r_long.csv',
            'h_session_r': 'data/decoder_output/main/session_r_main.csv',
            'h_session_manifest': 'data/decoder_output/main/session_manifest_main.csv',
            'h_abd_cohort': 'data/decoder_output/main/abd_cohort_main.npz',
            'h_spectra': 'data/decoder_output/main/spectra_main.npz',
            'h_cross_cohort_iw': 'data/decoder_output/main/cross_cohort_main_IW.npz',
            'h_cross_cohort_rest': 'data/decoder_output/main/cross_cohort_main_REST.npz',
            'h_realunit_iw': 'data/decoder_output/main/cohort_main_realunit_mi.npz',
            'h_global_basis': 'data/decoder_output/main/basis_hglobal_cdm_pca30/shared_pca_basis.mat',
            'h_lso_basis_dir': 'data/decoder_output/main/basis_subjout_cdm_pca30',
            'h_rw_dump_dir': 'data/decoder_output/main/h_rw_dump',
            'h_rw_summary_csv': 'data/decoder_output/main/h_rw_session_summary.csv',
            'h_rw_trial_r_long': 'data/decoder_output/main/h_rw_trial_r_long.csv',
            'h_rw_session_manifest': 'data/decoder_output/main/session_manifest_main_rw.csv',
            'h_rw_lso_basis_dir': 'data/decoder_output/main/basis_subjout_exec_cdm_pca30',
            'h_rest_dump_dir': 'data/decoder_output/main/h_rest_testonly_dump',
            'h_rest_summary_csv': 'data/decoder_output/main/h_rest_testonly_session_summary.csv',
            'h_rest_trial_r_long': 'data/decoder_output/main/h_rest_testonly_trial_r_long.csv',
            'h_labelshuf_dir': 'data/decoder_output/main/h_labelshuf',
            'h_cadshuf_summary_csv': 'data/decoder_output/main/h_cadshuf_session_summary.csv',
            'h_blockcv_summary_csv': 'data/decoder_output/main/h_blockcv_session_summary.csv',
            'h_xtask_mi2exec_summary_csv': 'data/decoder_output/main/h_xtask_mi2exec_session_summary.csv',
            'h_xtask_exec2mi_summary_csv': 'data/decoder_output/main/h_xtask_exec2mi_session_summary.csv',
            'h_iw_10fold_summary_csv': 'data/decoder_output/main/h_iw_10fold_pca30_merged.csv',
            'p_dump_dir': 'data/decoder_output/main/pat_post_operative_iw',
            'p_pre_operative_dump_dir': 'data/decoder_output/main/pat_pre_operative_iw',
            'p_trial_r_long': 'data/decoder_output/main/pat_trial_r_long.csv',
            'p_summary_csv': 'data/decoder_output/main/pat_session_summary.csv',
            'p_margin_ci': 'data/decoder_output/main/pat_margin_ci.csv',
            'p_rest_dump_dir': 'data/decoder_output/main/pat_rest_testonly',
            'p_abd_cohort': 'data/decoder_output/main/abd_cohort_pat_post_operative.npz',
            'p_spectra': 'data/decoder_output/main/spectra_pat_post_operative.npz',
            'p_cross_cohort_iw': 'data/decoder_output/main/cross_cohort_pat_post_operative_IW.npz',
            'p_cross_cohort_rest': 'data/decoder_output/main/cross_cohort_pat_rest_testonly.npz',
            'p_cycavg': 'data/decoder_output/main/cycavg_pat.npz',
        },
    ),
}


def resolve(value: str) -> str:
    v = (value or "").strip()
    if v not in ALIASES:
        raise SystemExit(f"{BUILD_ID_ENV}={value!r}: this repository carries one decoder build, "
                         f"{sorted(ALIASES)}; leave {BUILD_ID_ENV} unset")
    return ALIASES[v]


BUILD_ID = resolve(os.environ.get(BUILD_ID_ENV, "main"))
E = BUILD_IDS[BUILD_ID]
TAG = E["healthy_tag"]
PAT_TAG = E["patient_tag"]
NEURAL_BUILD_ID = E["neural_build_id"]
NEURAL_TAG = BUILD_IDS[NEURAL_BUILD_ID]["healthy_tag"]
GAIT_BUILD_ID = E["gait_build_id"]
IS_DEFAULT = False
SCOPED = E["scoped"]
OUT_DIRNAME = BUILD_ID
COHORTS = E["cohorts"]
DUMP_FORMAT = E["dump_format"]
DECODER_BUILDS = ("main",)
MAIN = BUILD_ID

SIM_INPUT_BUILD_ID = BUILD_ID
SIM_IN_DIRNAME = OUT_DIRNAME


def sim_in(p) -> Path:
    """Location of a simulation table: unchanged (one build)."""
    return Path(p)


SCORE_MASK = "after_calibration"
SCORE_MASK_CSV = ROOT / "03_figures" / "scoring_mask.csv"
MASK_CONDITIONS = ("IW", "RW", "REST")
RUN_CONDITION = {
    "h_iw_dump": "IW", "h_iw_10fold_pca30": "IW", "h_blockcv_dump": "IW", "h_cadshuf_dump": "IW",
    "h_xtask_exec2mi_dump": "IW", "pat_post_operative_iw": "IW", "pat_pre_operative_iw": "IW", "pat_circshift_null": "IW",
    "pat_zero_null": "IW", "h_rw_dump": "RW", "h_xtask_mi2exec_dump": "RW", "h_rest_testonly_dump": "REST",
    "pat_rest_testonly": "REST",
}
XTASK_RUNS = ("h_xtask_mi2exec_dump", "h_xtask_exec2mi_dump")
_MASK_CACHE = None


def mask_ordinal(run) -> str:
    """Column that carries the mask's trial ordinal in a run's trial_r.csv / dump: test_ordinal for cross-task runs,
    else trial_idx."""
    p = Path(run)
    return "test_ordinal" if (p.name in XTASK_RUNS or p.parent.name in XTASK_RUNS) else "trial_idx"


def _mask():
    global _MASK_CACHE
    if _MASK_CACHE is None:
        import csv as _csv
        keep, known = {}, {}
        with open(SCORE_MASK_CSV, newline="") as fh:
            for r in _csv.DictReader(fh):
                k = (r["session"], r["condition"])
                known.setdefault(k, set()).add(int(r["trial_idx"]))
                if r["keep"] == "1":
                    keep.setdefault(k, set()).add(int(r["trial_idx"]))
        _MASK_CACHE = (keep, known)
    return _MASK_CACHE


def mask_condition(run) -> str:
    """Test condition (IW / RW / REST) of a decoder run folder, or of a file inside one."""
    p = Path(run)
    for name in (p.name, p.parent.name):
        if name in RUN_CONDITION:
            return RUN_CONDITION[name]
    raise SystemExit(f"analysis_config.mask_condition: {run} is not a known decoder run folder ({sorted(RUN_CONDITION)})")


def keep_trial(session, condition: str, trial_idx) -> bool:
    """True when the trial is scored. A (session, condition, trial_idx) that the mask table does not list stops the run."""
    if SCORE_MASK is None:
        return True
    if condition not in MASK_CONDITIONS:
        raise SystemExit(f"analysis_config.keep_trial: condition {condition!r} not in {MASK_CONDITIONS}")
    keep, known = _mask()
    k, t = (str(session), condition), int(float(trial_idx))
    if t not in known.get(k, ()):
        raise SystemExit(f"analysis_config.keep_trial: {k} trial_idx {t} is not in {SCORE_MASK_CSV.name}")
    return t in keep.get(k, ())


def filter_trials(df, condition: str, session_col: str = "session", trial_col: str = "trial_idx"):
    """Rows of a DataFrame (one per trial, or per trial x joint) whose trial is scored; df itself without a mask."""
    if SCORE_MASK is None:
        return df
    m = [keep_trial(s, condition, t) for s, t in zip(df[session_col], df[trial_col])]
    return df.loc[[bool(x) for x in m]]


def require(*eras: str) -> str:
    allowed = {resolve(e) for e in eras}
    if BUILD_ID not in allowed:
        raise SystemExit(f"{Path(sys.argv[0]).name}: needs decoder build {sorted(allowed)}, "
                         f"this repository carries {BUILD_ID}")
    return BUILD_ID


def require_dump_format(fmt: str) -> None:
    if DUMP_FORMAT != fmt:
        raise SystemExit(f"{Path(sys.argv[0]).name}: reads decoder dumps as {fmt!r}, "
                         f"the decoder writes {DUMP_FORMAT!r}")


def rel(key: str) -> str:
    """Path of a decoder output, relative to the repository root."""
    try:
        return E["artifacts"][key]
    except KeyError:
        raise SystemExit(f"no decoder output {key!r} is registered in analysis_config.py") from None


def path(key: str) -> Path:
    return ROOT / rel(key)


def scoped(p) -> Path:
    """Location of a rendered panel, table or Source Data file: unchanged (one build)."""
    return Path(p)


def sd_group(name: str) -> str:
    m = re.match(r"(?:sup_)?fig([1-7])", name)
    return f"fig{m.group(1)}" if (m and not name.startswith("sup")) else "supplementary"


def sd(name: str) -> Path:
    """Source Data file 04_source_data/<group>/<name>."""
    return SD / sd_group(name) / name


def sd_in(name: str) -> Path:
    return sd(name)


DECODER_SD: frozenset = frozenset()

F7_SIM = os.environ.get("GAIT_F7_SIM", "causal_from2s").strip()
if F7_SIM != "causal_from2s":
    raise SystemExit(f"GAIT_F7_SIM={F7_SIM!r}: this repository carries one simulation build, "
                     "causal_from2s; leave GAIT_F7_SIM unset")
F7_SIM_MD = ("f7_simulation", "Published simulation build: the biped is driven from trial 2.0 s, with no set-point hold and no standing period, by the gait phase that the causal extractor reads from each condition's own joined input [trial 2-4 s ; main stream]: Walker = its own goniometer from 2 s (window-matched); Decoded = [decoder stage-1 output A_pred over trial 2-4 s ; level-matched final output C' = C - C[onset] + A_pred[onset - 1], per joint]; Coordination / In-phase null = per-joint / channel-shared phase randomisation of the WHOLE joined Decoded stream (2-4 s drive included; seed = trial_idx). Causal extractor = forward 2nd-order Butterworth [0.5, 1.5] f0 -> causal notches at 2 f0 and 3 f0 (Q 1) -> quadrature from the backward difference (half sample and filter phase at f0 compensated in closed form), f0 = per-session LOSO cadence prior (02_decoding/simulation/f0_cadence_prior.csv), started at the first sample (no input before 2.0 s); template de-mean; rig torque clamp 280 N*m (kp 1200). Every outcome is measured from the onset of the final decoder output C_pred (trial 4 s; time re-zeroed 2.0 s into the commanded stream): forward displacement 3.0 s after that onset, cadence / asymmetry / step length from the onset to the trial end (durations = the C_pred span); falls counted over the whole rollout from trial 2 s.")


def anchor(key: str):
    """(value, tolerance, pinned) of a reference value of the published decoder; None = not recorded."""
    return E["anchors"].get(key), E["anchor_tol"], True


def check_anchor(key: str, observed: float, what: str = "") -> None:
    """Stop when an aggregate of the inputs differs from the published decoder's value."""
    val, tol, _ = anchor(key)
    who = Path(sys.argv[0]).name
    if val is None:
        print(f"[analysis_config] {who}: no reference value for {key!r}; observed {observed:.6f} "
              f"({what}) not checked", file=sys.stderr)
        return
    if abs(observed - val) > tol:
        raise SystemExit(f"ANCHOR GUARD ({who}): {what or key} = {observed:.6f}, but the published "
                         f"decoder gives {key} = {val} (tol {tol}); the inputs are not its output. "
                         "Refusing to write.")


def check_tag(s, kind: str = "decoder", what: str = "") -> None:
    """Decoder outputs sit under data/decoder_output/main/ and their sidecars carry its
    tag; Source Data under 04_source_data/ come from it by construction; the zero-phase neural
    inputs (kind="neural") do not depend on the decoder."""
    s = str(s)
    if kind == "neural" or TAG in s or "04_source_data" in s:
        return
    raise SystemExit(f"INPUT CHECK ({Path(sys.argv[0]).name}): {what or 'input'} {s!r} is not an "
                     f"output of the published decoder.")


def describe() -> str:
    return f"{BUILD_ID} ({E['label']})"

