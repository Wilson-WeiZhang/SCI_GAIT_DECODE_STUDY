"""Figure 5 — Decoding imagined walking after spinal cord injury

c,d - the single held-out patient trial that panels c and d both show

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations

import csv
import os
import sys

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import project_palette as P
import analysis_config as ec

ec.require("main", "main")
SNAPS_DEFAULT = "onestride" if ec.BUILD_ID in ec.DECODER_BUILDS else "minimax"
if ec.BUILD_ID == ec.MAIN:
    SNAPS_DEFAULT = "twostride"
MAIN_SNAPS = os.environ.get("MAIN_SNAPS", SNAPS_DEFAULT)
if (MAIN_SNAPS not in ("minimax", "onestride", "twostride") or (ec.IS_DEFAULT and "MAIN_SNAPS" in os.environ)
        or (MAIN_SNAPS == "twostride" and ec.BUILD_ID != ec.MAIN)):
    raise SystemExit("MAIN_SNAPS: values minimax | onestride | twostride (default twostride)")
ONESTRIDE = MAIN_SNAPS == "onestride"
TWOSTRIDE = MAIN_SNAPS == "twostride"
CANDIDATE = MAIN_SNAPS != SNAPS_DEFAULT
SESS = "pat02_sess06"
FOLD = "6"
ORD = 2
SNAPS = np.array([4.39, 4.66, 4.93, 5.20, 5.47, 5.74, 6.01, 6.28, 6.55])
if not ec.IS_DEFAULT:
    import dump_h5
    TRIAL_IDX = 26
    FOLD = "33"
    ORD = None
    SNAPS = np.array([4.23, 4.33, 4.43, 4.53, 4.63, 4.73, 4.83, 4.93, 5.03])
    if ONESTRIDE:
        SNAPS = np.array([4.28, 4.43, 4.58, 4.73, 4.88, 5.03, 5.18, 5.33, 5.48])
    if ec.BUILD_ID in ec.DECODER_BUILDS:
        FOLD = "3"
        SNAPS = np.array([4.59, 4.67, 4.75, 4.83, 4.91, 4.99, 5.07, 5.15, 5.23])
        if ONESTRIDE:
            SNAPS = np.array([4.55, 4.70, 4.85, 5.00, 5.15, 5.30, 5.45, 5.60, 5.75])
    if ec.BUILD_ID in ec.DECODER_BUILDS:
        SNAPS = np.array([4.11, 4.19, 4.27, 4.35, 4.43, 4.51, 4.59, 4.67, 4.75])
        if ONESTRIDE:
            SNAPS = np.array([5.87, 6.02, 6.17, 6.32, 6.47, 6.62, 6.77, 6.92, 7.07])
        if TWOSTRIDE:
            SNAPS = np.array([4.11, 4.40, 4.69, 4.98, 5.27, 5.56, 5.85, 6.14, 6.43])
AMP = "calib"
W_MM, H_MM = 183.0, 66.0
FS = 250.0
PAD_L_MM = 14.0
FS_SCALE, JNFS = 2.0, 4.6
TFS = JNFS * FS_SCALE
BODYFS = TFS * 0.8
YLX, YLX_R = -0.085, -0.030
GS_L = PAD_L_MM / W_MM
WCOL = "#141414"

DUMP = os.path.join(ec.path("p_dump_dir"),
                    f"{SESS}_fold{FOLD}_trial_dump.mat" if ec.IS_DEFAULT else f"{SESS}_dump.h5")
NPZ = os.path.join(ROOT, "data", "decoder_output", "cycavg_patient.npz")
NPZ_TEMPLATE = os.path.join(ROOT, "data", "decoder_output", "cycavg_pat02_sess06_loto_f6o2.npz")
if not ec.IS_DEFAULT:
    NPZ_TEMPLATE = os.fspath(ec.scoped(os.path.join(ROOT, "data", "decoder_output", ec.BUILD_ID,
                                                f"cycavg_{SESS}_loto_t{TRIAL_IDX}.npz")))
    ec.check_tag(DUMP, what="trial dump")
    ec.check_tag(NPZ_TEMPLATE, what="calibration template")
OUT_PNG = os.fspath(ec.scoped(os.path.join(ROOT, "figures_out", "fig5", "fig5d_patient_trial_traces.png")))
SD_DIR = os.path.join(ROOT, "04_source_data", "fig5")
OUT_SD = os.fspath(ec.sd("fig5_cd_patient_trial_example_source_data.csv"))
OUT_MD = os.fspath(ec.sd("fig5_cd_patient_trial_example_metadata.csv"))
if CANDIDATE:
    CAND_DIR = os.path.join(ROOT, "figures_out", "fig5", "_candidates", ec.OUT_DIRNAME)
    os.makedirs(CAND_DIR, exist_ok=True)
    OUT_PNG = os.path.join(CAND_DIR, f"fig5d_patient_trial_traces_{MAIN_SNAPS}.png")
    OUT_SD = os.path.join(CAND_DIR, f"fig5_cd_patient_trial_example_{MAIN_SNAPS}_source_data.csv")
    OUT_MD = os.path.join(CAND_DIR, f"fig5_cd_patient_trial_example_{MAIN_SNAPS}_metadata.csv")

plt.rcParams.update({"font.family": "Arial", "font.size": 6.0, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "xtick.direction": "out", "ytick.direction": "out",
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.default": "it"})

if ec.IS_DEFAULT:
    ec.require_dump_format("matlab_fold_mat")
    with h5py.File(DUMP, "r") as h:
        Y = np.asarray(h[h["Y_true"][ORD, 0]]).T
        C = np.asarray(h[h["C_pred"][ORD, 0]]).T
        TS = np.asarray(h[h["time_sec"][ORD, 0]]).ravel()
        _tl = h[h["meta/target_labels"][0, 0]]
        TARGET_LABELS = [''.join(chr(c) for c in np.asarray(h[_tl[j, 0]]).ravel()) for j in range(6)]
else:
    ec.require_dump_format("pyport_session_h5")
    _tr = dump_h5.load_trial(DUMP, TRIAL_IDX, ("Y_true", "C_pred"))
    assert str(_tr["fold"]) == FOLD, "fold id of the picked trial changed"
    Y, C, TS = _tr["Y_true"].T, _tr["C_pred"].T, _tr["time_sec"]
    TARGET_LABELS = list(_tr["joints"])
m = np.isfinite(C).all(1) & np.isfinite(Y).all(1)
Yf, Cf, t = Y[m], C[m], TS[m]
mY = np.isfinite(Y).all(1)
tW, W = TS[mY], Y[mY]

zl = np.load(NPZ_TEMPLATE, allow_pickle=True)
assert str(zl["sessions"][0]) == SESS and str(zl["excluded"]) == (
    f"{FOLD}:{ORD}" if ec.IS_DEFAULT else str(TRIAL_IDX))
assert zl["joint_ok"].all()
if not ec.IS_DEFAULT:
    ec.check_tag(str(zl["source_dump"]), what="calibration template source dump")
TEMPLATE_TRIALS = [str(s) for s in zl["trials_used"]]
TEMPLATE_NCYC = int(zl["ncyc"][0])
T = zl["obs"][0]
gains = [float(T[j].std() / max(Cf[:, j].std(), 1e-9)) for j in range(6)]
Acal = np.stack([T[j].mean() + gains[j] * (Cf[:, j] - Cf[:, j].mean()) for j in range(6)], 1)
D = Acal if AMP == "calib" else Cf

RJ = [float(np.corrcoef(Yf[:, j], Cf[:, j])[0, 1]) for j in range(6)]

S = np.array([+1, -1, +1, -1, +1, -1], float)
W, D, Craw = W * S, D * S, Cf * S

print(f"trial {SESS} " + (f"fold{FOLD} ORD{ORD}" if ec.IS_DEFAULT else f"trial_idx {TRIAL_IDX} (fold {FOLD})")
      + f": {len(t)} samples {t[0]:.2f}-{t[-1]:.2f} s; AMP={AMP}")
print(f"gain-fit template: {SESS} leave-this-trial-out, {len(TEMPLATE_TRIALS)} trials, {TEMPLATE_NCYC} cycles")
print("gains " + " ".join(f"{g:.3f}" for g in gains))
print("per-joint r " + "  ".join(
    f"{TARGET_LABELS[j].replace('W X', '').strip()} {RJ[j]:.2f}" for j in range(6)))

if not ec.IS_DEFAULT:
    import hashlib
    P2_POST_OPERATIVE = ("pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08", "pat02_sess09")
    _acc = {}
    with open(ec.path("p_trial_r_long"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["session"] in P2_POST_OPERATIVE and r["model"] == "calib":
                _a = _acc.setdefault((r["session"], int(r["trial_idx"])), [int(r["n_samples"]), {}])
                _a[1][r["joint"]] = float(r["rpt_r"] or "nan")
    PICK_STAT = {k: (v[0], float(np.nanmean(list(v[1].values())))) for k, v in _acc.items()}
    PICK_MEDIAN = float(np.median([s[1] for s in PICK_STAT.values()]))
    PICK_CANDS = sorted(((abs(s[1] - PICK_MEDIAN), k) for k, s in PICK_STAT.items() if s[0] >= 1000))
    PICK_CANDS = [(k, PICK_STAT[k][1]) for _, k in PICK_CANDS]
    assert PICK_CANDS[0][0] == (SESS, TRIAL_IDX), f"trial rule now picks {PICK_CANDS[0]}"
    _tab = _acc[(SESS, TRIAL_IDX)][1]
    assert np.allclose(RJ, [_tab[lab] for lab in TARGET_LABELS], atol=1e-9), "per-joint r != table rpt_r"
    M81_CSV = os.path.join(ROOT, "methods", "simulation", "simulation",
                           f"m81_snap_window_search_{SESS}_t{TRIAL_IDX}_proximal_{ec.BUILD_ID}.csv")
    _w = np.loadtxt(M81_CSV, delimiter=",", skiprows=1)
    if ONESTRIDE:
        _w = _w[np.abs(_w[:, 6] - 1.0) <= 0.12]
    if TWOSTRIDE:
        _w = _w[np.abs(_w[:, 6] - 2.0) <= 0.12]
    M81_ROW = _w[np.argmin(_w[:, 4])]
    assert np.allclose(M81_ROW[0] + M81_ROW[1] * np.arange(len(SNAPS)), SNAPS, atol=1e-6), "SNAPS != m81 minimax"
    with open(DUMP, "rb") as fh:
        DUMP_MD5 = hashlib.md5(fh.read()).hexdigest()
    print(f" pick: {len(PICK_STAT)} P2 post-op trials, median {PICK_MEDIAN:.6f}; eligible (>= 1000 samples) "
          + "; ".join(f"{k[0]} t{k[1]} {v:.6f}" for k, v in PICK_CANDS))
    print(f"m81 minimax: start {M81_ROW[0]:.2f} step {M81_ROW[1]:.2f} worst {M81_ROW[4]:.2f} cm "
          f"mean {M81_ROW[3]:.2f} cm span {M81_ROW[6]:.2f} stride")

TRIM_END_S = float(os.environ.get("FIG5D_TRIM_END_S", "1.0"))
T_FULL0, T_FULL1 = float(t[0]), float(t[-1])
T0, T1 = T_FULL0, T_FULL1 - TRIM_END_S
assert T1 > SNAPS[-1], "the trimmed window must still contain every t1-t9 instant"
sel = (t >= T0) & (t <= T1)
selW = (tW >= T0) & (tW <= T1)

SIDE = ["L", "L", "L", "R", "R", "R"]
JOINT = ["hip", "knee", "ankle", "hip", "knee", "ankle"]
os.makedirs(SD_DIR, exist_ok=True)
snap_idx = [int(np.argmin(np.abs(t - s))) for s in SNAPS]
with open(OUT_SD, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["row_type", "instant", "time_s", "side", "joint", "reference_deg",
                "decoded_calibrated_deg", "decoded_raw_deg", "trial_r", "amplitude_gain"])
    for j in range(6):
        for i in np.nonzero(sel)[0]:
            w.writerow(["series", "", f"{t[i]:.4f}", SIDE[j], JOINT[j],
                        f"{W[selW][i, j]:.6f}", f"{D[i, j]:.6f}", f"{Craw[i, j]:.6f}", "", ""])
    for k, i in enumerate(snap_idx):
        for j in range(6):
            w.writerow(["instant", f"t{k + 1}", f"{t[i]:.4f}", SIDE[j], JOINT[j],
                        f"{W[selW][i, j]:.6f}", f"{D[i, j]:.6f}", f"{Craw[i, j]:.6f}", "", ""])
    for j in range(6):
        w.writerow(["joint_summary", "", "", SIDE[j], JOINT[j], "", "", "",
                    f"{RJ[j]:.6f}", f"{gains[j]:.6f}"])
print("wrote", OUT_SD)


def _meta_base():
    """Metadata of the panel."""
    rj = lambda side: " / ".join(f"{RJ[j]:.2f}" for j in range(6) if SIDE[j] == side)
    snaps = ",".join(f"{s:.2f}" for s in SNAPS)
    rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, "/")
    panel_c = ("figures_out/fig5/_candidates/main/fig5c_patient_trial_filmstrip_onestride.png" if CANDIDATE
               else "figures_out/fig5/main/fig5c_patient_trial_filmstrip.png")
    status = [("status", "CANDIDATE: t1-t9 = the m81 minimax among windows "
                         "spanning about one stride")]
    return (status if CANDIDATE else []) + [
        ("figure", "Fig.5 c (skeleton filmstrip, nine instants) and d (six joint-angle traces) - "
                   "one held-out imagined-walking trial"),
        ("cohort", "not applicable - this is a SINGLE held-out trial, not a cohort statistic; the trial was "
                   "selected from the patient 2 trials of cohort post_operative"),
        ("decode_tag", ec.PAT_TAG),
        ("decoder", ec.E["label"]),
        ("basis", "frozen all-80-healthy pooled shared PCA basis, 30 components, on the causal "
                  "complex-demodulation features cdm_o1_c0p18_causal"),
        ("readout", "patient-trained random 10-fold CV (representation transfer, not zero-shot); "
                    "every trial is its own fold"),
        ("trial", f"{SESS} (patient 2, post-operative session 3 of the 5 analysed), trial_idx {TRIAL_IDX} "
                  f"(fold {FOLD}, its only test trial)"),
        ("n_samples", str(int(sel.sum()))),
        ("sampling_rate_hz", "250"),
        ("snap_instants_s", snaps),
        ("snap_instants_note", "t1-t9 as drawn; the CSV's instant rows carry the nearest sample, "
                               "which is what both panels draw."),
        ("amplitude_calibration", "decoded_calibrated = template_mean + (template_sd / decoded_sd) * "
                                  "(decoded - decoded_mean), per joint. Template = the gait-cycle "
                                  "average of the gain_fit_cohort trials (template_mean / template_sd "
                                  "= mean / SD of that 50-bin curve). Timing and shape are "
                                  "the decoder's own; only amplitude and offset are set by the template."),
        ("amplitude_gains", ", ".join(f"{SIDE[j]} {JOINT[j]} {gains[j]:.3f}" for j in range(6))
                            + f" (range x{min(gains):.2f}-{max(gains):.2f})"),
        ("per_joint_r", ", ".join(f"{SIDE[j]} {JOINT[j]} {RJ[j]:.3f}" for j in range(6))
                        + f"; rounded to 2 dp for the legend this reads {rj('L')} (left) and {rj('R')} "
                          f"(right hip, knee, ankle)"),
        ("r_definition", "trial-level Pearson r of C_pred against Y_true over the finite window. Under "
                         "one-trial folds the trial is its own fold, so this equals the trial's rpt_r "
                         "in result/main/pat_trial_r_long.csv (asserted to 1e-9 by this script). "
                         "Amplitude calibration and the display sign are both per-channel linear, so neither "
                         "moves r."),
        ("display_sign", "S = [+1,-1,+1,-1,+1,-1] applied to all series in this CSV: undoes the "
                         "mirrored right-side goniometer mounting and flips the knee, giving the "
                         "clinical convention (hip flexion +, knee flexion +, ankle dorsiflexion +). "
                         "Decoding runs on the RAW channels throughout; this is display only."),
        ("panel_c_png", panel_c),
        ("panel_d_png", rel(OUT_PNG)),
        ("scope", "n=2 SCI patients, case-level, descriptive; this is ONE example trial and is NOT the "
                  "best trial and NOT a cohort statistic. No population claim."),
        ("output_csv", rel(OUT_SD)),
    ]


def _meta_tenfold():
    """Metadata with the random 10-fold CV provenance."""
    rep = [("main", ec.BUILD_ID), ("GAIT_BUILD_ID=", "GAIT_BUILD_ID="),
           (f"(fold {FOLD}, its only test trial)", f"(random 10-fold CV, fold {FOLD})"),
           ("Under one-trial folds the trial is its own fold, so this equals the trial's rpt_r",
            "rpt_r is the per-trial r of the held-out trial (computed per trial in every CV scheme), so this "
            "equals the trial's rpt_r")]
    out = []
    for k, v in _meta_base():
        if k == "decode_run":
            v = ("random 10-fold, causal 6 Hz low-pass on the stage-1 output; reproduced byte-identically by "
                 "02_decoding/pyport/run_session.py with pyport/configs/tenfold_pca30_lpf6.json; "
                 f"dumps in data/decoder_output/{ec.BUILD_ID}/pat_post_operative_iw/")
        elif k == "readout":
            v = "patient-trained random 10-fold CV (representation transfer, not zero-shot)"
        for a, b in rep:
            v = v.replace(a, b)
        out.append((k, v))
    snaps = ",".join(f"{s:.2f}" for s in SNAPS)
    m81_rel = os.path.relpath(M81_CSV, ROOT).replace(os.sep, "/")
    _wa = np.loadtxt(M81_CSV, delimiter=",", skiprows=1)
    _glob = _wa[np.argmin(_wa[:, 4])]
    _one = _wa[np.abs(_wa[:, 6] - 1.0) <= 0.12]
    _one = _one[np.argmin(_one[:, 4])]
    _alt = _glob if ONESTRIDE else _one
    _alt_txt = (f"start {_alt[0]:.2f} s, step {_alt[1]:.2f} s, span {_alt[6]:.2f} stride, worst frame "
                f"{_alt[4]:.2f} cm")
    strip = (f"Selected strip start {M81_ROW[0]:.2f} s, step {M81_ROW[1]:.2f} s, worst frame {M81_ROW[4]:.2f} cm, "
             f"mean {M81_ROW[3]:.2f} cm, span {M81_ROW[2]:.2f} s = {M81_ROW[6]:.2f} stride; table {m81_rel}. The "
             f"search calibrates with the leave-this-trial-out template above.")
    search = (f"m81_snap_window_search.py (not distributed), SESS={SESS} TRIAL={TRIAL_IDX} "
              f"NSNAP=9 T0=4.0 T1=8.0 LANDMARKS=proximal: nine equally spaced instants; criterion = the WORST "
              f"frame's mean knee + ankle joint-centre distance between the posed walker and the posed calibrated "
              f"decode (OpenSim gait2354, pelvis frame), minimised ")
    if ONESTRIDE:
        sel_txt = (search + "among the windows whose span is within 12% of one stride (|span in strides - 1| <= "
                   "0.12). "
                    + strip)
    else:
        sel_txt = (search + "over all windows. "
                   
                   + strip)
    if ONESTRIDE:
        panel_c = f"figures_out/fig5/{ec.OUT_DIRNAME}/fig5c_patient_trial_filmstrip.png"
        render = (f"OUT=figures_out/fig5/_candidates/{ec.OUT_DIRNAME}/fig5c_patient_trial_filmstrip_onestride.png")
    else:
        panel_c = f"figures_out/fig5/_candidates/{ec.OUT_DIRNAME}/fig5c_patient_trial_filmstrip_minimax.png"
        render = (f"default OUT = figures_out/fig5/{ec.OUT_DIRNAME}/fig5c_patient_trial_filmstrip.png")
    with open(os.path.join(ROOT, panel_c), "rb") as fh:
        panel_c_md5 = hashlib.md5(fh.read()).hexdigest()
    rendering = (f"The filmstrip PNG is rendered by render_skeleton_filmstrip.py (not distributed) on the compute server "
                 f"(SESS={SESS} TRIAL={TRIAL_IDX} SNAPS={snaps} SKEL=1 HEADER=t, {render}; the render log prints exactly "
                 f"these nine instants). {panel_c} is a byte copy of that render (md5 {panel_c_md5}). OpenSim "
                 f"gait2354 forward kinematics. The renderer reads the same "
                 f"dump and the same leave-this-trial-out template and applies the same calibration, so the "
                 f"row_type=instant rows here ARE its input. Trunk, head and condition are a fixed template (nothing above "
                 f"the pelvis is recorded); the pelvis is pinned and no forward progression is drawn.")
    new = {"snap_selection": sel_txt, "panel_c_rendering": rendering, "panel_c_png": panel_c,
           "status": ("CANDIDATE: t1-t9 = the m81 global minimax over all windows "
                      "; the official  build uses the "
                      "minimax among windows spanning about one stride")}
    out = [(k, new.get(k, v)) for k, v in out]
    assert [k for k, _ in out].count("status") == int(CANDIDATE)
    return out


def _meta_zero_phase_prep():
    """The metadata of the ten-fold build on the zero-phase preprocessing."""
    out = []
    for k, v in _meta_tenfold():
        if k == "decode_run":
            v = 'random 10-fold, causal 6 Hz low-pass on the stage-1 output, harmonic band table estimated per fold from the training trials, on the zero-phase preprocessing'
        out.append((k, v))
    return out


def _meta_main():
    """The metadata of the published decoder (causal preprocessing, per-fold band table)."""
    out = []
    for k, v in _meta_zero_phase_prep():
        if k == "decode_run":
            v = ('random 10-fold, causal 6 Hz low-pass on the stage-1 output, harmonic band table estimated per fold from the training trials, on the strictly causal preprocessing: every fold decoded from the feature export of its band table (02_decoding/run/export_features.sh, job list 02_decoding/run/make_job_lists.py) and the folds merged per session (02_decoding/run/merge_runs.py); 02_decoding/pyport/run_session.py with pyport/configs/tenfold_pca30_lpf6.json'
                 f"; dumps in data/decoder_output/{ec.BUILD_ID}/pat_post_operative_iw/")
        out.append((k, v))
    return out


def _meta_scored():
    """"""
    out = _meta_main()
    if not TWOSTRIDE:
        return out
    snaps = ",".join(f"{s:.2f}" for s in SNAPS)
    m81_rel = os.path.relpath(M81_CSV, ROOT).replace(os.sep, "/")
    _wa = np.loadtxt(M81_CSV, delimiter=",", skiprows=1)
    _one = _wa[np.abs(_wa[:, 6] - 1.0) <= 0.12]
    _one = _one[np.argmin(_one[:, 4])]
    panel_c = f"figures_out/fig5/{ec.OUT_DIRNAME}/fig5c_patient_trial_filmstrip.png"
    with open(os.path.join(ROOT, panel_c), "rb") as fh:
        panel_c_md5 = hashlib.md5(fh.read()).hexdigest()
    new = {
        "snap_selection": (
            f"m81_snap_window_search.py (not distributed), SESS={SESS} TRIAL={TRIAL_IDX} NSNAP=9 "
            f"T0=4.0 T1=8.0 LANDMARKS=proximal: nine equally spaced instants; criterion = the WORST frame's mean knee + "
            f"ankle joint-centre distance between the posed walker and the posed calibrated decode (OpenSim gait2354, "
            f"pelvis frame), minimised among the windows whose span is within 12% of two strides (|span in strides - 2| "
            f"<= 0.12). The published strip spans at least 2 s (t1-t9). "
            
            f"Selected strip start {M81_ROW[0]:.2f} s, step {M81_ROW[1]:.2f} s, worst "
            f"frame {M81_ROW[4]:.2f} cm, mean {M81_ROW[3]:.2f} cm, span {M81_ROW[2]:.2f} s = {M81_ROW[6]:.2f} stride; "
            f"table {m81_rel}. The search calibrates with the leave-this-trial-out template above."),
        "panel_c_rendering": (
            f"The filmstrip PNG is rendered by render_skeleton_filmstrip.py (not distributed) on the compute server "
            f"(SESS={SESS} TRIAL={TRIAL_IDX} SNAPS={snaps} SKEL=1 HEADER=t, OUT=figures_out/fig5/_candidates/"
            f"main/fig5c_patient_trial_filmstrip_twostride.png" f"). "
            f"{panel_c} is a byte copy of that render "
            f"(md5 {panel_c_md5}). OpenSim gait2354 forward kinematics. "
            f"The renderer reads the same dump and the same leave-this-trial-out template and applies the same calibration, so "
            f"the row_type=instant rows here ARE its input. Trunk, head and condition are a fixed template (nothing above the "
            f"pelvis is recorded); the pelvis is pinned and no forward progression is drawn."),
        "panel_c_png": panel_c}
    return [(k, new.get(k, v)) for k, v in out]


META = ((_meta_scored() if ec.BUILD_ID == ec.MAIN else _meta_main() if ec.BUILD_ID in ec.DECODER_BUILDS else
         _meta_zero_phase_prep() if ec.BUILD_ID == "zero_phase_prep" else _meta_tenfold()) if ec.BUILD_ID in ec.DECODER_BUILDS
        else _meta_base()) if not ec.IS_DEFAULT else [
    ("figure", "Fig.5 c (skeleton filmstrip, nine instants) and d (six joint-angle traces) - "
               "one held-out imagined-walking trial"),
    ("cohort", "not applicable - this is a SINGLE held-out trial, not a cohort statistic"),
    ("decode_tag", "patientdump" if ec.IS_DEFAULT else ec.PAT_TAG),
    ("readout", "patient-trained 10-fold CV (representation transfer, not zero-shot)"),
    ("trial", f"{SESS} (patient 2, post-operative session 3 of the 5 analysed), fold {FOLD}, "
              f"trial ordinal {ORD}"),
    ("n_samples", str(int(sel.sum()))),
    ("sampling_rate_hz", "250"),
    ("snap_instants_s", ",".join(f"{s:.2f}" for s in SNAPS)),
    ("snap_instants_note", "t1-t9 as drawn; the CSV's instant rows carry the nearest sample, "
                           "which is what both panels draw."),
    ("amplitude_calibration", "decoded_calibrated = template_mean + (template_sd / decoded_sd) * "
                              "(decoded - decoded_mean), per joint. Template = the gait-cycle "
                              "average of the gain_fit_cohort trials (template_mean / template_sd "
                              "= mean / SD of that 50-bin curve). Timing and shape are "
                              "the decoder's own; only amplitude and offset are set by the template."),
    ("per_joint_r", ", ".join(f"{SIDE[j]} {JOINT[j]} {RJ[j]:.3f}" for j in range(6))
                    
                      ),
    ("display_sign", "S = [+1,-1,+1,-1,+1,-1] applied to all series in this CSV: undoes the "
                     "mirrored right-side goniometer mounting and flips the knee, giving the "
                     "clinical convention (hip flexion +, knee flexion +, ankle dorsiflexion +). "
                     "Decoding runs on the RAW channels throughout; this is display only."
                     
                     ),
    ("panel_d_png", "figures_out/fig5/fig5d_patient_trial_traces.png"),
    ("scope", "n=2 SCI patients, case-level, descriptive; this is ONE example trial and is NOT the "
              "best trial and NOT a cohort statistic. No population claim."),
    ("rendered_by", "03_figures/fig5_cd_patient_trial_example.py"),
    ("output_csv", "fig5_cd_patient_trial_example_source_data.csv"),
]
with open(OUT_MD, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["key", "value"])
    w.writerows(META)
print("wrote", OUT_MD)

fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
gs = gridspec.GridSpec(3, 2, figure=fig, left=GS_L, right=0.995, top=0.905, bottom=0.135,
                       wspace=0.115, hspace=0.46)
_cell = lambda r, c: fig.add_subplot(gs[r, c])
_LW_SNAP, _LW_WALK, _LW_DEC, _FS_TICK, _FS_T, _FS_TITLE = 0.45, 0.75, 1.0, BODYFS, TFS, BODYFS
JR = [("hip", P.HIP, 0, 3), ("knee", P.KNEE, 1, 4), ("ankle", P.ANKLE, 2, 5)]
for r, (nm, col, jl, jr) in enumerate(JR):
    vv = np.r_[W[selW, jl], D[sel, jl], W[selW, jr], D[sel, jr]]
    pad = 0.12 * (vv.max() - vv.min())
    ylo, yhi = vv.min() - pad, vv.max() + pad
    for c, j in enumerate((jl, jr)):
        ax = _cell(r, c)
        for x in SNAPS:
            ax.axvline(x, color="0.6", lw=_LW_SNAP, ls=(0, (2.0, 1.8)), zorder=1)
        ax.plot(tW[selW], W[selW, j], "-", color=WCOL, lw=_LW_WALK, zorder=3)
        ax.plot(t[sel], D[sel, j], "-", color=col, lw=_LW_DEC, zorder=4)
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(T0, T1)
        st = 10 if (yhi - ylo) < 55 else 20
        tk = [q for q in np.arange(-180, 181, st)
              if ylo + 0.05 * (yhi - ylo) < q < yhi - 0.05 * (yhi - ylo)]
        ax.set_yticks([tk[0], tk[-1]] if len(tk) >= 2 else tk)
        ax.tick_params(length=2, pad=1.2, labelsize=BODYFS)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if c == 1:
            ax.set_yticklabels([])
        if r < 2:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=1.2)
        else:
            ax.set_xlabel("Time (s)", fontsize=_FS_TITLE, labelpad=2.0)
        if r == 0:
            for i2, x in enumerate(SNAPS):
                ax.text(x, 1.02, rf"$t_{{{i2 + 1}}}$", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=_FS_T)
        ax.set_ylabel(f"{('L', 'R')[c]} {nm} (deg)", color="black", fontsize=_FS_TITLE, labelpad=1.0)
        ax.yaxis.set_label_coords(YLX if c == 0 else YLX_R, 0.5)
fig.text(0.995, 0.012, "decoded: amplitude-matched", ha="right", va="bottom", fontsize=BODYFS, color="black")
fig.savefig(OUT_PNG, dpi=600)
print("wrote", OUT_PNG)
