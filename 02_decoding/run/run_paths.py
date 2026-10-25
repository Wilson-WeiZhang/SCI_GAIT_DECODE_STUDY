"""Decoder — run drivers (02_decoding/run)

Roots of the decode chain (Python twin of run_paths.sh): the causal triple roots, the result root, the basis,
export and job folders, and the fixed inputs the chain reuses (band tables, session manifests, the trial layout
of the run on the zero-phase preprocessing). Every value can be overridden from the environment.
Usage: python run_paths.py [--check-sh]  (prints the values; --check-sh compares them with run_paths.sh)

Reads:
  the aligned per-session recordings (not distributed)

Part of the decoder's run chain (README.md, Decoding, Steps 07-09): it builds job lists,
inputs or merged outputs; it fits nothing.
"""

import os
import subprocess
import sys

E = os.environ.get
TAG = "main"
GAIT = E("MAIN_GAIT", E("GAIT_PROJECT_ROOT", ""))
DMI = GAIT + "/02_decoding"
CODE = DMI + "/run"
H_PREP = E("MAIN_H_PREP", GAIT + "/causal_prep_output/healthy")
P_PREP = E("MAIN_P_PREP", GAIT + "/causal_prep_output/patient")
H_TRIPLE = E("MAIN_H_TRIPLE", H_PREP + "/causal_triple")
P_TRIPLE = E("MAIN_P_TRIPLE", P_PREP + "/causal_triple")
R = E("MAIN_R", "result/" + TAG)
P_REST_TRIPLE = E("MAIN_P_REST_TRIPLE", DMI + "/" + R + "/triples/pat_restgood")
BASIS_IW = R + "/bases/iw_lso"
BASIS_EXEC = R + "/bases/exec_lso"
BASIS_HG = R + "/bases/hglobal"
EXPORTS = R + "/exports"
Q = R + "/_queue"
JOBS = R + "/_jobs"
R14 = "result/zero_phase_prep"
R13 = "result/session_band_table"
PY = E("MAIN_PY", E("PYTHON", "python"))
MATLAB = E("MAIN_MATLAB", E("MATLAB", "matlab"))
SNAPSHOT_TAR = DMI + "/pyport"
DECODER_MD5 = "9a17ee13c766c03f41e9fe56517c4b40"

H_IW_BANDS = "result/diag_goniometer_psd_peaks/goniometer_peak_bands.csv"
H_EXEC_BANDS = "result/diag_main_exec_goni_psd_peaks/main_exec_goni_peak_bands.csv"
H_MANIFEST = "result/session_manifest_main.csv"
H_MANIFEST_RW = "result/session_manifest_real_walking.csv"
F0F = "result/main/fold_tables/imagined_walking"
FT14 = "result/main/fold_tables"
PX = "result/pyport_export"
P_GROUP = {"pat02_sess08": "sess08", "pat02_sess09": "sess09"}
P_GROUP_MANIFEST = {"patient set": "result/patient_session_manifest.csv",
                    "sess08": "result/patient_session_manifest_sess08.csv",
                    "sess09": "result/patient_session_manifest_sess09.csv"}
P_GROUP_BANDS = {"patient set": "result/diag_patient_goni_psd_peaks/patient_goni_peak_bands.csv",
                 "sess08": "result/diag_patient_goni_psd_peaks_sess08/patient_goni_peak_bands.csv",
                 "sess09": "result/diag_patient_goni_psd_peaks_sess09/patient_goni_peak_bands.csv"}
P_REST_GROUP = {s: "rest7" for s in ("pat01_sess04", "pat01_sess06", "pat01_sess07", "pat01_sess09", "pat02_sess03",
                                      "pat02_sess05", "pat02_sess06")}
P_REST_GROUP.update({s: "add4" for s in ("pat01_sess05", "pat01_sess08", "pat02_sess08", "pat02_sess09")})
P_REST_MANIFEST = {"rest7": "result/patient_session_manifest_rest7.csv",
                   "add4": "result/patient_session_manifest_rest_add4.csv"}
P_REST_BANDS = {"rest7": "result/diag_patient_goni_psd_peaks/patient_goni_peak_bands.csv",
                "add4": "result/diag_patient_goni_psd_peaks_rest_add4/patient_goni_peak_bands.csv"}
P_REST_ASRUN_ROOT = {"rest7": GAIT + "/patient_prep_restgood_v4_7",
                     "add4": GAIT + "/patient_prep_restgood_add4"}
GONI_QC_OVERRIDE = GAIT + "/01_preprocessing/patient/result/qc/goni_qc_trial_override.csv"
PRE_OPERATIVE = ("pat01_sess01", "pat01_sess02", "pat02_sess01", "pat02_sess02")
POST_OPERATIVE = ("pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08", "pat01_sess09",
            "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08", "pat02_sess09")


def p_group(s):
    return P_GROUP.get(s, "patient set")


def _items():
    return {k: v for k, v in globals().items() if k.isupper() and isinstance(v, str)}


if __name__ == "__main__":
    for k, v in sorted(_items().items()):
        print("%-16s %s" % (k, v))
    if "--check-sh" in sys.argv:
        sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_paths.sh")
        keys = ("TAG", "GAIT", "DMI", "CODE", "H_PREP", "P_PREP", "H_TRIPLE", "P_TRIPLE", "R", "P_REST_TRIPLE", "BASIS_IW",
                "BASIS_EXEC", "BASIS_HG", "EXPORTS", "Q", "JOBS", "R14", "PY", "MATLAB", "SNAPSHOT_TAR", "DECODER_MD5")
        out = subprocess.run(["bash", "-c", "source '%s'; %s" % (sh, "; ".join('echo "%s=$MAIN_%s"' % (k, k) for k in keys))],
                             capture_output=True, text=True, check=True).stdout.split("\n")
        got = dict(l.split("=", 1) for l in out if "=" in l)
        bad = [(k, got.get(k), globals()[k]) for k in keys if got.get(k) != globals()[k]]
        print("sh == py:", "PASS" if not bad else "FAIL %s" % bad)
        sys.exit(1 if bad else 0)
