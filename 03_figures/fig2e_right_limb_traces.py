# -*- coding: utf-8 -*-
"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

panel e — right-limb example traces (RHip / RKnee / RAnkle)

Renders one panel or table; it performs no
preprocessing and fits no model.
"""


from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import fig2c_plot_individual_ypred_yreal as f2c
import analysis_config as _ec
_ec.require("main", "main")

ROOT = CODE_DIR.parent
JOINTS = ["RHipW X", "RKneW X", "RAnkW X"]
PRED_FIELD = "C_pred"
TRIAL_ORDINAL = 5
NORMALIZE = "zscore"
ANNOTATE_R = False
FIG_WIDTH = 5.67
FIG_HEIGHT = 2.75
DPI = 300
BUILD_ID = _ec.TAG
PANEL = "Fig. 2e"
PAIRED_PANEL = "Fig. 2d (same trial, left limb)"

MAT = f2c.DEFAULT_MAT
OUT_PNG = _ec.scoped(ROOT / "figures_out" / "fig2" / "fig2e_right_limb_traces.png")
SD_DIR = ROOT / "04_source_data" / "fig2"
SOURCE_CSV = _ec.sd("fig2e_right_limb_traces_source_data.csv")
META_CSV = _ec.sd("fig2e_right_limb_traces_metadata.csv")


def zscored(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, float).ravel().copy()
    ok = np.isfinite(a)
    if ok.sum() > 1:
        a -= np.nanmean(a[ok])
        sd = np.nanstd(a[ok])
        if sd > 0:
            a /= sd
    return a


def main() -> None:
    mat, sel = MAT, TRIAL_ORDINAL
    if not _ec.IS_DEFAULT:
        mat = _ec.path("h_iw_dump_dir") / f"{f2c.MAIN_EXAMPLE[0]}_dump.h5"
        sel = f2c.MAIN_EXAMPLE[1]
    else:
        _ec.require_dump_format("matlab_fold_mat")
    traces = [f2c.extract_trace(mat, sel, joint, PRED_FIELD) for joint in JOINTS]
    for trace in traces:
        trace["normalize"] = NORMALIZE

    f2c.draw_traces(traces, OUT_PNG, DPI, FIG_WIDTH, FIG_HEIGHT, annotate_r=ANNOTATE_R)

    SD_DIR.mkdir(parents=True, exist_ok=True)
    rel_mat = mat.relative_to(ROOT).as_posix()

    with META_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "session", "trial_ordinal", "trial_idx", "joint",
                    "pred_field", "r", "build_id", "panel", "paired_panel"])
        for trace in traces:
            w.writerow([rel_mat, trace["session"], trace["trial_ordinal"], trace["trial_idx"],
                        trace["joint"], trace["pred_field"], float(trace["r"]),
                        BUILD_ID, PANEL, PAIRED_PANEL])

    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as f:
        sw = csv.writer(f)
        sw.writerow(["joint", "sample", "time_s", "reference", "decoded"])
        for trace in traces:
            t = np.asarray(trace["time"], float).ravel()
            obs = zscored(trace["y_true"])
            dec = zscored(trace["y_pred"])
            jn = str(trace["joint"]).replace("W X", "")
            for i in range(len(t)):
                sw.writerow([jn, i, f"{t[i]:.4f}", f"{obs[i]:.6f}", f"{dec[i]:.6f}"])

    print(f"source={rel_mat}")
    print(f"session={traces[0]['session']} trial_ordinal={traces[0]['trial_ordinal']} "
          f"trial_idx={traces[0]['trial_idx']}")
    for trace in traces:
        print(f"{trace['joint']} r={float(trace['r']):.6f}")
    print(f"wrote={OUT_PNG}")
    print(f"wrote={SOURCE_CSV}")
    print(f"wrote={META_CSV}")


if __name__ == "__main__":
    main()
