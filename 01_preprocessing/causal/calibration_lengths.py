"""Preprocessing — strictly causal variant (01_preprocessing/causal)

Calibration length of every session: trial onsets from the session's feature export; a run boundary is a gap
between consecutive onsets longer than GAP_S; n_cal = the boundary count closest to n/3 if within n/6 of it, else
round(n/3). Writes calib.csv (session, cohort, n_ev, boundaries, n_cal, rule), which healthy/cpc_plan.py and
patient/cpp_plan.py read. Run from 02_decoding/ (it imports pyport).

Reads:
  decoder output in data/decoder_output/ (not distributed)

Part of the causal preprocessing: it plans or runs the frozen preprocessing stages with the
causal switches (README.md, Preprocessing); it fits nothing.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from pyport.decoder import Session

R = os.path.join(os.environ.get("GAIT_PROJECT_ROOT", ""), "data", "decoder_output")
OUT = os.path.join(R, "calibration", "calib.csv")
GAP_S = 120.0
PRE_OPERATIVE = {"pat01_sess01", "pat01_sess02", "pat02_sess01", "pat02_sess02"}
rows = []
for cohort, pat in (("healthy", f"{R}/pyport_export/h_cdm_o1_c0p18_causal_ownlso/*_export.mat"),
                    ("patient", f"{R}/pyport_export/cdm_o1_c0p18_causal_ownhglobal/pat*_export.mat")):
    for p in sorted(glob.glob(pat)):
        S = Session.from_export(p)
        on = np.asarray(S.et0, float) / S.fs
        assert np.all(np.diff(on) > 0)
        n = S.n_ev
        gaps = np.diff(on)
        bnd = [int(i + 1) for i in np.where(gaps > GAP_S)[0]]
        third = n / 3.0
        cand = [b for b in bnd if abs(b - third) <= n / 6.0]
        if cand:
            n_cal, rule = min(cand, key=lambda b: abs(b - third)), "run boundary"
        else:
            n_cal, rule = int(round(third)), "n/3"
        rows.append(dict(session=S.label, cohort=cohort if cohort == "healthy" else ("pre_operative" if S.label in PRE_OPERATIVE else "post_operative"),
                         n_ev=n, boundaries=";".join(str(b) for b in bnd), n_cal=n_cal, n_test=n - n_cal, rule=rule))
d = pd.DataFrame(rows)
d.to_csv(OUT, index=False)
pd.set_option("display.width", 200, "display.max_rows", 200)
print(d.to_string(index=False))
print(d.groupby(["cohort", "rule"]).agg(sessions=("session", "size"), n_cal_min=("n_cal", "min"), n_cal_max=("n_cal", "max"),
                                        n_test_sum=("n_test", "sum")))
