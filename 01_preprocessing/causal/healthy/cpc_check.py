"""Preprocessing — strictly causal variant (01_preprocessing/causal)

Last stage of the causal preprocessing of a healthy session: its causal triple against the triple of the
frozen (zero-phase) chain. Every goniometer array and every event field must be equal (except the two event
fields computed from the EEG itself), the EEG must have the same shape and differ (per-channel correlation
reported), the stage info files must exist and the calibration step2 must hold an ICA. Writes
<out>/check/<label>.csv; the decode chain (02_decoding/run/) uses a session only when its check passed.
Usage: python cpc_check.py --sid <SID> --plan plan.csv --out <out_root>; python cpc_check.py --summary --out <out_root>

Reads:
  the aligned per-session recordings (not distributed)

Part of the causal preprocessing: it plans or runs the frozen preprocessing stages with the
causal switches (README.md, Preprocessing); it fits nothing.
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

AS_RUN = os.path.join(os.environ.get("CPC_ROOT", os.environ.get("GAIT_PROJECT_ROOT", "")), "data", "healthy_prep")
EEG_DERIVED = ("eeg_pre1s_mean", "eeg_pre1s_std")


def h5arrays(path):
    out = {}
    with h5py.File(path, "r") as h:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fiu":
                out[name] = np.array(obj)
        h.visititems(walk)
    return out


def load(d, L, kind):
    p = f"{d}/{L}/{L}_{kind}.mat"
    try:
        return {"_mat": sio.loadmat(p, squeeze_me=True, struct_as_record=False)[kind]}
    except NotImplementedError:
        return h5arrays(p)


def same_event(a, b, skip=()):
    ea, eb = np.atleast_1d(a["_mat"]), np.atleast_1d(b["_mat"])
    if len(ea) != len(eb):
        return False, f"n events {len(ea)} vs {len(eb)}"
    for i, (x, y) in enumerate(zip(ea, eb)):
        if x._fieldnames != y._fieldnames:
            return False, f"event {i + 1} fields differ"
        for f in x._fieldnames:
            if f in skip:
                continue
            u, v = getattr(x, f), getattr(y, f)
            ok = u == v if isinstance(u, str) else np.array_equal(np.asarray(u), np.asarray(v), equal_nan=True)
            if not ok:
                return False, f"event {i + 1} field {f}"
    return True, f"{len(ea)} events, all fields equal" + (" (EEG-derived skipped)" if skip else "")


def same_arrays(a, b):
    keys = sorted(set(a) | set(b))
    bad = [k for k in keys if k not in a or k not in b or not np.array_equal(a[k], b[k], equal_nan=True)]
    return not bad, (f"{len(keys)} arrays equal" if not bad else f"differ: {bad[:5]}")


def check(sid, L, out, ref, est_task=None):
    C = {k: load(f"{out}/causal_triple", L, k) for k in ("eeg", "event", "goni")}
    A = {k: load(AS_RUN, L, k) for k in ("eeg", "event", "goni")}
    row = dict(session=L, sid=sid)
    row["goni_ok"], row["goni_msg"] = same_arrays(C["goni"], A["goni"])
    row["event_ok"], row["event_msg"] = same_event(C["event"], A["event"], EEG_DERIVED)
    xa, xc = A["eeg"]["eeg/x"], C["eeg"]["eeg/x"]
    row["eeg_shape_ok"] = xa.shape == xc.shape
    row["eeg_differs"] = bool(row["eeg_shape_ok"] and not np.array_equal(xa, xc))
    if row["eeg_shape_ok"]:
        r = [np.corrcoef(xa[:, c], xc[:, c])[0, 1] if xa[:, c].std() > 0 and xc[:, c].std() > 0 else np.nan
             for c in range(xa.shape[1])]
        row["eeg_r_median"], row["eeg_r_min"] = float(np.nanmedian(r)), float(np.nanmin(r))
    info = {k: f"{out}/info/{sid}_{k}_info.txt" for k in ("step1", "step2_cal", "step3_causal")}
    row["info_ok"] = all(os.path.exists(p) for p in info.values())
    if row["info_ok"]:
        s2 = open(info["step2_cal"]).read()
        m = re.search(r"calibration step2: (\d+) segments, ([\d.]+) s, bad ch (\d+) \{([^}]*)\}, rank (\d+), ASR segments (\d+), ICs (\d+)", s2)
        row.update(cal_segments=int(m.group(1)), cal_task_s=float(m.group(2)), cal_bad_ch=m.group(4),
                   cal_rank=int(m.group(5)), cal_ics=int(m.group(7)))
        s3 = open(info["step3_causal"]).read()
        m = re.search(r"rejected ICs: this (.+?) of (\d+) \((.*?)\); as-run (.+?) of (\d+) \((.*?)\)\s*$", s3, re.M)
        row.update(rejected_ics=m.group(1), rejected_labels=m.group(3), asrun_rejected_ics=m.group(4),
                   asrun_rejected_labels=m.group(6))
        row["info_ok"] = row["cal_ics"] > 0
        if est_task is not None:
            row["plan_est_cal_task_s"] = float(est_task)
            row["cal_task_diff_s"] = row["cal_task_s"] - float(est_task)
            row["info_ok"] = row["info_ok"] and abs(row["cal_task_diff_s"]) < 1.0
    if ref:
        R = {k: load(ref, L, k) for k in ("eeg", "event", "goni")}
        e1, e2, e3 = same_arrays(C["eeg"], R["eeg"]), same_arrays(C["goni"], R["goni"]), same_event(C["event"], R["event"])
        row["ref_ok"] = e1[0] and e2[0] and e3[0]
        row["ref_msg"] = f"eeg {e1[1]}; goni {e2[1]}; event {e3[1]}"
        xr = R["eeg"]["eeg/x"]
        if xr.shape == xc.shape:
            rr = [np.corrcoef(xr[:, c], xc[:, c])[0, 1] if xr[:, c].std() > 0 and xc[:, c].std() > 0 else np.nan
                  for c in range(xr.shape[1])]
            row.update(ref_eeg_maxabs_diff=float(np.max(np.abs(xr - xc))), ref_eeg_maxabs=float(np.max(np.abs(xr))),
                       ref_eeg_r_min=float(np.nanmin(rr)), ref_eeg_r_median=float(np.nanmedian(rr)))
    row["ok"] = bool(row["goni_ok"] and row["event_ok"] and row["eeg_differs"] and row["info_ok"] and row.get("ref_ok", True))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid")
    ap.add_argument("--plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/check", exist_ok=True)
    if a.summary:
        d = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{a.out}/check/*.csv"))], ignore_index=True)
        d.to_csv(f"{a.out}/check_summary.csv", index=False)
        print(d.to_string(index=False))
        print(f"\n{int(d.ok.sum())} / {len(d)} sessions ok; failed: {d.loc[~d.ok, 'session'].tolist()}")
        return
    P = pd.read_csv(a.plan)
    r = P.loc[P.sid == a.sid]
    assert len(r) == 1, a.sid
    row = check(a.sid, r.session.iloc[0], a.out, a.ref, r.est_cal_task_s.iloc[0])
    pd.DataFrame([row]).to_csv(f"{a.out}/check/{row['session']}.csv", index=False)
    for k, v in row.items():
        print(f"{k:22s} {v}")
    sys.exit(0 if row["ok"] else 1)


if __name__ == "__main__":
    main()
