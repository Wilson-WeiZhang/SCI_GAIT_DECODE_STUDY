"""Preprocessing — strictly causal variant (01_preprocessing/causal)

Last stage of the causal preprocessing of a patient session: as cpc_check.py, against the patient triple of the
frozen chain (patient events carry no EEG-derived field, so every event field must be equal).
Usage: python cpp_check.py --label <label> --plan plan.csv --out <out_root>; python cpp_check.py --summary --out <out_root>

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


def h5arrays(path):
    out = {}
    with h5py.File(path, "r") as h:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fiu":
                out[name] = np.array(obj)
        h.visititems(walk)
    return out


def h5str(path, key):
    with h5py.File(path, "r") as h:
        return "".join(chr(c) for c in np.asarray(h[key][()]).ravel())


def events(path):
    return np.atleast_1d(sio.loadmat(path, squeeze_me=True, struct_as_record=False)["event"])


def same_event(ea, eb):
    if len(ea) != len(eb):
        return False, f"n events {len(ea)} vs {len(eb)}"
    for i, (x, y) in enumerate(zip(ea, eb)):
        if x._fieldnames != y._fieldnames:
            return False, f"event {i + 1} fields differ"
        for f in x._fieldnames:
            u, v = getattr(x, f), getattr(y, f)
            ok = u == v if isinstance(u, str) else np.array_equal(np.asarray(u), np.asarray(v), equal_nan=True)
            if not ok:
                return False, f"event {i + 1} field {f}"
    n_good = sum(1 for e in ea if str(e.code) == "gaitimagery" and str(e.qc) == "good")
    return True, f"{len(ea)} events, all fields equal ({n_good} gaitimagery good before the decode-time QC override)"


def same_arrays(a, b):
    keys = sorted(set(a) | set(b))
    bad = [k for k in keys if k not in a or k not in b or not np.array_equal(a[k], b[k], equal_nan=True)]
    return not bad, (f"{len(keys)} arrays equal" if not bad else f"differ: {bad[:5]}")


def check(L, out, asrun_root):
    cd, ad = f"{out}/causal_triple/{L}", f"{asrun_root}/{L}"
    row = dict(session=L, asrun=os.path.realpath(f"{ad}/{L}_event.mat"))
    row["goni_ok"], row["goni_msg"] = same_arrays(h5arrays(f"{cd}/{L}_goni.mat"), h5arrays(f"{ad}/{L}_goni.mat"))
    row["event_ok"], row["event_msg"] = same_event(events(f"{cd}/{L}_event.mat"), events(f"{ad}/{L}_event.mat"))
    with h5py.File(f"{cd}/{L}_eeg.mat", "r") as hc, h5py.File(f"{ad}/{L}_eeg.mat", "r") as ha:
        xc, xa = np.array(hc["eeg/x"]), np.array(ha["eeg/x"])
    if xa.shape[0] < xa.shape[1]:
        xa, xc = xa.T, xc.T
    row["eeg_shape_ok"] = xa.shape == xc.shape
    row["eeg_differs"] = bool(row["eeg_shape_ok"] and not np.array_equal(xa, xc))
    row["eeg_source"] = h5str(f"{cd}/{L}_eeg.mat", "eeg/source_set")
    row["eeg_status"] = h5str(f"{cd}/{L}_eeg.mat", "eeg/pipeline_status")
    if row["eeg_shape_ok"]:
        r = [np.corrcoef(xa[:, c], xc[:, c])[0, 1] if xa[:, c].std() > 0 and xc[:, c].std() > 0 else np.nan
             for c in range(xa.shape[1])]
        row["eeg_r_median"], row["eeg_r_min"] = float(np.nanmedian(r)), float(np.nanmin(r))
    info = {k: f"{out}/info/{L}_{k}_info.txt" for k in ("step1", "step2_cal", "step3_causal")}
    row["info_ok"] = all(os.path.exists(p) for p in info.values())
    if row["info_ok"]:
        m = re.search(r"calibration step2: (\d+) segments, ([\d.]+) s, bad ch (\d+) \{([^}]*)\}, rank (\d+), ASR segments (\d+), "
                      r"ICs (\d+)", open(info["step2_cal"]).read())
        row.update(cal_segments=int(m.group(1)), cal_task_s=float(m.group(2)), cal_bad_ch=m.group(4),
                   cal_rank=int(m.group(5)), cal_ics=int(m.group(7)))
        m = re.search(r"rejected ICs: this (.+?) of (\d+) \((.*?)\); as-run (.+?) of (\d+) \((.*?)\)\s*$",
                      open(info["step3_causal"]).read(), re.M)
        row.update(rejected_ics=m.group(1), rejected_labels=m.group(3), asrun_rejected_ics=m.group(4),
                   asrun_rejected_labels=m.group(6))
        row["info_ok"] = row["cal_ics"] > 0
    row["ok"] = bool(row["goni_ok"] and row["event_ok"] and row["eeg_differs"] and row["info_ok"]
                     and row["eeg_source"] == "step3" and row["eeg_status"] == "ok")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--plan")
    ap.add_argument("--out", required=True)
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
    r = P.loc[P.sid == a.label]
    assert len(r) == 1, a.label
    row = check(a.label, a.out, r.triple_root.iloc[0])
    pd.DataFrame([row]).to_csv(f"{a.out}/check/{a.label}.csv", index=False)
    for k, v in row.items():
        print(f"{k:22s} {v}")
    sys.exit(0 if row["ok"] else 1)


if __name__ == "__main__":
    main()
