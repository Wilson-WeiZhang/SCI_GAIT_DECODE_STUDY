"""Decoder — run drivers (02_decoding/run)

Check of one feature export (one row of exports.csv) before any fold is decoded from it: against the export of
the same table on the zero-phase preprocessing, the trial set, targets, trial windows, fold map, joints and
goniometer QC must be equal, the band table must be the row's band csv, the triple root must be the causal root,
and the EEG features must differ; a rest-window export must also attach to its paired imagined-walking export.
On a pass it writes <export>/EXPORT_READY (its md5 is checked again by merge_runs.py), else EXPORT_FAIL.
Usage (from 02_decoding/): python run/check_export.py <export_id>

Part of the decoder's run chain (README.md, Decoding, Steps 07-09): it builds job lists,
inputs or merged outputs; it fits nothing.
"""

import hashlib
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_paths as P
from pyport import matio
from pyport.config import Config
from pyport.decoder import Session, _find_export_dict

XCSV = os.environ.get("MAIN_EXPORTS_CSV", os.path.join(HERE, "exports.csv"))
REST_CFG = os.path.join(os.path.dirname(HERE), "pyport", "configs", "tenfold_pca30_lpf6_rest_input.json")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def raw_dict(p):
    return _find_export_dict(matio.loadmat_any(p))


HOST_ROOT = [None]


def host_root_of(path):
    """mirror root of an export made on a remote export host (its REMOTE_HOST.txt), else None"""
    rh = os.path.join(os.path.dirname(path), "REMOTE_HOST.txt")
    if not os.path.exists(rh):
        return None
    return dict(l.split(" ", 1) for l in open(rh).read().strip().split("\n"))["host_root"].strip()


def unhost(x, h=None):
    h = h if h is not None else HOST_ROOT[0]
    if x and h and (x == h or x.startswith(h.rstrip("/") + "/")):
        return P.GAIT + x[len(h.rstrip("/")):]
    return x


def root_of(d):
    return unhost(matio.to_str(d["gait_triple_root_env"])) if "gait_triple_root_env" in d else None


def gate_h1(row, f15, f14, want_root, self_test):
    a, b = Session.from_export(f15), Session.from_export(f14)
    c = {"n_ev": a.n_ev == b.n_ev, "label": a.label == b.label == row.session,
         "Y_full": a.n_ev == b.n_ev and all(eq(x, y) for x, y in zip(a.Y_full, b.Y_full)),
         "et0": eq(a.et0, b.et0), "Lk": eq(a.Lk, b.Lk), "fa_fold": eq(a.fa_fold, b.fa_fold), "fa_seed": a.fa_seed == b.fa_seed,
         "joints": list(a.joint_labels) == list(b.joint_labels)}
    if isinstance(a.meta.get("goni_qc_flags_env"), str):
        a.meta["goni_qc_flags_env"] = unhost(a.meta["goni_qc_flags_env"])
    if isinstance(b.meta.get("goni_qc_flags_env"), str):
        b.meta["goni_qc_flags_env"] = unhost(b.meta["goni_qc_flags_env"], host_root_of(f14))
    for k in ("ev_trial_idx", "Tk", "joint_idx", "valid_trial", "bp_band_feature_mode", "goni_qc_flags_env"):
        c[k] = eq(a.meta.get(k), b.meta.get(k)) if not isinstance(a.meta.get(k), str) else a.meta.get(k) == b.meta.get(k)
    da = raw_dict(f15)
    with h5py.File(f15, "r") as h:
        eb = np.asarray(h["bands"][()], dtype=float)
    bt = pd.read_csv(os.path.join(P.DMI, row.band_csv))
    bt = bt[bt.session == row.session].sort_values("harmonic_k")
    c["bands"] = len(bt) == 3 and np.array_equal(eb[0], [1, 2, 3]) and \
        np.allclose(eb[1], bt.band_lo, rtol=0, atol=1e-12) and np.allclose(eb[2], bt.band_hi, rtol=0, atol=1e-12)
    c["root"] = root_of(da) == want_root
    same_u = a.n_ev == b.n_ev and all(eq(x, y) for x, y in zip(a.U, b.U))
    if row.kind in ("wf", "wfp"):
        c["U_differs"] = True
    else:
        c["U_differs"] = same_u if self_test else not same_u
    return c, "n_ev %d tested %d root %s" % (a.n_ev, int((np.asarray(a.meta.get("Tk", [])) > 0).sum()), root_of(da))


def gate_rest(row, f15, f14, want_root, self_test):
    a, b = matio.loadmat_any(f15), matio.loadmat_any(f14)
    c = {"label": matio.to_str(a["session_label"]) == matio.to_str(b["session_label"]) == row.session}
    for k in ("n_ev", "rest_ok", "Tk", "et0", "lag_samples", "input_control_seed"):
        c[k] = eq(matio.vec(a[k]), matio.vec(b[k]))
    c["root"] = root_of(a) == want_root
    ua, ub = matio.cell_list(a["U_rest"]), matio.cell_list(b["U_rest"])
    same_u = len(ua) == len(ub) and all(eq(np.asarray(x, float), np.asarray(y, float)) for x, y in zip(ua, ub))
    c["U_rest_differs"] = same_u if self_test else not same_u
    E = pd.read_csv(XCSV, keep_default_na=False).set_index("export_id")
    pr = E.loc[row.pair_export]
    if self_test:
        pf = os.path.join(P.DMI, pr.ref_export)
    else:
        mk = os.path.join(P.DMI, P.EXPORTS, row.pair_export, "EXPORT_READY")
        t0 = time.time()
        while not os.path.exists(mk) and time.time() - t0 < 1800:
            if os.path.exists(os.path.join(P.DMI, P.EXPORTS, row.pair_export, "EXPORT_FAIL")):
                break
            time.sleep(30)
        pf = os.path.join(P.DMI, P.EXPORTS, row.pair_export, pr.out_file)
        if not os.path.exists(mk):
            c["pair_ready"] = False
            return c, "pair %s not READY" % row.pair_export
    s = Session.from_export(pf)
    try:
        s.attach_rest(f15, Config.from_json(REST_CFG).stage1_lags(s.fs))
        c["attach_rest"] = True
        n_ok = s.rest_meta["n_rest_ok"]
    except ValueError as e:
        c["attach_rest"] = False
        n_ok = "ERR %s" % e
    return c, "rest_ok %s root %s pair %s" % (n_ok, root_of(a), row.pair_export)


def main():
    eid = sys.argv[1]
    self_test = "--self-test" in sys.argv
    E = pd.read_csv(XCSV, keep_default_na=False).set_index("export_id")
    row = E.loc[eid]
    f14 = os.path.join(P.DMI, row.ref_export)
    f15 = f14 if self_test else os.path.join(P.DMI, P.EXPORTS, eid, row.out_file)
    rh = os.path.join(P.DMI, P.EXPORTS, eid, "REMOTE_HOST.txt")
    if not self_test and os.path.exists(rh):
        HOST_ROOT[0] = dict(l.split(" ", 1) for l in open(rh).read().strip().split("\n"))["host_root"].strip()
    if self_test:
        want = root_of(matio.loadmat_any(f14) if row.kind in ("resth", "restp") else raw_dict(f14))
    else:
        want = {"pat": P.P_TRIPLE, "wfp": P.P_TRIPLE, "restp": P.P_REST_TRIPLE}.get(row.kind, P.H_TRIPLE)
    if not os.path.exists(f15):
        c, info = {"exists": False}, f15
    else:
        c, info = (gate_rest if row.kind in ("resth", "restp") else gate_h1)(row, f15, f14, want, self_test)
    bad = [k for k, v in c.items() if not v]
    line = "gate E %s %s: %s | %s%s%s" % (eid, "PASS" if not bad else "FAIL", ",".join(bad) or "all %d checks" % len(c), info,
                                          " [self-test vs itself]" if self_test else "",
                                          " [remote, host root %s]" % HOST_ROOT[0] if HOST_ROOT[0] else "")
    print(line, flush=True)
    if self_test:
        sys.exit(1 if bad else 0)
    d = os.path.join(P.DMI, P.EXPORTS, eid)
    mk = os.path.join(d, "EXPORT_READY" if not bad else "EXPORT_FAIL")
    with open(mk + ".tmp", "w") as fh:
        if not bad:
            fh.write("md5 %s\nfile %s\n" % (md5(f15), os.path.relpath(f15, P.DMI)))
        fh.write(line + "\n")
    os.replace(mk + ".tmp", mk)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
