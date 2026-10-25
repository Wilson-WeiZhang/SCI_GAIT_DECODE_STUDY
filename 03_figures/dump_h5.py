"""Shared figure settings

Reader for the per-session decoder outputs `<session>_dump.h5` + `<session>_trial_r.csv` (02_decoding/pyport/).

Imported by the figure scripts; it renders no panel of its own.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

JOINTS = ["LHipW X", "LKneW X", "LAnkW X", "RHipW X", "RKneW X", "RAnkW X"]


def session_files(dump_dir) -> dict:
    """{session: <dump_dir>/<session>_dump.h5} for every dump in dump_dir, sorted by session."""
    d = Path(dump_dir)
    out = {p.name[:-len("_dump.h5")]: p for p in sorted(d.glob("*_dump.h5"))}
    if not out:
        raise SystemExit(f"dump_h5: no *_dump.h5 in {d}")
    return out


def _ec():
    try:
        import analysis_config
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import analysis_config
    return analysis_config


def load_session(h5_path, fields=("Y_true", "C_pred")) -> dict:
    """dict(session, fs, joints, trials=[dict(fold, trial_idx, test_ordinal, time_sec, <field>: J x T)]),
    the scored trials are returned."""
    with h5py.File(h5_path, "r") as f:
        sess = f.attrs["session"]
        sess = sess.decode() if isinstance(sess, bytes) else str(sess)
        fs = float(f.attrs["fs"])
        joints = str(f.attrs["joints"]).split(",")
        if joints != JOINTS:
            raise SystemExit(f"dump_h5: {h5_path}: joints {joints} differ from {JOINTS}")
        tr = {k: f["trials/" + k][()] for k in ("fold", "trial_idx", "test_ordinal", "row_offset", "n_rows")}
        rows = {k: f["rows/" + k][()] for k in tuple(fields) + ("time_sec",)}
    order = np.argsort(tr["trial_idx"], kind="stable")
    trials = []
    for i in order:
        a, n = int(tr["row_offset"][i]), int(tr["n_rows"][i])
        t = dict(fold=int(tr["fold"][i]), trial_idx=int(tr["trial_idx"][i]),
                 test_ordinal=int(tr["test_ordinal"][i]), time_sec=rows["time_sec"][a:a + n].copy())
        for k in fields:
            t[k] = np.ascontiguousarray(rows[k][a:a + n].T)
        trials.append(t)
    ec = _ec()
    if ec.SCORE_MASK is not None:
        cond, col = ec.mask_condition(h5_path), ec.mask_ordinal(h5_path)
        trials = [t for t in trials if ec.keep_trial(sess, cond, t[col])]
    return dict(session=sess, fs=fs, joints=joints, trials=trials)


def load_trial(h5_path, trial_idx: int, fields=("Y_true", "C_pred", "A_pred")) -> dict:
    """One trial of a session dump (same dict as a load_session() trial, plus session / fs / joints)."""
    ec = _ec()
    if ec.SCORE_MASK is not None:
        sess = Path(h5_path).name[:-len("_dump.h5")]
        if ec.mask_ordinal(h5_path) != "trial_idx":
            raise SystemExit(f"dump_h5.load_trial: {h5_path} numbers its trials train-then-test; not supported under the mask")
        if not ec.keep_trial(sess, ec.mask_condition(h5_path), trial_idx):
            raise SystemExit(f"dump_h5: {sess} trial_idx {trial_idx} lies inside the calibration segment and is not "
                             f"scored (03_figures/scoring_mask.csv)")
    S = load_session(h5_path, fields)
    for t in S["trials"]:
        if t["trial_idx"] == int(trial_idx):
            return dict(t, session=S["session"], fs=S["fs"], joints=S["joints"])
    raise SystemExit(f"dump_h5: trial_idx {trial_idx} not in {h5_path}")


def dump_path(dump_dir, session: str) -> Path:
    p = Path(dump_dir) / f"{session}_dump.h5"
    if not p.exists():
        raise SystemExit(f"dump_h5: {p} missing")
    return p


if __name__ == "__main__":
    import sys
    S = load_session(sys.argv[1])
    print(S["session"], S["fs"], len(S["trials"]), "trials;", "first", S["trials"][0]["trial_idx"],
          S["trials"][0]["Y_true"].shape, os.path.basename(sys.argv[1]))
