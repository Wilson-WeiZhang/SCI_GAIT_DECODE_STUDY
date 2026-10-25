"""Goniometer QC verdict of the patient trials whose only QC failure is a reference channel without signal.

In one patient session the reference walker's left-knee goniometer channel carries no
signal. The goniometer QC verdict is per trial, so every imagined-walking trial of that
session is 'problem' through that one channel. The decoder reconstructs the six joints
independently and does not score a joint whose reference is missing, so the other five
joints remain usable.

This script sets to 'ok' only the trials whose entire QC reason is the flat left-knee reference
channel. A trial that also fails on length or on another channel stays 'problem'.

Output: goni_qc_trial_override.csv (label, trial, status), passed to the decoder through
the environment variable GAIT_GONI_QC_TRIAL_FLAGS.
"""

import csv
import os
import re
import collections
from pathlib import Path

import h5py
import numpy as np

GONI_DIR = Path(os.environ["GAIT_DATA_ROOT"]) / "patient_prep" / "goni"
OUT = Path(os.environ["GAIT_PREP_OUTPUT"]) / "patient" / "result" / "qc" / "goni_qc_trial_override.csv"

DEAD_CHAN_RE = re.compile(r"^LKneW X std:0\.00<[\d.]+$")
TARGETS = ["pat01_sess05"]


def mat_str(f, ref) -> str:
    a = np.asarray(f[ref]).ravel()
    return "".join(chr(int(c)) for c in a if int(c) > 0)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in TARGETS:
        p = GONI_DIR / f"{label}_goni.mat"
        with h5py.File(p, "r") as f:
            qc = f["qc"]
            n = qc["status"].shape[0]
            typ = [mat_str(f, qc["type"][i, 0]) for i in range(n)]
            sta = [mat_str(f, qc["status"][i, 0]) for i in range(n)]
            rea = [mat_str(f, qc["reason"][i, 0]) for i in range(n)]

        stats = collections.Counter()
        for i in range(n):
            if typ[i] != "imagine" or sta[i] != "problem":
                continue
            parts = [x.strip() for x in rea[i].split(";") if x.strip()]
            only_dead = bool(parts) and all(DEAD_CHAN_RE.match(x) for x in parts)
            if only_dead:
                rows.append({"label": label, "trial": i + 1, "status": "ok"})
                stats["promoted"] += 1
            else:
                stats["kept_problem"] += 1
                for x in parts:
                    if not DEAD_CHAN_RE.match(x):
                        stats[f"kept_because::{x.split(':')[0]}"] += 1

        print(f"{label}: {n} qc entries")
        for k, v in sorted(stats.items()):
            print(f"   {k:<34} {v}")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "trial", "status"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} trials promoted to ok)")


if __name__ == "__main__":
    main()
