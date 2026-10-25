"""Decoder — run drivers (02_decoding/run)

Rest-window patient triple root for the resting-input control, built from the causal patient triples:
EEG and goniometer linked, and an event file whose rest events are marked usable (patient events carry no EEG-derived field). Checks per session before linking: the events
of the causal triple equal those of the source triple, and the rest-good file differs from them only in the
rest flags. Writes <root>/READY. Usage: python patient_rest_root.py [--only s1,s2]

Part of the decoder's run chain (README.md, Decoding, Steps 07-09): it builds job lists,
inputs or merged outputs; it fits nothing.
"""

import os
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_paths as P


def h5str(ds):
    return "".join(chr(c) for c in np.asarray(ds[()]).ravel() if c)


def load_events(p):
    try:
        e = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
        ev = np.atleast_1d(e["event"] if "event" in e else e["events"])
        return [(str(x.code), str(x.qc), int(x.gtstart), int(x.gtstop)) for x in ev]
    except NotImplementedError:
        with h5py.File(p, "r") as h:
            g = h["event"] if "event" in h else h["events"]
            col = lambda f: [h[r] for r in np.asarray(g[f][()]).ravel()]
            return [(h5str(c), h5str(q), int(np.ravel(a[()])[0]), int(np.ravel(b[()])[0]))
                    for c, q, a, b in zip(col("code"), col("qc"), col("gtstart"), col("gtstop"))]


def main():
    dst = P.P_REST_TRIPLE
    if os.path.exists(dst):
        sys.exit("REFUSE: %s exists" % dst)
    plan, bad = [], []
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else P.POST_OPERATIVE
    for s in only:
        g = P.P_REST_GROUP[s]
        chk = os.path.join(P.P_PREP, "check", s + ".csv")
        c = pd.read_csv(chk).iloc[0] if os.path.exists(chk) else None
        if c is None or not (bool(c.ok) and bool(c.event_ok)):
            bad.append((s, "prep check missing / not ok"))
            continue
        ct = os.path.join(P.P_TRIPLE, s)
        rg = os.path.join(P.P_REST_ASRUN_ROOT[g], s, s + "_event.mat")
        src = os.path.dirname(os.path.realpath(os.path.join(P.P_REST_ASRUN_ROOT[g], s, s + "_eeg.mat")))
        ec, es, er = (load_events(p) for p in (os.path.join(ct, s + "_event.mat"), os.path.join(src, s + "_event.mat"), rg))
        if ec != es:
            bad.append((s, "causal events != source events"))
            continue
        flips = [i for i, (a, b) in enumerate(zip(es, er)) if a != b]
        ok = len(es) == len(er) and all(es[i][0] == "rest" and es[i][1] == "skip" and er[i][1] == "good"
                                        and es[i][2:] == er[i][2:] for i in flips)
        if not ok:
            bad.append((s, "rest-good events differ from the source beyond rest skip->good"))
            continue
        links = {s + "_event.mat": rg}
        for suf in ("_eeg.mat", "_goni.mat", "_goni_preped.mat"):
            if os.path.exists(os.path.join(ct, s + suf)):
                links[s + suf] = os.path.join(ct, s + suf)
        plan.append((s, g, links, len(flips), sum(e[0] == "rest" for e in es)))
    if bad:
        sys.exit("REFUSE (nothing written): %s" % bad)
    os.makedirs(dst)
    lines = []
    for s, g, links, nflip, nrest in plan:
        os.makedirs(os.path.join(dst, s))
        for name, target in sorted(links.items()):
            os.symlink(target, os.path.join(dst, s, name))
        lines.append("%s group %s rest events %d, skip->good %d; %s" % (
            s, g, nrest, nflip, "; ".join("%s -> %s" % kv for kv in sorted(links.items()))))
        print(lines[-1])
    with open(os.path.join(dst, "READY.tmp"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(os.path.join(dst, "READY.tmp"), os.path.join(dst, "READY"))
    print("built %s (%d sessions)" % (dst, len(plan)))


if __name__ == "__main__":
    main()
