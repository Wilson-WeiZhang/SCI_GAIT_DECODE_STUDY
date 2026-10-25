"""Figure 7 — physics simulation and walk-intent gate

Gate session set on the causal preprocessing: the gate session set of build_gate_session_set.sh with each
session's EEG replaced by the causal patient triple (01_preprocessing/causal/patient/). The event files are
linked unchanged. Checked per session before anything is written: 250 Hz on both sides, equal sample count
and channel labels, the events of the causal triple equal to the gate event file, every imagined-walking and
rest event usable, and the gate loader (gate_features.load_session) returning the same events on both roots.
Writes <root>/<session>/{<session>_eeg.mat, <session>_event.mat} links and <root>/READY.

Reads:
  decoder output in data/decoder_output/ (not distributed)
  the aligned per-session recordings (not distributed)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os
import sys

import h5py
import numpy as np
import scipy.io as sio

PW = os.environ.get("GAIT_PROJECT_ROOT", "") + "/02_decoding/simulation"
sys.path.insert(0, PW)
import gate_features as pgf

G11 = os.environ.get("GAIT_PROJECT_ROOT", "") + "/patient_prep_gate"
CT = os.environ.get("GAIT_PROJECT_ROOT", "") + "/patient_prep_causal"
DST = os.environ.get("GAIT_PROJECT_ROOT", "") + "/data/decoder_output/main/triples/pat_gate"
SESS11 = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08", "pat01_sess09",
          "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08", "pat02_sess09"]


def h5str(h, ref):
    return "".join(chr(c) for c in np.array(h[ref]).ravel())


def eeg_meta(p):
    with h5py.File(p, "r") as h:
        fs = float(np.array(h["eeg/s"]).ravel()[0])
        shape = h["eeg/x"].shape
        lab = h["eeg/c/labels"]
        ch = [h5str(h, lab[i, 0]) for i in range(lab.shape[0])]
    return fs, shape, ch


def events_all(p):
    try:
        m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
        ev = np.atleast_1d(m["event"] if "event" in m else m["events"])
        return [(str(e.code), float(e.etstart), float(e.etstop), str(e.qc)) for e in ev]
    except NotImplementedError:
        with h5py.File(p, "r") as h:
            g = h["event"] if "event" in h else h["events"]
            col = lambda f: [h[r] for r in np.asarray(g[f][()]).ravel()]
            sv = lambda d: "".join(chr(c) for c in np.asarray(d[()]).ravel() if c)
            return [(sv(c), float(np.ravel(a[()])[0]), float(np.ravel(b[()])[0]), sv(q))
                    for c, a, b, q in zip(col("code"), col("etstart"), col("etstop"), col("qc"))]


def main():
    if os.path.exists(DST):
        sys.exit(f"REFUSE: {DST} exists")
    plan, bad, lines = [], [], []
    for s in SESS11:
        ce, ge = os.path.join(CT, s, s + "_eeg.mat"), os.path.realpath(os.path.join(G11, s, s + "_eeg.mat"))
        cv, gv = os.path.join(CT, s, s + "_event.mat"), os.path.join(G11, s, s + "_event.mat")
        fc, shc, chc = eeg_meta(ce)
        fg, shg, chg = eeg_meta(ge)
        if not (fc == fg == 250.0 and shc == shg and chc == chg):
            bad.append((s, f"EEG meta differ: fs {fc}/{fg} shape {shc}/{shg} labels equal {chc == chg}"))
            continue
        ec, eg = events_all(cv), events_all(gv)
        if [e[:3] for e in ec] != [e[:3] for e in eg]:
            bad.append((s, "event (code, etstart, etstop) differ between causal triple and gate_sessions"))
            continue
        nq = [e for e in eg if e[0] in ("gaitimagery", "rest") and e[3] != "good"]
        if nq:
            bad.append((s, f"{len(nq)} gate events of gate_sessions not qc good"))
            continue
        with h5py.File(ce, "r") as a, h5py.File(ge, "r") as b:
            xc, xg = np.array(a["eeg/x"]), np.array(b["eeg/x"])
        dmax = float(np.max(np.abs(xc - xg)))
        plan.append((s, ce, os.path.realpath(gv)))
        n_iw = sum(e[0] == "gaitimagery" for e in eg)
        n_rest = sum(e[0] == "rest" for e in eg)
        lines.append(f"{s}: fs 250, x {shc}, {len(chc)} labels equal; events {len(ec)} (iw {n_iw}, rest {n_rest}) timing equal; "
                     f"EEG max |causal - as-run| {dmax:.4g}; eeg -> {ce}; event -> {os.path.realpath(gv)}")
        print(lines[-1], flush=True)
    if bad:
        sys.exit(f"REFUSE (nothing written): {bad}")
    os.makedirs(DST)
    for s, ce, gv in plan:
        os.makedirs(os.path.join(DST, s))
        os.symlink(ce, os.path.join(DST, s, s + "_eeg.mat"))
        os.symlink(gv, os.path.join(DST, s, s + "_event.mat"))
    for s, _ce, _gv in plan:
        _x1, f1, c1, e1 = pgf.load_session(DST, s)
        _x2, f2, c2, e2 = pgf.load_session(G11, s)
        assert f1 == f2 and c1 == c2 and e1 == e2, f"{s}: loader view differs"
    with open(os.path.join(DST, "READY"), "w") as fh:
        fh.write("\n".join(lines) + "\nloader check: gate_features.load_session identical events / fs / labels on "
                 "the causal and the zero-phase gate roots for all 11 sessions\n")
    print(f"GATE ROOT PASS: {len(plan)} sessions -> {DST}")


if __name__ == "__main__":
    main()
