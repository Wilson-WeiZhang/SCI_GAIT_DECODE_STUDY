"""Preprocessing — strictly causal variant (01_preprocessing/causal)

Calibration plan of the healthy sessions (writes plan.csv, read by cpc_session.m and cpc_step1.m).

Per session: the raw parts and their order in the merged step1 recording, the decoder trials in export
order, the number of calibration trials n_cal (the trials of the first run; else the first third) and the
calibration cut in step1 samples: the end of the last task or rest segment that starts before trial n_cal + 1,
plus a 2 s margin, and at most the start of trial n_cal + 1 minus 1 s (its baseline window stays outside).
Checks per session (straddle, bound, n_cal trials before the cut, at least 5 min of task segments before the
cut, raw files present, part composition); a session with several parts, no S11 marker or a failed check
needs a reviewed approval (--overrides plan_overrides.csv: sid, cut_step1, n_cal, approve, note).
Usage: python cpc_plan.py --out plan.csv [--overrides plan_overrides.csv] [--margin-s 2]

Reads:
  the aligned per-session recordings (not distributed)

Part of the causal preprocessing: it plans or runs the frozen preprocessing stages with the
causal switches (README.md, Preprocessing); it fits nothing.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio

G = os.environ.get("CPC_ROOT", os.environ.get("GAIT_PROJECT_ROOT", ""))
EEG1 = f"{G}/data/prep_data/healthy/eeg"
TRIP = f"{G}/data/healthy_prep"
LOGS = f"{G}/01_preprocessing/healthy/logs"
REG = f"{G}/01_preprocessing/healthy/shared/build_session_registry.m"
RAW = f"{G}/data/raw_data"
CALIB = f"{G}/data/decoder_output/calibration/calib.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
FS = 250
PAIRS = {"S  1": "S  2", "S  4": "S  5", "S  7": "S  8"}
MIN_TASK_S = 300.0


def registry():
    reg = {}
    for ln in open(REG):
        m = re.match(r"\s*'(SUB_\d+_sess\d+[a-z]?)',\s*'([^']+\.vhdr)'", ln)
        if m:
            assert m.group(1) not in reg, f"registry lists {m.group(1)} twice"
            reg[m.group(1)] = m.group(2)
    return reg


def prep_log(sid):
    p = f"{LOGS}/prep_{sid}.log"
    parts, off = [], {}
    if os.path.exists(p):
        for ln in open(p, errors="replace"):
            m = re.search(r"sub-parts: \d+ \((.*)\)", ln)
            if m:
                parts, off = [x.strip() for x in m.group(1).split(",")], {}
            m = re.search(r"Timebase: part (\d+) step1.*offset = (-?\d+) samples", ln)
            if m:
                off[int(m.group(1))] = int(m.group(2))
    return parts, off


def set_info(path):
    m = sio.loadmat(path, variable_names=["pnts", "srate", "event"], squeeze_me=True, struct_as_record=False)
    ev = np.atleast_1d(m["event"])
    return int(m["pnts"]), float(m["srate"]), [str(e.type) for e in ev], \
        np.array([float(e.latency) for e in ev])


def segments(types, lats, s11):
    seg = []
    for st, en in PAIRS.items():
        ends = np.sort(lats[[t == en for t in types]])
        for s in lats[[t == st for t in types]]:
            if s11 is not None and s < s11:
                continue
            nxt = ends[ends > s]
            seg.append((s, nxt[0] if len(nxt) else np.inf, st))
    return sorted(seg)


def trimmed_task_s(seg, types, lats, cut, drop_s12):
    """Task time prep_step2 keeps from the calibration copy [1, cut] (seg is unused, kept for the call signature):
    its trim_s11_s12 (round(first 'S 11') .. round(LAST 'S 12') when both exist, first S11 .. end
    without S12, everything without S11), then extract_segments_no_bl  on the kept events: per
    start marker the first end marker after it; skipped when another start of the same kind lies before that end, when
    the duration is < 3 s or > 300 s, or when a 'boundary' event (raw data break, part merge) lies inside."""
    t = np.array(types, dtype=object)
    keep = lats <= cut
    if drop_s12:
        keep &= t != "S 12"
    t, l = t[keep], lats[keep]
    s11, s12 = l[t == "S 11"], l[t == "S 12"]
    lo, hi = 1, cut
    if len(s11):
        lo = max(1, round(s11[0]))
        if len(s12):
            hi = min(cut, round(s12[-1]))
    w = (l >= lo) & (l <= hi)
    t, l = t[w], l[w]
    bnd = l[t == "boundary"]
    total = 0.0
    for st, en in PAIRS.items():
        starts, ends = l[t == st], l[t == en]
        for sl in starts:
            nxt = ends[ends > sl]
            if not len(nxt):
                continue
            nxt = nxt[0]
            if np.any((starts > sl) & (starts < nxt)):
                continue
            dur = (nxt - sl) / FS
            if dur < 3 or dur > 300 or np.any((bnd > sl) & (bnd < nxt)):
                continue
            total += dur
    return total


def raw_files_ok(v):
    """The raw part exists: the .vhdr and the DataFile / MarkerFile it names (BrainVision header, same folder)."""
    if not v or not os.path.isfile(f"{RAW}/{v}"):
        return False
    d = os.path.dirname(f"{RAW}/{v}")
    names = re.findall(r"^(?:DataFile|MarkerFile)=(.+?)\s*$", open(f"{RAW}/{v}", errors="replace").read(), re.M)
    return len(names) == 2 and all(os.path.isfile(os.path.join(d, n)) for n in names)


def check_cut(cut, seg, s11, ev_step1, n_cal, B, types, lats):
    """ev_step1 = [(start, stop)] of the decoder trials on the step1 axis; B = step1 start of trial n_cal + 1."""
    straddle = all(e < cut for s, e, _ in seg if s < cut)
    n_before = sum(1 for s, e in ev_step1 if e < cut)
    bound = int(np.floor(B)) - FS - 1
    task = trimmed_task_s(seg, types, lats, cut, True)
    return dict(straddle_ok=straddle, bound_step1=int(bound), bound_ok=cut <= bound,
                count_ok=(n_before == n_cal and B > cut), est_cal_task_s=round(task, 1),
                est_cal_task_unfixed_s=round(trimmed_task_s(seg, types, lats, cut, False), 1),
                s12_in_cal=int(sum(1 for t, l in zip(types, lats) if t == "S 12" and l <= cut)),
                dur_ok=task >= MIN_TASK_S, cut_after_s11=(s11 is None or cut > s11))


def plan_session(row, reg, margin, overrides, tl):
    L, n_ev, n_cal = row.session, int(row.n_ev), int(row.n_cal)
    m = re.match(r"subj(\d+)_sess(\d+)", L)
    sid = f"SUB_{m.group(1)}_sess{m.group(2)}"
    notes = []
    ov = overrides.get(sid)
    n_cal_calib = n_cal
    if ov and ov[1] is not None:
        n_cal = ov[1]
        notes.append(f"n_cal override {n_cal_calib} -> {n_cal} ({ov[2]})")
    exp_parts, log_off = prep_log(sid)
    pnts, srate, types, lats = set_info(f"{EEG1}/{sid}_step1.set")
    assert srate == FS
    pfiles = sorted(glob.glob(f"{EEG1}/{sid}[a-z]_step1.set"))
    pnames = [os.path.basename(p)[:-len("_step1.set")] for p in pfiles]
    ppnts = [set_info(p)[0] for p in pfiles]
    if pfiles:
        cs = np.cumsum(ppnts)
        k = [i + 1 for i in range(len(cs)) if cs[i] == pnts]
        step1_parts = pnames[:k[0]] if k else []
        parts_ok = bool(k)
        if not k:
            notes.append("no prefix of the part files sums to the merged pnts")
    else:
        step1_parts, parts_ok = [sid], True
    vhdr = [reg.get(p, "") for p in step1_parts]
    raw_ok = bool(vhdr) and all(raw_files_ok(v) for v in vhdr)
    if not raw_ok:
        notes.append("raw vhdr missing")
    s11i = [i for i, t in enumerate(types) if t == "S 11"]
    s11 = lats[s11i[0]] if s11i else None
    off_trim = int(round(s11)) - 1 if s11 is not None else 0
    if s11 is None:
        notes.append("no S11 in step1 (offset 0)")

    ev = sio.loadmat(f"{TRIP}/{L}/{L}_event.mat", squeeze_me=True, struct_as_record=False)["event"]
    ev = [e for e in np.atleast_1d(ev) if str(e.code) == "gaitimagery" and str(e.qc) == "good"]
    assert len(ev) == n_ev, f"{L}: {len(ev)} good imagined-walking trials vs calib n_ev {n_ev}"
    et = np.array([[float(e.etstart), float(e.etstop)] for e in ev])
    assert np.all(np.diff(et[:, 0]) > 0), f"{L}: onsets not monotone"
    bnd = np.cumsum([0] + ppnts[:len(step1_parts)]) if (pfiles and parts_ok) else np.array([0, pnts])
    part_of = np.clip(np.searchsorted(bnd, et[:, 0] + off_trim, side="right"), 1, len(bnd) - 1)
    off = np.array([log_off.get(int(p), off_trim) for p in part_of])
    if log_off and max(abs(v - off_trim) for v in log_off.values()) > 2:
        notes.append(f"logged offsets {log_off} differ from the S11 trim {off_trim}")
    if not log_off:
        notes.append("no Timebase line in the prep log (offset = S11 trim)")
    s1 = np.sort(lats[[t == "S  1" for t in types]])
    mk = np.array([s1[np.argmin(np.abs(s1 - (a + o)))] for a, o in zip(et[:, 0], off)])
    resid = mk - (et[:, 0] + off)
    ev_step1 = [(a + o, b + o) for (a, b), o in zip(et, off)]
    seg = segments(types, lats, s11)
    A, B = mk[n_cal - 1], min(mk[n_cal], ev_step1[n_cal][0])
    cyc = [(s, e, t) for s, e, t in seg if A <= s < B - 1]
    last_end = max(e for _, e, _ in cyc)
    cut_rule = int(np.floor(last_end)) + 1 + int(round(margin * FS))
    bound = int(np.floor(B)) - FS - 1
    if cut_rule > bound and np.floor(last_end) + 1 <= bound:
        notes.append(f"margin clipped at the bound ({(bound - np.floor(last_end) - 1) / FS:.1f} s left)")
        cut_rule = bound
    cut = ov[0] if ov and ov[0] is not None else cut_rule
    if cut != cut_rule:
        notes.append(f"cut override ({ov[2]})")
    chk = check_cut(cut, seg, s11, ev_step1, n_cal, B, types, lats)
    multipart = len(step1_parts) > 1 or len(exp_parts) > 1
    ok = all([parts_ok, raw_ok, chk["straddle_ok"], chk["bound_ok"], chk["count_ok"], chk["dur_ok"], chk["cut_after_s11"],
              np.max(np.abs(resid)) <= 2])
    review = multipart or s11 is None or not ok or any("differ" in n for n in notes)
    out = dict(session=L, sid=sid, rule=row.rule, n_ev=n_ev, n_cal_calib=n_cal_calib, n_cal=n_cal,
               step1_parts=";".join(step1_parts) if pfiles else "", export_parts=";".join(exp_parts),
               part_files=";".join(pnames), part_pnts=";".join(str(p) for p in ppnts), step1_pnts=pnts,
               raw_vhdr=";".join(vhdr), s11_step1=None if s11 is None else round(float(s11), 3), offset_trim=off_trim,
               offset_log=";".join(f"{k}:{v}" for k, v in sorted(log_off.items())), map_resid_max=float(np.max(np.abs(resid))),
               cal_trial_export_start=int(et[n_cal - 1, 0]), cal_trial_export_stop=int(et[n_cal - 1, 1]),
               cal_trial_end_step1=int(ev_step1[n_cal - 1][1]), cycle_segments=len(cyc),
               last_seg_end_step1=round(float(last_end), 3), next_trial_start_step1=int(B), margin_s=margin,
               cut_rule_step1=cut_rule, cut_override_step1=ov[0] if ov else None, cut_step1=int(cut),
               cut_s=round((cut - 1) / FS, 2), cut_export_s=round((cut - off[n_cal - 1] - 1) / FS, 2), **chk,
               parts_ok=parts_ok, raw_ok=raw_ok, checks_ok=bool(ok), multipart=multipart,
               approved=int(not review), notes="; ".join(notes))
    if review:
        tl.append(f"\n=== {L} ({sid}) step1 parts {step1_parts} pnts {ppnts or [pnts]} (merged {pnts}); export parts "
                  f"{exp_parts}; offsets log {log_off} trim {off_trim}; S11 {s11}")
        for i, b in enumerate(bnd[1:-1]):
            tl.append(f"    part boundary {i + 1}|{i + 2} at step1 {b} ({(b - 1) / FS:.1f} s)")
        for k, ((a, b), r, p) in enumerate(zip(ev_step1, resid, part_of)):
            mark = " <- calibration ends (n_cal)" if k == n_cal - 1 else (" <- first test trial" if k == n_cal else "")
            tl.append(f"    trial {k + 1:3d} part {p} step1 {(a - 1) / FS:8.1f} - {(b - 1) / FS:8.1f} s  S1 resid {r:+.0f}{mark}")
        tl.append(f"    cycle segments after trial n_cal: " + ", ".join(f"{t.strip()} {(s - 1) / FS:.1f}-{(e - 1) / FS:.1f}"
                                                                      for s, e, t in cyc))
        tl.append(f"    proposed cut step1 {cut} ({(cut - 1) / FS:.1f} s), bound {bound} ({(bound - 1) / FS:.1f} s), "
                  f"calibration task {chk['est_cal_task_s']} s; checks {chk}; notes {notes}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-s", type=float, default=2.0)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "plan.csv"))
    ap.add_argument("--only", default="")
    ap.add_argument("--approve", default="", help="comma-separated SIDs to approve after review")
    a = ap.parse_args()
    ov = {}
    if a.overrides:
        for r in pd.read_csv(a.overrides).to_dict("records"):
            num = lambda k: int(r[k]) if k in r and pd.notna(r[k]) else None
            ov[r["sid"]] = (num("cut_step1"), num("n_cal"), str(r.get("note", "")))
            if num("approve") == 1:
                a.approve = ",".join(filter(None, [a.approve, r["sid"]]))
    c = pd.read_csv(CALIB)
    c = c[c.cohort == "healthy"]
    assert len(c) == 80, len(c)
    reg, tl, rows = registry(), [], []
    only = set(a.only.split(",")) if a.only else None
    for r in c.itertuples():
        if only and not ({r.session, "SUB_%s_sess%s" % tuple(re.match(r"subj(\d+)_sess(\d+)", r.session).groups())} & only):
            continue
        rows.append(plan_session(r, reg, a.margin_s, ov, tl))
    d = pd.DataFrame(rows)
    for sid in filter(None, a.approve.split(",")):
        assert sid in set(d.sid), sid
        assert bool(d.loc[d.sid == sid, "checks_ok"].iloc[0]), f"{sid}: checks failed, cannot approve"
        d.loc[d.sid == sid, "approved"] = 1
        d.loc[d.sid == sid, "notes"] += "; approved by --approve"
    tmp = a.out + ".tmp"
    d.to_csv(tmp, index=False)
    os.replace(tmp, a.out)
    with open(os.path.splitext(a.out)[0] + "_timeline.txt", "w") as fh:
        fh.write("\n".join(tl) + "\n")
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 40)
    print(d[["session", "rule", "n_ev", "n_cal", "step1_parts", "offset_trim", "offset_log", "map_resid_max", "cut_s",
             "cut_export_s", "est_cal_task_s", "checks_ok", "multipart", "approved", "notes"]].to_string(index=False))
    print("\nby rule:", d.groupby("rule").size().to_dict(), "| approved", int(d.approved.sum()), "/", len(d),
          "| checks failed:", d.loc[~d.checks_ok, "session"].tolist())
    print("review sessions (timeline in %s):" % (os.path.splitext(a.out)[0] + "_timeline.txt"),
          d.loc[d.approved == 0, "session"].tolist())


if __name__ == "__main__":
    main()
