"""Preprocessing — strictly causal variant (01_preprocessing/causal)

Calibration plan of the patient sessions (writes plan.csv, read by cpp_session.m and the cpp_* stages).

Per session: the preprocessing lane of the frozen patient chain (marker protocol, connector fix, rest
relabel), the decoder trials, the calibration trials n_cal (the first natural segment where its task
segments reach 180 s, else the smallest n_cal that reaches 180 s) and the calibration cut in step1 samples
(as cpc_plan.py: end of the last segment before trial n_cal + 1, plus 2 s, at most the next trial's start
minus 1 s), with the same checks.
Usage: python cpp_plan.py --out plan.csv

Reads:
  decoder output in data/decoder_output/ (not distributed)
  the aligned per-session recordings (not distributed)

Part of the causal preprocessing: it plans or runs the frozen preprocessing stages with the
causal switches (README.md, Preprocessing); it fits nothing.
"""

import argparse
import os
import re

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

G = os.environ.get("CPC_ROOT", os.environ.get("GAIT_PROJECT_ROOT", ""))
P = f"{G}/data/prep_data/patient"
CALIB = f"{G}/data/decoder_output/calibration/calib.csv"
EXPORT = f"{G}/data/decoder_output/pyport_export/cdm_o1_c0p18_causal_ownhglobal"
QC_OVERRIDE = f"{G}/01_preprocessing/patient/result/qc/goni_qc_trial_override.csv"
FS = 250
HERE = os.path.dirname(os.path.abspath(__file__))
GAP_REST_S = 30.0

LANES = {
    "two_marker":  dict(step1=f"{P}/eeg", step23=f"{P}/eeg_two_marker", goni=f"{P}/goni_two_marker", export="export_session", threads=8,
                    root=f"{G}/patient_prep_two_marker", runner=""
                    ),
    "connector_fix":  dict(step1=f"{P}/eeg_connector_fix", step23=f"{P}/eeg_connector_fix", goni=f"{P}/goni_connector_fix", export="export_session",
                    threads=32,
                    root=f"{G}/patient_prep_connector_fix", runner=""
                    ),
    "rest_relabel": dict(step1=f"{P}/eeg", step23=f"{P}/eeg", goni=f"{P}/goni", export="export_session", threads=8,
                    root=f"{G}/patient_prep", runner=""),
    "sess08":  dict(step1=f"{P}/eeg", step23=f"{P}/eeg", goni=f"{P}/goni", export="export_session", threads=24,
                    root=f"{G}/patient_prep", runner=""),
    "sess09":  dict(step1=f"{P}/eeg", step23=f"{P}/eeg", goni=f"{P}/goni", export="export_session_relabel", threads=24,
                    root=f"{G}/patient_prep", runner=""
                    ),
}
SESSIONS = {
    "pat01_sess01": ("two_marker", 0, "", "patient set"), "pat01_sess02": ("two_marker", 0, "", "patient set"),
    "pat01_sess04": ("connector_fix", 1, "", "patient set"), "pat02_sess01": ("connector_fix", 1, "", "patient set"),
    "pat02_sess02": ("connector_fix", 1, "", "patient set"),
    "pat01_sess05": ("rest_relabel", 0, "", "patient set"), "pat01_sess06": ("rest_relabel", 0, "", "patient set"),
    "pat01_sess07": ("rest_relabel", 0, "", "patient set"), "pat01_sess08": ("rest_relabel", 0, "", "patient set"),
    "pat01_sess09": ("rest_relabel", 0, "", "patient set"), "pat02_sess03": ("rest_relabel", 0, "", "patient set"),
    "pat02_sess05": ("rest_relabel", 0, "", "patient set"), "pat02_sess06": ("rest_relabel", 0, "", "patient set"),
    "pat02_sess08": ("sess08", 0, "", "sess08"), "pat02_sess09": ("sess09", 0, "[sensor id];LAnkS;RAnkW", "sess09"),
}
RULE_N_CAL = {"pat02_sess05": (15, "first natural segment")}
FALLBACK_N_CAL = {"pat01_sess01": 18}


def set_events(path):
    m = sio.loadmat(path, variable_names=["pnts", "srate", "event", "etc"], squeeze_me=True, struct_as_record=False)
    ev = np.atleast_1d(m["event"])
    etc = m["etc"]
    return (int(m["pnts"]), float(m["srate"]), [str(e.type) for e in ev], np.array([float(e.latency) for e in ev]),
            str(getattr(etc, "marker_protocol", "standard")))


def split_rest(types, lats):
    """prep_patient prep_step2 split_rest_events: S4/S5 within 30 s after the preceding S2 -> 'S  4r' / 'S  5r'."""
    mi_end = np.sort(lats[[t == "S  2" for t in types]])
    out = list(types)
    for i, (t, l) in enumerate(zip(types, lats)):
        if t in ("S  4", "S  5"):
            prev = mi_end[mi_end < l]
            if len(prev) and (l - prev[-1]) / FS <= GAP_REST_S:
                out[i] = t + "r"
    return out


def trial_defs(protocol, types):
    """prep_patient prep_step2 get_patient_trial_defs (+ choose_walk_markers)."""
    if protocol == "legacy":
        return [("S  1", "S  2", "MI")]
    if "S  6" in types and "S  7" in types:
        walk = ("S  6", "S  7")
    elif "S  8" in types:
        walk = ("S  7", "S  8")
    else:
        walk = ("S  7", 120.0)
    return [("S  1", "S  2", "MI"), ("S  4r", "S  5r", "Rest"), ("S  4", "S  5", "S2S"), (walk[0], walk[1], "Walk2min")]


def task_duration(types, lats, pnts, protocol, cut):
    """prep_patient prep_step2 total_dur on the calibration copy: pop_select [1 cut], split rest, trim S11 (- last S12),
    extract_segments_no_bl."""
    keep = lats <= cut
    t = [x for x, k in zip(types, keep) if k]
    lat = lats[keep]
    t = split_rest(t, lat)
    defs = trial_defs(protocol, t)
    s11 = [l for x, l in zip(t, lat) if x == "S 11"]
    s12 = [l for x, l in zip(t, lat) if x == "S 12"]
    lo, hi = (max(1, round(s11[0])) if s11 else 1), (min(cut, round(s12[-1])) if (s11 and s12) else cut)
    sel = (lat >= lo) & (lat <= hi)
    t = [x for x, k in zip(t, sel) if k]
    lat = lat[sel]
    bnd = lat[[x == "boundary" for x in t]]
    total, n, per = 0.0, 0, {}
    for smk, emk, name in defs:
        s = lat[[x == smk for x in t]]
        e = lat[[x == emk for x in t]] if isinstance(emk, str) else s + emk * FS
        for i, sl in enumerate(s):
            if isinstance(emk, str):
                nxt = e[e > sl]
                if not len(nxt):
                    continue
                nxt = nxt[0]
                if np.any((s > sl) & (s < nxt)):
                    continue
            else:
                nxt = e[i]
                if round(nxt) > hi:
                    continue
            dur = (nxt - sl) / FS
            if dur < 3 or dur > 300 or np.any((bnd > sl) & (bnd < nxt)):
                continue
            total += dur
            n += 1
            per[name] = per.get(name, 0) + 1
    return total, n, per


def segments(types, lats, s11):
    seg = []
    for st, en in (("S  1", "S  2"), ("S  4", "S  5")):
        ends = np.sort(lats[[t == en for t in types]])
        for s in lats[[t == st for t in types]]:
            if s11 is not None and s < s11:
                continue
            nxt = ends[ends > s]
            seg.append((s, nxt[0] if len(nxt) else np.inf, st))
    return sorted(seg)


def decoder_trials(L, triple):
    with h5py.File(f"{EXPORT}/{L}_export.mat", "r") as h:
        ev_idx = np.asarray(h["ev_trial_idx"][()]).ravel().astype(int)
        lab = "".join(chr(c) for c in np.asarray(h["session_label"][()]).ravel())
    assert lab == L, (lab, L)
    ev = np.atleast_1d(sio.loadmat(f"{triple}/{L}/{L}_event.mat", squeeze_me=True, struct_as_record=False)["event"])
    qc = [str(e.qc) for e in ev]
    for _, r in pd.read_csv(QC_OVERRIDE).iterrows():
        if r["label"] == L and 1 <= int(r["trial"]) <= len(qc):
            qc[int(r["trial"]) - 1] = {"ok": "good", "problem": "bad", "skip": "skip"}[r["status"]]
    pos = [j + 1 for j, e in enumerate(ev) if str(e.code) == "gaitimagery" and qc[j] == "good"]
    assert np.array_equal(pos, ev_idx), f"{L}: export ev_trial_idx != gaitimagery & good after the QC override"
    et = np.array([[float(ev[j - 1].etstart), float(ev[j - 1].etstop)] for j in ev_idx])
    assert np.all(np.diff(et[:, 0]) > 0), f"{L}: onsets not monotone"
    return et, len(ev)


def plan_session(row, margin, min_task_s, overrides, tl, n_cal_force=None):
    """n_cal_force = (n_cal, rule text) or None"""
    L = row.session
    lane, connector_fix, relabel, troot = SESSIONS[L]
    ln = LANES[lane]
    triple_root = {"patient set": f"{G}/patient_prep", "sess08": f"{G}/patient_prep_sess08",
                   "sess09": f"{G}/patient_prep_sess09"}[troot]
    real = os.path.realpath(f"{triple_root}/{L}/{L}_event.mat")
    lane_ok = real == f"{ln['root']}/{L}/{L}_event.mat"
    notes = [] if lane_ok else [f"as-run triple resolves to {real}, not the lane root {ln['root']}"]
    s1 = f"{ln['step1']}/{L}_step1.set"
    pnts, srate, types, lats, protocol = set_events(s1)
    assert srate == FS
    et, n_events = decoder_trials(L, triple_root)
    n_ev = len(et)
    assert n_ev == int(row.n_ev), f"{L}: {n_ev} decoder trials vs calib n_ev {row.n_ev}"
    n_cal_calib = int(row.n_cal)
    n_cal, why = n_cal_calib, "calib.csv"
    if L in RULE_N_CAL:
        n_cal, why = RULE_N_CAL[L]
    ov = overrides.get(L)
    if ov and ov[1] is not None:
        n_cal, why = ov[1], f"override ({ov[2]})"
    if n_cal_force is not None:
        n_cal, why = n_cal_force
    s11i = [i for i, t in enumerate(types) if t == "S 11"]
    s11 = lats[s11i[0]] if s11i else None
    off = int(round(s11)) - 1 if s11 is not None else 0
    if s11 is None:
        notes.append("no S11 in step1 (offset 0)")
    ev1 = et + off
    s1m = np.sort(lats[[t == "S  1" for t in types]])
    mk = np.array([s1m[np.argmin(np.abs(s1m - a))] for a in ev1[:, 0]])
    resid = mk - ev1[:, 0]
    seg = segments(types, lats, s11)
    A, B = mk[n_cal - 1], min(mk[n_cal], ev1[n_cal, 0])
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
    straddle = all(e < cut for s, e, _ in seg if s < cut)
    n_before = int(sum(1 for a, b in ev1 if b < cut))
    task_s, n_seg, per = task_duration(types, lats, pnts, protocol, cut)
    span_s = (ev1[n_cal - 1, 1] - ev1[0, 0]) / FS
    full_s, _, _ = task_duration(types, lats, pnts, protocol, pnts)
    m2 = sio.loadmat(f"{ln['step23']}/{L}_step2.set", variable_names=["etc"], squeeze_me=True, struct_as_record=False)
    asrun_s = float(m2["etc"].step2_meta.trial_dur_sec)
    est_ok = abs(full_s - asrun_s) < 1e-6
    n_need = None
    for k in range(n_cal, n_ev):
        Ak, Bk = mk[k - 1], min(mk[k], ev1[k, 0])
        ck = [(s, e) for s, e, _ in seg if Ak <= s < Bk - 1]
        cutk = min(int(np.floor(max(e for _, e in ck))) + 1 + int(round(margin * FS)), int(np.floor(Bk)) - FS - 1)
        if task_duration(types, lats, pnts, protocol, cutk)[0] >= min_task_s:
            n_need = k
            break
    bnds = lats[[t == "boundary" for t in types]]
    chk = dict(straddle_ok=straddle, bound_step1=bound, bound_ok=cut <= bound, count_ok=(n_before == n_cal and B > cut),
               cut_after_s11=(s11 is None or cut > s11), resid_ok=bool(np.max(np.abs(resid)) <= 2),
               dur_ok=task_s >= min_task_s)
    ok = lane_ok and est_ok and all(chk[k] for k in ("straddle_ok", "bound_ok", "count_ok", "cut_after_s11", "resid_ok", "dur_ok"))
    review = (not ok) or (n_cal != n_cal_calib) or (cut != cut_rule)
    if n_cal != n_cal_calib:
        notes.append(f"n_cal {n_cal_calib} -> {n_cal}: {why}")
    out = dict(session=L, sid=L, cohort=row.cohort, rule=row.rule, boundaries=row.boundaries, n_ev=n_ev,
               n_events_triple=n_events, n_cal_calib=n_cal_calib, n_cal=n_cal, n_cal_source=why,
               lane=lane, connector_fix=connector_fix, relabel_override=relabel, marker_protocol=protocol,
               rest_split=int(protocol == "standard"), dur_tol_s=0.050, export_fn=ln["export"],
               step1_dir=ln["step1"], step23_dir=ln["step23"], goni_dir=ln["goni"], lane_root=ln["root"],
               triple_root=triple_root, lane_runner=ln["runner"], asrun_step23_threads=ln["threads"], step1_pnts=pnts,
               merge_boundaries=";".join(f"{b:.1f}" for b in bnds),
               s11_step1=None if s11 is None else round(float(s11), 3), offset_trim=off,
               map_resid_max=float(np.max(np.abs(resid))), cal_trial_export_start=int(et[n_cal - 1, 0]),
               cal_trial_export_stop=int(et[n_cal - 1, 1]), cycle_segments=len(cyc),
               last_seg_end_step1=round(float(last_end), 3), next_trial_start_step1=int(B), margin_s=margin,
               cut_rule_step1=cut_rule, cut_override_step1=ov[0] if ov else None, cut_step1=int(cut),
               cut_s=round((cut - 1) / FS, 2), cut_export_s=round((cut - off - 1) / FS, 2),
               cal_span_s=round(span_s, 1), est_cal_task_s=round(task_s, 1), est_cal_segments=n_seg,
               est_cal_per_type=";".join(f"{k}:{v}" for k, v in per.items()), min_task_s=min_task_s,
               n_cal_for_min_task=n_need, est_full_task_s=round(full_s, 3), asrun_step2_task_s=round(asrun_s, 3),
               estimator_ok=est_ok, **chk,
               lane_ok=lane_ok, checks_ok=bool(ok), approved=int(not review), notes="; ".join(notes))
    tl.append(f"\n=== {L} lane {lane} (connector_fix {connector_fix}, relabel '{relabel}', protocol {protocol}) step1 {s1} pnts {pnts}; "
              f"S11 {s11}; offset {off}; merge boundaries {out['merge_boundaries'] or '-'}")
    for k, ((a, b), r) in enumerate(zip(ev1, resid)):
        mark = " <- calibration ends (n_cal)" if k == n_cal - 1 else (" <- first test trial" if k == n_cal else "")
        gap = "" if k == 0 else f" gap {(a - ev1[k - 1, 0]) / FS:6.1f} s"
        tl.append(f"    trial {k + 1:3d} step1 {(a - 1) / FS:8.1f} - {(b - 1) / FS:8.1f} s{gap}  S1 resid {r:+.0f}{mark}")
    tl.append("    cycle segments after trial n_cal: " + ", ".join(f"{t.strip()} {(s - 1) / FS:.1f}-{(e - 1) / FS:.1f}"
                                                                   for s, e, t in cyc))
    tl.append(f"    cut step1 {cut} ({(cut - 1) / FS:.1f} s), bound {bound} ({(bound - 1) / FS:.1f} s); calibration "
              f"task {task_s:.1f} s in {n_seg} segments {per} (span first start -> trial n_cal end {span_s:.1f} s); "
              f"checks {chk}; notes {notes}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-s", type=float, default=2.0)
    ap.add_argument("--min-task-min", type=float, default=3.0)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "plan.csv"))
    ap.add_argument("--only", default="")
    ap.add_argument("--approve", default="")
    ap.add_argument("--plan", action="store_true", help="print every timeline")
    a = ap.parse_args()
    ov = {}
    if a.overrides:
        for r in pd.read_csv(a.overrides).to_dict("records"):
            num = lambda k: int(r[k]) if k in r and pd.notna(r[k]) else None
            ov[r["sid"]] = (num("cut_step1"), num("n_cal"), str(r.get("note", "")))
            if num("approve") == 1:
                a.approve = ",".join(filter(None, [a.approve, r["sid"]]))
    c = pd.read_csv(CALIB)
    c = c[c.cohort.isin(["pre_operative", "post_operative"])]
    assert sorted(c.session) == sorted(SESSIONS), sorted(c.session)
    only = set(a.only.split(",")) if a.only else None
    tl, rows = [], []
    min_task_s = 60.0 * a.min_task_min
    for r in c.itertuples():
        if only and r.session not in only:
            continue
        sub = []
        o = plan_session(r, a.margin_s, min_task_s, ov, sub)
        if not o["dur_ok"] and r.session not in ov and (r.session in FALLBACK_N_CAL or o["n_cal_for_min_task"]):
            force = ((FALLBACK_N_CAL[r.session], "next run boundary (reviewed)") if r.session in FALLBACK_N_CAL
                     else (int(o["n_cal_for_min_task"]), f"smallest n_cal reaching {min_task_s:.0f} s"))
            sub.append(f"    -> duration check failed at n_cal {o['n_cal']} ({o['est_cal_task_s']} s); re-planned: n_cal "
                       f"{force[0]} ({force[1]})")
            tl += sub
            sub = []
            o = plan_session(r, a.margin_s, min_task_s, ov, sub, n_cal_force=force)
        o["n_cal_rule"] = o["n_cal_source"] if o["n_cal_source"] != "calib.csv" else "first natural segment"
        if o["n_cal_rule"].startswith("first natural segment") and o["n_cal_source"] != "calib.csv":
            o["n_cal_rule"] = "first natural segment (pat02_sess05 = 15, reviewed)"
        tl += sub
        rows.append(o)
    d = pd.DataFrame(rows)
    for sid in filter(None, a.approve.split(",")):
        assert sid in set(d.sid), sid
        assert bool(d.loc[d.sid == sid, "checks_ok"].iloc[0]), f"{sid}: checks failed, cannot approve"
        d.loc[d.sid == sid, "approved"] = 1
        d.loc[d.sid == sid, "notes"] += "; approved by --approve"
    tmp = a.out + ".tmp"
    d.to_csv(tmp, index=False)
    os.replace(tmp, a.out)
    tlp = os.path.splitext(a.out)[0] + "_timeline.txt"
    with open(tlp + ".tmp", "w") as fh:
        fh.write("\n".join(tl) + "\n")
    os.replace(tlp + ".tmp", tlp)
    if a.plan:
        print("\n".join(tl))
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 40)
    print(d[["session", "lane", "connector_fix", "rule", "n_ev", "n_cal", "offset_trim", "map_resid_max", "cut_s", "cut_export_s",
             "est_cal_task_s", "n_cal_rule", "n_cal_for_min_task", "estimator_ok", "checks_ok", "approved", "notes"]].to_string(index=False))
    print("\napproved", int(d.approved.sum()), "/", len(d), "| checks failed:", d.loc[~d.checks_ok, "session"].tolist())
    print("timelines:", tlp)


if __name__ == "__main__":
    main()
