"""Figure 7 — physics simulation and walk-intent gate

The walk-intent gate stream scored after the calibration recording: the stream of gate_scores_causal.py
(result/main/; trained on every event of the session, within-session 10-fold by event) restricted to the events
whose onset is at or after the session's calibration cut (cut_export_s of
01_preprocessing/causal/patient/plan.csv). The gate is not re-fitted: score, fold and event rows are copied
unchanged for the kept events, the threshold tables byte for byte. Checks that no event straddles the cut.
Writes result/main/scored/ (read by the figure scripts) with gate_scored_events.csv, one row per event.

Reads:
  simulation and gate output in 02_decoding/simulation/ (not distributed)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import hashlib
import os
import shutil

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC = os.path.join(ROOT, "02_decoding/simulation/result/main")
DST = os.path.join(ROOT, "02_decoding/simulation/result/main/scored")
PLAN = os.path.join(ROOT, "01_preprocessing/causal/patient/plan.csv")


def md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main():
    if os.path.exists(os.path.join(DST, "gate_within_session_scores_T3.csv")):
        raise SystemExit(f"{DST} already holds a masked stream; refusing to overwrite")
    os.makedirs(DST, exist_ok=True)
    plan = pd.read_csv(PLAN).set_index("session")
    ev = pd.read_csv(os.path.join(SRC, "gate_within_session_events.csv"))
    ev["cut_export_s"] = ev.session.map(plan.cut_export_s)
    assert ev.cut_export_s.notna().all(), "a gate session has no calibration cut"
    straddle = ev[(ev.t_start_s < ev.cut_export_s) & (ev.t_stop_s > ev.cut_export_s)]
    assert straddle.empty, f"events straddle the cut: {straddle[['session', 'event']].values.tolist()}"
    ev["keep"] = (ev.t_start_s >= ev.cut_export_s).astype(int)
    keep = set(zip(ev.session[ev.keep == 1], ev.event[ev.keep == 1]))
    lines = []
    for s, g in ev.groupby("session"):
        iw_b = int(((g.label == 1) & (g.keep == 0)).sum())
        lines.append(f"{s}: IW {int((g.label == 1).sum())} -> {int(((g.label == 1) & (g.keep == 1)).sum())}, rest "
                     f"{int((g.label == 0).sum())} -> {int(((g.label == 0) & (g.keep == 1)).sum())}; IW before the cut {iw_b}, "
                     f"n_cal {int(plan.n_cal[s])}{'' if iw_b == int(plan.n_cal[s]) else '  <- differs'}")
        print(lines[-1])

    def write(name, df):
        p = os.path.join(DST, name)
        df.to_csv(p + ".tmp", index=False)
        os.replace(p + ".tmp", p)
        return p

    ins = []
    for name in ("gate_within_session_scores_T3.csv", "gate_within_session_scores_T4.csv",
                 "gate_within_session_events.csv", "gate_within_session_folds.csv"):
        src = os.path.join(SRC, name)
        d = pd.read_csv(src, dtype=str, keep_default_na=False)
        m = [(s, e) in keep for s, e in zip(d.session, d.event)]
        write(name, d.loc[m])
        ins.append((name, md5(src), len(d), int(sum(m))))
        print(f"{name}: {len(d)} -> {int(sum(m))} rows")
    for name in ("gate_within_session_thresholds_T3.csv", "gate_within_session_thresholds_T4.csv"):
        shutil.copyfile(os.path.join(SRC, name), os.path.join(DST, name + ".tmp"))
        os.replace(os.path.join(DST, name + ".tmp"), os.path.join(DST, name))
        ins.append((name, md5(os.path.join(SRC, name)), "copied", "copied"))
    write("gate_scored_events.csv", ev[["session", "event", "label", "t_start_s", "t_stop_s", "cut_export_s", "keep"]])
    md = [("script", "02_decoding/simulation/gate_scored_events.py"), ("build_id", "main"),
          ("rule", "event scored iff t_start_s >= cut_export_s (plan.csv); training, folds, reference rests and "
                   "thresholds unchanged (the stream of gate_scores_causal.py, all events)"),
          ("plan", f"{os.path.relpath(PLAN, ROOT)} (md5 {md5(PLAN)})"),
          ("events", f"{int(ev.keep.sum())} of {len(ev)} scored (IW {int(((ev.label == 1) & (ev.keep == 1)).sum())} of "
                     f"{int((ev.label == 1).sum())}, rest {int(((ev.label == 0) & (ev.keep == 1)).sum())} of {int((ev.label == 0).sum())})"),
          ("per_session", "; ".join(lines))]
    md += [(f"input {n}", f"{os.path.relpath(SRC, ROOT)}/{n} md5 {h}; rows {a} -> {b}") for n, h, a, b in ins]
    write("gate_scored_events_metadata.csv", pd.DataFrame(md, columns=["key", "value"]))
    print("wrote", os.path.relpath(DST, ROOT))


if __name__ == "__main__":
    main()
