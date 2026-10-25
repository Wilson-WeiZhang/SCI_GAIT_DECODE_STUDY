"""Figure 7 — physics simulation and walk-intent gate

gate_working_point — the gate's clinical working point: TPR, FP.min-1, detection latency

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "result")
TIMED = os.path.join(RES, "gate_scores_timed.csv")
THRESH = os.path.join(RES, "gate_thresholds.csv")
FUSE2 = os.path.join(RES, "gate_scores_fuse2.csv")
OUT = os.path.join(RES, "gate_working_point.csv")

SESSIONS = ["pat01_sess04", "pat01_sess06", "pat01_sess07", "pat01_sess09",
            "pat02_sess03", "pat02_sess05", "pat02_sess06"]

WIN_S, HOP_S = 4.0, 4.0
N_ON = 1
CLOSE_M, CLOSE_N = 2, 3
DWELL_S = 8.0
IQR_K, OFF_Q, BASE_MIN_S = 0.25, 90, 30.0

TRACE_RULE = "session with the most evaluated decision windows"


def hysteresis(t_end, score, theta_on, theta_off):
    """gate.hysteresis, decision-indexed instead of sample-indexed. Rules verbatim."""
    cur, above, t_open = 0, 0, -1e9
    hist, events, state = [], [], []
    for te, sc in zip(t_end, score):
        if cur == 0:
            above = above + 1 if sc > theta_on else 0
            if above >= N_ON:
                cur, t_open, above, hist = 1, te, 0, []
                events.append((te, "open", sc))
        else:
            hist.append(bool(sc < theta_off))
            if len(hist) > CLOSE_N:
                hist.pop(0)
            if sum(hist) >= CLOSE_M and (te - t_open) >= DWELL_S:
                cur, hist = 0, []
                events.append((te, "close", sc))
        state.append(cur)
    return state, events


def load():
    win = list(csv.DictReader(open(TIMED, encoding="utf-8")))
    for r in win:
        for k in ("win", "label", "ev_orig_idx", "ev_a_samp", "ev_b_samp", "n_win_ev"):
            r[k] = int(r[k])
        for k in ("ev_dur_s", "t_start_ev_s", "t_end_ev_s", "t_end_sess_s", "score"):
            r[k] = float(r[k])
    thr = {(r["convention"], r["session"]): {k: (float(v) if k not in
                                                 ("convention", "session", "K_calib_events",
                                                  "n_calib_win") else v)
                                             for k, v in r.items()}
           for r in csv.DictReader(open(THRESH, encoding="utf-8"))}
    for v in thr.values():
        v["K_calib_events"] = int(v["K_calib_events"])
        v["n_calib_win"] = int(v["n_calib_win"])
    fuse = {(r["session"], r["event"], int(r["win"])): float(r["score_fuse"])
            for r in csv.DictReader(open(FUSE2, encoding="utf-8"))}
    return win, thr, fuse


def events_of(win, conv, sess, label, thr):
    """Evaluated events of one session as ordered lists of (event, [(t_end_ev, score, win)])."""
    rows = [r for r in win if r["convention"] == conv and r["session"] == sess
            and r["label"] == label]
    if conv == "dumpbase" and label == 0:
        K = thr[(conv, sess)]["K_calib_events"]
        rows = [r for r in rows if r["ev_orig_idx"] >= K]
    ev = {}
    for r in rows:
        ev.setdefault(r["event"], []).append(r)
    out = []
    for name, rs in ev.items():
        rs.sort(key=lambda r: r["win"])
        out.append((name, rs))
    out.sort(key=lambda kv: kv[1][0]["ev_a_samp"])
    return out


def run_convention(win, thr, fuse, conv, iqr_k=IQR_K):
    detail, summary = [], []
    for sess in SESSIONS:
        T = thr[(conv, sess)]
        th_on = T["calib_max"] + iqr_k * T["calib_iqr"]
        th_off = T["theta_off"]

        iw = events_of(win, conv, sess, 1, thr)
        lat, n_det = [], 0
        for name, rs in iw:
            t = [r["t_end_ev_s"] for r in rs]
            s = [r["score"] for r in rs]
            _st, evs = hysteresis(t, s, th_on, th_off)
            opens = [e for e in evs if e[1] == "open"]
            det = len(opens) > 0
            n_det += int(det)
            if det:
                lat.append(opens[0][0])
            fu = [fuse.get((sess, name, r["win"])) for r in rs]
            fu = [x for x in fu if x is not None]
            detail.append(dict(row_type="iw_trial", convention=conv, session=sess, event=name,
                               n_win=len(rs), ev_dur_s=rs[0]["ev_dur_s"],
                               theta_on=th_on, theta_off=th_off, max_score=max(s),
                               detected=int(det),
                               first_open_win=(int(round(opens[0][0] / WIN_S)) if det else ""),
                               latency_s=(opens[0][0] if det else ""),
                               max_score_fuse2=(max(fu) if fu else "")))

        rest = events_of(win, conv, sess, 0, thr)
        t, s, tag = [], [], []
        for name, rs in rest:
            for r in rs:
                t.append(len(t) * WIN_S + WIN_S)
                s.append(r["score"])
                tag.append((name, r["win"]))
        _st, evs = hysteresis(t, s, th_on, th_off)
        fp = [e for e in evs if e[1] == "open"]
        rest_s = len(t) * WIN_S
        for name, rs in rest:
            detail.append(dict(row_type="rest_event", convention=conv, session=sess, event=name,
                               n_win=len(rs), ev_dur_s=rs[0]["ev_dur_s"],
                               theta_on=th_on, theta_off=th_off,
                               max_score=max(r["score"] for r in rs),
                               n_win_above_on=sum(r["score"] > th_on for r in rs)))
        for i, (te, _k, sc) in enumerate(fp):
            j = int(round(te / WIN_S)) - 1
            detail.append(dict(row_type="fp_event", convention=conv, session=sess,
                               event=tag[j][0], n_win=1, theta_on=th_on, theta_off=th_off,
                               max_score=sc, latency_s=te, first_open_win=i + 1))

        allev = sorted(iw + rest, key=lambda kv: kv[1][0]["ev_a_samp"])
        ut, us = [], []
        for _name, rs in allev:
            for r in rs:
                ut.append(len(ut) * WIN_S + WIN_S)
                us.append(r["score"])
        ust, uev = hysteresis(ut, us, th_on, th_off)
        u_open_iw = 0
        k = 0
        for _name, rs in allev:
            for r in rs:
                k += 1
        for te, kind, _sc in uev:
            if kind != "open":
                continue
            j = int(round(te / WIN_S)) - 1
            lab = [r["label"] for _n, rs in allev for r in rs][j]
            u_open_iw += int(lab == 1)

        n_iw = len(iw)
        summary.append(dict(
            row_type="session_summary", convention=conv, session=sess,
            n_iw_trials=n_iw, n_detected=n_det, tpr=(n_det / n_iw if n_iw else np.nan),
            n_iw_win=sum(len(rs) for _n, rs in iw),
            n_rest_events=len(rest), n_rest_win=len(t), rest_min=rest_s / 60.0,
            n_fp_events=len(fp), fp_per_min=(len(fp) / (rest_s / 60.0) if rest_s else np.nan),
            fp_win_rate=(sum(np.array(s) > th_on) / len(s) if s else np.nan),
            latency_median_s=(float(np.median(lat)) if lat else np.nan),
            latency_q1_s=(float(np.percentile(lat, 25)) if lat else np.nan),
            latency_q3_s=(float(np.percentile(lat, 75)) if lat else np.nan),
            n_latency=len(lat),
            theta_on=th_on, theta_off=th_off,
            K_calib_events=T["K_calib_events"], calib_cum_s=T["calib_cum_s"],
            n_calib_win=T["n_calib_win"],
            unified_n_win=len(ut), unified_n_open=sum(1 for e in uev if e[1] == "open"),
            unified_n_open_on_iw=u_open_iw,
            unified_open_frac=(float(np.mean(ust)) if ust else np.nan)))
    return detail, summary


def aggregate(detail, summary, conv, scope, sessions):
    S = [r for r in summary if r["convention"] == conv and r["session"] in sessions]
    lat = [float(r["latency_s"]) for r in detail
           if r["row_type"] == "iw_trial" and r["convention"] == conv
           and r["session"] in sessions and r["latency_s"] != ""]
    n_tr = sum(r["n_iw_trials"] for r in S)
    n_det = sum(r["n_detected"] for r in S)
    rest_min = sum(r["rest_min"] for r in S)
    n_fp = sum(r["n_fp_events"] for r in S)
    return dict(row_type="overall_summary", convention=conv, session=scope,
                n_sessions=len(S), n_iw_trials=n_tr, n_detected=n_det,
                tpr=n_det / n_tr, tpr_session_mean=float(np.mean([r["tpr"] for r in S])),
                n_rest_win=sum(r["n_rest_win"] for r in S), rest_min=rest_min,
                n_fp_events=n_fp, fp_per_min=n_fp / rest_min,
                fp_per_min_session_mean=float(np.mean([r["fp_per_min"] for r in S])),
                fp_win_rate=float(np.mean([r["fp_win_rate"] for r in S])),
                latency_median_s=float(np.median(lat)), latency_q1_s=float(np.percentile(lat, 25)),
                latency_q3_s=float(np.percentile(lat, 75)), n_latency=len(lat),
                latency_median_win=float(np.median(lat)) / WIN_S)


def main():
    win, thr, fuse = load()
    detail, summary, agg, expl = [], [], [], []
    for conv in ("demobase", "dumpbase"):
        d, s = run_convention(win, thr, fuse, conv)
        detail += d
        summary += s
        for scope, ss in (("ALL7", SESSIONS),
                          ("P1", [x for x in SESSIONS if x.startswith("pat01")]),
                          ("P2", [x for x in SESSIONS if x.startswith("pat02")])):
            agg.append(aggregate(d, s, conv, scope, ss))

    for conv in ("demobase", "dumpbase"):
        for k in (0.0, 0.25, 0.5, 1.0):
            d, s = run_convention(win, thr, fuse, conv, iqr_k=k)
            a = aggregate(d, s, conv, "ALL7", SESSIONS)
            a["row_type"] = "exploratory_threshold_sensitivity"
            a["iqr_k"] = k
            a["frozen"] = int(k == IQR_K)
            expl.append(a)

    rows = detail + summary + agg + expl
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    head = ["row_type", "convention", "session", "event"]
    cols = head + [c for c in cols if c not in head]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)

    for a in agg:
        print(f"{a['convention']:9s} {a['session']:5s}  TPR {a['tpr']:.3f} "
              f"({a['n_detected']}/{a['n_iw_trials']})  FP {a['fp_per_min']:.2f}/min "
              f"({a['n_fp_events']} in {a['rest_min']:.1f} min)  "
              f"lat {a['latency_median_s']:.1f} s "
              f"[{a['latency_q1_s']:.1f}-{a['latency_q3_s']:.1f}]", flush=True)
    print(f"\nwrote {OUT}: {len(rows)} rows")
    return rows


if __name__ == "__main__":
    main()
