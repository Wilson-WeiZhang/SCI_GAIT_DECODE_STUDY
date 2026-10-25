"""Figure 7 — physics simulation and walk-intent gate

gate_working_point_symmetric — the gate working point of the fused (V1) classifier.

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import csv
import os

import numpy as np

import gate as GATE
from gate_dense_hop import ordered, rest_stream, run_consec

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "result")
OUT = os.path.join(RES, "gate_working_point_symmetric.csv")
SESSIONS = ["pat01_sess04", "pat01_sess06", "pat01_sess07", "pat01_sess09",
            "pat02_sess03", "pat02_sess05", "pat02_sess06"]
P1 = [s for s in SESSIONS if s.startswith("pat01")]
P2 = [s for s in SESSIONS if s.startswith("pat02")]
WIN_S = 4.0
SUP = dict(tpr=208 / 270, fp_true=0.7393, lat_med=4.0, lat_mean=4.2692, dec=4.0481,
           first_ok=208, n_rest_ev=224)


def assert_config():
    want = dict(HOP_S=1.0, CLOSE_M=3, CLOSE_N=3, N_ON=1, DWELL_S=8.0, IQR_K=0.25, OFF_Q=90,
                BASE_MIN_S=30.0)
    got = {k: getattr(GATE, k) for k in want}
    assert got == want, f"gate.py {got} != adopted densegrid grid {want}"
    return got


def load():
    sc, thr = {}, {}
    for r in csv.DictReader(open(os.path.join(RES, "gate_fused_scores.csv"),
                                 encoding="utf-8")):
        sc.setdefault((r["session"], r["label"] == "1"), {}).setdefault(r["event"], []).append(
            dict(t=float(r["t_end_ev_s"]), s=float(r["fuse_V1"]), dur=float(r["ev_dur_s"]),
                 n_dec=int(r["n_dec_ev"])))
    for v in sc.values():
        for rs in v.values():
            rs.sort(key=lambda r: r["t"])
    for r in csv.DictReader(open(os.path.join(RES, "gate_fused_thresholds.csv"),
                                 encoding="utf-8")):
        if r["variant"] != "V1":
            continue
        t = ((0.0 - float(r["centre_A"])) / float(r["sd_A"])
             + (0.0 - float(r["centre_B"])) / float(r["sd_B"])) / 2.0
        d = GATE.IQR_K * float(r["calib_iqr"])
        thr[r["session"]] = (t + d, t - d)
    return sc, thr


def main():
    cfg = assert_config()
    print(f"[config] live gate.py grid = {cfg}", flush=True)
    sc, thr = load()
    rows, per = [], {}

    for sess in SESSIONS:
        th_on, th_off = thr[sess]
        iw = ordered(sc[(sess, True)])
        n_det, lat, dec_iw = 0, [], 0
        for name, rs in iw:
            t = [r["t"] for r in rs]
            s = [r["s"] for r in rs]
            dec_iw += len(rs)
            op = [e[0] for e in run_consec(t, s, th_on, th_off, GATE.N_ON) if e[1] == "open"]
            if op:
                n_det += 1
                lat.append(op[0])
            rows.append(dict(row_type="iw_trial", grid="symdg", session=sess, event=name,
                             n_dec=len(rs), ev_dur_s=rs[0]["dur"], detected=int(bool(op)),
                             latency_s=(op[0] if op else ""), theta_on=th_on, theta_off=th_off))
        rest = ordered(sc[(sess, False)])
        t, s, span = rest_stream(rest)
        evs = run_consec(t, s, th_on, th_off, GATE.N_ON)
        opens = [e[0] for e in evs if e[1] == "open"]
        first_wrong = [n for n, rs in rest if rs[0]["s"] > th_on]
        true_rest = sum(rs[0]["dur"] for _n, rs in rest)
        for i, te in enumerate(opens):
            rows.append(dict(row_type="fp_event", grid="symdg", session=sess, fp_idx=i + 1,
                             t_open_s=te))
        per[sess] = dict(n_iw=len(iw), n_det=n_det, lat=lat, dec_iw=dec_iw,
                         n_rest_ev=len(rest), n_dec_rest=len(t), rest_s=span,
                         true_rest_s=true_rest, n_fp=len(opens), n_first_wrong=len(first_wrong))
        rows.append(dict(row_type="session_summary", grid="symdg", session=sess,
                         n_iw_trials=len(iw), n_detected=n_det, tpr=n_det / len(iw),
                         n_dec_iw=dec_iw, dec_per_trial=dec_iw / len(iw),
                         n_rest_events=len(rest), n_dec_rest=len(t),
                         rest_min=span / 60.0, rest_min_true=true_rest / 60.0,
                         n_fp_events=len(opens), fp_per_min=len(opens) / (span / 60.0),
                         fp_per_min_true_rest=len(opens) / (true_rest / 60.0),
                         n_first_wrong=len(first_wrong),
                         first_correct=len(rest) - len(first_wrong),
                         first_correct_frac=(len(rest) - len(first_wrong)) / len(rest),
                         latency_median_s=(float(np.median(lat)) if lat else np.nan),
                         latency_q1_s=(float(np.percentile(lat, 25)) if lat else np.nan),
                         latency_q3_s=(float(np.percentile(lat, 75)) if lat else np.nan),
                         latency_mean_s=(float(np.mean(lat)) if lat else np.nan),
                         theta_on=th_on, theta_off=th_off))

    def agg(name, ss):
        det = sum(per[s]["n_det"] for s in ss)
        ntr = sum(per[s]["n_iw"] for s in ss)
        lat = [x for s in ss for x in per[s]["lat"]]
        nfp = sum(per[s]["n_fp"] for s in ss)
        rs_ = sum(per[s]["rest_s"] for s in ss)
        tr = sum(per[s]["true_rest_s"] for s in ss)
        nre = sum(per[s]["n_rest_ev"] for s in ss)
        nfw = sum(per[s]["n_first_wrong"] for s in ss)
        return dict(row_type="overall_summary", grid="symdg", session=name, n_sessions=len(ss),
                    n_iw_trials=ntr, n_detected=det, tpr=det / ntr,
                    n_dec_iw=sum(per[s]["dec_iw"] for s in ss),
                    dec_per_trial=sum(per[s]["dec_iw"] for s in ss) / ntr,
                    n_rest_events=nre, n_dec_rest=sum(per[s]["n_dec_rest"] for s in ss),
                    rest_min=rs_ / 60.0, rest_min_true=tr / 60.0, n_fp_events=nfp,
                    fp_per_min=nfp / (rs_ / 60.0), fp_per_min_true_rest=nfp / (tr / 60.0),
                    n_first_wrong=nfw, first_correct=nre - nfw,
                    first_correct_frac=(nre - nfw) / nre,
                    latency_median_s=float(np.median(lat)),
                    latency_mean_s=float(np.mean(lat)),
                    latency_q1_s=float(np.percentile(lat, 25)),
                    latency_q3_s=float(np.percentile(lat, 75)),
                    latency_min_s=float(np.min(lat)), latency_max_s=float(np.max(lat)),
                    n_latency=len(lat))

    A = agg("ALL7", SESSIONS)
    rows += [A, agg("P1", P1), agg("P2", P2)]
    lat_all = [x for s in SESSIONS for x in per[s]["lat"]]
    hist = {v: lat_all.count(v) for v in sorted(set(lat_all))}
    for v, c in hist.items():
        rows.append(dict(row_type="latency_hist", grid="symdg", latency_s=v, n_trials=c))
    for k, new, old in (("tpr", A["tpr"], SUP["tpr"]),
                        ("fp_per_min_true_rest", A["fp_per_min_true_rest"], SUP["fp_true"]),
                        ("latency_median_s", A["latency_median_s"], SUP["lat_med"]),
                        ("latency_mean_s", A["latency_mean_s"], SUP["lat_mean"]),
                        ("first_correct", A["first_correct"], SUP["first_ok"])):
        rows.append(dict(row_type="build_id_comparison", grid="symdg", metric=k, symdg=new,
                         fuse2dg_previous=old, delta=new - old))

    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    head = ["row_type", "grid", "session"]
    cols = head + [c for c in cols if c not in head]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)

    print("\n== gate working point ==")
    print(f"  TPR {A['tpr']:.4f} ({A['n_detected']}/{A['n_iw_trials']})  "
          f"FP {A['fp_per_min_true_rest']:.3f}/min true rest ({A['n_fp_events']} events)")
    print(f"  latency median {A['latency_median_s']:.1f}s mean {A['latency_mean_s']:.3f}s "
          f"range {A['latency_min_s']:.0f}-{A['latency_max_s']:.0f}s")
    print(f"  rest first-decision correct {A['first_correct']}/{A['n_rest_events']} = "
          f"{A['first_correct_frac']:.4f}   ")
    print(f"  latency hist {hist}")
    for r in rows:
        if r["row_type"] == "overall_summary" and r["session"] in ("P1", "P2"):
            print(f"  {r['session']}: TPR {r['tpr']:.4f}  FP {r['fp_per_min_true_rest']:.3f}")
    print(f"\nwrote {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
