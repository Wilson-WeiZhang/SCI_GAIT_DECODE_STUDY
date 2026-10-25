"""Figure 7 — physics simulation and walk-intent gate

gate_fusion_working_point — the working point of the fused gate classifier and its FP / TPR trade curve.

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate as GATE
from gate_dense_hop import (DWELL_S, HOP_S, N_OFF, SESSIONS, WIN_S,
                                  ordered, run_consec)

RES = os.path.join(HERE, "result")
OUT = os.path.join(RES, "gate_fused.csv")
VARIANTS = ("V1", "V2", "V3")
PRIMARY = "V1"

LDA_WP = dict(tpr=173 / 270, n_det=173, n_trials=270, fp_true=0.5750022, n_fp=14,
              true_rest_s=1461.0, latency_median=4.0, latency_mean=4.6473988,
              first_correct=211, n_rest_ev=224)
K_GRID = np.round(np.arange(4.0, -1.5001, -0.02), 4)
FP_BUDGETS = (0.10, 0.25, 0.40, 0.575, 0.75, 1.00)


def assert_config():
    want = dict(HOP_S=1.0, CLOSE_M=3, CLOSE_N=3, N_ON=1, DWELL_S=8.0, IQR_K=0.25, OFF_Q=90,
                BASE_MIN_S=30.0, WIN_S=4.0)
    got = {k: getattr(GATE, k) for k in want}
    assert got == want, f"gate.py config {got} != frozen densegrid {want}"
    assert (WIN_S, HOP_S, DWELL_S, N_OFF) == (4.0, 1.0, 8.0, 3), "replay helper constants drifted"
    return got


def load_fuse():
    sc, thr = {}, {}
    for r in csv.DictReader(open(os.path.join(RES, "gate_fused_scores.csv"),
                                 encoding="utf-8")):
        e = dict(t=float(r["t_end_ev_s"]), dur=float(r["ev_dur_s"]),
                 **{v: float(r[f"fuse_{v}"]) for v in VARIANTS})
        sc.setdefault((r["session"], r["label"] == "1"), {}).setdefault(r["event"], []).append(e)
    for v in sc.values():
        for rs in v.values():
            rs.sort(key=lambda r: r["t"])
    for r in csv.DictReader(open(os.path.join(RES, "gate_fused_thresholds.csv"),
                                 encoding="utf-8")):
        thr[(r["session"], r["variant"])] = {k: float(r[k]) for k in
                                             ("calib_max", "calib_iqr", "theta_on", "theta_off",
                                              "n_calib_win")}
    return sc, thr


def load_lda():
    sc, thr = {}, {}
    for r in csv.DictReader(open(os.path.join(RES, "gate_dense_hop_scores.csv"),
                                 encoding="utf-8")):
        sc.setdefault((r["session"], r["label"] == "1"), {}).setdefault(r["event"], []).append(
            dict(t=float(r["t_end_ev_s"]), s=float(r["score"]), dur=float(r["ev_dur_s"])))
    for v in sc.values():
        for rs in v.values():
            rs.sort(key=lambda r: r["t"])
    for r in csv.DictReader(open(os.path.join(RES, "gate_dense_hop_thresholds.csv"),
                                 encoding="utf-8")):
        thr[r["session"]] = {k: float(v) for k, v in r.items() if k != "session"}
    return sc, thr


def replay(sc, thr, key, iqr_k=None):
    """One full two-condition replay. `key` selects the score field; iqr_k=None uses the frozen theta."""
    per = {}
    for s in SESSIONS:
        T = thr[s]
        th_on, th_off = T["theta_on"], T["theta_off"]
        if iqr_k is not None:
            th_on = T["calib_max"] + iqr_k * T["calib_iqr"]
            th_off = min(T["theta_off"], th_on - 1e-6)
        iw = ordered(sc[(s, True)])
        n_det, lat, n_dec = 0, [], 0
        trials = []
        for name, rs in iw:
            t = [r["t"] for r in rs]
            v = [r[key] for r in rs]
            n_dec += len(rs)
            op = [e[0] for e in run_consec(t, v, th_on, th_off, GATE.N_ON) if e[1] == "open"]
            if op:
                n_det += 1
                lat.append(op[0])
            trials.append((name, bool(op), (op[0] if op else None), len(rs), rs[0]["dur"]))
        rest = ordered(sc[(s, False)])
        t, v, span = [], [], 0.0
        for _n, rs in rest:
            for j, r in enumerate(rs):
                t.append(span + WIN_S + j * HOP_S)
                v.append(r[key])
            span += len(rs) + 3.0
        opens = [e[0] for e in run_consec(t, v, th_on, th_off, GATE.N_ON) if e[1] == "open"]
        first_wrong = [n for n, rs in rest if rs[0][key] > th_on]
        per[s] = dict(n_iw=len(iw), n_det=n_det, lat=lat, n_dec_iw=n_dec, trials=trials,
                      n_rest_ev=len(rest), n_dec_rest=len(t), rest_s=span,
                      true_rest_s=sum(rs[0]["dur"] for _n, rs in rest),
                      n_fp=len(opens), fp_t=opens, n_first_wrong=len(first_wrong),
                      theta_on=th_on, theta_off=th_off)
    return per


def agg(per, ss):
    det = sum(per[s]["n_det"] for s in ss)
    ntr = sum(per[s]["n_iw"] for s in ss)
    lat = [x for s in ss for x in per[s]["lat"]]
    nfp = sum(per[s]["n_fp"] for s in ss)
    tr = sum(per[s]["true_rest_s"] for s in ss)
    sp = sum(per[s]["rest_s"] for s in ss)
    nre = sum(per[s]["n_rest_ev"] for s in ss)
    nfw = sum(per[s]["n_first_wrong"] for s in ss)
    return dict(n_sessions=len(ss), n_iw_trials=ntr, n_detected=det, tpr=det / ntr,
                n_dec_iw=sum(per[s]["n_dec_iw"] for s in ss),
                dec_per_trial=sum(per[s]["n_dec_iw"] for s in ss) / ntr,
                n_rest_events=nre, n_dec_rest=sum(per[s]["n_dec_rest"] for s in ss),
                rest_min=sp / 60.0, rest_min_true=tr / 60.0, n_fp_events=nfp,
                fp_per_min=nfp / (sp / 60.0), fp_per_min_true_rest=nfp / (tr / 60.0),
                n_first_wrong=nfw, first_correct=nre - nfw,
                first_correct_frac=(nre - nfw) / nre,
                latency_median_s=(float(np.median(lat)) if lat else np.nan),
                latency_mean_s=(float(np.mean(lat)) if lat else np.nan),
                latency_q1_s=(float(np.percentile(lat, 25)) if lat else np.nan),
                latency_q3_s=(float(np.percentile(lat, 75)) if lat else np.nan),
                latency_min_s=(float(np.min(lat)) if lat else np.nan),
                latency_max_s=(float(np.max(lat)) if lat else np.nan), n_latency=len(lat))


def curve(sc, thr, key, tag, rows):
    """FP/TPR trade curve over the IQR_K scan + its Pareto frontier (best TPR at FP <= budget)."""
    pts = []
    for k in K_GRID:
        A = agg(replay(sc, thr, key, iqr_k=float(k)), SESSIONS)
        pts.append((float(k), A["fp_per_min_true_rest"], A["tpr"], A["n_fp_events"],
                    A["n_detected"]))
        rows.append(dict(row_type="trade_curve", condition=tag, iqr_k=float(k),
                         fp_per_min_true_rest=A["fp_per_min_true_rest"], tpr=A["tpr"],
                         n_fp_events=A["n_fp_events"], n_detected=A["n_detected"],
                         latency_median_s=A["latency_median_s"]))
    front = []
    for _k, f, t, _nf, _nd in sorted(pts, key=lambda c: (c[1], c[2])):
        if not front or t > front[-1][1]:
            front.append((f, t))
    for f, t in front:
        rows.append(dict(row_type="pareto_frontier", condition=tag, fp_per_min_true_rest=f, tpr=t))
    return pts, ([p[0] for p in front], [p[1] for p in front])


def at_fp(front, target):
    """TPR reachable at FP <= target by re-thresholding alone (frontier interpolation)."""
    fx, fy = front
    if target < fx[0]:
        return float("nan")
    if target > fx[-1]:
        return fy[-1]
    return float(np.interp(target, fx, fy))


def _mcnemar(b, c):
    """Exact two-sided McNemar (binomial on the discordant pairs). Descriptive only here."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def nearest_k(pts, target):
    """The scanned threshold whose FP is closest to `target` (ties -> higher TPR)."""
    best = min(pts, key=lambda p: (abs(p[1] - target), -p[2]))
    return best


def main():
    cfg = assert_config()
    print(f"[config] live gate.py = {cfg}", flush=True)
    fsc, fthr = load_fuse()
    lsc, lthr = load_lda()
    rows = []

    for r in csv.DictReader(open(os.path.join(RES, "gate_fused_auc.csv"), encoding="utf-8")):
        rows.append(dict(row_type="auc", **r))

    per_v = {}
    for v in VARIANTS:
        thr_v = {s: fthr[(s, v)] for s in SESSIONS}
        per = replay(fsc, thr_v, v)
        per_v[v] = per
        A = agg(per, SESSIONS)
        rows.append(dict(row_type="working_point", condition=f"fuse2_{v}", session="ALL7",
                         frozen_theta=1, **A))
        for scope, ss in (("P1", [s for s in SESSIONS if s.startswith("pat01")]),
                          ("P2", [s for s in SESSIONS if s.startswith("pat02")])):
            rows.append(dict(row_type="working_point", condition=f"fuse2_{v}", session=scope,
                             frozen_theta=1, **agg(per, ss)))
        for s in SESSIONS:
            p = per[s]
            rows.append(dict(row_type="wp_session", condition=f"fuse2_{v}", session=s,
                             n_iw_trials=p["n_iw"], n_detected=p["n_det"],
                             tpr=p["n_det"] / p["n_iw"], n_dec_iw=p["n_dec_iw"],
                             n_rest_events=p["n_rest_ev"], n_dec_rest=p["n_dec_rest"],
                             rest_min_true=p["true_rest_s"] / 60.0, n_fp_events=p["n_fp"],
                             fp_per_min_true_rest=p["n_fp"] / (p["true_rest_s"] / 60.0),
                             first_correct=p["n_rest_ev"] - p["n_first_wrong"],
                             latency_median_s=(float(np.median(p["lat"])) if p["lat"]
                                               else np.nan),
                             theta_on=p["theta_on"], theta_off=p["theta_off"],
                             n_calib_win=fthr[(s, v)]["n_calib_win"]))
        if v == PRIMARY:
            for s in SESSIONS:
                for name, det, lat, nd, dur in per[s]["trials"]:
                    rows.append(dict(row_type="iw_trial", condition=f"fuse2_{v}", session=s,
                                     event=name, detected=int(det),
                                     latency_s=(lat if det else ""), n_dec=nd, ev_dur_s=dur))
                for i, t in enumerate(per[s]["fp_t"]):
                    rows.append(dict(row_type="fp_event", condition=f"fuse2_{v}", session=s,
                                     fp_idx=i + 1, t_open_s=t))
            lat = [x for s in SESSIONS for x in per[s]["lat"]]
            for u in sorted(set(lat)):
                rows.append(dict(row_type="latency_hist", condition=f"fuse2_{v}", latency_s=u,
                                 n_trials=lat.count(u)))
        print(f"[working point] fuse2 {v}: TPR {A['tpr']:.4f} ({A['n_detected']}/"
              f"{A['n_iw_trials']}), FP {A['fp_per_min_true_rest']:.3f}/min true rest "
              f"({A['n_fp_events']} ev), latency median {A['latency_median_s']:.1f}s "
              f"mean {A['latency_mean_s']:.2f}s", flush=True)

    perL = replay(lsc, lthr, "s")
    AL = agg(perL, SESSIONS)
    rows.append(dict(row_type="working_point", condition="lda_frozen_densegrid", session="ALL7",
                     frozen_theta=1, **AL))
    for scope, ss in (("P1", [s for s in SESSIONS if s.startswith("pat01")]),
                      ("P2", [s for s in SESSIONS if s.startswith("pat02")])):
        rows.append(dict(row_type="working_point", condition="lda_frozen_densegrid", session=scope,
                         frozen_theta=1, **agg(perL, ss)))
    assert AL["n_detected"] == LDA_WP["n_det"] and AL["n_fp_events"] == LDA_WP["n_fp"], \
        f"LDA comparator {AL['n_detected']}/{AL['n_fp_events']} != reference"
    assert abs(AL["fp_per_min_true_rest"] - LDA_WP["fp_true"]) < 1e-6
    assert AL["latency_median_s"] == LDA_WP["latency_median"]
    assert abs(AL["latency_mean_s"] - LDA_WP["latency_mean"]) < 1e-6
    assert AL["first_correct"] == LDA_WP["first_correct"]
    assert AL["n_rest_events"] == LDA_WP["n_rest_ev"]
    print(f"[comparator] frozen LDA densegrid reproduced field for field: TPR {AL['tpr']:.4f} "
          f"({AL['n_detected']}/{AL['n_iw_trials']}) FP {AL['fp_per_min_true_rest']:.4f}/min, "
          f"latency {AL['latency_median_s']:.1f}/{AL['latency_mean_s']:.4f}s, "
          f"first-decision {AL['first_correct']}/{AL['n_rest_events']} OK", flush=True)

    ptsL, frontL = curve(lsc, lthr, "s", "lda_frozen_densegrid", rows)
    ptsF, frontF = {}, {}
    for v in VARIANTS:
        thr_v = {s: fthr[(s, v)] for s in SESSIONS}
        ptsF[v], frontF[v] = curve(fsc, thr_v, v, f"fuse2_{v}", rows)
    print(f"[curves] LDA frontier FP {frontL[0][0]:.3f}-{frontL[0][-1]:.3f} "
          f"TPR {frontL[1][0]:.3f}-{frontL[1][-1]:.3f}; "
          f"fuse2 {PRIMARY} frontier FP {frontF[PRIMARY][0][0]:.3f}-{frontF[PRIMARY][0][-1]:.3f} "
          f"TPR {frontF[PRIMARY][1][0]:.3f}-{frontF[PRIMARY][1][-1]:.3f}", flush=True)

    print(f"\n{'FP budget':>10s} {'LDA TPR':>8s} " +
          " ".join(f"{'fuse2 ' + v:>10s}" for v in VARIANTS) + f" {'d(V1-LDA)':>10s}")
    for bud in FP_BUDGETS:
        tl = at_fp(frontL, bud)
        tf = {v: at_fp(frontF[v], bud) for v in VARIANTS}
        rows.append(dict(row_type="trade_summary", condition="frontier_at_fp_budget",
                         fp_per_min_true_rest=bud, tpr=tl, reference_tpr=tl,
                         **{f"tpr_{v}": tf[v] for v in VARIANTS},
                         d_tpr=tf[PRIMARY] - tl))
        print(f"{bud:10.3f} {tl:8.4f} " + " ".join(f"{tf[v]:10.4f}" for v in VARIANTS) +
              f" {tf[PRIMARY] - tl:+10.4f}")

    for v in VARIANTS:
        k, fp, tpr, nfp, ndet = nearest_k(ptsF[v], LDA_WP["fp_true"])
        A = agg(replay(fsc, {s: fthr[(s, v)] for s in SESSIONS}, v, iqr_k=k), SESSIONS)
        rows.append(dict(row_type="prenamed_matched_fp", condition=f"fuse2_{v}",
                         session="fuse2_at_LDA_FP", iqr_k=k,
                         fp_per_min_true_rest=fp, fp_target=LDA_WP["fp_true"],
                         fp_delta=fp - LDA_WP["fp_true"], n_fp_events=nfp,
                         tpr=tpr, reference_tpr=LDA_WP["tpr"], d_tpr=tpr - LDA_WP["tpr"],
                         n_detected=ndet, reference_n_detected=LDA_WP["n_det"],
                         latency_median_s=A["latency_median_s"],
                         reference_latency_s=LDA_WP["latency_median"],
                         tpr_from_frontier=at_fp(frontF[v], LDA_WP["fp_true"]),
                         note="fuse2 re-thresholded to the LDA FP " "(0.575/min true rest)"))
        Af = agg(per_v[v], SESSIONS)
        kL, fpL, tprL, nfpL, ndetL = nearest_k(ptsL, Af["fp_per_min_true_rest"])
        rows.append(dict(row_type="prenamed_matched_fp", condition="lda_at_fuse2_FP",
                         session=f"vs_fuse2_{v}", iqr_k=kL, fp_per_min_true_rest=fpL,
                         fp_target=Af["fp_per_min_true_rest"],
                         fp_delta=fpL - Af["fp_per_min_true_rest"], n_fp_events=nfpL,
                         tpr=tprL, reference_tpr=Af["tpr"], d_tpr=Af["tpr"] - tprL,
                         n_detected=ndetL, reference_n_detected=Af["n_detected"],
                         tpr_from_frontier=at_fp(frontL, Af["fp_per_min_true_rest"]),
                         note="reverse reading: the frozen LDA re-thresholded to the fused "
                              "gate's own FP; d_tpr > 0 means the fusion is above the LDA curve"))
        perM = replay(fsc, {s: fthr[(s, v)] for s in SESSIONS}, v, iqr_k=k)
        b = c = 0
        for s in SESSIONS:
            dl = {n: d for n, d, _l, _nd, _du in perL[s]["trials"]}
            for n, d, _l, _nd, _du in perM[s]["trials"]:
                b += int(d and not dl[n])
                c += int(dl[n] and not d)
            rows.append(dict(row_type="matched_fp_session", condition=f"fuse2_{v}", session=s,
                             n_iw_trials=perM[s]["n_iw"], n_detected=perM[s]["n_det"],
                             tpr=perM[s]["n_det"] / perM[s]["n_iw"],
                             reference_n_detected=perL[s]["n_det"],
                             reference_tpr=perL[s]["n_det"] / perL[s]["n_iw"],
                             d_tpr=(perM[s]["n_det"] - perL[s]["n_det"]) / perM[s]["n_iw"],
                             n_fp_events=perM[s]["n_fp"],
                             reference_n_fp=perL[s]["n_fp"], iqr_k=k))
        p = _mcnemar(b, c)
        rows.append(dict(row_type="matched_fp_paired", condition=f"fuse2_{v}", session="ALL7",
                         iqr_k=k, n_iw_trials=270, n_gain_only=b, n_loss_only=c,
                         d_tpr=(b - c) / 270.0, mcnemar_exact_p=p,
                         note="descriptive: the 270 trials are nested in 7 sessions from "
                              "n = 2 patients"))
        d = tpr - LDA_WP["tpr"]
        print(f" fuse2 {v} @ FP {fp:.3f}/min (target 0.575, k={k:+.2f}): "
              f"TPR {tpr:.4f} vs LDA {LDA_WP['tpr']:.4f} -> {d:+.4f} | paired discordant "
              f"gain {b} / loss {c}, McNemar exact p = {p:.3g} (descriptive)", flush=True)

    for v in VARIANTS:
        Af = agg(per_v[v], SESSIONS)
        ref = at_fp(frontL, Af["fp_per_min_true_rest"])
        rows.append(dict(row_type="curve_position", condition=f"fuse2_{v}", session="ALL7",
                         fp_per_min_true_rest=Af["fp_per_min_true_rest"], tpr=Af["tpr"],
                         reference_tpr=ref, d_tpr=Af["tpr"] - ref,
                         note="fused gate at its own frozen theta vs the TPR the frozen LDA "
                              "reaches at the SAME FP by re-thresholding alone"))
        print(f"[curve position] fuse2 {v} frozen theta: FP {Af['fp_per_min_true_rest']:.3f} "
              f"TPR {Af['tpr']:.4f} vs LDA curve {ref:.4f} -> {Af['tpr'] - ref:+.4f}", flush=True)

    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    head = ["row_type", "condition", "session", "event"]
    cols = head + [c for c in cols if c not in head]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
