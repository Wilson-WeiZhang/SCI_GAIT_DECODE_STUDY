"""Figure 7 — physics simulation and walk-intent gate

gate_dense_hop — four switching logics over the 1 s-hop inference stream

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import csv
import os

import numpy as np

from gate_working_point import hysteresis

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "result")
OUT = os.path.join(RES, "gate_dense_hop.csv")

SESSIONS = ["pat01_sess04", "pat01_sess06", "pat01_sess07", "pat01_sess09",
            "pat02_sess03", "pat02_sess05", "pat02_sess06"]
P1 = [s for s in SESSIONS if s.startswith("pat01")]
WIN_S, HOP_S, DWELL_S, N_OFF = 4.0, 1.0, 8.0, 3


def load():
    sc = {}
    for r in csv.DictReader(open(os.path.join(RES, "gate_dense_hop_scores.csv"),
                                 encoding="utf-8")):
        sc.setdefault((r["session"], r["label"] == "1"), {}).setdefault(r["event"], []).append(
            dict(t=float(r["t_end_ev_s"]), s=float(r["score"]), dur=float(r["ev_dur_s"]),
                 n_dec=int(r["n_dec_ev"])))
    for v in sc.values():
        for rs in v.values():
            rs.sort(key=lambda r: r["t"])
    thr = {r["session"]: {k: float(v) for k, v in r.items() if k != "session"}
           for r in csv.DictReader(open(os.path.join(RES,
                                                     "gate_dense_hop_thresholds.csv"),
                                        encoding="utf-8"))}
    return sc, thr


def ordered(ev):
    return [(k, ev[k]) for k in sorted(ev, key=lambda e: int(e.lstrip("irestw")))]


def run_consec(t, s, th_on, th_off, n_on):
    """Open on n_on CONSECUTIVE decisions above theta_on; close on N_OFF consecutive below
    theta_off, never inside DWELL_S of opening."""
    cur, above, below, t_open, ev = 0, 0, 0, -1e9, []
    for te, v in zip(t, s):
        if cur == 0:
            above = above + 1 if v > th_on else 0
            if above >= n_on:
                cur, t_open, above, below = 1, te, 0, 0
                ev.append((te, "open"))
        else:
            below = below + 1 if v < th_off else 0
            if below >= N_OFF and (te - t_open) >= DWELL_S:
                cur, below = 0, 0
                ev.append((te, "close"))
    return ev


def run_integrator(t, s, th_on, th_off, lam, bound):
    """Leaky evidence accumulator; opening and closing are symmetric. Zero hand-tuned numbers:
    lam and bound come from IQR(calibration)."""
    cur, acc, dacc, t_open, ev = 0, 0.0, 0.0, -1e9, []
    for te, v in zip(t, s):
        if cur == 0:
            acc = max(0.0, acc + v - th_on - lam)
            if acc > bound:
                cur, t_open, acc, dacc = 1, te, 0.0, 0.0
                ev.append((te, "open"))
        else:
            dacc = max(0.0, dacc + (th_off - v) - lam)
            if dacc > bound and (te - t_open) >= DWELL_S:
                cur, dacc, acc = 0, 0.0, 0.0
                ev.append((te, "close"))
    return ev


def logic(name, t, s, T):
    if name == "A0":
        return run_consec(t, s, T["theta_on"], T["theta_off"], 1)
    if name == "A2":
        return run_consec(t, s, T["theta_on"], T["theta_off"], 2)
    if name == "A3":
        return run_consec(t, s, T["theta_on"], T["theta_off"], 3)
    if name == "B":
        return run_integrator(t, s, T["theta_on"], T["theta_off"], T["lam"], T["bound"])
    raise ValueError(name)


def rest_stream(rest_events):
    """Concatenate rest events onto one timeline. An event with n decisions monitors n+3 s
    (its first decision covers [0,4), each later one adds 1 s)."""
    t, s, x0 = [], [], 0.0
    for _name, rs in rest_events:
        for j, r in enumerate(rs):
            t.append(x0 + WIN_S + j * HOP_S)
            s.append(r["s"])
        x0 += len(rs) + 3.0
    return t, s, x0


def evaluate(sc, thr, name, four_s_grid=False, frozen_logic=False, iqr_k=None):
    per = {}
    for sess in SESSIONS:
        T = dict(thr[sess])
        if iqr_k is not None:
            T["theta_on"] = T["calib_max"] + iqr_k * T["calib_iqr"]
            T["theta_off"] = min(T["theta_off"], T["theta_on"] - 1e-6)
        iw = ordered(sc[(sess, True)])
        rest = ordered(sc[(sess, False)])
        if four_s_grid:
            keep = (lambda rs: [r for r in rs if abs(r["t"] / WIN_S - round(r["t"] / WIN_S))
                                < 1e-9])
            iw = [(n, keep(rs)) for n, rs in iw]
            rest = [(n, keep(rs)) for n, rs in rest]

        n_det, lat = 0, []
        for _n, rs in iw:
            t = [r["t"] for r in rs]
            s = [r["s"] for r in rs]
            ev = (hysteresis(t, s, T["theta_on"], T["theta_off"])[1] if frozen_logic
                  else logic(name, t, s, T))
            op = [e[0] for e in ev if e[1] == "open"]
            if op:
                n_det += 1
                lat.append(op[0])

        if four_s_grid:
            t, s, span = [], [], 0.0
            for _n, rs in rest:
                for j, r in enumerate(rs):
                    t.append(span + WIN_S + j * WIN_S)
                    s.append(r["s"])
                span += len(rs) * WIN_S
        else:
            t, s, span = rest_stream(rest)
        ev = (hysteresis(t, s, T["theta_on"], T["theta_off"])[1] if frozen_logic
              else logic(name, t, s, T))
        opens = [e[0] for e in ev if e[1] == "open"]
        closes = [e[0] for e in ev if e[1] == "close"]
        runs, ci = [], 0
        for o in opens:
            nxt = [c for c in closes[ci:] if c > o]
            end = nxt[0] if nxt else (t[-1] if t else o)
            runs.append(end - o)
            ci = closes.index(nxt[0]) + 1 if nxt else len(closes)
        frozen_span = sum(len([r for r in rs if abs(r["t"] / WIN_S - round(r["t"] / WIN_S))
                               < 1e-9]) for _n, rs in ordered(sc[(sess, False)])) * WIN_S
        true_span = sum(rs[0]["dur"] for _n, rs in ordered(sc[(sess, False)]))
        per[sess] = dict(n_iw=len(iw), n_det=n_det, lat=lat, n_dec_iw=sum(len(r) for _n, r in iw),
                         n_dec_rest=len(t), rest_s=span, frozen_rest_s=frozen_span,
                         true_rest_s=true_span,
                         n_fp=len(opens), runs=runs,
                         open_s=float(sum(runs)))
    return per


def summarise(name, per, note=""):
    det = sum(p["n_det"] for p in per.values())
    ntr = sum(p["n_iw"] for p in per.values())
    lat = [x for p in per.values() for x in p["lat"]]
    nfp = sum(p["n_fp"] for p in per.values())
    rs = sum(p["rest_s"] for p in per.values())
    fz = sum(p["frozen_rest_s"] for p in per.values())
    tr = sum(p["true_rest_s"] for p in per.values())
    runs = [x for p in per.values() for x in p["runs"]]
    return dict(row_type="logic_summary", logic=name, note=note,
                n_iw_trials=ntr, n_detected=det, tpr=det / ntr,
                n_dec_iw=sum(p["n_dec_iw"] for p in per.values()),
                dec_per_trial=sum(p["n_dec_iw"] for p in per.values()) / ntr,
                n_dec_rest=sum(p["n_dec_rest"] for p in per.values()),
                rest_min=rs / 60.0, rest_min_frozen_exposure=fz / 60.0,
                rest_min_true=tr / 60.0,
                n_fp_events=nfp, fp_per_min=nfp / (rs / 60.0),
                fp_per_min_frozen_exposure=nfp / (fz / 60.0),
                fp_per_min_true_rest=nfp / (tr / 60.0),
                fp_open_s_total=float(sum(runs)),
                fp_open_frac_of_rest=(float(sum(runs)) / rs if rs else np.nan),
                fp_run_median_s=(float(np.median(runs)) if runs else np.nan),
                fp_run_q1_s=(float(np.percentile(runs, 25)) if runs else np.nan),
                fp_run_q3_s=(float(np.percentile(runs, 75)) if runs else np.nan),
                fp_run_max_s=(float(np.max(runs)) if runs else np.nan),
                latency_median_s=(float(np.median(lat)) if lat else np.nan),
                latency_q1_s=(float(np.percentile(lat, 25)) if lat else np.nan),
                latency_q3_s=(float(np.percentile(lat, 75)) if lat else np.nan),
                latency_min_s=(float(np.min(lat)) if lat else np.nan),
                latency_max_s=(float(np.max(lat)) if lat else np.nan),
                n_latency=len(lat))


def trial_detail(sc, thr, name, iqr_k=None):
    """Per-trial detection + latency for one logic/threshold, for the latency-grid comparison."""
    out = []
    for sess in SESSIONS:
        T = dict(thr[sess])
        if iqr_k is not None:
            T["theta_on"] = T["calib_max"] + iqr_k * T["calib_iqr"]
            T["theta_off"] = min(T["theta_off"], T["theta_on"] - 1e-6)
        for nm, rs in ordered(sc[(sess, True)]):
            t = [r["t"] for r in rs]
            v = [r["s"] for r in rs]
            op = [e[0] for e in logic(name, t, v, T) if e[1] == "open"]
            out.append(dict(session=sess, event=nm, detected=int(bool(op)),
                            latency_s=(op[0] if op else ""), n_dec=len(rs),
                            ev_dur_s=rs[0]["dur"]))
    return out


def main():
    sc, thr = load()
    rows, summ, pers = [], {}, {}

    pg = evaluate(sc, thr, "frozen", four_s_grid=True, frozen_logic=True)
    Sg = summarise("A0_4sgrid_frozenlogic", pg,
                   "degenerate control: dense stream subsampled to the 4 s grid, frozen "
                   "open-1 / close-2-of-3 logic")
    Sg["row_type"] = "degenerate_control"
    rows.append(Sg)

    for name in ("A0", "A2", "A3", "B"):
        per = evaluate(sc, thr, name)
        pers[name] = per
        S = summarise(name, per)
        S["prenamed"] = int(name == "A2")
        summ[name] = S
        rows.append(S)
        for s in SESSIONS:
            p = per[s]
            rows.append(dict(row_type="logic_session", logic=name, session=s,
                             n_iw_trials=p["n_iw"], n_detected=p["n_det"],
                             tpr=p["n_det"] / p["n_iw"], n_dec_iw=p["n_dec_iw"],
                             n_dec_rest=p["n_dec_rest"], rest_min=p["rest_s"] / 60.0,
                             n_fp_events=p["n_fp"], fp_per_min=p["n_fp"] / (p["rest_s"] / 60.0),
                             fp_open_s_total=p["open_s"],
                             latency_median_s=(float(np.median(p["lat"])) if p["lat"]
                                               else np.nan),
                             theta_on=thr[s]["theta_on"], theta_off=thr[s]["theta_off"],
                             lam=thr[s]["lam"], bound=thr[s]["bound"]))
        for s in SESSIONS:
            for i, r in enumerate(per[s]["runs"]):
                rows.append(dict(row_type="fp_run", logic=name, session=s, fp_run_idx=i + 1,
                                 fp_run_s=r))
        for v, c in sorted({x: sum(1 for y in [z for p in per.values() for z in p["lat"]]
                                   if y == x)
                            for x in {z for p in per.values() for z in p["lat"]}}.items()):
            rows.append(dict(row_type="latency_hist", logic=name, latency_s=v, n_trials=c))

    curve = []
    for k in np.round(np.arange(3.0, -1.0001, -0.05), 3):
        pk = evaluate(sc, thr, "frozen", four_s_grid=True, frozen_logic=True, iqr_k=float(k))
        Sk = summarise(f"4sgrid_k{k}", pk)
        curve.append((float(k), Sk["fp_per_min_true_rest"], Sk["tpr"]))
        rows.append(dict(row_type="fourS_threshold_curve", logic="frozen_4s", iqr_k=float(k),
                         fp_per_min_true_rest=Sk["fp_per_min_true_rest"], tpr=Sk["tpr"],
                         n_fp_events=Sk["n_fp_events"], n_detected=Sk["n_detected"]))
    front = []
    for _k, f, t in sorted(curve, key=lambda c: (c[1], c[2])):
        if not front or t > front[-1][1]:
            front.append((f, t))
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    assert all(b >= a for a, b in zip(fx, fx[1:])) and all(b > a for a, b in zip(fy, fy[1:]))
    print(f"[4 s curve] {len(curve)} swept points -> {len(front)} on the Pareto frontier, "
          f"FP {fx[0]:.3f}-{fx[-1]:.3f}/min, TPR {fy[0]:.3f}-{fy[-1]:.3f}", flush=True)

    for n in ("A0", "A2", "A3", "B"):
        S = summ[n]
        fp = S["fp_per_min_true_rest"]
        ref = float(np.interp(fp, fx, fy)) if fx[0] <= fp <= fx[-1] else float("nan")
        rows.append(dict(row_type="curve_comparison", logic=n,
                         session="dense_vs_4s_threshold_curve",
                         fp_per_min_true_rest=fp, tpr=S["tpr"], reference=ref,
                         d_tpr=(S["tpr"] - ref if ref == ref else float("nan")),
                         note="TPR of the dense logic vs the TPR the window-classifier gate reaches at "
                              "the SAME FP by re-thresholding alone; both per minute of true "
                              "rest. > 0 means continuity buys something re-thresholding "
                              "cannot; ~0 means it lands on the existing curve"))
        summ[n]["curve_tpr"] = ref

    a2, a0 = summ["A2"], summ["A0"]

    target = summ_ctrl_fp = None
    ctrl = [r for r in rows if r["row_type"] == "degenerate_control"][0]
    target = ctrl["fp_per_min_true_rest"]
    best = None
    for k in np.round(np.arange(1.5, -1.0001, -0.05), 3):
        pk = evaluate(sc, thr, "A0", iqr_k=float(k))
        Sk = summarise("A0_matched", pk)
        d = abs(Sk["fp_per_min_true_rest"] - target)
        if best is None or d < best[0]:
            best = (d, float(k), Sk)
    dfp, kbest, Sbest = best
    assert dfp <= 0.15, f"no A0 threshold within 0.15/min of {target:.3f} (best {dfp:.3f})"
    det = trial_detail(sc, thr, "A0", iqr_k=kbest)
    lat = [d["latency_s"] for d in det if d["detected"]]
    Sbest = dict(Sbest, row_type="matched_fp_summary", logic="A0_matched_fp", iqr_k=kbest,
                 fp_target_true_rest=target, fp_delta=Sbest["fp_per_min_true_rest"] - target,
                 latency_mean_s=float(np.mean(lat)),
                 note=" "
                      "used only to show latency granularity"
                      )
    rows.append(Sbest)
    for d in det:
        rows.append(dict(row_type="matched_fp_trial", logic="A0_matched_fp", iqr_k=kbest, **d))
    print(f"[FP-matched A0] IQR_K={kbest:+.2f} -> FP {Sbest['fp_per_min_true_rest']:.3f}/min "
          f"vs frozen {target:.3f} (delta {Sbest['fp_delta']:+.3f}), TPR {Sbest['tpr']:.3f}, "
          f"latency median {Sbest['latency_median_s']:.1f}s mean {Sbest['latency_mean_s']:.2f}s",
          flush=True)

    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    head = ["row_type", "logic", "session"]
    cols = head + [c for c in cols if c not in head]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'logic':6s} {'TPR':>6s} {'det/n':>8s} {'FP/min':>7s} {'FPfz':>6s} {'nFP':>4s} "
          f"{'open%rest':>9s} {'runmed':>7s} {'runmax':>7s} {'lat med':>8s} {'lat rng':>9s} "
          f"{'dec/trial':>9s}")
    for n in ("A0", "A2", "A3", "B"):
        S = summ[n]
        print(f"{n:6s} {S['tpr']:.3f} {S['n_detected']:4d}/{S['n_iw_trials']:3d} "
              f"{S['fp_per_min']:7.2f} {S['fp_per_min_frozen_exposure']:6.2f} "
              f"{S['n_fp_events']:4d} {S['fp_open_frac_of_rest'] * 100:8.1f}% "
              f"{S['fp_run_median_s']:7.1f} {S['fp_run_max_s']:7.1f} "
              f"{S['latency_median_s']:7.1f}s "
              f"{str(int(S['latency_min_s'])) + '-' + str(int(S['latency_max_s'])):>9s} "
              f"{S['dec_per_trial']:9.2f}")
    print(f"\nwrote {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
