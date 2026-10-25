"""Figure 7 — physics simulation and walk-intent gate

gate_transition — gate response across imagined-walking to rest transitions

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import csv
import os
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gate_features import SESSIONS, bandpower, build, load_session, windows
from gate_fusion import (BASE_MIN_S, LAM_RA, RES, ROOT, WIN_S, band_covs,
                                  cov_inventory, fit_ts, ra_feat, session_layout, whiteners)

HOP_S, WARMUP_S, DWELL_S, N_OFF = 1.0, 4.0, 8.0, 3
IQR_K, OFF_Q = 0.25, 90
P1 = [s for s in SESSIONS if s.startswith("pat01")]
ANCHOR = dict(start_rate=0.635, n_start=169, n_pairs=266, closure_evidence=0.778, km_median=6.66)


def score_span(rows_bp, keys, covs, calib):
    y = np.array([r["label"] for r in rows_bp])
    Xbp = np.vstack([r["corrected"] for r in rows_bp])
    sess = np.array([r["session"] for r in rows_bp])
    whit = {s: whiteners(calib[s], LAM_RA) for s in SESSIONS}
    Xra = np.vstack([ra_feat(c, whit[k[0]], LAM_RA) for c, k in zip(covs, keys)])
    Xcal = {s: np.vstack([ra_feat(c, whit[s], LAM_RA) for c in calib[s]]) for s in SESSIONS}

    out, thr = [], {}
    for s in SESSIONS:
        te, tr = sess == s, sess != s
        sa = StandardScaler().fit(Xbp[tr])
        cla = LDA(solver="lsqr", shrinkage="auto").fit(sa.transform(Xbp[tr]), y[tr])
        sb = StandardScaler().fit(Xra[tr])
        clb, _C = fit_ts(sb.transform(Xra[tr]), y[tr], sess[tr])

        X, fs, cal, _ = session_layout(ROOT, s, BASE_MIN_S)
        bw = np.vstack([bandpower(w, fs) for e in cal for w in windows(X, e["a"], e["b"], fs)])
        P_base = np.median(bw, axis=0)
        cA = cla.decision_function(sa.transform(np.log10((bw + 1e-20) / (P_base + 1e-20))))
        cB = clb.decision_function(sb.transform(Xcal[s]))
        sdA = float(cla.decision_function(sa.transform(Xbp[tr])).std())
        sdB = float(clb.decision_function(sb.transform(Xra[tr])).std())
        mA, mB = float(np.median(cA)), float(np.median(cB))
        fc = ((cA - mA) / sdA + (cB - mB) / sdB) / 2
        iqrF = float(np.percentile(fc, 75) - np.percentile(fc, 25))
        iqrL = float(np.percentile(cA, 75) - np.percentile(cA, 25))
        thr[s] = dict(fuse_on=float(fc.max() + IQR_K * iqrF),
                      fuse_off=float(np.percentile(fc, OFF_Q)),
                      lda_on=float(cA.max() + IQR_K * iqrL),
                      lda_off=float(np.percentile(cA, OFF_Q)))

        _X, fs, _ch, ev = load_session(ROOT, s)
        calib_a = {e["a"] for e in cal}
        n_win, hop = int(round(WIN_S * fs)), int(round(HOP_S * fs))
        iw_evs = [e for e in ev if e["code"] == "gaitimagery"]
        pid = 0
        for k_iw, e in enumerate(iw_evs):
            nxt = [x for x in ev if x["a"] > e["a"]]
            if not nxt or nxt[0]["code"] != "rest":
                continue
            r = nxt[0]
            pid += 1
            s0 = e["a"] - int(round(WARMUP_S * fs))
            if s0 < 0:
                continue
            meta, fb, fr = [], [], []
            ss, m = s0, 0
            while ss + n_win <= min(r["b"], len(X)):
                t_abs = ss + n_win
                t_rel = (t_abs - e["a"]) / fs
                zone = ("pre" if t_abs <= e["a"] else "iw" if t_abs <= e["b"] else
                        "gap" if t_abs <= r["a"] else "rest")
                w = X[ss:ss + n_win]
                fb.append(np.log10((bandpower(w, fs) + 1e-20) / (P_base + 1e-20)))
                fr.append(ra_feat(band_covs(w), whit[s], LAM_RA))
                meta.append((m, round(t_rel, 4), zone))
                ss += hop
                m += 1
            if not meta:
                continue
            gA = cla.decision_function(sa.transform(np.vstack(fb)))
            gB = clb.decision_function(sb.transform(np.vstack(fr)))
            for (m_i, t_rel, zone), va, vb in zip(meta, gA, gB):
                out.append(dict(pair_id=f"{s}|p{pid:03d}", session=s, iw_event=f"iw{k_iw}",
                                rest_is_calibration=int(r["a"] in calib_a), m=m_i,
                                t_rel_iw_s=t_rel, zone=zone, score_lda=float(va),
                                score_fuse=float(((va - mA) / sdA + (vb - mB) / sdB) / 2),
                                gap_s=round((r["a"] - e["b"]) / fs, 4),
                                iw_end_rel_s=round((e["b"] - e["a"]) / fs, 4),
                                rest_start_rel_s=round((r["a"] - e["a"]) / fs, 4),
                                rest_end_rel_s=round((r["b"] - e["a"]) / fs, 4)))
        print(f"  {s}: {pid} pairs, {len(out)} decisions cumulative", flush=True)
    return out, thr


def label_pure(dec):
    iw_end, ra, rb = dec[0]["iw_end"], dec[0]["rest_a"], dec[0]["rest_b"]
    iw = [d for d in dec if d["t"] - WIN_S >= -1e-9 and d["t"] <= iw_end + 1e-9]
    rest = [d for d in dec if d["t"] - WIN_S >= ra - 1e-9 and d["t"] <= rb + 1e-9]
    return iw, rest


def run_clean(iw, rest, key, th_on, th_off):
    cur, below, t_open, ev = 0, 0, None, []
    for phase, seq in (("iw", iw), ("rest", rest)):
        if phase == "rest":
            below = 0
        for d in seq:
            if cur == 0:
                if d[key] > th_on:
                    cur, t_open, below = 1, d["t"], 0
                    ev.append((d["t"], "open", phase))
            else:
                below = below + 1 if d[key] < th_off else 0
                if below >= N_OFF and (d["t"] - t_open) >= DWELL_S:
                    cur, below = 0, 0
                    ev.append((d["t"], "close", phase))
    return ev


def km(times, events):
    order = np.argsort(times)
    t, e = np.asarray(times)[order], np.asarray(events)[order]
    S, grid, surv = 1.0, [], []
    for u in np.unique(t[e == 1]):
        at_risk = int((t >= u - 1e-9).sum())
        d = int(((t == u) & (e == 1)).sum())
        if at_risk:
            S *= (1 - d / at_risk)
        grid.append(u)
        surv.append(S)
    return np.array(grid), np.array(surv), len(t)


def quantile(grid, surv, q):
    idx = np.where(surv <= 1 - q + 1e-12)[0]
    return float(grid[idx[0]]) if len(idx) else None


def analyse(span, thr, key, on_k, off_k, condition, rows):
    pairs = {}
    for r in span:
        pairs.setdefault(r["pair_id"], []).append(
            dict(t=r["t_rel_iw_s"], score_lda=r["score_lda"], score_fuse=r["score_fuse"],
                 session=r["session"], cal=r["rest_is_calibration"],
                 iw_end=r["iw_end_rel_s"], rest_a=r["rest_start_rel_s"],
                 rest_b=r["rest_end_rel_s"]))
    for v in pairs.values():
        v.sort(key=lambda d: d["t"])

    per = {s: dict(n=0, start=0, openrest=0, stop=0, ev=[], t=[], e=[], lat=[]) for s in SESSIONS}
    n_noeval = 0
    for pid, dec in sorted(pairs.items()):
        s = dec[0]["session"]
        th_on, th_off = thr[s][on_k], thr[s][off_k]
        iw, rest = label_pure(dec)
        ra, rb = dec[0]["rest_a"], dec[0]["rest_b"]
        ev = run_clean(iw, rest, key, th_on, th_off)
        start = any(k == "open" and z == "iw" for _t, k, z in ev)
        state, t_open_at_rest = 0, None
        for t, k, _z in ev:
            if t <= ra:
                state = 1 if k == "open" else 0
                if k == "open":
                    t_open_at_rest = t
        open_at_rest = state == 1
        closes_rest = [t for t, k, z in ev if k == "close" and z == "rest"]
        stop = open_at_rest and bool(closes_rest)
        lat = (closes_rest[0] - ra) if stop else None

        p = per[s]
        p["n"] += 1
        p["start"] += int(start)
        p["openrest"] += int(open_at_rest)
        p["stop"] += int(stop)
        if lat is not None:
            p["lat"].append(lat)
        if start:
            if rest:
                allb = all(d[key] < th_off for d in rest)
                p["ev"].append(allb)
            else:
                n_noeval += 1
            p["t"].append(lat if stop else (rb - ra))
            p["e"].append(1 if stop else 0)
        rows.append(dict(row_type="pair", condition=condition, session=s, pair_id=pid,
                         n_dec_iw=len(iw), n_dec_rest=len(rest), start=int(start),
                         open_at_rest_onset=int(open_at_rest), stop=int(stop),
                         closing_latency_s=(lat if lat is not None else ""),
                         rest_dur_s=round(rb - ra, 3), rest_is_calibration=dec[0]["cal"]))

    def summarise(name, ss):
        n = sum(per[s]["n"] for s in ss)
        st = sum(per[s]["start"] for s in ss)
        orz = sum(per[s]["openrest"] for s in ss)
        sp = sum(per[s]["stop"] for s in ss)
        evs = [x for s in ss for x in per[s]["ev"]]
        ts = [x for s in ss for x in per[s]["t"]]
        es = [x for s in ss for x in per[s]["e"]]
        lat = [x for s in ss for x in per[s]["lat"]]
        g, S, nkm = km(ts, es) if ts else (np.array([]), np.array([]), 0)
        med = quantile(g, S, 0.5) if len(g) else None
        return dict(row_type="summary", condition=condition, session=name, n_pairs=n, n_start=st,
                    start_rate=st / n if n else np.nan, n_open_at_rest=orz, n_stop=sp,
                    stop_rate_of_start=(sp / st if st else np.nan),
                    n_evaluable=len(evs), n_all_below=int(sum(evs)),
                    closure_evidence_rate=(sum(evs) / len(evs) if evs else np.nan),
                    n_km=nkm, n_events=int(sum(es)), n_censored=int(len(es) - sum(es)),
                    km_median_s=(med if med is not None else ""),
                    km_S_at_last_event=(float(S[-1]) if len(S) else np.nan),
                    latency_median_s=(float(np.median(lat)) if lat else np.nan),
                    latency_q1_s=(float(np.percentile(lat, 25)) if lat else np.nan),
                    latency_q3_s=(float(np.percentile(lat, 75)) if lat else np.nan),
                    n_latency=len(lat), n_not_evaluable=n_noeval)

    for s in SESSIONS:
        rows.append(summarise(s, [s]))
    rows.append(summarise("P1", P1))
    rows.append(summarise("P2", [s for s in SESSIONS if s not in P1]))
    A = summarise("ALL", SESSIONS)
    rows.append(A)
    return A


def main():
    print("== building demobase rows + covariance inventory ==", flush=True)
    rows_bp, _m = build(ROOT, base_min_s=BASE_MIN_S)
    keys, covs, calib = cov_inventory(ROOT, BASE_MIN_S)
    assert [(r["session"], r["event"], r["win"]) for r in rows_bp] == keys
    print("== scoring the continuous IW->rest spans ==", flush=True)
    span, thr = score_span(rows_bp, keys, covs, calib)

    frozen = {(r["session"], r["event"], round(float(r["t_end_ev_s"]), 4)): float(r["fuse_V1"])
              for r in csv.DictReader(open(os.path.join(RES, "gate_fused_scores.csv"),
                                           encoding="utf-8"))}
    n_chk, worst = 0, 0.0
    for r in span:
        if r["m"] >= 4:
            key = (r["session"], r["iw_event"], round((r["m"] - 4) * HOP_S + WIN_S, 4))
            if key in frozen:
                worst = max(worst, abs(r["score_fuse"] - frozen[key]))
                n_chk += 1
    assert n_chk > 0 and worst < 1e-9, f"span does not reproduce the P-F fused stream ({worst:.2e})"
    print(f"[GATE] continuous span vs P-F dense fused stream: n={n_chk}, "
          f"max |delta| {worst:.3e} OK", flush=True)

    rows = []
    AL = analyse(span, thr, "score_lda", "lda_on", "lda_off", "lda_densegrid", rows)
    AF = analyse(span, thr, "score_fuse", "fuse_on", "fuse_off", "fuse2_V1", rows)

    assert AL["n_pairs"] == ANCHOR["n_pairs"], f"pairs {AL['n_pairs']} != {ANCHOR['n_pairs']}"
    assert AL["n_start"] == ANCHOR["n_start"], f"START {AL['n_start']} != {ANCHOR['n_start']}"
    assert abs(AL["closure_evidence_rate"] - ANCHOR["closure_evidence"]) < 1e-3
    assert abs(float(AL["km_median_s"]) - ANCHOR["km_median"]) < 1e-2
    print(f"[GATE] LDA transition comparator reproduced: START {AL['start_rate']:.3f} "
          f"({AL['n_start']}/{AL['n_pairs']}), closure evidence "
          f"{AL['closure_evidence_rate']:.3f}, KM median {AL['km_median_s']}s OK", flush=True)

    for k in ("start_rate", "closure_evidence_rate", "stop_rate_of_start", "km_median_s",
              "latency_median_s"):
        a, b = AL[k], AF[k]
        rows.append(dict(row_type="comparison", condition="fuse2_V1_vs_lda", session="ALL", metric=k,
                         lda=a, fuse2=b,
                         delta=((float(b) - float(a)) if a != "" and b != "" else "")))
        print(f"  [{k:24s}] LDA {a}  ->  fuse2 {b}", flush=True)

    for name, data in (("gate_transition_scores.csv", span),
                       ("gate_transition.csv", rows)):
        cols, seen = [], set()
        for r in data:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
        with open(os.path.join(RES, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(data)
        print(f"wrote {name}: {len(data)} rows", flush=True)


if __name__ == "__main__":
    main()
