"""Figure 7 — physics simulation and walk-intent gate

rhythm_metrics — the rhythm-metric battery R2-R6 (+ cadence annex)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os
import sys
import csv
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import biped_model as BM
import data_io as IO
import replay_rig as RR
import analysis_config as ec

LEVEL = "L1p"
PHASE_SRC, LAM = "k1", 0.0
NULLS = {"iscr": IO.scramble, "uscr": IO.uscramble}
R2_BAND = (0.3, 2.5)
R6_TOL = 0.40
ANNEX_THRESH = [0.02, 0.03]

METRICS = [("r1_rate_track_err", "R1 rate-tracking error"),
           ("r2_peak_hz", "R2 contact-rhythm peak (Hz)"),
           ("r3_gated_swing_med_s", "R3 gated swing duration (s)"),
           ("r5_lr_asym", "R5 L/R advancing-step asymmetry"),
           ("r6_rhythmic_frac", "R6 rhythmic-segment time fraction"),
           ("annex_cadence_2cm", "annex cadence @2 cm (steps/s)"),
           ("annex_cadence_3cm", "annex cadence @3 cm (steps/s)")]



CSV_COLS = ["stage", "condition", "phase_src", "lam", "level", "session", "trial_idx", "fold",
            "ordinal", "dur_s", "n_samples", "n_eval", "f0_hz",
            "cadence_hz", "progress_rate", "dx_m", "n_support_transitions",
            "n_steps_max_streak", "fell", "survival_s",
            "chk_streak_1cm", "chk_ntrans",
            "r2_peak_hz", "r2_prominence", "r3_gated_swing_med_s", "r4_step_interval_cv",
            "r5_lr_asym", "r6_rhythmic_frac", "annex_cadence_2cm", "annex_cadence_3cm",
            "n_adv_events", "n_adv_l", "n_adv_r"]


def _epochs(contact):
    """The frozen `_step_streak` epoch construction, verbatim in logic."""
    single = np.where(contact.sum(1) == 1, np.argmax(contact, axis=1), -1)
    single[contact.sum(1) != 1] = -1
    ep = []
    for i, s in enumerate(single):
        if s >= 0 and (not ep or ep[-1][0] != s):
            ep.append((int(s), int(i)))
    return single, ep


def _transitions(contact, fx, thresh=RR.STEP_MIN_ADVANCE):
    """[(idx, foot_in, foot_out, advancing)] for every consecutive epoch pair."""
    _, ep = _epochs(contact)
    out = []
    for a, b in zip(ep, ep[1:]):
        adv = fx[b[1], b[0]] - fx[b[1], a[0]]
        out.append((b[1], b[0], a[0], bool(adv >= thresh)))
    return out


def _streak(contact, fx, thresh=RR.STEP_MIN_ADVANCE):
    """The frozen streak / transition count, parameterised only in the advance threshold."""
    best = cur = 0
    tr = _transitions(contact, fx, thresh)
    for _, _, _, adv in tr:
        if adv:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best, len(tr)


def rhythm_metrics(contact, fx, dur_s):
    dt = BM.DT
    n_eval = len(contact)
    out = {}

    s = contact[:, 0].astype(float) - contact[:, 1].astype(float)
    s = s - s.mean()
    if n_eval >= 8 and np.any(s != 0):
        w = np.hanning(n_eval)
        F = np.fft.rfft(s * w)
        fr = np.fft.rfftfreq(n_eval, dt)
        pw = np.abs(F) ** 2
        band = (fr >= R2_BAND[0]) & (fr <= R2_BAND[1])
        if band.any():
            pb = pw[band]
            out["r2_peak_hz"] = float(fr[band][int(np.argmax(pb))])
            out["r2_prominence"] = float(pb.max() / max(pb.mean(), 1e-30))
        else:
            out["r2_peak_hz"] = out["r2_prominence"] = float("nan")
    else:
        out["r2_peak_hz"] = out["r2_prominence"] = float("nan")

    single, ep = _epochs(contact)
    tr = _transitions(contact, fx)

    dur = []
    for (a, b, t) in zip(ep, ep[1:], tr):
        if not t[3]:
            continue
        seg = single[a[1]:b[1]]
        dur.append(float(np.count_nonzero(seg == a[0]) * dt))
    out["r3_gated_swing_med_s"] = float(np.median(dur)) if dur else float("nan")

    adv = [(i * dt, foot_in, pos) for pos, (i, foot_in, _, a) in enumerate(tr) if a]
    times = np.array([a[0] for a in adv], float)
    out["n_adv_events"] = len(adv)
    out["n_adv_l"] = sum(1 for a in adv if a[1] == 0)
    out["n_adv_r"] = sum(1 for a in adv if a[1] == 1)

    if len(adv) >= 3:
        iv = np.diff(times)
        out["r4_step_interval_cv"] = float(iv.std(ddof=1) / iv.mean()) if iv.mean() > 0 \
            else float("nan")
    else:
        out["r4_step_interval_cv"] = float("nan")

    nL, nR = out["n_adv_l"], out["n_adv_r"]
    out["r5_lr_asym"] = abs(nL - nR) / (nL + nR) if (nL + nR) else float("nan")

    frac = 0.0
    if len(adv) >= 3:
        med = float(np.median(np.diff(times)))
        lo, hi = med * (1 - R6_TOL), med * (1 + R6_TOL)
        chains, cur = [], [adv[0]]
        for prev, nxt in zip(adv, adv[1:]):
            if nxt[2] == prev[2] + 1:
                cur.append(nxt)
            else:
                chains.append(cur)
                cur = [nxt]
        chains.append(cur)
        total = 0.0
        for ch in chains:
            if len(ch) < 3:
                continue
            t = np.array([c[0] for c in ch], float)
            ok = (np.diff(t) >= lo) & (np.diff(t) <= hi)
            i = 0
            while i < len(ok):
                if not ok[i]:
                    i += 1
                    continue
                j = i
                while j < len(ok) and ok[j]:
                    j += 1
                if (j - i) >= 2:
                    total += float(t[j] - t[i])
                i = j
        frac = total / max(n_eval * dt, 1e-12)
    out["r6_rhythmic_frac"] = float(frac)

    b1, nt1 = _streak(contact, fx)
    out["chk_streak_1cm"], out["chk_ntrans"] = int(b1), int(nt1)
    for th in ANNEX_THRESH:
        b, _ = _streak(contact, fx, th)
        out[f"annex_cadence_{int(round(th * 100))}cm"] = float(b) / dur_s
    return out


def _job(J):
    t, sp, tg, tn = J["trial"], J["setpoint"], J["tmpl_g"], J["tmpl_n"]
    Y = t["Y"]
    f0 = IO.f0_of(Y)
    if IO.F7_FROM2S:
        W, Y = IO.f7_from2s_inputs(IO.F7_FROM2S_COND[J["condition"]], t)
        tgt = IO.f5_targets(Y, tg, tn, J["delta"], f0, PHASE_SRC, LAM, sp, J["session"], warm=W)
    else:
        if J["null"]:
            Y = NULLS[J["null"]](Y, seed=t["trial_idx"])
        tgt = IO.f5_targets(Y, tg, tn, J["delta"], f0, PHASE_SRC, LAM, sp, J["session"])
    r = RR.rollout(tgt, LEVEL)
    dur = t["dur_s"]
    row = dict(stage=J["stage"], condition=J["condition"], phase_src=PHASE_SRC, lam=LAM, level=LEVEL,
               session=J["session"], trial_idx=t["trial_idx"], fold=t["fold"],
               ordinal=t["ordinal"], dur_s=dur, n_samples=t["n"],
               n_eval=r["n_eval_steps"], f0_hz=f0,
               cadence_hz=r["n_steps_max_streak"] / dur, progress_rate=r["dx_m"] / dur,
               dx_m=r["dx_m"], n_support_transitions=r["n_support_transitions"],
               n_steps_max_streak=r["n_steps_max_streak"], fell=r["fell"],
               survival_s=r["survival_s"])
    row.update(rhythm_metrics(r["contact_lr"], r["foot_x"], dur))
    return row


def replay(args):
    sessions = [s for s in args.sessions.split(",") if s]
    tmpl_n = IO.fukuchi_template()
    jobs = []
    for s in sessions:
        sp = IO.device_stance_ankle(IO.loso_setpoint(args.npz, s)[0])
        tg = IO.loso_template(args.npz, s)
        delta, _ = IO.global_delta(tg, tmpl_n)
        ctx = dict(setpoint=sp, tmpl_g=tg, tmpl_n=tmpl_n, delta=delta, session=s)
        for t in ([] if IO.F7_FROM2S else IO.load_session_trials(args.dump, s, stream="Y_true")):
            jobs.append(dict(stage="calib", condition="wphase", trial=t, null=None, **ctx))
        for t in IO.load_session_trials(args.dump, s, stream="C_pred"):
            for condition, nl in (("dphase", None), ("dphase_iscr", "iscr"),
                            ("dphase_uscr", "uscr")):
                jobs.append(dict(stage="confirm", condition=condition, trial=t, null=nl, **ctx))

    rows = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(min(args.workers, len(jobs))) as pool:
            for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
                rows.append(row)
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(jobs)}", flush=True)
    else:
        for j in jobs:
            rows.append(_job(j))

    rows.sort(key=lambda r: (r["stage"], r["condition"], r["session"], r["trial_idx"]))
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "rhythm_metrics.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    bad = [r for r in rows if r["chk_streak_1cm"] != r["n_steps_max_streak"]
           or r["chk_ntrans"] != r["n_support_transitions"]]
    print(f"wrote {p}: {len(rows)} rows; internal consistency mismatches: {len(bad)}",
          flush=True)


def _fits(df, vc):
    import numpy as _np
    import statsmodels.formula.api as smf
    fits = []
    for meth in ("lbfgs", "powell", "cg"):
        kw = dict(groups=df["session"], re_formula="~1")
        if vc:
            kw["vc_formula"] = {"trial": "0 + C(trial)"}
        m = smf.mixedlm("y ~ armc", df, **kw).fit(reml=True, method=meth, maxiter=2000)
        if m.converged and _np.isfinite(m.bse["armc"]) and _np.isfinite(m.llf):
            fits.append((float(m.llf), (float(m.params["armc"]), float(m.bse["armc"]),
                                        float(m.pvalues["armc"]), meth)))
    if not fits:
        raise RuntimeError("LME did not converge")
    return max(fits, key=lambda t: t[0])[1]


def lme_p(rows_a, rows_b, key):
    """Paired LME, y ~ condition + (1|session) + (1|trial); pairwise-complete."""
    import pandas as pd
    A = {(r["session"], r["trial_idx"]): float(r[key]) for r in rows_a}
    B = {(r["session"], r["trial_idx"]): float(r[key]) for r in rows_b}
    recs = []
    n = 0
    for k in sorted(set(A) & set(B)):
        if not (np.isfinite(A[k]) and np.isfinite(B[k])):
            continue
        n += 1
        tid = f"{k[0]}_{k[1]}"
        recs.append(dict(y=A[k], armc=1.0, session=k[0], trial=tid))
        recs.append(dict(y=B[k], armc=0.0, session=k[0], trial=tid))
    if n < 3:
        return (float("nan"),) * 3 + ("n/a", n)
    return _fits(pd.DataFrame(recs), True) + (n,)


def lme_unpaired_p(rows_a, rows_b, key):
    """Unpaired LME, y ~ condition + (1|session)."""
    import pandas as pd
    a = [float(r[key]) for r in rows_a]
    b = [float(r[key]) for r in rows_b]
    recs = [dict(y=v, armc=1.0, session=r["session"])
            for v, r in zip(a, rows_a) if np.isfinite(v)] + \
           [dict(y=v, armc=0.0, session=r["session"])
            for v, r in zip(b, rows_b) if np.isfinite(v)]
    na = sum(1 for v in a if np.isfinite(v))
    nb = sum(1 for v in b if np.isfinite(v))
    if na < 3 or nb < 3:
        return (float("nan"),) * 3 + ("n/a", na, nb)
    return _fits(pd.DataFrame(recs), False) + (na, nb)


def _wil(rows_a, rows_b, key):
    from scipy.stats import wilcoxon
    A = {(r["session"], r["trial_idx"]): float(r[key]) for r in rows_a}
    B = {(r["session"], r["trial_idx"]): float(r[key]) for r in rows_b}
    a, b = [], []
    for k in sorted(set(A) & set(B)):
        if np.isfinite(A[k]) and np.isfinite(B[k]):
            a.append(A[k])
            b.append(B[k])
    a, b = np.array(a), np.array(b)
    if len(a) < 3 or not np.any(a - b != 0):
        return len(a), float("nan")
    return len(a), float(wilcoxon(a, b).pvalue)


def _med(rows, key):
    v = np.array([float(r[key]) for r in rows], float)
    v = v[np.isfinite(v)]
    return (float(np.median(v)), len(v)) if len(v) else (float("nan"), 0)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="replay", choices=["replay"])
    ap.add_argument("--dump", default=str(ec.path("p_dump_dir")))
    ap.add_argument("--npz", default=str(ec.path("p_cycavg")))
    ap.add_argument("--out", default=os.path.join(HERE, "result"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sessions", default=",".join(IO.G0_SESSIONS))
    args = ap.parse_args()
    args.out = IO.build_id_out(args.out)
    if args.mode == "replay":
        IO.build_id_inputs(dump=args.dump, npz=args.npz)
        if not args.dump or not args.npz:
            raise SystemExit("--dump and --npz are required in replay mode")
        print(BM.rig_report(), flush=True)
        replay(args)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
