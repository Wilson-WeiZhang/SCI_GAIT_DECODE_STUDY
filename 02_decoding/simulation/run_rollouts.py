"""Figure 7 — physics simulation and walk-intent gate

Figure 7 - physics simulation rollouts

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os
import sys
import csv
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import biped_model as BM
import data_io as IO
import replay_rig as RR
import analysis_config as ec

RUN_LEVEL = "L1p"
R0_LEVELS = [RUN_LEVEL, "L0"]
R0_HOLD_S, R0_MAX_DRIFT = 8.0, 0.05
ALT_GAINS = [1.0, 2.0, 3.5]
METRONOME_HZ = 0.90
HICKS_RESIDUAL_NM = 75.0
PHASE_SOURCES = ["k1", "k2", "k12"]
LAMBDAS = [0.0, 0.15, 0.30, 0.45, 0.60]
if ec.F7_SIM:
    PHASE_SOURCES, LAMBDAS = ["k1"], [0.0]
NULL_DEAD_ABS = 0.15
NULL_DEAD_FRAC = 0.25
NULLS = {"iscr": IO.scramble, "uscr": IO.uscramble}

CSV_COLS = ["stage", "rung", "condition", "src_stream", "phase_src", "lam", "gain_mode", "level",
            "session", "trial_idx", "fold", "ordinal", "dur_s", "n_samples",
            "cadence_hz", "progress_rate", "passed", "fell", "t_fall", "survival_s",
            "n_steps_max_streak", "n_support_transitions", "dx_m", "clamp_frac",
            "tau_int", "tau_mean", "tau_peak", "frac_double_support", "frac_flight",
            "body_ground", "stand_height_m", "pitch_after_settle_deg", "final_pitch_deg",
            "final_hip_z_m", "fidelity_r", "f0_hz", "delta_cycle", "gain_vec"]


def _job(J):
    """J is a plain dict so the job schema stays readable as conditions multiply."""
    t, sp, tg, tn = J["trial"], J["setpoint"], J["tmpl_g"], J["tmpl_n"]
    gain = np.ones(6)
    f0 = float("nan")
    if J["condition"] == "r0":
        tgt = np.radians(np.tile(sp * BM.SIGN, (t["n"], 1)))
    elif J["condition"].startswith("metron"):
        tgt = IO.to_targets(IO.metronome(t["n"], tn, J["delta"], METRONOME_HZ), sp)
    else:
        Y = t["Y"]
        f0 = IO.f0_of(Y)
        from2s = IO.F7_FROM2S and J["rung"] == "f5n"
        if J["null"] and not from2s:
            Y = NULLS[J["null"]](Y, seed=t["trial_idx"])
        if from2s:
            W, Y = IO.f7_from2s_inputs(IO.F7_FROM2S_COND[J["condition"]], t)
            tgt = IO.f5_targets(Y, tg, tn, J["delta"], f0, J["phase_src"], J["lam"], sp, J["session"], warm=W)
        elif J["rung"] == "f5n":
            tgt = IO.f5_targets(Y, tg, tn, J["delta"], f0, J["phase_src"], J["lam"], sp,
                                J["session"])
        else:
            gain = (IO.loso_gain(J["npz"], J["session"], Y) if J["gain_mode"] == "loso"
                    else np.full(6, float(J["gain_mode"][1:])))
            tgt = IO.to_targets(Y, sp, gain=gain)
    r = RR.rollout(tgt, J["level"])
    dur = t["dur_s"]
    row = {k: r.get(k) for k in CSV_COLS if k in r}
    row.update(stage=J["stage"], rung=J["rung"], condition=J["condition"], src_stream=J["src_stream"],
               phase_src=J["phase_src"], lam=J["lam"], gain_mode=J["gain_mode"],
               level=J["level"], session=J["session"], trial_idx=t["trial_idx"],
               fold=t["fold"], ordinal=t["ordinal"], dur_s=dur, n_samples=t["n"],
               cadence_hz=r["n_steps_max_streak"] / dur, progress_rate=r["dx_m"] / dur,
               tau_mean=r["tau_int"] / max(r["survival_s"], 1e-9),
               fidelity_r=J["fid"] if J["fid"] is not None else "",
               f0_hz=f0, delta_cycle=J["delta"],
               gain_vec="|".join(f"{g:.3f}" for g in np.atleast_1d(gain)))
    if J["condition"] == "r0":
        row["passed"] = int((not r["fell"]) and abs(r["dx_m"]) < R0_MAX_DRIFT)
    keep = J.get("keep_trace", False)
    return row, ((r["trace"], r["site_order"]) if keep else None)


def _mk(stage, rung, condition, src_stream, session, trial, ctx, phase_src="n/a", lam=0.0,
        gain_mode="n/a", null=None, level=RUN_LEVEL, fid=None, keep_trace=False):
    return dict(stage=stage, rung=rung, condition=condition, src_stream=src_stream, session=session,
                trial=trial, phase_src=phase_src, lam=lam, gain_mode=gain_mode, null=null,
                level=level, fid=fid, keep_trace=keep_trace, **ctx)


def _run(jobs, workers, label):
    print(f"\n{label}: {len(jobs)} rollouts", flush=True)
    rows, traces = [], {}
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(min(workers, len(jobs))) as pool:
            for i, (row, tr) in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
                rows.append(row)
                if tr is not None:
                    traces[_key(row)] = tr
                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1}/{len(jobs)}", flush=True)
    else:
        for j in jobs:
            row, tr = _job(j)
            rows.append(row)
            if tr is not None:
                traces[_key(row)] = tr
    return rows, traces


def _key(r):
    return (r["condition"], r["phase_src"], r["lam"], r["gain_mode"], r["session"], r["trial_idx"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ec.path("p_dump_dir")))
    ap.add_argument("--rest-dump", default="")
    ap.add_argument("--npz", default=str(ec.path("p_cycavg")))
    ap.add_argument("--out", default=os.path.join(HERE, "result"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sessions", default=",".join(IO.G0_SESSIONS))
    args = ap.parse_args()
    IO.build_id_inputs(dump=args.dump, rest_dump=args.rest_dump, npz=args.npz)
    args.out = IO.build_id_out(args.out)

    os.makedirs(args.out, exist_ok=True)
    strip_dir = os.path.join(args.out, "filmstrip")
    os.makedirs(strip_dir, exist_ok=True)
    sessions = [s for s in args.sessions.split(",") if s]

    print(BM.rig_report(), flush=True)
    tmpl_n = IO.fukuchi_template()
    ctxs, w_tr, d_tr, r_tr, fid, rfid, inv = {}, {}, {}, {}, {}, {}, {}
    for s in sessions:
        sp_raw, keep = IO.loso_setpoint(args.npz, s)
        sp = IO.device_stance_ankle(sp_raw)
        tg = IO.loso_template(args.npz, s)
        delta, cc = IO.global_delta(tg, tmpl_n)
        ctxs[s] = dict(setpoint=sp, tmpl_g=tg, tmpl_n=tmpl_n, delta=delta, npz=args.npz)
        w_tr[s] = IO.load_session_trials(args.dump, s, stream="Y_true")
        d_tr[s] = IO.load_session_trials(args.dump, s, stream="C_pred")
        fid[s] = IO.fidelity_by_trial(w_tr[s], d_tr[s])
        r_tr[s], rfid[s] = [], {}
        if args.rest_dump:
            rw = IO.load_session_trials(args.rest_dump, s, stream="Y_true")
            r_tr[s] = IO.load_session_trials(args.rest_dump, s, stream="C_pred")
            rfid[s] = IO.fidelity_by_trial(rw, r_tr[s])
        fv = list(fid[s].values())
        rv = list(rfid[s].values())
        inv[s] = dict(n_walker=len(w_tr[s]), n_decoded=len(d_tr[s]), n_rest=len(r_tr[s]),
                      loso=keep, delta_cycle=round(delta, 4), delta_corr=round(cc, 4),
                      walker_dur_s=[round(t["dur_s"], 2) for t in w_tr[s]],
                      decoded_dur_s=[round(t["dur_s"], 2) for t in d_tr[s]],
                      fidelity_r_median=round(float(np.median(fv)), 4) if fv else None,
                      rest_fidelity_r_median=round(float(np.median(rv)), 4) if rv else None)
        print(f"{s}: walker {len(w_tr[s])}, decoded {len(d_tr[s])}, rest {len(r_tr[s])}; "
              f"delta {delta:.2f} (corr {cc:.2f}); fidelity r med "
              f"{(np.median(fv) if fv else float('nan')):.3f} | rest "
              f"{(np.median(rv) if rv else float('nan')):.3f}", flush=True)

    n_hold = int(round(R0_HOLD_S * IO.FS))
    hold = dict(trial_idx=-1, fold=-1, ordinal=-1, n=n_hold, dur_s=R0_HOLD_S)
    r0_jobs = [_mk("R0", "R0", "r0", "static", s, hold, ctxs[s], level=lv)
               for s in sessions for lv in R0_LEVELS]
    r0_rows, _ = _run(r0_jobs, args.workers, f"R0' rig validity ({R0_HOLD_S:.0f} s)")
    gated = [r for r in r0_rows if r["level"] == RUN_LEVEL]
    r0_fail = [r for r in gated if not r["passed"]]
    print(f"R0': {len(gated) - len(r0_fail)}/{len(gated)} at {RUN_LEVEL}"
          + ("" if not r0_fail else "  -> R0' FAIL, gates not run"), flush=True)
    if r0_fail:
        _write(r0_rows, [], inv, sessions, args.out, None, True)
        return

    cal_jobs = []
    for s in sessions:
        for t in ([] if IO.F7_FROM2S else w_tr[s]):
            f = fid[s].get(t["trial_idx"])
            for src in PHASE_SOURCES:
                for lam in LAMBDAS:
                    for nm, nl in (("wphase", None), ("wphase_iscr", "iscr"),
                                   ("wphase_uscr", "uscr")):
                        cal_jobs.append(_mk("calib", "f5n", nm, "walker", s, t, ctxs[s],
                                            phase_src=src, lam=lam, null=nl, fid=f,
                                            keep_trace=(src == "k1" and lam == 0.0)))
    cal_rows, cal_traces = _run(cal_jobs, args.workers,
                                "stage A — BLIND calibration on walker phase conditions only")
    best, grid = _select(cal_rows)
    src_star, lam_star = best
    print(f"\nblind winner: phase source {src_star}, lambda {lam_star}", flush=True)

    combos = [("k1", 0.0)] + ([best] if best != ("k1", 0.0) else [])
    jobs = []
    for s in sessions:
        C = ctxs[s]
        for t in w_tr[s]:
            f = fid[s].get(t["trial_idx"])
            jobs.append(_mk("confirm", "f1", "walker", "walker", s, t, C,
                            gain_mode="g1.0", fid=f, keep_trace=True))
            jobs.append(_mk("confirm", "f5n", "metron_w", "metronome", s, t, C,
                            phase_src=f"metronome_{METRONOME_HZ}", fid=f, keep_trace=True))
        for t in d_tr[s]:
            f = fid[s].get(t["trial_idx"])
            jobs.append(_mk("confirm", "f1", "decoded", "decoded", s, t, C,
                            gain_mode="loso", fid=f, keep_trace=True))
            for g in ALT_GAINS:
                jobs.append(_mk("confirm", "f1", "decoded", "decoded", s, t, C,
                                gain_mode=f"g{g}", fid=f))
            jobs.append(_mk("confirm", "f1", "decoded_iscr", "decoded", s, t, C,
                            gain_mode="loso", null="iscr", fid=f))
            jobs.append(_mk("confirm", "f5n", "metron_d", "metronome", s, t, C,
                            phase_src=f"metronome_{METRONOME_HZ}", fid=f))
            for src, lam in combos:
                for nm, nl in (("dphase", None), ("dphase_iscr", "iscr"),
                               ("dphase_uscr", "uscr")):
                    jobs.append(_mk("confirm", "f5n", nm, "decoded", s, t, C,
                                    phase_src=src, lam=lam, null=nl, fid=f,
                                    keep_trace=(nm == "dphase")))
        for t in r_tr[s]:
            f = rfid[s].get(t["trial_idx"])
            jobs.append(_mk("confirm", "f1", "rest_f1", "rest", s, t, C,
                            gain_mode="loso", fid=f, keep_trace=True))
            for src, lam in combos:
                jobs.append(_mk("confirm", "f5n", "rest_f5n", "rest", s, t, C,
                                phase_src=src, lam=lam, fid=f, keep_trace=True))
    b_rows, b_traces = _run(jobs, args.workers, "stage B — confirmatory conditions")

    rows = r0_rows + cal_rows + b_rows
    traces = {**cal_traces, **b_traces}
    _write(rows, traces, inv, sessions, args.out, (best, grid), False, strip_dir)


def _select(cal_rows):
    """Blind selection on walker conditions only."""
    grid = []
    for src in PHASE_SOURCES:
        for lam in LAMBDAS:
            cell = {}
            for condition in ("wphase", "wphase_iscr", "wphase_uscr"):
                v = [r["progress_rate"] for r in cal_rows
                     if r["condition"] == condition and r["phase_src"] == src and r["lam"] == lam]
                c = [r["cadence_hz"] for r in cal_rows
                     if r["condition"] == condition and r["phase_src"] == src and r["lam"] == lam]
                cell[condition] = (float(np.median(v)) if v else float("nan"),
                             float(np.median(c)) if c else float("nan"), len(v))
            real = cell["wphase"][0]
            dead = all(cell[a][0] < NULL_DEAD_ABS and cell[a][0] < NULL_DEAD_FRAC * real
                       for a in ("wphase_iscr", "wphase_uscr"))
            grid.append(dict(src=src, lam=lam, real=real, cad=cell["wphase"][1],
                             iscr=cell["wphase_iscr"][0], uscr=cell["wphase_uscr"][0],
                             n=cell["wphase"][2], admissible=bool(dead)))
    ok = [g for g in grid if g["admissible"]]
    if not ok:
        return ("k1", 0.0), grid
    b = max(ok, key=lambda g: g["real"])
    return (b["src"], b["lam"]), grid


def _wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.any(a - b != 0):
        return len(a), float("nan"), float("nan")
    from scipy.stats import wilcoxon
    res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return len(a), float(res.statistic), float(res.pvalue)


def _sel(rows, condition, src=None, lam=None, gm=None, stage=None):
    return [r for r in rows if r["condition"] == condition
            and (src is None or r["phase_src"] == src)
            and (lam is None or r["lam"] == lam)
            and (gm is None or r["gain_mode"] == gm)
            and (stage is None or r["stage"] == stage)]


def _write(rows, traces, inv, sessions, out, calib, r0_failed, strip_dir=None):
    with open(os.path.join(out, "g_v5_rollouts.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["stage"], r["rung"], r["condition"],
                                             str(r["phase_src"]), r["lam"],
                                             r["gain_mode"], r["session"], r["trial_idx"])):
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    print(f"wrote {out}/g_v5_rollouts.csv: {len(rows)} rows", flush=True)
    with open(os.path.join(out, "g_v5_inventory.json"), "w", encoding="utf-8") as fh:
        json.dump(inv, fh, indent=1)
    strips = _filmstrips(rows, traces, strip_dir, calib) if strip_dir and calib else []
    


def _filmstrips(rows, traces, out_dir, calib):
    (src_star, lam_star), _ = calib
    combos = [("walker", "n/a", 0.0, "g1.0"), ("decoded", "n/a", 0.0, "loso"),
              ("rest_f1", "n/a", 0.0, "loso"), ("wphase", "k1", 0.0, "n/a"),
              ("dphase", "k1", 0.0, "n/a"), ("rest_f5n", "k1", 0.0, "n/a"),
              ("metron_w", f"metronome_{METRONOME_HZ}", 0.0, "n/a")]
    if (src_star, lam_star) != ("k1", 0.0):
        combos += [("wphase", src_star, lam_star, "n/a"), ("dphase", src_star, lam_star, "n/a"),
                   ("rest_f5n", src_star, lam_star, "n/a")]
    made = []
    for condition, src, lam, gm in combos:
        sel = _sel(rows, condition, src, lam, gm)
        if not sel:
            continue
        sel.sort(key=lambda r: (r["progress_rate"], r["cadence_hz"]))
        for tag, r in [("worst", sel[0]), ("median", sel[len(sel) // 2]), ("best", sel[-1])]:
            k = _key(r)
            if k not in traces:
                continue
            tr, order = traces[k]
            nm = f"v5_{condition}_{src}_{lam:g}_{tag}_{r['session']}_t{r['trial_idx']}.png"
            p = os.path.join(out_dir, nm.replace("/", "_"))
            RR.filmstrip(dict(trace=tr, site_order=order), p,
                         f"{condition} [{r['rung']} src={src} lam={lam:g}] {tag}: {r['session']} "
                         f"trial {r['trial_idx']} — cadence {r['cadence_hz']:.2f} steps/s, "
                         f"progress {r['progress_rate']:+.2f}, {r['dur_s']:.1f} s, "
                         f"fell={r['fell']}")
            made.append((condition, src, lam, tag,
                         os.path.relpath(p, os.path.dirname(out_dir)).replace("\\", "/"), r))
    return made




def _condition_list(src_star, lam_star):
    A = [("f1", "walker", "n/a", 0.0, "g1.0", "walker"),
         ("f1", "decoded", "n/a", 0.0, "loso", "decoded"),
         ("f1", "decoded_iscr", "n/a", 0.0, "loso", "decoded_iscr"),
         ("f1", "rest_f1", "n/a", 0.0, "loso", "**rest-f1**"),
         ("f5n", "wphase", "k1", 0.0, "n/a", "wphase"),
         ("f5n", "wphase_iscr", "k1", 0.0, "n/a", "wphase_iscr"),
         ("f5n", "wphase_uscr", "k1", 0.0, "n/a", "wphase_uscr"),
         ("f5n", "dphase", "k1", 0.0, "n/a", "dphase"),
         ("f5n", "dphase_iscr", "k1", 0.0, "n/a", "dphase_iscr"),
         ("f5n", "dphase_uscr", "k1", 0.0, "n/a", "dphase_uscr"),
         ("f5n", "rest_f5n", "k1", 0.0, "n/a", "**rest-f5n**"),
         ("f5n", "metron_w", f"metronome_{METRONOME_HZ}", 0.0, "n/a", "metron_w"),
         ("f5n", "metron_d", f"metronome_{METRONOME_HZ}", 0.0, "n/a", "metron_d")]
    if (src_star, lam_star) != ("k1", 0.0):
        A += [("f5n", "wphase", src_star, lam_star, "n/a", "wphase\\*"),
              ("f5n", "wphase_iscr", src_star, lam_star, "n/a", "wphase\\*_iscr"),
              ("f5n", "wphase_uscr", src_star, lam_star, "n/a", "wphase\\*_uscr"),
              ("f5n", "dphase", src_star, lam_star, "n/a", "dphase\\*"),
              ("f5n", "dphase_iscr", src_star, lam_star, "n/a", "dphase\\*_iscr"),
              ("f5n", "dphase_uscr", src_star, lam_star, "n/a", "dphase\\*_uscr"),
              ("f5n", "rest_f5n", src_star, lam_star, "n/a", "**rest-f5n\\***")]
    return A


if __name__ == "__main__":
    main()
