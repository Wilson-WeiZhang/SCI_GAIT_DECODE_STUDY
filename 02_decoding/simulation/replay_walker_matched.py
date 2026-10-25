"""Figure 7 — physics simulation and walk-intent gate

replay_walker_matched — the walker's own goniometer phase replayed on exactly the decoded window of every decoded trial.

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os, sys, csv, gzip, glob, argparse
import numpy as np
import h5py
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import data_io as IO
import rhythm_metrics as PR
import dump_traces_patient as DT
import run_rollouts as G5
import analysis_config as ec

CONDITION = "wphase_matched"
REPRO_COLS = ["cadence_hz", "progress_rate", "survival_s", "n_steps_max_streak", "n_support_transitions", "dx_m",
              "clamp_frac", "tau_int", "tau_mean", "tau_peak", "frac_double_support", "frac_flight",
              "stand_height_m", "pitch_after_settle_deg", "final_pitch_deg", "final_hip_z_m", "fell"]


def matched_trials(dump, session):
    if ec.DUMP_FORMAT == "pyport_session_h5":
        out = []
        for r in IO.h5_session_trials(dump, session):
            C, Y, tid = r["C_pred"], r["Y_true"], r["trial_idx"]
            assert C.shape == Y.shape, f"{session} trial {tid}: C_pred {C.shape} vs Y_true {Y.shape}"
            a, b = IO._longest_finite_run(C)
            if (b - a) < int(round(IO.MIN_DUR_S * IO.FS)):
                continue
            Yw = Y[a:b]
            assert np.isfinite(Yw).all(), f"{session} trial {tid}: Y_true not finite on the decoded window"
            out.append(dict(session=session, trial_idx=tid, fold=r["fold"], ordinal=r["ordinal"], t0_sec=float(a) / IO.FS,
                            n=int(b - a), dur_s=float((b - a) / IO.FS) - IO._F7_WIN, Y=Yw))
            if IO.F7_WARM or IO.F7_FROM2S:
                out[-1]["W"] = IO.f7_warm_input(Y, a)
        return sorted(out, key=lambda r: r["trial_idx"])
    ec.require_dump_format("matlab_fold_mat")
    files = sorted(glob.glob(os.path.join(dump, f"{session}_fold*_trial_dump.mat")),
                   key=lambda p: int(os.path.basename(p).split("_fold")[1].split("_")[0]))
    seen, out = set(), []
    for f in files:
        fold = int(os.path.basename(f).split("_fold")[1].split("_")[0])
        with h5py.File(f, "r") as h:
            idx = np.array(h["meta/original_trial_indices"]).ravel().astype(int)
            rc, ry = h["C_pred"], h["Y_true"]
            for o in range(rc.shape[0]):
                tid = int(idx[o])
                if tid in seen:
                    continue
                C = np.array(h[rc[o, 0]]).T
                Y = np.array(h[ry[o, 0]]).T
                assert C.shape == Y.shape, f"{session} trial {tid}: C_pred {C.shape} vs Y_true {Y.shape}"
                a, b = IO._longest_finite_run(C)
                if (b - a) < int(round(IO.MIN_DUR_S * IO.FS)):
                    continue
                seen.add(tid)
                Yw = Y[a:b]
                assert np.isfinite(Yw).all(), f"{session} trial {tid}: Y_true not finite on the decoded window"
                out.append(dict(session=session, trial_idx=tid, fold=fold, ordinal=o, t0_sec=float(a) / IO.FS,
                                n=int(b - a), dur_s=float((b - a) / IO.FS) - IO._F7_WIN, Y=Yw))
    return sorted(out, key=lambda r: r["trial_idx"])


def g5_job(stage, condition, session, trial, ctx):
    return G5._mk(stage, "f5n", condition, "Y_true", session, trial, ctx, phase_src="k1", lam=0.0, level=G5.RUN_LEVEL)


def g5_run(J):
    return G5._job(J)[0]


def write_csv(path, cols, rows):
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ec.path("p_dump_dir"))); ap.add_argument("--npz", default=str(ec.path("p_cycavg")))
    ap.add_argument("--out", required=True); ap.add_argument("--sessions", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    IO.build_id_inputs(dump=a.dump, npz=a.npz)
    a.out = IO.build_id_out(a.out)
    tmpl_n = IO.fukuchi_template()
    jobs, g5_matched, g5_repro = [], [], []
    for s in [x for x in a.sessions.split(",") if x]:
        sp = IO.device_stance_ankle(IO.loso_setpoint(a.npz, s)[0]); tg = IO.loso_template(a.npz, s)
        delta, _ = IO.global_delta(tg, tmpl_n)
        ctx = dict(setpoint=sp, tmpl_g=tg, tmpl_n=tmpl_n, delta=delta)
        mt = matched_trials(a.dump, s)
        dec = IO.load_session_trials(a.dump, s, stream="C_pred")
        assert [(t["trial_idx"], t["n"]) for t in mt] == [(t["trial_idx"], t["n"]) for t in dec], \
            f"{s}: matched walker trials differ from the decoded condition"
        for t in mt:
            jobs.append(dict(stage="calib", condition=CONDITION, trial=t, null=None, session=s, **ctx))
            g5_matched.append(g5_job("calib", CONDITION, s, t, ctx))
        for t in ([] if IO.F7_FROM2S else IO.load_session_trials(a.dump, s, stream="Y_true")):
            g5_repro.append(g5_job("calib", "wphase", s, t, ctx))
    print(f"MIN_DUR_S={IO.MIN_DUR_S}  window-matched walker rollouts {len(jobs)}; reproduction-gate rollouts "
          f"{len(g5_repro)}", flush=True)
    import multiprocessing as mp
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        repro = list(pool.imap_unordered(g5_run, g5_repro, chunksize=8))
        if IO.F7_FROM2S:
            print("reproduction gate vs the unmatched walker rows of g_v5_rollouts.csv: NOT DEFINED under "
                  "GAIT_F7_SIM=causal_from2s (no unmatched walker is simulated); the rollouts-vs-rhythm cross-check "
                  "below still runs", flush=True)
        ref = {}
        with open(os.path.join(a.out, "g_v5_rollouts.csv"), newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (r["stage"], r["condition"], r["rung"], r["phase_src"]) == ("calib", "wphase", "f5n", "k1") \
                        and float(r["lam"]) == 0.0:
                    ref[(r["session"], int(r["trial_idx"]))] = r
        assert len(ref) == len(repro), f"reproduction gate: {len(repro)} rollouts vs {len(ref)} frozen rows"
        worst = 0.0
        for r in repro:
            f = ref[(r["session"], int(r["trial_idx"]))]
            for c in REPRO_COLS:
                worst = max(worst, abs(float(r[c]) - float(f[c])))
        assert worst < 1e-9, f"reproduction gate FAILED: max |diff| {worst}"
        assert not (IO.F7_FROM2S and (ref or repro)), "causal_from2s: unexpected unmatched walker rows"
        if not IO.F7_FROM2S:
            print(f"reproduction gate PASS: {len(repro)} walker rollouts, max |diff| over {len(REPRO_COLS)} outcomes "
                  f"{worst:.1e}", flush=True)
        g5_rows = list(pool.imap_unordered(g5_run, g5_matched, chunksize=4))
        rows = list(pool.imap_unordered(PR._job, jobs, chunksize=4))
        trace_rows = [r for rr in pool.imap_unordered(DT._job, jobs, chunksize=4) for r in rr]
    os.makedirs(a.out, exist_ok=True)
    g5_rows.sort(key=lambda r: (r["session"], r["trial_idx"]))
    write_csv(os.path.join(a.out, "walker_matched_rollouts.csv"), G5.CSV_COLS, g5_rows)
    rows.sort(key=lambda r: (r["session"], r["trial_idx"]))
    write_csv(os.path.join(a.out, "walker_matched_rhythm.csv"), PR.CSV_COLS, rows)
    bad = [r for r in rows if r["chk_streak_1cm"] != r["n_steps_max_streak"] or r["chk_ntrans"] != r["n_support_transitions"]]
    cross = max(abs(float(g["cadence_hz"]) - float(r["cadence_hz"])) + abs(float(g["dx_m"]) - float(r["dx_m"]))
                for g, r in zip(g5_rows, rows))
    assert cross < 1e-9, f"rollouts vs rhythm replay disagree: {cross}"
    q = os.path.join(a.out, "walker_matched_traces_x.csv.gz")
    with gzip.open(q + ".tmp", "wt", newline="") as fh:
        w = csv.writer(fh); w.writerow(["stage", "condition", "session", "trial_idx", "t_s", "x_m"]); w.writerows(trace_rows)
    os.replace(q + ".tmp", q)
    print(f"wrote walker_matched_rollouts.csv {len(g5_rows)} rows; walker_matched_rhythm.csv {len(rows)} rows "
          f"(internal consistency mismatches {len(bad)}; rollouts vs rhythm max |diff| {cross:.1e}); "
          f"{q}: {len(trace_rows)} samples", flush=True)


if __name__ == "__main__":
    main()
