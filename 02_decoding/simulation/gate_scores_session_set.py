"""Figure 7 — physics simulation and walk-intent gate

gate_scores_session_set — the fused (V1, converged Karcher mean) decision stream on the 11-session gate session set.

Reads:
  the aligned per-session recordings (not distributed)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import csv
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import karcher_checked as KC
import gate_features as GF
import gate_fusion as PF

RES = os.path.join(HERE, "result")
ROOT_V47 = "${GAIT_PROJECT_ROOT}/patient_prep_restgood_v4_7"
ROOT_G11 = "${GAIT_PROJECT_ROOT}/patient_prep_gate"

SESS7 = ["pat01_sess04", "pat01_sess06", "pat01_sess07", "pat01_sess09",
         "pat02_sess03", "pat02_sess05", "pat02_sess06"]
SESS11 = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
          "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06",
          "pat02_sess08", "pat02_sess09"]

CHECKED_SCORES = os.path.join(RES, "gate_fused_checked_scores.csv")
OUT_SCORES = os.path.join(RES, "gate_event_set_scores.csv")
OUT_SCORES_B = os.path.join(RES, "gate_event_set_scores_condition_b.csv")
OUT_THR = os.path.join(RES, "gate_event_set_thresholds.csv")
OUT_CHK = os.path.join(RES, "gate_event_set_selfcheck.csv")

TOL_A = 1e-9
HEADLINE_CHECKED7 = 0.932871
SCORE_COLS = ["score_lda", "score_ts", "fuse_V1", "fuse_V2", "fuse_V3"]
IQR_K = 0.25


def log(m):
    print(m, flush=True)


def bind(sessions, root):
    """Rebind the session list and the data root IN THIS PROCESS ONLY."""
    GF.SESSIONS = list(sessions)
    PF.SESSIONS = list(sessions)
    PF.ROOT = root


def loader_inventory(sessions, root):
    """Per-session event/window inventory straight from the frozen loader, plus anomaly flags."""
    inv = []
    for s in sessions:
        X, fs, ch, ev = GF.load_session(root, s)
        rest = [e for e in ev if e["code"] == "rest"]
        iw = [e for e in ev if e["code"] == "gaitimagery"]
        cal, eval_rest = GF.calib_split(rest, fs, PF.BASE_MIN_S)
        cal_win = sum(len(GF.windows(X, e["a"], e["b"], fs)) for e in cal)
        cal_s = sum(e["b"] - e["a"] for e in cal) / fs
        t_iw = sum(len(GF.windows(X, e["a"], e["b"], fs)) for e in iw)
        t_rest = sum(len(GF.windows(X, e["a"], e["b"], fs)) for e in eval_rest)
        d_iw = sum(len(PF.dense_windows(X, e["a"], e["b"], fs)) for e in iw)
        d_rest = sum(len(PF.dense_windows(X, e["a"], e["b"], fs)) for e in eval_rest)
        short = [f"{lab}{k}" for lab, evs in (("iw", iw), ("rest", eval_rest))
                 for k, e in enumerate(evs) if not GF.windows(X, e["a"], e["b"], fs)]
        anom = []
        if abs(fs - 250.0) > 1e-9:
            anom.append(f"fs={fs}")
        if len(ch) != 60:
            anom.append(f"n_ch={len(ch)}")
        if cal_win < 3:
            anom.append(f"calib_windows={cal_win}<3")
        if not iw or not eval_rest:
            anom.append("empty class")
        if short:
            anom.append(f"events_without_a_4s_window={len(short)} ({','.join(short)})")
        inv.append(dict(session=s, n_iw_ev=len(iw), n_rest_ev=len(rest),
                        n_calib_ev=len(cal), calib_s=round(cal_s, 1), n_calib_win=cal_win,
                        n_eval_rest_ev=len(eval_rest), n_tiled_iw=t_iw, n_tiled_rest=t_rest,
                        n_dense_iw=d_iw, n_dense_rest=d_rest, n_dense=d_iw + d_rest,
                        anomalies="; ".join(anom)))
        log(f"  [inv] {s}: IW {len(iw)} ev / {d_iw} dec, rest {len(eval_rest)}(+{len(cal)} calib) "
            f"ev / {d_rest} dec, calib {cal_s:.1f}s / {cal_win} win"
            + (f"   ANOMALY: {'; '.join(anom)}" if anom else ""))
    return inv


def run_condition(name, sessions, root):
    log(f"\n=========== CONDITION {name}: {len(sessions)} sessions, root {os.path.basename(root)} ===")
    bind(sessions, root)
    rows, _meta = PF.build(root, base_min_s=PF.BASE_MIN_S)
    keys, covs, calib = PF.cov_inventory(root, PF.BASE_MIN_S)
    assert [(r["session"], r["event"], r["win"]) for r in rows] == keys, \
        f"condition {name}: cov inventory diverged from build()"
    PF.whiteners = KC.make_whiteners("checked")
    auc, dense, thr, cal = PF.deployable(rows, keys, covs, calib, PF.LAM_RA, dense=True)
    return auc, dense, thr, cal


def tiled_auc(dense_rows, sessions, col="fuse_V1"):
    out = {}
    for s in sessions:
        y = [r["label"] for r in dense_rows
             if r["session"] == s and abs(r["t_end_ev_s"] % 4.0) < 1e-9]
        d = [r[col] for r in dense_rows
             if r["session"] == s and abs(r["t_end_ev_s"] % 4.0) < 1e-9]
        out[s] = float(roc_auc_score(y, d))
    return out


def check_row_counts(name, dense_rows, inv, chk):
    """Self-check (ii): dense rows per session == the loader's independently counted decisions."""
    per = {}
    for r in dense_rows:
        per[r["session"]] = per.get(r["session"], 0) + 1
    worst = 0
    for d in inv:
        got, want = per.get(d["session"], 0), d["n_dense"]
        worst = max(worst, abs(got - want))
        chk.append(dict(check=f"{name}_row_count", key=d["session"], n=got,
                        max_abs_delta=abs(got - want), tol=0, passed=int(got == want)))
        assert got == want, f"condition {name} {d['session']}: {got} dense rows, loader says {want}"
    ev = {}
    for r in dense_rows:
        ev[(r["session"], r["event"])] = ev.get((r["session"], r["event"]), 0) + 1
    for r in dense_rows:
        assert ev[(r["session"], r["event"])] == r["n_dec_ev"], \
            f"condition {name}: {r['session']}|{r['event']} rows != n_dec_ev"
    log(f"[{name} row counts] PASS  {len(dense_rows)} rows over {len(per)} sessions, "
        f"max |delta| vs loader {worst}")


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    chk = []

    log("== loader inventory, gate_sessions root, 11 sessions ==")
    inv11 = loader_inventory(SESS11, ROOT_G11)
    for d in inv11:
        chk.append(dict(check="inventory_gate", key=d["session"], **{
            k: v for k, v in d.items() if k != "session"}))

    aucA, denseA, _thrA, _calA = run_condition("A", SESS7, ROOT_V47)
    ref = list(csv.DictReader(open(CHECKED_SCORES, encoding="utf-8")))
    assert len(ref) == len(denseA), f"CONDITION A FAIL row count {len(denseA)} vs {len(ref)}"
    worst = {c: 0.0 for c in SCORE_COLS}
    for i, (a, b) in enumerate(zip(denseA, ref)):
        assert (a["session"], a["event"]) == (b["session"], b["event"]) and \
            abs(a["t_end_ev_s"] - float(b["t_end_ev_s"])) < 1e-9 and \
            int(a["label"]) == int(b["label"]), f"CONDITION A FAIL grid/order mismatch at row {i}"
        for c in SCORE_COLS:
            worst[c] = max(worst[c], abs(a[c] - float(b[c])))
    for c in SCORE_COLS:
        log(f"  [A] {c}: max |delta| over {len(ref)} rows = {worst[c]:.3e}")
        chk.append(dict(check="A_codepath_regression", key=c, n=len(ref),
                        max_abs_delta=worst[c], tol=TOL_A, passed=int(worst[c] <= TOL_A)))
    assert worst["fuse_V1"] <= TOL_A, f"CONDITION A FAIL fuse_V1 {worst['fuse_V1']:.3e} > {TOL_A}"
    aA = tiled_auc(denseA, SESS7)
    mA = float(np.mean([aA[s] for s in SESS7]))
    assert abs(mA - HEADLINE_CHECKED7) < 5e-7, f"CONDITION A FAIL mean7 {mA:.9f}"
    log(f"[CONDITION A] PASS  reproduces the checked stream (fuse_V1 max |delta| {worst['fuse_V1']:.3e}), "
        f"mean-of-7 {mA:.6f} == {HEADLINE_CHECKED7}")
    for s in SESS7:
        chk.append(dict(check="A_tiled_auc", key=s, auc=aA[s]))
    chk.append(dict(check="A_tiled_auc", key="MEAN7", auc=mA))

    aucB, denseB, thrB, _calB = run_condition("B", SESS7, ROOT_G11)
    invB = loader_inventory(SESS7, ROOT_G11)
    check_row_counts("B", denseB, invB, chk)
    aB = tiled_auc(denseB, SESS7)
    mB = float(np.mean([aB[s] for s in SESS7]))
    for s in SESS7:
        chk.append(dict(check="B_tiled_auc", key=s, auc=aB[s], max_abs_delta=abs(aB[s] - aA[s])))
        log(f"  [B] {s}: {aB[s]:.6f}  (A {aA[s]:.6f}, delta {aB[s] - aA[s]:+.6f})")
    chk.append(dict(check="B_tiled_auc", key="MEAN7", auc=mB, max_abs_delta=abs(mB - mA)))
    log(f"[CONDITION B] loading-rule-only mean-of-7 {mB:.6f}  (A {mA:.6f}, delta {mB - mA:+.6f})")

    aucC, denseC, thrC, _calC = run_condition("C", SESS11, ROOT_G11)
    check_row_counts("C", denseC, inv11, chk)
    aC = tiled_auc(denseC, SESS11)
    mC = float(np.mean([aC[s] for s in SESS11]))
    for s in SESS11:
        d = f"  (B {aB[s]:.6f}, delta {aC[s] - aB[s]:+.6f})" if s in aB else "   [NEW SESSION]"
        log(f"  [C] {s}: {aC[s]:.6f}{d}")
        chk.append(dict(check="C_tiled_auc", key=s, auc=aC[s],
                        max_abs_delta=(abs(aC[s] - aB[s]) if s in aB else "")))
    chk.append(dict(check="C_tiled_auc", key="MEAN11", auc=mC))
    m7inC = float(np.mean([aC[s] for s in SESS7]))
    chk.append(dict(check="C_tiled_auc", key="MEAN7_subset_of_LOSO11", auc=m7inC))
    log(f"[CONDITION C] LOSO-11 mean-of-11 {mC:.6f}; the 7-session subset of LOSO-11 reads "
        f"{m7inC:.6f} ")

    for row in thrC:
        if row["variant"] == "V1":
            t = ((0.0 - row["centre_A"]) / row["sd_A"]
                 + (0.0 - row["centre_B"]) / row["sd_B"]) / 2.0
            d = IQR_K * row["calib_iqr"]
            row["theta_sym"] = t
            row["delta"] = d
            row["theta_on_s1"] = t + d
            row["theta_off_s1"] = t - d
        else:
            row["theta_sym"] = row["delta"] = row["theta_on_s1"] = row["theta_off_s1"] = ""

    PF.write(OUT_SCORES, denseC)
    PF.write(OUT_SCORES_B, denseB)
    PF.write(OUT_THR, thrC)
    PF.write(OUT_CHK, chk)
    tagged = ([dict(condition="A_frozen7_v4_7", **r) for r in aucA]
              + [dict(condition="B_reference_gate", **r) for r in aucB]
              + [dict(condition="C_post_operative_gate", **r) for r in aucC])
    PF.write(os.path.join(RES, "gate_event_set_auc.csv"), tagged)
    log(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
