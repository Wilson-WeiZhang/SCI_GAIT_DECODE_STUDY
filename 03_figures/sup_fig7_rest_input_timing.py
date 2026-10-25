"""Supplementary item (Figure 7 companion)

sup_fig7_rest_input_timing — resting-EEG decoder input in the Fig. 7 trials: timing against the reference walker

Reads:
  simulation and gate output in 02_decoding/simulation/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dump_h5
os.environ.setdefault("GAIT_F7_SIM", "causal_from2s")
import analysis_config as ec
ec.require("main")

SD = os.path.join(ROOT, "04_source_data", "supplementary")
ROLL = str(ec.scoped(os.path.join(SD, "sup_fig7_sim_per_patient_gate3s_source_data.csv")))
PROBE = os.path.join(ROOT, "02_decoding", "simulation", "simulation_rollouts_gate3s", f"{ec.SIM_IN_DIRNAME}_restprobe", "stats",
                     "f7r_per_trial_common.csv")
D_IW, D_REST = str(ec.path("p_dump_dir")), str(ec.path("p_rest_dump_dir"))
WIN_S = 3.0


def write(kind, df):
    p = str(ec.scoped(os.path.join(SD, f"sup_fig7_rest_input_timing_{kind}.csv")))
    df.to_csv(p + ".tmp", index=False)
    assert len(pd.read_csv(p + ".tmp")) == len(df)
    os.replace(p + ".tmp", p)
    return p


def z(x):
    return (x - x.mean(0)) / x.std(0)


def lead(q):
    return np.sign(q[:, 0] + q[:, 3])


def main():
    roll = pd.read_csv(ROLL)
    trials = roll[roll.condition == "Decoded"][["patient", "session", "trial_idx", "x3_m"]]
    probe = pd.read_csv(PROBE)
    rest_x = probe[probe.condition == "Rest"].set_index(["session", "trial_idx"]).disp_3s_m
    rows, rest_ids = [], {}
    for r0 in trials.itertuples(index=False):
        if not os.path.exists(os.path.join(D_REST, f"{r0.session}_dump.h5")):
            continue
        if r0.session not in rest_ids:
            S = dump_h5.load_session(dump_h5.dump_path(D_REST, r0.session), ("C_pred",))
            rest_ids[r0.session] = {t["trial_idx"] for t in S["trials"]}
        if int(r0.trial_idx) not in rest_ids[r0.session]:
            continue
        a = dump_h5.load_trial(dump_h5.dump_path(D_IW, r0.session), int(r0.trial_idx), fields=("Y_true", "C_pred"))
        b = dump_h5.load_trial(dump_h5.dump_path(D_REST, r0.session), int(r0.trial_idx), fields=("Y_true", "C_pred"))
        Y, C, Cr = a["Y_true"].T, a["C_pred"].T, b["C_pred"].T
        if not np.isfinite(Cr).any():
            continue
        ok = np.isfinite(C).all(1) & np.isfinite(Y).all(1) & np.isfinite(Cr).all(1)
        n3 = int(round(WIN_S * a["fs"]))
        Y, C, Cr = Y[ok][:n3], C[ok][:n3], Cr[ok][:n3]
        assert len(Y) == n3, (r0.session, r0.trial_idx, len(Y))
        lw, ld, lr = lead(z(Y)), lead(z(C)), lead(z(Cr))
        rows.append(dict(patient=r0.patient, session=r0.session, trial_idx=int(r0.trial_idx),
                         agreement_imagined_walking=float((ld == lw).mean()),
                         agreement_resting_input=float((lr == lw).mean()),
                         r6_imagined_walking=float(np.mean([np.corrcoef(Y[:, j], C[:, j])[0, 1] for j in range(6)])),
                         r6_resting_input=float(np.mean([np.corrcoef(Y[:, j], Cr[:, j])[0, 1] for j in range(6)])),
                         x3_m_decoded=float(r0.x3_m),
                         x3_m_resting_input_probe=float(rest_x.get((r0.session, int(r0.trial_idx)), np.nan))))
    d = pd.DataFrame(rows)
    assert d.x3_m_resting_input_probe.notna().all(), "a trial with a resting-input output has no probe rollout"
    q = lambda s: (float(s.median()), float(s.quantile(.25)), float(s.quantile(.75)))
    st = [("n_trials", len(d), "", ""), ("n_sessions", d.session.nunique(), "", ""),
          ("n_trials patient 1 / patient 2", f"{(d.patient == d.patient.min()).sum()} / {(d.patient == d.patient.max()).sum()}", "", "")]
    for k in ("agreement_imagined_walking", "agreement_resting_input", "r6_imagined_walking", "r6_resting_input",
              "x3_m_decoded", "x3_m_resting_input_probe"):
        m, lo, hi = q(d[k])
        st.append((f"{k}: median over trials", m, "", f"quartiles {lo:.6f} / {hi:.6f}"))
        for pt, g in d.groupby("patient"):
            st.append((f"{k}: median, {pt}", float(g[k].median()), "", f"n = {len(g)}"))
    da = d.agreement_imagined_walking - d.agreement_resting_input
    st += [("trials with agreement imagined walking > resting input", int((da > 0).sum()), "", f"of {len(d)}"),
           ("agreement imagined walking minus resting input: Wilcoxon signed-rank", float(stats.wilcoxon(da).statistic),
            float(stats.wilcoxon(da).pvalue), "two-sided, trials, uncorrected"),
           ("agreement resting input vs 0.5: Wilcoxon signed-rank", float(stats.wilcoxon(d.agreement_resting_input - .5).statistic),
            float(stats.wilcoxon(d.agreement_resting_input - .5).pvalue), "two-sided, trials, uncorrected"),
           ("agreement imagined walking vs 0.5: Wilcoxon signed-rank", float(stats.wilcoxon(d.agreement_imagined_walking - .5).statistic),
            float(stats.wilcoxon(d.agreement_imagined_walking - .5).pvalue), "two-sided, trials, uncorrected")]
    dx = d.x3_m_resting_input_probe - d.x3_m_decoded
    st += [("x3_m resting input minus decoded: median difference (m)", float(dx.median()), "", ""),
           ("x3_m resting input minus decoded: Wilcoxon signed-rank", float(stats.wilcoxon(dx).statistic),
            float(stats.wilcoxon(dx).pvalue), "two-sided, trials, uncorrected"),
           ("fraction of resting-input rollouts with x3_m > 0", float((d.x3_m_resting_input_probe > 0).mean()), "", "")]
    ses = d.groupby("session")[["agreement_imagined_walking", "agreement_resting_input"]].median()
    ds = ses.agreement_imagined_walking - ses.agreement_resting_input
    st += [("sessions with agreement imagined walking > resting input (session medians)", int((ds > 0).sum()), "", f"of {len(ses)}"),
           ("session medians, agreement imagined walking minus resting input: exact Wilcoxon signed-rank",
            float(stats.wilcoxon(ds, mode="exact").statistic), float(stats.wilcoxon(ds, mode="exact").pvalue),
            f"two-sided, n = {len(ses)} sessions")]
    st = pd.DataFrame(st, columns=["quantity", "value", "p", "note"])
    md = pd.DataFrame([
        ("cohort", "post_operative; simulation trials with a resting-input decoder output; 2 patients, case-level"),
        ("input_rollouts", os.path.relpath(ROLL, ROOT)), ("input_dump_imagined_walking", os.path.relpath(D_IW, ROOT)),
        ("input_dump_resting_input", os.path.relpath(D_REST, ROOT)),
        ("window", f"{WIN_S} s after the onset of the final decoder output"),
        ("leading_leg", "sign of zL_hip + zR_hip (raw channels, z-scored within the window); chance agreement 0.5"),
        ("level", "decoder output (the drive of the simulation), not the simulated biped"),
    ], columns=["key", "value"])
    for k, df in (("source_data", d), ("stats", st), ("metadata", md)):
        print("wrote", os.path.relpath(write(k, df), ROOT))
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
    print(st.to_string(index=False))


if __name__ == "__main__":
    main()
