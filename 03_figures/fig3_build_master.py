"""Figure 3 — Gait-phase locking of the imagined-walking representation

master table builder (per-session locking, decoding, ERD and questionnaire scores)

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import analysis_config as ec

ec.require("main", "main")
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "decoder_output"
SD = ROOT / "04_source_data" / "fig3"
SD.mkdir(parents=True, exist_ok=True)
OUT = ec.sd("fig3_master_source_data.csv")
META = ec.sd("fig3_master_main_metadata.csv")


def _cabs_by_k(path):
    ec.check_tag(path.parent.name, "neural", "fig3 master |c| input")
    d = pd.read_csv(path, usecols=["session", "harmonic_k", "real", "imag"])
    d["cabs"] = np.hypot(d.real, d.imag)
    t = d.groupby(["session", "harmonic_k"])["cabs"].mean().unstack("harmonic_k")
    t.columns = [f"k{int(c)}" for c in t.columns]
    return t.reset_index()


def build():
    ec.check_tag(ec.rel("h_session_r"), "decoder", "fig3 master decoding r")
    r = pd.read_csv(ec.path("h_session_r")).rename(columns={"r": "r_reference"})

    mi = _cabs_by_k(R / "diag_harmonic_phase_responder_split"
                    / "persession_harmonic_phase_cycle9_complex.csv")
    mi = mi.rename(columns={c: f"mi_{c}" for c in mi.columns if c.startswith("k")})
    rest = _cabs_by_k(R / "diag_rest_yoked"
                      / "persession_harmonic_phase_cycle9_complex.csv")
    rest = rest.rename(columns={c: f"rest_{c}" for c in rest.columns if c.startswith("k")})

    m = r.merge(mi, on="session").merge(rest, on="session")
    for k in ("k1", "k2", "k3"):
        m[f"excess_{k}"] = m[f"mi_{k}"] - m[f"rest_{k}"]

    _erd = R / "v4_fold1_following_rest_erd_main_session.csv"
    ec.check_tag(_erd.name, "neural", "fig3 master ERD input")
    erd = pd.read_csv(_erd,
                      usecols=["session", "mu_erd_mean", "beta_erd_mean", "mubeta_erd_mean"])
    m = m.merge(erd, on="session", how="left")

    F = ROOT / "data_descrption_from_others" / "Intend to Walk - EEG Gait Decoding - all subjects - scored.xlsx"
    q = pd.read_excel(F, sheet_name="scores")
    cc = list(q.columns)
    ren = {cc[3]: "Q1_intent", cc[4]: "Q2_sync", cc[5]: "Q3_still",
           cc[6]: "Q4_pace", cc[7]: "Q5_relax", cc[8]: "Q6_avoid"}
    q = q.rename(columns=ren)
    q["Q3_still"] = q["Q3_still"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    q["subj_int"] = q["subj_number"].str.extract(r"SUB_0*(\d+)").astype(int)
    q["sess_int"] = q["session_number"].astype(int)
    qa = q.groupby(["subj_int", "sess_int"])[
        ["Q1_intent", "Q2_sync", "Q3_still", "Q4_pace", "Q5_relax", "Q6_avoid"]].mean().reset_index()
    m["subj_int"] = m.session.str.extract(r"subj0*(\d+)").astype(int)
    m["sess_int"] = m.session.str.extract(r"sess0*(\d+)").astype(int)
    m = m.merge(qa, on=["subj_int", "sess_int"], how="left")

    m.to_csv(OUT, index=False)

    meta = pd.DataFrame([
        ("n_sessions", len(m)),
        ("n_subjects", m.subj_int.nunique()),
        ("rest95_floor_k1", round(float(np.quantile(m.rest_k1, 0.95)), 6)),
        ("n_with_Q6", int(m.Q6_avoid.notna().sum())),
        ("locking_metric", "unit-phasor gait-phase concentration |c|, amplitude divided out, decoder-independent"),
        ("excess_def", "excess_k = mi_k - rest_k (yoked-REST trial-matched chance floor)"),
        ("ERD_sign", "mu/beta_erd_mean negative = desynchronization (stronger classic MI marker)"),
        ("Q6_def", "Q6_avoid = self-reported avoidance of imagery during REST (higher = better imagery control)"),
    ], columns=["key", "value"])
    meta = meta.sort_values("key", kind="stable").reset_index(drop=True)
    meta.to_csv(META, index=False)

    print(f"wrote {OUT}  ({len(m)} sessions, {m.subj_int.nunique()} subjects, cohort r={m.r_reference.mean():.4f})")
    print(f"wrote {META}")
    return m


if __name__ == "__main__":
    build()
