"""Supplementary item

Resting-input control as a trial-level hierarchical model (participant as a random effect)

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and does not fit the decoder. It fits a linear mixed model
for the statistics reported with that panel or table.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import analysis_config as ec

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data/decoder_output"
SRC = ROOT / "04_source_data/supplementary"
STEM = "sup_restinput_trial_lme"
BUILD_ID = ec.TAG
BOOT_B, BOOT_SEED = 2000, 20260915


def one(key: str) -> Path:
    """The build_id's registry input; it must exist and carry the build_id tag (name or folder)."""
    p = ec.path(key)
    assert p.is_file() and (BUILD_ID in p.name or BUILD_ID in p.parent.name), f"{key}: {p}"
    ec.check_tag(ec.rel(key), "decoder", key)
    return p


def fit(t: pd.DataFrame, col: str) -> dict:
    best = None
    for meth in ("lbfgs", "powell", "cg"):
        try:
            r = smf.mixedlm(f"{col} ~ 1", t, groups=t["participant"], re_formula="~1",
                            vc_formula={"session": "0 + C(session)"}).fit(reml=True, method=meth, maxiter=4000)
        except Exception:
            continue
        if np.isfinite(r.llf) and (best is None or r.llf > best[0].llf):
            best = (r, meth)
    assert best is not None, f"{col}: no optimiser converged"
    r, meth = best
    ci = r.conf_int().loc["Intercept"]
    return {"mean": float(r.params["Intercept"]), "se": float(r.bse["Intercept"]),
            "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "z": float(r.tvalues["Intercept"]),
            "p": float(r.pvalues["Intercept"]),
            "sd_participant": float(np.sqrt(max(r.cov_re.iloc[0, 0], 0.0))),
            "sd_session": float(np.sqrt(max(r.vcomp[0], 0.0))),
            "sd_trial": float(np.sqrt(r.scale)), "converged": bool(r.converged), "optimizer": meth}


def main() -> None:
    warnings.filterwarnings("ignore")
    f_real, f_rest = one("h_iw_trial_r_long"), one("h_rest_trial_r_long")
    a = pd.read_csv(f_real); b = pd.read_csv(f_rest)
    a = a[a.model == "calib"]; b = b[b.model == "calib"]
    m = a.merge(b, on=["session", "trial_idx", "joint"], suffixes=("_real", "_rest"))
    if ec.SCORE_MASK is None:
        assert len(m) == len(a) == len(b), "real and rest dumps are not row-paired"
    else:
        assert len(m) == len(set(map(tuple, a[["session", "trial_idx", "joint"]].values))
                             & set(map(tuple, b[["session", "trial_idx", "joint"]].values))), "masked pairing"
    assert (m.fold_real == m.fold_rest).all() and (m.n_samples_real == m.n_samples_rest).all()
    m = m.dropna(subset=["rpt_r_real", "rpt_r_rest"])
    z = lambda r: np.arctanh(np.clip(r, -0.999, 0.999))
    m["z_real"], m["z_rest"] = z(m.rpt_r_real), z(m.rpt_r_rest)
    t = (m.groupby(["session", "trial_idx"])
          .agg(n_joints=("joint", "size"), r_real=("rpt_r_real", "mean"), r_rest=("rpt_r_rest", "mean"),
               z_real=("z_real", "mean"), z_rest=("z_rest", "mean")).reset_index())
    t["participant"] = t.session.str.extract(r"^(subj\d+)_")[0]
    assert t.participant.notna().all()
    t["d_r"], t["d_z"] = t.r_real - t.r_rest, t.z_real - t.z_rest
    n_t, n_s, n_p = len(t), t.session.nunique(), t.participant.nunique()
    N_EXP = 4756 if ec.SCORE_MASK is None else 1968
    assert (n_t, n_s, n_p) == (N_EXP, 80, 42), (n_t, n_s, n_p)

    lr, lz = fit(t, "d_r"), fit(t, "d_z")
    pm = t.groupby("participant").d_r.mean(); sm = t.groupby("session").d_r.mean()
    groups = {p: t.loc[t.participant == p, "d_r"].to_numpy() for p in pm.index}
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.array([np.mean([groups[p].mean() for p in rng.choice(pm.index.to_numpy(), size=n_p, replace=True)])
                     for _ in range(BOOT_B)])
    b_lo, b_hi = np.percentile(boot, [2.5, 97.5])

    SRC.mkdir(parents=True, exist_ok=True)
    cols = ["participant", "session", "trial_idx", "n_joints", "r_real", "r_rest", "d_r", "z_real", "z_rest", "d_z"]
    t[cols].to_csv(ec.sd(f"{STEM}_source_data.csv"), index=False, float_format="%.6f")
    meta = [
        ("analysis", "Imagined-walking decoding with the trial's own EEG versus a resting-input window, "
                     "trial-level paired difference, participant and session-within-participant random intercepts"),
        ("n_trials", n_t), ("n_sessions", n_s), ("n_participants", n_p),
        ("metric", "per-trial rpt_r (model=calib) averaged over walker joints finite in both inputs; trial scale, "
                   "NOT the session-mean per-joint rpt_calib of Supplementary Fig. 1a-c / Supplementary Table 2"),
        ("real_source", f_real.name), ("rest_source", f_rest.name),
        ("model", "d ~ 1 + (1|participant) + (1|participant:session); REML; statsmodels MixedLM"),
        ("trial_mean_r_real", f"{t.r_real.mean():.6f}"), ("trial_mean_r_rest", f"{t.r_rest.mean():.6f}"),
        ("lme_r_mean_difference", f"{lr['mean']:.6f}"), ("lme_r_ci95", f"{lr['ci_lo']:.6f} to {lr['ci_hi']:.6f}"),
        ("lme_r_se", f"{lr['se']:.6f}"), ("lme_r_z", f"{lr['z']:.4f}"), ("lme_r_p", f"{lr['p']:.3e}"),
        ("lme_r_p_reported", "< 0.001" if lr["p"] < 0.001 else f"{lr['p']:.3f}"),
        ("lme_r_sd_participant", f"{lr['sd_participant']:.6f}"), ("lme_r_sd_session_within_participant", f"{lr['sd_session']:.6f}"),
        ("lme_r_sd_trial_residual", f"{lr['sd_trial']:.6f}"), ("lme_r_optimizer", lr["optimizer"]), ("lme_r_converged", lr["converged"]),
        ("lme_fisherz_mean_difference", f"{lz['mean']:.6f}"), ("lme_fisherz_ci95", f"{lz['ci_lo']:.6f} to {lz['ci_hi']:.6f}"),
        ("lme_fisherz_p", f"{lz['p']:.3e}"), ("lme_fisherz_optimizer", lz["optimizer"]), ("lme_fisherz_converged", lz["converged"]),
        ("bootstrap_participant_equal_mean_difference", f"{pm.mean():.6f}"),
        ("bootstrap_ci95", f"{b_lo:.6f} to {b_hi:.6f}"), ("bootstrap_B", BOOT_B), ("bootstrap_seed", BOOT_SEED),
        ("bootstrap_fraction_le_0", f"{(boot <= 0).mean():.4f}"),
        ("participants_mean_difference_positive", f"{int((pm > 0).sum())}/{n_p}"),
        ("sessions_mean_difference_positive", f"{int((sm > 0).sum())}/{n_s}"),
        ("trials_difference_positive", f"{int((t.d_r > 0).sum())}/{n_t}"),
        ("source_script", "03_figures/sup_restinput_trial_lme.py"),
    ]
    with open(ec.sd(f"{STEM}_metadata.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["key", "value"]); w.writerows(meta)
    for k, v in meta:
        if k.startswith(("lme_r_mean", "lme_r_ci", "lme_r_p_rep", "lme_r_sd", "lme_fisherz_mean", "lme_fisherz_ci",
                         "bootstrap_part", "bootstrap_ci", "participants_", "sessions_", "trials_")):
            print(f"{k:<45} {v}")
    print(f"-> {ec.sd(STEM).parent}/{STEM}_{{source_data,metadata}}.csv")


if __name__ == "__main__":
    main()
