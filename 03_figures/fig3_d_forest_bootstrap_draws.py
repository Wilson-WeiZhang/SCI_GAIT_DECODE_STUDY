#!/usr/bin/env python3
"""Figure 3 — Gait-phase locking of the imagined-walking representation

Export the per-resample Spearman rho draws behind the Fig. 3 forest CIs

Writes:
  fig3_d_forest_bootstrap_draws.csv (not distributed)     (1000 rows x 4 predictor columns)
  04_source_data/fig3/fig3_d_forest_bootstrap_draws_metadata.csv

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import analysis_config as ec

ec.require("main", "main")
ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data" / "fig3"
N_BOOT = 1000
N_EXPORT = 1000

rng = np.random.default_rng(0)

m = pd.read_csv(ec.sd("fig3_master_source_data.csv"))
resp = pd.read_csv(ec.sd("fig3_def_resnonres_source_data.csv"))[["session", "resp"]]
m = m.merge(resp, on="session", how="left")
m["resp"] = m.resp.fillna(False).astype(bool)


def boot_rho(x, y, n=N_BOOT):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    r0, p = spearmanr(x, y)
    i = np.arange(len(x))
    rs = np.array([spearmanr(x[s], y[s])[0] for s in (rng.choice(i, len(i), True) for _ in range(n))])
    return r0, rs


draws, point = {}, {}
for k in (1, 2, 3):
    point[f"k{k}_taskevoked"], draws[f"k{k}_taskevoked"] = boot_rho(m[f"excess_k{k}"], m.r_reference)
for k in (1, 2, 3):
    point[f"k{k}_restcontrol"], draws[f"k{k}_restcontrol"] = boot_rho(m[f"rest_k{k}"], m.r_reference)
point["mubeta_erd"], draws["mubeta_erd"] = boot_rho(m.mubeta_erd_mean, m.r_reference)

PANEL = ["k1_taskevoked", "k2_taskevoked", "k3_taskevoked", "mubeta_erd"]

stats = pd.read_csv(ec.sd("fig3_d_forest_stats.csv")).set_index("predictor")
print(f"{'predictor':16s} {'rho':>9s} | {'CI(1000) = published':>21s} | {'CI(exported rows)':>21s} | max|delta|")
rows = []
for name in PANEL:
    d = draws[name]
    lo5, hi5 = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
    lo1, hi1 = np.nanpercentile(d[:N_EXPORT], 2.5), np.nanpercentile(d[:N_EXPORT], 97.5)
    pub_lo, pub_hi = float(stats.loc[name, "ci_lo"]), float(stats.loc[name, "ci_hi"])
    assert abs(lo5 - pub_lo) < 1e-12 and abs(hi5 - pub_hi) < 1e-12, f"{name}: {N_BOOT}-draw CI does not reproduce"
    print(f"{name:16s} {point[name]:9.6f} | {lo5:+.5f} .. {hi5:+.5f} | {lo1:+.5f} .. {hi1:+.5f} | "
          f"{max(abs(lo1 - lo5), abs(hi1 - hi5)):.5f}")
    rows.append((name, lo5, hi5, lo1, hi1))

out = pd.DataFrame({name: draws[name][:N_EXPORT] for name in PANEL})
out.insert(0, "resample", np.arange(1, N_EXPORT + 1))
out.to_csv(ec.sd("fig3_d_forest_bootstrap_draws.csv"), index=False, float_format="%.6f")

meta = [
    ("unit_of_analysis", "session"),
    ("n_sessions", 80),
    ("resampling", "session-level bootstrap with replacement, np.random.default_rng(0)"),
    ("n_draws_in_file", N_EXPORT),
    ("n_draws_behind_published_CI", N_BOOT),
    ("published_CI_source", f"{ec.sd('fig3_d_forest_stats.csv').relative_to(ROOT).as_posix()} (ci_lo/ci_hi)"),
    ("generator", "03_figures/fig3_d_forest_bootstrap_draws.py"),
]
for name, lo5, hi5, lo1, hi1 in rows:
    meta.append((f"{name}_ci_published", f"{lo5:.6f},{hi5:.6f}"))
    meta.append((f"{name}_ci_exported{N_EXPORT}", f"{lo1:.6f},{hi1:.6f}"))
pd.DataFrame(meta, columns=["key", "value"]).to_csv(
    ec.sd("fig3_d_forest_bootstrap_draws_metadata.csv"), index=False)

print(f"\nwrote {ec.sd('fig3_d_forest_bootstrap_draws.csv')} ({N_EXPORT} rows x {len(PANEL)} predictors)")
print(f"wrote {ec.sd('fig3_d_forest_bootstrap_draws_metadata.csv')}")
