#!/usr/bin/env python3
"""Figure 3 — Gait-phase locking of the imagined-walking representation

Export the per-resample Spearman rho draws behind the Fig. 3 self-report forest CIs

Reads:
  04_source_data/fig3/fig3_master_source_data.csv
  04_source_data/fig3/fig3_def_resnonres_source_data.csv

Writes:
  fig3_h_forest_bootstrap_draws.csv (not distributed)     (1000 rows x 6 item columns)
  04_source_data/fig3/fig3_h_forest_bootstrap_draws_metadata.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from safe_write import safe_write

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data" / "fig3"
N_BOOT = 1000
N_EXPORT = 1000

rng = np.random.default_rng(0)

m = pd.read_csv(SD / "fig3_master_source_data.csv")
resp = pd.read_csv(SD / "fig3_def_resnonres_source_data.csv")[["session", "resp"]]
m = m.merge(resp, on="session", how="left")
m["resp"] = m.resp.fillna(False).astype(bool)

ITEMS = [("Q1_intent", "intent to move"), ("Q2_sync", "sync with walker"), ("Q3_still", "stayed still (0-1)"),
         ("Q4_pace", "matched pace"), ("Q5_relax", "felt relaxed"), ("Q6_avoid", "volitional control")]


def boot_rho(x, y, n=N_BOOT):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r0, p = spearmanr(x, y)
    i = np.arange(len(x))
    rs = np.array([spearmanr(x[s], y[s])[0] for s in (rng.choice(i, len(i), True) for _ in range(n))])
    rs = rs[np.isfinite(rs)]
    return r0, rs, len(x)


draws, point, nvalid = {}, {}, {}
for col, lab in ITEMS:
    point[col], draws[col], nvalid[col] = boot_rho(m[col], m.excess_k1)

stats = pd.read_csv(SD / "fig3_h_forest_stats.csv").set_index("item")
order = list(stats.sort_values("rho", ascending=False).index)

print(f"{'item':11s} {'label':20s} {'rho':>7s} {'finite draws':>12s} | {'CI(1000) = published':>21s} | "
      f"{'CI(exported rows)':>21s} | max|d|")
rows = []
for col in order:
    d = draws[col]
    lo5, hi5 = np.percentile(d, [2.5, 97.5])
    lo1, hi1 = np.percentile(d[:N_EXPORT], [2.5, 97.5])
    pub_lo, pub_hi = float(stats.loc[col, "ci_lo"]), float(stats.loc[col, "ci_hi"])
    assert abs(lo5 - pub_lo) < 1e-12 and abs(hi5 - pub_hi) < 1e-12, f"{col}: {N_BOOT}-draw CI does not reproduce"
    print(f"{col:11s} {stats.loc[col,'label']:20s} {point[col]:7.3f} {len(d):12d} | {lo5:+.5f} .. {hi5:+.5f} | "
          f"{lo1:+.5f} .. {hi1:+.5f} | {max(abs(lo1-lo5), abs(hi1-hi5)):.5f}")
    rows.append((col, str(stats.loc[col, "label"]), lo5, hi5, lo1, hi1, len(d)))

out = pd.DataFrame({col: pd.Series(draws[col][:N_EXPORT]) for col in order}).reindex(range(N_EXPORT))
out.insert(0, "resample", np.arange(1, N_EXPORT + 1))
safe_write(SD / "fig3_h_forest_bootstrap_draws.csv",
           out.to_csv(index=False, float_format="%.6f", lineterminator="\n"))

meta = [
    ("paper_location", "Fig. 3 panel i - per-resample Spearman rho draws behind the 95% bootstrap CIs of the "
                       "self-report forest (six items vs rest-corrected k1 locking, n = 76 sessions)"),
    ("unit_of_analysis", "session"),
    ("n_sessions_with_questionnaire", int(stats["n"].iloc[0])),
    ("statistic", "Spearman rho of each self-report item vs excess_k1 (TASK-EVOKED k1 LOCKING)"),
    ("resampling", "session-level bootstrap with replacement, np.random.default_rng(0)"),
    ("n_draws_in_file", N_EXPORT),
    ("n_draws_behind_published_CI", N_BOOT),
    ("degenerate_draws", "resamples returning NaN rho (constant 0-1 item) are dropped, as in the renderer"),
    ("column_order", "top-to-bottom as plotted (descending rho)"),
    ("published_CI_source", "04_source_data/fig3/fig3_h_forest_stats.csv (ci_lo/ci_hi)"),
    ("significance_rule", "BH-FDR q<0.05 across the 6 items, NOT the raw bootstrap CI"),
    ("generator", "03_figures/fig3_h_forest_bootstrap_draws.py"),
]
for col, lab, lo5, hi5, lo1, hi1, nfin in rows:
    meta.append((f"{col}_label", lab))
    meta.append((f"{col}_finite_draws_of_{N_BOOT}", nfin))
    meta.append((f"{col}_ci_published", f"{lo5:.6f},{hi5:.6f}"))
    meta.append((f"{col}_ci_exported{N_EXPORT}", f"{lo1:.6f},{hi1:.6f}"))
safe_write(SD / "fig3_h_forest_bootstrap_draws_metadata.csv",
           pd.DataFrame(meta, columns=["key", "value"]).to_csv(index=False, lineterminator="\n"))

print(f"\nwrote {SD / 'fig3_h_forest_bootstrap_draws.csv'} ({N_EXPORT} rows x {len(order)} items)")
print(f"wrote {SD / 'fig3_h_forest_bootstrap_draws_metadata.csv'}")
