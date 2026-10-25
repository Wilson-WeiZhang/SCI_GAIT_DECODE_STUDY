#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary item

Is responder status participant-dependent? — numbers-only producer (no figure)

Reads:
  04_source_data/fig3/fig3_def_resnonres_metadata.csv
  04_source_data/fig3/fig3_def_resnonres_source_data.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import fisher_exact, spearmanr

SD = Path(__file__).resolve().parents[1] / "04_source_data" / "supplementary"
STEM = "sup_responder_participant_consistency"
N_PERM, SEED = 100_000, 20260925

meta_in = pd.read_csv(SD.parent / "fig3" / "fig3_def_resnonres_metadata.csv").set_index("key")["value"]
BUILD_ID = meta_in["build_id"]
m = pd.read_csv(SD.parent / "fig3" / "fig3_def_resnonres_source_data.csv")
m["participant"] = m.session.str.extract(r"^(subj\d+)")[0]
m["sess"] = m.session.str.extract(r"_sess(\d+)")[0].astype(int)
n_per = m.groupby("participant").size()
assert len(m) == 80 and len(n_per) == 42 and n_per.max() == 2, "healthy cohort / 42 participants expected"

pairs = m[m.participant.isin(n_per[n_per == 2].index)].sort_values(["participant", "sess"])
w = pairs.groupby("participant").agg(session_1=("session", "first"), session_2=("session", "last"),
                                      resp_1=("resp", "first"), resp_2=("resp", "last"),
                                      z_1=("z", "first"), z_2=("z", "last")).reset_index()
r1, r2 = w.resp_1.values.astype(bool), w.resp_2.values.astype(bool)
tab = np.array([[np.sum(~r1 & ~r2), np.sum(~r1 & r2)], [np.sum(r1 & ~r2), np.sum(r1 & r2)]])
odds, p_fisher = fisher_exact(tab)
po = np.mean(r1 == r2)
pe = r1.mean() * r2.mean() + (1 - r1.mean()) * (1 - r2.mean())
kappa = (po - pe) / (1 - pe)

lab = m.resp.values.astype(bool)
pos = {p: np.flatnonzero(m.participant.values == p) for p in w.participant}
i1 = np.array([pos[p][0] for p in w.participant]); i2 = np.array([pos[p][1] for p in w.participant])
rng = np.random.default_rng(SEED)
null = np.array([(L[i1] == L[i2]).sum() for L in (rng.permutation(lab) for _ in range(N_PERM))])
obs = int((r1 == r2).sum())
p_perm = (np.sum(null >= obs) + 1) / (N_PERM + 1)
rho, p_rho = spearmanr(w.z_1, w.z_2)

rows = [
    ("n_sessions", len(m)), ("n_participants", len(n_per)), ("n_pairs", len(w)),
    ("n_single_session", int((n_per == 1).sum())),
    ("n_single_session_responder", int(m[m.participant.isin(n_per[n_per == 1].index)].resp.sum())),
    ("n_concordant", obs), ("n_both_resp", int(tab[1, 1])), ("n_both_non", int(tab[0, 0])),
    ("n_resp_to_non", int(tab[1, 0])), ("n_non_to_resp", int(tab[0, 1])),
    ("kappa", kappa), ("fisher_odds_ratio", odds), ("fisher_p", p_fisher),
    ("perm_n", N_PERM), ("perm_seed", SEED), ("perm_null_mean_concordant", null.mean()),
    ("perm_p_one_sided", p_perm),
    ("spearman_rho_z", rho), ("spearman_p_z", p_rho),
]
for name, df in (("source_data", w), ("metadata", pd.DataFrame(rows, columns=["key", "value"]))):
    out = SD / f"{STEM}_{name}.csv"; tmp = out.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False); os.replace(tmp, out)
print(f"[{STEM}] build_id {BUILD_ID}: {obs}/{len(w)} concordant ({tab[1,1]} RR, {tab[0,0]} NN), kappa {kappa:.3f}, "
      f"Fisher p {p_fisher:.4f} (OR {odds:.2f}), perm p {p_perm:.5f}, Spearman z rho {rho:.3f} p {p_rho:.2e}")
