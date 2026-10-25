"""Shared figure settings

The Figure 7 simulation tables reduced to the scored trials: keep_sim keeps the rows of a scored
imagined-walking trial (03_figures/scoring_mask.csv) and the static rows (trial_idx < 0)

Imported by the figure scripts; it renders no panel of its own.
"""

import analysis_config as ec


def keep_sim(df, col="trial_idx"):
    if ec.SCORE_MASK is None:
        return df
    m = [int(float(t)) < 0 or ec.keep_trial(s, "IW", t) for s, t in zip(df["session"], df[col])]
    return df.loc[m]
