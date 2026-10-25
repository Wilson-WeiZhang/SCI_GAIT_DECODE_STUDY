"""Supplementary item

Cadence sensitivity of the decoder: trial-to-trial variation of the reference walker's cadence around the fold
calibration, and its association with decoding accuracy

Reads:
  04_source_data/fig3/fig3e_locking_boxwhisker_source_data.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and does not fit the decoder. It fits a linear mixed model
for the statistics reported with that panel or table.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import signal, stats

import analysis_config as ec
from safe_write import safe_write

ec.require(ec.MAIN)
ROOT = Path(__file__).resolve().parents[1]
STEM = "sup_cadence_sensitivity"
FS = 250.0
GRID = np.round(np.arange(0, 4.0 + 1e-9, 0.01), 2)
F0_BAND, HARM_HALF, DEMOD_HALF = (0.45, 1.15), 0.25, 0.18
RESPONDER_SD = ROOT / "04_source_data/fig3/fig3e_locking_boxwhisker_source_data.csv"


def trace_psd(x):
    x = x[np.isfinite(x)]
    if len(x) < 2 * FS:
        return None
    x = x - x.mean()
    win = int(round(FS * min(4.0, max(2.0, len(x) / FS / 2))))
    f, p = signal.welch(x, fs=FS, window="hamming", nperseg=win, noverlap=win // 2, nfft=8192, detrend=False)
    m = (f >= 0.05) & (f <= 4.0)
    return np.interp(GRID, f, p / np.trapz(p[m], f[m]))


def trial_psd(y):
    ps = [p for p in (trace_psd(y[:, j]) for j in range(y.shape[1])) if p is not None]
    return np.mean(ps, axis=0) if ps else None


def argmax_in(p, lo, hi):
    m = (GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)
    return float(GRID[m][np.argmax(p[m])])


def stride_freq(y, f0):
    fr = []
    for j in (0, 3):
        x = y[:, j]
        if np.isfinite(x).sum() < 2 * FS:
            continue
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        pk, _ = signal.find_peaks(x, distance=int(0.6 * FS / f0), prominence=0.3 * np.std(x))
        if len(pk) >= 3:
            fr.append(FS / np.median(np.diff(pk)))
    return float(np.mean(fr)) if fr else np.nan


def cadence_table() -> pd.DataFrame:
    rows = []
    dumps = sorted(ec.path("h_iw_dump_dir").glob("*_dump.h5"))
    assert len(dumps) == 80, len(dumps)
    for path in dumps:
        sess = path.name.replace("_dump.h5", "")
        with h5py.File(path, "r") as f:
            Y = f["rows/Y_true"][:]
            off, nr = f["trials/row_offset"][:], f["trials/n_rows"][:]
            tidx, fold = f["trials/trial_idx"][:], f["trials/fold"][:]
        psds = [trial_psd(Y[o:o + n]) for o, n in zip(off, nr)]
        for k in np.unique(fold):
            p = np.mean([q for q, fo in zip(psds, fold) if fo != k and q is not None], axis=0)
            f0 = argmax_in(p, *F0_BAND)
            fc = [argmax_in(p, h * f0 - HARM_HALF, h * f0 + HARM_HALF) for h in (1, 2, 3)]
            for i in np.where(fold == k)[0]:
                ft = stride_freq(Y[off[i]:off[i] + nr[i]], f0)
                rows.append(dict(session=sess, trial_idx=int(tidx[i]), fold=int(k), f0_fold=f0,
                                 fc_k1=fc[0], fc_k2=fc[1], fc_k3=fc[2], f_trial=ft,
                                 **{f"offset_k{h}": abs(h * ft - fc[h - 1]) for h in (1, 2, 3)}))
    return pd.DataFrame(rows)


def fit_lme(t: pd.DataFrame):
    best = None
    for meth in ("lbfgs", "powell", "cg"):
        try:
            r = smf.mixedlm("r ~ abs_dev", t, groups=t["participant"], re_formula="~1",
                            vc_formula={"session": "0 + C(session)"}).fit(reml=True, method=meth, maxiter=4000)
        except Exception:
            continue
        if np.isfinite(r.llf) and (best is None or r.llf > best[0].llf):
            best = (r, meth)
    assert best is not None, "no optimiser converged"
    return best


def main() -> None:
    warnings.filterwarnings("ignore")
    d = cadence_table()
    d["participant"] = d.session.str.extract(r"^(subj\d+)_")[0]
    d["dev"] = d.f_trial - d.f0_fold
    d["abs_dev"] = d.dev.abs()
    d["inside_all_bands"] = (d[["offset_k1", "offset_k2", "offset_k3"]] <= DEMOD_HALF).all(axis=1)
    assert len(d) == 4756 and d.f_trial.notna().all() and d.participant.nunique() == 42, (len(d), d.f_trial.isna().sum())

    r = pd.read_csv(ec.sd_in("fig2g_trial_r_long.csv"))[["session", "trial_idx", "All"]].rename(columns={"All": "r"})
    assert len(r) == 3172, len(r)
    d = d.merge(r, on=["session", "trial_idx"], how="left")
    d["scored"] = d.r.notna()
    assert d.scored.sum() == 3172
    resp = pd.read_csv(RESPONDER_SD)[["session", "responder"]]
    assert len(resp) == 80 and resp.responder.sum() == 55
    d = d.merge(resp, on="session", how="left")

    s = d[d.scored].copy()
    lme, meth = fit_lme(s)
    ci = lme.conf_int().loc["abs_dev"]
    rho = s.groupby("session").apply(lambda g: stats.spearmanr(g.abs_dev, g.r)[0])
    sess = d.groupby("session").agg(sd_f=("f_trial", "std"), resp=("responder", "first"))
    mw = stats.mannwhitneyu(sess[sess.resp].sd_f, sess[~sess.resp].sd_f, alternative="two-sided")
    q = d.abs_dev.quantile([0.5, 0.9])
    rel = d.abs_dev / d.f0_fold

    cols = ["participant", "session", "trial_idx", "fold", "f0_fold", "f_trial", "dev",
            "offset_k1", "offset_k2", "offset_k3", "inside_all_bands", "scored", "r", "responder"]
    safe_write(ec.sd(f"{STEM}_source_data.csv"), d[cols].to_csv(index=False, float_format="%.6f", lineterminator="\n"), newline="\n")
    meta = [
        ("analysis", "Reference-walker cadence per imagined-walking trial versus the decoder's fold calibration; "
                     "trial-level association with decoding accuracy"),
        ("n_trials_cadence", len(d)), ("n_trials_scored", int(d.scored.sum())),
        ("n_sessions", d.session.nunique()), ("n_participants", d.participant.nunique()),
        ("trial_cadence", "1 / median interval between hip-flexion peaks, left and right hip averaged"),
        ("fold_calibration", "f0 = PSD peak 0.45-1.15 Hz of the fold's training trials; fc_k = peak in k f0 +/- 0.25 Hz"),
        ("band_test", "harmonic k inside when |k f_trial - fc_k| <= 0.18 Hz (demodulator half-power cut-off)"),
        ("f0_fold_median_hz", f"{d.f0_fold.median():.3f}"),
        ("abs_deviation_median_hz", f"{q[0.5]:.4f}"), ("abs_deviation_p90_hz", f"{q[0.9]:.4f}"),
        ("abs_deviation_median_pct", f"{100 * rel.median():.2f}"), ("abs_deviation_p90_pct", f"{100 * rel.quantile(0.9):.2f}"),
        ("within_session_sd_median_hz", f"{sess.sd_f.median():.4f}"),
        ("trials_all_harmonics_inside", f"{int(d.inside_all_bands.sum())}/{len(d)}"),
        ("trials_all_harmonics_inside_pct", f"{100 * d.inside_all_bands.mean():.2f}"),
        ("trials_k1_inside", f"{int((d.offset_k1 <= DEMOD_HALF).sum())}/{len(d)}"),
        ("trials_k2_inside", f"{int((d.offset_k2 <= DEMOD_HALF).sum())}/{len(d)}"),
        ("trials_k3_inside", f"{int((d.offset_k3 <= DEMOD_HALF).sum())}/{len(d)}"),
        ("model", "r ~ |deviation| + (1|participant) + (1|participant:session); REML; scored trials"),
        ("lme_beta_r_per_hz", f"{lme.params['abs_dev']:.4f}"),
        ("lme_ci95", f"{ci[0]:.4f} to {ci[1]:.4f}"), ("lme_z", f"{lme.tvalues['abs_dev']:.3f}"),
        ("lme_p", f"{lme.pvalues['abs_dev']:.3e}"),
        ("lme_r_change_at_median_deviation", f"{lme.params['abs_dev'] * d.loc[d.scored, 'abs_dev'].median():.4f}"),
        ("lme_optimizer", meth), ("lme_converged", bool(lme.converged)),
        ("within_session_spearman_median", f"{rho.median():.3f}"),
        ("within_session_spearman_iqr", f"{rho.quantile(0.25):.3f} to {rho.quantile(0.75):.3f}"),
        ("within_session_spearman_wilcoxon_p", f"{stats.wilcoxon(rho.dropna())[1]:.3e}"),
        ("cadence_sd_median_responder_hz", f"{sess[sess.resp].sd_f.median():.4f}"),
        ("cadence_sd_median_nonresponder_hz", f"{sess[~sess.resp].sd_f.median():.4f}"),
        ("cadence_sd_responder_vs_nonresponder", f"Mann-Whitney U {mw.statistic:.1f}, p {mw.pvalue:.3f}, n = 55 and 25 sessions"),
        ("source_script", "03_figures/sup_cadence_sensitivity.py"),
    ]
    safe_write(ec.sd(f"{STEM}_metadata.csv"), pd.DataFrame(meta, columns=["key", "value"]).to_csv(index=False, lineterminator="\n"), newline="\n")
    for k, v in meta[11:]:
        print(f"{k:<42} {v}")
    print(f"-> {ec.sd(STEM).parent}/{STEM}_{{source_data,metadata}}.csv")


if __name__ == "__main__":
    main()
