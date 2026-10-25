"""Figure 4 — Shared locomotor representation across imagined and real walking

panel c - locking strength |c| INDEXES decodability (not causes), condition-stratified

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and does not fit the decoder. It fits a least-squares regression
for the statistics reported with that panel or table.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _c in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    if Path(_c).exists():
        fm.fontManager.addfont(_c)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS
PS.apply(plt)
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
import analysis_config as ec

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/decoder_output"
import os as _os
R_CONV = _os.environ.get("GAIT_R_CONV", "trial_median")
if R_CONV not in ("session_mean", "trial_median"):
    raise SystemExit(f"GAIT_R_CONV must be 'session_mean' or 'trial_median', got {R_CONV!r}")
TM_CSV = ec.sd("fig4_trialmedian_source_data.csv")
CPLX = {
    "IW": R / "diag_harmonic_phase_responder_split/persession_harmonic_phase_cycle9_complex.csv",
    "RW": R / "diag_real_walking_persession/persession_harmonic_phase_cycle9_complex.csv",
    "REST": R / "diag_rest_yoked/persession_harmonic_phase_cycle9_complex.csv",
}
FUSION = {
    "IW": ec.path("h_iw_summary_csv"),
    "RW": ec.path("h_rw_summary_csv"),
}
for _k, _p in CPLX.items():
    ec.check_tag(_p, "neural", f"|c| input {_k}")
for _k, _p in FUSION.items():
    ec.check_tag(_p, "decoder", f"fusion r input {_k}")
import project_palette as P
IW_Y, RW_R = P.IW_HUE, P.RW_HUE
COL = {"IW": IW_Y, "RW": RW_R, "REST": P.REST}
EDGE = {"IW": "none", "RW": "none", "REST": P.REST_EDGE}
LAB = {"IW": "imagined (IW)", "RW": "real walking (RW)", "REST": "rest (null)"}


def session_absc(path, ks=(1, 2, 3)):
    """per-session phase-locking strength = mean |complex| over k1+k2+k3 x channels x cycles."""
    d = pd.read_csv(path); d = d[d.harmonic_k.isin(ks)]
    d["absc"] = np.hypot(d["real"], d["imag"])
    return d.groupby("session")["absc"].mean()


def session_r(path):
    return pd.read_csv(path).groupby("session")["rpt_calib"].mean()


absc = {k: session_absc(p) for k, p in CPLX.items()}
if R_CONV == "trial_median":
    _tm = pd.read_csv(TM_CSV).set_index("session")
    r = {"IW": _tm["iw_median"], "RW": _tm["rw_median"]}
else:
    r = {"IW": session_r(FUSION["IW"]), "RW": session_r(FUSION["RW"])}


def xsess_heldout(ac, rv):
    """Cross-session held-out Spearman rho: |c|(session A) -> decode r(session B), same subject, both directions."""
    m = pd.DataFrame({"absc": ac, "r": rv}).dropna()
    m["subj"] = [s.split("_")[0] for s in m.index]
    pts = []
    for _, g in m.groupby("subj"):
        ss = list(g.index)
        for i in range(len(ss)):
            for j in range(len(ss)):
                if i != j:
                    pts.append((g.loc[ss[i], "absc"], g.loc[ss[j], "r"]))
    if len(pts) < 4:
        return float("nan"), 0
    pts = np.array(pts)
    nsub = sum(1 for _, g in m.groupby("subj") if len(g) >= 2)
    rho, p = stats.spearmanr(pts[:, 0], pts[:, 1])
    return float(rho), float(p), nsub, len(pts)


XCOL = "abs_c_rest_corrected"
absc_corr = {cond: absc[cond] - absc["REST"].reindex(absc[cond].index) for cond in ("IW", "RW")}

XSESS_RHO, XSESS_P, XSESS_NSUB, XSESS_NPAIR = xsess_heldout(absc_corr["IW"], r["IW"])

rows = []
for cond in ["IW", "RW", "REST"]:
    for s, c in absc[cond].items():
        rr = float(r[cond].get(s, np.nan)) if cond in r else float("nan")
        rc = float(absc_corr[cond].get(s, np.nan)) if cond in absc_corr else float("nan")
        rows.append(dict(condition=cond, session=s, abs_c=c, abs_c_rest_corrected=rc, decode_r=rr,
                         plotted=(cond in ("IW", "RW"))))
df = pd.DataFrame(rows)
dfp = df[df.plotted]
assert dfp[XCOL].notna().all(), "a plotted session has no rest control"

fig, ax = PS.panel_axes(plt, left_mm=15.5, bottom_mm=13.0)
_S_PT, _S_MEAN, _LW_DOT, _LW_MEAN, _LW_FIT = ((5, 26, 0.3, 0.6, 1.3))
for cond in ["IW", "RW"]:
    g = dfp[dfp.condition == cond]
    ax.scatter(g[XCOL], g.decode_r, s=_S_PT, c=COL[cond], alpha=0.75,
               edgecolors="#1A1A1A", linewidths=_LW_DOT, label=LAB[cond], zorder=3)
    mx, my = g[XCOL].mean(), g.decode_r.mean()
    ax.scatter([mx], [my], s=_S_MEAN, c=COL[cond], edgecolors="#1A1A1A", linewidths=_LW_MEAN, zorder=5)
_fx = dfp[XCOL].to_numpy(float)
_fy = dfp.decode_r.to_numpy(float)
_fsub = dfp.session.str.split("_sess").str[0].to_numpy()
bx = np.polyfit(_fx, _fy, 1)
_r2 = float(np.corrcoef(np.polyval(bx, _fx), _fy)[0, 1] ** 2)
_B = 1000


def _cluster_boot_fit(x, y, sub, grid):
    """OLS of y on x + 95% band from a cluster bootstrap over the participants in `sub`
    (1000 draws, seed 20260901): resampling participants keeps their sessions together."""
    rng = np.random.default_rng(20260901)
    subs = np.unique(sub)
    idx_by_sub = {u: np.flatnonzero(sub == u) for u in subs}
    draws = np.empty((_B, grid.size))
    for i in range(_B):
        pick = np.concatenate([idx_by_sub[u] for u in rng.choice(subs, subs.size, replace=True)])
        draws[i] = np.polyval(np.polyfit(x[pick], y[pick], 1), grid)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return lo, hi, subs.size


FIT_BY_STATE = ec.sd("fig4d_locking_indexes_decoding_fit_by_state.csv")
STATE_FIT, _fit_rows = {}, []
for cond in ["IW", "RW"]:
    m = dfp.condition.to_numpy() == cond
    x, y, sub = _fx[m], _fy[m], _fsub[m]
    bs = np.polyfit(x, y, 1)
    r2s = float(np.corrcoef(np.polyval(bs, x), y)[0, 1] ** 2)
    rho_s, p_s = stats.spearmanr(x, y)
    grid = np.linspace(x.min(), x.max(), 200)
    lo, hi, nsub = _cluster_boot_fit(x, y, sub, grid)
    ax.fill_between(grid, lo, hi, color=COL[cond], alpha=0.20, lw=0, zorder=2)
    ax.plot(grid, np.polyval(bs, grid), color=COL[cond], lw=_LW_FIT, ls=(0, (4, 2)), zorder=4)
    STATE_FIT[cond] = dict(slope=bs[0], intercept=bs[1], r2=r2s, rho=float(rho_s), p=float(p_s),
                           n=int(m.sum()), nsub=nsub)
    _fit_rows.append(pd.DataFrame({"condition": cond, XCOL: grid, "fit_decode_r": np.polyval(bs, grid),
                                   "ci95_lo": lo, "ci95_hi": hi}))
    print(f"fit {cond}: r = {bs[0]:+.3f}*|c| {bs[1]:+.3f}, R2 = {r2s:.3f}, rho = {rho_s:.3f} "
          f"(p = {p_s:.3g}), n = {m.sum()} sessions ({nsub} participants, {_B} cluster-bootstrap draws)")
if not ec.IS_DEFAULT:
    ec.check_anchor("fig4c_state_rho_iw", STATE_FIT["IW"]["rho"], "Fig. 4c IW state rho")
    ec.check_anchor("fig4c_state_rho_rw", STATE_FIT["RW"]["rho"], "Fig. 4c RW state rho")
pd.concat(_fit_rows).to_csv(FIT_BY_STATE, index=False, float_format="%.6f")
PS.stat_note(ax, [f"IW $\\it{{\\rho}}$ = {STATE_FIT['IW']['rho']:.2f}",
                  f"RW $\\it{{\\rho}}$ = {STATE_FIT['RW']['rho']:.2f}"], loc="lower right")
ax.set_xlabel(r"Gait-phase locking |$c$|")
ax.set_ylabel("Decoding accuracy ($r$)")
b = bx
_band_top = float(pd.concat(_fit_rows)["ci95_hi"].max())
ax.set_ylim(-0.05, max(0.95, float(dfp.decode_r.max()) + 0.05, _band_top + 0.01))
PS.finish(ax)
OUT = ec.scoped(ROOT / "figures_out/fig4/fig4d_locking_indexes_decoding.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=PS.DPI, facecolor="white")
figL = plt.figure(figsize=(46 * PS.MM, 14 * PS.MM), dpi=PS.DPI)
from matplotlib.lines import Line2D
_h = [Line2D([0], [0], marker="o", color="none", mfc=COL[c], mec="#1A1A1A", mew=0.6, ms=5.1, label=LAB[c])
      for c in ("IW", "RW")]
_lg = figL.legend(handles=_h, loc="center", fontsize=PS.NOTE_PT, frameon=True, edgecolor="#1A1A1A",
                  facecolor="white", fancybox=False, framealpha=1.0,
                  handletextpad=0.4, labelspacing=0.35, ncol=1, borderpad=0.5)
_lg.get_frame().set_linewidth(0.5)
figL.savefig(OUT.with_name("fig4d_locking_indexes_decoding_legend.png"), bbox_inches="tight",
             pad_inches=0.02, dpi=PS.DPI, facecolor="white")
plt.close(figL)
plt.close(fig)

print("per-condition mean |c| / r:")
for cond in ["IW", "RW", "REST"]:
    g = df[df.condition == cond]
    print(f"  {cond:5s} n={len(g)} |c|={g.abs_c.mean():.3f}  rest-corrected={g[XCOL].mean():.3f}  "
          f"r={g.decode_r.mean():.3f}")
print(f"regression slope={b[0]:.3f} | cross-session rho={XSESS_RHO:+.4f} p={XSESS_P:.3g} "
      f"(n_subjects={XSESS_NSUB}, n_pairs={XSESS_NPAIR})")
SD = ec.sd("fig4d_locking_indexes_decoding_source_data.csv")
df.to_csv(SD, index=False)

PW = ec.sd("fig4d_locking_indexes_decoding_wide.csv")
wide = pd.DataFrame({XCOL: dfp[XCOL].values})
for cond, col in (("IW", "r_imagined"), ("RW", "r_realwalking")):
    wide[col] = np.where(dfp.condition.values == cond, dfp.decode_r.values, np.nan)
wide.to_csv(PW, index=False, na_rep="")
PWM = ec.sd("fig4d_locking_indexes_decoding_means.csv")
cm = dfp.groupby("condition").agg(x=(XCOL, "mean"), y=("decode_r", "mean"))
pd.DataFrame(
    [dict(row_type="condition_mean", label=c, x=float(cm.loc[c, "x"]), y=float(cm.loc[c, "y"]))
     for c in ("IW", "RW")]
).to_csv(PWM, index=False, na_rep="")
print("wrote", PW.name, "+", PWM.name)
FITMETA = []
_SUP = ""
FITMETA = [(k, _SUP + v) for k, v in FITMETA] + [
    ("panel_c_fit_by_state", "one OLS of decode r on |c| within each state (IW, RW), each drawn over "
                             "that state's own |c| range; 95% CI band per state from a cluster "
                             "bootstrap over the participants present in that state, 1000 draws, "
                             f"seed 20260901; curves in {FIT_BY_STATE.name}"),
    ("IW_spearman_rho", f"{STATE_FIT['IW']['rho']:.4f}"),
    ("IW_spearman_p", f"{STATE_FIT['IW']['p']:.4g}"),
    ("RW_spearman_rho", f"{STATE_FIT['RW']['rho']:.4f}"),
    ("RW_spearman_p", f"{STATE_FIT['RW']['p']:.4g}"),
    ("n_sessions_per_state", STATE_FIT["IW"]["n"] if STATE_FIT["IW"]["n"] == STATE_FIT["RW"]["n"]
     else f"IW {STATE_FIT['IW']['n']}; RW {STATE_FIT['RW']['n']}"),
    ("IW_ols_slope", f"{STATE_FIT['IW']['slope']:.4f}"),
    ("IW_ols_r2", f"{STATE_FIT['IW']['r2']:.4f}"),
    ("RW_ols_slope", f"{STATE_FIT['RW']['slope']:.4f}"),
    ("RW_ols_r2", f"{STATE_FIT['RW']['r2']:.4f}"),
]

from datetime import date as _date


def _libvers(_mods):
    """Runtime library versions, so a re-run that moves the last digits is explainable."""
    import importlib as _il
    import sys as _sys
    _out = ["python " + _sys.version.split()[0]]
    for _m in _mods:
        try:
            _out.append(_m + " " + _il.import_module(_m).__version__)
        except Exception:
            _out.append(_m + " unavailable")
    return "; ".join(_out)


TM_BUILD_ID = "undetermined_from_inputs"
if R_CONV == "trial_median":
    _tmm = TM_CSV.with_name(TM_CSV.name.replace("_source_data.csv", "_metadata.csv"))
    if _tmm.exists():
        _m = pd.read_csv(_tmm).set_index("key")["value"]
        TM_BUILD_ID = str(_m.get("build_id", "undetermined_from_inputs"))
    for _k, _p in CPLX.items():
        ec.check_tag(_p, "neural", f"|c| input {_k}")
    if TM_BUILD_ID not in ("undetermined_from_inputs", ec.TAG):
        raise SystemExit(f"INPUT CHECK: the |c| inputs are {ec.NEURAL_TAG} (neural) but the trial-median r "
                         f"input is {TM_BUILD_ID}, not the decoder build_id {ec.TAG}; one cohort+version per figure.")

MD = ec.sd("fig4d_locking_indexes_decoding_metadata.csv")
_ERA_COHORT = (" healthy cohort (main, tag main)" if ec.IS_DEFAULT
               else f"healthy cohort (decoder tag {ec.TAG}; |c| tag {ec.NEURAL_TAG})")
pd.DataFrame([
    ("panel", "Fig. 4 panel c - locking strength |c| indexes decodability"),
    ("cohort", f"{_ERA_COHORT}; {df[df.condition == 'IW'].session.nunique()} "
               "sessions per condition"),
    ("locking_metric", "mean |complex| over k1+k2+k3 x channels x cycle points (band-unselected)"),
    ("x_axis", "abs_c_rest_corrected = abs_c of the task minus abs_c of the same session's rest control "
               "(same metric); abs_c = uncorrected value, kept for the comparison with rest"),
] + ([] if R_CONV == "session_mean" else [
    ("generated", _date.today().isoformat()),
    ("library_versions", _libvers(("numpy", "scipy", "pandas", "matplotlib"))),
    ("r_convention", R_CONV),
    ("input_trial_median", str(TM_CSV.relative_to(ROOT)).replace("\\", "/")),
    ("input_trial_median_build_id", TM_BUILD_ID),
    ("r_convention_definition",
     "per-session median over trials of the per-trial mean over the 6 joints "
     "(= Fig.2c 'All' median); source " + TM_CSV.relative_to(ROOT).as_posix()),
    ("cohort_median_of_session_r_IW", f"{float(np.nanmedian(r['IW'].values)):.6f}"),
    ("cohort_median_of_session_r_RW", f"{float(np.nanmedian(r['RW'].values)):.6f}"),
] ) + [
    ("xsess_heldout_rho_pooled_IW", f"{XSESS_RHO:+.4f}"),
    ("xsess_heldout_p_pooled_IW", f"{XSESS_P:.4g}"),
    ("xsess_heldout_n_subjects", XSESS_NSUB),
    ("xsess_heldout_n_pairs", XSESS_NPAIR),
    ("xsess_heldout_method",
     "Spearman over (|c| session A, decode r session B), same subject, both directions; "
     "|c| and r come from different sessions"
     ""),
    ("xsess_heldout_scope", "full-cohort healthy, imagined walking, pooled k1+k2+k3 rest-corrected |c|"),
] + FITMETA, columns=["key", "value"]).to_csv(MD, index=False)
print("wrote", OUT, "+", SD, "+", MD)
