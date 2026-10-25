"""Figure 7 — Intent gate and physics simulation of the decoded gait phase

_sim_trajectories — Fig. 7 four-condition SIMULATED-TRAJECTORY panel: forward hip displacement x(t) of every rollout, one small axes per condition on a shared 0-3 s window

Reads:
  simulation and gate output in 02_decoding/simulation/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and does not fit the decoder. It fits a linear mixed model and a least-squares regression
for the statistics reported with that panel or table.
"""

import csv
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import project_palette as P
os.environ.setdefault("GAIT_F7_SIM", "causal_from2s")
import analysis_config as ec
import f7_scored_trials as F7_SCORED
ec.require("main", "main")

TRACES = os.path.join(ROOT, "02_decoding", "simulation", "simulation_rollouts", "traces_x.csv.gz")
OUT_PNG = os.path.join(ROOT, "figures_out", "fig7", "fig7_sim_trajectories.png")
OUT_PNG_L = os.path.join(ROOT, "figures_out", "fig7", "fig7_sim_trajectories_left.png")
OUT_PNG_R = os.path.join(ROOT, "figures_out", "fig7", "fig7_sim_trajectories_right.png")
SD = os.path.join(ROOT, "04_source_data", "fig7", "fig7_sim_trajectories_source_data.csv")
ST = os.path.join(ROOT, "04_source_data", "fig7", "fig7_sim_trajectories_stats.csv")
MD = os.path.join(ROOT, "04_source_data", "fig7", "fig7_sim_trajectories_metadata.csv")

def _md_fix(v):
    """Metadata value rewriter; the identity at the default support level (L1p)."""
    return v


SUPPORT = os.environ.get("GAIT_SUPPORT", "L1p")
assert SUPPORT in ("L1p", "L3b"), f"GAIT_SUPPORT must be L1p or L3b, got {SUPPORT!r}"
SUPPORT_MD = None
if SUPPORT == "L3b":
    _CAND = os.path.join(ROOT, "figures_out", "fig7", "_candidates")
    _SDL3B = os.path.join(_CAND, "sd_L3b")
    os.makedirs(_SDL3B, exist_ok=True)

    def _variant_png(p):
        return os.path.join(_CAND, os.path.splitext(os.path.basename(p))[0] + "_L3b.png")

    def _variant_source_data(p):
        return os.path.join(_SDL3B, os.path.basename(p))

    TRACES = TRACES.replace(os.sep + "simulation_rollouts" + os.sep,
                            os.sep + "simulation_rollouts_variant" + os.sep)
    OUT_PNG = _variant_png(OUT_PNG)
    OUT_PNG_L, OUT_PNG_R = _variant_png(OUT_PNG_L), _variant_png(OUT_PNG_R)
    SD, ST, MD = _variant_source_data(SD), _variant_source_data(ST), _variant_source_data(MD)

    def _md_fix(v):
        return (v.replace("simulation_rollouts/", "simulation_rollouts_variant/")
                if isinstance(v, str) else v)
    SUPPORT_MD = ("support_level",
                  "an alternative support level; not the published build")

GATE = "3s"
assert SUPPORT != "L3b", "GAIT_SUPPORT=L3b is not available in this repository"
TMAX = 3.0
if ec.F7_SIM == "causal":
    TMAX -= 1.0
XTICKS = [int(v) if float(v).is_integer() else v for v in (0, TMAX / 2, TMAX)]
DRAW_FULL = os.environ.get("GAIT_SIM_DRAW_FULL", "0") == "1"
XMAX = TMAX
GATE_MD = None
if GATE == "3s":
    TRACES = TRACES.replace(os.sep + "simulation_rollouts" + os.sep,
                            os.sep + "simulation_rollouts_gate3s" + os.sep)
    GATE_MD = ("usable_duration_gate",
               "3 s (PHYS_MIN_DUR_S=3; a trial is included when at least 3 s of decoder output remain after the decoder's 4 s start-up)")

if ec.SCOPED:
    TRACES = str(ec.sim_in(TRACES))
    OUT_PNG, OUT_PNG_L, OUT_PNG_R, SD, ST, MD = (str(ec.scoped(p)) for p in (OUT_PNG, OUT_PNG_L, OUT_PNG_R, SD, ST, MD))

WALKER = os.environ.get("GAIT_SIM_WALKER", "matched" if GATE == "3s" else "unmatched")
assert WALKER in ("matched", "unmatched"), f"GAIT_SIM_WALKER must be matched or unmatched, got {WALKER!r}"
MATCHED = WALKER == "matched"
if not MATCHED and not ec.IS_DEFAULT:
    raise SystemExit("fig7_sim_trajectories: the unmatched-walker metadata (META rows "
                     "common_window and input_selfcheck) is not available for the published decoder")
WALKER_CONDITION = "wphase_matched" if MATCHED else "wphase"
WALKER_TRACES = os.path.join(os.path.dirname(TRACES), "walker_matched_traces_x.csv.gz")
WALKER_RHY = os.path.join(os.path.dirname(TRACES), "walker_matched_rhythm.csv")

TEST = os.environ.get("GAIT_SIM_TEST", "lme" if MATCHED else "perm")
assert TEST in ("lme", "perm"), f"GAIT_SIM_TEST must be lme or perm, got {TEST!r}"
assert not (TEST == "lme" and not MATCHED), "GAIT_SIM_TEST=lme needs the window-matched (paired) walker"
LME = TEST == "lme"
TEST_RULE = ("Each contrast is tested on the per-trial "
             "paired difference d = A - B (pairs matched on session x trial_idx; in a vs-zero test d is the value "
             "itself), NaN dropped, with a linear mixed model d ~ 1 + (1 | session) (session ids are unique across "
             "the two patients), fitted by REML in statsmodels MixedLM (optimizers lbfgs, powell and cg; of the fits "
             "that converge with a finite intercept SE and log-likelihood, the one with the largest log-likelihood is "
             "kept; session variance at the REML boundary 0 for every optimizer -> its intercept-only limit, "
             "labelled in the optimizer column): the fixed intercept is the mean difference and the session random "
             "intercept accounts for trials within a session being correlated. Session alone, not nested in "
             "patient: with two patients the patient variance is not estimable. Per-patient "
             "rows fit the same model to that patient's trials (4 and 5 sessions). Its Wald p is p_used for every "
             "contrast; Shapiro-Wilk on the residuals and the two-sided Wilcoxon signed-rank p are reported in the "
             "Source Data only. Stars *** / ** / * / ns at 0.001 / 0.01 / 0.05 on p_used. Every box, line and "
             "in-figure number still shows MEDIANS, descriptively; the paired t-test p and the median of d are "
             "recorded in the stats CSV descriptively and never set a star.")
FDR_STARS = LME and ec.F7_SIM == "causal_from2s"
if FDR_STARS:
    TEST_RULE = TEST_RULE.replace(
        "Stars *** / ** / * / ns at 0.001 / 0.01 / 0.05 on p_used.",
        "Stars *** / ** / * / ns at 0.001 / 0.01 / 0.05: the stats column star on p_used; the stars drawn in panel c "
        "(stats column star_drawn) on p_fdr_bh, Benjamini-Hochberg over the seven displacement tests (the first seven "
        "rows of the stats CSV).")

FS = 1.35
MIN_N = 5
GRID_W, GRID_H = 4.4, 3.0
THIN_LW = 0.55
ALPHA_BASE = 0.38
ALPHA_FLOOR = 0.12
MED_LW = 2.4
MED_DARKEN = 0.30

plt.rcParams.update({"font.family": "Arial", "font.size": 6.5 * FS, "axes.linewidth": 1.0,
                     "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42})

ROWS = [(WALKER_CONDITION, "Walker", P.WALKER_HUE), ("dphase", "Decoded", P.IW_HUE),
        ("dphase_iscr", "CoordinationNull", "#9aa0a6"), ("dphase_uscr", "InPhaseNull", "#9aa0a6")]
DISPLAY = {"Walker": "Walker", "Decoded": "Decoded",
           "CoordinationNull": "Coordination null", "InPhaseNull": "In-phase null"}

d = F7_SCORED.keep_sim(pd.read_csv(TRACES))
N_WALKER_UNMATCHED = d[d.condition == "wphase"].groupby(["session", "trial_idx"]).ngroups
if MATCHED:
    d = pd.concat([d[d.condition != "wphase"], F7_SCORED.keep_sim(pd.read_csv(WALKER_TRACES))], ignore_index=True)
D_FULL = d.copy()
d = d[d.t_s <= TMAX + 1e-9]
if DRAW_FULL:
    XMAX = float(np.ceil(D_FULL.t_s.max() * 2) / 2)
    XTICKS = [int(v) for v in range(0, int(XMAX) + 1, 2)]
    lo, hi = D_FULL.x_m.min(), D_FULL.x_m.max()
else:
    lo, hi = d.x_m.min(), d.x_m.max()
pad = 0.05 * (hi - lo)

fig, axes = plt.subplots(2, 2, figsize=(GRID_W, GRID_H), dpi=300, sharex=True, sharey=True)
axes = axes.ravel()
_n = {condition: d[d.condition == condition].groupby(["session", "trial_idx"]).ngroups for condition, _, _ in ROWS}
_nmin = min(_n.values())

rep, sd_traces, sd_medians, sd_stats = [], [], [], []
END = {}
DRAW = {}
ANNOT = {}

ANNOT_X = 0.97
ANNOT_VA = {"Walker": "bottom", "Decoded": "bottom", "CoordinationNull": "top", "InPhaseNull": "top"}
ANNOT_Y = {"top": 0.93, "bottom": 0.04}


def _darken(hex_col, f):
    """Mix a hex colour f of the way toward project_palette.INK (hue kept, value lowered)."""
    a = [int(hex_col[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(P.INK[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x * (1 - f) + y * f) for x, y in zip(a, b))


def _draw_condition(ax, lab):
    """Draw one condition's axes. Called once for the combined 2 x 2 and once more for the split
    half-panel, so the split PNGs are the same lines with the same styling, not a re-analysis."""
    s, col, alpha, med = DRAW[lab]
    for _, tr in s.groupby(["session", "trial_idx"]):
        ax.plot(tr.t_s, tr.x_m, color=col, lw=THIN_LW, alpha=alpha, zorder=2)
    ax.plot(med.index, med.values, color=_darken(col, MED_DARKEN), lw=MED_LW, zorder=4,
            solid_capstyle="round")
    ax.axhline(0, color="#30343a", lw=0.7, ls=(0, (1.6, 1.6)), zorder=1)
    if DRAW_FULL:
        ax.axvline(TMAX, color="#30343a", lw=0.7, ls=(0, (1.6, 1.6)), zorder=1)
    ax.set_ylim(lo - pad, hi + pad)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=1.0)
    ax.text(0.015, 0.93, DISPLAY.get(lab, lab), transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5 * FS, color="#30343a")
    ax.text(ANNOT_X, ANNOT_Y[ANNOT_VA[lab]], ANNOT[lab], transform=ax.transAxes, ha="right",
            va=ANNOT_VA[lab], fontsize=6.5 * FS, color=P.INK)


for condition, lab, col in ROWS:
    s = d[d.condition == condition]
    alpha = max(ALPHA_BASE * (_nmin / _n[condition]), ALPHA_FLOOR)
    piv = s.pivot_table(index="t_s", columns=["session", "trial_idx"], values="x_m")
    n = piv.notna().sum(axis=1)
    med = piv.median(axis=1)[n >= MIN_N]
    s_draw = D_FULL[D_FULL.condition == condition] if DRAW_FULL else s
    DRAW[lab] = (s_draw, col, alpha, med)
    for _, tr in s_draw.groupby(["session", "trial_idx"]):
        sd_traces.append(tr.assign(condition_label=lab))
    ntr = s.groupby(["session", "trial_idx"]).ngroups
    end = s.sort_values("t_s").groupby(["session", "trial_idx"]).x_m.last()
    END[lab] = end
    rep.append(f"{lab:10s} n={ntr:3d}  x at {TMAX:.1f} s: median {end.median():+.2f} m  "
               f"IQR {end.quantile(.25):+.2f}..{end.quantile(.75):+.2f}  "
               f"backward {100 * (end < 0).mean():.0f}%")
    for t, v in med.items():
        sd_medians.append((lab, t, v, int(n.loc[t])))
    sd_stats.append((lab, ntr, alpha, end.median(), end.quantile(.25), end.quantile(.75),
                     100 * (end < 0).mean()))

B = 10000
SEED = 0
RNG = np.random.default_rng(SEED)


def perm_unpaired_median(x, y, key):
    """Walker vs Decoded: condition labels permuted WITHIN session, shared sessions only."""
    dx = x[["session", key]].dropna(); dy = y[["session", key]].dropna()
    sess = sorted(set(dx.session) & set(dy.session)); dx = dx[dx.session.isin(sess)]; dy = dy[dy.session.isin(sess)]
    v = np.r_[dx[key].astype(float), dy[key].astype(float)]; lab = np.r_[np.ones(len(dx)), np.zeros(len(dy))]; s = np.r_[dx.session, dy.session]
    obs = np.median(v[lab == 1]) - np.median(v[lab == 0]); idx = [np.where(s == k)[0] for k in sess]; c = 0
    for _ in range(B):
        l = lab.copy()
        for ii in idx: l[ii] = RNG.permutation(l[ii])
        c += abs(np.median(v[l == 1]) - np.median(v[l == 0])) >= abs(obs) - 1e-12
    return obs, (c + 1) / (B + 1), len(dx), len(dy), sess


def perm_paired_median(x, y, key):
    """Decoded vs a null: sign flip within trial on the per-trial difference."""
    A = x.set_index(["session", "trial_idx"])[key].astype(float); Bv = y.set_index(["session", "trial_idx"])[key].astype(float)
    d = (A - Bv).dropna().values; obs = np.median(d); c = 0
    for _ in range(B):
        c += abs(np.median(d * RNG.choice([-1, 1], len(d)))) >= abs(obs) - 1e-12
    return obs, (c + 1) / (B + 1), len(d), len(d), None


def perm_onesample_median(x, key):
    """endpoint displaced from zero at all? EXACTLY the paired machinery above with the null side set
    to 0, so the per-rollout endpoint itself is the "paired difference" and the permutation is the
    same within-trial sign flip on the median."""
    return perm_paired_median(x, x.assign(**{key: 0.0}), key)


def stars(p): return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def lme_paired_test(a, b, key, patient=None):
    """Per-trial paired difference d = a - b matched on (session, trial_idx); b None = the vs-zero test (d is
    a's value itself)."""
    A = a.set_index(["session", "trial_idx"])[key].astype(float)
    dd = (A if b is None else A - b.set_index(["session", "trial_idx"])[key].astype(float)).dropna()
    sess = dd.index.get_level_values("session").astype(str)
    df = pd.DataFrame({"d": dd.to_numpy(), "subject": sess.str[:5], "session": sess})
    if patient is not None:
        df = df[df.subject == patient].reset_index(drop=True)
    multi = df.session.nunique() >= 2
    fits = []
    for m in ("lbfgs", "powell", "cg") if multi else ():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = smf.mixedlm("d ~ 1", df, groups=df["session"], re_formula="~1").fit(reml=True, method=m, maxiter=2000)
        if f.converged and np.isfinite(f.bse_fe["Intercept"]) and np.isfinite(f.llf):
            fits.append((m, f))
    if fits:
        m, f = max(fits, key=lambda t: t[1].llf)
        beta, se, revar = float(f.fe_params["Intercept"]), float(f.bse_fe["Intercept"]), float(f.cov_re.iloc[0, 0])
    else:
        if multi and not _lme_at_boundary(df):
            raise RuntimeError(f"LME d ~ 1 + (1 | session) converged with no optimizer ({key}, n = {len(df)})")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = smf.ols("d ~ 1", df).fit()
        m = "ols (session variance 0)" if multi else "ols (one session: intercept only)"
        beta, se, revar = float(f.params["Intercept"]), float(f.bse["Intercept"]), 0.0
    p_lme, p_sw = float(f.pvalues["Intercept"]), float(sps.shapiro(f.resid).pvalue)
    p_wx = float(sps.wilcoxon(df.d, zero_method="wilcox").pvalue)
    p = p_lme
    return dict(n=len(df), n_subjects=df.subject.nunique(), n_sessions=df.session.nunique(),
                mean_difference_lme=beta, se_lme=se,
                p_lme=p_lme, session_variance=revar, optimizer=m, shapiro_p=p_sw,
                p_wilcoxon=p_wx, p_paired_t=float(sps.ttest_1samp(df.d, 0).pvalue),
                median_difference=float(np.median(df.d)),
                test_used="LME" if patient is None else "LME (within patient)", p_used=p, star=stars(p))


def _lme_at_boundary(df):
    """variance at the REML boundary (< 1e-4 of var(d)), i.e. the fit equals its intercept-only limit."""
    res = []
    for m in ("lbfgs", "powell", "cg"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = smf.mixedlm("d ~ 1", df, groups=df["session"], re_formula="~1").fit(reml=True, method=m, maxiter=2000)
        res.append(bool(f.converged) and bool(np.isfinite(f.llf))
                   and float(f.cov_re.iloc[0, 0]) < 1e-4 * float(np.var(df.d)))
    return all(res)

LME_COLS = ["comparison", "n", "n_subjects", "n_sessions", "mean_difference_lme", "se_lme", "p_lme",
            "session_variance", "optimizer", "shapiro_p", "p_wilcoxon", "p_paired_t", "median_difference",
            "test_used", "p_used", "star", "scheme"]


def lme_csv_row(comparison, r, scheme):
    """One stats-CSV row in LME_COLS order (differences 6 decimals, p values / SE / variance 6 significant)."""
    f6, g6 = "{:.6f}".format, "{:.6g}".format
    return [comparison, r["n"], r["n_subjects"], r["n_sessions"], f6(r["mean_difference_lme"]), g6(r["se_lme"]),
            g6(r["p_lme"]), g6(r["session_variance"]), r["optimizer"], g6(r["shapiro_p"]), g6(r["p_wilcoxon"]),
            g6(r["p_paired_t"]), f6(r["median_difference"]), r["test_used"], g6(r["p_used"]), r["star"], scheme]


LME_SCHEME = ("paired difference d = A - B per trial (session x trial_idx), NaN dropped; LME d ~ 1 + (1 | session), "
              "REML, session random intercept, intercept = mean difference, Wald p = p_used; Shapiro-Wilk and Wilcoxon "
              "reported in the Source Data only; star from p_used")


def intercept_paired_test(a, b, key, subject):
    """first 5 characters of the session id; 4 and 5 sessions), same pairing, NaN rule, optimizers and boundary rule;
    test_used 'LME (within patient)'."""
    return lme_paired_test(a, b, key, patient=subject)


PP_TAIL = ("within a single patient the test is the mixed model of the test_rule row fitted to that patient's "
           "trials, d ~ 1 + (1 | session) (4 and 5 sessions): mean_difference_lme = the LME intercept, se_lme its SE, "
           "p_lme = p_used = its Wald p (test_used 'LME (within patient)'); session_variance its session variance; "
           "shapiro_p, p_wilcoxon, p_paired_t and median_difference are descriptive and never set a star. n is the "
           "patient's rollout count. Rows added; the pooled rows above and everything drawn in the "
           "panel do not depend on them.")


def _annot_text(obs, star):
    """The ONE line each axes prints: signed median TMAX displacement, two decimals, " m", then
    that condition's vs-zero star."""
    return f"{obs:+.2f}".replace("-", "−") + f" m {star}"


ENDF = {lab: e.reset_index() for lab, e in END.items()}
if MATCHED:
    assert set(END["Walker"].index) == set(END["Decoded"].index), "matched walker != decoded trial set"
KEY = "x_m"
if LME:
    pw = lme_paired_test(ENDF["Walker"], ENDF["Decoded"], KEY)
    pn = [lme_paired_test(ENDF["Decoded"], ENDF[n], KEY) for n in ("CoordinationNull", "InPhaseNull")]
    pz = {lab: lme_paired_test(ENDF[lab], None, KEY) for _, lab, _ in ROWS}
    if FDR_STARS:
        from statsmodels.stats.multitest import multipletests
        _fam = [pw] + pn + list(pz.values())
        _q = multipletests(np.asarray([float("{:.6g}".format(r["p_used"])) for r in _fam], float),
                           alpha=0.05, method="fdr_bh")[1]
        for r, q in zip(_fam, _q):
            r["p_fdr_bh"], r["star_drawn"] = float(q), stars(q)
    ANNOT.update({lab: _annot_text(pz[lab]["median_difference"], pz[lab]["star_drawn" if FDR_STARS else "star"])
                  for _, lab, _ in ROWS})
    PATIENTS = sorted({str(s)[:5] for lab in ENDF for s in ENDF[lab].session})
    PP_ZERO_SCHEME = (f"per-patient one-sample: d = the per-rollout {TMAX:.1f} s endpoint itself, over that "
                      "patient's rollouts only, NaN dropped; " + PP_TAIL)
    PP_SCHEME = ("per-patient split of the pooled row of the same name, subject = patient (first 5 characters "
                 "of the session id): paired difference d = A - B per trial (session x trial_idx) over that "
                 "patient's rollouts only, NaN dropped; " + PP_TAIL)
    PP = []
    for _sub in PATIENTS:
        for _, lab, _ in ROWS:
            PP.append((f"{lab}_vs_zero_{_sub}",
                       intercept_paired_test(ENDF[lab], None, KEY, _sub), PP_ZERO_SCHEME))
        PP.append((f"Walker_minus_Decoded_{_sub}",
                   intercept_paired_test(ENDF["Walker"], ENDF["Decoded"], KEY, _sub), PP_SCHEME))
        for _nul in ("CoordinationNull", "InPhaseNull"):
            PP.append((f"Decoded_minus_{_nul}_{_sub}",
                       intercept_paired_test(ENDF["Decoded"], ENDF[_nul], KEY, _sub), PP_SCHEME))
else:
    pw = (perm_paired_median if MATCHED else perm_unpaired_median)(ENDF["Walker"], ENDF["Decoded"], KEY)
    pn = [perm_paired_median(ENDF["Decoded"], ENDF[n], KEY) for n in ("CoordinationNull", "InPhaseNull")]
    pz = {lab: perm_onesample_median(ENDF[lab], KEY) for _, lab, _ in ROWS}
    ANNOT.update({lab: _annot_text(pz[lab][0], stars(pz[lab][1])) for _, lab, _ in ROWS})

if LME:
    _K = LME_COLS.index("star") + 1

    def _st_row(comparison, r, scheme):
        row = lme_csv_row(comparison, r, scheme)
        if not FDR_STARS:
            return row
        fdr = ["{:.6g}".format(r["p_fdr_bh"]), r["star_drawn"]] if "p_fdr_bh" in r else ["", ""]
        return row[:_K] + fdr + row[_K:]

    with open(ST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LME_COLS[:_K] + ["p_fdr_bh", "star_drawn"] + LME_COLS[_K:] if FDR_STARS else LME_COLS)
        w.writerow(_st_row("Walker_minus_Decoded", pw, LME_SCHEME))
        for name, r in zip(("CoordinationNull", "InPhaseNull"), pn):
            w.writerow(_st_row(f"Decoded_minus_{name}", r, LME_SCHEME))
        for lab, r in pz.items():
            w.writerow(_st_row(f"{lab}_vs_zero", r, f"one-sample: d = the per-rollout {TMAX:.1f} s endpoint "
                                                    "itself; " + LME_SCHEME.split("; ", 1)[1]))
        for name, r, sch in PP:
            w.writerow(_st_row(name, r, sch))
    print(f"wrote {os.path.relpath(ST, ROOT)}")
    print(f"x at {TMAX:.1f} s: " + " | ".join(
        f"{c} mean {r['mean_difference_lme']:+.3f} m {r['test_used']} p={r['p_used']:.2g} {r['star']} (n {r['n']})"
        for c, r in [("W-D", pw), ("D-iscr", pn[0]), ("D-uscr", pn[1])]))
    print("vs zero: " + " | ".join(f"{lab} mean {r['mean_difference_lme']:+.3f} m {r['test_used']} p={r['p_used']:.2g} "
                                   f"{r['star']} (n {r['n']}) drawn {ascii(ANNOT[lab])} at {ANNOT_VA[lab]} right"
                                   for lab, r in pz.items()))
    print("per patient: " + " | ".join(f"{name} mean {r['mean_difference_lme']:+.3f} m LME (within patient) "
                                       f"p={r['p_used']:.2g} {r['star']} (n {r['n']})" for name, r, _ in PP))
else:
    with open(ST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "statistic", "observed", "p_perm", "star", "n_a", "n_b", "B", "seed", "scheme"])
        obs, p, na, nb, sess = pw
        w.writerow(["Walker_minus_Decoded", "median of the paired difference" if MATCHED else "median difference",
                    f"{obs:.6f}", f"{p:.6f}", stars(p), na, nb, B, SEED,
                    "paired within-trial sign-flip permutation on the per-trial difference (the window-matched "
                    "walker replays the SAME trials over exactly the decoded samples); two-sided on |median|; "
                    "P = (count + 1) / (B + 1)" if MATCHED else
                    "unpaired session-stratified permutation: condition labels permuted WITHIN session over the "
                    f"{len(sess)} shared sessions ({'; '.join(sess)}); two-sided on |median difference|; "
                    "P = (count + 1) / (B + 1)"])
        for name, (obs, p, na, nb, _) in zip(("CoordinationNull", "InPhaseNull"), pn):
            w.writerow([f"Decoded_minus_{name}", "median of the paired difference", f"{obs:.6f}", f"{p:.6f}",
                        stars(p), na, nb, B, SEED,
                        "paired within-trial sign-flip permutation on the per-trial difference (the null re-runs "
                        "the SAME trials with the phases swapped); two-sided on |median|; P = (count + 1) / (B + 1)"])
        for lab, (obs, p, na, _nb, _) in pz.items():
            w.writerow([f"{lab}_vs_zero", f"median of the per-rollout {TMAX:.1f} s endpoint", f"{obs:.6f}",
                        f"{p:.6f}", stars(p), na, 0, B, SEED,
                        f"one-sample sign-flip permutation of the per-rollout {TMAX:.1f} s endpoint against 0 "
                        "(median), B = 10,000, seed 0; two-sided"])
    print("wrote", os.path.relpath(ST, ROOT))
    print(f"x at {TMAX:.1f} s: W-D median-perm obs={pw[0]:+.3f} m p={pw[1]:.3g} {stars(pw[1])} "
          f"(n {pw[2]}/{pw[3]}) | D-iscr obs={pn[0][0]:+.3f} m p={pn[0][1]:.3g} {stars(pn[0][1])} "
          f"| D-uscr obs={pn[1][0]:+.3f} m p={pn[1][1]:.3g} {stars(pn[1][1])}")
    # print it (the FILES are always written encoding="utf-8", so only this diagnostic is escaped).
    print("vs zero: " + " | ".join(f"{lab} obs={o:+.3f} m p={p:.3g} {stars(p)} (n {na}) drawn "
                                   f"{ascii(ANNOT[lab])} at {ANNOT_VA[lab]} right"
                                   for lab, (o, p, na, _, _) in pz.items()))

for ax, (_condition, lab, _col) in zip(axes, ROWS):
    _draw_condition(ax, lab)

for ax in axes:
    ax.set_xlim(0, XMAX)
    ax.set_xticks(XTICKS)
    ax.xaxis.set_major_formatter("{x:g}")
    ax.set_yticks([-2.5, 0, 2.5])
axes[2].set_xlabel("Time (s)")
axes[3].set_xlabel("Time (s)")
for ax in (axes[0], axes[2]):
    ax.set_ylabel("Forward displacement (m)", alpha=0)
fig.tight_layout(h_pad=0.4, w_pad=0.4)
fig.supylabel("Forward displacement (m)", fontsize=6.5 * FS,
              y=0.5 * (axes[0].get_position().y1 + axes[2].get_position().y0))
_cell_w_in = axes[0].get_position().width * GRID_W
fig.savefig(OUT_PNG, dpi=300, facecolor="white")
plt.close(fig)
print("wrote", os.path.relpath(OUT_PNG, ROOT))


def _render_split(labels, out_png):
    fw = GRID_W / 2.0
    for _ in range(4):
        f2, ax2 = plt.subplots(2, 1, figsize=(fw, GRID_H), dpi=300, sharex=True, sharey=True)
        for a, lab in zip(ax2, labels):
            _draw_condition(a, lab)
            a.set_xlim(0, XMAX)
            a.set_xticks(XTICKS)
            a.xaxis.set_major_formatter("{x:g}")
            a.set_yticks([-2.5, 0, 2.5])
            a.set_ylabel("Forward displacement (m)", alpha=0)
        ax2[1].set_xlabel("Time (s)")
        f2.tight_layout(h_pad=0.4, w_pad=0.4)
        err = _cell_w_in - ax2[0].get_position().width * fw
        if abs(err) < 1e-6:
            break
        fw += err
        plt.close(f2)
    f2.supylabel("Forward displacement (m)", fontsize=6.5 * FS,
                 y=0.5 * (ax2[0].get_position().y1 + ax2[1].get_position().y0))
    f2.savefig(out_png, dpi=300, facecolor="white")
    plt.close(f2)
    print("wrote", os.path.relpath(out_png, ROOT),
          f"(axes {ax2[0].get_position().width * fw:.4f} x "
          f"{ax2[0].get_position().height * GRID_H:.4f} in)")


_render_split(["Walker", "CoordinationNull"], OUT_PNG_L)
_render_split(["Decoded", "InPhaseNull"], OUT_PNG_R)
print("\n".join(rep))

COLS = ["row_type", "condition_label", "stage", "condition", "session", "trial_idx", "t_s", "x_m",
        "x_m_median", "n_rollouts"]
with open(SD, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    tr_all = pd.concat(sd_traces, ignore_index=True)
    for r in tr_all.itertuples(index=False):
        w.writerow(["trace", r.condition_label, r.stage, r.condition, r.session, r.trial_idx,
                    f"{r.t_s:.2f}", f"{r.x_m:.6f}", "", ""])
    for lab, t, v, nn in sd_medians:
        w.writerow(["median", lab, "", "", "", "", f"{t:.2f}", "", f"{v:.6f}", nn])
print(f"wrote {os.path.relpath(SD, ROOT)}  ({len(tr_all)} trace rows + {len(sd_medians)} median rows)")

def _dec_mix(sessions):
    """How the Decoded condition splits over the two patients (stats / scope text)."""
    n = len(sessions); p1 = int(sum(str(x).startswith("pat01") for x in sessions))
    return (f"is P1-dominated ({p1} of {n} trials)" if p1 >= 0.9 * n
            else f"holds both patients (P1 {p1} + P2 {n - p1} = {n} trials) but only two people")


DEC_MIX = _dec_mix(ENDF["Decoded"].session)

if LME:
    STATS_MD = (
        f"04_source_data/fig7/fig7_sim_trajectories_stats.csv — SEVEN tests on the {TMAX:.1f} s forward displacement "
        "(the per-rollout endpoints summarised by the endpoint_stats row), each under the test_rule row: the "
        "per-trial paired difference, LME d ~ 1 + (1 | session), REML, Wald p (LME only; Shapiro-Wilk and Wilcoxon in the Source Data only). THREE BETWEEN-CONDITION contrasts, all PAIRED on (session, trial_idx): (1) "
        f"Walker_minus_Decoded, because the window-matched walker replays the SAME {pw['n']} trials over exactly "
        "the decoded samples (walker_window row); (2, 3) Decoded_minus_CoordinationNull and Decoded_minus_InPhaseNull, "
        "because each null re-runs the SAME trials with the phases swapped. FOUR ONE-SAMPLE rows — Walker_vs_zero, "
        "Decoded_vs_zero, CoordinationNull_vs_zero, InPhaseNull_vs_zero — back the in-panel annotation (in_figure_annotation "
        f"row): d is that condition's {TMAX:.1f} s endpoint itself. Results (mean = the LME intercept; p = p_used"
        + ("; pFDR = p_fdr_bh): " if FDR_STARS else "): ")
        + "; ".join(f"{c} mean {r['mean_difference_lme']:+.6f} m, {r['test_used']} p = {r['p_used']:.3g} "
                    + (f"pFDR = {r['p_fdr_bh']:.3g} ({r['star_drawn']})" if FDR_STARS else f"({r['star']})")
                    + f", n = {r['n']}"
                    for c, r in [("Walker_minus_Decoded", pw), ("Decoded_minus_CoordinationNull", pn[0]),
                                 ("Decoded_minus_InPhaseNull", pn[1])] + [(f"{lab}_vs_zero", r) for lab, r in pz.items()])
        + ". ONLY THE FOUR VS-ZERO STARS ARE DRAWN, one per axes, beside that condition's median endpoint; the three "
          "between-condition tests are reported here and in the caption only, with no bracket in the panel. "
        + ("Benjamini-Hochberg correction across these seven displacement tests: each drawn star is set on that "
           "row's p_fdr_bh (stats column star_drawn), the same values as 03_figures/results_statistics.py item 15 (family "
           "F15-displacement); the stats column star keeps the uncorrected star of p_used. "
           if FDR_STARS else "Uncorrected: ")
        + ("The " if FDR_STARS else "the ")
        + "two nulls are re-runs of the same trials and the vs-zero tests re-use the same endpoints, so the "
          f"seven tests are NOT independent. The Decoded condition {DEC_MIX}; the session random intercept absorbs the "
          "within-session correlation of trials, but with two patients these remain TRIAL-level tests, not "
          "population-level ones.")
else:
    STATS_MD = (
    f"04_source_data/fig7/fig7_sim_trajectories_stats.csv — SEVEN permutation tests on the {TMAX:.1f} s forward "
    "displacement (the per-rollout endpoints summarised by the endpoint_stats row), "
    f"B = {B}, seed {SEED}, two-sided, P = (count + 1) / (B + 1), stars *** / ** / * / ns at 1e-3 / "
    "1e-2 / 0.05. "
    + ("THREE BETWEEN-CONDITION tests, all PAIRED on the pairing key (session, trial_idx): sign flip within "
       "trial on the per-trial difference, statistic = the median of the paired difference. (1) "
       f"Walker_minus_Decoded, because the window-matched walker replays the SAME {pw[2]} trials over "
       "exactly the decoded samples (walker_window row). (2, 3) Decoded_minus_CoordinationNull and "
       "Decoded_minus_InPhaseNull, because each null re-runs the SAME trials with the phases swapped."
       if MATCHED else
       "THREE BETWEEN-CONDITION tests on the MEDIAN DIFFERENCE, session-stratified. (1) "
       "Walker_minus_Decoded is UNPAIRED: condition labels permuted WITHIN session over the "
       f"{len(pw[4])} sessions the two conditions share ({'; '.join(pw[4])}). (2, 3) Decoded_minus_CoordinationNull "
       "and Decoded_minus_InPhaseNull are PAIRED: sign flip within trial on the per-trial difference, "
       "pairing key (session, trial_idx), because each null re-runs the SAME trials with the phases "
       "swapped.")
    + " FOUR ONE-SAMPLE tests, one per condition — Walker_vs_zero, Decoded_vs_zero, "
    "CoordinationNull_vs_zero, InPhaseNull_vs_zero — added to back the in-panel annotation (see the "
    f"in_figure_annotation row): is that condition's {TMAX:.1f} s endpoint displaced from zero at all? Same paired "
    "machinery with the null side set to 0, so the per-rollout endpoint itself is the paired "
    "difference; n_a = that condition's rollouts and n_b = 0 because there is no second sample. ONE seeded "
    "RNG stream, consumed in file order: the three between-condition tests first, then the four vs-zero "
    "tests. "
    "Scheme for the between-condition tests as in "
    "03_figures/fig7_sim_cadence_symmetry.py so panel c is tested exactly like panels d and e. "
    "Results: "
    + "; ".join([f"Walker_minus_Decoded {pw[0]:+.6f} m, P = {pw[1]:.6f} ({stars(pw[1])})"]
                + [f"Decoded_minus_{n} {o:+.6f} m, P = {p:.6f} ({stars(p)})"
                   for n, (o, p, _, _, _) in zip(("CoordinationNull", "InPhaseNull"), pn)]
                + [f"{lab}_vs_zero {o:+.6f} m, P = {p:.6f} ({stars(p)}), n = {na}"
                   for lab, (o, p, na, _, _) in pz.items()])
    + ". ONLY THE FOUR VS-ZERO STARS ARE DRAWN, one per axes, beside that condition's median endpoint "
      "; the three between-condition tests "
      "are reported here and in the caption only, with no bracket in the panel. Uncorrected: the two "
      "nulls are re-runs of the same trials and the vs-zero tests re-use the same endpoints, so the "
      f"seven tests are NOT independent. The Decoded condition {DEC_MIX}, so these "
      "are TRIAL-level tests, not population-level ones.")

PER_PATIENT_MD = ""
if LME:
    PER_PATIENT_MD = (
        "04_source_data/fig7/fig7_sim_trajectories_stats.csv, rows whose comparison ends in _pat01 / _pat02. Each of the seven "
        "pooled contrasts is re-run inside ONE patient (patient = the first 5 characters of the session id), "
        f"{len(PP)} rows appended after the seven pooled rows, in the same columns: four one-sample rows per "
        f"patient (<Condition>_vs_zero_patNN — d is that condition's {TMAX:.1f} s per-rollout endpoint itself, tested against "
        "zero) and the three paired contrasts per patient (Walker_minus_Decoded_patNN, "
        "Decoded_minus_CoordinationNull_patNN, Decoded_minus_InPhaseNull_patNN — d is the per-trial difference, paired on "
        "(session, trial_idx)). TEST: every per-patient row "
        "fits the pooled model d ~ 1 + (1 | session) to that patient's trials (test_used 'LME (within patient)'): "
        "mean_difference_lme = the LME intercept, se_lme its SE, p_lme = p_used = its Wald p, which sets the star; "
        "session_variance its session variance; shapiro_p, p_wilcoxon, p_paired_t and median_difference are descriptive. Results: "
        + "; ".join(f"{name} mean {r['mean_difference_lme']:+.6f} m, p = {r['p_used']:.3g} ({r['star']}), n = "
                    f"{r['n']} over {r['n_sessions']} sessions" for name, r, _ in PP)
        + ". NOTHING IS DRAWN from these rows — the four stars in the panel still come from the POOLED "
          "<Condition>_vs_zero rows (in_figure_annotation row), the pooled rows, the figure and the *_source_data.csv "
          "are unchanged, and these rows exist only at GAIT_SIM_TEST=lme (the permutation stats CSV has different "
          "columns). Uncorrected and not independent of the pooled rows, which re-use the same endpoints. "
          "One patient is ONE PERSON: a per-patient row is a within-person trial-level test, never an estimate "
          "for patients in general.")


def _n_over(lab):
    s = ENDF[lab].session
    return f"n = {len(s)} rollouts over {s.nunique()} sessions: " + "; ".join(sorted(s.unique()))


CONDITION_ROWS = {
    "condition_walker": (("condition == 'wphase_matched' (calib; "
                    + os.path.relpath(WALKER_TRACES, ROOT).replace(os.sep, "/")
                    + ") — the pacing walker's own phase = the CEILING, not a null, replayed on exactly the "
                    "decoded window of each decoded trial (walker_window row). " + _n_over("Walker")
                    + ". Same (k1, lam = 0) cell as the Decoded condition.") if MATCHED else
                   ("condition == 'wphase' (calib) — the pacing walker's own phase = the CEILING, not a "
                    "null. " + _n_over("Walker") + ".")),
    "condition_decoded": ("condition == 'dphase' (confirm) — EEG-decoded per-joint phase driving the "
                    "template. " + _n_over("Decoded") + "."),
    "condition_scrambled": ("condition == 'dphase_iscr' (confirm) — inter-joint phase relationship scrambled. " f"n = {len(ENDF['CoordinationNull'])}."),
    "condition_shifted": ("condition == 'dphase_uscr' (confirm) — In-phase null: one random Fourier phase "
                    "spectrum shared by all six joints of the decoded trajectory "
                    "(02_decoding/simulation/data_io.py uscramble; each joint keeps its amplitude "
                    "spectrum), so the joints move in phase and the decoded timing is destroyed. "
                    f"n = {len(ENDF['InPhaseNull'])}."),
}
STAT_LINES = "; ".join(
    f"{lab}: median {m:+.2f} m, IQR {q1:+.2f}..{q3:+.2f} m, backward {b:.0f}%"
    for lab, _, _, m, q1, q3, b in sd_stats)
INK_LINES = "; ".join(f"{lab} n = {nt}, alpha = {al:.4f}"
                      for lab, nt, al, *_ in sd_stats)
ENV_NOTE = ""
if GATE == "3s":
    _yl, _yh = lo - pad, hi + pad
    _BOX = {"bottom": (0.04, 0.14), "top": (0.83, 0.93)}
    _env, _clear = [], True
    for _condition, _lab, _ in ROWS:
        _r3 = d[(d.condition == _condition) & (d.t_s >= 2 * TMAX / 3 - 1e-9)].x_m
        _f0, _f1 = (_r3.min() - _yl) / (_yh - _yl), (_r3.max() - _yl) / (_yh - _yl)
        _env.append(f"{_lab} {_f0:.3f}-{_f1:.3f}")
        _clear &= (_f0 > _BOX["bottom"][1]) if ANNOT_VA[_lab] == "bottom" else (_f1 < _BOX["top"][0])
    ENV_NOTE = (" Measured on this "
                f"{TMAX:.0f} s build over the right third of the window (t >= {2 * TMAX / 3:.1f} s), the rollout "
                "clouds span " + "; ".join(_env) + " of the axes height, against text boxes at 0.04-0.14 "
                "(bottom) and 0.83-0.93 (top) of the axes height, so "
                + ("every text box stays clear of its condition's cloud." if _clear
                   else "at least one text box overlaps its condition's cloud."))

ANNOT_MD = (
    "ONE text line inside EACH of the four axes. " f"It reads: that condition's MEDIAN {TMAX:.1f} s forward displacement in METRES — signed, "
    "two decimals, real minus sign U+2212, followed by ' m' — then the star of THAT CONDITION'S "
    "<Condition>_vs_zero row in 04_source_data/fig7/fig7_sim_trajectories_stats.csv. Exactly, as drawn: "
    + "; ".join((f'{lab} “{ANNOT[lab]}” (median {pz[lab]["median_difference"]:+.6f} m, descriptive; star from '
                 f'{lab}_vs_zero: {pz[lab]["test_used"]} p = {pz[lab]["p_used"]:.3g}'
                 + (f', pFDR = {pz[lab]["p_fdr_bh"]:.3g} (BH over the seven displacement tests)' if FDR_STARS else '')
                 + f', n = {pz[lab]["n"]})') if LME else
                (f'{lab} “{ANNOT[lab]}” (from {lab}_vs_zero: observed {pz[lab][0]:+.6f} m, '
                 f'P = {pz[lab][1]:.6f}, n = {pz[lab][2]})') for _, lab, _ in ROWS)
    + (". The printed value is the MEDIAN endpoint, shown descriptively; the star is that condition's vs-zero test "
       "under the test_rule row (the LME intercept = the mean endpoint with a session random intercept), not a test of the median." if LME else
       ". The printed value IS the tested statistic (the vs-zero observed), not a separately computed "
       "number.")
    + " NOTHING ELSE is drawn from the statistics: no bracket, and the three BETWEEN-CONDITION "
      "tests (Walker_minus_Decoded, Decoded_minus_CoordinationNull, Decoded_minus_InPhaseNull) appear only in "
      "the stats CSV and the caption — every star that IS in the panel has a row in that CSV. "
      "Typography: Arial at 6.5 pt x 1.35 (the same size as the condition label at the top left), "
      "project_palette.INK, right-aligned at x = 0.97 of the axes. PLACEMENT: the right edge, in "
      "the half of the axes that condition's rollout cloud leaves empty — BOTTOM right (y = 0.04, "
      "va='bottom') for Walker and Decoded, TOP right (y = 0.93, va='top', the condition label's own y) "
      "for CoordinationNull and InPhaseNull. "
      
      
      
      
      "The rule lives in _draw_condition, so the combined 2 x 2 and "
      "both split half-panels print the same string in the same place. METRES ONLY: no speed is "
      "printed or implied (units_rule)." + ENV_NOTE)

WALKER_IN = WALKER_FILT = WALKER_CHECK = WALKER_MD = ""
if MATCHED:
    _wrel, _rrel = (os.path.relpath(p, ROOT).replace(os.sep, "/") for p in (WALKER_TRACES, WALKER_RHY))
    _WR = F7_SCORED.keep_sim(pd.read_csv(WALKER_RHY))
    assert (_WR.phase_src == "k1").all() and (_WR.lam.astype(float) == 0.0).all(), WALKER_RHY
    _WDUR = (_WR.dur_s.min(), _WR.dur_s.max())
    WALKER_MD = ("walker_window",
                 f"window-matched: the walker's own phase replayed on exactly the decoded window of each of the "
                 f"{_n['dphase']} decoded trials (after the decoder's 2 s causal start-up), produced by "
                 "02_decoding/simulation/replay_walker_matched.py via run_walker_matched_gate3s.sh on the compute server")
    if ec.F7_SIM == "causal_from2s":
        WALKER_MD = (WALKER_MD[0], WALKER_MD[1].split("; the unmatched build")[0]
                     + "; under GAIT_F7_SIM=causal_from2s it is driven from trial 2 s by its own goniometer (no "
                       "unmatched walker is simulated)")
    WALKER_IN = (f" The Walker condition is read instead from {_wrel} (condition 'wphase_matched', same columns and "
                 "40 ms grid; walker_window row), which replaces the condition 'wphase' rows of that file.")
    WALKER_FILT = (f" In the window-matched build the Walker condition is condition == 'wphase_matched' (stage 'calib') of "
                   f"{_wrel}; its companion {_rrel} records phase_src 'k1' and lam 0 on all {len(_WR)} rows.")
    _wl = (F7_SCORED.keep_sim(pd.read_csv(WALKER_TRACES)).sort_values("t_s", kind="stable")
           .groupby(["session", "trial_idx"]).x_m.last().reset_index()
           .merge(_WR[["session", "trial_idx", "dx_m"]], on=["session", "trial_idx"], how="inner"))
    _wd = (_wl.x_m - _wl.dx_m.astype(float)).abs()
    WALKER_CHECK = (f" Window-matched walker: trace endpoint vs the per-rollout dx_m of {_rrel}, matched on "
                    f"(session, trial_idx) for {len(_wl)} of {len(_WR)} rollouts: median |difference| "
                    f"{_wd.median():.3f} m, max {_wd.max():.3f} m, computed at build time.")

LINEAGE = SELFCHECK = None
if GATE == "3s":
    _rdir = os.path.dirname(TRACES)
    _rel = os.path.relpath(_rdir, ROOT).replace(os.sep, "/")
    LINEAGE = ("Produced by 02_decoding/simulation/dump_traces_patient.py via "
               "02_decoding/simulation/run_gate3s_replays.sh on the compute server (PHYS_MIN_DUR_S=3), replaying the FROZEN "
               f"rollouts of {_rel}/g_v5_rollouts.csv (the 3 s run)."
               " The traces dump adds no analysis — it records x(t) of rollouts that already "
               "existed.")
    _R = pd.read_csv(os.path.join(_rdir, "g_v5_rollouts.csv"), low_memory=False)
    _R = _R[(_R.rung == "f5n") & (_R.phase_src == "k1") & (_R.lam.astype(float) == 0.0)]
    _K = ["stage", "condition", "session", "trial_idx"]
    _last = pd.read_csv(TRACES).sort_values("t_s", kind="stable").groupby(_K).x_m.last().reset_index()
    _mm = _last.merge(_R[_K + ["dx_m"]], on=_K, how="inner")
    _dd = (_mm.x_m - _mm.dx_m.astype(float)).abs()
    SELFCHECK = ("Trace endpoint (last sample of the untruncated trace) vs the frozen per-rollout dx_m of "
                 f"{_rel}/g_v5_rollouts.csv, matched on (stage, condition, session, trial_idx) for {len(_mm)} of "
                 f"{len(_last)} rollouts: median |difference| {_dd.median():.3f} m, max {_dd.max():.3f} m "
                 "(traces are recorded on a 40 ms grid), computed at build time. The traces therefore "
                 "reproduce the frozen outcome file to within that margin." + WALKER_CHECK)

META = [
    ("paper_location", "Fig. 7 simulation block."
                       ),
    ("build_id", ec.COHORTS["post_operative"]),
    ("cohort", "post_operative (P1 sess04-09 x6 + P2 sess03/05/06/08/09 x5), n = 2 patients, "
               "case-level. The simulation uses 9 of the 11 sessions. "
               "Eleven sessions "
               "from two people is not a population estimate."),
    ("n_sessions_contributing", "; ".join(f"{lab} {ENDF[lab].session.nunique()}" for _, lab, _ in ROWS)),
    ("input_file", os.path.relpath(TRACES, ROOT).replace(os.sep, "/") + " — columns stage, "
                   "condition, session, trial_idx, t_s, x_m; hip-x forward displacement of every f5n "
                   "main-cell rollout on a 40 ms grid." + WALKER_IN),
    ("baseline_definition", "x_m = 0 at the first main-loop sample after the 0.5 s settle; every "
                            "trace is that rollout's displacement from its own post-settle hip x."),
    ("input_filter", "stage == 'confirm' for the Decoded / CoordinationNull / InPhaseNull conditions and "
                     "stage == 'calib' for the Walker condition (the walker-phase conditions live in the "
                     "calib sweep); then rung == 'f5n', phase_src == 'k1', lam == '0.0' on every "
                     "condition — inherited from the frozen rollout file, this script re-filters only "
                     "by `condition`." + WALKER_FILT),
    ("condition_walker", CONDITION_ROWS["condition_walker"]),
    ("condition_decoded", CONDITION_ROWS["condition_decoded"]),
    ("condition_scrambled", CONDITION_ROWS["condition_scrambled"]),
    ("condition_shifted", CONDITION_ROWS["condition_shifted"]),
    ("n_rollouts", "; ".join(f"{lab} {_n[condition]}" for condition, lab, _ in ROWS) + " (rollout = one "
                   "session x trial_idx group)."),
    ("common_window", f"0-{TMAX:.1f} s, applied to EVERY condition. Trials are not equally long — "
                      + (f"every condition, the window-matched walker included, spans the decoded window of the same "
                         f"trials, {_WDUR[0]:.1f}-{_WDUR[1]:.1f} s —" if MATCHED else
                         "walker trials run to 8.4 s while the decoded conditions stop at 6.4 s —")
                      + " so an untruncated "
                      f"final displacement would partly measure trial duration. {TMAX:.1f} s is the "
                      "MINIMUM trial length in the set, so every rollout covers the whole window "
                      "and the four axes are directly comparable."),
    ("median_rule", f"Thick line (lw {MED_LW}, condition hue darkened {MED_DARKEN:.0%} toward INK, no halo) = the median over rollouts at each 40 ms time point, "
                    "drawn ONLY where >= 5 rollouts are still running (trials differ in length). "
                    "row_type=median carries exactly the plotted points with their n_rollouts."),
    ("zero_line", "Dotted horizontal line at x = 0 (#30343a, lw 0.7) = no net forward progress. "
                  "It is a reference, not a threshold or a test."),
    ("endpoint_stats", f"Forward displacement at {TMAX:.1f} s (last sample within the window) per condition — "
                       + STAT_LINES + ". Computed over ALL rollouts of the condition."),
    ("stats", STATS_MD),
    ("reproducibility", "No RNG. figsize 4.4 x 3.0 in at 300 dpi, Arial, base font 6.5 pt x 1.35, "
                        f"shared x and y, x ticks [{', '.join(map(str, XTICKS))}], y ticks [-2.5, 0, 2.5], no in-figure "
                        "title."),
    ("term_definition", "'Forward displacement' = the simulated hip x of the physics rollout. It "
                        "is NOT gait speed and NOT a clinical walking distance."),
    ("producer", "03_figures/fig7_sim_trajectories.py"),
]
if GATE_MD:
    META.insert([k for k, _ in META].index("input_file"), GATE_MD)
if ec.F7_SIM:
    META.insert([k for k, _ in META].index("input_file"), ec.F7_SIM_MD)
if ec.F7_SIM == "causal_from2s":
    META = [(k, v.replace(f"it matters for the unmatched build, whose walker condition holds {N_WALKER_UNMATCHED} rollouts.",
                          "it matters only for an unmatched walker build, which this build does not simulate."))
            for k, v in META]
if DRAW_FULL:
    META.insert([k for k, _ in META].index("input_file"),
                ("drawn_window", f"GAIT_SIM_DRAW_FULL=1: each rollout's thin line is drawn to its own "
                 f"end (x axis 0-{XMAX:g} s; the trace rows of the Source Data carry every drawn sample, t_s > {TMAX:.1f} s "
                 f"included); the median line, the endpoint at {TMAX:.1f} s, every statistic and the in-figure number are "
                 f"unchanged and use the common window 0-{TMAX:.1f} s only; a dotted vertical line marks {TMAX:.1f} s."))
if MATCHED:
    META.insert([k for k, _ in META].index("input_file"), WALKER_MD)
if LME:
    META.insert([k for k, _ in META].index("stats"), ("test_rule", TEST_RULE))
    META.insert([k for k, _ in META].index("stats") + 1, ("per_patient_rows", PER_PATIENT_MD))
with open(MD, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["key", "value"])
    if SUPPORT_MD:
        w.writerow(SUPPORT_MD)
    for k, v in META:
        w.writerow([k, _md_fix(v)])
print("wrote", os.path.relpath(MD, ROOT))
