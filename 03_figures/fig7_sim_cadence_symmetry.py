"""Figure 7 — Intent gate and physics simulation of the decoded gait phase

_sim_cadence_symmetry — Fig. 7 simulation-rhythm pair: does the EEG-parameterized template
walk keep the pacing walker's STEP RHYTHM (cadence) and its LEFT-RIGHT STEP SYMMETRY, and do the
two input-side nulls destroy both?

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

ROLLOUTS = "02_decoding/simulation/simulation_rollouts/g_v5_rollouts.csv"
RHYTHM = "02_decoding/simulation/simulation_rollouts/rhythm_metrics.csv"
FS = 1.35
B = 10000
SEED = 0
BUILD_ID = ec.PAT_TAG
COHORT = ec.COHORTS["post_operative"]
CAND_CLI = "rendered by 03_figures/fig7_sim_cadence_symmetry.py"
PAIR_PNG = "figures_out/fig7/fig7_sim_cadence_symmetry.png"

def _md_fix(v):
    """Metadata value rewriter; the identity at the default support level (L1p)."""
    return v


SUPPORT = os.environ.get("GAIT_SUPPORT", "L1p")
assert SUPPORT in ("L1p", "L3b"), f"GAIT_SUPPORT must be L1p or L3b, got {SUPPORT!r}"
SUPPORT_MD = None
PNG_DIR, PNG_SUF, _SD_OVERRIDE = "figures_out/fig7", "", None
if SUPPORT == "L3b":
    ROLLOUTS = ROLLOUTS.replace("/simulation_rollouts/", "/simulation_rollouts_variant/")
    RHYTHM = RHYTHM.replace("/simulation_rollouts/", "/simulation_rollouts_variant/")
    PNG_DIR, PNG_SUF = "figures_out/fig7/_candidates", "_L3b"
    PAIR_PNG = f"{PNG_DIR}/fig7_sim_cadence_symmetry{PNG_SUF}.png"
    _SD_OVERRIDE = os.path.join(ROOT, "figures_out", "fig7", "_candidates", "sd_L3b")
    os.makedirs(_SD_OVERRIDE, exist_ok=True)

    def _md_fix(v):
        return (v.replace("simulation_rollouts/", "simulation_rollouts_variant/")
                if isinstance(v, str) else v)
    SUPPORT_MD = ("support_level",
                  "an alternative support level; not the published build")

GATE = "3s"
assert SUPPORT != "L3b", "GAIT_SUPPORT=L3b is not available in this repository"
TMAX = 3.0
GATE_MD = None
if GATE == "3s":
    ROLLOUTS = ROLLOUTS.replace("/simulation_rollouts/", "/simulation_rollouts_gate3s/")
    RHYTHM = RHYTHM.replace("/simulation_rollouts/", "/simulation_rollouts_gate3s/")
    GATE_MD = ("usable_duration_gate",
               "3 s (PHYS_MIN_DUR_S=3; a trial is included when at least 3 s of decoder output remain after the decoder's 4 s start-up)")

WALKER = os.environ.get("GAIT_SIM_WALKER", "matched" if GATE == "3s" else "unmatched")
assert WALKER in ("matched", "unmatched"), f"GAIT_SIM_WALKER must be matched or unmatched, got {WALKER!r}"
MATCHED = WALKER == "matched"
WALKER_RHY = RHYTHM.replace("rhythm_metrics.csv", "walker_matched_rhythm.csv")


def _out(p):
    return str(ec.scoped(p)) if ec.SCOPED else p


if ec.SCOPED:
    ROLLOUTS, RHYTHM, WALKER_RHY = (os.path.relpath(ec.sim_in(os.path.join(ROOT, p)), ROOT).replace(os.sep, "/")
                                    for p in (ROLLOUTS, RHYTHM, WALKER_RHY))

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

DE = os.environ.get("GAIT_SIM_DE", "rate_length" if (GATE == "3s" and MATCHED and LME) else "streak_asym")
assert DE in ("rate_length", "streak_asym"), f"GAIT_SIM_DE must be rate_length or streak_asym, got {DE!r}"
assert not (DE == "rate_length" and not (GATE == "3s" and MATCHED and LME)), \
    "GAIT_SIM_DE=rate_length exists only for GAIT_SIM_WALKER=matched + GAIT_SIM_TEST=lme"
RATE = DE == "rate_length"

CONDITION_ORDER =("Walker", "Decoded", "CoordinationNull", "InPhaseNull")
DISPLAY = {"Walker": "Walker", "Decoded": "Decoded",
           "CoordinationNull": "Coordination null", "InPhaseNull": "In-phase null"}
AXIS_LABEL = {"Walker": "Walker", "Decoded": "Decoded",
              "Coordination null": "Coord.\nnull", "In-phase null": "In-phase\nnull"}

plt.rcParams.update({"font.family": "Arial", "font.size": 6.5 * FS, "axes.linewidth": 1.0,
                     "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42})
RNG = np.random.default_rng(SEED)

COL = {"Walker": P.WALKER_HUE, "Decoded": P.IW_HUE, "CoordinationNull": P.MID, "InPhaseNull": P.MID}
FACE = {"Walker": P.WALKER_LIGHT, "Decoded": P.IW_LIGHT, "CoordinationNull": P.NULL_FILL, "InPhaseNull": P.NULL_FILL}


def conditions(df):
    f = lambda st, ar: df[(df.stage == st) & (df.condition == ar) & (df.phase_src == "k1") & (df.lam.astype(float) == 0.0)]
    return {"Walker": f("calib", "wphase"), "Decoded": f("confirm", "dphase"),
            "CoordinationNull": f("confirm", "dphase_iscr"), "InPhaseNull": f("confirm", "dphase_uscr")}


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


def _star(t):
    """Bracket text: in GAIT_SIM_TEST=lme a test result is a dict carrying its own star; in perm it is the
    permutation tuple (obs, P, ...), starred from P."""
    if isinstance(t, dict) and "star_drawn" in t:
        return t["star_drawn"]
    return t["star"] if LME else stars(t[1])


def bracket(ax, i, j, y, h, txt, fs=10.2 * FS):
    _lw, _fs_ns = (1.1, fs * 0.65)
    ax.plot([i, i, j, j], [y, y + h, y + h, y], color="#30343a", lw=_lw, clip_on=False, zorder=5)
    ax.annotate(txt, ((i + j) / 2, y + h), xytext=(0, -0.503 * fs if txt != "ns" else 1), textcoords="offset points",
                ha="center", va="bottom", fontsize=fs if txt != "ns" else _fs_ns, color="#30343a")


_roll = F7_SCORED.keep_sim(pd.read_csv(os.path.join(ROOT, ROLLOUTS)))
_rhy = F7_SCORED.keep_sim(pd.read_csv(os.path.join(ROOT, RHYTHM)))
N_WALKER_UNMATCHED = len(conditions(_roll)["Walker"])
if MATCHED:
    _wm = F7_SCORED.keep_sim(pd.read_csv(os.path.join(ROOT, WALKER_RHY)))
    _WM = _wm[(_wm.stage == "calib") & (_wm.condition == "wphase_matched") & (_wm.phase_src == "k1")
              & (_wm.lam.astype(float) == 0.0)]
if RATE:
    def _rate_length(df):
        n = df.n_adv_events.astype(float)
        return df.assign(cadence_steps_per_s=n / df.dur_s.astype(float),
                         step_length_m=df.dx_m.astype(float) / n.where(n > 0))
    _rhy, _WM = _rate_length(_rhy), _rate_length(_WM)
_ASYM = dict(slot="e", stem="fig7_sim_step_symmetry", df=_rhy, src=RHYTHM, key="r5_lr_asym",
             ylab="Advancing-step count imbalance",
             metric="r5_lr_asym = |n_L - n_R| / (n_L + n_R) over the advancing steps of the rollout "
                    "(rhythm probe, columns n_adv_l / n_adv_r). 0 = perfectly alternating left/right "
                    "stepping; 1 = every advancing step on the same side.")
_ADV = ("n_adv_events = the number of advancing left/right support transitions of the rollout (a transition "
        "advances when the incoming support foot lands >= 0.01 m in front of the outgoing one)")
if RATE:
    PANELS = [
        dict(slot="d", stem="fig7_sim_cadence", df=_rhy, src=RHYTHM, key="cadence_steps_per_s",
             ylab="Cadence (steps s$^{-1}$)",
             scheme_pre="cadence = n_adv_events / dur_s (advancing steps per second) per rollout; ",
             metric="cadence_steps_per_s = n_adv_events / dur_s per rollout (rhythm-probe columns), in advancing "
                    f"steps per second: the conventional step rate over the whole rollout. {_ADV}; dur_s = the "
                    "rollout duration. The Walker condition is read from walker_matched_rhythm.csv with the same "
                    "formula. It differs from cadence_hz = n_steps_max_streak / dur_s (the longest alternating advancing-step streak / rollout duration)."),
        _ASYM,
        dict(slot="SI", stem="sup_fig7_sim_step_length", png_dir="figures_out/supplementary", df=_rhy, src=RHYTHM,
             key="step_length_m", ylab="Step length (m)", zero=True,
             scheme_pre="step length = dx_m / n_adv_events (m) per rollout, NaN where n_adv_events == 0; ",
             metric="step_length_m = dx_m / n_adv_events per rollout (rhythm-probe columns), in metres: the mean "
                    "forward progression per advancing step. dx_m = the rollout's net forward hip displacement; "
                    f"{_ADV}. A rollout with n_adv_events == 0 has no step length (NaN) and is dropped. Negative "
                    "values = net backward progression; the dotted line marks 0 m. The Walker condition is read from "
                    "walker_matched_rhythm.csv with the same formula. Fig. 7e is the left-right step asymmetry "
                    "(figures_out/fig7/fig7_sim_step_symmetry.png)."),
    ]
else:
    PANELS = [
        dict(slot="d", stem="fig7_sim_cadence", df=_roll, src=ROLLOUTS, key="cadence_hz",
             ylab="Cadence (steps s$^{-1}$)",
             metric="cadence_hz = n_steps_max_streak / dur_s (run_rollouts.py) — the step rhythm of the "
                    f"longest advancing-step streak of the {TMAX:.0f} s rollout, in steps per second."),
        _ASYM,
    ]

if MATCHED and not RATE:
    PANELS[0]["metric"] += (" The Walker condition's cadence_hz is read from walker_matched_rhythm.csv, computed with the "
                            "same formula (on the Decoded condition the rhythm replay reproduces g_v5_rollouts.csv "
                            "cadence_hz exactly).")
for pan in PANELS:
    A = conditions(pan["df"])
    if MATCHED:
        A["Walker"] = _WM
        assert (set(zip(_WM.session, _WM.trial_idx)) == set(zip(A["Decoded"].session, A["Decoded"].trial_idx))), \
            "matched walker != decoded trial set"
    pan["A"] = A
    if LME:
        pan["pw"] = lme_paired_test(A["Walker"], A["Decoded"], pan["key"])
        pan["pn"] = [lme_paired_test(A["Decoded"], A[n], pan["key"]) for n in ("CoordinationNull", "InPhaseNull")]
        for r in [pan["pw"]] + pan["pn"]:
            for _ in range(B):
                RNG.choice([-1, 1], r["n"])
    else:
        pan["pw"] = (perm_paired_median if MATCHED else perm_unpaired_median)(A["Walker"], A["Decoded"], pan["key"])
        pan["pn"] = [perm_paired_median(A["Decoded"], A[n], pan["key"]) for n in ("CoordinationNull", "InPhaseNull")]
    pan["vals"], pan["jit"] = {}, {}
    for i, lab in enumerate(CONDITION_ORDER):
        v = A[lab][pan["key"]].astype(float).dropna().to_numpy()
        pan["vals"][lab] = v
        pan["jit"][lab] = np.full_like(v, i, dtype=float) + RNG.uniform(-0.16, 0.16, v.size)

FDR_STARS = ec.F7_SIM == "causal_from2s"
FDR_FAM = {"d": "cadence", "e": "asymmetry"}
_FDR_BOOT = (FDR_STARS and os.environ.get("GAIT_F7_FDR_BOOTSTRAP", "").strip() == "1"
             and not ec.sd("results_statistics_source_data.csv").exists())
if FDR_STARS:
    from statsmodels.stats.multitest import multipletests
    assert LME and RATE, "GAIT_F7_SIM=causal_from2s FDR stars expect the LME + rate_length build"
    _rsn = None if _FDR_BOOT else pd.read_csv(ec.sd("results_statistics_source_data.csv"))
    for pan in PANELS:
        if pan["slot"] not in FDR_FAM:
            continue
        _rs = [pan["pw"]] + pan["pn"]
        _q = multipletests(np.asarray([float("{:.6g}".format(r["p_used"])) for r in _rs], float),
                           alpha=0.05, method="fdr_bh")[1]
        for c, r, q in zip(("Walker_minus_Decoded", "Decoded_minus_CoordinationNull", "Decoded_minus_InPhaseNull"), _rs, _q):
            if not _FDR_BOOT:
                ref = _rsn.loc[_rsn.quantity == f"{FDR_FAM[pan['slot']]} {c}: pFDR", "value"]
                assert len(ref) == 1 and abs(float(ref.iloc[0]) - q) <= 1e-6, \
                    f"pFDR {FDR_FAM[pan['slot']]} {c}: {q} here vs {ref.tolist()} in results_statistics_source_data.csv"
            r["p_fdr_bh"], r["star_drawn"] = float(q), stars(q)
    if _FDR_BOOT:
        print("BOOTSTRAP pass (GAIT_F7_FDR_BOOTSTRAP=1): pFDR NOT cross-checked against results_statistics (file absent); "
              "re-run without the switch after 03_figures/results_statistics.py and confirm byte-identical outputs", flush=True)


def draw(ax, pan):
    if pan.get("zero"):
        ax.axhline(0, color="#30343a", lw=0.7, ls=(0, (1.6, 1.6)), zorder=1)
    allv = []
    _msn, _msf, _lwb, _lwm, _ptx = ((1.6, 2.6, 1.3, 2.2, 5.6 * FS))
    for i, lab in enumerate(CONDITION_ORDER):
        v = pan["vals"][lab]; allv.append(v)
        ax.plot(pan["jit"][lab], v, "o", ms=_msn if v.size > 30 else _msf, mew=0, color=COL[lab],
                alpha=0.35 if v.size > 30 else 0.75, zorder=2)
        ax.boxplot([v], positions=[i], widths=0.52, whis=(10, 90), showfliers=False, zorder=3, patch_artist=True,
                   medianprops=dict(color=COL[lab], lw=_lwm), boxprops=dict(facecolor=FACE[lab], edgecolor=COL[lab], lw=_lwb),
                   whiskerprops=dict(color=COL[lab], lw=_lwb), capprops=dict(color=COL[lab], lw=_lwb))
    allv = np.concatenate(allv); ymax, ymin = allv.max(), allv.min(); span = max(ymax - ymin, 1e-9)
    bracket(ax, 0, 1, ymax + span * 0.06, span * 0.03, _star(pan["pw"]))
    for step, j in enumerate((2, 3)):
        bracket(ax, 1, j, ymax + span * (0.20 + 0.14 * step), span * 0.03, _star(pan["pn"][step]))
    ax.set_ylim(ymin - span * 0.05, ymax + span * 0.52); ax.set_xticks(range(4))
    ax.set_xticklabels([AXIS_LABEL[DISPLAY[l]] for l in CONDITION_ORDER], fontsize=_ptx, rotation=0,
                       ha="center", linespacing=1.0)
    ax.set_ylabel(pan["ylab"]); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=3, width=1.0)


def tight_rect(pan, width):
    """axes box (left, bottom, right, top) that tight_layout gives one panel drawn alone"""
    fig, ax = plt.subplots(figsize=(width, 2.7), dpi=300); draw(ax, pan); fig.tight_layout()
    p = ax.get_position(); plt.close(fig)
    return p.x0, p.y0, p.x1, p.y1


def render(path, panels, width, rect=None):
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 2.7), dpi=300)
    for ax, pan in zip(np.atleast_1d(axes), panels):
        draw(ax, pan)
    if rect is None: fig.tight_layout()
    else: fig.subplots_adjust(left=rect[0], bottom=rect[1], right=rect[2], top=rect[3])
    out = _out(os.path.join(ROOT, path))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)
    print("wrote", path)


PAIR = [PANELS[0], PANELS[1]]
render(PAIR_PNG, PAIR, 2.35 * len(PAIR))
_r = [tight_rect(pan, 2.35) for pan in PAIR]
DE_RECT = (max(r[0] for r in _r), max(r[1] for r in _r), min(r[2] for r in _r), min(r[3] for r in _r))
for pan in PANELS:
    render(f"{pan.get('png_dir', PNG_DIR)}/{pan['stem']}{PNG_SUF}.png", [pan], 2.35,
           DE_RECT if any(pan is q for q in PAIR) else None)

SD = _SD_OVERRIDE or os.path.join(ROOT, "04_source_data/fig7")
_A0 = PANELS[0]["A"]


def _n_over(lab):
    s = _A0[lab].session
    return f"n = {len(s)} rollouts over {s.nunique()} sessions: " + "; ".join(sorted(s.unique()))


def _dec_mix(sessions):
    """How the Decoded condition splits over the two patients (scope text)."""
    n = len(sessions); p1 = int(sum(str(x).startswith("pat01") for x in sessions))
    return (f"is P1-dominated ({p1} of {n} trials)" if p1 >= 0.9 * n
            else f"holds both patients (P1 {p1} + P2 {n - p1} = {n} trials) but only two people")


CONDITION_DEF = {
    "condition_walker": (f"condition == 'wphase_matched' (calib; {WALKER_RHY}) — the pacing walker's own phase = the CEILING, not a "
                   "null, replayed on exactly the decoded window of each decoded trial (walker_window row). "
                   + _n_over("Walker") + ". Same (k1, lam = 0) cell as the Decoded condition.") if MATCHED else
                  "condition == 'wphase' (calib) — the pacing walker's own phase = the CEILING, not a null. " + _n_over("Walker") + ". ",
    "condition_decoded": "condition == 'dphase' (confirm) — EEG-decoded per-joint phase driving the template. " + _n_over("Decoded") + ".",
    "condition_scrambled": f"condition == 'dphase_iscr' (confirm) — inter-joint phase relationship scrambled. n = {len(_A0['CoordinationNull'])}.",
    "condition_shifted": f"condition == 'dphase_uscr' (confirm) — In-phase null: one random Fourier phase spectrum shared by all six joints of the decoded trajectory (02_decoding/simulation/data_io.py uscramble; each joint keeps its amplitude spectrum), so the joints move in phase and the decoded timing is destroyed . n = {len(_A0['InPhaseNull'])}.",
}
DEC_MIX = _dec_mix(_A0["Decoded"].session)
WALKER_MD = ("walker_window",
             f"window-matched: the walker's own phase replayed on exactly the decoded window of each of the "
             f"{len(_A0['Decoded'])} decoded trials (after the decoder's 2 s causal start-up), produced by "
             "02_decoding/simulation/replay_walker_matched.py via run_walker_matched_gate3s.sh on the compute server")
if ec.F7_SIM == "causal_from2s":
    WALKER_MD = (WALKER_MD[0], WALKER_MD[1].split("; the unmatched build")[0]
                 + "; under GAIT_F7_SIM=causal_from2s it is driven from trial 2 s by its own goniometer (no unmatched "
                   "walker is simulated)")
WALKER_IN = (f" (Decoded / CoordinationNull / InPhaseNull conditions); the Walker condition is read from {WALKER_RHY} (window-matched; "
             "walker_window row)") if MATCHED else ""
WALKER_COMP = (f" In both panels the Walker condition is read from {WALKER_RHY} (window-matched)."
               if MATCHED else "")
if RATE:
    WALKER_COMP = (f" In all three panels (d, e and the Supplementary step-length panel) the Walker condition is read "
                   f"from {WALKER_RHY} (window-matched).")
    RATE_MD = {
        "status": ("panel d shows the conventional cadence n_adv_events / dur_s, "
                   "panel e the left-right step asymmetry, and the step length "
                   "dx_m / n_adv_events is a Supplementary panel."),
        "companion": (f"the Fig. 7 panels are figures_out/fig7/fig7_sim_cadence.png (d, cadence) and "
                      "figures_out/fig7/fig7_sim_step_symmetry.png (e, advancing-step count imbalance); the Supplementary panel "
                      "figures_out/supplementary/sup_fig7_sim_step_length.png draws the step length; all three read "
                      f"{RHYTHM}. The pair PNG {PAIR_PNG} is d and e side by side. This panel uses "
                      "the file named in input_file."),
        "rationale": ("The star tests the per-trial paired difference "
                      "(test_rule row), with a session random intercept for trials within a session; the boxes "
                      "still show MEDIANS, descriptively."),
        "multiple": ("reported uncorrected; the three panels (d, e and the Supplementary step-length panel) share "
                     "the same rollouts and the two nulls are re-runs of the same trials, so the tests are NOT "
                     "independent."),
        "repro": (f"ONE shared numpy default_rng({SEED}), used for the point jitter only (no statistic in this build "
                  "draws from it). The panels are computed in the order cadence, left-right asymmetry, step length; "
                  "before each panel's jitter the stream is advanced by B sign-flip draws per paired test; step length continues the same stream. "
                  
                  
                  "The jitter is cached so the "
                  "single-panel PNGs reuse the pair's exact point positions. Arial, font.size 6.5 x 1.35, no "
                  "in-figure title; pair 4.7 x 2.7 in and each single panel 2.35 x 2.7 in, 300 dpi."),
    }
if LME:
    _e = PANELS[1]; _vw, _vd = _e["vals"]["Walker"], _e["vals"]["Decoded"]; _r = _e["pw"]
    SYM_CAVEAT = (f"Walker median {np.median(_vw):.3f} vs Decoded median {np.median(_vd):.3f} (the boxes, descriptive "
                  "only). The drawn star is the test_rule test of the per-trial Walker - Decoded difference (stats row "
                  f"Walker_minus_Decoded): {_r['test_used']}, p = {_r['p_used']:.3g} ({_r['star']}); LME mean difference "
                  f"{_r['mean_difference_lme']:+.6f} (SE {_r['se_lme']:.6f}, Wald p = {_r['p_lme']:.3g}, session variance "
                  f"{_r['session_variance']:.3g}), Shapiro-Wilk p = {_r['shapiro_p']:.3g}; median of the per-trial "
                  f"difference {_r['median_difference']:+.6f}. The star refers to the paired per-trial differences, not "
                  f"to the gap between the two box medians. Upper tail: {100 * np.mean(_vw >= 0.33):.1f} % of Walker "
                  f"trials vs {100 * np.mean(_vd >= 0.33):.1f} % of Decoded trials have an imbalance >= 0.33.")
elif MATCHED:
    _e = PANELS[1]; _vw, _vd = _e["vals"]["Walker"], _e["vals"]["Decoded"]
    SYM_CAVEAT = (f"Walker median {np.median(_vw):.3f} vs Decoded median {np.median(_vd):.3f}; the drawn star is the "
                  "PAIRED within-trial sign-flip test on the median of the per-trial Walker - Decoded difference "
                  f"(observed {_e['pw'][0]:+.6f}, P = {_e['pw'][1]:.4f}, {stars(_e['pw'][1])}; stats row "
                  "Walker_minus_Decoded; the window-matched walker replays the same trials). The box shows the "
                  "per-condition medians; the paired statistic is the median of the per-trial difference, which need not "
                  f"equal the difference of the two medians. Upper tail: {100 * np.mean(_vw >= 0.33):.1f} % of "
                  f"Walker trials vs {100 * np.mean(_vd >= 0.33):.1f} % of Decoded trials have an imbalance >= 0.33. "
                  "No mean-based test is computed here.")
else:
    _e = PANELS[1]; _vw, _vd = _e["vals"]["Walker"], _e["vals"]["Decoded"]
    SYM_CAVEAT = (f"Walker median {np.median(_vw):.3f} vs Decoded median {np.median(_vd):.3f}; the drawn star is the "
                  f"permutation test on the MEDIAN difference (P = {_e['pw'][1]:.4f}, {stars(_e['pw'][1])}; stats row "
                  "Walker_minus_Decoded), because the box shows medians. The two conditions also differ in the UPPER "
                  f"TAIL: {100 * np.mean(_vw >= 0.33):.1f} % of Walker trials vs {100 * np.mean(_vd >= 0.33):.1f} % of "
                  "Decoded trials have an imbalance >= 0.33. No mean-based test is computed here; the drawn star is the median test.")
if FDR_STARS:
    _r = PANELS[1]["pw"]
    SYM_CAVEAT = (SYM_CAVEAT
                  .replace("The drawn star is the test_rule test of", "The drawn star (bracket_star_rule row) is the "
                           "BH-FDR p over the panel's three contrasts of the test_rule test of")
                  .replace(f"p = {_r['p_used']:.3g} ({_r['star']})",
                           f"p = {_r['p_used']:.3g}, pFDR = {_r['p_fdr_bh']:.3g} ({_r['star_drawn']})"))

report = []
for pan in PANELS:
    key, stem, A = pan["key"], pan["stem"], pan["A"]
    med = {l: float(np.median(pan["vals"][l])) for l in CONDITION_ORDER}
    TESTS = [("Walker_minus_Decoded", pan["pw"]), ("Decoded_minus_CoordinationNull", pan["pn"][0]),
             ("Decoded_minus_InPhaseNull", pan["pn"][1])]
    if LME:
        report.append(f"{key}: medians W {med['Walker']:.3f} D {med['Decoded']:.3f} iscr {med['CoordinationNull']:.3f} "
                      f"uscr {med['InPhaseNull']:.3f} | "
                      + " | ".join(f"{c} mean {r['mean_difference_lme']:+.3f} {r['test_used']} p={r['p_used']:.2g} "
                                   f"{r['star']}" for c, r in TESTS) + f" | n D={len(A['Decoded'])}")
        if FDR_STARS and pan["slot"] in FDR_FAM:
            report.append(f"  {key} DRAWN (BH-FDR over 3): " + " | ".join(
                f"{c} p_used {r['p_used']:.6g} ({r['star']}) -> pFDR {r['p_fdr_bh']:.6g} ({r['star_drawn']})"
                for c, r in TESTS))
    else:
        report.append(f"{key}: medians W {med['Walker']:.3f} D {med['Decoded']:.3f} iscr {med['CoordinationNull']:.3f} "
                      f"uscr {med['InPhaseNull']:.3f} | W-D median-perm p={pan['pw'][1]:.3g} {stars(pan['pw'][1])} | "
                      f"D-iscr p={pan['pn'][0][1]:.3g} {stars(pan['pn'][0][1])} | D-uscr p={pan['pn'][1][1]:.3g} "
                      f"{stars(pan['pn'][1][1])} | n D={len(A['Decoded'])}")

    with open(_out(os.path.join(os.path.dirname(SD), ec.sd_group(stem), f"{stem}_source_data.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["condition_label", "stage", "condition", "session", "trial_idx", "value"])
        for lab in CONDITION_ORDER:
            d = A[lab][["stage", "condition", "session", "trial_idx", key]].dropna(subset=[key])
            for r in d.itertuples(index=False):
                w.writerow([lab, r.stage, r.condition, r.session, int(r.trial_idx), f"{float(getattr(r, key)):.6f}"])

    with open(_out(os.path.join(os.path.dirname(SD), ec.sd_group(stem), f"{stem}_stats.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if LME and FDR_STARS and pan["slot"] in FDR_FAM:
            _k = LME_COLS.index("star") + 1
            w.writerow(LME_COLS[:_k] + ["p_fdr_bh", "star_drawn"] + LME_COLS[_k:])
            for c, r in TESTS:
                _row = lme_csv_row(c, r, pan.get("scheme_pre", "") + LME_SCHEME)
                w.writerow(_row[:_k] + ["{:.6g}".format(r["p_fdr_bh"]), r["star_drawn"]] + _row[_k:])
        elif LME:
            w.writerow(LME_COLS)
            w.writerows(lme_csv_row(c, r, pan.get("scheme_pre", "") + LME_SCHEME) for c, r in TESTS)
        else:
            w.writerow(["comparison", "statistic", "observed", "p_perm", "star", "n_a", "n_b", "B", "seed", "scheme"])
            obs, p, na, nb, sess = pan["pw"]
            w.writerow(["Walker_minus_Decoded", "median of the paired difference" if MATCHED else "median difference",
                        f"{obs:.6f}", f"{p:.6f}", stars(p), na, nb, B, SEED,
                        "paired within-trial sign-flip permutation on the per-trial difference (the window-matched "
                        "walker replays the SAME trials over exactly the decoded samples); two-sided on |median|; "
                        "P = (count + 1) / (B + 1)" if MATCHED else
                        "unpaired session-stratified permutation: condition labels permuted WITHIN session over the "
                        f"{len(sess)} shared sessions ({'; '.join(sess)}); two-sided on |median difference|; "
                        "P = (count + 1) / (B + 1)"])
            for name, (obs, p, na, nb, _) in zip(("CoordinationNull", "InPhaseNull"), pan["pn"]):
                w.writerow([f"Decoded_minus_{name}", "median of the paired difference", f"{obs:.6f}", f"{p:.6f}",
                            stars(p), na, nb, B, SEED,
                            "paired within-trial sign-flip permutation on the per-trial difference (the null re-runs "
                            "the SAME trials with the phases swapped); two-sided on |median|; P = (count + 1) / (B + 1)"])

    md = [("figure", (f"Fig. 7 panel {pan['slot']} — {pan['ylab']} over four conditions -> figures_out/fig7/{stem}.png; "
                      f"also drawn as the left/right half of the pair {PAIR_PNG}. "
                      "") if not RATE else
                     (f"Supplementary panel — {pan['ylab']} over four conditions -> supplementary/{stem}.png."
                      )
                     if pan["slot"] == "SI" else
                     (f"Fig. 7 panel {pan['slot']} — {pan['ylab']} over four conditions -> figures_out/fig7/{stem}.png"
                      + f"; also drawn as the {'left' if pan['slot'] == 'd' else 'right'} half of the d,e pair "
                        f"{PAIR_PNG}"
                      + ". Panel d shows the conventional cadence and panel e the left-right step "
                        "asymmetry; the step length is a Supplementary panel.")),
          ("build_id", COHORT),
          ("cohort", f"{COHORT} (P1 sess04-09 x6 + P2 sess03/05/06/08/09 x5); the simulation uses 9 of the 11 sessions. "
                     
                     "n = 2 patients, case-level."),
          ] + ([GATE_MD] if GATE_MD else []) + ([WALKER_MD] if MATCHED else []) + ([ec.F7_SIM_MD] if ec.F7_SIM else []) + [
          ("input_file", pan["src"] + WALKER_IN),
          ("input_filter", ("stage == 'calib' and condition == 'wphase_matched' for the Walker condition (the window-matched "
                            "walker replay) " if MATCHED else
                            "stage == 'calib' for the Walker condition (the walker-phase conditions live in the calib sweep) ")
                           + "and stage == 'confirm' for Decoded / CoordinationNull / InPhaseNull; then phase_src == 'k1' and "
                           "lam == 0.0 on every condition."),
          ] + list(CONDITION_DEF.items())
    md += [("condition_display_labels", "; ".join(f"{l} -> {DISPLAY[l]!r} (axis {AXIS_LABEL[DISPLAY[l]]!r})" for l in CONDITION_ORDER)
            + " (the nulls are abbreviated on the axis only, because the full name collides"
            + " with Decoded at this panel width; the internal keys are unchanged)"),
           ("n_per_condition", "; ".join(f"{l} {len(pan['vals'][l])}" for l in CONDITION_ORDER)),
           ("n_sessions_contributing", "; ".join(f"{l} {A[l].session.nunique()}" for l in CONDITION_ORDER)),
           ("sessions_per_condition", " | ".join(f"{l}: " + "; ".join(sorted(A[l].session.unique())) for l in CONDITION_ORDER)),
           ("metric_definition", pan["metric"] + ("" if key != "step_length_m" else
                                  " Dropped for n_adv_events == 0: "
                                  + "; ".join(f"{l} {int(A[l][key].isna().sum())}" for l in CONDITION_ORDER) + ".")),
           ("medians_per_condition", "; ".join(f"{l} {med[l]:.6f}" for l in CONDITION_ORDER)),
           ("multiple_comparisons", RATE_MD["multiple"] if RATE else
                                    "reported uncorrected; the two panels share the same rollouts and the two "
                                    "nulls are re-runs of the same trials, so the tests are NOT independent."),
           ("producer", "03_figures/fig7_sim_cadence_symmetry.py")]
    if key == "r5_lr_asym":
        md.insert(len(md) - 1, ("caveat_median_vs_mean", SYM_CAVEAT))
    if pan["slot"] == "SI":
        md.insert(1, ("paper_location", "Supplementary Fig. 8"))
    if LME:
        md.insert([k for k, _ in md].index("test_definition"), ("test_rule", TEST_RULE))
    if FDR_STARS and pan["slot"] in FDR_FAM:
        _ks = [k for k, _ in md]
        md[_ks.index("test_definition")] = ("test_definition", md[_ks.index("test_definition")][1].replace(
            ". Stars *** / ** / * / ns at 1e-3 / 1e-2 / 0.05 on p_used. Table: ",
            ". Bracket stars *** / ** / * / ns at 1e-3 / 1e-2 / 0.05 on p_fdr_bh (BH-FDR over this panel's 3 "
            "contrasts; bracket_star_rule row); the stats column star keeps the uncorrected "
            "star of p_used. Table: "))
        assert "on p_fdr_bh" in md[_ks.index("test_definition")][1]
        md[_ks.index("multiple_comparisons")] = (
            "multiple_comparisons",
            "BH-FDR over this panel's three contrasts (Walker_minus_Decoded, Decoded_minus_CoordinationNull, "
            "Decoded_minus_InPhaseNull) sets the DRAWN bracket star (bracket_star_rule row; stats "
            "columns p_fdr_bh / star_drawn); p_used and star are uncorrected. The three panels (d, e and the "
            "Supplementary step-length panel) share the same rollouts and the two nulls are re-runs of the same "
            "trials, so the tests are NOT independent; the Supplementary step-length panel stays uncorrected.")
        md.insert(_ks.index("test_rule") + 1, (
            "bracket_star_rule",
            "GAIT_F7_SIM=causal_from2s only: the bracket stars of Fig. 7d and 7e are drawn from p_fdr_bh = "
            "Benjamini-Hochberg FDR over this panel's three contrasts, computed from p_used as written in the stats "
            "CSV with statsmodels multipletests(method='fdr_bh') - the same family and computation as "
            f"03_figures/results_statistics.py (family F15-{FDR_FAM[pan['slot']]}), asserted equal to its Source Data "
            "results_statistics_source_data.csv to 1e-6; *** / ** / * / ns at 0.001 / 0.01 / 0.05 on p_fdr_bh (stats "
            "column star_drawn). The stats column star and the test_rule sentence on p_used remain the uncorrected "
            "star (the rule, which still applies to every other build and to the Supplementary "
            "step-length panel). Drawn: "
            + "; ".join(f"{c} p_used {r['p_used']:.6g} ({r['star']}) -> pFDR {r['p_fdr_bh']:.6g} ({r['star_drawn']})"
                        for c, r in TESTS)))
    with open(_out(os.path.join(os.path.dirname(SD), ec.sd_group(stem), f"{stem}_metadata.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["key", "value"])
        w.writerows([(k, _md_fix(v)) for k, v in
                     ((md + [SUPPORT_MD]) if SUPPORT_MD else md)])
    print("wrote", f"{os.path.relpath(SD, ROOT)}/{stem}_{{source_data,stats,metadata}}.csv")

print("\n".join(report))
