"""Figure 4 — Shared locomotor representation across imagined and real walking

RW vs IW decoding gain + subject x task test (trial-wise r as sample)

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
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _c in ("/System/Library/Fonts/Supplemental/Arial.ttf",
           "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    if Path(_c).exists():
        fm.fontManager.addfont(_c)
import panel_style as PS
PS.apply(plt)
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
import analysis_config as ec

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data/decoder_output"
import os as _os
R_CONV = _os.environ.get("GAIT_R_CONV", "trial_median")
if R_CONV not in ("session_mean", "trial_median"):
    raise SystemExit(f"GAIT_R_CONV must be 'session_mean' or 'trial_median', got {R_CONV!r}")
TM_CSV = ec.sd("fig4_trialmedian_source_data.csv")
IW_CSV = ec.path("h_iw_summary_csv")
RW_CSV = ec.path("h_rw_summary_csv")
import re as _re
_tags = set(_re.findall(r"main(?:)?|" + _re.escape(ec.TAG),
                        ec.rel("h_iw_summary_csv") + " " + ec.rel("h_rw_summary_csv")))
if not _tags:
    raise SystemExit("INPUT CHECK: no decode tag in the IW/RW input filenames; "
                     "refusing to stamp an build_id.")
if len(_tags) > 1:
    raise SystemExit(f"INPUT CHECK: mixed-build_id IW/RW inputs {sorted(_tags)}; "
                     "one cohort+version per figure.")
BUILD_ID_DERIVED = _tags.pop()
ec.check_tag(BUILD_ID_DERIVED, "decoder", "IW/RW fusion input build_id")
OUT = ec.scoped(ROOT / "figures_out/fig4/fig4_decoding_gain.png")
SD = ec.sd("fig4_decoding_gain_source_data.csv")
from matplotlib.colors import to_rgba
import project_palette as P
IWC, RWC = P.IW_HUE, P.RW_HUE
STYLE = {t: dict(edge=c, fill=to_rgba(c, 0.32), median=c, dot_face=to_rgba(c, 0.85), dot_edge="none", lw=1.2)
         for t, c in (("IW", IWC), ("RW", RWC))}


def load(p, task):
    d = pd.read_csv(p)[["session", "fold", "joint", "rpt_calib"]].copy()
    d["task"] = task
    d["subject"] = d.session.str.extract(r"(subj\d+)")
    return d.dropna(subset=["rpt_calib"])


iw, rw = load(IW_CSV, "IW"), load(RW_CSV, "RW")
df = pd.concat([iw, rw], ignore_index=True)

def _pfmt(p):
    return "<1e-300" if p == 0 else f"{p:.3e}"


mw_U, mw_p = stats.mannwhitneyu(iw.rpt_calib, rw.rpt_calib, alternative="less")
if R_CONV == "trial_median":
    _tm = pd.read_csv(TM_CSV).set_index("session")
    si = _tm["iw_median"]; sr = _tm["rw_median"]
else:
    si = iw.groupby("session").rpt_calib.mean(); sr = rw.groupby("session").rpt_calib.mean()
common = si.index.intersection(sr.index)
t_p = stats.ttest_rel(sr[common], si[common])
w_res = stats.wilcoxon(sr[common], si[common], alternative="two-sided")
bracket_p = float(w_res.pvalue) if R_CONV == "trial_median" else float(t_p.pvalue)
bracket_test = "session_wilcoxon" if R_CONV == "trial_median" else "session_paired_t"
n_up = int((sr[common] > si[common]).sum())
gain_trial = rw.rpt_calib.mean() - iw.rpt_calib.mean()

aov = sm.stats.anova_lm(smf.ols("rpt_calib ~ C(task)*C(subject)", data=df).fit(), typ=2)
F_task = aov.loc["C(task)", "F"]; p_task = aov.loc["C(task)", "PR(>F)"]
F_int = aov.loc["C(task):C(subject)", "F"]; p_int = aov.loc["C(task):C(subject)", "PR(>F)"]
if R_CONV == "trial_median":
    _sub = pd.DataFrame({"IW": si[common], "RW": sr[common]})
    _sub["subject"] = pd.Series(common, index=common).str.extract(r"(subj\d+)", expand=False)
    g = _sub.groupby("subject")[["IW", "RW"]].mean()
else:
    g = df.groupby(["subject", "task"]).rpt_calib.mean().unstack()
g["gain"] = g["RW"] - g["IW"]
w_sub = stats.wilcoxon(g["RW"], g["IW"], alternative="two-sided")
t_sub = stats.ttest_rel(g["RW"], g["IW"])
n_subj = len(g); n_subj_up = int((g["gain"] > 0).sum())

print(f"trial-wise IW {iw.rpt_calib.mean():.3f} -> RW {rw.rpt_calib.mean():.3f}  (gain +{gain_trial:.3f})")
print(f"  Mann-Whitney (trial-wise) p={mw_p:.2e}")
print(f"  r convention: {R_CONV}")
print(f"  session paired-t p={t_p.pvalue:.2e}  ({n_up}/{len(common)} sessions RW>IW)")
print(f"  session Wilcoxon W={w_res.statistic:.1f} p={w_res.pvalue:.2e}"
      f"  | cohort median IW {si[common].median():.4f} RW {sr[common].median():.4f}")
print(f"  2-way ANOVA: task F={F_task:.1f} p={p_task:.1e} | task x subject F={F_int:.2f} p={p_int:.2e}")
print(f"  per-subject gain: {n_subj_up}/{n_subj} subjects RW>IW | gain mean {g.gain.mean():.3f} sd {g.gain.std():.3f} range [{g.gain.min():.2f},{g.gain.max():.2f}]")

COND_FILL_ALPHA = 0.32
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

sess = pd.DataFrame({"IW": si[common], "RW": sr[common]}).reset_index().rename(columns={"index": "session"})
sess["subject"] = sess.session.str.extract(r"(subj\d+)")
rng = np.random.default_rng(0)
X = {"IW": 0.0, "RW": 1.0}
DOT_OFF = {"IW": -0.42, "RW": 0.42}
jit = rng.uniform(-0.07, 0.07, len(sess))
xdot = {t: X[t] + DOT_OFF[t] + jit for t in ("IW", "RW")}
half, cap = 0.20, 0.09

fig, ax = PS.panel_axes(plt, left_mm=15.5, bottom_mm=13.0)
_LW_PAIR, _S_PT, _LW_MED, _LW_BRK, _PT_STAR = ((0.45, 5, 1.6, 1.0, PS.STAR_PT))
for k_ in range(len(sess)):
    ax.plot([xdot["IW"][k_], xdot["RW"][k_]], [sess.IW.iloc[k_], sess.RW.iloc[k_]],
            color="#B3B3B3", lw=_LW_PAIR, alpha=0.7, zorder=1, solid_capstyle="round")
box_stats = {}
for t, col in (("IW", IWC), ("RW", RWC)):
    v = sess[t].values
    p10, q25, med, q75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    box_stats[t] = dict(p10=p10, q25=q25, median=med, q75=q75, p90=p90, mean=v.mean(), n=len(v))
    x = X[t]
    st = STYLE[t]
    ax.scatter(xdot[t], v, s=_S_PT, facecolor=st["dot_face"], edgecolor=st["dot_edge"], linewidths=0.4, zorder=2)
    for a, z in ((p10, q25), (q75, p90)):
        ax.plot([x, x], [a, z], color=st["edge"], lw=st["lw"], zorder=3, solid_capstyle="butt")
    for c in (p10, p90):
        ax.plot([x - cap, x + cap], [c, c], color=st["edge"], lw=st["lw"], zorder=3)
    ax.add_patch(Rectangle((x - half, q25), 2 * half, q75 - q25,
                           facecolor=st["fill"], edgecolor=st["edge"], lw=st["lw"], zorder=4))
    ax.plot([x - half, x + half], [med, med], color=st["median"], lw=_LW_MED, zorder=5, solid_capstyle="butt")
ytop = max(sess.RW.max(), sess.IW.max()) + 0.04
ax.plot([X["IW"], X["IW"], X["RW"], X["RW"]], [ytop, ytop + 0.025, ytop + 0.025, ytop],
        color="#1A1A1A", lw=_LW_BRK, zorder=6, solid_capstyle="butt")
ax.text((X["IW"] + X["RW"]) / 2, ytop + 0.03, stars(bracket_p), ha="center", va="bottom",
        fontsize=_PT_STAR, color="#1A1A1A")
ax.set_xlim(-0.75, 1.75); ax.set_xticks([0, 1])
ax.set_xticklabels(["Imagined\nwalking", "Real\nwalking"])
ax.set_ylabel("Decoding accuracy ($r$)")
ax.set_ylim(0.0, 1.05); ax.set_yticks(np.arange(0, 1.01, 0.2))
PS.finish(ax)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=PS.DPI, facecolor="white")
plt.close(fig)
print("wrote", OUT)
print("  session boxes: " + "  ".join(f"{t} median {b['median']:.3f} IQR {b['q25']:.3f}-{b['q75']:.3f} "
                                         f"p10-p90 {b['p10']:.3f}-{b['p90']:.3f}" for t, b in box_stats.items()))

SD_SESS = SD.with_name("fig4_decoding_gain_sessions_source_data.csv")
sess_out = sess[["session", "subject", "IW", "RW"]].copy()
sess_out["x_dot_IW"], sess_out["x_dot_RW"] = xdot["IW"], xdot["RW"]
sess_out.to_csv(SD_SESS, index=False, float_format="%.6f")
with open(SD_SESS, "a") as fh:
    _unit = "session means" if R_CONV == "session_mean" else "per-session trial medians"
    fh.write(f"\n# box statistics ({_unit}, n=80 each)\n")
    for t, b in box_stats.items():
        fh.write(f"# {t},median,{b['median']:.6f},q25,{b['q25']:.6f},q75,{b['q75']:.6f},p10,{b['p10']:.6f},p90,{b['p90']:.6f},mean,{b['mean']:.6f},n,{b['n']}\n")
    fh.write(f"# bracket,{bracket_test}_p,{bracket_p:.3e},stars,{stars(bracket_p)},sessions_RW_gt_IW,{n_up}/{len(common)}\n")
    fh.write(f"# build_id,{BUILD_ID_DERIVED}\n# colours,IW {IWC} / RW {RWC}, dark edge with 32 % light fill\n")
    if R_CONV != "session_mean":
        fh.write(f"# r_convention,{R_CONV}\n")
        fh.write("# session_value,per-session median over trials of the per-trial mean over 6 joints (= Fig.2c All median)\n")
        fh.write(f"# session_value_source,{TM_CSV.name}\n")
        fh.write(f"# cohort_median_IW,{si[common].median():.6f},cohort_median_RW,{sr[common].median():.6f}\n")
        fh.write(f"# session_wilcoxon_W,{w_res.statistic:.1f},p,{w_res.pvalue:.3e}\n")
        fh.write(f"# session_paired_t,{t_p.statistic:.3f},df,{len(common)-1},p,{t_p.pvalue:.3e},reference_only\n")
print("wrote", SD_SESS)

SD.parent.mkdir(parents=True, exist_ok=True)
g.reset_index()[["subject", "IW", "RW", "gain"]].to_csv(SD, index=False)
with open(SD, "a") as fh:
    fh.write("\n# provenance\n")
    fh.write(f"# cohort,{BUILD_ID_DERIVED},sessions,{df.session.nunique()},subjects,{df.subject.nunique()}\n")
    fh.write(f"# iw_input,{IW_CSV.name}\n# rw_input,{RW_CSV.name}\n")
    fh.write(f"# rows_entering_stats_IW,{len(iw)},rows_entering_stats_RW,{len(rw)}\n")
    fh.write("\n# stats\n")
    fh.write(f"# trial_wise_gain,{gain_trial:.4f}\n# mannwhitney_p,{_pfmt(mw_p)}\n")
    fh.write(f"# session_paired_t_p,{t_p.pvalue:.3e},sessions_RW_gt_IW,{n_up}/{len(common)}\n")
    if R_CONV != "session_mean":
        fh.write(f"# r_convention,{R_CONV}\n")
        fh.write(f"# session_wilcoxon_W,{w_res.statistic:.1f},p,{w_res.pvalue:.3e}\n")
        fh.write("# subject_unit,per-participant mean of that participant's session medians\n")
        fh.write(f"# subject_wilcoxon_W,{w_sub.statistic:.1f},p,{w_sub.pvalue:.3e},subject_paired_t,{t_sub.statistic:.3f},df,{len(g)-1},p,{t_sub.pvalue:.3e}\n")
        fh.write(f"# subjects_RW_gt_IW,{n_subj_up}/{n_subj}\n")
    fh.write(f"# anova_task_F,{F_task:.2f},p,{_pfmt(p_task)}\n# anova_taskXsubject_F,{F_int:.3f},p,{_pfmt(p_int)}\n")
print("wrote", SD)


import csv as _csv
import os as _osw
from datetime import date as _date


def _write_meta(_path, _rows):
    _tmp = _path.parent / (_path.name + ".tmp")
    with _tmp.open("w", newline="", encoding="utf-8") as _f:
        _w = _csv.writer(_f)
        _w.writerow(["key", "value"])
        _w.writerows([(k, v) for k, v in _rows])
    _osw.replace(_tmp, _path)
    print("wrote", _path)


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


_SHARED = [
    ("producer", "03_figures/fig4_decoding_gain.py"),
    ("generated", _date.today().isoformat()),
    ("library_versions", _libvers(("numpy", "scipy", "pandas", "matplotlib", "statsmodels"))),
    ("cohort", f"healthy; {df.session.nunique()} sessions / {df.subject.nunique()} participants"),
    ("input_iw", str(IW_CSV.relative_to(ROOT)).replace("\\", "/")),
    ("input_rw", str(RW_CSV.relative_to(ROOT)).replace("\\", "/")),
    ("r_convention", R_CONV),
    ("session_statistic", "per-session median over trials of the per-trial mean over the six "
                          "joints (= the Fig. 2c 'All' median)" if R_CONV == "trial_median"
                          else "mean of that session's per-fold x per-joint rpt_calib"),
    ("bracket_test", bracket_test + f" (p = {bracket_p:.3e}); the bracket quotes the test that "
                     "matches the plotted statistic"),
    ("n_sessions", int(df.session.nunique())),
    ("n_subjects", int(df.subject.nunique())),
    ("n_rows_entering_stats_iw", len(iw)),
    ("n_rows_entering_stats_rw", len(rw)),
]
if R_CONV == "trial_median":
    _SHARED.append(("input_trial_median",
                    str(TM_CSV.relative_to(ROOT)).replace("\\", "/")))
_write_meta(SD.parent / "fig4_decoding_gain_metadata.csv", _SHARED + [
    ("figure", "Fig. 4 panel b - RW vs IW decoding gain; THIS file carries the "
               "PER-PARTICIPANT IW / RW / gain values"),
    ("rows", "one per participant"),
    ("n_rows", len(g)),
    ("outputs", OUT.relative_to(ROOT).as_posix() + "; "
                + SD.relative_to(ROOT).as_posix()),
])
_write_meta(SD.parent / "fig4_decoding_gain_sessions_metadata.csv", _SHARED + [
    ("figure", "Fig. 4 panel b - RW vs IW decoding gain; THIS file carries the "
               "PLOTTED PER-SESSION values and the box statistics"),
    ("rows", "one per session"),
    ("n_rows", len(sess_out)),
    ("outputs", OUT.relative_to(ROOT).as_posix() + "; "
                + SD_SESS.relative_to(ROOT).as_posix()),
])
