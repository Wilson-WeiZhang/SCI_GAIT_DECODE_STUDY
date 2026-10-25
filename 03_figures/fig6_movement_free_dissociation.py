#!/usr/bin/env python3
"""Figure 6 — Representational geometry of patient imagined walking

fig6_movement_free_dissociation.py

Reads:
  decoder output in data/decoder_output/ (not distributed)
  the aligned per-session recordings (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import csv
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

import project_palette as P
import panel_style as PS
from sup_panel_export import PANEL_MODE

for _c in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
           "/System/Library/Fonts/Supplemental/Arial.ttf"):
    if Path(_c).exists():
        fm.fontManager.addfont(_c)
import fig6_panel_style as S
plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.linewidth": 2.2,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic"})
if PANEL_MODE:
    PS.apply(plt)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "decoder_output"
NSUR_TAG = os.environ.get("GAIT_TASK3_NSUR", "1000")
_T3DIRS = {"200": ("diag_patient_task_expression",
                   "diag_patient_task_expression_session08",
                   "diag_patient_task_expression_session09"),
           "1000": ("diag_task3_adaptive_ovr_nsur1000",
                    "diag_task3_adaptive_sess08_ovr_nsur1000",
                    "diag_task3_adaptive_sess09_ovr_nsur1000")}[NSUR_TAG]
T3 = RES / _T3DIRS[0] / "task3_adaptive.csv"
ERD = RES / "diag_erd_similarity_ovr" / "erd_roi.csv"
T3_S08 = RES / _T3DIRS[1] / "task3_adaptive.csv"
ERD_S08 = RES / "diag_erd_similarity_sess08_ovr" / "erd_roi.csv"
T3_S09 = RES / _T3DIRS[2] / "task3_adaptive.csv"
ERD_S09 = RES / "diag_erd_similarity_sess09_ovr" / "erd_roi.csv"
SIDECARS = {"pat02_sess08": (T3_S08, ERD_S08), "pat02_sess09": (T3_S09, ERD_S09)}
OUT = ROOT / "figures_out/fig6/fig6_movement_free_dissociation.png"
OUT_LEGEND = ROOT / "figures_out/fig6/fig6_movement_free_dissociation_legend.png"
OUT_PHASELOCK = ROOT / "figures_out/fig6/fig6_movement_free_dissociation_phaselock.png"
SRC = ROOT / "04_source_data/fig6/fig6_movement_free_dissociation_source_data.csv"
META = ROOT / "04_source_data/fig6/fig6_movement_free_dissociation_metadata.csv"

def _ms_mm(v):
    return (round(v / 25.4 * 600) + 0.25) / 600 * 25.4

DIAG_TAG = (f"{_T3DIRS[0]} + "
            "diag_erd_similarity_ovr "
            f"+ pat02_sess08 sidecar {_T3DIRS[1]} + "
            "diag_erd_similarity_sess08_ovr (same diagnostics, sess08 prep root) "
            f"+ pat02_sess09 sidecar {_T3DIRS[2]} + "
            "diag_erd_similarity_sess09_ovr (same diagnostics, sess09 prep root)"
            + (f" [task3 surrogate draws NSUR={NSUR_TAG}; the three task3 dirs named are the "
               f"_nsur1000 runs]" if NSUR_TAG != "200" else ""))
PREP = ("01_preprocessing/patient prep (frozen) "
        "+ the separate patient_prep_sess08 root for pat02_sess08 "
        "+ the separate patient_prep_sess09 root for pat02_sess09")

POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
            "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
            "pat02_sess09"]
PRE_OPERATIVE = ["pat01_sess01", "pat01_sess02", "pat02_sess01", "pat02_sess02"]

TASKS = ["IW", "RW", "S2S"]
KS = [1, 2, 3]
TCOL = {"IW": P.IW_HUE, "RW": P.RW_HUE, "S2S": "#7B4F2E"}
FLOOR_COL = P.NULL_FILL
TLAB = {"IW": "imagined\nwalking", "RW": "real\nwalking", "S2S": "sit-to-\nstand"}
TFULL = {"IW": "imagined walking", "RW": "real walking", "S2S": "sit-to-stand"}
KLAB = {1: "$\\mathrm{k}_1$\n(stride)", 2: "$\\mathrm{k}_2$\n(step)", 3: "$\\mathrm{k}_3$\n(3rd)"}
LN2DB = 10.0 * np.log10(np.e)
BANDS = [("mu", "Mu (8–13 Hz)"), ("beta", "Beta (13–30 Hz)")]


def _panel_spec_check(fig, axes, allow_texts=()):
    """only. Raises before anything is written if a title, a legend, or any text other than the
    explicitly allowed axis-label-like entries is still inside the image."""
    bad = []
    for i, ax in enumerate(axes):
        if ax.get_title():
            bad.append(f"axes[{i}] title {ax.get_title()!r}")
        if ax.get_legend() is not None:
            bad.append(f"axes[{i}] legend present")
        extra = [t.get_text() for t in ax.texts if t.get_text() not in allow_texts]
        if extra:
            bad.append(f"axes[{i}] texts {extra}")
    if fig.texts:
        bad.append(f"fig.texts {[t.get_text() for t in fig.texts]}")
    if bad:
        raise SystemExit("PANEL SPEC VIOLATION: " + "; ".join(bad))
    print(f"panel spec OK: {len(axes)} axes, no title / legend / in-figure text")


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def agg_c(df, task, k):
    """Unchanged aggregation: across-session mean |c| and mean surrogate, SEM over sessions,
    and the MEDIAN of the per-session surrogate p."""
    m = df[(df.cond == task) & (df.harmonic_k == k) & df.c_abs.notna()]
    return dict(c=m.c_abs.mean(), sem=m.c_abs.sem(), surr=m.c_surrogate.mean(),
                p=m.surrogate_p.median(), n=len(m), sess=sorted(m.session),
                n_cycles=m.n_cycles.mean(), n_seg=m.n_seg.mean(),
                n_sig=int((m.surrogate_p < 0.05).sum()))


def agg_erd_db(df, task, band):
    """ROI ERD in dB log-ratio, matching Fig.3b's metric. p = two-sided Wilcoxon signed-rank
    of the per-session values against 0 (sessions as units; case-level descriptive, n=2
    patients — a session-level consistency statement, never a population test)."""
    from scipy.stats import wilcoxon
    m = df[(df.task == task) & (df.band == band) & df.roi_erd.notna()]
    v = m.roi_erd * LN2DB
    return dict(db=v.mean(), sem=v.sem(), n=len(m), sess=sorted(m.session),
                n_neg=int((v < 0).sum()), p=wilcoxon(v).pvalue,
                per=sorted(zip(m.session.tolist(), v.tolist())))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SRC.parent.mkdir(parents=True, exist_ok=True)
    streams = {"task3": [pd.read_csv(T3)], "erd_roi": [pd.read_csv(ERD)]}
    for _sess, (_t3s, _erds) in SIDECARS.items():
        _a, _b = pd.read_csv(_t3s), pd.read_csv(_erds)
        for _n, _d, _f in (("task3", _a, _t3s), ("erd_roi", _b, _erds)):
            _ids = set(_d.session.unique())
            if _ids != {_sess}:
                raise SystemExit(f"{_f.name} ({_n}) must hold exactly {_sess}, got: {sorted(_ids)}")
        streams["task3"].append(_a)
        streams["erd_roi"].append(_b)
    for _n, _parts in streams.items():
        _seen: set[str] = set()
        for _d in _parts:
            _ids = set(_d.session.unique())
            ov = sorted(_seen & _ids)
            if ov:
                raise SystemExit(f"{_n}: session present in MORE THAN ONE run: {ov}")
            _seen |= _ids
    t3_all = pd.concat(streams["task3"], ignore_index=True)
    erd_all = pd.concat(streams["erd_roi"], ignore_index=True)

    missing = sorted(set(POST_OPERATIVE) - set(t3_all.session.unique()))
    if missing:
        raise SystemExit(f"post_operative sessions missing from the task3 inputs: {missing}")
    missing_e = sorted(set(POST_OPERATIVE) - set(erd_all.session.unique()))
    if missing_e:
        raise SystemExit(f"post_operative sessions missing from the erd_roi inputs: {missing_e}")
    t3 = t3_all[t3_all.session.isin(POST_OPERATIVE)]
    erd = erd_all[erd_all.session.isin(POST_OPERATIVE)]

    cvals = {t: {k: agg_c(t3, t, k) for k in KS} for t in TASKS}
    evals = {t: {b: agg_erd_db(erd, t, b) for b, _ in BANDS} for t in TASKS}
    from statsmodels.stats.multitest import multipletests
    _keys = [(t, b) for t in TASKS for b, _ in BANDS]
    for (t, b), q in zip(_keys, multipletests([evals[t][b]["p"] for t, b in _keys], method="fdr_bh")[1]):
        evals[t][b]["p_fdr"] = q

    miss = t3[(t3.harmonic_k == 1) & t3.c_abs.isna()]
    missing_nseg_max = int(miss.n_seg.max()) if len(miss) else 0
    erd_not_c, c_not_erd = [], []
    for t in TASKS:
        sc, se = set(cvals[t][1]["sess"]), set(evals[t]["mu"]["sess"])
        if se - sc:
            erd_not_c.append(f"{t}: " + ", ".join(sorted(se - sc)))
        if sc - se:
            c_not_erd.append(f"{t}: " + ", ".join(sorted(sc - se)))

    figL, axL = plt.subplots(figsize=(6.0, 4.8))

    gx = np.arange(len(KS), dtype=float)
    w = 0.26
    offs = {"IW": -w, "RW": 0.0, "S2S": +w}
    if PANEL_MODE:
        _rng = np.random.default_rng(0)
        _top = 0.0
        _marks = []
        for t in TASKS:
            for gi, k in enumerate(KS):
                d = cvals[t][k]
                xi = gx[gi] + offs[t]
                _v = t3[(t3.cond == t) & (t3.harmonic_k == k)
                        & t3.c_abs.notna()].c_abs.to_numpy(float)
                axL.plot([xi - 0.11, xi + 0.11], [d["surr"], d["surr"]], color="#8C8C8C",
                         lw=1.6, solid_capstyle="butt", zorder=2)
                axL.scatter(xi + (_rng.random(_v.size) - 0.5) * 0.16, _v, s=14, color=TCOL[t],
                            alpha=0.85, edgecolor="none", zorder=3)
                axL.plot([xi - 0.11, xi + 0.11], [d["c"], d["c"]], color="#222222", lw=1.6,
                         solid_capstyle="butt", zorder=4)
                axL.errorbar(xi, d["c"], yerr=d["sem"], fmt="none", ecolor="#222222",
                             elinewidth=1.2, capsize=2.5, capthick=1.2, zorder=4)
                _gtop = max(d["c"] + (d["sem"] if np.isfinite(d["sem"]) else 0.0),
                            _v.max() if _v.size else 0.0, d["surr"])
                _marks.append((xi, _gtop, d["p"]))
                _top = max(_top, _gtop)
    else:
        for t in TASKS:
            for gi, k in enumerate(KS):
                d = cvals[t][k]
                xi = gx[gi] + offs[t]
                floor = min(d["surr"], d["c"])
                excess = max(d["c"] - d["surr"], 0.0)
                axL.bar(xi, floor, w, color=FLOOR_COL, alpha=0.55, linewidth=0,
                        label="surrogate (chance)" if (t == "IW" and k == 1) else None)
                axL.bar(xi, excess, w, bottom=floor, color=TCOL[t], linewidth=0,
                        label=f"{t} (n = {d['n']})" if k == 1 else None)
                axL.errorbar(xi, d["c"], yerr=d["sem"], fmt="none", ecolor="#333",
                             elinewidth=1.4, capsize=3.0, capthick=1.4)
                sig = d["p"] < 0.05
                axL.text(xi, d["c"] + d["sem"] + 0.012, stars(d["p"]), ha="center", va="bottom",
                         rotation=0 if sig else 90, fontsize=15 if sig else 10,
                         fontweight="bold" if sig else "normal",
                         color=TCOL[t] if sig else "#888")
    axL.set_xticks(gx)
    axL.set_xticklabels([KLAB[k] for k in KS])
    axL.set_ylabel("Gait-phase locking  $|c|$", fontsize=16, labelpad=6)
    axL.set_ylim(*((-0.02 * _top, _top * 1.16) if PANEL_MODE else (0, 0.92)))
    if PANEL_MODE:
        _STAR_TXT = []
        for _xi, _gtop, _pv in _marks:
            _mk = PS.stars(_pv)
            _sig = _mk != "n.s."
            axL.text(_xi, _gtop + 0.022 * _top, _mk, ha="center", va="bottom",
                     fontsize=15 if _sig else 9, fontweight="bold" if _sig else "normal",
                     color=PS.INK if _sig else "#888888", zorder=5)
            _STAR_TXT.append(_mk)
    else:
        axL.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    axL.spines[["top", "right"]].set_visible(False)
    axL.tick_params(axis="y", direction="out", length=5, width=2.2, labelsize=14)
    axL.tick_params(axis="x", length=0, pad=6, labelsize=14)
    if not PANEL_MODE:
        axL.legend(frameon=False, fontsize=11.5, loc="upper left", ncol=2, handlelength=1.1,
                   borderaxespad=0.3, labelspacing=0.35, handletextpad=0.5, columnspacing=1.2)
        axL.text(0.98, 0.845,
                 "at $\\mathrm{k}_1$ and $\\mathrm{k}_2$, only IW\nclears its chance floor",
                 transform=axL.transAxes, ha="right", va="top", fontsize=11.5,
                 color="#444444", style="italic", linespacing=1.35)

    _rc = S.RC
    _lw_bar, _lw_err, _cap, _s_pt, _lw_pt, _lw_zero = 0.9, 1.1, 2.0, 6, 0.5, 1.1
    _pt_mark, _pt_ns, _pt_tick, _pt_title, _tick_len, _tick_w = (S.STAR_PT, S.NOTE_PT, S.TICK_PT, S.BASE_PT,
                                                                 3.0, 1.4)
    with plt.rc_context(_rc):
        figR, axR = S.panel_axes(plt)
        x = np.arange(len(TASKS), dtype=float)
        bw = 0.36
        for j, (band, _lab) in enumerate(BANDS):
            xs = x + (j - 0.5) * bw
            vals = [evals[t][band]["db"] for t in TASKS]
            sems = [evals[t][band]["sem"] for t in TASKS]
            axR.bar(xs, vals, bw, color=[TCOL[t] for t in TASKS],
                    alpha=1.0 if band == "mu" else 0.5, edgecolor=S.INK, linewidth=_lw_bar,
                    hatch=None if band == "mu" else "///")
            axR.errorbar(xs, vals, yerr=sems, fmt="none", ecolor=S.INK, elinewidth=_lw_err,
                         capsize=_cap, capthick=_lw_err, zorder=5)
            pts = {t: np.array([val for _s, val in evals[t][band]["per"]]) for t in TASKS}
            for t, xi in zip(TASKS, xs):
                _off = ((np.argsort(np.argsort(pts[t])) % 3) - 1) * bw * 0.22
                axR.scatter(xi + _off, pts[t], s=_s_pt, facecolor="white", edgecolor=S.INK, linewidths=_lw_pt,
                            zorder=3)
            for t, xi, v, s in zip(TASKS, xs, vals, sems):
                pv = evals[t][band]["p_fdr"]
                if v < 0:
                    axR.text(xi, min(v - s, pts[t].min()) - 0.35, stars(pv), ha="center", va="top",
                             fontsize=_pt_mark if pv < 0.05 else _pt_ns,
                             fontweight="bold", color=S.INK, zorder=6)
                else:
                    axR.text(xi, max(v + s, pts[t].max()) + 0.3, stars(pv), ha="center", va="bottom",
                             fontsize=_pt_mark if pv < 0.05 else _pt_ns,
                             fontweight="bold", color=S.INK, zorder=6)
        axR.axhline(0, color=S.INK, lw=_lw_zero, zorder=1)
        axR.set_xticks(x)
        axR.set_xticklabels(["IW", "AW", "S2S"], fontsize=_pt_tick)
        axR.set_ylabel("Power change vs rest (dB)", fontsize=_pt_title, labelpad=2)
        _allp = np.concatenate([[val for _s, val in evals[t][b]["per"]] for t in TASKS for b, _ in BANDS])
        _lo, _hi = min(-3.6, _allp.min() - 1.5), max(8.6, _allp.max() + 1.6)
        axR.set_ylim(_lo, _hi)
        axR.set_yticks(np.arange(np.ceil(_lo / 2) * 2, _hi, 2))
        axR.spines[["top", "right"]].set_visible(False)
        axR.tick_params(axis="y", direction="out", length=_tick_len, width=_tick_w, labelsize=_pt_tick)
        axR.tick_params(axis="x", length=0, pad=3, labelsize=_pt_tick)

        if not PANEL_MODE:
            figR.savefig(OUT, dpi=S.DPI, facecolor="white")
    plt.close(figR)

    with plt.rc_context(S.RC):
        figLg, axLg = plt.subplots(figsize=(38 * S.MM, 12 * S.MM), dpi=S.DPI)
        axLg.axis("off")
        axLg.legend(handles=[Patch(facecolor="#777777", label=BANDS[0][1]),
                             Patch(facecolor="#777777", alpha=0.5, hatch="///",
                                   label=BANDS[1][1])],
                    frameon=False, fontsize=S.NOTE_PT, loc="center", handlelength=1.2,
                    labelspacing=0.35, handletextpad=0.5)
        if not PANEL_MODE:
            figLg.savefig(OUT_LEGEND, dpi=S.DPI, facecolor="white",
                          bbox_inches="tight")
    plt.close(figLg)
    if not PANEL_MODE:
        print(f"wrote {OUT_LEGEND.name} (band legend strip)")
    if not PANEL_MODE:
        print("wrote", OUT)
    figL.subplots_adjust(left=0.13, right=0.99, top=0.985, bottom=0.155)
    if PANEL_MODE:
        print("panel mode: no current SI panel for the |c| sub-panel; no panel file written")
    else:
        figL.savefig(OUT_PHASELOCK, dpi=300, facecolor="white", bbox_inches="tight")
        print("wrote", OUT_PHASELOCK, "")

    with SRC.open("w", newline="", encoding="utf-8") as f:
        w2 = csv.writer(f)
        w2.writerow(["subpanel", "cohort", "session", "task", "harmonic_or_band", "value",
                     "sem", "surrogate_or_ref", "p_value", "n_sessions"])
        for t in TASKS:
            for k in KS:
                d = cvals[t][k]
                w2.writerow([f"left_phaselock_k{k}", "post_operative", "cohort_mean", t, f"k{k}",
                             f"{d['c']:.6f}", f"{d['sem']:.6f}", f"{d['surr']:.6f}",
                             f"{d['p']:.4f}", d["n"]])
        for t in TASKS:
            for band, _lab in BANDS:
                e = evals[t][band]
                w2.writerow([f"right_erd_{band}_dB", "post_operative", "cohort_mean", t, band,
                             f"{e['db']:.6f}", f"{e['sem']:.6f}", "vs_REST",
                             f"{e['p']:.4f}", e["n"]])
        for t in TASKS:
            for band, _lab in BANDS:
                for s_, val in evals[t][band]["per"]:
                    w2.writerow([f"right_erd_{band}_dB_persession", "post_operative", s_, t, band,
                                 f"{val:.6f}", "", "vs_REST", "", 1])
        post = t3_all[t3_all.session.isin(POST_OPERATIVE) & t3_all.cond.isin(TASKS)
                      & t3_all.c_abs.notna()].sort_values(["cond", "session", "harmonic_k"])
        for _, r in post.iterrows():
            w2.writerow([f"left_phaselock_persession_k{int(r.harmonic_k)}", "post_operative", r.session,
                         r.cond, f"k{int(r.harmonic_k)}", f"{r.c_abs:.6f}", "",
                         f"{r.c_surrogate:.6f}", f"{r.surrogate_p:.4f}", 1])
        pre_iw = t3_all[t3_all.session.isin(PRE_OPERATIVE) & (t3_all.cond == "IW")
                        & t3_all.c_abs.notna()].sort_values(["session", "harmonic_k"])
        for _, r in pre_iw.iterrows():
            w2.writerow([f"extra_pre_operative_persession_k{int(r.harmonic_k)}", "pre_operative", r.session,
                         "IW", f"k{int(r.harmonic_k)}", f"{r.c_abs:.6f}", "",
                         f"{r.c_surrogate:.6f}", f"{r.surrogate_p:.4f}", 1])

    iw_c = {k: cvals["IW"][k] for k in KS}
    with META.open("w", newline="", encoding="utf-8") as f:
        w2 = csv.writer(f); w2.writerow(["key", "value"])
        for k, v in [
            ("prep", PREP),
            ("post_operative_set", "pat01 sess04-09 + pat02 sess03/05/06/08/09 (11 sessions) — THE DRAWN COHORT"),
            ("pre_operative_set", "pat01 sess01/02 + pat02 sess01/02 (4 sessions) — per-session IW |c| appears in the "
                              "Source Data tagged cohort=pre_operative as a descriptive extra ONLY; not drawn, not in the claim"),
            ("excluded_not_plotted", "pat01_sess03, pat02_sess04 and pat02_sess07 (recording hardware failure)"),
            ("cohort_selection", "explicit session list (POST_OPERATIVE / PRE_OPERATIVE in the script), never an id prefix; the "
                                 "script aborts if a listed session is absent from either input CSV"),
            ("left_metric", "adaptive per-cycle gait-phase-binned unit-phasor |c| at k1/k2/k3; per-task "
                            f"phase-shift surrogate null ({NSUR_TAG}x, one-sided p = share of surrogates >= observed); bar = chance "
                            "floor (surrogate mean) + signal excess; error = across-session SEM; p = MEDIAN of the "
                            "per-session surrogate p (a summary, not a group test; the manuscript reports the "
                            "per-session counts in left_sessions_above_surrogate)"),
            ("left_sessions_above_surrogate", "; ".join(
                f"{t} k{k} {int((post[(post.cond == t) & (post.harmonic_k == k)].surrogate_p < 0.05).sum())}/"
                f"{int(((post.cond == t) & (post.harmonic_k == k)).sum())}" for t in TASKS for k in KS)),
            ("left_n_sessions", "; ".join(f"{t} n={cvals[t][1]['n']}" for t in TASKS)),
            ("left_sessions_IW", ", ".join(cvals["IW"][1]["sess"])),
            ("left_sessions_RW", ", ".join(cvals["RW"][1]["sess"])),
            ("left_sessions_S2S", ", ".join(cvals["S2S"][1]["sess"])),
            ("right_metric", "sensorimotor ROI (21 ch C/CP/FC) ERD as dB log-ratio 10*log10(P_task/P_rest) — "
                             "SAME metric and band edges as Fig.3 panel b (Mu 8-13 Hz, Beta 13-30 Hz); "
                             "negative dB = desynchronization (lab convention)"),
            ("right_n_sessions", "; ".join(f"{t} n={evals[t]['mu']['n']}" for t in TASKS)),
            ("right_task_labels", "tick labels IW / AW / S2S; AW = assisted walking, internal task code RW "
                                  "(the patient overt-walking task; not Fig. 4's real walking)"),
            ("right_missing_conditions", f"of the {len(POST_OPERATIVE)} post_operative sessions: RW ERD is undefined for "
                                   + ", ".join(sorted(set(POST_OPERATIVE) - set(evals["RW"]["mu"]["sess"])))
                                   + "; S2S ERD is undefined for "
                                   + ", ".join(sorted(set(POST_OPERATIVE) - set(evals["S2S"]["mu"]["sess"])) or ["none"])
                                   + f". IW is defined in all {len(POST_OPERATIVE)}."
                                   
                                   ),
            ("style", "unified Fig.3/4 style: Arial 13 pt base, axes.linewidth 2.2, tick width 2.2, top/right spines "
                      "off, harmonics as mathtext k1/k2/k3, 300 dpi PNG, no in-figure title, no panel letter"),
            ("input_task3_csv", T3.relative_to(ROOT).as_posix()),
            ("input_erd_csv", ERD.relative_to(ROOT).as_posix()),
            ("output_png", "figures_out/fig6/fig6_movement_free_dissociation.png"),
        ]:
            w2.writerow([k, v])
    print("wrote", SRC)
    print("wrote", META)

    print(f"\npost_operative = {len(POST_OPERATIVE)} sessions")
    print(f"{'task':5s} {'k':3s} {'mean|c|':>9s} {'sem':>8s} {'surrogate':>10s} {'med p':>7s} {'n':>3s} {'n p<.05':>8s}")
    for t in TASKS:
        for k in KS:
            d = cvals[t][k]
            print(f"{t:5s} k{k:<2d} {d['c']:9.6f} {d['sem']:8.6f} {d['surr']:10.6f} "
                  f"{d['p']:7.4f} {d['n']:3d} {d['n_sig']:5d}/{d['n']}")
    print(f"\n{'task':5s} {'band':6s} {'dB':>9s} {'sem':>8s} {'n':>3s} {'n<0':>6s}")
    for t in TASKS:
        for band, _lab in BANDS:
            e = evals[t][band]
            print(f"{t:5s} {band:6s} {e['db']:+9.6f} {e['sem']:8.6f} {e['n']:3d} {e['n_neg']:4d}/{e['n']}")


if __name__ == "__main__":
    main()
