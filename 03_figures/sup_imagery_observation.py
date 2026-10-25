# -*- coding: utf-8 -*-
"""Supplementary item

figure: imagery, not observation, carries the decoded gait representation

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import csv
import json
import sys
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel_style as PS
import si_panel_style as SI
import project_palette as PAL
from sup_panel_export import PANEL_DIR, export_legend
import analysis_config as ec

BASE = Path(__file__).resolve().parents[1]
TRIALS = ec.path("s5_trials_all")
LME_JSON = ec.path("s5_lme_json")
CMP_JSON = BASE / "gait_prep_supplementary/result/s5m_compare_all.json"
SD = ec.sd("sup_imagery_observation_source_data.csv")
META = ec.sd("sup_imagery_observation_metadata.csv")

SLUG = "supplementary_imagery_observation"
COL = {"MI+AO": PAL.IW_HUE, "AO": PAL.MID}
LIGHT = {"MI+AO": PAL.IW_LIGHT, "AO": "#D6D6D6"}
CONDS = ("MI+AO", "AO")
MAIN = ec.BUILD_ID == "main"
N_PART = 6
N_CONDITION = 20 if MAIN else 30
N_TRIALS = N_PART * 2 * N_CONDITION
NAME = ({"MI+AO": "Imagined walking", "AO": "Observation only"} if MAIN
        else {"MI+AO": "Observation with imagery", "AO": "Observation alone"})
CORR = "BH-FDR" if MAIN else "Holm"


def short(participant):
    """SUB_09_sess03 -> S09: the label used in the Methods for a supplementary participant."""
    return "S" + participant.split("_")[1]


def load():
    rows = list(csv.DictReader(TRIALS.open(encoding="utf-8")))
    if len(rows) != N_TRIALS:
        raise SystemExit(f"{TRIALS} has {len(rows)} rows, expected {N_TRIALS} "
                         f"({N_PART} x 2 x {N_CONDITION})")
    for r in rows:
        r["rpt_calib"] = float(r["rpt_calib"])
        r["cycle_pair"] = int(r["cycle_pair"])
    parts = sorted({r["participant"] for r in rows}, key=lambda s: int(s.split("_")[1]))
    if len(parts) != N_PART:
        raise SystemExit(f"expected {N_PART} participants, found {parts}")
    bad = [r for r in rows if r.get("goni_qc", "good") != "good"]
    if bad:
        raise SystemExit(f"{len(bad)} trials carry goni_qc != good; the upstream table "
                         "should already have excluded them")
    return rows, parts


def per_participant(rows, parts):
    """Trial values, cycle-pair means and the paired differences, per participant."""
    out = {}
    for p in parts:
        pr = [r for r in rows if r["participant"] == p]
        trials = {c: np.array([r["rpt_calib"] for r in pr if r["cond"] == c]) for c in CONDS}
        for c in CONDS:
            if trials[c].size != N_CONDITION:
                raise SystemExit(f"{p} {c}: {trials[c].size} trials, expected {N_CONDITION}")
        pairs = sorted({r["cycle_pair"] for r in pr})
        cyc = {c: np.array([np.mean([r["rpt_calib"] for r in pr
                                     if r["cond"] == c and r["cycle_pair"] == k])
                            for k in pairs]) for c in CONDS}
        out[p] = {"trials": trials, "pairs": pairs,
                  "cycle": cyc, "diff": cyc["MI+AO"] - cyc["AO"]}
    return out


def holm(pvals):
    """Holm step-down over one family; returns corrected p in the input order."""
    order = np.argsort(pvals)
    m, running, corr = len(pvals), 0.0, np.empty(len(pvals))
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        corr[i] = min(1.0, running)
    return corr


def bh_fdr(pvals):
    """Benjamini-Hochberg step-up over one family; returns adjusted p in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = pvals[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    corr = np.empty(m)
    corr[order] = np.minimum(1.0, adj)
    return corr


correct = bh_fdr if MAIN else holm


def tests(D, parts, lme):
    """The within-participant paired family, the AO-vs-zero family, and the model."""
    raw = [stats.wilcoxon(D[p]["diff"], alternative="two-sided").pvalue for p in parts]
    p_corr = correct(np.array(raw))
    def tci(d):
        h = stats.t.ppf(0.975, d.size - 1) * np.std(d, ddof=1) / np.sqrt(d.size)
        return [float(np.mean(d) - h), float(np.mean(d) + h)]
    paired = [{"participant": p, "n_pairs": len(D[p]["diff"]),
               "mean_diff": float(np.mean(D[p]["diff"])), "ci95_t": tci(D[p]["diff"]),
               "median_diff": float(np.median(D[p]["diff"])),
               "n_positive": int(np.sum(D[p]["diff"] > 0)),
               "p_raw": float(r), "p_corr": float(h)}
              for p, r, h in zip(parts, raw, p_corr)]

    ao_raw = [stats.wilcoxon(D[p]["trials"]["AO"], alternative="two-sided").pvalue
              for p in parts]
    ao_corr = correct(np.array(ao_raw))
    ao = [{"participant": p, "median": float(np.median(D[p]["trials"]["AO"])),
           "n_positive": int(np.sum(D[p]["trials"]["AO"] > 0)),
           "p_raw": float(r), "p_corr": float(h)}
          for p, r, h in zip(parts, ao_raw, ao_corr)]

    md = np.array([t["mean_diff"] for t in paired])
    pre = stats.wilcoxon(md, alternative="greater")
    tt = stats.ttest_1samp(md, 0.0)
    return {"paired": paired, "ao_vs_zero": ao,
            "prereg": {"W": float(pre.statistic), "p_one_sided": float(pre.pvalue),
                       "n_positive": int(np.sum(md > 0)), "n": len(md)},
            "ttest": {"t": float(tt.statistic), "df": len(md) - 1,
                      "p_two_sided": float(tt.pvalue)},
            "model": lme}


def verify_lme(D, parts):
    """Read the model and prove it was fitted on THIS trial table."""
    j = json.loads(LME_JSON.read_text(encoding="utf-8"))["all"]
    m = j["M2_random_slope"]
    if m["n_trials"] != N_TRIALS or m["n_participants"] != N_PART:
        raise SystemExit(f"{LME_JSON}: model covers {m['n_trials']} trials / "
                         f"{m['n_participants']} participants, not {N_TRIALS} / {N_PART}")
    if not m["converged"]:
        raise SystemExit(f"{LME_JSON}: M2 did not converge; do not draw it")
    for p in parts:
        mine = float(np.mean(D[p]["trials"]["MI+AO"]) - np.mean(D[p]["trials"]["AO"]))
        theirs = j["participant_mean_diffs"][p]
        if abs(mine - theirs) > 1e-9:
            raise SystemExit(f"{p}: mean difference {mine:.9f} from {TRIALS.name} but "
                             f"{theirs:.9f} in {LME_JSON.name} — the model was fitted on "
                             "different data")
    return {"beta": m["beta_miao_minus_ao"], "se": m["se"], "z": m["z"], "p": m["p"],
            "ci": m["ci95"], "n_trials": m["n_trials"], "n_participants": m["n_participants"],
            "spec": "r ~ condition + (1 + condition | participant), REML, condition centred"}


def panel_d(D, parts, T):
    """Per-trial r by condition, five participants: raincloud pairs + corrected marks (CORR)."""
    fig, ax = SI.panel(plt, "half" if MAIN else "full", 55.0, left_mm=12.5 if MAIN else 11.0)
    xs = {}
    for i, p in enumerate(parts):
        base = 2.6 * i
        if i % 2 == 0:
            ax.axvspan(base - 0.65, base + 1.70, facecolor="#F4F4F4", edgecolor="none", zorder=0)
        for j, c in enumerate(CONDS):
            x = base + j * 1.05
            xs[(p, c)] = x
            SI.raincloud(ax, x, D[p]["trials"][c], COL[c], light=LIGHT[c], seed=10 * i + j,
                         box_w=0.20, point_spread=0.15, point_s=4.5, width=0.34)
    ax.axhline(0.0, color=PAL.PALE, lw=SI.LW_REF, zorder=1)
    ax.set_xticks([2.6 * i + 0.52 for i in range(len(parts))])
    ax.set_xticklabels([short(p) for p in parts])
    ax.set_ylabel("Decoding accuracy, $r$")
    ax.set_xlim(-0.85, 2.6 * (len(parts) - 1) + 1.90)
    SI.finish(ax)
    ink = np.concatenate([D[p]["trials"][c] for p in parts for c in CONDS])
    lo, hi = float(ink.min()), float(ink.max())
    rng = hi - lo
    ax.set_ylim(lo - 0.08 * rng, hi + 0.28 * rng)
    for t in T["paired"]:
        PS.sig_bracket(ax, xs[(t["participant"], "MI+AO")], xs[(t["participant"], "AO")],
                       t["p_corr"], hi + 0.06 * rng, tick=0.030 * rng, lw=SI.LW_REF,
                       star_pt=SI.TITLE_PT, ns_pt=SI.TICK_PT)
    return fig, ax


def panel_e_forest(D, parts, T):
    """sc: forest in the Fig. 3c grammar (03_figures/fig3_d_dissociation_forest.py). One row per participant = mean
    cycle-pair difference with its t 95% CI; filled ink = BH-FDR-corrected Wilcoxon p < 0.05, open grey = n.s.; the
    star column reads that p. Pooled row = the mixed model M2 (beta, its 95% CI, its p)."""
    fig, ax = SI.panel(plt, "half", 55.0, left_mm=9.0, bottom_mm=11.0, right_mm=7.0)
    m = T["model"]
    rows = [(short(t["participant"]), t["mean_diff"], t["ci95_t"], t["p_corr"]) for t in T["paired"]]
    rows.append(("All", m["beta"], m["ci"], m["p"]))
    n = len(rows)
    ys = [n - i + (0.0 if i < n - 1 else -0.35) for i in range(n)]
    ax.axvline(0.0, color=PAL.PALE, lw=SI.LW_REF * 1.4, ls=(0, (2, 2)), zorder=0)
    for (lab, est, (lo, hi), q), y in zip(rows, ys):
        col = PAL.INK if q < 0.05 else PAL.MID
        ax.plot([lo, hi], [y, y], color=col, lw=SI.LW_MAIN, solid_capstyle="butt", zorder=2)
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.16, y + 0.16], color=col, lw=SI.LW_MAIN, solid_capstyle="butt", zorder=2)
        ax.plot([est], [y], "o", ms=3.6, mfc=col if q < 0.05 else "white", mec=col,
                mew=SI.LW_MAIN * 0.9, zorder=3)
        st = PS.stars(q).replace("n.s.", "ns")
        ax.text(1.03, y, st, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=SI.TITLE_PT if st != "ns" else SI.TICK_PT, color=PAL.INK, clip_on=False)
    ax.axhline(1.25, color=PAL.PALE, lw=SI.LW_MINOR, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 0.7)
    lo = min(r[2][0] for r in rows)
    ax.set_xlim(min(-0.1, lo - 0.05), 0.8)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("$r$ difference\n" + NAME["MI+AO"].lower() + " \u2212 " + NAME["AO"].lower())
    SI.finish(ax)
    return fig, ax


def panel_e(D, parts, T):
    """Per-cycle paired difference per participant, with the across-participant model band."""
    if MAIN:
        return panel_e_forest(D, parts, T)
    fig, ax = SI.panel(plt, "half", 55.0, left_mm=12.5)
    m = T["model"]
    ax.axhspan(m["ci"][0], m["ci"][1], facecolor=PAL.IW_LIGHT, alpha=0.45,
               edgecolor="none", zorder=0)
    ax.axhline(m["beta"], color=PAL.IW_HUE, lw=SI.LW_MAIN, zorder=1)
    ax.axhline(0.0, color=PAL.INK, lw=SI.LW_REF, zorder=2)
    for i, p in enumerate(parts):
        SI.raincloud(ax, i, D[p]["diff"], PAL.INK, light="#DEDEDE", seed=100 + i,
                     box_w=0.18, point_spread=0.14, point_s=5.0, width=0.32, violin=False)
        ax.plot([i + 0.17], [np.mean(D[p]["diff"])], marker="D", ms=2.8,
                mfc=PAL.IW_HUE, mec=PAL.INK, mew=SI.LW_MINOR, zorder=8, clip_on=False)
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels([short(p) for p in parts])
    ax.set_ylabel("$r$ difference, MI+AO " + "−" + " AO")
    ax.set_xlim(-0.65, len(parts) - 0.45)
    SI.finish(ax)
    return fig, ax


def write_source_data(rows, D, parts, T):
    SD.parent.mkdir(parents=True, exist_ok=True)
    head = ["row_type", "panel", "participant", "condition", "cycle_pair", "slot",
            "metric", "value", "n", "p_raw", "p_corr", "ci_lo", "ci_hi", "note"]
    out = []

    def add(**kw):
        out.append([kw.get(k, "") for k in head])

    for r in rows:
        add(row_type="trial", panel="a", participant=r["participant"], condition=r["cond"],
            cycle_pair=r["cycle_pair"], slot=r.get("slot", ""), metric="rpt_calib",
            value=f"{r['rpt_calib']:.6f}")
    for p in parts:
        for k, mi, ao, d in zip(D[p]["pairs"], D[p]["cycle"]["MI+AO"],
                                D[p]["cycle"]["AO"], D[p]["diff"]):
            add(row_type="cycle_pair", panel="b", participant=p, cycle_pair=k,
                metric="miao_minus_ao", value=f"{d:.6f}",
                note=f"MI+AO {mi:.6f}; AO {ao:.6f}")
    for p in parts:
        for c in CONDS:
            v = D[p]["trials"][c]
            q = np.percentile(v, [10, 25, 50, 75, 90])
            add(row_type="condition_summary", panel="a", participant=p, condition=c,
                metric="trial_r", value=f"{np.median(v):.6f}", n=v.size,
                note=("box 25-75, whiskers 10-90: "
                      + ", ".join(f"{x:.6f}" for x in q)
                      + f"; mean {np.mean(v):.6f}; above zero {int(np.sum(v > 0))}/{v.size}"))
    for t in T["paired"]:
        add(row_type="paired_test", panel="a", participant=t["participant"],
            metric="wilcoxon_two_sided_on_cycle_pair_differences",
            value=f"{t['mean_diff']:.6f}", n=t["n_pairs"], p_raw=f"{t['p_raw']:.6g}",
            p_corr=f"{t['p_corr']:.6g}", ci_lo=f"{t['ci95_t'][0]:.6f}", ci_hi=f"{t['ci95_t'][1]:.6f}",
            note=(f"median {t['median_diff']:.6f}; positive {t['n_positive']}/{t['n_pairs']}; "
                  f"mark {PS.stars(t['p_corr'])}; {CORR} family = the {len(parts)} participants; "
                  "ci = t 95% CI of the mean pair difference (uncorrected)"))
    for a in T["ao_vs_zero"]:
        add(row_type="ao_vs_zero", panel="a", participant=a["participant"], condition="AO",
            metric="wilcoxon_two_sided_vs_zero", value=f"{a['median']:.6f}", n=N_CONDITION,
            p_raw=f"{a['p_raw']:.6g}", p_corr=f"{a['p_corr']:.6g}",
            note=f"trials above zero {a['n_positive']}/{N_CONDITION}; secondary, not drawn")
    m = T["model"]
    add(row_type="across_participants", panel="b", metric="lme_random_slope",
        value=f"{m['beta']:.6f}", n=m["n_trials"], p_raw=f"{m['p']:.6g}",
        ci_lo=f"{m['ci'][0]:.6f}", ci_hi=f"{m['ci'][1]:.6f}",
        note=(m["spec"] + f"; SE {m['se']:.6f}; z {m['z']:.4f}; "
              f"{m['n_participants']} participants; read from {LME_JSON.name}"))
    pr = T["prereg"]
    add(row_type="across_participants", panel="b",
        metric="paired_wilcoxon_over_participants",
        value=f"{np.mean([t['mean_diff'] for t in T['paired']]):.6f}", n=pr["n"],
        p_raw=f"{pr['p_one_sided']:.6g}",
        note=(f"one-sided exact W+ = {pr['W']:.0f}, positive {pr['n_positive']}/{pr['n']}; "
              "secondary across-participant test; "
              "reported, not drawn"))
    tt = T["ttest"]
    add(row_type="across_participants", panel="b",
        metric="one_sample_t_on_participant_mean_differences",
        value=f"{tt['t']:.4f}", n=pr["n"], p_raw=f"{tt['p_two_sided']:.6g}",
        note=f"df {tt['df']}; two-sided; secondary")

    with SD.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(out)
    print("wrote", SD, f"({len(out)} rows)")


def write_metadata(T):
    m = T["model"]
    rows = [
        ("figure", "Supplementary Fig. 11a-b"),
        ("panels", "a per-trial r by condition; b per-cycle paired difference + model"),
        ("note", "Supplementary Note 2 (imagery-observation dissociation)"),
        ("experiment", "supplementary (MI+AO vs AO), the supplementary experiment"),
        ("n", ""),
        ("basis", "leave-subject-out basis"),
        ("readout", "random 10-fold, inner-CV lambda; scored after the calibration cut"),
        ("decoding_target", "the reference walker's six sagittal joints"),
        ("scope", "case level: one session per participant, descriptive"),
        ("test_within", "two-sided Wilcoxon signed-rank on the per-cycle-pair differences, " "corrected over the participants; the drawn star reads the "
                        "corrected p"),
        ("correction_method", f"{CORR} over the participants, in both per-participant families (paired "
                              "MI+AO - AO Wilcoxon; AO against zero); Source Data column p_corr"),
        ("test_across", m["spec"]),
        ("test_across_result", f"beta = {m['beta']:.6f}, 95% CI "
                               f"[{m['ci'][0]:.6f}, {m['ci'][1]:.6f}], "
                               f"z = {m['z']:.4f}, p = {m['p']:.6g}"),
        ("box_convention", "box 25th-75th percentile, whiskers 10th-90th"),
        ("inputs", f"{ec.rel('s5_trials_all')}; "
                   f"{ec.rel('s5_lme_json')}"),
        ("renderer", "03_figures/sup_imagery_observation.py"),
        ("canvas", "16 cm SI contract (03_figures/si_panel_style.py): a full 160 mm, b half 78.5 mm, "
                   "both 55 mm high, 600 dpi, rendered at placement size"),
    ]
    if MAIN:
        upd = {
            "figure": "Supplementary Fig. 11a-b",
            "panels": "a per-trial r by condition; b forest of per-participant mean cycle-pair difference + model "
                      "(Fig. 3c grammar)",
            "build_id_detail": "decoder (causal prep, random 10-fold, per-fold band tables, leave-subject-out basis) "
                          "scored only on the trials after the calibration cut (cycles 11-30); the paper build_id",
            "cohort": "6 healthy cohort participants, one additional session each: SUB_01_sess05, SUB_09_sess03, "
                      "SUB_12_sess03, SUB_23_sess03, SUB_29_sess03, SUB_35_sess03",
            "n": f"{N_TRIALS} imagine trials = {N_PART} participants x ({N_CONDITION} + {N_CONDITION}); "
                 f"{N_PART * N_CONDITION // 2} cycle pairs",
            "basis": " leave-subject-out basis bases/iw_lso/subjNN_sess01_shared_delta_pca_lso.mat",
            "readout": " random 10-fold, inner-CV lambda; scored after the calibration cut",
            "task_names": f"MI+AO printed as '{NAME['MI+AO']}', AO as '{NAME['AO']}'",
            "scope": f"case level: one session per participant, {N_PART} participants, descriptive",
            "test_within": f"two-sided Wilcoxon signed-rank on the {N_CONDITION // 2} per-cycle-pair differences, "
                           f"{CORR}-corrected over the {N_PART} participants; drawn marker and star read the corrected "
                           "p; forest bar = t 95% CI of the mean pair difference (uncorrected)",
            "inputs": f"{ec.rel('s5_trials_all')}; {ec.rel('s5_lme_json')}",
            "canvas": "16 cm SI contract (03_figures/si_panel_style.py): a full 160 mm, b half, both 55 mm high, 600 dpi",
        }
        rows = [(k, upd.get(k, v)) for k, v in rows
                if k not in ("replaces", "test_across_primary_changed")] + \
               [(k, v) for k, v in upd.items() if k not in dict(rows)]
    with META.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)
    print("wrote", META)


def write_sidecars(D, parts, T):
    m = T["model"]
    lines_d = [
        "Per-trial decoding accuracy against the reference walker's six joints; "
        f"{N_CONDITION} {NAME['MI+AO'].lower()} and {N_CONDITION} {NAME['AO'].lower()} trials per participant.",
        "Box = 25th to 75th percentile, whiskers 10th to 90th; points are single trials.",
        f"Marks: two-sided Wilcoxon signed-rank on the {N_CONDITION // 2} per-cycle-pair differences, "
        f"{CORR}-corrected over the {len(parts)} participants.",
    ]
    for t in T["paired"]:
        lines_d.append(
            f"  {short(t['participant'])}: median difference "
            f"{PS.num(t['median_diff'], '+.3f')}, positive in {t['n_positive']}/"
            f"{t['n_pairs']} cycle pairs, {PS.pstr(t['p_corr'])} ({CORR}) "
            f"{PS.stars(t['p_corr'])}")
    lines_d.append(f"{NAME['AO']} against zero (secondary, two-sided Wilcoxon on {N_CONDITION} "
                   f"trials, {CORR} over {len(parts)}):")
    for a in T["ao_vs_zero"]:
        lines_d.append(f"  {short(a['participant'])}: median {PS.num(a['median'], '+.3f')}, "
                       f"{PS.pstr(a['p_corr'])} ({CORR})")

    lines_e = [
        f"Per-cycle paired difference, {NAME['MI+AO'].lower()} minus {NAME['AO'].lower()}; "
        f"{N_CONDITION // 2} cycle pairs per participant, {len(parts) * N_CONDITION // 2} in total.",
        (f"Row = that participant's mean difference with its t 95% CI (uncorrected); filled = {CORR}-corrected "
         "Wilcoxon p < 0.05, open = n.s.; pooled row = the model below." if MAIN
         else "Filled marker = that participant's mean difference."),
        f"{'Pooled row' if MAIN else 'Band and line'}: {m['spec']}.",
        f"  beta = {PS.num(m['beta'], '+.3f')} $r$, 95% CI "
        f"[{PS.num(m['ci'][0], '+.3f')}, {PS.num(m['ci'][1], '+.3f')}], "
        f"{PS.pstr(m['p'])}, {m['n_trials']} trials in {m['n_participants']} participants.",
        f"Participant-level one-sample $t$({T['ttest']['df']}) = {T['ttest']['t']:.2f}, "
        f"{PS.pstr(T['ttest']['p_two_sided'])}.",
        "Across-participant test:  "
        f"exact paired Wilcoxon, one-sided "
        f"{PS.pstr(T['prereg']['p_one_sided'])}, positive in "
        f"{T['prereg']['n_positive']}/{T['prereg']['n']} participants.",
    ]
    if MAIN:
        lines_e = [x for x in lines_e if not x.startswith("Across-participant test")]
        lines_e += [f"  {short(t['participant'])}: mean {PS.num(t['mean_diff'], '+.3f')} "
                    f"[{PS.num(t['ci95_t'][0], '+.3f')}, {PS.num(t['ci95_t'][1], '+.3f')}], "
                    f"{PS.pstr(t['p_corr'])} ({CORR}) {PS.stars(t['p_corr'])}" for t in T["paired"]]
    for letter, lines in (("a", lines_d), ("b", lines_e)):
        p = PANEL_DIR / f"S11{letter}_{SLUG}_stats.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("wrote", p)


def write_legends():
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    if MAIN:
        export_legend([Patch(facecolor=LIGHT["MI+AO"], edgecolor=COL["MI+AO"], lw=0.6),
                       Patch(facecolor=LIGHT["AO"], edgecolor=COL["AO"], lw=0.6)],
                      "S11", "a", SLUG, ncol=2, fontsize=7, figsize=(2.6, 0.4),
                      labels=[NAME["MI+AO"], NAME["AO"]])
        export_legend([Line2D([], [], marker="o", ls="-", color=PAL.INK, mfc=PAL.INK, ms=4, lw=0.8),
                       Line2D([], [], marker="o", ls="-", color=PAL.MID, mfc="white", ms=4, lw=0.8)],
                      "S11", "b", SLUG, ncol=1, fontsize=7, figsize=(2.6, 0.6),
                      labels=["Mean difference, 95% CI; $p_{\\mathrm{FDR}}$ < 0.05", "Not significant"])
        return
    export_legend([Patch(facecolor=LIGHT["MI+AO"], edgecolor=COL["MI+AO"], lw=0.6),
                   Patch(facecolor=LIGHT["AO"], edgecolor=COL["AO"], lw=0.6)],
                  "S11", "a", SLUG, ncol=2, fontsize=7, figsize=(2.6, 0.4),
                  labels=["Observation with imagery", "Observation alone"])
    export_legend([Line2D([], [], marker="o", ls="none", ms=4, mfc=PAL.IW_HUE, mec=PAL.INK,
                          mew=0.4),
                   Patch(facecolor=PAL.IW_LIGHT, edgecolor="none"),
                   Line2D([], [], color=PAL.INK, lw=0.8)],
                  "S11", "b", SLUG, ncol=1, fontsize=7, figsize=(2.6, 0.9),
                  labels=["Participant mean", "Mixed-model 95% CI", "No difference"])


def main():
    rows, parts = load()
    D = per_participant(rows, parts)
    lme = verify_lme(D, parts)
    T = tests(D, parts, lme)

    SI.apply(plt)
    for letter, fn in (("a", panel_d), ("b", panel_e)):
        fig, _ = fn(D, parts, T)
        SI.save(fig, PANEL_DIR / f"S11{letter}_{SLUG}.png")
        plt.close(fig)
    write_legends()
    write_sidecars(D, parts, T)
    write_source_data(rows, D, parts, T)
    write_metadata(T)

    print(f"\nper participant (mean difference, {CORR} p, mark):")
    for t in T["paired"]:
        print(f"  {short(t['participant'])}  {t['mean_diff']:+.4f}  "
              f"p_{CORR} {t['p_corr']:.3g}  {PS.stars(t['p_corr']) or 'n.s.'}")
    m = T["model"]
    print(f"model: beta {m['beta']:+.4f} [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}] p {m['p']:.3g}")
    print(f"across-participant test: one-sided p "
          f"{T['prereg']['p_one_sided']:.4f}, {T['prereg']['n_positive']}/"
          f"{T['prereg']['n']} positive")


if __name__ == "__main__":
    main()
