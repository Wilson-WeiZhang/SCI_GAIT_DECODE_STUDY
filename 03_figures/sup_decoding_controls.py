"""Supplementary item

Figure + Table — Decoding controls and chance level

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import analysis_config as ec
from sup_panel_export import PANEL_MODE
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "decoder_output"
SRCDIR = ROOT / "04_source_data" / "supplementary"
PNG = ROOT / "figures_out" / "supplementary" / "sup_decoding_controls.png"
XLSX = ROOT / "figures_out" / "supplementary" / "tables" / "decoding_controls_table.xlsx"


FINAL_TITLE = (
    "Decoder control analyses: real EEG versus matched nulls. "
    "Healthy cohort: {n_sessions} sessions, {n_subjects} participants. A session value is "
    "the mean of its finite fold \u00d7 joint cells (10 folds \u00d7 6 joints); the cohort "
    "value is the session-equal mean. Real versus null compared with two-sided Wilcoxon "
    "signed-rank tests across sessions."
)
FINAL_DROP_CONDITION_COLUMNS = ("Era / tag", "Status")
FINAL_DROP_CONDITION_CONTROLS = ("Zero-EEG cyclic template",)
FINAL_CONDITION_DEF = {
    "Real EEG": "Real imagined-walking EEG: the reported decoder.",
}
FINAL_PER_SESSION_HEADERS = {
    "session": "Session", "subject": "Subject",
    "label_shuffle sd": "Label-shuffle SD",
    "label_shuffle p95": "Label-shuffle 95th percentile",
    "label_shuffle p (empirical)": "Label-shuffle p (empirical)",
}
FINAL_P_COLUMN = "Label-shuffle p (empirical)"
FINAL_P_DECIMALS = 3
FINAL_DEFINITIONS_HEADER = ["Item", "Description"]

BUILD_ID_TAG = ec.TAG
COL = "rpt_calib"
EXPECT_SESSIONS = 80
EXPECT_SUBJECTS = 42
ANCHOR, ANCHOR_TOL, _ANCHOR_PINNED = ec.anchor("iw_session_mean")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_palette as P
import panel_style as PS


def one(key: str) -> Path:
    """The build_id's registry input (03_figures/analysis_config.py key); it must exist and carry the build_id tag."""
    p = ec.path(key)
    if not p.exists():
        raise SystemExit(f"INPUT GUARD: missing {ec.rel(key)} (build_id {ec.BUILD_ID}, key {key!r})")
    if BUILD_ID_TAG not in p.name and BUILD_ID_TAG not in p.parent.name:
        raise SystemExit(f"INPUT CHECK: {p.name} does not carry the build_id tag {BUILD_ID_TAG!r}.")
    ec.check_tag(ec.rel(key), "decoder", key)
    return p


def maybe_one(key: str) -> Path | None:
    p = ec.path(key)
    if not p.exists():
        return None
    if BUILD_ID_TAG not in p.name and BUILD_ID_TAG not in p.parent.name:
        raise SystemExit(f"INPUT CHECK: {p.name} does not carry the build_id tag {BUILD_ID_TAG!r}.")
    ec.check_tag(ec.rel(key), "decoder", key)
    return p


def fold_level(path: Path, label: str) -> dict[str, float]:
    """Session means from a fold-level merged CSV, with the shape guards."""
    by: dict[str, list[float]] = defaultdict(list)
    folds: dict[str, set] = defaultdict(set)
    n_rows = 0
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            folds[r["session"]].add(r["fold"])
            v = r.get(COL, "")
            if v in ("", "nan", "NaN"):
                continue
            fv = float(v)
            if np.isfinite(fv):
                by[r["session"]].append(fv)
    if len(folds) != EXPECT_SESSIONS:
        raise SystemExit(f"COHORT GUARD [{label}]: expected {EXPECT_SESSIONS} sessions in "
                         f"{path.name}, got {len(folds)}.")
    if not ec.IS_DEFAULT:
        seen, dup = set(), []
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                k = (r["session"], r["fold"], r.get("joint", ""))
                if k in seen:
                    dup.append(k)
                seen.add(k)
        bad = {s: sorted(v) for s, v in folds.items()
               if any(float(x) != int(float(x)) or int(float(x)) < 1 for x in v)}
        if bad or dup:
            raise SystemExit(f"FOLD GUARD [{label}]: non-integer fold ids in {list(bad)[:5]} or doubled "
                             f"cells {dup[:3]} in {path.name}")
    bad = {} if not ec.IS_DEFAULT else {s: sorted(v) for s, v in folds.items() if len(v) != 10}
    if bad:
        raise SystemExit(f"FOLD GUARD [{label}]: sessions without 10 folds in {path.name}: "
                         f"{list(bad)[:5]} ...")
    empty = [s for s in folds if not by.get(s)]
    if empty:
        raise SystemExit(f"FINITE GUARD [{label}]: sessions with no finite {COL}: {empty}")
    means = {s: st.mean(v) for s, v in by.items()}
    n_finite = sum(len(v) for v in by.values())
    print(f"  {label:<16} {path.name}")
    print(f"  {'':<16} sessions {len(means)}  finite cells {n_finite}/{n_rows}  "
          f"cohort mean {st.mean(means.values()):.6f}")
    return means


CONDITION_DEF = {
    "real": "Real imagined-walking EEG - the reported decoder (mainline).",
    "rest_input": ("Resting-window EEG at test: test-fold trial EEG replaced by a contiguous, "
                   "equally long window from that session's own resting trials; training "
                   "folds, goniometer targets and fold assignment unchanged."),
    "label_shuffle": ("Target-label shuffle: post-fit permutation of the held-out target "
                      "trajectories across trials, no refit; 200 permutations per fold."),
    "cadence_shuffle": ("Cadence-shuffle null: within-trial circular shift of the EEG relative "
                        "to its gait target (both spectra preserved), decoder refit."),
}
CONDITION_LABEL = {"real": "Real EEG", "rest_input": "Rest-input", "label_shuffle": "Label-shuffle",
             "cadence_shuffle": "Cadence-shuffle"}
CONDITION_WRAP = {"real": "Real\nEEG", "rest_input": "Rest\ninput",
            "label_shuffle": "Label\nshuffle", "cadence_shuffle": "Cadence\nshuffle"}
CONDITION_FACE = {"real": P.IW_LIGHT, "rest_input": P.REST_LIGHT, "label_shuffle": P.NULL_FILL,
            "cadence_shuffle": P.NULL_FILL}
CONDITION_EDGE = {"real": P.IW_HUE, "rest_input": P.REST_HUE, "label_shuffle": P.MID,
            "cadence_shuffle": P.INK}


def collect():
    print("inputs")
    real_f = one("h_iw_summary_csv")
    rest_f = one("h_rest_summary_csv")
    real = fold_level(real_f, "real")
    rest = fold_level(rest_f, "rest_input")

    cohort_real = st.mean(real.values())
    subj = sorted({s.split("_sess")[0] for s in real})
    if len(subj) != EXPECT_SUBJECTS:
        raise SystemExit(f"COHORT GUARD: expected {EXPECT_SUBJECTS} subjects, got {len(subj)}.")
    if abs(cohort_real - ANCHOR) > ANCHOR_TOL:
        raise SystemExit(f"ANCHOR GUARD: mainline cohort mean {cohort_real:.6f} != {ANCHOR} "
                         f"(tol {ANCHOR_TOL}). Refusing to write a wrong-build_id Source Data.")

    ls_dir = ec.path("h_labelshuf_dir")
    if not ls_dir.is_dir():
        raise SystemExit(f"INPUT GUARD: missing {ls_dir}")
    ls_meta = json.loads((ls_dir / "metadata.json").read_text())
    ls = {}
    with (ls_dir / "per_session.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            ls[r["session"]] = dict(mean=float(r["rpt_calib_shuf_mean"]),
                                    sd=float(r["rpt_calib_shuf_sd"]),
                                    p95=float(r["rpt_calib_shuf_p95"]),
                                    p_emp=float(r["p_empirical"]),
                                    true=float(r["rpt_calib_true"]))
    if len(ls) != EXPECT_SESSIONS:
        raise SystemExit(f"COHORT GUARD [label_shuffle]: {len(ls)} sessions, expected "
                         f"{EXPECT_SESSIONS}.")
    dev = {s: abs(ls[s]["true"] - real[s]) for s in real}
    max_dev_sess = max(dev, key=dev.get)
    max_dev = dev[max_dev_sess]
    if max_dev > 1e-5:
        raise SystemExit(f"SELF-CHECK GUARD: label-shuffle rpt_calib_true deviates from the "
                         f"mainline session mean by {max_dev:.3e} at {max_dev_sess}.")

    conditions = {"real": real, "rest_input": rest,
            "label_shuffle": {s: ls[s]["mean"] for s in ls}}
    sources = {"real": real_f.name, "rest_input": rest_f.name,
               "label_shuffle": f"{ls_dir.name}/per_session.csv"}
    pending = []

    for key, pat in (("cadence_shuffle", "h_cadshuf_summary_csv"),):
        f = maybe_one(pat)
        if f is None:
            pending.append(key)
            print(f"  {key:<16} NOT PRESENT -> condition skipped (metadata records 'pending'). "
                  f"Re-run this script once {ec.rel(pat)} lands.")
            continue
        conditions[key] = fold_level(f, key)
        sources[key] = f.name

    return conditions, sources, ls, ls_meta, pending, max_dev, max_dev_sess, cohort_real


def condition_stats(real: dict, null: dict):
    sess = sorted(set(real) & set(null))
    a = np.array([real[s] for s in sess])
    b = np.array([null[s] for s in sess])
    w, p = stats.wilcoxon(a, b, alternative="two-sided")
    return dict(n=len(sess), mean_real=float(a.mean()), mean_null=float(b.mean()),
                mean_diff=float((a - b).mean()), k_gt=int((a > b).sum()), W=float(w), P=float(p))


def pfmt(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def ptxt(p: float) -> str:
    """In-figure / log form: "p < 0.001" or "p = 0.123" (pfmt gives the bare table-cell form)."""
    v = pfmt(p)
    return f"p {v}" if v.startswith("<") else f"p = {v}"


def render(conditions, ls, order, stats_by_condition):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PS.apply(plt)
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic"})
    if PANEL_MODE:
        PS.apply(plt)
    MM = PS.MM
    INK, MID, PALE = PS.INK, PS.MID, PS.PALE
    nulls = [k for k in order if k != "real"]
    sess = sorted(conditions["real"])
    rng = np.random.default_rng(0)

    W_MM, H_MM = 183.0, 66.0
    BOX_H, BOT, TOP = 44.0, 12.0, 8.0
    LEFTS = (12.0, 12.0, 12.0)
    WIDTHS = (47.0, 45.0, 49.0)
    X0 = (0.0, 61.0, 121.0)
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM), dpi=PS.DPI)

    _AXES, _PFIGS = [], []
    if PANEL_MODE:
        import si_panel_style as SI
        SI.apply(plt)
        plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                             "mathtext.it": "Arial:italic"})
        LW_BOX, LW_MED, S_PT, LW_CONN, LW_ZERO, LW_LINE, LW_MEAN = 0.5, 0.6, 1.3, 0.25, SI.LW_REF, SI.LW_MAIN, SI.LW_REF
        TICK_FS, LAB_FS, STAR_FS, BR_LW = SI.TICK_PT, SI.TITLE_PT, SI.TITLE_PT, SI.LW_REF
        FINISH = SI.finish
        P_GUT = ((8.4, 6.6, 1.8), (9.5, 6.6, 1.5), (9.5, 10.6, 1.5))
        P_W_MM = (37.75, 35.15, 36.45)
    else:
        LW_BOX, LW_MED, S_PT, LW_CONN, LW_ZERO, LW_LINE, LW_MEAN = 1.2, 1.3, 4.5, .25, .8, 1.2, 1.0
        TICK_FS, LAB_FS, STAR_FS, BR_LW = PS.NOTE_PT, PS.BASE_PT, PS.STAR_PT, 1.1
        FINISH = PS.finish

    def box(i):
        if PANEL_MODE:
            f, ax = SI.panel(plt, P_W_MM[i], 38.0, left_mm=P_GUT[i][0], bottom_mm=P_GUT[i][1], top_mm=1.0,
                             right_mm=P_GUT[i][2])
            _PFIGS.append(f)
            _AXES.append(ax)
            return ax
        ax = fig.add_axes([(X0[i] + LEFTS[i]) / W_MM, BOT / H_MM,
                           WIDTHS[i] / W_MM, BOX_H / H_MM])
        _AXES.append(ax)
        return ax

    def letter(i, s):
        if PANEL_MODE:
            return
        fig.text((X0[i] + 1.0) / W_MM, (H_MM - 2.0) / H_MM, s, ha="left", va="top",
                 fontsize=PS.STAR_PT, fontweight="bold", color=INK)

    def boxplot(ax, x, vals, face, edge, bw=0.46):
        q1, md, q3 = np.percentile(vals, [25, 50, 75])
        w10, w90 = np.percentile(vals, [10, 90])
        ax.plot([x, x], [w10, q1], color=edge, lw=LW_BOX, zorder=3)
        ax.plot([x, x], [q3, w90], color=edge, lw=LW_BOX, zorder=3)
        for yy in (w10, w90):
            ax.plot([x - .12, x + .12], [yy, yy], color=edge, lw=LW_BOX,
                    solid_capstyle="butt", zorder=3)
        ax.add_patch(plt.Rectangle((x - bw / 2, q1), bw, q3 - q1, facecolor=face,
                                   edgecolor=edge, lw=LW_MED, zorder=4))
        ax.plot([x - bw / 2, x + bw / 2], [md, md], color=edge, lw=LW_MED,
                solid_capstyle="butt", zorder=5)

    ax = box(0)
    letter(0, "a")
    cols = ["real"] + nulls
    xs = {k: i for i, k in enumerate(cols)}
    jit = {k: xs[k] + rng.uniform(-.06, .06, len(sess)) for k in cols}
    for k in nulls:
        for i, s in enumerate(sess):
            ax.plot([jit["real"][i], jit[k][i]], [conditions["real"][s], conditions[k][s]],
                    color=PALE, lw=LW_CONN, alpha=.35, zorder=1)
    for k in cols:
        v = np.array([conditions[k][s] for s in sess])
        ax.scatter(jit[k], v, s=S_PT, facecolor=CONDITION_EDGE[k], alpha=.75, lw=0, zorder=2)
        boxplot(ax, xs[k], v, CONDITION_FACE[k], CONDITION_EDGE[k])
    ax.axhline(0, color=MID, lw=LW_ZERO, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(*((-.42, len(cols) - .58) if PANEL_MODE else (-.6, len(cols) - .4)))
    ax.set_xticks(list(range(len(cols))))
    ax.set_xticklabels([CONDITION_WRAP[k] for k in cols], fontsize=TICK_FS)
    ax.set_ylabel(r"Session-mean decoding $\mathit{r}$", fontsize=LAB_FS, labelpad=2)
    y_top = max(max(conditions[k][s] for s in sess) for k in cols)
    PS.sig_brackets(ax, [(xs["real"], xs[k], stats_by_condition[k]["P"]) for k in nulls],
                    y0=y_top + .035 * (ax.get_ylim()[1] - ax.get_ylim()[0]), star_pt=STAR_FS, lw=BR_LW)
    FINISH(ax)

    ax = box(1)
    letter(1, "b")
    for i, k in enumerate(nulls):
        d = np.array([conditions["real"][s] - conditions[k][s] for s in sess])
        ax.scatter(i + rng.uniform(-.09, .09, len(d)), d, s=S_PT, facecolor=CONDITION_EDGE[k],
                   alpha=.75, lw=0, zorder=2)
        boxplot(ax, i, d, CONDITION_FACE[k], CONDITION_EDGE[k], bw=0.40)
    dmin = min(min(conditions["real"][s] - conditions[k][s] for s in sess) for k in nulls)
    if ec.IS_DEFAULT or dmin >= 0:
        assert dmin >= 0, "panel b pins the y axis at 0; a negative difference would be clipped"
    else:
        ax.axhline(0, color=INK, lw=0.6, zorder=1)
    ax.set_xlim(-.6, len(nulls) - .4)
    ax.set_xticks(list(range(len(nulls))))
    ax.set_xticklabels([CONDITION_WRAP[k] for k in nulls], fontsize=TICK_FS)
    ax.set_ylabel(r"Real $-$ null ($\Delta\mathit{r}$)", fontsize=LAB_FS, labelpad=2)
    hi = ax.get_ylim()[1]
    diffs = {k: np.array([conditions["real"][s] - conditions[k][s] for s in sess]) for k in nulls}
    top = max(d.max() for d in diffs.values())
    for i, k in enumerate(nulls):
        mk = PS.stars(stats_by_condition[k]["P"])
        ax.text(i, top + .04 * hi, mk, ha="center", va="bottom",
                fontsize=STAR_FS if mk != "n.s." else STAR_FS * .65, color=INK, zorder=6)
    ax.set_ylim(0 if dmin >= 0 else dmin - .06 * hi,
                max(hi, top + .04 * hi + PS._pt_in_data(ax, STAR_FS) + .02 * hi))
    FINISH(ax)

    ax = box(2)
    letter(2, "c")
    ordr = sorted(sess, key=lambda s: ls[s]["mean"])
    x = np.arange(len(ordr))
    m = np.array([ls[s]["mean"] for s in ordr])
    sd = np.array([ls[s]["sd"] for s in ordr])
    ax.fill_between(x, m - sd, m + sd, color=P.NULL_FILL, lw=0, zorder=1)
    ax.plot(x, m, color=P.MID, lw=LW_LINE, zorder=3)
    ax.scatter(x, [conditions["real"][s] for s in ordr], s=S_PT, facecolor=P.IW_HUE, alpha=.8,
               lw=0, zorder=4)
    cm_real = float(np.mean([conditions["real"][s] for s in ordr]))
    cm_null = float(m.mean())
    for y, c, lab in ((cm_real, P.IW_HUE, f"real {cm_real:.3f}"),
                      (cm_null, P.MID, f"shuffle floor {cm_null:.3f}")):
        ax.axhline(y, color=c, lw=LW_MEAN, ls=(0, (3, 2)), zorder=2)
        if not PANEL_MODE:
            ax.text(len(ordr) - 1, y, " " + lab, ha="right", va="bottom",
                    fontsize=PS.NOTE_PT, color=c)
    ax.set_xlim(-1, len(ordr))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Session (ranked by\nshuffle mean)" if PANEL_MODE else "Session (ranked by shuffle mean)",
                  fontsize=LAB_FS, labelpad=2)
    ax.set_ylabel(r"Session-mean decoding $\mathit{r}$", fontsize=LAB_FS, labelpad=2)
    FINISH(ax)

    if PANEL_MODE:
        from sup_panel_export import PANEL_DIR
        for _f, _l in zip(_PFIGS, "abc"):
            SI.save(_f, PANEL_DIR / f"S1{_l}_decoding_controls.png")
            plt.close(_f)
        print("--- statistics text for S1a/b/c (copy into the layout) ---")
        for k, st in stats_by_condition.items():
            print(f"  {k}: {st}")
        print(f"  panel c: real mean {cm_real:.3f}, shuffle floor {cm_null:.3f}")
    else:
        png = ec.scoped(PNG)
        fig.savefig(png, dpi=PS.DPI, facecolor="white")
        print(f"wrote {png}")
    plt.close(fig)


def write_xlsx(condition_rows, per_session_rows, per_session_hdr, definitions):
    """Writes the delivered workbook: the same values as the Source Data."""
    import openpyxl
    from openpyxl.styles import Alignment, Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Controls"
    title = FINAL_TITLE.format(n_sessions=EXPECT_SESSIONS, n_subjects=EXPECT_SUBJECTS)
    hdr = ["Control", "Definition", "n sessions", "Cohort mean (real)", "Cohort mean (null)",
           "Mean paired difference", "Sessions real > null", "Wilcoxon W", "p",
           "Era / tag", "Status"]
    hdr = [h for h in hdr if h not in FINAL_DROP_CONDITION_COLUMNS]
    ws.append([title])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(hdr))
    ws.cell(1, 1).font = Font(italic=True, size=9)
    ws.cell(1, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[1].height = 58
    ws.append(hdr)
    for c in range(1, len(hdr) + 1):
        ws.cell(2, c).font = Font(bold=True)
    printed = [r for r in condition_rows if r["Control"] not in FINAL_DROP_CONDITION_CONTROLS]
    assert len(printed) == len(condition_rows) - len(FINAL_DROP_CONDITION_CONTROLS)
    for r in printed:
        ws.append([FINAL_CONDITION_DEF.get(r["Control"], r[h]) if h == "Definition" else r[h]
                   for h in hdr])
    for col, w in zip("ABCDEFGHI", (16, 62, 11, 18, 18, 21, 19, 12, 13)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=3, min_col=2, max_col=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")

    ws2 = wb.create_sheet("per_session")
    printed_hdr = [FINAL_PER_SESSION_HEADERS.get(h, h) for h in per_session_hdr]
    p_col = printed_hdr.index(FINAL_P_COLUMN)
    ws2.append(printed_hdr)
    for c in range(1, len(printed_hdr) + 1):
        ws2.cell(1, c).font = Font(bold=True)
    for r in per_session_rows:
        r = list(r)
        r[p_col] = round(r[p_col], FINAL_P_DECIMALS)
        ws2.append(r)

    ws3 = wb.create_sheet("Definitions")
    ws3.append(FINAL_DEFINITIONS_HEADER)
    ws3.cell(1, 1).font = Font(bold=True)
    ws3.cell(1, 2).font = Font(bold=True)
    for k, v in definitions:
        ws3.append([k, v])
    ws3.column_dimensions["A"].width = 30
    ws3.column_dimensions["B"].width = 120
    wb.save(ec.scoped(XLSX))
    print(f"wrote {ec.scoped(XLSX)}")


def main() -> None:
    (conditions, sources, ls, ls_meta, pending, max_dev, max_dev_sess,
     cohort_real) = collect()

    order = [k for k in ("real", "rest_input", "label_shuffle", "cadence_shuffle")
             if k in conditions]
    nulls = [k for k in order if k != "real"]
    sess = sorted(conditions["real"])
    subj = sorted({s.split("_sess")[0] for s in sess})

    stats_by_condition = {k: condition_stats(conditions["real"], conditions[k]) for k in nulls}

    fig_rows = []
    for s in sess:
        for k in order:
            row = dict(session=s, subject=s.split("_sess")[0], condition=k,
                       condition_label=CONDITION_LABEL[k], value=f"{conditions[k][s]:.6f}",
                       shuffle_sd="", shuffle_p95="", p_empirical="")
            if k == "label_shuffle":
                row["shuffle_sd"] = f"{ls[s]['sd']:.6f}"
                row["shuffle_p95"] = f"{ls[s]['p95']:.6f}"
                row["p_empirical"] = f"{ls[s]['p_emp']:.6f}"
            fig_rows.append(row)
    f_fields = ["session", "subject", "condition", "condition_label", "value", "shuffle_sd",
                "shuffle_p95", "p_empirical"]
    with ec.sd("sup_decoding_controls_source_data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=f_fields)
        w.writeheader()
        w.writerows(fig_rows)

    condition_rows = [{
        "Control": CONDITION_LABEL["real"], "Definition": CONDITION_DEF["real"], "n sessions": len(sess),
        "Cohort mean (real)": round(cohort_real, 6), "Cohort mean (null)": "",
        "Mean paired difference": "", "Sessions real > null": "", "Wilcoxon W": "", "p": "",
        "Era / tag": f"{BUILD_ID_TAG} / {sources['real']}", "Status": "reported decoder",
    }]
    for k in nulls:
        s_ = stats_by_condition[k]
        condition_rows.append({
            "Control": CONDITION_LABEL[k], "Definition": CONDITION_DEF[k], "n sessions": s_["n"],
            "Cohort mean (real)": round(s_["mean_real"], 6),
            "Cohort mean (null)": round(s_["mean_null"], 6),
            "Mean paired difference": round(s_["mean_diff"], 6),
            "Sessions real > null": f"{s_['k_gt']}/{s_['n']}",
            "Wilcoxon W": round(s_["W"], 1), "p": pfmt(s_["P"]),
            "Era / tag": f"{BUILD_ID_TAG} / {sources[k]}", "Status": "run",
        })
    for k in pending:
        condition_rows.append({
            "Control": CONDITION_LABEL[k], "Definition": CONDITION_DEF[k], "n sessions": "",
            "Cohort mean (real)": "", "Cohort mean (null)": "",
            "Mean paired difference": "", "Sessions real > null": "", "Wilcoxon W": "",
            "p": "", "Era / tag": "n/a",
            "Status": "not available",
        })
    condition_rows.append({
        "Control": "Zero-EEG cyclic template", "Definition": CONDITION_DEF.get(
            "zero_eeg", "Zero-EEG cyclic template floor: r(real EEG) vs r(template)."),
        "n sessions": "", "Cohort mean (real)": "", "Cohort mean (null)": "",
        "Mean paired difference": "", "Sessions real > null": "", "Wilcoxon W": "", "p": "n/a",
        "Era / tag": "n/a",
        "Status": "not included: no healthy cohort product (patient cohort only)",
    })

    t_fields = list(condition_rows[0].keys())
    with ec.sd("sup_decoding_controls_table_source_data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=t_fields)
        w.writeheader()
        w.writerows(condition_rows)

    ps_hdr = (["session", "subject"] + [f"{CONDITION_LABEL[k]} r" for k in order]
              + ["label_shuffle sd", "label_shuffle p95", "label_shuffle p (empirical)"])
    ps_rows = [[s, s.split("_sess")[0]] + [round(conditions[k][s], 6) for k in order]
               + [round(ls[s]["sd"], 6), round(ls[s]["p95"], 6), round(ls[s]["p_emp"], 6)]
               for s in sess]

    floor_note = (
        "The target-label-shuffle null does NOT sit at r = 0: its cohort floor is "
        f"{np.mean([ls[s]['mean'] for s in sess]):.4f}. Gait targets are quasi-periodic and every "
        "session's trials share a cadence spectrum, so a prediction permuted onto a different "
        "trial still correlates with that trial's waveform. The primary control is the cadence-shuffle "
        "(within-trial circular shift, decoder refit), which breaks the EEG-to-gait alignment "
        "while preserving both spectra. Read the label-shuffle control as the empirical chance "
        "level of this metric, not as zero.")

    meta = [
        ("figure", "Supplementary Fig. 1a-c (Decoding controls and chance level)"),
        ("note", "Supplementary Note 1"),
        ("build_id_evidence", "derived from the decode result tag in every input filename; every glob "
                         f"must resolve to exactly one file carrying '{BUILD_ID_TAG}' or the script exits"),
        ("n_sessions", str(len(sess))),
        ("n_subjects", str(len(subj))),
        ("scope", "full-cohort healthy (session-level, session-equal weighting)"),
        ("metric", "rpt_calib (calibrated per-trial reconstruction r); session = mean of the "
                   "finite fold x joint cells (10 folds x 6 joints)"),
        ("test", "two-sided Wilcoxon signed-rank across sessions, real vs each null (paired)"),
        ("labelshuffle_selfcheck", f"max |rpt_calib_true - mainline session mean| = "
                                   f"{max_dev:.3e} at {max_dev_sess} (guard 1e-5)"),
        ("label_shuffle_n_perm", str(ls_meta.get("n_perm"))),
        ("label_shuffle_seed", str(ls_meta.get("seed"))),
        ("condition_zero_eeg", "no healthy cohort product (patient condition only)"),
    ]
    for k in order:
        meta.append((f"condition_{k}", f"RUN | {CONDITION_DEF[k]} | source {sources[k]}"))
    for k in pending:
        meta.append(("condition_cadshuf", "pending"))
    meta += [
        ("outputs", "figures_out/supplementary/sup_decoding_controls.png (default mode) or "
                    "figures_out/supplementary/_panels/S1{a,b,c}_decoding_controls.png (SI_PANEL_SPEC=panel); "
                    "04_source_data/sup_decoding_controls_{source_data,metadata}.csv; "
                    "04_source_data/sup_decoding_controls_table_{source_data,metadata}.csv"),
    ]
    for path in (ec.sd("sup_decoding_controls_metadata.csv"),
                 ec.sd("sup_decoding_controls_table_metadata.csv")):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["key", "value"])
            w.writerows(meta)

    definitions = [
        ("Cohort", f"healthy analysis cohort, {len(sess)} sessions, {len(subj)} participants"),
        ("Sessions", str(len(sess))),
        ("Participants", str(len(subj))),
        ("Scope", "full-cohort healthy, session level, session-equal weighting"),
        ("Metric", "calibrated per-trial reconstruction r; a session value is the mean of its "
                   "finite fold \u00d7 joint cells (10 folds \u00d7 6 joints)"),
        ("Test", "two-sided Wilcoxon signed-rank across sessions, real versus each null (paired)"),
        ("Label-shuffle permutations", f"{ls_meta.get('n_perm')} per fold"),
        ("Label-shuffle floor",
         "The target-label-shuffle null does not reach r = 0 (cohort floor "
         f"{np.mean([ls[s]['mean'] for s in sess]):.4f}) because the gait targets are "
         "quasi-periodic and a session's trials share a cadence spectrum; the cadence shuffle "
         "is the primary control."),
    ]
    render(conditions, ls, order, stats_by_condition)

    print()
    print(f"mainline self-check : {cohort_real:.6f}      # Result: {ANCHOR}")
    print(f"labelshuf self-check: max |true - mainline| = {max_dev:.3e} at {max_dev_sess}")
    print(f"sessions / subjects : {len(sess)} / {len(subj)}   # Result: 80 / 42")
    for k in nulls:
        s_ = stats_by_condition[k]
        print(f"{CONDITION_LABEL[k]:<16} null {s_['mean_null']:.6f}  diff {s_['mean_diff']:+.6f}  "
              f"real>null {s_['k_gt']}/{s_['n']}  W={s_['W']:.1f}  {ptxt(s_['P'])}")
    for k in pending:
        print(f"{CONDITION_LABEL[k]:<16} pending (merged CSV absent)")
    print(f"wrote {ec.sd('sup_decoding_controls_source_data.csv')}")
    print(f"wrote {ec.sd('sup_decoding_controls_table_source_data.csv')}")


if __name__ == "__main__":
    main()
