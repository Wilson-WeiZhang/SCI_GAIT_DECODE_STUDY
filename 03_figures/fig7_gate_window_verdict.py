"""Figure 7 — Intent gate and physics simulation of the decoded gait phase

_gate_window_verdict — Fig. 7 gate panel: what every 1 s decision window was, and what
the gate called it

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sup_panel_export import PANEL_MODE, PANEL_DIR, export_stats
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import Divider, Size

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_style as PS
import project_palette as PAL
import fig7_gate_timeline as GT
from fig7_gate_accuracy_table import read_thresholds, s1_states

ROOT = GT.ROOT
GATE_SET = GT.GATE_SET
GATE_CV, GATE_WIN, WIN, RULE_COL = GT.GATE_CV, GT.GATE_WIN, GT.WIN, GT.RULE_COL
SESS, SCORES, THR, ANCHORS = GT.SESS, GT.SCORES, GT.THR, GT.ANCHORS
HOP_S, WIN_S = GT.HOP_S, GT.WIN_S
EXCERPT_SESSION = GT.EXCERPT_SESSION
EXCERPT_S = GT.EXCERPT_S

OUT = f"{ROOT}/figures_out/fig7/fig7_gate_window_verdict.png"
OUT_LEG = f"{ROOT}/figures_out/fig7/fig7_gate_window_verdict_legend.png"
OUT_PREV = f"{ROOT}/figures_out/fig7/_candidates/f7_gate_window_verdict_preview.png"
SD = f"{ROOT}/04_source_data/fig7/fig7_gate_window_verdict_source_data.csv"
MD = f"{ROOT}/04_source_data/fig7/fig7_gate_window_verdict_metadata.csv"
ACC_SD = f"{ROOT}/04_source_data/fig7/fig7_gate_accuracy_source_data.csv"
import gate_config as GE
OUT, OUT_LEG, OUT_PREV, SD, MD, ACC_SD = (GE.scoped(p) for p in (OUT, OUT_LEG, OUT_PREV, SD, MD, ACC_SD))
TL_SD = GE.scoped(f"{ROOT}/04_source_data/fig7/fig7_gate_timeline_source_data.csv")

C_IW, C_REST, C_EDGE = PAL.IW, PAL.REST, PAL.REST_EDGE
INK, MID, PALE, NULLF = PAL.INK, PAL.MID, PAL.PALE, PAL.NULL_FILL

MM = PS.MM
H_TRUTH, H_SCORE, H_RASTER, H_GATE = 4.0, 24.0, 3.2, 4.0
GAP_MM = 1.8
LABEL_LINESPACING = 0.9
LEFT_MM, RIGHT_MM = 22.0, 4.5
BOTTOM_MM, NOTE_MM, TOP_MM = 10.0, 9.0, 2.0
FS_NOTE, FS_TICK, FS_BASE, FULL_W = PS.NOTE_PT, PS.TICK_PT, PS.BASE_PT, PS.TIERS["full"]
LW_STEP, LW_SYM = 0.7, 0.8
DPI_OUT = PS.DPI
XTICK_STEP_S = 30.0
SQ_W_S, SQ_H_FRAC = 0.80, 0.22


def short(s):
    return GT.short(s)


def build_windows(sess, sc, lab, dur, anchors, th):
    """Every 1 s decision of one session on the CONCATENATED clock, with its verdict."""
    sym, on, off = th[sess]
    evs = sorted([k for k in sc if k[0] == sess], key=lambda k: anchors[k])
    rows, x = [], 0.0
    for k in evs:
        t = np.array([a for a, _ in sc[k]])
        v = np.array([b for _, b in sc[k]])
        is_iw = lab[k]
        st, _tr = s1_states(v, on, off, start_open=(not is_iw))
        for te, val, s1 in zip(t, v, st):
            cls = bool(val >= sym)
            verdict = ("hit" if cls else "miss") if is_iw else \
                      ("false_alarm" if cls else "correct_rest")
            rows.append(dict(session=sess, event=k[1], instructed="IW" if is_iw else "REST",
                             t_start_s=x + te, t_end_s=x + te + HOP_S, score=float(val),
                             theta_sym=sym, delta=on - sym, classified_iw=cls,
                             verdict=verdict, s1_open=bool(s1)))
        x += dur[k]
    return pd.DataFrame(rows), (sym, on, off)


def metrics(d):
    """Agreement block for a set of windows. S0 at theta_sym; S1 open fractions."""
    n_iw = int((d.instructed == "IW").sum())
    n_rest = int((d.instructed == "REST").sum())
    hit = int((d.verdict == "hit").sum())
    miss = int((d.verdict == "miss").sum())
    fa = int((d.verdict == "false_alarm").sum())
    cr = int((d.verdict == "correct_rest").sum())
    assert hit + miss + fa + cr == len(d), "verdict classes do not partition the windows"
    s1_iw = float(d.s1_open[d.instructed == "IW"].mean())
    s1_rest = float(d.s1_open[d.instructed == "REST"].mean())
    return dict(n_windows=len(d), n_iw_windows=n_iw, n_rest_windows=n_rest,
                n_hit=hit, n_miss=miss, n_false_alarm=fa, n_correct_rest=cr,
                s0_accuracy=(hit + cr) / len(d), s0_sensitivity=hit / n_iw,
                s0_specificity=cr / n_rest,
                s1_open_fraction_iw=s1_iw, s1_open_fraction_rest=s1_rest,
                s1_sensitivity=s1_iw, s1_specificity=1.0 - s1_rest)


def build_axes():
    """Four stacked axes sharing one x axis, each pinned to an exact physical height."""
    box_w = FULL_W - LEFT_MM - RIGHT_MM
    box_h = H_TRUTH + H_SCORE + H_RASTER + H_GATE + 3 * GAP_MM
    w_mm = LEFT_MM + box_w + RIGHT_MM
    note_mm = 0.0 if PANEL_MODE else NOTE_MM
    h_mm = BOTTOM_MM + note_mm + box_h + TOP_MM
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM), dpi=DPI_OUT)
    hs = [Size.Fixed(LEFT_MM * MM), Size.Fixed(box_w * MM)]
    vs = [Size.Fixed((BOTTOM_MM + note_mm) * MM)]
    ny = {}
    for name, h in (("gate", H_GATE), ("raster", H_RASTER), ("score", H_SCORE),
                    ("truth", H_TRUTH)):
        if len(vs) > 1:
            vs.append(Size.Fixed(GAP_MM * MM))
        ny[name] = len(vs)
        vs.append(Size.Fixed(h * MM))
    div = Divider(fig, (0, 0, 1, 1), hs, vs, aspect=False)
    axes = {}
    for name in ("gate", "raster", "score", "truth"):
        axes[name] = fig.add_axes(div.get_position(),
                                  axes_locator=div.new_locator(nx=1, ny=ny[name]))
    return fig, axes, w_mm, h_mm


def draw(fig, axes, W, B, sym, on, off, t0, t1, es):
    ax_t, ax_s, ax_r, ax_g = axes["truth"], axes["score"], axes["raster"], axes["gate"]
    seams = [s for s in B["seams"] if t0 < s < t1]

    def strip(ax, hide_all=True):
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, 1)
        for sp in ("top", "right", "left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([])
        if hide_all:
            ax.tick_params(axis="y", length=0, labelsize=FS_NOTE)

    for cat, face, edge in (("REST", C_REST, C_EDGE), ("IW", C_IW, "none")):
        xr = [(max(a, t0), min(b, t1) - max(a, t0))
              for a, b, c in B["blocks"] if c == cat and a < t1 and b > t0]
        if xr:
            ax_t.broken_barh(xr, (0, 1), facecolors=face, edgecolors=edge,
                             linewidth=0.25 if edge != "none" else 0.0, zorder=3)
    strip(ax_t)
    ax_t.set_yticks([0.5])
    ax_t.set_yticklabels(["Instructed\ntask"])
    ax_t.get_yticklabels()[0].set_linespacing(LABEL_LINESPACING)

    lo = min(W.score.min(), off) - 0.25
    hi = max(W.score.max(), on) + 0.25
    ax_s.axhspan(off, on, facecolor=PALE, edgecolor="none", alpha=0.55, zorder=1)
    style = {"hit": dict(facecolor=C_IW, edgecolor="none", hatch=None),
             "miss": dict(facecolor=INK, edgecolor="none", hatch=None),
             "false_alarm": dict(facecolor="white", edgecolor=INK, hatch="////")}
    for _i, r in W.iterrows():
        if r.verdict == "correct_rest":
            continue
        st = style[r.verdict]
        ax_s.add_patch(Rectangle((r.t_start_s, min(r.score, sym)), HOP_S,
                                 abs(r.score - sym), linewidth=0.35, zorder=2, **st))
    ax_s.axhline(sym, color=INK, lw=LW_SYM, ls=(0, (1.6, 1.6)), zorder=4)
    for ev, d in W.groupby("event", sort=False):
        d = d.sort_values("t_start_s")
        xs = list(d.t_start_s) + [d.t_end_s.iloc[-1]]
        ys = list(d.score) + [d.score.iloc[-1]]
        ax_s.plot(xs, ys, drawstyle="steps-post", color=INK, lw=LW_STEP, zorder=5,
                  solid_joinstyle="miter")
    ax_s.set_xlim(t0, t1)
    ax_s.set_ylim(lo, hi)
    for sp in ("top", "right"):
        ax_s.spines[sp].set_visible(False)
    ax_s.set_xticks([])
    ax_s.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(4, prune="both"))
    ax_s.tick_params(axis="y", direction="out", labelsize=FS_TICK)
    ax_s.set_ylabel("Gate score (\u03b4)" if GATE_CV == "within10" else "Gate score (z)",
                    fontsize=FS_BASE, labelpad=1.5)

    for _i, r in W.iterrows():
        c = (r.t_start_s + r.t_end_s) / 2.0
        face, edge = (C_IW, "none") if r.classified_iw else (C_REST, C_EDGE)
        ax_r.add_patch(Rectangle((c - SQ_W_S / 2, 0.5 - SQ_H_FRAC / 2), SQ_W_S, SQ_H_FRAC,
                                 facecolor=face, edgecolor=edge, linewidth=0.3, zorder=3))
        if r.verdict in ("miss", "false_alarm"):
            ax_r.plot([c], [0.5], marker="x", color=INK, ms=3.0, mew=0.75, zorder=4,
                      linestyle="none")
    strip(ax_r)
    ax_r.set_yticks([0.5])
    ax_r.set_yticklabels(["Per-window\ndecision"])
    ax_r.get_yticklabels()[0].set_linespacing(LABEL_LINESPACING)

    ax_g.broken_barh([(t0, t1 - t0)], (0, 1), facecolors=NULLF, edgecolors="none", zorder=2)
    for st, face, edge in ((False, C_REST, C_EDGE), (True, C_IW, "none")):
        xr = [(max(a, t0), min(b, t1) - max(a, t0))
              for a, b, s in B["gate"] if s == st and a < t1 and b > t0]
        if xr:
            ax_g.broken_barh(xr, (0, 1), facecolors=face, edgecolors=edge,
                             linewidth=0.25 if edge != "none" else 0.0, zorder=3)
    ax_g.set_xlim(t0, t1)
    ax_g.set_ylim(0, 1)
    for sp in ("top", "right", "left"):
        ax_g.spines[sp].set_visible(False)
    ax_g.set_xticks(np.arange(t0, t1 + 1e-9, XTICK_STEP_S))
    ax_g.set_xticklabels([f"{v - t0:.0f}" for v in np.arange(t0, t1 + 1e-9, XTICK_STEP_S)])
    ax_g.tick_params(axis="x", direction="out", labelsize=FS_TICK)
    ax_g.tick_params(axis="y", length=0, labelsize=FS_NOTE)
    ax_g.set_yticks([0.5])
    ax_g.set_yticklabels(["Gate\nstate"])
    ax_g.get_yticklabels()[0].set_linespacing(LABEL_LINESPACING)
    ax_g.set_xlabel("Concatenated time over imagined-walking and rest events (s)",
                    fontsize=FS_BASE, labelpad=1.5)

    for ax in (ax_t, ax_s, ax_r, ax_g):
        for s in seams:
            ax.axvline(s, color=PALE, lw=0.35, zorder=6)


def note_block(fig, w_mm, h_mm, m, es, m_all=None):
    """Agreement text block below the axis."""
    l1 = (f"{short(es)} · {m['n_windows']} decision windows "
          f"({m['n_iw_windows']} imagined walking, {m['n_rest_windows']} rest) · "
          f"{m['n_hit']} hit · {m['n_miss']} miss · {m['n_false_alarm']} false alarm · "
          f"{m['n_correct_rest']} correct rest")
    l2 = (f"at $\\theta_{{sym}}$: agreement {m['s0_accuracy']:.3f} · "
          f"sensitivity {m['s0_sensitivity']:.3f} · specificity {m['s0_specificity']:.3f} · "
          f"S1 gate open for {m['s1_open_fraction_iw']:.3f} of imagined walking, "
          f"{m['s1_open_fraction_rest']:.3f} of rest")
    if PANEL_MODE:
        pat = "Patient 1" if es.startswith("pat01") else "Patient 2"
        lines = [
            f"Excerpt ({EXCERPT_S:.0f} s of one {pat} session): $n$ = {m['n_windows']} decision "
            f"windows ({m['n_iw_windows']} imagined walking, {m['n_rest_windows']} rest);",
            f"    {m['n_hit']} hit, {m['n_miss']} miss, {m['n_false_alarm']} false alarm, "
            f"{m['n_correct_rest']} correct rest.",
            f"    At the symmetric threshold: agreement {m['s0_accuracy']:.3f}, sensitivity "
            f"{m['s0_sensitivity']:.3f}, specificity {m['s0_specificity']:.3f}.",
            f"    Gate state (S1): open for {m['s1_open_fraction_iw']:.3f} of imagined walking "
            f"and {m['s1_open_fraction_rest']:.3f} of rest.",
        ]
        if m_all is not None:
            lines += [
                f"Whole session: $n$ = {m_all['n_windows']} decision windows "
                f"({m_all['n_iw_windows']} imagined walking, {m_all['n_rest_windows']} rest);",
                f"    at the symmetric threshold: agreement {m_all['s0_accuracy']:.3f}, "
                f"sensitivity {m_all['s0_sensitivity']:.3f}, specificity "
                f"{m_all['s0_specificity']:.3f};",
                f"    gate state (S1): sensitivity {m_all['s1_sensitivity']:.3f}, specificity "
                f"{m_all['s1_specificity']:.3f}.",
            ]
        lines.append(f"{WIN:.0f} s decision windows, 1 s hop; the excerpt is an illustration of "
                     f"one post-operative session, not the reported statistic.")
        export_stats(lines, "S1", "e", "gate_window_verdict", width=7.6)
        (PANEL_DIR / "S1e_gate_window_verdict_stats.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        print("wrote", PANEL_DIR / "S1e_gate_window_verdict_stats.txt")
        return
    fig.text(LEFT_MM / w_mm, 7.0 / h_mm, l1, ha="left", va="top",
             fontsize=FS_NOTE, color=INK)
    fig.text(LEFT_MM / w_mm, 3.5 / h_mm, l2, ha="left", va="top",
             fontsize=FS_NOTE, color=INK)


def legend_png(delta=None, out=None):
    h = [Patch(facecolor=C_IW, edgecolor="none"),
         Patch(facecolor=C_REST, edgecolor=C_EDGE, linewidth=0.5),
         Line2D([0], [0], color=INK, lw=LW_STEP),
         Line2D([0], [0], color=INK, lw=LW_SYM, ls=(0, (1.6, 1.6))),
         Patch(facecolor=PALE, edgecolor="none", alpha=0.55),
         Patch(facecolor=INK, edgecolor="none"),
         Patch(facecolor="white", edgecolor=INK, hatch="////", linewidth=0.35),
         Line2D([0], [0], marker="x", color=INK, ms=3.0, mew=0.75, linestyle="none"),
         Patch(facecolor=NULLF, edgecolor="none")]
    lab = ["Imagined walking / gate open / hit",
           "Rest / gate closed / correct rest",
           "Gate score (1 s steps)",
           r"$\theta_{sym}$ threshold",
           r"$\pm\delta$ hysteresis band",
           "Miss",
           "False alarm",
           "Wrong decision",
           "Not scored"]
    lf = plt.figure(figsize=(6.22, 1.15), dpi=DPI_OUT)
    lf.legend(h, lab, loc="center", frameon=False, ncol=4, handlelength=1.6,
              columnspacing=1.4, labelspacing=0.5, fontsize=FS_NOTE)
    out = OUT_LEG if out is None else out
    lf.savefig(out, dpi=DPI_OUT, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(lf)
    from PIL import Image
    lw_mm = Image.open(out).size[0] / DPI_OUT * 25.4
    assert lw_mm <= FULL_W + 0.5, f"legend {lw_mm:.1f} mm wider than the {FULL_W:.0f} mm column"
    print(f"wrote {out}  ({lw_mm:.1f} mm wide)")


def preview_png():
    """Panel with its legend stacked below."""
    from PIL import Image
    a, b = Image.open(OUT).convert("RGB"), Image.open(OUT_LEG)
    bg = Image.new("RGB", b.size, "white")
    bg.paste(b, mask=b.split()[3] if b.mode == "RGBA" else None)
    b = bg
    if b.width != a.width:
        b = b.resize((a.width, max(1, round(b.height * a.width / b.width))), Image.LANCZOS)
    pad = 24
    out = Image.new("RGB", (a.width, a.height + b.height + pad), "white")
    out.paste(a, (0, 0))
    out.paste(b, (0, a.height + pad))
    os.makedirs(os.path.dirname(OUT_PREV), exist_ok=True)
    out.save(OUT_PREV)
    print(f"wrote {OUT_PREV}  ({out.width} x {out.height} px)")


def main():
    PS.apply(plt)
    plt.rcParams["hatch.linewidth"] = 0.4
    if PANEL_MODE:
        global FS_NOTE, FS_TICK, FS_BASE, FULL_W, LW_STEP, LW_SYM, H_TRUTH, H_SCORE, H_RASTER, H_GATE, GAP_MM
        global LEFT_MM, RIGHT_MM, BOTTOM_MM, TOP_MM, DPI_OUT, SI
        import si_panel_style as SI
        DPI_OUT = SI.DPI
        SI.apply(plt)
        plt.rcParams["hatch.linewidth"] = 0.4
        FS_NOTE, FS_TICK, FS_BASE, FULL_W = SI.TICK_PT, SI.TICK_PT, SI.TITLE_PT, 117.1
        LW_STEP, LW_SYM = 0.75, SI.LW_REF
        H_TRUTH, H_SCORE, H_RASTER, H_GATE, GAP_MM = 3.2, 33.5, 2.8, 3.2, 1.4
        LEFT_MM, RIGHT_MM, BOTTOM_MM, TOP_MM = 18.0, 2.0, 8.6, 1.5
    sc, lab, dur = GT.load_scores(SCORES)
    th = read_thresholds(THR)
    anchors = GT.load_anchors(ANCHORS)
    have = sorted({k[0] for k in sc})
    assert have == sorted(SESS), f"{SCORES}: sessions {have} != {sorted(SESS)}"

    es = EXCERPT_SESSION if EXCERPT_SESSION in have else SESS[0]
    B = GT.build_session(es, sc, lab, dur, anchors, th)
    Wall, (sym, on, off) = build_windows(es, sc, lab, dur, anchors, th)
    bl = B["blocks"]
    t0 = next((a for a, b, c in bl if c == "IW" and a + EXCERPT_S <= bl[-1][1]), 0.0)
    t1 = t0 + EXCERPT_S
    W = Wall[(Wall.t_start_s < t1) & (Wall.t_end_s > t0)].reset_index(drop=True)

    m_ex, m_all = metrics(W), metrics(Wall)

    print(f"== fig7_gate_window_verdict  GATE_SET={GATE_SET}  excerpt={es} "
          f"t = {t0:.0f}-{t1:.0f} s ==")
    print(f"  windows in excerpt      {m_ex['n_windows']}  "
          f"({m_ex['n_iw_windows']} IW + {m_ex['n_rest_windows']} rest)")
    print(f"  hit / miss / FA / CR    {m_ex['n_hit']} / {m_ex['n_miss']} / "
          f"{m_ex['n_false_alarm']} / {m_ex['n_correct_rest']}  = "
          f"{m_ex['n_hit'] + m_ex['n_miss'] + m_ex['n_false_alarm'] + m_ex['n_correct_rest']}"
          f" (n = {m_ex['n_windows']})  OK")
    print(f"  S0 at theta_sym         acc {m_ex['s0_accuracy']:.6f}  "
          f"sens {m_ex['s0_sensitivity']:.6f}  spec {m_ex['s0_specificity']:.6f}")
    print(f"  S1 open fraction        IW {m_ex['s1_open_fraction_iw']:.6f}  "
          f"rest {m_ex['s1_open_fraction_rest']:.6f}")
    print(f"  whole session           n {m_all['n_windows']}  acc {m_all['s0_accuracy']:.6f}  "
          f"sens {m_all['s0_sensitivity']:.6f}  spec {m_all['s0_specificity']:.6f}  "
          f"S1 sens {m_all['s1_sensitivity']:.6f}  spec {m_all['s1_specificity']:.6f}")

    acc = pd.read_csv(ACC_SD)
    ar = acc[(acc.session == es) & (acc.row_type == "session")]
    assert len(ar) == 1, f"{ACC_SD}: {len(ar)} session rows for {es}"
    ar = ar.iloc[0]
    assert int(ar.n_decisions) == m_all["n_windows"], (
        f"window set differs: table {int(ar.n_decisions)} vs here {m_all['n_windows']}")
    for a_key, b_key in (("s0_accuracy", "s0_accuracy"), ("s0_sensitivity", "s0_sensitivity"),
                         ("s0_specificity", "s0_specificity"),
                         ("s1_sensitivity", "s1_sensitivity"),
                         ("s1_specificity", "s1_specificity")):
        d = abs(float(ar[a_key]) - m_all[b_key])
        assert d < 1e-6, f"cross-check {a_key}: table {ar[a_key]} vs here {m_all[b_key]} (d={d})"
        print(f"  [cross-check] {a_key:16s} {m_all[b_key]:.6f} == table {float(ar[a_key]):.6f}  OK")

    tl = pd.read_csv(TL_SD)
    tg = tl[(tl.session == es) & (tl.row_type == "gate_S1")]
    mine = [(a, b, "open" if s else "closed") for a, b, s in B["gate"]]
    assert len(tg) == len(mine), f"S1 intervals: timeline {len(tg)} vs here {len(mine)}"
    for (a, b, s), (_i, r) in zip(mine, tg.iterrows()):
        assert abs(a - r.t_start_s) < 1e-3 and abs(b - r.t_end_s) < 1e-3 and s == r.state, \
            f"S1 interval mismatch: here ({a:.4f},{b:.4f},{s}) vs timeline " \
            f"({r.t_start_s},{r.t_end_s},{r.state})"
    n_ex_iv = sum(1 for a, b, _s in mine if a < t1 and b > t0)
    print(f"  [cross-check] S1 intervals    {len(mine)} identical to the timeline "
          f"(to 1e-3 s; {n_ex_iv} of them inside the excerpt)  OK")

    fig, axes, w_mm, h_mm = build_axes()
    draw(fig, axes, W, B, sym, on, off, t0, t1, es)
    note_block(fig, w_mm, h_mm, m_ex, es, m_all)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for t in fig.texts:
        boxes.append(("note", t.get_window_extent(r)))
    for name, ax in axes.items():
        for t in ax.get_yticklabels() + ax.get_xticklabels():
            if t.get_text():
                boxes.append((f"{name}:tick", t.get_window_extent(r)))
        for t in (ax.yaxis.label, ax.xaxis.label):
            if t.get_text():
                boxes.append((f"{name}:label", t.get_window_extent(r)))
    W_px, H_px = fig.get_size_inches() * fig.dpi
    TOL = 2.0
    bad = [n for n, b in boxes if b.x0 < -TOL or b.y0 < -TOL or b.x1 > W_px + TOL
           or b.y1 > H_px + TOL]
    assert not bad, f"text outside the canvas: {bad}"
    ov = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (n1, b1), (n2, b2) = boxes[i], boxes[j]
            w = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            hh = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if w > 1.0 and hh > 1.0:
                ov.append((n1, n2, round(w, 1), round(hh, 1)))
    assert not ov, f"overlapping text: {ov}"
    print(f"  [layout] {w_mm:.1f} x {h_mm:.1f} mm, {len(boxes)} text boxes, "
          f"0 overlaps, all inside the canvas  OK")

    if PANEL_MODE:
        for nm, ax_ in axes.items():
            assert ax_.get_title() == "" and ax_.get_title("left") == "" \
                and ax_.get_title("right") == "", f"S1e/{nm}: axes title still set"
            assert ax_.get_legend() is None, f"S1e/{nm}: legend still inside the panel"
            assert list(ax_.texts) == [], \
                f"S1e/{nm}: text still inside the panel: {[t.get_text() for t in ax_.texts]}"
        assert list(fig.texts) == [], \
            f"figure text left: {[t.get_text() for t in fig.texts]}"
        print(f"[panel spec] {len(axes)}/{len(axes)} axes: no title, no legend, no in-axes "
              f"text, no figure text  OK")
        PANEL_DIR.mkdir(parents=True, exist_ok=True)
        _pp = PANEL_DIR / "S1e_gate_window_verdict.png"
        SI.save(fig, _pp)
        plt.close(fig)
        legend_png(on - sym, out=PANEL_DIR / "S1e_gate_window_verdict_legend.png")
        return

    fig.savefig(OUT, dpi=PS.DPI)
    plt.close(fig)
    print(f"wrote {OUT}  ({w_mm:.1f} x {h_mm:.1f} mm @ {PS.DPI} dpi)")
    legend_png(on - sym)
    preview_png()

    cols = ["session", "event", "instructed", "t_start_s", "t_end_s", "score", "theta_sym",
            "delta", "classified_iw", "verdict", "s1_open"]
    rows = [dict(row_type="window", plotted=True, **{c: r[c] for c in cols})
            for _i, r in W.iterrows()]
    for scope, mm_, plotted in (("excerpt", m_ex, True), ("whole_session", m_all, False)):
        rows.append(dict(row_type=f"summary_{scope}", session=es, plotted=plotted, **mm_))
    pd.DataFrame(rows).to_csv(SD, index=False, float_format="%.6f")

    def rel(p):
        return os.path.relpath(p, ROOT).replace(os.sep, "/")

    def md5(p):
        import hashlib
        return hashlib.md5(open(p, "rb").read()).hexdigest()

    meta = [
        ("window_s", f"{WIN:.0f}"),
        ("score_column", (f"{RULE_COL} -- the fold-standardised score, in units of the delta of "
                          "the fold that emitted each decision, so theta_sym = 0 and "
                          "theta_on/off = +/-1 for every session. The per-fold (theta_sym, delta) "
                          "spread is in fig7_gate_accuracy_source_data.csv (not distributed) "
                          "(row_type=fold_thresholds). The Source Data column is named `score` "
                          "for exactly this reason." if GATE_CV == "within10"
                          else "fuse_V1, with the per-session (theta_sym, delta).")),
        ("gate_set", f"{GATE_SET} ({len(SESS)} sessions: {', '.join(SESS)}); the panel shows ONE "
                     "of them, the whole set is fig7_gate_timeline.png."),
        ("rule_S0", "memory-less per-window classifier, open iff s(t) >= theta_sym. This is what "
                    "the verdict fill and the decision raster show: hit = IW and s >= theta_sym, "
                    "miss = IW and s < theta_sym, false alarm = rest and s >= theta_sym, correct "
                    "rest = rest and s < theta_sym (drawn with NO fill)."),
        ("thresholds", f"session {es}: theta_sym = {sym:.6f}, delta = {on - sym:.6f}, "
                       f"theta_on = {on:.6f}, theta_off = {off:.6f}. theta_sym is the S1a natural "
                       "zero ((0 - centre_A)/sd_A + (0 - centre_B)/sd_B)/2; delta = 0.25 * IQR of "
                       "the session's calibration fused scores."),
        ("excerpt", f"{es}, concatenated t = {t0:.0f}-{t1:.0f} s ({EXCERPT_S:.0f} s). Same "
                    "session and same t0 rule as figures_out/fig7/fig7_gate_timeline_excerpt.png (the first "
                    "imagined-walking block that leaves a full 180 s inside the session), so the "
                    "two panels show the same stretch."),
        ("whole_session_metrics", f"n = {m_all['n_windows']} windows; S0 agreement "
                                  f"{m_all['s0_accuracy']:.6f}, sensitivity "
                                  f"{m_all['s0_sensitivity']:.6f}, specificity "
                                  f"{m_all['s0_specificity']:.6f}; S1 sensitivity "
                                  f"{m_all['s1_sensitivity']:.6f}, specificity "
                                  f"{m_all['s1_specificity']:.6f}. Asserted equal to 1e-6 to the "
                                  f"{es} row of fig7_gate_accuracy_source_data.csv (not distributed) "
                                  "(same window set, n_decisions checked first) before anything "
                                  "is written."),
        ("clock", "CONCATENATED, not session wall-clock: the evaluated IW and rest events only, "
                  "back to back in true chronological order at their true durations. Dropped and "
                  "invisible: real walking, sit-to-stand, every inter-event gap and pre-cue "
                  f"interval, and any event too short for one {WIN:.0f} s window"
                  + ("." if GATE_CV == "within10"
                     else ", plus the >= 30 s calibration rest segment.")
                  + (" Under within10 NO rest event is reserved as calibration, so every "
                     "imagined-walking and rest event of the session is on this clock."
                     if GATE_CV == "within10" else "")
                  + " There is NO null category. Thin pale verticals are event seams."),
        ("colours", "contract palette, achromatic: IW / gate-open / classified-IW #8C8C8C, REST / "
                    "gate-closed / classified-rest white with a #9AA0A6 edge, not-scored #D9D9D9, "
                    "ink #1A1A1A, hysteresis band #B3B3B3. Verdict is carried by solid vs hollow "
                    "vs hatch and by the x mark, not by hue."),
        ("cohort", "n = 2 patients, case-level. One session of one patient is an illustration, "
                   "not an estimate."),
        ("producer", "03_figures/fig7_gate_window_verdict.py"),
    ]
    meta = GE.meta(meta)
    with open(MD, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(meta)
    print("wrote", SD, MD)


if __name__ == "__main__":
    main()
