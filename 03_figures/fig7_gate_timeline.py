"""Figure 7 — Intent gate and physics simulation of the decoded gait phase

_gate_timeline — Fig. 7 gate panel: the whole-trial gate-state timeline

Reads:
  simulation and gate output in 02_decoding/simulation/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import Divider, Size

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_style as PS
import project_palette as PAL
from fig7_gate_accuracy_table import (GATE_CV, GATE_SET, GATE_WIN, RULE_COL,
                                      SESS, WIN, read_thresholds, s1_states)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PW = f"{ROOT}/02_decoding/simulation/result"

SCORES, THR, ANCHORS = (
    (f"{PW}/gate_within_session_scores_T{GATE_WIN}.csv",
     f"{PW}/gate_within_session_thresholds_T{GATE_WIN}.csv",
     f"{PW}/gate_within_session_events.csv") if GATE_CV == "within10" else
    {"post_operative": (f"{PW}/gate_event_set_scores.csv", f"{PW}/gate_event_set_thresholds.csv",
                  f"{PW}/gate_event_set_events.csv"),
     "post_operative_subset": (f"{PW}/gate_fused_checked_scores.csv",
                 f"{PW}/gate_fused_checked_thresholds.csv",
                 f"{PW}/gate_scores_timed.csv")}[GATE_SET])

OUT = f"{ROOT}/figures_out/fig7/fig7_gate_timeline.png"
OUT_EX = f"{ROOT}/figures_out/fig7/fig7_gate_timeline_excerpt.png"
OUT_LEG = f"{ROOT}/figures_out/fig7/fig7_gate_timeline_legend.png"
SD = f"{ROOT}/04_source_data/fig7/fig7_gate_timeline_source_data.csv"
MD = f"{ROOT}/04_source_data/fig7/fig7_gate_timeline_metadata.csv"
import gate_config as GE
SCORES, THR, ANCHORS, OUT, OUT_EX, OUT_LEG, SD, MD = (GE.scoped(p) for p in (
    SCORES, THR, ANCHORS, OUT, OUT_EX, OUT_LEG, SD, MD))

C_IW, C_REST, C_EDGE = PAL.IW, PAL.REST, PAL.REST_EDGE
INK, MID, PALE, NULLF = PAL.INK, PAL.MID, PAL.PALE, PAL.NULL_FILL

MM = PS.MM
BAND_MM = 5.0
BANDGAP_MM = 0.4
ROWGAP_MM = 2.6
PATGAP_MM = 3.4
LEFT_MM, BOTTOM_MM, TOP_MM = 17.0, 11.0, 2.0
HOP_S, WIN_S = 1.0, WIN

EXCERPT_SESSION = os.environ.get("GATE_EXCERPT_SESSION", "pat01_sess09")
EXCERPT_S = 180.0


def short(s):
    from patient_session_labels import label as _plabel
    return _plabel(s, with_patient=True)


def load_scores(path, col=None):
    """Per (session, event), the decision stream used by the state machine."""
    col = RULE_COL if col is None else col
    sc = defaultdict(list)
    lab, dur = {}, {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        k = (r["session"], r["event"])
        sc[k].append((float(r["t_end_ev_s"]), float(r[col])))
        lab[k] = r["label"] == "1"
        dur[k] = float(r["ev_dur_s"])
    for v in sc.values():
        v.sort()
    return sc, lab, dur


def load_anchors(path):
    """(session, event) -> event onset in session samples. Two accepted layouts."""
    a = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if "ev_a_samp" in r:
            if r.get("convention") != "demobase":
                continue
            a[(r["session"], r["event"])] = int(r["ev_a_samp"])
        else:
            if r["role"] != "evaluated":
                continue
            a[(r["session"], r["event"])] = int(r["a"])
    return a


def build_session(sess, sc, lab, dur, anchors, th):
    """Concatenated stream of one session: instructed blocks, gate intervals, seams."""
    sym, on, off = th[sess]
    evs = sorted([k for k in sc if k[0] == sess], key=lambda k: anchors[k])
    blocks, gate, seams, rows = [], [], [], []
    x = 0.0
    for k in evs:
        d = dur[k]
        t = np.array([a for a, _ in sc[k]])
        v = np.array([b for _, b in sc[k]])
        is_iw = lab[k]
        blocks.append((x, x + d, "IW" if is_iw else "REST"))
        st, _tr = s1_states(v, on, off, start_open=(not is_iw))
        cur, seg_a = None, None
        for te, s in zip(t, st):
            a0, a1 = x + te, x + te + HOP_S
            if s != cur:
                if cur is not None:
                    gate.append((seg_a, a0, cur))
                cur, seg_a = bool(s), a0
        if cur is not None:
            gate.append((seg_a, x + t[-1] + HOP_S, cur))
        rows.append(dict(session=sess, event=k[1], t_start_s=x, t_end_s=x + d,
                         instructed="IW" if is_iw else "REST",
                         scored_from_s=x + t[0], scored_to_s=x + t[-1] + HOP_S))
        x += d
        seams.append(x)
    merged = []
    for a, b, s in gate:
        if merged and merged[-1][2] == s and abs(merged[-1][1] - a) < 1e-9:
            merged[-1] = (merged[-1][0], b, s)
        else:
            merged.append((a, b, s))
    return dict(blocks=blocks, gate=merged, seams=seams[:-1], total=x, events=rows,
                scored=[(r["scored_from_s"], r["scored_to_s"]) for r in rows])


def draw_row(ax, y_state, y_gate, B, t_lo, t_hi):
    def clipx(a, b):
        a, b = max(a, t_lo), min(b, t_hi)
        return (a / 60.0, (b - a) / 60.0) if b > a else None

    for cat, face, edge in (("REST", C_REST, C_EDGE), ("IW", C_IW, "none")):
        xr = [z for z in (clipx(a, b) for a, b, c in B["blocks"] if c == cat) if z]
        if xr:
            ax.broken_barh(xr, (y_state, BAND_MM), facecolors=face,
                           edgecolors=edge, linewidth=0.25 if edge != "none" else 0.0, zorder=3)
    xr = [z for z in (clipx(0.0, B["total"]),) if z]
    ax.broken_barh(xr, (y_gate, BAND_MM), facecolors=NULLF, edgecolors="none", zorder=2)
    for st, face, edge in ((False, C_REST, C_EDGE), (True, C_IW, "none")):
        xr = [z for z in (clipx(a, b) for a, b, s in B["gate"] if s == st) if z]
        if xr:
            ax.broken_barh(xr, (y_gate, BAND_MM), facecolors=face, edgecolors=edge,
                           linewidth=0.25 if edge != "none" else 0.0, zorder=3)


def build_axes(n_rows, extra_gap):
    box_w = PS.TIERS["full"] - LEFT_MM - PS.RIGHT_MM
    row_h = 2 * BAND_MM + BANDGAP_MM
    box_h = n_rows * row_h + (n_rows - 1) * ROWGAP_MM + extra_gap
    w_mm, h_mm = LEFT_MM + box_w + PS.RIGHT_MM, BOTTOM_MM + box_h + TOP_MM
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM), dpi=PS.DPI)
    div = Divider(fig, (0, 0, 1, 1),
                  [Size.Fixed(LEFT_MM * MM), Size.Fixed(box_w * MM)],
                  [Size.Fixed(BOTTOM_MM * MM), Size.Fixed(box_h * MM)], aspect=False)
    ax = fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=1))
    return fig, ax, box_h, row_h


def main():
    PS.apply(plt)
    sc, lab, dur = load_scores(SCORES)
    th = read_thresholds(THR)
    anchors = load_anchors(ANCHORS)
    have = sorted({k[0] for k in sc})
    assert have == sorted(SESS), f"{SCORES}: sessions {have} != {sorted(SESS)}"
    missing = [k for k in sc if k not in anchors]
    assert not missing, f"{ANCHORS}: no onset for {missing[:5]} ({len(missing)} events)"

    B = {s: build_session(s, sc, lab, dur, anchors, th) for s in SESS}
    tmax = max(b["total"] for b in B.values()) / 60.0

    n_p1 = sum(1 for s in SESS if s.startswith("pat01"))
    fig, ax, box_h, row_h = build_axes(len(SESS), PATGAP_MM)
    yt, ylab = [], []
    y = box_h
    for i, s in enumerate(SESS):
        if i == n_p1:
            y -= PATGAP_MM
        y_state = y - BAND_MM
        y_gate = y_state - BANDGAP_MM - BAND_MM
        draw_row(ax, y_state, y_gate, B[s], 0.0, B[s]["total"])
        yt.append(y_gate + BAND_MM + BANDGAP_MM / 2.0)
        ylab.append(short(s))
        y = y_gate - ROWGAP_MM
    ax.set_xlim(0, tmax * 1.005)
    ax.set_ylim(0, box_h)
    ax.set_yticks(yt)
    ax.set_yticklabels(ylab)
    ax.set_xlabel("concatenated time over evaluated imagined-walking and rest events (min)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=PS.TICK_PT)
    ax.tick_params(axis="x", direction="out", labelsize=PS.TICK_PT)
    fig.savefig(OUT, dpi=PS.DPI)
    plt.close(fig)
    print(f"wrote {OUT}")

    es = EXCERPT_SESSION if EXCERPT_SESSION in B else SESS[0]
    bl = B[es]["blocks"]
    t0 = next((a for a, b, c in bl if c == "IW" and a + EXCERPT_S <= bl[-1][1]), 0.0)
    t1 = t0 + EXCERPT_S
    figx, axx, box_hx, _rh = build_axes(1, 0.0)
    y_state = box_hx - BAND_MM
    y_gate = y_state - BANDGAP_MM - BAND_MM
    draw_row(axx, y_state, y_gate, B[es], t0, t1)
    axx.set_xlim(t0 / 60.0, t1 / 60.0)
    axx.set_ylim(0, box_hx)
    axx.set_yticks([y_gate + BAND_MM + BANDGAP_MM / 2.0])
    axx.set_yticklabels([short(es)])
    axx.set_xlabel("concatenated time (min)")
    for sp in ("top", "right", "left"):
        axx.spines[sp].set_visible(False)
    axx.tick_params(axis="y", length=0, labelsize=PS.TICK_PT)
    axx.tick_params(axis="x", direction="out", labelsize=PS.TICK_PT)
    figx.savefig(OUT_EX, dpi=PS.DPI)
    plt.close(figx)
    n_iw_ex = sum(1 for a, b, c in bl if c == "IW" and a < t1 and b > t0)
    n_rest_ex = sum(1 for a, b, c in bl if c == "REST" and a < t1 and b > t0)
    print(f"wrote {OUT_EX}  ({es}, {t0:.0f}-{t1:.0f} s, {n_iw_ex} IW / {n_rest_ex} rest events)")

    handles = [Patch(facecolor=C_IW, edgecolor="none"),
               Patch(facecolor=C_REST, edgecolor=C_EDGE, linewidth=0.5),
               Patch(facecolor=NULLF, edgecolor="none"),
               Line2D([0], [0], color="none")]
    labels = ["imagined walking / gate open", "rest / gate closed", "not scored", ""]
    lf = plt.figure(figsize=(2.3, 0.9), dpi=PS.DPI)
    lf.legend(handles[:3], labels[:3], loc="center", frameon=False, handlelength=1.5,
              fontsize=PS.NOTE_PT, title="upper band: instructed  ·  lower band: gate state (S1)",
              title_fontsize=PS.NOTE_PT)
    lf.savefig(OUT_LEG, dpi=PS.DPI, bbox_inches="tight", transparent=True)
    plt.close(lf)
    print(f"wrote {OUT_LEG}")

    rows = []
    for s in SESS:
        for a, b, c in B[s]["blocks"]:
            rows.append(dict(row_type="instructed", session=s, t_start_s=a, t_end_s=b,
                             state=c, plotted=True))
        for a, b, st in B[s]["gate"]:
            rows.append(dict(row_type="gate_S1", session=s, t_start_s=a, t_end_s=b,
                             state="open" if st else "closed", plotted=True))
        for r in B[s]["events"]:
            rows.append(dict(row_type="event", plotted=False, **r))
    pd.DataFrame(rows).to_csv(SD, index=False, float_format="%.4f")

    tot_open_iw = sum(max(0.0, min(b, bb) - max(a, aa))
                      for s in SESS for a, b, st in B[s]["gate"] if st
                      for aa, bb, c in B[s]["blocks"] if c == "IW")
    tot_iw = sum(b - a for s in SESS for a, b, c in B[s]["blocks"] if c == "IW")
    meta = [
        ("gate_set", f"{GATE_SET} ({len(SESS)} sessions: {', '.join(SESS)})"),
        ("window_s", f"{WIN:.0f}"),
        ("score_column", (f"{RULE_COL} -- under within10 the drawn state machine runs on the "
                          "fold-standardised score, so the thresholds are the uniform "
                          "(0, +1, -1) in units of each decision own delta; the per-fold "
                          "(theta_sym, delta) spread is in fig7_gate_accuracy_source_data.csv "
                          "(row_type=fold_thresholds)." if GATE_CV == "within10"
                          else "fuse_V1 with the per-session (theta_sym, delta).")),
        ("karcher_solver", "checked (step-halving fixed point, tol 1e-10, itmax 20000)."),
        ("excerpt", f"{es}, concatenated t = {t0:.0f}-{t1:.0f} s ({EXCERPT_S:.0f} s), "
                    f"{n_iw_ex} imagined-walking and {n_rest_ex} rest events."),
        ("open_time_in_iw", f"{tot_open_iw / 60.0:.2f} min of gate-open time inside "
                            f"{tot_iw / 60.0:.2f} min of instructed imagined walking "
                            "(the rates themselves are in fig7_gate_accuracy_source_data.csv; "
                            "this key is a drawing cross-check, not the reported statistic)."),
        ("cohort", "n = 2 patients, case-level."),
        ("producer", "03_figures/fig7_gate_timeline.py"),
    ]
    meta = GE.meta(meta)
    with open(MD, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(meta)
    print("wrote", SD, MD)


if __name__ == "__main__":
    main()
