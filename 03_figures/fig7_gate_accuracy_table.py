"""Figure 7 — Intent gate and physics simulation of the decoded gait phase

_gate_accuracy_table — Fig. 7 gate panel: the working-point accuracy table

Reads:
  simulation and gate output in 02_decoding/simulation/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_style as PS
from project_palette import INK, MID, NULL_FILL, PALE

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PW = f"{ROOT}/02_decoding/simulation/result"

GATE_CV = os.environ.get("GATE_CV", "within10")
assert GATE_CV in ("within10", "loso"), f"GATE_CV must be within10|loso, got {GATE_CV!r}"
GATE_WIN = os.environ.get("GATE_WIN", "3" if GATE_CV == "within10" else "4")
assert GATE_WIN in ("3", "4"), f"GATE_WIN must be 3|4, got {GATE_WIN!r}"
WIN = float(GATE_WIN)
GATE_SET = os.environ.get("GATE_SET", "post_operative")
assert GATE_SET in ("post_operative", "post_operative_subset"), f"GATE_SET must be post_operative|post_operative_subset, got {GATE_SET!r}"
assert not (GATE_CV == "within10" and GATE_SET != "post_operative"), \
    "unsupported GATE_CV / GATE_SET combination"
assert not (GATE_CV == "loso" and GATE_WIN != "4"), \
    "unsupported GATE_CV / GATE_WIN combination"

RULE_COL = "e_sigma" if GATE_CV == "within10" else "fuse_V1"
AUC_COL = "fuse_V1"

SESS = {
    "post_operative": ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
                 "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06",
                 "pat02_sess08", "pat02_sess09"],
    "post_operative_subset": ["pat01_sess04", "pat01_sess06", "pat01_sess07", "pat01_sess09",
                "pat02_sess03", "pat02_sess05", "pat02_sess06"],
}[GATE_SET]
SCORES, THR = (
    (f"{PW}/gate_within_session_scores_T{GATE_WIN}.csv",
     f"{PW}/gate_within_session_thresholds_T{GATE_WIN}.csv") if GATE_CV == "within10" else
    {"post_operative": (f"{PW}/gate_event_set_scores.csv", f"{PW}/gate_event_set_thresholds.csv"),
     "post_operative_subset": (f"{PW}/gate_fused_checked_scores.csv",
                 f"{PW}/gate_fused_checked_thresholds.csv")}[GATE_SET])

OUT = f"{ROOT}/figures_out/fig7/fig7_gate_accuracy_table.png"
SD = f"{ROOT}/04_source_data/fig7/fig7_gate_accuracy_source_data.csv"
MD = f"{ROOT}/04_source_data/fig7/fig7_gate_accuracy_metadata.csv"
import gate_config as GE
SCORES, THR, OUT, SD, MD = (GE.scoped(p) for p in (SCORES, THR, OUT, SD, MD))

IQR_K = 0.25
HOP_S = 1.0


def short(s):
    from patient_session_labels import label as _plabel
    return _plabel(s, with_patient=True)


def read_thresholds(path):
    """(theta_sym, theta_on, theta_off) per session."""
    if GATE_CV == "within10":
        return {s: (0.0, 1.0, -1.0) for s in SESS}
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["variant"] != "V1":
            continue
        t = ((0.0 - float(r["centre_A"])) / float(r["sd_A"])
             + (0.0 - float(r["centre_B"])) / float(r["sd_B"])) / 2.0
        d = IQR_K * float(r["calib_iqr"])
        out[r["session"]] = (t, t + d, t - d)
    return out


def fold_threshold_spread(path, rung="L5_headline"):
    """Per session, the fold-to-fold spread of the within10 (theta_sym, delta) pairs."""
    if GATE_CV != "within10":
        return {}
    per = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("rung") != rung:
            continue
        per.setdefault(r["session"], []).append((float(r["theta_sym"]), float(r["delta"]),
                                                 float(r["ref_dur_s"]), int(r["n_ref_events"])))
    out = {}
    for s, v in per.items():
        th = np.array([a for a, _b, _c, _d in v])
        dl = np.array([b for _a, b, _c, _d in v])
        rd = np.array([c for _a, _b, c, _d in v])
        out[s] = dict(n_folds=len(v), theta_min=float(th.min()), theta_max=float(th.max()),
                      theta_sd=float(th.std(ddof=1)), delta_min=float(dl.min()),
                      delta_max=float(dl.max()), delta_sd=float(dl.std(ddof=1)),
                      ref_dur_min_s=float(rd.min()), ref_dur_max_s=float(rd.max()),
                      n_folds_ref_under_30s=int((rd < 30.0 - 1e-9).sum()))
    return out


def read_events(path, rule_col=None, auc_col=AUC_COL):
    """{(session, event): (t_end, score_for_the_RULE, score_for_the_AUC, label)} by t_end."""
    rule_col = RULE_COL if rule_col is None else rule_col
    ev = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        ev.setdefault((r["session"], r["event"]), []).append(
            (float(r["t_end_ev_s"]), float(r[rule_col]), float(r[auc_col]), int(r["label"])))
    out = {}
    for k, v in ev.items():
        v.sort()
        out[k] = (np.array([a for a, _, _, _ in v]), np.array([b for _, b, _, _ in v]),
                  np.array([c for _, _, c, _ in v]), v[0][3])
    return out


def s1_states(x, on, off, start_open):
    """The declared symmetric single-window rule. Returns (state per decision, n transitions)."""
    st, prev, out, tr = bool(start_open), bool(start_open), np.empty(len(x), bool), 0
    for i, v in enumerate(x):
        if st and v < off:
            st = False
        elif (not st) and v >= on:
            st = True
        out[i] = st
        tr += int(st != prev)
        prev = st
    return out, tr


def session_metrics(ev, sess, th):
    """Every declared readout for one session, on its concatenated valid IW + rest stream."""
    sym, on, off = th[sess]
    keys = [k for k in ev if k[0] == sess]
    n = n_iw_dec = n_rest_dec = 0
    s0_hit = s1_hit = s0_iw = s1_iw = s0_rest = s1_rest = 0
    n_iw_ev = n_rest_ev = n_det = n_never = n_split = extra = 0
    lat, ys, ss = [], [], []
    for k in keys:
        t, x, xa, y = ev[k]
        ys.append(np.full(len(x), y))
        ss.append(xa)
        p0 = x >= sym
        p1, tr = s1_states(x, on, off, start_open=(y == 0))
        n += len(x)
        s0_hit += int((p0 == (y == 1)).sum())
        s1_hit += int((p1 == (y == 1)).sum())
        extra += max(0, tr - 1)
        if y == 1:
            n_iw_ev += 1
            n_iw_dec += len(x)
            s0_iw += int(p0.sum())
            s1_iw += int(p1.sum())
            n_det += int(p1.any())
            bouts = int(p1[0]) + int(np.sum(p1[1:] & ~p1[:-1]))
            n_split += int(bouts > 1)
        else:
            n_rest_ev += 1
            n_rest_dec += len(x)
            s0_rest += int(p0.sum())
            s1_rest += int(p1.sum())
            closed = np.where(~p1)[0]
            if len(closed):
                lat.append(float(t[closed[0]]))
            else:
                n_never += 1
    y_all, s_all = np.concatenate(ys), np.concatenate(ss)
    sens0, spec0 = s0_iw / n_iw_dec, 1 - s0_rest / n_rest_dec
    sens1, spec1 = s1_iw / n_iw_dec, 1 - s1_rest / n_rest_dec
    tiled = np.array([abs(ev[k][0][i] % WIN) < 1e-9 for k in keys for i in range(len(ev[k][0]))])
    return dict(
        session=sess, n_iw_events=n_iw_ev, n_rest_events=n_rest_ev, n_decisions=n,
        n_iw_decisions=n_iw_dec, n_rest_decisions=n_rest_dec, stream_min=n * HOP_S / 60.0,
        auc_tiled=float(roc_auc_score(y_all[tiled], s_all[tiled])),
        auc_dense=float(roc_auc_score(y_all, s_all)),
        theta_sym=sym, theta_on=on, theta_off=off,
        s0_accuracy=s0_hit / n, s0_sensitivity=sens0, s0_specificity=spec0,
        s0_balanced_accuracy=0.5 * (sens0 + spec0),
        s1_accuracy=s1_hit / n, s1_sensitivity=sens1, s1_specificity=spec1,
        s1_balanced_accuracy=0.5 * (sens1 + spec1),
        s1_detected=n_det, s1_detection_rate=n_det / n_iw_ev,
        s1_close_latency_median_s=(float(np.median(lat)) if lat else np.nan),
        s1_rests_never_closed=n_never, s1_frac_rests_never_closed=n_never / n_rest_ev,
        s1_extra_flips=extra, s1_extra_flips_per_min=extra / (n * HOP_S / 60.0),
        s1_iw_events_split=n_split, s1_frac_iw_events_split=n_split / n_iw_ev,
        _s0_hit=s0_hit, _s1_hit=s1_hit, _s0_iw=s0_iw, _s1_iw=s1_iw,
        _s0_rest=s0_rest, _s1_rest=s1_rest, _lat=lat)


def pooled(per):
    """Decision-weighted pooled row (NOT a mean of session means, except where noted)."""
    g = lambda k: sum(p[k] for p in per)
    n, niw, nrest = g("n_decisions"), g("n_iw_decisions"), g("n_rest_decisions")
    sens0, spec0 = g("_s0_iw") / niw, 1 - g("_s0_rest") / nrest
    sens1, spec1 = g("_s1_iw") / niw, 1 - g("_s1_rest") / nrest
    lat = [x for p in per for x in p["_lat"]]
    return dict(
        session="POOLED", n_iw_events=g("n_iw_events"), n_rest_events=g("n_rest_events"),
        n_decisions=n, n_iw_decisions=niw, n_rest_decisions=nrest,
        stream_min=n * HOP_S / 60.0,
        auc_tiled=float(np.mean([p["auc_tiled"] for p in per])),
        auc_dense=float(np.mean([p["auc_dense"] for p in per])),
        theta_sym=np.nan, theta_on=np.nan, theta_off=np.nan,
        s0_accuracy=g("_s0_hit") / n, s0_sensitivity=sens0, s0_specificity=spec0,
        s0_balanced_accuracy=0.5 * (sens0 + spec0),
        s1_accuracy=g("_s1_hit") / n, s1_sensitivity=sens1, s1_specificity=spec1,
        s1_balanced_accuracy=0.5 * (sens1 + spec1),
        s1_detected=g("s1_detected"), s1_detection_rate=g("s1_detected") / g("n_iw_events"),
        s1_close_latency_median_s=float(np.median(lat)) if lat else np.nan,
        s1_rests_never_closed=g("s1_rests_never_closed"),
        s1_frac_rests_never_closed=g("s1_rests_never_closed") / g("n_rest_events"),
        s1_extra_flips=g("s1_extra_flips"),
        s1_extra_flips_per_min=g("s1_extra_flips") / (n * HOP_S / 60.0),
        s1_iw_events_split=g("s1_iw_events_split"),
        s1_frac_iw_events_split=g("s1_iw_events_split") / g("n_iw_events"))


def run(scores_path, thr_path, sessions, rule_col=None):
    ev = read_events(scores_path, rule_col=rule_col)
    th = read_thresholds(thr_path)
    have = sorted({k[0] for k in ev})
    assert have == sorted(sessions), f"{scores_path}: sessions {have} != {sorted(sessions)}"
    per = [session_metrics(ev, s, th) for s in sessions]
    return per, pooled(per)


COLS = [("session", "session", "{}"), ("n_decisions", "n dec", "{:d}"),
        ("auc_tiled", "AUC", "{:.3f}"),
        ("s0_accuracy", "acc", "{:.3f}"), ("s0_sensitivity", "sens", "{:.3f}"),
        ("s0_specificity", "spec", "{:.3f}"), ("s0_balanced_accuracy", "bacc", "{:.3f}"),
        ("s1_accuracy", "acc", "{:.3f}"), ("s1_sensitivity", "sens", "{:.3f}"),
        ("s1_specificity", "spec", "{:.3f}"), ("s1_balanced_accuracy", "bacc", "{:.3f}"),
        ("s1_detection_rate", "detect", "{:.2f}"),
        ("s1_close_latency_median_s", "close s", "{:.1f}"),
        ("s1_frac_rests_never_closed", "open rest", "{:.2f}"),
        ("s1_extra_flips_per_min", "flips/min", "{:.1f}"),
        ("s1_frac_iw_events_split", "split IW", "{:.2f}")]
GROUPS = [(1, 3, ""), (3, 7, "S0  memory-less"), (7, 16, "S1  symmetric open/close")]


def render(per, po):
    PS.apply(plt)
    rows = per + [po]
    ncol, nrow = len(COLS), len(rows) + 2
    col_mm = [20.0, 9.0, 9.0] + [9.0] * 4 + [9.0] * 4 + [11.0, 11.5, 12.5, 12.5, 11.5]
    w_mm, h_mm, top_mm, row_mm = sum(col_mm) + 4.0, 0.0, 3.0, 4.6
    h_mm = top_mm + nrow * row_mm + 3.0
    fig = plt.figure(figsize=(w_mm * PS.MM, h_mm * PS.MM), dpi=PS.DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.axis("off")
    x0 = [2.0]
    for c in col_mm:
        x0.append(x0[-1] + c)

    def y(i):
        return h_mm - top_mm - (i + 0.5) * row_mm

    for a, b, lab in GROUPS:
        if not lab:
            continue
        xc = (x0[a] + x0[b]) / 2
        ax.text(xc, y(0), lab, ha="center", va="center", fontsize=PS.NOTE_PT, color=MID)
        ax.plot([x0[a] + 0.6, x0[b] - 0.6], [y(0) - row_mm * 0.42] * 2, color=PALE, lw=0.6)
    for j, (_k, lab, _f) in enumerate(COLS):
        ax.text(x0[j] + (0.0 if j == 0 else col_mm[j] - 1.0), y(1), lab,
                ha="left" if j == 0 else "right", va="center", fontsize=PS.NOTE_PT, color=INK)
    ax.plot([x0[0], x0[-1]], [y(1) - row_mm * 0.42] * 2, color=INK, lw=0.9)
    for i, r in enumerate(rows):
        yy = y(i + 2)
        pooled_row = r["session"] == "POOLED"
        if pooled_row:
            ax.plot([x0[0], x0[-1]], [yy + row_mm * 0.5] * 2, color=INK, lw=0.9)
        elif i % 2 == 1:
            ax.add_patch(plt.Rectangle((x0[0], yy - row_mm * 0.44), x0[-1] - x0[0],
                                       row_mm * 0.88, facecolor=NULL_FILL, alpha=0.35,
                                       lw=0, zorder=0))
        for j, (k, _lab, f) in enumerate(COLS):
            v = r[k]
            if k == "session":
                txt = "pooled" if pooled_row else short(v)
            elif isinstance(v, float) and np.isnan(v):
                txt = "—"
            else:
                txt = f.format(int(v) if f == "{:d}" else v)
            ax.text(x0[j] + (0.0 if j == 0 else col_mm[j] - 1.0), yy, txt,
                    ha="left" if j == 0 else "right", va="center", fontsize=PS.NOTE_PT,
                    color=INK, fontweight="bold" if pooled_row else "normal")
    fig.savefig(OUT, dpi=PS.DPI)
    plt.close(fig)
    print(f"wrote {OUT}  ({w_mm:.0f} x {h_mm:.0f} mm)")


def main():

    print(f"\n== GATE_CV={GATE_CV} window={WIN:.0f}s GATE_SET={GATE_SET}: "
          f"{len(SESS)} sessions ==")
    per, po = run(SCORES, THR, SESS)
    spread = fold_threshold_spread(THR)
    for s, d in sorted(spread.items()):
        print(f"  [folds] {s:14s} {d['n_folds']} folds  theta_sym "
              f"{d['theta_min']:+.3f}..{d['theta_max']:+.3f} (sd {d['theta_sd']:.3f})  "
              f"delta {d['delta_min']:.3f}..{d['delta_max']:.3f}  reference "
              f"{d['ref_dur_min_s']:.1f}-{d['ref_dur_max_s']:.1f} s"
              + (f"   {d['n_folds_ref_under_30s']} folds under 30 s"
                 if d["n_folds_ref_under_30s"] else ""))
    for r in per + [po]:
        print(f"  {r['session']:14s} n={r['n_decisions']:5d} AUC {r['auc_tiled']:.3f} | "
              f"S0 acc {r['s0_accuracy']:.3f} sens {r['s0_sensitivity']:.3f} "
              f"spec {r['s0_specificity']:.3f} | S1 acc {r['s1_accuracy']:.3f} "
              f"sens {r['s1_sensitivity']:.3f} spec {r['s1_specificity']:.3f} "
              f"det {r['s1_detected']}/{r['n_iw_events']} "
              f"flips {r['s1_extra_flips_per_min']:.1f}/min")

    render(per, po)

    out = []
    for r in per + [po]:
        out.append({k: v for k, v in r.items() if not k.startswith("_")}
                   | dict(row_type="session" if r["session"] != "POOLED" else "pooled",
                          gate_set=GATE_SET, plotted=True))
    for s, d in sorted(spread.items()):
        out.append(dict(row_type="fold_thresholds", session=s, gate_set=GATE_SET,
                        plotted=False, **d))
    pd.DataFrame(out).to_csv(SD, index=False, float_format="%.6f")

    n_under = sum(d["n_folds_ref_under_30s"] for d in spread.values())
    meta = [
        ("window_s", f"{WIN:.0f}"),
        ("per_fold_thresholds",
         "Successive decisions inside one session can come from different folds and hence from "
         "different (theta_sym, delta) pairs. The state machine is run on `e_sigma`, each "
         "decision already standardised by its emitting fold, so the rule constants are uniform "
         "(+1 / -1) even though the underlying thresholds are not. The fold-to-fold spread per "
         f"session is in the source data (row_type=fold_thresholds)."
         
         if GATE_CV == "within10" else "One (theta_sym, delta) per session."),
         ]
    meta += [
        ("gate_set", f"{GATE_SET} ({len(SESS)} sessions: {', '.join(SESS)})"),
        ("rule_S0", "memory-less per-window classifier, open iff s(t) >= theta_sym."),
        ("thresholds", "theta_sym = S1a natural zero = ((0 - centre_A)/sd_A + (0 - centre_B)/sd_B)/2; "
                       "delta = 0.25 * IQR(session calibration fused scores); theta_on = theta_sym "
                       "+ delta, theta_off = theta_sym - delta. Recomputed here from the thresholds "
                       "CSV."
                       ),
        ("time_weighting", "Each decision is one 1 s hop, so a fraction of decisions is a "
                           "time-weighted fraction of the concatenated valid IW + rest stream. "
                           "The pooled row re-aggregates raw counters (it is NOT a mean of session "
                           "means); the pooled AUC columns ARE means over sessions, as everywhere "
                           "else in this figure."),
        ("input", os.path.relpath(SCORES, ROOT).replace(os.sep, "/") + " + "
                  + os.path.relpath(THR, ROOT).replace(os.sep, "/")),
        ("cohort", "n = 2 patients, case-level. Eleven sessions from two people is not a "
                   "population estimate."),
        ("producer", "03_figures/fig7_gate_accuracy_table.py"),
    ]
    meta = GE.meta(meta)
    with open(MD, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(meta)
    print("wrote", SD, MD)
    write_xlsx(out, meta)


def write_xlsx(out, meta):
    """Sheet `working_points`: the session + pooled rows, readable headers first (the PNG's
    columns, in its order), then every raw counter the Source Data carries."""
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    xlsx = GE.scoped(f"{ROOT}/figures_out/supplementary/Supplementary_Table_Gate_Working_Points.xlsx")
    wp = pd.DataFrame([r for r in out if r["row_type"] in ("session", "pooled")])
    ft = pd.DataFrame([r for r in out if r["row_type"] == "fold_thresholds"])
    ft = ft.dropna(axis=1, how="all")
    label = {k: lab for k, lab, _f in COLS}
    label.update({"n_decisions": "n decisions", "auc_tiled": "AUC (tiled)",
                  "s0_accuracy": "S0 accuracy", "s0_sensitivity": "S0 sensitivity",
                  "s0_specificity": "S0 specificity", "s0_balanced_accuracy": "S0 balanced acc",
                  "s1_accuracy": "S1 accuracy", "s1_sensitivity": "S1 sensitivity",
                  "s1_specificity": "S1 specificity", "s1_balanced_accuracy": "S1 balanced acc",
                  "s1_detection_rate": "S1 detection rate",
                  "s1_close_latency_median_s": "S1 close latency median (s)",
                  "s1_frac_rests_never_closed": "S1 rests never closed (fraction)",
                  "s1_extra_flips_per_min": "S1 extra flips / min",
                  "s1_frac_iw_events_split": "S1 IW events split (fraction)"})
    first = [k for k, _l, _f in COLS if k in wp.columns]
    wp = wp[first + [c for c in wp.columns if c not in first]]
    wp = wp.rename(columns=label)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        wp.to_excel(xw, sheet_name="working_points", index=False)
        ft.to_excel(xw, sheet_name="fold_thresholds", index=False)
        pd.DataFrame(meta, columns=["key", "value"]).to_excel(xw, sheet_name="metadata",
                                                              index=False)
        for ws in xw.book.worksheets:
            for c in ws[1]:
                c.font = Font(bold=True)
                c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "B2"
            for j, col in enumerate(ws.columns, start=1):
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[get_column_letter(j)].width = min(max(10, width + 2), 60)
            if ws.title == "metadata":
                ws.column_dimensions["B"].width = 120
                for row in ws.iter_rows(min_row=2):
                    row[1].alignment = Alignment(wrap_text=True, vertical="top")
    print("wrote", xlsx, f"({len(wp)} working-point rows, {len(ft)} fold rows, "
          f"{len(meta)} metadata rows)")


if __name__ == "__main__":
    main()
