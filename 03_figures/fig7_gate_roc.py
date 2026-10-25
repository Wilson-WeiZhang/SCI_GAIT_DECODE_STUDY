"""Figure 7 — Intent gate and physics simulation of the decoded gait phase

_gate_roc — Fig. 7 gate panel: rest-vs-imagined-walking ROC of the deployable input-side gate

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
from sup_panel_export import PANEL_MODE, PANEL_DIR, export_stats
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_style as PS
from project_palette import MID, NULL_FILL, PATIENT_P1, PATIENT_P2

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

GATE_CV = os.environ.get("GATE_CV", "within10")
assert GATE_CV in ("within10", "loso"), f"GATE_CV must be within10|loso, got {GATE_CV!r}"
GATE_WIN = os.environ.get("GATE_WIN", "3" if GATE_CV == "within10" else "4")
assert GATE_WIN in ("3", "4"), f"GATE_WIN must be 3|4, got {GATE_WIN!r}"
WIN = float(GATE_WIN)
GATE_SET = os.environ.get("GATE_SET", "post_operative")
assert GATE_SET in ("post_operative", "post_operative_subset"), f"GATE_SET must be post_operative|post_operative_subset, got {GATE_SET!r}"
KARCHER = os.environ.get("GATE_KARCHER", "checked")
assert KARCHER in ("checked", "frozen"), f"GATE_KARCHER must be checked|frozen, got {KARCHER!r}"
assert not (GATE_SET == "post_operative" and KARCHER == "frozen"), \
    "unsupported GATE_SET / GATE_KARCHER combination"
assert not (GATE_CV == "within10" and (GATE_SET != "post_operative" or KARCHER != "checked")), \
    "unsupported GATE_CV / GATE_SET / GATE_KARCHER combination"
assert not (GATE_CV == "loso" and GATE_WIN != "4"), \
    "unsupported GATE_CV / GATE_WIN combination"
SCORES = f"{ROOT}/02_decoding/simulation/result/" + (
    f"gate_within_session_scores_T{GATE_WIN}.csv" if GATE_CV == "within10" else
    "gate_event_set_scores.csv" if GATE_SET == "post_operative" else
    "gate_fused_checked_scores.csv" if KARCHER == "checked" else
    "gate_fused_scores.csv")
OUT = f"{ROOT}/figures_out/fig7/fig7_gate_roc.png"
OUT_LEG = f"{ROOT}/figures_out/fig7/fig7_gate_roc_legend.png"
OUT_STATS = f"{ROOT}/figures_out/fig7/fig7_gate_roc_stats.txt"
OUT_SI = f"{ROOT}/figures_out/supplementary/sup_fig7_gate_auc_bootstrap.png"
SD = f"{ROOT}/04_source_data/fig7/fig7_gate_roc_source_data.csv"
MD = f"{ROOT}/04_source_data/fig7/fig7_gate_roc_metadata.csv"
SD_SI = f"{ROOT}/04_source_data/supplementary/sup_fig7_gate_auc_bootstrap_source_data.csv"
MD_SI = f"{ROOT}/04_source_data/supplementary/sup_fig7_gate_auc_bootstrap_metadata.csv"
import gate_config as GE
SCORES, OUT, OUT_LEG, OUT_STATS, OUT_SI, SD, MD, SD_SI, MD_SI = (GE.scoped(p) for p in (
    SCORES, OUT, OUT_LEG, OUT_STATS, OUT_SI, SD, MD, SD_SI, MD_SI))

SESS_ORDER = {
    "post_operative": ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
                 "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06",
                 "pat02_sess08", "pat02_sess09"],
    "post_operative_subset": ["pat01_sess04", "pat01_sess06", "pat01_sess07", "pat01_sess09",
                "pat02_sess03", "pat02_sess05", "pat02_sess06"],
}[GATE_SET]
N_SESS = len(SESS_ORDER)
MEANKEY = f"MEAN{N_SESS}"
SCORE_COLS = {"fuse_V1": "fuse2 V1 (deployable, headline)", "score_lda": "band-power LDA (comparator)"}
B = 1000
SEED = 0
LOSO_COMPARATOR = (f"{ROOT}/04_source_data/fig7/"
                   "fig7_gate_roc_leave_one_session_out_source_data.csv")


def loso11_comparator():
    """(auc, ci_lo, ci_hi) of the leave-one-session-out comparator, or None."""
    if not os.path.exists(LOSO_COMPARATOR):
        return None
    for r in csv.DictReader(open(LOSO_COMPARATOR, encoding="utf-8")):
        if (r.get("row_type") == "auc" and r.get("score") == "fuse_V1"
                and r.get("convention") == "tiled" and r.get("session") == "MEAN11"):
            return float(r["auc"]), float(r["ci_lo"]), float(r["ci_hi"])
    return None
KARCHER_NOTE = {
    "checked": "checked (step-halving fixed point, tol 1e-10, itmax 20000)",
    "frozen": "frozen (undamped 60-iteration fixed point)",
}[KARCHER]
TIER = os.environ.get("FIG7_GATE_TIER", "quarter")


def _shade(hex_color, t):
    """Lighten `hex_color` toward white by fraction t (0 = the colour itself)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(c + (255 - c) * t)) for c in (r, g, b))


_P1S = [s for s in SESS_ORDER if s.startswith("pat01")]
_P2S = [s for s in SESS_ORDER if s.startswith("pat02")]


def _ladder(n):
    hi = 0.58
    return np.linspace(0.0, hi, n) if n > 1 else np.array([0.0])


def _darken(hex_color, t):
    """Darken `hex_color` toward black by fraction t (0 = the colour itself)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(c * (1 - t))) for c in (r, g, b))


_MIX = _darken if os.environ.get("SI_PANEL_SPEC") == "panel" else _shade
SESS_COLOR = {s: _MIX(PATIENT_P1, t) for s, t in zip(_P1S, _ladder(len(_P1S)))}
SESS_COLOR.update({s: _MIX(PATIENT_P2, t) for s, t in zip(_P2S, _ladder(len(_P2S)))})


def short(sess):
    from patient_session_labels import label as _plabel
    return _plabel(sess, with_patient=True)


def auc_tiled(g, col):
    z = g[g.t_end_ev_s % WIN == 0]
    return roc_auc_score(z.label, z[col])


def auc_moment(g, col):
    w = 1.0 / g.groupby("event").event.transform("size")
    return roc_auc_score(g.label, g[col], sample_weight=w)


def _session_pools(g, col):
    """Per label: list of per-event score arrays (dense) and (tiled)."""
    pools = {}
    for lab in (0, 1):
        dense, tiled = [], []
        for _, ge in g[g.label == lab].groupby("event", sort=False):
            dense.append(ge[col].to_numpy())
            tiled.append(ge[ge.t_end_ev_s % WIN == 0][col].to_numpy())
        pools[lab] = (dense, tiled)
    return pools


def bootstrap_session(pools, rng, variant, B):
    out = np.empty(B)
    for b in range(B):
        ys, ss = [], []
        for lab in (0, 1):
            dense, tiled = pools[lab]
            n = len(dense)
            idx = rng.integers(n, size=n)
            if variant == "tiled":
                sc = np.concatenate([tiled[i] for i in idx])
            else:
                sc = np.array([dense[i][rng.integers(len(dense[i]))] for i in idx])
            ss.append(sc)
            ys.append(np.full(len(sc), lab))
        out[b] = roc_auc_score(np.concatenate(ys), np.concatenate(ss))
    return out


def main():
    df = pd.read_csv(SCORES)
    assert set(df.session) == set(SESS_ORDER), sorted(set(df.session))
    n_iw = df[df.label == 1].groupby("session").event.nunique().sum()
    n_rest = df[df.label == 0].groupby("session").event.nunique().sum()
    if GATE_SET == "post_operative_subset" and GATE_CV == "loso":
        assert (n_iw, n_rest, len(df)) == (270, 224, 1774), (n_iw, n_rest, len(df))
    else:
        assert n_iw > 270 and n_rest > 224 and len(df) > 1774, (n_iw, n_rest, len(df))
    if GATE_CV == "within10":
        assert "fold" in df.columns and "e_sigma" in df.columns,             f"{SCORES} does not look like a within10 stream"
        assert df.groupby(["session", "event"]).fold.nunique().max() == 1,             "an event carries rows from more than one fold"
    print(f"[input] GATE_CV={GATE_CV} window={WIN:.0f}s GATE_SET={GATE_SET} "
          f"solver={KARCHER}: {N_SESS} sessions, {n_iw} IW + {n_rest} rest events, "
          f"{len(df)} decisions")

    point = {(c, v): np.array([(auc_tiled if v == "tiled" else auc_moment)(df[df.session == s], c)
                               for s in SESS_ORDER])
             for c in SCORE_COLS for v in ("tiled", "moment")}

    rng = np.random.default_rng(SEED)
    draws = {}
    for c in SCORE_COLS:
        pools = {s: _session_pools(df[df.session == s], c) for s in SESS_ORDER}
        for v in ("tiled", "moment"):
            draws[(c, v)] = np.column_stack([bootstrap_session(pools[s], rng, v, B) for s in SESS_ORDER])

    summ = {}
    for (c, v), M in draws.items():
        m7 = M.mean(axis=1)
        summ[(c, v, MEANKEY)] = (point[(c, v)].mean(), m7.mean(), *np.percentile(m7, [2.5, 97.5]))
        for j, s in enumerate(SESS_ORDER):
            summ[(c, v, s)] = (point[(c, v)][j], M[:, j].mean(), *np.percentile(M[:, j], [2.5, 97.5]))
    hp, hb, hlo, hhi = summ[("fuse_V1", "tiled", MEANKEY)]
    mp, mb, mlo, mhi = summ[("fuse_V1", "moment", MEANKEY)]

    off_rows = []
    for t in sorted(df.t_end_ev_s.unique()):
        for c in SCORE_COLS:
            a = []
            for s in SESS_ORDER:
                g = df[(df.session == s) & (df.t_end_ev_s == t)]
                if g.label.nunique() == 2 and min((g.label == 0).sum(), (g.label == 1).sum()) >= 3:
                    a.append(roc_auc_score(g.label, g[c]))
            off_rows.append((t, c, len(a), int((df.t_end_ev_s == t).sum()), np.mean(a) if a else np.nan))

    PS.apply(plt)
    fig, ax = PS.panel_axes(plt, tier=TIER, aspect=1.0)
    handles, labels, roc_rows = [], [], []
    for s in SESS_ORDER:
        z = df[(df.session == s) & (df.t_end_ev_s % WIN == 0)]
        fpr, tpr, thr = roc_curve(z.label, z.fuse_V1)
        auc = roc_auc_score(z.label, z.fuse_V1)
        (h,) = ax.plot(fpr, tpr, color=SESS_COLOR[s],
                       lw=1.0 if PANEL_MODE else 1.4, solid_capstyle="round")
        handles.append(h); labels.append(f"{short(s)}   {auc:.2f}")
        for f_, t_, th in zip(fpr, tpr, thr):
            roc_rows.append(dict(row_type="roc_point", session=s, score="fuse_V1", convention="tiled",
                                 fpr=f_, tpr=t_, threshold=th, auc=auc, plotted=True))
    ax.plot([0, 1], [0, 1], color=MID, lw=1.0, ls="--", zorder=0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    PS.finish(ax)
    if not PANEL_MODE:
        fig.savefig(OUT, dpi=PS.DPI)
    plt.close(fig)

    lf = plt.figure(figsize=(1.6, 1.7), dpi=PS.DPI)
    lf.legend(handles, labels, loc="center", frameon=False, handlelength=1.6,
              fontsize=PS.NOTE_PT, title="session   AUC", title_fontsize=PS.NOTE_PT)
    if not PANEL_MODE:
        lf.savefig(OUT_LEG, dpi=PS.DPI, bbox_inches="tight", transparent=True)
    plt.close(lf)

    if not PANEL_MODE:
        PS.write_stats(OUT_STATS, [
            f"mean AUC = {hp:.3f}",
            f"95% CI {hlo:.3f}–{hhi:.3f}",
            (f"(fuse2 V1, deployable; {N_SESS} post-operative sessions [{GATE_SET}], "
             + ("within-session 10-fold by event;" if GATE_CV == "within10"
                else "leave-one-session-out;")),
            f" {WIN:.0f} s decision windows;",
            f" event-level stratified bootstrap, B = {B})",
        ])

    if PANEL_MODE:
        import si_panel_style as SI
        SI.apply(plt)
        fig, bx = SI.panel(plt, "quarter", 38.0, left_mm=8.4, bottom_mm=10.6, top_mm=1.0, right_mm=1.8)
    else:
        fig, bx = PS.panel_axes(plt, tier="half", aspect=0.62)
    _all = np.concatenate([draws[("fuse_V1", v)].mean(axis=1) for v in ("tiled", "moment")])
    bins = np.linspace(np.floor(_all.min() * 200) / 200, np.ceil(_all.max() * 200) / 200, 49)
    _leg_h, _leg_lab, _stat_lines = [], [], []
    for v, color, lab in (("tiled", PATIENT_P1, f"fixed {WIN:.0f} s tiling"),
                          ("moment", PATIENT_P2, f"random {WIN:.0f} s moment")):
        m7 = draws[("fuse_V1", v)].mean(axis=1)
        p, _, lo, hi = summ[("fuse_V1", v, MEANKEY)]
        (_h,) = bx.hist(m7, bins=bins, color=color, alpha=0.6, lw=0,
                        label=f"{lab}: {p:.3f} [{lo:.3f}, {hi:.3f}]")[2][:1]
        bx.axvline(p, color=color, lw=SI.LW_MAIN if PANEL_MODE else 1.2)
        _leg_h.append(_h)
        _leg_lab.append(lab)
        _stat_lines.append(
            ("Tiled" if v == "tiled" else "One random")
            + f" {WIN:.0f} s window" + (f"s: " if v == "tiled" else " per event: ")
            + f"mean AUC {p:.3f} (95% CI {lo:.3f} to {hi:.3f})")
    bx.set_xlabel(f"Mean AUC over {N_SESS} sessions" + ("\n" if PANEL_MODE else " ")
                  + "(bootstrap draws)")
    bx.set_ylabel("Draws")
    if PANEL_MODE:
        from matplotlib.ticker import MaxNLocator
        bx.xaxis.set_major_locator(MaxNLocator(4))
    if PANEL_MODE:
        assert bx.get_title() == "" and bx.get_title("left") == "" \
            and bx.get_title("right") == "", "S1f: axes title still set"
        assert bx.get_legend() is None, "S1f: legend still inside the panel"
        assert list(bx.texts) == [], \
            f"S1f: text still inside the panel: {[t.get_text() for t in bx.texts]}"
        assert list(fig.texts) == [], f"figure text left: {[t.get_text() for t in fig.texts]}"
        print("[panel spec] 1/1 axes: no title, no legend, no in-axes text, no figure text  OK")
        SI.finish(bx)
        PANEL_DIR.mkdir(parents=True, exist_ok=True)
        _pp = PANEL_DIR / "S1f_gate_auc_bootstrap.png"
        SI.save(fig, _pp)
        plt.close(fig)
        for _h, _l in zip(_leg_h, _leg_lab):
            _h.set_label(_l)
        SI.save_legend(_leg_h, PANEL_DIR / "S1f_gate_auc_bootstrap_legend.png", handlelength=1.2,
                       handletextpad=0.4, labelspacing=0.3)
        _stat_lines.append(f"{N_SESS} post-operative sessions, {B} bootstrap draws "
                           f"(events resampled with replacement, stratified by session and "
                           f"condition; the model is never refit)")
        export_stats(_stat_lines, "S1", "f", "gate_auc_bootstrap", width=5.6)
        (PANEL_DIR / "S1f_gate_auc_bootstrap_stats.txt").write_text(
            "\n".join(_stat_lines) + "\n", encoding="utf-8")
        print("wrote", PANEL_DIR / "S1f_gate_auc_bootstrap_stats.txt")
        print("[panel mode] figures_out/fig7 outputs and every Source Data CSV left untouched")
        return
    bx.legend(loc="upper left", frameon=False, fontsize=PS.NOTE_PT)
    PS.finish(bx)
    fig.savefig(OUT_SI, dpi=PS.DPI)
    plt.close(fig)

    rows = list(roc_rows)
    for (c, v, s), (p, bm, lo, hi) in summ.items():
        rows.append(dict(row_type="auc", session=s, score=c, convention=v, auc=p, boot_mean=bm,
                         ci_lo=lo, ci_hi=hi, B=B, seed=SEED,
                         plotted=(c == "fuse_V1" and v == "tiled" and s != MEANKEY)))
    for t, c, ns, nd, a in off_rows:
        rows.append(dict(row_type="auc_by_offset", score=c, convention="dense", t_end_s=t,
                         n_sessions=ns, n_decisions=nd, auc=a, plotted=False))
    for s in SESS_ORDER:
        g = df[df.session == s]
        rows.append(dict(row_type="inventory", session=s, n_iw_events=g[g.label == 1].event.nunique(),
                         n_rest_events=g[g.label == 0].event.nunique(), n_decisions=len(g),
                         n_tiled_windows=int((g.t_end_ev_s % WIN == 0).sum()), plotted=False))
    pd.DataFrame(rows).to_csv(SD, index=False, float_format="%.6f")

    pd.DataFrame({f"{c}_{v}": M.mean(axis=1) for (c, v), M in draws.items()}).to_csv(SD_SI, index=False, float_format="%.6f")

    N_IW_EV, N_REST_EV, N_DEC = int(n_iw), int(n_rest), len(df)
    lc = loso11_comparator()
    meta = [
        ("window_s", f"{WIN:.0f}"),
          ]
    meta += [
        ("headline_ci", f"95% CI {hlo:.4f}–{hhi:.4f}; event-level stratified (session x label) bootstrap, B = {B}, "
                        f"seed {SEED}, percentile interval, frozen scores, model never refit."),
        ("producer", "03_figures/fig7_gate_roc.py"),
    ]
    meta = GE.meta(meta)
    with open(MD, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["key", "value"]); w.writerows(meta)
    with open(MD_SI, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["key", "value"])
        w.writerows(GE.meta([
            ("columns", f"one column per score x convention; each row is one bootstrap draw's mean-of-{N_SESS}-sessions AUC "
                        f"(B = {B}, seed {SEED})."),
            ("tiled", f"{hp:.4f} [{hlo:.4f}, {hhi:.4f}]"), ("moment", f"{mp:.4f} [{mlo:.4f}, {mhi:.4f}]"),
            ("producer", "03_figures/fig7_gate_roc.py (same run as fig7_gate_roc_source_data.csv)"),
        ]))

    for (c, v), M in draws.items():
        p, bm, lo, hi = summ[(c, v, MEANKEY)]
        print(f"{c:10s} {v:7s} point {p:.4f} boot {bm:.4f} CI {lo:.4f}–{hi:.4f}")
    print("wrote", OUT, OUT_LEG, OUT_STATS, OUT_SI, SD, MD, SD_SI, MD_SI)


if __name__ == "__main__":
    main()
