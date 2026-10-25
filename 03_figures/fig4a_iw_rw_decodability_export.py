"""Figure 4 — Shared locomotor representation across imagined and real walking

statistics -- IW vs RW decoding gain: the values behind the panel

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
from pathlib import Path
import csv
import glob
import statistics as st
from collections import defaultdict

from scipy import stats

import analysis_config as ec

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "decoder_output"
SRC = ROOT / "04_source_data" / "fig4"

COL = "rpt_calib"

def session_means(path: Path, col: str = COL) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            v = r.get(col, "")
            if v in ("", "nan", "NaN"):
                continue
            by[r["session"]].append(float(v))
    return {s: st.mean(v) for s, v in by.items() if v}


def main() -> None:
    iw_f, rw_f = ec.path("h_iw_summary_csv"), ec.path("h_rw_summary_csv")
    IW, RW = session_means(iw_f), session_means(rw_f)

    common = sorted(set(IW) & set(RW))
    if len(common) != 80:
        raise SystemExit(f"COHORT GUARD: expected 80 common sessions, got {len(common)}. "
                         f"IW={len(IW)} RW={len(RW)}. Refusing to write a wrong-cohort Source Data.")

    rows = [dict(session=s, iw=IW[s], rw=RW[s], subj=s.split("_sess")[0]) for s in common]
    subj = sorted({r["subj"] for r in rows})
    if len(subj) != 42:
        raise SystemExit(f"COHORT GUARD: expected 42 subjects, got {len(subj)}.")

    with open(ec.sd("fig4_iw_rw_decodability_source_data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["session", "iw", "rw", "subj"])
        w.writeheader()
        w.writerows(rows)

    iw_v = [r["iw"] for r in rows]
    rw_v = [r["rw"] for r in rows]
    t, p = stats.ttest_rel(rw_v, iw_v)
    sess_gt = sum(a > b for a, b in zip(rw_v, iw_v))

    sIW: dict[str, list[float]] = defaultdict(list)
    sRW: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        sIW[r["subj"]].append(r["iw"])
        sRW[r["subj"]].append(r["rw"])
    subj_iw = [st.mean(sIW[k]) for k in subj]
    subj_rw = [st.mean(sRW[k]) for k in subj]
    st_, sp = stats.ttest_rel(subj_rw, subj_iw)
    subj_gt = sum(a > b for a, b in zip(subj_rw, subj_iw))

    ec.check_tag(ec.rel("h_iw_summary_csv"), "decoder", "IW input")
    ec.check_tag(ec.rel("h_rw_summary_csv"), "decoder", "RW input")

    meta = [
        ("figure", "Fig. 4 panel b - paired IW vs RW decoding accuracy over sessions"),
        ("claim", "Fixed linear decoder reconstructs joint angles ~2x better in real than imagined walking"),
        ("cohort", "healthy, 80 sessions, 42 participants"),
        ("n_sessions", str(len(rows))),
        ("n_subjects", str(len(subj))),
        ("metric", "rpt_calib (calibrated per-trial reconstruction r), session = mean of finite folds"),
        ("weighting", "session-equal"),
        ("iw_session_mean", f"{st.mean(iw_v):.4f}"),
        ("rw_session_mean", f"{st.mean(rw_v):.4f}"),
        ("sessions_rw_gt_iw", f"{sess_gt}/{len(rows)}"),
        ("paired_t_session", f"t={t:.4f}, P={p:.3e}"),
        ("subjects_rw_gt_iw", f"{subj_gt}/{len(subj)}"),
        ("paired_t_subject", f"t={st_:.4f}, P={sp:.3e}"),
        ("iw_source", iw_f.name),
        ("rw_source", rw_f.name),
        ("scope", "full-cohort healthy"),
    ]
    with open(ec.sd("fig4_iw_rw_decodability_metadata.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(meta)

    print(f"sessions            : {len(rows)}")
    print(f"subjects            : {len(subj)}")
    print(f"IW session-mean     : {st.mean(iw_v):.4f}")
    print(f"RW session-mean     : {st.mean(rw_v):.4f}")
    print(f"sessions RW > IW    : {sess_gt}/{len(rows)}")
    print(f"paired-t (session)  : t={t:.3f}, P={p:.3e}")
    print(f"subjects RW > IW    : {subj_gt}/{len(subj)}")
    print(f"paired-t (subject)  : t={st_:.3f}, P={sp:.3e}")
    print(f"\nwrote {ec.sd('fig4_iw_rw_decodability_source_data.csv')}")


def render() -> None:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(ec.sd("fig4_iw_rw_decodability_source_data.csv"))
    assert len(d) == 80, f"expected the 80-session Source Data, got {len(d)}"
    iw, rw = d.iw.to_numpy(), d.rw.to_numpy()
    t, p = stats.ttest_rel(rw, iw)
    stars = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "n.s."

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import panel_style as PS
    import project_palette as P
    INK, MID, PALE = PS.INK, PS.MID, PS.PALE
    FS = PS.FS
    PS.apply(plt)
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic"})
    PNG = ec.scoped(ROOT / "figures_out/fig4/fig4a_iw_rw_paired.png")
    rng = np.random.default_rng(0)
    fig, ax = PS.panel_axes(plt, left_mm=14.5, bottom_mm=10.0, top_mm=6.5)
    BW = .30
    IW_C, RW_C = P.IW, P.RW
    for x, v, face in ((0, iw, IW_C), (1, rw, RW_C)):
        q1, md, q3 = np.percentile(v, [25, 50, 75])
        w10, w90 = np.percentile(v, [10, 90])
        ax.plot([x, x], [w10, q1], color=INK, lw=1.3, zorder=2)
        ax.plot([x, x], [q3, w90], color=INK, lw=1.3, zorder=2)
        for yy in (w10, w90):
            ax.plot([x - .09, x + .09], [yy, yy], color=INK, lw=1.3,
                    solid_capstyle="butt", zorder=2)
        ax.add_patch(plt.Rectangle((x - BW / 2, q1), BW, q3 - q1, facecolor=face,
                                   edgecolor=INK, lw=1.4, zorder=3))
        ax.plot([x - BW / 2, x + BW / 2], [md, md], color="white" if face in (P.RW, P.IW) else INK, lw=1.2,
                solid_capstyle="butt", zorder=4)
    xs_iw = -.40 + rng.uniform(-.055, .055, len(iw))
    xs_rw = 1.40 + rng.uniform(-.055, .055, len(rw))
    for a_, b_, p_, q_ in zip(iw, rw, xs_iw, xs_rw):
        ax.plot([p_, q_], [a_, b_], color=PALE, lw=.3, alpha=.40, zorder=1)
    ax.scatter(xs_iw, iw, s=5, facecolor=IW_C, alpha=.85, lw=0, zorder=5)
    ax.scatter(xs_rw, rw, s=5, facecolor=RW_C, alpha=.85, lw=0, zorder=5)

    yb, tick = 1.005, .018
    ax.plot([0, 0, 1, 1], [yb, yb + tick, yb + tick, yb], color=INK, lw=1.1,
            solid_capstyle="butt", clip_on=False, zorder=6)
    ax.text(.5, yb + tick + .012, stars, ha="center", va="bottom",
            fontsize=PS.STAR_PT, color=INK, clip_on=False, zorder=6)

    ax.set_xlim(-.70, 1.70)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Imagined\nwalking", "Real\nwalking"], fontsize=PS.NOTE_PT)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_ylabel(r"Decoding performance ($\mathit{r}$)", fontsize=PS.BASE_PT, labelpad=2)
    PS.finish(ax)
    fig.savefig(PNG, dpi=PS.DPI, facecolor="white")
    plt.close(fig)
    print(f"panel               : {stars}  (paired t across sessions t={t:.2f}, P={p:.2e})")
    print(f"wrote {PNG}")


if __name__ == "__main__":
    main()
    render()
