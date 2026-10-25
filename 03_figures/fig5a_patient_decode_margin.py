#!/usr/bin/env python3
"""Figure 5 — Decoding imagined walking after spinal cord injury

SCI-patient per-session imagined-walking decoding, pre- and post-operative, with
bootstrap CIs over the no-EEG floor

Reads:
  decoder output in data/decoder_output/ (not distributed)

Writes:
  figures_out/fig5/fig5a_patient_decode_margin.png
  04_source_data/fig5a_patient_decode_margin_{source_data,metadata}.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import analysis_config as ec

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "decoder_output"
if ec.IS_DEFAULT:
    DEFAULT_POST_OPERATIVE_CSV = ec.path("p_margin_ci_post_operative")
    DEFAULT_PRE_OPERATIVE_CSV = ec.path("p_margin_ci_pre_operative")
    POST_OPERATIVE_SIDECAR_CSVS = {
        "pat02_sess08": ec.path("p_margin_ci_sess08"),
        "pat02_sess09": ec.path("p_margin_ci_sess09"),
    }
else:
    DEFAULT_POST_OPERATIVE_CSV = DEFAULT_PRE_OPERATIVE_CSV = ec.path("p_margin_ci")
    POST_OPERATIVE_SIDECAR_CSVS = {}
DEFAULT_OUT_PNG = ROOT / "figures_out" / "fig5" / "fig5a_patient_decode_margin.png"
DEFAULT_SRC = ROOT / "04_source_data" / "fig5" / "fig5a_patient_decode_margin_source_data.csv"
DEFAULT_META = ROOT / "04_source_data" / "fig5" / "fig5a_patient_decode_margin_metadata.csv"

BUILD_ID = ec.TAG
DECODE_TAG = "patientdump" if ec.IS_DEFAULT else ec.PAT_TAG
BASIS = ("  "
         "(shared_pca_basis)")
READOUT = "patient-trained 10-fold CV (representation transfer, not zero-shot)"
if ec.BUILD_ID in ec.DECODER_BUILDS:
    BASIS = "frozen all-80-healthy pooled shared PCA30 on causal complex-demodulation features (basis_hglobal_cdm_pca30), built on the causal triples"

POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
            "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06",
            "pat02_sess08", "pat02_sess09"]
PRE_OPERATIVE = ["pat01_sess01", "pat01_sess02", "pat02_sess01", "pat02_sess02"]
CONTROL = "pat01_sess01iw"
ANCHOR_POST_OPERATIVE = ec.anchor("pat_post_operative_session_mean")[0]
ANCHOR_PRE_OPERATIVE = ec.anchor("pat_pre_operative_session_mean")[0]
ANCHOR_TOL = 5e-6 if ec.IS_DEFAULT else ec.anchor("pat_post_operative_session_mean")[1]

if ec.IS_DEFAULT:
    FOLD_PARALLEL = [
        ec.path("p_summary_csv"),
        ec.path("p_summary_csv_sess08"),
        ec.path("p_summary_csv_sess09"),
    ]
else:
    FOLD_PARALLEL = [ec.path("p_summary_csv")]


def fold_level_session_means(paths) -> dict:
    """Session means of full-precision `rpt_calib` over the 10 CV folds x 6 joints (NaN ignored)."""
    vals: dict = {}
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing {p} — needed for the full-precision fold-level anchor.")
        for r in csv.DictReader(p.open(newline="", encoding="utf-8-sig")):
            try:
                v = float(r["rpt_calib"])
            except (ValueError, KeyError):
                continue
            if math.isfinite(v):
                vals.setdefault(r["session"], []).append(v)
    return {s: sum(v) / len(v) for s, v in vals.items()}

S_LABEL = {"pat01_sess01": "pre1", "pat01_sess02": "pre2",
           "pat01_sess04": "S1", "pat01_sess05": "S2", "pat01_sess06": "S3",
           "pat01_sess07": "S4", "pat01_sess08": "S5", "pat01_sess09": "S6",
           "pat02_sess01": "pre1", "pat02_sess02": "pre2",
           "pat02_sess03": "S1", "pat02_sess05": "S3", "pat02_sess06": "S4",
           "pat02_sess08": "S6", "pat02_sess09": "S7"}

BLUE, GREY = "#0072B2", "#9aa0a6"
REF, SHADE = "#525152", "#f0f0f0"
BAND_FC = "#d9d9d9"
PMARK = {"pat01": "o", "pat02": "s"}
SESS_RE = re.compile(r"(pat\d+)_sess(\d+)")

REST_BAND_CSV = ec.sd("fig2g_rest_null_band.csv")
REST_BAND_PANEL, REST_BAND_CONV = "All", "session_mean"


def load_rest_band() -> dict:
    if not REST_BAND_CSV.exists():
        raise SystemExit(f"Missing {REST_BAND_CSV} — run 03_figures/fig2g_trial_r_export.py first "
                         "(it derives the healthy rest-input null band).")
    for row in csv.DictReader(REST_BAND_CSV.open(newline="")):
        if row["panel"] == REST_BAND_PANEL and row["convention"] == REST_BAND_CONV:
            return {"lo": float(row["band_lo_p2p5"]), "hi": float(row["band_hi_p97p5"]),
                    "center": float(row["rest_center"]), "n_sessions": int(row["n_sessions"])}
    raise SystemExit(f"row panel={REST_BAND_PANEL}, convention={REST_BAND_CONV} not found in {REST_BAND_CSV}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fig.5 per-session patient decoding ()")
    p.add_argument("--post_operative-ci-csv", type=Path, default=DEFAULT_POST_OPERATIVE_CSV)
    p.add_argument("--pre_operative-ci-csv", type=Path, default=DEFAULT_PRE_OPERATIVE_CSV)
    p.add_argument("--post_operative-sidecar-csv", type=Path, nargs="*",
                   default=list(POST_OPERATIVE_SIDECAR_CSVS.values()),
                   help="single-session margin/CI files merged into the post-operative condition "
                        "(default: the pat02_sess08 + pat02_sess09 sidecars)")
    p.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    p.add_argument("--src-csv", type=Path, default=DEFAULT_SRC)
    p.add_argument("--meta-csv", type=Path, default=DEFAULT_META)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load(path: Path, cohort: str, expect: list[str],
         sidecars: list[Path] | None = None) -> pd.DataFrame:
    """Read one cohort CSV (+ optional single-session sidecars) and check membership exactly."""
    if not path.exists():
        raise SystemExit(f"Missing {path} — the margin/CI compute step must run first.")
    parts = [pd.read_csv(path)]
    for side in sidecars or []:
        if not side.exists():
            raise SystemExit(f"Missing sidecar {side} — its single-session margin/CI must be "
                             "computed on the compute server before this cohort can be rendered.")
        d = pd.read_csv(side)
        dup = set(d.session) & set(parts[0].session)
        if dup:
            raise SystemExit(f"{side.name} duplicates {sorted(dup)} already in {path.name} — "
                             "merging would double-count them in the cohort mean")
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d[d.session != CONTROL].copy()
    if not ec.IS_DEFAULT:
        if sorted(d.session) != sorted(POST_OPERATIVE + PRE_OPERATIVE):
            raise SystemExit(f"{path.name} must hold exactly pre_operative + post_operative (15 sessions), got: "
                             f"{sorted(d.session)}")
        d = d[d.session.isin(expect)].copy()
    got = sorted(d.session)
    if got != sorted(expect):
        raise SystemExit(f"{cohort} membership mismatch in {path.name} (+{len(sidecars or [])} "
                         f"sidecars): {got} != {sorted(expect)}")
    d["cohort"] = cohort
    d["ci_excludes_zero"] = d.ci_excludes_zero.astype(str).str.strip().isin(["True", "TRUE", "1"])
    return d


def main() -> None:
    args = parse_args()
    post = load(args.post_operative_ci_csv, "post_operative", POST_OPERATIVE, args.post_operative_sidecar_csv)
    pre = load(args.pre_operative_ci_csv, "pre_operative", PRE_OPERATIVE)
    df = pd.concat([pre, post], ignore_index=True)
    df["pat"] = df.session.str.extract(r"(pat\d+)")
    df["snum"] = df.session.str.extract(r"sess(\d+)").astype(int)
    df["s_label"] = df.session.map(S_LABEL)
    if df.s_label.isna().any():
        raise SystemExit(f"unlabelled sessions: {list(df.session[df.s_label.isna()])}")

    fold_sess_mean = fold_level_session_means(FOLD_PARALLEL)
    missing = [s for s in list(post.session) + list(pre.session) if s not in fold_sess_mean]
    if missing:
        raise SystemExit(f"fold tables do not cover {missing} — cannot compute the anchor.")
    m_post = sum(fold_sess_mean[s] for s in post.session) / len(post)
    m_pre = sum(fold_sess_mean[s] for s in pre.session) / len(pre)
    for name, got, want in (("post_operative", m_post, ANCHOR_POST_OPERATIVE), ("pre_operative", m_pre, ANCHOR_PRE_OPERATIVE)):
        if want is None:
            print(f"[anchor] {name} = {got:.6f} - NO recorded {ec.BUILD_ID} anchor yet, not checked")
            continue
        if abs(got - want) >= ANCHOR_TOL:
            raise SystemExit(f"ANCHOR MISMATCH {name}: computed {got:.6f} vs recorded {ec.BUILD_ID} "
                             f"{want:.6f} (|d|={abs(got - want):.2e} >= {ANCHOR_TOL:g}). "
                             f"Inputs are not the {ec.BUILD_ID} build.")

    df["blk"] = np.where(df.cohort == "pre_operative", 0, 1)
    df = df.sort_values(["pat", "blk", "snum"]).reset_index(drop=True)
    GAP_BLOCK, GAP_PATIENT = 0.55, 1.30
    xs, x, prev = [], 0.0, None
    for _, r in df.iterrows():
        key = (r.pat, r.blk)
        if prev is not None:
            x += 1.0 + (GAP_PATIENT if key[0] != prev[0] else (GAP_BLOCK if key[1] != prev[1] else 0.0))
        prev = key
        xs.append(x)
    df["x"] = xs

    plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.linewidth": 2.2,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic"})
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    Y_LO, Y_HI = -0.13, 0.80

    blocks = []
    for (pat, blk), g in df.groupby(["pat", "blk"], sort=False):
        blocks.append((pat, blk, g.x.min(), g.x.max()))
    for pat, blk, lo, hi in blocks:
        if blk == 0:
            ax.add_patch(Rectangle((lo - 0.5, Y_LO), (hi - lo) + 1.0, Y_HI - Y_LO,
                                   facecolor=SHADE, edgecolor="none", zorder=0))
        ax.text((lo + hi) / 2, Y_HI - 0.015, "pre-operative" if blk == 0 else "post-operative",
                ha="center", va="top", fontsize=11, color="#6b6b6b")
    for pat in df.pat.unique():
        g = df[df.pat == pat]
        pre_hi, post_lo = g[g.blk == 0].x.max(), g[g.blk == 1].x.min()
        ax.plot([(pre_hi + post_lo) / 2] * 2, [Y_LO, Y_HI - 0.075], color="#b8b8b8",
                lw=1.2, ls=(0, (4, 3)), zorder=1)

    band = load_rest_band()
    ax.axhspan(band["lo"], band["hi"], facecolor=BAND_FC, edgecolor="none", alpha=0.55, zorder=0.6)
    ax.text(df.x.max() + 0.95, -0.040, "healthy rest-input\nnull (combined)",
            ha="left", va="center", fontsize=9.5, color="#6b6b6b")

    ax.axhline(0.0, color=GREY, lw=1.2, zorder=1)

    pat_means = {}
    for pat in df.pat.unique():
        g = df[(df.pat == pat) & (df.blk == 1)]
        mp = float(g.r_all.mean())
        pat_means["P" + pat[-1]] = mp
        ax.plot([g.x.min() - 0.42, g.x.max() + 0.42], [mp] * 2, color=REF, lw=1.3,
                ls=(0, (1.6, 2.2)), zorder=2)
        ax.text(g.x.max() + 0.52, mp, f"mean {mp:.2f}", ha="left", va="center",
                fontsize=10, color=REF)

    src_rows = []
    for _, r in df.iterrows():
        col = BLUE if r.ci_excludes_zero else GREY
        lo, hi = r.floor + r.margin_lo, r.floor + r.margin_hi
        ax.plot([r.x, r.x], [r.floor, r.r_all], color=col, lw=3.4, solid_capstyle="butt", zorder=3)
        ax.plot([r.x, r.x], [lo, hi], color=col, lw=1.6, solid_capstyle="butt", zorder=4)
        for y in (lo, hi):
            ax.plot([r.x - 0.13, r.x + 0.13], [y, y], color=col, lw=1.6, zorder=4)
        ax.plot(r.x, r.r_all, marker=PMARK[r.pat], ms=9.5,
                mfc=col if r.ci_excludes_zero else "white",
                mec=col, mew=2.0, lw=0, zorder=5)
        src_rows.append({"patient": "P" + r.pat[-1], "cohort": r.cohort, "session": r.session,
                         "s_label": r.s_label, "r_all": round(float(r.r_all), 6),
                         "zero_all": round(float(r.zero_all), 6),
                         "circ_all": round(float(r.circ_all), 6),
                         "floor": round(float(r.floor), 6), "margin": round(float(r.margin), 6),
                         "margin_lo": round(float(r.margin_lo), 6),
                         "margin_hi": round(float(r.margin_hi), 6),
                         "ci_excludes_zero": bool(r.ci_excludes_zero)})

    ax.set_xlim(df.x.min() - 1.0, df.x.max() + 4.2)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(df.x.to_numpy(), df.s_label.tolist())
    ax.set_ylabel("Decoding accuracy ($r$)", fontsize=16, labelpad=6)
    ax.tick_params(axis="y", direction="out", length=5, width=2.2, labelsize=14)
    ax.tick_params(axis="x", length=0, pad=4, labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)

    for pat in df.pat.unique():
        g = df[df.pat == pat]
        ax.text((g.x.min() + g.x.max()) / 2, -0.10, "P" + pat[-1], ha="center", va="top",
                fontsize=16, fontweight="bold", color="#222222",
                transform=ax.get_xaxis_transform(), clip_on=False)

    h = [Line2D([0], [0], marker="o", color=BLUE, mfc=BLUE, mec=BLUE, mew=2.0, lw=0, ms=9.5,
                label="CI excludes floor"),
         Line2D([0], [0], marker="o", color=GREY, mfc="white", mec=GREY, mew=2.0, lw=0, ms=9.5,
                label="CI includes floor"),
         Line2D([0], [0], marker="o", color="none", mfc="#666666", mec="#666666", mew=2.0, lw=0,
                ms=8.5, label="P1"),
         Line2D([0], [0], marker="s", color="none", mfc="#666666", mec="#666666", mew=2.0, lw=0,
                ms=8.5, label="P2")]
    ax.legend(handles=h, loc="upper right", frameon=False, fontsize=11, handlelength=1.0,
              borderaxespad=0.3, labelspacing=0.35, handletextpad=0.5)

    fig.subplots_adjust(left=0.085, right=0.995, top=0.985, bottom=0.155)
    args.out_png, args.src_csv, args.meta_csv = (ec.scoped(args.out_png), ec.scoped(args.src_csv),
                                                 ec.scoped(args.meta_csv))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=args.dpi, facecolor="white")
    plt.close(fig)

    args.src_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.src_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(src_rows[0].keys()))
        w.writeheader()
        w.writerows(src_rows)

    med_post, med_pre = float(np.median(post.margin)), float(np.median(pre.margin))
    n_post, n_pre = int(post.ci_excludes_zero.sum()), int(pre.ci_excludes_zero.sum())
    with args.meta_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in [
            ("figure", "Fig.5 — per-session SCI-patient imagined-walking decoding, pre- and post-operative"),
            ("decode_tag", DECODE_TAG),
            ("basis", BASIS),
            ("readout", READOUT),
            ("scope", "n=2 SCI patients, case-level, descriptive; per-session margin +/- bootstrap CI, NO p-value, no population claim"),
            ("metric", "rpt_calib, All = mean over the 6 Walker-X joints, per session (10 CV folds x 6 joints)"),
            ("post_operative_set", "pat01 sess04-09 + pat02 sess03/05/06/08/09 (11 sessions)"),
            ("pre_operative_set", "pat01 sess01/02 + pat02 sess01/02 (4 sessions)"),
            ("post_operative_mean_r", round(m_post, 6)),
            ("pre_operative_mean_r", round(m_pre, 6)),
            ("median_margin_post_operative", round(med_post, 6)),
            ("median_margin_pre_operative", round(med_pre, 6)),
            ("n_ci_excludes_zero_post_operative", f"{n_post}/{len(post)}"),
            ("n_ci_excludes_zero_pre_operative", f"{n_pre}/{len(pre)}"),
            ("post_operative_mean_r_P1", round(pat_means.get("P1", float("nan")), 6)),
            ("post_operative_mean_r_P2", round(pat_means.get("P2", float("nan")), 6)),
            ("healthy_rest_band", f"grey background band = healthy rest-input null, panel {REST_BAND_PANEL} (combined), convention {REST_BAND_CONV}: 2.5-97.5 pct of the {band['n_sessions']} healthy session values = [{band['lo']:.6f}, {band['hi']:.6f}], center {band['center']:.6f}"),
            ("patient_marker_convention", "P1 = circle, P2 = square (same shape convention as fig6_rsa_quantification_forest.py); fill still encodes the CI criterion"),
            ("margin_def", "r_all - floor; floor = max(zero_eeg, circshift) null"),
            ("marker_semantics", "DATA-DRIVEN from ci_excludes_zero: filled blue #0072B2 = CI excludes the no-EEG floor; open grey #9aa0a6 = CI includes it. Nothing hardcoded per session."),
            ("input_post_operative_csv", args.post_operative_ci_csv.relative_to(ROOT).as_posix()),
            ("input_pre_operative_csv", args.pre_operative_ci_csv.relative_to(ROOT).as_posix()),
            ("input_post_operative_sidecar_csvs", "; ".join(pp.relative_to(ROOT).as_posix() for pp in args.post_operative_sidecar_csv) or "none"),
            ("output_png", ec.scoped(DEFAULT_OUT_PNG).relative_to(ROOT).as_posix()),
        ]:
            w.writerow([k, v])

    print(f"post_operative n={len(post)} mean_r={m_post:.6f} median_margin={med_post:.6f} CI-excl-floor {n_post}/{len(post)}")
    print(f"pre_operative  n={len(pre)} mean_r={m_pre:.6f} median_margin={med_pre:.6f} CI-excl-floor {n_pre}/{len(pre)}")
    print(f"wrote {args.out_png}")
    print(f"wrote {args.src_csv}")
    print(f"wrote {args.meta_csv}")


if __name__ == "__main__":
    main()
