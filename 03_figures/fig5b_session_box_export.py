"""Figure 5 — Decoding imagined walking after spinal cord injury

per-session patient decoding, drawn as Fig. 2c-style boxes (export only)

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import analysis_config as ec

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data" / "fig5"
RESULT = ROOT / "data" / "decoder_output"
if ec.IS_DEFAULT:
    SIDECARS = {
        "pat02_sess08": (ec.path("p_trial_r_long_sess08"), ec.path("p_margin_ci_sess08")),
        "pat02_sess09": (ec.path("p_trial_r_long_sess09"), ec.path("p_margin_ci_sess09")),
    }
    FOLD_PARALLEL = [
        ec.path("p_summary_csv"),
        ec.path("p_summary_csv_sess08"),
        ec.path("p_summary_csv_sess09"),
    ]
else:
    SIDECARS = {}
    FOLD_PARALLEL = [ec.path("p_summary_csv")]

BUILD_ID = ec.TAG
DECODE_TAG = ("patientdump (+ patientdump_sess08 for S6, "
              "+ patientdump_sess09 for S7)") if ec.IS_DEFAULT else ec.PAT_TAG

JOINTS_ALL = ["LHipW X", "RHipW X", "LKneW X", "RKneW X", "LAnkW X", "RAnkW X"]

from patient_session_labels import SESSION_LABEL as _SL

PLOT_ORDER = [
    (sess, _SL[sess][0], coh, _SL[sess][1]) for sess, coh in [
        ("pat01_sess01", "pre_operative"), ("pat01_sess02", "pre_operative"),
        ("pat01_sess04", "post_operative"), ("pat01_sess05", "post_operative"), ("pat01_sess06", "post_operative"),
        ("pat01_sess07", "post_operative"), ("pat01_sess08", "post_operative"), ("pat01_sess09", "post_operative"),
        ("pat02_sess01", "pre_operative"), ("pat02_sess02", "pre_operative"),
        ("pat02_sess03", "post_operative"), ("pat02_sess05", "post_operative"), ("pat02_sess06", "post_operative"),
        ("pat02_sess08", "post_operative"), ("pat02_sess09", "post_operative"),
    ]
]


def quantile(vals, q):
    """Linear-interpolation quantile, numpy 'linear' convention (same helper as fig2g)."""
    v = sorted(vals)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def fold_level_session_means(paths) -> dict[str, float]:
    """Session means of full-precision `rpt_calib` over the 10 CV folds x 6 joints (NaN ignored)."""
    import math
    vals = defaultdict(list)
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing {p} - needed for the full-precision fold-level anchor.")
        for r in csv.DictReader(p.open(newline="", encoding="utf-8-sig")):
            try:
                v = float(r["rpt_calib"])
            except (ValueError, KeyError):
                continue
            if math.isfinite(v):
                vals[r["session"]].append(v)
    return {s: sum(v) / len(v) for s, v in vals.items()}


def aggregate_sidecar(path: Path, session: str) -> list[float]:
    """Per-trial All = mean of the 6 Walker-X joints, for ONE single-session sidecar dump."""
    per: dict[int, dict[str, float | None]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    seen = {r["session"] for r in rows}
    if seen != {session}:
        raise SystemExit(f"{path.name} must hold exactly {session}, got: {sorted(seen)}")
    for row in rows:
        if row["model"] != "calib":
            continue
        try:
            v = float(row["rpt_r"])
            if v != v:
                v = None
        except ValueError:
            v = None
        per[int(row["trial_idx"])][row["joint"]] = v
    out = []
    for jr in per.values():
        vals = [jr.get(j) for j in JOINTS_ALL]
        vals = [v for v in vals if v is not None]
        if vals:
            out.append(mean(vals))
    return out


def main() -> None:
    trials: dict[str, list[float]] = defaultdict(list)
    with open(ec.sd("fig5g_trial_r_long.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            v = row["All"]
            if v not in ("", "NA", "NaN"):
                trials[row["session"]].append(float(v))
    for sess, (trial_src, _) in SIDECARS.items():
        trials[sess] = aggregate_sidecar(trial_src, sess)

    anchor: dict[str, dict] = {}
    with open(ec.sd("fig5a_patient_decode_margin_source_data.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            anchor[row["session"]] = row
    for sess, (_, margin_src) in SIDECARS.items():
        with open(margin_src, newline="") as fh:
            side_rows = list(csv.DictReader(fh))
        seen = {r["session"] for r in side_rows}
        if seen != {sess}:
            raise SystemExit(f"{margin_src.name} must hold exactly {sess}, got: {sorted(seen)}")
        for row in side_rows:
            anchor[row["session"]] = row

    band = None
    with open(ec.sd("fig2g_rest_null_band.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            if row["panel"] == "All" and row["convention"] == "session_median":
                band = row
    if band is None:
        raise SystemExit("rest-null band row All/session_median not found")
    band_lo, band_hi = float(band["band_lo_p2p5"]), float(band["band_hi_p97p5"])
    band_center = float(band["rest_center"])

    rows = []
    for i, (sess, pat, cohort, label) in enumerate(PLOT_ORDER, start=1):
        v = trials[sess]
        if not v:
            raise SystemExit(f"no trials for {sess}")
        a = anchor[sess]
        med = median(v)
        rows.append(
            {
                "order": i,
                "patient": pat,
                "cohort": cohort,
                "session": sess,
                "x_label": label,
                "n_trials": len(v),
                "p10": quantile(v, 0.10),
                "q25": quantile(v, 0.25),
                "median": med,
                "q75": quantile(v, 0.75),
                "p90": quantile(v, 0.90),
                "mean": mean(v),
                "sd": pstdev(v) if len(v) > 1 else 0.0,
                "min": min(v),
                "max": max(v),
                "median_above_rest_band": med > band_hi,
                "r_all_fold": a["r_all"],
                "ci_lo_fold": a["margin_lo"],
                "ci_hi_fold": a["margin_hi"],
                "ci_excludes_zero_fold": a["ci_excludes_zero"],
            }
        )

    fields = list(rows[0].keys())
    with open(ec.sd("fig5b_session_box_source_data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    header = [f"{pat} {label}" for _, pat, _, label in PLOT_ORDER]
    cols = [trials[s] for s, _, _, _ in PLOT_ORDER]
    nmax = max(len(c) for c in cols)
    with open(ec.sd("fig5b_trial_matrix.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(nmax):
            w.writerow([f"{c[i]:.6f}" if i < len(c) else "" for c in cols])

    post = [r for r in rows if r["cohort"] == "post_operative"]
    n_above = sum(r["median_above_rest_band"] is True for r in rows)
    n_above_post = sum(r["median_above_rest_band"] is True for r in post)
    fold_sess_mean = fold_level_session_means(FOLD_PARALLEL)
    missing = [r["session"] for r in post if r["session"] not in fold_sess_mean]
    if missing:
        raise SystemExit(f"no fold-level rows for {missing} in the parallel tables")
    post_operative_anchor = mean(fold_sess_mean[r["session"]] for r in post)
    meta = [
        ("figure", "Fig.5b - per-session patient imagined-walking decoding, Fig.2c-style boxes"),
        ("decode_tag", DECODE_TAG),
        ("readout", "patient-trained 10-fold CV (representation transfer, not zero-shot)"),
        ("scope", "n=2 SCI patients, case-level, descriptive; no population claim, no pre/post test"),
        ("metric", "rpt_r, model=calib, All = mean of the 6 Walker-X joints, one value per held-out trial"),
        ("plot_statistic", "box = median + IQR; whiskers = 10th-90th percentile (identical to Fig.2c)"),
        ("cohort", "pre_operative + post_operative (P1 6 + P2 5 post-operative sessions)"),
        ("pre_operative_set", "pat01 sess01/02 + pat02 sess01/02 (4 sessions)"),
        ("n_trials_total", sum(r["n_trials"] for r in rows)),
        ("n_sessions", len(rows)),
        ("column_order", "P1 pre -> P1 post -> P2 pre -> P2 post, by session number; NOT ranked by performance"),
        ("rest_band_convention",
         "session_median (Fig.2c's row) - NOT session_mean. The box is centred on the session "
         "median, so the null must be the median-convention band."),
        ("rest_band_all", f"[{band_lo:.6f}, {band_hi:.6f}], centre {band_center:.6f}"),
        ("rest_band_semantics",
         "NULL reference (the frozen decoder fed resting EEG), NOT a healthy performance benchmark. "
         "No cross-cohort performance comparison is drawn or claimed."),
        ("n_session_medians_above_band",
         f"{n_above}/{len(rows)} all sessions; {n_above_post}/{len(post)} post-operative"),
        ("output_csv", "fig5b_session_box_source_data.csv; fig5b_trial_matrix.csv"),
        ("rendered_by", "03_figures/fig5b_session_box.py -> figures_out/fig5/fig5b_session_box.png; the same CSVs are "
                        "the Prism inputs, as for Fig.2c"),
    ]
    with open(ec.sd("fig5b_session_box_metadata.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value"])
        w.writerows(meta)

    print(f"{len(rows)} sessions = pre_operative + post_operative; trials {sum(r['n_trials'] for r in rows)}")
    print(f"post_operative fold-level anchor {post_operative_anchor:.6f}  "
          "")
    print(f"rest band (All, session_median) [{band_lo:.6f}, {band_hi:.6f}]")
    print(f"session medians above band: {n_above}/{len(rows)} "
          f"(post-operative {n_above_post}/{len(post)})")
    for r in rows:
        print(f"  {r['patient']} {r['x_label']:5s} n={r['n_trials']:3d} "
              f"med={r['median']:.3f} IQR[{r['q25']:.3f},{r['q75']:.3f}] "
              f"p10-90[{r['p10']:.3f},{r['p90']:.3f}] above_band={r['median_above_rest_band']}")


if __name__ == "__main__":
    main()
