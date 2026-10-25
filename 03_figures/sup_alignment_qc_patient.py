"""Supplementary item

figure: EEG-goniometer synchronization (alignment) QC — PATIENT condition

Reads:
  preprocessing output in $GAIT_PREP_OUTPUT (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""
PREP_OUTPUT = __import__("pathlib").Path(__import__("os").environ.get("GAIT_PREP_OUTPUT", "preprocessing_output"))
_release_rel = lambda p, r: p.relative_to(r) if p.is_relative_to(r) else p

from pathlib import Path
from datetime import date
import csv
import json
import sys
import numpy as np
from sup_panel_export import PANEL_MODE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import panel_style as PS


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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_palette import REST, REST_EDGE

ROOT = Path(__file__).resolve().parents[1]
PREP_PATIENT = PREP_OUTPUT / "patient" / "result"
OUT_PNG = ROOT / "figures_out" / "supplementary" / "sup_alignment_qc_patient.png"
SD = ROOT / "04_source_data" / "supplementary" / "sup_alignment_qc_patient_source_data.csv"
MD = ROOT / "04_source_data" / "supplementary" / "sup_alignment_qc_patient_metadata.csv"
FS = 250.0

BUILD_ID = "pre_operative;post_operative"
BUILD_ID_NOTE = ("prep_patient frozen patient prep (no decoder; per-session alignment QC). "
            "The tokens name the two patient cohorts the QC covers.")
COHORT = "pre_operative + post_operative (15 retained patient sessions)"

PRE_OPERATIVE = ["pat01_sess01", "pat01_sess02", "pat02_sess01", "pat02_sess02"]
POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
            "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
            "pat02_sess09"]
CONDITION = {s: "pre_operative" for s in PRE_OPERATIVE} | {s: "post_operative" for s in POST_OPERATIVE}
COHORT_SESSIONS = PRE_OPERATIVE + POST_OPERATIVE

EXCLUDED = {"pat01_sess03", "pat02_sess04", "pat02_sess07"}
EXCLUSION_REASON = {
    "pat01_sess03": "recording hardware failure",
    "pat02_sess04": "recording hardware failure",
    "pat02_sess07": "recording hardware failure",
}

BUILD = {s: "qc" for s in COHORT_SESSIONS}
BUILD.update({"pat01_sess01": "qc_two_marker", "pat01_sess02": "qc_two_marker",
              "pat01_sess04": "qc_connector_fix", "pat02_sess01": "qc_connector_fix",
              "pat02_sess02": "qc_connector_fix"})
PHASE_A = {b: PREP_PATIENT / b / "phase_a_summary.csv" for b in ("qc", "qc_connector_fix", "qc_two_marker")}


def matched_residuals(d):
    """Per-TTL residual (ms) for the matched TTLs only, with their goni-file index."""
    res = d.get("per_ttl_residual_ms")
    matched = d.get("per_ttl_matched")
    if res is None or matched is None:
        return None, None
    gfile = d.get("goni_file_per_ttl")
    out_r, out_f = [], []
    for i, (r, m) in enumerate(zip(res, matched)):
        if not m or r is None:
            continue
        out_r.append(float(r))
        out_f.append(int(gfile[i]) if (gfile and gfile[i] is not None) else 1)
    if not out_r:
        return None, None
    return np.array(out_r), np.array(out_f)


def deoffset_jitter(resid, fidx):
    """De-offset each goni file (subtract that file's median residual) -> |jitter| ms."""
    jit = np.full(resid.size, np.nan)
    for fv in np.unique(fidx):
        sel = fidx == fv
        jit[sel] = resid[sel] - np.median(resid[sel])
    return np.abs(jit)


def registry_labels():
    """Every label the QC snapshots know about, so a new registry row cannot go unnoticed."""
    labs = set()
    for path in PHASE_A.values():
        with path.open(newline="") as f:
            labs.update(r["label"] for r in csv.DictReader(f))
    return labs


def main():
    unaccounted = registry_labels() - set(COHORT_SESSIONS) - EXCLUDED
    if unaccounted:
        raise SystemExit(f"registry labels not in cohort or exclusion list: {sorted(unaccounted)}")

    rows, missing_input = [], []
    for lab in COHORT_SESSIONS:
        build = BUILD[lab]
        jp = PREP_PATIENT / build / "align_log" / f"{lab}.json"
        if not jp.exists():
            missing_input.append(f"{lab}(no align log in {build})")
            continue
        d = json.loads(jp.read_text())
        verdict = d.get("auto_verdict", "")
        sct = d.get("special_case_type", "") or ""
        method = d.get("match_method") or d.get("alignment_algorithm", "")
        n_matched = d.get("n_matched")
        is_multi = sct == "multi_goni_file"

        resid, fidx = matched_residuals(d)
        if resid is None:
            rows.append(dict(session=lab, condition=CONDITION[lab], build=build, verdict=verdict, method=method,
                             n_matched=n_matched, is_multi=False,
                             nominal_med=np.nan, median=np.nan, p95=np.nan, mx=np.nan))
            print(f"{lab} [{build}]: verdict={verdict} method={method} -> no per-TTL residual "
                  f"(legacy anchors)")
            continue

        nominal_med = float(np.median(np.abs(resid)))
        if is_multi:
            aj = deoffset_jitter(resid, fidx)
            tag = "de-offset jitter (multi_goni per-file constant offset removed)"
        else:
            aj = np.abs(resid)
            tag = "synchronization residual"
        med, p95, mx = float(np.median(aj)), float(np.percentile(aj, 95)), float(aj.max())
        rows.append(dict(session=lab, condition=CONDITION[lab], build=build, verdict=verdict, method=method,
                         n_matched=n_matched, is_multi=is_multi, nominal_med=nominal_med,
                         median=med, p95=p95, mx=mx))
        extra = f" | nominal median {nominal_med:,.0f} ms (per-file offset)" if is_multi else ""
        print(f"{lab} [{build}]: verdict={verdict} method={method} n={n_matched} {tag}: "
              f"median={med:.1f} p95={p95:.1f} max={mx:.1f} ms{extra}")

    SD.parent.mkdir(parents=True, exist_ok=True)
    with SD.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "cohort_condition", "align_build", "verdict", "match_method", "n_matched",
                    "multi_goni_special_case", "nominal_median_abs_residual_ms",
                    "sync_median_abs_residual_ms", "sync_p95_abs_residual_ms",
                    "sync_max_abs_residual_ms"])
        for r in rows:
            fmt = lambda v: "" if (v is None or not np.isfinite(v)) else f"{v:.4f}"
            w.writerow([r["session"], r["condition"], r["build"], r["verdict"], r["method"],
                        r["n_matched"], "yes" if r["is_multi"] else "no",
                        fmt(r["nominal_med"]), fmt(r["median"]), fmt(r["p95"]), fmt(r["mx"])])

    plotted = [r for r in rows if np.isfinite(r["median"])]
    plotted.sort(key=lambda r: (r["is_multi"], r["session"]))
    labels = [r["session"].replace("pat", "P").replace("_sess", " s") for r in plotted]
    x = np.arange(len(plotted))
    med = np.array([r["median"] for r in plotted])
    p95 = np.array([r["p95"] for r in plotted])
    mx = np.array([r["mx"] for r in plotted])
    multi = np.array([r["is_multi"] for r in plotted])

    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    if PANEL_MODE:
        import si_panel_style as SI
        SI.apply(plt)
        fig, ax = SI.panel(plt, "full", 52.0, left_mm=13.0, bottom_mm=14.0)
    else:
        fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for xi, m_, p_, x_, isml in zip(x, med, p95, mx, multi):
        ax.vlines(xi, m_, x_, color="#cfcfcf", lw=2.2 if PANEL_MODE else 3.0, zorder=1)
        ax.scatter(xi, p_, s=14 if PANEL_MODE else 16, marker="_", color="#1e6fd9",
                   zorder=3, lw=1.0 if PANEL_MODE else 1.3)
    ax.scatter(x[~multi], med[~multi], s=22, color=REST, edgecolor=REST_EDGE, linewidth=0.9,
               zorder=4, label="median |residual|")
    ax.scatter(x[multi], med[multi], s=26, color=REST, edgecolor="#d1495b", linewidth=1.3,
               zorder=4, label="multi-goni (per-file offset removed)")
    ax.axhline(1000.0 / FS, color="#aaa", lw=SI.LW_REF if PANEL_MODE else 0.7, ls="--",
               zorder=0)
    if not PANEL_MODE:
        ax.text(len(plotted) - 0.5, 1000.0 / FS, "1 frame (4 ms)", fontsize=6, color="#888",
            va="bottom", ha="right")
    ax.scatter([], [], s=16, marker="_", color="#1e6fd9", lw=1.3, label="95th pct |residual|")
    ax.scatter([], [], s=30, marker="|", color="#cfcfcf", label="max |residual|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=SI.TICK_PT if PANEL_MODE else 6.5)
    ax.set_ylabel("EEG-goni alignment residual (ms)")
    ax.set_xlim(-0.6, len(plotted) - 0.4)
    ax.set_ylim(0, max(mx.max(), 1000.0 / FS) * 1.18)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    if not PANEL_MODE:
        ax.legend(loc="upper left", fontsize=6, frameon=False, handletextpad=0.4,
                  borderaxespad=0.2)
    n_multi = int(multi.sum())
    if not PANEL_MODE:
        ax.text(0.98, 0.98, f"n = {len(plotted)} sessions ({n_multi} multi-goni)",
                transform=ax.transAxes, va="top", ha="right", fontsize=6.5, color="#5f5e5a")
    if PANEL_MODE:
        SI.finish(ax)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    if PANEL_MODE:
        _panel_spec_check(fig, [ax])
        def _rng(a, f="{:.1f}"):
            return f.format(a.min()) + "–" + f.format(a.max())
        lines = [
            "EEG-goniometer alignment residual over the matched trigger pairs, per session:",
            f"  single goniometer file ($n$ = {int((~multi).sum())} sessions): "
            f"median {_rng(med[~multi])} ms, 95th percentile {_rng(p95[~multi])} ms, "
            f"maximum {_rng(mx[~multi])} ms",
            f"  restarted goniometer, per-file constant offset removed "
            f"($n$ = {n_multi} sessions): median {_rng(med[multi])} ms, "
            f"95th percentile {_rng(p95[multi])} ms, maximum {_rng(mx[multi])} ms",
            f"  {sum(1 for r in rows if not np.isfinite(r['median']))} further sessions used "
            f"legacy anchors and log no per-trigger residual; they are not plotted.",
            "Dashed line = one goniometer frame (4 ms) at 250 Hz.",
        ]
        print("panel mode: no Supplementary figure panel for the patient alignment QC - "
              "nothing exported; statistics:")
        for _ln in lines:
            print("   ", _ln)
    else:
        fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    legacy = [r["session"] for r in rows if not np.isfinite(r["median"])]
    multi_labels = [r["session"] for r in rows if r["is_multi"]]

    with MD.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["build_id", BUILD_ID])
        w.writerow(["build_id_note", BUILD_ID_NOTE])
        w.writerow(["cohort", COHORT])
        w.writerow
        w.writerow(["sessions_included", ";".join(r["session"] for r in rows)])
        w.writerow(["n_sessions_included", len(rows)])
        w.writerow(["sessions_plotted", ";".join(r["session"] for r in plotted)])
        w.writerow(["n_sessions_plotted", len(plotted)])
        w.writerow(["sessions_legacy_anchor_no_residual", ";".join(legacy) if legacy else "none"])
        w.writerow(["sessions_multi_goni_file", ";".join(multi_labels)])
        w.writerow(["sessions_excluded_by_cohort",
                    ";".join(f"{s}({EXCLUSION_REASON[s]})" for s in sorted(EXCLUDED))])
        w.writerow(["sessions_missing_input", ";".join(missing_input) if missing_input else "none"])
        w.writerow(["single_goni_median_abs_residual_ms", f"{np.median(med[~multi]):.2f}"])
        w.writerow(["single_goni_worst_max_ms", f"{mx[~multi].max():.2f}"])
        w.writerow(["multi_goni_deoffset_median_ms", f"{np.median(med[multi]):.2f}"])
        w.writerow(["multi_goni_worst_max_ms", f"{mx[multi].max():.2f}"])
        w.writerow(["inputs", ";".join(
            f"01_preprocessing/patient/result/{b}/align_log/" for b in ("qc", "qc_connector_fix", "qc_two_marker"))])
        w.writerow(["renderer", "03_figures/sup_alignment_qc_patient.py"])
        w.writerow(["output_png", _release_rel(OUT_PNG, ROOT).as_posix()])
        w.writerow(["paper_location", "Supplementary Note 11, text only (patient synchronization sentence; no Supplementary figure panel)"])
        w.writerow(["generated", date.today().isoformat()])

    print(f"\ncohort {len(rows)}/{len(COHORT_SESSIONS)} sessions | {len(plotted)} plotted with IOI "
          f"residual | {len(legacy)} legacy-anchor (no residual): {legacy} | "
          f"{len(multi_labels)} multi_goni: {multi_labels}")
    print(f"missing input: {missing_input or 'none'}")
    print(f"single-goni sessions: median|res| {np.median(med[~multi]):.1f} ms | "
          f"worst max {mx[~multi].max():.1f} ms")
    print(f"multi_goni de-offset jitter: median {np.median(med[multi]):.1f} ms | "
          f"worst max {mx[multi].max():.1f} ms (matches PASS -> per-file offset, not desync)")
    print("wrote", OUT_PNG)
    print("wrote", SD)
    print("wrote", MD)


if __name__ == "__main__":
    main()
