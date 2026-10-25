"""Figure 4 — Shared locomotor representation across imagined and real walking

Fig. 4b / Fig. 4c - per-session IW vs RW decoding r in the TRIAL-MEDIAN convention

Reads:
  decoder output in data/decoder_output/ (not distributed)

Writes:
  04_source_data/fig4/fig4_trialmedian_source_data.csv   one row per session

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy import stats

import analysis_config as ec

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data/decoder_output"
IW_SRC = ec.path("h_iw_trial_r_long")
RW_SRC = ec.path("h_rw_trial_r_long")
OUTDIR = ROOT / "04_source_data" / "fig4"

_ERA_RE = re.compile("(" + re.escape(ec.TAG) + r"|v\d+_\d+(?:)?)[_/]")
BUILD_ID = _ERA_RE.search(ec.rel("h_iw_trial_r_long")).group(1)
assert _ERA_RE.search(ec.rel("h_rw_trial_r_long")).group(1) == BUILD_ID, "IW/RW build_id tags disagree"
ec.check_tag(BUILD_ID, "decoder", "IW/RW trial-r build_id")



def session_mean_convention(task: str) -> float:
    """sessions of the session mean of the finite rpt_calib over its fold x joint rows."""
    acc = defaultdict(list)
    src = ec.path(f"h_{task}_summary_csv")
    for row in csv.DictReader(src.open(newline="")):
        v = float(row["rpt_calib"]) if row["rpt_calib"] not in ("", "NaN") else float("nan")
        if np.isfinite(v):
            acc[row["session"]].append(v)
    assert len(acc) == 80, src
    m = mean(mean(v) for v in acc.values())
    if task == "iw":
        ec.check_anchor("iw_session_mean", m, "IW session-mean rpt_calib")
    return m


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nanmean(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def per_session_trial_values(src: Path):
    """-> (dict session -> list of per-trial All values, sorted joint names found)."""
    per = defaultdict(dict)
    for row in csv.DictReader(src.open(newline="")):
        if row["model"] != "calib":
            continue
        try:
            v = float(row["rpt_r"])
            if v != v:
                v = None
        except ValueError:
            v = None
        per[(row["session"], int(row["trial_idx"]))][row["joint"]] = v

    joints = sorted({j for jr in per.values() for j in jr})
    assert len(joints) == 6, "expected 6 joints in " + src.name + ", got " + repr(joints)

    bysess = defaultdict(list)
    for (sess, _tidx), jr in per.items():
        allv = nanmean([jr.get(j) for j in joints])
        if allv is None:
            continue
        bysess[sess].append(allv)
    return bysess, joints


def paired_tests(a, b):
    """b vs a, two-sided. Returns Wilcoxon W/p and paired t/df/p."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = stats.wilcoxon(b, a, alternative="two-sided")
    t = stats.ttest_rel(b, a)
    return dict(wilcoxon_W=float(w.statistic), wilcoxon_p=float(w.pvalue),
                t=float(t.statistic), df=int(len(a) - 1), t_p=float(t.pvalue),
                n=int(len(a)), n_b_gt_a=int((b > a).sum()))


def main() -> None:
    iw_bysess, iw_joints = per_session_trial_values(IW_SRC)
    rw_bysess, rw_joints = per_session_trial_values(RW_SRC)

    sessions = sorted(set(iw_bysess) & set(rw_bysess))
    assert len(sessions) == 80, "expected 80 paired sessions, got " + str(len(sessions))
    assert set(iw_bysess) == set(rw_bysess) == set(sessions), "IW/RW session sets differ"

    rows = []
    for s in sessions:
        iv, rv = iw_bysess[s], rw_bysess[s]
        rows.append(dict(session=s, subject=s.split("_sess")[0],
                         n_trials_iw=len(iv), n_trials_rw=len(rv),
                         iw_median=median(iv), rw_median=median(rv),
                         iw_mean=mean(iv), rw_mean=mean(rv),
                         rw_minus_iw_median=median(rv) - median(iv)))

    ref = ec.sd("fig2g_session_summary.csv")
    ref_med = {r["session"]: float(r["All_median"]) for r in csv.DictReader(ref.open(newline=""))}
    worst = 0.0
    for r in rows:
        d = abs(r["iw_median"] - ref_med[r["session"]])
        worst = max(worst, d)
        assert d < 1e-6, (r["session"] + ": iw_median " + format(r["iw_median"], ".8f") +
                          " != fig2g All_median " + format(ref_med[r["session"]], ".8f"))
    print("sanity vs fig2g_session_summary.csv All_median: max |delta| = "
          + format(worst, ".2e") + " over " + str(len(rows)) + " sessions  OK")

    iw_med = [r["iw_median"] for r in rows]
    rw_med = [r["rw_median"] for r in rows]
    cohort_iw = median(iw_med)
    cohort_rw = median(rw_med)

    pooled_iw = median([v for s in sessions for v in iw_bysess[s]])
    pooled_rw = median([v for s in sessions for v in rw_bysess[s]])
    print("cohort pooled-trial median IW = " + format(pooled_iw, ".4f")
          + "   RW = " + format(pooled_rw, ".4f") + "   (Supplementary Note 1 convention)")
    print("cohort median of per-session IW medians = " + format(cohort_iw, ".4f"))
    print("cohort median of per-session RW medians = " + format(cohort_rw, ".4f"))
    if not ec.IS_DEFAULT:
        ec.check_anchor("iw_median_of_session_medians", cohort_iw, "IW cohort median of per-session medians")
    else:
        pass

    sess_stats = paired_tests(iw_med, rw_med)

    bysub = defaultdict(lambda: {"iw": [], "rw": []})
    for r in rows:
        bysub[r["subject"]]["iw"].append(r["iw_median"])
        bysub[r["subject"]]["rw"].append(r["rw_median"])
    subs = sorted(bysub)
    assert len(subs) == 42, "expected 42 participants, got " + str(len(subs))
    sub_iw = [mean(bysub[s]["iw"]) for s in subs]
    sub_rw = [mean(bysub[s]["rw"]) for s in subs]
    subj_stats = paired_tests(sub_iw, sub_rw)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    SD = ec.scoped(OUTDIR / "fig4_trialmedian_source_data.csv")
    with SD.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "subject", "n_trials_iw", "n_trials_rw",
                    "iw_median", "rw_median", "iw_mean", "rw_mean", "rw_minus_iw_median"])
        for r in rows:
            w.writerow([r["session"], r["subject"], r["n_trials_iw"], r["n_trials_rw"],
                        format(r["iw_median"], ".6f"), format(r["rw_median"], ".6f"),
                        format(r["iw_mean"], ".6f"), format(r["rw_mean"], ".6f"),
                        format(r["rw_minus_iw_median"], ".6f")])
    print("wrote " + str(SD))

    if ec.IS_DEFAULT:
        convention_note = (
            "the SI quotes cohort_pooled_trial_median_*"
            "; the Fig. 4b panel plots one point per "
            "session and its legend quotes cohort_median_of_session_medians_*.  "
            ""
            "")
    else:
        convention_note = (
            "the SI keeps cohort_pooled_trial_median_* (IW "
            + format(pooled_iw, ".3f") + ", RW " + format(pooled_rw, ".3f")
            + "); the Results sentence, the Fig. 4b panel (one point per "
            "session) and its legend quote cohort_median_of_session_medians_* (the headline convention; IW "
            + format(cohort_iw, ".3f") + ", RW " + format(cohort_rw, ".3f")
            + "). At 2 dp the two conventions give IW " + format(pooled_iw, ".2f") + " vs "
            + format(cohort_iw, ".2f") + " and RW " + format(pooled_rw, ".2f") + " vs "
            + format(cohort_rw, ".2f") + "")

    if ec.IS_DEFAULT:
        rw_provenance = (
            "concatenated the per-session "
            "trial_dump_real_walking/*_trial_r.csv; no decoder re-run")
    else:
        rw_provenance = (
            "build_tables.py (set rw) concatenated the " + str(len(rw_bysess))
            + " per-session *_trial_r.csv files of " + ec.rel("h_rw_dump_dir")
            + " (causal complex-demodulation features, real-walking leave-subject-out PCA30 basis, "
            + "causal 6 Hz stage-1 low-pass, random 10-fold readout" + "); no decoder re-run")

    MD = ec.scoped(OUTDIR / "fig4_trialmedian_metadata.csv")
    md = [
        ("panel", "Fig. 4b (paired IW vs RW session plot) and Fig. 4c (|c| -> r scatter), "
                  "trial-median convention"),
        ("r_convention", "trial_median"),
        ("r_convention_definition",
         "per trial: nanmean of calib rpt_r over the 6 joints; per session: median over trials "
         "(identical to 03_figures/fig2g_trial_r_export.py 'All')"),
        ("iw_joints", " ".join(iw_joints)), ("rw_joints", " ".join(rw_joints)),
        ("cohort", str(len(sessions)) + " sessions / " + str(len(subs))
                   + " participants (healthy, full cohort)"),
        ("n_trials_iw_total", sum(r["n_trials_iw"] for r in rows)),
        ("n_trials_rw_total", sum(r["n_trials_rw"] for r in rows)),
        ("cohort_median_of_session_medians_iw", format(cohort_iw, ".6f")),
        ("cohort_median_of_session_medians_rw", format(cohort_rw, ".6f")),
        ("cohort_pooled_trial_median_iw", format(pooled_iw, ".6f")),
        ("cohort_pooled_trial_median_rw", format(pooled_rw, ".6f")),
        ("cohort_mean_of_session_medians_iw", format(mean(iw_med), ".6f")),
        ("cohort_mean_of_session_medians_rw", format(mean(rw_med), ".6f")),
        ("sessions_rw_gt_iw", str(sess_stats["n_b_gt_a"]) + "/" + str(sess_stats["n"])),
        ("sessions_wilcoxon_W", format(sess_stats["wilcoxon_W"], ".1f")),
        ("sessions_wilcoxon_p", format(sess_stats["wilcoxon_p"], ".3e")),
        ("sessions_paired_t", format(sess_stats["t"], ".3f")),
        ("sessions_paired_t_df", sess_stats["df"]),
        ("sessions_paired_t_p", format(sess_stats["t_p"], ".3e")),
        ("participants_unit", "per-participant mean of that participant's session medians"),
        ("participants_n", subj_stats["n"]),
        ("participants_rw_gt_iw", str(subj_stats["n_b_gt_a"]) + "/" + str(subj_stats["n"])),
        ("participants_wilcoxon_W", format(subj_stats["wilcoxon_W"], ".1f")),
        ("participants_wilcoxon_p", format(subj_stats["wilcoxon_p"], ".3e")),
        ("participants_paired_t", format(subj_stats["t"], ".3f")),
        ("participants_paired_t_df", subj_stats["df"]),
        ("participants_paired_t_p", format(subj_stats["t_p"], ".3e")),
        ("tests_note", "all tests two-sided; Wilcoxon signed-rank is the headline (medians), "
                       "the paired t is reported for reference only"),
    ]
    with MD.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(md)
    print("wrote " + str(MD))

    print("sessions: RW>IW " + str(sess_stats["n_b_gt_a"]) + "/" + str(sess_stats["n"])
          + "  Wilcoxon W=" + format(sess_stats["wilcoxon_W"], ".1f")
          + " p=" + format(sess_stats["wilcoxon_p"], ".3e")
          + "  paired t(" + str(sess_stats["df"]) + ")=" + format(sess_stats["t"], ".2f")
          + " p=" + format(sess_stats["t_p"], ".3e"))
    print("participants (n=" + str(subj_stats["n"]) + "): RW>IW " + str(subj_stats["n_b_gt_a"])
          + "/" + str(subj_stats["n"])
          + "  Wilcoxon W=" + format(subj_stats["wilcoxon_W"], ".1f")
          + " p=" + format(subj_stats["wilcoxon_p"], ".3e")
          + "  paired t(" + str(subj_stats["df"]) + ")=" + format(subj_stats["t"], ".2f")
          + " p=" + format(subj_stats["t_p"], ".3e"))


if __name__ == "__main__":
    main()
