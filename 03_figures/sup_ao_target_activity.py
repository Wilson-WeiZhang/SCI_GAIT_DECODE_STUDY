"""Supplementary item

Reference-walker joint excursion under imagined walking (MI+AO) vs observation only (AO), six supplementary participants

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.io as sio
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from safe_write import safe_write_csv

GONI_DIR = ROOT / "data/supplementary/prep/goni"
RES = ROOT / "gait_prep_supplementary/result"
SD = ROOT / "04_source_data/supplementary"
SESSIONS = [("S01", "sub01", ["SUB_01_sess05"]),
            ("S09", "sub09", ["SUB_09_sess03"]),
            ("S12", "sub12", ["SUB_12_sess03a", "SUB_12_sess03b"]),
            ("S23", "sub23", ["SUB_23_sess03a", "SUB_23_sess03b"]),
            ("S29", "sub29", ["SUB_29_sess03a", "SUB_29_sess03b", "SUB_29_sess03c"]),
            ("S35", "sub35", ["SUB_35_sess03a", "SUB_35_sess03b"])]


def provenance(key):
    if key == "sub09":
        return RES / "s5_export_event_conditions.csv"
    return RES / key / "s5m_export_event_conditions.csv"


def walker_excursion(labels):
    exc, qcs = [], []
    for lab in labels:
        G = sio.loadmat(GONI_DIR / f"{lab}_goni.mat", squeeze_me=True, struct_as_record=False)
        for t, q in zip(np.atleast_1d(G["trials"]), np.atleast_1d(G["qc"])):
            if str(t.type) != "imagine":
                continue
            names = [" ".join(str(c).split()) for c in np.atleast_1d(t.goni_labels)]
            w = [i for i, c in enumerate(names) if c.endswith("W X")]
            assert len(w) == 6, f"{lab}: walker channels {len(w)} != 6"
            a = np.asarray(t.joint_angles, dtype=float)[:, w]
            exc.append(float(np.nanmean(np.nanstd(a, axis=0))))
            qcs.append(str(q.status))
    return np.array(exc), np.array(qcs)


def main():
    rows = []
    for pid, key, labels in SESSIONS:
        exc, qcs = walker_excursion(labels)
        with open(provenance(key), newline="") as f:
            prov = [r for r in csv.DictReader(f) if r["code"] == "gaitimagery"]
        prov.sort(key=lambda r: int(r["etstart"]))
        cond = np.array([r["cond"] for r in prov])
        assert len(cond) == len(exc), f"{key}: {len(cond)} provenance vs {len(exc)} goniometer imagine trials"
        a, b = exc[cond == "MI+AO"], exc[cond == "AO"]
        assert len(a) == 30 and len(b) == 30, f"{key}: MI+AO {len(a)} / AO {len(b)} trials"
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        ok = qcs[(cond == "MI+AO") | (cond == "AO")]
        rows.append([pid, key, "+".join(labels), len(a), len(b),
                     f"{np.median(a):.6f}", f"{np.median(b):.6f}", f"{np.mean(a):.6f}", f"{np.mean(b):.6f}",
                     f"{np.median(a) / np.median(b):.6f}", f"{u.statistic:.1f}", f"{u.pvalue:.6f}",
                     int((ok == "ok").sum()), len(ok)])
        print(f"{pid} {key}: walker excursion MI+AO {np.median(a):.2f} vs AO {np.median(b):.2f} deg, "
              f"ratio {np.median(a) / np.median(b):.3f}, Mann-Whitney p = {u.pvalue:.3f}, goni QC ok {rows[-1][-2]}/{len(ok)}")

    header = ["participant", "session_key", "recordings", "n_trials_miao", "n_trials_ao",
              "walker_excursion_median_miao_deg", "walker_excursion_median_ao_deg",
              "walker_excursion_mean_miao_deg", "walker_excursion_mean_ao_deg",
              "ratio_of_medians_miao_over_ao", "mannwhitney_u", "mannwhitney_p_two_sided",
              "n_goni_qc_ok", "n_trials_both"]
    safe_write_csv(SD / "sup_ao_target_activity_source_data.csv", header, rows)

    ratios = [float(r[9]) for r in rows]; ps = [float(r[11]) for r in rows]
    meta = [
        ("paper_location", "Supplementary Note 2, target-availability paragraph: 'all ratios within 0.7%, all p > 0.65'"),
        ("cohort", "six supplementary participants (S01, S09, S12, S23, S29, S35), one additional session each; "
                   "30 imagined-walking (MI+AO) and 30 observation-only (AO) trials per participant"),
        ("excursion", "per trial: mean over the six walker sagittal goniometer channels (labels ending 'W X') of the "
                      "channel's standard deviation over the trial, degrees"),
        ("test", "ratio of medians MI+AO / AO; two-sided Mann-Whitney U over all 30 vs 30 trials per participant "
                 "(calibration-segment trials included), uncorrected"),
        ("max_abs_ratio_deviation", f"{max(abs(r - 1) for r in ratios):.6f}"),
        ("min_p", f"{min(ps):.6f}"),
        ("inputs", "data/supplementary/prep/goni/<label>_goni.mat; gait_prep_supplementary/result/<sess>/s5m_export_event_conditions.csv "
                   "(sub09: gait_prep_supplementary/result/s5_export_event_conditions.csv)"),
        ("method_source", "gait_prep_supplementary/code/target_activity.py "),
        ("software_environment", f"python {sys.version.split()[0]}; numpy {np.__version__}; scipy {scipy.__version__}"),
    ]
    safe_write_csv(SD / "sup_ao_target_activity_metadata.csv", ["key", "value"], meta)
    print(f"max |ratio - 1| = {max(abs(r - 1) for r in ratios):.4f}; min p = {min(ps):.3f}")


if __name__ == "__main__":
    main()
