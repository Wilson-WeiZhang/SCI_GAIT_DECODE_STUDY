"""Supplementary item

Supplementary Table 5 — per-session trial exclusion before decoder fitting (healthy + patient)

Reads:
  04_source_data/fig5/fig5b_session_box_source_data.csv
  preprocessing output in $GAIT_PREP_OUTPUT (not distributed)

Writes:
  04_source_data/supplementary/sup_goni_qc_exclusion_source_data.csv
  04_source_data/supplementary/sup_goni_qc_exclusion_metadata.csv
  supplementary/tables/Supplementary_Table_S5_Goni_QC_Trial_Exclusion.xlsx

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
PREP_OUTPUT = __import__("pathlib").Path(__import__("os").environ.get("GAIT_PREP_OUTPUT", "preprocessing_output"))
_release_rel = lambda p, r: p.relative_to(r) if p.is_relative_to(r) else p

import csv
import os
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

ROOT = Path(__file__).resolve().parents[1]
HEALTHY = PREP_OUTPUT / "healthy/result/qc/prep_healthy_cohort_trial_inventory.csv"
PATIENT = PREP_OUTPUT / "patient/result/qc/prep_patient_patient15_trial_inventory.csv"
HEALTHY_DECODED = ROOT / "sup_healthy_trial_inventory_source_data.csv (not distributed)"
PATIENT_DECODED = ROOT / "04_source_data/fig5/fig5b_session_box_source_data.csv"
SD = ROOT / "04_source_data/supplementary"
OUT = SD / "sup_goni_qc_exclusion_source_data.csv"
OUT_META = SD / "sup_goni_qc_exclusion_metadata.csv"
XLSX = ROOT / "figures_out/supplementary/tables/Supplementary_Table_S5_Goni_QC_Trial_Exclusion.xlsx"
FIELDS = ["cohort", "patient_phase", "session", "n_iw_trials", "n_rw_trials", "n_trials_assessed",
          "n_excluded", "pct_excluded"]


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def healthy_rows():
    decoded = {r["session"]: r for r in read(HEALTHY_DECODED) if r["row_type"] == "session"}
    out = []
    for r in read(HEALTHY):
        iw, rw = int(r["n_gi_events"]), int(r["n_rw_events"])
        good = int(r["n_gi_qc_good"]) + int(r["n_rw_qc_good"])
        d = decoded[r["session"]]
        assert good == int(float(d["n_iw_heldout_decoded"])) + int(float(d["n_rw_heldout_decoded"])), r["session"]
        out.append(dict(cohort="healthy", patient_phase="", session=r["session"], n_iw_trials=iw,
                        n_rw_trials=rw, n_trials_assessed=iw + rw, n_excluded=iw + rw - good))
    assert len(out) == 80
    return out


def patient_rows():
    decoded = {r["session"]: int(r["n_trials"]) for r in read(PATIENT_DECODED)}
    out = []
    for r in read(PATIENT):
        assert int(r["n_iw_qc_good"]) == decoded[r["session"]], r["session"]
        n = int(r["n_iw_events"])
        out.append(dict(cohort="patient", patient_phase=r["cohort"], session=r["session"], n_iw_trials=n,
                        n_rw_trials="", n_trials_assessed=n, n_excluded=int(r["n_iw_excluded"])))
    assert len(out) == 15 and set(decoded) == {r["session"] for r in out}
    return out


def summary(rows):
    n = sum(r["n_trials_assessed"] for r in rows)
    x = sum(r["n_excluded"] for r in rows)
    pct = [100.0 * r["n_excluded"] / r["n_trials_assessed"] for r in rows]
    return dict(n_sessions=len(rows), n_assessed=n, n_excluded=x, pooled=100.0 * x / n,
                lo=min(pct), hi=max(pct))


HFILL = PatternFill("solid", fgColor="525152")
THIN = Border(*[Side(style="thin", color="DDDDDD")] * 4)


def sheet(wb, name, caption, header, body):
    ws = wb.create_sheet(name)
    ws.append([caption])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(header))
    ws.cell(1, 1).font = Font(italic=True, size=9)
    ws.cell(1, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[1].height = 46
    ws.append(header)
    for c in ws[2]:
        c.font = Font(bold=True, color="FFFFFF", size=9)
        c.fill = HFILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = THIN
    for col, w in zip("ABCDEF", [16, 14, 14, 14, 14, 12]):
        ws.column_dimensions[col].width = w
    for row in body:
        ws.append(row)
    ws.freeze_panes = "A3"


def main():
    h, p = healthy_rows(), patient_rows()
    for r in h + p:
        r["pct_excluded"] = f"{100.0 * r['n_excluded'] / r['n_trials_assessed']:.2f}"
    tmp = OUT.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(h + p)
    os.replace(tmp, OUT)

    hs, ps = summary(h), summary(p)
    post = summary([r for r in p if r["patient_phase"] == "post_operative"])
    fmt = lambda s: f"{s['n_excluded']} of {s['n_assessed']} ({s['pooled']:.2f}%; per-session range {s['lo']:.2f}-{s['hi']:.2f}%)"
    meta = [
        ("healthy_scope", "80 sessions / 42 participants; imagined-walking + real-walking trials"),
        ("healthy_source", _release_rel(HEALTHY, ROOT).as_posix()),
        ("healthy_summary", fmt(hs)),
        ("patient_scope", "15 retained sessions (4 pre-operative + 11 post-operative), N = 2; imagined-walking trials"),
        ("patient_source", _release_rel(PATIENT, ROOT).as_posix()),
        ("patient_summary", fmt(ps)),
        ("patient_post_operative_summary", fmt(post)),
        ("assertions", "qc-good = decoded trial count in 80/80 healthy sessions (IW + RW) and 15/15 patient sessions"),
    ]
    tmp = OUT_META.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("key", "value")] + meta)
    os.replace(tmp, OUT_META)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    sheet(wb, "patient",
          "Patient cohort, 15 retained sessions, imagined-walking trials",
          ["Session", "Phase", "Trials assessed", "Trials excluded", "% excluded"],
          [[r["session"], "pre-operative" if r["patient_phase"] == "pre_operative" else "post-operative",
            r["n_trials_assessed"], r["n_excluded"], float(r["pct_excluded"])] for r in p])
    wb.save(XLSX)

    print("healthy", fmt(hs))
    print("patient", fmt(ps))
    print("post_operative", fmt(post))


if __name__ == "__main__":
    main()
