"""Supplementary item

Healthy participant demographics for the Methods "Participants" sentence

Reads:
  preprocessing output in $GAIT_PREP_OUTPUT (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
PREP_OUTPUT = __import__("pathlib").Path(__import__("os").environ.get("GAIT_PREP_OUTPUT", "preprocessing_output"))
_release_rel = lambda p, r: p.relative_to(r) if p.is_relative_to(r) else p
from pathlib import Path
import csv
import io
from statistics import mean, stdev, median

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from safe_write import safe_write

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/healthy_demographics.csv"
MANIFEST = PREP_OUTPUT / "healthy/note/cohort_manifest_healthy.csv"
SD = ROOT / "04_source_data/supplementary"
XLSX = ROOT / "figures_out/supplementary/tables/participant_demographics_table.xlsx"


FINAL_TITLE = ("Healthy participant demographics. One row per "
               "recruited participant; 1 marks the {n_part} participants in the analysis "
               "cohort ({n_sessions} sessions, {n_part} participants). All recruited "
               "participants are retained, so recruited n and analysis n coincide.")

FINAL_SUMMARY_LABELS = {
    "n": "Participants (n)", "n_male": "Men (n)", "n_female": "Women (n)",
    "age_mean": "Mean age (years)", "age_sd": "Age SD (years)",
    "age_min": "Minimum age (years)", "age_max": "Maximum age (years)",
    "age_median": "Median age (years)",
    "age_male_mean": "Mean age, men (years)", "age_male_sd": "Age SD, men (years)",
    "age_female_mean": "Mean age, women (years)", "age_female_sd": "Age SD, women (years)",
    "n_sessions": "Sessions (n)",
    "excluded_from_cohort": "Participants excluded from the cohort",
}
FINAL_SUMMARY_SCOPES = ("retained analysis cohort",)
FINAL_NONE_LABEL = "none"

FINAL_DEFINITIONS_HEADER = ["Item", "Description"]

EXPECT_SESSIONS = 80
EXPECT_PARTICIPANTS = 42
BUILD_ID = "main"
BUILD_ID_NOTE = ("decoder-independent (sex and age records and the prep cohort manifest only; no decoder product is "
            "read); the token names the healthy cohort (prep) analysis set, 80 sessions / 42 participants")
PAPER_LOCATION = ("Methods, Healthy participants (age mean +/- s.d. and range, 21 female / 21 male, of "
                  "the 42 participants); Supplementary Note 11 (the same with the median and the per-sex means; "
                  "per-participant sex and age in participant_demographics_source_data.csv)")


def load_demo() -> dict[str, tuple[str, int]]:
    out = {}
    with DEMO.open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[r["subject"]] = (r["sex"].strip().upper(), int(r["age"]))
    return out


def retained_subjects() -> tuple[set[str], int]:
    """Return (subjects with >=1 retained session, number of retained sessions)."""
    out, n_sess = set(), 0
    with MANIFEST.open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("status", "").strip() == "include":
                out.add(r["subject"])
                n_sess += 1
    return out, n_sess


def summarize(subjects, demo):
    rows = [demo[s] for s in subjects if s in demo]
    ages = [a for _s, a in rows]
    nM = sum(1 for s, _a in rows if s == "M")
    nF = sum(1 for s, _a in rows if s == "F")
    aM = [a for s, a in rows if s == "M"]
    aF = [a for s, a in rows if s == "F"]
    return {
        "n": len(rows), "n_male": nM, "n_female": nF,
        "age_mean": mean(ages), "age_sd": stdev(ages) if len(ages) > 1 else 0.0,
        "age_min": min(ages), "age_max": max(ages), "age_median": median(ages),
        "age_male_mean": mean(aM) if aM else float("nan"), "age_male_sd": stdev(aM) if len(aM) > 1 else 0.0,
        "age_female_mean": mean(aF) if aF else float("nan"), "age_female_sd": stdev(aF) if len(aF) > 1 else 0.0,
    }


HFILL = PatternFill("solid", fgColor="525152")
THIN = Border(*[Side(style="thin", color="DDDDDD")] * 4)


def _header(ws, row=1):
    for c in ws[row]:
        c.font = Font(bold=True, color="FFFFFF", size=9)
        c.fill = HFILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = THIN


def write_workbook(demo, cohort, analysis, recruited, excluded, n_sessions) -> None:
    """SI deliverable. Writes ONLY values already computed above -- nothing recomputed."""
    a = analysis
    wb = openpyxl.Workbook()

    ws = wb.active
    ws.title = "participants"
    title = FINAL_TITLE.format(n_part=EXPECT_PARTICIPANTS, n_sessions=n_sessions)
    ws.append([title])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=4)
    ws.cell(1, 1).font = Font(italic=True, size=9)
    ws.cell(1, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[1].height = 60
    ws.append(["Participant", "Sex", "Age (years)", "In analysis cohort"])
    _header(ws, 2)
    for s in sorted(demo):
        ws.append([s, demo[s][0], demo[s][1], 1 if s in cohort else 0])
    for col, w_ in zip("ABCD", [16, 8, 14, 20]):
        ws.column_dimensions[col].width = w_
    ws.freeze_panes = "A3"

    ws2 = wb.create_sheet("summary")
    ws2.append(["Scope", "Metric", "Value"])
    _header(ws2)
    for scope, summ in (("retained analysis cohort", analysis), ("all recruited", recruited)):
        if scope not in FINAL_SUMMARY_SCOPES:
            continue
        for k, v in summ.items():
            ws2.append([scope, FINAL_SUMMARY_LABELS[k], round(v, 4) if isinstance(v, float) else v])
    ws2.append(["retained analysis cohort", FINAL_SUMMARY_LABELS["n_sessions"], n_sessions])
    ws2.append(["retained analysis cohort", FINAL_SUMMARY_LABELS["excluded_from_cohort"],
                ";".join(excluded) or FINAL_NONE_LABEL])
    for col, w_ in zip("ABC", [26, 24, 18]):
        ws2.column_dimensions[col].width = w_
    ws2.freeze_panes = "A2"

    ws3 = wb.create_sheet("Definitions")
    ws3.append(FINAL_DEFINITIONS_HEADER)
    _header(ws3)
    for k, v in [
        ("Scope", "full-cohort healthy, participant level"),
        ("Cohort", "healthy analysis cohort"),
        ("Sessions", str(n_sessions)),
        ("Participants", str(a["n"])),
        ("Sex", f"{a['n_female']} women / {a['n_male']} men"),
        ("Age", f"mean {a['age_mean']:.1f} \u00b1 {a['age_sd']:.1f} SD years, "
                f"range {a['age_min']}-{a['age_max']}, median {a['age_median']:.0f}"),
        ("Age by sex", f"men {a['age_male_mean']:.1f} \u00b1 {a['age_male_sd']:.1f}; "
                       f"women {a['age_female_mean']:.1f} \u00b1 {a['age_female_sd']:.1f}"),
        ("SD definition", "sample SD (n-1 denominator)"),
        ("Demographics source", "participant records of sex and age collected at enrolment"),
        ("Inclusion", "cohort membership is decided by alignment and goniometer quality "
                      "control only; no EEG-component gate and no demographic criterion "
                      "enters inclusion"),
    ]:
        ws3.append([k, v])
    for col, w_ in zip("AB", [26, 110]):
        ws3.column_dimensions[col].width = w_
    for row in ws3.iter_rows(min_row=2, min_col=2, max_col=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    ws3.freeze_panes = "A2"

    XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(XLSX)


def main() -> None:
    demo = load_demo()
    cohort, n_sessions = retained_subjects()
    if n_sessions != EXPECT_SESSIONS or len(cohort) != EXPECT_PARTICIPANTS:
        raise SystemExit(
            f"cohort manifest is not healthy cohort: {n_sessions} retained sessions / "
            f"{len(cohort)} participants (expected {EXPECT_SESSIONS} / {EXPECT_PARTICIPANTS})")
    missing = sorted(cohort - set(demo))
    if missing:
        raise SystemExit(f"cohort subjects lack demographics (refusing to report incomplete): {missing}")

    analysis = summarize(cohort, demo)
    recruited = summarize(set(demo), demo)
    excluded = sorted(set(demo) - cohort)

    SD.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(["subject", "sex", "age", "in_analysis"])
    for s in sorted(demo):
        w.writerow([s, demo[s][0], demo[s][1], 1 if s in cohort else 0])
    safe_write(SD / "participant_demographics_source_data.csv", buf.getvalue())

    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(["scope", "metric", "value", "build_id", "build_id_note", "paper_location"])
    for scope, summ in (("retained_analysis_cohort", analysis), ("all_recruited", recruited)):
        for k, v in summ.items():
            w.writerow([scope, k, f"{v:.4f}" if isinstance(v, float) else v, BUILD_ID, BUILD_ID_NOTE, PAPER_LOCATION])
    w.writerow(["retained_analysis_cohort", "excluded_from_cohort", ";".join(excluded), BUILD_ID, BUILD_ID_NOTE,
                PAPER_LOCATION])
    safe_write(SD / "participant_demographics_metadata.csv", buf.getvalue())


    a = analysis
    print(f"=== retained analysis cohort (n={a['n']}) ===")
    print(f"  sex: {a['n_female']} women, {a['n_male']} men")
    print(f"  age: mean {a['age_mean']:.1f} +/- {a['age_sd']:.1f} SD, range {a['age_min']}-{a['age_max']}, median {a['age_median']:.0f}")
    print(f"  M {a['age_male_mean']:.1f}+/-{a['age_male_sd']:.1f} | F {a['age_female_mean']:.1f}+/-{a['age_female_sd']:.1f}")
    print(f"  excluded from cohort (recruited but not retained): {excluded}")
    print(f"\n  recruited total: n={recruited['n']} ({recruited['n_female']}F/{recruited['n_male']}M), "
          f"age {recruited['age_mean']:.1f}+/-{recruited['age_sd']:.1f}")
    print("\nMethods sentence:")
    print(f'  "EEG was recorded from {a["n"]} healthy adults ({a["n_female"]} women; '
          f'mean age {a["age_mean"]:.1f} +/- {a["age_sd"]:.1f} years, range {a["age_min"]}-{a["age_max"]})."')
    print("\nwrote participant_demographics_source_data.csv + _metadata.csv")


if __name__ == "__main__":
    main()
