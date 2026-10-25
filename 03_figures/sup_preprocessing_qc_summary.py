#!/usr/bin/env python3
"""Supplementary item

Preprocessing quality-control summary: automatic vs manual decisions and their effect

Reads:
  04_source_data/supplementary/sup_session_exclusions_source_data.csv
  preprocessing output in $GAIT_PREP_OUTPUT (not distributed)

Writes:
  sup_preprocessing_qc_summary_source_data.csv (not distributed)        + _metadata.csv
  sup_preprocessing_qc_interventions_source_data.csv (not distributed)  + _metadata.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
PREP_OUTPUT = __import__("pathlib").Path(__import__("os").environ.get("GAIT_PREP_OUTPUT", "preprocessing_output"))
_release_rel = lambda p, r: p.relative_to(r) if p.is_relative_to(r) else p

import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data" / "supplementary"
H_ALIGN = PREP_OUTPUT / "healthy/result/qc/prep_align_verdicts.csv"
H_REGISTRY = PREP_OUTPUT / "healthy/code/shared/build_session_registry.m"
H_MANUAL = PREP_OUTPUT / "healthy/code/01_align/manual_align_corrections.m"
H_MANIFEST = PREP_OUTPUT / "healthy/note/cohort_manifest_healthy.csv"
H_INVENTORY = PREP_OUTPUT / "healthy/result/qc/prep_healthy_cohort_trial_inventory.csv"
H_GONI = PREP_OUTPUT / "healthy/result/qc/prep_goni_qc_recompute_files.csv"
P_ALIGN_ALL = PREP_OUTPUT / "patient/result/qc/phase_a_summary.csv"
P_ALIGN_USED = SD / "sup_alignment_qc_patient_source_data.csv"
EXCLUSIONS = SD / "sup_session_exclusions_source_data.csv"
P_INVENTORY = PREP_OUTPUT / "patient/result/qc/prep_patient_patient15_trial_inventory.csv"


EXCLUSION_REASON = {
    "subj10_sess01": "Insufficient valid recording",
    "subj14_sess01": "Goniometer failure",
    "subj33_sess02": "EEG-goniometer alignment failure",
    "pat01_sess03": "recording hardware failure",
    "pat02_sess04": "recording hardware failure",
    "pat02_sess07": "recording hardware failure",
}
PATIENT_REPAIRS = [
    ("pat01_sess04", "EEG channel mapping", "amplifier module connectors swapped at recording; channel order restored before bad-channel detection"),
    ("pat02_sess01", "EEG channel mapping", "amplifier module connectors swapped at recording; channel order restored before bad-channel detection"),
    ("pat02_sess02", "EEG channel mapping", "amplifier module connectors swapped at recording; channel order restored before bad-channel detection"),
    ("pat02_sess09", "goniometer channel labels", "one reference-walker ankle sensor was labelled as a participant sensor at recording; label corrected by sensor serial number"),
]


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    assert len(read(path)) == len(rows)


def healthy_label(sub_sess: str) -> str:
    """SUB_07_sess01 -> subj07_sess01 (the export naming used in Source Data)."""
    m = re.fullmatch(r"SUB_(\d+)_(sess\d+)", sub_sess)
    return f"subj{m.group(1)}_{m.group(2)}"


def spurious_overrides() -> dict[str, list[int]]:
    text = H_REGISTRY.read_text(encoding="utf-8")
    out = {}
    for label, idx in re.findall(r"'(SUB_\d+_sess\d+[a-e]?)',\s*'[^']*',\s*'[^']*',\s*\[([0-9 ]+)\]", text):
        out[label] = [int(x) for x in idx.split()]
    return out


def manual_pairings() -> list[str]:
    text = H_MANUAL.read_text(encoding="utf-8")
    labels = []
    for case in re.findall(r"^\s*case\s+(.+?)\s*(?:%.*)?$", text, re.M):
        labels += re.findall(r"'([^']+)'", case)
    return labels


def main() -> None:
    summary: list[dict] = []
    interventions: list[dict] = []

    def add(cohort, stage, measure, n, denom, unit, decision, source):
        pct = f"{100.0 * n / denom:.2f}" if denom else ""
        summary.append(dict(cohort=cohort, stage=stage, measure=measure, n=n, denominator=denom,
                            percent=pct, unit=unit, decision=decision, source=source))

    seg = read(H_ALIGN)
    sessions = sorted({r["session"] for r in seg})
    assert len(sessions) == 83 and len(seg) == 98, (len(sessions), len(seg))
    spur = spurious_overrides()
    pairs = manual_pairings()
    assert set(spur) <= {r["segment"] for r in seg}, set(spur) - {r["segment"] for r in seg}
    assert set(pairs) <= {r["segment"] for r in seg}, set(pairs) - {r["segment"] for r in seg}

    manual_seg = set(spur) | set(pairs)
    wide = {r["segment"] for r in seg if r["ioi_match_tolerance_ms"] == "1200"}
    auto_pass = [r for r in seg if r["segment"] not in manual_seg and r["auto_verdict"] == "PASS"]
    auto_fail = [r for r in seg if r["segment"] not in manual_seg and r["auto_verdict"] != "PASS"]
    add("healthy", "alignment", "recorded sessions", len(sessions), len(sessions), "session", "input", "alignment logs")
    add("healthy", "alignment", "recording segments", len(seg), len(seg), "segment", "input", "alignment logs")
    add("healthy", "alignment", "segments passed with no manual input", len(auto_pass), len(seg), "segment", "automatic", "alignment logs")
    add("healthy", "alignment", "of which on the wider-tolerance rule for the second acquisition system",
        len([r for r in auto_pass if r["segment"] in wide]), len(seg), "segment", "automatic", "alignment logs")
    add("healthy", "alignment", "segments not passed with no manual input", len(auto_fail), len(seg), "segment", "automatic", "alignment logs")
    add("healthy", "alignment", "segments with a hand-verified marker pairing", len(pairs), len(seg), "segment", "manual", "hand-verified pairing table")
    add("healthy", "alignment", "segments with spurious goniometer edges removed", len(spur), len(seg), "segment", "manual", "session registry")
    add("healthy", "alignment", "segments with any manual input", len(manual_seg), len(seg), "segment", "manual", "session registry + hand-verified pairing table")
    add("healthy", "alignment", "sessions with any manual input",
        len({re.sub(r"[a-e]$", "", s) for s in manual_seg}), len(sessions), "session", "manual", "session registry + hand-verified pairing table")
    final = defaultdict(int)
    for r in seg:
        final[r["auto_verdict"]] += 1
    for v in ("PASS", "SPECIAL_CASE", "FAIL"):
        add("healthy", "alignment", f"final segment verdict {v}", final[v], len(seg), "segment", "result", "alignment logs")

    by_seg = {r["segment"]: r for r in seg}
    for s in sorted(manual_seg):
        r = by_seg[s]
        steps = []
        if s in pairs:
            steps.append("hand-verified marker pairing")
        if s in spur:
            steps.append(f"removed goniometer edge(s) {' '.join(map(str, spur[s]))} with no EEG counterpart")
        interventions.append(dict(
            cohort="healthy", session=healthy_label(re.sub(r"[a-e]$", "", s)),
            segment=s[-1] if re.search(r"sess\d+[a-e]$", s) else "", stage="alignment",
            intervention="; ".join(steps),
            result=f"{r['auto_verdict']} ({r['n_matched']}/{r['n_eeg_ttl']} markers matched)"
                   + ("; segment not exported or analysed" if r["auto_verdict"] == "FAIL" else "")))

    man = read(H_MANIFEST)
    assert len(man) == 83
    status = defaultdict(int)
    for r in man:
        status[r["status"]] += 1
    excl = [r for r in read(EXCLUSIONS) if r["cohort"] == "healthy"]
    assert status["include"] == 80 and status["excluded"] == 3 == len(excl), (status, len(excl))
    participants = {r["subject"] for r in man if r["status"] == "include"}
    for r in excl:
        why = EXCLUSION_REASON[r["session"]]
        add("healthy", "session inclusion", f"excluded: {why}", 1, 83, "session", "manual", "session exclusion table")
        interventions.append(dict(cohort="healthy", session=r["session"], segment="", stage="session inclusion",
                                  intervention="whole session excluded", result=why))
    add("healthy", "session inclusion", "retained sessions", 80, 83, "session", "result", "session manifest")
    add("healthy", "session inclusion", "retained participants", len(participants), len(participants), "participant", "result", "session manifest")

    inv = read(H_INVENTORY)
    assert len(inv) == 80
    goni = read(H_GONI)
    stored = defaultdict(int)
    for r in goni:
        stored[healthy_label(r["base_label"])] += int(r["stored_problem"])
    tot = defaultdict(int)
    flagged = dup = 0
    for r in inv:
        base = re.sub(r"_[a-e]+$", "", r["session"])
        for k in ("n_gi_events", "n_gi_qc_good", "n_rw_events", "n_rw_qc_good", "n_rest_events", "n_rest_qc_good", "n_qc_bad"):
            tot[k] += int(r[k])
        task_bad = (int(r["n_gi_events"]) - int(r["n_gi_qc_good"])) + (int(r["n_rw_events"]) - int(r["n_rw_qc_good"]))
        flagged += min(stored[base], task_bad)
        dup += task_bad - min(stored[base], task_bad)
    iw_x = tot["n_gi_events"] - tot["n_gi_qc_good"]
    rw_x = tot["n_rw_events"] - tot["n_rw_qc_good"]
    rest_x = tot["n_rest_events"] - tot["n_rest_qc_good"]
    assert iw_x + rw_x + rest_x == tot["n_qc_bad"]
    task = tot["n_gi_events"] + tot["n_rw_events"]
    add("healthy", "goniometer trial QC", "imagined + real walking trials exported", task, task, "trial", "input", "exported trial inventory")
    add("healthy", "goniometer trial QC", "trials excluded by the goniometer MAD rules", flagged, task, "trial", "automatic", "goniometer QC table")
    add("healthy", "goniometer trial QC", "trials excluded by the EEG-goniometer duration check", dup, task, "trial", "automatic", "exported trial inventory")
    add("healthy", "goniometer trial QC", "imagined walking trials excluded", iw_x, tot["n_gi_events"], "trial", "result", "exported trial inventory")
    add("healthy", "goniometer trial QC", "real walking trials excluded", rw_x, tot["n_rw_events"], "trial", "result", "exported trial inventory")
    add("healthy", "goniometer trial QC", "imagined walking trials reaching the decoder", tot["n_gi_qc_good"], tot["n_gi_events"], "trial", "result", "exported trial inventory")
    add("healthy", "goniometer trial QC", "real walking trials reaching the decoder", tot["n_rw_qc_good"], tot["n_rw_events"], "trial", "result", "exported trial inventory")
    frac = sorted((int(r["n_gi_events"]) - int(r["n_gi_qc_good"]) + int(r["n_rw_events"]) - int(r["n_rw_qc_good"]))
                  / (int(r["n_gi_events"]) + int(r["n_rw_events"])) for r in inv)
    add("healthy", "goniometer trial QC", "sessions with no excluded walking trial", sum(1 for f in frac if f == 0), 80, "session", "result", "exported trial inventory")

    used = {r["session"]: r for r in read(P_ALIGN_USED)}
    allrows = {r["label"]: r for r in read(P_ALIGN_ALL)}
    assert len(used) == 15 and len(allrows) == 18
    pexcl = {r["session"]: r for r in read(EXCLUSIONS) if r["cohort"] == "patient"}
    assert len(pexcl) == 3 and not set(pexcl) & set(used)
    verdict = {s: r["verdict"] for s, r in used.items()}
    verdict.update({s: allrows[s]["verdict"] for s in pexcl})
    assert len(verdict) == 18
    vc = defaultdict(int)
    for v in verdict.values():
        vc[v] += 1
    add("patient", "alignment", "recorded sessions", 18, 18, "session", "input", "alignment logs")
    for v, label in (("PASS", "passed on the standard marker protocol"),
                     ("PASS_LEGACY", "passed on the two-marker protocol of the first two recordings"),
                     ("SPECIAL_CASE", "passed on the multi-file rule (at least 80% of markers matched)"),
                     ("FAIL", "not passed")):
        add("patient", "alignment", f"sessions {label}", vc[v], 18, "session", "automatic", "alignment logs")
    fails = sorted(s for s, v in verdict.items() if v == "FAIL")
    kept_fail = [s for s in fails if s in used]
    add("patient", "alignment", "not passed but retained after review", len(kept_fail), 18, "session", "manual", "alignment logs")
    for s in kept_fail:
        interventions.append(dict(cohort="patient", session=s, segment="", stage="alignment",
                                  intervention="retained after review: the goniometer recording ends before the EEG, so the unmatched markers are the final trials only",
                                  result=f"FAIL ({used[s]['n_matched']} markers matched)"))

    assert set(EXCLUSION_REASON) == {r["session"] for r in excl} | set(pexcl)
    n_why = defaultdict(int)
    for s in sorted(pexcl):
        why = EXCLUSION_REASON[s]
        n_why[why] += 1
        interventions.append(dict(cohort="patient", session=s, segment="", stage="session inclusion",
                                  intervention="whole session excluded", result=why))
    for why, n in n_why.items():
        add("patient", "session inclusion", f"excluded: {why}", n, 18, "session", "manual", "session exclusion table")
    for s, stage, what in PATIENT_REPAIRS:
        assert s in used, s
        interventions.append(dict(cohort="patient", session=s, segment="", stage=stage,
                                  intervention=what, result="retained"))
    add("patient", "EEG channel mapping", "sessions with connector order restored", 3, 18, "session", "manual", "preprocessing notes")
    add("patient", "goniometer channel labels", "sessions with a sensor label corrected", 1, 18, "session", "manual", "preprocessing notes")
    add("patient", "session inclusion", "retained sessions", 15, 18, "session", "result", "alignment logs")
    pinv = read(P_INVENTORY)
    assert len(pinv) == 15
    ev = sum(int(r["n_iw_events"]) for r in pinv)
    good = sum(int(r["n_iw_qc_good"]) for r in pinv)
    restored = sum(int(r["n_iw_override_restored"]) for r in pinv)
    add("patient", "goniometer trial QC", "imagined walking trials exported", ev, ev, "trial", "input", "exported trial inventory")
    add("patient", "goniometer trial QC", "trials re-admitted: failing only on a dead reference-walker channel the decoder does not use",
        restored, ev, "trial", "manual", "exported trial inventory")
    add("patient", "goniometer trial QC", "imagined walking trials excluded", ev - good, ev, "trial", "result", "exported trial inventory")
    add("patient", "goniometer trial QC", "imagined walking trials reaching the decoder", good, ev, "trial", "result", "exported trial inventory")
    for condition, label in (("pre_operative", "pre-operative"), ("post_operative", "post-operative")):
        rows = [r for r in pinv if r["cohort"] == condition]
        e = sum(int(r["n_iw_events"]) for r in rows)
        g = sum(int(r["n_iw_qc_good"]) for r in rows)
        add("patient", "goniometer trial QC", f"{label} imagined walking trials reaching the decoder ({len(rows)} sessions)",
            g, e, "trial", "result", "exported trial inventory")
    for r in pinv:
        if int(r["n_iw_override_restored"]):
            interventions.append(dict(cohort="patient", session=r["session"], segment="", stage="goniometer trial QC",
                                      intervention="trials failing goniometer QC only on the dead reference-walker left-knee channel re-admitted; that joint is not scored",
                                      result=f"{r['n_iw_override_restored']} trials re-admitted"))

    write(SD / "sup_preprocessing_qc_summary_source_data.csv", summary)
    write(SD / "sup_preprocessing_qc_interventions_source_data.csv", interventions)
    meta = [
        dict(key="scope", value="healthy recorded sessions (83) and patient recorded sessions (18); trial counts are for retained sessions"),
        dict(key="healthy_cohort", value="80 sessions / 42 participants"),
        dict(key="patient_cohort", value="15 sessions / 2 participants (case-level)"),
        dict(key="build_id", value="main;pre_operative;post_operative"),
        dict(key="build_id_note", value="Prep-side and DECODER-INDEPENDENT: every input is a prep (healthy) or prep_patient (patient) alignment / goniometer-QC / manifest artefact; no decode product is read. The tokens name the cohorts the counts are scored against - healthy prep healthy cohort and the retained patient set pre_operative + post_operative."),
        dict(key="decision_column", value="input = count entering the stage; automatic = decided by a coded rule; manual = decided or corrected by a person; result = after both"),
        dict(key="generator", value="sup_preprocessing_qc_summary.py"),
    ]
    write(SD / "sup_preprocessing_qc_summary_metadata.csv", meta)
    write(SD / "sup_preprocessing_qc_interventions_metadata.csv", meta)
    for r in summary:
        print(f"{r['cohort']:8s} {r['stage']:20s} {r['decision']:9s} {r['n']:>6} / {r['denominator']:<6} {r['percent']:>6}  {r['measure']}")
    print(f"interventions: {len(interventions)}")


if __name__ == "__main__":
    main()
