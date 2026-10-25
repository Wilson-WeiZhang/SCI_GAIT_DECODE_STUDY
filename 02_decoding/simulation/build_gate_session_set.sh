#!/usr/bin/env bash
#
# Directories and symlinks of the gate session set, one directory per session containing:
#   <sess>_event.mat  written by make_gate_events.m (qc -> 'good' for gaitimagery + rest only)
# This script only makes the directories and the symlinks; run make_gate_events.m afterwards:
set -euo pipefail
B=${GAIT_PROJECT_ROOT}
D=$B/patient_prep_gate

pairs="
pat01_sess04 patient_prep_restgood_v4_7
pat01_sess05 patient_prep_restgood
pat01_sess06 patient_prep_restgood_v4_7
pat01_sess07 patient_prep_restgood_v4_7
pat01_sess08 patient_prep_restgood
pat01_sess09 patient_prep_restgood_v4_7
pat02_sess03 patient_prep_restgood_v4_7
pat02_sess05 patient_prep_restgood_v4_7
pat02_sess06 patient_prep_restgood_v4_7
pat02_sess08 patient_prep
pat02_sess09 patient_prep
"

mkdir -p "$D"
echo "$pairs" | while read -r s r; do
  [ -z "$s" ] && continue
  mkdir -p "$D/$s"
  src=$(readlink -f "$B/$r/$s/${s}_eeg.mat")
  ln -sfn "$src" "$D/$s/${s}_eeg.mat"
  printf '%-14s -> %s\n' "$s" "$src"
done
echo "symlinks done in $D"
