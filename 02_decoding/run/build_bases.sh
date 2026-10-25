#!/usr/bin/env bash
#
# The three basis builds of Step 07 in series (iw, hglobal, exec), each checked by run/check_basis.py;
# stops at the first failure.
# Usage (from 02_decoding/): bash run/build_bases.sh <cpu list> <parallel workers>
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/run_paths.sh"
for k in iw hglobal exec; do
  bash "$HERE/build_basis.sh" "$k" "$1" "$2" || { echo "[basis] $(date +%F_%T) STOP: $k failed"; exit 2; }
done
echo "[basis] $(date +%F_%T) ALL BASES READY"
