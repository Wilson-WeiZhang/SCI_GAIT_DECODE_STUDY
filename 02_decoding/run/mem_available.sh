#!/usr/bin/env bash
#
# Available memory in MB for the memory gates of the drivers: MemAvailable plus the reclaimable part of
# a ZFS ARC cache (size - c_min), which free counts as used.
ma=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
arc=0
if [ -r /proc/spl/kstat/zfs/arcstats ]; then
  arc=$(awk '$1=="size" {s=$3} $1=="c_min" {m=$3} END {d=(s-m)/1048576; if (d<0) d=0; print int(d)}' /proc/spl/kstat/zfs/arcstats)
fi
echo $((ma + arc))
