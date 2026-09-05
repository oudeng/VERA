#!/usr/bin/env bash
# Chat ruling 2026-08-29 item 2, scheduling clause: the A3 single-thread
# timing probes take the idle window FIRST (their load>2 guard requires an
# empty machine); the NP-MIMIC 15-seed expansion pool launches only after
# A3 completes. This script encodes that ordering mechanically -- launch it
# once the last training pool has drained.
set -u
cd "$(dirname "$0")/.."
PY="${SNI_PY:-python}"
OUT=results/A3_cost_context
mkdir -p "$OUT" results/T3_faithfulness/logs_expansion

echo "=== waiting for the 1-min load to decay below 2.0 ==="
until awk '{exit !($1 < 2.0)}' /proc/loadavg; do sleep 60; done
echo "=== idle at $(date '+%H:%M:%S'), load $(cut -d' ' -f1 /proc/loadavg) ==="

echo "=== A3 probes (sequential, single-thread, /usr/bin/time -v) ==="
A3_RC=0
for ds in MIMIC eICU; do
  for obj in P MF SNI; do
    echo "--- $ds $obj $(date '+%H:%M:%S')"
    env PYTHONHASHSEED=2025 SNI_NUM_THREADS=1 /usr/bin/time -v \
      "$PY" experiments/cost_probe.py --dataset "$ds" --object "$obj" \
      2> "$OUT/${ds}_${obj}_time.txt" || { echo "A3 FAIL: $ds $obj"; A3_RC=1; }
  done
done
echo "=== A3 done rc=$A3_RC $(date '+%H:%M:%S') ==="

echo "=== NP-MIMIC expansion pool (seeds 13..987, 10 workers) ==="
for s in 13 21 34 55 89 144 233 377 610 987; do echo "MIMIC $s NoPrior"; done \
  > results/T3_faithfulness/npmimic_expansion_jobs.txt
xargs -a results/T3_faithfulness/npmimic_expansion_jobs.txt -L1 -P 10 bash -c \
  'env PYTHONHASHSEED=2025 SNI_NUM_THREADS=2 "${SNI_PY:-python}" experiments/faithfulness.py --stage train --datasets "$0" --seeds "$1" --variant "$2" > "results/T3_faithfulness/logs_expansion/${2}_${0}_seed${1}.log" 2>&1'
NP_RC=$?
echo "=== NP-MIMIC pool exit $NP_RC $(date '+%H:%M:%S') ==="
exit $((A3_RC + NP_RC))
