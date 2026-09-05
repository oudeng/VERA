#!/bin/bash
# Ordered regeneration of every generated artifact.
#
# The order is not cosmetic. Three artifacts feed each other in one
# direction, and a wrong order leaves a stale copy that no lexical gate can
# see (it cost us a figure printing a superseded cost headline beside the
# corrected prose):
#
#   cost probes -> table_cost_primary.py -> a3_macros.tex
#                                        -> t_final.py (copies the macros)
#                                        -> fig_scoreboard.py (prints the copy)
#
# facts_gate.numeric_checks() now asserts that copy is current, so a wrong
# order is red rather than silent -- but running this script is the way to
# be right in the first place.
set -e
cd "$(dirname "$0")/.."
# Tenth review P0-3: no private absolute path. This script remains a
# REPOSITORY-ENVIRONMENT PROCEDURE -- it calls generators a review
# package deliberately does not carry -- and ships for inspection, not
# for execution inside the package. SNI_PY overrides the interpreter.
P="${SNI_PY:-python3}"
run () { echo "  -> $*"; env PYTHONHASHSEED=2025 "$P" "$@" > /dev/null; }

echo "[1/5] cost: probes -> primary table -> macros"
run reporting/table_cost_primary.py
run reporting/table_cost.py --fit-from-grid results/P2_main_grid

echo "[2/5] single evidence source (reads the cost macros)"
run experiments/t_final.py

echo "[3/5] tables and ESM fragments"
for g in table_faithfulness table_recovery table_leakage \
         table_fiveway_stability table_d_stability table_realgap \
         table_ceiling table_sign_test grid_scale esm_detail_tables esm_leakage \
         esm_noprior esm_probe2 esm_realpattern esm_rss esm_tapfam \
         audit_target_roles; do
  run "reporting/$g.py"
done
run reporting/table_avg_rank_r1.py --final
run reporting/table_impute_predict.py --long results/T4_downstream/t44_long.csv
run reporting/esm_sections.py --all

echo "[4/5] figures (fig_scoreboard reads t_final, so it goes last)"
# fig_vera.py is SUPERSEDED-BY-ASSET (2026-09-01): Fig. 1 is now the
# delivered design asset registered in docs/figure_assets.json, and
# regenerating it here would overwrite nothing the manuscript uses while
# making the gates' identity check fail on a file nobody asked for.
# run reporting/fig_vera.py
run reporting/fig_leakage.py
run reporting/fig_scoreboard.py

rm -f reporting/out/*_SELFTEST.tex reporting/out/*_SELFTEST.pdf

echo "[5/5] the two documents' directories and the submission views"
# P7-A SS2: each document compiles in its own directory, which means it
# CONTAINS the fragments it pulls. Regenerating without this step leaves
# those copies one round behind -- and the document still builds, from
# the stale local copy, which is the silent half of the failure.
run experiments/sync_submission_sources.py
echo "[ok] regeneration complete"
grep -l -- "-dirty" reporting/out/*.tex 2>/dev/null \
  && echo "WARNING: outputs carry a dirty commit -- commit first, then rerun" \
  || echo "[ok] no dirty provenance"
