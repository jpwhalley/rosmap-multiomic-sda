#!/usr/bin/env bash
# Rebuild the figure panels and the composites, in dependency order.
# Set the G_ROS_* variables first, in the environment or in the file named by G_ROS_CONFIG (README.md).
# Usage: build_all.sh [panels|composites|all]     (default: all)
# Each step's output goes to $G_ROS_FIGURE_OUT/logs/<step>.log; the script stops at the first failure.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# the composite layouts refer to $G_ROS_FIGURE_OUT: take it from figure_paths.py (environment, G_ROS_CONFIG or default)
G_ROS_FIGURE_OUT="$(cd "$HERE" && "$PY" -c 'import figure_paths; print(figure_paths.FIGOUT)')"
export G_ROS_FIGURE_OUT
LOGS="$G_ROS_FIGURE_OUT/logs"
mkdir -p "$LOGS"
STAGE="${1:-all}"

run() {   # run <script relative to this folder> [args...]
  local script="$1"; shift
  local log="$LOGS/$(echo "${script%.py} $*" | tr ' /,:' '____' | sed 's/_*$//').log"
  echo "== $script $*"
  (cd "$HERE/$(dirname "$script")" && "$PY" "$(basename "$script")" "$@") > "$log" 2>&1 \
    || { echo "failed: $script (see $log)"; exit 1; }
}

compose() {   # compose <figure folder> <layout file> <name>
  echo "== compose $1 -> $3"
  "$PY" "$HERE/composites/compose.py" "$HERE/composites/$1/$2" --workdir "$G_ROS_FIGURE_OUT/$1/build" --name "$3" \
    > "$LOGS/compose_$3.log" 2>&1 || { echo "failed: compose $1 (see $LOGS/compose_$3.log)"; exit 1; }
}

if [[ "$STAGE" == panels || "$STAGE" == all ]]; then
  # stand-alone panels (statistics in the logs); p04 and p06 logs are read by the S4 and S9 builders
  for p in p01_structure p02_covariates p03_c74 p03b_fig2g_compact p04_c278 p05_neuronal p06_c21_single; do
    run "panels/$p.py"
  done
  run panels/cell_type_bars/make_relink_panels.py "$G_ROS_FIGURE_OUT/cell_type_bars"
  run panels/round1/fig1c_modality_coverage.py
  run panels/round1/fig4c_labelled.py
  run replot/kegg_panels/03_replot.py
  run replot/sample_panels/replot_v23labels.py
fi

if [[ "$STAGE" == composites || "$STAGE" == all ]]; then
  run composites/S1/s1_panels_final_size_v2.py
  run composites/S2/s2_panels_at_cell_size.py
  run composites/S3/s3_final_size.py
  run composites/S4/s4_panels.py
  run composites/S5/s5_panels_at_cell_size.py
  run composites/S6/s6_panels_at_cell_size.py
  run composites/S7/s7_panels_final_size.py
  run composites/S7/s7fg_fix.py
  run composites/S8/s8_panels.py
  run composites/S8/s8_fix_cdef.py
  run composites/S9/s9_panels_fix_v2.py
  run composites/S10/s10_panels_at_cell_size.py
  run composites/S10/fix_c_al121.py 1.5 0.3 AL121890.2,AL031666.2
  run composites/S11/s11_panels_at_cell_size.py
  run composites/S11/s11c_relabel.py
  run composites/Fig2/fig2_panels.py b c f g
  ( export FIG2_NUDGE1="DDX3Y:-3:-0.4"; run composites/Fig2/fig2_panels.py a d )
  ( export FIG2_EYMIN=-6.5 FIG2_EDGEB=1.8 FIG2_LCLEAR=1.5 FIG2_K=800 FIG2_SEED3=1 FIG2_SIDEPEN=100000
    run composites/Fig2/fig2_panels.py e )
  run composites/Fig3/f3_panels.py
  run composites/Fig3/f3f_fix_italic.py
  run composites/Fig4/fig4_panels.py
  run composites/Fig5/f5_panels.py
  compose S1 layout_S1_v2.json FigS1
  compose S2 layout_S2.json FigS2
  compose S3 layout_S3.json FigS3
  compose S4 layout_S4.json FigS4
  compose S5 layout_S5.json FigS5
  compose S6 layout_S6.json FigS6
  compose S7 layout_S7_v2.json FigS7
  compose S8 layout_S8_v2.json FigS8
  compose S9 layout_S9.json FigS9
  compose S10 layout_S10_v2.json FigS10
  compose S11 v23_figS11_py_v2.json FigS11
  compose Fig2 layout_Fig2.json Fig2
  compose Fig3 layout_Fig3_v2.json Fig3
  compose Fig4 layout_fig4.json Fig4
  compose Fig5 layout_Fig5_v2.json Fig5
fi
echo "done: $G_ROS_FIGURE_OUT"
