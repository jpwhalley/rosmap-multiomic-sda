#!/bin/bash
# sequential, one SDA run at a time; BLAS threads = G_ROS_THREADS (default 2); primary fit = analytic second moments,
# mean-field A variance. Usage: run_all.sh 1 2 ... 10   (/usr/bin/time -l is the macOS form; use -v with GNU time)
T=${G_ROS_THREADS:-2}
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T VECLIB_MAXIMUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
PY=${PYTHON:-python}
cd "$(dirname "$0")" || exit 1
CACHE=$($PY -c "import vp_common; print(vp_common.CACHE)") || exit 1
LOGS=$($PY -c "import vp_common; print(vp_common.LOGS)") || exit 1
for r in "$@"; do
  mkdir -p "$CACHE/run$r"
  /usr/bin/time -l $PY 04_solve.py $r full 1 mf > "$LOGS/04_solve_run${r}_full_mf.log" 2>&1 || { echo "run $r failed"; exit 1; }
  $PY 05_eval.py $r fit_full_r1_mf >> "$LOGS/05_eval_mf.log" 2>&1
  echo "run $r done $(date +%T) $(grep 'maximum resident' "$LOGS/04_solve_run${r}_full_mf.log")"
done
