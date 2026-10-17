#!/bin/bash
# sequential, one SDA run at a time; BLAS threads = G_ROS_THREADS (default 2). Usage: run_all.sh 1 2 ... 10
T=${G_ROS_THREADS:-2}
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T VECLIB_MAXIMUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
PY=${PYTHON:-python}
cd "$(dirname "$0")" || exit 1
LOGS=$($PY -c "import pi_common; print(pi_common.LOGS)") || exit 1
for r in "$@"; do
  $PY 03_suffstats.py $r > "$LOGS/03_suffstats_run$r.log" 2>&1 || { echo "run $r suffstats failed"; exit 1; }
  $PY 05_run.py $r > "$LOGS/05_run$r.log" 2>&1 || { echo "run $r scenarios failed"; exit 1; }
  echo "run $r done $(date +%T)"; tail -2 "$LOGS/05_run$r.log"
done
