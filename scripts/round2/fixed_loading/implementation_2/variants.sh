#!/bin/bash
# method-challenge variants, runs 3 and 7, sequential (one SDA run at a time); BLAS threads = G_ROS_THREADS (default 2)
T=${G_ROS_THREADS:-2}
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T VECLIB_MAXIMUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
PY=${PYTHON:-python}
cd "$(dirname "$0")" || exit 1
LOGS=$($PY -c "import vp_common; print(vp_common.LOGS)") || exit 1
$PY save_run_npy.py 7 > "$LOGS/save_run7.log" 2>&1
go() { r=$1; shift; tag=$(echo "$@" | tr ' ' '_'); /usr/bin/time -l $PY 04_solve.py $r "$@" > "$LOGS/04_solve_run${r}_v_${tag}.log" 2>&1 || echo "FAILED $r $@"; echo "variant run $r $@ done $(date +%T) $(grep 'maximum resident' "$LOGS/04_solve_run${r}_v_${tag}.log")"; }
go 3 ss 1 inv
go 3 full 1 mf emp
go 3 pooled 1 mf
go 3 rate 1 mf
go 3 flat 1 mf
go 3 full 0 mf
go 3 full 10 mf
go 7 plain 1 inv
go 7 ss 1 inv
go 7 full 1 inv
go 7 full 1 none
go 7 full 1 mf emp
go 7 pooled 1 mf
go 7 rate 1 mf
go 7 flat 1 mf
go 7 full 0 mf
go 7 full 10 mf
echo "variants all done"
