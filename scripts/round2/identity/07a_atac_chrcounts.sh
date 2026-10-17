#!/bin/bash
# per-chromosome fragment counts for every MIT snATAC library; G_ROS_THREADS parallel jobs (default 2)
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
ROS=$($PY -c "import sys; sys.path.insert(0, '..'); import config; print(config.ROS)") || exit 1
export ATAC_OUT=$($PY -c "import sys; sys.path.insert(0, '..'); import config; print(config.out_dir(config.WORK, 'identity_checks', 'cache', 'atac'))") || exit 1
ls "$ROS"/mit/atac/*.fragments.tsv.gz | xargs -n 1 -P "${G_ROS_THREADS:-2}" "$PWD/07a_atac_one.sh"
echo done
