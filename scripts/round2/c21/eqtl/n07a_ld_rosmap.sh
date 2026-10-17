#!/bin/bash
# Local LD around rs6664332 in the ROSMAP array genotypes, read by n07_c21_public_eqtl.py. Run by hand in the round-2
# analysis; the command is reconstructed from the PLINK log. Output: $G_ROS_RESULTS_DIR/revision_analyses/neuronal_c21/cache.
set -e
PY=${PYTHON:-python}
cd "$(dirname "$0")"
ROS=$($PY -c "import sys; sys.path.insert(0, '../..'); import config; print(config.ROS)")
OUT=$($PY -c "import sys; sys.path.insert(0, '../..'); import config; print(config.out_dir(config.WORK, 'revision_analyses', 'neuronal_c21', 'cache'))")
cd "$OUT"
plink --bfile "$ROS/oth/snps/ROSMAP_arrayGenotype" --ld-snp rs6664332 --ld-window 99999 --ld-window-kb 600 --ld-window-r2 0 \
  --memory 2000 --r2 --threads "${G_ROS_THREADS:-2}" --out ld_rs6664332_rosmap
