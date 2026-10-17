#!/bin/bash
# GEMMA 0.96 for the three C21 phenotype columns written by 12a_build_fam_c21.py (archived phenotype; RNA-discordant
# donor set to NA; all flagged donors set to NA). Run by hand in the round-2 analysis; reconstructed from the GEMMA logs.
# Working folder: $G_ROS_RESULTS_DIR/identity_checks/gwas_c21.
set -e
PY=${PYTHON:-python}; GEMMA=${GEMMA:-gemma}
T=${G_ROS_THREADS:-2}; export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
cd "$(dirname "$0")"
W=$($PY -c "import sys; sys.path.insert(0, '../..'); import config; print(config.out_dir(config.WORK, 'identity_checks', 'gwas_c21'))")
cd "$W"
for k in 1 2 3; do
  "$GEMMA" -bfile geno -n $k -lmm 4 -k relatedness_matrix.txt -o C21_col$k > run_col$k.stdout
done
