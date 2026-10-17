#!/bin/bash
# GEMMA 0.96 univariate LMM (-lmm 4) for C21, C74, C351 (validation against the archived runs) and C547, C548 (new),
# with the archived KING relatedness matrix. Run by hand in the 10-08 audit; the commands are reconstructed from the
# GEMMA logs ("Command Line Input"). Run after 01_build_fam.py. Working folder: $G_ROS_AUDIT_DIR/gwas547.
set -e
PY=${PYTHON:-python}; GEMMA=${GEMMA:-gemma}
T=${G_ROS_THREADS:-2}; export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
cd "$(dirname "$0")"
SNP=$($PY -c "import sys; sys.path.insert(0, '../..'); import config; print(config.SNP)")
W=$($PY -c "import sys; sys.path.insert(0, '../..'); import config; print(config.out_dir(config.AUD, 'gwas547'))")
cd "$W"
ln -sf "$SNP/ROSMAP_arrayGenotype_maf.bed" geno547.bed
ln -sf "$SNP/ROSMAP_arrayGenotype_maf.bim" geno547.bim
ln -sf "$SNP/gemma/relatedness_matrix.txt" relatedness_matrix.txt
for nc in "1 C21" "2 C74" "3 C351" "4 C547" "5 C548"; do
  set -- $nc
  "$GEMMA" -bfile geno547 -n "$1" -lmm 4 -k relatedness_matrix.txt -o "local_$2" > "run_$2.stdout"
done
