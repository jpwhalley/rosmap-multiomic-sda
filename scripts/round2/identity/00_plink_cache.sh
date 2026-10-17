#!/bin/bash
# PLINK 1.9 (v1.90b7.2) genotype cache for the identity checks. These commands were run by hand in the round-2 analysis;
# they are reconstructed here from the PLINK logs ("Options in effect"), in the order run. --threads (G_ROS_THREADS,
# default 2) is the only addition. Output: $G_ROS_RESULTS_DIR/identity_checks/cache/plink/ (per-person genotypes).
# 02_genotype_sex_identity.py reads genome_full.genome; qtl_common.py reads auto_qc.{bed,bim,fam}.
set -e
PY=${PYTHON:-python}
cd "$(dirname "$0")"
ROS=$($PY -c "import sys; sys.path.insert(0, '..'); import config; print(config.ROS)")
OUT=$($PY -c "import sys; sys.path.insert(0, '..'); import config; print(config.out_dir(config.WORK, 'identity_checks', 'cache', 'plink'))")
G=$ROS/oth/snps/ROSMAP_arrayGenotype
T="--threads ${G_ROS_THREADS:-2}"
awk 'BEGIN{OFS=" "}{$5=0; print}' "$G.fam" > "$OUT/full_nosex.fam"     # recorded sex set to unknown (0)
NS="--bed $G.bed --bim $G.bim --fam $OUT/full_nosex.fam"
plink --allow-no-sex --bfile "$G" --check-sex ycount 0.2 0.8 5 50 --out "$OUT/full_checksex" $T
plink --allow-no-sex $NS --check-sex ycount 0.2 0.8 5 50 --out "$OUT/full_nosex_checksex" $T
plink --allow-no-sex $NS --chr 23,24 --missing --out "$OUT/full_nosex_xy" $T
plink --allow-no-sex $NS --chr 23 --missing --out "$OUT/full_nosex_x" $T
plink --allow-no-sex $NS --chr 23 --het --out "$OUT/full_nosex_xhet" $T
plink --allow-no-sex $NS --chr 24 --missing --out "$OUT/full_nosex_y" $T
plink --allow-no-sex $NS --chr 1-22 --geno 0.05 --maf 0.05 --indep-pairwise 200 50 0.2 --out "$OUT/prune" $T
plink --allow-no-sex $NS --extract "$OUT/prune.prune.in" --genome --min 0.1 --out "$OUT/genome_full" $T
plink --allow-no-sex $NS --extract "$OUT/prune.prune.in" --missing --out "$OUT/prune_miss" $T
plink --allow-no-sex $NS --chr 1-22 --geno 0.05 --maf 0.05 --make-bed --out "$OUT/auto_qc" $T
