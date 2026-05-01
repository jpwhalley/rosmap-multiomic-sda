"""
08_gwas_preprocessing.py
========================

Prepare ROSMAP genotype data for the per-component GWAS by (a) restricting
the published genotype set to the snRNA-seq cohort, (b) computing a kinship
matrix with KING, and (c) writing a `.fam` file whose phenotype columns
carry the per-component sample loading scores from
`07_cluster_components.py`. The downstream GWAS itself was run with GEMMA
on the cluster (not in this script).

Manuscript: Whalley J.P. et al., *Multi-omic Glial Programs in Brain Aging*
(under review). Methods §"GWAS" (downstream of the SDA decomposition).

Original filename in the working tree: `ros4_GWAS_association_study_maf.py`.

External dependencies:
- PLINK 1.07 (or compatible): https://www.cog-genomics.org/plink/
- KING 2.0.9 (or compatible): http://people.virginia.edu/~wc9c/KING/
- ROSMAP genotype `.bed/.bim/.fam` (controlled access via DUC).

Pipeline:
1. Read the SDA cohort's `md_samples.txt` (the 276 projids), and the full
   ROSMAP genotype `.fam` file (whose ID column is `<Study><projid>`,
   e.g. `ROS50410319`).
2. Identify the genotype-cohort entries whose ID strips to a snRNA-seq
   projid; write them as `samples_in_common.txt`.
3. Run PLINK to subset the genotype set to those samples (`--keep`,
   `--make-bed`).
4. Run KING on the subsetted genotypes to compute a relatedness matrix,
   then build a square (n × n) symmetric kinship matrix in the order of
   the new `.fam` and write to `relatedness_matrix.txt`.
5. Read `samples_final.csv` (component × sample) and append the per-
   component sample loadings as phenotype columns of the `.fam` so GEMMA
   can run a GWAS per component using `--pheno-col`.

Outputs:
- `oth/snps/samples_in_common.txt`
- `oth/snps/ROSMAP_arrayGenotype_maf.{bed,bim,fam}` (subsetted)
- `oth/snps/gemma/relatedness_matrix.txt`
"""
from __future__ import annotations
import os
import subprocess

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. ROSMAP raw genotype data are obtained
# from the AD Knowledge Portal under DUC.
# ----------------------------------------------------------------------------
SAMPLES_IN_COMMON_TXT = "sda/md_samples.txt"           # output of 05_export_sda_inputs.py
GENOTYPE_PREFIX = "oth/snps/ROSMAP_arrayGenotype"      # original .bed/.bim/.fam prefix
SUBSET_PREFIX = "oth/snps/ROSMAP_arrayGenotype_maf"    # PLINK --out target
CLINICAL_CSV = "oth/meta/ROSMAP_clinical.csv"          # syn3219045/Data/Metadata
SAMPLES_FINAL_CSV = "results/samples_final.csv"        # output of 07_cluster_components.py

OUTPUT_DIR = "oth/snps/"
GEMMA_DIR = "oth/snps/gemma/"

# External binaries — adapt to your install
PLINK_BIN = "/apps/well/plink/1.07/plink"
KING_BIN = "/apps/well/KING/2.0.9/./king"


def main() -> None:
    os.makedirs(GEMMA_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Cross snRNA-seq cohort against the genotype .fam
    # ------------------------------------------------------------------------
    samps_in_common = open(SAMPLES_IN_COMMON_TXT).read().splitlines()
    fam = pd.read_csv(f"{GENOTYPE_PREFIX}.fam", sep=r"\s+", header=None, index_col=1)
    fam.columns = ["FID", "PatID", "MatID", "Sex", "Phenotype"]

    clin = pd.read_csv(CLINICAL_CSV, index_col=0)
    clin_combined = clin["Study"].astype(str) + clin.index.astype(str)

    mismatches = fam.index[~fam.index.isin(clin_combined)]
    if not mismatches.empty:
        print(f"WARNING: {len(mismatches)} fam rows do not match any "
              f"<Study><projid> string in clinical metadata: {list(mismatches)[:5]}…")
    else:
        print("All fam rows match a <Study><projid> in clinical metadata.")

    # Strip the leading 'ROS' or 'MAP' prefix to test against snRNA-seq projids
    in_common = fam.index[
        fam.index.str.replace(r"^(ROS|MAP)", "", regex=True).isin(samps_in_common)
    ]
    keep_path = os.path.join(OUTPUT_DIR, "samples_in_common.txt")
    with open(keep_path, "w") as f:
        for samp in in_common:
            f.write(f"{fam.FID.loc[samp]}\t{samp}\n")
    print(f"  wrote {keep_path}: {len(in_common)} samples in common with snRNA-seq")

    # ------------------------------------------------------------------------
    # 2. PLINK subset
    # ------------------------------------------------------------------------
    plink_cmd = (
        f"{PLINK_BIN} --bfile {GENOTYPE_PREFIX} "
        f"--keep {keep_path} --make-bed --out {SUBSET_PREFIX}"
    )
    print(f"  running: {plink_cmd}")
    subprocess.run(plink_cmd, shell=True, check=True)

    n_snps = int(subprocess.check_output(["wc", "-l", f"{SUBSET_PREFIX}.bim"]).split()[0])
    print(f"  retained {n_snps} SNPs after subsetting")

    # ------------------------------------------------------------------------
    # 3. KING kinship
    # ------------------------------------------------------------------------
    king_cmd = (
        f"{KING_BIN} -b {SUBSET_PREFIX}.bed --kinship "
        f"--prefix {os.path.join(GEMMA_DIR, 'relatedness')}"
    )
    print(f"  running: {king_cmd}")
    subprocess.run(king_cmd, shell=True, check=True)

    # KING writes a long-format pairwise file (relatedness.kin0). Build a
    # square symmetric kinship matrix from it, in the .fam row order, with
    # 0.5 on the diagonal (self-kinship).
    fam_subset = open(f"{SUBSET_PREFIX}.fam").read().splitlines()
    long_samps = [line.split()[1] for line in fam_subset]
    n = len(long_samps)
    samp_idx = {s: i for i, s in enumerate(long_samps)}

    relate_array = np.zeros((n, n), dtype=float)
    for i in range(n):
        relate_array[i, i] = 0.5

    kin_lines = open(os.path.join(GEMMA_DIR, "relatedness.kin0")).read().splitlines()
    for line in kin_lines[1:]:  # skip header
        parts = line.split("\t")
        a, b, kin = parts[1], parts[3], float(parts[7])
        if a in samp_idx and b in samp_idx:
            i, j = samp_idx[a], samp_idx[b]
            relate_array[i, j] = kin
            relate_array[j, i] = kin

    relate_path = os.path.join(GEMMA_DIR, "relatedness_matrix.txt")
    with open(relate_path, "w") as f:
        for row in relate_array:
            f.write("\t".join(str(v) for v in row) + "\n")
    print(f"  wrote {relate_path}: {n} × {n} kinship matrix")

    # ------------------------------------------------------------------------
    # 4. Append per-component sample loadings as phenotype columns to the
    #    subsetted .fam, so GEMMA can run one GWAS per component (using
    #    --pheno-col on the column index).
    # ------------------------------------------------------------------------
    samples_final = pd.read_csv(SAMPLES_FINAL_CSV, index_col=0).T
    print(f"  appending {samples_final.shape[1]} component phenotypes to .fam")

    # Preserve a backup of the original .fam (PLINK overwrote it with a 6-column
    # version; we want the original for re-running).
    backup = f"{SUBSET_PREFIX}_before.fam"
    if not os.path.exists(backup):
        os.rename(f"{SUBSET_PREFIX}.fam", backup)

    src_lines = open(backup).read().splitlines()
    with open(f"{SUBSET_PREFIX}.fam", "w") as f:
        for line in src_lines:
            # Preserve the first five FAM fields, then replace the original
            # single phenotype column with the 546 component scores.
            parts = line.split()
            samp_id = parts[1]
            if samp_id.startswith(("ROS", "MAP")):
                samp_id = samp_id[3:]
            comp_scores = [str(score) for score in samples_final.loc[samp_id]]
            f.write(" ".join(parts[:5] + comp_scores) + "\n")
    print(f"  wrote {SUBSET_PREFIX}.fam with component phenotypes")


if __name__ == "__main__":
    main()
