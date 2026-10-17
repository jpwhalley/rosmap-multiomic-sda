"""
05_export_sda_inputs.py
=======================

Concatenate the four prepared per-omic matrices into the input format SDA
expects, write per-omic metadata files, and emit a SLURM batch script that
launches the ten SDA runs.

Manuscript: Whalley J.P. et al., *Multi-omic Analysis of Sex and Alzheimer's
Disease in the Aged Human Cortex* (under review at npj Aging).
Methods §"Tensor and matrix decomposition".

Original filename in the working tree: `ros1_export_data.py`.

Pipeline:
1. Read the per-cell-type RPM-normalised pseudobulks from `01_prepare_snrna`
   and concatenate row-wise into `rna.txt` (an (n_cell_types × n_samples) ×
   n_genes table — SDA reads it as a tensor of (samples × cell_types ×
   genes) given the row-major concatenation order).
2. Transpose and write the methylation, ChIP, and proteomics matrices into
   the two-dimensional (samples × features) layout SDA expects, as
   `meth.txt`, `chip.txt`, `prot.txt`.
3. Write small metadata files (`md_samples.txt`, `genes.txt`, `cell_types.txt`,
   `md_cpg.txt`, `md_chip.txt`, `md_prot.txt`) carrying the row/column
   labels so `07_cluster_components.py` can reattach them to SDA's numeric
   output.
4. Emit `matd.sh`, a SLURM array script that runs SDA ten times under
   `SLURM_ARRAY_TASK_ID = 1..10`, each into its own `results${id}/`
   directory. (See scripts/README.md, step 06. The runs were launched with
   `--max_iter 3000`; each was stopped by the cluster's 10-day time limit
   after 1,803–2,006 iterations, and the components were taken from the
   iteration-1,500 checkpoint, the last one common to all ten runs. The
   script passes no seed option, so all ten runs use the same seed; the
   logs record `random seed: 44610 90489`.)

Outputs (in `OUTPUT_DIR`):
- `rna.txt`, `meth.txt`, `chip.txt`, `prot.txt`
- `md_samples.txt`, `genes.txt`, `cell_types.txt`, `md_cpg.txt`,
  `md_chip.txt`, `md_prot.txt`
- `matd.sh`
"""
from __future__ import annotations
import os
import pandas as pd

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. The four prepared matrices below come
# from scripts 01–04. The pseudobulk directory must contain one
# `*_rpm_normalized.csv` per cell type in `CELL_TYPE_ORDER`.
# ----------------------------------------------------------------------------
PSEUDOBULK_DIR = "mit/rna/pseudobulk/"            # output of 01_prepare_snrna.py
METH_INPUT = "oth/meth/meth_crossed_with_sc.csv"  # output of 03_prepare_meth.py
CHIP_INPUT = "oth/chip/chip_crossed_with_sc.csv"  # output of 02_prepare_chip.py
PROT_INPUT = "prot/tmt/prot_crossed_with_sc.csv"  # output of 04_prepare_proteomics.py

OUTPUT_DIR = "sda/"
CELL_TYPE_ORDER = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]  # matches the deposited treatment/cell-type matrix

# ----------------------------------------------------------------------------
# SDA invocation parameters (used to compose the SLURM batch script). These
# match the manuscript Methods §"Tensor and matrix decomposition".
# ----------------------------------------------------------------------------
SDA_BINARY = os.environ.get("G_ROS_SDA_BINARY", "sda")  # SDA v1.1 executable; adapt to your install
SDA_NUM_COMPS = 2000
SDA_NUM_BLOCKS = 8
SDA_NUM_OPENMP_THREADS = 12
SDA_SAVE_FREQ = 500
SDA_MAX_ITER = 3000                                # the runs were stopped by the time limit first; components taken at it1500 (07_cluster_components.py)
SDA_NUM_RUNS = 10                                  # SLURM array size

# Cluster-side base directory used to compose absolute paths in matd.sh
# (G_ROS_CLUSTER_WORKDIR; default the current directory). It is written into
# the generated SLURM script verbatim.
CLUSTER_WORKDIR = os.environ.get("G_ROS_CLUSTER_WORKDIR", os.getcwd())
SLURM_ACCOUNT = "combat.prj"
SLURM_PARTITION = "long"
SLURM_JOB_NAME = "ros"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def write_lines(path: str, items) -> None:
    """Write each item as its own line of `path`."""
    with open(path, "w") as f:
        for item in items:
            f.write(f"{item}\n")


def export_2d_matrix(matrix_path: str, output_path: str, samples_order: list) -> None:
    """Read a (features × samples) matrix CSV, transpose to (samples ×
    features), align columns to `samples_order`, and write as a
    space-separated SDA input (no header, no index, NaN as `NA`)."""
    df = pd.read_csv(matrix_path, index_col=0)
    df.columns = df.columns.astype(int)
    df = df[samples_order]
    df.T.to_csv(output_path, header=False, index=False, sep=" ", na_rep="NA")


def compose_slurm_script(output_path: str, n_samples: int) -> None:
    """Compose the SLURM array script that runs SDA `SDA_NUM_RUNS` times."""
    sda_data_args = " ".join(
        f"{CLUSTER_WORKDIR}/sda/{name}.txt"
        for name in ("rna", "meth", "chip", "prot")
    )
    sda_cmd = (
        f"{SDA_BINARY} --data {sda_data_args} "
        f"--N {n_samples} "
        f"--out {CLUSTER_WORKDIR}/sda/results${{SLURM_ARRAY_TASK_ID}} "
        f"--num_openmp_threads {SDA_NUM_OPENMP_THREADS} "
        f"--num_blocks {SDA_NUM_BLOCKS} "
        f"--eigen_parallel true "
        f"--num_comps {SDA_NUM_COMPS} "
        f"--save_freq {SDA_SAVE_FREQ} "
        f"--remove_zero_comps true "
        f"--max_iter {SDA_MAX_ITER} "
        f"--ignore_missing true"
    )
    script = f"""#!/bin/bash

#SBATCH -A {SLURM_ACCOUNT}
#SBATCH -J {SLURM_JOB_NAME}
#SBATCH -o {SLURM_JOB_NAME}-%j.out
#SBATCH -e {SLURM_JOB_NAME}-%j.err
#SBATCH --array 1-{SDA_NUM_RUNS}
#SBATCH -c {SDA_NUM_OPENMP_THREADS}
#SBATCH -p {SLURM_PARTITION}

echo "#######################################################################"
echo `date`: Executing task ${{SLURM_ARRAY_TASK_ID}} of job ${{SLURM_ARRAY_JOB_ID}} on `hostname` as user ${{USER}}
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "#######################################################################"

    mkdir -p {CLUSTER_WORKDIR}/sda/results${{SLURM_ARRAY_TASK_ID}}
export OPENMP_NUM_THREADS={SDA_NUM_OPENMP_THREADS}

{sda_cmd}

echo "#########################################################################"
echo "Finished at: "`date`
echo "#########################################################################"
exit 0
"""
    with open(output_path, "w") as f:
        f.write(script)


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Concatenate per-cell-type RNA pseudobulks into a single tensor-shaped
    #    table (rows = cell_type × sample in row-major order, cols = genes).
    # ------------------------------------------------------------------------
    cell_types: list[str] = []
    per_celltype_dfs: list[pd.DataFrame] = []

    samples: list | None = None
    genes: list | None = None
    for cell_type in CELL_TYPE_ORDER:
        fname = f"{cell_type.replace(':', '_')}_rpm_normalized.csv"
        path = os.path.join(PSEUDOBULK_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected pseudobulk file for cell type {cell_type!r}: {path}"
            )
        df = pd.read_csv(path, index_col=0)
        cell_types.append(cell_type)
        per_celltype_dfs.append(df)
        print(f"  loaded {fname}: {df.shape}  (cell type {cell_type!r})")

        # Sanity-check that all per-cell-type pseudobulks share the same rows
        # (samples) and columns (genes); abort loudly if they diverge.
        if samples is None:
            samples = df.index.tolist()
            genes = df.columns.tolist()
        else:
            if df.index.tolist() != samples:
                raise ValueError(
                    f"Sample order mismatch in {fname}; expected {samples[:5]}…"
                )
            if df.columns.tolist() != genes:
                raise ValueError(f"Gene order mismatch in {fname}")

    if not per_celltype_dfs:
        raise FileNotFoundError(
            f"No `*_rpm_normalized.csv` files found in {PSEUDOBULK_DIR}; "
            f"did `01_prepare_snrna.py` run?"
        )

    rna_concat = pd.concat(per_celltype_dfs, axis=0)
    rna_concat.to_csv(
        os.path.join(OUTPUT_DIR, "rna.txt"),
        header=False,
        index=False,
        sep=" ",
        na_rep="NA",
    )
    print(f"  wrote sda/rna.txt: {rna_concat.shape}")

    # ------------------------------------------------------------------------
    # 2. Per-omic 2D matrices (samples × features), with NaN preserved
    # ------------------------------------------------------------------------
    export_2d_matrix(METH_INPUT, os.path.join(OUTPUT_DIR, "meth.txt"), samples)
    export_2d_matrix(CHIP_INPUT, os.path.join(OUTPUT_DIR, "chip.txt"), samples)
    export_2d_matrix(PROT_INPUT, os.path.join(OUTPUT_DIR, "prot.txt"), samples)
    print("  wrote sda/{meth,chip,prot}.txt")

    # ------------------------------------------------------------------------
    # 3. Metadata sidecars (so 07_cluster_components can reattach labels)
    # ------------------------------------------------------------------------
    write_lines(os.path.join(OUTPUT_DIR, "md_samples.txt"), samples)
    write_lines(os.path.join(OUTPUT_DIR, "genes.txt"), genes)
    write_lines(os.path.join(OUTPUT_DIR, "cell_types.txt"), cell_types)

    meth = pd.read_csv(METH_INPUT, index_col=0, usecols=[0])
    write_lines(os.path.join(OUTPUT_DIR, "md_cpg.txt"), meth.index)

    chip = pd.read_csv(CHIP_INPUT, index_col=0, usecols=[0])
    write_lines(os.path.join(OUTPUT_DIR, "md_chip.txt"), chip.index)

    prot = pd.read_csv(PROT_INPUT, index_col=0, usecols=[0])
    write_lines(os.path.join(OUTPUT_DIR, "md_prot.txt"), prot.index)
    print("  wrote sda/md_{samples,cpg,chip,prot}.txt + genes.txt + cell_types.txt")

    # ------------------------------------------------------------------------
    # 4. SLURM array submission script
    # ------------------------------------------------------------------------
    compose_slurm_script(os.path.join(OUTPUT_DIR, "matd.sh"), n_samples=len(samples))
    print(f"  wrote sda/matd.sh (SLURM array, {SDA_NUM_RUNS} runs)")


if __name__ == "__main__":
    main()
