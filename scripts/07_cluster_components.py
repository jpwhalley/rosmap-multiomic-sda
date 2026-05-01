"""
07_cluster_components.py
========================

Pool the components from all ten SDA runs and apply stability-based
clustering to identify the final 546 reproducible components. Produces the
component-loading matrices that go into the Synapse deposit.

Manuscript: Whalley J.P. et al., *Multi-omic Glial Programs in Brain Aging*
(under review). Methods §"Tensor and matrix decomposition".

Original filename in the working tree: `ros2_cluster.py`. The original
runtime was approximately 4 hours wall-clock on a 16-core machine; the
linkage step is the dominant cost (`scipy.cluster.hierarchy.linkage` on
~10,000 pooled components).

Pipeline:
1. For each of `NUM_RUNS` SDA runs (default 10), load the per-iteration
   factor matrices: A (sample loadings), B1 (cell-type loadings; only
   present for the RNA tensor), and X1..X4 / S1..S4 (feature loadings and
   PIPs for RNA / methylation / ChIP / proteomics respectively).
2. Drop components that converged to all-zero (`--remove_zero_comps true`
   was passed to SDA, but a small number of edge cases survive); reindex
   the surviving components into a global numbering across runs.
3. Cluster the pooled components by hierarchical linkage on (1 − |Pearson
   correlation of sample loadings|), flat-cut at cophenetic distance
   `COPHENETIC_THRESHOLD`.
4. Retain only clusters drawing components from `MIN_RUNS_PER_CLUSTER` or
   more of the `NUM_RUNS` independent runs (the manuscript's
   stability criterion).
5. For each retained cluster, sign-align components to the first member
   (flip if Pearson correlation is negative) and average across components
   to produce a single representative loading vector. PIPs are probabilities
   and are therefore aggregated by median without sign alignment.
6. Write the deposited matrices: `samples_final.csv`, `treatment_final.csv`
   (same data later renamed to `cell_type.csv` for the deposit), and per
   omic `genes_final_<omic>.csv` + `pip_final_<omic>.csv`.

Outputs (in `OUTPUT_DIR`):
- `samples_final.csv` — 546 components × 276 samples
- `treatment_final.csv` — 546 components × 6 cell types
- `genes_final_RNA.csv`, `genes_final_meth.csv`, `genes_final_chip.csv`,
  `genes_final_proteins.csv` — feature × component loadings (unmasked)
- `pip_final_RNA.csv`, `pip_final_meth.csv`, `pip_final_chip.csv`,
  `pip_final_proteins.csv` — feature × component PIP values

The PIP > 0.5 mask used in the deposit's `*.txt` files is applied
**after** this script — it is the post-clustering mask that produces
`RNA.txt`, `meth.txt`, `chip.txt`, `proteins.txt` from the unmasked
`genes_final_*.csv` and the `pip_final_*.csv`.

Diagnostic plots (the dendrogram + cluster-size histogram from the original
script) are produced only when `MAKE_DIAGNOSTIC_PLOTS = True`. They are
useful for QC but do not appear in the manuscript.
"""
from __future__ import annotations
import os
from collections import Counter

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
from scipy import stats

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. SDA outputs come from the SLURM array
# submitted by `05_export_sda_inputs.py`. Each run writes into
# `SDA_RESULTS_BASE{i}/it{ITERATION}/{A,B1,X1..X4,S1..S4}` for
# `i = 1..NUM_RUNS`.
# ----------------------------------------------------------------------------
SDA_RESULTS_BASE = "/well/combat/users/bru597/ros/sda/results"  # run i is at f"{SDA_RESULTS_BASE}{i}/"
SDA_INPUT_DIR = "sda/"            # output dir of 05_export_sda_inputs.py; carries the metadata sidecars
OUTPUT_DIR = "results/"           # downstream scripts expect clustered outputs here

# ----------------------------------------------------------------------------
# Run / clustering parameters — match the manuscript Methods.
# ----------------------------------------------------------------------------
NUM_RUNS = 10                       # SDA array size
ITERATION = 1500                    # which iteration's checkpoint to read; matches `--max_iter` in 05_export_sda_inputs.py
COPHENETIC_THRESHOLD = 0.4          # flat-cluster cut on (1 − |Pearson|) dissimilarity
MIN_RUNS_PER_CLUSTER = 5            # keep cluster only if it draws from ≥ 5 of the 10 runs
MAKE_DIAGNOSTIC_PLOTS = False       # toggle to write dendrogram + cluster-size histogram

OMIC_NAMES = {1: "RNA", 2: "meth", 3: "chip", 4: "proteins"}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def correlation_distance(x, y) -> float:
    """1 − |Pearson correlation| as a hierarchical-clustering metric."""
    return 1.0 - abs(stats.pearsonr(x, y)[0])


def load_sda_run(run_dir: str, sample_labels, cell_type_labels, feature_labels: dict) -> dict:
    """Read one SDA run's checkpoint matrices and drop all-zero components.

    Parameters
    ----------
    run_dir
        Directory like `f"{SDA_RESULTS_BASE}{i}/it{ITERATION}/"`.
    sample_labels
        List of sample IDs in the column order SDA wrote them. Used to
        attach column labels to the (n_components × n_samples) `A` matrix.
    cell_type_labels
        List of cell-type names in the column order SDA wrote them. Used
        for the (n_components × n_cell_types) `B1` matrix.
    feature_labels
        Mapping `{1: rna_genes, 2: cpg_islands, 3: chip_peaks, 4: proteins}`.

    Returns
    -------
    dict with keys `samples`, `treatment`, `genes`, `pip`. The values are
    DataFrames whose row index is a generic 0..n_kept_components−1 range
    (caller is responsible for assigning a global index across runs).
    """
    # A: components × samples (transposed because SDA writes samples × components)
    samples = pd.read_csv(os.path.join(run_dir, "A"), sep=" ", header=None).T
    samples.columns = sample_labels
    keep = samples[(samples.T != 0).any()].index.values  # non-zero components
    samples = samples.loc[keep].reset_index(drop=True)

    # B1: cell-type loadings (transposed similarly; only the RNA tensor has a B
    # matrix because B is the third dimension of the snRNA-seq tensor)
    treatment = pd.read_csv(os.path.join(run_dir, "B1"), sep=" ", header=None).T
    treatment.columns = cell_type_labels
    treatment = treatment.loc[keep].reset_index(drop=True)

    # X1..X4 (feature loadings) and S1..S4 (PIP values)
    genes: dict = {}
    pip: dict = {}
    for j in range(1, 5):
        x = pd.read_csv(os.path.join(run_dir, f"X{j}"), sep=" ", header=None)
        x.columns = feature_labels[j]
        genes[j] = x.loc[keep].reset_index(drop=True)

        s = pd.read_csv(os.path.join(run_dir, f"S{j}"), sep=" ", header=None)
        s.columns = feature_labels[j]
        pip[j] = s.loc[keep].reset_index(drop=True)

    return {
        "samples": samples,
        "treatment": treatment,
        "genes": genes,
        "pip": pip,
        "n_kept": len(keep),
    }


def sign_align_and_mean(block: pd.DataFrame) -> pd.Series:
    """Sign-align all rows of `block` to the first row (flip if anti-correlated)
    and return the mean loading vector."""
    if len(block) == 0:
        raise ValueError("Empty block passed to sign_align_and_mean")
    first = block.iloc[0]
    aligned = block.copy()
    for idx in aligned.index[1:]:
        if stats.pearsonr(first.values, aligned.loc[idx].values)[0] < 0:
            aligned.loc[idx] = aligned.loc[idx] * -1
    return aligned.mean(axis=0)


def median_pip(block: pd.DataFrame) -> pd.Series:
    """Aggregate PIP values across a retained cluster without sign alignment."""
    if len(block) == 0:
        raise ValueError("Empty block passed to median_pip")
    return block.median(axis=0)


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Load metadata sidecars produced by 05_export_sda_inputs
    # ------------------------------------------------------------------------
    samps_in_common = open(os.path.join(SDA_INPUT_DIR, "md_samples.txt")).read().splitlines()
    cell_types     = open(os.path.join(SDA_INPUT_DIR, "cell_types.txt")).read().splitlines()
    rna_genes      = open(os.path.join(SDA_INPUT_DIR, "genes.txt")).read().splitlines()
    cpg_islands    = open(os.path.join(SDA_INPUT_DIR, "md_cpg.txt")).read().splitlines()
    chip_peaks     = open(os.path.join(SDA_INPUT_DIR, "md_chip.txt")).read().splitlines()
    proteins       = open(os.path.join(SDA_INPUT_DIR, "md_prot.txt")).read().splitlines()

    feature_labels = {1: rna_genes, 2: cpg_islands, 3: chip_peaks, 4: proteins}

    # ------------------------------------------------------------------------
    # 2. Pool SDA runs. Build a list of per-run dicts then concat at the end —
    #    cleaner than the original first/subsequent-iteration if/else split.
    # ------------------------------------------------------------------------
    per_run_data = []
    indices_per_run: list[list[int]] = []
    next_global_index = 0

    for i in range(1, NUM_RUNS + 1):
        run_dir = f"{SDA_RESULTS_BASE}{i}/it{ITERATION}"
        run = load_sda_run(run_dir, samps_in_common, cell_types, feature_labels)

        # Re-index this run's surviving components into a global numbering
        global_idx = list(range(next_global_index, next_global_index + run["n_kept"]))
        next_global_index += run["n_kept"]

        for key in ("samples", "treatment"):
            run[key].index = global_idx
        for j in range(1, 5):
            run["genes"][j].index = global_idx
            run["pip"][j].index = global_idx

        per_run_data.append(run)
        indices_per_run.append(global_idx)
        print(f"  run {i:>2}: {run['n_kept']} non-zero components retained")

    # Concatenate
    samples = pd.concat([r["samples"] for r in per_run_data], axis=0)
    treatment = pd.concat([r["treatment"] for r in per_run_data], axis=0)
    genes = {j: pd.concat([r["genes"][j] for r in per_run_data], axis=0)
             for j in range(1, 5)}
    pip = {j: pd.concat([r["pip"][j] for r in per_run_data], axis=0)
           for j in range(1, 5)}
    print(f"  pooled: {len(samples)} components across {NUM_RUNS} runs")

    # ------------------------------------------------------------------------
    # 3. Hierarchical clustering on (1 − |Pearson|) of sample loadings
    # ------------------------------------------------------------------------
    Z = hac.linkage(samples, method="complete", metric=correlation_distance)
    cluster_assignments = hac.fcluster(Z, t=COPHENETIC_THRESHOLD, criterion="distance")
    cluster_sizes = Counter(cluster_assignments)
    print(f"  {len(cluster_sizes)} flat clusters at cophenetic ≤ {COPHENETIC_THRESHOLD}")

    # ------------------------------------------------------------------------
    # 4. Filter clusters: must draw from ≥ MIN_RUNS_PER_CLUSTER unique runs
    # ------------------------------------------------------------------------
    samples_final = pd.DataFrame(columns=samples.columns.values)
    treatment_final = pd.DataFrame(columns=treatment.columns.values)
    genes_final = {j: pd.DataFrame(columns=genes[j].columns.values) for j in range(1, 5)}
    pip_final = {j: pd.DataFrame(columns=pip[j].columns.values) for j in range(1, 5)}

    final_index = 1
    for cluster_id, size in cluster_sizes.items():
        # Cheap pre-filter: a cluster cannot draw from more runs than its size
        if size < MIN_RUNS_PER_CLUSTER:
            continue

        cluster_member_indices = set(samples.index.values[cluster_assignments == cluster_id])
        n_unique_runs = sum(
            1 for run_idx_list in indices_per_run
            if cluster_member_indices.intersection(run_idx_list)
        )
        if n_unique_runs < MIN_RUNS_PER_CLUSTER:
            continue

        # Sign-align and aggregate within the cluster
        in_cluster = pd.Series(cluster_assignments == cluster_id, index=samples.index.values)
        samples_final.loc[final_index] = sign_align_and_mean(samples[in_cluster])
        treatment_final.loc[final_index] = sign_align_and_mean(treatment[in_cluster])
        for j in range(1, 5):
            genes_final[j].loc[final_index] = sign_align_and_mean(genes[j][in_cluster])
            pip_final[j].loc[final_index] = median_pip(pip[j][in_cluster])
        final_index += 1

    print(f"  retained {final_index - 1} stable clusters "
          f"(≥ {MIN_RUNS_PER_CLUSTER}/{NUM_RUNS} runs each)")

    # ------------------------------------------------------------------------
    # 5. Write outputs
    # ------------------------------------------------------------------------
    samples_final.to_csv(os.path.join(OUTPUT_DIR, "samples_final.csv"))
    treatment_final.to_csv(os.path.join(OUTPUT_DIR, "treatment_final.csv"))
    for j, name in OMIC_NAMES.items():
        genes_final[j].to_csv(os.path.join(OUTPUT_DIR, f"genes_final_{name}.csv"))
        pip_final[j].to_csv(os.path.join(OUTPUT_DIR, f"pip_final_{name}.csv"))
    print(f"  wrote samples_final / treatment_final / genes_final_* / pip_final_* "
          f"to {OUTPUT_DIR}")

    # ------------------------------------------------------------------------
    # 6. Optional diagnostic plots
    # ------------------------------------------------------------------------
    if MAKE_DIAGNOSTIC_PLOTS:
        import matplotlib.pyplot as plt
        os.makedirs("figs/td/all/", exist_ok=True)

        plt.figure(figsize=(25, 10))
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("sample index")
        plt.ylabel("distance")
        plt.yticks(np.arange(0, 2.01, 0.2))
        hac.dendrogram(Z, leaf_rotation=90.0, leaf_font_size=8.0)
        plt.axhline(y=COPHENETIC_THRESHOLD, c="k")
        plt.savefig("figs/td/all/sample_components_cluster.pdf")
        plt.close()

        sizes = pd.Series(list(cluster_sizes.values()))
        vc = sizes.value_counts().sort_index()
        vc.plot(kind="bar", colormap="ocean")
        plt.xlabel("Cluster size")
        plt.ylabel("Frequency")
        plt.savefig("figs/td/all/no_of_clusters.pdf")
        plt.close()
        print("  wrote diagnostic plots in figs/td/all/")


if __name__ == "__main__":
    main()
