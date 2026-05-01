"""
01_prepare_snrna.py
===================

Build the three-dimensional gene-expression tensor from the published ROSMAP
snRNA-seq AnnData object.

Manuscript: Whalley J.P. et al., *Multi-omic Glial Programs in Brain Aging*
(under review). Methods §"Single-Nucleus RNA-Sequencing".

Original filename in the working tree: `ros1a_prepare_data_sc.py`.

Pipeline:
1. Load the Mathys 2023 ROSMAP DLPFC AnnData.
2. Drop two QC-failing samples (projids `21157370`, `79590778`) and all
   computationally predicted doublets.
3. Restrict to the six major cell types (Ast, Oli, Inh, OPC, Exc, Mic) and to
   participants with ≥ 100 cells in *each* of those types (yields 276
   participants).
4. For each cell type, generate a participant × gene pseudobulk by summing raw
   counts across cells of that type within each participant.
5. Apply a gene filter (≥ 10 counts in ≥ 10 pseudobulk samples) on the
   concatenated raw-count table.
6. Re-pseudobulk the surviving genes and library-size normalise to RPM
   (Counts Per Million scaled by 1e6).
7. Write per-cell-type CSVs to `pseudobulk/`.

Outputs (in `pseudobulk/`):
- `<cell_type>_raw_counts.csv` — pre-filter raw pseudobulk counts (one CSV
  per cell type, rows = projids, cols = all genes).
- `<cell_type>_rpm_normalized.csv` — post-filter RPM-normalised pseudobulks
  (one CSV per cell type, rows = projids, cols = the genes that survive the
  ≥10-counts-in-≥10-samples filter).

The resulting six `*_rpm_normalized.csv` files together form the (276 × 6 ×
25,098) tensor input that `05_export_sda_inputs.py` concatenates and writes
out as `rna.txt` for SDA.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import scanpy as sc

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. The doublet-labelled AnnData is
# produced by the prep step described in `_document_doublet_labelled_anndata`
# below; that step was run once on the raw Mathys 2023 download (Synapse
# syn52293417) and the labelled AnnData was cached.
# ----------------------------------------------------------------------------
INPUT_ANNDATA = "mit_all_doublets_labelled.h5ad"   # output of doublet-labelling step
PSEUDOBULK_DIR = "mit/rna/pseudobulk/"              # output dir; 05_export_sda_inputs.py reads from here

# ----------------------------------------------------------------------------
# Constants from the manuscript Methods
# ----------------------------------------------------------------------------
DESIRED_CELL_TYPES = ["Exc", "Oli", "Inh", "Ast", "OPC", "Mic"]
QC_FAILED_PROJIDS = [21157370, 79590778]
MIN_CELLS_PER_TYPE = 100        # ≥ 100 cells per cell type per participant
GENE_FILTER_MIN_COUNTS = 10     # ≥ 10 counts...
GENE_FILTER_MIN_SAMPLES = 10    # ...in ≥ 10 pseudobulk samples


# ----------------------------------------------------------------------------
# Doublet labelling (already-done step, kept here as documentation)
# ----------------------------------------------------------------------------
def _document_doublet_labelled_anndata():
    """Run scrublet per projid on the raw Mathys 2023 AnnData and write the
    labelled object out. This step was run once upstream of the rest of this
    script; its output (`mit_all_doublets_labelled.h5ad`) is the input for
    the main pipeline below.

    Pseudocode (the original block run once on the raw download):

        adata = sc.read('PFC427_raw_data.h5ad')            # raw Mathys 2023
        adata.obs['doublet_scores'] = np.nan
        adata.obs['predicted_doublets'] = False
        for projid in adata.obs['projid'].unique():
            subset = adata[adata.obs['projid'] == projid]
            scrub = scr.Scrublet(subset.X)
            scores, calls = scrub.scrub_doublets()
            adata.obs.loc[adata.obs['projid'] == projid, 'doublet_scores'] = scores
            adata.obs.loc[adata.obs['projid'] == projid, 'predicted_doublets'] = calls
        adata.write('mit_all_doublets_labelled.h5ad')

    Re-running this requires `scrublet` and the raw Mathys 2023 AnnData;
    it is a separate concern from the rest of this pipeline and is provided
    here for completeness only. The cached labelled object is what the rest
    of this script consumes.
    """
    raise NotImplementedError(
        "This function documents the doublet-labelling step that was run "
        "upstream. Re-implement against the raw Mathys 2023 AnnData if you "
        "need to regenerate `mit_all_doublets_labelled.h5ad`."
    )


# ----------------------------------------------------------------------------
# Pseudobulk helper — used twice: once for the raw-counts pass that drives
# the gene filter, once for the RPM-normalised post-filter pass.
# ----------------------------------------------------------------------------
def pseudobulk_per_celltype(
    adata,
    cell_type: str,
    gene_subset: list[str] | None = None,
    rpm_normalise: bool = False,
) -> pd.DataFrame:
    """Sum gene counts across all cells of `cell_type` per projid.

    Returns a (n_donors × n_genes) DataFrame, indexed by projid.
    If `gene_subset` is given, only those genes are summed. If
    `rpm_normalise` is True, each donor's vector is scaled to counts per
    million.

    Note: this loop was originally run on a HPC node with ~100 GB RAM.
    The full Mathys 2023 AnnData plus the accumulating per-cell-type
    DataFrames will OOM-kill on a typical workstation; see the README
    for compute-resource notes on the scripts pipeline.
    """
    rows = []
    gene_names = pd.Index(adata.var.gene_names)
    for projid in adata.obs["projid"].unique():
        mask = (
            (adata.obs["projid"] == projid)
            & (adata.obs["major_cell_type"] == cell_type)
        ).to_numpy()
        counts = np.asarray(adata.X[mask].sum(axis=0)).squeeze()
        if gene_subset is not None:
            keep_genes = gene_names.isin(gene_subset)
            counts = counts[keep_genes]
            cols = gene_names[keep_genes]
        else:
            cols = gene_names
        if rpm_normalise:
            total = counts.sum()
            counts = (counts / total) * 1e6 if total > 0 else counts
        rows.append(pd.DataFrame(counts.reshape(1, -1), index=[projid], columns=cols))
    return pd.concat(rows, axis=0)


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main() -> None:
    sc.settings.verbosity = 3
    sc.logging.print_header()

    # ------------------------------------------------------------------------
    # 1. Load doublet-labelled AnnData and drop QC failures + doublets
    # ------------------------------------------------------------------------
    adata = sc.read(INPUT_ANNDATA)
    adata = adata[~adata.obs["projid"].isin(QC_FAILED_PROJIDS)]
    adata = adata[adata.obs["predicted_doublets"] == False]  # noqa: E712
    adata = adata[adata.obs["major_cell_type"].isin(DESIRED_CELL_TYPES)]

    # ------------------------------------------------------------------------
    # 2. Per-projid cell-type counts; require ≥ MIN_CELLS_PER_TYPE in each of
    #    the six cell types
    # ------------------------------------------------------------------------
    counts_per_type = (
        adata.obs.groupby("projid")["major_cell_type"]
        .value_counts()
        .unstack(fill_value=0)
    )
    excluded = {
        projid: [ct for ct in DESIRED_CELL_TYPES if row[ct] < MIN_CELLS_PER_TYPE]
        for projid, row in counts_per_type.iterrows()
        if any(row[ct] < MIN_CELLS_PER_TYPE for ct in DESIRED_CELL_TYPES)
    }
    valid_projids = (
        counts_per_type[counts_per_type >= MIN_CELLS_PER_TYPE]
        .dropna(axis=0, how="any")
        .index
    )
    adata = adata[adata.obs["projid"].isin(valid_projids)]

    print(f"Excluded {len(excluded)} projid(s) for failing the "
          f"≥{MIN_CELLS_PER_TYPE}-cells-per-type criterion:")
    for projid, types in excluded.items():
        print(f"  projid {projid}: failed cell types {types}")
    print(f"Retained {len(valid_projids)} projids × {len(DESIRED_CELL_TYPES)} cell types.")

    os.makedirs(PSEUDOBULK_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # 3. Raw-count pseudobulks (one CSV per cell type) — used to derive the
    #    gene filter
    # ------------------------------------------------------------------------
    raw_pseudobulks: list[pd.DataFrame] = []
    for cell_type in DESIRED_CELL_TYPES:
        pb = pseudobulk_per_celltype(adata, cell_type, rpm_normalise=False)
        safe_name = cell_type.replace(":", "_")
        pb.to_csv(os.path.join(PSEUDOBULK_DIR, f"{safe_name}_raw_counts.csv"))
        raw_pseudobulks.append(pb)
        print(f"  wrote {safe_name}_raw_counts.csv  ({pb.shape})")

    # ------------------------------------------------------------------------
    # 4. Gene filter: ≥ GENE_FILTER_MIN_COUNTS counts in
    #    ≥ GENE_FILTER_MIN_SAMPLES pseudobulk samples (concatenating across
    #    cell types so a gene that is informative anywhere passes the filter)
    # ------------------------------------------------------------------------
    pooled = pd.concat(raw_pseudobulks, axis=0)
    genes_to_keep = (
        pooled.columns[
            pooled.ge(GENE_FILTER_MIN_COUNTS).sum(axis=0) >= GENE_FILTER_MIN_SAMPLES
        ].tolist()
    )
    print(f"Gene filter retains {len(genes_to_keep)} of {pooled.shape[1]} genes "
          f"(≥{GENE_FILTER_MIN_COUNTS} counts in ≥{GENE_FILTER_MIN_SAMPLES} pseudobulks).")

    # ------------------------------------------------------------------------
    # 5. RPM-normalised pseudobulks on the filtered gene set
    # ------------------------------------------------------------------------
    for cell_type in DESIRED_CELL_TYPES:
        pb = pseudobulk_per_celltype(
            adata, cell_type, gene_subset=genes_to_keep, rpm_normalise=True
        )
        safe_name = cell_type.replace(":", "_")
        pb.to_csv(os.path.join(PSEUDOBULK_DIR, f"{safe_name}_rpm_normalized.csv"))
        print(f"  wrote {safe_name}_rpm_normalized.csv  ({pb.shape})")


if __name__ == "__main__":
    main()
