"""Donor-level snRNA-seq quality covariates for the 276 SDA donors (Reviewer 1, Major 4).

What exists (checked 2026-10-13):
- MIT_ROSMAP_Multiomics_assay_snRNAseq_metadata.csv: RIN, rnaBatch, libraryBatch, sequencingBatch, numberCells,
  medianUMIs, DV200 are EMPTY for every one of the 6,248 specimen rows (only platform/run-type fields are filled).
  So no RIN and no library/sequencing batch is available for this snRNA-seq release.
- Available instead (derived from the processed object exactly as the pseudobulk preparation did):
  * post-QC nuclei per donor in the six classes (revision_r2/donor_celltype_counts_from_h5ad.csv, written by
    nuclei_counts/10_r2_13_composition_marker_sensitivity.py; read from the h5ad
    obs with the ros0 filters: two failed donors removed, predicted doublets removed, >=100 cells per class);
  * pseudobulk library size = total raw UMIs per donor summed over the six class pseudobulks
    (mit/rna/pseudobulk/<class>_raw_counts.csv, all genes, before the gene filter);
  * UMIs per nucleus = library size / nuclei (a per-nucleus capture/RNA-quality proxy).
Writes $G_ROS_RESULTS_DIR/covariates/rna_qc_donor.csv (per donor)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
B = cfg.ROS
OUT = cfg.out_dir(cfg.WORK, "covariates")
CT = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
don = pd.read_csv(f"{B}/sda/samples_final.csv", index_col=0, nrows=1).columns.astype(str)

# 1. confirm the MIT assay metadata has no RIN / batch values
meta = pd.read_csv(f"{B}/mit/meta/MIT_ROSMAP_Multiomics_assay_snRNAseq_metadata.csv")
empty = {c: int(meta[c].notna().sum()) for c in ["RIN", "rnaBatch", "libraryBatch", "sequencingBatch", "numberCells", "medianUMIs", "DV200"]}
print("MIT snRNA-seq assay metadata, non-missing values per QC field (of %d rows):" % len(meta), empty)

# 2. nuclei per donor (six classes, post-QC)
dc = pd.read_csv(f"{cfg.WORK}/revision_r2/donor_celltype_counts_from_h5ad.csv", index_col=0)
dc.index = dc.index.astype(str); dc = dc.loc[don, CT]
# 3. pseudobulk library size per donor
umi = {}
for c in CT:
    x = pd.read_csv(f"{B}/mit/rna/pseudobulk/{c}_raw_counts.csv", index_col=0)
    x.index = x.index.astype(str)
    umi[c] = x.sum(axis=1)
umi = pd.DataFrame(umi).loc[don, CT]
qc = pd.DataFrame({"nuclei": dc.sum(axis=1), "umis": umi.sum(axis=1)}, index=don)
qc["umis_per_nucleus"] = qc.umis / qc.nuclei
for c in CT:
    qc[f"nuclei_{c}"] = dc[c]
print("missing:", int(qc.isna().sum().sum()))
print(qc[["nuclei", "umis", "umis_per_nucleus"]].describe().round(1).to_string())
print("Spearman between QC covariates:\n", qc[["nuclei", "umis", "umis_per_nucleus"]].corr("spearman").round(2).to_string())
qc.to_csv(f"{OUT}/rna_qc_donor.csv")
