# Sample identity checks

Calls sex from each assay (snRNA-seq, H3K9ac, 450K, TMT, snATAC) and from the genotypes, and scores every profile against all 1,708 genotyped participants through QTLs discovered with cross-fitting. Flagged donors get pseudonymous labels (`RNA-1`, `CHIP-1`, ...) in a private key; printed output uses labels and counts only.

**Inputs**: `G_ROS_ROSMAP_DIR` (SDA inputs, clinical, biospecimen and assay metadata, MIT metadata and snATAC fragments, ChIP read counts and QC, 450K raw and imputed data, TMT matrix, array genotypes), `G_ROS_AUDIT_DIR/c547`.
**Outputs**: the donor key (`donor_key.csv`) and per-sample tables in `G_ROS_PRIVATE_DIR`; the PLINK and 450K caches in `G_ROS_RESULTS_DIR/identity_checks/cache`; summaries by assay and by flagged label in `G_ROS_RESULTS_DIR/identity_checks/results`.

**Order**: `00_plink_cache.sh`, `01_rna_chip_sex.py`, `02_genotype_sex_identity.py`, `03a_extract_meth_raw.py`, `03_meth_sex_identity.py`, `01b_key_extra.py`, `04_prot_sex.py`, `05_chip_ids_sex.py`, `06_metadata.py`, `07a_atac_chrcounts.sh`, `07b_atac_sex.py`, `08_dominance_gwas.py`, `09_rna_detail.py`, `10_qtl_identity_rna.py`, `10b_rna_mismatch_structure.py`, `10c_rna_duplicates.py`, `11_qtl_identity_chip_prot.py chip`, `11b_chip_swap_structure.py`, `11_qtl_identity_chip_prot.py prot`, `11c_prot_summary.py`, `01c_key_identity.py`, `03f_meth_best_match.py`, `13_all_donor_table.py`, `14_meqtl_identity_meth.py`, `15_flagged_summary.py`, `16_cross_modal_rna_meth.py`, `17_identity_by_assay.py`. `01b`, `01c`, `10b` extend the donor key. The C21 GWAS without the flagged donors is in `c21/sensitivity/`.

**Manuscript** (Methods: Sample identity checks): 2 snRNA-seq, 4 H3K9ac and, under the corrected mapping, 3 TMT profiles are another person's; all 191 genotyped methylation profiles match (`17_identity_by_assay.py`, `15_flagged_summary.py`); none of these donors dominates a described component (`08_dominance_gwas.py`).

`00_plink_cache.sh` repeats PLINK commands that were run by hand; they are taken from the PLINK logs.
