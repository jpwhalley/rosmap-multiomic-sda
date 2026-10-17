# Covariate-adjusted phenotype models

Reproduces the catalogue-wide Mann-Whitney / Kruskal-Wallis tests and fits covariate-adjusted models for all 547 components: proportional-odds (cumulative-logit) LRTs (PO0 no covariates, PO1 age, PMI, cohort and sex, PO2 + log nuclei and log UMIs per nucleus, PO3 alternative coding) and OLS on the rank-based inverse-normal score. `check/` is an independent implementation.

**Inputs**: `G_ROS_ROSMAP_DIR` (scores, clinical, snRNA-seq object and pseudobulk counts, assay metadata), `G_ROS_AUDIT_DIR/c547`, `G_ROS_TABLES_DIR/clinical_BH_547_family.csv`, `G_ROS_DATA_S1_V22`, `G_ROS_QC_EXCLUDE_PATH`.
**Outputs** (under `G_ROS_RESULTS_DIR`): `covariates/` (component-level tables and the per-donor QC table `rna_qc_donor.csv`), `verify_covariates/`, `revision_r2/` (nuclei per donor and class).

**Order**: `nuclei_counts/10_r2_13_composition_marker_sensitivity.py` (nuclei per donor and class from the snRNA-seq object; also used by `source_data/` and `dossier/check/`), `models/00` to `models/04` in order, then `check/01_reproduce_and_po.py`, `03_central_five.py`, `04_loo.py Sex`, `04_loo.py ceradsc`, `04_loo.py cogdx`, `05b_modality_batch.py` and `t1_tmt_mapping_check.py`.

**Manuscript** (Methods: Association with phenotypes; Results: Decomposition structure): 57 of the 73 associations remain at FDR < 0.05 after adjustment and all 73 keep their direction (`models/03_po_adjusted.py`, `po_tests_547.csv`, read by `concentration/` and Data S1 `*_adj_q`); for C278 the FDR ranges 0.0076 to 0.021 (proportional odds) and 0.028 to 0.088 (rank-based) across models and single-donor removals, and C277 reaches 0.084 with the most extreme donor removed (`check/03_central_five.py`, `check/04_loo.py`); TMT mapping counts (126 shared identifiers, 112 of 400 columns, 29 of 105 proteomes, 28 left out; `check/t1_tmt_mapping_check.py`; Methods: Proteomics).
