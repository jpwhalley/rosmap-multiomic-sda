# Source-data views

Caches of the SDA inputs and loadings, the proteomics mapping checks, and the statistics of the source-data views: batch tests, the C74 intermediate donors and the named features of the donor-concentrated components.

**Inputs**: `G_ROS_ROSMAP_DIR` (SDA inputs and loadings, clinical and assay metadata, TMT matrix, pseudobulk counts), `G_ROS_TABLES_DIR/cell_type_loadings_corrected.csv`, `G_ROS_RESULTS_DIR/revision_r2/donor_celltype_counts_from_h5ad.csv` (`covariates/nuclei_counts/`), `G_ROS_CATALOGUE_DIR/samples_final.csv`; `named/` also reads the identity key and the fixed-loading caches.
**Outputs** (under `G_ROS_RESULTS_DIR`): `{verify_source_panels,source_panels}/{cache,results}` (the caches hold the SDA inputs and loadings) and `v23_review/points_1_3_6/out`.

**Order**
1. `verify_source_panels/01_load.py` (caches read by `concentration/`, `fixed_loading/` and others), `03_tmt_mapping.py`, `08_circularity_null.py`.
2. `source_panels/01_extract.py`, `01b_extract_random.py`, `02_meta.py`, `05f_corrected_prot.py`, `07_c177_depth_check.py`.
3. `verify_source_panels/09_batch.py`, `12_c74_intermediate.py`.
4. After `identity/` and `fixed_loading/`: `named/p6_named_features.py`, `named/p3_completeness.py`.

**Manuscript**: TMT columns matched to the wrong study and the corrected protein rows (`03_tmt_mapping.py`, `05f_corrected_prot.py`; Methods: Proteomics); batch and cohort tests with BH within each component (`09_batch.py`; Methods: Source-data views); the four C74 donors between the sex clusters (`12_c74_intermediate.py`); IGKC (C351), the chromosome 22 H3K9ac peaks (C2) and VTCN1 (C87) without the dominant donor, and the IGKC ranks of the C351 donor (`p6_named_features.py`; Results: Donor-concentrated components); donors with all four layers (`p3_completeness.py`).

The per-panel statistics of the figures are computed by the figure code (`scripts/figures/`).
