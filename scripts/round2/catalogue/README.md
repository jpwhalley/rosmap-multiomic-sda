# Catalogue

The 2026-10-08 correction of the consensus step: flat clusters of the ten SDA runs, the run count, the cell-type sign rule, the two new components (C547, C548) and the corrected Data S1 (`Data_S1_corrected_547.csv`, 547 rows). `scripts/07_cluster_components.py` implements the same aggregation for the release.

**Inputs**: `G_ROS_ROSMAP_DIR` (`sda/results1..10/it1500`, `it2000` of runs 8 and 10, version-1 `*_final.csv`, SDA inputs), `G_ROS_SEAAD_DIR`, `G_ROS_DATA_S1_V22`.
**Outputs**: `G_ROS_AUDIT_DIR` (`audit/`, `cell_type_evidence/` and `c547/` steps), `G_ROS_TABLES_DIR` (`tables/`), `G_ROS_RESULTS_DIR/r1_numbers/out` (`counts/`).

**Order**
1. `audit/01_recluster.py`: pooled components, complete linkage, run count under both label formulas.
2. `audit/02_sign_audit.py`: per-member flips; corrected consensus (`B1_mean_cons.npy`). Then `audit/04_consensus.py` and `audit/07_bar_direction_test.py` (bar directions of the figure components).
3. `cell_type_evidence/s1_rebuild.py`, `s2_cacheX1.py`, `s3_cos.py`: run-level evidence for each cell-type direction (`cos_all.csv`).
4. `tables/build_DataS1_new_rows_C547_C548.py`: C547/C548 vectors and the BH values over the 547 components (`c547/`).
5. `kegg/kegg547/` and `seaad/` (see their READMEs).
6. `tables/00_collect_inputs.py`, then `tables/make_DataS1_cell_types_corrected.py` and `tables/make_DataS1_547.py`.
7. `counts/01_modality_coverage.py`, `02_filter_stages.py`, `04_donor_dominance_all.py`, `07_run_counts.py`, `06_ckpt_A_check.py`.

**Manuscript**: 547 components; 294 in all ten runs, 27 in nine, 35 in eight, 59 in seven, 47 in six, 85 in five, median 13 members (5 to 124) (`07_run_counts.py`); 99.5% and 99.6% of members with |r| > 0.99 between iterations 1,500 and 2,000 (`06_ckpt_A_check.py`); Data S1 cell-type labels shown only where the runs back them; C520 retired, C547 and C548 added; phenotype FDR over the 547 components.

**Checks**: `make_DataS1_cell_types_corrected.py` and `make_DataS1_547.py` reproduce the files used byte for byte (`Data_S1_corrected_547.csv` sha256 `79da373c...`). `tables/00_collect_inputs.py` repeats copy steps done by hand; its `cell_type_loadings_corrected.csv` agrees with the file used to 1.8e-15 and gives identical labels.
