# SEA-AD catalogue and partners

Re-derives the SEA-AD consensus from the ten SEA-AD SDA runs (as `dlpfc2_cluster.py` did), reproduces the deposited SEA-AD matrices, counts runs correctly (SEA-AD 9 and 128 retired; 141 components kept) and re-matches the Data S1 SEA-AD partners against the 141.

**Inputs**: `G_ROS_SEAAD_DIR` (`gene_info.csv`; `sda/` with `results1..10/it3000`, `md_samples.txt`, `cell_types.txt`, `genes.txt`, the deposited `*_final.csv`), `G_ROS_ROSMAP_DIR/sda`, `G_ROS_AUDIT_DIR/c547`, `G_ROS_TABLES_DIR/Data_S1_corrected_547.csv`.
**Outputs**: `G_ROS_AUDIT_DIR/seaad141/`.

**Order**: `01_recluster_seaad.py`, then `02_partner.py`. `02_partner.py` reads Data S1 for the 547 IDs and the partners it compares against; the re-match itself uses only the loadings. On a first build, place this repository's `Data_S1.csv` at `G_ROS_TABLES_DIR/Data_S1_corrected_547.csv` (same IDs and SEA-AD columns), run `02_partner.py`, then `catalogue/tables/00_collect_inputs.py`.

**Manuscript**: 141 SEA-AD components (Methods: Validation in an independent cohort); 15 Data S1 rows change partner (C3, 86, 104, 106, 140, 163, 204, 208, 227, 310, 389, 394, 403, 438, 548). The SEA-AD Spearman, permutation null and replication criterion are in `concentration/structure/02_seaad.py`.

The SEA-AD preprocessing and the SEA-AD SDA runs (800 components per run) are in `scripts/seaad/`.
