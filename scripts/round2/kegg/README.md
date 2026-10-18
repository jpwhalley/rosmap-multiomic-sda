# KEGG enrichment

`kegg547/` fetches the KEGG_2021_Human library once, re-implements the original Enrichr background-mode call offline (`kegg_local.py`; network blocked after the fetch), validates it against Data S1, and computes the C547/C548 enrichments. `threshold/` repeats the reported enrichments at other PIP thresholds; `cpg_backgrounds/` identifies the background of every archived CpG table.

**Inputs**: `G_ROS_ROSMAP_DIR/sda` (PIPs, feature lists), `G_ROS_AUDIT_DIR/c547`, `G_ROS_REFERENCES_DIR` (Ensembl 100 gene table, methylation manifest), `G_ROS_ARCHIVE_DIR` (version-1 Enrichr tables, for validation), `G_ROS_DATA_S1_V22`, `G_ROS_DATA_S1`. The library is written to `G_ROS_KEGG_GMT` (default `G_ROS_AUDIT_DIR/kegg547/KEGG_2021_Human.gmt`) and is not bundled.
**Outputs**: `G_ROS_AUDIT_DIR/kegg547/`; under `G_ROS_RESULTS_DIR`: `r1_numbers/out/kegg_*` (threshold) and `v23_review/point_5_analysis/out/01_*` (CpG backgrounds).

**Order**: `kegg547/01_fetch_library.py` (network), `02_pip_lists.py`, `05_validate_dataS1.py`, `06_compute_c547_c548.py`; then `threshold/09_kegg_threshold.py` and `cpg_backgrounds/01_cpg_kegg_backgrounds.py`.

**Manuscript** (Methods: Pathway enrichment analysis): the 25,098-gene RNA background; CpG tables against the 19,872 RefGene genes except C74 (Enrichr's default), with the C74 top term unchanged on the 19,872-gene background (`01_cpg_kegg_backgrounds.py`); the terms named in the Results stay enriched at PIP thresholds 0.45 to 0.75 (`09_kegg_threshold.py`); Data S1 `GeneExpression_Pathway` for C547/C548 (`06_compute_c547_c548.py`).

Notes: the original run of `09_kegg_threshold.py` also re-tested the withdrawn protein enrichment (version-22 Fig 2d); those sets are removed here, the C277 claims add Calcium signaling pathway (named in version 23; five rows, follow-up to v2.0.0), and the other rows equal the record. The order of genes in the `Genes` column follows Python's set order and can differ between runs; all values agree.
