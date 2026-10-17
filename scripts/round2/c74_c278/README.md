# Components 74 and 278

`views/`: measured data against the scores, SEA-AD comparison of C74, C278 marker modules and effects. `check/`: independent implementation (several reported values come from here). `overlap/`: the C74 cross-layer overlap tests with matched backgrounds. `round1/`: the C74 SEA-AD outlier check of the first revision.

**Inputs**: `G_ROS_ROSMAP_DIR`, `G_ROS_SEAAD_DIR`, `G_ROS_DEPOSIT_DIR` (feature sidecars; methylation and ChIP identical in deposit versions 1 and 2, RNA chromosome column identical), `G_ROS_REFERENCES_DIR`, `G_ROS_SEQUENCES_DIR`, `G_ROS_TABLES_DIR`, `G_ROS_KEGG_GMT`, the identity key, and the outputs of `covariates/` and `dossier/`.
**Outputs** (under `G_ROS_RESULTS_DIR`): `revision_analyses/{c74_c278,verify_c74_c278}/{out,panels,cache}`, `v23_review/point_5_analysis/out`, `revision_r2/`. Draft panels plot one point per donor without IDs.

**Order**: `views/20_c74_views.py`, `21_c74_seaad.py`, `30_c278_views.py`, `32_c278_markers.py`; `check/v00_cache_inputs.py`, `v10_gene_annot.py`, `v30_c74_seaad.py`, `v31_seaad_zero_baserate.py`, `v60_c278_views.py`, `v61_c278_po.py`, `v62_c278_effect_noflag.py`, `v80_c278_markers.py`; `overlap/02` to `05` and `r5_c74_overlaps.py`; `round1/01_fig4c_outlier_check.py`.

**Manuscript** (Results: Components 74 and 278; Methods: Validation, Sensitivity and cross-layer analyses; Figures 2, 3, S3, S4):
- C74: XIST tracks the score in every cell type and the top CpG near AMDHD2 (`views/20_c74_views.py`); SEA-AD r = -0.982 over 138 genes, carried by the 62 X- and Y-linked genes, with the 76 autosomal genes compared by a one-sided binomial test against 0.539 (`views/21_c74_seaad.py`, `check/v30_c74_seaad.py`, `check/v31_seaad_zero_baserate.py`); top-5 removal and the 138-gene jackknife (`round1/01_fig4c_outlier_check.py`); gene overlap 2.3-fold (p = 9.7e-4) and peak overlap 2.6-fold (p = 2.2e-7) against matched strata (`overlap/r5_c74_overlaps.py`; builders `overlap/03_cpg_in_peak_density.py` and `overlap/05_rna_cpg_chromosome.py`).
- C278: FKBP5 and the 1,114-gene aggregate track the score in OPCs and oligodendrocytes, FKBP5 higher in women (`views/30_c278_views.py`); OPC share of oligodendrocyte-lineage nuclei and the marker modules (Figure S4d; `check/v80_c278_markers.py`); the effect without the flagged donors (`check/v62_c278_effect_noflag.py`).

`overlap/05_rna_cpg_chromosome.py` carries the documented fix for genes without a chromosome (see its docstring); with it the script gives the reported expected 6.10, fold 2.29, p 9.67e-4.
