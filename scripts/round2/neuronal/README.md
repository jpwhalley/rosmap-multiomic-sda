# Components 172 and 277

`models/`: nucleus counts by subtype, joint phenotype models, composition and source-data views (reusing the loaders of `dossier/dc_common.py`). `check/`: independent implementation, which gives the composition and stacked-model numbers. `rivals/`: the joint models for all distributed components and the within-component test.

**Inputs**: `G_ROS_ROSMAP_DIR` (scores, clinical, the labelled snRNA-seq object `mit/rna/mit_all_doublets_labelled.h5ad`), `G_ROS_QC_EXCLUDE_PATH`, `G_ROS_DATA_S1`, `G_ROS_TABLES_DIR`, `G_ROS_AUDIT_DIR/c547`, the identity key, and the outputs of `covariates/` and `dossier/`.
**Outputs** (under `G_ROS_RESULTS_DIR`): `revision_analyses/{neuronal_c21,verify_neuronal_c21}/{out,cache}` (the caches hold per-donor counts and frames) and `selection_review/are_C172_C277_distinctive/out`.

**Order**: `models/n02_subtype_counts.py`, `n03_phenotype_joint.py`, `n04_c277_composition.py`, `n05_source_data.py`, `n05b_po_sensitivity.py`; `check/v00_frame.py`, `v01_polr.R` (Rscript with MASS; reads `G_ROS_RESULTS_DIR` from the environment), `v02_counts.py`, `v03_composition.py`, `v05_dissociation.py`; `rivals/s01_joint_rivals.py`.

**Manuscript** (Results: Components 172 and 277; Methods: Sensitivity and cross-layer analyses; Figures 4, 5, S6, S7): joint models with CERAD, Braak and cognitive diagnosis (C172 CERAD 0.49, 0.27 to 0.87; C277 AD dementia 0.36, 0.20 to 0.64; `models/n03_phenotype_joint.py`); stacked comparison of the two components, p = 0.0018 (`check/v05_dissociation.py`); within C172, CERAD against AD dementia, p = 0.37 (`rivals/s01_joint_rivals.py`); composition adjustment of C277: 19 to 34% over the six composition measures, 27% (2% to 53%) for the six cell-class proportions, -0.44 SD (-0.69 to -0.19) (`check/v03_composition.py`); LRP8 and PLCB1 against the scores (`models/n05_source_data.py`).

`check/v00_frame.py` rebuilds a per-donor frame that was written inline in the original analysis (byte-identical).
