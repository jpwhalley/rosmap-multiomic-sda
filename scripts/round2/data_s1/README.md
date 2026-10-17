# Data S1

`build_Data_S1.py` builds Data S1 of this version (547 rows, 29 columns) from `Data_S1_corrected_547.csv` (`catalogue/`): it drops the three protein columns and adds the concentration, adjusted-q, prioritisation, protein-flag, protein-sensitivity and provenance columns. Every retained field is copied byte for byte, and every new value is checked against `concentration/structure/07_dataS1.py`'s draft.

**Inputs**: `G_ROS_TABLES_DIR/Data_S1_corrected_547.csv`; under `G_ROS_RESULTS_DIR`: `revision_analyses/structure/out/{05_master_prioritisation.csv,DataS1_additions_draft.csv}`, `protein_influence/results/consensus_547_scenarios.csv`, `verify_protein_influence/results/{consensus,members}_fit_full_r1_mf.csv`.
**Outputs**: `G_ROS_RESULTS_DIR/v23_ds1/Data_S1.csv` and `build.log`. The output is identical to this repository's `Data_S1.csv` (sha256 `18f0188f...`).

Run after `concentration/` and `fixed_loading/`.
