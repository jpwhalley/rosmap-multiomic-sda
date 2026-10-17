# Fixed-loading re-estimation (proteomics input)

With each run's fitted loadings held fixed, every donor's sample scores are re-estimated with the proteomics input as used, without the proteomics layer, without the 29 mismatched protein rows, and with the protein rows rebuilt under the corrected mapping; run-level scores are then combined into the 547 consensus components. Two implementations: `implementation_1/` is calibrated to the published scores, `implementation_2/` is built from the SDA variational updates.

**Inputs**: `G_ROS_ROSMAP_DIR` (`sda/results1..10/it1500`, SDA inputs, TMT matrix and metadata, clinical), `G_ROS_CATALOGUE_DIR` (`cluster_members.csv`, `samples_final.csv`, protein loadings and PIPs from `scripts/07_cluster_components.py`), `G_ROS_RESULTS_DIR/verify_source_panels/cache` (`source_data/verify_source_panels/01_load.py`), `G_ROS_AUDIT_DIR/c547`.
**Outputs** (under `G_ROS_RESULTS_DIR`): `{protein_influence,verify_protein_influence}/{results,logs}`, with per-run caches and per-donor scores in their `cache/`.

**Order**
- `implementation_1/`: `00_check_inputs.py`, `01_cache_run.py 1`, `02_prot_rebuild.py`, `run_all.sh 1 2 3 4 5 6 7 8 9 10` (03 and 05 per run), `04_calibrate.py 1`, `04c_calib_weighted.py 9`, `04c_calib_weighted.py 1`, `06_analyse.py`, `07_protein_share.py`.
- `implementation_2/`: `01_inputs.py`, `02_prot.py`, `run_all.sh 1 2 3 4 5 6 7 8 9 10` (04 and 05 per run), `06_members.py fit_full_r1_mf 1 2 3 4 5 6 7 8 9 10`, `07_consensus.py fit_full_r1_mf`, `09_x4_origin.py`, `10_layer_share.py`; method variants: `variants.sh`, then `08_variants.py 3 7`.
- `compare/p1_reproduction.py` after both.

**Manuscript** (Methods: Sensitivity of component scores to the proteomics input): re-estimated scores reproduce the published scores (median r = 0.999); for C74, C278, C172, C277, C21 and C545, r >= 0.998 against the published scores and r >= 0.9999 between each alternative input and the as-used re-estimate in both implementations (`compare/p1_reproduction.py`); Data S1 `score_r_corrected_protein` (implementation 1, `consensus_547_scenarios.csv`, `V3_d_r`) and the `protein_flag` codes (rule in `concentration/structure/07_dataS1.py`).

Not in the record as scripts: the exploratory checks logged as implementation 1 `04x`, `04y`, `06b`, `08`, `09`, `10` and implementation 2 `02b`, `11`, `12`, `13`. No reported value depends on them. `run_all.sh` and `variants.sh` use `/usr/bin/time -l` (macOS; `-v` with GNU time).
