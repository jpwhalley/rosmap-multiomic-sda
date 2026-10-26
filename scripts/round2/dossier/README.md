# Component dossier

Per-component facts for the described components: effect estimates with confidence intervals, score concentration, the C21 genotype checks, sex-chromosome gene shares and composition correlations. `check/` re-derives the effects with its own code.

**Inputs**: `G_ROS_ROSMAP_DIR` (scores, SDA inputs and loadings, clinical, genotypes), `G_ROS_SEQUENCES_DIR` (C21 genome-build check), `G_ROS_REFERENCES_DIR`, `G_ROS_SEAAD_DIR`, `G_ROS_TABLES_DIR`, `G_ROS_AUDIT_DIR/c547`, and the outputs of `covariates/` and `source_data/`.
**Outputs** (under `G_ROS_RESULTS_DIR`): `component_dossier/{out,cache}` and `verify_dossier/{out,cache}`; the caches hold the SDA inputs and loadings read by `c74_c278/`, `neuronal/` and `scripts/figures/`.

**Order**: `00_cache_loadings.py`, `03_protein_proxy.py` (also writes the input caches), then `01`, `02`, `09`, `04`, `06`, `08`; `check/v02a_qc.py`, `check/v02_effects.py 74`, then `278`, `545`, `172` and `277` (one table per component; read by `c74_c278/views/30_c278_views.py` and Figure S3a), `check/v05_component_facts.py`.

**Manuscript**: covariate-adjusted odds ratios with profile-likelihood CIs and rank-based differences (C278 2.38, 1.56 to 3.66; C172 0.43; C277 0.31 and MCI 0.92; `02_effects_ci.py`, `09_effect_table.py`; Figure S3a draws the `check/v02_effects.py` values); shares of the dominant donor (98.4%, 94.4%, 98.4%, 83.5%; `01_concentration.py`); the C351 donor's proteome row (`03_protein_proxy.py`); C21: +0.89 SD per A allele, 81 kb from the OR2L3 gene start (`04_c21_checks.py`); composition correlations of C277, C172 and C545 (`08_composition.py`).
