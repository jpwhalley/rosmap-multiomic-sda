# Round-2 analysis code

Code for the analyses added in the second revision and for the 2026-10-08 correction of the component catalogue. Each folder holds one analysis; its README lists inputs, outputs and the manuscript results it produces. The code is as run, apart from the changes listed at the end.

## Setup

- Python: pandas 2, as in the original analysis: `requirements.txt` pins it (pandas 2.3.3, the other versions of `uv.lock`, and `h5py` for the snRNA-seq object metadata). `uv venv .venv-round2 --python 3.11`, then `uv pip install -p .venv-round2 -r scripts/round2/requirements.txt`. Under pandas 3 (`uv.lock`), `config.py` prints a note. The steps re-run for the release (covariate models, effects, donor concentration and prioritisation, ranking, Data S1) gave the record's results under both, apart from floating-point differences below 2e-15. PLINK 1.9 and GEMMA 0.96 must be on `PATH` for the genotype steps; `curl` for the Bryois download.
- Paths: `config.py` reads environment variables, or a `KEY=VALUE` file named by `G_ROS_CONFIG` (template: `paths.env.example`).
- Threads: `G_ROS_THREADS` (default 2) sets the BLAS threads and worker processes.
- Run a script from anywhere: `python scripts/round2/<folder>/<script>.py`. Shell steps: `bash <script>.sh`.

| Variable | Contents |
|---|---|
| `G_ROS_ROSMAP_DIR` | ROSMAP data from the AD Knowledge Portal and the SDA files: `sda/` (SDA inputs, `results1..10/`, version-1 consensus `*_final.csv`), `oth/meta/`, `oth/snps/` (array genotypes, archived GEMMA runs), `oth/chip/`, `oth/meth/`, `meth/`, `prot/tmt/`, `mit/meta/`, `mit/rna/`, `mit/atac/`. `G_ROS_SDA_DIR`, if set, must point to its `sda/` |
| `G_ROS_SEAAD_DIR` | SEA-AD DLPFC: `gene_info.csv` and the SDA folder `sda/` |
| `G_ROS_DEPOSIT_DIR`, `G_ROS_REFERENCES_DIR` | Synapse deposit (version 2) and the reference tables, as for the notebooks |
| `G_ROS_SEQUENCES_DIR` | GRCh37 FASTA (+ `.fai`), Ensembl 100 GTF and cDNA, dbSNP 135 (hg18) VCF |
| `G_ROS_ARCHIVE_DIR` | Version-1 Enrichr result tables (`enrichr/`, `cpg_enrichr/`) |
| `G_ROS_DATA_S1_V22`, `G_ROS_DATA_S1` | Data S1 as submitted in version 22 (input of the 10-08 correction; default `data_s1/Data_S1_v22.csv`, shipped) and this version |
| `G_ROS_QC_EXCLUDE_PATH` | The two snRNA-seq projids that failed QC, one per line (default `qc_failed_projids.txt` of the Synapse deposit) |
| `G_ROS_RESULTS_DIR` | Round-2 outputs, in the folder layout of the original analysis (default `results/` in the repository, ignored by git); `scripts/figures/` reads them from here |
| `G_ROS_AUDIT_DIR`, `G_ROS_TABLES_DIR` | Catalogue audit of 2026-10-08 and the corrected Data S1 tables (defaults `results/audit_1008`, `results/catalogue_1008`) |
| `G_ROS_CATALOGUE_DIR` | Outputs of `scripts/07_cluster_components.py` (default `results/sda_catalogue`; run 07 with `--output-dir` set to this) |
| `G_ROS_PRIVATE_DIR` | Donor key and per-donor tables of the identity checks |

## Run order

| Step | Folder | Analysis | Manuscript |
|---|---|---|---|
| 1 | `catalogue/` (with `kegg/kegg547/` and `seaad/`) | Corrected aggregation, 547-component catalogue, run counts, 10-08 Data S1 | Methods: Tensor and matrix decomposition; Data S1 |
| 2 | `kegg/` (rest) | PIP thresholds, CpG backgrounds | Methods: Pathway enrichment |
| 3 | `covariates/nuclei_counts/` | Nuclei per donor and cell class | Methods: Association with phenotypes |
| 4 | `source_data/` (steps 1 to 3) | Input and loading caches, protein mapping, batch tests | Methods: Proteomics, Source-data views |
| 5 | `identity/` | Sample identity checks | Methods: Sample identity checks |
| 6 | `covariates/` | Covariate-adjusted models | Methods: Association with phenotypes |
| 7 | `dossier/` | Effect estimates with CIs, concentration, C21 checks | Results: the five components; Figure S2 |
| 8 | `fixed_loading/` | Fixed-loading re-estimation, two implementations | Methods: Sensitivity to the proteomics input |
| 9 | `c21/` | GWAS of all 547 components, C21 sensitivity and eQTL comparison | Results: Component 21; Figure S7 |
| 10 | `concentration/` | Donor concentration, SEA-AD statistics, prioritisation, selection of the five | Results: Decomposition structure; Figure S1 |
| 11 | `c74_c278/` | Components 74 and 278 | Results; Figures 2, 3, S3, S4 |
| 12 | `neuronal/` | Components 172 and 277 | Results; Figures 4, 5, S5, S6 |
| 13 | `source_data/named/` | Named features of the donor-concentrated components | Results: Donor-concentrated components |
| 14 | `data_s1/` | Data S1 of this version | Data S1 |

`scripts/07_cluster_components.py` runs before step 1 (`--output-dir` set to `G_ROS_CATALOGUE_DIR`). `check/` and `implementation_2/` folders hold the second, independent implementations; several reported values come from them (see the folder READMEs).

## Not in this folder

- The SEA-AD preprocessing and SEA-AD SDA runs (800 components per run): `scripts/seaad/`. The runs are read from `G_ROS_SEAAD_DIR/sda`.
- Exploratory checks of the fixed-loading analysis that were run without a saved script and that no reported value depends on (named in `fixed_loading/README.md`), and the downloads of the public eQTL data (recorded in `c21/eqtl/public_queries.txt`).
- Withdrawn analyses (motif enrichment, protein pathway figures, the glucocorticoid gene-set tests) and exploratory or verification-only scripts that no reported value depends on.
- Figure code (`scripts/figures/`).

## Changes from the analysis record

- Paths come from `config.py`; outputs that were written beside the scripts go to the output folders above. Run folders keep their `out/`, `cache/` and `private/` subfolders; the identity checks' `private/` folder is `G_ROS_PRIVATE_DIR`.
- Thread counts and worker pools follow `G_ROS_THREADS`.
- The two QC-failed projids are read from `G_ROS_QC_EXCLUDE_PATH`.
- `c21/gwas547/01_build_fam.py` prints the number of unmatched genotype IDs instead of the IDs.
- `c74_c278/overlap/05_rna_cpg_chromosome.py` labels genes without a chromosome explicitly (documented fix; see its docstring).
- Shell steps that were run by hand are reconstructed from their logs: `identity/00_plink_cache.sh`, `c21/gwas547/run_gemma.sh`, `c21/sensitivity/run_gemma_c21.sh`, `c21/eqtl/n07a_ld_rosmap.sh`.
- Two files that were written inline are rebuilt by new scripts: `neuronal/check/v00_frame.py` (byte-identical) and `catalogue/tables/00_collect_inputs.py` (copy steps; see its docstring).
- `c21/gwas547/04_new_summary.py` reads `03_catalogue.py`'s table instead of an unsaved variant of it (only the shared column is used).
- `concentration/check/v07_prioritise_verify.py` casts one table index to integers, so it runs with the locked pandas 3 (the record ran pandas 2); its log is unchanged.
