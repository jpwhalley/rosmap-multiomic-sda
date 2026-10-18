# `scripts/`: analysis pipeline upstream and downstream of the Synapse deposit

This directory contains the analysis code that was used to produce the figures
and tables reported in **Whalley J.P., Hunsberger H.C., Bennett D.A.,
Lamar M., Peterson D.A., Stutzmann G.E.** *Multi-omic Analysis of Sex and
Alzheimer's Disease in the Aged Human Cortex* (under review
at npj Aging).

The four notebooks in `notebooks/` (`00`–`03`)
reproduce the panels and tables that flow directly from the deposited
component-loading matrices. **Everything else** (the upstream pipeline that
produced those matrices, plus the downstream analyses that require
controlled-access data or external tools) is here as plain `.py` source so
that a committed reader, with appropriate access to the underlying datasets,
can reproduce the entire study end-to-end. This file describes scripts 01 to
13; the analyses added in this revision are in `round2/` and the figure code
in `figures/` (see "Round-2 analyses and figure code" below).

The release scripts are numbered by approximate **run order**: 01 → 13. Step
06 is the external SDA run rather than a Python script. One historical wrinkle:
`13_assemble_table_s1.py` writes the precursor of Data S1 (called Table S1 in
earlier drafts; the script names keep that name), then
`12_seaad_validation.py` appends SEA-AD cross-cohort columns and writes the
validated table in the version-1 format. Data S1 of this version is built by
`round2/data_s1/build_Data_S1.py`. The analysis-tree numbers from the original repository
(e.g. `ros1a`, `ros7di`) are noted in each script's header for traceability,
but the working filenames here follow the pipeline sequence as closely as the
historical dependencies allow.

---

## Pipeline overview

```
                          ┌──────────────────────────────────────────────────────┐
                          │ AD Knowledge Portal raw data (DUC required):         │
                          │   snRNA-seq    syn52293417  (Mathys et al. 2023)     │
                          │   methylation  syn3157275   (De Jager et al. 2014)   │
                          │   ChIP-seq     syn4896408   (Klein et al. 2019)      │
                          │   proteomics   syn17008935  (Johnson et al. 2020)    │
                          │   clinical     syn3219045/Data/Metadata              │
                          └──────────────────────────────────────────────────────┘
                                                  │
                  ┌───────────────────────────────┼───────────────────────────────┐
                  ▼                ▼              ▼              ▼
          01_prepare_snrna   02_prepare_chip  03_prepare_meth  04_prepare_proteomics
                  │                │              │              │
                  └────────────────┴──────┬───────┴──────────────┘
                                          ▼
                                 05_export_sda_inputs
                                          │
                                          ▼
                                 [06: run SDA on cluster]   ← see "Running SDA" below
                                          │
                                          ▼
                                 07_cluster_components
                                          │
                                          ▼
                              ┌───────────┴──────────────┐
                              ▼                          ▼
                     Synapse deposit               Notebooks 00–03
                     (component loadings)          (figure reproduction)
                              │
                              ▼
              ┌───────────────┴──────────────────────────────────────────┐
              ▼                  ▼                  ▼
          08, 09 GWAS        10 motifs           11 eQTL
              │                  │                  │
              └──────────────────┴────────┬─────────┘
                                          ▼
                              13_assemble_table_s1
                                          │
                                          ▼
                              12_seaad_validation
                                          │
                                          ▼
                         Data S1 (version-1 format)
```

Notebooks 00–03 only need the deposited matrices (`synapse/`) plus the
small public reference annotations in `references/`. Scripts 01–05, 07–13
require the upstream raw data and external tools listed in each script's
"External dependencies" section below.

---

## Compute resources

The four notebooks (`notebooks/00`–`03`) are designed to run on a personal
computer: they start from the deposited 547-component matrices and the
heaviest single object loaded in memory is on the order of a few hundred
megabytes.

The scripts in this directory are a different story. They were originally
run on a high-performance compute node with ~100 GB RAM and tens of CPU
cores; in particular `01_prepare_snrna.py` loads the full Mathys 2023
single-nucleus AnnData (millions of cells) and aggregates per-donor
pseudobulks across six cell types, which can OOM-kill on a typical
workstation. Once those pseudobulks already exist, `02`–`05` are practical
workstation-scale checks and were validated that way against the cluster
archive. `07_cluster_components.py` reads the ten SDA runs' it1500
checkpoints (~4.6 GB of text per run) one omic at a time; on a 12-core
workstation it took 13 minutes, with a peak of 8.6 GB of memory. The downstream GWAS / motif / eQTL / SEA-AD scripts
(`08`–`12`) have similar headroom assumptions plus external tool
dependencies. **If you intend to re-run `01` or `08`–`12` end-to-end, plan
to do so on a cluster or a comparable high-memory machine.**

The Synapse deposit exists precisely so that readers reproducing the
manuscript figures don't need that compute footprint; only the scripts'
authors needed it once, and the deposited outputs flow downstream from
there.

---

## Script-by-script

### Upstream of the deposit

#### `01_prepare_snrna.py` *(was `ros1a_prepare_data_sc.py`)*

Builds the three-dimensional gene-expression tensor from the published
ROSMAP snRNA-seq AnnData object.

- **Inputs:** Mathys et al. 2023 AnnData (Synapse `syn52293417`).
- **Processing:** exclude two QC-failing samples (projids in
  `qc_failed_projids.txt` of the Synapse deposit; `G_ROS_QC_EXCLUDE_PATH`
  overrides) and computationally predicted doublets;
  require ≥ 100 cells in each of the six major cell types per
  participant (Ast, Oli, Inh, OPC, Exc, Mic); generate cell-type-specific
  pseudobulk by summing raw counts; gene filter (≥ 10 counts in ≥ 10
  pseudobulk samples); library-size normalise to counts per million. The
  CPM values are the SDA input as they are (no log transform); the snRNA-seq
  log2(CPM+1) values are used only in the eQTL analysis in `11_eqtls.py`
  (plots and ANOVA tests).
- **Output:** tensor of shape (276 samples) × (6 cell types) × (25,098 genes),
  no missing samples.

#### `02_prepare_chip.py` *(was `ros1b_prepare_data_chip.py`)*

Harmonises the H3K9ac peak count matrix to the snRNA-seq cohort.

- **Inputs:** Klein et al. 2019 H3K9ac peak counts (Synapse `syn4896408`).
- **Processing:** subset to the 276 snRNA-seq projids (preserving missingness
  for participants without ChIP data); log2(CPM+1) normalise.
- **Output:** matrix of shape (276 samples) × (26,384 peaks); 83 samples
  carry NaN for the entire ChIP modality.

#### `03_prepare_meth.py` *(was `ros1c_prepare_data_imputed_meth.py`)*

Filters the imputed DNA methylation β-value matrix to variable CpG sites.

- **Inputs:** De Jager et al. 2014 imputed β-value matrix (Synapse
  `syn3157275`).
- **Processing:** subset to the 276 projids; binarise each β temporarily at
  0.5 (1 if β ≥ 0.5, else 0, NaN preserved); drop CpGs that are invariant
  after binarisation (all 0, all 1, or all missing); the surviving CpGs keep
  their **continuous β values** for SDA.
- **Output:** matrix of shape (276 samples) × (51,798 CpG sites); 60 samples
  carry NaN for the entire methylation modality.

#### `04_prepare_proteomics.py` *(was `ros1d_prepare_data_imputed_prot_tmt.py`)*

Subsets the log2-TMT proteomics matrix to no-missing-value proteins.

- **Inputs:** the log2-transformed TMT proteomics of ROSMAP round 1
  (400 individuals; Johnson et al. 2020; Synapse `syn17008935`, file `syn21266454`).
- **Processing:** subset to the 276 projids; remove any protein with a
  missing value in any of the 105 participants who do have proteomics data
  (no imputation).
- **Output:** matrix of shape (276 samples) × (5,230 proteins); 171 samples
  carry NaN for the entire proteomics modality.
- **Known error, kept as run:** the assay metadata covers two TMT studies
  that share 126 batch.channel identifiers, and the mapping keeps the last
  metadata row for each. 112 of the 400 dorsolateral prefrontal cortex
  columns are attributed to individuals of the other study, so 29 of the 105
  proteomes in the SDA input belong to other donors and 28 donors' own
  proteomes are left out (manuscript Methods, "Proteomics"). This is the
  input the decomposition used, so the code is unchanged; no protein-derived
  results are reported. `round2/covariates/check/t1_tmt_mapping_check.py`
  checks the counts, and `round2/fixed_loading/` uses the corrected mapping.

#### `05_export_sda_inputs.py` *(was `ros1_export_data.py`)*

Concatenates the four prepared matrices into the input format SDA expects
and writes a SLURM batch script that launches the ten SDA runs.

- **Inputs:** the four prepared matrices from scripts 01–04, plus per-omic
  metadata files (sample IDs, gene/CpG/peak/protein lists).
- **Outputs (in `sda/`):** `rna.txt`, `meth.txt`, `chip.txt`, `prot.txt`,
  `md_samples.txt`, `genes.txt`, `cell_types.txt`, `md_cpg.txt`, `md_chip.txt`,
  `md_prot.txt`, and `matd.sh` (the SLURM array submission script). Set
  `G_ROS_SDA_BINARY` and `G_ROS_CLUSTER_WORKDIR` for the paths written into
  `matd.sh`.

#### **06: Running SDA** *(external tool; no script in this directory)*

SDA is a Bayesian tensor-decomposition tool from
[Hore *et al.*, *Nature Genetics* 2016](https://doi.org/10.1038/ng.3582).
It is not bundled with this repository; install per the maintainer's
release. The runs used SDA v1.1 with this invocation (from `matd.sh`):

```bash
sda --data rna.txt meth.txt chip.txt prot.txt \
    --N 276 \
    --num_comps 2000 \
    --num_blocks 8 \
    --eigen_parallel true \
    --num_openmp_threads 12 \
    --save_freq 500 \
    --remove_zero_comps true \
    --max_iter 3000 \
    --ignore_missing true \
    --out results${SLURM_ARRAY_TASK_ID}
```

It was run as a 10-element SLURM array (`SLURM_ARRAY_TASK_ID = 1..10`), each
run writing to its own output directory. All ten runs used the same random
seed (the logs record `random seed: 44610 90489`). Each run was stopped by
the cluster's 10-day time limit, after 1,803–2,006 iterations. SDA saved a
checkpoint every 500 iterations, and the components were taken from the
iteration-1,500 checkpoint (`results${i}/it1500/`), the last one common to
all ten runs; `07_cluster_components.py` reads it.

#### `07_cluster_components.py` *(was `ros2_cluster.py`)*

Pools the components from the ten SDA runs and applies stability-based
clustering to identify the final 547 reproducible components (IDs 1–519 and
521–548).

- **Inputs:** the ten `results${i}/it1500/` directories produced by the SDA
  runs (under `G_ROS_SDA_DIR`, default `sda/`, or `--sda-results-base`); the
  metadata sidecars from `05`; `component_id_map.csv` (in this directory).
- **Processing:** compute pairwise dissimilarity between non-zero components
  as 1 − |Pearson correlation of sample loadings|; complete-linkage
  hierarchical clustering; flat-cluster cut at cophenetic distance 0.4;
  retain only clusters containing components from ≥ 5 of the 10 runs; within
  each retained cluster, align each component to the cluster's first
  component (sample and feature loadings each by the sign of their Pearson
  correlation with the first component's, and cell-type loadings by the
  product of the sample and gene flips, so that each component's
  sample × cell-type × gene product keeps its sign) and take the
  unweighted mean as the representative loading. PIPs are the median across
  the cluster's components.
- **Known residual (documented after v2.0.0):** the methylation, H3K9ac and
  protein blocks are aligned by their own correlations, not by the sample
  flip, so a member whose block flip differs from its sample flip enters the
  average with its sample × feature product reversed. In the components of
  the manuscript, aligning those blocks with the sample flip changes the
  selected C74 H3K9ac loadings by 0.045% of their norm with no sign change
  (the six members concerned are weak) and no other selected methylation or
  H3K9ac loading; the selected protein loadings of C431 and C490 (withdrawn
  outputs) change materially. PIP masks and sample scores do not depend on
  these flips. The released outputs keep the original alignment.
- **Component IDs:** IDs are assigned from `component_id_map.csv`, which
  keeps the published IDs: C520 is retired and the two components added by
  the corrected run count are C547 and C548. The map keys each component on
  its cluster's first member (run and SDA component index) and records the
  expected numbers of members and runs; the script stops if the clustering
  does not reproduce it. Component ID is not the row number.
- **Output:** unmasked cluster-aggregated matrices in `results/`
  (`samples_final.csv`, `treatment_final.csv`, `genes_final_*`,
  `pip_final_*`, rows indexed by component ID), plus `component_catalogue.csv`
  and `cluster_members.csv` (per-member runs and flips). The Synapse deposit
  is derived from these by applying the PIP > 0.5 mask to the four
  feature × component matrices.
- **Changes from the version that built the first deposit (Synapse
  `syn75275226` version 1):** that version counted the runs of a cluster
  with position labels that overlapped by one between consecutive runs,
  which kept historical C520 (4 runs) and missed the two 5-run clusters
  now numbered C547 and C548; and it aligned each component's cell-type
  loadings separately from its sample and gene loadings, which reversed
  the sample × cell-type × gene product of some components in the average.
  The other 545 components have the same members, sample loadings and
  feature loadings in both versions.

### Downstream of the deposit (controlled-access analyses)

These scripts produce results that feed into specific manuscript figures
and Data S1 columns. They are not reproduced from the notebook layer
because they require either restricted upstream data or external tools
not bundled with this repository.

#### `08_gwas_preprocessing.py` *(was `ros4_GWAS_association_study_maf.py`)*

Prepares genotype data for the per-component GWAS.

- **External dependencies:** PLINK 1.07 (genotype subsetting); KING 2.0.9
  (kinship), set by `G_ROS_PLINK` and `G_ROS_KING` or found on `PATH`;
  ROSMAP genotype array data in PLINK `.bed/.bim/.fam` format (Synapse
  `syn3157325`, controlled access via DUC).
- **Outputs:** PLINK BED/BIM/FAM files restricted to the snRNA-seq cohort
  (228 genotyped donors), the KING relatedness matrix, a `.fam` whose
  phenotype columns are the component sample loadings, and
  `gemma_phenotype_columns.txt` (the component ID of each phenotype column).
- **GWAS:** run per component with GEMMA v0.96 on the cluster, as a
  univariate linear mixed model with the KING relatedness matrix:
  `gemma -bfile ROSMAP_arrayGenotype_maf -n <column> -lmm 4 -k relatedness_matrix.txt -o results<ID>`.
  GEMMA's `-n` selects a phenotype column by position; with C520 retired,
  column n is not component n for IDs above 519, so take `<ID>` from line n
  of `gemma_phenotype_columns.txt`. Downstream scripts use the score-test
  P-value (`p_score`, the last column of `*.assoc.txt`).

#### `09_manhattan_plots.py` *(was `ros5_Manhattan_Plot_maf.py`)*

Manhattan and Q-Q plots from the GWAS association tables.

- **Inputs:** the GEMMA outputs of the GWAS run after `08` (`output/`); the
  abbreviated tables are read from `G_ROS_GWAS_DIR` (default `output/`).
- **External dependencies:** none beyond what `08` produced.

#### `10_chip_motifs.py` *(was `ros7di_chip_motifs.py`)*

Transcription-factor motif enrichment in H3K9ac peaks (PIP > 0.5). This is
the version-1 analysis behind the motif panels of the earlier version; this
version reports no motif results.

- **External dependencies:** HOMER (Perl; `G_ROS_HOMER_BIN_DIR`) with the
  genome FASTA used for motif extraction (`G_ROS_SEQUENCES_DIR`). The historical run used GRCh37.p13; use GRCh38 only after
  coordinate harmonisation / liftOver.
- **Outputs:** HOMER-format motif enrichment tables + cross-validation
  scatter plots against SEA-AD.

#### `11_eqtls.py` *(was `ros8_eQTLs.py`)*

Cell-type-specific eQTL analysis: tests SNP genotype against PIP > 0.5
gene loadings per cell type. Underlies Supplementary Figure S7e
(Component 21 olfactory eQTL).

- **External dependencies:** outputs of `08`; the abbreviated GWAS tables
  (`G_ROS_GWAS_DIR`); per-cell-type pseudobulk expression from
  `01_prepare_snrna`.
- **Cis/trans label:** the script compares SNP positions from the genotype
  `.bim` (NCBI36) with GRCh38 gene coordinates, so the `cis|trans` part of
  the output file names can be wrong. The Component 21 file says `trans`,
  but rs6664332 lies 81 kb from the OR2L3 gene start (GRCh38): a cis eQTL.
  Only the file name is affected. The Component 21 extensions (GWAS of all
  547 components, public eQTL comparison) are in `round2/c21/`.

#### `12_seaad_validation.py` *(was `synth1b_croos_clin_manuscript.py`)*

Cross-cohort reproducibility validation against the SEA-AD DLPFC dataset.
Gives the `SEA-AD_comp` and `SEA-AD_corr` columns of Data S1. The Spearman
correlation, the permutation null and the replication criterion of this
version are computed in `round2/concentration/structure/02_seaad.py`, and the
SEA-AD panels (Figs 2c, 3d, S1d, S3d) are drawn by `figures/`.

- **External dependencies:** SEA-AD processed snRNA-seq (the version used was
  downloaded from [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/collections/1ca90a2d-2943-483d-b678-b809bf464c30) on 5 October 2023; see `seaad/README.md`).
- **Inputs:** `results/significant_manu.csv` from `13_assemble_table_s1.py`.
- **SEA-AD catalogue:** 141 components. The SEA-AD decomposition was
  clustered with the same pipeline and has the same run-count error; counted
  correctly, SEA-AD components 9 and 128 come from only 4 of 10 runs. They
  are retired (`RETIRED_SEAAD_COMPONENTS`) and the other IDs are kept.
- **Outputs:** `results/significant_df_validated_manu.csv`, the Data S1
  precursor (version-1 format) with SEA-AD columns included.

#### `13_assemble_table_s1.py` *(was `ros9b_summarize_manuscript.py`)*

Integrates outputs from `06a` (clinical via Notebook 03), pathway
enrichment (`07ai`/`07bi`/`07ci` via Notebook 02), motif enrichment
(`10`), and top feature summaries into the precursor of Data S1.

- **Outputs:** `results/significant_manu.csv`, which is then consumed by
  `12_seaad_validation.py`.
- **Clinical precursor:** expects
  `results/significant_clinical_features_manu.csv`, produced by Notebook 03
  / the clinical-association pipeline.
- **Cell-type columns:** this script names the cell type with the largest
  positive / most negative loading. In the published Data S1 a label is
  shown only where the individual SDA runs support that direction, so a
  few labels this script writes are blank in Data S1.
- **Version-1 format:** the table has protein and `ChIP_Motif` columns that
  Data S1 of this version does not carry. Data S1 of this version is built
  by `round2/data_s1/build_Data_S1.py` from the corrected table, the
  round-2 results and the fixed-loading results.

---

## Round-2 analyses and figure code

- `round2/`: the analyses added in this revision (identity checks,
  covariate-adjusted models, fixed-loading re-estimation, donor concentration
  and prioritisation, the described components, the GWAS of all 547
  components, KEGG backgrounds, source-data views, the Data S1 build).
  `round2/README.md` gives the setup, the run order and what each folder
  produces.
- `seaad/`: the SEA-AD preprocessing (pseudobulk profiles from the
  open-access atlas), the SDA input and run script of the ten SEA-AD SDA
  runs, and their consensus (`seaad/README.md`).
- `figures/`: the panel and composite code of Figures 2 to 5 and S1 to S11
  (`figures/README.md`).

Both read every path from environment variables or a config file
(`round2/paths.env.example`).

---

## External dependencies summary

| Dependency | Required by | Source |
|---|---|---|
| ROSMAP raw modalities (snRNA-seq, methylation, ChIP, proteomics) | 01–05 | AD Knowledge Portal, DUC required |
| ROSMAP clinical metadata (`ROSMAP_clinical.csv`) | Notebook 03, parts of 11/13 | AD Knowledge Portal `syn3219045` / Data / Metadata, DUC required |
| ROSMAP genotype array data (PLINK format) | 08, 11 | AD Knowledge Portal `syn3157325`, DUC required |
| SDA binary (v1.1) | 06 | Hore *et al.* 2016, [https://jmarchini.org/software/](https://jmarchini.org/software/) |
| PLINK 1.07 | 08 | [https://www.cog-genomics.org/plink/](https://www.cog-genomics.org/plink/) |
| KING 2.0.9 | 08 | [https://www.kingrelatedness.com/](https://www.kingrelatedness.com/) |
| GEMMA 0.96 | GWAS (after 08) | [https://github.com/genetics-statistics/GEMMA](https://github.com/genetics-statistics/GEMMA) |
| HOMER | 10 | [http://homer.ucsd.edu/homer/](http://homer.ucsd.edu/homer/) |
| Genome FASTA (historical run used GRCh37.p13; use GRCh38 only after coordinate harmonisation) | 10 | Ensembl, UCSC, etc. |
| SEA-AD processed snRNA-seq | 12, `seaad/` | [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/collections/1ca90a2d-2943-483d-b678-b809bf464c30) (versions in `seaad/README.md`); also the [Allen SEA-AD AWS bucket](https://registry.opendata.aws/allen-sea-ad-atlas/) |
| Python packages not in `uv.lock`: `scanpy` (01, `seaad/` 00 and 01), `assocplots` (09), `pysnptools` (11), `beautifulsoup4` (13) | 01, 09, 11, 13, `seaad/` | PyPI |
| pandas 2 (`round2/requirements.txt`), PLINK 1.9, GEMMA 0.96, `h5py`, R (MASS); `xelatex` with TikZ, Ghostscript, poppler | `round2/`; `figures/` composites | see `round2/README.md` and `figures/README.md` |

---

## A note on idiosyncrasies

These scripts were written by a single author (J.P.W.) over the course of
the analysis and reflect that history: hardcoded cluster paths, repeated
near-identical blocks where a small loop would suffice, occasional
exploratory branches that never made it into the manuscript. They have
been lightly cleaned for this release (external paths read from
environment variables or relative to the working directory, dead
exploratory code removed, repetitive blocks loop-ified where doing so does
not obscure the science), but the control flow has been preserved as faithfully as possible to what was
actually run. A reader following the manuscript Methods step by step
should find the corresponding script's structure recognisable.
`07_cluster_components.py` is the exception: it carries the 2026-10
corrections described above and is restructured to run on a workstation.

The original analysis-tree filenames (`ros1a_prepare_data_sc.py`, etc.)
are listed in italics next to each script above so anyone cross-referencing
this repository against a future supplementary code archive or a previous
internal version can find the equivalent file.
