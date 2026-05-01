# `scripts/` — analysis pipeline upstream and downstream of the Synapse deposit

This directory contains the analysis code that was used to produce the figures
and tables reported in **Whalley J.P., Hunsberger H.C., Bennett D.A.,
Lamar M., Peterson D.A., Stutzmann G.E.** *Multi-omic Glial Programs in Brain
Aging* (manuscript under review).

The four notebooks in `notebooks/` (`00`–`03`)
reproduce the panels and tables that flow directly from the deposited
component-loading matrices. **Everything else** — the upstream pipeline that
produced those matrices, plus the downstream analyses that require
controlled-access data or external tools — is here as plain `.py` source so
that a committed reader, with appropriate access to the underlying datasets,
can reproduce the entire study end-to-end.

The release scripts are numbered by approximate **run order**: 01 → 13. Step
06 is the external SDA run rather than a Python script. One historical wrinkle:
`13_assemble_table_s1.py` writes the precursor Table S1 file, then
`12_seaad_validation.py` appends SEA-AD cross-cohort columns and writes the
final validated table. The analysis-tree numbers from the original repository
(e.g. `ros1a`, `ros7di`) are noted in each script's header for traceability,
but the working filenames here follow the pipeline sequence as closely as the
historical dependencies allow.

---

## Pipeline overview

```
                          ┌──────────────────────────────────────────────────────┐
                          │ AD Knowledge Portal raw data (DUC required):         │
                          │   snRNA-seq    syn52293417  (Mathys et al. 2023)     │
                          │   methylation  syn4896408   (De Jager et al. 2014)   │
                          │   ChIP-seq     syn3157275   (Klein et al. 2019)      │
                          │   proteomics   syn17008935  (Ping et al. 2020)       │
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
                              final Table S1
```

Notebooks 00–03 only need the deposited matrices (`synapse/`) plus the
small public reference annotations in `references/`. Scripts 01–05, 07–13
require the upstream raw data and external tools listed in each script's
"External dependencies" section below.

---

## Compute resources

The four notebooks (`notebooks/00`–`03`) are designed to run on a personal
computer — they start from the deposited 546-component matrices and the
heaviest single object loaded in memory is on the order of a few hundred
megabytes.

The scripts in this directory are a different story. They were originally
run on a high-performance compute node with ~100 GB RAM and tens of CPU
cores; in particular `01_prepare_snrna.py` loads the full Mathys 2023
single-nucleus AnnData (millions of cells) and aggregates per-donor
pseudobulks across six cell types, which can OOM-kill on a typical
workstation. Once those pseudobulks already exist, `02`–`05` are practical
workstation-scale checks and were validated that way against the cluster
archive. The downstream GWAS / motif / eQTL / SEA-AD scripts (`08`–`12`)
have similar headroom assumptions plus external tool dependencies. **If you
intend to re-run `01` or `08`–`12` end-to-end, plan to do so on a cluster
or a comparable high-memory machine.**

The Synapse deposit exists precisely so that readers reproducing the
manuscript figures don't need that compute footprint — only the scripts'
authors needed it once, and the deposited outputs flow downstream from
there.

---

## Script-by-script

### Upstream of the deposit

#### `01_prepare_snrna.py` *(was `ros1a_prepare_data_sc.py`)*

Builds the three-dimensional gene-expression tensor from the published
ROSMAP snRNA-seq AnnData object.

- **Inputs:** Mathys et al. 2023 AnnData (Synapse `syn52293417`).
- **Processing:** exclude two QC-failing samples and computationally predicted
  doublets; require ≥ 100 cells in each of the six major cell types per
  participant (Ast, Oli, Inh, OPC, Exc, Mic); generate cell-type-specific
  pseudobulk by summing raw counts; gene filter (≥ 10 counts in ≥ 10
  pseudobulk samples); library-size normalise to counts per million.
- **Output:** tensor of shape (276 samples) × (6 cell types) × (25,098 genes),
  no missing samples.

#### `02_prepare_chip.py` *(was `ros1b_prepare_data_chip.py`)*

Harmonises the H3K9ac peak count matrix to the snRNA-seq cohort.

- **Inputs:** Klein et al. 2019 H3K9ac peak counts (Synapse `syn3157275`).
- **Processing:** subset to the 276 snRNA-seq projids (preserving missingness
  for participants without ChIP data); log2(CPM+1) normalise.
- **Output:** matrix of shape (276 samples) × (26,384 peaks); 83 samples
  carry NaN for the entire ChIP modality.

#### `03_prepare_meth.py` *(was `ros1c_prepare_data_imputed_meth.py`)*

Filters the imputed DNA methylation β-value matrix to variable CpG sites.

- **Inputs:** De Jager et al. 2014 imputed β-value matrix (Synapse
  `syn4896408`).
- **Processing:** subset to the 276 projids; binarise each β temporarily at
  0.5 (1 if β ≥ 0.5, else 0, NaN preserved); drop CpGs that are invariant
  after binarisation (all 0, all 1, or all missing); the surviving CpGs keep
  their **continuous β values** for SDA.
- **Output:** matrix of shape (276 samples) × (51,798 CpG sites); 60 samples
  carry NaN for the entire methylation modality.

#### `04_prepare_proteomics.py` *(was `ros1d_prepare_data_imputed_prot_tmt.py`)*

Subsets the log2-TMT proteomics matrix to no-missing-value proteins.

- **Inputs:** Ping et al. 2020 log2-transformed TMT proteomics
  (Synapse `syn17008935`).
- **Processing:** subset to the 276 projids; remove any protein with a
  missing value in any of the 105 participants who do have proteomics data
  (no imputation).
- **Output:** matrix of shape (276 samples) × (5,230 proteins); 171 samples
  carry NaN for the entire proteomics modality.

#### `05_export_sda_inputs.py` *(was `ros1_export_data.py`)*

Concatenates the four prepared matrices into the input format SDA expects
and writes a SLURM batch script that launches the ten independent runs.

- **Inputs:** the four prepared matrices from scripts 01–04, plus per-omic
  metadata files (sample IDs, gene/CpG/peak/protein lists).
- **Outputs (in `sda/`):** `rna.txt`, `meth.txt`, `chip.txt`, `prot.txt`,
  `md_samples.txt`, `genes.txt`, `cell_types.txt`, `md_cpg.txt`, `md_chip.txt`,
  `md_prot.txt`, and `matd.sh` (the SLURM array submission script).

#### **06: Running SDA** *(external tool — no script in this directory)*

SDA is a Bayesian tensor-decomposition tool from
[Hore *et al.*, *Nature Genetics* 2016](https://doi.org/10.1038/ng.3582).
It is not bundled with this repository; install per the maintainer's
release. The exact invocation used in the manuscript was:

```bash
sda --data rna.txt meth.txt chip.txt prot.txt \
    --N 276 \
    --num_comps 2000 \
    --num_blocks 8 \
    --eigen_parallel true \
    --num_openmp_threads 12 \
    --save_freq 500 \
    --remove_zero_comps true \
    --max_iter 1500 \
    --ignore_missing true \
    --out results${SLURM_ARRAY_TASK_ID}
```

Run as a 10-element array (`SLURM_ARRAY_TASK_ID = 1..10`) so each run
starts from a different random seed and writes to its own output
directory. `--max_iter 1500` reflects the actual iteration count achieved
within compute budget (the manuscript Methods reports 1,500 iterations
for this reason; an earlier draft of the export script set the cap higher
but the runs were not extended).

#### `07_cluster_components.py` *(was `ros2_cluster.py`)*

Pools the components from all ten SDA runs and applies stability-based
clustering to identify the final 546 reproducible components.

- **Inputs:** the ten `results${i}/` directories produced by SDA runs.
- **Processing:** compute pairwise dissimilarity between non-zero components
  as 1 − |Pearson correlation of sample loadings|; hierarchical clustering;
  flat-cluster cut at cophenetic distance 0.4; retain only clusters
  containing components from ≥ 5 of the 10 runs; sign-align components
  within each retained cluster; mean-aggregate to a single representative
  loading per cluster.
- **Output:** unmasked cluster-aggregated matrices in `results/`
  (`samples_final.csv`, `treatment_final.csv`, `genes_final_*`,
  `pip_final_*`). The Synapse deposit is derived from these by applying
  the PIP > 0.5 mask to the four feature × component matrices.

### Downstream of the deposit (controlled-access analyses)

These scripts produce results that feed into specific manuscript figures
and Table S1 columns. They are not reproduced from the notebook layer
because they require either restricted upstream data or external tools
not bundled with this repository.

#### `08_gwas_preprocessing.py` *(was `ros4_GWAS_association_study_maf.py`)*

Prepares genotype data for the per-component GWAS.

- **External dependencies:** PLINK 2.0; ROSMAP genotype VCF (controlled
  access via DUC; AMP-AD genotype data).
- **Outputs:** PLINK BED/BIM/FAM files and relatedness matrix. GEMMA was run
  downstream on the cluster to produce the per-component association tables.

#### `09_manhattan_plots.py` *(was `ros5_Manhattan_Plot_maf.py`)*

Manhattan and Q-Q plots from the GWAS association tables.

- **Inputs:** outputs of `08_gwas_preprocessing`.
- **External dependencies:** none beyond what `08` produced.

#### `10_chip_motifs.py` *(was `ros7di_chip_motifs.py`)*

Transcription-factor motif enrichment in H3K9ac peaks (PIP > 0.5).
Used for the BARHL1-like motif panel (Sup Fig S2) and the PRDM4 panel
(Fig 4).

- **External dependencies:** HOMER (Perl) with the genome FASTA used for motif
  extraction. The historical run used GRCh37.p13; use GRCh38 only after
  coordinate harmonisation / liftOver.
- **Outputs:** HOMER-format motif enrichment tables + cross-validation
  scatter plots against SEA-AD.

#### `11_eqtls.py` *(was `ros8_eQTLs.py`)*

Cell-type-specific eQTL analysis: tests SNP genotype against PIP > 0.5
gene loadings per cell type. Underlies Sup Fig S1 (Component 21
olfactory eQTL).

- **External dependencies:** outputs of `08`; per-cell-type pseudobulk
  expression from `01_prepare_snrna`.

#### `12_seaad_validation.py` *(was `synth1b_croos_clin_manuscript.py`)*

Cross-cohort reproducibility validation against the SEA-AD DLPFC dataset.
Underlies the SEA-AD scatter panels of Fig 4 and Sup Figs S2 + S3.

- **External dependencies:** SEA-AD processed snRNA-seq from
  [Allen SEA-AD AWS bucket](https://registry.opendata.aws/allen-sea-ad-atlas/).
- **Inputs:** `results/significant_manu.csv` from `13_assemble_table_s1.py`.
- **Outputs:** `results/significant_df_validated_manu.csv`, the final Table S1
  precursor with SEA-AD columns included.

#### `13_assemble_table_s1.py` *(was `ros9b_summarize_manuscript.py`)*

Integrates outputs from `06a` (clinical via Notebook 03), pathway
enrichment (`07ai`/`07bi`/`07ci` via Notebook 02), motif enrichment
(`10`), and top feature summaries into the precursor Supplementary
Table 1.

- **Outputs:** `results/significant_manu.csv`, which is then consumed by
  `12_seaad_validation.py`.
- **Clinical precursor:** expects
  `results/significant_clinical_features_manu.csv`, produced by Notebook 03
  / the clinical-association pipeline.

---

## External dependencies summary

| Dependency | Required by | Source |
|---|---|---|
| ROSMAP raw modalities (snRNA-seq, methylation, ChIP, proteomics) | 01–05 | AD Knowledge Portal, DUC required |
| ROSMAP clinical metadata (`ROSMAP_clinical.csv`) | Notebook 03, parts of 11/13 | AD Knowledge Portal `syn3219045` / Data / Metadata, DUC required |
| ROSMAP genotype VCF | 08, 11 | AD Knowledge Portal, DUC required |
| SDA binary | 06 | Hore *et al.* 2016, [https://jmarchini.org/software/](https://jmarchini.org/software/) |
| PLINK 2.0 | 08 | [https://www.cog-genomics.org/plink/2.0/](https://www.cog-genomics.org/plink/2.0/) |
| HOMER | 10 | [http://homer.ucsd.edu/homer/](http://homer.ucsd.edu/homer/) |
| Genome FASTA (historical run used GRCh37.p13; use GRCh38 only after coordinate harmonisation) | 10 | Ensembl, UCSC, etc. |
| SEA-AD processed snRNA-seq | 12 | [Allen SEA-AD AWS bucket](https://registry.opendata.aws/allen-sea-ad-atlas/) |

---

## A note on idiosyncrasies

These scripts were written by a single author (J.P.W.) over the course of
the analysis and reflect that history: hardcoded cluster paths, repeated
near-identical blocks where a small loop would suffice, occasional
exploratory branches that never made it into the manuscript. They have
been lightly cleaned for this release — cluster-specific paths replaced
with placeholder comments, dead exploratory code removed, repetitive
blocks loop-ified where doing so does not obscure the science — but the
control flow has been preserved as faithfully as possible to what was
actually run. A reader following the manuscript Methods step by step
should find the corresponding script's structure recognisable.

The original analysis-tree filenames (`ros1a_prepare_data_sc.py`, etc.)
are listed in italics next to each script above so anyone cross-referencing
this repository against a future supplementary code archive or a previous
internal version can find the equivalent file.
