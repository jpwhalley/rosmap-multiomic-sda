# g_ros: analysis code for *Multi-omic Analysis of Sex and Alzheimer's Disease in the Aged Human Cortex*

This repository accompanies the manuscript

> **Whalley J.P., Hunsberger H.C., Bennett D.A., Lamar M., Peterson D.A.,
> Stutzmann G.E.** *Multi-omic Analysis of Sex and Alzheimer's Disease
> in the Aged Human Cortex*. Under review at
> *npj Aging*. Preprint (earlier version): bioRxiv,
> doi:[10.1101/2025.10.10.681695](https://doi.org/10.1101/2025.10.10.681695).

The study decomposed snRNA-seq, DNA methylation, H3K9ac ChIP-seq and TMT
proteomics data from 276 ROSMAP donors with Sparse Decomposition of Arrays
(SDA) into 547 components (IDs 1–519 and 521–548; see "Component IDs"). The
repository holds:

- `notebooks/`: four notebooks that start from the component-loading
  matrices deposited on Synapse and redraw the per-component panels, the
  KEGG enrichments and the phenotype tests;
- `scripts/` (01 to 13): the pipeline that built the SDA inputs and the
  547-component catalogue, and the GWAS, eQTL and SEA-AD steps
  (`scripts/README.md`);
- `scripts/round2/`: the analyses added in this revision (sample identity
  checks, covariate-adjusted models, fixed-loading re-estimation, donor
  concentration and prioritisation, the five reported components, Data S1;
  `scripts/round2/README.md`);
- `scripts/seaad/`: the SEA-AD preprocessing and SDA runs used for the
  cross-cohort validation (`scripts/seaad/README.md`);
- `scripts/figures/`: the code that draws the panels and assembles
  Figures 2 to 5 and S1 to S11 (`scripts/figures/README.md`).

> **Status.** The manuscript is under review. This version of the
> repository matches the revised manuscript and version 3 of the Synapse
> deposit `syn75275226` (see "Changes in this version"). The release DOI
> will be added on publication.

---

## Quick start

Three install paths are supported. Pick whichever matches your tooling.

### Option 1: `uv` (recommended for replication)

This path installs from the hash-pinned `uv.lock`, giving you the exact
package versions the notebooks were validated against.

```bash
# Install uv (https://docs.astral.sh/uv/) if you don't have it.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync.
git clone https://github.com/jpwhalley/rosmap-multiomic-sda.git
cd rosmap-multiomic-sda
uv sync --frozen
```

`uv sync` creates a virtualenv in `.venv/` and installs every dependency from
`uv.lock`. To run a notebook from the synced environment:

```bash
uv run jupyter lab notebooks/
```

### Option 2: conda

```bash
git clone https://github.com/jpwhalley/rosmap-multiomic-sda.git
cd rosmap-multiomic-sda
conda env create -f environment.yml
conda activate g_ros_release
jupyter lab notebooks/
```

The conda environment uses loose pins (no version locking), so it gets the
latest compatible packages each time you create it. For deterministic
replication, prefer Option 1.

### Option 3: plain pip

```bash
git clone https://github.com/jpwhalley/rosmap-multiomic-sda.git
cd rosmap-multiomic-sda
python -m venv .venv
source .venv/bin/activate
pip install -e .
jupyter lab notebooks/
```

`pip install -e .` reads `pyproject.toml` and installs the loose-pinned
dependencies plus the `g_ros_release` helper module in editable mode.

### After install: fetch the deposit and run

1. Download the Synapse deposit (see "Data availability" below) into a folder
   named `synapse/` at the root of your cloned repo. The notebooks default to
   `DEPOSIT_DIR = "../synapse"` (relative to `notebooks/`), so this layout
   works without further configuration.
2. Download the reference annotation files into `references/` per the
   instructions in `references/README.md` (the gene annotation, the
   methylation manifest, and the H3K9ac peak set). All four notebooks load
   them when they load the deposit, so all four need them.
3. Open the notebooks in order: `00_setup_and_data_loading.ipynb` first, then
   any of `01`, `02`, `03` depending on what you want to reproduce.

If you place the data somewhere else, edit the `DEPOSIT_DIR` and
`REFERENCES_DIR` paths near the top of each notebook accordingly, or set
the environment variables `G_ROS_DEPOSIT_DIR` and `G_ROS_REFERENCES_DIR`
(and `G_ROS_CLINICAL_PATH`, the clinical CSV for Notebook 03) before
starting Jupyter.

---

## Data availability

The derived data are AD Knowledge Portal dataset
[`syn75275226`](https://www.synapse.org/Synapse:syn75275226) (under the
ROSMAP project `syn3219045`). Use **version 3** (DOI: [10.7303/syn75275226.3](https://doi.org/10.7303/syn75275226.3)), which
holds the 547 components described here.
**Download via the
[AD Knowledge Portal](https://adknowledgeportal.synapse.org/Data%20Access)
after completing the data use certificate (DUC).**

| File | Shape | Description |
|---|---|---|
| `samples_final.csv` | 547 × 276 | Component × sample loadings |
| `cell_type.csv` | 547 × 6 | Component × cell-type loadings (Ast, Oli, Inh, OPC, Exc, Mic) |
| `RNA.txt` | 25,098 × 547 | Gene × component loadings (PIP > 0.5 masked) |
| `meth.txt` | 51,798 × 547 | CpG × component loadings (PIP > 0.5 masked) |
| `chip.txt` | 26,384 × 547 | H3K9ac peak × component loadings (PIP > 0.5 masked) |

The deposit README lists the other files it holds (Data S1 and the figure
source data). Version 1 of the deposit (DOI 10.7303/syn75275226.1; 546
components) is superseded by version 2, and version 3 renames the figure
source-data files to the final supplementary figure numbering (the data are
unchanged); see "Changes in this version".

**Proteomics.** The TMT proteomics layer was a modelled input of the
decomposition. Because of an identifier error in that input (TMT columns
matched to the wrong donors; manuscript Methods, "Proteomics"), no
protein-derived results are reported, and the deposit includes none.
`scripts/round2/fixed_loading/` measures how much the error moves the
reported component scores, and Data S1 flags the components it could affect.

**Primary data** are on the AD Knowledge Portal under its access terms:
snRNA-seq `syn52293417`, DNA methylation `syn3157275`, H3K9ac ChIP-seq
`syn4896408`, TMT proteomics `syn17008935`, array genotypes (GWAS and eQTL)
`syn3157325`, and the clinical metadata (`ROSMAP_clinical.csv`). SEA-AD is
public (Allen Brain Map). Donor-level source-assay measurements are not in
this repository or the deposit; the scripts regenerate them from these
accessions.

**Data S1.** `Data_S1.csv` at the repository root is Data S1 of the
manuscript: 547 rows (one per component ID) and 29 columns, built by
`scripts/round2/data_s1/build_Data_S1.py`. Next to the component
annotations (cell types, top features, KEGG terms, unadjusted clinical FDRs,
SEA-AD partner) it gives donor concentration (`top_donor_share`,
`participation_ratio`, `concentration_class`), covariate-adjusted FDRs
(`*_adj_q`), the SEA-AD Spearman correlation, the prioritisation
(`prioritised_2026`, `prioritisation_support`), the proteomics flags
(`protein_flag`, `score_r_corrected_protein`) and `provenance`. The protein
feature and pathway columns of earlier versions are withdrawn. Earlier
versions of this repository carried a draft of this table as
`Table_S1.csv`; `Data_S1.csv` replaces it.

In `Data_S1.csv`, `Top_positive_cell_type` / `Top_negative_cell_type` name
the cell type with the largest positive / most negative loading in
`cell_type.csv`, reported where that direction is supported by the
individual SDA runs; a blank cell means no cell type is reported for that
direction. A label recomputed from `cell_type.csv` alone can therefore
appear where Data S1 is blank.

The clinical metadata (`ROSMAP_clinical.csv`) is **controlled access** and is
not redistributed here. Notebook 03 requires it; users obtain it through the
AD Knowledge Portal DUC.

---

## Repository layout

```
rosmap-multiomic-sda/                       # this repo
├── README.md                               # this file
├── LICENSE                                 # MIT
├── pyproject.toml                          # PEP 621 project metadata + dependencies
├── uv.lock                                 # hash-pinned lock for `uv sync`
├── environment.yml                         # conda environment specification
├── .gitignore
├── g_ros_release.py                        # shared helper module imported by every notebook
├── Data_S1.csv                             # Data S1, the component-annotation table
├── notebooks/
│   ├── 00_setup_and_data_loading.ipynb     # environment + data sanity check
│   ├── 01_per_component_panels.ipynb       # sample scatter, cell-type bar,
│   │                                       # per-omic chromosome-scatter loadings
│   ├── 02_pathway_enrichment.ipynb         # KEGG enrichment for RNA and CpG features
│   └── 03_clinical_associations.ipynb      # sex / CERAD / cogdx tests per component
├── scripts/
│   ├── README.md                           # scripts 01-13, step by step
│   ├── 01_prepare_snrna.py ... 13_assemble_table_s1.py
│   ├── component_id_map.csv                # explicit component-ID map
│   ├── round2/                             # analyses of this revision (README.md, config.py,
│   │                                       # requirements.txt, data_s1/Data_S1_v22.csv)
│   ├── seaad/                              # SEA-AD preprocessing and SDA runs (README.md)
│   └── figures/                            # figure panels and composites (README.md, build_all.sh)
└── references/
    └── README.md                           # how to obtain the reference annotations
                                            # (no files are bundled; see README)
```

The Synapse deposit is **not** part of this repository. After cloning, create
a `synapse/` folder at the repo root and download the deposit files into it
(see "Data availability" above). The notebooks expect `DEPOSIT_DIR = "../synapse"`
by default (`synapse/` is gitignored). Outputs of `scripts/round2/` and
`scripts/figures/` go to `results/` by default, which is gitignored as well.

---

## What the notebooks reproduce

The notebooks operate on a configurable component list (`COMPONENTS = [...]`)
defined near the top of each notebook. The default is the nine components
shown in the manuscript figures:

| Component | Manuscript figure |
|---|---|
| 74  | Figure 2: sex-associated regulatory variation |
| 278 | Figure 3: sex-biased component shared by OPCs and mature oligodendrocytes |
| 172 | Figure 4: neuron-dominated component associated with neuritic plaque burden |
| 277 | Figure 5: neuron-dominated component associated with clinical Alzheimer's dementia |
| 21  | Supplementary Figure S2: ectopic olfactory receptor expression and its *cis*-genetic regulation |
| 351 | Supplementary Figure S8: a single-individual multi-omic profile |
| 2   | Supplementary Figure S9: a single-individual multi-omic profile |
| 87  | Supplementary Figure S10: a single-individual multi-omic profile |
| 177 | Supplementary Figure S11: a component dominated by one donor |

Notebooks 01 and 02 draw the building blocks of these figures (sample
scores, cell-type loadings, chromosome-scatter loadings, KEGG tables);
Notebook 03 reproduces the unadjusted clinical FDRs of Data S1. The composed
figures are built by `scripts/figures/`.

To explore other components or sweep all 547: change `COMPONENTS` to a list
of your choice, or to `list(deposit.samples.index)`. For full sweeps, set the
`SAVE_FIGURES_DIR` / `SAVE_DIR` variables so output is written to disk rather
than rendered inline.

### Component IDs

The component IDs are those used in the manuscript and in Data S1: 1–519
and 521–548. ID 520 is not used: that component was retired when the run
count was corrected (it was found in only 4 of the 10 SDA runs), and the
two components that meet the ≥ 5-run criterion under the corrected count
were added as 547 and 548. Every other ID is unchanged. Component ID is
therefore not the row number of a matrix: select and join by ID (for
example `deposit.samples.loc[548]`), never by position.
`scripts/component_id_map.csv` records the mapping.

### Sign convention

SDA components are sign-arbitrary, but only for sign changes that leave every
reconstructed product unchanged. For one component, the sample loadings can
be multiplied by −1 together with all of its feature loadings, which leaves
the sample × cell-type × gene product and each sample × feature product
unchanged; or the cell-type loadings can be multiplied by −1 together with
the gene loadings. Flipping the sample, cell-type and gene vectors all
together reverses the snRNA-seq product and is not a valid equivalence. The
deposit files contain one fixed sign choice per component, and the
manuscript figures plot the deposited values as they are, without sign
changes.

---

## Beyond the notebooks

Everything else in the manuscript is computed by `scripts/`, from the
controlled-access ROSMAP data and SEA-AD:

- **Pipeline** (`scripts/01` to `13`): SDA inputs, the ten SDA runs and the
  547-component consensus (`07_cluster_components.py`), GWAS with GEMMA 0.96
  and a KING 2.0.9 relatedness matrix (`08`, `09`), the eQTL tests (`11`) and
  SEA-AD matching (`12`). `10_chip_motifs.py` is the version-1 motif
  analysis, whose results are not reported in this version.
- **Analyses of this revision** (`scripts/round2/`): sample identity checks,
  covariate-adjusted models, the fixed-loading re-estimation (two
  implementations), donor concentration and prioritisation, Components 74,
  278, 172, 277 and 21, the GWAS of all 547 components, KEGG backgrounds and
  thresholds, the source-data views, and the build of `Data_S1.csv`. Its
  README gives the run order and the manuscript section of each folder.
- **SEA-AD** (`scripts/seaad/`): pseudobulk profiles from the open-access
  SEA-AD atlas, the SDA input and the ten SDA runs (800 components requested
  per run), and their consensus.
- **Figures** (`scripts/figures/`): every panel at its printed size, and the
  composites of Figures 2 to 5 and S1 to S11.

Paths come from environment variables or a config file
(`scripts/round2/paths.env.example`). The round-2 analyses ran under
pandas 2: `scripts/round2/requirements.txt` pins that environment (pandas
2.3.3, otherwise the versions of `uv.lock`, plus `h5py`). The round-2 code
also needs PLINK 1.9, GEMMA 0.96 and R (MASS); `scripts/seaad/` 00 and 01
need `scanpy`; the composites need `xelatex` with TikZ, Ghostscript and
poppler. The two QC-excluded snRNA-seq donors are listed in the deposit
(`qc_failed_projids.txt`), and Data S1 as submitted with the previous
version, the input of the catalogue correction, is
`scripts/round2/data_s1/Data_S1_v22.csv`. Exploratory checks that no
reported value depends on are not included (listed in
`scripts/round2/README.md`).

---

## Statistical conventions

The unadjusted clinical-association columns of `Data_S1.csv` (`Sex`,
`ceradsc`, `cogdx`):

- **Sex.** Mann–Whitney U on sample-level component loadings (Male vs Female).
- **ceradsc.** Mann–Whitney U on component loadings across a binarised CERAD
  score: `1, 2 → "AD"` (definite / probable AD) vs `3, 4 → "No"` (possible /
  no AD), per the CERAD scoring convention. Subjects with missing CERAD are
  dropped.
- **cogdx.** Kruskal–Wallis on component loadings restricted to cognitive
  diagnosis groups `1` (NCI), `2` (MCI), and `4` (probable AD). Other cogdx
  values (`3`, `5`, `6`) are excluded; subjects with missing cogdx are dropped.
- **Adjustment.** Benjamini–Hochberg FDR adjustment is applied **separately
  per clinical variable** across all 547 components. Notebook 03 calls
  `g_ros_release.clinical_association_table(..., correction='fdr_bh')` for
  exact parity with Data S1.

The covariate-adjusted columns (`*_adj_q`) come from proportional-odds
models with likelihood-ratio tests, adjusted for age at death, post-mortem
interval, cohort and (where not the phenotype) sex, with the same
per-phenotype BH correction over 547 components
(`scripts/round2/covariates/`).

The KEGG enrichment helper (`gr.kegg_enrichment`) wraps `gseapy.enrichr`
(library `KEGG_2021_Human`) with the PIP > 0.5 features as the gene list.
For RNA it passes the 25,098 genes of the RNA input as the background, as
in the manuscript. CpGs are mapped to every gene in the array manifest's
`RefGene` column and tested against Enrichr's default background. Data S1
used the 19,872 `RefGene` genes of the array as the CpG background, except
for Component 74 (Enrichr's default), so Notebook 02's CpG P-values match
Data S1 only for Component 74; the top terms agree with Data S1 for the nine
default components. `scripts/round2/kegg/` identifies the CpG backgrounds
and repeats the enrichments at other PIP thresholds.

ChIP-seq feature → gene assignment is **not** included in the helper module.
The H3K9ac peak manifest carries chromosome coordinates and a Promoter /
Enhancer / Other classification but no nearest-gene column, so pathway
analysis of peaks would need an external nearest-gene step (e.g.
`bedtools closest` against an Ensembl GTF). The manuscript reports no
pathway enrichment for H3K9ac peaks.

**Plot styling.** The clinical box plots (Figs 2a, 3a, 4a, 5a) are drawn by
`scripts/figures/qlabel_panels/`, which reproduces the seaborn < 0.13 box
style of version 1 and vendors statannotations 0.7.2 (MIT licence). Box
plots drawn directly with the locked seaborn 0.13.2 come out with thinner
lines (1.0 pt instead of 2.25 pt). Values and statistics are unaffected.

---

## Changes in this version

**Revised manuscript.**

- New title (above).
- **No protein-derived results.** The proteomics input matched 29 of its 105
  proteomes to the wrong donors (126 TMT batch.channel identifiers are shared
  between two studies; `scripts/04_prepare_proteomics.py` documents it). The
  layer stays in the decomposition as a modelled input; the protein loading
  matrix, the protein columns of Data S1 and the notebooks' protein panels
  and KEGG lists are withdrawn.
- **Data S1** has 29 columns (see "Data availability").
- **New code**: `scripts/round2/`, `scripts/seaad/` and `scripts/figures/`.
- **Notebooks** default to the nine components of the manuscript figures and
  were re-executed against version 2 of the deposit.
- Scripts read external paths from environment variables instead of
  hard-coded locations.

**Component catalogue** (Synapse `syn75275226` version 2):

- **547 components.** The step that counts how many of the 10 SDA runs
  contribute to a cluster miscounted the runs of some clusters. For three of
  them this changed the result: counted correctly, one published component
  (C520, 4 runs) is retired and two components (5 runs each) are added as
  C547 and C548. The other 545 components have the same members, sample
  loadings and feature loadings as before.
- **Cell-type loadings.** When the run-level components of a cluster were
  sign-aligned and averaged, each component's cell-type loadings were
  aligned separately from its sample and gene loadings. They are now flipped
  by the product of the component's sample and gene flips, so each run-level
  component's sample × cell-type × gene product keeps its sign in the
  average. Sample and feature loadings are unchanged. `cell_type.csv` and the
  Data S1 cell-type columns changed accordingly.
- **SEA-AD partners.** The SEA-AD decomposition has the same run-count
  error: SEA-AD components 9 and 128 come from only 4 runs and are retired
  (141 components, IDs otherwise unchanged). The 14 published Data S1 rows
  that named them, and the new C548, now name their best partner among the
  141.
- **Clinical q-values** in Data S1 are recomputed for the 547-component
  family. No significance call is lost; C548 gains a CERAD call.
- **Documentation** of the pipeline now states what was run: SDA ran for up
  to 3,000 iterations and was stopped by the time limit, and components were
  taken at iteration 1,500; all ten runs used the same seed; the RNA input
  to SDA was CPM; the GWAS used GEMMA (see `scripts/README.md`).

The scripts in `scripts/` implement the corrected aggregation
(`07_cluster_components.py`) and SEA-AD matching (`12_seaad_validation.py`).

---

## Reference annotations

The `references/` folder holds the public annotation files that the notebooks
need in addition to the Synapse deposit (gene annotation, methylation array
manifest, H3K9ac peak coordinates). To keep this repository code-only, no
data files are bundled; `references/README.md` gives the download source of
each file. All four notebooks load them with the deposit
(`load_deposit(..., load_references=True)`); Notebooks 01 and 02 use them,
and Notebook 03 also needs the controlled-access clinical CSV.

---

## Citing this code

If you use this code in derivative work, please cite the manuscript above
and the GitHub release tag (release DOI to be added on publication).

```bibtex
@unpublished{whalley_rosmap_sda,
  title  = {Multi-omic Analysis of Sex and Alzheimer's Disease in the
            Aged Human Cortex},
  author = {Whalley, Justin P. and Hunsberger, Holly C. and
            Bennett, David A. and Lamar, Melissa and
            Peterson, Daniel A. and Stutzmann, Grace E.},
  note   = {Under review at npj Aging. Preprint (earlier version): bioRxiv,
            doi:10.1101/2025.10.10.681695. Citation, year, journal, and DOI
            will be updated on publication.}
}
```

---

## License

MIT License. See `LICENSE`. The vendored copy of statannotations in
`scripts/figures/qlabel_panels/vendor/` keeps its own MIT licence.

---

## Contact

Questions about the code: open an issue on the GitHub repository.
Questions about the manuscript or the underlying data: justin.whalley@rosalindfranklin.edu.
