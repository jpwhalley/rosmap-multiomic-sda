# g_ros — figure-reproduction code for *Multi-omic Glial Programs in Brain Aging*

This repository accompanies the manuscript

> **Whalley J.P., Hunsberger H.C., Bennett D.A., Lamar M., Peterson D.A.,
> Stutzmann G.E.** *Multi-omic Glial Programs in Brain Aging*. (Manuscript
> under review; full citation will be filled in on publication.)

It contains Jupyter notebooks that reproduce the figures and the component-
annotation table in that manuscript, starting from the multi-omic Sparse
Decomposition of Arrays (SDA) component-loading matrices deposited on Synapse.
The SDA decomposition itself was performed on a high-performance cluster from
upstream snRNA-seq, methylation, ChIP-seq, and proteomics data; these
notebooks pick up from the deposited 546-component output and reproduce the
downstream analyses described in the manuscript Results.

> **Status.** This repository is in pre-publication preparation. The
> manuscript citation, Zenodo DOI, and Synapse sub-project accession are
> placeholders below until the editorial decision lands and the data deposit
> is finalised through the AD Knowledge Portal DCA intake.

---

## Quick start

Three install paths are supported. Pick whichever matches your tooling.

### Option 1 — `uv` (recommended for replication)

This path installs from the hash-pinned `uv.lock`, giving you the exact
package versions the notebooks were validated against.

```bash
# Install uv (https://docs.astral.sh/uv/) if you don't have it.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync.
git clone https://github.com/jpwhalley/glial-aging-multiomic.git
cd g_ros
uv sync --frozen
```

`uv sync` creates a virtualenv in `.venv/` and installs every dependency from
`uv.lock`. To run a notebook from the synced environment:

```bash
uv run jupyter lab notebooks/
```

### Option 2 — conda

```bash
git clone https://github.com/jpwhalley/glial-aging-multiomic.git
cd g_ros
conda env create -f environment.yml
conda activate g_ros_release
jupyter lab notebooks/
```

The conda environment uses loose pins (no version locking) — it will get the
latest compatible packages each time you create the env. For deterministic
replication, prefer Option 1.

### Option 3 — plain pip

```bash
git clone https://github.com/jpwhalley/glial-aging-multiomic.git
cd g_ros
python -m venv .venv
source .venv/bin/activate
pip install -e .
jupyter lab notebooks/
```

`pip install -e .` reads `pyproject.toml` and installs the loose-pinned
dependencies plus the `g_ros_release` helper module in editable mode.

### After install — fetch the deposit and run

1. Download the Synapse deposit (see "Data availability" below) into a folder
   named `synapse/` at the root of your cloned repo. The notebooks default to
   `DEPOSIT_DIR = "../synapse"` (relative to `notebooks/`), so this layout
   works without further configuration.
2. Download the reference annotation files into `references/` per the
   instructions in `references/README.md` (the gene annotation, the
   methylation manifest, and the H3K9ac peak set). Notebooks 01 and 02 need
   them; Notebook 00 does not.
3. Open the notebooks in order: `00_setup_and_data_loading.ipynb` first, then
   any of `01`, `02`, `03` depending on what you want to reproduce.

If you place the data somewhere else, edit the `DEPOSIT_DIR` and
`REFERENCES_DIR` paths near the top of each notebook accordingly.

---

## Data availability

The Synapse deposit (sub-project under `syn3219045`, accession TBD on
publication) contains the SDA component-loading matrices. **Download via the
[AD Knowledge Portal](https://adknowledgeportal.synapse.org/Data%20Access)
after completing the data use certificate (DUC).**

| File | Shape | Description |
|---|---|---|
| `samples_final.csv` | 546 × 276 | Component × sample loadings |
| `cell_type.csv` | 546 × 6 | Component × cell-type loadings (Ast, Oli, Inh, OPC, Exc, Mic) |
| `RNA.txt` | 25,098 × 546 | Gene × component loadings (PIP > 0.5 masked) |
| `meth.txt` | 51,798 × 546 | CpG × component loadings (PIP > 0.5 masked) |
| `chip.txt` | 26,384 × 546 | H3K9ac peak × component loadings (PIP > 0.5 masked) |
| `proteins.txt` | 5,230 × 546 | Protein × component loadings (PIP > 0.5 masked) |

Components annotated with manuscript-derived metadata (cell-type drivers, top
features per omic, KEGG pathway hits, clinical association p-values, SEA-AD
cross-cohort component, etc.) are reported in the published Supplementary
Table 1 (`Table_S1.csv`). The same file is bundled at the repo root as
`Table_S1.csv` for offline parity-checking against the notebooks' reproduced
columns.

The clinical metadata (`ROSMAP_clinical.csv`) is **controlled access** and is
not redistributed here. Notebook 03 requires it; users obtain it through the
AD Knowledge Portal DUC.

---

## Repository layout

```
g_ros/                                      # this repo
├── README.md                               # this file
├── LICENSE                                 # MIT
├── pyproject.toml                          # PEP 621 project metadata + dependencies
├── uv.lock                                 # hash-pinned lock for `uv sync`
├── environment.yml                         # conda environment specification
├── .gitignore
├── g_ros_release.py                        # shared helper module imported by every notebook
├── Table_S1.csv                            # published component-annotation table (parity reference)
├── notebooks/
│   ├── 00_setup_and_data_loading.ipynb     # environment + data sanity check
│   ├── 01_per_component_panels.ipynb       # sample scatter, cell-type bar,
│   │                                       # per-omic chromosome-scatter loadings
│   ├── 02_pathway_enrichment.ipynb         # KEGG enrichment for RNA / meth / proteins
│   └── 03_clinical_associations.ipynb      # sex / CERAD / cogdx tests per component
└── references/
    └── README.md                           # how to obtain the reference annotations
                                            # (no files are bundled — see README)
```

The Synapse deposit is **not** part of this repository. After cloning, create
a `synapse/` folder at the repo root and download the deposit files into it
(see "Data availability" below). The notebooks expect `DEPOSIT_DIR = "../synapse"`
by default (`synapse/` is gitignored).

---

## What the notebooks reproduce

The notebooks operate on a configurable component list (`COMPONENTS = [...]`)
defined near the top of each notebook. The default is the six headline
components featured in the published manuscript:

| Component | Manuscript role |
|---|---|
| 351 | Figure 2 — microglial immunometabolic program |
| 2   | Figure 3 — multi-omic regulatory features |
| 74  | Figure 4 — sex-differentiated programs |
| 87  | Figure 5 — astrocyte immune-modulation / epigenetic-reprogramming |
| 278 | Sup Fig S4 — female-biased OPC/precursor shift (single-omic) |
| 177 | Sup Fig S5 — microglial bioenergetic program |

To reproduce a manuscript figure: run Notebook 01 (and Notebook 02 if pathway
enrichment is wanted) with `COMPONENTS = [<headline_id>]`.

To explore a non-headline component or sweep all 546 components: change
`COMPONENTS` to a list of your choice, or to `list(range(1, 547))`. For full
sweeps, set the `SAVE_FIGURES_DIR` / `SAVE_DIR` variables so output is written
to disk rather than rendered inline.

### Sign convention

SDA components are sign-arbitrary — the sign of a component's loadings can be
flipped without changing the decomposition. Some manuscript figures display
loadings with the sign flipped relative to the deposit values for narrative
clarity (e.g., to make a "microglial" pole appear positive). When comparing a
notebook panel to a manuscript figure: if the manuscript's top driver is in
the column that this notebook reports as the most-negative cell type, the
published figure has been sign-flipped. Flip the loading vector to match.

### What is *not* reproduced from these notebooks

The following analyses are documented in the manuscript Methods but are not
reproduced here, either because they require external tools / datasets not
bundled with the deposit, or because the published figure is more usefully
re-created from the corresponding building-block notebook output:

- **GWAS / Manhattan plots** (manuscript Methods §"GWAS"). Run with PLINK on
  cluster-genotype data; not in the deposit.
- **eQTL analysis** (manuscript Sup Fig S1, Component 21). Requires the GWAS
  outputs above plus per-cell-type pseudobulk expression matrices.
- **HOMER motif enrichment** (manuscript Fig 4 PRDM4 panel; Sup Fig S2 BARHL1
  panel). Requires Perl + HOMER + a genome FASTA.
- **SEA-AD cross-cohort validation** (manuscript Fig 4 SEA-AD scatter; Sup
  Figs S2 + S3). Requires SEA-AD external download. The component-level SEA-AD
  cross-reference is, however, captured in `Table_S1.csv` columns
  `SEA-AD_comp` and `SEA-AD_corr`.

To recreate any of the above, follow the corresponding sections of the
manuscript Methods. Synapse upload and AD Knowledge Portal metadata submission
are handled outside this notebook release; see the AD Knowledge Portal data
contribution workflow for the current intake, certification, upload, metadata
validation, and release steps.

---

## Statistical conventions

For full reproducibility of `Table_S1.csv`'s clinical-association columns:

- **Sex.** Mann–Whitney U on sample-level component loadings (Male vs Female).
- **ceradsc.** Mann–Whitney U on component loadings across a binarised CERAD
  score: `1, 2 → "AD"` (definite / probable AD) vs `3, 4 → "No"` (possible /
  no AD), per the CERAD scoring convention. Subjects with missing CERAD are
  dropped.
- **cogdx.** Kruskal–Wallis on component loadings restricted to cognitive
  diagnosis groups `1` (NCI), `2` (MCI), and `4` (probable AD). Other cogdx
  values (`3`, `5`, `6`) are excluded; subjects with missing cogdx are dropped.
- **Adjustment.** Benjamini–Hochberg FDR adjustment is applied **separately
  per clinical variable** across all 546 components. Notebook 03 calls
  `g_ros_release.clinical_association_table(..., correction='fdr_bh')` for
  exact parity with Table S1.

The KEGG enrichment helper (`gr.kegg_enrichment`) wraps `gseapy.enrichr` with
PIP > 0.5 features as the foreground gene list and (for RNA) the full
deposited gene set as the background. Other omics use Enrichr's default human
background. Methylation features are mapped to genes via the array manifest's
`RefGene` column (first listed gene per CpG); proteins are stripped from
`SYMBOL|UniProt` to `SYMBOL`.

ChIP-seq feature → gene assignment is **not** included in the helper module.
The H3K9ac peak manifest carries chromosome coordinates and a Promoter /
Enhancer / Other classification but no nearest-gene column. Pathway-level
analysis on ChIP peaks therefore requires an external nearest-gene step
(e.g., `bedtools closest` against an Ensembl GTF, or HOMER's `annotatePeaks.pl`).
The published manuscript's ChIP-side analysis is motif enrichment via HOMER,
not pathway enrichment; this is described in the Methods.

---

## Reference annotations

The `references/` folder holds the public annotation files that the notebooks
need in addition to the Synapse deposit (gene annotation, methylation array
manifest, H3K9ac peak coordinates). To keep this repository code-only, no
data files are bundled directly — see `references/README.md` for the download
sources for each file. Notebook 00 runs without them; Notebook 01 and 02
need them; Notebook 03 needs only the controlled-access clinical CSV.

---

## Citing this code

If you use these notebooks or the helper module in derivative work, please
cite the manuscript above and the GitHub release tag (Zenodo DOI: TBD on
publication).

```bibtex
@unpublished{whalley_glial_aging,
  title  = {Multi-omic Glial Programs in Brain Aging},
  author = {Whalley, Justin P. and Hunsberger, Holly C. and
            Bennett, David A. and Lamar, Melissa and
            Peterson, Daniel A. and Stutzmann, Grace E.},
  note   = {Manuscript under review. Citation, year, journal, and DOI will
            be updated on publication.}
}
```

---

## License

MIT License. See `LICENSE`.

---

## Contact

Questions about the code: open an issue on the GitHub repository.
Questions about the manuscript or the underlying data: justin.whalley@rosalindfranklin.edu.
