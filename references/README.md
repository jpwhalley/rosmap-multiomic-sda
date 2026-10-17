# Reference annotations

The notebooks need three small public reference tables in addition to the
Synapse deposit. None are bundled in this repository (we keep the GitHub
release code-only); each is documented below with download instructions and
should be placed in this directory.

All four notebooks load all three tables when they load the deposit
(`load_deposit(..., load_references=True)`), so each notebook needs them in
this directory. Notebook 01 uses all three for the chromosome-scatter
loadings panels. Notebook 02 uses the methylation manifest (for the CpG gene
lists in KEGG enrichment). Notebooks 00 and 03 load them but do not use them;
Notebook 03 also needs the clinical CSV (see "Optional: clinical metadata" at
the bottom).

## Download yourself

### `Homo_sapiens.GRCh38.100.txt` (~10.6 MB)

Ensembl 100 gene annotation, formatted as a CSV indexed by Ensembl gene ID
with at least the columns `gene_name`, `chrom`, `chromStart`, `chromEnd`. Used
for placing RNA features on the chromosome-scatter loading plots in
Notebook 01.

Download from
[Ensembl release 100](https://ftp.ensembl.org/pub/release-100/gtf/homo_sapiens/):
the file `Homo_sapiens.GRCh38.100.gtf.gz` is the canonical GTF; run a small
extraction to derive the columns above. Public domain under
[the EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use).

In the file used for the published figures, `chromStart` / `chromEnd` are the
coordinates of a single sub-gene record of each gene (one row per gene: an
exon, UTR, CDS or start/stop-codon line of the Ensembl 100 GTF), not the gene
span. The notebooks use `chromStart` only to place each gene along its
chromosome in the loading plots, so a gene-span extraction gives the same
plots to within a gene's length. `scripts/11_eqtls.py` also reads
`chromStart` / `chromEnd` (for RNA and protein features), for its ±1 Mb
cis/trans label: the window is set around that record, not around the
whole gene, and the SNP positions it is compared with are on NCBI36, so the
label can be wrong (see `scripts/README.md`, step 11).

### `ROSMAP_arrayMethylation_metaData.tsv` (~34 MB)

Methylation array (Illumina HumanMethylation450; 420,132 probes) manifest,
joined to the ROSMAP-cohort CpG IDs with chromosome (`CHR`), genomic position
(`MAPINFO`), and gene annotation (`RefGene`) columns. Used for placing CpG
features on the chromosome-scatter loading plots in Notebook 01 and for
mapping CpGs to gene symbols (every `RefGene` gene of each CpG) for KEGG
enrichment in Notebook 02.

The full Illumina manifest is available from
[Illumina Infinium HumanMethylation450 product support](https://support.illumina.com/array/array_kits/infinium_humanmethylation450_beadchip_kit/downloads.html).
The ROSMAP-derived subset accompanies the AMP-AD methylation deposit
(Synapse folder `syn3157275`, "Epigenetics (DNA methylation array)", which
holds a file of this name, `syn3168775`); request through the
[AD Knowledge Portal DUC](https://adknowledgeportal.synapse.org/Data%20Access).

Place the downloaded file in this directory as
`ROSMAP_arrayMethylation_metaData.tsv`.

### `H3K9acDomains.csv` (~1.4 MB)

H3K9ac ChIP-seq peak coordinates from the Klein *et al.* ROSMAP H3K9ac study.
Columns: `Chr`, `Start`, `End`, `Width`, `Classification` (Promoter /
Enhancer / Other), `MedianCount`, `DetectedInNumSamples`. Used for placing
ChIP features on the chromosome-scatter loading plots in Notebook 01.

The peak set was originally published as supplementary data alongside

> Klein H.-U. *et al.* Epigenome-wide study uncovers large-scale changes in
> histone acetylation driven by tau pathology in aging and Alzheimer's human
> brains. *Nature Neuroscience*, 2019.
> [DOI: 10.1038/s41593-018-0291-1](https://doi.org/10.1038/s41593-018-0291-1).

Obtain through the AMP-AD H3K9ac deposit on the AD Knowledge Portal
(Synapse folder `syn4896408`, "Epigenetics (ChIP Seq)") and reformat to the
column names listed above. The CSV is small (~1.4 MB) so caching it locally
is straightforward.

## Optional: clinical metadata

`ROSMAP_clinical.csv` from the ROSMAP cohort is required by Notebook 03
(clinical-association tests) and optional for Notebook 01 (sex overlay on
sample scatter). It is **controlled access**; obtain it through the
[AD Knowledge Portal DUC](https://adknowledgeportal.synapse.org/Data%20Access)
and place it in this directory, set `G_ROS_CLINICAL_PATH`, or update the
`CLINICAL_PATH` variable in Notebook 03.

The notebooks degrade gracefully when the file is absent (Notebook 03 raises
a clear `FileNotFoundError` with the DUC link; Notebook 01 falls back to
sample scatter without sex/age annotation).
