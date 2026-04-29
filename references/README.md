# Reference annotations

The notebooks need three small public reference tables in addition to the
Synapse deposit. None are bundled in this repository (we keep the GitHub
release code-only); each is documented below with download instructions and
should be placed in this directory.

Notebook 00 runs without any of these. Notebook 01 needs all three for the
chromosome-scatter loadings panels. Notebook 02 needs the gene annotation and
the methylation manifest (for CpG-to-gene mapping in KEGG enrichment).
Notebook 03 needs none of these (only the clinical CSV — see "Optional:
clinical metadata" at the bottom).

## Download yourself

### `Homo_sapiens.GRCh38.100.txt` (~10.6 MB)

Ensembl 100 gene annotation, formatted as a CSV indexed by Ensembl gene ID
with at least the columns `gene_name`, `chrom`, `chromStart`, `chromEnd`. Used
for placing RNA and protein features on the chromosome-scatter loading plots
in Notebook 01 and for mapping protein features to gene symbols in Notebook 02.

Download from [Ensembl release 100]
(https://ftp.ensembl.org/pub/release-100/gtf/homo_sapiens/) — the file
`Homo_sapiens.GRCh38.100.gtf.gz` is the canonical GTF; run a small extraction
to derive the columns above. Public domain under [the EMBL-EBI terms of use]
(https://www.ebi.ac.uk/about/terms-of-use).

### `ROSMAP_arrayMethylation_metaData.tsv` (~34 MB)

Methylation array (Illumina MethylationEPIC v1.0 / 450K) manifest, joined to
the ROSMAP-cohort CpG IDs with chromosome (`CHR`), genomic position
(`MAPINFO`), and nearest-gene (`RefGene`) columns. Used for placing CpG
features on the chromosome-scatter loading plots in Notebook 01 and for
mapping CpGs to gene symbols for KEGG enrichment in Notebook 02.

The full Illumina manifests are available from
[Illumina Infinium MethylationEPIC product support]
(https://support.illumina.com/array/array_kits/infinium-methylationepic-beadchip-kit/downloads.html).
The ROSMAP-derived subset accompanies the AMP-AD methylation deposit; request
through the [AD Knowledge Portal DUC]
(https://adknowledgeportal.synapse.org/Data%20Access).

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
(syn4896408 sub-project) and reformat to the column names listed above. The
CSV is small (~1.4 MB) so caching it locally is straightforward.

## Optional: clinical metadata

`ROSMAP_clinical.csv` from the ROSMAP cohort is required by Notebook 03
(clinical-association tests) and optional for Notebook 01 (sex overlay on
sample scatter). It is **controlled access**; obtain it through the
[AD Knowledge Portal DUC]
(https://adknowledgeportal.synapse.org/Data%20Access) and place it in this
directory or update the `CLINICAL_PATH` variable in Notebook 03.

The notebooks degrade gracefully when the file is absent (Notebook 03 raises
a clear `FileNotFoundError` with the DUC link; Notebook 01 falls back to
sample scatter without sex/age annotation).
