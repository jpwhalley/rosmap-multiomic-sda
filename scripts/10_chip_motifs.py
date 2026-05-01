"""
10_chip_motifs.py
=================

For each SDA component, extract the H3K9ac peaks with PIP > 0.5, write them
as a BED file, and run HOMER `findMotifsGenome.pl` to test for transcription-
factor motif enrichment in those peak sequences.

Manuscript: Whalley J.P. et al., *Multi-omic Glial Programs in Brain Aging*
(under review). Methods §"H3K9ac ChIP-Sequencing" / motif-enrichment;
underlies Figure 4 (PRDM4 motif panel) and Supplementary Figure S2 (BARHL1-
like motif enrichment in Component 351).

Original filename in the working tree: `ros7di_chip_motifs.py`.

External dependencies:
- HOMER (Perl): http://homer.ucsd.edu/homer/ — specifically
  `findMotifsGenome.pl`. The HOMER `bin/` directory must be on PATH so that
  HOMER's helper scripts (configureHomer.pl, scanMotifGenomeWide.pl, etc.)
  resolve.
- A local genome FASTA. The original analysis used GRCh37.p13; if you
  are working in GRCh38 coordinates throughout the rest of the pipeline,
  liftOver the peaks first or use a GRCh38 FASTA. Either choice will produce
  slightly different motif p-values from the published figures.
Pipeline:
1. Load the deposited matrices and the public reference annotations
   (gene / methylation / ChIP). The prep block here mirrors what's in
   `notebooks/01_per_component_panels.ipynb` because the downstream
   pipeline expected the same in-memory state.
2. For each component, filter ChIP peaks with PIP > 0.5, write them as a
   BED file, and run HOMER. Results land in
   `results/chip/<component>/homer_results/` per component.

Outputs:
- `results/chip/<component>/significant_peaks.bed` — BED of PIP > 0.5 peaks
- `results/chip/<component>/homer_results/` — HOMER motif output
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
sns.set(style="white", context="talk")

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. The deposit and reference paths match
# those used by the figure-reproduction notebooks; see
# `references/README.md` for download sources.
# ----------------------------------------------------------------------------
GENES_FINAL_TEMPLATE = "results/genes_final_{omic}.csv"  # output of 07_cluster_components.py
PIP_FINAL_TEMPLATE = "results/pip_final_{omic}.csv"      # output of 07_cluster_components.py
CHIP_ANNOT_TSV = "data/H3K9acDomains.csv"

# HOMER configuration — adapt to your install
HOMER_BIN_DIR = "/Users/jpw/Documents/homer/bin"
HOMER_FIND_MOTIFS = os.path.join(HOMER_BIN_DIR, "findMotifsGenome.pl")
GENOME_FASTA = "/Users/jpw/Documents/sequences/GRCh37.p13.genome.fa"

CHIP_RESULTS_DIR = "results/chip"

OMIC_NAMES = {1: "RNA", 2: "meth", 3: "chip", 4: "proteins"}
PIP_THRESHOLD = 0.5


def load_genes_and_pip() -> tuple[dict, dict]:
    """Load the per-omic feature × component loading and PIP matrices."""
    genes = {}
    pip = {}
    for j, name in OMIC_NAMES.items():
        genes[j] = pd.read_csv(GENES_FINAL_TEMPLATE.format(omic=name), index_col=0)
        pip[j] = pd.read_csv(PIP_FINAL_TEMPLATE.format(omic=name), index_col=0)
    return genes, pip


def load_chip_info() -> pd.DataFrame:
    """Load and prepare the H3K9ac peak annotation."""
    chip_info = pd.read_csv(CHIP_ANNOT_TSV, index_col=0, sep="\t")
    chip_info["Chr"] = chip_info["Chr"].astype(str)
    chip_info["peak"] = chip_info.index
    # HOMER + many genome FASTAs use 'chr' prefixes; map plain chromosome
    # names to that convention. Non-numeric / non-X/Y values are left alone.
    chip_info["Chr"] = chip_info["Chr"].apply(
        lambda x: f"chr{x}" if str(x).isdigit() or x in ["X", "Y"] else x
    )
    return chip_info


def main() -> None:
    Path(CHIP_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Make HOMER's bin dir resolvable for any helper scripts that
    # `findMotifsGenome.pl` shells out to.
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + HOMER_BIN_DIR

    genes, pip = load_genes_and_pip()
    chip_info = load_chip_info()

    print(f"ChIP peak chromosome breakdown:\n{chip_info['Chr'].value_counts()}\n")

    # Iterate components — the third omic (chip) is the one we run motifs on.
    for k in genes[3].index.values:
        print(f"Processing Component {k}…")

        component_dir = f"{CHIP_RESULTS_DIR}/{k}"
        homer_output_dir = f"{component_dir}/homer_results"

        if Path(homer_output_dir).exists():
            print(f"  HOMER results already exist for component {k}; skipping.")
            continue

        # PIP-filter ChIP peaks for this component
        pip_filter = pip[3].T[pip[3].loc[k] > PIP_THRESHOLD].index.values
        if len(pip_filter) == 0:
            print(f"  No PIP > {PIP_THRESHOLD} peaks for component {k}; skipping.")
            continue

        Path(component_dir).mkdir(parents=True, exist_ok=True)

        # Subset peak metadata to the significant peaks and write a 3-column BED
        peaks_subset = chip_info.loc[pip_filter]
        bed_path = f"{component_dir}/significant_peaks.bed"
        peaks_subset[["Chr", "Start", "End"]].to_csv(
            bed_path, sep="\t", header=False, index=False
        )
        print(f"  wrote {bed_path} ({len(peaks_subset)} peaks)")

        # Run HOMER. The `findMotifsGenome.pl <bed> <genome> <out>` signature
        # accepts either an installed-genome name (e.g. `hg19`) or an
        # explicit FASTA path; we use the explicit FASTA so the genome
        # version is unambiguous.
        cmd = [HOMER_FIND_MOTIFS, bed_path, GENOME_FASTA, homer_output_dir]
        print(f"  running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✓ HOMER completed for component {k}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ HOMER failed for component {k}: {e}")


if __name__ == "__main__":
    main()
