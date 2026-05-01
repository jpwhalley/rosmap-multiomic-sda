"""
02_prepare_chip.py
==================

Harmonise the H3K9ac ChIP-seq peak count matrix to the snRNA-seq cohort.

Manuscript: Whalley J.P. et al., *Multi-omic Glial Programs in Brain Aging*
(under review). Methods §"H3K9ac ChIP-Sequencing".

Original filename in the working tree: `ros1b_prepare_data_chip.py`.

Pipeline:
1. Load the Klein 2019 H3K9ac peak read-count matrix (Synapse syn3157275),
   which is indexed by sequencing-run @RG IDs rather than ROSMAP individualIDs.
2. Use the BAM-header-derived `bam_id_mapping.csv` (see
   `_document_bam_id_mapping` below for how that file was generated) to rename
   the columns from RG IDs back to BAM filenames, whose prefixes are ROSMAP
   individualIDs.
3. Cross-reference to the snRNA-seq cohort via `ROSMAP_clinical.csv`
   (mapping individualID → projid) and reindex columns to the 276 snRNA-seq
   projids, with NaN for the 83 participants who lack ChIP-seq data.
4. Library-size normalise to log2(CPM+1), preserving NaN for absent samples.

Output: `oth/chip/chip_crossed_with_sc.csv` — a 26,384 peak × 276 sample
log2(CPM+1) matrix, written with peak IDs as the row index and ROSMAP
projids as columns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. ROSMAP raw data are obtained from the
# AD Knowledge Portal under the SynIDs cited in the manuscript Methods.
# ----------------------------------------------------------------------------
CLINICAL_CSV = "oth/meta/ROSMAP_clinical.csv"            # syn3219045/Data/Metadata
CHIP_READ_COUNTS = "oth/chip/ReadCounts.csv"             # Klein 2019, syn3157275 (TSV)
BAM_ID_MAPPING = "oth/chip/bam_id_mapping.csv"           # see _document_bam_id_mapping()
SNRNA_PSEUDOBULK = "mit/rna/pseudobulk/Oli_rpm_normalized.csv"  # one of the six output of `01_prepare_snrna.py`; used here as the canonical projid list
METH_CROSSED = "oth/meth/meth_crossed_with_sc.csv"       # output of `03_prepare_meth.py`; used for sanity-checking column alignment

OUTPUT_PATH = "oth/chip/chip_crossed_with_sc.csv"


# ----------------------------------------------------------------------------
# BAM → @RG ID mapping (already-done step, kept here as documentation)
# ----------------------------------------------------------------------------
def _document_bam_id_mapping():
    """Extract the @RG ID line from every ChIP-seq BAM header and write a
    CSV mapping BAM filename → @RG ID. The Klein 2019 read-count table has
    @RG IDs as columns, so this mapping is needed to relink columns to BAM
    filenames, whose prefixes are ROSMAP individualIDs.

    Pseudocode (the original block run once on the BAM directory):

        import os, csv, re, subprocess
        bam_id_dict = {}
        rg_id_pattern = re.compile(r'^@RG\\s+ID:(\\S+)')
        for bam in (f for f in os.listdir('.') if f.endswith('.bam')):
            result = subprocess.run(
                ['samtools', 'view', '-H', bam],
                stdout=subprocess.PIPE, text=True,
            )
            for line in result.stdout.splitlines():
                m = rg_id_pattern.match(line)
                if m:
                    bam_id_dict[bam.removesuffix('.bam')] = m.group(1)
                    break
        with open('bam_id_mapping.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['BAM_File', 'RG_ID'])
            for bam, rg in bam_id_dict.items():
                w.writerow([bam, rg])

    Re-running requires `samtools` on the path and access to the raw BAM
    files. The cached mapping (`bam_id_mapping.csv`) is what the rest of
    this script consumes.
    """
    raise NotImplementedError(
        "This function documents the upstream BAM-header parsing step. "
        "Re-implement against the raw BAMs if you need to regenerate "
        "`bam_id_mapping.csv`."
    )


def main() -> None:
    # ------------------------------------------------------------------------
    # 1. Load metadata + the snRNA-seq projid anchor
    # ------------------------------------------------------------------------
    clin = pd.read_csv(CLINICAL_CSV, index_col=0)
    sc = pd.read_csv(SNRNA_PSEUDOBULK, index_col=0)
    samples = list(sc.index.values)
    sc_clin = clin.loc[samples]

    # ------------------------------------------------------------------------
    # 2. Read ChIP read counts (TSV; columns are @RG IDs)
    # ------------------------------------------------------------------------
    chip_data = pd.read_csv(CHIP_READ_COUNTS, index_col=0, sep="\t")

    # ------------------------------------------------------------------------
    # 3. Rename columns: @RG ID → BAM filename → ROSMAP individualID → projid
    # ------------------------------------------------------------------------
    bam_to_rgid = pd.read_csv(BAM_ID_MAPPING, index_col=0)
    rgid_to_bam = bam_to_rgid.reset_index().set_index("RG_ID")["BAM_File"].to_dict()
    chip_renamed = chip_data.rename(columns=rgid_to_bam)

    # In the Klein 2019 BAM filenames the part before any version suffix is
    # the ROSMAP individualID. We filter to individuals with snRNA-seq data
    # and rename columns to projid.
    indid_to_projid = {
        ros: scid
        for scid, ros in zip(sc_clin.index.values, sc_clin.individualID.values)
        if ros in chip_renamed.columns
    }
    chip_filtered = chip_renamed[list(indid_to_projid.keys())]
    chip_by_projid = chip_filtered.rename(columns=indid_to_projid)

    # Reindex to the 276 snRNA-seq projids (NaN for the 83 missing)
    chip_aligned = chip_by_projid.reindex(columns=sc.index, fill_value=np.nan)

    # ------------------------------------------------------------------------
    # 4. log2(CPM+1) normalisation, preserving NaN for absent samples
    # ------------------------------------------------------------------------
    library_sizes = chip_aligned.sum()
    cpm = chip_aligned.div(library_sizes) * 1e6
    log2_cpm = np.log2(cpm + 1).where(~chip_aligned.isna(), np.nan)

    # ------------------------------------------------------------------------
    # 5. Sanity checks
    # ------------------------------------------------------------------------
    print(f"Output shape: {log2_cpm.shape}  (peaks × samples)")
    print(f"Columns aligned to snRNA-seq projid order: "
          f"{log2_cpm.columns.equals(sc.index)}")
    print(f"Samples with all-NaN ChIP signal: "
          f"{int(log2_cpm.isna().all().sum())} (should be 83 per Methods)")

    try:
        meth = pd.read_csv(METH_CROSSED, index_col=0, nrows=0)
        print(f"Methylation column order matches: "
              f"{log2_cpm.columns.astype(str).equals(meth.columns)}")
    except FileNotFoundError:
        print(f"(Methylation cross-check skipped — {METH_CROSSED} not present yet)")

    log2_cpm.to_csv(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
