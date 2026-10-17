"""
03_prepare_meth.py
==================

Filter the imputed DNA methylation β-value matrix to the 276 snRNA-seq
participants and to variable CpG sites.

Manuscript: Whalley J.P. et al., *Multi-omic Analysis of Sex and Alzheimer's
Disease in the Aged Human Cortex* (under review at npj Aging).
Methods §"DNA methylation".

Original filename in the working tree: `ros1c_prepare_data_imputed_meth.py`.

Pipeline:
1. Load the De Jager 2014 imputed β-value matrix (Synapse syn3157275),
   which is indexed by sequencing specimenIDs.
2. Use the biospecimen metadata to map specimenID → individualID → projid,
   reindex columns to the 276 snRNA-seq projids (NaN for the 60
   participants without methylation data).
3. Drop CpG sites that are uninformative under a binarisation criterion:
   compute a temporary binarised matrix (β ≥ 0.5 → methylated, β < 0.5 →
   unmethylated, NaN preserved); drop CpGs whose binarised values are
   constant across all non-NaN samples.
4. Surviving CpGs retain their **continuous β values** for SDA — the
   binarisation is *only* used as a variability filter.

Output: `oth/meth/meth_crossed_with_sc.csv` — a 51,798 CpG × 276 sample
β-value matrix, with NaN preserved for the 60 missing-methylation samples.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. ROSMAP raw data are obtained from the
# AD Knowledge Portal under the SynIDs cited in the manuscript Methods.
# ----------------------------------------------------------------------------
CLINICAL_CSV = "oth/meta/ROSMAP_clinical.csv"            # syn3219045/Data/Metadata
BIOSPECIMEN_CSV = "oth/meta/ROSMAP_biospecimen_metadata.csv"  # same source
METH_IMPUTED_TSV = "oth/meth/ROSMAP_arrayMethylation_imputed.tsv.gz"  # De Jager 2014, syn3157275
SNRNA_PSEUDOBULK = "mit/rna/pseudobulk/Oli_rpm_normalized.csv"  # output of `01_prepare_snrna.py`; canonical projid list

OUTPUT_PATH = "oth/meth/meth_crossed_with_sc.csv"


def main() -> None:
    # ------------------------------------------------------------------------
    # 1. Load metadata + the snRNA-seq projid anchor
    # ------------------------------------------------------------------------
    clin = pd.read_csv(CLINICAL_CSV, index_col=0)
    bio = pd.read_csv(BIOSPECIMEN_CSV, index_col=1)
    sc = pd.read_csv(SNRNA_PSEUDOBULK, index_col=0)
    samples = list(sc.index.values)
    sc_clin = clin.loc[samples]

    # ------------------------------------------------------------------------
    # 2. Load imputed β-values; columns are specimenIDs
    # ------------------------------------------------------------------------
    me = pd.read_csv(METH_IMPUTED_TSV, index_col=0, sep="\t")

    # ------------------------------------------------------------------------
    # 3. Map specimenID → individualID → projid; rename columns
    # ------------------------------------------------------------------------
    meth_bio = bio.loc[me.columns.values]
    meth_clin = clin.copy()
    meth_clin["projid"] = meth_clin.index
    meth_clin.index = meth_clin["individualID"]
    meth_clin = meth_clin.loc[meth_bio.individualID]

    indid_to_projid = meth_clin["projid"].to_dict()
    specimen_to_indid = meth_bio["individualID"].to_dict()
    specimen_to_projid = {
        spec: indid_to_projid[indid]
        for spec, indid in specimen_to_indid.items()
        if indid in indid_to_projid
    }

    me_renamed = me.rename(columns=specimen_to_projid)
    me_aligned = me_renamed.reindex(columns=samples, fill_value=np.nan)

    # ------------------------------------------------------------------------
    # 4. Variability filter via binarisation (vectorised — much faster than
    #    a per-cell .applymap on a 850K × 276 matrix). The binarised matrix
    #    is *only* used to identify variable CpGs; the surviving rows keep
    #    their continuous β values.
    # ------------------------------------------------------------------------
    binarised = (me_aligned >= 0.5).astype("float").where(me_aligned.notna())
    # A CpG has variation iff it has both 0s and 1s among its non-NaN values.
    has_zero = (binarised == 0).any(axis=1)
    has_one = (binarised == 1).any(axis=1)
    keep = has_zero & has_one
    final_df = me_aligned.loc[keep[keep].index]

    # ------------------------------------------------------------------------
    # 5. Sanity checks
    # ------------------------------------------------------------------------
    final_df = final_df.reindex(columns=sc.index)  # belt-and-braces re-order
    print(f"Output shape: {final_df.shape}  (CpGs × samples)")
    print(f"Columns aligned to snRNA-seq projid order: "
          f"{final_df.columns.equals(sc.index)}")
    print(f"Samples with all-NaN methylation: "
          f"{int(final_df.isna().all().sum())} (should be 60 per Methods)")

    final_df.to_csv(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
