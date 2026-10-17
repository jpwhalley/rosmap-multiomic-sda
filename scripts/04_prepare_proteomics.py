"""
04_prepare_proteomics.py
========================

Subset the log2-TMT proteomics matrix to the 276 snRNA-seq participants and
to proteins with no missing values among the participants who do have
proteomics data.

Manuscript: Whalley J.P. et al., *Multi-omic Analysis of Sex and Alzheimer's
Disease in the Aged Human Cortex* (under review at npj Aging).
Methods §"Proteomics".

Original filename in the working tree: `ros1d_prepare_data_imputed_prot_tmt.py`.

Pipeline:
1. Load the fully processed log2-TMT matrix of ROSMAP round 1 (400 individuals;
   Johnson et al. 2020; Synapse syn17008935, file syn21266454; columns are TMT
   batch.SampleIDs).
2. Use the proteomics assay metadata to map TMT batch.SampleID → R_ID, then
   to ROSMAP individualID and projid.
3. Reindex to the 276 snRNA-seq projids; 171 participants lack proteomics.
4. Apply a stringent feature filter: drop any protein with a missing value
   in any of the 105 participants who *do* have proteomics — i.e. require
   the protein to be quantified in every present sample (no imputation).

Output: `prot/tmt/prot_crossed_with_sc.csv` — a 5,230 protein × 276 sample
log2-TMT matrix, with NaN columns for the 171 missing-proteomics samples.

Known error, kept as run (manuscript Methods, "Proteomics"): the assay
metadata covers two TMT studies that share 126 batch.channel identifiers, and
the SampleID → R_ID dictionary in step 2 keeps the last metadata row for each.
112 of the 400 dorsolateral prefrontal cortex columns are therefore attributed
to individuals of the other study: 29 of the 105 proteomes in the SDA input
belong to other donors, and 28 donors' own proteomes are left out. The code is
unchanged because this is the input the decomposition used; no protein-derived
results are reported. The counts are checked in
`scripts/round2/covariates/check/t1_tmt_mapping_check.py`, and the corrected
mapping is used by the fixed-loading analysis in `scripts/round2/fixed_loading/`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. ROSMAP raw data are obtained from the
# AD Knowledge Portal under the SynIDs cited in the manuscript Methods.
# ----------------------------------------------------------------------------
CLINICAL_CSV = "oth/meta/ROSMAP_clinical.csv"            # syn3219045/Data/Metadata

# ROSMAP round-1 TMT proteomics (Johnson et al. 2020): the long filename below is the published quant
# matrix as distributed on Synapse (8817 proteins × 400 TMT channels).
TMT_MATRIX = (
    "prot/tmt/"
    "C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)"
    "-8817x400.csv"
)
TMT_ASSAY_META = "prot/tmt/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv"

SNRNA_PSEUDOBULK = "mit/rna/pseudobulk/Oli_rpm_normalized.csv"  # canonical projid list

OUTPUT_PATH = "prot/tmt/prot_crossed_with_sc.csv"


def main() -> None:
    # ------------------------------------------------------------------------
    # 1. Load metadata + the snRNA-seq projid anchor
    # ------------------------------------------------------------------------
    clin = pd.read_csv(CLINICAL_CSV, index_col=0)
    sc = pd.read_csv(SNRNA_PSEUDOBULK, index_col=0)
    samples = list(sc.index.values)
    sc_clin = clin.loc[samples]

    # ------------------------------------------------------------------------
    # 2. Load TMT matrix + assay metadata; build SampleID → R_ID mapping
    # ------------------------------------------------------------------------
    tmt = pd.read_csv(TMT_MATRIX, index_col=0)
    tmt_meta = pd.read_csv(TMT_ASSAY_META, index_col=0)

    # The metadata index has the form "...batchID.channelID.R_ID";
    # split into SampleID = "batchID.channelID" and R_ID = "R_ID".
    parts = tmt_meta.index.str.split(".")
    tmt_meta["R_ID"] = parts.str[-1]
    tmt_meta["SampleID"] = parts.str[-3] + "." + parts.str[-2]
    # Known error (see the module docstring): SampleIDs shared by the two TMT
    # studies are not unique here, and to_dict() keeps the last row for each.
    sampleid_to_rid = (
        tmt_meta[["R_ID", "SampleID"]].set_index("SampleID")["R_ID"].to_dict()
    )

    # ------------------------------------------------------------------------
    # 3. Rename TMT columns from SampleID → R_ID (= ROSMAP individualID)
    # ------------------------------------------------------------------------
    common_sample_ids = tmt.columns.intersection(pd.Index(list(sampleid_to_rid)))
    tmt_filtered = tmt[common_sample_ids].rename(columns=sampleid_to_rid)
    print(f"After SampleID→R_ID renaming: {tmt_filtered.shape}")

    # ------------------------------------------------------------------------
    # 4. Restrict to individualIDs that overlap the snRNA-seq cohort, drop
    #    proteins with any missing value in present samples, then reindex
    #    columns to all 276 projids and rename individualID → projid.
    # ------------------------------------------------------------------------
    common_individuals = [
        individual
        for individual in sc_clin["individualID"].values
        if individual in tmt_filtered.columns
    ]
    tmt_overlap = tmt_filtered[common_individuals].dropna(how="any")

    tmt_with_all_samples = tmt_overlap.reindex(
        columns=sc_clin["individualID"].values, fill_value=np.nan
    )
    individual_to_projid = dict(zip(sc_clin["individualID"], sc_clin.index))
    tmt_final = tmt_with_all_samples.rename(columns=individual_to_projid)

    # ------------------------------------------------------------------------
    # 5. Sanity checks
    # ------------------------------------------------------------------------
    print(f"Output shape: {tmt_final.shape}  (proteins × samples)")
    print(f"Columns aligned to snRNA-seq projid order: "
          f"{tmt_final.columns.equals(sc.index)}")
    print(f"Samples with all-NaN proteomics: "
          f"{int(tmt_final.isna().all().sum())} (should be 171 per Methods)")

    tmt_final.to_csv(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
