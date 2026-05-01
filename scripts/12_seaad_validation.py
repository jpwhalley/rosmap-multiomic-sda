"""
12_seaad_validation.py
======================

Cross-cohort validation of ROSMAP SDA components against the SEA-AD DLPFC
cohort: for each significant ROSMAP component, find the most-correlated (or
anti-correlated) SEA-AD component using shared PIP > 0.5 RNA features, plot
the scatter, and add `SEA-AD_comp` / `SEA-AD_corr` columns to the
component-annotation table that becomes Supplementary Table 1.

Manuscript: Whalley J.P. et al., *Multi-omic Glial Programs in Brain Aging*
(under review). Methods §"Cross-cohort validation"; underlies the SEA-AD
correlation panels of Figure 4 and Supplementary Figures S2 + S3.

Original filename in the working tree: `synth1b_croos_clin_manuscript.py`.

External dependencies:
- SEA-AD processed snRNA-seq from the Allen Institute SEA-AD AWS bucket
  (https://registry.opendata.aws/allen-sea-ad-atlas/). The original
  analysis decomposed the SEA-AD DLPFC slice with the same SDA pipeline as
  this study, producing `genes_final.csv` and a clinical-association
  `clin_attr_sig_results.csv` per SEA-AD component. Those outputs are the
  inputs here.

Pipeline:
1. Load SEA-AD per-omic SDA outputs and clinical-association table.
2. Load this study's ROSMAP per-omic outputs and the in-progress
   `significant_manu.csv` (the precursor table from
   `13_assemble_table_s1.py`).
3. Restrict both gene matrices to genes common across both cohorts.
4. For each significant ROSMAP component:
   - PIP-filter ROSMAP RNA features.
   - Compute Pearson correlation of the ROSMAP component's feature
     loadings against every SEA-AD component on the shared feature set.
   - Pick the SEA-AD component with the largest absolute correlation
     magnitude (positive or negative); record its ID, correlation value,
     and any associated clinical features.
   - Plot ROSMAP vs SEA-AD feature loadings as a scatter + regression line.
5. Add an `omics_count` column to the table by counting how many of the
   four `Top_Positive_*` / `Top_Negative_*` pairs have a non-null entry,
   then write `significant_df_validated_manu.csv` (this becomes Supplementary
   Table 1).

Outputs:
- `figs/outlier/<component>/<component>_dlpfc_correlation.pdf` per
  significant component.
- `results/significant_df_validated_manu.csv` — Supplementary Table 1.
"""
from __future__ import annotations
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="white", context="talk")

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. The SEA-AD output directory below is
# from a parallel SDA decomposition run on the Allen DLPFC slice; see the
# manuscript Methods §"Cross-cohort validation" for the SEA-AD pipeline.
# ----------------------------------------------------------------------------
SEAAD_FEATURE_NAMES_CSV = "../e_brain_td/data/dlpfc/gene_info.csv"
SEAAD_GENES_FINAL_CSV = "../e_brain_td/data/dlpfc/genes_final.csv"
SEAAD_CLIN_RESULTS_CSV = "../e_brain_td/results/dlpfc/clin_attr_sig_results.csv"

ROSMAP_GENES_FINAL_CSV = "results/genes_final_RNA.csv"
ROSMAP_PIP_FINAL_CSV = "results/pip_final_RNA.csv"

# `significant_manu.csv` is the precursor table written by
# `13_assemble_table_s1.py`. This script adds SEA-AD columns to it and
# writes the final `significant_df_validated_manu.csv`, which becomes
# Supplementary Table 1.
INPUT_SIGNIFICANT_CSV = "results/significant_manu.csv"
OUTPUT_TABLE_S1_CSV = "results/significant_df_validated_manu.csv"

PLOT_OUTPUT_TEMPLATE = "figs/outlier/{component}"

PIP_THRESHOLD = 0.5

OUTPUT_COLUMN_ORDER = [
    "omics_count",
    "Top_positive_cell_type", "Top_negative_cell_type",
    "Sex", "ceradsc", "cogdx",
    "SEA-AD_comp", "SEA-AD_corr",
    "Top_Positive_RNA", "Top_Negative_RNA", "GeneExpression_Pathway",
    "Top_Positive_meth", "Top_Negative_meth", "CpG_Site_Pathway",
    "Top_Positive_proteins", "Top_Negative_proteins", "Protein_Pathway",
    "Top_Positive_chip", "Top_Negative_chip", "ChIP_Motif",
]

# Column groups the omics_count tally is computed over
OMICS_COLUMN_GROUPS = {
    "RNA": ["Top_Positive_RNA", "Top_Negative_RNA"],
    "Meth": ["Top_Positive_meth", "Top_Negative_meth"],
    "Proteins": ["Top_Positive_proteins", "Top_Negative_proteins"],
    "ChIP": ["Top_Positive_chip", "Top_Negative_chip"],
}


def find_top_correlation(rosmap_vec: pd.Series, sead_df: pd.DataFrame):
    """Return (most-correlated SEA-AD component id, +corr value, most-anti-
    correlated id, −corr value) for a single ROSMAP component vector."""
    correlations = sead_df.apply(lambda row: row.corr(rosmap_vec), axis=1)
    return (
        correlations.idxmax(), correlations.max(),
        correlations.idxmin(), correlations.min(),
    )


def main() -> None:
    # ------------------------------------------------------------------------
    # 1. Load SEA-AD SDA outputs
    # ------------------------------------------------------------------------
    seaad_feature_names = pd.read_csv(SEAAD_FEATURE_NAMES_CSV, index_col=0)
    seaad_genes = pd.read_csv(SEAAD_GENES_FINAL_CSV, index_col=0)
    seaad_clin = pd.read_csv(SEAAD_CLIN_RESULTS_CSV, index_col=0)
    seaad_clin = seaad_clin[seaad_clin["reject"]]  # significant SEA-AD components only

    # SEA-AD's matrix indexes columns by internal feature IDs; convert to gene
    # symbols using the feature_names table so we can join with ROSMAP later.
    seaad_genes.rename(
        columns=seaad_feature_names["feature_name"].to_dict(),
        inplace=True,
    )

    # ------------------------------------------------------------------------
    # 2. Load this study's ROSMAP RNA loadings + PIPs + the in-progress table
    # ------------------------------------------------------------------------
    rosmap_genes = pd.read_csv(ROSMAP_GENES_FINAL_CSV, index_col=0)
    rosmap_pip = pd.read_csv(ROSMAP_PIP_FINAL_CSV, index_col=0)
    significant_df = pd.read_csv(INPUT_SIGNIFICANT_CSV, index_col=0)

    # ------------------------------------------------------------------------
    # 3. Restrict to common genes
    # ------------------------------------------------------------------------
    common_genes = sorted(set(seaad_genes.columns) & set(rosmap_genes.columns))
    seaad_genes = seaad_genes[common_genes]
    rosmap_genes = rosmap_genes[common_genes]
    rosmap_pip = rosmap_pip[common_genes]
    print(f"  common genes between cohorts: {len(common_genes)}")

    # ------------------------------------------------------------------------
    # 4. Per significant ROSMAP component: best SEA-AD partner + scatter
    # ------------------------------------------------------------------------
    significant_df[["SEA-AD_comp", "SEA-AD_corr", "SEA-AD_clin"]] = None

    for k in significant_df.index:
        pip_filter = rosmap_pip.T[rosmap_pip.loc[k] > PIP_THRESHOLD].index.values
        if len(pip_filter) <= 1:
            continue

        rosmap_vec = rosmap_genes[pip_filter].loc[k]
        seaad_subset = seaad_genes[pip_filter]

        # Pick whichever extreme (positive or negative correlation) has the
        # larger absolute magnitude — preserves SDA's sign-arbitrariness.
        d_max_id, d_max_corr, d_min_id, d_min_corr = find_top_correlation(
            rosmap_vec, seaad_subset
        )
        if abs(d_max_corr) >= abs(d_min_corr):
            seaad_id, seaad_corr = d_max_id, d_max_corr
        else:
            seaad_id, seaad_corr = d_min_id, d_min_corr

        # Look up any clinical features the SEA-AD component is tagged with
        clin_features = seaad_clin[seaad_clin["col"] == seaad_id]["clin_attr"]
        clin_str = ", ".join(clin_features.dropna().unique()) if not clin_features.empty else None

        significant_df.at[k, "SEA-AD_comp"] = seaad_id
        significant_df.at[k, "SEA-AD_corr"] = seaad_corr
        significant_df.at[k, "SEA-AD_clin"] = clin_str

        # ROSMAP-vs-SEA-AD scatter of the shared PIP-significant features
        plot_dir = PLOT_OUTPUT_TEMPLATE.format(component=k)
        os.makedirs(plot_dir, exist_ok=True)

        df_plot = pd.DataFrame({
            "ROSMAP": rosmap_vec.values,
            "SEA-AD": seaad_subset.loc[seaad_id].values,
        }, index=pip_filter)

        # Comp 74 panel is square (matches Fig 4 layout); others stay rectangular
        figsize = (6, 6) if k == 74 else (8, 6)
        fig, ax = plt.subplots(figsize=figsize)
        sns.regplot(x="ROSMAP", y="SEA-AD", data=df_plot, ax=ax, ci=None)

        corr_val = df_plot.corr().iloc[0, 1]
        ax.set_title(
            f"ROSMAP: {k} vs SEA-AD: {seaad_id}\n"
            f"Correlation (r) = {corr_val:.3f}"
        )
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{k}_dlpfc_correlation.pdf")
        plt.close(fig)

    # ------------------------------------------------------------------------
    # 5. Add omics_count + write Table S1
    # ------------------------------------------------------------------------
    omics_presence = pd.DataFrame({
        omic: significant_df[cols].notna().any(axis=1)
        for omic, cols in OMICS_COLUMN_GROUPS.items()
    })
    significant_df["omics_count"] = omics_presence.sum(axis=1)

    significant_df = significant_df[OUTPUT_COLUMN_ORDER]
    significant_df.index.name = "Component"
    significant_df.sort_index(inplace=True)
    significant_df.to_csv(OUTPUT_TABLE_S1_CSV, index=True)
    print(f"  wrote {OUTPUT_TABLE_S1_CSV}: {significant_df.shape}")


if __name__ == "__main__":
    main()
