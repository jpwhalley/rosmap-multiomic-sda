"""
13_assemble_table_s1.py
=======================

Assemble the per-component annotation table that becomes Data S1 of the
manuscript (called Table S1 in earlier drafts, hence the file name).
Integrates outputs from:

- ros6a_cross_clin_discrete (Sex / ceradsc / cogdx p-values; pre-computed
  by the clinical-association notebook 03 / equivalent pipeline)
- KEGG pathway enrichment per omic (`results/enrichr/all_kegg/`,
  `results/cpg_enrichr/`, `results/prot_enrichr/`) — produced by
  notebook 02 / pre-computed
- HOMER motif results from `10_chip_motifs.py` (parses the per-component
  `homer_results/homerResults.html`)
- Cell-type drivers from `treatment_final.csv` (`07_cluster_components.py`)
- Top features per omic, PIP-filtered, from `genes_final_*.csv` +
  `pip_final_*.csv` (also from `07`)

Manuscript: Whalley J.P. et al., *Multi-omic Analysis of Sex and Alzheimer's
Disease in the Aged Human Cortex* (under review at npj Aging).
This script builds the version-1 format of the table (with protein and
`ChIP_Motif` columns). Data S1 of this version is built by
`scripts/round2/data_s1/build_Data_S1.py`.

Original filename in the working tree: `ros9b_summarize_manuscript.py`.

External dependencies:
- `beautifulsoup4` for parsing HOMER's `homerResults.html` output table.

Pipeline:
1. Load the precursor `significant_clinical_features_manu.csv` carrying
   Sex / ceradsc / cogdx p-values per component. **This precursor file is
   produced by `notebooks/03_clinical_associations.ipynb`** — run that
   notebook first with the saving cell pointed at this filename. Expected
   columns: an index of `Component_<n>` strings (or integer component IDs
   after the `.replace("Component_", "").astype(int)` normalisation in
   step 1 of `main()` below) and at least the columns `Sex`, `ceradsc`,
   `cogdx` (the BH-adjusted p-values from the manuscript's clinical-
   association tests).
2. For each pathway omic (RNA / meth / proteins): pick the most-
   significant KEGG term per component (smallest Adjusted P-value below
   0.05) from the per-omic enrichr output directory.
3. For each component: parse `homer_results/homerResults.html` from
   `10_chip_motifs.py` and pick the top non-false-positive motif name.
4. Add cell-type drivers (most-positive / most-negative cell-type for each
   component from `treatment_final.csv`). In the published Data S1 a label
   is shown only where the individual SDA runs support that direction (see
   the Data S1 legend), so a few labels written here are blank there.
5. Add top positive / negative PIP > 0.5 feature per omic per component.
6. Reorder columns, write to `results/significant_manu.csv` (the input
   table for `12_seaad_validation.py`).

Output: `results/significant_manu.csv` — feeds into `12_seaad_validation.py`,
which appends SEA-AD columns and emits Data S1.
"""
from __future__ import annotations
import os

import pandas as pd
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. Many of the inputs are produced by
# preceding pipeline stages; see each path's comment for the producer.
# ----------------------------------------------------------------------------
INPUT_SIGNIFICANT_CLINICAL_CSV = "results/significant_clinical_features_manu.csv"  # from clinical-association pipeline
TREATMENT_FINAL_CSV = "results/treatment_final.csv"              # from 07_cluster_components.py
GENES_FINAL_TEMPLATE = "results/genes_final_{omic}.csv"          # from 07_cluster_components.py
PIP_FINAL_TEMPLATE = "results/pip_final_{omic}.csv"              # from 07_cluster_components.py

# KEGG enrichr outputs (one CSV per component per omic). Produced upstream
# (the per-omic enrichment scripts; ros7ai/bi/ci → notebook 02).
ENRICHR_DIRS = {
    "GeneExpression_Pathway": "results/enrichr/all_kegg/{component}_pathway_encrichment.csv",
    "CpG_Site_Pathway":       "results/cpg_enrichr/{component}_pathway_encrichment.csv",
    "Protein_Pathway":        "results/prot_enrichr/{component}_pathway_encrichment.csv",
}

# HOMER output directory (one subdir per component, from 10_chip_motifs.py)
HOMER_BASE_DIR = "results/chip"

# Clinical metadata for the optional summary printouts at the end
CLINICAL_CSV = "results/ROSMAP_clinical.csv"
SAMPLES_FINAL_CSV = "results/samples_final.csv"

OUTPUT_PATH = "results/significant_manu.csv"

ENRICHMENT_PADJ_THRESHOLD = 0.05

OMIC_NAMES = {1: "RNA", 2: "meth", 3: "chip", 4: "proteins"}

OUTPUT_COLUMN_ORDER = [
    "Top_positive_cell_type", "Top_negative_cell_type",
    "cogdx", "Sex", "ceradsc",
    "Top_Positive_RNA", "Top_Negative_RNA", "GeneExpression_Pathway",
    "Top_Positive_meth", "Top_Negative_meth", "CpG_Site_Pathway",
    "Top_Positive_proteins", "Top_Negative_proteins", "Protein_Pathway",
    "Top_Positive_chip", "Top_Negative_chip", "ChIP_Motif",
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def add_top_kegg_pathway(
    df: pd.DataFrame,
    column_name: str,
    enrich_path_template: str,
) -> None:
    """For each component in `df.index`, read its KEGG enrichment CSV and
    write the most-significant pathway (smallest Adjusted P-value, gated at
    `ENRICHMENT_PADJ_THRESHOLD`) into `df[column_name]`.

    The original code had three near-identical loops for RNA, meth, and
    proteins — this helper collapses them into a single call site each.
    """
    df[column_name] = pd.NA
    for k in df.index:
        path = enrich_path_template.format(component=k)
        try:
            enr = pd.read_csv(path, index_col=0)
        except FileNotFoundError:
            continue
        sig = enr[enr["Adjusted P-value"] < ENRICHMENT_PADJ_THRESHOLD]
        if not sig.empty:
            best = sig.loc[sig["Adjusted P-value"].idxmin()]
            df.at[k, column_name] = best["Term"]


def add_homer_top_motif(df: pd.DataFrame, base_dir: str) -> None:
    """For each component, parse `<base_dir>/<component>/homer_results/
    homerResults.html` and pick the top non-false-positive motif name."""
    df["ChIP_Motif"] = pd.NA
    for component in df.index:
        html = os.path.join(base_dir, str(component), "homer_results", "homerResults.html")
        if not os.path.exists(html):
            continue
        try:
            with open(html) as f:
                soup = BeautifulSoup(f, "html.parser")
            table = soup.find("table")
            if not table:
                continue
            for row in table.find_all("tr")[1:]:  # skip header
                cols = row.find_all("td")
                if not cols:
                    continue
                # HOMER marks false-positive hits with '*' in the rank column.
                # Skip those; take the first non-FP motif.
                if "*" in cols[0].get_text(strip=True):
                    continue
                df.at[component, "ChIP_Motif"] = cols[1].get_text(strip=True)
                break
        except Exception as exc:  # noqa: BLE001
            print(f"Error parsing HOMER HTML for component {component}: {exc}")


def add_cell_type_drivers(df: pd.DataFrame, treatment_path: str) -> None:
    """Add `Top_positive_cell_type` and `Top_negative_cell_type` columns
    based on the cell-type loadings matrix."""
    tissues = pd.read_csv(treatment_path, index_col=0)
    df["Top_positive_cell_type"] = pd.NA
    df["Top_negative_cell_type"] = pd.NA
    for component, row in tissues.iterrows():
        df.loc[component, "Top_positive_cell_type"] = row.idxmax() if row.max() > 0 else pd.NA
        df.loc[component, "Top_negative_cell_type"] = row.idxmin() if row.min() < 0 else pd.NA


def add_top_features_per_omic(df: pd.DataFrame, genes: dict, pip: dict) -> None:
    """For each omic, add `Top_Positive_<omic>` / `Top_Negative_<omic>`
    columns containing the strongest-loading PIP > 0.5 feature in each
    direction."""
    for j, omic in OMIC_NAMES.items():
        pos_col = f"Top_Positive_{omic}"
        neg_col = f"Top_Negative_{omic}"
        df[pos_col] = pd.NA
        df[neg_col] = pd.NA

        for k in df.index:
            k = int(k)
            pip_features = genes[j][pip[j].T[pip[j].loc[k] > 0.5].index.values]
            if pip_features.empty:
                continue
            row = pip_features.loc[k]
            top_pos = row[row > 0].nlargest(1).index.tolist()
            top_neg = row[row < 0].nsmallest(1).index.tolist()
            if top_pos:
                df.loc[k, pos_col] = ", ".join(top_pos)
            if top_neg:
                df.loc[k, neg_col] = ", ".join(top_neg)


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------------
    # 1. Load precursor table (already has Sex / ceradsc / cogdx p-values)
    # ------------------------------------------------------------------------
    df = pd.read_csv(INPUT_SIGNIFICANT_CLINICAL_CSV, index_col=0)
    df.index = df.index.str.replace("Component_", "").astype(int)

    # ------------------------------------------------------------------------
    # 2. Top KEGG pathway per omic (one helper call replaces three loops)
    # ------------------------------------------------------------------------
    for column_name, template in ENRICHR_DIRS.items():
        add_top_kegg_pathway(df, column_name, template)

    # ------------------------------------------------------------------------
    # 3. Top HOMER motif per component
    # ------------------------------------------------------------------------
    add_homer_top_motif(df, HOMER_BASE_DIR)

    # ------------------------------------------------------------------------
    # 4. Cell-type drivers
    # ------------------------------------------------------------------------
    add_cell_type_drivers(df, TREATMENT_FINAL_CSV)

    # ------------------------------------------------------------------------
    # 5. Top features per omic, PIP-filtered
    # ------------------------------------------------------------------------
    genes = {j: pd.read_csv(GENES_FINAL_TEMPLATE.format(omic=name), index_col=0)
             for j, name in OMIC_NAMES.items()}
    pip = {j: pd.read_csv(PIP_FINAL_TEMPLATE.format(omic=name), index_col=0)
           for j, name in OMIC_NAMES.items()}
    add_top_features_per_omic(df, genes, pip)

    # ------------------------------------------------------------------------
    # 6. Reorder + write
    # ------------------------------------------------------------------------
    df = df[OUTPUT_COLUMN_ORDER]
    df.to_csv(OUTPUT_PATH, index=True)
    print(f"  wrote {OUTPUT_PATH}: {df.shape}")

    # ------------------------------------------------------------------------
    # 7. Optional clinical-feature counts (sanity-check printout from the
    #    original; useful for the manuscript Methods cohort summary).
    # ------------------------------------------------------------------------
    if os.path.exists(CLINICAL_CSV) and os.path.exists(SAMPLES_FINAL_CSV):
        samples = pd.read_csv(SAMPLES_FINAL_CSV, index_col=0)
        clin = pd.read_csv(CLINICAL_CSV, index_col=0)
        samples.columns = samples.columns.astype(str)
        clin.index = clin.index.astype(str)
        metadata = clin.loc[samples.columns.values]
        for feature in ("cogdx", "msex", "ceradsc"):
            print(f"\n{feature} counts:")
            print(metadata[feature].value_counts())


if __name__ == "__main__":
    main()
