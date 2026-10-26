"""
11_eqtls.py
===========

Cell-type-specific eQTL analysis: for each (component, significant SNP,
high-PIP feature) combination, fit a linear model of feature value against
genotype within each cell type and plot a boxplot per tissue.

Manuscript: Whalley J.P. et al., *Multi-omic Analysis of Sex and Alzheimer's
Disease in the Aged Human Cortex* (under review at npj Aging).
Methods §"Genetic association and eQTL analysis". Underlies Supplementary
Figure S2e (Component 21 olfactory-receptor eQTL signal).

Known label error: `classify_qtl` compares the SNP position from the genotype
`.bim` (NCBI36/hg18) with GRCh38 gene coordinates, so the `cis|trans` part of
the RNA output file names can be wrong. For Component 21 the file is named
`..._trans_eQTL_tissues.pdf`, but rs6664332 lies 81 kb from the OR2L3 gene
start on GRCh38 (inside OR2AK2): a cis eQTL, as the manuscript states. The
label enters only the file name; the tests and plots are unaffected.

Original filename in the working tree: `ros8_eQTLs.py`.

External dependencies:
- `pysnptools.snpreader.Bed` to read PLINK bedfiles.
- `statsmodels` for the OLS / ANOVA test per (genotype, expression) cell.
- The output of `08_gwas_preprocessing.py` (PLINK `.bed/.bim/.fam`) plus
  the per-component `.abbr.txt` GWAS summaries from `09_manhattan_plots.py`.

Pipeline:
1. Load deposit + reference annotations (mirrors the prep block used in the
   notebooks and in `10_chip_motifs.py`; left standalone per the moderate-
   cleanup convention to keep each downstream script self-contained).
2. Load per-cell-type pseudobulks, the harmonised meth/ChIP/proteomics
   matrices, and the SNP genotype data (PLINK `.bed`).
3. For each omic and each component:
   - PIP-filter the features for that component.
   - Read the per-component GWAS abbreviated table; keep SNPs above
     `LOG10_P_THRESHOLD`.
   - For RNA: for the top-loading features, plot expression vs. genotype
     in each of the six cell types. Save 2 × 3 panels per (component,
     feature, SNP) combination if any tissue's ANOVA p < 0.05.
   - For meth/chip/protein: a single boxplot per (component, feature, SNP).

Outputs (one file per panel set):
- `figs/eQTL/<comp>_<feature>_<snp>_<cis|trans>_eQTL_tissues.pdf`  (RNA)
- `figs/methQTL/<comp>_<feature>_<snp>.pdf`
- `figs/chipQTL/<comp>_<feature>_<snp>.pdf`
- `figs/pQTL/<comp>_<feature>_<snp>.pdf`
"""
from __future__ import annotations
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pysnptools.snpreader import Bed
from statsmodels.formula.api import ols

mpl.rcParams["pdf.fonttype"] = 42
sns.set(style="white", context="poster")

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. The genes/pip CSVs come from
# `07_cluster_components.py`; the per-omic prepared matrices come from
# scripts 01–04; the SNP and GWAS files come from `08_gwas_preprocessing`
# and `09_manhattan_plots` respectively.
# ----------------------------------------------------------------------------
SAMPLES_FINAL_CSV = "results/samples_final.csv"
TISSUES_FINAL_CSV = "results/treatment_final.csv"
GENES_FINAL_TEMPLATE = "results/genes_final_{omic}.csv"
PIP_FINAL_TEMPLATE = "results/pip_final_{omic}.csv"
CLINICAL_CSV = "results/ROSMAP_clinical.csv"
GENE_ANNOT_CSV = "results/Homo_sapiens.GRCh38.100.txt"
METH_ANNOT_TSV = "data/ROSMAP_arrayMethylation_metaData.tsv"
CHIP_ANNOT_TSV = "data/H3K9acDomains.csv"

PSEUDOBULK_TEMPLATE = "data/scrna/{cell_type}_rpm_normalized.csv"  # output of 01_prepare_snrna.py
CHIP_CROSSED = "data/chip/chip_crossed_with_sc.csv"   # output of 02_prepare_chip.py
METH_CROSSED = "data/meth/meth_crossed_with_sc.csv"   # output of 03_prepare_meth.py
PROT_CROSSED = "data/prot/prot_crossed_with_sc.csv"   # output of 04_prepare_proteomics.py

SNP_BED_PREFIX = "data/snp/ROSMAP_arrayGenotype_maf"  # output of 08_gwas_preprocessing.py
GWAS_PATH_TEMPLATE = os.path.join(os.environ.get("G_ROS_GWAS_DIR", "output/"), "results{comp}.abbr.txt")  # 09_manhattan_plots.py output

OMIC_NAMES = {1: "RNA", 2: "meth", 3: "chip", 4: "proteins"}
QTL_OUTPUT_DIRS = {
    "RNA": "figs/eQTL",
    "meth": "figs/methQTL",
    "chip": "figs/chipQTL",
    "proteins": "figs/pQTL",
}

# ----------------------------------------------------------------------------
# Per-test parameters — match the manuscript Methods.
# ----------------------------------------------------------------------------
LOG10_P_THRESHOLD = 15      # GWAS −log10 p threshold for considering a SNP
MAX_FEATURES_PER_COMP = 5   # cap features tested per (omic, component) — runtime guard
PIP_THRESHOLD = 0.5
MIN_GENOTYPE_GROUP = 5      # require ≥ this many samples per genotype group for ANOVA
CIS_WINDOW_BP = 1_000_000   # ± 1 Mb around the feature for cis vs trans classification
ANOVA_SIG_THRESHOLD = 0.05  # tissue-level significance flag


def load_genes_and_pip() -> tuple[dict, dict]:
    genes = {j: pd.read_csv(GENES_FINAL_TEMPLATE.format(omic=name), index_col=0)
             for j, name in OMIC_NAMES.items()}
    pip = {j: pd.read_csv(PIP_FINAL_TEMPLATE.format(omic=name), index_col=0)
           for j, name in OMIC_NAMES.items()}
    return genes, pip


def load_annotations() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load gene / methylation / ChIP annotations."""
    expr_info = pd.read_csv(GENE_ANNOT_CSV, index_col=0, sep=",")
    expr_info["new_index"] = expr_info["gene_name"]
    expr_info.set_index("new_index", inplace=True)

    meth_info = pd.read_csv(METH_ANNOT_TSV, index_col=0, sep="\t")
    meth_info["CHR"] = meth_info["CHR"].astype(str)

    chip_info = pd.read_csv(CHIP_ANNOT_TSV, index_col=0, sep="\t")
    chip_info["Chr"] = chip_info["Chr"].astype(str)

    return expr_info, meth_info, chip_info


def classify_qtl(feature_chr, feature_start, feature_end, snp_chr, snp_pos) -> str:
    """`cis` if the SNP is within `CIS_WINDOW_BP` of the feature, else `trans`."""
    if str(feature_chr) == str(snp_chr) and \
       (feature_start - CIS_WINDOW_BP) <= snp_pos <= (feature_end + CIS_WINDOW_BP):
        return "cis"
    return "trans"


def main() -> None:
    samples = pd.read_csv(SAMPLES_FINAL_CSV, index_col=0)
    tissues = pd.read_csv(TISSUES_FINAL_CSV, index_col=0)
    samples.columns = samples.columns.astype(str)

    genes, pip = load_genes_and_pip()
    expr_info, meth_info, chip_info = load_annotations()

    # Per-cell-type pseudobulks, log-transformed (only used for RNA eQTLs)
    cell_types = list(tissues.columns)
    gene_expression = {}
    for ct in cell_types:
        gene_expression[ct] = np.log2(
            pd.read_csv(PSEUDOBULK_TEMPLATE.format(cell_type=ct), index_col=0) + 1
        )

    # Per-omic harmonised feature matrices (transposed so rows = samples)
    chip_data = pd.read_csv(CHIP_CROSSED, index_col=0).T
    meth_data = pd.read_csv(METH_CROSSED, index_col=0).T
    prot_data = pd.read_csv(PROT_CROSSED, index_col=0).T

    # SNP genotype data
    snp_data = Bed(SNP_BED_PREFIX + ".bed")
    snp_bim = pd.read_csv(SNP_BED_PREFIX + ".bim", sep=r"\s+", header=None)
    snp_fam = pd.read_csv(SNP_BED_PREFIX + ".fam", sep=r"\s+", header=None)

    snp_samples = snp_fam[1].astype(str)
    snp_samples_clean = snp_samples.str.replace(r"^(ROS|MAP)", "", regex=True).astype(int)

    for d in QTL_OUTPUT_DIRS.values():
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------------
    # Per omic, per component
    # ------------------------------------------------------------------------
    for j, omic in OMIC_NAMES.items():
        print(f"\n=== {omic} ===")
        genes_data = genes[j]
        pip_data = pip[j]

        for k in genes_data.index.values:
            pip_features = genes_data[
                pip_data.T[pip_data.loc[k] > PIP_THRESHOLD].index.values
            ]
            if pip_features.empty:
                continue

            gwas_file = GWAS_PATH_TEMPLATE.format(comp=k)
            if not os.path.isfile(gwas_file):
                continue
            gwas_df = pd.read_csv(
                gwas_file, sep=" ", header=None,
                names=["chromosome", "position", "SNP", "pval"],
            )
            gwas_df["-log10(pval)"] = -np.log10(gwas_df["pval"])
            sig = gwas_df[gwas_df["-log10(pval)"] > LOG10_P_THRESHOLD]
            if sig.empty:
                continue

            print(f"  comp {k}: {len(sig)} SNPs above -log10 p > {LOG10_P_THRESHOLD}")

            if omic == "RNA":
                _eqtl_rna(
                    k, pip_features, sig, expr_info, snp_data, snp_bim,
                    snp_samples_clean, gene_expression, cell_types,
                )
            else:
                _eqtl_other(
                    k, omic, pip_features, sig, expr_info, meth_info, chip_info,
                    snp_data, snp_bim, snp_samples_clean,
                    {"meth": meth_data, "chip": chip_data, "proteins": prot_data}[omic],
                )


def _eqtl_rna(
    k, pip_features, sig_snps, expr_info, snp_data, snp_bim, snp_samples_clean,
    gene_expression, cell_types,
) -> None:
    """RNA eQTL path: 2x3 grid of cell-type panels per (component, feature, SNP)."""
    feature_loadings = pip_features.loc[k].abs().sort_values(ascending=False)
    tested = 0
    cell_types_sorted = sorted(cell_types)

    for feature in feature_loadings.index:
        if tested >= MAX_FEATURES_PER_COMP:
            break
        if feature not in expr_info.index:
            continue
        feature_chr = expr_info.loc[feature, "chrom"]
        feature_start = expr_info.loc[feature, "chromStart"]
        feature_end = expr_info.loc[feature, "chromEnd"]

        for _, snp in sig_snps.iterrows():
            snp_id = snp["SNP"]
            idx_arr = np.where(snp_bim[1].values == snp_id)[0]
            if len(idx_arr) == 0:
                continue
            snp_index = idx_arr[0]

            qtl_type = classify_qtl(
                feature_chr, feature_start, feature_end,
                snp["chromosome"], snp["position"],
            )

            fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
            axes = axes.flatten()
            significant_tissues = []

            a1 = snp_bim.iloc[snp_index, 5]
            a2 = snp_bim.iloc[snp_index, 4]
            genotype_order = [f"{a1}{a1}", f"{a1}{a2}", f"{a2}{a2}"]

            for i, tissue in enumerate(cell_types_sorted):
                shared = snp_samples_clean[snp_samples_clean.isin(gene_expression[tissue].index)]
                expr = gene_expression[tissue].loc[shared, feature]
                geno = snp_data[:, snp_index].read().val.flatten()
                geno = pd.Series(geno, index=snp_samples_clean).loc[shared]
                geno = geno.map({0: f"{a2}{a2}", 1: f"{a1}{a2}", 2: f"{a1}{a1}"})

                df = pd.DataFrame({"expression": expr, "genotype": geno}).dropna()
                if df.empty or df["genotype"].value_counts().min() < MIN_GENOTYPE_GROUP:
                    continue

                pval = sm.stats.anova_lm(
                    ols("expression ~ C(genotype)", data=df).fit(), typ=2
                )["PR(>F)"][0]

                ax = axes[i]
                sns.boxplot(x="genotype", y="expression", data=df, ax=ax,
                            showfliers=False, palette="Set2", order=genotype_order)
                sns.stripplot(x="genotype", y="expression", data=df, ax=ax,
                              color="gray", alpha=0.5, jitter=True, order=genotype_order)
                ax.set_title(f"{tissue} (p = {pval:.2e})")
                ax.set_xlabel("Genotype")
                ax.set_ylabel("Expression")

                if pval < ANOVA_SIG_THRESHOLD:
                    significant_tissues.append(tissue)

            if significant_tissues:
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fname = (f"{QTL_OUTPUT_DIRS['RNA']}/"
                         f"{k}_{feature}_{snp_id}_{qtl_type}_eQTL_tissues.pdf")
                plt.savefig(fname)
                plt.close(fig)
            else:
                plt.close(fig)

            tested += 1
            if tested >= MAX_FEATURES_PER_COMP:
                break


def _eqtl_other(
    k, omic, pip_features, sig_snps, expr_info, meth_info, chip_info,
    snp_data, snp_bim, snp_samples_clean, feature_matrix,
) -> None:
    """Non-RNA QTL path (methQTL / chipQTL / pQTL): one panel per
    (component, feature, SNP)."""
    feature_loadings = pip_features.loc[k].abs().sort_values(ascending=False)
    tested = 0

    for feature in feature_loadings.index:
        if tested >= MAX_FEATURES_PER_COMP:
            break

        try:
            if omic == "meth":
                feature_chr = meth_info.loc[feature, "CHR"]
                feature_start = meth_info.loc[feature, "MAPINFO"]
                feature_end = feature_start
            elif omic == "chip":
                feature_chr = chip_info.loc[feature, "Chr"]
                feature_start = chip_info.loc[feature, "Start"]
                feature_end = chip_info.loc[feature, "End"]
            elif omic == "proteins":
                # Strip the |UniProt suffix and look up via gene symbol
                gene_symbol = feature.split("|")[0]
                feature_chr = expr_info.loc[gene_symbol, "chrom"]
                feature_start = expr_info.loc[gene_symbol, "chromStart"]
                feature_end = expr_info.loc[gene_symbol, "chromEnd"]
            else:
                continue
        except KeyError:
            continue

        for _, snp in sig_snps.iterrows():
            qtl_type = classify_qtl(
                feature_chr, feature_start, feature_end,
                snp["chromosome"], snp["position"],
            )

            shared = snp_samples_clean[snp_samples_clean.isin(feature_matrix.index)]
            feat_vec = feature_matrix.loc[shared, feature]

            snp_index = np.where(snp_bim[1].values == snp["SNP"])[0]
            if len(snp_index) == 0:
                continue
            geno = pd.Series(
                snp_data[:, snp_index[0]].read().val.flatten(),
                index=snp_samples_clean,
            ).loc[shared]

            df = pd.DataFrame({"expression": feat_vec, "genotype": geno}).dropna()
            if df.empty or df["genotype"].value_counts().min() < MIN_GENOTYPE_GROUP:
                continue

            plt.figure(figsize=(10, 6))
            sns.boxplot(x="genotype", y="expression", data=df,
                        showfliers=False, palette="Set2")
            sns.stripplot(x="genotype", y="expression", data=df,
                          color="gray", alpha=0.5, jitter=True)
            plt.xlabel(f"{snp['SNP']} Genotype")
            plt.ylabel(f"{feature} Expression Level")
            plt.tight_layout()

            fname = f"{QTL_OUTPUT_DIRS[omic]}/{k}_{feature}_{snp['SNP']}.pdf"
            plt.savefig(fname)
            plt.close()

            tested += 1
            break  # one SNP per feature is enough


if __name__ == "__main__":
    main()
