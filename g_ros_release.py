"""
g_ros_release.py
================

Shared helpers for the figure-reproduction notebooks accompanying

    Whalley J.P., Hunsberger H.C., Bennett D.A., Lamar M.,
    Peterson D.A., Stutzmann G.E.
    *Multi-omic Analysis of Sex and Alzheimer's Disease in the Aged Human
    Cortex.* Under review at npj Aging; full citation will be
    filled in on publication.

The notebooks in this repository each reproduce one or more manuscript figures
or tables. They consume the multi-omic Sparse Decomposition of Arrays (SDA)
component-loading matrices deposited on Synapse (AD Knowledge Portal dataset
syn75275226, version 3, under the ROSMAP project syn3219045) and a small set
of public reference annotations the user downloads into `references/` (see
references/README.md for sources).

This module provides:
- `load_deposit()` : read the Synapse deposit files into a single namespace.
- `plot_sample_scatter()` : the per-component sample-loading scatter with
  broken axis for outliers (ros3b style).
- `plot_celltype_bar()` : the consistent cell-type bar plot (Ast/Exc/Inh/
  Mic/OPC/Oli) used in every figure.
- `plot_loadings_by_chromosome()` : the per-omic chromosome-scatter "Manhattan-
  style" loadings plot (ros3b style).
- `kegg_enrichment()` : KEGG pathway enrichment wrapper (replaces ros7ai/bi/ci
  boilerplate).
- Module-level constants: cell-type color palette, headline component map,
  expected file names, default plot styling.

Scope note: panels that need controlled-access data or external tools
(GWAS/eQTL, SEA-AD cross-cohort validation, the round-2 analyses) are produced
by the code in scripts/ (see scripts/README.md), not by these notebooks. The
notebooks reproduce the panels that follow directly from the Synapse deposit
files plus public reference annotations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Cell-type colour palette used across every figure in the manuscript.
# (Source: ros3b_plot_outliers.py.)
CELL_TYPE_COLORS: dict[str, str] = {
    "Ast": "#D85396",  # pink     — astrocytes
    "Oli": "#8BC34A",  # green    — oligodendrocytes
    "Inh": "#303F9F",  # dark blue — inhibitory neurons
    "OPC": "#E5A43A",  # orange   — oligodendrocyte precursors
    "Exc": "#D32F2F",  # red      — excitatory neurons
    "Mic": "#4DB6E7",  # sky blue — microglia
}

# Cell-type plotting order used in manuscript figures (NOT the column order in
# `cell_type.csv`, which is Ast/Oli/Inh/OPC/Exc/Mic).
CELL_TYPE_PLOT_ORDER: list[str] = ["Ast", "Exc", "Inh", "Mic", "OPC", "Oli"]

# Per-omic colour palettes for the chromosome-scatter plots (4-colour cycle
# applied across chromosomes 1–22 + X + Y, repeating every 4).
OMIC_PALETTES: dict[str, list[str]] = {
    "RNA":      ["#6495ED", "#778899", "#008080", "#20B2AA"],
    "meth":     ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"],
    "chip":     ["#f1eef6", "#d7b5d8", "#df65b0", "#ce1256"],
    "proteins": ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"],
}

# Sex colour mapping for the sample scatters (used when clinical metadata are given).
SEX_COLORS: dict[str, str] = {
    "Male":   "#4878D0",  # seaborn muted[0]
    "Female": "#D65F5F",  # seaborn muted[2]
}

# Maximum chromosome positions (GRCh38, used for x-axis layout in chromosome
# scatter plots). From ros3b.
CHR_MAX_POS: dict[str, float] = {
    "1": 249230899.0, "2": 243064438.0, "3": 197955065.0, "4": 191012160.0,
    "5": 180794288.0, "6": 170886450.0, "7": 158820866.0, "8": 146277826.0,
    "9": 141069493.0, "10": 135497182.0, "11": 134248398.0, "12": 133758108.0,
    "13": 115079831.0, "14": 107282052.0, "15": 102501016.0, "16": 90244125.0,
    "17": 81037567.0, "18": 77915115.0, "19": 59086812.0, "20": 62958768.0,
    "21": 48110787.0, "22": 51205920.0, "X": 155249967.0, "Y": 59330252.0,
}

# Headline components → manuscript figure (used for cross-referencing across
# notebooks and for the Data S1 partial-reproduction notebook).
HEADLINE_COMPONENTS: dict[int, str] = {
    74:  "Figure 2: sex-associated regulatory variation",
    278: "Figure 3: sex-biased component shared by OPCs and mature oligodendrocytes",
    172: "Figure 4: neuron-dominated component associated with neuritic plaque burden",
    277: "Figure 5: neuron-dominated component associated with clinical Alzheimer's dementia",
    21:  "Supplementary Figure S2: ectopic olfactory receptor expression and its cis-genetic regulation",
    351: "Supplementary Figure S8: a single-individual multi-omic profile",
    2:   "Supplementary Figure S9: a single-individual multi-omic profile",
    87:  "Supplementary Figure S10: a single-individual multi-omic profile",
    177: "Supplementary Figure S11: a component dominated by one donor",
}

# Default IQR multiplier for outlier detection in sample scatter (broken-axis).
# From ros3b.
OUTLIER_IQR_MULTIPLIER: float = 30.0

# Expected deposit file names. Users get these from the Synapse sub-project.
DEPOSIT_FILES: dict[str, str] = {
    "samples":   "samples_final.csv",
    "cell_type": "cell_type.csv",
    "RNA":       "RNA.txt",
    "meth":      "meth.txt",
    "chip":      "chip.txt",
    "proteins":  "proteins.txt",
}

# Reference annotation files (downloaded by the user into `references/`, NOT
# part of the Synapse deposit). See references/README.md for download sources.
REFERENCE_FILES: dict[str, str] = {
    "gene_annot": "Homo_sapiens.GRCh38.100.txt",
    "meth_annot": "ROSMAP_arrayMethylation_metaData.tsv",
    "chip_annot": "H3K9acDomains.csv",
}


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------

def apply_plot_style() -> None:
    """Apply the manuscript's matplotlib + seaborn style. Call once per notebook."""
    mpl.rcParams["pdf.fonttype"] = 42  # editable text in vector PDFs
    sns.set(style="white", context="talk")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

@dataclass
class Deposit:
    """Container for the Synapse deposit files plus optional reference annotations.

    Attributes
    ----------
    samples : pd.DataFrame
        Component × sample loadings. Index is component ID (1–519 and
        521–548; ID 520 is retired, so ID is not the row number). Columns are
        ROSMAP sample IDs (string).
    cell_type : pd.DataFrame
        Component × cell-type loadings. Index is component ID. Columns are the
        six cell types (Ast, Oli, Inh, OPC, Exc, Mic — note this is the
        file's column order, not the figure plotting order).
    genes : dict[str, pd.DataFrame]
        Per-omic feature × component matrices (PIP > 0.5 masked). Keys are
        "RNA", "meth", "chip", and "proteins" only if the deposit has
        proteins.txt (version 1). Each DataFrame has features as rows
        (gene symbol / CpG ID / peak ID / protein|UniProt) and component IDs
        as columns. Entries below the PIP threshold are NaN.
    deposit_dir : Path
        Directory holding the deposit files.
    references : dict[str, pd.DataFrame]
        Optional reference annotation tables (gene/meth/chip metadata), keyed
        by short name. Empty unless `load_references=True` was passed to
        `load_deposit`.
    """
    samples: pd.DataFrame
    cell_type: pd.DataFrame
    genes: dict[str, pd.DataFrame]
    deposit_dir: Path
    references: dict[str, pd.DataFrame] = field(default_factory=dict)

    def has_component(self, component_id: int) -> bool:
        return component_id in self.samples.index

    def pip_features(self, omic: str, component_id: int) -> pd.Series:
        """Return the non-NaN (PIP > 0.5) feature loadings for one component
        in one omic. Index is feature ID, values are loadings."""
        col = self.genes[omic][component_id]
        return col.dropna()


def load_deposit(
    deposit_dir: str | os.PathLike,
    references_dir: Optional[str | os.PathLike] = None,
    load_references: bool = False,
) -> Deposit:
    """Read the Synapse deposit into a `Deposit` object.

    Parameters
    ----------
    deposit_dir
        Directory containing the deposit files (samples_final.csv,
        cell_type.csv, RNA.txt, meth.txt, chip.txt; proteins.txt is read
        only if present, as in version 1).
    references_dir
        Optional. If `load_references=True`, this directory must contain the
        reference annotation files (Homo_sapiens.GRCh38.100.txt,
        ROSMAP_arrayMethylation_metaData.tsv, H3K9acDomains.csv).
    load_references
        Whether to read the reference annotation files. All four release
        notebooks pass `load_references=True`; notebooks 01 and 02 use the
        tables.

    Returns
    -------
    Deposit
        See `Deposit` docstring.
    """
    deposit_dir = Path(deposit_dir).expanduser().resolve()

    # Component × sample
    samples = pd.read_csv(deposit_dir / DEPOSIT_FILES["samples"], index_col=0)
    # Component × cell-type
    cell_type = pd.read_csv(deposit_dir / DEPOSIT_FILES["cell_type"], index_col=0)

    # Per-omic feature × component (PIP > 0.5 masked, tab-separated)
    genes: dict[str, pd.DataFrame] = {}
    for omic in ("RNA", "meth", "chip", "proteins"):
        if omic == "proteins" and not (deposit_dir / DEPOSIT_FILES[omic]).exists():
            continue  # proteins.txt is optional (version 1 has it); the notebooks report no protein-derived results
        df = pd.read_csv(
            deposit_dir / DEPOSIT_FILES[omic],
            sep="\t",
            index_col=0,
        )
        # The deposit files have integer component IDs as column headers; force
        # int dtype on columns for consistent indexing later.
        df.columns = df.columns.astype(int)
        genes[omic] = df

    references: dict[str, pd.DataFrame] = {}
    if load_references:
        if references_dir is None:
            raise ValueError(
                "load_references=True but references_dir is None. "
                "Pass the path to the references/ directory (see references/README.md "
                "for which files are expected)."
            )
        ref_dir = Path(references_dir).expanduser().resolve()
        # Gene annotation
        references["gene_annot"] = pd.read_csv(
            ref_dir / REFERENCE_FILES["gene_annot"],
            index_col=0,
            sep=",",
        )
        # Methylation manifest
        references["meth_annot"] = pd.read_csv(
            ref_dir / REFERENCE_FILES["meth_annot"],
            index_col=0,
            sep="\t",
        )
        # ChIP peaks
        references["chip_annot"] = pd.read_csv(
            ref_dir / REFERENCE_FILES["chip_annot"],
            index_col=0,
            sep="\t",
        )

    return Deposit(
        samples=samples,
        cell_type=cell_type,
        genes=genes,
        deposit_dir=deposit_dir,
        references=references,
    )


# -----------------------------------------------------------------------------
# Plotting helpers (extracted from ros3b_plot_outliers.py)
# -----------------------------------------------------------------------------

def plot_celltype_bar(
    deposit: Deposit,
    component_id: int,
    ax: Optional[plt.Axes] = None,
    order: list[str] = CELL_TYPE_PLOT_ORDER,
) -> plt.Axes:
    """Plot the cell-type loading bar for one component.

    Parameters
    ----------
    deposit
        Deposit container.
    component_id
        Component ID to plot (1–519 or 521–548).
    ax
        Axes to draw on. If None, create a new figure.
    order
        Cell-type order along the x-axis (defaults to manuscript figure order).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    values = deposit.cell_type.loc[component_id, order]
    colors = [CELL_TYPE_COLORS[ct] for ct in order]
    y_pos = np.arange(len(order))
    ax.bar(y_pos, values.values, align="center", color=colors)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(order)
    ax.set_ylabel("loading / score")
    ax.set_title(f"Component {component_id} — cell-type loadings")
    return ax


def plot_sample_scatter(
    deposit: Deposit,
    component_id: int,
    sample_metadata: Optional[pd.DataFrame] = None,
    ax: Optional[plt.Axes] = None,
    iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER,
) -> plt.Axes:
    """Plot the per-sample scatter for one component.

    Samples are sorted by `age_death` (when `sample_metadata` is supplied) and
    coloured by sex. When outliers exist (defined by ±`iqr_multiplier` × IQR
    around the median), the y-axis is broken to keep both ranges legible.

    Parameters
    ----------
    deposit
        Deposit container.
    component_id
        Component to plot.
    sample_metadata
        Optional DataFrame indexed by sample ID with at least 'msex' and
        'age_death' columns (from `ROSMAP_clinical.csv`). If None, samples are
        plotted in their default order without sex colouring.
    ax
        Axes to draw on. If None, create a new figure.
    iqr_multiplier
        Outlier detection threshold. Defaults to 30 (the manuscript's setting).
    """
    # Component loadings per sample as a series indexed by sample ID
    series = deposit.samples.loc[component_id].copy()
    series.index = series.index.astype(str)

    df = pd.DataFrame({"loading": series})
    if sample_metadata is not None:
        meta = sample_metadata.copy()
        meta.index = meta.index.astype(str)
        df = df.join(meta[["msex", "age_death"]], how="left")
        df["Sex"] = df["msex"].map({1.0: "Male", 0.0: "Female"})
        # '90+' is treated as 90 for sorting
        df["age_num"] = pd.to_numeric(df["age_death"], errors="coerce").fillna(90)
        df = df.sort_values("age_num")
    df["x_pos"] = np.arange(len(df))

    # Outlier bounds (IQR × multiplier)
    q1, q3 = df["loading"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    has_outliers = ((df["loading"] < lower) | (df["loading"] > upper)).any()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Capture the figure now — if we go down the broken-axis path below we
    # call ax.remove(), which detaches ax from its figure and makes
    # `ax.figure` None. We need a stable handle for the legend call later.
    fig = ax.figure

    if has_outliers:
        # Lazy import — brokenaxes is an optional dependency.
        from brokenaxes import brokenaxes

        normal = df[(df["loading"] >= lower) & (df["loading"] <= upper)]
        outliers = df[(df["loading"] < lower) | (df["loading"] > upper)]
        n_min, n_max = normal["loading"].min(), normal["loading"].max()
        o_min, o_max = outliers["loading"].min(), outliers["loading"].max()
        n_buf = max((n_max - n_min) * 0.05, 0.02)
        o_buf = max(abs(o_min) * 0.05, 0.01) if o_min == o_max else max((o_max - o_min) * 0.05, 0.01)
        ylims = sorted([(n_min - n_buf, n_max + n_buf), (o_min - o_buf, o_max + o_buf)],
                       key=lambda t: t[0])
        # brokenaxes returns its own axes; replace the supplied ax.
        ax.remove()
        bax = brokenaxes(ylims=ylims, hspace=0.05, fig=fig)
        target = bax
    else:
        target = ax

    if "Sex" in df.columns and df["Sex"].notna().any():
        target.scatter(df["x_pos"], df["loading"],
                       c=df["Sex"].map(SEX_COLORS).fillna("#888888"),
                       edgecolors="k", linewidth=0.0, alpha=0.8)
        legend_handles = [
            mlines.Line2D([], [], color=SEX_COLORS["Male"],   marker="o", linestyle="None", markersize=8, label="Male"),
            mlines.Line2D([], [], color=SEX_COLORS["Female"], marker="o", linestyle="None", markersize=8, label="Female"),
        ]
        fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.01, 1))
    else:
        target.scatter(df["x_pos"], df["loading"],
                       c="#444444", edgecolors="k", linewidth=0.0, alpha=0.8)

    # without metadata the samples stay in deposit column order
    xlabel = "Samples (sorted by age at death)" if sample_metadata is not None else "Samples"
    if has_outliers:
        for sub_ax in target.axs:
            sub_ax.set_xticks([])
        target.set_xlabel(xlabel)
    else:
        target.set_xticks([])
        target.set_xlabel(xlabel)
        target.set_ylabel("loading / score")
    return target


def plot_loadings_by_chromosome(
    deposit: Deposit,
    component_id: int,
    omic: str,
    feature_annot: pd.DataFrame,
    chrom_col: str,
    pos_col: str,
    label_col: str,
    n_extreme_to_label: int = 8,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot per-feature loadings laid out by chromosome (Manhattan-style).

    Used for the per-omic loading panels of Notebook 01.

    Parameters
    ----------
    deposit
        Deposit container.
    component_id
        Component to plot.
    omic
        One of {"RNA", "meth", "chip", "proteins"}.
    feature_annot
        DataFrame indexed by feature ID (matching the row index of
        `deposit.genes[omic]`) with chromosome / position / display-label
        columns. Caller is responsible for joining the relevant reference
        annotation onto the deposit features before calling.
    chrom_col, pos_col, label_col
        Column names in `feature_annot` for chromosome, start position, and
        the human-readable label (gene symbol, CpG ID, etc.).
    n_extreme_to_label
        Top + bottom N loadings (by absolute value) to label inline.
    ax
        Axes to draw on. If None, create a new figure.
    """
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    # Get the masked feature loadings for this component
    feature_loadings = deposit.pip_features(omic, component_id)
    if feature_loadings.empty:
        ax.text(0.5, 0.5, f"No features above PIP threshold for component {component_id}",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    # Restrict annotation to the features we actually have
    annot = feature_annot.loc[feature_annot.index.intersection(feature_loadings.index)]
    if annot.empty:
        ax.text(0.5, 0.5, "Feature annotation does not cover the loaded features",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    palette = OMIC_PALETTES[omic]
    chr_list = [str(i) for i in range(1, 23)] + ["X", "Y"]
    shift = np.array([0.0])

    # Choose features to label inline (top + bottom by absolute loading)
    extreme = feature_loadings.abs().nlargest(2 * n_extreme_to_label)
    label_set = set(extreme.index)

    texts: list = []
    for ii, chrom in enumerate(chr_list):
        in_chrom = annot[annot[chrom_col].astype(str) == chrom]
        if in_chrom.empty:
            shift = np.append(shift, shift[-1] + CHR_MAX_POS.get(chrom, 0))
            continue
        x = shift[-1] + in_chrom[pos_col].values
        y = feature_loadings.loc[in_chrom.index].values
        ax.plot(x, y, ".", color=palette[ii % len(palette)])
        shift = np.append(shift, x.max() if len(x) else shift[-1])
        # Inline labels for extreme loadings
        for xi, yi, fid in zip(x, y, in_chrom.index):
            if fid in label_set:
                texts.append(ax.text(xi, yi, in_chrom.loc[fid, label_col],
                                     color="black" if yi > 0 else "gray", fontsize=8))
    if adjust_text is not None and texts:
        # a fixed iteration budget instead of adjustText's default 1-second time limit, so that the label
        # placement does not depend on the speed of the machine (2,000 is about what 1 s allowed when the
        # notebook outputs were regenerated)
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5), iter_lim=2000)

    midpoints = (shift[1:] + shift[:-1]) / 2.0
    ax.set_xticks(midpoints)
    ax.set_xticklabels(chr_list, rotation=90)
    ax.set_xlabel(f"{omic} features by chromosome")
    ax.set_ylabel("loading / score")
    return ax


# -----------------------------------------------------------------------------
# Pathway enrichment (extracted from ros7ai/bi/ci)
# -----------------------------------------------------------------------------

def features_to_genes(
    deposit: Deposit,
    component_id: int,
    omic: str,
) -> list[str]:
    """Extract a unique gene-symbol list from the PIP-significant features of
    one component in one omic.

    - RNA: feature index already gene symbols → returned as-is.
    - proteins: feature index is "SYMBOL|UniProt" → strip to SYMBOL; when
      `deposit.references["gene_annot"]` is loaded, keep only symbols that are
      an Ensembl 100 `gene_name` (as for Fig 2d of version 1; this version
      reports no protein-derived results).
    - meth: requires `deposit.references["meth_annot"]`; uses every entry
      of the manifest's `RefGene` column per CpG (as for the Component 74 CpG
      enrichment).
    - chip: not supported; raises `NotImplementedError`. The peak table
      (`chip_annot`) has no gene column, so peaks would need an external
      nearest-gene step (e.g. bedtools closest or HOMER annotatePeaks.pl).
    """
    feats = deposit.pip_features(omic, component_id)
    if feats.empty:
        return []
    if omic == "RNA":
        return list(dict.fromkeys(feats.index))
    if omic == "proteins":
        symbols = [str(f).split("|")[0] for f in feats.index]
        if "gene_annot" in deposit.references:   # as in version 1 (Fig 2d): only symbols that are an Ensembl 100 gene_name
            known = set(deposit.references["gene_annot"]["gene_name"].dropna())
            symbols = [s for s in symbols if s in known]
        return list(dict.fromkeys(symbols))
    if omic == "meth":
        if "meth_annot" not in deposit.references:
            raise ValueError("meth → gene mapping needs deposit.references['meth_annot']; "
                             "call load_deposit(..., load_references=True).")
        annot = deposit.references["meth_annot"]
        genes: list[str] = []
        for cpg in feats.index:
            if cpg in annot.index:
                refgene = annot.loc[cpg, "RefGene"]
                if pd.notna(refgene):
                    genes.extend(str(refgene).split(";"))   # every listed gene, as for the Component 74 CpG enrichment
        return list(dict.fromkeys(genes))
    if omic == "chip":
        # H3K9ac peaks (`H3K9acDomains.csv`) carry only Chr / Start / End /
        # Classification (Promoter / Enhancer / Other), with no gene symbol or
        # nearest-gene column. Mapping ChIP peaks to genes requires an
        # external nearest-gene step (e.g., bedtools closest against an
        # Ensembl GTF, or HOMER's annotatePeaks.pl), which is out of scope
        # for these notebooks. The manuscript reports no pathway enrichment
        # for H3K9ac peaks.
        raise NotImplementedError(
            "ChIP peak → gene mapping is not provided by the deposit. "
            "The H3K9acDomains.csv manifest has no gene-symbol column. "
            "If you want a peak-level gene assignment, run nearest-gene "
            "annotation externally (bedtools / HOMER annotatePeaks.pl) "
            "against the chromosome + start coordinates in `chip_annot`."
        )
    raise ValueError(f"Unknown omic: {omic!r}")


def kegg_enrichment(
    deposit: Deposit,
    component_id: int,
    omic: str,
    gene_list: Optional[list[str]] = None,
    organism: str = "human",
    cutoff: float = 0.25,
    background: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run KEGG pathway enrichment for one component in one omic, using
    `gseapy.enrichr`. If Enrichr answers HTTP 429 (rate limit), the call is
    retried after 10, 20, 40, 80 and 160 s.

    Parameters
    ----------
    deposit
        Deposit container.
    component_id
        Component to enrich.
    omic
        One of {"RNA", "meth", "chip", "proteins"}.
    gene_list
        Optional explicit gene-symbol list. If None, derived from
        `features_to_genes(deposit, component_id, omic)`.
    organism
        gseapy organism string (default "human").
    cutoff
        Adjusted-p cutoff for reporting hits.
    background
        Optional gene-symbol list to use as the enrichment background. For RNA
        the natural background is all genes in the deposit's RNA matrix.

    Returns
    -------
    DataFrame
        gseapy's enrichment result table, sorted by adjusted p-value. Empty
        DataFrame if no features cross the PIP threshold or if no terms pass
        `cutoff`.
    """
    import gseapy as gp  # heavy import — keep lazy

    if gene_list is None:
        gene_list = features_to_genes(deposit, component_id, omic)
    if not gene_list:
        return pd.DataFrame()

    if background is None and omic == "RNA":
        background = list(deposit.genes["RNA"].index)

    import time
    for attempt in range(6):
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets="KEGG_2021_Human",
                organism=organism,
                background=background,
                outdir=None,
                cutoff=cutoff,
                no_plot=True,
            )
            break
        except Exception as e:  # Enrichr rate limit (HTTP 429): wait and retry
            if "429" not in str(e) or attempt == 5:
                raise
            time.sleep(10 * 2 ** attempt)
    if enr.results is None or enr.results.empty:
        return pd.DataFrame()
    return enr.results.sort_values("Adjusted P-value")


# -----------------------------------------------------------------------------
# Clinical associations (extracted from ros6a_cross_clin_discrete.py)
# -----------------------------------------------------------------------------

CERADSC_BINARY_MAP: dict[int, str] = {
    1: "AD",
    2: "AD",
    3: "No",
    4: "No",
}

COGDX_TABLE_S1_GROUPS: tuple[int, int, int] = (1, 2, 4)


def _bh_adjust(p_values: list[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(p)
    if not valid_mask.any():
        return adjusted

    valid_p = p[valid_mask]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    n = len(ranked)
    ranked_adjusted = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        running_min = min(running_min, ranked[i] * n / (i + 1))
        ranked_adjusted[i] = min(running_min, 1.0)

    valid_adjusted = np.empty(n, dtype=float)
    valid_adjusted[order] = ranked_adjusted
    adjusted[valid_mask] = valid_adjusted
    return adjusted


def clinical_association(
    deposit: Deposit,
    clinical_df: pd.DataFrame,
    component_id: int,
    test: str,
) -> dict:
    """Test the sample loadings of one component against a clinical variable.

    Parameters
    ----------
    deposit
        Deposit container.
    clinical_df
        DataFrame indexed by sample ID with the clinical columns we test
        against. Expected columns:
        - `msex` (1 = Male, 0 = Female) for `test='sex'`
        - `ceradsc` (CERAD neuropathology score) for `test='ceradsc'`
        - `cogdx` (cognitive diagnosis category) for `test='cogdx'`
        Sourced from `ROSMAP_clinical.csv` (controlled access).
    component_id
        Component to test.
    test
        One of {'sex', 'ceradsc', 'cogdx'}. 'sex' uses Mann–Whitney U
        (two-sided); 'ceradsc' maps scores 1/2 to AD and 3/4 to No, then uses
        Mann-Whitney U (two-sided); 'cogdx' uses Kruskal-Wallis across the
        manuscript groups 1, 2, and 4.

    Returns
    -------
    dict
        Keys: `test`, `n` (sample count after dropna), `statistic`, `p_value`,
        plus per-test extras (`n_male`/`n_female` for sex, `n_ad`/`n_no` for
        ceradsc, `n_groups` for cogdx).
    """
    from scipy import stats as ss

    series = deposit.samples.loc[component_id].copy()
    series.index = series.index.astype(str)
    df = pd.DataFrame({"loading": series})
    clin = clinical_df.copy()
    clin.index = clin.index.astype(str)
    df = df.join(clin, how="inner").dropna(subset=["loading"])

    if test == "sex":
        if "msex" not in df.columns:
            raise KeyError("clinical_df must have an 'msex' column for test='sex'.")
        df = df.dropna(subset=["msex"])
        male = df[df["msex"] == 1]["loading"].values
        female = df[df["msex"] == 0]["loading"].values
        if len(male) == 0 or len(female) == 0:
            return {"test": "sex", "n": len(df), "statistic": float("nan"),
                    "p_value": float("nan"), "n_male": int(len(male)),
                    "n_female": int(len(female))}
        u, p = ss.mannwhitneyu(male, female, alternative="two-sided")
        return {"test": "sex", "n": int(len(df)), "statistic": float(u),
                "p_value": float(p), "n_male": int(len(male)),
                "n_female": int(len(female))}

    if test == "ceradsc":
        if "ceradsc" not in df.columns:
            raise KeyError("clinical_df must have a 'ceradsc' column for test='ceradsc'.")
        df = df.dropna(subset=["ceradsc"]).copy()
        df["ceradsc_binary"] = df["ceradsc"].map(CERADSC_BINARY_MAP)
        df = df.dropna(subset=["ceradsc_binary"])
        ad = df[df["ceradsc_binary"] == "AD"]["loading"].values
        no = df[df["ceradsc_binary"] == "No"]["loading"].values
        if len(ad) == 0 or len(no) == 0:
            return {"test": "ceradsc", "n": int(len(df)), "statistic": float("nan"),
                    "p_value": float("nan"), "n_ad": int(len(ad)),
                    "n_no": int(len(no))}
        u, p = ss.mannwhitneyu(ad, no, alternative="two-sided")
        return {"test": "ceradsc", "n": int(len(df)), "statistic": float(u),
                "p_value": float(p), "n_ad": int(len(ad)), "n_no": int(len(no))}

    if test == "cogdx":
        if "cogdx" not in df.columns:
            raise KeyError("clinical_df must have a 'cogdx' column for test='cogdx'.")
        df = df.dropna(subset=["cogdx"])
        df = df[df["cogdx"].isin(COGDX_TABLE_S1_GROUPS)]
        groups = [df[df["cogdx"] == group]["loading"].values
                  for group in COGDX_TABLE_S1_GROUPS]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return {"test": test, "n": int(len(df)), "statistic": float("nan"),
                    "p_value": float("nan"), "n_groups": int(len(groups))}
        h, p = ss.kruskal(*groups)
        return {"test": test, "n": int(len(df)), "statistic": float(h),
                "p_value": float(p), "n_groups": int(len(groups))}

    raise ValueError(f"Unknown test: {test!r}. Expected 'sex', 'ceradsc', or 'cogdx'.")


def clinical_association_table(
    deposit: Deposit,
    clinical_df: pd.DataFrame,
    tests: tuple[str, ...] | list[str] = ("sex", "ceradsc", "cogdx"),
    components: Optional[list[int]] = None,
    correction: str | None = "fdr_bh",
) -> pd.DataFrame:
    """Run clinical association tests and add manuscript-style adjusted p-values.

    Data S1 reports Benjamini-Hochberg adjusted p-values calculated separately
    for each clinical variable across all 547 components. To reproduce those
    values exactly, leave `components=None` and filter the returned table for
    the components of interest after adjustment.

    Parameters
    ----------
    deposit
        Deposit container.
    clinical_df
        DataFrame indexed by sample ID with `msex`, `ceradsc`, and/or `cogdx`.
    tests
        Clinical tests to run.
    components
        Components to include in the multiple-testing family. Defaults to all
        components in `deposit.samples`.
    correction
        `fdr_bh` for Benjamini-Hochberg adjustment, or None/`none` for raw
        p-values only.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per component/test and columns including
        `p_value` and `p_adjusted`.
    """
    if components is None:
        components = [int(c) for c in deposit.samples.index]

    rows = []
    for component_id in components:
        if not deposit.has_component(component_id):
            raise KeyError(f"Component {component_id} not present in deposit.samples.")
        for test in tests:
            rows.append({
                "component_id": int(component_id),
                **clinical_association(deposit, clinical_df, component_id, test),
            })

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results["p_adjusted"] = np.nan
    if correction is None or correction == "none":
        return results
    if correction != "fdr_bh":
        raise ValueError("correction must be 'fdr_bh', 'none', or None.")

    for test in results["test"].dropna().unique():
        mask = results["test"] == test
        results.loc[mask, "p_adjusted"] = _bh_adjust(results.loc[mask, "p_value"])

    return results


# -----------------------------------------------------------------------------
# Convenience: per-component summary printout
# -----------------------------------------------------------------------------

def top_cell_types(loadings: pd.Series) -> tuple[Optional[str], Optional[str]]:
    """(top positive, top negative) cell type, as in the Data S1 columns: the
    cell type with the largest loading if any loading is > 0, and the one
    with the smallest loading if any loading is < 0; None otherwise. (Data S1
    additionally leaves a label blank where the individual SDA runs do not
    support its direction; that check needs the run-level outputs.)"""
    pos = loadings.idxmax() if loadings.max() > 0 else None
    neg = loadings.idxmin() if loadings.min() < 0 else None
    return pos, neg


def component_summary(deposit: Deposit, component_id: int) -> dict:
    """Compact summary of one component for notebook narrative cells."""
    top_pos, top_neg = top_cell_types(deposit.cell_type.loc[component_id])
    return {
        "component_id": component_id,
        "manuscript_role": HEADLINE_COMPONENTS.get(component_id, "(non-headline component)"),
        # n_features_RNA, n_features_meth, n_features_chip (and n_features_proteins for a version-1 deposit)
        **{f"n_features_{omic}": int(df[component_id].notna().sum()) for omic, df in deposit.genes.items()},
        "top_celltype_pos":    top_pos,
        "top_celltype_neg":    top_neg,
    }
