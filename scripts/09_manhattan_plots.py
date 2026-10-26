"""
09_manhattan_plots.py
=====================

Generate Manhattan and Q-Q plots from the per-component GWAS association
results produced by GEMMA on the cluster.

Manuscript: Whalley J.P. et al., *Multi-omic Analysis of Sex and Alzheimer's
Disease in the Aged Human Cortex* (under review at npj Aging).
Methods §"Genetic association and eQTL analysis"; Component 21 in
Supplementary Figure S2.

Original filename in the working tree: `ros5_Manhattan_Plot_maf.py`.

External dependencies:
- `assocplots` (https://github.com/khramts/assocplots) for `manhattan` and
  `qqplot`.
- `adjustText` for inline label de-overlap.

Pipeline:
1. For each `<results_id>.assoc.txt` GEMMA output: write an abbreviated
   space-separated file (`chrom pos snp pval`) for downstream tooling.
2. For each abbreviated file: build a Manhattan plot annotated with the
   most-extreme SNP per chromosome (where −log10 p ≥ 8), and a Q-Q plot
   under the theoretical-error null.

Inputs:
- `output/*.assoc.txt` — raw GEMMA per-component association tables (the
  `output/` directory is the GEMMA `--out` location set up by `08`).
- `<GWAS_INPUT_DIR>/results<comp>.abbr.txt` — the abbreviated space-
  separated files this script writes in step 1.

Outputs:
- `figs/GWAS/<comp>_Manhattan_Component.png`
- `figs/GWAS/qq_<comp>_Component_theoretical_error.png`
"""
from __future__ import annotations
import os
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hex2color

from adjustText import adjust_text
from assocplots.manhattan import manhattan
from assocplots.qqplot import qqplot

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout. The GEMMA `output/` directory and the
# GWAS abbreviated-file directory below are kept separate because in the
# original analysis they lived on different machines (cluster vs. laptop).
# ----------------------------------------------------------------------------
GEMMA_OUTPUT_DIR = "output/"                                # where GEMMA wrote .assoc.txt
GWAS_INPUT_DIR = os.environ.get("G_ROS_GWAS_DIR", GEMMA_OUTPUT_DIR)  # where .abbr.txt files live (default: where step 1 writes them)
PLOT_OUTPUT_DIR = "figs/GWAS/"                              # destination for Manhattan / QQ PNGs

LOG10_P_LABEL_THRESHOLD = 8.0   # only label SNPs with -log10(p) ≥ this on the Manhattan
LOG10_P_NOTICE_THRESHOLD = 8.0  # also used as the gate for printing significant hits

# Default: process every available component. The original analysis short-
# circuited at component 291; set this to "291" (or any component string) to
# replicate the exact set of figures included in the manuscript run, leave at
# None to plot all components.
STOP_AT_COMPONENT: str | None = None


def abbreviate_assoc_files() -> None:
    """Step 1: write `<base>.abbr.txt` next to each `<base>.assoc.txt` in
    `GEMMA_OUTPUT_DIR`. The abbreviated files are space-separated with
    columns `chrom pos snp pval` (matching what the rest of this script and
    `11_eqtls.py` expect)."""
    for txt in glob(os.path.join(GEMMA_OUTPUT_DIR, "*.assoc.txt")):
        base = txt.rsplit(".assoc.txt", 1)[0]
        out_path = f"{base}.abbr.txt"
        with open(txt) as src, open(out_path, "w") as dst:
            for line in src.read().splitlines()[1:]:  # skip header
                parts = line.split("\t")
                # GEMMA `lmm` output column convention:
                #   parts[0] = chr, parts[1] = rs, parts[2] = pos, parts[-1] = pval
                dst.write(f"{parts[0]} {parts[2]} {parts[1]} {parts[-1]}\n")


def plot_manhattan_and_qq(abbr_path: str) -> None:
    """Step 2: render Manhattan + Q-Q for one abbreviated file."""
    base = abbr_path.rsplit(".abbr.txt", 1)[0]
    comp = os.path.basename(base).removeprefix("results")

    out_manhattan = os.path.join(PLOT_OUTPUT_DIR, f"{comp}_Manhattan_Component.png")
    if os.path.exists(out_manhattan):
        return  # already plotted

    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 150
    mpl.rcParams["figure.figsize"] = (7.375, 3.375)

    hip = np.genfromtxt(abbr_path, dtype=None)

    chrs = [str(i) for i in range(1, 23)]
    chrs_names = np.array(chrs, dtype=object)
    chrs_names[1::2] = ""  # show every-other label for readability

    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    colors = [hex2color(c) for c in palette]

    # ---- Manhattan ----
    manhattan(
        hip["f3"], hip["f1"], hip["f0"].astype(str), " ",
        chrs_plot=chrs,
        chrs_names=chrs_names,
        cut=0,
        title=" ",
        xlabel="chromosome",
        ylabel="-log10(p-value)",
        colors=colors,
    )

    # Annotate the most-extreme SNP per chromosome where −log10 p ≥ threshold
    if max(-np.log10(hip["f3"])) >= LOG10_P_LABEL_THRESHOLD:
        print(comp)
        texts = []
        shift = np.array([0.0])
        for chrom in chrs:
            sel = np.where(hip["f0"].astype(str) == chrom)[0]
            x = shift[-1] + hip["f1"][sel]
            y = -np.log10(hip["f3"][sel])
            z = hip["f2"][sel].astype(str)
            extreme = [j for j, p in enumerate(y) if p >= LOG10_P_NOTICE_THRESHOLD]
            if extreme:
                j = extreme[int(np.argmax(y[extreme]))]
                texts.append(plt.text(x[j], y[j], z[j], size=7, color="black"))
                print(f"  chr{chrom}: {z[j]}")
            shift = np.append(shift, np.max(x) if len(x) else shift[-1])
        if texts:
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))

    plt.tight_layout()
    plt.savefig(out_manhattan, dpi=150)
    plt.close()

    # ---- Q-Q ----
    mpl.rcParams["figure.dpi"] = 100
    mpl.rcParams["savefig.dpi"] = 100
    mpl.rcParams["figure.figsize"] = (5.375, 5.375)

    qqplot(
        [hip["f3"]],
        ["Component"],
        color=["b"],
        fill_dens=[0.2],
        error_type="theoretical",
        distribution="beta",
        title="",
    )
    plt.savefig(
        os.path.join(PLOT_OUTPUT_DIR, f"qq_{comp}_Component_theoretical_error.png"),
        dpi=150,
    )
    plt.close()


def main() -> None:
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    abbreviate_assoc_files()

    for abbr in glob(os.path.join(GWAS_INPUT_DIR, "*.abbr.txt")):
        comp = os.path.basename(abbr).rsplit(".abbr.txt", 1)[0].removeprefix("results")
        if STOP_AT_COMPONENT is not None and comp == STOP_AT_COMPONENT:
            break  # historical short-circuit; set STOP_AT_COMPONENT = None to disable
        plot_manhattan_and_qq(abbr)


if __name__ == "__main__":
    main()
