"""00_sample_by_cell.py: nuclei per donor and cell type in the SEA-AD DLPFC and MTG atlases.

Writes `{tissue}_sample_by_cell.csv` (donor x cell-type ontology term) for both regions;
01_pseudobulk.py uses the two tables to choose the donors and cell types. Paths: `_paths.py`.

From the original analysis (td0_prepare_data.py, 2023; Python 3.10.8). Changes: paths, and the
colon missing after the `for` statement. Needs scanpy.
"""
import scanpy as sc
import numpy as np
import pandas as pd
from _paths import SEAAD_DIR, MTG_DIR, COUNTS_DIR

# Setup
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

results = {}

for tissue, folder in [('mtg', MTG_DIR), ('dlpfc', SEAAD_DIR)]:
    # Read data
    adata = sc.read(folder + '/local.h5ad')
    # bdata = sc.read('dlpfc/local.h5ad')

    # # Where the raw count data lives
    # adata.raw.X
    #
    # # Names of the columns in the metadata
    # for col in adata.obs:
    #     print(col)

    # Donor IDs
    adata.obs['donor_id'].value_counts()

    # Cell Onotology Types
    adata.obs['cell_type_ontology_term_id'].value_counts()

    # Cross-tabulation of donor_id and cell_type_ontology_term_id
    df_counts = pd.crosstab(adata.obs['donor_id'], adata.obs['cell_type_ontology_term_id'])
    df_counts.to_csv(COUNTS_DIR + '/' + tissue + '_sample_by_cell.csv')
    del adata

    # print(df_counts)
    results[tissue] = df_counts

# Need to do:
# Compare this to the other cell type. Ideally want samples x cell types with at least 100 cells in each cross for both tissue types
# Pseudobulk counts were generated for each combination of gene and sample at cell type level by summing together the within-group gene counts. These were then converted to reads-per-million (RPM) by normalizing by the total count across all genes for each combination of sample and cell type.
