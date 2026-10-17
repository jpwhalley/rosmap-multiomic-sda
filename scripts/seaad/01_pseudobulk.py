"""01_pseudobulk.py: SEA-AD DLPFC pseudobulk profiles per donor and cell type (reads per million).

Donors: those in both the DLPFC and the MTG atlases, less H21.33.020 (82 donors). Cell types: the
DLPFC cell-type ontology terms less the eight listed below (10 cell types). For each cell type and
donor, raw counts are summed over nuclei; genes with at least 10 counts in at least 10 donor-by-cell-type
profiles are kept and each profile is scaled to reads per million. Writes `<cell type>_raw_counts.csv`,
`<cell type>_rpm_normalized.csv`, `metadata.csv` and `gene_info.csv` in G_ROS_SEAAD_DIR. Paths: `_paths.py`.

From the original analysis (dlpfc0_prepare_data.py, 2023; Python 3.10.8, pandas 1.x). Changes: paths,
and `DataFrame.append` (removed in pandas 2) written as the equivalent `pd.concat`. The donor order is the
iteration order of a Python set, so it can differ between runs; the order used is in `sda/md_samples.txt`.
Needs scanpy.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import os
from _paths import SEAAD_DIR, COUNTS_DIR

# Setup
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

ff_counts = pd.read_csv(COUNTS_DIR + '/mtg_sample_by_cell.csv', index_col = 0)
ef_counts = pd.read_csv(COUNTS_DIR + '/dlpfc_sample_by_cell.csv', index_col = 0)

samples_to_keep = list(set(ff_counts.index.values).intersection(ef_counts.index.values))

df_counts = pd.read_csv(COUNTS_DIR + '/dlpfc_sample_by_cell.csv', index_col = 0)
df_counts = df_counts.loc[samples_to_keep]

samp_to_delete = 'H21.33.020'
cells_to_delete = ['CL:1001602', 'CL:4023041', 'CL:4023051', 'CL:4023070', 'CL:4023036', 'CL:4023038', 'CL:4023012', 'CL:4023013']

df_counts.drop(samp_to_delete, inplace=True)
df_counts.drop(cells_to_delete, inplace=True, axis=1)
print(df_counts)

# # Read data
adata = sc.read(SEAAD_DIR + '/local.h5ad')

# Extract desired donor_ids and cell types
desired_donor_ids = df_counts.index.tolist()
desired_cell_types = df_counts.columns.tolist()

save_directory = SEAAD_DIR + "/"

# Zeroeth up, get the metadata
results ={}
# Iterate through each donor_id
for donor_id in desired_donor_ids:
    results[donor_id] = {}
    # Filter the cells that belong to this donor_id
    cells_for_donor = adata.obs[adata.obs['donor_id'] == donor_id]
    
    # Dictionary to store metadata with consistent values for this donor_id
    consistent_values = {}
    
    # Go through each column in the adata.obs DataFrame
    for col in adata.obs.columns:
        unique_vals = cells_for_donor[col].unique()
        # If there's only one unique value, store it
        if len(unique_vals) == 1:
            results[donor_id][col] = unique_vals[0]

metadata = pd.DataFrame(results).T
# metadata.drop('Specimen ID', axis=1, inplace=True)
metadata.to_csv(save_directory + 'metadata.csv')
adata.var.to_csv(save_directory + 'gene_info.csv')

# First up get the raw counts done
to_concat = []
for cell_type in desired_cell_types:
    # Initialize empty DataFrames to accumulate counts and RPMs for all donors for this cell type
    all_donors_counts = pd.DataFrame(columns=adata.raw.var_names)

    for donor_id in desired_donor_ids:
        # Get a mask of the rows (cells) that belong to the current donor_id
        donor_mask = adata.obs['donor_id'] == donor_id
        
        # Get a mask of the rows (cells) that belong to the current cell_type
        cell_type_mask = adata.obs['cell_type_ontology_term_id'] == cell_type
        
        # Find the intersection of cells that belong to both the donor_id and the cell_type
        combined_mask = donor_mask & cell_type_mask
        
        # Sum gene counts across all cells of this type for this donor (pseudobulk)
        pseudobulk_counts = adata.raw.X[combined_mask].sum(axis=0)
        pseudobulk_counts = np.asarray(pseudobulk_counts).squeeze()  # Convert to 1D numpy array
        
        # Normalize to RPM
        
        # Convert pseudobulk_counts and rpm to DataFrames and store
        df_counts = pd.DataFrame(pseudobulk_counts.reshape(1, -1), index=[donor_id], columns=adata.raw.var_names)
        
        # Append donor's data to the collective DataFrames
        all_donors_counts = pd.concat([all_donors_counts, df_counts])

    # Save the DataFrames after iterating through all donors for a particular cell type
    safe_cell_type = cell_type.replace(':', '_')
    counts_file_path = os.path.join(save_directory, f"{safe_cell_type}_raw_counts.csv")
    
    all_donors_counts.to_csv(counts_file_path)
    to_concat.append(all_donors_counts)
    
concatenated_df = pd.concat(to_concat, axis=0)
# Get genes that have at least 10 counts in at least 10 rows
genes_to_keep = concatenated_df.columns[concatenated_df.ge(10).sum(axis=0) >= 10].tolist()

# Now do the normalized counts   
for cell_type in desired_cell_types:
    # Initialize empty DataFrames to accumulate counts and RPMs for all donors for this cell type
    all_donors_rpm = pd.DataFrame(columns=genes_to_keep)  # Initialize with genes_to_keep only

    for donor_id in desired_donor_ids:
        # Get a mask of the rows (cells) that belong to the current donor_id
        donor_mask = adata.obs['donor_id'] == donor_id
        
        # Get a mask of the rows (cells) that belong to the current cell_type
        cell_type_mask = adata.obs['cell_type_ontology_term_id'] == cell_type
        
        # Find the intersection of cells that belong to both the donor_id and the cell_type
        combined_mask = donor_mask & cell_type_mask
        
        # Sum gene counts across all cells of this type for this donor (pseudobulk)
        pseudobulk_counts = adata.raw.X[combined_mask].sum(axis=0)
        pseudobulk_counts = np.asarray(pseudobulk_counts).squeeze()  # Convert to 1D numpy array
        
        # Filter the unwanted genes
        pseudobulk_counts = pseudobulk_counts[adata.raw.var_names.isin(genes_to_keep)]
        
        # Normalize to RPM
        total_count = pseudobulk_counts.sum()
        rpm = (pseudobulk_counts / total_count) * 1e6
        
        # Convert rpm to DataFrame and store
        df_rpm = pd.DataFrame(rpm.reshape(1, -1), index=[donor_id], columns=genes_to_keep)
        
        # Append donor's data to the collective DataFrames
        all_donors_rpm = pd.concat([all_donors_rpm, df_rpm])

    # Save the DataFrames after iterating through all donors for a particular cell type
    safe_cell_type = cell_type.replace(':', '_')
    rpm_file_path = os.path.join(save_directory, f"{safe_cell_type}_rpm_normalized.csv")
    
    all_donors_rpm.to_csv(rpm_file_path)
