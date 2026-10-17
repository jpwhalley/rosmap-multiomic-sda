"""Folder layout shared by the SEA-AD preprocessing scripts in this folder.

G_ROS_SEAAD_DIR      the SEA-AD DLPFC folder: the open-access AnnData file `local.h5ad`;
                     the pseudobulk tables, `metadata.csv`, `gene_info.csv` and `sda/` are written here
                     (the same folder that scripts/round2 reads as G_ROS_SEAAD_DIR)
G_ROS_SEAAD_MTG_DIR  the SEA-AD MTG folder with its `local.h5ad` (default: `mtg` next to the DLPFC folder)

The donor-by-cell-type count tables `{dlpfc,mtg}_sample_by_cell.csv` go in the folder above both,
as in the original analysis.
"""
import os, sys

SEAAD_DIR = os.environ.get("G_ROS_SEAAD_DIR")
if not SEAAD_DIR:
    sys.exit("Set G_ROS_SEAAD_DIR to the SEA-AD DLPFC folder (see scripts/seaad/README.md).")
SEAAD_DIR = os.path.abspath(SEAAD_DIR)
MTG_DIR = os.path.abspath(os.environ.get("G_ROS_SEAAD_MTG_DIR", os.path.join(os.path.dirname(SEAAD_DIR), "mtg")))
COUNTS_DIR = os.path.dirname(SEAAD_DIR)
