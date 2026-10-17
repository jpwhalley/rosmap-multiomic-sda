# SEA-AD preprocessing and SDA runs

The SEA-AD decomposition used for the cross-cohort validation (Methods: Validation in an independent cohort), from the open-access SEA-AD atlases to the ten SDA runs and their consensus. These scripts were run in 2023 (Python 3.10.8, pandas 1.x); the changes are listed in each docstring (paths from environment variables, `DataFrame.append` written as `pd.concat`, and two small fixes).

**Inputs**: the SEA-AD single-nucleus AnnData files (`local.h5ad`) of the dorsolateral prefrontal cortex (DLPFC) and the middle temporal gyrus (MTG), downloaded on 5 October 2023 from [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/collections/1ca90a2d-2943-483d-b678-b809bf464c30): "Whole Taxonomy - DLPFC" version 63e83c64-ac87-49db-b19a-87d0e8c181ec and "Whole Taxonomy - MTG" version 4aca65b8-bc4b-4c68-833e-36af8f43fbfa (schema 3.1.0; the files match these versions in byte size). SEA-AD is also distributed through the Allen Brain Map and its [AWS bucket](https://registry.opendata.aws/allen-sea-ad-atlas/), whose current files may differ. Layout:

```
<folder>/dlpfc/local.h5ad    G_ROS_SEAAD_DIR (the folder that scripts/round2 reads)
<folder>/mtg/local.h5ad      G_ROS_SEAAD_MTG_DIR (default: mtg next to dlpfc)
```

**Order**

1. `00_sample_by_cell.py`: nuclei per donor and cell type in both regions (`<folder>/{dlpfc,mtg}_sample_by_cell.csv`).
2. `01_pseudobulk.py`: donors in both regions less one (82), ten cell types; pseudobulk counts per donor and cell type, genes with at least 10 counts in at least 10 profiles (31,648), reads per million.
3. `02_export_sda_inputs.py`: the SDA input `sda/rna.txt` (82 donors x 31,648 genes x 10 cell types) and `sda/matd.sh`, a SLURM array of ten SDA v1.1 runs (800 components requested, 3,000 iterations). Set `G_ROS_SDA_BINARY` and `G_ROS_CLUSTER_WORKDIR` as for `scripts/05_export_sda_inputs.py`.
4. Run `sda/matd.sh` on the cluster.
5. `03_cluster_components.py`: the consensus of the ten runs as deposited in 2023 (`sda/*_final.csv`). `scripts/round2/seaad/01_recluster_seaad.py` then reproduces it and applies the corrected run count (141 components), and `02_partner.py` matches the ROSMAP components to them.

Scripts 00 and 01 need `scanpy` (not in `uv.lock`) and enough memory for the DLPFC atlas (53 GB on disk).

**Checked against the original files (2026-10-17)**: the donor-by-cell-type tables equal a cross-tabulation of the atlases' cell metadata; the donor and cell-type selection gives the 82 donors and 10 cell types of `sda/md_samples.txt` and `sda/cell_types.txt`; pseudobulk sums for three sampled donor and cell-type pairs equal the original count tables; the gene filter gives the 31,648 genes of `sda/genes.txt` and the reads-per-million tables are bit-identical; `02_export_sda_inputs.py` on the original tables writes `md_samples.txt` and `genes.txt` byte for byte and an `rna.txt` whose ten cell-type blocks are byte-identical to the original, in directory-listing order (the order used is in `sda/cell_types.txt`); `scripts/round2/seaad/01_recluster_seaad.py`, which follows `03_cluster_components.py` step by step, reproduces the deposited `*_final.csv`. The donor order of `01_pseudobulk.py` follows Python set iteration and can differ between runs; the order used is in `sda/md_samples.txt`.
