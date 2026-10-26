# Figure code

Code that draws the figure panels of the manuscript and assembles the composites.

- Figure 1 is a schematic drawn in Illustrator. Its panel b (modality coverage) is drawn by `panels/round1/fig1c_modality_coverage.py`.
- Figures 2 to 5 and Supplementary Figures S1 to S11 are built by `composites/Fig2` ... `Fig5` and `composites/S1` ... `S11`: every panel is drawn at its printed size, then the panels are placed and lettered by `composites/compose.py`.

Run against the inputs of the v23 figures (2026-10-16), `build_all.sh` reproduced every panel file of these figures (identical apart from PDF creation dates), and the composites render pixel for pixel as the final figure PDFs.

## Requirements

- The Python environment of the repository (`uv.lock`). Arial must be installed.
- For the composites: `xelatex` with TikZ, Ghostscript (`gs`) and poppler (`pdftotext`, `pdffonts`, `pdfinfo`, `pdftoppm`).

## Inputs and outputs

Locations are read by `figure_paths.py` from environment variables or from the file named by `G_ROS_CONFIG`, with the names and defaults of `scripts/round2/config.py` (see `scripts/round2/paths.env.example`). Its docstring lists every variable. Inputs are only read; every output goes under `G_ROS_FIGURE_OUT` (default `$G_ROS_RESULTS_DIR/figures`).

Figure-only variables, in addition to those of `scripts/round2`:

```bash
G_ROS_CLINICAL_PATH=...      # ROSMAP_clinical.csv (default references/, as for the notebooks)
G_ROS_ENRICHR_DIR=...        # Enrichr KEGG tables (default $G_ROS_ARCHIVE_DIR/enrichr)
G_ROS_SOURCE_DATA_DIR=...    # source-data tables, Figure S9 (default $G_ROS_DEPOSIT_DIR/source_data)
G_ROS_ROUND1_DIR=...         # fig4c_points.csv (Figure 2c)
G_ROS_V1_PANELS_DIR=...      # version-1 panel files, for the builders' checks
G_ROS_V23_PANELS_DIR=...     # v23 panel files, for the same checks
G_ROS_MAIN_FIGURES_DIR=...   # Illustrator main figures (replot/kegg_panels/02 only)
G_ROS_FIGURE_OUT=...         # outputs
```

The inputs from `G_ROS_RESULTS_DIR` are, relative to it: `revision_analyses/{structure,verify_structure,c74_c278,verify_c74_c278,neuronal_c21,verify_neuronal_c21}/...`, `covariates/`, `verify_dossier/out/` and `component_dossier/cache/` (the SDA input matrices and loadings).

## Build

```bash
bash scripts/figures/build_all.sh            # panels, then composites; logs in $G_ROS_FIGURE_OUT/logs
bash scripts/figures/build_all.sh panels     # or composites
```

`build_all.sh` lists the run order. Computation is capped at two threads.

## Layout

| Folder | What it draws |
| --- | --- |
| `panels/` | `vp_common.py` (shared loaders and style). `p01` S1, `p02` S3, `p03` S4, `p03b` Fig 2g, `p04` Fig 3f and S5, `p05` Fig 4d, 5d, S6 and S7, `p06` S2f-g, S8f, S9f, S10f, S11e. Each writes a log with the statistics quoted in the legends. `p99_check.py <folder>` checks fonts, clipping, withdrawn terms and donor-ID tokens in panel PDFs. |
| `panels/cell_type_bars/` | Corrected cell-type loading bars (`make_relink_panels.py OUTDIR`). |
| `panels/round1/` | Figure 1b and Figure 2c (Illustrator version) as drawn for the first revision. They list Helvetica first; matplotlib falls back to Arial or DejaVu Sans where it is missing. |
| `qlabel_panels/` | Port of the version-1 clinical box panels (Fig 2a, 3a, 4a, 5a), used as a module by the Fig 2 to 5 builders; `python qlabel_panels.py --qfamily corrected --jitter recovered_jitter.json --outdir DIR` draws the four corrected panels. `recovered_jitter.json` holds the strip-plot jitter recovered from the published PDFs (`recover_jitter.py`). `vendor/statannotations` is statannotations 0.7.2, unmodified (MIT licence included). |
| `replot/` | KEGG bars of Fig 3e, 4e and 5e without in-bar gene lists (`kegg_panels/03_replot.py`; `01` and `02` regenerate `orig_spec.json` and `placement.json`), and the relabelled Fig 4a and 5a (`sample_panels/`). |
| `composites/compose.py` | Places panel PDFs on a page from a layout JSON (xelatex + TikZ, letters in Arial Bold), then checks page size, printed type sizes and fonts. Layout paths may use `$G_ROS_FIGURE_OUT`. Pass `--workdir` (otherwise it writes next to the layout). |
| `composites/S1` ... `S11`, `Fig2` ... `Fig5` | The final panel scripts and the layout JSON of each figure. |

Notes on the composite folders:

- Final single-panel fixes run after the main script and write to `panels_fix/` (S2 f and g to `panels/`): `S2/s2fg_fix.py`, `S8/s8_fix_cdef.py`, `S10/fix_c_al121.py` (run with `1.5 0.3 AL121890.2,AL031666.2`), `S11/s11c_relabel.py`, `Fig3/f3f_fix_italic.py`.
- `Fig2/fig2_panels.py` is run three times with different label-placement options (listed in its docstring and in `build_all.sh`).
- `S5/make_cells.py`, `Fig2/make_layout.py` and `Fig4/fig4_layout.py` rewrite the layout JSON next to them; the S9 script also writes its layout to the output folder. The committed layouts are the ones used for the final figures.
- Several scripts compare their output with the version-1 or v23 panel files and stop if points, bars or labels differ.
