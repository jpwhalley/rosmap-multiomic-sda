"""Input and output locations for the figure code.

Each location is read from an environment variable, or from the KEY=VALUE file named by G_ROS_CONFIG
(scripts/round2/paths.env.example); environment variables win over the file. Names and defaults are those of
scripts/round2/config.py, plus the figure-only ones below. A variable that has no default and is not set is left
as the literal "$NAME", so a missing input names the variable in the error message. Inputs are only read; every
output (panel PDFs, logs, composite builds) goes under G_ROS_FIGURE_OUT.

  G_ROS_REFERENCES_DIR    reference tables (Homo_sapiens.GRCh38.100.txt, H3K9acDomains.csv,
                          ROSMAP_arrayMethylation_metaData.tsv); default <repo>/references
  G_ROS_CLINICAL_PATH     ROSMAP_clinical.csv; default <references>/ROSMAP_clinical.csv
  G_ROS_ROSMAP_DIR        controlled ROSMAP data; the figures read oth/snps/ (PLINK genotypes, GEMMA output in
                          gemma/output/) and mit/rna/pseudobulk/; default <repo>/rosmap
  G_ROS_SDA_DIR           SDA inputs and consensus outputs (md_samples.txt, samples_final.csv, genes_final_*.csv,
                          pip_final_*.csv); default <rosmap>/sda
  G_ROS_SEAAD_DIR         SEA-AD DLPFC decomposition (gene_info.csv, sda/genes_final.csv); default <repo>/seaad
  G_ROS_RESULTS_DIR       outputs of scripts/round2, in the folder layout they write; default <repo>/results
  G_ROS_PRIVATE_DIR       private outputs of the identity checks (donor_key.csv); default <results>/private
  G_ROS_TABLES_DIR        catalogue tables (cell_type_loadings_corrected.csv, ROSMAP_cell_type_panels_values.csv,
                          clinical_BH_547_family.csv, FigS2c_SEAAD_C21_cell_type_values.csv);
                          default <results>/catalogue_1008
  G_ROS_ARCHIVE_DIR       version-1 result tables; default <repo>/version1_results
  G_ROS_ENRICHR_DIR       Enrichr KEGG tables, <component>_pathway_encrichment.csv; default <archive>/enrichr
  G_ROS_DEPOSIT_DIR       Synapse deposit, version 3; default <repo>/synapse
  G_ROS_SOURCE_DATA_DIR   source-data tables (Figure S9; S9a also reads controlled/); default <deposit>/source_data
  G_ROS_ROUND1_DIR        first-revision outputs (fig4c_points.csv, Figure 2c)
  G_ROS_V1_PANELS_DIR     version-1 panel files (outlier/<k>/, enrichr/, GWAS/); several builders check their
                          redraw against them
  G_ROS_V23_PANELS_DIR    the v23 panel files, used by the same checks
  G_ROS_MAIN_FIGURES_DIR  the Illustrator main figures (replot/kegg_panels/02_measure_placement.py)
  G_ROS_FIGURE_OUT        where everything is written; default <results>/figures. The composite layouts refer to
                          $G_ROS_FIGURE_OUT, so set it in the environment before running composites/compose.py.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))          # scripts/figures
REPO = os.path.dirname(os.path.dirname(HERE))


def _read_file(path):
    vals = {}
    if path and os.path.isfile(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    return vals


_FILE = _read_file(os.environ.get("G_ROS_CONFIG", ""))


def _env(name, default=None):
    v = (os.environ.get(name) or _FILE.get(name) or "").strip()
    if v:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(v)))
    return default if default is not None else f"${name}"


REFERENCES = _env("G_ROS_REFERENCES_DIR", os.path.join(REPO, "references"))
CLINICAL = _env("G_ROS_CLINICAL_PATH", os.path.join(REFERENCES, "ROSMAP_clinical.csv"))
ROSMAP = _env("G_ROS_ROSMAP_DIR", os.path.join(REPO, "rosmap"))
SDA = _env("G_ROS_SDA_DIR", os.path.join(ROSMAP, "sda"))
SEAAD = _env("G_ROS_SEAAD_DIR", os.path.join(REPO, "seaad"))
RESULTS = _env("G_ROS_RESULTS_DIR", os.path.join(REPO, "results"))
PRIVATE = _env("G_ROS_PRIVATE_DIR", os.path.join(RESULTS, "private"))
TABLES = _env("G_ROS_TABLES_DIR", os.path.join(RESULTS, "catalogue_1008"))
ARCHIVE = _env("G_ROS_ARCHIVE_DIR", os.path.join(REPO, "version1_results"))
ENRICHR = _env("G_ROS_ENRICHR_DIR", os.path.join(ARCHIVE, "enrichr"))
DEPOSIT = _env("G_ROS_DEPOSIT_DIR", os.path.join(REPO, "synapse"))
SOURCE_DATA = _env("G_ROS_SOURCE_DATA_DIR", os.path.join(DEPOSIT, "source_data"))
ROUND1 = _env("G_ROS_ROUND1_DIR")
V1_PANELS = _env("G_ROS_V1_PANELS_DIR")
V23_PANELS = _env("G_ROS_V23_PANELS_DIR")
MAIN_FIGURES = _env("G_ROS_MAIN_FIGURES_DIR")
FIGOUT = _env("G_ROS_FIGURE_OUT", os.path.join(RESULTS, "figures"))

# an output folder inside the repository ignores its own contents
if os.path.commonpath([FIGOUT, REPO]) == REPO:
    os.makedirs(FIGOUT, exist_ok=True)
    if not os.path.exists(os.path.join(FIGOUT, ".gitignore")):
        open(os.path.join(FIGOUT, ".gitignore"), "w").write("*\n")
