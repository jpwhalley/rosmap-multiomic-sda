"""Paths and thread settings shared by the round-2 analysis scripts.

Every location is read from an environment variable, or from a KEY=VALUE file named by G_ROS_CONFIG
(see paths.env.example); environment variables win over the file. Defaults follow the release layout
(folders beside the repository's own files). Output folders that sit inside the repository get a
`.gitignore` of `*`, so derived files there are never committed. The variable names match scripts/figures.

Inputs
  G_ROS_ROSMAP_DIR      controlled ROSMAP data (default <repo>/rosmap), laid out as sda/ (SDA inputs,
                        results1..10/, version-1 consensus *_final.csv), oth/meta/, oth/snps/ (array genotypes,
                        gemma/ with the archived GEMMA runs), oth/chip/, oth/meth/, meth/, prot/tmt/, mit/meta/,
                        mit/rna/, mit/atac/
  G_ROS_SDA_DIR         default $G_ROS_ROSMAP_DIR/sda; keep it there, as several scripts read sda/ through
                        G_ROS_ROSMAP_DIR
  G_ROS_SEAAD_DIR       SEA-AD DLPFC folder: gene_info.csv and sda/ (default <repo>/seaad)
  G_ROS_DEPOSIT_DIR     Synapse deposit, version 2 (default <repo>/synapse)
  G_ROS_REFERENCES_DIR  reference tables (default <repo>/references)
  G_ROS_ARCHIVE_DIR     version-1 Enrichr tables: enrichr/, cpg_enrichr/ (default <repo>/version1_results)
  G_ROS_SEQUENCES_DIR   GRCh37.p13.genome.fa(.fai), Homo_sapiens.GRCh38.100.gtf.gz, Homo_sapiens.GRCh38.cdna.all.fa.gz,
                        dbsnp_135.hg18.sort.vcf (default <repo>/sequences)
  G_ROS_KEGG_GMT        default $G_ROS_AUDIT_DIR/kegg547/KEGG_2021_Human.gmt (fetched by kegg/kegg547/01_fetch_library.py)
  G_ROS_DATA_S1_V22     Data S1 as submitted in version 22 (default scripts/round2/data_s1/Data_S1_v22.csv, shipped)
  G_ROS_DATA_S1         Data S1, this version (default <repo>/Data_S1.csv)
  G_ROS_QC_EXCLUDE_PATH snRNA-seq projids that failed QC, one per line: qc_failed_projids.txt of the Synapse deposit
                        (default $G_ROS_DEPOSIT_DIR/qc_failed_projids.txt)
  G_ROS_CLINICAL_PATH   ROSMAP_clinical.csv (default $G_ROS_REFERENCES_DIR/ROSMAP_clinical.csv, as for the notebooks)
Outputs
  G_ROS_RESULTS_DIR     round-2 outputs, in the folder layout of the original analysis (default <repo>/results)
  G_ROS_AUDIT_DIR       catalogue audit of 2026-10-08 (default $G_ROS_RESULTS_DIR/audit_1008)
  G_ROS_TABLES_DIR      corrected Data S1 tables of 2026-10-08 (default $G_ROS_RESULTS_DIR/catalogue_1008)
  G_ROS_CATALOGUE_DIR   outputs of scripts/07_cluster_components.py (default $G_ROS_RESULTS_DIR/sda_catalogue)
  G_ROS_PRIVATE_DIR     identity checks: donor key and per-donor identity tables (default $G_ROS_RESULTS_DIR/private)
Threads
  G_ROS_THREADS         BLAS/OpenMP threads and worker processes (default 2)
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.dirname(os.path.abspath(__file__))   # scripts/round2


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


def _get(name, default):
    v = os.environ.get(name) or _FILE.get(name) or str(default)
    return os.path.abspath(os.path.expanduser(os.path.expandvars(v)))


THREADS = str(int(os.environ.get("G_ROS_THREADS") or _FILE.get("G_ROS_THREADS") or 2))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS)

# inputs
ROS = _get("G_ROS_ROSMAP_DIR", os.path.join(REPO, "rosmap"))
SDA = _get("G_ROS_SDA_DIR", os.path.join(ROS, "sda"))
SNP = os.path.join(ROS, "oth", "snps")         # array genotypes; gemma/ holds the archived GEMMA runs
SEAAD = _get("G_ROS_SEAAD_DIR", os.path.join(REPO, "seaad"))
DEPOSIT = _get("G_ROS_DEPOSIT_DIR", os.path.join(REPO, "synapse"))
REFS = _get("G_ROS_REFERENCES_DIR", os.path.join(REPO, "references"))
ARCHIVE = _get("G_ROS_ARCHIVE_DIR", os.path.join(REPO, "version1_results"))
SEQ = _get("G_ROS_SEQUENCES_DIR", os.path.join(REPO, "sequences"))
# outputs
RESULTS = _get("G_ROS_RESULTS_DIR", os.path.join(REPO, "results"))
WORK = RESULTS                                  # round-2 layout root (scripts use cfg.WORK)
AUD = _get("G_ROS_AUDIT_DIR", os.path.join(RESULTS, "audit_1008"))
TABLES = _get("G_ROS_TABLES_DIR", os.path.join(RESULTS, "catalogue_1008"))
CATALOGUE = _get("G_ROS_CATALOGUE_DIR", os.path.join(RESULTS, "sda_catalogue"))
PRIVATE = _get("G_ROS_PRIVATE_DIR", os.path.join(RESULTS, "private"))
# single files
KEGG_GMT = _get("G_ROS_KEGG_GMT", os.path.join(AUD, "kegg547", "KEGG_2021_Human.gmt"))
DS1_V22 = _get("G_ROS_DATA_S1_V22", os.path.join(CODE, "data_s1", "Data_S1_v22.csv"))
DS1_V23 = _get("G_ROS_DATA_S1", os.path.join(REPO, "Data_S1.csv"))
QC_EXCLUDE = _get("G_ROS_QC_EXCLUDE_PATH", os.path.join(DEPOSIT, "qc_failed_projids.txt"))
CLINICAL = _get("G_ROS_CLINICAL_PATH", os.path.join(REFS, "ROSMAP_clinical.csv"))


# The round-2 analyses ran under pandas 2 (requirements.txt pins 2.3.3); pandas 3 can change dtypes and stop a script.
try:
    from importlib.metadata import version as _version
    if int(_version("pandas").split(".")[0]) >= 3:
        import sys as _sys
        _sys.stderr.write(f"note: scripts/round2 was run with pandas 2; pandas {_version('pandas')} is installed "
                          "(see scripts/round2/requirements.txt)\n")
except Exception:
    pass


def _guard(root):
    """Inside the repository, an output root ignores its own contents."""
    root = os.path.abspath(root)
    if os.path.commonpath([root, REPO]) == REPO:
        os.makedirs(root, exist_ok=True)
        gi = os.path.join(root, ".gitignore")
        if not os.path.exists(gi):
            open(gi, "w").write("*\n")


def out_dir(*parts):
    """Create (if needed) and return an output folder."""
    p = os.path.join(*map(str, parts))
    for root in (RESULTS, AUD, TABLES, CATALOGUE, PRIVATE):
        if os.path.commonpath([os.path.abspath(p), root]) == root:
            _guard(root)
    os.makedirs(p, exist_ok=True)
    return p


def qc_failed_projids():
    """The snRNA-seq projids excluded after QC (ints), read from G_ROS_QC_EXCLUDE_PATH."""
    return [int(l.split()[0]) for l in open(QC_EXCLUDE) if l.strip() and not l.startswith("#")]
