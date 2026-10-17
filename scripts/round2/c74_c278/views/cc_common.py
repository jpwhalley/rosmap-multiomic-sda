"""Shared loaders for the C74 / C278 revision analyses (2026-10-13 round 2, revision_analyses/c74_c278).
Read-only on the SDA archive ($G_ROS_ROSMAP_DIR), SEA-AD ($G_ROS_SEAAD_DIR), the reference tables, the corrected tables
and the earlier round-2 outputs (whose caches are reused read-only). Writes only under
$G_ROS_RESULTS_DIR/revision_analyses/c74_c278/ (out/: aggregates, no donor IDs; panels/: draft panels with per-donor points)
and $G_ROS_RESULTS_DIR/revision_analyses/c74_c278/cache."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats as ss

os.environ.setdefault("OMP_NUM_THREADS", cfg.THREADS); os.environ.setdefault("OPENBLAS_NUM_THREADS", cfg.THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", cfg.THREADS)

ROS = cfg.ROS
SDA = cfg.SDA
R2 = cfg.WORK
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = f"{R2}/revision_analyses/c74_c278"
OUT = f"{BASE}/out"; PAN = f"{BASE}/panels"; CACHE = f"{cfg.WORK}/revision_analyses/c74_c278/cache"; HOM = f"{BASE}/homer"
for d in (OUT, PAN, CACHE, HOM): cfg.out_dir(d)
DCACHE = f"{cfg.WORK}/component_dossier/cache"            # dossier caches (validated; read-only reuse)
NEXT = cfg.TABLES
BCOR = f"{NEXT}/cell_type_loadings_corrected.csv"
REFS = cfg.REFS
GTF100 = f"{REFS}/Homo_sapiens.GRCh38.100.txt"
METH_ANN = f"{REFS}/ROSMAP_arrayMethylation_metaData.tsv"
PEAKS = f"{ROS}/oth/chip/H3K9acDomains.csv"
SEA = cfg.SEAAD
KEGG_GMT = cfg.KEGG_GMT
GENOME = f"{cfg.SEQ}/GRCh37.p13.genome.fa"
CTS = [l.strip() for l in open(f"{SDA}/cell_types.txt") if l.strip()]   # rna.txt block order: Ast Oli Inh OPC Exc Mic
assert CTS == ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
CTNAME = {"Ast": "astrocytes", "Oli": "oligodendrocytes", "Inh": "inhibitory neurons", "OPC": "OPCs",
          "Exc": "excitatory neurons", "Mic": "microglia"}


def donors():
    return [l.strip() for l in open(f"{SDA}/md_samples.txt") if l.strip()]


def score(k):
    """Deposited consensus sample score of component k (IDs 74 and 278 unchanged in the 547 catalogue)."""
    S = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0)
    S.columns = S.columns.astype(str); assert list(S.columns) == donors()
    return S.loc[k].values.astype(float)


def clinical():
    c = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv"); c.index = c.projid.astype(int).astype(str)
    return c.loc[donors()]


def female():
    return (clinical().msex.values == 0)


def flags():
    """Sample-identity flags from identity_checks (private key; never written out).
    Returns dict label -> boolean donor mask (276)."""
    key = pd.read_csv(f"{cfg.PRIVATE}/donor_key.csv")
    d = np.array(donors()); out = {}
    for lab, pid in zip(key.label, key.projid.astype(str)):
        out[lab] = d == pid
        assert out[lab].sum() == 1, lab
    return out


def excl_mask(modality, extra_same_sex=False):
    """Donors excluded from source-data views of the affected modality: the four sex-discordant samples
    (RNA-1, RNA-2: snRNA of another person; CHIP-1, CHIP-2: H3K9ac of another person). With extra_same_sex the
    two same-sex H3K9ac swaps (CHIPID-1/2) are excluded too (sensitivity)."""
    F = flags(); m = np.zeros(276, bool)
    if modality == "RNA":
        for l in ("RNA-1", "RNA-2"): m |= F[l]
    elif modality == "chip":
        for l in ("CHIP-1", "CHIP-2"): m |= F[l]
        if extra_same_sex:
            for l in ("CHIPID-1", "CHIPID-2"): m |= F[l]
    return m


def four_flag_mask():
    F = flags(); m = np.zeros(276, bool)
    for l in ("RNA-1", "RNA-2", "CHIP-1", "CHIP-2"): m |= F[l]
    return m


def loadings():
    return dict(np.load(f"{DCACHE}/loadings.npz", allow_pickle=True))


def comp_loadings(k, m):
    L = loadings(); ci = list(L["comps"]).index(k)
    return pd.DataFrame({"X": L[f"{m}_X"][ci], "P": L[f"{m}_P"][ci]}, index=list(L[f"{m}_feats"]))


def rna_input():
    """SDA RNA input (CPM), shape (6 cell types, 276 donors, 25,098 genes)."""
    return np.load(f"{DCACHE}/input_RNA.npy", mmap_mode="r").reshape(6, 276, -1)


def mod_input(m):
    return np.load(f"{DCACHE}/input_{m}.npy", mmap_mode="r")


def gene_chrom():
    g = pd.read_csv(GTF100, usecols=["chrom", "gene_name"]).drop_duplicates("gene_name").set_index("gene_name").chrom.astype(str)
    return g


def meth_ann():
    return pd.read_csv(METH_ANN, sep="\t", index_col=0, usecols=["TargetID", "CHR", "MAPINFO", "RefGene"])


def peaks():
    return pd.read_csv(PEAKS, sep="\t", index_col=0)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(adj, 1); return out


def spearman_ci(x, y, B=2000, seed=1):
    """Spearman rho with a percentile bootstrap CI (donor resampling)."""
    ok = ~np.isnan(x) & ~np.isnan(y); x = np.asarray(x)[ok]; y = np.asarray(y)[ok]; n = len(x)
    rho, p = ss.spearmanr(x, y)
    rng = np.random.default_rng(seed); bs = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n); bs[b] = ss.spearmanr(x[i], y[i])[0]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(n=n, rho=float(rho), p=float(p), lo=float(lo), hi=float(hi))


def zagg(M, w):
    """Loading-weighted mean of column-standardised values (donors x features); rows all-NaN stay NaN."""
    M = np.asarray(M, float)
    mu = np.nanmean(M, 0); sd = np.nanstd(M, 0); keep = sd > 0
    Z = (M[:, keep] - mu[keep]) / sd[keep]
    w = np.asarray(w, float)[keep]
    return Z @ w / np.abs(w).sum(), int(keep.sum())


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.2g}" if p < 0.01 else f"{p:.2f}"
