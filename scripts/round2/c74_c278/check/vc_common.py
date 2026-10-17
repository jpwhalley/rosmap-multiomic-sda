"""Shared loaders for the independent verification of the C74/C278 revision analyses (verify_c74_c278).
Reads raw SDA files (rna/meth/chip.txt via the v00 cache; genes_final_*/pip_final_* directly), the deposited
samples_final.csv, clinical CSV, corrected cell types, identity key (read-only; never written out).
Writes only under $G_ROS_RESULTS_DIR/revision_analyses/verify_c74_c278/."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", cfg.THREADS); os.environ.setdefault("OPENBLAS_NUM_THREADS", cfg.THREADS); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", cfg.THREADS)
from scipy import stats as ss
ROS = cfg.ROS; SDA = cfg.SDA
R2 = cfg.WORK
BASE = f"{R2}/revision_analyses/verify_c74_c278"
OUT = cfg.out_dir(BASE, "out"); PAN = cfg.out_dir(BASE, "panels"); CACHE = cfg.out_dir(cfg.WORK, "revision_analyses", "verify_c74_c278", "cache"); HOM = f"{BASE}/homer"
OTHER = f"{R2}/revision_analyses/c74_c278"
NEXT = cfg.TABLES
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
DON = [l.strip() for l in open(f"{SDA}/md_samples.txt") if l.strip()]
GENES = [l.strip() for l in open(f"{SDA}/genes.txt") if l.strip()]
CPGS = [l.strip() for l in open(f"{SDA}/md_cpg.txt") if l.strip()]
PEAKS = [l.strip() for l in open(f"{SDA}/md_chip.txt") if l.strip()]
GENOME = f"{cfg.SEQ}/GRCh37.p13.genome.fa"

def rna():  # (6, 276, 25098) CPM; block order = cell_types.txt
    return np.load(f"{CACHE}/rna.npy", mmap_mode="r").reshape(6, 276, -1)
def meth(): return np.load(f"{CACHE}/meth.npy", mmap_mode="r")
def chip(): return np.load(f"{CACHE}/chip.npy", mmap_mode="r")

_SC = None
def score(k):
    global _SC
    if _SC is None:
        _SC = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0); _SC.columns = _SC.columns.astype(str)
        assert list(_SC.columns) == DON
    return _SC.loc[k].values.astype(float)

_LD = {}
def loadings(k, mod):
    """gene loading X and PIP for component k, modality in RNA/meth/chip/proteins (deposited finals)."""
    key = (k, mod)
    if key not in _LD:
        def row(fn):
            with open(fn) as f:
                hdr = f.readline().rstrip("\n").split(",")[1:]
                for l in f:
                    i = l.find(",")
                    if l[:i] == str(k):
                        return hdr, np.array(l.rstrip("\n").split(",")[1:], float)
            raise KeyError(k)
        h1, x = row(f"{SDA}/genes_final_{mod}.csv"); h2, p = row(f"{SDA}/pip_final_{mod}.csv")
        assert h1 == h2
        _LD[key] = pd.DataFrame({"X": x, "P": p}, index=h1)
    return _LD[key]

def clinical():
    c = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv"); c.index = c.projid.astype(int).astype(str)
    return c.loc[DON]
def female(): return clinical().msex.values == 0

def idflags():
    k = pd.read_csv(f"{cfg.PRIVATE}/donor_key.csv"); d = np.array(DON); out = {}
    for lab, pid in zip(k.label, k.projid.astype(str)): out[lab] = d == pid
    return out
def excl(mod):
    F = idflags(); m = np.zeros(276, bool)
    for l in {"RNA": ("RNA-1", "RNA-2"), "chip": ("CHIP-1", "CHIP-2"), "meth": ()}[mod]: m |= F[l]
    return m

def ct_corrected():
    return pd.read_csv(f"{NEXT}/cell_type_loadings_corrected.csv", index_col=0)

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out

def rho_ci(x, y, B=2000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    r, p = ss.spearmanr(x, y); rng = np.random.default_rng(seed); n = len(x); bs = np.empty(B)
    rx = None
    for b in range(B):
        i = rng.integers(0, n, n); bs[b] = ss.spearmanr(x[i], y[i])[0]
    return dict(n=n, rho=r, p=p, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5))

def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    return ss.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))
