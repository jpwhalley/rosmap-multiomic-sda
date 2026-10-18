"""Recheck point 5 numbers quoted in Results l.138 with an independent implementation (interval tree via sorted search,
own stratification and convolution; chromosomes from rna_chrom.py, shared with 05). Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = cfg.THREADS
import numpy as np, pandas as pd
from scipy import stats as ss
SDA = cfg.SDA; SYN = cfg.DEPOSIT   # deposit feature sidecars
def pip(fname, k="74"):
    with open(f"{SDA}/{fname}") as f:
        hdr = f.readline().rstrip("\n").split(",")[1:]
        for line in f:
            a, b = line.split(",", 1)
            if a == k: return pd.Series(np.array(b.rstrip("\n").split(","), float), index=hdr)
meth = pd.read_csv(f"{SYN}/meth_features.csv", index_col=0); chip = pd.read_csv(f"{SYN}/chip_features.csv", index_col=0)
rna = pd.read_csv(f"{SYN}/RNA_features.csv", index_col=0)
pm, pc, pr = pip("pip_final_meth.csv"), pip("pip_final_chip.csv"), pip("pip_final_RNA.csv")
assert list(pm.index) == list(meth.index) and list(pc.index) == list(chip.index)
def conv_test(strata, a_mask, b_mask, k):
    dist = np.array([1.0])
    for s in np.unique(strata):
        m = strata == s; u, a, b = int(m.sum()), int((a_mask & m).sum()), int((b_mask & m).sum())
        if a and b: dist = np.convolve(dist, ss.hypergeom.pmf(np.arange(min(a, b) + 1), u, a, b))
    E = float((np.arange(len(dist)) * dist).sum()); return E, k / E, float(dist[k:].sum())
# ---- genes
G = meth.RefGene.dropna().astype(str).apply(lambda v: {x.strip() for x in v.split(";") if x.strip()})
cnt = {}
for gs in G:
    for g in gs: cnt[g] = cnt.get(g, 0) + 1
from rna_chrom import rna_chrom, AUTOSOMES
chrom = rna_chrom(cfg, rna)   # stable-ID join; NaN = unresolved
U = [g for g in pr.index if g in cnt and chrom.get(g) in AUTOSOMES]
U = sorted(set(U)); Us = set(U)
A = set(pr.index[pr.values > 0.5]) & Us
B = set().union(*[G[c] for c in pm.index[pm.values > 0.5] if c in G.index]) & Us
inA = np.array([g in A for g in U]); inB = np.array([g in B for g in U]); k = int((inA & inB).sum())
d = np.array([cnt[g] for g in U]); dbin = np.where(d >= 4, 2, np.where(d >= 2, 1, 0))
ch = np.array([str(chrom[g]) for g in U])
print(f"genes: U {len(U)} A {inA.sum()} B {inB.sum()} k {k}; plain hypergeom fold {k/(inA.sum()*inB.sum()/len(U)):.4f} p {ss.hypergeom.sf(k-1, len(U), inA.sum(), inB.sum()):.4g}")
E, f, p = conv_test(pd.factorize(pd.Series(ch) + "_" + dbin.astype(str))[0], inA, inB, k)
print(f"genes chromosome x input-CpG (1/2-3/4+): expected {E:.4f} fold {f:.4f} p {p:.5g}")
# ---- peaks
auto = ~chip.Chr.astype(str).isin(["X", "Y"]); P = chip[auto]; c74pk = (pc[auto.values].values > 0.5)
mchr = meth.CHR.astype(str).values; mpos = meth.MAPINFO.values.astype(np.int64); c74cpg = pm.values > 0.5
isl = (meth.Island_Feature.astype(str).values == "Island")
nsda = np.zeros(len(P), int); nc74 = np.zeros(len(P), int); hasisl = np.zeros(len(P), bool)
for chv in P.Chr.astype(str).unique():
    mi = np.where(mchr == chv)[0]; o = np.argsort(mpos[mi]); ps = mpos[mi][o]; mi = mi[o]
    pi = np.where(P.Chr.astype(str).values == chv)[0]
    lo = np.searchsorted(ps, P.Start.values[pi], "left"); hi = np.searchsorted(ps, P.End.values[pi], "right")
    for j, a, b in zip(pi, lo, hi):
        idx = mi[a:b]; nsda[j] = len(idx); nc74[j] = int(c74cpg[idx].sum()); hasisl[j] = bool(isl[idx].any())
u = nsda > 0
lab = c74pk[u]; has = nc74[u] > 0; k2 = int((lab & has).sum())
print(f"peaks: universe {u.sum()} C74 peaks {lab.sum()} with C74 CpG {has.sum()} k {k2}; CpG-level S1 {int(nc74[u][lab].sum())}")
nb = np.where(nsda[u] >= 4, 2, np.where(nsda[u] >= 2, 1, 0))
st = pd.factorize(pd.Series(P.Chr.astype(str).values[u]) + "_" + nb.astype(str) + "_" + P.Classification.astype(str).values[u] + "_" + hasisl[u].astype(str))[0]
E, f, p = conv_test(st, lab, has, k2)
print(f"peaks chromosome x input-CpG x class x island: strata {len(np.unique(st))} expected {E:.4f} fold {f:.4f} p {p:.5g}")
print("peak classes:", P.Classification.value_counts().to_dict())
