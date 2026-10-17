"""Point 5(b-i) addendum: the RNA/CpG-gene overlap with chromosome added to the density strata (exact, as N1 in 02).
Strata: chromosome; chromosome x d_sda (1 / 2-3 / 4+). Aggregates only.
Fix (round-2 record, 10-16 rerun): universe genes with no chromosome in the sidecar get the explicit label 'nan', so in
the chromosome x d_sda strata they are split by density like every other chromosome (expected 6.10, fold 2.29,
p 9.67e-4, as reported). The filed log pooled them into one stratum through pandas missing-value handling (6.14, 2.28, 1.03e-3)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys
import numpy as np, pandas as pd
from scipy import stats as ss
SDA = cfg.SDA; SYN = cfg.DEPOSIT   # deposit feature sidecars


def pip_row(fname, k=74):
    with open(f"{SDA}/{fname}") as f:
        hdr = f.readline().rstrip("\n").split(",")[1:]
        for line in f:
            key, rest = line.split(",", 1)
            if key == str(k): return pd.Series(np.array(rest.rstrip("\n").split(","), dtype=float), index=hdr)


meth = pd.read_csv(f"{SYN}/meth_features.csv", index_col=0); rna = pd.read_csv(f"{SYN}/RNA_features.csv", index_col=0)
pm = pip_row("pip_final_meth.csv"); pr = pip_row("pip_final_RNA.csv")
G = {c: {x.strip() for x in v.split(";") if x.strip()} for c, v in meth.RefGene.dropna().astype(str).items()}
annot = set().union(*G.values()); rna["c"] = rna.chrom.astype(str).str.replace("chr", "", regex=False)
auto = set(rna.index[~rna.c.isin(["X", "Y", "MT", "M"])])
U = sorted(set(pr.index) & annot & auto); ui = {g: i for i, g in enumerate(U)}
B = set().union(*[G.get(c, set()) for c in pm.index[pm > 0.5]]) & set(U); A = set(pr.index[pr > 0.5]) & set(U)
inA = np.array([g in A for g in U]); inB = np.array([g in B for g in U]); k = int((inA & inB).sum())
d = np.zeros(len(U), int)
for gs in G.values():
    for g in gs:
        if g in ui: d[ui[g]] += 1
chrom = np.array([str(x) for x in rna.c.reindex(U).fillna("nan")], dtype=object)   # explicit label for missing chromosome (fix above)
print(f"U {len(U)}, A {inA.sum()}, B {inB.sum()}, k {k}")
print("A (RNA) genes by chromosome:", pd.Series(chrom[inA]).value_counts().to_dict())
print("overlap genes by chromosome:", pd.Series(chrom[inA & inB]).value_counts().to_dict())


def exact(st):
    dist = np.array([1.0])
    for s in np.unique(st):
        m = st == s; u, a, n = m.sum(), int((inA & m).sum()), int((inB & m).sum())
        if a and n: dist = np.convolve(dist, ss.hypergeom.pmf(np.arange(min(a, n) + 1), u, a, n))
    x = np.arange(len(dist)); E = float((x * dist).sum())
    return E, k / E, float(dist[k:].sum())


coarse = np.digitize(d, [1, 2, 4]) - 1
for name, st in [("chromosome", pd.factorize(chrom)[0]), ("chromosome x d_sda (1/2-3/4+)", pd.factorize(pd.Series(chrom) + "_" + coarse.astype(str))[0])]:
    E, f, p = exact(st); print(f"{name}: expected {E:.2f}, fold {f:.2f}, exact p {p:.3g}")
