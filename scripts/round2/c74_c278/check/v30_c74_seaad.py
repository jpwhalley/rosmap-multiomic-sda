"""Verify C74 task 2: SEA-AD concordance scoped by chromosome class. Own code; chromosome from the GTF via Ensembl ID
(SEA-AD gene_info) where available. Also: alternative choices (Kendall tau; permutation p for Spearman; class by
ROSMAP-side gene name)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from vc_common import *
SEA = cfg.SEAAD
gi = pd.read_csv(f"{SEA}/gene_info.csv", index_col=0)
sg = pd.read_csv(f"{SEA}/sda/genes_final.csv", index_col=0); sg.index = sg.index.astype(int)
sp = pd.read_csv(f"{SEA}/sda/pip_final.csv", index_col=0); sp.index = sp.index.astype(int)
print("SEA-AD components:", len(sg)); sg = sg.drop(index=[9, 128]); sp = sp.drop(index=[9, 128])
name = gi.feature_name.reindex(sg.columns)
print("duplicated SEA-AD names among columns:", int(name.duplicated().sum()))
L = loadings(74, "RNA"); pipg = set(L.index[L.P > 0.5])
# map ROSMAP gene names to SEA-AD Ensembl columns by name
n2e = {}
for e, n in name.items():
    n2e.setdefault(n, []).append(e)
common = sorted(g for g in pipg if g in n2e)
amb = [g for g in common if len(n2e[g]) > 1]
print("C74 PIP>0.5 genes:", len(pipg), "| measured in SEA-AD (by name):", len(common), "| ambiguous names:", amb)
cols = [n2e[g][0] for g in common]
x = L.loc[common, "X"].values
Y = sg[cols].values
r = np.array([np.corrcoef(x, Y[i])[0, 1] for i in range(len(sg))])
o = np.argsort(-np.abs(r)); print("partner:", sg.index[o[0]], f"r {r[o[0]]:+.4f}; next |r| {abs(r[o[1]]):.3f} (C{sg.index[o[1]]})")
y = sg.loc[106, cols].values; yp = sp.loc[106, cols].values
GC = pd.read_csv(f"{CACHE}/gene_chrom_gtf100.csv").set_index("gene_id")
chr_e = np.array([GC.chrom.get(e.split(".")[0], "NA") for e in cols])
print("chromosome classes (by Ensembl ID):", pd.Series(chr_e).value_counts().to_dict())
na = [g for g, c in zip(common, chr_e) if c not in [str(i) for i in range(1, 23)] + ["X", "Y"]]
print("unannotated / non-canonical:", na)
orient = np.sign(r[list(sg.index).index(106)])
rng = np.random.default_rng(7)
rows = []
def conc(sel, lab):
    a_, b_ = x[sel], y[sel]; n = len(a_)
    pr, pp = ss.pearsonr(a_, b_); sr, spv = ss.spearmanr(a_, b_); kt = ss.kendalltau(a_, b_)
    bp, bs = [], []
    for _ in range(4000):
        i = rng.integers(0, n, n); bp.append(np.corrcoef(a_[i], b_[i])[0, 1]); bs.append(ss.spearmanr(a_[i], b_[i])[0])
    perm = np.array([ss.spearmanr(a_, rng.permutation(b_))[0] for _ in range(4000)])
    k = int((np.sign(a_) * np.sign(b_) == orient).sum()); bt = ss.binomtest(k, n, 0.5)
    rows.append(dict(cls=lab, n=n, pearson=pr, p_lo=np.nanpercentile(bp, 2.5), p_hi=np.nanpercentile(bp, 97.5), pearson_p=pp,
                     spearman=sr, s_lo=np.nanpercentile(bs, 2.5), s_hi=np.nanpercentile(bs, 97.5), spearman_p=spv,
                     spearman_perm_p=(1 + (np.abs(perm) >= abs(sr)).sum()) / 4001, kendall=kt.statistic, concord=k, binom_p=bt.pvalue,
                     sq_share=(a_ ** 2).sum() / (x ** 2).sum()))
isX = chr_e == "X"; isY = chr_e == "Y"; isA = np.isin(chr_e, [str(i) for i in range(1, 23)])
conc(np.ones(len(x), bool), "all"); conc(isX | isY, "sex chromosomes"); conc(isX, "chrX"); conc(isY, "chrY"); conc(isA, "autosomal")
oa = np.where(isA)[0][np.argsort(-np.abs(x[isA]))]
for d in (5, 10):
    s = np.zeros(len(x), bool); s[oa[d:]] = True; conc(s, f"autosomal minus top {d}")
conc(isA & (yp > 0.5), "autosomal, PIP>0.5 in both")
# also: does the XIST-only removal change the sex-chromosome Pearson?
s = (isX | isY) & (np.array(common) != "XIST"); conc(s, "sex chromosomes without XIST")
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/v30_c74_seaad.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250); print(T.to_string(index=False, float_format=lambda v: f"{v:+.3g}"))

# ---- base rate of sign concordance for genes NOT selected by C74 (null for the binomial 0.5 assumption) ----
allc = [g for g in L.index if g in n2e]
xa = L.loc[allc, "X"].values; pa = L.loc[allc, "P"].values; ya = sg.loc[106, [n2e[g][0] for g in allc]].values
cha = np.array([GC.chrom.get(n2e[g][0].split(".")[0], "NA") for g in allc]); auto_ = np.isin(cha, [str(i) for i in range(1, 23)])
for lab, sel in (("autosomal, C74 PIP<0.5", auto_ & (pa < 0.5)), ("autosomal, C74 PIP<0.1", auto_ & (pa < 0.1)),
                 ("autosomal, C74 PIP<0.5, |x| top decile", auto_ & (pa < 0.5) & (np.abs(xa) > np.quantile(np.abs(xa[auto_ & (pa < 0.5)]), 0.9)))):
    nz = sel & (xa != 0) & (ya != 0)
    k = int((np.sign(xa[nz]) * np.sign(ya[nz]) == orient).sum())
    print(f"base rate {lab}: {k}/{nz.sum()} = {k/nz.sum():.3f} concordant; Spearman {ss.spearmanr(xa[nz], ya[nz])[0]:+.3f}")
