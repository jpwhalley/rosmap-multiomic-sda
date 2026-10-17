"""Sign-concordance null for the autosomal C74 genes: exact zeros in SEA-AD 106 loadings, and binomial tests against
empirical base rates (non-selected autosomal genes) instead of 0.5."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from vc_common import *
SEA = cfg.SEAAD
gi = pd.read_csv(f"{SEA}/gene_info.csv", index_col=0); sg = pd.read_csv(f"{SEA}/sda/genes_final.csv", index_col=0); sg.index = sg.index.astype(int)
nm = gi.feature_name.reindex(sg.columns); n2e = {n: e for e, n in nm.items()}
GC = pd.read_csv(f"{CACHE}/gene_chrom_gtf100.csv").set_index("gene_id")
L = loadings(74, "RNA"); com = [g for g in L.index if g in n2e]
x = L.loc[com, "X"].values; p = L.loc[com, "P"].values; y = sg.loc[106, [n2e[g] for g in com]].values
ch = np.array([GC.chrom.get(n2e[g].split(".")[0], "NA") for g in com]); au = np.isin(ch, [str(i) for i in range(1, 23)])
sel = au & (p > 0.5)
print("autosomal PIP genes:", sel.sum(), "| SEA-AD loading exactly 0:", int((y[sel] == 0).sum()), "| C74 loading 0:", int((x[sel] == 0).sum()))
k = int((np.sign(x[sel]) * np.sign(y[sel]) == -1).sum()); n = int(sel.sum()); nz = int(((x[sel] != 0) & (y[sel] != 0)).sum())
print(f"concordant (opposite signs) {k}/{n}; among non-zero pairs {k}/{nz}")
for lab, bm in (("all non-PIP autosomal", au & (p < 0.5)), ("non-PIP autosomal, |C74 x| top decile", au & (p < 0.5) & (np.abs(x) > np.quantile(np.abs(x[au & (p < 0.5)]), 0.9))),
                ("non-PIP autosomal, |C74 x| matched (>= min |x| of PIP genes)", au & (p < 0.5) & (np.abs(x) >= np.abs(x[sel]).min()))):
    b = bm & (x != 0) & (y != 0); p0 = float((np.sign(x[b]) * np.sign(y[b]) == -1).mean())
    print(f"base rate [{lab}] = {p0:.3f} (n {b.sum()}); binomial {k}/{n} vs p0: p = {ss.binomtest(k, n, p0, alternative='greater').pvalue:.2g}")
print("vs 0.5 two-sided:", f"{ss.binomtest(k, n, 0.5).pvalue:.2g}")
