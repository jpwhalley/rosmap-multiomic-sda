"""Share of each component's squared PIP>0.5 RNA gene loading on chrX/chrY genes (GRCh38.100 gene table). Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from dc_common import *
L=dict(np.load(f"{CACHE}/loadings.npz",allow_pickle=True)); feats=L["RNA_feats"]
ref=pd.read_csv(f"{cfg.REFS}/Homo_sapiens.GRCh38.100.txt",usecols=["chrom","gene_name"]).drop_duplicates("gene_name").set_index("gene_name").chrom.astype(str)
for ci,k in enumerate(L["comps"]):
    P=L["RNA_P"][ci]>0.5; X=L["RNA_X"][ci]; g=feats[P]; ch=ref.reindex(g).fillna("NA").values
    nx=(ch=="X").sum(); ny=(ch=="Y").sum(); w=X[P]**2; wxy=w[(ch=="X")|(ch=="Y")].sum()/w.sum() if P.sum() else np.nan
    top=g[np.argsort(-np.abs(X[P]))[:8]]
    print(f"C{k}: PIP genes {P.sum()}, chrX {nx}, chrY {ny}, share of squared loading on X/Y {wxy:.3f}; XIST in set {'XIST' in set(g)}; top8 {list(map(str,top))}")
