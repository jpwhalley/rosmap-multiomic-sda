"""Autosomal GWAS minimum p per component (chr 1-22 only; PLINK codes 0 = unplaced, 23 X, 24 Y, 25 XY, 26 MT are set aside),
so that a sex-linked or unplaced probe tracking a sex component is not counted as genetic support.
Archived GEMMA 0.96 runs results{k}.abbr.txt (chr ps rs p_score) for 1..546; local C547/C548 abbr from the 2026-10-08 audit.
Reads every component (all 547).
Writes out/01b_gwas_autosomal.csv (aggregates only)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from multiprocessing import Pool
from st_common import *

ARC = f"{cfg.SNP}/gemma/output"
M = pd.read_csv(f"{OUT}/01_master_547.csv", index_col=0)
todo = [int(k) for k in M.index]

def one(k):
    f = f"{ARC}/results{k}.abbr.txt" if k <= 546 else f"{cfg.AUD}/gwas547/output/local_C{k}.abbr.txt"
    d = pd.read_csv(f, sep=" ", header=None, names=["chr", "ps", "rs", "p"], dtype={"chr": int, "rs": str, "p": float})
    assert len(d) == 631829
    auto = d.chr.between(1, 22)
    i = int(d.p.values.argmin()); ia = int(d.p[auto].idxmin())
    return dict(component=k, gwas_top_chr=int(d.chr.iat[i]), gwas_min_p_check=float(d.p.iat[i]),
                gwas_min_p_autosomal=float(d.p[ia]), gwas_top_chr_autosomal=int(d.chr[ia]),
                gwas_n_below_5e8_autosomal=int((d.p[auto] < 5e-8).sum()), gwas_n_below_5e8_nonautosomal=int((d.p[~auto] < 5e-8).sum()))

if __name__ == "__main__":
    with Pool(4) as pool:
        rows = pool.map(one, todo)
    G = pd.DataFrame(rows).set_index("component")
    assert np.allclose(G.gwas_min_p_check, M.loc[G.index, "gwas_min_p"], rtol=1e-6)
    G.drop(columns="gwas_min_p_check").to_csv(f"{OUT}/01b_gwas_autosomal.csv")
    print(f"components read: {len(G)}; top SNP not autosomal in {int((~G.gwas_top_chr.between(1, 22)).sum())}")
    print(G.loc[[k for k in [74, 21, 351, 2, 87, 177] if k in G.index]].to_string())
    th = 5e-8 / 547
    print("study-wide (5e-8/547): overall", int((M.gwas_min_p < th).sum()), "| autosomal", int((G.gwas_min_p_autosomal < th).sum()))
