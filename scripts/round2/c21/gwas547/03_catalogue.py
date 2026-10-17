"""Per-component min p_score etc. across the 546 archived GEMMA runs (read-only).

Uses results{k}.abbr.txt (chr ps rs p_score; written by ros5 from assoc's last
column). Checks every abbr has 631,829 rows, and for a subset (C21/C74/C351 and
every component passing 1e-15) that the abbr p column equals assoc p_score
byte-for-byte.
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool

ARC = f"{cfg.SNP}/gemma/output"
OUT = cfg.out_dir(cfg.AUD, "gwas547")


def one(k):
    d = pd.read_csv(f"{ARC}/results{k}.abbr.txt", sep=" ", header=None,
                    names=["chr", "ps", "rs", "p"], dtype={"rs": str, "p": float})
    p = d["p"].to_numpy()
    i = int(np.argmin(p))
    return dict(component=k, n_rows=len(d), min_p_score=p[i], top_rs=d["rs"].iat[i],
                top_chr=d["chr"].iat[i], top_ps=d["ps"].iat[i],
                n_below_1e15=int((p < 1e-15).sum()), n_below_5e8=int((p < 5e-8).sum()),
                n_nan=int(np.isnan(p).sum()))


def check_abbr_vs_assoc(k):
    a = pd.read_csv(f"{ARC}/results{k}.assoc.txt", sep="\t", dtype=str, usecols=["rs", "p_score"])
    b = pd.read_csv(f"{ARC}/results{k}.abbr.txt", sep=" ", header=None, dtype=str, names=["chr", "ps", "rs", "p"])
    return k, bool((a["rs"].values == b["rs"].values).all() and (a["p_score"].values == b["p"].values).all())


if __name__ == "__main__":
    with Pool(int(cfg.THREADS)) as pool:
        rows = pool.map(one, range(1, 547))
    df = pd.DataFrame(rows).sort_values("component")
    df.to_csv(f"{OUT}/catalogue_546_min_pscore.csv", index=False)
    print("row counts:", df["n_rows"].value_counts().to_dict(), "| NaN p:", int(df["n_nan"].sum()))
    passed = df.loc[df["min_p_score"] < 1e-15, "component"].tolist()
    chk = sorted(set([21, 74, 351] + passed))
    with Pool(int(cfg.THREADS)) as pool:
        res = pool.map(check_abbr_vs_assoc, chk)
    print("abbr p == assoc p_score (byte) for", len(res), "components:", all(r for _, r in res),
          [k for k, r in res if not r])
    print("done")
