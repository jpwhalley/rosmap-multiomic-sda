"""Compare local (macOS/Accelerate) GEMMA 0.96 output with the archived cluster run."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ARC = f"{cfg.SNP}/gemma/output"
OUT = cfg.out_dir(cfg.AUD, "gwas547")
IDCOLS = ["chr", "rs", "ps", "n_miss", "allele1", "allele0", "af"]
NUM = ["beta", "se", "l_remle", "l_mle", "p_wald", "p_lrt", "p_score"]
LOGKEYS = ["number of total individuals", "number of analyzed individuals", "number of covariates",
           "number of phenotypes", "number of total SNPs", "number of analyzed SNPs",
           "REMLE log-likelihood in the null model", "MLE log-likelihood in the null model",
           "pve estimate in the null model", "se(pve) in the null model", "vg estimate in the null model",
           "ve estimate in the null model", "beta estimate in the null model", "se(beta)"]


def logsum(path):
    d = {}
    for line in open(path):
        m = re.match(r"## (.+?) = (.+)$", line.rstrip())
        if m and m.group(1) in LOGKEYS:
            d[m.group(1)] = m.group(2).strip()
    return d


def compare(k):
    print(f"===== C{k} =====")
    a = pd.read_csv(f"{ARC}/results{k}.assoc.txt", sep="\t", dtype=str)
    b = pd.read_csv(f"{OUT}/output/local_C{k}.assoc.txt", sep="\t", dtype=str)
    print("columns identical:", list(a.columns) == list(b.columns), "| rows:", len(a), len(b))
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for c in IDCOLS:
        same = (a[c].values == b[c].values).all()
        print(f"  {c:8s} byte-identical: {same}")
    res = {"component": k, "n_snps_compared": len(a)}
    for c in NUM:
        sa, sb = a[c].values, b[c].values
        nbyte = int((sa == sb).sum())
        fa, fb = sa.astype(float), sb.astype(float)
        ad = np.abs(fa - fb)
        with np.errstate(divide="ignore", invalid="ignore"):
            rd = np.where(fa != 0, ad / np.abs(fa), np.where(fb == 0, 0.0, np.inf))
        line = (f"  {c:8s} identical strings {nbyte}/{len(sa)}  max|diff| {ad.max():.3e}  "
                f"max rel diff {np.nanmax(rd):.3e}  n(rel>1e-6) {(rd > 1e-6).sum()}")
        if c.startswith("p_"):
            ld = np.abs(np.log10(fa) - np.log10(fb))
            line += f"  max|dlog10 p| {ld.max():.3e}"
            res[f"max_abs_dlog10_{c}"] = float(ld.max())
        print(line)
        res[f"max_abs_diff_{c}"] = float(ad.max())
        res[f"max_rel_diff_{c}"] = float(np.nanmax(rd))
        res[f"n_identical_{c}"] = nbyte
    pa, pb = a["p_score"].astype(float).values, b["p_score"].astype(float).values
    res["min_p_score_archived"] = float(pa.min())
    res["min_p_score_local"] = float(pb.min())
    for thr in (1e-15, 5e-8, 1e-5):
        sa_, sb_ = set(a["rs"][pa < thr]), set(b["rs"][pb < thr])
        print(f"  p_score < {thr:g}: archived {len(sa_)} local {len(sb_)} same set: {sa_ == sb_}")
        res[f"same_set_below_{thr:g}"] = sa_ == sb_
        res[f"n_below_{thr:g}"] = (len(sa_), len(sb_))
    oa = np.argsort(pa, kind="stable")[:100]
    ob = np.argsort(pb, kind="stable")[:100]
    # ties in printed p can reorder; compare top lists by rank with tie awareness
    for n in (5, 20, 100):
        print(f"  top-{n} SNPs by p_score same order: {list(a['rs'].values[oa[:n]]) == list(b['rs'].values[ob[:n]])}"
              f"  same set: {set(a['rs'].values[oa[:n]]) == set(b['rs'].values[ob[:n]])}")
    res["top20_same_order"] = list(a["rs"].values[oa[:20]]) == list(b["rs"].values[ob[:20]])
    rho = spearmanr(pa, pb).correlation
    print(f"  Spearman rho p_score (all SNPs): {rho:.12f}")
    print("  top 5 archived / local:")
    for i, j in zip(oa[:5], ob[:5]):
        print(f"    {a['rs'].iat[i]:15s} {a['p_score'].iat[i]}   {b['rs'].iat[j]:15s} {b['p_score'].iat[j]}")
    la = logsum(f"{ARC}/results{k}.log.txt")
    lb = logsum(f"{OUT}/output/local_C{k}.log.txt")
    diffs = [(key, la.get(key), lb.get(key)) for key in LOGKEYS if la.get(key) != lb.get(key)]
    print("  log summary keys compared:", len(LOGKEYS), "| differing:", diffs if diffs else "none")
    res["log_diffs"] = diffs
    return res


if __name__ == "__main__":
    ks = [int(x) for x in sys.argv[1:]] or [21, 74, 351]
    allres = [compare(k) for k in ks]
    pd.DataFrame(allres).to_csv(f"{OUT}/validation_summary.csv", index=False)
