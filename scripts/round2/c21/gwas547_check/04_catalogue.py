"""Catalogue context: per-component min p_score over the 546 archived GEMMA runs (assoc last column),
abbr-vs-assoc identity for all 546, threshold counts, where C547/C548 fall, phenotype shape, lambda_GC."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import stats

ARC = f"{cfg.SNP}/gemma/output"
SNP = cfg.SNP
G547 = f"{cfg.AUD}/gwas547"   # GEMMA working folder (01_build_fam.py, run_gemma.sh)
OUT = cfg.out_dir(cfg.AUD, "gwas547_verify")
THR = (1e-5, 5e-8, 1e-10, 1e-15, 1e-20, 1e-30)


def summarise_assoc(path):
    with open(path, "rb") as fh:
        hdr = fh.readline().decode().rstrip("\n").split("\t")
        lines = fh.read().decode().split("\n")
    if lines[-1] == "":
        lines.pop()
    assert hdr[-1] == "p_score" and hdr[1] == "rs"
    last = [l.rsplit("\t", 1)[1] for l in lines]
    rs = [l.split("\t", 2)[1] for l in lines]
    p = np.array(last, dtype=float)
    return hdr, lines, last, rs, p


def worker(k):
    hdr, lines, last, rs, p = summarise_assoc(f"{ARC}/results{k}.assoc.txt")
    with open(f"{ARC}/results{k}.abbr.txt") as fh:
        ab = fh.read().split("\n")
    if ab[-1] == "":
        ab.pop()
    abp = [l.rsplit(" ", 1)[1] for l in ab]
    # ros5 format check on every row: chr ps rs p
    ros5 = [f"{t[0]} {t[2]} {t[1]} {t[-1]}" for t in (l.split("\t") for l in lines)]
    j = int(np.argmin(p))
    abf = np.array(abp, dtype=float)
    chi = stats.chi2.isf(p, 1)
    return dict(component=k, n_rows=len(p), n_nan=int(np.isnan(p).sum()), min_p=float(p[j]), min_p_str=last[j], top_rs=rs[j],
                abbr_rows=len(ab), abbr_p_identical=(abp == last), abbr_rows_identical_to_ros5=(ab == ros5),
                abbr_n_gt15=int((-np.log10(abf) > 15).sum()),
                lambda_gc=float(np.median(chi) / stats.chi2.ppf(0.5, 1)),
                **{f"n_below_{t:g}": int((p < t).sum()) for t in THR})


if __name__ == "__main__":
    with Pool(int(cfg.THREADS)) as pool:
        rows = pool.map(worker, range(1, 547), chunksize=4)
    cat = pd.DataFrame(rows).set_index("component")
    print("components read:", len(cat), "| rows per file:", sorted(cat.n_rows.unique()), "| any NaN:", int(cat.n_nan.sum()))
    print("abbr p column string-identical to assoc p_score: ", int(cat.abbr_p_identical.sum()), "/ 546")
    print("abbr file byte-equal to ros5 transform of assoc:  ", int(cat.abbr_rows_identical_to_ros5.sum()), "/ 546")
    print("ros8 criterion (-log10 abbr p > 15) components:", int((cat.abbr_n_gt15 > 0).sum()),
          "| equals p_score<1e-15 set:", set(cat.index[cat.abbr_n_gt15 > 0]) == set(cat.index[cat["n_below_1e-15"] > 0]))

    # phenotype shape from the archived .fam (228 genotyped donors) and for the new components from geno547.fam
    arch = [l.split() for l in open(f"{SNP}/ROSMAP_arrayGenotype_maf.fam")]
    Y = np.array([[float(x) for x in r[5:]] for r in arch])  # 228 x 546
    new = [l.split() for l in open(f"{G547}/geno547.fam")]
    Yn = np.array([[float(r[8]), float(r[9])] for r in new])

    def shape(v):
        z = (v - v.mean()) / v.std()
        return stats.kurtosis(v, fisher=True, bias=True), stats.kurtosis(v, fisher=False, bias=True), np.abs(z).max()

    sh = np.array([shape(Y[:, k - 1]) for k in cat.index])
    cat["kurt_excess"], cat["kurt_pearson"], cat["max_abs_z"] = sh[:, 0], sh[:, 1], sh[:, 2]

    newrows = {}
    for comp, col in (("C547", 0), ("C548", 1)):
        hdr, lines, last, rs, p = summarise_assoc(f"{G547}/output/local_{comp}.assoc.txt")
        j = int(np.argmin(p))
        ke, kp, mz = shape(Yn[:, col])
        chi = stats.chi2.isf(p, 1)
        with open(f"{G547}/output/local_{comp}.abbr.txt") as fh:
            ab = fh.read().split("\n")
        if ab[-1] == "":
            ab.pop()
        ros5 = [f"{t[0]} {t[2]} {t[1]} {t[-1]}" for t in (l.split("\t") for l in lines)]
        o = np.argsort(p, kind="stable")[:6]
        tops = []
        for i in o:
            t = lines[i].split("\t")
            tops.append(f"{t[1]} chr{t[0]}:{t[2]} af {t[6]} p={t[-1]}")
        newrows[comp] = dict(min_p=float(p[j]), min_p_str=last[j], top_rs=rs[j], n_rows=len(p),
                             lambda_gc=float(np.median(chi) / stats.chi2.ppf(0.5, 1)), kurt_excess=ke, kurt_pearson=kp, max_abs_z=mz,
                             abbr_equal_ros5=(ab == ros5), **{f"n_below_{t:g}": int((p < t).sum()) for t in THR})
        print(f"\n{comp}: rows {len(p)}; min p_score {last[j]} ({rs[j]}); counts " +
              ", ".join(f"<{t:g}: {int((p < t).sum())}" for t in THR) +
              f"; lambda_GC {newrows[comp]['lambda_gc']:.4f}; excess kurtosis {ke:.3f} (Pearson {kp:.3f}); max|z| {mz:.3f}; "
              f"abbr file == ros5 transform: {ab == ros5}")
        print("   lowest 6:", " | ".join(tops))
        nsm = int((cat.min_p < p[j]).sum()); neq = int((cat.min_p == p[j]).sum())
        print(f"   archived components with smaller min p: {nsm} (ties {neq}) -> rank {nsm + 1}/547; stronger than {546 - nsm - neq} of 546")

    mp = cat.min_p.values
    print("\nArchived 546: smallest", f"{mp.min():.4g} (C{cat.min_p.idxmin()})", "largest", f"{mp.max():.4g} (C{cat.min_p.idxmax()})")
    for q in (5, 25, 50, 75, 95):
        print(f"  {q}th percentile (numpy linear): {np.percentile(mp, q):.3g}")
    for t in THR:
        print(f"  components with min p < {t:g}: {int((mp < t).sum())}")
    passing = cat[cat.min_p < 1e-15]
    print("  total SNP rows with p_score < 1e-15 across 546:", int(cat["n_below_1e-15"].sum()))
    print("  passing list (C):", " ".join(str(k) for k in passing.index))
    nonp = cat[cat.min_p >= 1e-15]
    print(f"  non-passing: {len(nonp)}; median min p {np.median(nonp.min_p):.3g}")
    lp = -np.log10(cat.min_p)
    print(f"  Spearman -log10 min p vs excess kurtosis {stats.spearmanr(lp, cat.kurt_excess).statistic:.3f}; "
          f"vs max|z| {stats.spearmanr(lp, cat.max_abs_z).statistic:.3f}")
    print(f"  passing: median excess kurtosis {passing.kurt_excess.median():.1f} (Pearson {passing.kurt_pearson.median():.1f}); "
          f"median max|z| {passing.max_abs_z.median():.2f}; sqrt(227) = {np.sqrt(227):.3f}")
    print(f"  passing with excess kurtosis > 10: {int((passing.kurt_excess > 10).sum())}; those <= 10: "
          f"{list(passing.index[passing.kurt_excess <= 10])} (Pearson > 10: {int((passing.kurt_pearson > 10).sum())})")
    print(f"  passing with >=100 SNPs < 5e-8: {int((passing['n_below_5e-08'] >= 100).sum())}; median count {passing['n_below_5e-08'].median():.0f}")
    print(f"  non-passing: median excess kurtosis {nonp.kurt_excess.median():.2f}, median max|z| {nonp.max_abs_z.median():.2f}")
    print(f"  archived C268 (r=0.87 with C547): min p {cat.loc[268, 'min_p']:.3g}, excess kurtosis {cat.loc[268, 'kurt_excess']:.2f}, max|z| {cat.loc[268, 'max_abs_z']:.2f}")
    print(f"  archived lambda_GC range {cat.lambda_gc.min():.3f}..{cat.lambda_gc.max():.3f}, median {cat.lambda_gc.median():.3f}")
    cat.drop(columns=["abbr_p_identical", "abbr_rows_identical_to_ros5"]).assign(
        abbr_p_identical=cat.abbr_p_identical, abbr_eq_ros5=cat.abbr_rows_identical_to_ros5).to_csv(f"{OUT}/04_catalogue_546.csv")
    pd.DataFrame(newrows).T.to_csv(f"{OUT}/04_new_components.csv")
