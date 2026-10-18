"""Point 5(b-i): does 450K probe density explain the C74 RNA-gene / CpG-gene overlap (published 2.6-fold, 14 of 38)?

Step 0 reproduces the published test exactly (round-1 revision script 04_c74_cpg_followthrough.py, Q2): deposit sidecars
($G_ROS_DEPOSIT_DIR/{meth,RNA}_features.csv) + the SDA PIP rows of component 74 (pip_final_{RNA,meth}.csv,
byte-identical to the archive). A = C74 RNA PIP > 0.5 autosomal genes in the universe (38); B = genes annotated (RefGene,
split on ';') to C74 PIP > 0.5 CpGs, in the universe (1,296); U = RNA genes x genes annotated to any SDA CpG, autosomal
(9,192, as published: it includes 114 genes without a sidecar chromosome, 111 autosomal by stable ID and 3 unresolved;
05 and r5 use the 9,189 confirmed chromosome 1-22 genes). Statistic: k = |A n B| (14). A is held fixed; nulls resample B (or A) so that probe density is preserved.

Density per gene: d_sda = number of the 51,798 SDA CpGs annotated to the gene (the sampling frame of B);
                  d_450 = number of the 420,132 manifest probes annotated to the gene (array CpG count).
Nulls (all one-sided, statistic k; add-one empirical p; B and seed printed):
  N0  uniform gene permutation of B (sanity: = hypergeometric)
  N1  B stratified by d_sda bins (exact: convolution of per-bin hypergeometrics; also sampled for calibration report)
  N1b B stratified by d_450 decile bins (exact)
  N1c A stratified by d_sda bins (density-matched RNA gene sets vs fixed B; exact)
  N2  weighted gene resampling without replacement (Efraimidis-Spirakis), |B*| = 1,296, weights d_450 / d_sda / PWF
  N3  CpG resampling: random order of the 51,798 SDA CpGs; (a) first 3,435 CpGs; (b) CpGs until 1,296 universe genes
      are hit (size-matched). Each gene's hit probability then follows its own CpG count, as for the observed set.
  PB  gometh-style: PWF = isotonic P(g in B | log d_sda) [or log d_450] over U; exact Poisson-binomial P(k >= 14) over A.
Calibration: (i) size match and frac-with-hit for every resampling null (nullcal.py); (ii) type-I check: 1,000 random
CpG sets of 3,435 (uniform from the SDA CpGs; density only, no biology) tested against the real A with the uncorrected
hypergeometric and with N1 / PB; (iii) positive control: spike s of A's genes into density-matched null B sets and find
the s at which N1 detects at p < 0.05 in 80% / 100% of 500 spikes; detection thresholds in overlap units.
Outputs: out/02_* (aggregates and gene-level counts only; no donor data)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, json
os.environ.setdefault("OMP_NUM_THREADS", cfg.THREADS); os.environ.setdefault("OPENBLAS_NUM_THREADS", cfg.THREADS); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", cfg.THREADS)
import numpy as np, pandas as pd
from scipy import stats as ss, sparse
from scipy.optimize import isotonic_regression
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nullcal

SDA = cfg.SDA
SYN = cfg.DEPOSIT   # deposit feature sidecars (meth/chip identical in V1 and V2; RNA chrom column identical)
MAN = f"{cfg.REFS}/ROSMAP_arrayMethylation_metaData.tsv"
OUT = cfg.out_dir(cfg.WORK, "v23_review", "point_5_analysis", "out")
SEED = 20261014
rng = np.random.default_rng(SEED)
K = 74


def pip_row(fname, k=K):
    with open(f"{SDA}/{fname}") as f:
        hdr = f.readline().rstrip("\n").split(",")[1:]
        for line in f:
            key, rest = line.split(",", 1)
            if key == str(k):
                return pd.Series(np.array(rest.rstrip("\n").split(","), dtype=float), index=hdr)


# ---------------- step 0: reproduce the published test ----------------
meth_feat = pd.read_csv(f"{SYN}/meth_features.csv", index_col=0)
rna_feat = pd.read_csv(f"{SYN}/RNA_features.csv", index_col=0)
pm = pip_row("pip_final_meth.csv"); pr = pip_row("pip_final_RNA.csv")
assert list(pm.index) == list(meth_feat.index) and len(pm) == 51798
assert set(pr.index) == set(rna_feat.index) and len(pr) == 25098
meth_feat["chrom_str"] = meth_feat.CHR.astype(str).str.replace("chr", "", regex=False)
assert not meth_feat.chrom_str.isin(["X", "Y"]).any()


def cpg_genes(ids):
    s = meth_feat.loc[ids, "RefGene"].dropna().astype(str)
    return {i: sorted({x.strip() for x in v.split(";") if x.strip()}) for i, v in s.items()}


G_all = cpg_genes(meth_feat.index)
annot = set(g for v in G_all.values() for g in v)
rna_feat["chrom_str"] = rna_feat.chrom.astype(str).str.replace("chr", "", regex=False)
auto = set(rna_feat.index[~rna_feat.chrom_str.isin(["X", "Y", "MT", "M"])])
U = sorted((set(pr.index) & annot) & auto)
c74_cpgs = list(pm.index[pm > 0.5])
B = sorted(set(g for c in c74_cpgs for g in G_all.get(c, [])) & set(U))
A = sorted((set(pr.index[pr > 0.5]) & auto) & set(U))
kobs = len(set(A) & set(B)); nU, nA, nB = len(U), len(A), len(B)
exp0 = nA * nB / nU; p0 = ss.hypergeom.sf(kobs - 1, nU, nB, nA)
print(f"[0] reproduce: U {nU}, A (RNA) {nA}, B (CpG genes) {nB}, C74 CpGs {len(c74_cpgs)}, overlap {kobs}, "
      f"expected {exp0:.2f}, fold {kobs/exp0:.2f}, hypergeometric p {p0:.3g}   (published 9192/38/1296/14, 5.36, 2.61, 4.0e-4)")
assert (nU, nA, nB, kobs) == (9192, 38, 1296, 14)

ui = {g: i for i, g in enumerate(U)}
inA = np.zeros(nU, bool); inA[[ui[g] for g in A]] = True
inB = np.zeros(nU, bool); inB[[ui[g] for g in B]] = True
# CpG x universe-gene incidence (SDA CpGs)
cp_idx = {c: i for i, c in enumerate(meth_feat.index)}
r_, c_ = [], []
for c, gl in G_all.items():
    for g in gl:
        if g in ui: r_.append(cp_idx[c]); c_.append(ui[g])
CG = sparse.csc_matrix((np.ones(len(r_), np.int8), (r_, c_)), shape=(len(meth_feat), nU))
d_sda = np.asarray((CG > 0).sum(0)).ravel()
assert (d_sda >= 1).all()
# array CpG count (whole 420,132-probe manifest)
man = pd.read_csv(MAN, sep="\t", index_col=0, usecols=["TargetID", "RefGene"])
cnt = {}
for v in man.RefGene.dropna().astype(str):
    for g in {x.strip() for x in v.split(";") if x.strip()}:
        cnt[g] = cnt.get(g, 0) + 1
d_450 = np.array([cnt.get(g, 0) for g in U])
print(f"    universe genes with 0 manifest probes under this split: {(d_450 == 0).sum()} (should be 0)")
assert (d_450 >= d_sda).all()

# ---------------- descriptive: is either side density-biased? ----------------
desc = {}
for name, d in [("d_sda", d_sda), ("d_450", d_450)]:
    mw_A = ss.mannwhitneyu(d[inA], d[~inA], alternative="two-sided").pvalue
    mw_B = ss.mannwhitneyu(d[inB], d[~inB], alternative="two-sided").pvalue
    desc[name] = dict(U_mean=d.mean(), U_median=float(np.median(d)), A_mean=d[inA].mean(), A_median=float(np.median(d[inA])),
                      B_mean=d[inB].mean(), B_median=float(np.median(d[inB])), AB_mean=d[inA & inB].mean(),
                      AB_median=float(np.median(d[inA & inB])), A_notB_mean=d[inA & ~inB].mean(),
                      MW_A_vs_rest_p=mw_A, MW_B_vs_rest_p=mw_B)
    print(f"[desc] {name}: universe mean {d.mean():.2f} (median {np.median(d):.0f}); RNA A mean {d[inA].mean():.2f} "
          f"(median {np.median(d[inA]):.0f}; MW vs rest p {mw_A:.3g}); CpG-gene B mean {d[inB].mean():.2f} (median {np.median(d[inB]):.0f}; "
          f"MW p {mw_B:.3g}); A&B (the 14) mean {d[inA & inB].mean():.2f} (median {np.median(d[inA & inB]):.0f}); A not B mean {d[inA & ~inB].mean():.2f}")
# hit rate of B by d_sda bin
bins_sda = [1, 2, 3, 4, 5, 6, 8, 11, 16, 26, 10**6]
bsda = np.digitize(d_sda, bins_sda) - 1
tab = pd.DataFrame({"bin": [f"{bins_sda[i]}-{bins_sda[i+1]-1}" for i in range(len(bins_sda) - 1)],
                    "U": np.bincount(bsda, minlength=10), "B": np.bincount(bsda, weights=inB, minlength=10).astype(int),
                    "A": np.bincount(bsda, weights=inA, minlength=10).astype(int),
                    "AB": np.bincount(bsda, weights=inA & inB, minlength=10).astype(int)})
tab["P(B|bin)"] = tab.B / tab.U
print("[desc] by d_sda bin:\n" + tab.to_string(index=False))
tab.to_csv(f"{OUT}/02_density_bins_sda.csv", index=False)
q450 = np.unique(np.quantile(d_450, np.linspace(0, 1, 11)))
b450 = np.clip(np.searchsorted(q450, d_450, side="right") - 1, 0, len(q450) - 2)
gene_tab = pd.DataFrame({"gene": U, "in_A_RNA": inA, "in_B_CpG": inB, "d_sda": d_sda, "d_450": d_450})
gene_tab[gene_tab.in_A_RNA].sort_values(["in_B_CpG", "d_sda"], ascending=False).to_csv(f"{OUT}/02_rna_genes_density.csv", index=False)
print("[desc] the 14 overlap genes (d_sda, d_450):",
      ", ".join(f"{g} ({a},{b})" for g, a, b in gene_tab[gene_tab.in_A_RNA & gene_tab.in_B_CpG].sort_values("d_sda", ascending=False)[["gene", "d_sda", "d_450"]].values))


# ---------------- exact stratified nulls ----------------
def conv_hyper(Ub, Ab, nb):
    """Exact distribution of sum_b Hypergeom(Ub, Ab, nb) (independent strata)."""
    dist = np.array([1.0])
    for u, a, n in zip(Ub, Ab, nb):
        if a == 0 or n == 0: continue
        x = np.arange(0, min(a, n) + 1); pm_ = ss.hypergeom.pmf(x, u, a, n)
        dist = np.convolve(dist, pm_)
    return dist


def strat_exact(strata, sample_mask, fixed_mask, k):
    """Sample side = sample_mask (its per-stratum counts are preserved); fixed side = fixed_mask."""
    S = np.unique(strata)
    Ub = [int((strata == s).sum()) for s in S]; Ab = [int((fixed_mask & (strata == s)).sum()) for s in S]
    nb = [int((sample_mask & (strata == s)).sum()) for s in S]
    dist = conv_hyper(Ub, Ab, nb); x = np.arange(len(dist))
    mean = float((x * dist).sum()); p = float(dist[k:].sum()) if k < len(dist) else 0.0
    q95 = int(x[np.searchsorted(np.cumsum(dist), 0.95)]); q999 = int(x[np.searchsorted(np.cumsum(dist), 0.999)])
    return dict(mean=mean, fold=k / mean, p=p, q95=q95, q999=q999), dist


res = {}
res["hypergeometric (published)"] = dict(mean=exp0, fold=kobs / exp0, p=p0,
                                         q95=int(ss.hypergeom.ppf(0.95, nU, nB, nA)), q999=int(ss.hypergeom.ppf(0.999, nU, nB, nA)))
res["N1 B stratified by d_sda bins (exact)"], dist_N1 = strat_exact(bsda, inB, inA, kobs)
res["N1b B stratified by d_450 deciles (exact)"], _ = strat_exact(b450, inB, inA, kobs)
res["N1c A stratified by d_sda bins (exact)"], _ = strat_exact(bsda, inA, inB, kobs)
# finer: exact d_sda value up to 30, then pooled
fine = np.minimum(d_sda, 30)
res["N1d B stratified by exact d_sda (1..29, 30+) (exact)"], _ = strat_exact(fine, inB, inA, kobs)


# ---------------- PWF / Poisson-binomial ----------------
def pwf(hit, d):
    o = np.argsort(np.log(d), kind="mergesort")
    fit = isotonic_regression(hit[o].astype(float), increasing=True).x
    p = np.empty(len(d)); p[o] = fit
    return np.clip(p, 1e-9, 1 - 1e-9)


def poibin_sf(ps, k):
    dist = np.array([1.0])
    for p in ps: dist = np.convolve(dist, [1 - p, p])
    return float(dist[k:].sum()), float(np.sum(np.arange(len(dist)) * dist)), dist


for name, d in [("d_sda", d_sda), ("d_450", d_450)]:
    w = pwf(inB, d); pval, mean, dist = poibin_sf(w[inA], kobs); cdf = np.cumsum(dist)
    res[f"PB gometh-style PWF({name}), Poisson-binomial (exact)"] = dict(mean=mean, fold=kobs / mean, p=pval,
                                                                          q95=int(np.searchsorted(cdf, 0.95)), q999=int(np.searchsorted(cdf, 0.999)))
PWF_sda = pwf(inB, d_sda)


# ---------------- weighted gene resampling (Efraimidis-Spirakis) ----------------
def es_null(weights, n, ndraw, chunk=2000):
    lw = np.log(weights); out = np.empty(ndraw, int)
    for s in range(0, ndraw, chunk):
        m = min(chunk, ndraw - s)
        keys = np.log(rng.random((m, nU))) / weights[None, :]          # log(u)/w: larger = selected
        thr = np.partition(keys, nU - n, axis=1)[:, nU - n][:, None]
        out[s:s + m] = (keys[:, inA] >= thr).sum(1)
    return out


NDRAW = 100_000
cal = {}
for name, w in [("d_450 (array CpG count)", d_450.astype(float)), ("d_sda (SDA CpG count)", d_sda.astype(float)),
                ("PWF(d_sda)", PWF_sda), ("uniform (N0)", np.ones(nU))]:
    nd = es_null(w, nB, NDRAW)
    rep = nullcal.null_calibration_report(kobs, nd.tolist(), obs_size=nB, null_sizes=[nB] * 3, label=f"N2 weighted genes {name}")
    key = f"N2 weighted gene resampling, {name}" if not name.startswith("uniform") else "N0 uniform gene permutation"
    res[key] = dict(mean=float(nd.mean()), fold=kobs / nd.mean(), p=rep["p_empirical"], q95=int(np.quantile(nd, 0.95)),
                    q999=int(np.quantile(nd, 0.999)), B=NDRAW, frac_draws_with_hit=rep["frac_draws_with_hit"], verdict=rep["verdict"])
    cal[key] = nullcal.format_calibration_report(rep)


# ---------------- CpG resampling (first-hit ranks) ----------------
indptr, indices = CG.indptr, CG.indices          # CSC: rows (CpGs) of each gene column
ncpg = CG.shape[0]
def cpg_null(ndraw, nfix=len(c74_cpgs), ntarget=nB):
    k_fix = np.empty(ndraw, int); size_fix = np.empty(ndraw, int); k_sz = np.empty(ndraw, int); ncpg_sz = np.empty(ndraw, int)
    fold_fix = np.empty(ndraw)
    for b in range(ndraw):
        r = rng.permutation(ncpg)
        first = np.minimum.reduceat(r[indices], indptr[:-1])     # each gene's first-hit rank
        hit = first < nfix; size_fix[b] = hit.sum(); k_fix[b] = (hit & inA).sum()
        fold_fix[b] = k_fix[b] / (nA * size_fix[b] / nU)
        thr = np.partition(first, ntarget - 1)[ntarget - 1]
        hs = first <= thr; k_sz[b] = (hs & inA).sum(); ncpg_sz[b] = thr + 1
    return k_fix, size_fix, fold_fix, k_sz, ncpg_sz


NCPG = 20_000
k_fix, size_fix, fold_fix, k_sz, ncpg_sz = cpg_null(NCPG)
obs_fold = kobs / exp0
rep_a = nullcal.null_calibration_report(kobs, k_fix.tolist(), obs_size=nB, null_sizes=size_fix.tolist(), label="N3a CpG resampling, 3,435 CpGs (count k)")
rep_af = nullcal.null_calibration_report(obs_fold, fold_fix.tolist(), obs_size=nB, null_sizes=size_fix.tolist(), hit_threshold=1.0,
                                         label="N3a CpG resampling, 3,435 CpGs (fold k/E_hyper of each draw)")
rep_b = nullcal.null_calibration_report(kobs, k_sz.tolist(), obs_size=nB, null_sizes=[nB] * 3, label="N3b CpG resampling until 1,296 genes (count k)")
res["N3a CpG resampling, 3,435 SDA CpGs (count)"] = dict(mean=float(k_fix.mean()), fold=kobs / k_fix.mean(), p=rep_a["p_empirical"],
    q95=int(np.quantile(k_fix, 0.95)), q999=int(np.quantile(k_fix, 0.999)), B=NCPG, median_genes_hit=float(np.median(size_fix)), verdict=rep_a["verdict"])
res["N3a CpG resampling, 3,435 SDA CpGs (fold vs each draw's hypergeometric E)"] = dict(mean=float(fold_fix.mean()), fold=obs_fold / fold_fix.mean(),
    p=rep_af["p_empirical"], B=NCPG, verdict=rep_af["verdict"])
res["N3b CpG resampling until 1,296 universe genes (count)"] = dict(mean=float(k_sz.mean()), fold=kobs / k_sz.mean(), p=rep_b["p_empirical"],
    q95=int(np.quantile(k_sz, 0.95)), q999=int(np.quantile(k_sz, 0.999)), B=NCPG, median_cpgs_needed=float(np.median(ncpg_sz)), verdict=rep_b["verdict"])
for r_ in (rep_a, rep_af, rep_b):
    cal[r_["label"]] = nullcal.format_calibration_report(r_)
# also sampled version of N1 for the calibration report (frac with hit etc.)
S = np.unique(bsda); Ub = np.array([(bsda == s).sum() for s in S]); Ab = np.array([(inA & (bsda == s)).sum() for s in S])
nb = np.array([(inB & (bsda == s)).sum() for s in S])
n1 = sum(rng.hypergeometric(Ab[i], Ub[i] - Ab[i], nb[i], size=NDRAW) for i in range(len(S)) if Ab[i] > 0 and nb[i] > 0)
rep_n1 = nullcal.null_calibration_report(kobs, n1.tolist(), obs_size=nB, null_sizes=[nB] * 3, label="N1 B stratified by d_sda bins (sampled)")
cal[rep_n1["label"]] = nullcal.format_calibration_report(rep_n1)

print(f"\n[nulls] seed {SEED}; gene nulls B = {NDRAW:,}; CpG nulls B = {NCPG:,}")
R = pd.DataFrame(res).T[["mean", "fold", "p", "q95", "q999"] + [c for c in ["B", "median_genes_hit", "median_cpgs_needed", "frac_draws_with_hit", "verdict"] if c in pd.DataFrame(res).T.columns]]
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(R.to_string())
R.to_csv(f"{OUT}/02_overlap_nulls.csv")
print("\n[calibration]")
for v in cal.values(): print(v)


# ---------------- type-I check on random CpG sets (density only) ----------------
NT = 1000; fp = {"uncorrected hypergeometric": [], "N1 stratified d_sda (exact)": [], "PB PWF(d_sda)": []}; folds = []
for t in range(NT):
    r = rng.permutation(ncpg); first = np.minimum.reduceat(r[indices], indptr[:-1]); Br = first < len(c74_cpgs)
    kr = int((Br & inA).sum()); nbr = int(Br.sum())
    fp["uncorrected hypergeometric"].append(ss.hypergeom.sf(kr - 1, nU, nbr, nA)); folds.append(kr / (nA * nbr / nU))
    fp["N1 stratified d_sda (exact)"].append(strat_exact(bsda, Br, inA, kr)[0]["p"])
    fp["PB PWF(d_sda)"].append(poibin_sf(pwf(Br, d_sda)[inA], kr)[0])
t1 = {k: dict(frac_p_lt_0_05=float(np.mean(np.array(v) < 0.05)), frac_p_le_4e_4=float(np.mean(np.array(v) <= 4.015e-4)),
              median_p=float(np.median(v))) for k, v in fp.items()}
t1["random-set fold (uncorrected), mean / 95th pct"] = dict(mean=float(np.mean(folds)), q95=float(np.quantile(folds, 0.95)),
                                                            frac_ge_obs_fold=float(np.mean(np.array(folds) >= obs_fold)))
print(f"\n[type-I] {NT} random sets of {len(c74_cpgs)} SDA CpGs (median genes hit {np.median(size_fix):.0f}) tested against the real A:")
for k, v in t1.items(): print(f"   {k}: {v}")


# ---------------- positive control (spike-in) for N1 ----------------
pc = []
for s in range(1, 13):
    det = 0
    for rep_ in range(500):
        Bn = np.zeros(nU, bool)
        for i, st in enumerate(S):                               # density-matched null B
            idx = np.where(bsda == st)[0]; Bn[rng.choice(idx, nb[i], replace=False)] = True
        cand = np.where(inA & ~Bn)[0]; add = rng.choice(cand, min(s, len(cand)), replace=False)
        for g in add:                                            # swap in an A gene for a same-bin non-A B gene
            pool = np.where(Bn & ~inA & (bsda == bsda[g]))[0]
            if len(pool): Bn[rng.choice(pool)] = False; Bn[g] = True
        kk = int((Bn & inA).sum()); det += strat_exact(bsda, Bn, inA, kk)[0]["p"] < 0.05
    pc.append(dict(spiked_genes=s, detection_rate=det / 500))
pc = pd.DataFrame(pc); print("\n[positive control] N1 detection (p < 0.05) vs genes spiked into density-matched null B sets:\n" + pc.to_string(index=False))
pc.to_csv(f"{OUT}/02_positive_control.csv", index=False)
json.dump(dict(reproduce=dict(U=nU, A=nA, B=nB, k=kobs, expected=exp0, fold=kobs / exp0, p=p0), desc=desc,
               nulls={k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv) for kk, vv in v.items()} for k, v in res.items()},
               typeI=t1, seed=SEED), open(f"{OUT}/02_summary.json", "w"), indent=1, default=float)
