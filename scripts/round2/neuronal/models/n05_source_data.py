"""(1) Source-data views for C172 and C277: measured snRNA-seq pseudobulk values (SDA input, CPM) of the top gene and a
loading-weighted aggregate of the component's genes, against the sample score and against the phenotype.
  C172: LRP8 in inhibitory neurons (top |loading| gene; Inh = dominant cell type); second gene SYN3 (top opposite sign)
  C277: PLCB1 in excitatory neurons (top |loading| gene; Exc = dominant cell type); second gene LRP1B
Aggregate: top 50 PIP > 0.5 genes by |loading| in the dominant cell type, each z-standardised over the analysed donors,
weighted by gene loading x corrected cell-type loading (the source_panels definition), so that the aggregate rises with
the score by construction of sign (the sign of an SDA loading is arbitrary; only products are interpretable).
The two snRNA samples that are another person's material (identity_checks RNA-1, RNA-2) are excluded: n = 274.
Phenotype tests on measured values: rank-INT OLS with M1 covariates (age, PMI, cohort, sex); C172 vs CERAD AD/non-AD
(n = 274) and C277 vs cogdx (NCI/MCI/AD dementia, n = 252). Per-donor arrays cached locally for the draft panels."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss
from nc_common import *
from source_panels_bridge import feature_values_by_name, rna_block, loadings

S = scores547(); D = full_frame(); don = list(S.columns)
excl = set(rna_mismatch_donors()); keep = np.array([d not in excl for d in don])
COV = ["age", "pmi", "ros", "sex"]
SPEC = {172: dict(ct="Inh", genes=["LRP8", "SYN3"], ph="CERAD"), 277: dict(ct="Exc", genes=["PLCB1", "LRP1B"], ph="cogdx")}
def robust_z(x):
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826; return (x - med) / mad

rows, prow, cache = [], [], {}
for k, sp in SPEC.items():
    X, P, feats, Bc = loadings(k); ct = sp["ct"]
    idx = np.where(P > 0.5)[0]; idx = idx[np.argsort(-np.abs(X[idx]))][:50]
    M = rna_block(ct, idx)[keep]
    mu, sd = M.mean(0), M.std(0); ok = sd > 0
    Z = (M[:, ok] - mu[ok]) / sd[ok]; w = X[idx][ok] * Bc[ct]
    agg = Z @ w / np.abs(w).sum()
    a = S.loc[k].values.astype(float)[keep]; ext = int(np.argmax(np.abs(robust_z(a))))
    views = {g: feature_values_by_name(g, ct)[keep] for g in sp["genes"]}; views["aggregate"] = agg
    n_pos = int(((X[idx][ok] * Bc[ct]) > 0).sum())
    print(f"C{k}: dominant cell type {ct} (corrected loading {Bc[ct]:+.2f}; next {sorted(Bc.items(), key=lambda t: -abs(t[1]))[1]}); "
          f"PIP>0.5 genes {int((P > 0.5).sum())}; aggregate uses {int(ok.sum())} genes ({n_pos} rising with the score, {int(ok.sum()) - n_pos} falling)")
    for g in sp["genes"]:
        j = list(feats).index(g); print(f"   {g}: gene loading {X[j]:+.3g}, PIP {P[j]:.2f}, product with {ct} loading {X[j] * Bc[ct]:+.3g} -> "
                                        f"{'rises' if X[j] * Bc[ct] > 0 else 'falls'} with the score; mean {views[g].mean():.1f} CPM")
    for v, y in views.items():
        r = ss.spearmanr(y, a); pr = ss.pearsonr(y, a); m = np.ones(len(a), bool); m[ext] = False
        r1 = ss.spearmanr(y[m], a[m])
        rows.append(dict(component=k, view=v, cell_type=ct, n=len(a), rho=r[0], p_rho=r[1], r=pr[0], rho_without_most_extreme=r1[0]))
    # phenotype on measured values
    dk = [d for d in don if d not in excl]
    if sp["ph"] == "CERAD":
        sub = dk; terms = ["cerad_ad"]
    else:
        sub = [d for d in dk if D.loc[d, "cogdx"] in (1, 2, 4)]; terms = ["cogdx_mci", "cogdx_ad"]
    pos = [dk.index(d) for d in sub]
    for v, y in list(views.items()) + [("score", a)]:
        yr = rint(np.asarray(y)[pos])
        T, _, _ = ols(yr, D.loc[sub, terms + COV].values.astype(float), terms + COV)
        for t in terms:
            prow.append(dict(component=k, view=v, phenotype=sp["ph"], term=t, n=len(sub), beta_SD=T.loc[t, "beta"], lo=T.loc[t, "lo"], hi=T.loc[t, "hi"], p=T.loc[t, "p"]))
    cache[k] = dict(a=a, ext=ext, views=views, donors_idx=np.where(keep)[0], ct=ct, n_agg=int(ok.sum()))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/n05_source_vs_score.csv", index=False, float_format="%.4g")
Pt = pd.DataFrame(prow); Pt.to_csv(f"{OUT}/n05_source_vs_phenotype.csv", index=False, float_format="%.4g")
np.save(f"{CACHE}/n05_views.npy", cache, allow_pickle=True)
pd.set_option("display.width", 200)
print("\nsource data vs score (RNA-mismatched donors excluded):"); print(R.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
print("\nsource data vs phenotype (rank-INT OLS, M1; beta in SD of the measured value):"); print(Pt.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
