"""Is a positive feature-aggregate vs score correlation among the NON-extreme donors evidence of an axis,
or does it arise by construction (the fitted score of every other donor is a projection of its data on the
component's loadings)? Empirical reference: all single-donor-dominated components (eff. donors < 1.5) in the
546, same aggregate (rank-INT, signed, all PIP>0.5 features), extreme donor removed.
Also: the same quantity for a cross-modal pair (aggregate of modality 1 vs aggregate of modality 2), which
does not share the projection. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); C = cfg.out_dir(cfg.WORK, "verify_source_panels", "cache"); R = cfg.out_dir(cfg.WORK, "verify_source_panels", "results")
A = np.load(f"{C}/A.npy")
IN = {m: np.load(f"{C}/{m}_input.npy") for m in ["meth", "chip", "proteins"]}
X = {m: np.load(f"{C}/{m}_X.npy", mmap_mode="r") for m in IN}
P = {m: np.load(f"{C}/{m}_P.npy", mmap_mode="r") for m in IN}
INT = {}
for m, V in IN.items():
    has = ~np.isnan(V).all(1); Y = V[has]
    Rk = np.apply_along_axis(stats.rankdata, 0, Y); INT[m] = (has, stats.norm.ppf((Rk - 0.5) / Y.shape[0]))

def eff(a):
    d2 = (a - a.mean()) ** 2; return d2.sum() ** 2 / (d2 ** 2).sum()

def rz(x):
    med = np.median(x); return (x - med) / (np.median(np.abs(x - med)) * 1.4826)

rows = []
for k in range(1, 547):
    a = A[k - 1]; ef = eff(a); e = int(np.argmax(np.abs(rz(a))))
    aggs = {}
    for m in IN:
        sel = np.where(np.array(P[m][k - 1]) > 0.5)[0]
        if len(sel) < 5: continue
        has, Z = INT[m]; s = np.sign(np.array(X[m][k - 1])[sel])
        agg = np.full(276, np.nan); agg[has] = Z[:, sel] @ s / len(sel); aggs[m] = agg
        ok = has.copy(); ok[e] = False
        rho = stats.spearmanr(agg[ok], a[ok])[0]
        rows.append(dict(component=k, eff_donors=ef, kind="within", modality=m, n_feats=len(sel), rho_drop_ext=rho))
    ms = list(aggs)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            ok = ~np.isnan(aggs[ms[i]]) & ~np.isnan(aggs[ms[j]]); ok[e] = False
            if ok.sum() < 30: continue
            rows.append(dict(component=k, eff_donors=ef, kind="cross", modality=f"{ms[i]} vs {ms[j]}", n_feats=np.nan,
                             rho_drop_ext=stats.spearmanr(aggs[ms[i]][ok], aggs[ms[j]][ok])[0]))
D = pd.DataFrame(rows); D.to_csv(f"{R}/07_circularity_null.csv", index=False)
sd = D[D.eff_donors < 1.5]
print("single-donor components (eff<1.5) with >=5 PIP features in a 2-D layer:", sd.component.nunique())
for (kind, m), g in sd.groupby(["kind", "modality"]):
    if len(g) < 3: continue
    q = g.rho_drop_ext.quantile([0.05, 0.5, 0.95]).round(3).tolist()
    big = g[g.n_feats >= 100] if kind == "within" else g
    print(f"{kind:6s} {m:22s} n={len(g):3d}  rho (5%,50%,95%) {q}" + (f" | >=100 feats: n={len(big)}, median {big.rho_drop_ext.median():.3f}" if kind == "within" and len(big) else ""))
print("\nC351 / C2 / C87 / C177 rows:")
print(D[D.component.isin([351, 2, 87, 177])].round(3).to_string(index=False))
# relation of within-layer rho to number of PIP features among single-donor components
w = sd[sd.kind == "within"]
print("\nSpearman(n_feats, rho_drop_ext) among single-donor within-layer rows:", stats.spearmanr(w.n_feats, w.rho_drop_ext).round(3) if len(w) > 5 else None)
