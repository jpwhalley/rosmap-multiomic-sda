"""C177: is the broad low-count-gene RNA signal explained by sequencing depth / nuclei number?
Partial Spearman of the unfiltered top-50 PIP>0.5 aggregate (per cell type) vs score,
adjusting for log library size and log nuclei of that class, total library size, cohort, PMI.
Also the fraction of donors with zero counts for the top genes. Aggregate output only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats
import importlib.util, sys
from common import *
spec = importlib.util.spec_from_file_location("ana", f"{HERE}/04_analysis.py")
S, R, A, B, T = load()
a = A[176].astype(float); rz = robust_z(a)
r = comp_row(S, 177); X = S["RNA_X"][r]; P = S["RNA_P"][r]; Bk = B.loc[177]
idx = np.where(P > 0.5)[0]; idx = idx[np.argsort(-np.abs(X[idx]))][:50]
tot = np.log(T[[f"rna_libsize_{c}" for c in CTS]].sum(1).values)
rows = []
for ct in CTS:
    M = np.column_stack([feature_values(S, "RNA", j, ct) for j in idx])
    agg, _ = aggregate(M, X[idx] * Bk[ct])
    covs = np.column_stack([np.ones(276), stats.rankdata(np.log(T[f"rna_libsize_{ct}"].values)), stats.rankdata(np.log(T[f"rna_ncells_{ct}"].values)),
                            stats.rankdata(tot), (T.Study.values == "ROS").astype(float), stats.rankdata(T.pmi.values)])
    def res(y):
        yr = stats.rankdata(y); b, *_ = np.linalg.lstsq(covs, yr, rcond=None); return yr - covs @ b
    keep = np.ones(276, bool); keep[np.argsort(-np.abs(rz))[:3]] = False
    pr = stats.pearsonr(res(agg)[keep], res(a)[keep])
    rows.append(dict(cell_type=ct, rho=stats.spearmanr(agg, a)[0], partial_rho_depth_drop1pct=pr[0], p=pr[1],
                     median_frac_donors_zero_top10=float(np.median((M[:, :10] == 0).mean(0)))))
D = pd.DataFrame(rows); print(D.round(4).to_string(index=False))
D.to_csv(f"{ROOT}/results/h_c177_depth_check.csv", index=False)
