"""PO odds ratios (M1) for the paper's contrasts with and without the two donors whose snRNA profile is another
person's (identity_checks RNA-1/RNA-2), for the panel captions. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from nc_common import *
from nc_po import PO
S = scores547(); D = full_frame(); excl = set(rna_mismatch_donors()); COV = ["age", "pmi", "ros", "sex"]
rows = []
for k, terms, base in ((172, ["cerad_ad"], list(D.index)), (172, ["cerad_burden"], list(D.index)),
                       (277, ["cogdx_mci", "cogdx_ad"], list(D.index[D.cogdx.isin([1, 2, 4])]))):
    for lab, sub in (("all", base), ("minus_RNA_mismatch", [d for d in base if d not in excl])):
        y = S.loc[k, sub].values.astype(float); X = D.loc[sub, terms + COV].values.astype(float)
        po = PO(y, X); ll, beta, ok = po.full()
        for j, t in enumerate(terms):
            lo, hi = po.ci_profile(j)
            rows.append(dict(component=k, term=t, sample=lab, n=len(sub), OR=np.exp(beta[j]), OR_lo=np.exp(lo), OR_hi=np.exp(hi), converged=ok))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/n05b_po_sensitivity.csv", index=False, float_format="%.4g")
print(T.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
