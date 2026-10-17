"""How many of the 547 components (and of the 11 prioritised) have sample scores dominated by a single donor?
Sample scores: samples_final.csv (historical 1..546; row 520 dropped) + C547/C548 consensus A (2026-10-08 audit).
Metrics per component: top donor's share of sum of squares, participation-ratio n_eff, robust z of top donor,
excess kurtosis. Aggregates only.
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import json, os
import numpy as np, pandas as pd
from scipy.stats import kurtosis

S = cfg.SDA
C547 = f"{cfg.AUD}/c547"
OUT = cfg.out_dir(cfg.WORK, "r1_numbers", "out")
A = pd.read_csv(f"{S}/samples_final.csv", index_col=0).drop(index=520)
for k in (547, 548):
    A.loc[k] = np.load(f"{C547}/C{k}_A.npy")
assert len(A) == 547
counts = pd.read_csv(os.path.join(OUT, "pip_counts_547.csv"), index_col=0)
eleven = [2, 74, 76, 87, 99, 130, 177, 291, 345, 351, 416]
rows = {}
for k, a in A.iterrows():
    a = a.values.astype(float); s2 = a ** 2; tot = s2.sum()
    med = np.median(a); mad = 1.4826 * np.median(np.abs(a - med))
    i = int(np.argmax(np.abs(a)))
    rows[k] = {"top_share_SS": s2[i] / tot, "n_eff": tot ** 2 / np.sum(a ** 4),
               "top_robust_z": abs(a[i] - med) / mad if mad > 0 else np.inf,
               "excess_kurtosis": kurtosis(a, fisher=True, bias=True)}
D = pd.DataFrame(rows).T
D["omics_count"] = counts["omics_count_recomputed"]
D.to_csv(os.path.join(OUT, "donor_dominance_547.csv"))
def summ(sub):
    return {"n": int(len(sub)), "top_share>0.5": int((sub.top_share_SS > 0.5).sum()),
            "top_share>0.8": int((sub.top_share_SS > 0.8).sum()),
            "n_eff<5": int((sub.n_eff < 5).sum()), "n_eff<20": int((sub.n_eff < 20).sum()),
            "excess_kurtosis>10": int((sub.excess_kurtosis > 10).sum()),
            "median_n_eff": float(sub.n_eff.median())}
res = {"all_547": summ(D), "omics>=3 (61)": summ(D[D.omics_count >= 3]), "omics>=2": summ(D[D.omics_count >= 2]),
       "omics==1": summ(D[D.omics_count == 1]), "the_11": summ(D.loc[eleven]),
       "the_11_detail": {int(k): {"top_share_SS": round(float(D.loc[k, "top_share_SS"]), 3),
                                  "n_eff": round(float(D.loc[k, "n_eff"]), 1),
                                  "top_robust_z": round(float(D.loc[k, "top_robust_z"]), 1)} for k in eleven}}
json.dump(res, open(os.path.join(OUT, "donor_dominance_summary.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
