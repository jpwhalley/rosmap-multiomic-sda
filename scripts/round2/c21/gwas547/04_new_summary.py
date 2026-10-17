"""Summarise local GEMMA runs for C547/C548 and place them in the 546-run catalogue."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np
import pandas as pd
from scipy.stats import chi2
OUT = cfg.out_dir(cfg.AUD, "gwas547")
# the run read catalogue_546_min_pscore_with_shape.csv (03_catalogue.py's table plus shape columns added in an unsaved
# step); only min_p_score is used here, which 03_catalogue.py writes
cat = pd.read_csv(f"{OUT}/catalogue_546_min_pscore.csv")
rows = []
for k in (547, 548):
    a = pd.read_csv(f"{OUT}/output/local_C{k}.assoc.txt", sep="\t", dtype={"rs": str})
    p = a["p_score"].to_numpy()
    o = np.argsort(p, kind="stable")
    lam = np.median(chi2.isf(p, 1)) / chi2.ppf(0.5, 1)
    print(f"===== C{k} =====  n SNPs {len(a)}  NaN p {np.isnan(p).sum()}  min p_score {p.min():.6e}")
    print(f"  n<1e-15 {(p<1e-15).sum()}  n<5e-8 {(p<5e-8).sum()}  n<1e-5 {(p<1e-5).sum()}  lambda_GC(p_score) {lam:.3f}")
    print(a.iloc[o[:10]][["chr", "rs", "ps", "af", "n_miss", "beta", "se", "p_wald", "p_lrt", "p_score"]].to_string(index=False))
    mp = p.min()
    rank = int((cat["min_p_score"] < mp).sum()) + 1
    print(f"  catalogue: {int((cat['min_p_score'] < mp).sum())} of 546 archived components have smaller min p "
          f"(rank {rank}/547 by min p, smallest=1); percentile of -log10 min p among 546: "
          f"{(cat['min_p_score'] > mp).mean()*100:.1f}%")
    rows.append(dict(component=k, min_p_score=mp, n_below_1e15=int((p < 1e-15).sum()),
                     n_below_5e8=int((p < 5e-8).sum()), n_below_1e5=int((p < 1e-5).sum()), lambda_gc=lam,
                     top5="; ".join(f"{a['rs'].iat[i]} chr{a['chr'].iat[i]}:{a['ps'].iat[i]} p={a['p_score'].iat[i]:.3e}" for i in o[:5]),
                     catalogue_rank_by_min_p=rank))
    # write abbr in the ros5 format for downstream use (chr ps rs p_score)
    with open(f"{OUT}/output/local_C{k}.abbr.txt", "w") as f:
        for line in open(f"{OUT}/output/local_C{k}.assoc.txt").read().splitlines()[1:]:
            t = line.split("\t")
            f.write(t[0] + " " + t[2] + " " + t[1] + " " + t[-1] + "\n")
pd.DataFrame(rows).to_csv(f"{OUT}/new_components_summary.csv", index=False)
# catalogue median lambda context is not computed (would need all 546 full files)
print("non-passer group (min p >= 1e-15): n", int((cat['min_p_score'] >= 1e-15).sum()),
      "median min p", f"{cat.loc[cat['min_p_score']>=1e-15,'min_p_score'].median():.3e}")
