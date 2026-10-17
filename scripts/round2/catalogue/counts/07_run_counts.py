"""Minor 1: true run count (number of the 10 runs contributing >=1 member) for each of the 547 components.
Historical IDs from members_buggy_max_it1500.csv (reproduces the deposit to 4.4e-16; row 520 retired) plus the two
new clusters (1465 -> C547, 716 -> C548) from members_corrected_max1_it1500.csv. Also split by omics_count and
single-donor dominance."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import json, os, numpy as np, pandas as pd
AUD = cfg.AUD
OUT = cfg.out_dir(cfg.WORK, "r1_numbers", "out")
b = pd.read_csv(f"{AUD}/members_buggy_max_it1500.csv").set_index("buggy_max_id")
c = pd.read_csv(f"{AUD}/members_corrected_max1_it1500.csv").set_index("cluster")
T = b.drop(index=520)[["cluster", "size", "true_runs"]].copy()
for k, clu in ((547, 1465), (548, 716)):
    T.loc[k] = [clu, c.loc[clu, "size"], c.loc[clu, "true_runs"]]
T = T.astype(int); assert len(T) == 547 and T.true_runs.min() >= 5
dom = pd.read_csv(os.path.join(OUT, "donor_dominance_547.csv"), index_col=0)
cnt = pd.read_csv(os.path.join(OUT, "pip_counts_547.csv"), index_col=0)
T["omics_count"] = cnt["omics_count_recomputed"]; T["single_donor"] = dom["top_share_SS"] > 0.8
T.to_csv(os.path.join(OUT, "run_counts_547.csv"))
res = {"true_runs_distribution": {str(r): int((T.true_runs == r).sum()) for r in range(5, 11)},
       "members_per_component_quantiles_0_25_50_75_100": [float(x) for x in np.percentile(T["size"], [0, 25, 50, 75, 100])],
       "components_with_more_members_than_runs": int((T["size"] > T.true_runs).sum()),
       "median_true_runs": float(T.true_runs.median()),
       "true_runs_distribution_omics>=3": {str(r): int(((T.true_runs == r) & (T.omics_count >= 3)).sum()) for r in range(5, 11)},
       "true_runs_distribution_single_donor": {str(r): int(((T.true_runs == r) & T.single_donor).sum()) for r in range(5, 11)},
       "true_runs_distribution_not_single_donor": {str(r): int(((T.true_runs == r) & ~T.single_donor).sum()) for r in range(5, 11)},
       "headline": {int(k): {"true_runs": int(T.loc[k, "true_runs"]), "members": int(T.loc[k, "size"])} for k in [351, 2, 74, 87, 278, 177, 172, 277, 21, 76, 99, 130, 291, 345, 416]}}
json.dump(res, open(os.path.join(OUT, "run_counts_summary.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
