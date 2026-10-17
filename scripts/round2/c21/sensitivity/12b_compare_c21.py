"""Compare C21 GEMMA runs: col1 (archived phenotype, should reproduce local_C21), col2 (RNA-discordant donor set to NA),
col3 (all four sex-discordant + RNA-mixed donors in the GWAS set NA). Top-SNP p-values and hit sets; SNP-level output stays
in the GEMMA folder ($G_ROS_RESULTS_DIR/identity_checks/gwas_c21)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, re, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); OUT = f"{cfg.WORK}/identity_checks/gwas_c21/output"; RS = cfg.out_dir(cfg.WORK, "identity_checks", "results")
ref = pd.read_csv(f"{cfg.AUD}/gwas547/output/local_C21.assoc.txt", sep="\t")
runs = {k: pd.read_csv(f"{OUT}/C21_col{k}.assoc.txt", sep="\t") for k in (1, 2, 3)}
for k in (1, 2, 3):
    lg = open(f"{OUT}/C21_col{k}.log.txt").read(); n = re.search(r"number of analyzed individuals = (\d+)", lg).group(1)
    print(f"col{k}: analysed individuals {n}")
c1 = runs[1]; print("col1 reproduces local_C21: same SNP order", bool((c1.rs.values == ref.rs.values).all()), "| max |dlog10 p_score|", float(np.max(np.abs(np.log10(c1.p_score) - np.log10(ref.p_score)))))
top = ref.nsmallest(20, "p_score").rs.tolist()
rows = []
for k in (1, 2, 3):
    d = runs[k].set_index("rs")
    rows.append(dict(run={1: "archived phenotype (n=228)", 2: "RNA-mismatched donor removed", 3: "4 flagged donors removed"}[k],
                     min_p=d.p_score.min(), top_snp_same=d.p_score.idxmin() == ref.set_index("rs").p_score.idxmin(),
                     p_at_archived_top=d.loc[ref.set_index("rs").p_score.idxmin(), "p_score"],
                     beta_at_archived_top=d.loc[ref.set_index("rs").p_score.idxmin(), "beta"],
                     n_below_1e15=int((d.p_score < 1e-15).sum()), n_below_5e8=int((d.p_score < 5e-8).sum()), n_below_1e5=int((d.p_score < 1e-5).sum()),
                     same_5e8_set_as_archived=set(d.index[d.p_score < 5e-8]) == set(ref.rs[ref.p_score < 5e-8]),
                     top20_overlap=len(set(d.p_score.nsmallest(20).index) & set(top)),
                     null_ratio_median_neg2lnp=float(np.median(-2 * np.log(d.p_score)) / 1.386)))   # median of -2 ln p over its chi2(2) null median; comparison only, not lambda_GC
R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(f"{RS}/12_c21_gwas_sensitivity.csv", index=False)
