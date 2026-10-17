"""Which covariate moves the CERAD / cogdx calls in the PO framework, and are the lost calls attenuated or just
re-ranked under BH? Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, warnings, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = cfg.out_dir(cfg.WORK, "covariates")
exec(open(f"{HERE}/_common.py").read()); exec(open(f"{HERE}/_po.py").read())
warnings.filterwarnings("ignore")
src = open(f"{HERE}/03_po_adjusted.py").read(); exec(src[src.index("PH = ["):src.index("rawU = ")])
po = pd.read_csv(f"{OUT}/po_tests_547.csv", header=[0, 1], index_col=0)
for ph in ["ceradsc", "cogdx"]:
    print(f"=== {ph} ===")
    for cov in [["age"], ["pmi"], ["ros"], ["sex"], ["age", "pmi", "ros", "sex"]]:
        ps = [po_lrt(S.loc[k, [x for x in don if x in set(SUB[ph])]].values, phX(ph, [x for x in don if x in set(SUB[ph])]),
                     covX(cov, [x for x in don if x in set(SUB[ph])]))[0] for k in S.index]
        q = pd.Series(bh(ps), index=S.index); sig = set(q.index[q < 0.05])
        print(f"  PO + {'+'.join(cov)}: significant {len(sig)}, published kept {len(sig & pub[ph])}/{len(pub[ph])}")
    bcol = "PO0_beta" if ph != "cogdx" else "PO0_beta_AD_vs_NCI"
    o = po[ph]; pc = o[o.published_sig.astype(str) == "True"]
    lost = pc[pc.PO1_q >= 0.05]; kept = pc[pc.PO1_q < 0.05]
    r_l = lost[bcol.replace("PO0", "PO1")] / lost[bcol]; r_k = kept[bcol.replace("PO0", "PO1")] / kept[bcol]
    print(f"  beta ratio PO1/PO0: lost calls median {r_l.median():.2f} (range {r_l.min():.2f}-{r_l.max():.2f}); kept calls median {r_k.median():.2f}")
    print(f"  raw p PO0 -> PO1, lost calls: median {lost.PO0_p.median():.2g} -> {lost.PO1_p.median():.2g}; kept: {kept.PO0_p.median():.2g} -> {kept.PO1_p.median():.2g}")
    print(f"  BH threshold (largest p called): PO0 {o.PO0_p[o.PO0_q < 0.05].max():.2g}; PO1 {o.PO1_p[o.PO1_q < 0.05].max():.2g}")
    # do lost calls track age?
    a = pd.Series([abs(ss.spearmanr(S.loc[k], D.age)[0]) for k in pc.index], index=pc.index)
    print(f"  |Spearman(score, age)|: lost calls median {a[lost.index].median():.2f}, kept calls median {a[kept.index].median():.2f}")
