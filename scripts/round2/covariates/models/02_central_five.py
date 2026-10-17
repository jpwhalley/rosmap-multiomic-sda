"""Central associations in the text: C74 sex, C278 sex, C545 sex, C172 CERAD, C277 cogdx.
For each: published unadjusted test; RINT models M0-M3 and single-QC-covariate variants; stronger versions
(CERAD 1-4 ordinal: linear trend and proportional-odds logit; cogdx: all six categories, and NCI<MCI<AD trend);
outlier checks (drop the single most extreme donor by score; drop the 3 most extreme; worst case over all 276
leave-one-out fits). Family q for a perturbed component = BH over 547 with only that component's p replaced.
Outputs: central_five.csv, central_five_loo.csv, 02_central_five.log. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, warnings, numpy as np, pandas as pd
from scipy import stats as ss
from statsmodels.miscmodels.ordinal_model import OrderedModel
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = cfg.out_dir(cfg.WORK, "covariates")
exec(open(f"{HERE}/_common.py").read())
warnings.filterwarnings("ignore")

CENTRAL = [(74, "Sex"), (278, "Sex"), (545, "Sex"), (172, "ceradsc"), (277, "cogdx")]
PIDX = {"Sex": 0, "ceradsc": 1, "cogdx": 2}
base = lambda ph: ["age", "pmi", "ros"] + ([] if ph == "Sex" else ["sex"])
MOD = {"M0": lambda ph: [], "M1": base, "M1+lognuc": lambda ph: base(ph) + ["lognuc"], "M1+logupn": lambda ph: base(ph) + ["logupn"],
       "M2": lambda ph: base(ph) + ["lognuc", "logupn"],
       "M3": lambda ph: ["age", "age90p", "logpmi", "ros"] + ([] if ph == "Sex" else ["sex"])}

# family p-vectors (all 547) for substitution
rawU = pd.DataFrame([tests(S.loc[i].values) for i in S.index], index=S.index, columns=["Sex", "ceradsc", "cogdx"])
famP = {}
for ph in ["Sex", "ceradsc", "cogdx"]:
    for m, f in MOD.items():
        famP[(ph, m)] = pd.Series(nested_F(PHTERM[ph], f(ph), SUB[ph])[0], index=S.index)
    famP[(ph, "U")] = rawU[ph]
def qsub(ph, m, k, newp):
    v = famP[(ph, m)].copy(); v.loc[k] = newp; return float(pd.Series(bh(v.values), index=v.index).loc[k])

def fit_one(k, ph, cov, idx, phterm=None):
    """RINT recomputed on the donors in idx; returns p, betas."""
    y = pd.DataFrame({k: rint(S.loc[k, idx].values)}, index=idx)
    sub = [d for d in idx if d in set(SUB[ph])] if phterm is None or phterm == PHTERM[ph] else list(idx)
    return nested_F(phterm or PHTERM[ph], cov, sub, Yd=y)

rows = []
for k, ph in CENTRAL:
    sc = S.loc[k]
    r = dict(component=k, phenotype=ph)
    pU = tests(sc.values)[PIDX[ph]]
    r["published_q"] = float(ref.loc[k, ph]); r["U_p"] = pU; r["U_q"] = qsub(ph, "U", k, pU)
    # effect sizes for the unadjusted test
    if ph == "Sex":
        a, b = sc[D.sex == 1], sc[D.sex == 0]; U = ss.mannwhitneyu(a, b).statistic; r["AUC_male_vs_female"] = U / (len(a) * len(b))
    elif ph == "ceradsc":
        a, b = sc[D.cerad_ad == 1], sc[D.cerad_ad == 0]; U = ss.mannwhitneyu(a, b).statistic; r["AUC_AD_vs_nonAD"] = U / (len(a) * len(b))
    else:
        sub = SUB[ph]; H = ss.kruskal(*[sc[D.cogdx == g] for g in (1, 2, 4)]).statistic; r["KW_H"] = H; r["KW_epsilon2"] = H / (len(sub) - 1)
    for m, f in MOD.items():
        p, betas, dff = fit_one(k, ph, f(ph), list(don))
        r[f"{m}_p"] = p[0]; r[f"{m}_q"] = qsub(ph, m, k, p[0])
        for bn, bv in betas.items(): r[f"{m}_beta_{bn}"] = bv[0]
    # partial association of the score with each covariate (in M2, phenotype included), as R2 increments (aggregates)
    for cov in ["age", "pmi", "ros", "lognuc", "logupn"] + ([] if ph == "Sex" else ["sex"]):
        others = [c for c in MOD["M2"](ph) if c != cov]
        p, betas, _ = nested_F([cov], PHTERM[ph] + others, SUB[ph], Yd=Ydf[[k]])
        r[f"cov_{cov}_p_in_M2"] = p[0]
    # ---- stronger versions -----------------------------------------------------------------------------------
    y = rint(sc.values); d = D.copy(); d["y"] = y
    # CERAD 1-4 ordinal
    rho, prho = ss.spearmanr(sc.values, D.ceradsc); r["CERAD4_spearman_rho"] = rho; r["CERAD4_spearman_p"] = prho
    for m in ["M1", "M2"]:
        cov = [c for c in MOD[m]("ceradsc")]
        p, betas, _ = nested_F(["ceradsc"], cov, list(don), Yd=Ydf[[k]]); r[f"CERAD4_linear_{m}_p"] = p[0]; r[f"CERAD4_linear_{m}_beta"] = betas["ceradsc"][0]
        ex = d[["y"] + cov]
        om = OrderedModel(d["ceradsc"].astype(int), ex, distr="logit").fit(method="bfgs", disp=False, maxiter=2000)
        r[f"CERAD4_POlogit_{m}_p"] = float(om.pvalues["y"]); r[f"CERAD4_POlogit_{m}_OR_perSD"] = float(np.exp(om.params["y"]))
    # cogdx: all six categories (n=276) and NCI<MCI<AD trend
    for m in ["M0", "M1", "M2"]:
        cov = MOD[m]("cogdx")
        p, _, _ = nested_F(["C(cogdx6)"], cov, list(don), Yd=Ydf[[k]]); r[f"cogdx6_cat_{m}_p"] = p[0]
        d3 = D.loc[SUB["cogdx"]].copy(); d3["cog3"] = d3.cogdx.map({1.0: 0.0, 2.0: 1.0, 4.0: 2.0})
        p, betas, _ = nested_F(["cog3"], cov, SUB["cogdx"], Yd=Ydf[[k]], Dx=d3); r[f"cogdx3_trend_{m}_p"] = p[0]; r[f"cogdx3_trend_{m}_beta"] = betas["cog3"][0]
    # ---- outlier checks ----------------------------------------------------------------------------------------
    z = (sc - sc.median()).abs()
    order = z.sort_values(ascending=False).index
    for ndrop in (1, 3):
        keep = [x for x in don if x not in set(order[:ndrop])]
        pU2 = tests(sc.loc[keep].values, idx=pd.Index(keep))[PIDX[ph]]
        r[f"drop{ndrop}_U_p"] = pU2; r[f"drop{ndrop}_U_q"] = qsub(ph, "U", k, pU2)
        for m in ["M1", "M2"]:
            p, _, _ = fit_one(k, ph, MOD[m](ph), keep); r[f"drop{ndrop}_{m}_p"] = p[0]; r[f"drop{ndrop}_{m}_q"] = qsub(ph, m, k, p[0])
    # how extreme is the most extreme donor? (robust z; no donor ID or value printed)
    mad = ss.median_abs_deviation(sc.values, scale="normal"); r["max_robust_z"] = float(z.max() / mad)
    r["most_extreme_in_phenotype_group"] = {"Sex": {1.0: "male", 0.0: "female"}, "ceradsc": {1.0: "CERAD-AD", 0.0: "non-AD"},
                                           "cogdx": {1.0: "NCI", 2.0: "MCI", 4.0: "AD", 3.0: "excluded", 5.0: "excluded", 6.0: "excluded"}}[ph][
        D.loc[order[0], {"Sex": "sex", "ceradsc": "cerad_ad", "cogdx": "cogdx"}[ph]]]
    # leave-one-out worst case (unadjusted test and M1)
    loo = []
    for x in don:
        keep = pd.Index([y_ for y_ in don if y_ != x])
        pu = tests(sc.loc[keep].values, idx=keep)[PIDX[ph]]
        p1, _, _ = fit_one(k, ph, MOD["M1"](ph), list(keep))
        loo.append((pu, p1[0]))
    loo = np.array(loo)
    r["LOO_worst_U_p"] = loo[:, 0].max(); r["LOO_worst_U_q"] = qsub(ph, "U", k, loo[:, 0].max())
    r["LOO_worst_M1_p"] = loo[:, 1].max(); r["LOO_worst_M1_q"] = qsub(ph, "M1", k, loo[:, 1].max())
    rows.append(r)

res = pd.DataFrame(rows).set_index("component")
res.to_csv(f"{OUT}/central_five.csv")
pd.set_option("display.width", 250)
def fmt(x):
    return f"{x:.2g}" if isinstance(x, float) else str(x)
for k, r in res.iterrows():
    print(f"\n===== C{k} {r.phenotype} =====")
    print(f"published q {r.published_q:.3g} | reproduced U p {r.U_p:.3g} q {r.U_q:.3g}")
    print("  " + " | ".join(f"{m}: p {r[f'{m}_p']:.2g} q {r[f'{m}_q']:.2g}" for m in MOD))
    bcols = [c for c in r.index if "_beta_" in c and c.startswith(("M0_", "M1_", "M2_"))]
    print("  betas (RINT SD units): " + ", ".join(f"{c}={r[c]:.3f}" for c in bcols))
    print("  covariate p within M2: " + ", ".join(f"{c[4:-6]}={r[c]:.2g}" for c in r.index if c.startswith("cov_")))
    print(f"  CERAD 1-4: Spearman rho {r.CERAD4_spearman_rho:.3f} (p {r.CERAD4_spearman_p:.2g}); linear M1 p {r.CERAD4_linear_M1_p:.2g}, M2 p {r.CERAD4_linear_M2_p:.2g}; "
          f"PO-logit M1 p {r.CERAD4_POlogit_M1_p:.2g} (OR/SD {r.CERAD4_POlogit_M1_OR_perSD:.2f}), M2 p {r.CERAD4_POlogit_M2_p:.2g}")
    print(f"  cogdx 6-cat: M0 p {r.cogdx6_cat_M0_p:.2g}, M1 p {r.cogdx6_cat_M1_p:.2g}, M2 p {r.cogdx6_cat_M2_p:.2g}; NCI<MCI<AD trend M0 p {r.cogdx3_trend_M0_p:.2g}, M1 p {r.cogdx3_trend_M1_p:.2g}, M2 p {r.cogdx3_trend_M2_p:.2g}")
    print(f"  most extreme donor: robust z {r.max_robust_z:.1f} ({r.most_extreme_in_phenotype_group}); drop1: U q {r.drop1_U_q:.2g}, M1 q {r.drop1_M1_q:.2g}, M2 q {r.drop1_M2_q:.2g}; "
          f"drop3: U q {r.drop3_U_q:.2g}, M1 q {r.drop3_M1_q:.2g}, M2 q {r.drop3_M2_q:.2g}")
    print(f"  leave-one-out worst case over 276: U p {r.LOO_worst_U_p:.2g} (q {r.LOO_worst_U_q:.2g}); M1 p {r.LOO_worst_M1_p:.2g} (q {r.LOO_worst_M1_q:.2g})")
