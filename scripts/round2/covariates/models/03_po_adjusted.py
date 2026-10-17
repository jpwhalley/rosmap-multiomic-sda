"""Rank-based adjusted analysis: proportional-odds (cumulative-logit) regression of each component's score, treated as
ordinal (one category per distinct value), on phenotype + covariates. With no covariates this is the regression form
of the published Mann-Whitney / Kruskal-Wallis tests, so the step published -> PO0 -> PO1 separates the change of test
(small here) from the effect of the covariates. Phenotype p = likelihood-ratio test (1 df sex/CERAD; 2 df cogdx 1/2/4).
  PO0 none | PO1 age + pmi + Study [+ msex] | PO2 PO1 + log nuclei + log UMIs/nucleus | PO3 age + I(90+) + log pmi + Study [+ msex]
BH per phenotype across 547. Central five also: drop the most extreme donor, and leave-one-out worst case (PO1, PO2).
Outputs: po_tests_547.csv, po_calls.csv, po_central_five.csv, 03_po_adjusted.log. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, warnings, numpy as np, pandas as pd
from scipy import stats as ss
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = cfg.out_dir(cfg.WORK, "covariates")
exec(open(f"{HERE}/_common.py").read()); exec(open(f"{HERE}/_po.py").read())
warnings.filterwarnings("ignore")
PH = ["Sex", "ceradsc", "cogdx"]
pub = {c: set(ref.index[ref[c] < 0.05]) for c in PH}
base = lambda ph: ["age", "pmi", "ros"] + ([] if ph == "Sex" else ["sex"])
MOD = {"PO0": lambda ph: [], "PO1": base, "PO2": lambda ph: base(ph) + ["lognuc", "logupn"],
       "PO3": lambda ph: ["age", "age90p", "logpmi", "ros"] + ([] if ph == "Sex" else ["sex"])}
def phX(ph, idx):
    d = D.loc[idx]
    if ph == "Sex": return d[["sex"]].values
    if ph == "ceradsc": return d[["cerad_ad"]].values
    return np.column_stack([(d.cogdx == 2).astype(float), (d.cogdx == 4).astype(float)])
def covX(cols, idx):
    return D.loc[idx, cols].values.astype(float) if cols else np.zeros((len(idx), 0))
def run(k, ph, m, idx):
    idx = [x for x in idx if x in set(SUB[ph])]
    return po_lrt(S.loc[k, idx].values, phX(ph, idx), covX(MOD[m](ph), idx))

rawU = pd.DataFrame([tests(S.loc[i].values) for i in S.index], index=S.index, columns=PH)
res = {}; nonconv = 0
for ph in PH:
    o = pd.DataFrame(index=S.index); o["U_p"] = rawU[ph]; o["U_q"] = bh(rawU[ph].values)
    for m in MOD:
        ps, bs = [], []
        for k in S.index:
            p, b, c = run(k, ph, m, list(don)); ps.append(p); bs.append(b[-1]); nonconv += (not c)
        o[f"{m}_p"] = ps; o[f"{m}_q"] = bh(ps); o[f"{m}_beta" + ("_AD_vs_NCI" if ph == "cogdx" else "")] = bs
    o["published_sig"] = o.index.isin(pub[ph]); res[ph] = o
print(f"PO fits not flagged converged: {nonconv}")
allt = pd.concat(res, axis=1); allt.to_csv(f"{OUT}/po_tests_547.csv")

print("\nPO0 (no covariates) vs published Mann-Whitney/Kruskal-Wallis p:")
for ph in PH:
    o = res[ph]; l0 = -np.log10(o.PO0_p); lu = -np.log10(o.U_p)
    sig0 = set(o.index[o.PO0_q < 0.05])
    print(f"  {ph}: Spearman(-log10 p) {ss.spearmanr(l0, lu)[0]:.4f}; median p ratio PO0/U {np.median(o.PO0_p / o.U_p):.3f}; "
          f"PO0 calls {len(sig0)} (published kept {len(sig0 & pub[ph])}/{len(pub[ph])}, gained {len(sig0 - pub[ph])})")

calls = []
for ph in PH:
    o = res[ph]
    print(f"\n=== {ph} ===")
    for m in MOD:
        sig = set(o.index[o[f"{m}_q"] < 0.05])
        lost = sorted(pub[ph] - sig); gained = sorted(sig - pub[ph])
        print(f"  {m} [{'+'.join(MOD[m](ph)) or 'none'}]: significant {len(sig)}; published kept {len(sig & pub[ph])}/{len(pub[ph])}; lost {lost}; gained {gained}")
    for k in sorted(pub[ph] | set().union(*[set(o.index[o[f'{m}_q'] < 0.05]) for m in MOD])):
        r = dict(component=k, phenotype=ph, published=k in pub[ph], U_q=o.loc[k, "U_q"])
        for m in MOD: r[f"{m}_p"] = o.loc[k, f"{m}_p"]; r[f"{m}_q"] = o.loc[k, f"{m}_q"]
        bc = [c for c in o.columns if c.startswith("PO0_beta")][0]
        r["beta_PO0"] = o.loc[k, bc]; r["beta_PO1"] = o.loc[k, bc.replace("PO0", "PO1")]; r["beta_PO2"] = o.loc[k, bc.replace("PO0", "PO2")]
        calls.append(r)
calls = pd.DataFrame(calls); calls.to_csv(f"{OUT}/po_calls.csv", index=False)
pc = calls[calls.published]
print(f"\nAll {len(pc)} calls of the 547 family:")
for m in MOD:
    s = pc[f"{m}_q"] < 0.05
    print(f"  {m}: survive q<0.05 {int(s.sum())}/{len(pc)} (Sex {int(s[pc.phenotype=='Sex'].sum())}/3, CERAD {int(s[pc.phenotype=='ceradsc'].sum())}/42, cogdx {int(s[pc.phenotype=='cogdx'].sum())}/28); "
          f"nominal p<0.05 {int((pc[f'{m}_p'] < 0.05).sum())}, p<0.01 {int((pc[f'{m}_p'] < 0.01).sum())}; "
          f"gained {int(((~calls.published) & (calls[f'{m}_q'] < 0.05)).sum())}")
for m in ["PO1", "PO2"]:
    same = (np.sign(pc.beta_PO0) == np.sign(pc[f"beta_{m}"])).sum(); ratio = pc[f"beta_{m}"] / pc.beta_PO0
    print(f"  {m}: phenotype effect same sign as PO0 in {same}/{len(pc)}; beta ratio {m}/PO0 median {ratio.median():.2f} (IQR {ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f}; range {ratio.min():.2f}-{ratio.max():.2f})")
    lost = pc[pc[f"{m}_q"] >= 0.05]
    if len(lost):
        print(f"     lost calls: {m} q range {lost[f'{m}_q'].min():.3f}-{lost[f'{m}_q'].max():.3f}; published q of lost calls: median {lost.U_q.median():.3f} "
              f"(published q of kept calls: median {pc[pc[f'{m}_q'] < 0.05].U_q.median():.3g})")

# ---------------- central five: PO models, outlier checks ----------------------------------------------------------
CENTRAL = [(74, "Sex"), (278, "Sex"), (545, "Sex"), (172, "ceradsc"), (277, "cogdx")]
def qsub(ph, m, k, newp):
    v = res[ph][f"{m}_p"].copy(); v.loc[k] = newp; return float(pd.Series(bh(v.values), index=v.index).loc[k])
rows = []
for k, ph in CENTRAL:
    o = res[ph]; r = dict(component=k, phenotype=ph, published_q=float(ref.loc[k, ph]))
    for m in MOD: r[f"{m}_p"] = o.loc[k, f"{m}_p"]; r[f"{m}_q"] = o.loc[k, f"{m}_q"]
    sc = S.loc[k]; order = (sc - sc.median()).abs().sort_values(ascending=False).index
    for m in ["PO1", "PO2"]:
        keep = [x for x in don if x != order[0]]
        p, _, _ = run(k, ph, m, keep); r[f"drop1_{m}_p"] = p; r[f"drop1_{m}_q"] = qsub(ph, m, k, p)
        worst = max(run(k, ph, m, [x for x in don if x != y])[0] for y in don)
        r[f"LOO_worst_{m}_p"] = worst; r[f"LOO_worst_{m}_q"] = qsub(ph, m, k, worst)
    rows.append(r)
cf = pd.DataFrame(rows).set_index("component"); cf.to_csv(f"{OUT}/po_central_five.csv")
print("\nCentral five (PO, LRT):")
for k, r in cf.iterrows():
    print(f"  C{k} {r.phenotype}: published q {r.published_q:.3g} | " + " | ".join(f"{m} p {r[f'{m}_p']:.2g} q {r[f'{m}_q']:.2g}" for m in MOD)
          + f" | drop1 PO1 q {r.drop1_PO1_q:.2g}, PO2 q {r.drop1_PO2_q:.2g} | LOO worst PO1 q {r.LOO_worst_PO1_q:.2g}, PO2 q {r.LOO_worst_PO2_q:.2g}")
