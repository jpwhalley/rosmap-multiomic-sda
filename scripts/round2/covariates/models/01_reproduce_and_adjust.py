"""Reviewer 1 Major 4: covariate-adjusted sensitivity analysis of the phenotype associations (547-component family).

(1) Reproduce the published unadjusted tests (sex MWU; CERAD {1,2} vs {3,4} MWU; cogdx KW over 1,2,4; BH per variable
    across 547) and check them against clinical_BH_547_family.csv ($G_ROS_TABLES_DIR).
(2) Rank-inverse-normal (Blom) transform of each component's donor scores (over all 276 donors), then OLS:
      M0  RINT ~ phenotype                                    (same donors, no covariates: isolates the change of test)
      M1  RINT ~ phenotype + age_death + pmi + Study [+ msex]  (primary adjusted; msex only when the phenotype is not sex)
      M2  M1 + log(post-QC nuclei) + log(UMIs per nucleus)     (RNA-quality proxies; no RIN/batch exists, see 00_)
      M3  M1 with age coded as min(age,90) + I(90+) and log(pmi)  (covariate-coding robustness)
    Phenotype terms: sex = msex; CERAD = binary AD {1,2} vs {3,4}; cogdx = categorical over 1,2,4 (2-df F test).
    P from the nested-model F test for the phenotype term(s) (= t^2 for 1 df). BH per phenotype across 547.
Outputs (aggregates only): all_tests_547.csv, rint_calls.csv, covariate_balance.csv, covariate_only_counts.csv,
01_reproduce_and_adjust.log (stdout)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats as ss
import statsmodels.formula.api as smf

B = cfg.ROS; AUD = cfg.AUD
REF = f"{cfg.TABLES}/clinical_BH_547_family.csv"
OUT = cfg.out_dir(cfg.WORK, "covariates")
PH = ["Sex", "ceradsc", "cogdx"]

# ---------------- data -------------------------------------------------------------------------------------------
dep = pd.read_csv(f"{B}/sda/samples_final.csv", index_col=0); dep.columns = dep.columns.astype(str)
S = dep.drop(index=520).copy()
for k in (547, 548):
    S.loc[k] = np.load(f"{AUD}/c547/C{k}_A.npy")
don = S.columns
clin = pd.read_csv(f"{B}/oth/meta/ROSMAP_clinical.csv"); clin.index = clin.projid.astype(int).astype(str); clin = clin.loc[don]
qc = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str); qc = qc.loc[don]

D = pd.DataFrame(index=don)
D["sex"] = clin.msex.astype(float)                                   # 1 = male
D["cerad_ad"] = clin.ceradsc.map({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0})    # 1 = CERAD-AD
D["ceradsc"] = clin.ceradsc.astype(float)
D["cogdx"] = clin.cogdx.astype(float)
age_raw = clin.age_death.astype(str)
D["age90p"] = (age_raw == "90+").astype(float)
D["age"] = pd.to_numeric(age_raw.replace("90+", "90"), errors="raise").clip(upper=90.0)
D["pmi"] = clin.pmi.astype(float); D["logpmi"] = np.log(D.pmi)
D["ros"] = (clin.Study == "ROS").astype(float)
D["lognuc"] = np.log(qc.nuclei); D["logupn"] = np.log(qc.umis_per_nucleus)
assert D.notna().all().all()
print(f"Donors {len(D)}; missing covariate values: 0; age 90+ (censored, coded 90): {int(D.age90p.sum())}")

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o]
    adj = np.minimum.accumulate((r * n / np.arange(1, n + 1))[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(adj, 1); return out

def rint(x):
    x = np.asarray(x, float); n = len(x); return ss.norm.ppf((ss.rankdata(x) - 0.375) / (n + 0.25))

# ---------------- (1) reproduce the published unadjusted tests -----------------------------------------------------
CER = {1: "AD", 2: "AD", 3: "No", 4: "No"}
def tests(v, d=D, c=clin):
    s = pd.Series(np.asarray(v, float), index=d.index); m = c.msex.loc[d.index]; cb = c.ceradsc.loc[d.index].map(CER); cg = c.cogdx.loc[d.index]
    return (ss.mannwhitneyu(s[m == 1], s[m == 0], alternative="two-sided").pvalue,
            ss.mannwhitneyu(s[cb == "AD"], s[cb == "No"], alternative="two-sided").pvalue,
            ss.kruskal(*[s[cg == k] for k in (1, 2, 4)]).pvalue)
rawU = pd.DataFrame([tests(S.loc[i].values) for i in S.index], index=S.index, columns=PH)
qU = rawU.apply(bh)
ref = pd.read_csv(REF, index_col=0)
assert list(ref.index) == list(qU.index)
print("Reproduction vs clinical_BH_547_family.csv: max |q diff| per variable",
      {c: float((qU[c] - ref[c]).abs().max()) for c in PH})
pub = {c: set(ref.index[ref[c] < 0.05]) for c in PH}
for c in PH:
    assert pub[c] == set(qU.index[qU[c] < 0.05])
print("Published significant calls (547 family):", {c: len(pub[c]) for c in PH}, "total", sum(len(v) for v in pub.values()))
print("  CERAD includes C548:", 548 in pub["ceradsc"])

# ---------------- (2) adjusted models ---------------------------------------------------------------------------
Y = np.column_stack([rint(S.loc[i].values) for i in S.index])   # donors x 547
Ydf = pd.DataFrame(Y, index=don, columns=S.index)

def design(cols, sub):
    X = [np.ones(len(sub))]
    names = ["Intercept"]
    for c in cols:
        if c == "C(cogdx)":
            for lev in (2.0, 4.0):
                X.append((D.loc[sub, "cogdx"] == lev).astype(float).values); names.append(f"cogdx{int(lev)}")
        else:
            X.append(D.loc[sub, c].values); names.append(c)
    return np.column_stack(X), names

def nested_F(ph_terms, cov_terms, sub):
    """F test of ph_terms given cov_terms, vectorised over all 547 components. Returns p, betas (dict)."""
    Xf, nf = design(ph_terms + cov_terms, sub); Xr, _ = design(cov_terms, sub)
    Ys = Ydf.loc[sub].values
    bf, *_ = np.linalg.lstsq(Xf, Ys, rcond=None); rf = Ys - Xf @ bf; rssf = (rf ** 2).sum(0)
    br, *_ = np.linalg.lstsq(Xr, Ys, rcond=None); rr = Ys - Xr @ br; rssr = (rr ** 2).sum(0)
    q = Xf.shape[1] - Xr.shape[1]; dff = len(sub) - Xf.shape[1]
    F = ((rssr - rssf) / q) / (rssf / dff); p = ss.f.sf(F, q, dff)
    betas = {nf[j]: bf[j] for j in range(1, 1 + q)}   # phenotype terms come first after the intercept
    return p, betas, dff

PHTERM = {"Sex": ["sex"], "ceradsc": ["cerad_ad"], "cogdx": ["C(cogdx)"]}
SUB = {"Sex": list(don), "ceradsc": list(don), "cogdx": list(D.index[D.cogdx.isin([1, 2, 4])])}
def covs(model, ph):
    base = ["age", "pmi", "ros"] + ([] if ph == "Sex" else ["sex"])
    if model == "M0": return []
    if model == "M1": return base
    if model == "M2": return base + ["lognuc", "logupn"]
    if model == "M3": return ["age", "age90p", "logpmi", "ros"] + ([] if ph == "Sex" else ["sex"])
MODELS = ["M0", "M1", "M2", "M3"]

res = {}
rows = []
for ph in PH:
    sub = SUB[ph]
    out = pd.DataFrame(index=S.index)
    out["U_p"] = rawU[ph]; out["U_q"] = qU[ph]
    for m in MODELS:
        p, betas, dff = nested_F(PHTERM[ph], covs(m, ph), sub)
        out[f"{m}_p"] = p; out[f"{m}_q"] = bh(p)
        for bn, bv in betas.items():
            out[f"{m}_beta_{bn}"] = bv
    out["published_sig"] = out.index.isin(pub[ph])
    res[ph] = out
    print(f"\n=== {ph}: n = {len(sub)} donors ===")
    for m in MODELS:
        sig = set(out.index[out[f"{m}_q"] < 0.05])
        print(f"  {m} [{'+'.join(covs(m, ph)) or 'none'}]: significant {len(sig)}; of {len(pub[ph])} published kept {len(sig & pub[ph])}, "
              f"lost {len(pub[ph] - sig)}, gained {len(sig - pub[ph])}")

# validate the vectorised F test against statsmodels for a few cells
chk = []
for ph, comp in [("Sex", 74), ("ceradsc", 172), ("cogdx", 277), ("Sex", 278)]:
    sub = SUB[ph]; d = D.loc[sub].copy(); d["y"] = Ydf.loc[sub, comp]
    f1 = "y ~ " + " + ".join([t for t in PHTERM[ph]] + covs("M2", ph))
    f0 = "y ~ " + (" + ".join(covs("M2", ph)) if covs("M2", ph) else "1")
    full = smf.ols(f1, d).fit(); red = smf.ols(f0, d).fit()
    p_sm = full.compare_f_test(red)[1]
    chk.append(abs(p_sm - res[ph].loc[comp, "M2_p"]) / p_sm)
print("\nVectorised F-test vs statsmodels compare_f_test (M2, 4 cells): max relative p diff", f"{max(chk):.1e}")

# ---------------- tables ----------------------------------------------------------------------------------------
allt = pd.concat({ph: res[ph] for ph in PH}, axis=1)
allt.to_csv(f"{OUT}/all_tests_547.csv")

calls = []
for ph in PH:
    o = res[ph]
    for k in sorted(pub[ph] | set().union(*[set(o.index[o[f"{m}_q"] < 0.05]) for m in MODELS])):
        r = dict(component=k, phenotype=ph, published=k in pub[ph], U_q=o.loc[k, "U_q"])
        for m in MODELS:
            r[f"{m}_q"] = o.loc[k, f"{m}_q"]; r[f"{m}_sig"] = bool(o.loc[k, f"{m}_q"] < 0.05)
        bcols = [c for c in o.columns if c.startswith("M0_beta_")]
        for bc in bcols:
            b0 = o.loc[k, bc]; b1 = o.loc[k, bc.replace("M0_", "M1_")]; b2 = o.loc[k, bc.replace("M0_", "M2_")]
            nm = bc.replace("M0_beta_", "")
            r[f"beta_{nm}_M0"] = b0; r[f"beta_{nm}_M1"] = b1; r[f"beta_{nm}_M2"] = b2
        calls.append(r)
calls = pd.DataFrame(calls)
calls.to_csv(f"{OUT}/rint_calls.csv", index=False)

pubc = calls[calls.published]
print(f"\nPublished calls: {len(pubc)}")
for m in MODELS:
    lost = pubc[~pubc[f"{m}_sig"]]
    gained = calls[(~calls.published) & calls[f"{m}_sig"]]
    print(f"  {m}: survive {int(pubc[f'{m}_sig'].sum())}/{len(pubc)}; lost {len(lost)} "
          f"{[(int(a), b) for a, b in zip(lost.component, lost.phenotype)]}; gained {len(gained)} "
          f"{[(int(a), b) for a, b in zip(gained.component, gained.phenotype)]}")

# sign consistency of the phenotype effect M0 -> M1 / M2 for published calls (binary phenotypes; cogdx AD-vs-NCI)
for ph, bn in [("Sex", "sex"), ("ceradsc", "cerad_ad"), ("cogdx", "cogdx4")]:
    o = res[ph].loc[sorted(pub[ph])]
    same1 = (np.sign(o[f"M0_beta_{bn}"]) == np.sign(o[f"M1_beta_{bn}"])).sum(); same2 = (np.sign(o[f"M0_beta_{bn}"]) == np.sign(o[f"M2_beta_{bn}"])).sum()
    ratio1 = (o[f"M1_beta_{bn}"] / o[f"M0_beta_{bn}"]); ratio2 = (o[f"M2_beta_{bn}"] / o[f"M0_beta_{bn}"])
    print(f"  {ph}: effect sign unchanged M0->M1 {same1}/{len(o)}, M0->M2 {same2}/{len(o)}; "
          f"|beta| ratio M1/M0 median {ratio1.median():.2f} [range {ratio1.min():.2f}-{ratio1.max():.2f}], "
          f"M2/M0 median {ratio2.median():.2f} [range {ratio2.min():.2f}-{ratio2.max():.2f}]")

# ---------------- covariate balance across phenotype groups (aggregates) -----------------------------------------
bal = []
grp = {"Sex": ("sex", {1.0: "male", 0.0: "female"}), "ceradsc": ("cerad_ad", {1.0: "CERAD-AD", 0.0: "non-AD"}),
       "cogdx": ("cogdx", {1.0: "NCI", 2.0: "MCI", 4.0: "AD dementia"})}
for ph, (col, lab) in grp.items():
    d = D[D[col].isin(lab.keys())]
    for cov in ["age", "age90p", "pmi", "ros", "sex", "lognuc", "logupn"]:
        if cov == col or (ph == "Sex" and cov == "sex"): continue
        g = [d.loc[d[col] == k, cov] for k in lab]
        if cov in ("age90p", "ros", "sex"):
            tab = pd.crosstab(d[col], d[cov]); p = ss.chi2_contingency(tab)[1]
        else:
            p = ss.kruskal(*g).pvalue
        r = dict(phenotype=ph, covariate=cov, p_difference=p)
        for k, x in zip(lab.values(), g):
            r[f"{k}_n"] = len(x); r[f"{k}_mean"] = x.mean(); r[f"{k}_sd"] = x.std()
        bal.append(r)
bal = pd.DataFrame(bal); bal.to_csv(f"{OUT}/covariate_balance.csv", index=False)
print("\nCovariate balance across phenotype groups (group means; p from Kruskal-Wallis or chi-square):")
for _, r in bal.iterrows():
    means = "; ".join(f"{k[:-5]} {r[k]:.3g}" for k in r.index if k.endswith("_mean") and pd.notna(r[k]))
    print(f"  {r.phenotype:8s} {r.covariate:8s} p={r.p_difference:.2g}   {means}")

# ---------------- how many components do the covariates themselves explain? -------------------------------------
cc = []
for cov in ["age", "pmi", "ros", "lognuc", "logupn", "sex", "cerad_ad"]:
    X = np.column_stack([np.ones(len(D)), D[cov].values]); b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r = Y - X @ b; rss = (r ** 2).sum(0); tss = ((Y - Y.mean(0)) ** 2).sum(0)
    F = (tss - rss) / (rss / (len(D) - 2)); p = ss.f.sf(F, 1, len(D) - 2)
    cc.append(dict(covariate=cov, n_components_q_lt_0_05=int((bh(p) < 0.05).sum()), max_R2=float((1 - rss / tss).max())))
cc = pd.DataFrame(cc); cc.to_csv(f"{OUT}/covariate_only_counts.csv", index=False)
print("\nComponents associated with each variable alone (RINT ~ variable; BH over 547):")
print(cc.to_string(index=False))
