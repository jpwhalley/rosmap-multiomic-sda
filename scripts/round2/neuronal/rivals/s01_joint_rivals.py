"""Are C172 (plaque) and C277 (clinical AD dementia) distinctive? (selection_review, 2026-10-15)
Fits the v23 joint model -- exactly the neuronal_c21/n03 'cogdx124' CERAD+Braak+cogdx model (proportional odds of the
score treated as ordinal, one level per distinct value; terms CERAD AD vs not, Braak stage, MCI vs NCI, AD dementia vs NCI;
covariates age (90+ -> 90), PMI, cohort, sex; n = 254 donors with cogdx 1/2/4; profile-likelihood 95% CIs) -- reusing the
neuronal_c21 loaders (nc_common -> dc_common) and PO class (nc_po), for every distributed component (top-donor share < 0.10;
193). Adds: block LR p per phenotype, a within-component Wald test of log OR(CERAD) = log OR(AD dementia) (observed
information), rank-INT OLS of the same model, the all-276 coding (any dementia 4/5/6, any MCI 2/3) for the named set, and
stacked rank-INT models (donor-clustered SE, as verify_neuronal_c21/v05) comparing each named component with C172 and C277.
Read-only everywhere else; writes only under $G_ROS_RESULTS_DIR/selection_review/are_C172_C277_distinctive/. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as ss
warnings.filterwarnings("ignore")
R2 = cfg.WORK
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))   # nc_common.py, nc_po.py
from nc_common import scores547, full_frame, rint, ols          # noqa: E402 (validated loaders, reused)
from nc_po import PO, _negll                                      # noqa: E402
ME = f"{R2}/selection_review/are_C172_C277_distinctive"; OUT = cfg.out_dir(ME, "out")
DS1 = cfg.DS1_V23
BCOR = f"{cfg.TABLES}/cell_type_loadings_corrected.csv"
AUDC = f"{cfg.AUD}/c547"
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]

S = scores547(); D = full_frame()
D["mci_any"] = D.cogdx.isin([2, 3]).astype(float); D["dementia_any"] = D.cogdx.isin([4, 5, 6]).astype(float)
COV = ["age", "pmi", "ros", "sex"]
SUB124 = list(D.index[D.cogdx.isin([1, 2, 4])]); SUB276 = list(D.index)
PH124 = ["cerad_ad", "braak", "cogdx_mci", "cogdx_ad"]; PH276 = ["cerad_ad", "braak", "mci_any", "dementia_any"]
BLK = {"CERAD": [0], "Braak": [1], "cogdx": [2, 3]}

# ---------------- catalogue facts: class, dominant cell type, SEA-AD, single-phenotype adjusted (PO1) ------------------
d1 = pd.read_csv(DS1).set_index("Component")
B = pd.read_csv(BCOR, index_col=0)[CTS]
for k in (547, 548):
    B.loc[k] = np.load(f"{AUDC}/C{k}_Bcor.npy")                  # corrected cell-type loadings of the two appended rows
po1 = pd.read_csv(f"{R2}/covariates/po_tests_547.csv", header=[0, 1], index_col=0)
dist = list(d1.index[d1.concentration_class == "distributed"]); assert len(dist) == 193
Ba = B.loc[dist].abs()
cat = pd.DataFrame(index=pd.Index(dist, name="component"))
zero = Ba.sum(axis=1) == 0                                          # C333, C350, C497: all six corrected loadings are 0
cat["dominant_cell_type"] = Ba.idxmax(axis=1).where(~zero, "none")
cat["dominant_abs_share"] = Ba.max(axis=1) / Ba.sum(axis=1)
cat["neuronal_abs_share"] = (Ba.Exc + Ba.Inh) / Ba.sum(axis=1)
srt = np.sort(Ba.values, axis=1); cat["second_cell_type"] = [CTS[i] for i in np.argsort(Ba.values, axis=1)[:, -2]]
cat.loc[zero, "second_cell_type"] = "none"
print("components with all-zero corrected cell-type loadings:", list(Ba.index[zero]))
cat["second_over_first"] = srt[:, -2] / srt[:, -1]
cat["neuron_led"] = cat.dominant_cell_type.isin(["Exc", "Inh"])
cat["top_donor_share"] = d1.loc[dist, "top_donor_share"]
cat["cerad_q_unadj"] = d1.loc[dist, "ceradsc"]; cat["cogdx_q_unadj"] = d1.loc[dist, "cogdx"]
cat["cerad_PO1_p"] = po1[("ceradsc", "PO1_p")].loc[dist]; cat["cerad_PO1_q"] = po1[("ceradsc", "PO1_q")].loc[dist]
cat["cerad_PO1_OR"] = np.exp(po1[("ceradsc", "PO1_beta")].loc[dist])
cat["cogdx_PO1_p"] = po1[("cogdx", "PO1_p")].loc[dist]; cat["cogdx_PO1_q"] = po1[("cogdx", "PO1_q")].loc[dist]
cat["cogdx_AD_PO1_OR"] = np.exp(po1[("cogdx", "PO1_beta_AD_vs_NCI")].loc[dist])
cat["cerad_rank_dist"] = cat.cerad_PO1_p.rank().astype(int); cat["cogdx_rank_dist"] = cat.cogdx_PO1_p.rank().astype(int)
cat["seaad_partner"] = d1.loc[dist, "SEA-AD_comp"]; cat["seaad_pearson"] = d1.loc[dist, "SEA-AD_corr"]
cat["seaad_spearman"] = d1.loc[dist, "SEA-AD_spearman"]
cat["prioritised_2026"] = d1.loc[dist, "prioritised_2026"].fillna("no")
cat["prioritisation_support"] = d1.loc[dist, "prioritisation_support"].fillna("")
cat["seaad_support_B"] = cat.prioritisation_support.str.contains("SEA-AD")


def po_cov(po):
    """observed-information covariance of the phenotype/covariate coefficients (original scale)."""
    n = len(po.par); H = np.zeros((n, n)); off = np.zeros(len(po.yk)); h = 1e-5
    for i in range(n):
        e = np.zeros(n); e[i] = h
        H[i] = (_negll(po.par + e, po.yk, po.Xs, po.K, off)[1] - _negll(po.par - e, po.yk, po.Xs, po.K, off)[1]) / (2 * h)
    H = (H + H.T) / 2; C = np.linalg.inv(H)[po.K - 1:, po.K - 1:]
    return C / np.outer(po.sd, po.sd)


def joint(k, sub, ph, profile=True):
    names = ph + COV; X = D.loc[sub, names].values.astype(float); y = S.loc[k, sub].values.astype(float)
    po = PO(y, X); ll, beta, ok = po.full(); C = po_cov(po); r = dict(component=k, n=len(sub), converged=bool(ok))
    for j, t in enumerate(["cerad", "braak", "mci", "dem"]):
        lo, hi = po.ci_profile(j) if profile else (beta[j] - 1.96 * np.sqrt(C[j, j]), beta[j] + 1.96 * np.sqrt(C[j, j]))
        r[f"{t}_OR"], r[f"{t}_lo"], r[f"{t}_hi"] = np.exp(beta[j]), np.exp(lo), np.exp(hi)
        r[f"{t}_wald_p"] = 2 * ss.norm.sf(abs(beta[j]) / np.sqrt(C[j, j]))
    for b, cols in BLK.items():
        keep = [i for i in range(X.shape[1]) if i not in cols]
        ll0 = PO(y, X[:, keep]).full()[0]; r[f"{b}_block_LR_p"] = ss.chi2.sf(max(2 * (ll - ll0), 0), len(cols))
    L = np.zeros(len(beta)); L[0], L[3] = 1, -1                     # log OR(CERAD) - log OR(AD dementia)
    r["contrast_logOR"] = float(L @ beta); r["contrast_se"] = float(np.sqrt(L @ C @ L))
    r["contrast_p"] = 2 * ss.norm.sf(abs(r["contrast_logOR"]) / r["contrast_se"])
    # rank-INT OLS of the same model (secondary)
    T, _, dfr = ols(rint(y), X, names)
    for j, t in enumerate(["cerad", "braak", "mci", "dem"]):
        r[f"{t}_SD"], r[f"{t}_SD_lo"], r[f"{t}_SD_hi"], r[f"{t}_SD_p"] = T.iloc[j][["beta", "lo", "hi", "p"]]
    return r


def pattern(r, pre=""):
    c = not (r[pre + "cerad_lo"] <= 1 <= r[pre + "cerad_hi"]); dm = not (r[pre + "dem_lo"] <= 1 <= r[pre + "dem_hi"])
    return {(True, False): "plaque-only", (False, True): "dementia-only", (True, True): "both", (False, False): "neither"}[(c, dm)]


rows = []
for i, k in enumerate(dist):
    r = joint(k, SUB124, PH124); r["pattern"] = pattern(r); rows.append(r)
    if i % 25 == 0:
        print(f"  fitted {i + 1}/193", flush=True)
J = pd.DataFrame(rows).set_index("component")
A = cat.join(J)
# orientation for display: sign so that disease (CERAD AD + AD dementia, joint) goes with a LOWER score, like C172/C277
A["flip"] = (np.log(A.cerad_OR) + np.log(A.dem_OR)) > 0
A.to_csv(f"{OUT}/s01_joint_all_distributed.csv", float_format="%.6g")
print("non-converged joint fits:", int((~A.converged).sum()))

# ---------------- named set ----------------------------------------------------------------------------------------
REF = [172, 277]
NEURO_CERAD = list(A.index[A.neuron_led & (A.cerad_PO1_q <= 1.5 * A.loc[172, "cerad_PO1_q"])])
NEURO_CERAD_p = list(A.index[A.neuron_led & (A.cerad_PO1_p <= 1.5 * A.loc[172, "cerad_PO1_p"])])
NEURO_COG = list(A.index[A.neuron_led & (A.cogdx_PO1_q <= 1.5 * A.loc[277, "cogdx_PO1_q"])])
NEURO_COG_p = list(A.index[A.neuron_led & (A.cogdx_PO1_p <= 1.5 * A.loc[277, "cogdx_PO1_p"])])
NEURO_COG_any = list(A.index[A.neuron_led & (A.cogdx_PO1_q < 0.05)])
TASK = [134, 542, 204, 231, 519, 454]; GLIA = [31, 32, 48, 515, 500]
print(f"neuron-led within 1.5x of C172 CERAD PO1 q ({A.loc[172,'cerad_PO1_q']:.6g}): {NEURO_CERAD}; by p: {NEURO_CERAD_p}")
print(f"neuron-led within 1.5x of C277 cogdx PO1 q ({A.loc[277,'cogdx_PO1_q']:.6g}): {NEURO_COG}; by p: {NEURO_COG_p}")
print(f"neuron-led with cogdx PO1 q < 0.05 (no neuron-led component is within 1.5x of C277, so these are listed): {NEURO_COG_any}")
NAMED = list(dict.fromkeys(REF + TASK + NEURO_CERAD + NEURO_CERAD_p + NEURO_COG + NEURO_COG_any + GLIA))
pd.Series(NAMED, name="component").to_csv(f"{OUT}/s01_named_set.csv", index=False)

# all-276 coding for the named set
r276 = []
for k in NAMED:
    r = joint(k, SUB276, PH276); r = {("a276_" + kk if kk != "component" else kk): v for kk, v in r.items()}
    r["a276_pattern"] = pattern(r, "a276_"); r276.append(r)
R276 = pd.DataFrame(r276).set_index("component")

# stacked rank-INT comparison with C172 and C277 (v05 model; donor-clustered SE); rival oriented as above
d = D.loc[SUB124].copy(); d["donor"] = np.arange(len(d))
FORM = "y ~ (cerad_ad + braak + cogdx_mci + cogdx_ad + age + pmi + ros + sex) * cR"
st = []
for k in NAMED:
    sgn = -1.0 if A.loc[k, "flip"] else 1.0
    for ref in REF:
        if k == ref:
            continue
        a = d.assign(y=rint(S.loc[ref, SUB124].values), cR=0.0); b = d.assign(y=rint(sgn * S.loc[k, SUB124].values), cR=1.0)
        m = smf.ols(FORM, pd.concat([a, b], ignore_index=True)).fit(cov_type="cluster", cov_kwds={"groups": np.r_[a.donor, b.donor]})
        L = np.zeros(len(m.params)); nm = list(m.params.index); L[nm.index("cerad_ad:cR")] = 1; L[nm.index("cogdx_ad:cR")] = -1
        t = m.t_test(L)
        st.append(dict(component=k, reference=ref, flipped=bool(A.loc[k, "flip"]),
                       cerad_diff=m.params["cerad_ad:cR"], cerad_diff_p=m.pvalues["cerad_ad:cR"],
                       dem_diff=m.params["cogdx_ad:cR"], dem_diff_p=m.pvalues["cogdx_ad:cR"],
                       pattern_diff=float(np.squeeze(t.effect)), pattern_diff_p=float(np.squeeze(t.pvalue))))
ST = pd.DataFrame(st)
ST.to_csv(f"{OUT}/s01_stacked_vs_C172_C277.csv", index=False, float_format="%.6g")
NM = A.loc[NAMED].join(R276)
NM.to_csv(f"{OUT}/s01_named_joint.csv", float_format="%.6g")

# ---------------- summary ------------------------------------------------------------------------------------------
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
f = lambda r, t: f"{r[t+'_OR']:.3f} [{r[t+'_lo']:.3f}, {r[t+'_hi']:.3f}]"
print("\nNamed set, joint PO model (n 254; CERAD + Braak + cogdx + age, PMI, cohort, sex; profile 95% CI):")
for k in NAMED:
    r = NM.loc[k]
    print(f"C{k:<4d} {r.dominant_cell_type} ({r.dominant_abs_share:.2f}; 2nd {r.second_cell_type} {r.second_over_first:.2f}) "
          f"| single PO1: CERAD q {r.cerad_PO1_q:.4g} (rank {r.cerad_rank_dist}), cogdx q {r.cogdx_PO1_q:.4g} (rank {r.cogdx_rank_dist}) "
          f"| joint CERAD {f(r,'cerad')}; Braak {f(r,'braak')}; MCI {f(r,'mci')}; AD dem {f(r,'dem')} "
          f"| CERAD-vs-dem Wald p {r.contrast_p:.3g} | {r.pattern} | all-276 {r.a276_pattern} | SEA-AD {'B' if r.seaad_support_B else '-'} "
          f"(partner {r.seaad_partner:.0f}, r {r.seaad_pearson:.3f}, rho {r.seaad_spearman:.3f})")
print("\nStacked comparisons (rank-INT; (CERAD - AD dementia) difference, component minus reference):")
for _, s in ST.iterrows():
    print(f"  C{int(s.component):<4d} vs C{int(s.reference)}: CERAD diff {s.cerad_diff:+.3f} (p {s.cerad_diff_p:.3g}); "
          f"AD-dem diff {s.dem_diff:+.3f} (p {s.dem_diff_p:.3g}); pattern diff {s.pattern_diff:+.3f} (p {s.pattern_diff_p:.3g})")

print("\nPattern counts, joint PO model, all 193 distributed:")
print(pd.crosstab(A.pattern, A.dominant_cell_type.map(lambda c: "neuron-led" if c in ("Exc", "Inh") else "glia-led" if c != "none" else "no cell type"), margins=True).to_string())
sup = A[(A.cerad_PO1_q < 0.05) | (A.cogdx_PO1_q < 0.05)]
print(f"\nPhenotype-supported (CERAD or cogdx PO1 q < 0.05; n {len(sup)}):")
print(pd.crosstab(sup.pattern, sup.dominant_cell_type.map(lambda c: "neuron-led" if c in ("Exc", "Inh") else "glia-led" if c != "none" else "no cell type"), margins=True).to_string())
for pat in ("plaque-only", "dementia-only"):
    x = A[A.pattern == pat].sort_values("cerad_wald_p" if pat == "plaque-only" else "dem_wald_p")
    print(f"\n{pat} ({len(x)}):")
    print(x[["dominant_cell_type", "dominant_abs_share", "cerad_PO1_q", "cogdx_PO1_q", "cerad_OR", "cerad_lo", "cerad_hi",
             "dem_OR", "dem_lo", "dem_hi", "mci_OR", "contrast_p", "seaad_support_B", "prioritised_2026"]].to_string(float_format=lambda v: f"{v:.4g}"))
print("\nWithin-component CERAD-vs-AD-dementia Wald contrast, all 193 distributed: p < 0.05 in",
      int((A.contrast_p < 0.05).sum()), f"(expected under no difference about {0.05 * len(A):.1f}); smallest:")
print(A.sort_values("contrast_p")[["dominant_cell_type", "cerad_OR", "dem_OR", "contrast_p", "pattern"]].head(8).to_string(float_format=lambda v: f"{v:.4g}"))
print("non-converged component(s):", list(A.index[~A.converged]))
