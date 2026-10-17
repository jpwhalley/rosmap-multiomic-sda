"""VERIFY part 5: adjusted phenotype calls with independent code and two models.
(1) Own proportional-odds implementation via statsmodels OrderedModel (score as ordinal outcome, one level per distinct
    value; phenotype + age(90+ -> 90) + PMI + cohort [+ sex]; LRT) on every component whose structure PO1 p < 0.05 in any
    phenotype plus a random 40 others -> checks the structure p-values directly.
(2) A different reasonable model for all 547 x 3: phenotype as outcome. Sex and CERAD AD-vs-non-AD: logistic regression on
    rank-normal score + covariates (LRT, 1 df). cogdx 1/2/4: multinomial logit (LRT, 2 df). BH per phenotype over 547.
Unadjusted: Mann-Whitney / Kruskal-Wallis, BH over 547 -> compare with Data S1 calls.
Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import time, warnings, numpy as np, pandas as pd
from scipy import stats as ss
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
warnings.filterwarnings("ignore")
t0 = time.time()
V = cfg.out_dir(cfg.WORK, "revision_analyses", "verify_structure"); cfg.out_dir(V, "out")
cfg.out_dir(V, "private")
ROS = cfg.ROS
AUD = cfg.AUD
DS1 = f"{cfg.TABLES}/Data_S1_corrected_547.csv"
don = [l.strip() for l in open(f"{ROS}/sda/md_samples.txt") if l.strip()]
S = pd.read_csv(f"{ROS}/sda/samples_final.csv", index_col=0); S.columns = S.columns.astype(str); S = S.drop(index=520)
for k in (547, 548): S.loc[k] = np.load(f"{AUD}/c547/C{k}_A.npy")
cl = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv"); cl.index = cl.projid.astype(int).astype(str); cl = cl.loc[don]
age = pd.to_numeric(cl.age_death.astype(str).str.replace("+", "", regex=False)).values.astype(float)
cov = pd.DataFrame({"age": age, "pmi": cl.pmi.values.astype(float), "ros": (cl.Study == "ROS").astype(float).values,
                    "sex": cl.msex.values.astype(float)}, index=don)
cer = cl.ceradsc.map({1: 1, 2: 1, 3: 0, 4: 0}).values.astype(float)
cog = cl.cogdx.values
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum(1, np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]); return q
def rn(x):
    return ss.norm.ppf((ss.rankdata(x) - 0.5) / len(x))

# ---------------- unadjusted
U = {}
U["Sex"] = [ss.mannwhitneyu(S.loc[k].values[cov.sex == 1], S.loc[k].values[cov.sex == 0]).pvalue for k in S.index]
U["ceradsc"] = [ss.mannwhitneyu(S.loc[k].values[cer == 1], S.loc[k].values[cer == 0]).pvalue for k in S.index]
U["cogdx"] = [ss.kruskal(*[S.loc[k].values[cog == g] for g in (1, 2, 4)]).pvalue for k in S.index]
ds1 = pd.read_csv(DS1, index_col=0)
for ph in U:
    q = bh(U[ph]); mine = set(S.index[q < 0.05]); theirs = set(ds1.index[ds1[ph].notna()])
    print(f"unadjusted {ph}: my calls {len(mine)}, Data S1 {len(theirs)}, sym diff {sorted(mine ^ theirs)}")

# ---------------- alternative model: phenotype as outcome
SEP = []
def llf_logit(y, X):
    try:
        return sm.Logit(y, X).fit(disp=0, method="newton", maxiter=200).llf
    except Exception:
        SEP.append(1)   # (quasi-)complete separation: likelihood -> 0 deviance; LRT p effectively 0
        return 0.0
def llf_mn(y, X):
    return sm.MNLogit(y, X).fit(disp=0, method="newton", maxiter=200).llf
alt = {ph: [] for ph in U}
Xs = sm.add_constant(cov[["age", "pmi", "ros"]].values)
Xc = sm.add_constant(cov[["age", "pmi", "ros", "sex"]].values)
sub = np.isin(cog, [1, 2, 4]); ycog = pd.Series(cog[sub]).map({1: 0, 2: 1, 4: 2}).values
l0s = llf_logit(cov.sex.values, Xs); l0c = llf_logit(cer, Xc); l0g = llf_mn(ycog, Xc[sub])
for k in S.index:
    z = rn(S.loc[k].values)
    alt["Sex"].append(ss.chi2.sf(2 * (llf_logit(cov.sex.values, np.column_stack([Xs, z])) - l0s), 1))
    alt["ceradsc"].append(ss.chi2.sf(2 * (llf_logit(cer, np.column_stack([Xc, z])) - l0c), 1))
    alt["cogdx"].append(ss.chi2.sf(2 * (llf_mn(ycog, np.column_stack([Xc[sub], rn(S.loc[k].values[sub])])) - l0g), 2))
print(f"separation cases (p set to ~0): {len(SEP)}"); print(f"alt model done [{time.time()-t0:.0f}s]", flush=True)
po = pd.read_csv(f"{cfg.WORK}/covariates/po_tests_547.csv", header=[0, 1], index_col=0)
A = pd.DataFrame(index=S.index)
for ph in U:
    A[f"alt_{ph}_p"] = alt[ph]; A[f"alt_{ph}_q"] = bh(alt[ph]); A[f"po1_{ph}_q"] = po[(ph, "PO1_q")].values; A[f"po1_{ph}_p"] = po[(ph, "PO1_p")].values
    mine = set(A.index[A[f"alt_{ph}_q"] < 0.05]); theirs = set(A.index[A[f"po1_{ph}_q"] < 0.05])
    print(f"adjusted {ph}: alt-model calls {len(mine)}, PO1 calls {len(theirs)}, both {len(mine & theirs)}, only alt {sorted(mine - theirs)}, only PO1 {sorted(theirs - mine)}; "
          f"Spearman(-log p) {ss.spearmanr(A[f'alt_{ph}_p'], A[f'po1_{ph}_p'])[0]:.3f}")

A.to_csv(f"{V}/out/v05_pheno_547.csv")
print(f"done [{time.time()-t0:.0f}s]")
