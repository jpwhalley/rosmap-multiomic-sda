"""Is the C172 (plaques) vs C277 (clinical dementia) dissociation formally supported? Stacked rank-INT OLS of both
scores with component x phenotype interactions, donor-clustered SEs; plus a paired donor bootstrap of the joint-model
contrasts. Also alternative Braak/CERAD codings. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import warnings, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
from scipy import stats as ss
from v_common import *
F = pd.read_csv(f"{CACHE}/frame.csv"); d = F[F.cogdx.isin([1, 2, 4])].reset_index(drop=True); d["donor"] = np.arange(len(d))
def rintb(x): x = np.asarray(x, float); return ss.norm.ppf((ss.rankdata(x) - 0.375) / (len(x) + 0.25))
d["y172"] = rintb(d.s172); d["y277"] = rintb(d.s277)
long = pd.concat([d.assign(y=d.y172, c277=0.0), d.assign(y=d.y277, c277=1.0)], ignore_index=True)
f = "y ~ (cerad_ad + braak + mci + addem + age + pmi + ros + male) * c277"
m = smf.ols(f, long).fit(cov_type="cluster", cov_kwds={"groups": long.donor})
print("Stacked joint model (n 254 donors x 2 scores; donor-clustered SE). Interaction = C277 minus C172:")
for t in ["cerad_ad:c277", "braak:c277", "mci:c277", "addem:c277"]:
    print(f"  {t:14s} {m.params[t]:+.3f} [{m.conf_int().loc[t, 0]:+.3f}, {m.conf_int().loc[t, 1]:+.3f}] p {m.pvalues[t]:.3g}")
# contrast of contrasts: (CERAD - ADdem) differs between components
L = np.zeros(len(m.params)); names = list(m.params.index)
L[names.index("cerad_ad:c277")] = 1; L[names.index("addem:c277")] = -1
t = m.t_test(L); print(f"  (CERAD - AD dementia) C277 minus C172: {float(np.squeeze(t.effect)):+.3f}, p {float(np.squeeze(t.pvalue)):.3g}")
# paired bootstrap of the joint-model coefficients
rng = np.random.default_rng(11); B = []
X = ["cerad_ad", "braak", "mci", "addem", "age", "pmi", "ros", "male"]
for b in range(4000):
    ii = rng.integers(0, len(d), len(d)); db = d.iloc[ii]
    Xb = np.column_stack([np.ones(len(db)), db[X].values])
    c1 = np.linalg.lstsq(Xb, rintb(db.s172), rcond=None)[0]; c2 = np.linalg.lstsq(Xb, rintb(db.s277), rcond=None)[0]
    B.append([c1[1], c1[4], c2[1], c2[4]])
B = np.array(B)
for lab, v in (("C172 CERAD", B[:, 0]), ("C172 AD dementia", B[:, 1]), ("C277 CERAD", B[:, 2]), ("C277 AD dementia", B[:, 3]),
               ("CERAD: C172 - C277", B[:, 0] - B[:, 2]), ("AD dementia: C277 - C172", B[:, 3] - B[:, 1])):
    print(f"  bootstrap {lab:26s} 95% CI [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]; P(>0) {np.mean(v > 0):.3f}")
# alternative codings in the joint model (rank-INT OLS)
print("\nJoint-model alternatives (rank-INT OLS, M1 covariates, n 254):")
d["braak_hi"] = (d.braak >= 5).astype(float); d["braak3"] = pd.cut(d.braak, [-1, 2, 4, 6], labels=False).astype(float)
for k in (172, 277):
    for lab, fm in (("CERAD AD + Braak>=5 + cogdx", "cerad_ad + braak_hi + mci + addem"),
                    ("CERAD 4-level + Braak 3-level + cogdx", "cerad_burden + braak3 + mci + addem"),
                    ("CERAD AD + Braak + AD dementia only (MCI pooled w NCI)", "cerad_ad + braak + addem")):
        mm = smf.ols(f"y{k} ~ {fm} + age + pmi + ros + male", d).fit()
        print(f"  C{k} {lab}: " + "; ".join(f"{t} {mm.params[t]:+.2f} [{mm.conf_int().loc[t,0]:+.2f}, {mm.conf_int().loc[t,1]:+.2f}] p {mm.pvalues[t]:.2g}" for t in fm.split(" + ")))
