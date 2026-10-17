"""(2) Plaque burden vs clinical diagnosis: associate the C172 and C277 scores with CERAD, Braak and cogdx, alone and
jointly, so that it is clear which phenotype carries the association when the others are in the model.
Global AD pathology (gpath) is NOT on this host: the AD Knowledge Portal ROSMAP_clinical.csv has CERAD, Braak, cogdx,
dcfdx and MMSE only (gpath is in the RADC 'basic' dataset). Last-visit MMSE is used as a continuous cognition
sensitivity instead; no pathology composite is invented.
Models: every model includes the dossier's M1 covariates (age at death 90+ -> 90, PMI, cohort, sex).
  primary set   : cogdx in {1 NCI, 2 MCI, 4 AD dementia} (the paper's cognition set), n = 254
  sensitivity   : all 276 donors, cognition coded as any dementia (cogdx 4/5/6) and any MCI (2/3) vs NCI
Phenotype codings: CERAD AD (definite/probable) vs not (the paper's coding) or CERAD plaque burden 1-4 (linear);
Braak stage 0-6 (per stage); cogdx MCI vs NCI and AD dementia vs NCI.
Estimates: rank-INT OLS (score -> rank-based inverse normal over the analysed donors; beta in SD of the score, t CI;
block F test) and proportional-odds regression of the score (OR = odds of a HIGHER score per unit/index group;
profile-likelihood CI; block LR test). Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import itertools, numpy as np, pandas as pd
from scipy import stats as ss
from nc_common import *
from nc_po import PO

S = scores547(); D = full_frame()
COV = ["age", "pmi", "ros", "sex"]
D["mci_any"] = D.cogdx.isin([2, 3]).astype(float)
D["mmse_per5"] = -D.mmse_lv / 5.0                       # per 5 points LOWER MMSE (worse cognition)
SETS = {"cogdx124": D.index[D.cogdx.isin([1, 2, 4])], "all276": D.index}
BLOCKS = {
    "cogdx124": {"CERAD": ["cerad_ad"], "Braak": ["braak"], "cogdx": ["cogdx_mci", "cogdx_ad"]},
    "all276": {"CERAD": ["cerad_ad"], "Braak": ["braak"], "cogdx": ["mci_any", "dementia_any"]},
}
ALT = {"CERADburden": ["cerad_burden"], "MMSE": ["mmse_per5"]}

def fit(k, sub, blocks, label):
    y = S.loc[k, sub].values.astype(float); yr = rint(y)
    names = sum(blocks.values(), []) + COV
    X = D.loc[sub, names].values.astype(float)
    T, _, _ = ols(yr, X, names)
    po = PO(y, X); ll, beta, ok = po.full()
    rows = []
    for bname, cols in blocks.items():
        others = [c for c in names if c not in cols]
        Xr = D.loc[sub, others].values.astype(float)
        F, pF, pr2 = nested_F(yr, X, Xr)
        ll0 = PO(y, Xr).full()[0]; pLR = ss.chi2.sf(max(2 * (ll - ll0), 0), len(cols))
        for c in cols:
            j = names.index(c); lo, hi = po.ci_profile(j)
            rows.append(dict(component=k, set=label, model="+".join(blocks), n=len(sub), block=bname, term=c,
                             beta_SD=T.loc[c, "beta"], SD_lo=T.loc[c, "lo"], SD_hi=T.loc[c, "hi"], p_term_ols=T.loc[c, "p"],
                             block_F_p=pF, block_partial_R2=pr2, OR=np.exp(beta[j]), OR_lo=np.exp(lo), OR_hi=np.exp(hi),
                             block_LR_p=pLR, converged=ok))
    return rows

rows = []
for k in (172, 277):
    for label, sub in SETS.items():
        B = BLOCKS[label]
        for r in (1, 2, 3):
            for combo in itertools.combinations(B, r):
                rows += fit(k, list(sub), {b: B[b] for b in combo}, label)
        # alternative codings, joint
        rows += fit(k, list(sub), {"CERAD": ALT["CERADburden"], "Braak": B["Braak"], "cogdx": B["cogdx"]}, label + "|CERADburden")
        rows += fit(k, list(sub), {"CERAD": B["CERAD"], "Braak": B["Braak"], "MMSE": ALT["MMSE"]}, label + "|MMSE")
        rows += fit(k, list(sub), {"MMSE": ALT["MMSE"]}, label + "|MMSE")
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/n03_joint_models.csv", index=False, float_format="%.4g")

# phenotype inter-correlation and collinearity in the primary set
s = D.loc[SETS["cogdx124"]]
print("Spearman among phenotypes (n = %d; cogdx 1/2/4):" % len(s))
print(s[["cerad_burden", "braak", "cogdx", "mmse_lv"]].corr("spearman").round(2).to_string())
Xv = s[["cerad_ad", "braak", "cogdx_mci", "cogdx_ad"] + COV].values.astype(float)
vif = {c: 1 / (1 - (1 - ((Xv[:, i] - np.column_stack([np.ones(len(s)), np.delete(Xv, i, 1)]) @ np.linalg.lstsq(np.column_stack([np.ones(len(s)), np.delete(Xv, i, 1)]), Xv[:, i], rcond=None)[0]) ** 2).sum() / ((Xv[:, i] - Xv[:, i].mean()) ** 2).sum()))
       for i, c in enumerate(["cerad_ad", "braak", "cogdx_mci", "cogdx_ad"])}
print("VIF in the joint model:", {k_: round(v, 2) for k_, v in vif.items()})

pd.set_option("display.width", 250)
show = ["component", "set", "model", "n", "term", "beta_SD", "SD_lo", "SD_hi", "p_term_ols", "block_F_p", "OR", "OR_lo", "OR_hi", "block_LR_p"]
for k in (172, 277):
    print(f"\n==== C{k}")
    print(T[T.component == k][show].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
