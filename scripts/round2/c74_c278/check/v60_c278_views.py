"""Verify C278 task 5 numbers with own code: leading genes in OPC/Oli (all donors and within sex), aggregates (two
definitions), measured female-minus-male differences (bootstrap CI + Welch t + MW), leading genes in the other cell
types, corrected cell-type loadings, and the M1 rank-INT effect (own OLS). Outputs out/v60_* (aggregates)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
from vc_common import *
K = 278; a = score(K); fem = female(); ex = excl("RNA"); keep = ~ex
R = rna(); L = loadings(K, "RNA"); B = ct_corrected().loc[K]
print("corrected cell-type loadings:", B.round(3).to_dict())
pip = L.P.values > 0.5; print("C278 PIP>0.5 genes:", int(pip.sum()))
rows = []
rng = np.random.default_rng(5)
def rec(view, y):
    ok = keep & np.isfinite(y)
    r = rho_ci(y[ok], a[ok], B=1500, seed=1); rm = ss.spearmanr(y[ok & ~fem], a[ok & ~fem])[0]; rf = ss.spearmanr(y[ok & fem], a[ok & fem])[0]
    yf, ym = y[ok & fem], y[ok & ~fem]
    bs = [rng.choice(yf, len(yf)).mean() - rng.choice(ym, len(ym)).mean() for _ in range(3000)]
    rows.append(dict(view=view, n=int(ok.sum()), nF=len(yf), nM=len(ym), rho=r["rho"], lo=r["lo"], hi=r["hi"], rho_M=rm, rho_F=rf,
                     FmM=yf.mean() - ym.mean(), FmM_lo=np.percentile(bs, 2.5), FmM_hi=np.percentile(bs, 97.5),
                     p_welch=ss.ttest_ind(yf, ym, equal_var=False).pvalue, p_MW=ss.mannwhitneyu(yf, ym).pvalue))
for ct in CTS:
    c = CTS.index(ct)
    for g in ("FKBP5", "SGK1", "ZBTB16"):
        rec(f"{g} {ct}", np.log2(np.asarray(R[c, :, GENES.index(g)], float) + 1))
lead = np.isin(GENES, ["FKBP5", "SGK1", "ZBTB16"])
for ct in ("OPC", "Oli"):
    c = CTS.index(ct); E = np.log2(np.asarray(R[c], float) + 1)
    for lab, sel in (("all PIP", pip), ("PIP minus 3 leading", pip & ~lead)):
        Es = E[:, sel]; sd = Es.std(0); kk = sd > 0; Z = (Es[:, kk] - Es[:, kk].mean(0)) / sd[kk]; w = L.X.values[sel][kk] * B[ct]
        rec(f"agg weighted {lab} {ct} ({kk.sum()} genes)", Z @ w / np.abs(w).sum())
        Rk = np.apply_along_axis(ss.rankdata, 0, Es[:, kk]); Rk = (Rk - Rk.mean(0)) / Rk.std(0)
        rec(f"agg sign-only rank {lab} {ct}", Rk @ np.sign(w) / len(w))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/v60_c278_views.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250); print(T.to_string(index=False, float_format=lambda v: f"{v:+.3g}"))
# own rank-INT OLS, M1 (age 90+ -> 90, PMI, cohort)
cl = clinical(); age = pd.to_numeric(cl.age_death.astype(str).replace("90+", "90"), errors="coerce").clip(upper=90).values
pmi = cl.pmi.astype(float).values; ros = (cl.Study == "ROS").astype(float).values
y = ss.norm.ppf((ss.rankdata(a) - 0.375) / (len(a) + 0.25))
X = np.column_stack([np.ones(276), fem.astype(float), age, pmi, ros]); ok = np.isfinite(X).all(1)
print("M1 rank-INT: missing covariates", int((~ok).sum()))
if (~ok).any():  # median-impute as a check (count reported)
    X2 = X.copy()
    for j in range(2, 5): X2[~np.isfinite(X2[:, j]), j] = np.nanmedian(X2[:, j])
else: X2 = X
f = sm.OLS(y, X2).fit(); print(f"M1 rank-INT female effect {f.params[1]:+.3f} SD [{f.conf_int()[1][0]:+.3f}, {f.conf_int()[1][1]:+.3f}], p {f.pvalues[1]:.2e}")
f0 = sm.OLS(y, X2[:, :2]).fit(); print(f"M0 rank-INT female effect {f0.params[1]:+.3f} [{f0.conf_int()[1][0]:+.3f}, {f0.conf_int()[1][1]:+.3f}] p {f0.pvalues[1]:.2e}; "
                                         f"MW p {ss.mannwhitneyu(a[fem], a[~fem]).pvalue:.2e}; AUC F>M {auc(a[fem], a[~fem]):.3f}")
np.save(f"{CACHE}/c278_M1_design.npy", X2)
