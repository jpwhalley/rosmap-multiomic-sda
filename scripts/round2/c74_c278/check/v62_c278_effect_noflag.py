"""Sensitivity not in the first implementation's forest: C278 M1 effect (rank-INT OLS and PO) excluding the four sex-discordant
identity-flagged samples (their snRNA or H3K9ac is another person's; scores were fitted with those inputs)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, numpy as np, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
from vc_common import *
src = open(os.path.join(os.path.dirname(__file__), "v61_c278_po.py")).read().split("for lab, X in")[0]
F = idflags(); m = ~(F["RNA-1"] | F["RNA-2"] | F["CHIP-1"] | F["CHIP-2"])
a_all = score(278); X_all = np.load(f"{CACHE}/c278_M1_design.npy")
a = a_all[m]; X2 = X_all[m]
y = ss.norm.ppf((ss.rankdata(a) - 0.375) / (len(a) + 0.25)); f = sm.OLS(y, X2).fit()
print(f"n {m.sum()}: M1 rank-INT female {f.params[1]:+.3f} [{f.conf_int()[1][0]:+.3f}, {f.conf_int()[1][1]:+.3f}], p {f.pvalues[1]:.2e}")
ns = dict(np=np, ss=ss, so=None, score=lambda k: a, female=lambda: X2[:, 1] > 0, CACHE=CACHE)
from scipy import optimize as so
yr = ss.rankdata(a, method="dense").astype(int) - 1; J = yr.max() + 1
exec(src.split("yr = ss.rankdata")[1].split("\n", 1)[1], globals())
X = (X2[:, 1:] - np.r_[0, X2[:, 2:].mean(0)]) / np.r_[1, X2[:, 2:].std(0)]
r = fit(X); b = r.x[J - 1]; lr0 = 2 * (fit(X, 0.0) - r.fun)
print(f"n {m.sum()}: M1 PO OR female {np.exp(b):.3f}; LRT p {ss.chi2.sf(lr0, 1):.2e}")
