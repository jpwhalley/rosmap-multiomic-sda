"""Own proportional-odds (cumulative logit) fit for the C278 sex effect, M0 and M1, by direct maximum likelihood
(thresholds parameterised as cumulative exp-increments; scipy BFGS on the analytic negative log-likelihood).
Profile CI for the female coefficient by root-finding on the likelihood-ratio statistic."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, numpy as np
from scipy import optimize as so
sys.path.insert(0, os.path.dirname(__file__))
from vc_common import *
a = score(278); X2 = np.load(f"{CACHE}/c278_M1_design.npy"); fem = female()
yr = ss.rankdata(a, method="dense").astype(int) - 1; J = yr.max() + 1
def nll(par, X):
    th = np.concatenate([[par[0]], par[0] + np.cumsum(np.exp(par[1:J - 1]))]); b = par[J - 1:]
    eta = X @ b
    up = np.where(yr < J - 1, 1 / (1 + np.exp(-(np.append(th, np.inf)[yr] - eta))), 1.0)
    lo = np.where(yr > 0, 1 / (1 + np.exp(-(np.append(-np.inf, th)[yr] - eta))), 0.0)
    return -np.sum(np.log(np.clip(up - lo, 1e-300, None)))
def fit(X, fix=None):
    p0 = np.concatenate([[-3], np.full(J - 2, np.log(6 / J)), np.zeros(X.shape[1])])
    if fix is not None:
        f = lambda q: nll(np.concatenate([q[:J - 1], [fix], q[J - 1:]]), X)
        r = so.minimize(f, np.delete(p0, J - 1), method="L-BFGS-B", options=dict(maxiter=20000, maxfun=10**6)); return r.fun
    r = so.minimize(lambda q: nll(q, X), p0, method="L-BFGS-B", options=dict(maxiter=20000, maxfun=10**6)); return r
for lab, X in (("M0", X2[:, 1:2]), ("M1", (X2[:, 1:] - np.r_[0, X2[:, 2:].mean(0)]) / np.r_[1, X2[:, 2:].std(0)])):
    r = fit(X); b = r.x[J - 1]; ll = r.fun
    lr0 = 2 * (fit(X, 0.0) - ll)
    g = lambda v: 2 * (fit(X, v) - ll) - 3.841
    lo = so.brentq(g, b - 1.5, b); hi = so.brentq(g, b, b + 1.5)
    print(f"{lab}: OR female {np.exp(b):.3f} [{np.exp(lo):.3f}, {np.exp(hi):.3f}] (profile); LRT p {ss.chi2.sf(lr0, 1):.2e}; converged {r.success}")
