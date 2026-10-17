# Cumulative-logit (proportional-odds) regression of a continuous score treated as ordinal (one category per distinct
# value): the covariate-adjusted generalisation of the Wilcoxon / Mann-Whitney and Kruskal-Wallis tests.
# Fit by L-BFGS with an analytic gradient; thresholds kept ordered by theta_1 = a_1, theta_j = theta_{j-1} + exp(a_j).
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy import stats as _ss

def _negll(par, yk, X, K):
    a = par[:K - 1]; b = par[K - 1:]
    theta = np.cumsum(np.concatenate([[a[0]], np.exp(a[1:])]))           # K-1 increasing cut-points
    eta = X @ b if X.shape[1] else np.zeros(len(yk))
    th_hi = np.concatenate([theta, [np.inf]])[yk]                         # theta_y   (y = 0..K-1)
    th_lo = np.concatenate([[-np.inf], theta])[yk]                        # theta_{y-1}
    Fh = expit(th_hi - eta); Fl = expit(th_lo - eta)
    p = np.clip(Fh - Fl, 1e-300, None)
    fh = Fh * (1 - Fh); fl = Fl * (1 - Fl)
    ll = np.log(p).sum()
    g_th = np.zeros(K - 1)
    hi = yk < K - 1; lo = yk > 0
    np.add.at(g_th, yk[hi], fh[hi] / p[hi])
    np.add.at(g_th, yk[lo] - 1, -fl[lo] / p[lo])
    g_b = -(X * ((fh - fl) / p)[:, None]).sum(0) if X.shape[1] else np.zeros(0)
    # chain rule to a: dtheta_j/da_1 = 1 (all j); dtheta_j/da_m = exp(a_m) for j >= m (m >= 2)
    rc = np.cumsum(g_th[::-1])[::-1]                                       # sum_{j>=m} g_th_j
    g_a = rc.copy(); g_a[1:] *= np.exp(a[1:])
    return -ll, -np.concatenate([g_a, g_b])

def po_fit(score, X):
    """score: 1-D array; X: n x p covariate matrix (no intercept). Returns (loglik, beta, converged).
    Columns are centred and scaled internally (the likelihood is invariant); beta is returned on the original scale."""
    X = np.asarray(X, float); mu = X.mean(0) if X.shape[1] else np.zeros(0); sd = X.std(0) if X.shape[1] else np.ones(0)
    sd = np.where(sd > 0, sd, 1.0); X = (X - mu) / sd
    u, yk = np.unique(score, return_inverse=True); K = len(u); n = len(score)
    cum = np.cumsum(np.bincount(yk, minlength=K))[:-1] / n
    th0 = np.log(cum / (1 - cum))
    a0 = np.concatenate([[th0[0]], np.log(np.diff(th0))])
    par0 = np.concatenate([a0, np.zeros(X.shape[1])])
    r = minimize(_negll, par0, args=(yk, X, K), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, maxfun=40000, gtol=1e-8, ftol=1e-14))
    return -r.fun, r.x[K - 1:] / sd, bool(r.success)

def po_lrt(score, Xph, Xcov):
    """LR test of the phenotype columns Xph given Xcov. Returns p, beta_ph, converged."""
    Xf = np.column_stack([Xph, Xcov]) if Xcov.shape[1] else Xph
    llf, bf, cf = po_fit(score, Xf)
    llr, _, cr = po_fit(score, Xcov)
    lr = max(2 * (llf - llr), 0.0)
    return _ss.chi2.sf(lr, Xph.shape[1]), bf[:Xph.shape[1]], cf and cr
