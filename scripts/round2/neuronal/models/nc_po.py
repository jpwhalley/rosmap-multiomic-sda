"""Proportional-odds (cumulative-logit) model with the component score as the ordinal response, copied verbatim from
component_dossier/scripts/02_effects_ci.py (validated there against covariates/03_po_adjusted.py and verify_covariates)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit
from scipy import stats as ss

def _negll(par, yk, X, K, off):
    a = par[:K - 1]; b = par[K - 1:]
    theta = np.cumsum(np.concatenate([[a[0]], np.exp(a[1:])]))
    eta = (X @ b if X.shape[1] else np.zeros(len(yk))) + off
    th_hi = np.concatenate([theta, [np.inf]])[yk]; th_lo = np.concatenate([[-np.inf], theta])[yk]
    Fh = expit(th_hi - eta); Fl = expit(th_lo - eta)
    p = np.clip(Fh - Fl, 1e-300, None)
    fh = Fh * (1 - Fh); fl = Fl * (1 - Fl)
    ll = np.log(p).sum()
    g_th = np.zeros(K - 1); hi = yk < K - 1; lo = yk > 0
    np.add.at(g_th, yk[hi], fh[hi] / p[hi]); np.add.at(g_th, yk[lo] - 1, -fl[lo] / p[lo])
    g_b = -(X * ((fh - fl) / p)[:, None]).sum(0) if X.shape[1] else np.zeros(0)
    rc = np.cumsum(g_th[::-1])[::-1]; g_a = rc.copy(); g_a[1:] *= np.exp(a[1:])
    return -ll, -np.concatenate([g_a, g_b])

class PO:
    def __init__(self, y, X):
        self.u, self.yk = np.unique(y, return_inverse=True); self.K = len(self.u); n = len(y)
        X = np.asarray(X, float); self.mu = X.mean(0) if X.shape[1] else np.zeros(0)
        self.sd = X.std(0) if X.shape[1] else np.ones(0); self.sd = np.where(self.sd > 0, self.sd, 1.0)
        self.Xs = (X - self.mu) / self.sd; self.X = X
        cum = np.cumsum(np.bincount(self.yk, minlength=self.K))[:-1] / n; th0 = np.log(cum / (1 - cum))
        self.a0 = np.concatenate([[th0[0]], np.log(np.diff(th0))])
    def fit(self, Xs, off, start=None):
        par0 = np.concatenate([self.a0, np.zeros(Xs.shape[1])]) if start is None else start
        r = minimize(_negll, par0, args=(self.yk, Xs, self.K, off), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=20000, maxfun=40000, gtol=1e-9, ftol=1e-15))
        return -r.fun, r.x, r.success
    def full(self):
        ll, x, ok = self.fit(self.Xs, np.zeros(len(self.yk))); self.ll, self.par = ll, x
        self.beta = x[self.K - 1:] / self.sd; return ll, self.beta, ok
    def profile(self, j, b):
        """max loglik with coefficient j (original scale) fixed at b"""
        keep = [i for i in range(self.Xs.shape[1]) if i != j]
        off = self.X[:, j] * b - (self.mu[j] * b)           # centring absorbed in thresholds
        start = np.concatenate([self.par[:self.K - 1], self.par[self.K - 1:][keep]])
        ll, _, _ = self.fit(self.Xs[:, keep], off, start); return ll
    def ci_profile(self, j, crit=ss.chi2.ppf(0.95, 1)):
        bh_, f = self.beta[j], lambda b: 2 * (self.ll - self.profile(j, b)) - crit
        se = self.wald_se(j); step = max(se, 0.05)
        out = []
        for sgn in (-1, 1):
            lo, k = bh_, 1
            while True:
                hi = bh_ + sgn * step * k
                if f(hi) > 0 or k > 60: break
                lo = hi; k *= 2
            out.append(brentq(f, min(lo, hi), max(lo, hi), xtol=1e-6) if f(hi) > 0 else np.nan)
        return out[0], out[1]
    def wald_se(self, j, h=1e-5):
        n = len(self.par); H = np.zeros((n, n)); off = np.zeros(len(self.yk))
        for i in range(n):
            e = np.zeros(n); e[i] = h
            gp = _negll(self.par + e, self.yk, self.Xs, self.K, off)[1]; gm = _negll(self.par - e, self.yk, self.Xs, self.K, off)[1]
            H[i] = (gp - gm) / (2 * h)
        H = (H + H.T) / 2
        try: cov = np.linalg.inv(H)
        except np.linalg.LinAlgError: return np.nan
        return float(np.sqrt(cov[self.K - 1 + j, self.K - 1 + j])) / self.sd[j]

