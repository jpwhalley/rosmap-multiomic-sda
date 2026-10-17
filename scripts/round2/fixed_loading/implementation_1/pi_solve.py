"""Fixed-loadings re-estimation of donor scores for one run (shared by calibration and scenarios).
For donor n with layer set M_n:  a_n = (I + sum_{j in M_n} G_j + diag(sum_{j in M_n} D_j))^{-1} sum_{j in M_n} h_nj
  G_RNA = sum_c (B1_c B1_c^T) o (X1 W_c X1^T);  G_j = X_j W_j X_j^T;  h from the inputs and the same weights.
  D_j (per component, per layer) stands in for the posterior second-moment terms of the loadings
  (E[x^2] - E[x]^2) that SDA uses in its score update but does not save."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls
from pi_common import *


def load_stats(r, wmode="resid"):
    st = dict(np.load(f"{CACHE}/run{r}/stats.npz" if wmode == "resid" else f"{CACHE}/run{r}/stats_{wmode}.npz"))
    A = np.load(f"{CACHE}/run{r}/A.npy"); B1 = np.load(f"{CACHE}/run{r}/B1.npy")
    st["G1"] = np.einsum("ck,cl,ckl->kl", B1, B1, st["G1c"], optimize=True)
    st["Q1"] = np.einsum("ck,ck->k", B1 ** 2, st["Q1c"])
    return st, A, B1


def solve(st, H, has, D=None, extra_diag=None, ridge=1.0, G4key="G4"):
    """H: dict j->(N x K) rhs (0 rows where absent); has: dict j->bool mask (j=2,3,4); D: dict j->K diag terms."""
    Ahat = np.zeros((N, K))
    pats = {}
    for n in range(N):
        pats.setdefault((bool(has[2][n]), bool(has[3][n]), bool(has[4][n])), []).append(n)
    for (m2, m3, m4), idx in pats.items():
        P = st["G1"].copy()
        d = np.full(K, ridge)
        if D is not None: d = d + D[1]
        for j, mj in ((2, m2), (3, m3), (4, m4)):
            if mj:
                P += st[G4key] if j == 4 else st[f"G{j}"]
                if D is not None: d = d + D[j]
        P[np.diag_indices(K)] += d
        rhs = H[1][idx].copy()
        for j, mj in ((2, m2), (3, m3), (4, m4)):
            if mj: rhs += H[j][idx]
        cf = cho_factor(P, lower=False, check_finite=False)
        Ahat[idx] = cho_solve(cf, rhs.T, check_finite=False).T
    return Ahat


def fixed_point_residual(st, A, H, has, ridge=1.0):
    """e_n = h_n - (ridge I + G_n) a_n for the published A (what the unsaved diagonal terms must explain)."""
    E = np.zeros((N, K))
    E += H[1] - A @ st["G1"] - ridge * A
    for j in (2, 3, 4):
        m = has[j]
        E[m] += H[j][m] - A[m] @ st[f"G{j}"]
    return E


def fit_diag(A, E, has, rows=None):
    """per component NNLS of e_nk on a_nk * 1[j in M_n] (j = RNA, meth, chip, prot) -> D[j][k] >= 0."""
    rows = np.arange(N) if rows is None else rows
    Z = np.stack([np.ones(N), has[2], has[3], has[4]], 1).astype(float)[rows]
    D = {j: np.zeros(K) for j in (1, 2, 3, 4)}
    for k in range(K):
        a = A[rows, k]
        if not np.any(a): continue
        M = Z * a[:, None]
        cols = np.where(np.abs(M).sum(0) > 0)[0]
        coef, _ = nnls(M[:, cols], E[rows, k])
        for c, v in zip(cols, coef): D[c + 1][k] = v
    return D


def comp_metrics(Ahat, A):
    """per component: Pearson r across donors, relative error ||Ahat-A|| / ||A||, sd of A."""
    a0 = A - A.mean(0); b0 = Ahat - Ahat.mean(0)
    den = np.sqrt((a0 ** 2).sum(0) * (b0 ** 2).sum(0))
    r = np.where(den > 0, (a0 * b0).sum(0) / np.where(den > 0, den, 1), np.nan)
    nA = np.linalg.norm(A, axis=0)
    rel = np.where(nA > 0, np.linalg.norm(Ahat - A, axis=0) / np.where(nA > 0, nA, 1), np.nan)
    return r, rel, A.std(0)


def pattern_diag(st, has, ridge=1.0, D=None):
    """per donor x component diagonal of the score precision for that donor's layer pattern (N x K)."""
    P = np.zeros((N, K)) + ridge + np.diag(st["G1"])[None, :]
    if D is not None: P += D[1][None, :]
    for j in (2, 3, 4):
        add = np.diag(st[f"G{j}"]) + (D[j] if D is not None else 0)
        P[has[j]] += add[None, :]
    return P


def fit_diag_w(A, E, has, Pd, rows=None, base=None):
    """as fit_diag, but each donor's equation is divided by its pattern's diagonal precision (errors in score units);
    base: optional fixed diagonal (dict j->K) already included (then E must already have base*a subtracted)."""
    rows = np.arange(N) if rows is None else rows
    Z = np.stack([np.ones(N), has[2], has[3], has[4]], 1).astype(float)[rows]
    D = {j: np.zeros(K) for j in (1, 2, 3, 4)}
    for k in range(K):
        a = A[rows, k]
        if not np.any(a): continue
        wgt = 1.0 / Pd[rows, k]
        M = Z * (a * wgt)[:, None]
        cols = np.where(np.abs(M).sum(0) > 0)[0]
        coef, _ = nnls(M[:, cols], E[rows, k] * wgt)
        for c, v in zip(cols, coef): D[c + 1][k] = v
    if base is not None:
        D = {j: D[j] + base[j] for j in D}
    return D
