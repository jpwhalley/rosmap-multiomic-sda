"""Independent re-implementation helpers (verify_covariates, 2026-10-13).

Written from scratch for the verification of the round-2 covariate sensitivity; does not import
anything from ../covariates.  Read-only on its inputs. Outputs are written to the working directory, which is set
here to $G_ROS_RESULTS_DIR/verify_covariates.
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss
from scipy.special import expit

B = cfg.ROS
C547 = f"{cfg.AUD}/c547"
os.chdir(cfg.out_dir(cfg.WORK, "verify_covariates"))   # outputs (vc_*.csv) are written relative to this folder
CT = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]


# ------------------------------------------------------------------ data
def load_scores():
    """547 x 276 component scores: published rows minus C520, plus C547/C548 consensus A vectors."""
    s = pd.read_csv(f"{B}/sda/samples_final.csv", index_col=0)
    s.columns = s.columns.astype(str)
    s = s.drop(index=520)
    for k in (547, 548):
        a = np.load(f"{C547}/C{k}_A.npy")
        assert a.shape == (s.shape[1],)
        s.loc[k] = a
    return s


def load_clin(donors):
    c = pd.read_csv(f"{B}/oth/meta/ROSMAP_clinical.csv", dtype={"projid": str, "age_death": str})
    c.index = c.projid.astype(int).astype(str)
    c = c.loc[list(donors)].copy()
    c["age90"] = (c.age_death.str.strip() == "90+").astype(float)
    c["age"] = pd.to_numeric(c.age_death.replace("90+", "90"), errors="coerce")
    c["is_map"] = (c.Study == "MAP").astype(float)
    c["male"] = c.msex.astype(float)
    c["cerad_ad"] = c.ceradsc.map({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0})
    return c


def load_qc(donors):
    """Per-donor snRNA-seq QC proxies built here: post-QC nuclei over the six classes (h5ad-derived count
    table, read-only), pseudobulk UMIs summed over the six classes (raw counts in the archive)."""
    dc = pd.read_csv(f"{cfg.WORK}/revision_r2/donor_celltype_counts_from_h5ad.csv", index_col=0)
    dc.index = dc.index.astype(str)
    dc = dc.loc[list(donors), CT]
    umi = {}
    for c in CT:
        x = pd.read_csv(f"{B}/mit/rna/pseudobulk/{c}_raw_counts.csv", index_col=0)
        x.index = x.index.astype(int).astype(str)
        umi[c] = x.sum(axis=1)
    umi = pd.DataFrame(umi).loc[list(donors), CT]
    q = pd.DataFrame(index=list(donors))
    q["nuclei"] = dc.sum(axis=1)
    q["umis"] = umi.sum(axis=1)
    q["upn"] = q.umis / q.nuclei
    for c in CT:
        q[f"n_{c}"] = dc[c]
    return q


# ------------------------------------------------------------------ stats helpers
def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(r, 1.0)
    return out


def rint(x, c=3.0 / 8):
    r = ss.rankdata(x)
    return ss.norm.ppf((r - c) / (len(x) - 2 * c + 1))


def published_tests(v, clin):
    """Original ros6a tests: sex MWU; CERAD {1,2} vs {3,4} MWU; cogdx KW over 1, 2, 4."""
    v = np.asarray(v, float)
    m = clin.msex.values
    cb = clin.cerad_ad.values
    cg = clin.cogdx.values
    p1 = ss.mannwhitneyu(v[m == 1], v[m == 0], alternative="two-sided").pvalue
    p2 = ss.mannwhitneyu(v[cb == 1], v[cb == 0], alternative="two-sided").pvalue
    p3 = ss.kruskal(*[v[cg == k] for k in (1, 2, 4)]).pvalue
    return p1, p2, p3


# ------------------------------------------------------------------ proportional odds (own Newton fitter)
def _cdf_diff(u, l):
    """F(u) - F(l) computed stably (F = logistic)."""
    out = np.where(l > 0, expit(-l) - expit(-u), expit(u) - expit(l))
    return out


def po_fit(y, X, maxit=300, tol=1e-9):
    """Cumulative-logit model P(Y <= k | x) = F(a_k - x'b), Y an ordinal outcome taken from the ranks of y
    (ties share a category). Returns dict(ll, beta, converged, iters). X: n x p (no intercept), may be p=0."""
    y = np.asarray(y, float)
    n = len(y)
    _, cat = np.unique(y, return_inverse=True)
    K = cat.max() + 1
    X = np.zeros((n, 0)) if X is None else np.asarray(X, float).reshape(n, -1)
    p = X.shape[1]
    cnt = np.bincount(cat, minlength=K)
    cum = np.cumsum(cnt)[:-1] / n
    a = np.log(cum / (1 - cum))
    b = np.zeros(p)
    hi = cat < K - 1          # has upper cut (a_cat)
    lo = cat > 0              # has lower cut (a_{cat-1})
    iu = np.where(hi, cat, 0)
    il = np.where(lo, cat - 1, 0)

    def ll_of(a, b):
        if np.any(np.diff(a) <= 0):
            return -np.inf
        eta = X @ b if p else np.zeros(n)
        u = np.where(hi, a[iu] - eta, np.inf)
        l = np.where(lo, a[il] - eta, -np.inf)
        pr = _cdf_diff(u, l)
        if np.any(pr <= 0):
            return -np.inf
        return np.log(pr).sum()

    ll = ll_of(a, b)
    conv = False
    it = 0
    for it in range(1, maxit + 1):
        eta = X @ b if p else np.zeros(n)
        u = np.where(hi, a[iu] - eta, np.inf)
        l = np.where(lo, a[il] - eta, -np.inf)
        pr = _cdf_diff(u, l)
        fu = np.where(hi, expit(u) * expit(-u), 0.0)
        fl = np.where(lo, expit(l) * expit(-l), 0.0)
        dfu = fu * (expit(-u) - expit(u))
        dfl = fl * (expit(-l) - expit(l))
        g_u = fu / pr
        g_l = fl / pr
        gg = g_u - g_l
        # gradient
        ga = np.zeros(K - 1)
        np.add.at(ga, iu[hi], g_u[hi])
        np.add.at(ga, il[lo], -g_l[lo])
        gb = -(X * gg[:, None]).sum(0) if p else np.zeros(0)
        grad = np.concatenate([ga, gb])
        # Hessian
        m = K - 1 + p
        H = np.zeros((m, m))
        huu = dfu / pr - g_u ** 2
        hll = -dfl / pr - g_l ** 2
        hul = g_u * g_l
        np.add.at(H, (iu[hi], iu[hi]), huu[hi])
        np.add.at(H, (il[lo], il[lo]), hll[lo])
        both = hi & lo
        np.add.at(H, (iu[both], il[both]), hul[both])
        np.add.at(H, (il[both], iu[both]), hul[both])
        if p:
            wbb = (dfu - dfl) / pr - gg ** 2
            H[K - 1:, K - 1:] = (X * wbb[:, None]).T @ X
            cu = -(dfu / pr - g_u * gg)       # d2/da_upper db = x * cu
            cl = (dfl / pr - g_l * gg)        # d2/da_lower db = x * cl
            Hab = np.zeros((K - 1, p))
            np.add.at(Hab, iu[hi], X[hi] * cu[hi, None])
            np.add.at(Hab, il[lo], X[lo] * cl[lo, None])
            H[:K - 1, K - 1:] = Hab
            H[K - 1:, :K - 1] = Hab.T
        try:
            step = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, -grad, rcond=None)[0]
        t = 1.0
        while True:
            a2 = a + t * step[:K - 1]
            b2 = b + t * step[K - 1:]
            ll2 = ll_of(a2, b2)
            if ll2 >= ll - 1e-12 or t < 1e-10:
                break
            t /= 2
        dll = ll2 - ll
        a, b, ll = a2, b2, ll2
        if np.max(np.abs(grad)) < 1e-7 or (abs(dll) < tol and t == 1.0):
            conv = True
            break
    return dict(ll=ll, beta=b, alpha=a, converged=conv, iters=it)


def po_lrt(y, Xcov, Xph):
    """LRT of the phenotype columns Xph given covariates Xcov in the cumulative-logit model."""
    n = len(y)
    X0 = Xcov if Xcov is not None and Xcov.shape[1] else np.zeros((n, 0))
    X1 = np.column_stack([Xph, X0]) if X0.shape[1] else np.asarray(Xph, float).reshape(n, -1)
    f0 = po_fit(y, X0)
    f1 = po_fit(y, X1)
    k = np.asarray(Xph).reshape(n, -1).shape[1]
    stat = max(2 * (f1["ll"] - f0["ll"]), 0.0)
    return dict(p=ss.chi2.sf(stat, k), stat=stat, beta=f1["beta"][:k], conv=f0["converged"] and f1["converged"],
                ll0=f0["ll"], ll1=f1["ll"])


# ------------------------------------------------------------------ OLS F test, vectorised over many outcomes
def ols_F(Y, Xcov, Xph):
    """Y: n x m outcomes. Returns (p, beta_ph) for the joint F test of Xph given intercept + Xcov."""
    n = Y.shape[0]
    one = np.ones((n, 1))
    X0 = np.column_stack([one] + ([Xcov] if Xcov is not None and Xcov.shape[1] else []))
    Xph = np.asarray(Xph, float).reshape(n, -1)
    X1 = np.column_stack([X0, Xph])
    def rss(X):
        bta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        r = Y - X @ bta
        return (r ** 2).sum(0), bta
    r0, _ = rss(X0)
    r1, b1 = rss(X1)
    k = Xph.shape[1]
    df1 = n - X1.shape[1]
    F = ((r0 - r1) / k) / (r1 / df1)
    return ss.f.sf(F, k, df1), b1[X0.shape[1]:, :]
