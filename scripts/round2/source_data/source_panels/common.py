"""Shared loaders and statistics for the source-data panels (read-only on inputs)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = f"{cfg.WORK}/source_panels"                       # results/ (aggregates)
CACHE = f"{cfg.WORK}/source_panels/cache"            # per-donor slices and covariates
cfg.out_dir(ROOT, "results"); cfg.out_dir(CACHE)
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]          # rna.txt block order (cell_types.txt)
BCOR = f"{cfg.TABLES}/cell_type_loadings_corrected.csv"
SDA = cfg.SDA
MODS = ["RNA", "meth", "chip", "proteins"]


def load():
    S = dict(np.load(f"{CACHE}/slices.npz", allow_pickle=True))
    R = dict(np.load(f"{CACHE}/random_bg.npz", allow_pickle=True))
    A = np.load(f"{CACHE}/A_all.npy", allow_pickle=True)
    B = pd.read_csv(BCOR, index_col=0)
    T = pd.read_csv(f"{CACHE}/donor_covariates.csv", index_col=0)
    assert [str(x) for x in T.index] == [str(x) for x in S["samples"]]
    return S, R, A, B, T


def comp_row(S, k):
    return int(np.where(S["comps"] == k)[0][0])


def feature_values(S, m, j, ct=None):
    """Input (as fed to SDA) values of global feature index j in modality m, 276 donors."""
    sel = S[f"{m}_sel"]; pos = int(np.where(sel == j)[0][0])
    V = S[f"{m}_V"]
    if m == "RNA":
        c = CTS.index(ct)
        return V[c * 276:(c + 1) * 276, pos].astype(float)
    return V[:, pos].astype(float)


def robust_z(x):
    med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / mad


def score_stats(a):
    rz = robust_z(a)
    i = int(np.nanargmax(np.abs(rz)))
    d = a - a.mean(); share = d[i] ** 2 / np.sum(d ** 2)
    order = np.argsort(-np.abs(rz))
    q1, q3 = np.percentile(a, [25, 75]); iqr = q3 - q1
    n30 = int(np.sum((a < q1 - 30 * iqr) | (a > q3 + 30 * iqr)))
    top3 = np.sum(d[order[:3]] ** 2) / np.sum(d ** 2)
    return dict(n=len(a), excess_kurtosis=float(stats.kurtosis(a)), max_abs_robust_z=float(np.abs(rz[i])),
                extreme_sign=int(np.sign(rz[i])), second_abs_robust_z=float(np.abs(rz[order[1]])),
                third_abs_robust_z=float(np.abs(rz[order[2]])),
                share_var_top1=float(share), share_var_top3=float(top3),
                n_beyond_30IQR=n30, idx_extreme=i)


def corr_pair(x, y):
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return dict(n=int(len(x)), rho=np.nan, p_rho=np.nan, r=np.nan, p_r=np.nan)
    rho, prho = stats.spearmanr(x, y); r, pr = stats.pearsonr(x, y)
    return dict(n=int(len(x)), rho=float(rho), p_rho=float(prho), r=float(r), p_r=float(pr))


def robustness(x, a, rz_all):
    """x: feature (NaN = no data); a: score; rz_all: robust z of score over all 276 donors."""
    ok = ~np.isnan(x)
    idx = np.where(ok)[0]
    full = corr_pair(x, a)
    ordr = idx[np.argsort(-np.abs(rz_all[idx]))]
    m1 = ok.copy(); m1[ordr[0]] = False
    d1 = corr_pair(np.where(m1, x, np.nan), a)
    k = int(np.ceil(0.01 * len(idx)))
    mk = ok.copy(); mk[ordr[:k]] = False
    dk = corr_pair(np.where(mk, x, np.nan), a)
    # leave-one-out
    xs, as_ = x[idx], a[idx]
    rr = stats.rankdata
    lo_rho, lo_r = [], []
    for t in range(len(idx)):
        keep = np.ones(len(idx), bool); keep[t] = False
        lo_rho.append(stats.spearmanr(xs[keep], as_[keep])[0]); lo_r.append(np.corrcoef(xs[keep], as_[keep])[0, 1])
    lo_rho = np.array(lo_rho); lo_r = np.array(lo_r)
    t_rho = int(np.nanargmax(np.abs(lo_rho - full["rho"]))); t_r = int(np.nanargmax(np.abs(lo_r - full["r"])))
    return dict(n=full["n"], rho=full["rho"], p_rho=full["p_rho"], r=full["r"], p_r=full["p_r"],
                rho_drop1=d1["rho"], p_rho_drop1=d1["p_rho"], r_drop1=d1["r"], p_r_drop1=d1["p_r"],
                k_top1pct=k, rho_drop1pct=dk["rho"], p_rho_drop1pct=dk["p_rho"], r_drop1pct=dk["r"], p_r_drop1pct=dk["p_r"],
                loo_max_drho=float(np.abs(lo_rho - full["rho"])[t_rho]),
                loo_max_dr=float(np.abs(lo_r - full["r"])[t_r]),
                loo_r_most_influential_is_score_extreme=bool(idx[t_r] == ordr[0]),
                extreme_has_data=bool(ok[int(np.argmax(np.abs(rz_all)))]))


def zstd(M):
    """column-standardise (donors x features), NaN rows kept NaN; drop zero-variance columns."""
    mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0)
    keep = sd > 0
    return (M[:, keep] - mu[keep]) / sd[keep], keep


def aggregate(M, w):
    """loading-weighted mean of standardised values; M donors x features, w weights (sign = product)."""
    Z, keep = zstd(M)
    w = w[keep]
    agg = Z @ w / np.sum(np.abs(w))   # rows of missing donors are all-NaN -> NaN
    return agg, int(keep.sum())
