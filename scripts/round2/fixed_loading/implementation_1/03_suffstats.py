"""Per-run sufficient statistics for re-estimating donor scores with the fitted loadings held fixed.
Model (one run): y[n, c, g] ~ sum_k A_nk B1_ck X1_kg (RNA, 6 cell-type blocks); y_j[n, l] ~ sum_k A_nk Xj_kl (meth, chip, prot).
Noise precision per feature (per cell type x gene for RNA) = 1 / mean residual^2 of the published reconstruction
over donors with data (SDA's precisions are not saved).
Saves cache/run{r}/stats.npz:
  G1c[c]  = X1 W_c X1^T   (6 x K x K)           -> RNA Gram for cell type c (B1 applied at solve time)
  G2,G3,G4 = Xj W_j Xj^T                          (meth, chip, prot as fed)
  G4cor   = prot Gram without the md_prot features that are all-NaN in the corrected build
  H1      = sum_c B1_c * ((Y_c W_c) X1^T)        (N x K)
  H2,H3,H4 (as fed), H4cor (corrected prot rows) (N x K; 0 for donors without that layer)
  Q1c[c]  = sum_g w_cg X1_kg^2 (1/s_kg - 1)       (6 x K)  spike-slab second-moment correction (slab variance ignored)
  Q2,Q3,Q4,Q4cor analogous
  w-summaries; presence masks."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, os, numpy as np
from pi_common import *

r = int(sys.argv[1])
WMODE = next((a.split("=")[1] for a in sys.argv[2:] if a.startswith("--wmode=")), "resid")
# resid : w = 1/mean residual^2 per feature (per cell type x gene for RNA)  [default]
# pooled: RNA w per gene pooled over the six cell types
# rate  : w = (n/2) / (1e6 + SSR/2)  (reading prior_lambda2 = 1e6 as a Gamma rate; diagnostic only)
# f1    : as resid, floor on the residual variance at the 1st percentile of the block (sensitivity)
# x0    : as resid, features with ~zero residual variance get zero weight (sensitivity)
# rv    : w = 1/(mean residual^2 + loading second-moment term), the term being
#         mean_n sum_k A_nk^2 B_ck^2 X_kl^2 (1/s_kl - 1)  (spike-slab E[x^2]-E[x]^2 with the slab variance ignored),
#         i.e. the part of SDA's expected squared residual that the posterior means alone miss
cd = f"{CACHE}/run{r}"
out = f"{cd}/stats.npz" if WMODE == "resid" else f"{cd}/stats_{WMODE}.npz"
Y = load_inputs()
os.makedirs(cd, exist_ok=True)
for w in ("A", "B1"):
    if not os.path.exists(f"{cd}/{w}.npy"): np.save(f"{cd}/{w}.npy", get_mat(r, w))
A = np.load(f"{cd}/A.npy"); B1 = np.load(f"{cd}/B1.npy")
assert A.shape == (N, K) and B1.shape == (6, K)
hasS = True
res = {}
Pfed = np.load(f"{CACHE}/prot_fed.npy"); Pcor = np.load(f"{CACHE}/prot_corrected.npy")
assert np.array_equal(np.isnan(Pfed), np.isnan(np.asarray(Y["proteins"]))) and np.nanmax(np.abs(Pfed - Y["proteins"])) == 0
cor_has = ~np.isnan(Pcor).all(1)
nanfeat = np.load(f"{CACHE}/prot_corrected_allnan_feat.npy")
assert np.array_equal(np.isnan(Pcor[cor_has]), np.broadcast_to(nanfeat, (cor_has.sum(), len(nanfeat)))), "corrected NaNs not confined to all-NaN features"


CH = 4000   # feature chunk (memory)


def gram(X, w):
    G = np.zeros((X.shape[0], X.shape[0]))
    for i in range(0, X.shape[1], CH):
        Xw = X[:, i:i + CH] * np.sqrt(w[i:i + CH])[None, :]
        G += Xw @ Xw.T
    return G


def qcorr(X, S, w):
    q = np.zeros(X.shape[0])
    for i in range(0, X.shape[1], CH):
        s = np.clip(S[:, i:i + CH].astype(np.float64), 1e-12, 1.0)
        q += ((X[:, i:i + CH] ** 2) * (1.0 / s - 1.0)) @ w[i:i + CH]
    return q


def vterm(X, S, a2):
    """sum_k a2_k X_kl^2 (1/s_kl - 1) per feature (chunked)."""
    out = np.zeros(X.shape[1])
    for i in range(0, X.shape[1], CH):
        s = np.clip(S[:, i:i + CH].astype(np.float64), 1e-12, 1.0)
        out[i:i + CH] = a2 @ ((X[:, i:i + CH] ** 2) * (1.0 / s - 1.0))
    return out


def prec(v, tag, n=None):
    """1/residual variance with a floor (features with ~zero residual would get unbounded weight)."""
    if WMODE == "rate":
        return (n / 2.0) / (1e6 + n * v / 2.0), 0
    if WMODE == "x0":      # sensitivity: features with ~zero residual get zero weight instead of a capped weight
        bad = v < 1e-6 * np.median(v)
        w = np.where(bad, 0.0, 1.0 / np.where(bad, 1.0, v)); return w, int(bad.sum())
    fl = 1e-6 * np.median(v) if WMODE != "f1" else np.percentile(v, 1)   # f1: floor at the block's 1st percentile
    nfl = int((v < fl).sum())
    if nfl: print(f"   {tag}: {nfl} features floored", flush=True)
    return 1.0 / np.maximum(v, fl), nfl


def r2f(Y, R):
    sst = np.sum((Y - Y.mean(0)) ** 2, 0); ok = sst > 0
    return float(np.median(1 - np.sum(R ** 2, 0)[ok] / sst[ok]))


# ---------------- RNA
X1 = get_mat(r, "X1").astype(np.float64)
S1 = get_mat(r, "S1") if hasS else None
YR = np.asarray(Y["RNA"])
G1c = np.zeros((6, K, K)); H1 = np.zeros((N, K)); Q1c = np.zeros((6, K)); W1 = np.zeros((6, X1.shape[1]))
if WMODE == "pooled":
    vp = np.zeros(X1.shape[1])
    for c in range(6):
        vp += np.mean((YR[c * N:(c + 1) * N] - (A * B1[c][None, :]) @ X1) ** 2, axis=0) / 6
for c in range(6):
    Yc = YR[c * N:(c + 1) * N]
    Rc = Yc - (A * B1[c][None, :]) @ X1
    v = np.mean(Rc ** 2, axis=0) if WMODE != "pooled" else vp
    if WMODE == "rv":
        vv = vterm(X1, S1, np.mean(A ** 2, 0) * B1[c] ** 2); res[f"RNA_{CTS[c]}_vterm_share_median"] = float(np.median(vv / (v + vv))); v = v + vv
    w, nfl = prec(v, f"RNA {CTS[c]}", N); W1[c] = w; res[f"RNA_{CTS[c]}_nfloor"] = nfl
    G1c[c] = gram(X1, w)
    H1 += B1[c][None, :] * ((Yc * w[None, :]) @ X1.T)
    if hasS: Q1c[c] = qcorr(X1, S1, w)
    res[f"RNA_{CTS[c]}_R2"] = r2f(Yc, Rc)
    tlog(f"run{r} RNA {CTS[c]} done; R2={res[f'RNA_{CTS[c]}_R2']:.3f}")
del X1, S1, YR

# ---------------- matrix modalities
out_d = dict(G1c=G1c, H1=H1, Q1c=Q1c, W1=W1)
for j, m in [(2, "meth"), (3, "chip"), (4, "proteins")]:
    X = get_mat(r, f"X{j}").astype(np.float64)
    S = get_mat(r, f"S{j}") if hasS else None
    V = np.asarray(Y[m]); has = ~np.isnan(V).all(1)
    Rm = V[has] - A[has] @ X
    v = np.mean(Rm ** 2, axis=0)
    if WMODE == "rv":
        vv = vterm(X, S, np.mean(A[has] ** 2, 0)); res[f"{m}_vterm_share_median"] = float(np.median(vv / (v + vv))); v = v + vv
    w, nfl = prec(v, m, int(has.sum())); res[f"{m}_nfloor"] = nfl
    out_d[f"W{j}"] = w
    out_d[f"G{j}"] = gram(X, w)
    H = np.zeros((N, K)); H[has] = (V[has] * w[None, :]) @ X.T
    out_d[f"H{j}"] = H
    out_d[f"has{j}"] = has
    if hasS: out_d[f"Q{j}"] = qcorr(X, S, w)
    res[f"{m}_R2"] = r2f(V[has], Rm)
    if j == 4:
        keep = ~nanfeat
        out_d["G4cor"] = gram(X[:, keep], w[keep])
        H = np.zeros((N, K)); H[cor_has] = (Pcor[cor_has][:, keep] * w[keep][None, :]) @ X[:, keep].T
        out_d["H4cor"] = H; out_d["has4cor"] = cor_has
        if hasS: out_d["Q4cor"] = qcorr(X[:, keep], S[:, keep], w[keep])
    tlog(f"run{r} {m} done; donors {int(has.sum())}; R2={res[f'{m}_R2']:.3f}")
    del X, S, V, Rm
import json; json.dump({k: float(v) for k, v in res.items()}, open(f"{cd}/stats_fit_{WMODE}.json", "w"), indent=1)
np.savez(out, **out_d)
print("median per-feature R2 of published reconstruction (as fed); floored-feature counts:", {k: round(float(v), 4) for k, v in res.items()})
