"""Independent fixed-loadings re-estimation of SDA donor scores for one run, with SDA's unsaved second moments
reconstructed ANALYTICALLY from the model's own variational updates (no fitting to the published A).

Model (SDA v1.1, Gauss mix, SS2 source, ignore_missing): y_ntj = sum_c A_nc B_tc X_cj + e, e ~ N(0, 1/lam_tj);
A_nc ~ N(0,1); x_cj ~ (1-s) d0 + s N(0, 1/beta_c) (per component, per data type); B_tc ~ N(0,1) (RNA only).
VB fixed point for a donor's score vector (layers the donor has):
   (ridge*I + sum_d [G_d + diag D_d]) a_n = sum_d h_nd
   G_d = sum_t (b_t b_t') o (X Lam_t X'),   h_nd = sum_t b_t o (X Lam_t y_nt)
   D_d = sum_t sum_j lam_tj (E[b_tc^2] E[x_cj^2] - b_tc^2 X_cj^2),  E[x^2] = X^2/S + S v   (X = s*m saved)
   v_cj = 1/(beta_c + sum_t lam_tj E[b_tc^2] sum_{n obs} E[a_nc^2]);  beta_c = sum_j S / sum_j (X^2/S + S v)
   lam_tj = n_obs / (sum_n (y - yhat)^2 + sum_n Var(yhat))       (Gamma prior negligible: shape 1e-6, rate 1e-6)
   Var(a_n) = diag((ridge I + ...)^-1)   (iterated: sweeps)
Variants (argv[2]):  full (default) | plain (lam from residuals only, D = 0) | ss (lam from residuals, D = spike-slab part only)
   | rate (full, but Gamma prior read as rate 1e6) | pooled (full, lam per gene pooled over the 6 cell types)
Outputs (cache only, per-donor): cache/run{r}/fit_{variant}.npz with Ahat for scenarios a,b,c,d and per-layer pieces."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, numpy as np
from scipy.linalg import cho_factor, cho_solve
from vp_common import *

r = int(sys.argv[1]); VAR = sys.argv[2] if len(sys.argv) > 2 else "full"
RIDGE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
SIG = sys.argv[4] if len(sys.argv) > 4 else "inv"        # posterior var of A: inv = diag(M^-1) | mf = 1/diag(M) | none = 0
NSWEEP = 1 if VAR in ("plain", "ss") else 3
CH = 6000
Y = inputs()
A = run_mat(r, "A"); B1 = run_mat(r, "B1")
cat = np.load(f"{CACHE}/prot_cat.npy"); Pc = np.load(f"{CACHE}/prot_corr.npy")
obs = {1: np.ones(N, bool), 2: ~np.isnan(Y["meth"]).all(1), 3: ~np.isnan(Y["chip"]).all(1), 4: ~np.isnan(Y["prot"]).all(1)}
YD = {2: Y["meth"], 3: Y["chip"], 4: Y["prot"]}
mis = np.isin(cat, [2, 3])
log(f"run {r} variant {VAR} ridge {RIDGE} sig {SIG}; donors per layer", {d: int(o.sum()) for d, o in obs.items()})


def lam_from(ss, n):
    if VAR == "rate":
        return (1e-6 + n / 2) / (1e6 + ss / 2)
    return (1e-6 + n / 2) / (1e-6 + ss / 2)


def ex2(X, S, v):
    q = np.divide(X * X, S, out=np.zeros_like(X), where=S > 0)
    return q + S * v if v is not None else q


state = {d: dict(beta=None, lam=None, Eb2=None) for d in (1, 2, 3, 4)}
sig2 = np.zeros((N, K))                                   # posterior variance of A (diag), from previous sweep


def layer(d, final=False, extra=None):
    """one pass over layer d: update lam, beta (v), Eb2; return G, D, H. extra: corrected-protein pieces for scenario d."""
    st = state[d]
    X = run_mat(r, f"X{d}"); S = run_mat(r, f"S{d}"); L = X.shape[1]
    o = obs[d]
    EA2 = A ** 2 + sig2
    sA2 = (A[o] ** 2).sum(0); sEA2 = EA2[o].sum(0)
    if d == 1:
        T = 6; bl = [B1[t] for t in range(T)]
        Yl = [Y["rna"][t * N:(t + 1) * N] for t in range(T)]
        Eb2 = st["Eb2"] if st["Eb2"] is not None else [b ** 2 for b in bl]
    else:
        T = 1; bl = [np.ones(K)]; Yl = [YD[d][o]]; Eb2 = [np.ones(K)]
    Ao = A if d == 1 else A[o]
    # ---- pass 1: lam (uses v from previous beta/lam) ----
    lam = np.zeros((T, L))
    for j0 in range(0, L, CH):
        sl = slice(j0, min(L, j0 + CH)); Xc = X[:, sl]; Sc = S[:, sl]
        if st["beta"] is not None and VAR not in ("plain", "ss"):
            prec = sum((Eb2[t] * sEA2)[:, None] * st["lam"][t][None, sl] for t in range(T))
            v = 1.0 / (st["beta"][:, None] + prec)
        else:
            v = None
        E2 = ex2(Xc, Sc, v) if VAR not in ("plain", "ss") else None
        for t in range(T):
            R = Yl[t][:, sl] - (Ao * bl[t]) @ Xc
            ss = (R * R).sum(0)
            if E2 is not None:
                ss = ss + (sEA2 * Eb2[t]) @ E2 - (sA2 * bl[t] ** 2) @ (Xc * Xc)
            lam[t, sl] = ss
    n_o = int(o.sum())
    if VAR == "pooled" and d == 1:
        lam = np.repeat(lam_from(lam.sum(0), T * n_o)[None, :], T, 0)
    elif VAR == "flat":                                  # homoscedastic: one precision per layer and context
        lam = np.repeat(lam_from(lam.mean(1), n_o)[:, None], L, 1)
    else:
        lam = lam_from(lam, n_o)
    lam[~np.isfinite(lam)] = 0.0
    cap = np.quantile(lam[lam > 0], 0.999) * 10        # zero-variance features (all-zero expression): cap the weight
    nc = int((lam > cap).sum()); lam = np.minimum(lam, cap)
    st["lam"] = lam
    # ---- beta / v fixed point ----
    sS = S.sum(1); sQ = np.zeros(K)
    for j0 in range(0, L, CH):
        sl = slice(j0, min(L, j0 + CH)); sQ += ex2(X[:, sl], S[:, sl], None).sum(1)
    beta = st["beta"] if st["beta"] is not None else np.maximum(sS, 1e-6) / np.maximum(sQ, 1e-300)
    if VAR not in ("plain", "ss"):
        for it in range(30):
            sv = np.zeros(K)
            for j0 in range(0, L, CH):
                sl = slice(j0, min(L, j0 + CH))
                prec = sum((Eb2[t] * sEA2)[:, None] * lam[t][None, sl] for t in range(T))
                sv += (S[:, sl] / (beta[:, None] + prec)).sum(1)
            nb = (1e-6 + 0.5 * sS) / (1e-6 + 0.5 * (sQ + sv))
            dmax = np.max(np.abs(np.log(nb) - np.log(beta)))
            beta = nb
            if dmax < 1e-6: break
    st["beta"] = beta
    # ---- pass 2: G, D, H (and new Eb2) ----
    G = np.zeros((K, K)); D = np.zeros(K); H = np.zeros((N, K)); xl = [np.zeros(K) for _ in range(T)]
    for j0 in range(0, L, CH):
        sl = slice(j0, min(L, j0 + CH)); Xc = X[:, sl]; Sc = S[:, sl]
        if VAR in ("plain", "ss"):
            E2 = ex2(Xc, Sc, None)
        else:
            prec = sum((Eb2[t] * sEA2)[:, None] * lam[t][None, sl] for t in range(T))
            E2 = ex2(Xc, Sc, 1.0 / (beta[:, None] + prec))
        X2 = Xc * Xc
        for t in range(T):
            lt = lam[t, sl]
            Xw = Xc * np.sqrt(lt)
            Gt = Xw @ Xw.T
            G += np.outer(bl[t], bl[t]) * Gt if d == 1 else Gt
            if VAR != "plain":
                D += Eb2[t] * (E2 @ lt) - bl[t] ** 2 * (X2 @ lt)
            xl[t] += E2 @ lt
            Ht = (Yl[t][:, sl] * lt) @ Xc.T
            if d == 1: H += Ht * bl[t]
            else: H[o] += Ht
    if d == 1 and VAR not in ("plain", "ss"):
        st["Eb2"] = [bl[t] ** 2 + 1.0 / (1.0 + sEA2 * xl[t]) for t in range(T)]
    out = dict(G=G, D=D, H=H)
    if extra is not None and d == 4:
        # scenario d: corrected rows; the 9 features absent after correction get weight 0 for everybody
        keep = ~np.isnan(Pc).all(0)
        lt = lam[0] * keep
        Xw = X * np.sqrt(lt)
        E2 = ex2(X, S, None if VAR in ("plain", "ss") else 1.0 / (beta[:, None] + (sEA2[:, None] * lam[0][None, :])))
        out["Gc"] = Xw @ Xw.T
        out["Dc"] = (E2 @ lt - (X * X) @ lt) if VAR != "plain" else np.zeros(K)
        Hc = np.zeros((N, K)); oc = ~np.isnan(Pc).all(1)
        Hc[oc] = (np.nan_to_num(Pc[oc]) * lt) @ X.T
        out["Hc"] = Hc
    log(f"  layer {d}: L={L} lam median {np.median(lam):.3g} capped {nc}; beta median {np.median(beta):.3g}; "
        f"G diag median {np.median(np.diag(G)):.3g}; D median {np.median(D):.3g}")
    return out


def solve(parts, has, ridge=RIDGE, want_var=False, contrib=None):
    """parts: d -> (G, D, H); has: d -> bool mask over donors. Returns Ahat (and diag of inverse per donor)."""
    Ah = np.zeros((N, K)); var = np.zeros((N, K)) if want_var else None
    C = {d: np.zeros((N, K)) for d in parts} if contrib is not None else None
    pats = {}
    for n in range(N):
        pats.setdefault(tuple(bool(has[d][n]) for d in sorted(parts)), []).append(n)
    ds = sorted(parts)
    for p, idx in pats.items():
        M = np.eye(K) * max(ridge, 1e-8)               # tiny jitter only for the no-ridge variant
        rhs = np.zeros((len(idx), K))
        for d, on in zip(ds, p):
            if on:
                G, Dd, H = parts[d]
                M += G; M[np.diag_indices(K)] += Dd; rhs += H[idx]
        cf = cho_factor(M, check_finite=False)
        Ah[idx] = cho_solve(cf, rhs.T, check_finite=False).T
        if want_var:
            var[idx] = (np.diag(cho_solve(cf, np.eye(K), check_finite=False)) if SIG == "inv" else
                        1.0 / np.diag(M) if SIG == "mf" else np.zeros(K))[None, :]
        if C is not None:
            for d, on in zip(ds, p):
                if on: C[d][idx] = cho_solve(cf, parts[d][2][idx].T, check_finite=False).T
    return Ah, var, C


def corr_cols(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    den = np.sqrt((P * P).sum(0) * (Q * Q).sum(0))
    return np.divide((P * Q).sum(0), den, out=np.full(P.shape[1], np.nan), where=den > 0)


for sw in range(NSWEEP):
    final = sw == NSWEEP - 1
    parts = {}
    for d in (1, 2, 3, 4):
        o = layer(d, final, extra=True if final else None)
        parts[d] = o
    base = {d: (parts[d]["G"], parts[d]["D"], parts[d]["H"]) for d in (1, 2, 3, 4)}
    Ah, var, _ = solve(base, obs, want_var=True)
    sig2 = var * np.vstack([np.ones(K)] * N)
    act = (A ** 2).mean(0) > 1e-4
    rc = corr_cols(Ah[:, act], A[:, act])
    rel = np.linalg.norm(Ah - A, axis=0)[act] / np.linalg.norm(A, axis=0)[act]
    log(f"sweep {sw}: active comps {int(act.sum())}; r(Ahat, A) median {np.nanmedian(rc):.4f} 1st pct {np.nanquantile(rc, .01):.4f}; "
        f"rel err median {np.median(rel):.3f} 95th {np.quantile(rel, .95):.3f}")

EMP = len(sys.argv) > 5 and sys.argv[5] == "emp"
if EMP:   # mechanism check: add an empirical per-component, per-layer NNLS diagonal fitted to the published A (as the V3 calibration)
    from scipy.optimize import nnls
    E = -RIDGE * A.copy(); Pd = np.full((N, K), RIDGE)
    for d in (1, 2, 3, 4):
        mk = obs[d]; G, Dd, H = base[d]
        E[mk] += H[mk] - A[mk] @ G - Dd * A[mk]; Pd[mk] += np.diag(G) + Dd
    Z = np.stack([np.ones(N), obs[2], obs[3], obs[4]], 1).astype(float)
    Dx = {d: np.zeros(K) for d in (1, 2, 3, 4)}
    for k in range(K):
        a = A[:, k]
        if not np.any(a): continue
        w = 1.0 / Pd[:, k]; M = Z * (a * w)[:, None]; cols = np.where(np.abs(M).sum(0) > 0)[0]
        coef, _ = nnls(M[:, cols], E[:, k] * w)
        for c, v in zip(cols, coef): Dx[c + 1][k] = v
    for d in (1, 2, 3, 4):
        parts[d]["D"] = parts[d]["D"] + Dx[d]
    parts[4]["Dc"] = parts[4]["Dc"] + Dx[4]
    base = {d: (parts[d]["G"], parts[d]["D"], parts[d]["H"]) for d in (1, 2, 3, 4)}
    log("emp: extra NNLS diagonal added; median extra/analytic D4", float(np.median(Dx[4] / np.maximum(parts[4]["D"] - Dx[4], 1e-300))))

# ---- scenarios (loadings, weights and diagonal fixed at the final sweep) ----
has_b = dict(obs); has_b[4] = np.zeros(N, bool)
has_c = dict(obs); has_c[4] = obs[4] & ~mis
oc = ~np.isnan(Pc).all(1)
partsd = dict(base); partsd[4] = (parts[4]["Gc"], parts[4]["Dc"], parts[4]["Hc"])
has_d = dict(obs); has_d[4] = oc
Aa, _, Ca = solve(base, obs, contrib=True)
Ab, _, _ = solve(base, has_b)
Ac, _, _ = solve(base, has_c)
Ad, _, Cd = solve(partsd, has_d, contrib=True)
np.savez(f"{CACHE}/run{r}/fit_{VAR}_r{RIDGE:g}" + ("" if SIG == "inv" else f"_{SIG}") + ("_emp" if EMP else "") + ".npz", Aa=Aa, Ab=Ab, Ac=Ac, Ad=Ad, sig2=sig2,
         Ca1=Ca[1], Ca2=Ca[2], Ca3=Ca[3], Ca4=Ca[4], Cd4=Cd[4],
         D1=parts[1]["D"], D2=parts[2]["D"], D3=parts[3]["D"], D4=parts[4]["D"],
         G4diag=np.diag(parts[4]["G"]), G1diag=np.diag(parts[1]["G"]), G2diag=np.diag(parts[2]["G"]), G3diag=np.diag(parts[3]["G"]),
         beta4=state[4]["beta"], lam4=state[4]["lam"][0])
log("saved scenarios a-d; check Aa == sweep solve:", float(np.abs(Aa - Ah).max()), "(differs only with emp)")
