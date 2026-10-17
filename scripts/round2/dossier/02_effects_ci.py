"""Effect estimates with 95% CIs for the phenotype associations of the dossier components.
Primary: proportional-odds (cumulative-logit) regression of the score treated as ordinal (one category per distinct
value) -- the model of covariates/03_po_adjusted.py (same likelihood and parameterisation, re-implemented here with an
offset so the phenotype coefficient can be profiled). OR = exp(beta): odds that a donor in the index group has a HIGHER
score than one in the reference group (male vs female; CERAD AD vs non-AD; cogdx MCI vs NCI and AD-dementia vs NCI).
95% CI = profile likelihood (LR chi2_1 = 3.841); Wald CI from the observed information as a check.
Secondary: rank-INT OLS (score -> rank-based inverse normal), coefficient in SD units with t-based 95% CI.
Models: M0 none | M1 age(90+->90) + PMI + cohort [+ sex] | M2 M1 + log nuclei + log UMIs/nucleus | M3 age + I(90+) +
log PMI + cohort [+ sex] | drop1-M1/M2: M1/M2 without the donor furthest from the median | extra rows as labelled.
q for the published family (547 components, BH per phenotype): from covariates/po_tests_547.csv and
verify_covariates/vc_rint_547.csv; for drop1 the component's p is substituted into the family (others fixed; the
verification found this equals the exact family refit for PO M1). Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import warnings, numpy as np, pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit
from scipy import stats as ss
from dc_common import *
warnings.filterwarnings("ignore")

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

S = scores547(); D = design_frame(); don = list(S.columns)
PH = {"Sex": ["sex"], "ceradsc": ["cerad_ad"], "cogdx": ["cogdx2", "cogdx4"], "cogdx_trend": ["cogdx_trend"],
      "cerad4": ["ceradsc"]}
D["cogdx2"] = (D.cogdx == 2).astype(float); D["cogdx4"] = (D.cogdx == 4).astype(float)
D["cogdx_trend"] = D.cogdx.map({1: 0.0, 2: 1.0, 4: 2.0})
for _l in (2, 3, 4, 5, 6): D[f"cgd{_l}"] = (D.cogdx == _l).astype(float)   # all six cogdx categories (NCI = reference)
COG6 = [f"cgd{_l}" for _l in (2, 3, 4, 5, 6)]
D["cerad4_rev"] = 5 - D.ceradsc     # CERAD 1 = definite AD; reversed so higher = more plaques
PH["cerad4"] = ["cerad4_rev"]
SUBSET = {"cogdx": lambda d: d.cogdx.isin([1, 2, 4]), "cogdx_trend": lambda d: d.cogdx.isin([1, 2, 4])}
def base(ph): return ["age", "pmi", "ros"] + ([] if ph == "Sex" else ["sex"])
MODELS = {"M0": lambda ph: [], "M1": base, "M2": lambda ph: base(ph) + ["lognuc", "logupn"],
          "M3": lambda ph: ["age", "age90p", "logpmi", "ros"] + ([] if ph == "Sex" else ["sex"])}
COMP_PRT = ["prop_Exc", "prop_Inh", "prop_Oli", "prop_OPC", "prop_Mic"]   # Ast = reference (proportions sum to 1)
TASKS = [(74, "Sex"), (278, "Sex"), (545, "Sex"), (172, "ceradsc"), (277, "cogdx"),
         (172, "cerad4"), (277, "cogdx_trend")]
EXTRA = {   # (component, phenotype): {label: covariate list}
    (278, "Sex"): {"M1+CERAD+cogdx6": base("Sex") + ["cerad_ad"] + COG6, "M1+composition": base("Sex") + COMP_PRT},
    (545, "Sex"): {"M1+CERAD+cogdx6": base("Sex") + ["cerad_ad"] + COG6, "M1+composition": base("Sex") + COMP_PRT},
    (172, "ceradsc"): {"M1+composition": base("ceradsc") + COMP_PRT},
    (277, "cogdx"): {"M1+composition": base("cogdx") + COMP_PRT, "M2+composition": base("cogdx") + ["lognuc", "logupn"] + COMP_PRT,
                     "M1+CERAD": base("cogdx") + ["cerad_ad"]},
    (277, "cogdx_trend"): {"M1+composition": base("cogdx") + COMP_PRT},
    (278, "cerad4"): {},
}
fam_po = pd.read_csv(f"{R2}/covariates/po_tests_547.csv", header=[0, 1], index_col=0)
fam_ri = pd.read_csv(f"{R2}/verify_covariates/vc_rint_547.csv")
def q_sub(fam_p, k, p):
    v = fam_p.copy(); v.loc[k] = p; return float(pd.Series(bh(v.values), index=v.index).loc[k])

def rint_family_q(k, fam_ph, rm, pool, sub, phcols, cov):
    """exact refit: RINT every component over `pool`, nested F of the phenotype block in `sub`, BH over 547."""
    Y = np.column_stack([rint(S.loc[c, pool].values.astype(float)) for c in S.index])
    Yd = pd.DataFrame(Y, index=pool, columns=S.index).loc[sub].values
    Xf = np.column_stack([np.ones(len(sub)), D.loc[sub, phcols + cov].values.astype(float)])
    Xr = np.column_stack([np.ones(len(sub))] + ([D.loc[sub, cov].values.astype(float)] if cov else []))
    bf, *_ = np.linalg.lstsq(Xf, Yd, rcond=None); rf = ((Yd - Xf @ bf) ** 2).sum(0)
    br, *_ = np.linalg.lstsq(Xr, Yd, rcond=None); rr = ((Yd - Xr @ br) ** 2).sum(0)
    q = Xf.shape[1] - Xr.shape[1]; dff = len(sub) - Xf.shape[1]
    p = ss.f.sf(((rr - rf) / q) / (rf / dff), q, dff)
    return float(pd.Series(bh(p), index=S.index).loc[k])

rows = []
for k, ph in TASKS:
    phcols = PH[ph]; sub0 = D.index[SUBSET[ph](D)] if ph in SUBSET else D.index
    sc = S.loc[k]; ext = (sc - sc.median()).abs().idxmax()
    specs = [(m, MODELS[m](ph), list(sub0)) for m in MODELS]
    specs += [("drop1-M1", MODELS["M1"](ph), [x for x in sub0 if x != ext]), ("drop1-M2", MODELS["M2"](ph), [x for x in sub0 if x != ext])]
    specs += [(lab, cov, list(sub0)) for lab, cov in EXTRA.get((k, ph), {}).items()]
    for lab, cov, sub in specs:
        dropped = [x for x in don if x not in set(sub) and x == ext]
        pool = [x for x in don if x not in set(dropped)]
        yr_all = pd.Series(rint(sc.loc[pool].values.astype(float)), index=pool)
        y = sc.loc[sub].values.astype(float); X = D.loc[sub, phcols + cov].values.astype(float)
        po = PO(y, X); ll, beta, ok = po.full()
        # LRT p for the phenotype block
        po0 = PO(y, D.loc[sub, cov].values.astype(float)) if cov else None
        ll0 = po0.full()[0] if cov else PO(y, np.zeros((len(y), 0))).fit(np.zeros((len(y), 0)), np.zeros(len(y)))[0]
        p_lrt = ss.chi2.sf(max(2 * (ll - ll0), 0), len(phcols))
        # rank-INT OLS
        yr = yr_all.loc[sub].values; Xo = np.column_stack([np.ones(len(y)), X]); bo, *_ = np.linalg.lstsq(Xo, yr, rcond=None)
        res = yr - Xo @ bo; dfr = len(y) - Xo.shape[1]; s2 = res @ res / dfr
        cov_o = s2 * np.linalg.inv(Xo.T @ Xo)
        # nested F for the phenotype block (OLS)
        Xr = np.column_stack([np.ones(len(y))] + ([D.loc[sub, cov].values.astype(float)] if cov else []))
        br, *_ = np.linalg.lstsq(Xr, yr, rcond=None); rssr = ((yr - Xr @ br) ** 2).sum()
        F = ((rssr - res @ res) / len(phcols)) / s2; p_ols = ss.f.sf(F, len(phcols), dfr)
        for j, term in enumerate(phcols):
            lo, hi = po.ci_profile(j); se = po.wald_se(j)
            seo = np.sqrt(cov_o[1 + j, 1 + j]); tq = ss.t.ppf(0.975, dfr)
            r = dict(component=k, phenotype=ph, model=lab, term=term, n=len(y), converged=ok,
                     OR=np.exp(beta[j]), OR_lo=np.exp(lo), OR_hi=np.exp(hi),
                     OR_wald_lo=np.exp(beta[j] - 1.96 * se), OR_wald_hi=np.exp(beta[j] + 1.96 * se), p_PO_LRT=p_lrt,
                     beta_rint_SD=bo[1 + j], rint_lo=bo[1 + j] - tq * seo, rint_hi=bo[1 + j] + tq * seo, p_rint_F=p_ols)
            # family q where the model is one of the published-family runs
            fam_ph = {"Sex": "Sex", "ceradsc": "ceradsc", "cogdx": "cogdx"}.get(ph)
            mm = {"M0": "PO0", "M1": "PO1", "M2": "PO2", "M3": "PO3", "drop1-M1": "PO1", "drop1-M2": "PO2"}.get(lab)
            if fam_ph and mm:
                fp = fam_po[(fam_ph, f"{mm}_p")]
                r["q_PO_family"] = q_sub(fp, k, p_lrt) if lab.startswith("drop1") else float(fam_po.loc[k, (fam_ph, f"{mm}_q")])
                rm = lab.replace("drop1-", "")
                fr = fam_ri[(fam_ri.pheno == fam_ph) & (fam_ri.model == rm)].set_index("comp").p
                r["q_rint_family"] = q_sub(fr, k, p_ols) if not lab.startswith("drop1") else rint_family_q(k, fam_ph, rm, pool, sub, phcols, cov)
            rows.append(r)
        print(f"C{k} {ph} {lab}: n {len(y)} " + "; ".join(f"{t} OR {np.exp(beta[j]):.3g}" for j, t in enumerate(phcols)) + f"; p {p_lrt:.2g}", flush=True)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/02_effects_ci.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
print(T[["component", "phenotype", "model", "term", "n", "OR", "OR_lo", "OR_hi", "OR_wald_lo", "OR_wald_hi", "p_PO_LRT", "q_PO_family",
         "beta_rint_SD", "rint_lo", "rint_hi", "p_rint_F", "q_rint_family"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
