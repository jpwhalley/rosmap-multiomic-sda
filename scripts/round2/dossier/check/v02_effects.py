"""V02: independent effect estimates + 95% CIs for the central associations.
PO: statsmodels OrderedModel (logit) on dense ranks of the score (one category per distinct value); Wald CI from the
fitted covariance; profile-likelihood CI by refitting with the phenotype coefficient as an offset (models with >=1
covariate). LRT p vs the covariate-only model (M0: analytic thresholds-only log-likelihood).
RINT: OLS of rank-based inverse normal (two offsets: 0.5 and Blom 3/8) on the same design; t CI; nested F.
Family q (547, BH per phenotype): RINT recomputed for all 547 components here (both offsets); PO by substituting this
p into covariates/po_tests_547.csv (others' p from file). Contrast coding is direct (female = 1, AD = 1, ...), no flips.
Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, warnings, time, numpy as np, pandas as pd
from scipy import stats as ss
from scipy.optimize import brentq
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
from v00_common import *
warnings.filterwarnings("ignore")
t0 = time.time()
S = scores(); don = donors(); c = clin()
q = pd.read_csv(f"{CACHE}/qc_donor.csv", index_col=0); q.index = q.index.astype(str); q = q.loc[don]
D = pd.DataFrame(index=don)
D["female"] = (c.msex == 0).astype(float); D["male"] = 1 - D.female
D["ad"] = c.ceradsc.isin([1, 2]).astype(float)
D["cerad_more"] = 5 - c.ceradsc                        # higher = more neuritic plaques
a = c.age_death.astype(str)
D["age"] = pd.to_numeric(a.str.replace("+", "", regex=False)).clip(upper=90).astype(float)
D["age90p"] = (a == "90+").astype(float)
D["pmi"] = c.pmi.astype(float); D["logpmi"] = np.log(D.pmi)
D["ros"] = (c.Study == "ROS").astype(float)
D["lognuc"] = np.log(q.nuclei); D["logupn"] = np.log(q.upn)
for k_ in ("Exc", "Inh", "Oli", "OPC", "Mic"):
    D[f"p_{k_}"] = q[f"prop_{k_}"]
D["cog"] = c.cogdx.astype(float)
D["mci"] = (D.cog == 2).astype(float); D["adem"] = (D.cog == 4).astype(float)
D["cogstep"] = D.cog.map({1: 0.0, 2: 1.0, 4: 2.0})
for l in (2, 3, 4, 5, 6): D[f"cg{l}"] = (D.cog == l).astype(float)
COMP5 = ["p_Exc", "p_Inh", "p_Oli", "p_OPC", "p_Mic"]
COG6 = [f"cg{l}" for l in (2, 3, 4, 5, 6)]
print("checks: n female", int(D.female.sum()), "| CERAD AD", int(D.ad.sum()), "| cogdx 1/2/4:", int((D.cog == 1).sum()), int((D.cog == 2).sum()), int((D.cog == 4).sum()))

def M(ph, name):
    sx = [] if ph == "sex" else ["male"]
    return {"M0": [], "M1": ["age", "pmi", "ros"] + sx, "M2": ["age", "pmi", "ros"] + sx + ["lognuc", "logupn"],
            "M3": ["age", "age90p", "logpmi", "ros"] + sx}[name]

PH = {"sex": ["female"], "cerad": ["ad"], "cog3": ["mci", "adem"], "cerad_step": ["cerad_more"], "cog_step": ["cogstep"]}
SUBS = {"cog3": D.index[D.cog.isin([1, 2, 4])], "cog_step": D.index[D.cog.isin([1, 2, 4])]}

def ll_null(yk):
    cnt = np.bincount(yk); cnt = cnt[cnt > 0]; n = cnt.sum(); return float(np.sum(cnt * np.log(cnt / n)))

def po(y, X, names, off=None, start=None):
    yk = ss.rankdata(y, method="dense").astype(int) - 1
    if X.shape[1] == 0:
        return ll_null(yk), None, None
    m = OrderedModel(yk, X, distr="logit", offset=off)
    r = m.fit(start_params=start, method="bfgs", maxiter=20000, disp=False, gtol=1e-7)
    return r.llf, r.params[:X.shape[1]], r

def fit_one(k, ph, cov, sub, extra_label):
    y = S.loc[k, sub].values.astype(float)
    Xp = D.loc[sub, PH[ph]].values; Xc = D.loc[sub, cov].values
    X = np.column_stack([Xp, Xc]) if cov else Xp
    llf, b, res = po(y, X, PH[ph] + cov)
    ll0 = po(y, Xc, cov)[0] if cov else ll_null(ss.rankdata(y, method="dense").astype(int) - 1)
    lr = max(2 * (llf - ll0), 0); p_lrt = ss.chi2.sf(lr, len(PH[ph]))
    cv = res.cov_params()
    out = []
    for j, t in enumerate(PH[ph]):
        se = float(np.sqrt(cv[j, j])) if not hasattr(cv, "iloc") else float(np.sqrt(cv.iloc[j, j]))
        lo_w, hi_w = b[j] - 1.96 * se, b[j] + 1.96 * se
        lo_p = hi_p = np.nan
        if len(PH[ph]) + len(cov) >= 2 and extra_label in PROFILE_MODELS:
            others = [i for i in range(X.shape[1]) if i != j]
            st = np.delete(np.asarray(res.params), j)
            def f(bb):
                ll = po(y, X[:, others], None, off=X[:, j] * bb, start=st)[0]
                return 2 * (llf - ll) - ss.chi2.ppf(0.95, 1)
            try:
                step = 2.5 * se
                lo_p = brentq(f, b[j] - 4 * step, b[j], xtol=1e-5) if f(b[j] - 4 * step) > 0 else np.nan
                hi_p = brentq(f, b[j], b[j] + 4 * step, xtol=1e-5) if f(b[j] + 4 * step) > 0 else np.nan
            except Exception as e:
                print("profile failed", k, ph, t, e)
        out.append(dict(comp=k, pheno=ph, model=extra_label, term=t, n=len(sub), OR=np.exp(b[j]),
                        OR_wald_lo=np.exp(lo_w), OR_wald_hi=np.exp(hi_w), OR_prof_lo=np.exp(lo_p), OR_prof_hi=np.exp(hi_p),
                        p_PO_LRT=p_lrt))
    # RINT OLS, both offsets, RINT over the analysed pool (all donors minus any dropped donor), then subset
    pool = [x for x in don if x not in DROPPED]
    for nm, off in (("r05", 0.5), ("r38", 0.375)):
        v = S.loc[k, pool].values.astype(float)
        rv = pd.Series(ss.norm.ppf((ss.rankdata(v) - off) / (len(v) + 1 - 2 * off)), index=pool).loc[sub].values
        Xo = sm.add_constant(X, has_constant="add"); fo = sm.OLS(rv, Xo).fit()
        Xr = sm.add_constant(Xc, has_constant="add") if cov else np.ones((len(sub), 1)); fr = sm.OLS(rv, Xr).fit()
        F = ((fr.ssr - fo.ssr) / len(PH[ph])) / (fo.ssr / fo.df_resid); pF = ss.f.sf(F, len(PH[ph]), fo.df_resid)
        ci = fo.conf_int()
        for j, t in enumerate(PH[ph]):
            out[j][f"{nm}_b"] = fo.params[1 + j]; out[j][f"{nm}_lo"] = ci[1 + j, 0]; out[j][f"{nm}_hi"] = ci[1 + j, 1]
            out[j][f"{nm}_pF"] = pF
    return out

PROFILE_MODELS = {"M1", "M2"}
DROPPED = set()
TASKS = [(74, "sex"), (278, "sex"), (545, "sex"), (172, "cerad"), (277, "cog3"), (172, "cerad_step"), (277, "cog_step")]
EXTRA = {(278, "sex"): {"M1+CERAD+cogdx6": M("sex", "M1") + ["ad"] + COG6, "M1+composition": M("sex", "M1") + COMP5},
         (545, "sex"): {"M1+composition": M("sex", "M1") + COMP5},
         (172, "cerad"): {"M1+composition": M("cerad", "M1") + COMP5},
         (277, "cog3"): {"M1+composition": M("cog3", "M1") + COMP5, "M2+composition": M("cog3", "M2") + COMP5,
                         "M1+CERAD": M("cog3", "M1") + ["ad"]}}
only = [int(x) for x in sys.argv[1:]]
rows = []
for k, ph in TASKS:
    if only and k not in only: continue
    sub0 = list(SUBS.get(ph, D.index))
    sc = S.loc[k]
    ext_all = (sc - sc.median()).abs().idxmax()
    ext_sub = (sc.loc[sub0] - sc.loc[sub0].median()).abs().idxmax()
    specs = [(m, M(ph if ph in ("sex",) else "x", m), sub0, set()) for m in ("M0", "M1", "M2", "M3")]
    for m in ("M1", "M2"):
        specs.append((f"drop1-{m}", M(ph if ph == "sex" else "x", m), [x for x in sub0 if x != ext_all], {ext_all}))
    if ext_sub != ext_all:
        print(f"C{k} {ph}: most-extreme donor differs between all-donor and subset medians")
    specs += [(lab, cv, sub0, set()) for lab, cv in EXTRA.get((k, ph), {}).items()]
    for lab, cv, sub, drp in specs:
        DROPPED = drp
        rows += fit_one(k, ph, cv, sub, lab)
        r = rows[-1]
        print(f"C{k} {ph} {lab}: OR {r['OR']:.3g} [{r['OR_prof_lo']:.3g},{r['OR_prof_hi']:.3g}] wald [{r['OR_wald_lo']:.3g},{r['OR_wald_hi']:.3g}] p {r['p_PO_LRT']:.2g} | rint {r['r38_b']:+.3f} [{r['r38_lo']:+.3f},{r['r38_hi']:+.3f}] ({time.time()-t0:.0f}s)", flush=True)
T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/v02_effects{'_' + '_'.join(map(str, only)) if only else ''}.csv", index=False, float_format="%.5g")
print("done", f"{time.time()-t0:.0f}s")
