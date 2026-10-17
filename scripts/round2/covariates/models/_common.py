# Shared loading and helpers (same definitions as 01_reproduce_and_adjust.py). exec()'d by 01b_/02_/03_.
# Paths come from the calling script's `cfg` (scripts/round2/config.py).
import numpy as np, pandas as pd
from scipy import stats as ss
B = cfg.ROS; AUD = cfg.AUD
REF = f"{cfg.TABLES}/clinical_BH_547_family.csv"
dep = pd.read_csv(f"{B}/sda/samples_final.csv", index_col=0); dep.columns = dep.columns.astype(str)
S = dep.drop(index=520).copy()
for _k in (547, 548):
    S.loc[_k] = np.load(f"{AUD}/c547/C{_k}_A.npy")
don = S.columns
clin = pd.read_csv(f"{B}/oth/meta/ROSMAP_clinical.csv"); clin.index = clin.projid.astype(int).astype(str); clin = clin.loc[don]
_qc = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); _qc.index = _qc.index.astype(str); _qc = _qc.loc[don]
ref = pd.read_csv(REF, index_col=0)
D = pd.DataFrame(index=don)
D["sex"] = clin.msex.astype(float)
D["cerad_ad"] = clin.ceradsc.map({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0})
D["ceradsc"] = clin.ceradsc.astype(float)
D["cogdx"] = clin.cogdx.astype(float)
_age = clin.age_death.astype(str)
D["age90p"] = (_age == "90+").astype(float)
D["age"] = pd.to_numeric(_age.replace("90+", "90")).clip(upper=90.0)
D["pmi"] = clin.pmi.astype(float); D["logpmi"] = np.log(D.pmi)
D["ros"] = (clin.Study == "ROS").astype(float)
D["lognuc"] = np.log(_qc.nuclei); D["logupn"] = np.log(_qc.umis_per_nucleus)

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o]
    adj = np.minimum.accumulate((r * n / np.arange(1, n + 1))[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(adj, 1); return out

def rint(x):
    x = np.asarray(x, float); n = len(x); return ss.norm.ppf((ss.rankdata(x) - 0.375) / (n + 0.25))

CER = {1: "AD", 2: "AD", 3: "No", 4: "No"}
def tests(v, idx=None):
    idx = don if idx is None else idx
    s = pd.Series(np.asarray(v, float), index=idx); m = clin.msex.loc[idx]; cb = clin.ceradsc.loc[idx].map(CER); cg = clin.cogdx.loc[idx]
    return (ss.mannwhitneyu(s[m == 1], s[m == 0], alternative="two-sided").pvalue,
            ss.mannwhitneyu(s[cb == "AD"], s[cb == "No"], alternative="two-sided").pvalue,
            ss.kruskal(*[s[cg == k] for k in (1, 2, 4)]).pvalue)

Y = np.column_stack([rint(S.loc[i].values) for i in S.index]); Ydf = pd.DataFrame(Y, index=don, columns=S.index)

def design(cols, sub, Dx=None):
    Dx = D if Dx is None else Dx
    X = [np.ones(len(sub))]; names = ["Intercept"]
    for c in cols:
        if c == "C(cogdx)":
            for lev in (2.0, 4.0):
                X.append((Dx.loc[sub, "cogdx"] == lev).astype(float).values); names.append(f"cogdx{int(lev)}")
        elif c == "C(cogdx6)":
            for lev in (2.0, 3.0, 4.0, 5.0, 6.0):
                X.append((Dx.loc[sub, "cogdx"] == lev).astype(float).values); names.append(f"cogdx{int(lev)}")
        else:
            X.append(Dx.loc[sub, c].values); names.append(c)
    return np.column_stack(X), names

def nested_F(ph_terms, cov_terms, sub, Yd=None, Dx=None):
    Yd = Ydf if Yd is None else Yd
    Xf, nf = design(ph_terms + cov_terms, sub, Dx); Xr, _ = design(cov_terms, sub, Dx)
    Ys = Yd.loc[sub].values
    bf, *_ = np.linalg.lstsq(Xf, Ys, rcond=None); rssf = ((Ys - Xf @ bf) ** 2).sum(0)
    br, *_ = np.linalg.lstsq(Xr, Ys, rcond=None); rssr = ((Ys - Xr @ br) ** 2).sum(0)
    q = Xf.shape[1] - Xr.shape[1]; dff = len(sub) - Xf.shape[1]
    F = ((rssr - rssf) / q) / (rssf / dff); p = ss.f.sf(F, q, dff)
    return p, {nf[j]: bf[j] for j in range(1, 1 + q)}, dff

PHTERM = {"Sex": ["sex"], "ceradsc": ["cerad_ad"], "cogdx": ["C(cogdx)"]}
SUB = {"Sex": list(don), "ceradsc": list(don), "cogdx": list(D.index[D.cogdx.isin([1, 2, 4])])}
