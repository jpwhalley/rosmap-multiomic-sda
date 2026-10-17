"""Independent re-derivation of the C277 composition decomposition (statsmodels OLS on rank-INT score; own bootstrap),
plus alternative reasonable composition specifications and the RNA-mismatch exclusion. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import warnings, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
from scipy import stats as ss
from v_common import *
F = pd.read_csv(f"{CACHE}/frame.csv"); C = pd.read_csv(f"{CACHE}/counts.csv")
CL = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
tot = C[[f"cls|{c}" for c in CL]].sum(1)
for c in CL: F[f"p_{c}"] = C[f"cls|{c}"] / tot
F["p_neu"] = F.p_Exc + F.p_Inh
exc = [c for c in C.columns if c.startswith("Exc|")]; inh = [c for c in C.columns if c.startswith("Inh|")]
fe = C[exc].div(C[exc].sum(1), axis=0); fi = C[inh].div(C[inh].sum(1), axis=0)
F["l23"] = fe["Exc|Exc L2-3 CBLN2 LINC02306"]; F["car3"] = fe["Exc|Exc L5/6 IT Car3"]; F["themis"] = fe["Exc|Exc L6 THEMIS NFIA"]
L = np.log(C[exc] + 0.5); clr = L.sub(L.mean(1), axis=0)
Z = (clr - clr.mean()) / clr.std(); U, s, Vt = np.linalg.svd(Z.values, full_matrices=False)
for i in range(5): F[f"ePC{i+1}"] = U[:, i] * s[i]
for j, c in enumerate(exc[:-1]): F[f"eclr{j}"] = clr[c]          # 13 free CLR coordinates (drop one)
COV = ["age", "pmi", "ros", "male"]; PH = ["mci", "addem"]
C6 = ["p_Exc", "p_Inh", "p_Oli", "p_OPC", "p_Mic"]
SETS = {"C6": C6, "NEU": ["p_neu"], "SUB5": C6 + [f"ePC{i}" for i in range(1, 6)], "C6+L23": C6 + ["l23"],
        "C6+allExcCLR": C6 + [f"eclr{j}" for j in range(13)], "NEU+L23": ["p_neu", "l23"]}
def rintb(x):  # Blom offset, as the original
    x = np.asarray(x, float); return ss.norm.ppf((ss.rankdata(x) - 0.375) / (len(x) + 0.25))
def fitb(y, X):
    m = sm.OLS(y, sm.add_constant(X)).fit(); return m
rows = []
rng = np.random.default_rng(7)
for k in (277, 172):
    for samp in ("cogdx124", "cogdx124_noRNAmm"):
        d = F[F.cogdx.isin([1, 2, 4])]
        if samp.endswith("noRNAmm"): d = d[d.rna_mismatch == 0]
        d = d.reset_index(drop=True); y = rintb(d[f"s{k}"].values)
        mt = fitb(y, d[PH + COV]); bt = mt.params["addem"]; cit = mt.conf_int().loc["addem"].values
        for lab, comp in SETS.items():
            md = fitb(y, d[PH + COV + comp]); bd = md.params["addem"]; cid = md.conf_int().loc["addem"].values
            m0 = fitb(y, d[COV]); mc = fitb(y, d[COV + comp]); r2 = 1 - mc.ssr / m0.ssr
            B = []
            for b in range(2000):
                ii = rng.integers(0, len(d), len(d)); db = d.iloc[ii]; yb = rintb(db[f"s{k}"].values)
                Xt = np.column_stack([np.ones(len(db)), db[PH + COV].values]); Xd = np.column_stack([Xt, db[comp].values])
                b1 = np.linalg.lstsq(Xt, yb, rcond=None)[0][2]; b2 = np.linalg.lstsq(Xd, yb, rcond=None)[0][2]
                B.append((b1 - b2, (b1 - b2) / b1))
            B = np.array(B)
            rows.append(dict(component=k, sample=samp, comp=lab, n=len(d), total=bt, total_lo=cit[0], total_hi=cit[1], total_p=mt.pvalues["addem"],
                             direct=bd, direct_lo=cid[0], direct_hi=cid[1], direct_p=md.pvalues["addem"], share=(bt - bd) / bt,
                             share_lo=np.percentile(B[:, 1], 2.5), share_hi=np.percentile(B[:, 1], 97.5),
                             ind_lo=np.percentile(B[:, 0], 2.5), ind_hi=np.percentile(B[:, 0], 97.5), R2_comp=r2))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/v03_total_direct.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
print(R.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

print("\nSpearman of scores with composition (all 276 | excluding 2 RNA-mismatched, n 274):")
for k in (277, 172):
    for lab, d in (("276", F), ("274", F[F.rna_mismatch == 0])):
        print(f"  C{k} n{lab}: " + ", ".join(f"{c} {ss.spearmanr(d[f's{k}'], d[c])[0]:+.3f}" for c in ["p_Exc", "p_Inh", "p_Oli", "p_Mic", "p_neu", "l23", "car3", "themis", "ePC1"]))

print("\nComposition ~ MCI + AD dementia + covariates (n 254; beta in SD of the feature), with BH over the 15 features tested by neuronal_c21:")
d = F[F.cogdx.isin([1, 2, 4])].reset_index(drop=True)
feat = {"p_Ast": d.p_Ast, "p_Oli": d.p_Oli, "p_Inh": d.p_Inh, "p_OPC": d.p_OPC, "p_Exc": d.p_Exc, "p_Mic": d.p_Mic, "p_neu": d.p_neu,
        "ePC1": d.ePC1, "ePC2": d.ePC2, "l23": d.l23, "car3": d.car3, "themis": d.themis,
        "Inh LAMP5 RELN": fi.loc[F.cogdx.isin([1, 2, 4]).values, "Inh|Inh LAMP5 RELN"].values,
        "Inh FBN2 EPB41L4A": fi.loc[F.cogdx.isin([1, 2, 4]).values, "Inh|Inh FBN2 EPB41L4A"].values}
rr = []
for nm, v in feat.items():
    v = np.asarray(v, float); v = (v - v.mean()) / v.std()
    m = fitb(v, d[PH + COV]); rr.append(dict(feature=nm, beta_AD=m.params["addem"], lo=m.conf_int().loc["addem", 0], hi=m.conf_int().loc["addem", 1], p=m.pvalues["addem"]))
T = pd.DataFrame(rr)
from statsmodels.stats.multitest import multipletests
T["q_BH_14"] = multipletests(T.p, method="fdr_bh")[1]
T.to_csv(f"{OUT}/v03_composition_vs_cogdx.csv", index=False, float_format="%.4g")
print(T.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
F.to_csv(f"{CACHE}/frame_comp.csv", index=False)
