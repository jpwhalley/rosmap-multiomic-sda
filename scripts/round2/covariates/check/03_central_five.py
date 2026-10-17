"""Central five associations: published q, own PO and RINT-OLS adjusted q (family-wide BH over 547), ORs,
unadjusted effect sizes, ordinal / categorical codings, and dropping the single most extreme donor (whole family refit).
Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss
from vc_common import *

s = load_scores(); don = s.columns; cl = load_clin(don); qc = load_qc(don)
un = pd.read_csv("vc_unadj.csv", index_col=0); po = pd.read_csv("vc_po_547.csv"); ri = pd.read_csv("vc_rint_547.csv")
FIVE = [(74, "Sex"), (278, "Sex"), (545, "Sex"), (172, "ceradsc"), (277, "cogdx")]
age, pmi, mp, male = cl.age.values, cl.pmi.values, cl.is_map.values, cl.male.values
lnuc, lupn = np.log(qc.nuclei.values), np.log(qc.upn.values)
cog = cl.cogdx.values; sub = np.isin(cog, [1, 2, 4])

def design(ph, model, keep):
    base = {"M0": [], "M1": [age, pmi, mp], "M2": [age, pmi, mp, lnuc, lupn], "M3": [age, cl.age90.values, np.log(pmi), mp]}[model]
    if ph != "Sex" and model != "M0":
        base = base + [male]
    Xc = np.column_stack(base)[keep] if base else np.zeros((keep.sum(), 0))
    if ph == "Sex": Xp = male[keep][:, None]
    elif ph == "ceradsc": Xp = cl.cerad_ad.values[keep][:, None]
    else: Xp = np.column_stack([cog[keep] == 2, cog[keep] == 4]).astype(float)
    return Xc, Xp

def family_po(ph, model, keep):
    """Refit every component on donors `keep` (phenotype subset applied) and BH across 547."""
    k2 = keep & (sub if ph == "cogdx" else True)
    Xc, Xp = design(ph, model, k2)
    p = np.array([po_lrt(s.loc[k].values[k2], Xc, Xp)["p"] for k in s.index])
    return pd.Series(bh(p), index=s.index), pd.Series(p, index=s.index)

def family_rint(ph, model, keep):
    R = np.column_stack([rint(s.loc[k].values[keep]) for k in s.index])  # RINT over the retained donors
    m = (sub[keep] if ph == "cogdx" else np.ones(keep.sum(), bool))
    k2 = keep.copy(); k2[keep] = m
    Xc, Xp = design(ph, model, k2)
    p, _ = ols_F(R[m], Xc, Xp)
    return pd.Series(bh(p), index=s.index)

def family_pub(ph, keep):
    c2 = cl[keep]
    p = np.array([published_tests(s.loc[k].values[keep], c2)[["Sex", "ceradsc", "cogdx"].index(ph)] for k in s.index])
    return pd.Series(bh(p), index=s.index)

rows = []
for k, ph in FIVE:
    g = lambda df, m, col: df[(df.comp == k) & (df.pheno == ph) & (df.model == m)][col].iloc[0]
    y = s.loc[k].values
    r = dict(comp=k, pheno=ph, q_pub=un.loc[k, f"q_{ph}"], p_pub=un.loc[k, f"p_{ph}"],
             q_po1=g(po, "M1", "q"), q_po2=g(po, "M2", "q"), q_po3=g(po, "M3", "q"), q_po0=g(po, "M0", "q"),
             OR_po0=np.exp(g(po, "M0", "beta")), OR_po1=np.exp(g(po, "M1", "beta")), OR_po2=np.exp(g(po, "M2", "beta")),
             q_ri1=g(ri, "M1", "q"), q_ri2=g(ri, "M2", "q"), q_ri0=g(ri, "M0", "q"),
             b_ri0=g(ri, "M0", "beta"), b_ri1=g(ri, "M1", "beta"), b_ri2=g(ri, "M2", "beta"))
    # unadjusted effect sizes
    if ph in ("Sex", "ceradsc"):
        grp = male if ph == "Sex" else cl.cerad_ad.values
        U = ss.mannwhitneyu(y[grp == 1], y[grp == 0]).statistic
        r["AUC(group1>group0)"] = U / ((grp == 1).sum() * (grp == 0).sum())
        if k == 74:
            thr_rank = np.sort(y)
            # number of donors on the wrong side of the best single threshold
            best = min(min(((y > t) != (grp == 1)).sum(), ((y > t) != (grp == 0)).sum()) for t in np.unique(y))
            r["min_misclassified"] = best
    else:
        H = ss.kruskal(*[y[cog == c] for c in (1, 2, 4)]).statistic
        r["KW_H"] = H; r["eps2"] = H / (sub.sum() - 1)
        r["OR_MCI_po1"] = np.exp(po[(po.comp == k) & (po.pheno == ph) & (po.model == "M1")].beta_mci.iloc[0])
    # most extreme donor (robust z) -> drop and refit whole family
    med = np.median(y); mad = ss.median_abs_deviation(y, scale="normal")
    z = np.abs(y - med) / mad; j = int(np.argmax(z)); keep = np.ones(276, bool); keep[j] = False
    r["max_robust_z"] = z[j]
    r["drop1_q_pub"] = family_pub(ph, keep)[k]
    r["drop1_q_po1"] = family_po(ph, "M1", keep)[0][k]
    r["drop1_q_po2"] = family_po(ph, "M2", keep)[0][k]
    r["drop1_q_ri1"] = family_rint(ph, "M1", keep)[k]
    r["drop1_q_ri2"] = family_rint(ph, "M2", keep)[k]
    rows.append(r)
    print({kk: (f"{v:.3g}" if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv("vc_central_five.csv", index=False)

# ---------------- ordinal / categorical codings ----------------
print("\n=== ordinal / categorical codings ===")
cer = cl.ceradsc.values
y172 = s.loc[172].values
rho, p = ss.spearmanr(y172, cer); print(f"C172 vs CERAD 1-4 Spearman rho {rho:.3f} p {p:.2g}")
R = np.column_stack([rint(s.loc[k].values) for k in s.index]); idx = list(s.index)
for model, base in [("M1", [age, pmi, mp, male]), ("M2", [age, pmi, mp, male, lnuc, lupn])]:
    pf, b = ols_F(R, np.column_stack(base), cer[:, None]); qf = bh(pf)
    print(f"  RINT ~ CERAD(1-4 linear) + {model} covariates: C172 p {pf[idx.index(172)]:.2g}, family q {qf[idx.index(172)]:.2g}")
    # CERAD as the ordinal outcome, score + covariates as predictors
    Xc = np.column_stack(base)
    r = po_lrt(cer, Xc, y172[:, None]); print(f"  PO logit, CERAD 1-4 as outcome, score + {model} covariates: C172 LRT p {r['p']:.2g}")
y277 = s.loc[277].values
trend = np.select([cog == 1, cog == 2, cog == 4], [0, 1, 2], np.nan)
for model, base in [("M1", [age, pmi, mp, male]), ("M2", [age, pmi, mp, male, lnuc, lupn])]:
    pf, b = ols_F(R[sub], np.column_stack(base)[sub], trend[sub][:, None]); qf = bh(pf)
    print(f"  RINT ~ cogdx trend NCI<MCI<AD + {model}: C277 p {pf[idx.index(277)]:.2g}, family q {qf[idx.index(277)]:.2g}")
    D6 = np.column_stack([cog == c for c in (2, 3, 4, 5, 6)]).astype(float)
    pf, b = ols_F(R, np.column_stack(base), D6); qf = bh(pf)
    print(f"  RINT ~ all six cogdx (5 df, n=276) + {model}: C277 p {pf[idx.index(277)]:.2g}, family q {qf[idx.index(277)]:.2g}; n in cogdx 3/5/6 = {int(np.isin(cog, [3, 5, 6]).sum())}")
    pf, b = ols_F(R[sub], np.column_stack(base)[sub], np.column_stack([cog[sub] == 2, cog[sub] == 4]).astype(float))
    print(f"  RINT {model} C277 betas: MCI {b[0, idx.index(277)]:.3f}, AD {b[1, idx.index(277)]:.3f}")
# C277 nuclei association and AD beta
print("C277 score vs log nuclei: Spearman p", f"{ss.spearmanr(y277, lnuc)[1]:.2g}", "| RINT OLS p (with M1 covs, cogdx subset)",
      f"{ols_F(R[sub][:, [idx.index(277)]], np.column_stack([age, pmi, mp, male, (cog == 2), (cog == 4), lupn])[sub], lnuc[sub][:, None])[0][0]:.2g}")
