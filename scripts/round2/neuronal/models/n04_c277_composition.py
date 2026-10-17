"""(3) C277 composition: split the AD-dementia association of the C277 score into a part that runs through donor
cell-type composition and a part that does not (expression within excitatory neurons).
Pieces (rank-INT OLS, M1 covariates age/PMI/cohort/sex; cogdx 1/2/4, n = 254 unless stated):
  a. score vs composition: Spearman with the six class proportions (nuclei per class / total, ros0 filters) and with
     the within-class subtype fractions of Exc and Inh nuclei (cell_type_high_resolution);
  b. composition vs cogdx: each proportion ~ MCI + AD dementia + M1 (beta in SD of the proportion);
  c. score ~ cogdx + M1 (total) vs score ~ cogdx + M1 + composition (direct). Composition sets:
       C6  : five class proportions (Ast reference; the dossier's set),
       NEU : neuronal fraction (Exc + Inh) alone,
       SUB : C6 + first 5 PCs of centred-log-ratio within-Exc subtype fractions,
     indirect = total - direct, with a donor bootstrap (2,000 resamples) for the CI and the share via composition;
  d. how much of the score variance composition explains (R2), and whether the residual score (composition removed)
     still separates AD dementia from NCI;
  e. the same for C172 (comparison) and for PLCB1 in Exc (the top gene), as a gene-level check.
The two donors whose snRNA profile is another person's are dropped in a sensitivity row. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy import stats as ss
from nc_common import *

S = scores547(); D = full_frame()
SUBC = pd.read_csv(f"{CACHE}/subtype_counts.csv", index_col=0); SUBC.index = SUBC.index.astype(str); SUBC = SUBC.loc[D.index]
COV = ["age", "pmi", "ros", "sex"]
for c in ("Exc", "Inh"):
    cols = [x for x in SUBC.columns if x.startswith(c + "|")]
    frac = SUBC[cols].div(SUBC[cols].sum(1), axis=0)
    for x in cols:
        D[f"sf_{x}"] = frac[x]
    if c == "Exc":
        L = np.log(SUBC[cols] + 0.5); clr = L.sub(L.mean(1), axis=0); Z = (clr - clr.mean()) / clr.std()
        U, s_, Vt = np.linalg.svd(Z.values, full_matrices=False)
        for i in range(5):
            D[f"excPC{i+1}"] = U[:, i] * s_[i]
        print("within-Exc subtype CLR PCA: variance explained by PC1-5:", np.round(s_[:5] ** 2 / (s_ ** 2).sum(), 3).tolist())
D["prop_neuron"] = D.prop_Exc + D.prop_Inh
C6 = ["prop_Exc", "prop_Inh", "prop_Oli", "prop_OPC", "prop_Mic"]
SETS = {"C6": C6, "NEU": ["prop_neuron"], "SUB": C6 + [f"excPC{i}" for i in range(1, 6)]}
sub124 = D.index[D.cogdx.isin([1, 2, 4])]
PH = ["cogdx_mci", "cogdx_ad"]

def score_vec(k, idx):
    return pd.Series(rint(S.loc[k, idx].values.astype(float)), index=idx)

# ---- a. score vs composition
print("\n[a] Spearman of score with class proportions (n = 276):")
for k in (277, 172):
    a = S.loc[k].values
    print(f"  C{k}: " + ", ".join(f"{c} {ss.spearmanr(a, D[f'prop_{c}'])[0]:+.2f}" for c in CTS) + f", neuronal (Exc+Inh) {ss.spearmanr(a, D.prop_neuron)[0]:+.2f}")
rows = []
for k in (277, 172):
    a = S.loc[k].values
    for x in [c for c in D.columns if c.startswith("sf_")] + [f"excPC{i}" for i in range(1, 6)]:
        r, p = ss.spearmanr(a, D[x]); rows.append(dict(component=k, feature=x.replace("sf_", ""), rho=r, p=p))
A = pd.DataFrame(rows); A["q"] = A.groupby("component").p.transform(bh); A.to_csv(f"{OUT}/n04a_score_vs_subtypes.csv", index=False, float_format="%.4g")
for k in (277, 172):
    t = A[A.component == k].reindex(A[A.component == k].rho.abs().sort_values(ascending=False).index).head(8)
    print(f"  C{k} strongest subtype-composition correlates:\n" + t[["feature", "rho", "q"]].to_string(index=False, float_format=lambda x: f"{x:.2g}"))

# ---- b. composition vs cogdx
print("\n[b] composition ~ MCI + AD dementia + M1 (n = %d); beta in SD of the proportion; and medians by group" % len(sub124))
rows = []
for c in ["prop_" + x for x in CTS] + ["prop_neuron", "excPC1", "excPC2", "sf_Exc|Exc L2-3 CBLN2 LINC02306", "sf_Exc|Exc L5/6 IT Car3", "sf_Exc|Exc L6 THEMIS NFIA", "sf_Inh|Inh LAMP5 RELN", "sf_Inh|Inh FBN2 EPB41L4A"]:
    y = D.loc[sub124, c].values.astype(float); y = (y - y.mean()) / y.std()
    T, _, _ = ols(y, D.loc[sub124, PH + COV].values.astype(float), PH + COV)
    med = D.loc[sub124].groupby("cogdx")[c].median()
    rows.append(dict(feature=c, beta_AD=T.loc["cogdx_ad", "beta"], lo=T.loc["cogdx_ad", "lo"], hi=T.loc["cogdx_ad", "hi"], p_AD=T.loc["cogdx_ad", "p"],
                     beta_MCI=T.loc["cogdx_mci", "beta"], p_MCI=T.loc["cogdx_mci", "p"], med_NCI=med.get(1.0), med_MCI=med.get(2.0), med_AD=med.get(4.0)))
Bt = pd.DataFrame(rows); Bt.to_csv(f"{OUT}/n04b_composition_vs_cogdx.csv", index=False, float_format="%.4g")
print(Bt.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# ---- c/d. total vs direct, bootstrap
rng = np.random.default_rng(20261013)
def total_direct(k, idx, comp):
    y = score_vec(k, idx).values
    Tt, _, _ = ols(y, D.loc[idx, PH + COV].values.astype(float), PH + COV)
    Td, _, _ = ols(y, D.loc[idx, PH + COV + comp].values.astype(float), PH + COV + comp)
    return Tt.loc["cogdx_ad"], Td.loc["cogdx_ad"]
rows = []
excl = set(rna_mismatch_donors())
for k in (277, 172):
    for label, comp in SETS.items():
        for samp, idx in (("cogdx124", list(sub124)), ("cogdx124_minus_RNAmismatch", [x for x in sub124 if x not in excl])):
            tt, td = total_direct(k, idx, comp)
            boots = []
            if samp == "cogdx124":
                for b in range(2000):
                    bi = list(rng.choice(idx, size=len(idx), replace=True))
                    y = rint(S.loc[k, bi].values.astype(float))
                    Xt = np.column_stack([np.ones(len(bi)), D.loc[bi, PH + COV].values.astype(float)])
                    Xd = np.column_stack([Xt, D.loc[bi, comp].values.astype(float)])
                    bt = np.linalg.lstsq(Xt, y, rcond=None)[0][2]; bd = np.linalg.lstsq(Xd, y, rcond=None)[0][2]
                    boots.append((bt, bd, bt - bd, (bt - bd) / bt if bt != 0 else np.nan))
            boots = np.array(boots) if boots else np.full((1, 4), np.nan)
            # composition R2 and residual-score contrast
            y = score_vec(k, idx).values
            Xc = D.loc[idx, comp + COV].values.astype(float); _, rss_c, _ = ols(y, Xc, comp + COV)
            _, rss_0, _ = ols(y, D.loc[idx, COV].values.astype(float), COV)
            rows.append(dict(component=k, composition=label, sample=samp, n=len(idx),
                             total_SD=tt.beta, total_lo=tt.lo, total_hi=tt.hi, total_p=tt.p,
                             direct_SD=td.beta, direct_lo=td.lo, direct_hi=td.hi, direct_p=td.p,
                             indirect_SD=tt.beta - td.beta, indirect_lo=np.nanpercentile(boots[:, 2], 2.5), indirect_hi=np.nanpercentile(boots[:, 2], 97.5),
                             share_via_composition=(tt.beta - td.beta) / tt.beta,
                             share_lo=np.nanpercentile(boots[:, 3], 2.5), share_hi=np.nanpercentile(boots[:, 3], 97.5),
                             R2_composition_beyond_covariates=1 - rss_c / rss_0))
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/n04c_total_direct.csv", index=False, float_format="%.4g")
print("\n[c] AD dementia vs NCI, rank-INT score (SD units): total (M1) vs direct (M1 + composition); bootstrap 2,000")
print(C.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# ---- e. PLCB1 (Exc) gene-level check; and the PO OR with composition from the dossier for reference
from source_panels_bridge import feature_values_by_name
y_pl = feature_values_by_name("PLCB1", "Exc")
idx = [x for x in sub124 if x not in excl]
pos = [list(D.index).index(x) for x in idx]
yv = rint(y_pl[pos])
rows = []
for label, comp in [("none", [])] + list(SETS.items()):
    T, _, _ = ols(yv, D.loc[idx, PH + COV + comp].values.astype(float), PH + COV + comp)
    rows.append(dict(gene="PLCB1", cell_type="Exc", composition=label, n=len(idx), beta_AD_SD=T.loc["cogdx_ad", "beta"],
                     lo=T.loc["cogdx_ad", "lo"], hi=T.loc["cogdx_ad", "hi"], p=T.loc["cogdx_ad", "p"]))
G = pd.DataFrame(rows); G.to_csv(f"{OUT}/n04e_plcb1_cogdx.csv", index=False, float_format="%.4g")
print("\n[e] PLCB1 in Exc (SDA input CPM, rank-INT), AD dementia vs NCI, M1 +/- composition (RNA-mismatched donors excluded):")
print(G.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
ref = pd.read_csv(f"{R2}/component_dossier/out/02_effects_ci.csv")
print("\n[ref] dossier PO OR (AD dementia vs NCI) for C277:")
print(ref[(ref.component == 277) & (ref.phenotype == "cogdx") & (ref.term == "cogdx4")][["model", "n", "OR", "OR_lo", "OR_hi", "p_PO_LRT"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
