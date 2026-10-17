"""Proteomics-influence PROXY per component (the SDA refit is the real test). Aggregates only.
29 donors: prot.txt rows that are another individual's proteome (ROSMAP210 batch.channel collision; mask from
source_panels/05f). 'Gained' donors: have their own ROSMAP.DLPFC TMT column under the correct mapping but no row in prot.txt.
For each component:
 (a) is the dominant donor (top share > 0.5) among the 29? does it have an own DLPFC proteome?
 (b) Mann-Whitney of scores: 29 vs the other 76 proteomics donors, and 29 vs all 247 others (AUC = P(29 > other)); also on
     |score - median| (the sign of a score is arbitrary).
 (c) loading energy by modality: raw sum of squared loadings, and the standardised reconstruction energy
     E_m = sum_{donors observed in m} a_i^2 * sum_f x_f^2 / v_f (RNA: x = b_c * x_g over cell type x gene), v_f = feature
     variance across observed donors; share of E per modality.
 (d) fixed-loading projection: precision-weighted (1/v_f) least-squares estimate of each donor's score from this
     component's loadings alone, per modality and combined over the donor's observed modalities; recomputed with the
     mis-mapped protein rows replaced by the correct DLPFC rows (or NA) and the gained donors' rows added.
     Validated by Pearson/Spearman of the combined estimate with the SDA score. Change reported in SD units of the estimate.
 (e) the donor whose standardised protein row best matches the component's protein loadings (|r|), and whether it is one of the 29."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import time, numpy as np, pandas as pd
from scipy import stats as ss
from dc_common import *
t0 = time.time()
L = dict(np.load(f"{CACHE}/loadings.npz", allow_pickle=True))
S = scores547(); don = S.columns
mis = np.load(f"{cfg.WORK}/source_panels/cache/prot_rows_misattributed.npy")
Pc = np.load(f"{cfg.WORK}/source_panels/cache/prot_corrected_full.npy")
def load_input(m):
    f = f"{CACHE}/input_{m}.npy"
    if os.path.exists(f): return np.load(f)
    V = pd.read_csv(f"{SDA}/{INPUT[m]}", sep=" ", header=None, na_values="NA", dtype=np.float32).values
    np.save(f, V); return V
V = {m: load_input(m).astype(np.float64) for m in MODS}; print("inputs", {m: V[m].shape for m in V}, f"{time.time()-t0:.0f}s", flush=True)
Po = V["proteins"]
has_o = ~np.isnan(Po).all(1); has_c = ~np.isnan(Pc).all(1)
gained = has_c & ~has_o
own_among_mis = mis & has_c
print(f"prot.txt donors {has_o.sum()}; mis-attributed {mis.sum()} (of which own DLPFC column exists {own_among_mis.sum()});"
      f" correct-mapping donors {has_c.sum()}; gained {gained.sum()}; correct rows unchanged {(has_o & ~mis).sum()}")
# per-modality centring / variance (observed donors), RNA per cell-type block
def prep2d(Y):
    mu = np.nanmean(Y, 0); v = np.nanvar(Y, 0); w = np.where(v > 0, 1 / np.where(v > 0, v, 1), 0.0)
    return mu, w
mu = {}; w = {}
for m in ["meth", "chip"]:
    mu[m], w[m] = prep2d(V[m])
mu["proteins"], w["proteins"] = prep2d(Po)                 # as fed
mu_c, w_c = prep2d(Pc)                                      # corrected
R = V["RNA"].reshape(6, 276, -1)                            # cell type x donor x gene
mu["RNA"] = R.mean(1); vR = R.var(1); w["RNA"] = np.where(vR > 0, 1 / np.where(vR > 0, vR, 1), 0.0)
obs = {m: ~np.isnan(V[m]).all(1) for m in ["meth", "chip"]}; obs["RNA"] = np.ones(276, bool)

def proj_2d(Y, x, mu_, w_):
    Z = np.nan_to_num(Y - mu_)                              # partial NAs (none expected) -> 0
    num = Z @ (w_ * x); den = np.sum(w_ * x * x)
    ok = ~np.isnan(Y).all(1)
    return np.where(ok, num, 0.0), np.where(ok, den, 0.0)

rows, prj = [], []
for ci, k in enumerate(L["comps"]):
    a = S.loc[k].values.astype(float); d2 = (a - a.mean()) ** 2; top = int(np.argmax(d2)); share = d2[top] / d2.sum()
    x = {m: L[f"{m}_X"][ci] for m in MODS}; pipn = {m: int((L[f"{m}_P"][ci] > 0.5).sum()) for m in MODS}
    b = L["B_cor"][ci]
    r = dict(component=int(k), top_share=share, dominant=share > 0.5)
    r["dominant_in_29"] = bool(mis[top]) if share > 0.5 else None
    r["dominant_has_prot_row"] = bool(has_o[top]) if share > 0.5 else None
    r["dominant_has_own_DLPFC_proteome"] = bool(has_c[top]) if share > 0.5 else None
    # (b) score comparisons
    oth76 = has_o & ~mis; others = ~mis
    for lab, grp in [("vs76", oth76), ("vsAll247", others)]:
        u = ss.mannwhitneyu(a[mis], a[grp], alternative="two-sided")
        r[f"MW_p_{lab}"] = u.pvalue; r[f"AUC_{lab}"] = u.statistic / (mis.sum() * grp.sum())
        dev = np.abs(a - np.median(a)); u2 = ss.mannwhitneyu(dev[mis], dev[grp], alternative="two-sided")
        r[f"MWabs_p_{lab}"] = u2.pvalue; r[f"AUCabs_{lab}"] = u2.statistic / (mis.sum() * grp.sum())
    # rank-biserial = 2*AUC - 1
    # (c) loading energy
    for m in MODS:
        r[f"SSload_{m}"] = float(np.sum(x[m] ** 2)); r[f"nPIP_{m}"] = pipn[m]
    r["SSload_RNA_x_B"] = float(np.sum(x["RNA"] ** 2) * np.sum(b ** 2))
    E = {}
    E["RNA"] = np.sum(a ** 2) * float(np.sum((b[:, None] ** 2) * (x["RNA"][None, :] ** 2) * w["RNA"]))
    for m in ["meth", "chip"]:
        E[m] = np.sum(a[obs[m]] ** 2) * float(np.sum(x[m] ** 2 * w[m]))
    E["proteins"] = np.sum(a[has_o] ** 2) * float(np.sum(x["proteins"] ** 2 * w["proteins"]))
    E_corr_prot = np.sum(a[has_c] ** 2) * float(np.nansum(x["proteins"] ** 2 * w_c))
    tot = sum(E.values())
    for m in MODS: r[f"Eshare_{m}"] = E[m] / tot
    r["Eshare_proteins_if_correct_rows"] = E_corr_prot / (tot - E["proteins"] + E_corr_prot)
    # (d) fixed-loading projections
    num = np.zeros(276); den = np.zeros(276)
    XR = (b[:, None] * x["RNA"][None, :])                      # 6 x G
    numR = np.einsum("cig,cg->i", R - mu["RNA"][:, None, :], w["RNA"] * XR); denR = float(np.sum(w["RNA"] * XR * XR))
    num += numR; den += denR
    for m in ["meth", "chip"]:
        n_, d_ = proj_2d(V[m], x[m], mu[m], w[m]); num += n_; den += d_
    nP, dP = proj_2d(Po, x["proteins"], mu["proteins"], w["proteins"])
    Pc_full = np.where(np.isnan(Pc), np.nan, Pc)
    xs = np.where(np.isnan(w_c), 0, x["proteins"])
    nPc, dPc = proj_2d(np.nan_to_num(Pc_full, nan=0.0) * 0 + Pc_full, xs, np.nan_to_num(mu_c), np.nan_to_num(w_c))
    est_fed = (num + nP) / (den + dP); est_cor = (num + nPc) / (den + dPc)
    est_noP = num / den
    pr = ss.pearsonr(est_fed, a)[0]; sr = ss.spearmanr(est_fed, a)[0]
    r["proxy_vs_SDA_pearson"] = pr; r["proxy_vs_SDA_spearman"] = sr
    sd = np.std(est_fed)
    dlt = (est_cor - est_fed) / sd
    r["delta_SD_median_29"] = float(np.median(np.abs(dlt[mis]))); r["delta_SD_max_29"] = float(np.max(np.abs(dlt[mis])))
    r["delta_SD_median_gained"] = float(np.median(np.abs(dlt[gained]))); r["delta_SD_max_gained"] = float(np.max(np.abs(dlt[gained])))
    r["delta_SD_max_unchanged76"] = float(np.max(np.abs(dlt[has_o & ~mis])))
    if share > 0.5:
        r["dominant_delta_SD"] = float(dlt[top]); r["dominant_est_SD_fed"] = float((est_fed[top] - est_fed.mean()) / sd)
        r["dominant_est_SD_cor"] = float((est_cor[top] - est_cor.mean()) / np.std(est_cor))
        r["dominant_protein_frac_of_den_fed"] = float(dP[top] / (den[top] + dP[top]))
        r["dominant_protein_frac_of_num_fed"] = float(nP[top] / (num[top] + nP[top])) if (num[top] + nP[top]) != 0 else np.nan
    # protein-only projection
    pP = np.where(has_o, nP / np.where(dP > 0, dP, 1), np.nan); pC = np.where(has_c, nPc / np.where(dPc > 0, dPc, 1), np.nan)
    sdp = np.nanstd(pP)
    r["protproj_sd"] = sdp
    r["protproj_median_abs_SD_29_fed"] = float(np.nanmedian(np.abs(pP[mis] - np.nanmean(pP)) / sdp))
    if own_among_mis.any():
        r["protproj_change_SD_own_col_donor"] = float(np.abs(pC[own_among_mis] - pP[own_among_mis]).max() / sdp)
    rho_c = ss.spearmanr(pP[has_o], a[has_o])[0]; r["protproj_vs_score_spearman_fed"] = rho_c
    okc = has_c; r["protproj_vs_score_spearman_cor"] = ss.spearmanr(pC[okc], a[okc])[0]
    # (e) best-matching donor row for the protein loadings
    Zf = (Po[has_o] - mu["proteins"]) * np.sqrt(w["proteins"]); idx = np.where(has_o)[0]
    rr = np.array([np.corrcoef(x["proteins"], Zf[j])[0, 1] for j in range(len(idx))])
    jb = int(np.nanargmax(np.abs(rr)))
    r["protload_best_donor_abs_r"] = float(abs(rr[jb])); r["protload_best_donor_in_29"] = bool(mis[idx[jb]])
    r["protload_best_donor_is_dominant"] = bool(idx[jb] == top)
    r["protload_median_abs_r_all_donors"] = float(np.nanmedian(np.abs(rr)))
    r["protload_best_r_among_29"] = float(np.nanmax(np.abs(rr[mis[idx]]))); r["protload_best_r_among_76"] = float(np.nanmax(np.abs(rr[~mis[idx]])))
    rows.append(r)
    print(f"C{k} done {time.time()-t0:.0f}s", flush=True)
T = pd.DataFrame(rows).set_index("component")
T.to_csv(f"{OUT}/03_protein_proxy.csv", float_format="%.4g")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
print(T.T.to_string(float_format=lambda v: f"{v:.3g}"))
