"""Figure S7 (Component 21 source data) and the source-data panels of Figures S8-S11 (C351, C2, C87, C177).
Definitions as RA/source_panels/scripts/04_analysis.py + 06_panels.py (aggregate: top 50 PIP > 0.5 features by |loading|,
column-standardised, weighted by gene loading x corrected cell-type loading; C2: the five chr22 peaks weighted by loading),
now with the identity-flagged profiles of the affected modality excluded (snRNA-seq RNA-1/RNA-2 -> n 274; H3K9ac CHIP-1/2,
CHIPID-1/2 -> n 189) and without titles or in-panel statistics. None of the excluded profiles is a dominant donor of these
components (identity_checks 15_flagged_donors_summary 'dominates' column).
 S7a  OR2L3 (Exc, CPM) vs C21 score; S7b top-50 Exc aggregate vs score
 S8f  IGKC (CPM, symmetric-log axis) in each cell type vs C351 score; top-50 microglial aggregate
 S9f  H3K9ac (log2(CPM + 1)) at the five chr22 peaks and their weighted mean z vs C2 score
 S10f VTCN1 (Ast, CPM) and top-50 astrocyte aggregate vs C87 score
 S11e NDUFA13 (Mic, CPM) and top-50 microglial aggregate vs C177 score"""
import numpy as np, pandas as pd
from scipy import stats as ss
from vp_common import *

LOG = []
RNA = rna_input()
exR = excl_mask("RNA"); exC = excl_mask("chip"); keepR = ~exR; keepC = ~exC

def top50_agg(k, ct):
    L = comp_loadings(k, "RNA"); X = L.X.values; P = L.P.values
    idx = np.where(P > 0.5)[0]; idx = idx[np.argsort(-np.abs(X[idx]))][:50]
    M = np.asarray(RNA[CTS_BLOCK.index(ct)][:, idx], float)
    w = X[idx] * bcor(k)[ct]
    ag, n = zagg(M[keepR], w)            # standardised over the analysed donors
    out = np.full(276, np.nan); out[keepR] = ag
    return out, n

def gene(k, g, ct):
    j = list(comp_loadings(k, "RNA").index).index(g)
    y = np.asarray(RNA[CTS_BLOCK.index(ct), :, j], float).copy(); y[exR] = np.nan
    return y

def log_view(tag, y, a):
    ok = ~np.isnan(y); r, p, n = spearman(y, a); ext = most_extreme(a[ok]); ai, yi = a[ok], y[ok]
    m = np.ones(ok.sum(), bool); m[ext] = False
    r1 = ss.spearmanr(yi[m], ai[m])[0]; pr = ss.pearsonr(yi, ai)[0]; pr1 = ss.pearsonr(yi[m], ai[m])[0]
    LOG.append(f"{tag}: n {n}; Spearman {r:+.3f} (p {p:.2g}); without most extreme donor {r1:+.3f}; Pearson {pr:+.3f} -> {pr1:+.3f}")

def two_panel(k, feat_series, feat_label, agg, agg_label, folder, name, ylog=None, legend=None):
    a = score(k)
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.32)
    broken_x_scatter(fig, gs[0, 0], a, feat_series, f"Component {k} loadings/score", feat_label, ylog=ylog, legend=legend)
    broken_x_scatter(fig, gs[0, 1], a, [(agg, CTCOL[CT[k]], 26, None)], f"Component {k} loadings/score", agg_label)
    save(fig, folder, name, LOG)

CT = {21: "Exc", 351: "Mic", 87: "Ast", 177: "Mic"}

# ---------------------------------------------------------------- S7 (C21): two separate panels a and b
a21 = score(21)
y = gene(21, "OR2L3", "Exc"); agg, n = top50_agg(21, "Exc")
for letter, yy, lab, nm in (("a", y, "OR2L3 in Exc (CPM)", "S7a_C21_OR2L3_Exc_vs_score"),
                            ("b", agg, "top-50 gene aggregate in Exc\nweighted mean z", "S7b_C21_aggregate_Exc_vs_score")):
    fig = plt.figure(figsize=(8, 6)); gs = fig.add_gridspec(1, 1)
    broken_x_scatter(fig, gs[0, 0], a21, [(yy, CTCOL["Exc"], 30, None)], "Component 21 loadings/score", lab)
    save(fig, "FigS7", nm, LOG); log_view(nm, yy, a21)
LOG.append(f"S7 C21 score donors beyond 30 x IQR: {int(outliers_30iqr(a21).sum())}; aggregate genes {n}")

# ---------------------------------------------------------------- S8f (C351)
a351 = score(351)
series = [(gene(351, "IGKC", ct), CTCOL[ct], 18, ct) for ct in CT_ORDER]
agg, n = top50_agg(351, "Mic")
two_panel(351, series, "IGKC (CPM)", agg, "top-50 gene aggregate in Mic\nweighted mean z", "FigS8", "S8f_C351_IGKC_by_celltype_and_Mic_aggregate",
          ylog="symlog", legend="upper right")
for ct in CT_ORDER:
    yv = gene(351, "IGKC", ct); log_view(f"S8f IGKC {ct}", yv, a351)
    ok = ~np.isnan(yv); dom = int(np.argmax(np.abs(robust_z(a351))))
    LOG.append(f"S8f IGKC {ct}: dominant donor rank {int(np.sum(yv[ok] >= yv[dom]))} of {int(ok.sum())}")
log_view("S8f Mic aggregate", agg, a351)

# ---------------------------------------------------------------- S9f (C2): five chr22 peaks + aggregate
a2 = score(2)
LC = comp_loadings(2, "chip"); peaks = ["peak16865", "peak16866", "peak16867", "peak16868", "peak16869"]
CI = np.asarray(mod_input("chip"), float)
H = pd.read_csv(f"{REFS}/H3K9acDomains.csv", sep="\t", index_col=0)
jj = [list(LC.index).index(p) for p in peaks]
Mp = CI[:, jj].copy(); Mp[exC] = np.nan
okp = ~np.isnan(Mp).all(1)
ag, _ = zagg(Mp[okp], LC.X.values[jj]); aggC = np.full(276, np.nan); aggC[okp] = ag
CHIPCOL = "#ce1256"   # v22 ChIP colour (ros3b colors[3])
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.38)
for i, p in enumerate(peaks):
    broken_x_scatter(fig, gs[i // 3, i % 3], a2, [(Mp[:, i], CHIPCOL, 26, None)], "Component 2 loadings/score",
                     f"{p}\nH3K9ac, log2(CPM + 1)")
    log_view(f"S9f {p} (chr{H.loc[p, 'Chr']}:{H.loc[p, 'Start'] / 1e6:.3f}-{H.loc[p, 'End'] / 1e6:.3f} Mb, {H.loc[p, 'Classification']})", Mp[:, i], a2)
broken_x_scatter(fig, gs[1, 2], a2, [(aggC, "#7a0177", 26, None)], "Component 2 loadings/score", "five-peak aggregate\nweighted mean z")
save(fig, "FigS9", "S9f_C2_chr22_peaks_and_aggregate_vs_score", LOG)
log_view("S9f five-peak aggregate", aggC, a2)
dom = int(np.argmax(np.abs(robust_z(a2))))
LOG.append(f"S9f n H3K9ac {int(okp.sum())}; dominant donor lowest at every peak: "
           f"{all(np.nanargmin(Mp[:, i]) == dom for i in range(5))}; dominant donor excluded: {bool(exC[dom])}")

# ---------------------------------------------------------------- S10f (C87), S11e (C177)
for k, g, ct, folder, nm in ((87, "VTCN1", "Ast", "FigS10", "S10f_C87_VTCN1_Ast_and_aggregate"),
                             (177, "NDUFA13", "Mic", "FigS11", "S11e_C177_NDUFA13_Mic_and_aggregate")):
    a = score(k); y = gene(k, g, ct); agg, n = top50_agg(k, ct)
    two_panel(k, [(y, CTCOL[ct], 26, None)], f"{g} in {ct} (CPM)", agg, f"top-50 gene aggregate in {ct}\nweighted mean z", folder, nm)
    log_view(f"{nm.split('_')[0]} {g} {ct}", y, a); log_view(f"{nm.split('_')[0]} {ct} aggregate", agg, a)
    LOG.append(f"{nm.split('_')[0]} dominant donor excluded by the snRNA flag: {bool(exR[int(np.argmax(np.abs(robust_z(a))))])}")
open(f"{OUT}/p06_c21_single.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
