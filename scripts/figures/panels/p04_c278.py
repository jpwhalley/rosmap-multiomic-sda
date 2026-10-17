"""Figure 3f and Figure S4 (Component 278 source data). Definitions as RA/revision_analyses/c74_c278/scripts/30_c278_views.py
and verify_c74_c278/scripts/v80_c278_markers.py; snRNA-seq profiles RNA-1/RNA-2 excluded (n 274).
 Fig 3f  FKBP5 in OPCs and oligodendrocytes, CPM of the SDA input (linear axis), most extreme donor ringed
 S4a     FKBP5, SGK1, ZBTB16 in OPCs (top) and oligodendrocytes (bottom), log2(CPM + 1), coloured by recorded sex
 S4b     weighted mean z of all 1,114 PIP > 0.5 genes (left) and without the three leading genes (right), OPC / Oli
 S4c     female - male difference in mean log2(CPM + 1) of the three genes (95% bootstrap CI)
 S4d     d1 SOX6 and BCAS1 vs score; d2 lineage-marker Spearman rho (second implementation, v80); d3 OPC share of OPC + Oli nuclei;
         d4 marker modules (second-implementation v80 definition, the values in the text)"""
import numpy as np, pandas as pd
from scipy import stats as ss
from matplotlib.lines import Line2D
from vp_common import *

LOG = []
K = 278; a = score(K); fem = female(); sexlab = np.where(fem, "female", "male")
exR = excl_mask("RNA"); keep = ~exR
RNA = rna_input(); LR = comp_loadings(K, "RNA"); genes = list(LR.index)
Bc = bcor(K)[CTS_BLOCK]
LEAD = ["FKBP5", "SGK1", "ZBTB16"]
XL = "Component 278 loadings/score"
def cpm(g, ct): return np.asarray(RNA[CTS_BLOCK.index(ct), :, genes.index(g)], float)

# ---------------------------------------------------------------- Fig 3f
ak = a[keep]; ext = most_extreme(ak)
fig, ax = plt.subplots(figsize=(8, 6))
for ct in ("OPC", "Oli"):
    y = cpm("FKBP5", ct)[keep]
    ax.scatter(ak, y, s=30, color=CTCOL[ct], alpha=0.8, linewidths=0, label=ct, zorder=2)
    ax.scatter([ak[ext]], [y[ext]], s=110, facecolors="none", edgecolors=EXTRING, linewidths=1.6, zorder=3)
    r, p, n = spearman(y, ak); m = np.ones(len(ak), bool); m[ext] = False
    LOG.append(f"Fig3f FKBP5 {ct}: n {n}; rho {r:+.3f} (p {p:.2g}); without most extreme donor {ss.spearmanr(y[m], ak[m])[0]:+.3f}")
ax.set_xlabel(XL); ax.set_ylabel("FKBP5 (CPM)")
ax.legend(frameon=False, loc="upper left", markerscale=1.3); sns.despine(ax=ax)
fig.tight_layout(); save(fig, "Fig3", "Fig3f_C278_FKBP5_OPC_Oli_vs_score", LOG)

def sex_scatter(ax, y, ylab, s=24):
    for sx in ("male", "female"):
        m = keep & (sexlab == sx)
        ax.scatter(a[m], y[m], s=s, color=SEXCOL[sx], alpha=0.8, linewidths=0)
    ax.set_ylabel(ylab); sns.despine(ax=ax)

# ---------------------------------------------------------------- S4a
fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True)
for r_, ct in enumerate(("OPC", "Oli")):
    for c_, g in enumerate(LEAD):
        y = np.log2(cpm(g, ct) + 1)
        sex_scatter(axs[r_, c_], y, f"{g} in {ct}\nlog2(CPM + 1)")
        rr, p, n = spearman(y[keep], a[keep])
        LOG.append(f"S4a {g} {ct}: n {n}; rho {rr:+.3f}")
for ax in axs[1]: ax.set_xlabel(XL)
sexlegend(axs[0, 0], int((keep & ~fem).sum()), int((keep & fem).sum()), loc="lower right", fontsize=13)
fig.tight_layout(); save(fig, "FigS4", "S4a_leading_genes_OPC_Oli_vs_score", LOG)

# ---------------------------------------------------------------- S4b
pip = LR.P.values > 0.5; nolead = pip & ~np.isin(genes, LEAD)
fig, axs = plt.subplots(2, 2, figsize=(12, 9.5), sharex=True)
for r_, ct in enumerate(("OPC", "Oli")):
    Rc = np.log2(np.asarray(RNA[CTS_BLOCK.index(ct)], float) + 1); w = LR.X.values * Bc[ct]
    for c_, (sel, lab) in enumerate(((pip, f"{int(pip.sum()):,} genes"), (nolead, f"{int(nolead.sum()):,} genes,\nwithout FKBP5, SGK1, ZBTB16"))):
        ag, n_ = zagg(Rc[:, sel], w[sel])
        sex_scatter(axs[r_, c_], ag, f"{ct}, {lab}\nweighted mean z")
        rr, p, n = spearman(ag[keep], a[keep])
        LOG.append(f"S4b {ct} aggregate of {int(sel.sum())} PIP genes ({n_} with variance): n {n}; rho {rr:+.4f}")
for ax in axs[1]: ax.set_xlabel(XL)
sexlegend(axs[0, 0], loc="lower right", fontsize=13)
fig.tight_layout(); save(fig, "FigS4", "S4b_aggregates_with_and_without_leading_genes", LOG)

# ---------------------------------------------------------------- S4c
rows = []
for ct in ("OPC", "Oli"):
    for g in LEAD:
        y = np.log2(cpm(g, ct) + 1)
        yf, ym = y[keep & fem], y[keep & ~fem]
        rng = np.random.default_rng(24); bs = [rng.choice(yf, len(yf)).mean() - rng.choice(ym, len(ym)).mean() for _ in range(4000)]
        rows.append(dict(lab=f"{g}, {ct}", d=yf.mean() - ym.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         p=ss.mannwhitneyu(yf, ym).pvalue, nf=len(yf), nm=len(ym)))
D = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
yy = np.arange(len(D))[::-1]
ax.errorbar(D.d, yy, xerr=[D.d - D.lo, D.hi - D.d], fmt="o", color="k", ms=8, lw=1.8, capsize=0)
ax.axvline(0, color="grey", lw=0.8, ls=":")
ax.set_yticks(yy); ax.set_yticklabels(D.lab)
ax.set_xlabel("female − male, mean log2(CPM + 1)"); sns.despine(ax=ax)
fig.tight_layout(); save(fig, "FigS4", "S4c_female_minus_male_leading_genes", LOG)
for _, r in D.iterrows():
    LOG.append(f"S4c {r.lab}: {r.d:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]; MW p {r.p:.2g}; n {r.nf} female, {r.nm} male")

# ---------------------------------------------------------------- S4d
# d1 SOX6 and BCAS1
fig, axs = plt.subplots(2, 2, figsize=(12, 9.5), sharex=True)
for r_, g in enumerate(("SOX6", "BCAS1")):
    for c_, ct in enumerate(("OPC", "Oli")):
        y = np.log2(cpm(g, ct) + 1); sex_scatter(axs[r_, c_], y, f"{g} in {ct}\nlog2(CPM + 1)")
        rr, p, n = spearman(y[keep], a[keep]); LOG.append(f"S4d1 {g} {ct}: n {n}; rho {rr:+.3f}")
for ax in axs[1]: ax.set_xlabel(XL)
sexlegend(axs[0, 0], loc="lower left", fontsize=13)
fig.tight_layout(); save(fig, "FigS4", "S4d1_SOX6_BCAS1_vs_score", LOG)
# d2 lineage markers (second-implementation v80 table)
T = pd.read_csv(f"{RA}/verify_c74_c278/out/v80_c278_markers.csv").dropna(subset=["rho_OPC"])
GRP = {"OPC": "OPC", "committed": "committed", "mature": "mature"}
fig, ax = plt.subplots(figsize=(7.5, 12))
yy = np.arange(len(T))[::-1]
for off, ct in ((-0.18, "OPC"), (0.18, "Oli")):
    ax.errorbar(T[f"rho_{ct}"], yy + off, xerr=[T[f"rho_{ct}"] - T[f"lo_{ct}"], T[f"hi_{ct}"] - T[f"rho_{ct}"]], fmt="o", ms=6, lw=1.4,
                color=CTCOL[ct], capsize=0, label=f"in {ct}")
ax.axvline(0, color="grey", lw=0.8, ls=":")
ax.set_yticks(yy); ax.set_yticklabels([f"{g} ({GRP[gr]})" for g, gr in zip(T.gene, T.group)])
nb = T.group.value_counts()[["OPC", "committed"]].cumsum().values
for b in nb: ax.axhline(len(T) - b - 0.5, color="k", lw=0.6)
ax.set_xlabel("Spearman ρ with the score"); ax.legend(frameon=False, loc="lower right", fontsize=13); sns.despine(ax=ax)
fig.tight_layout(); save(fig, "FigS4", "S4d2_lineage_markers_rho", LOG)
LOG.append(f"S4d2 markers drawn: {len(T)} (CLDN11 not in the SDA gene list); SOX6 OPC {T.set_index('gene').loc['SOX6','rho_OPC']:+.2f}, "
           f"Oli {T.set_index('gene').loc['SOX6','rho_Oli']:+.2f}; BCAS1 {T.set_index('gene').loc['BCAS1','rho_OPC']:+.2f} / {T.set_index('gene').loc['BCAS1','rho_Oli']:+.2f}")
# d3 OPC share of OPC + Oli nuclei
qc = pd.read_csv(f"{R2}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str); qc = qc.loc[donors()]
share = (qc.nuclei_OPC / (qc.nuclei_OPC + qc.nuclei_Oli)).values
fig, ax = plt.subplots(figsize=(7, 6))
sex_scatter(ax, share, "OPC share of OPC + Oli nuclei"); ax.set_xlabel(XL)
fig.tight_layout(); save(fig, "FigS4", "S4d3_OPC_share_vs_score", LOG)
rr, p, n = spearman(share[keep], a[keep]); LOG.append(f"S4d3 OPC share: n {n}; rho {rr:+.3f}; p {p:.2g}")
# d4 modules (second implementation, v80)
Mo = pd.read_csv(f"{RA}/verify_c74_c278/out/v80_c278_modules.csv")
Mo = Mo[Mo.ct != "composition"]
fig, ax = plt.subplots(figsize=(7.5, 5))
yy = np.arange(len(Mo))[::-1]
for (_, r), y_ in zip(Mo.iterrows(), yy):
    ax.errorbar(r.rho, y_, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt="o", color=CTCOL[r.ct], ms=8, lw=1.6, capsize=0)
ax.axvline(0, color="grey", lw=0.8, ls=":")
ax.set_yticks(yy); ax.set_yticklabels([f"{m} module, in {c}" for m, c in zip(Mo.module, Mo.ct)])
ax.set_xlabel("Spearman ρ with the score"); sns.despine(ax=ax)
fig.tight_layout(); save(fig, "FigS4", "S4d4_marker_modules_rho", LOG)
for _, r in Mo.iterrows(): LOG.append(f"S4d4 {r.module} module in {r.ct}: rho {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]")
open(f"{OUT}/p04_c278.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
