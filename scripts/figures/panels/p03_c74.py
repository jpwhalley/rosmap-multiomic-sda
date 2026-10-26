"""Figure S4 (Component 74 source data). Same data and definitions as RA/revision_analyses/c74_c278/scripts/20_c74_views.py
(SDA input values against the deposited C74 sample score), restyled, with the identity-flagged profiles of each modality
excluded: snRNA-seq RNA-1/RNA-2 (n 274); H3K9ac CHIP-1/CHIP-2 and the two same-sex swaps CHIPID-1/2 (n 189);
methylation none (n 216).
 a  XIST in each cell type (log2(CPM + 1) of the SDA input), coloured by recorded sex
 b  cg26921482 methylation (beta); H3K9ac weighted mean z at the 537 chrX and the 120 autosomal C74 peaks
 c  Spearman rho with the score (95% bootstrap CI): all donors and within each sex
 d  C74 vs SEA-AD component 106 RNA loadings by chromosome class (second implementation's annotation by Ensembl ID: 62 sex-chromosome
    genes, 44 X + 18 Y; 76 autosomal), as RA/revision_analyses/verify_c74_c278/scripts/v30_c74_seaad.py"""
import numpy as np, pandas as pd
from scipy import stats as ss
from matplotlib.lines import Line2D
from vp_common import *

LOG = []
K = 74; a = score(K); fem = female(); sexlab = np.where(fem, "female", "male")
exR = excl_mask("RNA"); exC = excl_mask("chip"); nomask = np.zeros(276, bool)
assert exR.sum() == 2 and exC.sum() == 4
RNA = rna_input(); LR = comp_loadings(K, "RNA"); genes = list(LR.index)
XL = "Component 74 loadings/score"
rows = []

def stat(view, y, excl, seed):
    ok = ~np.isnan(y) & ~excl
    r = spearman_ci(y[ok], a[ok], seed=seed)
    rm = spearman_ci(y[ok & ~fem], a[ok & ~fem], seed=seed + 1); rf = spearman_ci(y[ok & fem], a[ok & fem], seed=seed + 2)
    d = dict(view=view, n=int(ok.sum()), n_male=int((ok & ~fem).sum()), n_female=int((ok & fem).sum()),
             rho=r["rho"], lo=r["lo"], hi=r["hi"], p=r["p"], rho_m=rm["rho"], lo_m=rm["lo"], hi_m=rm["hi"],
             rho_f=rf["rho"], lo_f=rf["lo"], hi_f=rf["hi"])
    rows.append(d); return d

def sex_scatter(ax, y, excl, ylab, s=26):
    ok = ~np.isnan(y) & ~excl
    for sx in ("male", "female"):
        m = ok & (sexlab == sx)
        ax.scatter(a[m], y[m], s=s, color=SEXCOL[sx], alpha=0.8, linewidths=0)
    ax.set_ylabel(ylab); sns.despine(ax=ax)

# ---------------------------------------------------------------- a  XIST per cell type
j = genes.index("XIST")
fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True)
for ax, ct in zip(axs.ravel(), CT_ORDER):
    y = np.log2(np.asarray(RNA[CTS_BLOCK.index(ct), :, j], float) + 1)
    d = stat(f"XIST, {ct}", y, exR, 11)
    sex_scatter(ax, y, exR, f"XIST in {ct}\nlog2(CPM + 1)")
    LOG.append(f"S4a XIST {ct}: n {d['n']} ({d['n_male']} male, {d['n_female']} female); rho {d['rho']:+.3f}")
for ax in axs[1]: ax.set_xlabel(XL)
sexlegend(axs[0, 0], int((~exR & ~fem).sum()), int((~exR & fem).sum()), loc="center", fontsize=13)
fig.tight_layout(); save(fig, "FigS4", "S4a_XIST_by_celltype_vs_score", LOG)

# ---------------------------------------------------------------- b  CpG and H3K9ac aggregates
MI = mod_input("meth"); LM = comp_loadings(K, "meth"); cpgs = list(LM.index)
LMp = LM[LM.P > 0.5]; top_cpg = LMp.X.abs().idxmax(); assert top_cpg == "cg26921482"
ym = np.asarray(MI[:, cpgs.index(top_cpg)], float)
CI = np.asarray(mod_input("chip"), float); LC = comp_loadings(K, "chip")
H = pd.read_csv(f"{REFS}/H3K9acDomains.csv", sep="\t", index_col=0).loc[list(LC.index)]
chrc = H.Chr.astype(str).values
selX = (LC.P.values > 0.5) & (chrc == "X"); selA = (LC.P.values > 0.5) & ~np.isin(chrc, ["X", "Y"]); selY = (LC.P.values > 0.5) & (chrc == "Y")
aggX, nX = zagg(CI[:, selX], LC.X.values[selX]); aggA, nA = zagg(CI[:, selA], LC.X.values[selA]); aggY, nY = zagg(CI[:, selY], LC.X.values[selY])
MA = np.asarray(MI, float); selM = LM.P.values > 0.5
aggM, nM = zagg(MA[:, selM], LM.X.values[selM])
fig, axs = plt.subplots(1, 3, figsize=(18, 5.8))
d = stat(f"cg26921482 (top autosomal CpG)", ym, nomask, 21); sex_scatter(axs[0], ym, nomask, "cg26921482 methylation (β)")
LOG.append(f"S4b cg26921482: n {d['n']}; rho {d['rho']:+.3f}")
d = stat(f"H3K9ac, {nX} chrX peaks (aggregate)", aggX, exC, 31); sex_scatter(axs[1], aggX, exC, f"H3K9ac, {nX} chrX peaks\nweighted mean z")
LOG.append(f"S4b H3K9ac chrX aggregate ({nX} peaks): n {d['n']} ({d['n_male']} male, {d['n_female']} female); rho {d['rho']:+.3f}")
d = stat(f"H3K9ac, {nA} autosomal peaks (aggregate)", aggA, exC, 41); sex_scatter(axs[2], aggA, exC, f"H3K9ac, {nA} autosomal peaks\nweighted mean z")
LOG.append(f"S4b H3K9ac autosomal aggregate ({nA} peaks): n {d['n']}; rho {d['rho']:+.3f}")
for ax in axs: ax.set_xlabel(XL)
sexlegend(axs[0], loc="lower left", fontsize=13)
fig.tight_layout(); save(fig, "FigS4", "S4b_cpg_and_H3K9ac_vs_score", LOG)
d = stat(f"H3K9ac, {nY} chrY peaks (aggregate)", aggY, exC, 51)
LOG.append(f"S4c H3K9ac chrY aggregate ({nY} peaks): n {d['n']}; rho {d['rho']:+.3f}")
d = stat(f"methylation, {nM:,} CpG sites (aggregate)", aggM, nomask, 61)
LOG.append(f"S4c methylation aggregate ({nM} CpGs): n {d['n']}; rho {d['rho']:+.3f}; within male {d['rho_m']:+.2f}, female {d['rho_f']:+.2f}")

# ---------------------------------------------------------------- c  within-sex forest
gtf = pd.read_csv(f"{RA}/verify_c74_c278/cache/gene_chrom_gtf100.csv").drop_duplicates("gene_name").set_index("gene_name").chrom.astype(str)
cc = pd.read_csv(f"{REFS}/Homo_sapiens.GRCh38.100.txt",
                 usecols=["chrom", "gene_name"]).drop_duplicates("gene_name").set_index("gene_name").chrom.astype(str)
LR["chr"] = [cc.get(g, "NA") for g in LR.index]
auto = LR[(LR.P > 0.5) & ~LR.chr.isin(["X", "Y", "NA"])]
top_auto = auto.X.abs().idxmax(); ja = genes.index(top_auto)
Bc = bcor(K)[CTS_BLOCK]
selRa = (LR.P.values > 0.5) & ~np.isin(LR.chr.values, ["X", "Y", "NA"])
for i, ct in enumerate(CT_ORDER):
    yk = np.log2(np.asarray(RNA[CTS_BLOCK.index(ct), :, ja], float) + 1); stat(f"{top_auto}, {ct}", yk, exR, 70 + i)
for i, ct in enumerate(CT_ORDER):
    Rc = np.log2(np.asarray(RNA[CTS_BLOCK.index(ct)], float) + 1)
    ag, _ = zagg(Rc[:, selRa], LR.X.values[selRa] * Bc[ct]); stat(f"autosomal RNA genes, {ct} (aggregate)", ag, exR, 80 + i)
LOG.append(f"S4c top autosomal RNA gene: {top_auto}; autosomal PIP genes in the RNA aggregate: {int(selRa.sum())}")
T = pd.DataFrame(rows).set_index("view")
order = ([f"XIST, {c}" for c in CT_ORDER] + [f"{top_auto}, {c}" for c in CT_ORDER] + ["cg26921482 (top autosomal CpG)",
         f"methylation, {nM:,} CpG sites (aggregate)", f"H3K9ac, {nX} chrX peaks (aggregate)", f"H3K9ac, {nA} autosomal peaks (aggregate)",
         f"H3K9ac, {nY} chrY peaks (aggregate)"] + [f"autosomal RNA genes, {c} (aggregate)" for c in CT_ORDER])
S = T.loc[order]
fig, ax = plt.subplots(figsize=(10, 0.45 * len(S) + 1.8))
yy = np.arange(len(S))[::-1]
for off, sx, col, pre in ((-0.15, "male", MALE, "m"), (0.15, "female", FEMALE, "f")):
    ax.errorbar(S[f"rho_{pre}"], yy + off, xerr=[S[f"rho_{pre}"] - S[f"lo_{pre}"], S[f"hi_{pre}"] - S[f"rho_{pre}"]],
                fmt="o", color=col, ms=6, lw=1.4, capsize=0, label=f"within {'males' if sx == 'male' else 'females'}")
ax.scatter(S.rho, yy, marker="|", s=160, color="k", label="all donors", zorder=3, linewidths=2)
ax.axvline(0, color="grey", lw=0.8, ls=":")
ax.set_yticks(yy); ax.set_yticklabels([f"{v}  ({int(S.loc[v, 'n_male'])} male, {int(S.loc[v, 'n_female'])} female)" for v in S.index], fontsize=13)
ax.set_xlabel("Spearman ρ with the Component 74 score")
ax.set_xlim(-1, 1)
ax.legend(fontsize=13, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
sns.despine(ax=ax); fig.tight_layout(); save(fig, "FigS4", "S4c_within_sex_correlations", LOG)
for v in S.index:
    r = S.loc[v]
    LOG.append(f"S4c {v}: n {int(r.n)}; all {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]; male {r.rho_m:+.2f}; female {r.rho_f:+.2f}")
LOG.append(f"S4c within-sex range over the aggregates (methylation, H3K9ac chrX/autosomal): "
           f"{min(S.loc[[order[13], order[14], order[15]], ['rho_m', 'rho_f']].min()):+.2f} to {max(S.loc[[order[13], order[14], order[15]], ['rho_m', 'rho_f']].max()):+.2f}")

# ---------------------------------------------------------------- d  SEA-AD by chromosome class (second implementation's annotation)
SEA = SEAAD
gi = pd.read_csv(f"{SEA}/gene_info.csv", index_col=0)
sg = pd.read_csv(f"{SEA}/sda/genes_final.csv", index_col=0); sg.index = sg.index.astype(int); sg = sg.drop(index=[9, 128])
name = gi.feature_name.reindex(sg.columns)
n2e = {}
for e, nm in name.items(): n2e.setdefault(nm, []).append(e)
pipg = set(LR.index[LR.P > 0.5])
common = sorted(g for g in pipg if g in n2e)
cols = [n2e[g][0] for g in common]
x = LR.loc[common, "X"].values; y = sg.loc[106, cols].values
GC = pd.read_csv(f"{RA}/verify_c74_c278/cache/gene_chrom_gtf100.csv").set_index("gene_id")
chr_e = np.array([GC.chrom.get(e.split(".")[0], "NA") for e in cols]).astype(str)
isX = chr_e == "X"; isY = chr_e == "Y"; isA = np.isin(chr_e, [str(i) for i in range(1, 23)])
r_all = np.corrcoef(x, y)[0, 1]; orient = np.sign(r_all)
sexc = isX | isY
conc = lambda s: int((np.sign(x[s]) * np.sign(y[s]) == orient).sum())
# base rate among non-selected autosomal genes (second implementation, v31)
allc = [g for g in LR.index if g in n2e]
xa = LR.loc[allc, "X"].values; pa = LR.loc[allc, "P"].values; ya = sg.loc[106, [n2e[g][0] for g in allc]].values
cha = np.array([GC.chrom.get(n2e[g][0].split(".")[0], "NA") for g in allc]).astype(str); au = np.isin(cha, [str(i) for i in range(1, 23)])
nz = au & (pa < 0.5) & (xa != 0) & (ya != 0)
base = float((np.sign(xa[nz]) * np.sign(ya[nz]) == orient).mean())
kA = conc(isA); bt = ss.binomtest(kA, int(isA.sum()), base, alternative="greater")   # one-sided, as the second implementation (v31)
LOG.append(f"S4d genes measured in SEA-AD {len(common)}; overall Pearson {r_all:+.3f}; sex-chromosome {int(sexc.sum())} (X {int(isX.sum())}, "
           f"Y {int(isY.sum())}): Pearson {ss.pearsonr(x[sexc], y[sexc])[0]:+.3f}, Spearman {ss.spearmanr(x[sexc], y[sexc])[0]:+.3f}, "
           f"concordant {conc(sexc)}/{int(sexc.sum())}; autosomal {int(isA.sum())}: Pearson {ss.pearsonr(x[isA], y[isA])[0]:+.3f}, "
           f"Spearman {ss.spearmanr(x[isA], y[isA])[0]:+.3f}, concordant {kA}/{int(isA.sum())}; base rate {base:.3f} (n {int(nz.sum())}); "
           f"one-sided binomial vs base rate p {bt.pvalue:.2g} (two-sided {ss.binomtest(kA, int(isA.sum()), base).pvalue:.2g}; vs 0.5 two-sided {ss.binomtest(kA, int(isA.sum()), 0.5).pvalue:.2g}); unannotated {int((~isX & ~isY & ~isA).sum())}")
XCOL, YCOL, ACOL = "#7B3294", "#1B7837", "#666666"
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
ax = axs[0]
ax.scatter(x[isX], y[isX], s=40, color=XCOL, alpha=0.85, lw=0, label=f"chrX (n = {int(isX.sum())})")
ax.scatter(x[isY], y[isY], s=40, color=YCOL, alpha=0.85, lw=0, label=f"chrY (n = {int(isY.sum())})")
ax.set_ylabel("SEA-AD component 106 loadings")
ax.legend(frameon=False, fontsize=14, loc="lower left")
ax = axs[1]
ax.scatter(x[isA], y[isA], s=40, color=ACOL, alpha=0.85, lw=0, label=f"autosomal (n = {int(isA.sum())})")
ax.legend(frameon=False, fontsize=14, loc="upper right")
for ax in axs:
    ax.axhline(0, color="grey", lw=0.7, ls=":"); ax.axvline(0, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("ROSMAP Component 74 loadings"); sns.despine(ax=ax)
fig.tight_layout(); save(fig, "FigS4", "S4d_SEAAD_by_chromosome_class", LOG)

T.to_csv(f"{OUT}/p03_c74_view_stats.csv", float_format="%.4g")
open(f"{OUT}/p03_c74.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
