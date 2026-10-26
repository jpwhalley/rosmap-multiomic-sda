"""Figure S4 (Component 74 source data), panels redrawn at their final printed size for the composite.

Same data, exclusions, statistics, colours and label text as panels/p03_c74.py (the computation below is
copied from it verbatim, same bootstrap seeds); only the drawing changes: each figure is exactly its cell size in the
composite (placed at scale 1 in a 142.9 mm wide figure), with explicit Arial sizes (ticks 6 pt, axis labels 6.5 pt,
legends 6 pt) and thinner lines/smaller markers. Presentation-only differences from p03 (text unchanged):
line breaks in two long axis labels (b "cg26921482 / methylation (β)", d x "ROSMAP Component 74 / loadings");
a uses shared x like p03; c's legend is right-aligned above the plot; d's two plots are stacked (not side by side) with
one y label spanning both (p03 had one y label for both), MaxNLocator ticks, the chrX/chrY legend at upper right on an
opaque white box (p03 lower left, which would sit on the chrY points at this aspect) and the autosomal legend just above
its plot's top-right corner (p03 inside upper right, which would sit beside the top-left point at this size).

Writes only under <G_ROS_FIGURE_OUT>/S4/ (panels/, s4_final_size.log, s4_view_stats.csv). Never prints donor IDs or per-donor
values; the log holds aggregates only."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
from vp_common import *                      # loaders, palette, style (seaborn white, Arial, fonttype 42)
import numpy as np, pandas as pd
from scipy import stats as ss
from matplotlib.lines import Line2D

HERE = f"{FIGOUT}/S4"
PAN = f"{HERE}/panels"; os.makedirs(PAN, exist_ok=True)
MM = 1 / 25.4
TICK, LAB, LEG = 6.0, 6.5, 6.0
mpl.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "font.size": LAB, "axes.labelsize": LAB, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
                     "legend.fontsize": LEG, "axes.linewidth": 0.6, "lines.linewidth": 0.8,
                     "xtick.major.pad": 1.8, "ytick.major.pad": 1.5, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.labelpad": 2.0,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def fig_mm(w, h):
    return plt.figure(figsize=(w * MM, h * MM))


def ax_mm(fig, x, y, w, h, **kw):
    """Axes at x, y (top-left, mm, y down), size w x h mm."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def savep(fig, name):
    fig.savefig(f"{PAN}/{name}.pdf")            # exact figure size: no bbox_inches, the cell size is the page size
    plt.close(fig); LOG.append(f"wrote panels/{name}.pdf")


def sexhandles(n_m=None, n_f=None, ms=3.2):
    lm = "Male" if n_m is None else f"Male (n = {n_m})"
    lf = "Female" if n_f is None else f"Female (n = {n_f})"
    return [Line2D([], [], color=MALE, marker="o", ls="None", markersize=ms, markeredgewidth=0, label=lm),
            Line2D([], [], color=FEMALE, marker="o", ls="None", markersize=ms, markeredgewidth=0, label=lf)]


LEGKW = dict(frameon=False, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.3, borderpad=0.2, handlelength=1.0, scatteryoffsets=[0.5])

# ================================================================ computation (verbatim from p03_c74.py)
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

S_DOT = 4.0   # scatter marker area (pt^2) at final size (p03: 26 at about 2.8x this size)

def sex_scatter(ax, y, excl, ylab, s=S_DOT):
    ok = ~np.isnan(y) & ~excl
    for sx in ("male", "female"):
        m = ok & (sexlab == sx)
        ax.scatter(a[m], y[m], s=s, color=SEXCOL[sx], alpha=0.8, linewidths=0)
    ax.set_ylabel(ylab); sns.despine(ax=ax)

# ---------------------------------------------------------------- a  XIST per cell type   (cell 142.9 x 49.5 mm)
WA, HA = 142.9, 49.5
j = genes.index("XIST")
fig = fig_mm(WA, HA)
AXW, AXH = 36.6, 18.3
lefts = [10.0, 57.6, 105.2]; tops = [2.6, 23.9]
axs = np.array([[ax_mm(fig, l, t, AXW, AXH) for l in lefts] for t in tops])
for ax in axs.ravel()[1:]:
    ax.sharex(axs[0, 0])
for ax, ct in zip(axs.ravel(), CT_ORDER):
    y = np.log2(np.asarray(RNA[CTS_BLOCK.index(ct), :, j], float) + 1)
    d = stat(f"XIST, {ct}", y, exR, 11)
    sex_scatter(ax, y, exR, f"XIST in {ct}\nlog2(CPM + 1)")
    LOG.append(f"S4a XIST {ct}: n {d['n']} ({d['n_male']} male, {d['n_female']} female); rho {d['rho']:+.3f}")
for ax in axs[0]: ax.tick_params(labelbottom=False)
for ax in axs[1]: ax.set_xlabel(XL)
axs[0, 0].legend(handles=sexhandles(int((~exR & ~fem).sum()), int((~exR & fem).sum())), loc="center", **LEGKW)
savep(fig, "S4a_final")

# ---------------------------------------------------------------- b  CpG and H3K9ac aggregates   (cell 142.9 x 39 mm)
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
WB, HB = 142.9, 39.0
fig = fig_mm(WB, HB)
BW, BH = 35.6, 28.6
axs = [ax_mm(fig, l, 3.0, BW, BH) for l in (10.6, 58.2, 105.8)]
d = stat(f"cg26921482 (top autosomal CpG)", ym, nomask, 21); sex_scatter(axs[0], ym, nomask, "cg26921482\nmethylation (β)")
LOG.append(f"S4b cg26921482: n {d['n']}; rho {d['rho']:+.3f}")
d = stat(f"H3K9ac, {nX} chrX peaks (aggregate)", aggX, exC, 31); sex_scatter(axs[1], aggX, exC, f"H3K9ac, {nX} chrX peaks\nweighted mean z")
LOG.append(f"S4b H3K9ac chrX aggregate ({nX} peaks): n {d['n']} ({d['n_male']} male, {d['n_female']} female); rho {d['rho']:+.3f}")
d = stat(f"H3K9ac, {nA} autosomal peaks (aggregate)", aggA, exC, 41); sex_scatter(axs[2], aggA, exC, f"H3K9ac, {nA} autosomal peaks\nweighted mean z")
LOG.append(f"S4b H3K9ac autosomal aggregate ({nA} peaks): n {d['n']}; rho {d['rho']:+.3f}")
for ax in axs: ax.set_xlabel(XL)
axs[0].legend(handles=sexhandles(), loc="lower left", **LEGKW)
savep(fig, "S4b_final")
d = stat(f"H3K9ac, {nY} chrY peaks (aggregate)", aggY, exC, 51)
LOG.append(f"S4c H3K9ac chrY aggregate ({nY} peaks): n {d['n']}; rho {d['rho']:+.3f}")
d = stat(f"methylation, {nM:,} CpG sites (aggregate)", aggM, nomask, 61)
LOG.append(f"S4c methylation aggregate ({nM} CpGs): n {d['n']}; rho {d['rho']:+.3f}; within male {d['rho_m']:+.2f}, female {d['rho_f']:+.2f}")

# ---------------------------------------------------------------- c  within-sex forest   (cell 106 x 68 mm)
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
WC, HC = 106.0, 68.0
fig = fig_mm(WC, HC)
CX0, CTOP, CW, CH = 63.6, 5.2, 40.0, 55.6
ax = ax_mm(fig, CX0, CTOP, CW, CH)
yy = np.arange(len(S))[::-1]
for off, sx, col, pre in ((-0.15, "male", MALE, "m"), (0.15, "female", FEMALE, "f")):
    ax.errorbar(S[f"rho_{pre}"], yy + off, xerr=[S[f"rho_{pre}"] - S[f"lo_{pre}"], S[f"hi_{pre}"] - S[f"rho_{pre}"]],
                fmt="o", color=col, ms=2.0, lw=0.6, capsize=0, label=f"within {'males' if sx == 'male' else 'females'}")
ax.scatter(S.rho, yy, marker="|", s=22, color="k", label="all donors", zorder=3, linewidths=0.9)
ax.axvline(0, color="grey", lw=0.5, ls=":")
ax.set_yticks(yy); ax.set_yticklabels([f"{v}  ({int(S.loc[v, 'n_male'])} male, {int(S.loc[v, 'n_female'])} female)" for v in S.index], fontsize=TICK)
ax.set_ylim(-0.6, len(S) - 0.4)
ax.set_xlabel("Spearman ρ with the Component 74 score")
ax.set_xlim(-1, 1)
ax.legend(fontsize=LEG, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, handletextpad=0.3,
          columnspacing=1.0, borderaxespad=0.15, borderpad=0.1, handlelength=1.4)
sns.despine(ax=ax); savep(fig, "S4c_final")
for v in S.index:
    r = S.loc[v]
    LOG.append(f"S4c {v}: n {int(r.n)}; all {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]; male {r.rho_m:+.2f}; female {r.rho_f:+.2f}")

# ---------------------------------------------------------------- d  SEA-AD by chromosome class   (cell 36.4 x 68 mm)
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
allc = [g for g in LR.index if g in n2e]
xa = LR.loc[allc, "X"].values; pa = LR.loc[allc, "P"].values; ya = sg.loc[106, [n2e[g][0] for g in allc]].values
cha = np.array([GC.chrom.get(n2e[g][0].split(".")[0], "NA") for g in allc]).astype(str); au = np.isin(cha, [str(i) for i in range(1, 23)])
nz = au & (pa < 0.5) & (xa != 0) & (ya != 0)
base = float((np.sign(xa[nz]) * np.sign(ya[nz]) == orient).mean())
kA = conc(isA); bt = ss.binomtest(kA, int(isA.sum()), base, alternative="greater")
LOG.append(f"S4d genes measured in SEA-AD {len(common)}; overall Pearson {r_all:+.3f}; sex-chromosome {int(sexc.sum())} (X {int(isX.sum())}, "
           f"Y {int(isY.sum())}): Pearson {ss.pearsonr(x[sexc], y[sexc])[0]:+.3f}, Spearman {ss.spearmanr(x[sexc], y[sexc])[0]:+.3f}, "
           f"concordant {conc(sexc)}/{int(sexc.sum())}; autosomal {int(isA.sum())}: Pearson {ss.pearsonr(x[isA], y[isA])[0]:+.3f}, "
           f"Spearman {ss.spearmanr(x[isA], y[isA])[0]:+.3f}, concordant {kA}/{int(isA.sum())}; base rate {base:.3f} (n {int(nz.sum())}); "
           f"one-sided binomial vs base rate p {bt.pvalue:.2g}; unannotated {int((~isX & ~isY & ~isA).sum())}")
XCOL, YCOL, ACOL = "#7B3294", "#1B7837", "#666666"
WD, HD = 36.4, 68.0
fig = fig_mm(WD, HD)
DX0, DW = 10.2, 25.4
axs = [ax_mm(fig, DX0, 4.0, DW, 19.0), ax_mm(fig, DX0, 36.5, DW, 22.0)]
SD = 6.0
DLEG = dict(LEGKW, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)   # opaque white box over the 0 lines
ax = axs[0]
ax.scatter(x[isX], y[isX], s=SD, color=XCOL, alpha=0.85, lw=0, label=f"chrX (n = {int(isX.sum())})")
ax.scatter(x[isY], y[isY], s=SD, color=YCOL, alpha=0.85, lw=0, label=f"chrY (n = {int(isY.sum())})")
ax.legend(loc="upper right", markerscale=1.0, **DLEG)   # p03: lower left; moved off the chrY points at this aspect
ax = axs[1]
ax.scatter(x[isA], y[isA], s=SD, color=ACOL, alpha=0.85, lw=0, label=f"autosomal (n = {int(isA.sum())})")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), markerscale=1.0, **dict(LEGKW, borderaxespad=0.1))
# p03: inside, upper right. Here just above the plot's top-right corner: at this size an inside legend sits on or beside points
for ax in axs:
    ax.axhline(0, color="grey", lw=0.5, ls=":"); ax.axvline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("ROSMAP Component 74\nloadings"); sns.despine(ax=ax)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4)); ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
fig.text(0.6 * MM / (WD * MM), 1 - (4.0 + 36.5 + 22.0) / 2 / HD, "SEA-AD component 106 loadings", rotation=90,
         va="center", ha="left", fontsize=LAB, color=mpl.rcParams["axes.labelcolor"])
savep(fig, "S4d_final")

T.to_csv(f"{HERE}/s4_view_stats.csv", float_format="%.4g")
open(f"{HERE}/s4_final_size.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
