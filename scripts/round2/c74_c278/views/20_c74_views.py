"""Task 1: C74 source-data views by sex (SDA input values vs the C74 sample score), in the paper's style.
 a  XIST per cell type (6 panels)                      d  top autosomal CpG (cg26921482, AMDHD2)
 b  top autosomal PIP>0.5 gene (KANSL1) per cell type  e  H3K9ac aggregates: chrX C74 peaks vs autosomal C74 peaks
 c  score by sex (all 276 donors; the four identity-flagged samples marked)
 f  within-sex correlations (forest)                   g  corrected cell-type loadings (bars)
Exclusions: snRNA views drop the two samples whose snRNA belongs to another person (RNA-1, RNA-2); H3K9ac views drop
the two sex-discordant H3K9ac samples (CHIP-1, CHIP-2); methylation has no identity mismatches. Sensitivity: also
dropping the two same-sex H3K9ac swaps (CHIPID-1/2). All other donors with data are shown.
Outputs: panels/C74_*.pdf|png (per-donor points; stay on this host); out/20_c74_view_stats.csv (aggregates only)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as ss
from cc_common import *

mpl.rcParams["pdf.fonttype"] = 42
sns.set(style="white", context="talk")
mpl.rcParams["font.family"] = "sans-serif"; mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
CTCOL = {"Ast": "#D85396", "Oli": "#8BC34A", "Inh": "#303F9F", "OPC": "#E5A43A", "Exc": "#D32F2F", "Mic": "#4DB6E7"}
MUTED = sns.color_palette("muted"); SEXCOL = {"male": MUTED[0], "female": MUTED[2]}
K = 74; a = score(K); fem = female(); sexlab = np.where(fem, "female", "male")
XL = f"Component {K} sample loadings/score"
RNA = rna_input(); genes = list(comp_loadings(K, "RNA").index)
exR = excl_mask("RNA"); exC = excl_mask("chip"); exC6 = excl_mask("chip", extra_same_sex=True)
rows = []


def stats_row(view, y, excl, note=""):
    ok = ~np.isnan(y) & ~excl
    r_all = spearman_ci(y[ok], a[ok], seed=11)
    rm = spearman_ci(y[ok & ~fem], a[ok & ~fem], seed=12); rf = spearman_ci(y[ok & fem], a[ok & fem], seed=13)
    mw = ss.mannwhitneyu(y[ok & fem], y[ok & ~fem])
    auc = mw.statistic / ((ok & fem).sum() * (ok & ~fem).sum())
    d = dict(view=view, n=int(ok.sum()), n_male=int((ok & ~fem).sum()), n_female=int((ok & fem).sum()),
             n_excluded_identity=int((excl & ~np.isnan(y)).sum()),
             rho_all=r_all["rho"], rho_all_lo=r_all["lo"], rho_all_hi=r_all["hi"], p_all=r_all["p"],
             rho_male=rm["rho"], rho_male_lo=rm["lo"], rho_male_hi=rm["hi"], p_male=rm["p"],
             rho_female=rf["rho"], rho_female_lo=rf["lo"], rho_female_hi=rf["hi"], p_female=rf["p"],
             auc_female_gt_male=float(auc), p_sex_MW=float(mw.pvalue), note=note)
    rows.append(d); return d


def scatter(ax, y, excl, ylab, title, d, ycol=None):
    ok = ~np.isnan(y) & ~excl
    for s in ("male", "female"):
        m = ok & (sexlab == s)
        ax.scatter(a[m], y[m], s=16, color=SEXCOL[s], alpha=0.75, linewidths=0, label=f"{s} (n = {m.sum()})")
    ax.set_xlabel(XL, fontsize=11); ax.set_ylabel(ylab, fontsize=11); ax.set_title(title, fontsize=12, loc="left")
    ax.tick_params(labelsize=9); sns.despine(ax=ax)
    ax.text(0.02, 0.98, f"n = {d['n']}; ρ = {d['rho_all']:+.2f}\nwithin sex: ρ♂ {d['rho_male']:+.2f}, ρ♀ {d['rho_female']:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5)


def save(fig, name):
    fig.savefig(f"{PAN}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{PAN}/{name}.png", dpi=200, bbox_inches="tight")


figs = []
# ---- a/b: XIST and the top autosomal gene per cell type ----
gc = gene_chrom(); LR = comp_loadings(K, "RNA"); LR["chr"] = [gc.get(g, "NA") for g in LR.index]
auto = LR[(LR.P > 0.5) & ~LR.chr.isin(["X", "Y", "NA"])]
top_auto = auto.X.abs().idxmax()
for g, tag in (("XIST", "a_XIST"), (top_auto, "b_" + top_auto)):
    j = genes.index(g)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9)); axs = axs.ravel()
    for i, ct in enumerate(CTS):
        y = np.log2(np.asarray(RNA[i, :, j], float) + 1)
        d = stats_row(f"RNA {g} {ct}", y, exR)
        scatter(axs[i], y, exR, f"{g} (log2 CPM+1)", f"{g}, {CTNAME[ct]}", d)
    axs[0].legend(fontsize=8.5, frameon=False, loc="lower left", markerscale=1.2)
    chrom = "chrX" if g == "XIST" else f"chr{LR.loc[g, 'chr']}"
    fig.suptitle(f"Component {K}: {g} ({chrom}; loading {LR.loc[g, 'X']:+.0f}, PIP {LR.loc[g, 'P']:.2f}) in each cell type vs sample score. "
                 f"snRNA of 2 identity-flagged samples excluded.", fontsize=12, y=1.0)
    fig.tight_layout(); save(fig, f"C74_{tag}_by_celltype"); figs.append(fig)

# ---- c: score by sex, all donors, flagged marked ----
fl = four_flag_mask()
fig, ax = plt.subplots(figsize=(6, 6))
rs = np.random.default_rng(3)
jit = rs.uniform(-0.18, 0.18, 276)
for i, s in enumerate(("male", "female")):
    m = sexlab == s
    ax.scatter(i + jit[m], a[m], s=18, color=SEXCOL[s], alpha=0.8, linewidths=0)
    mf = m & fl
    ax.scatter(i + jit[mf], a[mf], s=70, facecolors="none", edgecolors="k", linewidths=1.4,
               label="identity-flagged sample (snRNA or H3K9ac of another person)" if i == 0 else None)
ax.set_xticks([0, 1]); ax.set_xticklabels([f"male\n(n = {(~fem).sum()})", f"female\n(n = {fem.sum()})"])
ax.set_ylabel("loadings/score"); ax.set_title(f"Component {K} score by recorded sex (all 276 donors)", fontsize=12, loc="left")
ax.legend(fontsize=8.5, frameon=False, loc="center"); sns.despine(ax=ax)
# separation summary
gap_m = np.sort(a[~fem & ~fl]); gap_f = np.sort(a[fem & ~fl])
sep_txt = f"unflagged donors: male range [{gap_m.min():.3f}, {gap_m.max():.3f}], female range [{gap_f.min():.3f}, {gap_f.max():.3f}]"
ax.text(0.0, -0.18, sep_txt, transform=ax.transAxes, fontsize=8)
fig.tight_layout(); save(fig, "C74_c_score_by_sex"); figs.append(fig)
n_between = int(((a > min(gap_m.max(), gap_f.max())) & (a < max(gap_m.min(), gap_f.min()))).sum()) if False else None

# ---- d/e: top autosomal CpG; H3K9ac aggregates (chrX vs autosomal C74 peaks) ----
MI = mod_input("meth"); LM = comp_loadings(K, "meth"); cpgs = list(LM.index)
LMp = LM[LM.P > 0.5]; top_cpg = LMp.X.abs().idxmax()
ym = np.asarray(MI[:, cpgs.index(top_cpg)], float)
CI = mod_input("chip"); LC = comp_loadings(K, "chip"); H = peaks().loc[list(LC.index)]
chrc = H.Chr.astype(str).values
selX = (LC.P.values > 0.5) & (chrc == "X"); selA = (LC.P.values > 0.5) & ~np.isin(chrc, ["X", "Y"]); selY = (LC.P.values > 0.5) & (chrc == "Y")
C_all = np.asarray(CI, float)
aggX, nX = zagg(C_all[:, selX], LC.X.values[selX]); aggA, nA = zagg(C_all[:, selA], LC.X.values[selA])
aggY, nY = zagg(C_all[:, selY], LC.X.values[selY])
# methylation aggregate (all C74 CpGs are autosomal: the SDA 450K input holds no chrX/chrY probes)
M_all = np.asarray(MI, float); selM = LM.P.values > 0.5
aggM, nM = zagg(M_all[:, selM], LM.X.values[selM])
nomask = np.zeros(276, bool)
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
d = stats_row(f"meth {top_cpg} (top autosomal CpG)", ym, nomask)
scatter(axs[0], ym, nomask, f"{top_cpg} methylation (β)", f"Top autosomal CpG ({top_cpg}, AMDHD2)", d)
d = stats_row(f"H3K9ac aggregate chrX C74 peaks ({nX})", aggX, exC)
scatter(axs[1], aggX, exC, f"weighted mean z, {nX} chrX peaks\n(sign by loading)", "H3K9ac: chrX C74 peaks", d)
d = stats_row(f"H3K9ac aggregate autosomal C74 peaks ({nA})", aggA, exC)
scatter(axs[2], aggA, exC, f"weighted mean z, {nA} autosomal peaks\n(sign by loading)", "H3K9ac: autosomal C74 peaks", d)
axs[0].legend(fontsize=8.5, frameon=False, loc="lower left")
fig.suptitle(f"Component {K}: methylation and H3K9ac source data vs sample score (H3K9ac of 2 sex-discordant samples excluded)", fontsize=12, y=1.02)
fig.tight_layout(); save(fig, "C74_de_cpg_h3k9ac"); figs.append(fig)
# extra aggregate rows (not drawn as scatter) + sensitivities
stats_row(f"H3K9ac aggregate chrY C74 peaks ({nY})", aggY, exC, note="chrY peaks: signal only in males")
stats_row(f"meth aggregate all C74 CpGs ({nM}, all autosomal)", aggM, nomask)
stats_row(f"H3K9ac aggregate chrX C74 peaks ({nX}) [also excl. 2 same-sex swaps]", aggX, exC6, note="sensitivity")
stats_row(f"H3K9ac aggregate autosomal C74 peaks ({nA}) [also excl. 2 same-sex swaps]", aggA, exC6, note="sensitivity")
selRxy = (LR.P.values > 0.5) & np.isin(LR.chr.values, ["X", "Y"]); selRa = (LR.P.values > 0.5) & ~np.isin(LR.chr.values, ["X", "Y", "NA"])
Bc = pd.read_csv(BCOR, index_col=0).loc[K, CTS]
for i, ct in enumerate(CTS):
    Rc = np.log2(np.asarray(RNA[i], float) + 1)
    w = LR.X.values * Bc[ct]
    ag, n1 = zagg(Rc[:, selRxy], w[selRxy]); stats_row(f"RNA aggregate X/Y PIP genes {ct}", ag, exR)
    ag, n2 = zagg(Rc[:, selRa], w[selRa]); stats_row(f"RNA aggregate autosomal PIP genes {ct}", ag, exR)
# unflagged-only check of the intermediate scores
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/20_c74_view_stats.csv", index=False, float_format="%.4g")

# ---- f: within-sex forest ----
show = [v for v in T.view if (v.startswith("RNA XIST") or v.startswith(f"RNA {top_auto}") or v.startswith("meth ") or
                               (v.startswith("H3K9ac aggregate") and "swaps" not in v) or v.startswith("RNA aggregate autosomal"))]
S = T.set_index("view").loc[show]
fig, ax = plt.subplots(figsize=(9, 0.42 * len(S) + 1.5))
yy = np.arange(len(S))[::-1]
for off, s, col in ((-0.15, "male", SEXCOL["male"]), (0.15, "female", SEXCOL["female"])):
    ax.errorbar(S[f"rho_{s}"], yy + off, xerr=[S[f"rho_{s}"] - S[f"rho_{s}_lo"], S[f"rho_{s}_hi"] - S[f"rho_{s}"]],
                fmt="o", color=col, ms=5, lw=1.2, capsize=0, label=f"within {s}s")
ax.scatter(S.rho_all, yy, marker="|", s=120, color="k", label="all donors (sex split included)", zorder=3)
ax.axvline(0, color="grey", lw=0.8, ls=":")
ax.set_yticks(yy); ax.set_yticklabels([f"{v}  (n {int(S.loc[v, 'n_male'])}♂/{int(S.loc[v, 'n_female'])}♀)" for v in S.index], fontsize=8.5)
ax.set_xlabel("Spearman ρ with Component 74 score (95% bootstrap CI)", fontsize=11)
ax.legend(fontsize=8.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
sns.despine(ax=ax); fig.tight_layout(); save(fig, "C74_f_within_sex"); figs.append(fig)

# ---- g: corrected cell-type bars ----
B = pd.read_csv(BCOR, index_col=0).loc[K, ["Ast", "Exc", "Inh", "Mic", "OPC", "Oli"]]
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(range(6), B.values, color=[CTCOL[c] for c in B.index]); ax.set_xticks(range(6)); ax.set_xticklabels(B.index)
ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("loadings/score"); ax.set_title(f"Component {K}: corrected cell-type loadings", fontsize=12, loc="left")
sns.despine(ax=ax); fig.tight_layout(); save(fig, "C74_g_celltype_bars_corrected"); figs.append(fig)

with PdfPages(f"{PAN}/C74_ALL_source_views_draft.pdf") as pdf:
    for f in figs: pdf.savefig(f, bbox_inches="tight")
pd.set_option("display.width", 250)
print(T[["view", "n", "n_male", "n_female", "rho_all", "rho_male", "rho_male_lo", "rho_male_hi", "rho_female", "rho_female_lo", "rho_female_hi", "auc_female_gt_male"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("top autosomal gene:", top_auto, "| top autosomal CpG:", top_cpg, "| H3K9ac peaks X/auto/Y:", nX, nA, nY)
print("corrected bars:", B.round(3).to_dict())
print(sep_txt)
fa = fl & np.isfinite(a)
lo_f, hi_m = (gap_f.max(), gap_m.min()) if np.median(gap_f) < np.median(gap_m) else (gap_m.max(), gap_f.min())
print(f"donors with score strictly between the unflagged sex clusters: {int(((a > lo_f) & (a < hi_m)).sum())}; of them flagged: {int(((a > lo_f) & (a < hi_m) & fl).sum())}")
