"""Task 5: C278 source-data views and effect estimates.
 a  FKBP5, SGK1, ZBTB16 in OPC and Oli (SDA input, log2 CPM+1) vs the C278 score, coloured by recorded sex
 b  aggregates: loading-weighted mean z of all C278 PIP > 0.5 genes (sign = gene loading x cell-type loading), in OPC and
    in Oli; and the same without the three leading genes
 c  score by sex (all 276 donors) and the measured female-male difference of the three genes (bootstrap CIs)
 d  forest: female vs male effect on the C278 score across covariate models (proportional-odds OR with profile-likelihood
    95% CI, and rank-INT OLS in SD units with t CI) -- reused from component_dossier/out/09_effect_table_oriented.csv
    (independently re-derived in verify_dossier/out/v02_effects_278.csv; agreement checked here)
snRNA of RNA-1/RNA-2 (another person's nuclei) excluded from every RNA view; all other donors shown.
Outputs: panels/C278_*; out/30_c278_view_stats.csv, out/30_c278_effects_forest.csv (aggregates only)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd, re, matplotlib as mpl
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
K = 278; a = score(K); fem = female(); sexlab = np.where(fem, "female", "male")
XL = f"Component {K} sample loadings/score"
RNA = rna_input(); LR = comp_loadings(K, "RNA"); genes = list(LR.index)
Bc = pd.read_csv(BCOR, index_col=0).loc[K, CTS]
exR = excl_mask("RNA"); rows = []
LEAD = ["FKBP5", "SGK1", "ZBTB16"]


def stats_row(view, y, excl):
    ok = ~np.isnan(y) & ~excl
    r = spearman_ci(y[ok], a[ok], seed=21); rm = spearman_ci(y[ok & ~fem], a[ok & ~fem], seed=22); rf = spearman_ci(y[ok & fem], a[ok & fem], seed=23)
    yf, ym = y[ok & fem], y[ok & ~fem]
    rng = np.random.default_rng(24); bs = [rng.choice(yf, len(yf)).mean() - rng.choice(ym, len(ym)).mean() for _ in range(4000)]
    d = dict(view=view, n=int(ok.sum()), n_male=int((ok & ~fem).sum()), n_female=int((ok & fem).sum()),
             rho_all=r["rho"], rho_all_lo=r["lo"], rho_all_hi=r["hi"], p_all=r["p"],
             rho_male=rm["rho"], rho_male_lo=rm["lo"], rho_male_hi=rm["hi"], rho_female=rf["rho"], rho_female_lo=rf["lo"], rho_female_hi=rf["hi"],
             female_minus_male=float(yf.mean() - ym.mean()), fmm_lo=float(np.percentile(bs, 2.5)), fmm_hi=float(np.percentile(bs, 97.5)),
             p_sex_MW=float(ss.mannwhitneyu(yf, ym).pvalue))
    rows.append(d); return d


def scatter(ax, y, excl, ylab, title, d):
    ok = ~np.isnan(y) & ~excl
    for s in ("male", "female"):
        m = ok & (sexlab == s)
        ax.scatter(a[m], y[m], s=16, color=SEXCOL[s], alpha=0.75, linewidths=0, label=f"{s} (n = {m.sum()})")
    ax.set_xlabel(XL, fontsize=11); ax.set_ylabel(ylab, fontsize=11); ax.set_title(title, fontsize=12, loc="left")
    ax.tick_params(labelsize=9); sns.despine(ax=ax)
    ax.text(0.02, 0.98, f"n = {d['n']}; ρ = {d['rho_all']:+.2f} [{d['rho_all_lo']:+.2f}, {d['rho_all_hi']:+.2f}]\n"
            f"within sex: ρ♂ {d['rho_male']:+.2f}, ρ♀ {d['rho_female']:+.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=8.5)


def save(fig, name):
    fig.savefig(f"{PAN}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{PAN}/{name}.png", dpi=200, bbox_inches="tight")


figs = []
# ---- a: three leading genes x OPC/Oli ----
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
for r_, ct in enumerate(("OPC", "Oli")):
    c = CTS.index(ct)
    for k_, g in enumerate(LEAD):
        y = np.log2(np.asarray(RNA[c, :, genes.index(g)], float) + 1)
        d = stats_row(f"RNA {g} {ct}", y, exR)
        scatter(axs[r_, k_], y, exR, f"{g}, {CTNAME[ct]} (log2 CPM+1)", f"{g} in {CTNAME[ct]} (loading {LR.loc[g,'X']:+.0f}, PIP {LR.loc[g,'P']:.2f})", d)
axs[0, 0].legend(fontsize=8.5, frameon=False, loc="lower right")
fig.suptitle(f"Component {K}: leading genes in OPCs and oligodendrocytes vs sample score (snRNA of 2 identity-flagged samples excluded; "
             f"corrected cell-type loadings OPC {Bc['OPC']:+.2f}, Oli {Bc['Oli']:+.2f})", fontsize=11.5, y=1.0)
fig.tight_layout(); save(fig, "C278_a_leading_genes_OPC_Oli"); figs.append(fig)

# ---- b: aggregates ----
pip = LR.P.values > 0.5; nolead = pip & ~np.isin(genes, LEAD)
fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.5))
for r_, ct in enumerate(("OPC", "Oli")):
    c = CTS.index(ct); Rc = np.log2(np.asarray(RNA[c], float) + 1); w = LR.X.values * Bc[ct]
    for k_, (sel, lab) in enumerate(((pip, f"all {pip.sum()} PIP > 0.5 genes"), (nolead, f"{nolead.sum()} PIP > 0.5 genes, FKBP5/SGK1/ZBTB16 removed"))):
        ag, n_ = zagg(Rc[:, sel], w[sel])
        d = stats_row(f"RNA aggregate {ct} ({lab})", ag, exR)
        scatter(axs[r_, k_], ag, exR, f"weighted mean z ({CTNAME[ct]})\n(sign by loading product)", f"{CTNAME[ct]}: {lab}", d)
axs[0, 0].legend(fontsize=8.5, frameon=False, loc="lower right")
fig.suptitle(f"Component {K}: aggregate of the component's RNA features in OPCs and oligodendrocytes vs score", fontsize=12, y=1.0)
fig.tight_layout(); save(fig, "C278_b_aggregates"); figs.append(fig)
for ct in ("Ast", "Inh", "Exc", "Mic"):     # other cell types, for the record (component's cell-type loadings ~0 there)
    c = CTS.index(ct); Rc = np.log2(np.asarray(RNA[c], float) + 1)
    ag, _ = zagg(Rc[:, pip], LR.X.values[pip] * np.sign(Bc["OPC"])); stats_row(f"RNA aggregate {ct} (all PIP genes, OPC-direction weights)", ag, exR)
    for g in LEAD:
        stats_row(f"RNA {g} {ct}", np.log2(np.asarray(RNA[c, :, genes.index(g)], float) + 1), exR)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/30_c278_view_stats.csv", index=False, float_format="%.4g")

# ---- c: score by sex + measured female-male differences ----
fig, axs = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw=dict(width_ratios=[1, 1.6]))
ax = axs[0]; rs = np.random.default_rng(3)
for i, s in enumerate(("male", "female")):
    m = sexlab == s
    ax.scatter(i + rs.uniform(-0.2, 0.2, m.sum()), a[m], s=16, color=SEXCOL[s], alpha=0.8, linewidths=0)
    ax.plot([i - 0.28, i + 0.28], [np.median(a[m])] * 2, color="k", lw=2)
ax.set_xticks([0, 1]); ax.set_xticklabels([f"male\n(n = {(~fem).sum()})", f"female\n(n = {fem.sum()})"])
ax.set_ylabel("loadings/score"); ax.set_title(f"Component {K} score by recorded sex\n(all 276 donors; bar = median)", fontsize=12, loc="left"); sns.despine(ax=ax)
E = pd.read_csv(f"{R2}/component_dossier/out/09_effect_table_oriented.csv"); E = E[E.component == K].drop_duplicates("model")
m1 = E[E.model == "M1"].iloc[0]
ax.text(0.0, -0.22, f"M1 (age, PMI, cohort): OR {m1.PO_OR_95CI}; {m1.RINT_SD_95CI} SD", transform=ax.transAxes, fontsize=9)
ax = axs[1]; S = T.set_index("view").loc[[f"RNA {g} {ct}" for ct in ("OPC", "Oli") for g in LEAD]]
yy = np.arange(len(S))[::-1]
ax.errorbar(S.female_minus_male, yy, xerr=[S.female_minus_male - S.fmm_lo, S.fmm_hi - S.female_minus_male], fmt="o", color="k", ms=6, capsize=0)
ax.axvline(0, color="grey", lw=0.8, ls=":"); ax.set_yticks(yy)
ax.set_yticklabels([f"{v.split()[1]} {v.split()[2]} (n {int(S.loc[v,'n_female'])}♀/{int(S.loc[v,'n_male'])}♂; MW p {fmt_p(S.loc[v,'p_sex_MW'])})" for v in S.index], fontsize=9)
ax.set_xlabel("female − male, mean log2(CPM+1) (95% bootstrap CI)", fontsize=11); ax.set_title("Measured expression by sex (source data)", fontsize=12, loc="left")
sns.despine(ax=ax); fig.tight_layout(); save(fig, "C278_c_score_and_measured_sex_difference"); figs.append(fig)

# ---- d: forest across models ----
def parse(s):
    v = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", s)]; return v[0], v[1], v[2]
F = []
for _, r in E.iterrows():
    o, lo, hi = parse(r.PO_OR_95CI); b, blo, bhi = parse(r.RINT_SD_95CI)
    F.append(dict(model=r.model, n=r.n, OR=o, OR_lo=lo, OR_hi=hi, p_PO=r.p_PO_LRT, q_PO_547=r.q_PO_family,
                  SD=b, SD_lo=blo, SD_hi=bhi, p_rint=r.p_rint_F, q_rint_547=r.q_rint_family))
F = pd.DataFrame(F)
V = pd.read_csv(f"{R2}/verify_dossier/out/v02_effects_278.csv").set_index("model")
F["verify_OR"] = F.model.map(V.OR); F["verify_SD"] = F.model.map(V.r38_b)
assert np.allclose(F.OR, F.verify_OR, rtol=0.01) and np.allclose(F.SD, F.verify_SD, atol=0.01)
F.to_csv(f"{OUT}/30_c278_effects_forest.csv", index=False, float_format="%.4g")
lab = {"M0": "M0 unadjusted", "M1": "M1 age + PMI + cohort", "M2": "M2 M1 + snRNA QC (nuclei, UMIs/nucleus)", "M3": "M3 age + 90+ flag + log PMI + cohort",
       "drop1-M1": "M1 without the most extreme donor", "drop1-M2": "M2 without the most extreme donor",
       "M1+CERAD+cogdx6": "M1 + CERAD + cogdx", "M1+composition": "M1 + cell-type composition"}
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
yy = np.arange(len(F))[::-1]
axs[0].errorbar(F.OR, yy, xerr=[F.OR - F.OR_lo, F.OR_hi - F.OR], fmt="s", color=MUTED[2], ms=7, capsize=0, lw=1.6)
axs[0].set_xscale("log"); axs[0].axvline(1, color="grey", ls=":", lw=0.8)
axs[0].set_xticks([1, 1.5, 2, 3, 4]); axs[0].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter()); axs[0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
axs[0].set_xlabel("female vs male, proportional-odds OR (95% profile CI)", fontsize=11)
axs[0].set_yticks(yy); axs[0].set_yticklabels([f"{lab.get(m, m)} (n {n})" for m, n in zip(F.model, F.n)], fontsize=9.5)
for y_, (_, r) in zip(yy, F.iterrows()):
    axs[0].text(4.6, y_, f"p {fmt_p(r.p_PO)}" + (f"; q {r.q_PO_547:.3f}" if pd.notna(r.q_PO_547) else ""), va="center", fontsize=8.5)
axs[0].set_xlim(0.9, 7.5)
axs[1].errorbar(F.SD, yy, xerr=[F.SD - F.SD_lo, F.SD_hi - F.SD], fmt="o", color=MUTED[2], ms=7, capsize=0, lw=1.6)
axs[1].axvline(0, color="grey", ls=":", lw=0.8); axs[1].set_xlabel("female − male, rank-INT score (SD; 95% CI)", fontsize=11)
for y_, (_, r) in zip(yy, F.iterrows()):
    axs[1].text(0.8, y_, f"p {fmt_p(r.p_rint)}" + (f"; q {r.q_rint_547:.3f}" if pd.notna(r.q_rint_547) else ""), va="center", fontsize=8.5)
axs[1].set_xlim(-0.1, 1.15)
for ax in axs: sns.despine(ax=ax); ax.tick_params(labelsize=9.5)
fig.suptitle(f"Component {K}: female vs male effect across models (q = BH over the 547-component family, where defined)", fontsize=12, y=1.02)
fig.tight_layout(); save(fig, "C278_d_effects_forest"); figs.append(fig)

with PdfPages(f"{PAN}/C278_ALL_source_views_draft.pdf") as pdf:
    for f in figs: pdf.savefig(f, bbox_inches="tight")
pd.set_option("display.width", 250)
print(T[["view", "n", "n_male", "n_female", "rho_all", "rho_all_lo", "rho_all_hi", "rho_male", "rho_female", "female_minus_male", "fmm_lo", "fmm_hi", "p_sex_MW"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print(F[["model", "n", "OR", "OR_lo", "OR_hi", "p_PO", "q_PO_547", "SD", "SD_lo", "SD_hi", "q_rint_547"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("cell-type loadings (corrected):", Bc.round(3).to_dict(), "| n female", int(fem.sum()), "male", int((~fem).sum()))
