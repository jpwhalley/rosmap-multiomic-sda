"""Task 7: SOX6/BCAS1 and OPC / oligodendrocyte-stage markers across donors vs the C278 score (support for withdrawing
'precursor-like'). Marker lists fixed before looking (standard lineage markers):
  OPC:            PDGFRA CSPG4 PTPRZ1 VCAN OLIG1 OLIG2 SOX6 PCDH15 LHFPL3 MEGF11 TNR
  committed/new:  GPR17 BCAS1 ENPP6 TCF7L2 ITPR2
  mature Oli:     MOG MBP PLP1 MOBP OPALIN MAG CNP ST18 CLDN11 ERMN KLK6
A precursor-like shift in high-score (female-biased) donors predicts OPC/committed markers UP and mature markers DOWN with
the score (in Oli nuclei especially), and/or a higher OPC share of oligodendrocyte-lineage nuclei.
snRNA of RNA-1/RNA-2 excluded. Outputs: out/32_* (aggregates), panels/C278_f_markers.*"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats as ss
from cc_common import *

mpl.rcParams["pdf.fonttype"] = 42
sns.set(style="white", context="talk")
mpl.rcParams["font.family"] = "sans-serif"; mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
CTCOL = {"Oli": "#8BC34A", "OPC": "#E5A43A"}
MUTED = sns.color_palette("muted"); SEXCOL = {"male": MUTED[0], "female": MUTED[2]}
MK = {"OPC": ["PDGFRA", "CSPG4", "PTPRZ1", "VCAN", "OLIG1", "OLIG2", "SOX6", "PCDH15", "LHFPL3", "MEGF11", "TNR"],
      "committed/newly formed": ["GPR17", "BCAS1", "ENPP6", "TCF7L2", "ITPR2"],
      "mature Oli": ["MOG", "MBP", "PLP1", "MOBP", "OPALIN", "MAG", "CNP", "ST18", "CLDN11", "ERMN", "KLK6"]}
K = 278; a = score(K); fem = female(); keep = ~excl_mask("RNA"); sexlab = np.where(fem, "female", "male")
RNA = rna_input(); LR = comp_loadings(K, "RNA"); genes = list(LR.index); fk = np.sign(LR.loc["FKBP5", "X"])
rows = []
for grp, gl in MK.items():
    for g in gl:
        if g not in genes: print("missing", g); continue
        j = genes.index(g); d = dict(group=grp, gene=g, C278_loading=LR.loc[g, "X"], C278_PIP=LR.loc[g, "P"],
                                     model_direction=("rises with score" if np.sign(LR.loc[g, "X"]) == fk else "falls with score"))
        for ct in ("OPC", "Oli"):
            y = np.log2(np.asarray(RNA[CTS.index(ct), :, j], float) + 1)
            r = spearman_ci(y[keep], a[keep], B=2000, seed=31)
            d.update({f"mean_log2cpm_{ct}": float(y[keep].mean()), f"rho_{ct}": r["rho"], f"rho_{ct}_lo": r["lo"], f"rho_{ct}_hi": r["hi"], f"p_{ct}": r["p"]})
        rows.append(d)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/32_c278_marker_rhos.csv", index=False, float_format="%.4g")

# module scores (mean z of each marker group, within each lineage)
mods = []
for grp, gl in MK.items():
    idx = [genes.index(g) for g in gl if g in genes]
    for ct in ("OPC", "Oli"):
        E = np.log2(np.asarray(RNA[CTS.index(ct)], float)[:, idx] + 1)
        sc, _ = zagg(E, np.ones(len(idx)))
        r = spearman_ci(sc[keep], a[keep], B=2000, seed=32)
        mods.append(dict(module=grp, cell_type=ct, n_genes=len(idx), n_donors=int(keep.sum()), rho=r["rho"], lo=r["lo"], hi=r["hi"], p=r["p"]))
# composition: OPC share of oligodendrocyte-lineage nuclei, OPC and Oli proportions of all nuclei
qc = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str); qc = qc.loc[donors()]
tot = qc[[f"nuclei_{c}" for c in CTS]].sum(1).values
comp = {"OPC / (OPC + Oli) nuclei": (qc.nuclei_OPC / (qc.nuclei_OPC + qc.nuclei_Oli)).values,
        "OPC proportion of all nuclei": qc.nuclei_OPC.values / tot, "Oli proportion of all nuclei": qc.nuclei_Oli.values / tot}
for k_, v in comp.items():
    r = spearman_ci(v[keep], a[keep], B=2000, seed=33)
    mods.append(dict(module=k_, cell_type="composition", n_genes=np.nan, n_donors=int(keep.sum()), rho=r["rho"], lo=r["lo"], hi=r["hi"], p=r["p"],
                     female_minus_male=float(np.median(v[keep & fem]) - np.median(v[keep & ~fem])), p_sex_MW=float(ss.mannwhitneyu(v[keep & fem], v[keep & ~fem]).pvalue)))
M = pd.DataFrame(mods); M.to_csv(f"{OUT}/32_c278_marker_modules_composition.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
print(T[["group", "gene", "C278_loading", "C278_PIP", "model_direction", "mean_log2cpm_OPC", "mean_log2cpm_Oli", "rho_OPC", "rho_OPC_lo", "rho_OPC_hi", "rho_Oli", "rho_Oli_lo", "rho_Oli_hi"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(M.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# ---- panel ----
fig = plt.figure(figsize=(21, 10.5)); gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.15, 1.0], wspace=0.62, hspace=0.45)
for r_, g in enumerate(("SOX6", "BCAS1")):
    for c_, ct in enumerate(("OPC", "Oli")):
        ax = fig.add_subplot(gs[r_, c_]); y = np.log2(np.asarray(RNA[CTS.index(ct), :, genes.index(g)], float) + 1)
        for s in ("male", "female"):
            m = keep & (sexlab == s); ax.scatter(a[m], y[m], s=12, color=SEXCOL[s], alpha=0.7, linewidths=0, label=f"{s} (n = {m.sum()})")
        rr = T.set_index("gene").loc[g]
        ax.text(0.02, 0.98, f"n = {keep.sum()}; ρ = {rr[f'rho_{ct}']:+.2f} [{rr[f'rho_{ct}_lo']:+.2f}, {rr[f'rho_{ct}_hi']:+.2f}]", transform=ax.transAxes, va="top", fontsize=8.5)
        ax.set_title(f"{g} in {CTNAME[ct]}", fontsize=11, loc="left"); ax.set_xlabel(f"Component {K} sample loadings/score", fontsize=10)
        ax.set_ylabel(f"{g} (log2 CPM+1)", fontsize=10); ax.tick_params(labelsize=8.5); sns.despine(ax=ax)
        if r_ == 0 and c_ == 0: ax.legend(fontsize=8, frameon=False, loc="lower left")
ax = fig.add_subplot(gs[:, 2])
TT = T.copy(); TT["lab"] = TT.gene + "  (" + TT.group.str.replace("committed/newly formed", "committed") + ")"
yy = np.arange(len(TT))[::-1]
for off, ct in ((-0.18, "OPC"), (0.18, "Oli")):
    ax.errorbar(TT[f"rho_{ct}"], yy + off, xerr=[TT[f"rho_{ct}"] - TT[f"rho_{ct}_lo"], TT[f"rho_{ct}_hi"] - TT[f"rho_{ct}"]], fmt="o", ms=4, lw=1,
                color=CTCOL[ct], label=f"in {CTNAME[ct]}")
ax.axvline(0, color="grey", lw=0.7, ls=":"); ax.set_yticks(yy); ax.set_yticklabels(TT.lab, fontsize=8)
for b in np.cumsum([len(v) for v in MK.values()])[:-1]:
    ax.axhline(len(TT) - b - 0.5, color="k", lw=0.5)
ax.set_xlabel("Spearman ρ with C278 score (95% CI)", fontsize=10); ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.tick_params(labelsize=8.5)
ax.set_title("Lineage markers vs score", fontsize=11, loc="left"); sns.despine(ax=ax)
ax = fig.add_subplot(gs[0, 3]); v = comp["OPC / (OPC + Oli) nuclei"]
for s in ("male", "female"):
    m = keep & (sexlab == s); ax.scatter(a[m], v[m], s=12, color=SEXCOL[s], alpha=0.7, linewidths=0)
rr = M.set_index("module").loc["OPC / (OPC + Oli) nuclei"]
ax.text(0.02, 0.98, f"n = {keep.sum()}; ρ = {rr.rho:+.2f} [{rr.lo:+.2f}, {rr.hi:+.2f}]", transform=ax.transAxes, va="top", fontsize=8.5)
ax.set_ylabel("OPC share of OPC + Oli nuclei", fontsize=10); ax.set_xlabel(f"Component {K} sample loadings/score", fontsize=10)
ax.set_title("Lineage composition", fontsize=11, loc="left"); ax.tick_params(labelsize=8.5); sns.despine(ax=ax)
ax = fig.add_subplot(gs[1, 3]); MM = M[M.cell_type != "composition"]
yy = np.arange(len(MM))[::-1]
ax.errorbar(MM.rho, yy, xerr=[MM.rho - MM.lo, MM.hi - MM.rho], fmt="o", color="k", ms=5, lw=1)
ax.axvline(0, color="grey", lw=0.7, ls=":"); ax.set_yticks(yy); ax.set_yticklabels([f"{m_}, {c_}" for m_, c_ in zip(MM.module.str.replace("committed/newly formed", "committed").str.replace("mature Oli", "mature"), MM.cell_type)], fontsize=8)
ax.set_xlabel("ρ with score (95% CI)", fontsize=10); ax.set_title("Marker modules (mean z)", fontsize=11, loc="left"); ax.tick_params(labelsize=8.5); sns.despine(ax=ax)
fig.suptitle(f"Component {K}: OPC / oligodendrocyte-stage markers and composition across donors (snRNA of 2 identity-flagged samples excluded)", fontsize=12, y=0.98)
fig.savefig(f"{PAN}/C278_f_markers.pdf", bbox_inches="tight"); fig.savefig(f"{PAN}/C278_f_markers.png", dpi=200, bbox_inches="tight")
