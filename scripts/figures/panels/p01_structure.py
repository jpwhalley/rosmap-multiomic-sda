"""Figure S1 (decomposition structure; catalogue level, one point per component, no donor data).
 a  top-donor share by class with simulated references (Gaussian, t(5), best-fitting t(12)); 20,000 simulations each
    (legend S1a); key without the t(5) expected count (a body-fit value, not reported in the text)
 b  top-donor share vs number of modalities with PIP > 0.5 features (adjusted phenotype calls; prioritised outlined)
 c  run recurrence by class
 d  SEA-AD |Pearson| vs |Spearman| (cross-cohort support circled); label overlaps fixed
 e  minimum autosomal GWAS p vs top-donor share
 f  prioritisation counts
Sources: RA/structure/out/05_master_prioritisation.csv (ST 06_panels.py drafts); RA/verify_structure/private/
v02_core_plus.csv (VST v09_panel.py draft for a)."""
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from vp_common import *

M = pd.read_csv(f"{RA}/structure/out/05_master_prioritisation.csv", index_col=0)
V = pd.read_csv(f"{RA}/verify_structure/private/v02_core_plus.csv", index_col=0)
assert (V.cls == M.conc_class).all() and np.allclose(V.top_share, M.top_share)
T1, T2 = 0.10, 0.50
n = M.conc_class.value_counts()
LOG = []
XT = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
def logx(ax, lo=0.004, hi=1.25):
    ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_xticks(XT); ax.set_xticklabels([str(t) for t in XT])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
XLAB = "top-donor share of score variance"

# ---------------------------------------------------------------- a  share histogram vs simulated references
fig, ax = plt.subplots(figsize=(11, 6))
bins = np.arange(-2.4, 0.05, 0.1)
lg = np.log10(V.top_share)
ax.hist([lg[V.cls == c] for c in CLS], bins=bins, stacked=True, color=list(CLS.values()),
        label=[f"{c} (n = {int((V.cls == c).sum())})" for c in CLS])
rng = np.random.default_rng(31)
NSIM = 20000
for nm, draw, ls in [("Gaussian", lambda: rng.standard_normal((NSIM, 276)), "-"),
                     ("t, 5 df", lambda: rng.standard_t(5, (NSIM, 276)), ":"),
                     ("t, 12 df (best fit)", lambda: rng.standard_t(12, (NSIM, 276)), "--")]:
    A = draw(); D = A - A.mean(1, keepdims=True); D2 = D ** 2; ts = D2.max(1) / D2.sum(1)
    nbody = (V.top_share < 0.075).sum() / np.mean(ts < 0.075)
    h, _ = np.histogram(np.log10(ts), bins=bins)
    exp10 = nbody * np.mean(ts >= 0.10)
    lab = f"{nm} reference" if nm.startswith("t, 5") else f"{nm} reference (expected ≥ 0.10: {exp10:.0f})"
    ax.step(bins[:-1], h / len(ts) * nbody, where="post", color="k", ls=ls, lw=1.5, label=lab)
    LOG.append(f"S1a reference {nm}: implied body {nbody:.1f}; expected >= 0.10 {exp10:.2f}; median share {np.median(ts):.4f}; "
               f"P(>=0.10) {np.mean(ts >= 0.10):.5f}")
    del A, D, D2
for t in (T1, T2):
    ax.axvline(np.log10(t), color="grey", ls=(0, (4, 3)), lw=0.9)
ax.set_xticks(np.log10(XT)); ax.set_xticklabels([str(t) for t in XT])
ax.set_xlabel(XLAB + " (log scale)"); ax.set_ylabel("components")
ax.set_ylim(0, 80)
n08 = int((V.top_share > 0.8).sum())
ax.text(np.log10(0.46), 77, f"{n08} above 0.8\n(bar truncated)", ha="right", va="top", fontsize=13)
ax.legend(fontsize=13, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0)); sns.despine(ax=ax)
fig.tight_layout(); save(fig, "FigS1", "S1a_top_share_vs_references", LOG)
LOG.append(f"S1a components > 0.8: {n08}")

# ---------------------------------------------------------------- b  overview
GREY = "#BBBBBB"
PH = {"none": (GREY, "o"), "sex": ("#0072B2", "D"), "CERAD": ("#E69F00", "^"), "diagnosis": ("#009E73", "s"),
      "CERAD and diagnosis": ("#D55E00", "P")}
def ph_cat(s):
    s = "" if pd.isna(s) else s
    if s == "": return "none"
    if s == "sex": return "sex"
    if s == "cerad": return "CERAD"
    if s == "cogdx": return "diagnosis"
    if set(s.split(";")) == {"cerad", "cogdx"}: return "CERAD and diagnosis"
    return s
M["ph"] = M.adj_calls.map(ph_cat); assert set(M.ph) <= set(PH)
rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(13, 6.8))
y = M.n_modalities + rng.uniform(-0.36, 0.36, len(M))
ax.axvspan(0.004, T1, color=CLS["distributed"], alpha=0.06, lw=0); ax.axvspan(T2, 1.25, color=CLS["concentrated"], alpha=0.06, lw=0)
pr = M.prioritised
for c in PH:
    col, mk = PH[c]
    for pp in (False, True):
        s_ = (M.ph == c) & (pr == pp)
        ax.scatter(M.top_share[s_], y[s_], c=col, marker=mk, s=(22 if c == "none" else 70), alpha=0.85 if c == "none" else 1.0,
                   edgecolors="k" if pp else "none", linewidths=0.9 if pp else 0, zorder=2 if c == "none" else 3)
OFF = {74: (0.012, 3.55), 278: (0.010, 2.25), 172: (0.020, 2.6), 277: (0.045, 2.3), 21: (0.075, 0.2),
       351: (0.42, 4.5), 2: (0.3, 3.75), 87: (0.42, 3.3), 177: (0.3, 2.75)}
for k, (tx, ty) in OFF.items():
    ax.annotate(f"C{k}", (M.top_share[k], y[k]), xytext=(tx, ty), fontsize=14, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color="k", lw=0.6, shrinkA=2, shrinkB=3), zorder=5)
for t in (T1, T2): ax.axvline(t, color="k", ls="--", lw=1)
logx(ax); ax.set_ylim(-0.6, 5.1); ax.set_yticks(range(5))
ax.set_xlabel(XLAB + " (log scale)"); ax.set_ylabel("modalities with PIP > 0.5 features")
for x, c in [(0.02, "distributed"), (np.sqrt(T1 * T2), "few-donor"), (np.sqrt(T2 * 1.25), "concentrated")]:
    ax.text(x, 4.85, f"{c}\n(n = {n[c]})", ha="center", va="center", fontsize=14)
h = [Line2D([], [], marker=PH[c][1], color=PH[c][0], ls="", markersize=9, label=c) for c in PH]
h.append(Line2D([], [], marker="o", mfc=GREY, mec="k", mew=0.9, ls="", markersize=8, label=f"prioritised (n = {int(pr.sum())})"))
ax.legend(handles=h, title="adjusted association\n(FDR < 0.05)", fontsize=13, title_fontsize=13,
          loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
ax.get_legend()._legend_box.align = "left"
sns.despine(ax=ax); fig.tight_layout(); save(fig, "FigS1", "S1b_share_vs_modalities", LOG)
LOG.append(f"S1b classes {n.to_dict()}; prioritised {int(pr.sum())}; adjusted-call categories {M.ph.value_counts().to_dict()}")
LOG.append(f"S1b >=3 modalities: {int((M.n_modalities >= 3).sum())}, of them concentrated {int(((M.n_modalities >= 3) & (M.conc_class == 'concentrated')).sum())}")

# ---------------------------------------------------------------- c  run recurrence
fig, ax = plt.subplots(figsize=(8, 5.5))
ct = pd.crosstab(M.true_runs, M.conc_class)[list(CLS)]
frac = ct / ct.sum(0)
x = np.arange(len(ct.index)); w = 0.27
for i, c in enumerate(frac.columns):
    ax.bar(x + (i - 1) * w, frac[c], width=w - 0.02, color=CLS[c], label=f"{c} (n = {int(ct[c].sum())})")
ax.set_xticks(x); ax.set_xticklabels(ct.index); ax.set_xlabel("runs (of 10) in which the component recurs")
ax.set_ylabel("fraction of class"); ax.legend(fontsize=13, frameon=False, loc="upper left"); sns.despine(ax=ax)
fig.tight_layout(); save(fig, "FigS1", "S1c_run_recurrence", LOG)
LOG.append("S1c fraction in all 10 runs: " + ", ".join(f"{c} {frac.loc[10, c]:.2f} ({ct.loc[10, c]}/{ct[c].sum()})" for c in CLS))

# ---------------------------------------------------------------- d  SEA-AD Pearson vs Spearman
fig, ax = plt.subplots(figsize=(7.5, 6.8))
ok = M.seaad_J.notna()
for c in ["concentrated", "few-donor", "distributed"]:
    s = ok & (M.conc_class == c)
    ax.scatter(M.seaad_pearson.abs()[s], M.seaad_spearman.abs()[s], c=CLS[c], s=26, alpha=0.8, lw=0)
bs = M.support_B_seaad.astype(bool)
ax.scatter(M.seaad_pearson.abs()[bs], M.seaad_spearman.abs()[bs], facecolors="none", edgecolors="k", s=90, lw=1)
# label offsets (points); C172 and C277 sit close together, so they are pulled apart with leader lines
LPOS = {74: (0.93, 0.80), 278: (0.30, 0.66), 172: (0.84, 0.45), 277: (0.70, 0.33), 21: (0.24, 0.02), 2: (0.52, 0.13),
        87: (0.44, 0.27), 177: (0.75, 0.24), 351: (0.80, 0.06)}
for k, (tx, ty) in LPOS.items():
    if pd.notna(M.seaad_J[k]):
        ax.annotate(f"C{k}", (abs(M.seaad_pearson[k]), abs(M.seaad_spearman[k])), xytext=(tx, ty), textcoords="data",
                    fontsize=13, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.7, shrinkA=0, shrinkB=2), zorder=6)
ax.axvline(0.4, color="k", ls="--", lw=1); ax.axhline(0.4, color="k", ls="--", lw=1)
ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_xlabel("|Pearson r| with SEA-AD partner"); ax.set_ylabel("|Spearman ρ| with SEA-AD partner")
h = [Line2D([], [], marker="o", color=CLS[c], ls="", markersize=8, label=c) for c in CLS]
h.append(Line2D([], [], marker="o", mfc="none", mec="k", ls="", markersize=10, label="cross-cohort support"))
ax.legend(handles=h, fontsize=13, frameon=False, loc="upper left"); sns.despine(ax=ax); fig.tight_layout()
save(fig, "FigS1", "S1d_seaad_pearson_vs_spearman", LOG)
LOG.append(f"S1d components with a partner: {int(ok.sum())}; meeting the SEA-AD criterion: {int(bs.sum())}")

# ---------------------------------------------------------------- e  GWAS vs share
fig, ax = plt.subplots(figsize=(10, 6))
for c in ["concentrated", "few-donor", "distributed"]:
    s_ = M.conc_class == c
    ax.scatter(M.top_share[s_], -np.log10(M.gwas_min_p_autosomal[s_]), c=CLS[c], s=24, alpha=0.85, lw=0, label=f"{c} (n = {n[c]})")
OFF5 = {74: (0.012, 9.0), 21: (0.012, 17.5), 351: (0.25, 18.5), 2: (0.25, 20.5), 87: (0.25, 22.5), 177: (0.25, 14.0)}
for k, (tx, ty) in OFF5.items():
    ax.annotate(f"C{k}", (M.top_share[k], -np.log10(M.gwas_min_p_autosomal[k])), xytext=(tx, ty), fontsize=14, ha="center",
                arrowprops=dict(arrowstyle="-", color="k", lw=0.6, shrinkB=3))
ax.axhline(15, color="k", ls=":", lw=1, label="p = 10$^{-15}$ (original screen)")
ax.axhline(-np.log10(5e-8 / 547), color="k", ls="--", lw=1, label="p = 5 × 10$^{-8}$ / 547")
ax.scatter([M.top_share[74]], [-np.log10(M.gwas_min_p[74])], marker="x", c="k", s=40,
           label="C74, unplaced sex-linked probes")
logx(ax); ax.set_xlabel(XLAB + " (log scale)"); ax.set_ylabel("−log$_{10}$ minimum GWAS p")
ax.legend(fontsize=13, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5)); sns.despine(ax=ax); fig.tight_layout()
save(fig, "FigS1", "S1e_gwas_vs_share", LOG)
below = M.gwas_min_p < 1e-15
LOG.append(f"S1e min p (all SNPs) < 1e-15: {int(below.sum())}; by class {M.conc_class[below].value_counts().to_dict()}")

# ---------------------------------------------------------------- f  prioritisation flow
fig, ax = plt.subplots(figsize=(10, 5))
popl = M.population_level.astype(bool); P = M[M.prioritised.astype(bool)]
tA = P.support_A1_phenotype_adj.astype(bool) | P.support_A2_gwas.astype(bool)
steps = [("catalogue", len(M)), (f"top-donor share < {T1}", int(popl.sum())), ("≥ 1 independent support", int(len(P)))]
yy = [2, 1, 0]
ax.barh(yy[0], len(M), color="#999999"); ax.barh(yy[1], int(popl.sum()), color=CLS["distributed"])
ax.barh(yy[2], int(tA.sum()), color="#009E73", label=f"phenotype or GWAS ({int(tA.sum())})")
ax.barh(yy[2], int((~tA).sum()), left=int(tA.sum()), color="#56B4E9", label=f"SEA-AD only ({int((~tA).sum())})")
for (lab, v), yv in zip(steps, yy):
    ax.text(v + 6, yv, str(v), va="center", fontsize=14)
ax.set_yticks(yy); ax.set_yticklabels([s_[0] for s_ in steps])
ax.set_xlabel("components"); ax.set_xlim(0, len(M) * 1.1)
ax.legend(fontsize=13, frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.02))
sns.despine(ax=ax); fig.tight_layout(); save(fig, "FigS1", "S1f_prioritisation", LOG)
LOG.append(f"S1f catalogue {len(M)}; distributed {int(popl.sum())}; prioritised {len(P)}: phenotype/GWAS {int(tA.sum())}, SEA-AD only {int((~tA).sum())}")
open(f"{OUT}/p01_structure.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
