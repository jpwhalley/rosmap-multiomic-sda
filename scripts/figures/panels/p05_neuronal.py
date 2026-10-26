"""Figure 4i, 4j, Figure S6 and Figure S7 (Components 172 and 277).
Per-donor views: RA/revision_analyses/neuronal_c21/cache/n05_views.npy (n05_source_data.py: SDA input CPM; top-50 PIP > 0.5
gene aggregate in the dominant cell type, weights = gene loading x corrected cell-type loading; snRNA-seq RNA-1/RNA-2
excluded, n 274; verified VNC v04). Joint models: neuronal_c21/out/n03_joint_models.csv (verified VNC v01 R polr).
Composition: verify_neuronal_c21/cache/frame_comp.csv (RNA-mismatched donors excluded, n 274) and
verify_neuronal_c21/out/v03_total_direct.csv (n 254 rows).
 Fig 4i  LRP8 (Inh) vs C172 score;  Fig 4j  PLCB1 (Exc) vs C277 score
 S6a,b   odds ratios, each phenotype alone (open) and jointly (filled), n 254
 S6c,d   C277 score vs neuronal fraction and vs layer 2-3 CBLN2 share of Exc nuclei, coloured by diagnosis
 S6e     AD dementia vs NCI effect on the rank-INT C277 score, total and after each composition set (n 254)
 S7a-l   C172 (Inh) and C277 (Exc) source data vs score and vs phenotype"""
import numpy as np, pandas as pd
from scipy import stats as ss
from matplotlib.lines import Line2D
from vp_common import *

LOG = []
V = np.load(f"{RA}/neuronal_c21/cache/n05_views.npy", allow_pickle=True).item()
clin = clinical()
SET2 = sns.color_palette("Set2"); GREYB = "#9e9e9e"
DXCOL = {1: SET2[0], 2: SET2[2], 4: SET2[1]}
GENE_LAB = {"aggregate": "top-50 gene aggregate"}

def scatter_view(k, view, fname, folder, size=(8, 6)):
    c = V[k]; a = c["a"]; y = np.asarray(c["views"][view], float); ct = c["ct"]; ext = most_extreme(a)
    assert ext == c["ext"]
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(a, y, s=30, color=CTCOL[ct], alpha=0.8, linewidths=0, zorder=2)
    ax.scatter([a[ext]], [y[ext]], s=110, facecolors="none", edgecolors=EXTRING, linewidths=1.6, zorder=3)
    ax.set_xlabel(f"Component {k} loadings/score")
    ax.set_ylabel(f"{view} in {ct} (CPM)" if view != "aggregate" else f"top-50 gene aggregate in {ct}\nweighted mean z")
    sns.despine(ax=ax); fig.tight_layout(); save(fig, folder, fname, LOG)
    r, p, n = spearman(y, a); m = np.ones(len(a), bool); m[ext] = False
    LOG.append(f"{fname}: n {n}; rho {r:+.3f} (p {p:.2g}); without most extreme donor {ss.spearmanr(y[m], a[m])[0]:+.3f}")

# ---------------------------------------------------------------- Fig 4i, 4j
scatter_view(172, "LRP8", "Fig4i_C172_LRP8_Inh_vs_score", "Fig4")
scatter_view(277, "PLCB1", "Fig4j_C277_PLCB1_Exc_vs_score", "Fig4")

# ---------------------------------------------------------------- S7 a-c, g-i scatters
for letter, view in zip("abc", ["LRP8", "SYN3", "aggregate"]):
    scatter_view(172, view, f"S7{letter}_C172_{view}_Inh_vs_score", "FigS7", size=(7, 5.5))
for letter, view in zip("ghi", ["PLCB1", "LRP1B", "aggregate"]):
    scatter_view(277, view, f"S7{letter}_C277_{view}_Exc_vs_score", "FigS7", size=(7, 5.5))

# ---------------------------------------------------------------- S7 d-f (CERAD), j-l (diagnosis)
def boxes(ax, groups, labels, colours, ylab):
    data = [g for g in groups]
    bp = ax.boxplot(data, positions=range(len(data)), widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#333333", lw=1.4), whiskerprops=dict(color="#333333"), capprops=dict(color="#333333"),
                    boxprops=dict(edgecolor="#333333"))
    for patch, col in zip(bp["boxes"], colours): patch.set_facecolor(col)
    rng = np.random.default_rng(5)
    for i, g in enumerate(data):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g, s=12, color="black", alpha=0.85, linewidths=0, zorder=3)
    ax.set_xticks(range(len(data))); ax.set_xticklabels(labels, fontsize=14); ax.set_ylabel(ylab); sns.despine(ax=ax)

for k, letters in ((172, "def"), (277, "jkl")):
    c = V[k]; idx = c["donors_idx"]; cl = clin.iloc[idx]; ct = c["ct"]
    for letter, view in zip(letters, list(c["views"])):
        y = np.asarray(c["views"][view], float)
        ylab = f"{view} in {ct} (CPM)" if view != "aggregate" else f"top-50 gene aggregate in {ct}\nweighted mean z"
        if k == 172:
            ad = cl.ceradsc.values <= 2
            g = [y[~ad], y[ad]]
            fig, ax = plt.subplots(figsize=(6, 6))
            boxes(ax, g, [f"non-AD\n(n = {int((~ad).sum())})", f"AD\n(n = {int(ad.sum())})"], [SET2[0], SET2[1]], ylab)
            ax.set_xlabel("CERAD score")
            LOG.append(f"S7{letter} C172 {view}: non-AD n {int((~ad).sum())}, AD n {int(ad.sum())}")
        else:
            order = [(1, "NCI"), (2, "MCI"), (4, "AD\ndementia"), (3, "MCI +\nother"), (5, "AD +\nother"), (6, "other\ndementia")]
            cg = cl.cogdx.values
            g = [y[cg == v] for v, _ in order]
            fig, ax = plt.subplots(figsize=(9, 6))
            boxes(ax, g, [f"{lab}\n(n = {int((cg == v).sum())})" for v, lab in order], [DXCOL[1], DXCOL[2], DXCOL[4]] + [GREYB] * 3, ylab)
            ax.axvline(2.5, color="grey", lw=0.8, ls=":")
            ax.set_xlabel("final clinical diagnosis")
            LOG.append(f"S7{letter} C277 {view}: " + ", ".join(f"{lab.replace(chr(10), ' ')} n {int((cg == v).sum())}" for v, lab in order))
        fig.tight_layout(); save(fig, "FigS7", f"S7{letter}_C{k}_{view}_{ct}_by_{'CERAD' if k == 172 else 'diagnosis'}", LOG)
Pt = pd.read_csv(f"{RA}/neuronal_c21/out/n05_source_vs_phenotype.csv")
for _, r in Pt.iterrows():
    LOG.append(f"S7 phenotype (n05, verified VNC v04): C{r.component} {r.view} {r.term} n {r.n}: {r.beta_SD:+.2f} SD [{r.lo:+.2f}, {r.hi:+.2f}], p {r.p:.2g}")

# ---------------------------------------------------------------- S6a,b joint models
J = pd.read_csv(f"{RA}/neuronal_c21/out/n03_joint_models.csv")
J = J[J.set == "cogdx124"]
Vp = pd.read_csv(f"{RA}/verify_neuronal_c21/out/v01_polr.csv"); Vp = Vp[Vp.set == "cogdx124"]
TERMS = [("cerad_ad", "CERAD AD vs non-AD", "CERAD", "cerad_ad"), ("braak", "Braak stage (per stage)", "Braak", "braak"),
         ("cogdx_mci", "MCI vs NCI", "cogdx", "mci"), ("cogdx_ad", "AD dementia vs NCI", "cogdx", "addem")]
for k, letter, col in ((172, "a", CTCOL["Inh"]), (277, "b", CTCOL["Exc"])):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (t, lab, alone, vt) in enumerate(TERMS):
        yb = len(TERMS) - 1 - i
        for joint, off in ((False, 0.15), (True, -0.15)):
            mdl = "CERAD+Braak+cogdx" if joint else alone
            r = J[(J.component == k) & (J.model == mdl) & (J.term == t)].iloc[0]
            v = Vp[(Vp.component == k) & (Vp.model == mdl) & (Vp.term == vt)].iloc[0]
            assert abs(r.OR - v.OR) < 0.005 and abs(r.OR_lo - v.OR_lo) < 0.005 and abs(r.OR_hi - v.OR_hi) < 0.01, (k, mdl, t)
            ax.errorbar(r.OR, yb + off, xerr=[[r.OR - r.OR_lo], [r.OR_hi - r.OR]], fmt="o", color=col, mfc=col if joint else "white",
                        mec=col, ms=9, lw=1.8, capsize=0, label=("all three in one model" if joint else "each phenotype alone") if i == 0 else None)
            LOG.append(f"S6{letter} C{k} {lab} {'joint' if joint else 'alone'} (n {r.n}): OR {r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]")
    ax.axvline(1, color="k", lw=0.8, ls="--"); ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2]); ax.set_xticklabels(["0.2", "0.5", "1", "2"]); ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(0.15, 2.5)
    ax.set_yticks(range(len(TERMS))); ax.set_yticklabels([t[1] for t in TERMS][::-1])
    ax.set_xlabel("odds ratio for a higher score"); ax.legend(frameon=False, fontsize=13, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    sns.despine(ax=ax); fig.tight_layout(); save(fig, "FigS6", f"S6{letter}_C{k}_odds_ratios_alone_vs_joint", LOG)

# ---------------------------------------------------------------- S6c,d composition scatters
F = pd.read_csv(f"{RA}/verify_neuronal_c21/cache/frame_comp.csv")       # rows in md_samples order (checked via the scores)
assert len(F) == 276 and np.allclose(F.s277.values, score(277)) and np.allclose(F.s172.values, score(172))
assert (F.rna_mismatch.values.astype(bool) == excl_mask("RNA")).all()
own = F[F.rna_mismatch == 0]
for letter, x, xl in (("c", "p_neu", "neuronal fraction of nuclei (Exc + Inh)"), ("d", "l23", "layer 2–3 CBLN2 share of Exc nuclei")):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for g, lab in ((1, "NCI"), (2, "MCI"), (4, "AD dementia")):
        m = own.cogdx == g; ax.scatter(own.loc[m, x], own.loc[m, "s277"], s=26, color=DXCOL[g], alpha=0.85, linewidths=0, label=lab)
    m = ~own.cogdx.isin([1, 2, 4]); ax.scatter(own.loc[m, x], own.loc[m, "s277"], s=26, color=GREYB, alpha=0.85, linewidths=0, label="other diagnosis")
    ax.set_xlabel(xl); ax.set_ylabel("Component 277 loadings/score"); sns.despine(ax=ax)
    if letter == "c": ax.legend(frameon=False, fontsize=13, loc="lower right", markerscale=1.3)
    fig.tight_layout(); save(fig, "FigS6", f"S6{letter}_C277_score_vs_{'neuronal_fraction' if x == 'p_neu' else 'L2-3_CBLN2_share'}", LOG)
    r, p, n = spearman(own[x].values, own.s277.values); r6, _, n6 = spearman(F[x].values, F.s277.values)
    LOG.append(f"S6{letter} C277 vs {x}: n {n}; rho {r:+.3f} (all 276: {r6:+.3f})")

# ---------------------------------------------------------------- S6e total vs direct (n 254)
T = pd.read_csv(f"{RA}/verify_neuronal_c21/out/v03_total_direct.csv")
T = T[(T.component == 277) & (T["sample"] == "cogdx124")].set_index("comp")
assert (T.n == 254).all()
LAB = {"NEU": "+ neuronal fraction", "C6": "+ six cell classes", "C6+L23": "+ six classes, layer 2–3 CBLN2 share",
       "NEU+L23": "+ neuronal fraction, layer 2–3 CBLN2 share", "SUB5": "+ six classes, five Exc-subtype components",
       "C6+allExcCLR": "+ six classes, all 14 Exc subtypes"}
order = ["NEU", "C6", "C6+L23", "NEU+L23", "SUB5", "C6+allExcCLR"]
for variant in ("", "_alt_with_shares"):
    fig, ax = plt.subplots(figsize=(10 if variant else 8.5, 6.5))
    r0 = T.iloc[0]; col = CTCOL["Exc"]
    ax.errorbar(r0.total, len(order), xerr=[[r0.total - r0.total_lo], [r0.total_hi - r0.total]], fmt="o", color=col, mfc=col, ms=10, lw=1.8, capsize=0)
    for i, c in enumerate(order):
        r = T.loc[c]; y = len(order) - 1 - i
        ax.errorbar(r.direct, y, xerr=[[r.direct - r.direct_lo], [r.direct_hi - r.direct]], fmt="o", color=col, mfc="white", ms=10, lw=1.8, capsize=0)
        if variant:
            ax.text(0.08, y, f"{r.share:.0%} [{r.share_lo:.0%}, {r.share_hi:.0%}]", fontsize=13, va="center")
    if variant:
        ax.text(0.08, len(order), "reduction on\nadjustment", fontsize=13, va="center")
        ax.set_xlim(-1.0, 0.65)
    else:
        ax.set_xlim(-1.0, 0.1)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(order) + 1)); ax.set_yticklabels([LAB[c] for c in order][::-1] + ["without composition"])
    ax.set_xlabel("AD dementia vs NCI\n(SD of rank-inverse-normal score)")
    sns.despine(ax=ax); fig.tight_layout(); save(fig, "FigS6", f"S6e_C277_dementia_effect_total_vs_direct{variant}", LOG)
LOG.append(f"S6e total {r0.total:+.2f} [{r0.total_lo:+.2f}, {r0.total_hi:+.2f}] (n {int(r0.n)}); " +
           "; ".join(f"{c}: direct {T.loc[c, 'direct']:+.2f} [{T.loc[c, 'direct_lo']:+.2f}, {T.loc[c, 'direct_hi']:+.2f}], share {T.loc[c, 'share']:.1%} [{T.loc[c, 'share_lo']:.1%}, {T.loc[c, 'share_hi']:.1%}]" for c in order))
open(f"{OUT}/p05_neuronal.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
