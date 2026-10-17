"""Figure S2 (covariate-adjusted associations and effect estimates; component-level aggregates only).
 a-c  published (unadjusted MW / KW) BH FDR vs covariate-adjusted (proportional-odds LRT, M1) BH FDR over 547 components,
      for sex, CERAD (AD vs non-AD) and final clinical diagnosis (NCI / MCI / AD dementia); published calls in blue
      Source: RA/covariates/po_tests_547.csv (CV 03_po_adjusted.py; verified VCV 06: 547 x 3 p agree to 1e-8)
 d    rank-inverse-normal effects (SD, OLS, Blom offset) with 95% CI for the five central associations, M0 / M1 / M2
      Source: verify_dossier/out/v02_effects_{74,278,545,172,277}.csv, columns r38_* (= CD 09_effect_table.log)
 d_alt (optional) the same with proportional-odds odds ratios (C74 omitted: near-separation)
 e    C278 female vs male across models: PO odds ratio (left) and RINT effect (right)
      Source: RA/revision_analyses/c74_c278/out/30_c278_effects_forest.csv (verify_OR / verify_SD columns agree)"""
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from vp_common import *

LOG = []
po = pd.read_csv(f"{R2}/covariates/po_tests_547.csv", header=[0, 1], index_col=0)
BLUE = "#0072B2"; GREY = "#BBBBBB"
CAP = 4.0
PHN = {"Sex": "sex", "ceradsc": "CERAD", "cogdx": "diagnosis"}
CENT = {"Sex": [74, 278, 545], "ceradsc": [172], "cogdx": [277]}
OFFS = {"Sex": {74: (-60, -40), 278: (-50, 18), 545: (-50, 18)}, "ceradsc": {172: (-60, 20)}, "cogdx": {277: (-60, 20)}}
for letter, ph in zip("abc", ["Sex", "ceradsc", "cogdx"]):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    xq = po[(ph, "U_q")]; yq = po[(ph, "PO1_q")]
    x = np.minimum(-np.log10(xq), CAP); y = np.minimum(-np.log10(yq), CAP)
    pub = po[(ph, "published_sig")].astype(str) == "True"
    ax.scatter(x[~pub], y[~pub], s=22, color=GREY, lw=0, zorder=1, label="other components")
    ax.scatter(x[pub], y[pub], s=34, color=BLUE, edgecolor="white", lw=0.4, zorder=2, label="published call")
    t = -np.log10(0.05)
    ax.axvline(t, color="k", lw=0.9, ls="--"); ax.axhline(t, color="k", lw=0.9, ls="--")
    for k in CENT[ph]:
        clipped = (-np.log10(xq[k]) > CAP) or (-np.log10(yq[k]) > CAP)
        ax.annotate(f"C{k}" + (" (off scale)" if clipped else ""), (x[k], y[k]), xytext=OFFS[ph][k], textcoords="offset points",
                    fontsize=14, ha="center", arrowprops=dict(arrowstyle="-", color="k", lw=0.6, shrinkA=0, shrinkB=3))
    lim = CAP * 1.06; ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("published FDR (−log$_{10}$)"); ax.set_ylabel("adjusted FDR (−log$_{10}$)")
    if letter == "a":
        ax.legend(frameon=False, fontsize=13, loc="lower right", markerscale=1.2)
    sns.despine(ax=ax); fig.tight_layout(); save(fig, "FigS2", f"S2{letter}_published_vs_adjusted_FDR_{PHN[ph]}", LOG)
    kept = int((pub & (yq < 0.05)).sum())
    same_dir = "n/a"
    LOG.append(f"S2{letter} {PHN[ph]}: published calls {int(pub.sum())}; keep adjusted FDR < 0.05: {kept}; "
               + ", ".join(f"C{k} q {xq[k]:.3g} -> {yq[k]:.3g}" for k in CENT[ph]))

# ---------------------------------------------------------------- d  RINT effects, five central associations
ROWS = [(74, "female", "C74, female vs male"), (278, "female", "C278, female vs male"), (545, "female", "C545, female vs male"),
        (172, "ad", "C172, CERAD AD vs non-AD"), (277, "adem", "C277, AD dementia vs NCI")]
MODELS = [("M0", "no covariates", dict(marker="o", mfc="white", mec="k", color="k")),
          ("M1", "main covariates", dict(marker="o", mfc="k", mec="k", color="k")),
          ("M2", "main covariates + quality proxies", dict(marker="s", mfc="#777777", mec="#777777", color="#777777"))]
E = {k: pd.read_csv(f"{R2}/verify_dossier/out/v02_effects_{k}.csv") for k, _, _ in ROWS}
fig, ax = plt.subplots(figsize=(9, 6))
off = {"M0": 0.22, "M1": 0.0, "M2": -0.22}
for i, (k, term, lab) in enumerate(ROWS):
    yb = len(ROWS) - 1 - i
    for m, mlab, kw in MODELS:
        r = E[k][(E[k].model == m) & (E[k].term == term)].iloc[0]
        ax.errorbar(r.r38_b, yb + off[m], xerr=[[r.r38_b - r.r38_lo], [r.r38_hi - r.r38_b]], fmt=kw["marker"], ms=8,
                    mfc=kw["mfc"], mec=kw["mec"], color=kw["color"], lw=1.6, capsize=0, label=mlab if i == 0 else None)
        LOG.append(f"S2d C{k} {term} {m} (n {int(r.n)}): {r.r38_b:+.2f} [{r.r38_lo:+.2f}, {r.r38_hi:+.2f}] SD")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([lab for _, _, lab in ROWS][::-1])
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlabel("difference in rank-inverse-normal score (SD)")
ax.legend(frameon=False, fontsize=13, loc="center left", bbox_to_anchor=(1.02, 0.5))
sns.despine(ax=ax); fig.tight_layout(); save(fig, "FigS2", "S2d_rint_effects_central_five", LOG)

# optional variant: proportional-odds odds ratios next to the RINT effects (C74 omitted from the OR side)
fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True, gridspec_kw=dict(wspace=0.08))
for i, (k, term, lab) in enumerate(ROWS):
    yb = len(ROWS) - 1 - i
    for m, mlab, kw in MODELS:
        r = E[k][(E[k].model == m) & (E[k].term == term)].iloc[0]
        lo = r.OR_prof_lo if pd.notna(r.OR_prof_lo) else r.OR_wald_lo
        hi = r.OR_prof_hi if pd.notna(r.OR_prof_hi) else r.OR_wald_hi
        if k != 74:
            axs[0].errorbar(r.OR, yb + off[m], xerr=[[r.OR - lo], [hi - r.OR]], fmt=kw["marker"], ms=8, mfc=kw["mfc"], mec=kw["mec"],
                            color=kw["color"], lw=1.6, capsize=0, label=mlab if i == 1 else None)
            LOG.append(f"S2d_alt C{k} {m}: OR {r.OR:.2f} [{lo:.2f}, {hi:.2f}] ({'profile' if pd.notna(r.OR_prof_lo) else 'Wald'})")
        axs[1].errorbar(r.r38_b, yb + off[m], xerr=[[r.r38_b - r.r38_lo], [r.r38_hi - r.r38_b]], fmt=kw["marker"], ms=8,
                        mfc=kw["mfc"], mec=kw["mec"], color=kw["color"], lw=1.6, capsize=0)
axs[0].set_xscale("log"); axs[0].axvline(1, color="k", lw=0.8, ls="--")
axs[0].set_xticks([0.2, 0.5, 1, 2, 5]); axs[0].set_xticklabels(["0.2", "0.5", "1", "2", "5"]); axs[0].xaxis.set_minor_locator(mpl.ticker.NullLocator())
axs[0].text(1, len(ROWS) - 1, "not estimable\n(near-separation)", ha="center", va="center", fontsize=12, color="#555555", bbox=dict(fc="white", ec="none", pad=1))
axs[0].set_xlabel("proportional-odds odds ratio"); axs[1].set_xlabel("difference in rank-inverse-normal score (SD)")
axs[1].axvline(0, color="k", lw=0.8, ls="--")
axs[0].set_yticks(range(len(ROWS))); axs[0].set_yticklabels([lab for _, _, lab in ROWS][::-1]); axs[0].set_ylim(-0.6, len(ROWS) - 0.4)
axs[1].legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=13, loc="center left", bbox_to_anchor=(1.02, 0.5))
for ax in axs: sns.despine(ax=ax)
fig.tight_layout(); save(fig, "FigS2", "S2d_alt_OR_and_rint_effects_central_five", LOG)

# ---------------------------------------------------------------- e  C278 across models
Fo = pd.read_csv(f"{RA}/c74_c278/out/30_c278_effects_forest.csv").set_index("model")
assert np.allclose(Fo.OR, Fo.verify_OR, atol=0.006) and np.allclose(Fo.SD, Fo.verify_SD, atol=0.006)
LAB = {"M0": "no covariates", "M1": "main covariates", "M2": "main covariates + quality proxies",
       "M3": "alternative covariate coding", "drop1-M1": "main covariates, most extreme donor removed",
       "drop1-M2": "with quality proxies, most extreme donor removed", "M1+CERAD+cogdx6": "main covariates + CERAD + diagnosis",
       "M1+composition": "main covariates + cell-type composition"}
order = list(LAB)
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True, gridspec_kw=dict(wspace=0.08))
for i, m in enumerate(order):
    r = Fo.loc[m]; yb = len(order) - 1 - i
    axs[0].errorbar(r.OR, yb, xerr=[[r.OR - r.OR_lo], [r.OR_hi - r.OR]], fmt="o", color=FEMALE, mfc=FEMALE, ms=9, lw=1.8, capsize=0)
    axs[1].errorbar(r.SD, yb, xerr=[[r.SD - r.SD_lo], [r.SD_hi - r.SD]], fmt="o", color=FEMALE, mfc=FEMALE, ms=9, lw=1.8, capsize=0)
    LOG.append(f"S2e {m} (n {int(r.n)}): OR {r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]; RINT {r.SD:+.2f} [{r.SD_lo:+.2f}, {r.SD_hi:+.2f}]; "
               f"q_PO {r.q_PO_547}; q_RINT {r.q_rint_547}")
axs[0].set_xscale("log"); axs[0].axvline(1, color="k", lw=0.8, ls="--")
axs[0].set_xticks([1, 1.5, 2, 3, 4]); axs[0].set_xticklabels(["1", "1.5", "2", "3", "4"]); axs[0].xaxis.set_minor_locator(mpl.ticker.NullLocator())
axs[0].set_xlim(0.9, 4.5)
axs[1].axvline(0, color="k", lw=0.8, ls="--"); axs[1].set_xlim(-0.1, 0.9)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([LAB[m] + (" (n = 275)" if m.startswith("drop1") else "") for m in order][::-1])
axs[0].set_xlabel("odds ratio, female vs male"); axs[1].set_xlabel("female − male, rank-inverse-normal score (SD)")
for ax in axs: sns.despine(ax=ax)
fig.tight_layout(); save(fig, "FigS2", "S2e_C278_effect_across_models", LOG)
LOG.append(f"S2e OR range {Fo.OR.min():.2f}-{Fo.OR.max():.2f}; RINT q range {Fo.q_rint_547.min():.3f}-{Fo.q_rint_547.max():.3f}; PO q range {Fo.q_PO_547.min():.4f}-{Fo.q_PO_547.max():.4f}")
open(f"{OUT}/p02_covariates.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
