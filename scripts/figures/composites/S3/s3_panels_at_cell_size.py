"""Figure S3 panels redrawn at their final printed size (v23 supplement, composite built with composites/compose.py).

Same data and the same plotting code as panels/p02_covariates.py (panels S3d and S3e there), with:
  - figsize = the composite cell size (no bbox_inches="tight"), axes placed in mm;
  - explicit print sizes: 7 pt Arial text, 0.6 pt spines, markers/lines scaled to print;
  - nothing else changed: data, models, colours (black/white/#777777; seaborn muted green = FEMALE), marker
    shapes, row order, tick positions/limits that p02 fixes, label texts, legend texts.
  The one layout-only change: the right-hand x label of panel b is broken onto two lines (same words), because
  at 7 pt it is wider (51.8 mm) than its axis.

  a  (was S3d) rint effects (SD, OLS) with 95% CI, five central associations, models M0 / M1 / M2
     Source: verify_dossier/out/v02_effects_{74,278,545,172,277}.csv, columns r38_*
  b  (was S3e) C278 female vs male across models: PO odds ratio (left) and RINT effect (right)
     Source: revision_analyses/c74_c278/out/30_c278_effects_forest.csv

Writes only under <G_ROS_FIGURE_OUT>/S3/panels/. Logs aggregates only (no donor IDs, no per-donor values)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
from vp_common import *          # style(): seaborn white/talk, Arial, pdf.fonttype 42; R2, RA, FEMALE ...
import numpy as np, pandas as pd

HERE = f"{FIGOUT}/S3"
PAN = f"{HERE}/panels"; os.makedirs(PAN, exist_ok=True)
MM = 1 / 25.4
FS = 7.0                          # all text, pt
W = 142.9                         # supplement \textwidth, mm
GEOM = {"a": dict(w=W, h=58.0), "b": dict(w=W, h=69.5)}

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 4,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 3.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
})
LOG = []


def ax_mm(fig, wmm, hmm, left, top, width, height, **kw):
    """Axes placed by its left/top edge and size in mm (origin top-left of the figure)."""
    return fig.add_axes([left / wmm, 1 - (top + height) / hmm, width / wmm, height / hmm], **kw)


def check_inside(fig, name):
    """Everything drawn (tick labels, axis labels, legend) must lie inside the figure (nothing clipped by the page).
    fig.get_tightbbox counts only the artists that are drawn (ticks outside the view limits are not)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(r); w, h = fig.get_size_inches()
    ext = [round(v * 25.4, 2) for v in (tb.x0, tb.y0, tb.x1, tb.y1)]
    ok = tb.x0 >= -0.005 and tb.y0 >= -0.005 and tb.x1 <= w + 0.005 and tb.y1 <= h + 0.005
    LOG.append(f"{name}: drawn extent (mm, from bottom-left) {ext} within {round(w * 25.4, 2)} x {round(h * 25.4, 2)}: {ok}")
    assert ok, ext


def save_cell(fig, name):
    check_inside(fig, name)
    fig.savefig(f"{PAN}/{name}.pdf")          # exact cell size: no bbox_inches
    plt.close(fig); LOG.append(f"wrote panels/{name}.pdf")


# ---------------------------------------------------------------- a  RINT effects, five central associations (p02 S3d)
ROWS = [(74, "female", "C74, female vs male"), (278, "female", "C278, female vs male"), (545, "female", "C545, female vs male"),
        (172, "ad", "C172, CERAD AD vs non-AD"), (277, "adem", "C277, AD dementia vs NCI")]
MODELS = [("M0", "no covariates", dict(marker="o", mfc="white", mec="k", color="k")),
          ("M1", "main covariates", dict(marker="o", mfc="k", mec="k", color="k")),
          ("M2", "main covariates + quality proxies", dict(marker="s", mfc="#777777", mec="#777777", color="#777777"))]
E = {k: pd.read_csv(f"{R2}/verify_dossier/out/v02_effects_{k}.csv") for k, _, _ in ROWS}
g = GEOM["a"]
fig = plt.figure(figsize=(g["w"] * MM, g["h"] * MM))
ax = ax_mm(fig, g["w"], g["h"], left=35.0, top=2.5, width=57.0, height=g["h"] - 2.5 - 8.5)
off = {"M0": 0.22, "M1": 0.0, "M2": -0.22}
for i, (k, term, lab) in enumerate(ROWS):
    yb = len(ROWS) - 1 - i
    for m, mlab, kw in MODELS:
        r = E[k][(E[k].model == m) & (E[k].term == term)].iloc[0]
        ax.errorbar(r.r38_b, yb + off[m], xerr=[[r.r38_b - r.r38_lo], [r.r38_hi - r.r38_b]], fmt=kw["marker"], ms=3.6,
                    mfc=kw["mfc"], mec=kw["mec"], mew=0.7, color=kw["color"], lw=0.9, capsize=0,
                    label=mlab if i == 0 else None)
        LOG.append(f"a C{k} {term} {m} (n {int(r.n)}): {r.r38_b:+.2f} [{r.r38_lo:+.2f}, {r.r38_hi:+.2f}] SD")
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([lab for _, _, lab in ROWS][::-1])
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlabel("difference in rank-inverse-normal score (SD)")
ax.legend(frameon=False, fontsize=FS, loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=2.0,
          handletextpad=0.6, labelspacing=0.7, borderaxespad=0.0)
sns.despine(ax=ax)
save_cell(fig, "S3a_rint_effects_central_five_cell")

# ---------------------------------------------------------------- b  C278 across models (p02 S3e)
Fo = pd.read_csv(f"{RA}/c74_c278/out/30_c278_effects_forest.csv").set_index("model")
assert np.allclose(Fo.OR, Fo.verify_OR, atol=0.006) and np.allclose(Fo.SD, Fo.verify_SD, atol=0.006)
LAB = {"M0": "no covariates", "M1": "main covariates", "M2": "main covariates + quality proxies",
       "M3": "alternative covariate coding", "drop1-M1": "main covariates, most extreme donor removed",
       "drop1-M2": "with quality proxies, most extreme donor removed", "M1+CERAD+cogdx6": "main covariates + CERAD + diagnosis",
       "M1+composition": "main covariates + cell-type composition"}
order = list(LAB)
g = GEOM["b"]
fig = plt.figure(figsize=(g["w"] * MM, g["h"] * MM))
AXW, GAP, LEFT, TOP, BOTTOM = 33.5, 4.5, 69.0, 2.5, 11.5
a0 = ax_mm(fig, g["w"], g["h"], LEFT, TOP, AXW, g["h"] - TOP - BOTTOM)
a1 = ax_mm(fig, g["w"], g["h"], LEFT + AXW + GAP, TOP, AXW, g["h"] - TOP - BOTTOM, sharey=a0)
axs = [a0, a1]
for i, m in enumerate(order):
    r = Fo.loc[m]; yb = len(order) - 1 - i
    axs[0].errorbar(r.OR, yb, xerr=[[r.OR - r.OR_lo], [r.OR_hi - r.OR]], fmt="o", color=FEMALE, mfc=FEMALE, ms=3.6, lw=0.9, capsize=0)
    axs[1].errorbar(r.SD, yb, xerr=[[r.SD - r.SD_lo], [r.SD_hi - r.SD]], fmt="o", color=FEMALE, mfc=FEMALE, ms=3.6, lw=0.9, capsize=0)
    LOG.append(f"b {m} (n {int(r.n)}): OR {r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]; RINT {r.SD:+.2f} [{r.SD_lo:+.2f}, {r.SD_hi:+.2f}]; "
               f"q_PO {r.q_PO_547}; q_RINT {r.q_rint_547}")
axs[0].set_xscale("log"); axs[0].axvline(1, color="k", lw=0.5, ls="--")
axs[0].set_xticks([1, 1.5, 2, 3, 4]); axs[0].set_xticklabels(["1", "1.5", "2", "3", "4"]); axs[0].xaxis.set_minor_locator(mpl.ticker.NullLocator())
axs[0].set_xlim(0.9, 4.5)
axs[1].axvline(0, color="k", lw=0.5, ls="--"); axs[1].set_xlim(-0.1, 0.9)
# p02 left these ticks to the locator, which gave 0.0-0.8 by 0.2 at its size; pinned so the printed panel shows the same ticks
axs[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8]); axs[1].set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8"])
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([LAB[m] + (" (n = 275)" if m.startswith("drop1") else "") for m in order][::-1])
axs[1].tick_params(labelleft=False)              # sharey=True in p02 hides the right-hand y tick labels
axs[0].set_xlabel("odds ratio, female vs male"); axs[1].set_xlabel("female − male,\nrank-inverse-normal score (SD)")
for ax in axs: sns.despine(ax=ax)
save_cell(fig, "S3b_C278_effect_across_models_cell")
LOG.append(f"b OR range {Fo.OR.min():.2f}-{Fo.OR.max():.2f}; RINT q range {Fo.q_rint_547.min():.3f}-{Fo.q_rint_547.max():.3f}; "
           f"PO q range {Fo.q_PO_547.min():.4f}-{Fo.q_PO_547.max():.4f}")
open(f"{PAN}/s3_panels_at_cell_size.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
