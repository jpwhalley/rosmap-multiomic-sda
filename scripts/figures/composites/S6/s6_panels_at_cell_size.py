"""Figure S6 panels redrawn at their final printed size (v23 supplement; composite built with composites/compose.py).

Same data and the same plotting code as panels/p05_neuronal.py (sections "S6a,b joint models",
"S6c,d composition scatters", "S6e total vs direct"), with:
  - figsize = the composite cell size (no bbox_inches="tight"), axes placed in mm;
  - explicit print sizes: 7 pt Arial text, 0.6 pt spines, markers/lines scaled to print;
  - nothing else changed: data, models, assertions against the verification tables, colours (CTCOL Inh / Exc for
    a, b, e; Set2 diagnosis colours and grey for c, d), open/filled marker coding, row order and offsets, axis
    scales, tick positions and limits that p05 fixes, label texts and in-panel key texts.
  Layout-only changes (same words): the a/b key is placed in the strip above the axes as in p05 but anchored
  where it fits the narrower cell; the c key is moved from inside the axes (lower right) to the strip above
  the axes when, at print size, it would cover donors (checked below and logged).

  a  C172 odds ratios, each phenotype alone (open) and all three in one model (filled); n 254
  b  the same for C277
     Source: revision_analyses/neuronal_c21/out/n03_joint_models.csv (asserted equal to verify v01_polr.csv)
  c  C277 score vs neuronal fraction of nuclei; d  vs layer 2-3 CBLN2 share of Exc nuclei; n 274, by diagnosis
     Source: revision_analyses/verify_neuronal_c21/cache/frame_comp.csv (RNA-mismatched profiles excluded)
  e  AD dementia vs NCI effect on the RINT C277 score, without and with each composition set (n 254)
     Source: revision_analyses/verify_neuronal_c21/out/v03_total_direct.csv

Writes only under <G_ROS_FIGURE_OUT>/S6/panels/. Logs aggregates only (no donor IDs, no per-donor values)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
from vp_common import *          # style(): seaborn white/talk, Arial, pdf.fonttype 42; R2, RA, CTCOL, score, excl_mask ...
import numpy as np, pandas as pd

HERE = f"{FIGOUT}/S6"
PAN = f"{HERE}/panels"; os.makedirs(PAN, exist_ok=True)
MM = 1 / 25.4
FS = 7.0                          # all text, pt
W = 142.9                         # supplement \textwidth, mm
GUT = 2.0                         # gutter between the two columns, mm
CW = (W - GUT) / 2                # column cell width, mm
GEOM = {"ab": dict(w=CW, h=42.0), "cd": dict(w=CW, h=54.0), "e": dict(w=W, h=48.0)}

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 4,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 3.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
})
LOG = []
SET2 = sns.color_palette("Set2"); GREYB = "#9e9e9e"           # as p05
DXCOL = {1: SET2[0], 2: SET2[2], 4: SET2[1]}


def ax_mm(fig, wmm, hmm, left, top, width, height, **kw):
    """Axes placed by its left/top edge and size in mm (origin top-left of the figure)."""
    return fig.add_axes([left / wmm, 1 - (top + height) / hmm, width / wmm, height / hmm], **kw)


def bbox_mm(fig, artist):
    """Drawn extent of an artist in mm from the top-left of the figure: (x0, y0_top, x1, y1_bottom)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bb = artist.get_window_extent(r); h = fig.get_size_inches()[1] * fig.dpi
    s = 25.4 / fig.dpi
    return (bb.x0 * s, (h - bb.y1) * s, bb.x1 * s, (h - bb.y0) * s)


def check_inside(fig, name, keep_clear_mm=(4.0, 3.0)):
    """Everything drawn must lie inside the figure, and the top-left corner (letter box, w x h mm) must be empty."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(r); w, h = fig.get_size_inches()
    ext = [round(v * 25.4, 2) for v in (tb.x0, tb.y0, tb.x1, tb.y1)]
    ok = tb.x0 >= -0.005 and tb.y0 >= -0.005 and tb.x1 <= w + 0.005 and tb.y1 <= h + 0.005
    LOG.append(f"{name}: drawn extent (mm, from bottom-left) {ext} within {round(w * 25.4, 2)} x {round(h * 25.4, 2)}: {ok}")
    assert ok, ext
    # letter corner: no text artist may enter the box [0, kw] x [0, kh] (mm from top-left)
    kw, kh = keep_clear_mm
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        x0, y0, x1, y1 = bbox_mm(fig, t)
        assert not (x0 < kw and y0 < kh), (name, t.get_text(), (x0, y0))
    for ax in fig.axes:
        x0, y0, x1, y1 = bbox_mm(fig, ax)
        assert not (x0 < kw and y0 < kh), (name, "axes in letter corner")


def save_cell(fig, name):
    check_inside(fig, name)
    fig.savefig(f"{PAN}/{name}.pdf")          # exact cell size: no bbox_inches
    plt.close(fig); LOG.append(f"wrote panels/{name}.pdf")


# ---------------------------------------------------------------- a, b  joint models (p05 S6a,b)
J = pd.read_csv(f"{RA}/neuronal_c21/out/n03_joint_models.csv")
J = J[J.set == "cogdx124"]
Vp = pd.read_csv(f"{RA}/verify_neuronal_c21/out/v01_polr.csv"); Vp = Vp[Vp.set == "cogdx124"]
TERMS = [("cerad_ad", "CERAD AD vs non-AD", "CERAD", "cerad_ad"), ("braak", "Braak stage (per stage)", "Braak", "braak"),
         ("cogdx_mci", "MCI vs NCI", "cogdx", "mci"), ("cogdx_ad", "AD dementia vs NCI", "cogdx", "addem")]
g = GEOM["ab"]
AB_LEFT, AB_TOP, AB_BOTTOM, AB_RIGHT = 29.0, 7.0, 8.5, 2.0
for k, letter, col in ((172, "a", CTCOL["Inh"]), (277, "b", CTCOL["Exc"])):
    fig = plt.figure(figsize=(g["w"] * MM, g["h"] * MM))
    ax = ax_mm(fig, g["w"], g["h"], AB_LEFT, AB_TOP, g["w"] - AB_LEFT - AB_RIGHT, g["h"] - AB_TOP - AB_BOTTOM)
    for i, (t, lab, alone, vt) in enumerate(TERMS):
        yb = len(TERMS) - 1 - i
        for joint, off in ((False, 0.15), (True, -0.15)):
            mdl = "CERAD+Braak+cogdx" if joint else alone
            r = J[(J.component == k) & (J.model == mdl) & (J.term == t)].iloc[0]
            v = Vp[(Vp.component == k) & (Vp.model == mdl) & (Vp.term == vt)].iloc[0]
            assert abs(r.OR - v.OR) < 0.005 and abs(r.OR_lo - v.OR_lo) < 0.005 and abs(r.OR_hi - v.OR_hi) < 0.01, (k, mdl, t)
            assert int(r.n) == 254
            ax.errorbar(r.OR, yb + off, xerr=[[r.OR - r.OR_lo], [r.OR_hi - r.OR]], fmt="o", color=col, mfc=col if joint else "white",
                        mec=col, ms=3.6, mew=0.7, lw=0.9, capsize=0,
                        label=("all three in one model" if joint else "each phenotype alone") if i == 0 else None)
            LOG.append(f"{letter} C{k} {lab} {'joint' if joint else 'alone'} (n {r.n}): OR {r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]")
    ax.axvline(1, color="k", lw=0.5, ls="--"); ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2]); ax.set_xticklabels(["0.2", "0.5", "1", "2"]); ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(0.15, 2.5)
    ax.set_yticks(range(len(TERMS))); ax.set_yticklabels([t[1] for t in TERMS][::-1])
    ax.set_xlabel("odds ratio for a higher score")
    # p05: key above the axes, two columns, anchored at the axes' left edge. At print size the two columns are wider
    # than the axes, so the key is anchored at the axes' right edge instead (same two columns, same texts).
    leg = fig.legend(frameon=False, fontsize=FS, loc="upper right", ncol=2,
                     bbox_to_anchor=(1 - AB_RIGHT / g["w"], 1 - 0.6 / g["h"]), borderaxespad=0.0, borderpad=0.0,
                     handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    sns.despine(ax=ax)
    lb = bbox_mm(fig, leg)
    LOG.append(f"{letter}: key extent x {lb[0]:.1f}-{lb[2]:.1f} mm, y {lb[1]:.1f}-{lb[3]:.1f} mm (cell {g['w']:.2f} x {g['h']:.1f})")
    save_cell(fig, f"S6{letter}_C{k}_odds_ratios_alone_vs_joint_cell")

# ---------------------------------------------------------------- c, d  composition scatters (p05 S6c,d)
F = pd.read_csv(f"{RA}/verify_neuronal_c21/cache/frame_comp.csv")       # rows in md_samples order (checked via the scores)
assert len(F) == 276 and np.allclose(F.s277.values, score(277)) and np.allclose(F.s172.values, score(172))
assert (F.rna_mismatch.values.astype(bool) == excl_mask("RNA")).all()
own = F[F.rna_mismatch == 0]
g = GEOM["cd"]
CD_LEFT, CD_TOP, CD_BOTTOM, CD_RIGHT = 12.5, 6.5, 8.5, 2.0
MS = 5.0                                                                   # scatter size, pt^2 (p05: 26 at talk size)
for letter, x, xl in (("c", "p_neu", "neuronal fraction of nuclei (Exc + Inh)"), ("d", "l23", "layer 2–3 CBLN2 share of Exc nuclei")):
    fig = plt.figure(figsize=(g["w"] * MM, g["h"] * MM))
    ax = ax_mm(fig, g["w"], g["h"], CD_LEFT, CD_TOP, g["w"] - CD_LEFT - CD_RIGHT, g["h"] - CD_TOP - CD_BOTTOM)
    for gg, lab in ((1, "NCI"), (2, "MCI"), (4, "AD dementia")):
        m = own.cogdx == gg; ax.scatter(own.loc[m, x], own.loc[m, "s277"], s=MS, color=DXCOL[gg], alpha=0.85, linewidths=0, label=lab)
    m = ~own.cogdx.isin([1, 2, 4]); ax.scatter(own.loc[m, x], own.loc[m, "s277"], s=MS, color=GREYB, alpha=0.85, linewidths=0, label="other diagnosis")
    ax.set_xlabel(xl); ax.set_ylabel("Component 277 loadings/score"); sns.despine(ax=ax)
    if letter == "c":
        # p05: legend(frameon=False, fontsize=13, loc="lower right", markerscale=1.3). Kept there unless it covers donors.
        leg = ax.legend(frameon=False, fontsize=FS, loc="lower right", markerscale=1.3, handletextpad=0.2,
                        labelspacing=0.3, borderaxespad=0.3, borderpad=0.2)
        fig.canvas.draw(); r_ = fig.canvas.get_renderer()
        bb = leg.get_window_extent(r_)
        pts = ax.transData.transform(np.c_[own[x].values, own.s277.values])
        under = int(((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum())
        LOG.append(f"c: key inside lower right would cover {under} donor points")
        if under > 0:
            # moved to the strip above the axes (one row, right-aligned to the axes' right edge): covers no donors
            leg.remove()
            leg = fig.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
                             frameon=False, fontsize=FS, loc="upper right", ncol=4, markerscale=1.3,
                             bbox_to_anchor=(1 - CD_RIGHT / g["w"], 1 - 0.6 / g["h"]), borderaxespad=0.0, borderpad=0.0,
                             handlelength=1.0, handletextpad=0.2, columnspacing=1.0)
            lb = bbox_mm(fig, leg)
            LOG.append(f"c: key moved above the axes: x {lb[0]:.1f}-{lb[2]:.1f} mm, y {lb[1]:.1f}-{lb[3]:.1f} mm; axes top {CD_TOP} mm")
            assert lb[3] < CD_TOP and lb[0] > 4.0
    save_cell(fig, f"S6{letter}_C277_score_vs_{'neuronal_fraction' if x == 'p_neu' else 'L2-3_CBLN2_share'}_cell")
    r, p, n = spearman(own[x].values, own.s277.values)
    assert n == 274
    LOG.append(f"{letter} C277 vs {x}: n {n}; rho {r:+.3f} (p {p:.2g})")

# ---------------------------------------------------------------- e  total vs direct (p05 S6e, main variant, not _alt)
T = pd.read_csv(f"{RA}/verify_neuronal_c21/out/v03_total_direct.csv")
T = T[(T.component == 277) & (T["sample"] == "cogdx124")].set_index("comp")
assert (T.n == 254).all()
LAB = {"NEU": "+ neuronal fraction", "C6": "+ six cell classes", "C6+L23": "+ six classes, layer 2–3 CBLN2 share",
       "NEU+L23": "+ neuronal fraction, layer 2–3 CBLN2 share", "SUB5": "+ six classes, five Exc-subtype components",
       "C6+allExcCLR": "+ six classes, all 14 Exc subtypes"}
order = ["NEU", "C6", "C6+L23", "NEU+L23", "SUB5", "C6+allExcCLR"]
g = GEOM["e"]
E_LEFT, E_TOP, E_BOTTOM, E_RIGHT = 56.0, 3.0, 12.0, 8.0
fig = plt.figure(figsize=(g["w"] * MM, g["h"] * MM))
ax = ax_mm(fig, g["w"], g["h"], E_LEFT, E_TOP, g["w"] - E_LEFT - E_RIGHT, g["h"] - E_TOP - E_BOTTOM)
r0 = T.iloc[0]; col = CTCOL["Exc"]
ax.errorbar(r0.total, len(order), xerr=[[r0.total - r0.total_lo], [r0.total_hi - r0.total]], fmt="o", color=col, mfc=col,
            ms=4.0, mew=0.7, lw=0.9, capsize=0)
for i, c in enumerate(order):
    r = T.loc[c]; y = len(order) - 1 - i
    ax.errorbar(r.direct, y, xerr=[[r.direct - r.direct_lo], [r.direct_hi - r.direct]], fmt="o", color=col, mfc="white",
                ms=4.0, mew=0.7, lw=0.9, capsize=0)
ax.set_xlim(-1.0, 0.1)
# p05 left these ticks to the locator, which gave -1.0, -0.5, 0.0 at its size; pinned so the printed panel shows the same ticks
ax.set_xticks([-1.0, -0.5, 0.0]); ax.set_xticklabels(["−1.0", "−0.5", "0.0"])
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_yticks(range(len(order) + 1)); ax.set_yticklabels([LAB[c] for c in order][::-1] + ["without composition"])
ax.set_xlabel("AD dementia vs NCI\n(SD of rank-inverse-normal score)")
sns.despine(ax=ax)
save_cell(fig, "S6e_C277_dementia_effect_total_vs_direct_cell")
LOG.append(f"e total {r0.total:+.2f} [{r0.total_lo:+.2f}, {r0.total_hi:+.2f}] (n {int(r0.n)}); " +
           "; ".join(f"{c}: direct {T.loc[c, 'direct']:+.2f} [{T.loc[c, 'direct_lo']:+.2f}, {T.loc[c, 'direct_hi']:+.2f}], share {T.loc[c, 'share']:.1%} [{T.loc[c, 'share_lo']:.1%}, {T.loc[c, 'share_hi']:.1%}]" for c in order))
open(f"{PAN}/s6_panels_at_cell_size.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
