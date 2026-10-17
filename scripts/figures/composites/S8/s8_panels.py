"""Figure S8 (Component 351) panels a-f redrawn at their final printed size (v23 supplement, 2026-10-16).

a  sample scores, broken y axis, coloured by sex          v22 351_samples.pdf        (ros3b_plot_outliers.py)
b  cell-type loadings (corrected)                         v23 351_cell_type.pdf      (ros3b, corrected input)
c  snRNA-seq gene loadings by chromosome (PIP > 0.5)       v22 351_RNA.pdf            (ros3b)
d  CpG loadings by chromosome (PIP > 0.5)                  v22 351_meth.pdf           (ros3b)
e  H3K9ac peak loadings by chromosome (PIP > 0.5)          v22 351_chip.pdf           (ros3b)
f  IGKC by cell type and top-50 Mic aggregate vs score     v23 S8f (panels/p06_c21_single.py)

a-e: data, colours, label sets and axis titles as ros3b_plot_outliers.py (5-Aug snapshot copy, read only), rebuilt by
s8_data.py; a check (not included) showed the rebuilt points land exactly (0.000 pt) on the v22 / corrected markers. Changes
here are presentation only: figure size (= the composite cell), explicit type sizes (6-7 pt Arial), marker sizes and
line widths scaled to the new size, and the gene / CpG / peak labels re-positioned by a deterministic placer with
leader lines (ros3b used adjustText, which is not in the locked environment). a carries the "loadings/score" y title of
the v22 Fig 2f composite (kept). The legend in a sits at the top right as in v22, in one row.
f: p06's S8f code (top50_agg, gene, two_panel -> broken_x_scatter) with the same data and exclusions (n = 274), at cell
size; the cell-type key sits just outside the right edge of the IGKC plot (inside it would cover points at this
size), the gap at the broken x axis is 0.12 instead of 0.07, and the aggregate's y title keeps p06's words on three lines.

Each panel is saved at exactly its cell size (no bbox_inches='tight'); axes are placed in millimetres so that the
left edges of a, c, d, e and f (left plot) and the right edges of b, c, d, e and f (right plot) line up.
Writes <G_ROS_FIGURE_OUT>/S8/panels/*.pdf and a log there (aggregates only; no donor IDs, no per-donor values).
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \\
     python s8_panels.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
from vp_common import *          # noqa: F401,F403  (loaders, CTCOL, CT_ORDER, robust_z, ...; sets v22 talk style)
import vp_common as VP
import s8_data as D
from scipy import stats as ss
from matplotlib.lines import Line2D

PAN = f"{FIGOUT}/S8/panels"
os.makedirs(PAN, exist_ok=True)
MM = 1 / 25.4
PT_PER_MM = 72 / 25.4

# ------------------------------------------------------------------ style: v22 seaborn 'white', sizes at print size
mpl.rcParams["pdf.fonttype"] = 42
sns.set(style="white", context="paper")
TICK, LAB, ANN, LEG = 6.5, 7.0, 6.0, 6.5
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "mathtext.default": "regular",
    "font.size": TICK, "axes.labelsize": LAB, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "legend.fontsize": LEG, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 1.6, "ytick.major.pad": 1.6, "axes.labelpad": 2.0, "lines.linewidth": 0.8,
    "legend.handletextpad": 0.3, "legend.columnspacing": 0.9, "legend.borderpad": 0.3,
    "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
    "savefig.facecolor": "white", "figure.facecolor": "white", "figure.dpi": 72,
})
LOG = []

# ------------------------------------------------------------------ geometry (mm)
W = 142.9
AXL = 12.0                    # common left axes edge (a, c, d, e, f-left)
AXR = 141.4                   # common right axes edge (b, c, d, e, f-right)
TOP = 2.5                     # cell top -> axes top (the panel letter sits in the cell's top-left corner)
ROWS = dict(ab=27.0, c=30.8, d=24.0, e=30.8, f=29.5)   # total 142.1 mm (S8 budget 142.3)
A_W = 80.0; B_X = 82.0        # a: x 0..80; b: x 82..142.9
GEOM = dict(W=W, AXL=AXL, AXR=AXR, TOP=TOP, ROWS=ROWS, A_W=A_W, B_X=B_X)


def cell(w_mm, h_mm):
    return plt.figure(figsize=(w_mm * MM, h_mm * MM), dpi=72)


def add_ax(fig, w_mm, h_mm, l_mm, top_mm, aw_mm, ah_mm, **kw):
    return fig.add_axes([l_mm / w_mm, 1 - (top_mm + ah_mm) / h_mm, aw_mm / w_mm, ah_mm / h_mm], **kw)


def savep(fig, name):
    fig.savefig(f"{PAN}/{name}.pdf")
    plt.close(fig)
    LOG.append(f"wrote panels/{name}.pdf")


# label placement: label_placer.place (replaces adjustText, which ros3b used; not installed here)
import label_placer


# ================================================================== a: sample scores (ros3b broken y axis)
def panel_a():
    w, h = A_W, ROWS["ab"]
    x, y, sex = D.samples()
    q1, q3 = np.quantile(y, [0.25, 0.75]); iqr = q3 - q1          # pandas .quantile (linear) as ros3b
    lo, hi = q1 - 30 * iqr, q3 + 30 * iqr
    out = (y < lo) | (y > hi); nrm = ~out
    nmin, nmax = y[nrm].min(), y[nrm].max(); nb = max((nmax - nmin) * 0.05, 0.02)
    omin, omax = y[out].min(), y[out].max()
    ob = max(abs(omin) * 0.05, 0.01) if omin == omax else max((omax - omin) * 0.05, 0.01)
    ylims = sorted([(nmin - nb, nmax + nb), (omin - ob, omax + ob)], key=lambda t: t[0])
    lower, upper = ylims                                           # brokenaxes: higher range on top
    fig = cell(w, h)
    aw = A_W - AXL - 1.0
    ah = h - TOP - 4.6
    gap = 1.6                                                      # brokenaxes hspace 0.05 would be 0.5 mm: tick labels would touch
    hu = (ah - gap) * (upper[1] - upper[0]) / ((upper[1] - upper[0]) + (lower[1] - lower[0]))
    hl = ah - gap - hu
    axu = add_ax(fig, w, h, AXL, TOP, aw, hu)
    axl = add_ax(fig, w, h, AXL, TOP + hu + gap, aw, hl)
    cmap = {"Male": sns.color_palette("muted")[0], "Female": sns.color_palette("muted")[2]}
    s = 0.6 * 81 * ((aw * PT_PER_MM) / 446.4) ** 2                 # v22 marker area scaled with the axes width (x 0.6:
                                                                   # the y range is 7 x more compressed than in v22)
    xl = (x.min() - 0.05 * np.ptp(x), x.max() + 0.05 * np.ptp(x))
    for ax, yl in ((axu, upper), (axl, lower)):
        # ros3b: bax.scatter(normal) then bax.scatter(outliers); brokenaxes draws both in every sub-axis
        for sel in (nrm, out):
            ax.scatter(x[sel], y[sel], c=[cmap[v] for v in sex[sel]], edgecolors="k", alpha=0.8, linewidth=0.0, s=s)
        ax.set_ylim(*yl); ax.set_xlim(*xl); ax.set_xticks([])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3, steps=[1, 2, 5, 10]))
        ax.tick_params(axis="y", left=True, length=2.0, width=0.6, color=".15")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axu.spines["bottom"].set_visible(False)
    # brokenaxes break marks (d = 0.015 of the axes, on the left spine)
    d = 0.015
    kw = dict(color="k", clip_on=False, lw=0.6)
    ru = hu / ah; rl = hl / ah
    axu.plot((-d, +d), (-d / ru * 1.0, +d / ru * 1.0), transform=axu.transAxes, **kw)
    axl.plot((-d, +d), (1 - d / rl, 1 + d / rl), transform=axl.transAxes, **kw)
    axl.set_xlabel("Samples")
    fig.text(1.2 / w, 1 - (TOP + ah / 2) / h, "loadings/score", rotation=90, ha="left", va="center",
             fontsize=LAB, color=".15")
    hnd = [Line2D([], [], color=cmap["Male"], marker="o", ls="None", markersize=3.6, label="Male"),
           Line2D([], [], color=cmap["Female"], marker="o", ls="None", markersize=3.6, label="Female")]
    axu.legend(handles=hnd, loc="upper right", bbox_to_anchor=(1.0, 1.06), ncol=2, frameon=True, fontsize=LEG,
               handletextpad=0.1, columnspacing=0.8, borderpad=0.25)
    savep(fig, "S8a_C351_samples")
    LOG.append(f"a: n {len(y)} ({int((sex == 'Female').sum())} female); donors beyond 30 x IQR: {int(out.sum())}; "
               f"marker s {s:.1f}")


# ================================================================== b: corrected cell-type loadings (ros3b bar)
def panel_b():
    w, h = W - B_X, ROWS["ab"]
    order, vals = D.celltype()
    fig = cell(w, h)
    l = 10.6; aw = (AXR - B_X) - l; ah = h - TOP - 4.6
    ax = add_ax(fig, w, h, l, TOP, aw, ah)
    pos = np.arange(len(order))
    ax.bar(pos, vals, align="center", color=[CTCOL[c] for c in order], edgecolor="white",
           linewidth=1.5 * (aw * PT_PER_MM) / 553.11)
    ax.set_xticks(pos); ax.set_xticklabels(order)
    ax.set_ylabel("loadings/score")
    ax.tick_params(length=0)
    savep(fig, "S8b_C351_cell_type")
    LOG.append("b: corrected cell-type loadings " + ", ".join(f"{c} {v:.3f}" for c, v in zip(order, vals)))


# ================================================================== c-e: loadings by chromosome (ros3b)
XLAB = {1: "Genes by Chromosome", 2: "CpG Sites by Chromosome", 3: "ChIP Peaks by Chromosome"}


def panel_manhattan(j, name, h, step=1.2, w_pts=6.0):
    d = D.manhattan(j)
    fig = cell(W, h)
    aw = AXR - AXL; ah = h - TOP - 7.0
    ax = add_ax(fig, W, h, AXL, TOP, aw, ah)
    scale = (aw * PT_PER_MM) / 750.75
    ms = 0.75 * 9 * scale                                          # talk markersize 9, scaled (x 0.75: y range compressed)
    # same draw order and colours as ros3b: one plot() per chromosome; empty chromosomes get an invisible spacer
    pts = np.array([(p[0], p[1]) for p in d["pts"]])
    # per-chromosome groups exactly as in s8_data.manhattan (same loop as ros3b)
    L = D.pip_features(j); info = D.annotation(j, L.index)
    shift = np.array([0.0])
    for ii, ch in enumerate(D.CHRS):
        sub = info[info["chrom"] == ch]
        col = D.COLORS[j][ii % 4]
        if len(sub) == 0:
            xx = shift[-1] + np.array([D.CHR_MAX[ch]]); ax.plot(xx, [0], ".", color=col, markersize=0)
        else:
            xx = shift[-1] + sub["pos"].values.astype(float); yy = L.loc[sub["fid"].values].values
            ax.plot(xx, yy, ".", color=col, markersize=ms)
        shift = np.append(shift, np.max(xx))
    ax.set_xticks(d["ticks"]); ax.set_xticklabels(d["chrs"], rotation=90)
    ax.set_ylim(*d["ylim"])
    ax.tick_params(length=0)
    ax.set_xlabel(XLAB[j]); ax.set_ylabel("loadings/score")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5 if ah > 15 else 4, steps=[1, 2, 5, 10]))
    # labels: ros3b label set and colours (black = largest, gray = smallest [+ IGHG1, CD68, HLA-DRB1 for C351])
    labs = d["labels"]
    order = list(np.argsort([-abs(l[1]) for l in labs]))
    labs = [labs[i] for i in order]
    res, tot = label_placer.place(fig, ax, pts, labs, fontsize=ANN, marker_r_pt=0.25 * ms + 0.2, step=step,
                                  w_pts=w_pts, K=600)
    savep(fig, name)
    LOG.append(f"{name}: {d['n_pip']} features PIP > 0.5, {d['n_points']} plotted, {len(labs)} labels; "
               f"marker size {ms:.2f}; longest leader {max(r['leader_pt'] for r in res):.1f} pt; "
               f"label overlaps {sum(r['overlaps_label'] for r in res)}; placement cost {tot:.0f}")


# ================================================================== f: p06 S8f at cell size
RNA = rna_input()
exR = excl_mask("RNA"); keepR = ~exR
CT = {351: "Mic"}


def top50_agg(k, ct):                                   # p06_c21_single.py, verbatim
    Lk = comp_loadings(k, "RNA"); X = Lk.X.values; Pp = Lk.P.values
    idx = np.where(Pp > 0.5)[0]; idx = idx[np.argsort(-np.abs(X[idx]))][:50]
    M = np.asarray(RNA[CTS_BLOCK.index(ct)][:, idx], float)
    w = X[idx] * bcor(k)[ct]
    ag, n = zagg(M[keepR], w)
    out = np.full(276, np.nan); out[keepR] = ag
    return out, n


def gene(k, g, ct):                                     # p06_c21_single.py, verbatim
    j = list(comp_loadings(k, "RNA").index).index(g)
    y = np.asarray(RNA[CTS_BLOCK.index(ct), :, j], float).copy(); y[exR] = np.nan
    return y


def broken_x_scatter_mm(fig, w, h, l_mm, top_mm, aw_mm, ah_mm, a, series, xlabel, ylabel, ring=True, ylog=None,
                        legend=None, ring_lw=0.6, sym_tick=None):
    """vp_common.broken_x_scatter with the axes placed in mm and the hard-coded sizes made explicit."""
    out = outliers_30iqr(a)
    ext = most_extreme(a)
    assert out.any()
    neg = a[out].mean() < np.median(a)
    wsp = 0.12                                            # p06 subgridspec wspace 0.07; 0.12 keeps the break marks
                                                          # clear of the first bulk tick label at this size
    ratios = [1, 4] if neg else [4, 1]
    unit = aw_mm / (sum(ratios) + wsp * len(ratios) / 2 * (len(ratios) - 1))
    gap = wsp * unit * sum(ratios) / len(ratios)
    w0 = unit * ratios[0]; w1 = unit * ratios[1]
    ax0 = add_ax(fig, w, h, l_mm, top_mm, w0, ah_mm)
    ax1 = add_ax(fig, w, h, l_mm + w0 + gap, top_mm, w1, ah_mm, sharey=ax0)
    axo, axb = (ax0, ax1) if neg else (ax1, ax0)
    axes = [axo, axb] if neg else [axb, axo]
    draw = [axb, axo]
    for (y, col, sz, lab) in series:
        ok = ~np.isnan(y)
        for ax in draw:
            ax.scatter(a[ok], y[ok], color=col, s=sz, alpha=0.8, linewidths=0, label=lab if ax is axb else None, zorder=2)
        if ring and not np.isnan(y[ext]):
            for ax in draw:
                ax.scatter([a[ext]], [y[ext]], s=sz * 3.2, facecolors="none", edgecolors=EXTRING, linewidths=ring_lw,
                           zorder=3)
    for ax in axes:
        sns.despine(ax=ax)
        ax.tick_params(length=0)
        if ylog == "symlog":
            ax.set_yscale("symlog", linthresh=1); ax.set_ylim(bottom=-0.3)
    bulk = a[~out]; pad = (bulk.max() - bulk.min()) * 0.05
    axb.set_xlim(bulk.min() - pad, bulk.max() + pad)
    axb.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    o = a[out]; span = max(np.abs(o).max() * 0.04, 1e-3)
    axo.set_xlim(o.min() - span, o.max() + span)
    axo.set_xticks([float(f"{o.mean():.2f}")])
    left, right = (axo, axb) if neg else (axb, axo)
    left.spines["right"].set_visible(False); right.spines["left"].set_visible(False)
    right.tick_params(labelleft=False, left=False)
    dd = 0.015; kw = dict(color="k", clip_on=False, linewidth=0.6)
    # vp_common uses the same fraction of each sub-axis; keep the marks the same physical size on both sides
    wl = (w0 if left is ax0 else w1); wr = (w1 if right is ax1 else w0)
    base = dd * (w0 + w1) / 2 * 1.6
    dl, dr = base / wl, base / wr
    dyv = 2 * dd * 1.6 * (w0 + w1) / 2 / ah_mm * 0.5
    right.plot((-dr, +dr), (-dyv, dyv), transform=right.transAxes, **kw)
    left.plot((1 - dl, 1 + dl), (-dyv, dyv), transform=left.transAxes, **kw)
    axb.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    if sym_tick is not None:
        axes[0].tick_params(axis="y", labelsize=sym_tick)
    if legend:
        # p06: inside, upper right; here just outside the right edge (inside it would cover points at this size)
        axb.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, markerscale=1.3, fontsize=LEG,
                   ncol=1, handletextpad=0.05, borderaxespad=0.3, labelspacing=0.35)
    return axes


def panel_f():
    w, h = W, ROWS["f"]
    k = 351
    a = score(k)
    sz1 = 3.2; sz2 = 4.6                                   # p06: 18 (cell-type series) and 26 (aggregate), scaled
    series = [(gene(k, "IGKC", ct), CTCOL[ct], sz1, ct) for ct in CT_ORDER]
    agg, n = top50_agg(k, "Mic")
    fig = cell(w, h)
    ah = h - TOP - 7.0
    lw_ = 62.5 - AXL                                       # the cell-type key sits outside, right of this plot
    r_l = 89.0
    AXR_F = W - 5.2                                        # room for the last x tick label (0.016)
    # the symmetric-log tick labels carry 10^n superscripts drawn at 0.7 x the tick size: 7.2 pt keeps them >= 5 pt
    broken_x_scatter_mm(fig, w, h, AXL, TOP, lw_, ah, a, series, f"Component {k} loadings/score", "IGKC (CPM)",
                        ylog="symlog", legend="upper right", sym_tick=7.2)
    # p06 label "top-50 gene aggregate in Mic\nweighted mean z", same words on three lines (two do not fit 20 mm)
    broken_x_scatter_mm(fig, w, h, r_l, TOP, AXR_F - r_l, ah, a, [(agg, CTCOL[CT[k]], sz2, None)],
                        f"Component {k} loadings/score", "top-50 gene\naggregate in Mic\nweighted mean z")
    savep(fig, "S8f_C351_IGKC_by_celltype_and_Mic_aggregate")
    # aggregates for the record (as p06 log_view)
    for ct in CT_ORDER:
        yv = gene(k, "IGKC", ct); r, p, nn = spearman(yv, a)
        LOG.append(f"f: IGKC {ct}: n {nn}; Spearman {r:+.3f} (p {p:.2g})")
    r, p, nn = spearman(agg, a); LOG.append(f"f: Mic top-50 aggregate: n {nn}, genes {n}; Spearman {r:+.3f}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "c", "d", "e", "f"]
    if "a" in which: panel_a()
    if "b" in which: panel_b()
    if "c" in which: panel_manhattan(1, "S8c_C351_RNA", ROWS["c"])
    if "d" in which: panel_manhattan(2, "S8d_C351_meth", ROWS["d"])
    if "e" in which: panel_manhattan(3, "S8e_C351_chip", ROWS["e"])
    if "f" in which: panel_f()
    open(f"{PAN}/s8_panels_{'_'.join(which)}.log", "w").write("\n".join(LOG) + "\n" + json.dumps(GEOM) + "\n")
    print("\n".join(LOG))
