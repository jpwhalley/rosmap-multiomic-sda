"""Figure S10 (Component 87), all six panels redrawn at their final printed size for the python composite
(v23 supplement; assembled and lettered by composites/compose.py). Each PDF is exactly its
composite cell and is placed at scale 1 in the 142.9 mm wide figure, so the sizes set here are the printed sizes.

Sources and recipes (all read only):
  a  87_samples.pdf (v22; outlier/87 in the version-1 panels) -> ros3b_plot_outliers.py l.226-317 (version-1 copy of
     20 Oct 2025): donors ordered by age at death ('90+' as 90), coloured by recorded sex (muted
     blue = Male, muted green = Female), broken y-axis (brokenaxes: two ranges, normal donors and the 30 x IQR outlier,
     each padded as l.245-252, heights proportional to the ranges, hspace 0.05, top/right spines off, '//' marks),
     no x ticks, x label 'Samples', framed Male/Female legend top right.
     ADDED (S10a): the y label 'loadings/score' (the original panel has none).
  b  corrected 87_cell_type.pdf (v23 panels: g_ros/figs/outlier/87/) -> ros3b l.326-340:
     bars of ROSMAP_cell_type_panels_values.csv cor_* (C87) in the order Ast Exc Inh Mic OPC Oli with
     the ros3b colours, y label 'loadings/score', full box.
  c-e 87_RNA.pdf, 87_meth.pdf, 87_chip.pdf (v22) -> ros3b l.319-607: PIP > 0.5 features at their genomic position
     (chromosomes laid end to end as ros3b, colours cycled per chromosome), the 8 largest (black) and 8 smallest
     (grey) labelled when > 16 features, else positive black / non-positive grey, the ros3b y limits, chromosome
     tick labels rotated 90 degrees, x labels as ros3b, full box.
  f  v23 S10f_C87_VTCN1_Ast_and_aggregate.pdf -> panels/p06_c21_single.py (S10f) with
     vp_common.broken_x_scatter: VTCN1 in Ast (CPM) and the top-50 Ast aggregate against the C87 score, n 274
     (snRNA-seq RNA-1/RNA-2 excluded), broken x-axis (30 x IQR outlier on the left), dominant donor ringed.

Print-size settings (only presentation changes; same words, colours, order):
  - explicit Arial sizes as the S3/S6 builds: tick labels 6 pt, axis labels and in-panel text 6.5 pt; axes lines
    0.6 pt; markers, rings and leader lines scaled with the panel;
  - c-e: adjustText (used by ros3b) is not in the locked environment, so the 16/2/2 feature labels
    are placed by a small deterministic search (nearest free position around each point, leader line when moved);
    the labelled features, their text and colours are those of ros3b;
  - f: the aggregate y label 'top-50 gene aggregate in Ast / weighted mean z' is broken into three lines
    ('top-50 gene / aggregate in Ast / weighted mean z') so it fits the axes height (as S6).
Checks: every point/bar/label of c-e and the bars of b are compared with the original PDFs (pdf_marks.py); the f
statistics are compared with the legend (n 274, rho -0.05, p 0.38). Logs aggregates only: no donor IDs, no per-donor
values. Writes only under <G_ROS_FIGURE_OUT>/S10/."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vp_common import *          # style(): seaborn white/talk, Arial, pdf.fonttype 42; score, rna_input, CTCOL, ...
import numpy as np, pandas as pd
import matplotlib.lines as mlines
from scipy import stats as ss
import s10_data as D
import pdf_marks as PM

from figure_paths import V1_PANELS, V23_PANELS
HERE = f"{FIGOUT}/S10"
PAN = f"{HERE}/panels"; os.makedirs(PAN, exist_ok=True)
CODING = f"{V1_PANELS}/outlier/87"
PNEW = V23_PANELS
MM = 1 / 25.4
TICK, LAB = 6.0, 6.5
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "font.size": LAB, "axes.labelsize": LAB, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
                     "legend.fontsize": TICK, "axes.linewidth": 0.6, "lines.linewidth": 0.8, "patch.linewidth": 0.3,
                     "xtick.major.pad": 1.8, "ytick.major.pad": 1.5, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.labelpad": 2.0,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

# ---------------------------------------------------------------- geometry (mm); the layout JSON uses the same numbers
W = 142.9
GUT_X = 2.5
HALF = (W - GUT_X) / 2
TOP = 3.0                         # strip above the axes for the panel letter
RIGHT = 1.5
GEOM = {  # name: (cell w, cell h, left, bottom, right) ; axes top = TOP
    "a": (HALF, 35.0, 11.0, 4.0, 1.5),
    "b": (HALF, 35.0, 11.0, 4.0, 1.5),
    "c": (W, 47.0, 11.0, 7.0, 1.5),
    "d": (W, 25.0, 11.0, 7.0, 1.5),
    "e": (W, 25.0, 11.0, 7.0, 1.5),
    "f": (W, 34.0, None, 5.8, 1.5),
}
LOG = []


def new_cell(name):
    w, h, left, bottom, right = GEOM[name]
    fig = plt.figure(figsize=(w * MM, h * MM), dpi=72)
    return fig, w, h, left, bottom, right


def add_axes_mm(fig, w, h, left, bottom, right, top=TOP):
    return fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - top - bottom) / h])


def bbox_mm(fig, artist):
    r = fig.canvas.get_renderer()
    bb = artist.get_window_extent(r); h = fig.get_size_inches()[1] * fig.dpi; s = 25.4 / fig.dpi
    return (bb.x0 * s, (h - bb.y1) * s, bb.x1 * s, (h - bb.y0) * s)


def drawn_ticklabels(axis):
    lo, hi = sorted(axis.get_view_interval())
    eps = (hi - lo) * 1e-9
    return [t for loc, t in zip(axis.get_majorticklocs(), axis.get_ticklabels())
            if lo - eps <= loc <= hi + eps and t.get_visible() and t.get_text().strip()]


def overlap(a_, b_, pad=0.0):
    return a_[0] < b_[2] + pad and b_[0] < a_[2] + pad and a_[1] < b_[3] + pad and b_[1] < a_[3] + pad


def check_cell(fig, name, keep_clear_mm=(3.0, 2.8), extra_texts=()):
    """Everything inside the cell; nothing in the letter corner; no overlapping tick labels; axis labels clear of
    tick labels; in-panel labels clear of each other and of the tick labels."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(r); w, h = fig.get_size_inches()
    ext = [round(v * 25.4, 2) for v in (tb.x0, tb.y0, tb.x1, tb.y1)]
    ok = tb.x0 >= -0.005 and tb.y0 >= -0.005 and tb.x1 <= w + 0.005 and tb.y1 <= h + 0.005
    LOG.append(f"{name}: drawn extent (mm, from bottom-left) {ext} within {w * 25.4:.2f} x {h * 25.4:.2f}: {ok}")
    assert ok, (name, ext)
    kw, kh = keep_clear_mm
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        x0, y0, x1, y1 = bbox_mm(fig, t)
        assert not (x0 < kw and y0 < kh), (name, t.get_text(), (x0, y0))
    ticks_all = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            bbs = [bbox_mm(fig, t) for t in drawn_ticklabels(axis)]
            ticks_all += bbs
            for i in range(len(bbs)):
                for j in range(i + 1, len(bbs)):
                    assert not overlap(bbs[i], bbs[j]), (name, "tick labels overlap", i, j)
        if ax.yaxis.label.get_text():
            yl = bbox_mm(fig, ax.yaxis.label)
            for t in drawn_ticklabels(ax.yaxis):
                assert yl[2] <= bbox_mm(fig, t)[0] + 0.01, (name, "y label overlaps tick labels", t.get_text())
        if ax.xaxis.label.get_text():
            xl = bbox_mm(fig, ax.xaxis.label)
            for t in drawn_ticklabels(ax.xaxis):
                assert xl[1] >= bbox_mm(fig, t)[3] - 0.01, (name, "x label overlaps tick labels", t.get_text())
    eb = [bbox_mm(fig, t) for t in extra_texts]
    for i in range(len(eb)):
        for j in range(i + 1, len(eb)):
            assert not overlap(eb[i], eb[j]), (name, "in-panel labels overlap", extra_texts[i].get_text(),
                                               extra_texts[j].get_text())
        for tbb in ticks_all:
            assert not overlap(eb[i], tbb), (name, "in-panel label overlaps a tick label", extra_texts[i].get_text())


def save_cell(fig, name, **kw):
    check_cell(fig, name, **kw)
    fig.savefig(f"{PAN}/{name}.pdf")          # exact cell size: no bbox_inches
    plt.close(fig); LOG.append(f"wrote panels/{name}.pdf")


# ================================================================ a: sample scores (ros3b l.226-317)
def panel_a():
    x, y, sex, out = D.samples()
    assert len(y) == 276 and out.sum() == 1
    normal = ~out
    nmin, nmax = y[normal].min(), y[normal].max()
    nbuf = max((nmax - nmin) * 0.05, 0.02)
    nyl = (nmin - nbuf, nmax + nbuf)
    omin, omax = y[out].min(), y[out].max()
    obuf = max(abs(omin) * 0.05, 0.01) if omin == omax else max((omax - omin) * 0.05, 0.01)
    oyl = (omin - obuf, omax + obuf)
    ylims = sorted([nyl, oyl], key=lambda v: v[0])                      # ascending; brokenaxes draws the highest on top
    fig, w, h, left, bottom, right = new_cell("a")
    axh = h - TOP - bottom
    gap = 0.05 * axh / (2 + 0.05)                                         # hspace 0.05 of the mean axes height
    spans = [hi - lo for lo, hi in ylims[::-1]]                           # top first
    hs = [(axh - gap) * s / sum(spans) for s in spans]
    axes = []
    y_top = h - TOP
    for i, (lo, hi) in enumerate(ylims[::-1]):
        b = y_top - hs[i]
        ax = fig.add_axes([left / w, b / h, (w - left - right) / w, hs[i] / h])
        ax.set_ylim(lo, hi)
        axes.append(ax)
        y_top = b - gap
    cmap = {'Male': sns.color_palette('muted')[0], 'Female': sns.color_palette('muted')[2]}
    cols = np.array([cmap[s] for s in sex])
    S = 9.0                                                               # rc s 81 (talk) x (70/203)^2
    for ax in axes:
        for m in (normal, out):                                           # normal data, then outliers (ros3b order)
            ax.scatter(x[m], y[m], c=cols[m], edgecolors='k', alpha=0.8, linewidth=0.0, s=S)
        ax.set_xlim(-0.05 * (len(y) - 1) - 0.0, (len(y) - 1) * 1.05)
        ax.set_xticks([]); ax.tick_params(axis="y", pad=2.8)       # tick labels clear of the break marks
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].spines["bottom"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    # brokenaxes diagonal marks on the left spine (d = 0.015 of the axes, tilt 45 deg), scaled to the panel
    d = 0.45                                                             # mm, half-length of a mark
    kw = dict(color="k", clip_on=False, linewidth=0.6)
    for ax, ydat in ((axes[0], 0.0), (axes[1], 1.0)):
        trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
        hmm = ax.get_position().height * h; wmm = ax.get_position().width * w
        ax.plot((-d / wmm, d / wmm), (ydat - d / hmm, ydat + d / hmm), transform=trans, **kw)
    big = fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - TOP - bottom) / h], frameon=False)
    big.set_xticks([]); big.set_yticks([]); big.patch.set_alpha(0)
    big.set_xlabel('Samples', labelpad=2.0)
    big.set_ylabel('loadings/score', labelpad=19.0)                       # ADDED (S10a)
    leg = [mlines.Line2D([], [], color=cmap['Male'], marker='o', linestyle='None', markersize=3.4, label='Male'),
           mlines.Line2D([], [], color=cmap['Female'], marker='o', linestyle='None', markersize=3.4, label='Female')]
    fig.legend(handles=leg, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=TICK, frameon=True,
               borderpad=0.4, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.3, handlelength=1.0)
    LOG.append(f"a: n {len(y)}; female {int((sex == 'Female').sum())}; beyond 30 x IQR {int(out.sum())}; "
               f"ylims {[tuple(round(float(v), 4) for v in t) for t in ylims]}")
    save_cell(fig, "S10a_C87_samples_cell")
    return dict(x=x, y=y, normal=normal, ylims=ylims)


# ================================================================ b: corrected cell-type bars (ros3b l.326-340)
def panel_b():
    order, vals = D.celltype()
    fig, w, h, left, bottom, right = new_cell("b")
    ax = add_axes_mm(fig, w, h, left, bottom, right)
    y_pos = np.arange(len(order))
    ax.bar(y_pos, vals, align='center', color=[D.CT_COL[c] for c in order])
    ax.set_xticks(y_pos); ax.set_xticklabels(order)
    ax.set_ylabel('loadings/score')
    LOG.append("b: " + ", ".join(f"{c} {v:+.4f}" for c, v in zip(order, vals)))
    save_cell(fig, "S10b_C87_cell_type_cell")
    return vals


# ================================================================ c-e: loadings by chromosome (ros3b l.319-607)
MS = 4.4          # '.' marker size; ros3b rc 9 (talk) x 0.469 (304.8 -> 142.9 mm)
LEAD_LW = 0.35    # adjustText arrowprops lw 0.5 x 0.469 (rounded up)
FEAT_FS = 6.0     # feature labels (ros3b: plt.text at the rc font size)


def _seg_box(p0, p1, box, pad=0.0):
    """Does the segment p0-p1 pass through the (padded) box? (Liang-Barsky clipping)"""
    x0, y0 = p0; dx, dy = p1[0] - x0, p1[1] - y0
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0 - (box[0] - pad)), (dx, (box[2] + pad) - x0), (-dy, y0 - (box[1] - pad)), (dy, (box[3] + pad) - y0)):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def _seg_seg(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _near_seg(P, a, b, r):
    """Number of points within r of the segment a-b (excluding its end points' own markers)."""
    a = np.asarray(a); b = np.asarray(b); ab = b - a; L2 = float(ab @ ab)
    if L2 == 0:
        return 0
    t = np.clip(((P - a) @ ab) / L2, 0, 1)
    proj = a + t[:, None] * ab
    dist = np.hypot(*(P - proj).T)
    return int(np.sum((dist < r) & (t > 0.02) & (t < 0.98)))


def _greedy(order, labels, P, LP, sizes, axbb, rad):
    mrad = MS / 2 * 0.5 + 0.3
    descend = [any(ch in l[2] for ch in 'gjpqy,()') for l in labels]
    dists = [1.2, 2.0, 3.0, 4, 5, 6, 7, 8.5, 10, 11, 12.5, 14, 16, 18, 20.5, 23, 26, 29, 32.5, 36, 40, 44, 49, 54, 60, 66, 73, 80, 88, 96]
    angs = np.deg2rad(np.arange(0, 360, 10.0))
    placed = {}; leaders = []; boxes = []; total = 0.0
    for i in order:
        p = LP[i]; w, h = sizes[i]
        best = None
        for dd in dists:
            for a in angs:
                cx, cy = np.cos(a), np.sin(a)
                ha = 'left' if cx > 0.38 else ('right' if cx < -0.38 else 'center')
                va = 'bottom' if cy > 0.38 else ('top' if cy < -0.38 else 'center')
                ax_ = p[0] + (dd + rad) * cx; ay_ = p[1] + (dd + rad) * cy
                x0 = ax_ if ha == 'left' else (ax_ - w if ha == 'right' else ax_ - w / 2)
                y0 = ay_ if va == 'bottom' else (ay_ - h if va == 'top' else ay_ - h / 2)
                box = (x0, y0, x0 + w, y0 + h)
                if not (box[0] >= axbb.x0 + 1 and box[2] <= axbb.x1 - 1 and box[1] >= axbb.y0 + 1
                        and box[3] <= axbb.y1 - 1):
                    continue
                if any(overlap(box, q, pad=0.8) for q in boxes):
                    continue
                # points under the glyphs: the ink box (matplotlib text box at 6 pt: 5.0 pt high with the baseline
                # 1.0 pt (20%) above its bottom; Arial caps/digits (0.716 em = 4.3 pt) reach 106% of the box height;
                # descenders reach the box bottom) plus the marker radius and 0.3 pt
                ib0 = box[1] + (0.0 if descend[i] else 0.20 * h); ib1 = box[1] + 1.06 * h
                cover = int(np.sum((P[:, 0] > box[0] - mrad) & (P[:, 0] < box[2] + mrad) &
                                   (P[:, 1] > ib0 - mrad) & (P[:, 1] < ib1 + mrad)))
                cost = dd + max(0.0, dd - 30.0) + COVER_W * cover
                q = None
                # ambiguity: another data point at least as close to the label as its own point (+2 pt) -> the
                # label gets a leader like a moved label, and a small penalty
                ddx = np.maximum(np.maximum(box[0] - P[:, 0], 0), P[:, 0] - box[2])
                ddy = np.maximum(np.maximum(box[1] - P[:, 1], 0), P[:, 1] - box[3])
                dist = np.hypot(ddx, ddy)
                own = np.hypot(max(box[0] - p[0], 0, p[0] - box[2]), max(box[1] - p[1], 0, p[1] - box[3]))
                amb = bool(np.sum(dist < own + 2.0) > 1)            # the own point is always counted once
                if amb:
                    cost += AMB_W
                if dd > LEAD_MIN or amb:
                    q = (min(max(p[0], box[0]), box[2]), min(max(p[1], box[1]), box[3]))
                    cost += LEADPT_W * _near_seg(P, p, q, rad + 0.6)
                    if any(_seg_box(p, q, bx, pad=0.8) for bx in boxes):
                        cost += 1e4
                    if any(_seg_seg(p, q, l0, l1) for l0, l1 in leaders):
                        cost += 1e4
                    # the leader must not run through another labelled point
                    for k_, lp in enumerate(LP):
                        if k_ != i and _near_seg(np.asarray([lp]), p, q, rad + 0.6):
                            cost += 1e4
                if any(_seg_box(l0, l1, box, pad=0.8) for l0, l1 in leaders):
                    cost += 1e4
                if any(overlap(box, (lp[0] - rad, lp[1] - rad, lp[0] + rad, lp[1] + rad)) for k_, lp in enumerate(LP) if k_ != i):
                    cost += 1e4                                              # never hide another labelled point
                if best is None or cost < best[0]:
                    best = (cost, ax_, ay_, ha, va, box, dd, cover, q)
            if best is not None and best[0] <= dd:
                break
        if best is None:
            return 1e9, None
        total += best[0]; placed[i] = best; boxes.append(best[5])
        if best[8] is not None:
            leaders.append((tuple(p), best[8]))
    return total, placed


LEAD_MIN = 2.6
COVER_W, LEADPT_W = 45.0, 15.0
AMB_W = 6.0


def place_labels(fig, ax, labels, pts_xy, fs=LAB):
    """Deterministic label placement (adjustText is not installed). For a label order, each label in turn takes the
    cheapest position around its point (16 directions, distances 1.2-80 pt) that stays inside the axes and does not
    overlap an earlier label; cost = distance (pt) + 6 per data point under the label + 2 per data point along the
    leader, and a prohibitive cost when the label or its leader would cross an earlier label or leader or another
    labelled point. The orders tried are black-first by |value|, then 400 seeded permutations; the cheapest total
    wins. A leader line (as adjustText's arrowprops: '-', black) joins the point and the nearest edge of the label
    when the label is not right next to its point."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    tr = ax.transData
    P = tr.transform(np.asarray(pts_xy, float))                          # display (pt, dpi 72)
    LP = [tuple(tr.transform((l[0], l[1]))) for l in labels]
    rad = MS / 2 * 0.5 + 0.4                                             # '.' circle radius + clearance, pt
    sizes = []
    for l in labels:
        t = ax.text(0, 0, l[2], fontsize=fs, ha='left', va='bottom', transform=None)
        bb = t.get_window_extent(r); t.remove(); sizes.append((bb.width, bb.height))
    base = sorted(range(len(labels)), key=lambda i: (labels[i][3] != "black", -abs(labels[i][1])))
    rng = np.random.default_rng(87)
    orders = [base, base[::-1]] + [list(rng.permutation(len(labels))) for _ in range(400 if len(labels) > 2 else 0)]
    best = None
    for o in orders:
        tot, pl = _greedy(o, labels, P, LP, sizes, axbb, rad)
        if pl is not None and (best is None or tot < best[0]):
            best = (tot, pl)
    assert best is not None and best[0] < 1e4, "no clean label placement"
    texts = []
    inv = tr.inverted()
    for i in range(len(labels)):
        x, y, txt, col = labels[i]
        cost, ax_, ay_, ha, va, box, dd, cover, q = best[1][i]
        dx, dy = inv.transform((ax_, ay_))
        t = ax.text(dx, dy, txt, fontsize=fs, color=col, ha=ha, va=va, zorder=4)
        texts.append(t)
        if q is not None:
            ex, ey = inv.transform(q)
            ax.plot([x, ex], [y, ey], color='k', lw=LEAD_LW, zorder=3, solid_capstyle='butt')
        LOG.append(f"   label {txt} ({col}): distance {dd} pt, covers {cover} points, cost {cost:.1f}")
    LOG.append(f"   placement total cost {best[0]:.1f} over {len(orders)} orders")
    return texts


def panel_manhattan(j, name):
    d = D.manhattan(j)
    fig, w, h, left, bottom, right = new_cell(name)
    ax = add_axes_mm(fig, w, h, left, bottom, right)
    xs = np.array([p[0] for p in d["pts"]]); ys = np.array([p[1] for p in d["pts"]])
    cols = [p[2] for p in d["pts"]]
    for colr in dict.fromkeys(cols):                                       # one '.' series per colour, as ros3b
        m = np.array([c == colr for c in cols])
        ax.plot(xs[m], ys[m], '.', color=colr, markersize=MS, markeredgewidth=0, zorder=2)
    # x limits as ros3b's autoscale: all plotted x (points and the markersize-0 spacers) with the 5% margin
    allx = np.r_[xs, d["spacer_x"]]
    lo, hi = allx.min(), allx.max(); m5 = 0.05 * (hi - lo)
    ax.set_xlim(lo - m5, hi + m5)
    ax.set_ylim(*d["ylim"])
    ax.set_xticks(d["ticks"]); ax.set_xticklabels(D.CHRS, rotation=90)
    ax.set_xlabel(D.XLAB[j]); ax.set_ylabel('loadings/score')
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    texts = place_labels(fig, ax, d["labels"], list(zip(xs, ys)), fs=FEAT_FS)
    LOG.append(f"{name} ({D.POS2TREAT[j]}): PIP > 0.5 {d['n_pip']}; plotted {len(xs)} (no annotation: "
               f"{len(d['unplaced'])}); labels {len(d['labels'])}; ylim {tuple(round(v, 4) for v in d['ylim'])}")
    save_cell(fig, {1: "S10c_C87_RNA_cell", 2: "S10d_C87_meth_cell", 3: "S10e_C87_chip_cell"}[j], extra_texts=texts)
    return d


# ================================================================ f: p06 S10f at cell size
RNA = rna_input()
exR = excl_mask("RNA"); keepR = ~exR


def top50_agg(k, ct):                       # p06_c21_single.py, verbatim
    L = comp_loadings(k, "RNA"); X = L.X.values; P = L.P.values
    idx = np.where(P > 0.5)[0]; idx = idx[np.argsort(-np.abs(X[idx]))][:50]
    M = np.asarray(RNA[CTS_BLOCK.index(ct)][:, idx], float)
    w = X[idx] * bcor(k)[ct]
    ag, n = zagg(M[keepR], w)
    out = np.full(276, np.nan); out[keepR] = ag
    return out, n


def gene(k, g, ct):                         # p06_c21_single.py, verbatim
    j = list(comp_loadings(k, "RNA").index).index(g)
    y = np.asarray(RNA[CTS_BLOCK.index(ct), :, j], float).copy(); y[exR] = np.nan
    return y


S_PT, RING_LW = 3.0, 0.6                    # p06 s 26 and ring s 26 x 3.2, lw 1.6 at its 15 in width


def broken_x_scatter_mm(fig, rect, a, series, xlabel, ylabel):
    """vp_common.broken_x_scatter with the axes placed in figure fractions (rect = l, b, w, h) and print-size marks:
    same 30 x IQR split, width ratios 1:4 (outlier side first when the outliers are negative), wspace 0.07,
    x limits, MaxNLocator(4) on the bulk, one tick at the outliers' mean (2 dp), ring on the most extreme donor,
    break marks, y label on the first axes, x label on the bulk axes."""
    out = outliers_30iqr(a); ext = most_extreme(a)
    assert out.any()
    neg = a[out].mean() < np.median(a)
    l, b, wd, ht = rect
    ws = 0.07; unit = wd / (5 + ws * 2.5)                                 # gridspec: wspace relative to mean width
    w1, w4 = unit * 1, unit * 4; gap = ws * unit * 2.5
    if neg:
        axo = fig.add_axes([l, b, w1, ht]); axb = fig.add_axes([l + w1 + gap, b, w4, ht], sharey=axo)
        axes = [axo, axb]
    else:
        axb = fig.add_axes([l, b, w4, ht]); axo = fig.add_axes([l + w4 + gap, b, w1, ht], sharey=axb)
        axes = [axb, axo]
    for (y, col, sz, lab) in series:
        ok = ~np.isnan(y)
        for ax in (axb, axo):
            ax.scatter(a[ok], y[ok], color=col, s=sz, alpha=0.8, linewidths=0, zorder=2)
        if not np.isnan(y[ext]):
            for ax in (axb, axo):
                ax.scatter([a[ext]], [y[ext]], s=sz * 3.2, facecolors="none", edgecolors=EXTRING,
                           linewidths=RING_LW, zorder=3)
    for ax in axes:
        sns.despine(ax=ax)
    bulk = a[~out]; pad = (bulk.max() - bulk.min()) * 0.05
    axb.set_xlim(bulk.min() - pad, bulk.max() + pad)
    axb.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    o = a[out]; span = max(np.abs(o).max() * 0.04, 1e-3)
    axo.set_xlim(o.min() - span, o.max() + span)
    axo.set_xticks([float(f"{o.mean():.2f}")])
    left_, right_ = (axo, axb) if neg else (axb, axo)
    left_.spines["right"].set_visible(False); right_.spines["left"].set_visible(False)
    right_.tick_params(labelleft=False, left=False)
    dd = 0.015; kw = dict(color="k", clip_on=False, linewidth=0.6)
    dl, dr = (4 * dd, dd) if neg else (dd, 4 * dd)
    right_.plot((-dr, +dr), (-dd * 2, dd * 2), transform=right_.transAxes, **kw)
    left_.plot((1 - dl, 1 + dl), (-dd * 2, dd * 2), transform=left_.transAxes, **kw)
    axb.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    return axes


def panel_f():
    k, g, ct = 87, "VTCN1", "Ast"
    a = score(k); y = gene(k, g, ct); agg, n = top50_agg(k, ct)
    fig, w, h, _, bottom, right = new_cell("f")
    axh = h - TOP - bottom
    # two halves as p06's gridspec(1, 2): left half and right half of the cell, each with its own y label
    L1, L2 = 11.0, 13.2                                                   # mm from each half's left edge (L1 = the left spine of a, c, d, e)
    halfw = (w - GUT_X) / 2
    r1 = (L1 / w, bottom / h, (halfw - L1 - 1.0) / w, axh / h)
    r2 = ((halfw + GUT_X + L2) / w, bottom / h, (halfw - L2 - right) / w, axh / h)
    ax1 = broken_x_scatter_mm(fig, r1, a, [(y, CTCOL[ct], S_PT, None)], f"Component {k} loadings/score",
                              f"{g} in {ct} (CPM)")
    ax2 = broken_x_scatter_mm(fig, r2, a, [(agg, CTCOL[ct], S_PT, None)], f"Component {k} loadings/score",
                              f"top-50 gene\naggregate in {ct}\nweighted mean z")
    r, p, nn = spearman(y, a)
    ra, pa, na = spearman(agg, a)
    LOG.append(f"f: VTCN1 Ast n {nn}, Spearman {r:+.3f} (p {p:.2g}); aggregate n {na}, Spearman {ra:+.3f}; "
               f"aggregate genes {n}; dominant donor excluded by the snRNA flag: {bool(exR[most_extreme(a)])}")
    assert nn == 274 and na == 274 and round(r, 2) == -0.05 and round(p, 2) == 0.38, (nn, r, p)
    save_cell(fig, "S10f_C87_VTCN1_Ast_and_aggregate_cell")
    return dict(r=r, p=p, n=nn)


# ================================================================ fidelity checks against the original panels
def fit_map(src, dst):
    """Least-squares x/y linear maps dst = s * src + t (separately per axis); returns max residual (pt)."""
    src = np.asarray(src, float); dst = np.asarray(dst, float); res = 0.0
    for c in (0, 1):
        A = np.c_[src[:, c], np.ones(len(src))]
        coef, *_ = np.linalg.lstsq(A, dst[:, c], rcond=None)
        res = max(res, float(np.abs(A @ coef - dst[:, c]).max()))
    return res


def check_rna_points(d):
    mk = PM.markers(f"{CODING}/87_RNA.pdf")
    assert len(mk) == len(d["pts"]), (len(mk), len(d["pts"]))
    # same multiset of points: sort both by x (positions are distinct), fit the linear page map, compare
    ours = sorted([(p[0], p[1]) for p in d["pts"]]); theirs = sorted([(m[0], m[1]) for m in mk])
    res = fit_map(ours, theirs)
    cols_o = [tuple(round(v, 3) for v in mpl.colors.to_rgb(p[2])) for p in sorted(d["pts"])]
    cols_t = [tuple(round(v, 3) for v in m[2]) for m in sorted(mk, key=lambda m: (m[0], m[1]))]
    same_col = sum(a_ == b_ for a_, b_ in zip(cols_o, cols_t))
    LOG.append(f"check c vs 87_RNA.pdf: {len(mk)} markers; max residual of the linear page map {res:.3f} pt; "
               f"colours equal {same_col}/{len(mk)}")
    assert res < 0.05 and same_col == len(mk)


def check_labels_in_original(j, d):
    """The original PDFs write each string as kerned TJ pieces; compare the glyph sequence with the label text."""
    path = {1: "87_RNA.pdf", 2: "87_meth.pdf", 3: "87_chip.pdf"}[j]
    os.makedirs(f"{HERE}/orig_png", exist_ok=True)
    txt = f"{HERE}/orig_png/{path}.txt"
    os.system(f'pdftotext -raw "{CODING}/{path}" "{txt}"')
    words = open(txt).read().split()
    miss = [l[2] for l in d["labels"] if l[2] not in words]
    LOG.append(f"check {path}: labels found in the original text {len(d['labels']) - len(miss)}/{len(d['labels'])}"
               + (f"; missing {miss}" if miss else ""))
    assert not miss


def check_bars(vals):
    rr = [q for q in PM.rects(f"{PNEW}/g_ros/figs/outlier/87/87_cell_type.pdf") if q[4] is not None]
    assert len(rr) == 6, len(rr)
    hts = [q[3] for q in rr]
    k_ = np.polyfit(vals, hts, 1)
    res = float(np.abs(np.polyval(k_, vals) - np.array(hts)).max())
    cols_ok = all(np.allclose(q[4], mpl.colors.to_rgb(D.CT_COL[c]), atol=2e-3) for q, c in zip(rr, D.CT_ORDER))
    LOG.append(f"check b vs corrected 87_cell_type.pdf: 6 bars, max residual of a linear map {res:.4f} pt; "
               f"colours and order equal: {cols_ok}")
    assert res < 0.05 and cols_ok


def check_samples(A):
    """Original 87_samples.pdf: the normal-range donors are 275 marker placements on the page (top axes); pair them
    with the replot's normal donors by x order and fit the linear page maps (x: age-at-death order, y: score)."""
    mk = np.array([(m[0], m[1]) for m in PM.markers(f"{CODING}/87_samples.pdf")])
    on = mk[(mk[:, 1] > 0) & (mk[:, 1] < 432)]
    nx = A["x"][A["normal"]]; ny = A["y"][A["normal"]]
    o = np.argsort(nx); ours = np.c_[nx[o], ny[o]]
    theirs = on[np.argsort(on[:, 0])]
    assert len(ours) == len(theirs), (len(ours), len(theirs))
    res = fit_map(ours, theirs)
    LOG.append(f"check a vs 87_samples.pdf: {len(theirs)} normal-range markers paired by x order; max residual of the "
               f"linear page maps {res:.3f} pt (order by age at death and scores reproduced)")
    assert res < 0.05


if __name__ == "__main__":
    A = panel_a()
    vals = panel_b()
    dd = {}
    for j, nm in ((1, "c"), (2, "d"), (3, "e")):
        dd[j] = panel_manhattan(j, nm)
    check_rna_points(dd[1])
    check_samples(A)
    for j in (1, 2, 3):
        check_labels_in_original(j, dd[j])
    check_bars(vals)
    F = panel_f()
    open(f"{HERE}/s10_panels.log", "w").write("\n".join(LOG) + "\n")
    print("\n".join(LOG))
