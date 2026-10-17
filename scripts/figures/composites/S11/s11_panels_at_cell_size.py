"""Figure S11 (Component 177), all five panels redrawn at their final printed size for the python composite
(v23 supplement; assembled and lettered by composites/compose.py). Each PDF is exactly its
composite cell and is placed at scale 1 in the 142.9 mm wide figure, so the sizes set here are the printed sizes.

Why redrawn: the existing files were drawn in seaborn talk style at 203-323 mm wide with 16.5 pt text; fitted into
the 182.3 mm S11 height budget they would print at about 4-5 pt (compose.py --panel-info: 5 pt needs a 61.6/69.3/
92.4/92.4/97.9 mm width for a-e, i.e. c, d and e each need most of a row, total > 182 mm with a+b).

Sources and recipes (all read only):
  a  177_samples.pdf (v22; outlier/177 in the version-1 panels) -> ros3b_plot_outliers.py l.226-317 (version-1
     copy): donors ordered by age at death ('90+' as 90), coloured by recorded sex (muted blue =
     Male, muted green = Female), broken y-axis (brokenaxes: two ranges, normal donors and the 30 x IQR outlier,
     each padded as l.245-252, heights proportional to the ranges, hspace 0.05, top/right spines off, '//' marks),
     no x ticks, x label 'Samples', framed Male/Female legend top right.
     ADDED (S11a): the y label 'loadings/score' (the original panel has none).
  b  corrected 177_cell_type.pdf (v23 panels: g_ros/figs/outlier/177/) -> ros3b l.326-340: bars of
     ROSMAP_cell_type_panels_values.csv cor_* (C177) in the order Ast Exc Inh Mic OPC Oli with the
     ros3b colours, y label 'loadings/score', full box.
  c, d 177_RNA.pdf, 177_meth.pdf (v22) -> ros3b l.319-517: PIP > 0.5 features at their genomic position
     (chromosomes laid end to end as ros3b, colours cycled per chromosome), the 8 largest (black) and 8 smallest
     (grey) labelled when > 16 features, else positive black / non-positive grey, chromosome tick labels rotated
     90 degrees, x labels as ros3b, full box. y limits as ros3b, except where noted in Y_HEADROOM below.
  e  v23 S11e_C177_NDUFA13_Mic_and_aggregate.pdf -> panels/p06_c21_single.py (S11e)
     with vp_common.broken_x_scatter: NDUFA13 in Mic (CPM) and the top-50 Mic aggregate against the C177 score,
     n 274 (snRNA-seq RNA-1/RNA-2 excluded), broken x-axis (30 x IQR outlier on the right), dominant donor ringed.

Print-size settings (presentation only; same words, colours, order), shared with the S10 build
(composites/S10/s10_panels_at_cell_size.py, from which the cell helpers and broken_x_scatter_mm are
adapted): tick labels 6 pt, axis labels, legend and in-panel text 6.5 pt; axes lines 0.6 pt; markers, rings and
leader lines scaled with the panel. adjustText (used by ros3b) is not in the locked environment,
so the c-d feature labels are placed by a deterministic search (place_labels); the labelled features, their text
and colours are those of ros3b. In e the aggregate y label is broken into three lines to fit the axes height.

Checks: every point/bar/label of a-d is compared with the original PDFs (pdf_marks.py); the e statistics are
compared with the legend (n 274, rho 0.28). Logs aggregates only: no donor IDs, no per-donor values.
Writes only under <G_ROS_FIGURE_OUT>/S11/."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vp_common import *          # style(): seaborn white/talk, Arial, pdf.fonttype 42; score, rna_input, CTCOL, ...
import numpy as np, pandas as pd
import matplotlib.lines as mlines
from scipy import stats as ss
import s11_data as D
import pdf_marks as PM

from figure_paths import V1_PANELS, V23_PANELS
HERE = f"{FIGOUT}/S11"
PAN = f"{HERE}/panels"; os.makedirs(PAN, exist_ok=True)
CODING = f"{V1_PANELS}/outlier/177"
PNEW = V23_PANELS
MM = 1 / 25.4
TICK, LAB = 6.0, 6.5
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "font.size": LAB, "axes.labelsize": LAB, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
                     "legend.fontsize": LAB, "axes.linewidth": 0.6, "lines.linewidth": 0.8, "patch.linewidth": 0.3,
                     "xtick.major.pad": 1.8, "ytick.major.pad": 1.5, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.labelpad": 2.0,
                     "axes.edgecolor": "black", "text.color": "black", "axes.labelcolor": "black",
                     "xtick.color": "black", "ytick.color": "black",
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

# ---------------------------------------------------------------- geometry (mm); the layout JSON uses the same numbers
W = 142.9
GUT_X = 3.0
HALF = (W - GUT_X) / 2
TOP = 3.0                         # strip above the axes for the panel letter
AXL = 11.0                        # common left edge of the a, c, d, e axes (mm from the cell's left edge)
GEOM = {  # name: (cell w, cell h, left, bottom, right) ; axes top = TOP
    "a": (HALF, 40.0, AXL, 4.6, 1.5),
    "b": (HALF, 40.0, 9.5, 4.6, 1.5),
    "c": (W, 50.0, AXL, 8.4, 1.5),
    "d": (W, 36.0, AXL, 8.4, 1.5),
    "e": (W, 40.0, AXL, 6.6, 1.5),
}
ROW_GAP = 2.0
LOG = []


def new_cell(name):
    w, h, left, bottom, right = GEOM[name]
    fig = plt.figure(figsize=(w * MM, h * MM), dpi=72)
    return fig, w, h, left, bottom, right


def add_axes_mm(fig, w, h, left, bottom, right, top=TOP):
    return fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - top - bottom) / h])


def bbox_mm(fig, artist):
    """Artist extent in mm, (x0, top, x1, bottom) measured from the cell's top-left (y down)."""
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


def check_cell(fig, name, keep_clear_mm=(3.2, 3.0), extra_texts=()):
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
        assert not (x0 < kw and y0 < kh), (name, "text in the letter corner", t.get_text(), (x0, y0))
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


def ylabel_left_of_ticks(fig, ax, text, gap_mm=1.2, x_mm=None):
    """y label placed gap_mm left of the widest drawn tick label (or at x_mm), centred on the axes."""
    fig.canvas.draw()
    w = fig.get_size_inches()[0] * 25.4
    if x_mm is None:
        x0 = min(bbox_mm(fig, t)[0] for t in drawn_ticklabels(ax.yaxis))
        x_mm = x0 - gap_mm
    pos = ax.get_position()
    ax.set_ylabel(text)
    ax.yaxis.set_label_coords((x_mm / w - pos.x0) / pos.width, 0.5)
    ax.yaxis.label.set_ha("center"); ax.yaxis.label.set_va("bottom")
    return x_mm


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
    S = 6.0                                                               # pt^2; ros3b rc s 81 (talk) scaled to the cell
    for ax in axes:
        for m in (normal, out):                                           # normal data, then outliers (ros3b order)
            ax.scatter(x[m], y[m], c=cols[m], edgecolors='k', alpha=0.8, linewidth=0.0, s=S)
        ax.set_xlim(-0.05 * (len(y) - 1), (len(y) - 1) * 1.05)
        ax.set_xticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.025))   # the original's tick step
    axes[0].spines["bottom"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    # brokenaxes diagonal marks on the left spine (tilt 45 deg), scaled to the panel
    d = 1.0                                                               # mm, half-length of a mark
    kw = dict(color="k", clip_on=False, linewidth=0.6)
    for ax, ydat in ((axes[0], 0.0), (axes[1], 1.0)):
        trans = ax.transAxes
        hmm = ax.get_position().height * h; wmm = ax.get_position().width * w
        ax.plot((-d / wmm, d / wmm), (ydat - d / hmm, ydat + d / hmm), transform=trans, **kw)
    big = fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - TOP - bottom) / h], frameon=False)
    big.set_xticks([]); big.set_yticks([]); big.patch.set_alpha(0)
    big.set_xlabel('Samples', labelpad=2.0)
    fig.canvas.draw()
    x0 = min(bbox_mm(fig, t)[0] for ax in axes for t in drawn_ticklabels(ax.yaxis))
    ylabel_left_of_ticks(fig, big, 'loadings/score', x_mm=x0 - 1.2)      # ADDED (S11a)
    leg = [mlines.Line2D([], [], color=cmap['Male'], marker='o', linestyle='None', markersize=3.2, label='Male'),
           mlines.Line2D([], [], color=cmap['Female'], marker='o', linestyle='None', markersize=3.2, label='Female')]
    lg = fig.legend(handles=leg, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=LAB, frameon=True,
                    borderpad=0.4, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.3, handlelength=1.0)
    lg.get_frame().set_linewidth(0.4)
    # the legend must not hide the outlier point
    fig.canvas.draw()
    lb = bbox_mm(fig, lg)
    po = axes[0].transData.transform((x[out][0], y[out][0])); hpt = h / 25.4 * 72
    pmm = (po[0] * 25.4 / 72, (hpt - po[1]) * 25.4 / 72)
    assert not (lb[0] - 1 < pmm[0] < lb[2] + 1 and lb[1] - 1 < pmm[1] < lb[3] + 1), "legend covers the outlier"
    LOG.append(f"a: n {len(y)}; female {int((sex == 'Female').sum())}; beyond 30 x IQR {int(out.sum())} "
               f"({list(sex[out])}); ylims {[tuple(round(v, 4) for v in t) for t in ylims]}")
    save_cell(fig, "S11a_C177_samples_cell")
    return dict(x=x, y=y, sex=sex, normal=normal, out=out, cmap=cmap)


# ================================================================ b: corrected cell-type bars (ros3b l.326-340)
def panel_b():
    order, vals = D.celltype()
    fig, w, h, left, bottom, right = new_cell("b")
    ax = add_axes_mm(fig, w, h, left, bottom, right)
    y_pos = np.arange(len(order))
    ax.bar(y_pos, vals, align='center', color=[D.CT_COL[c] for c in order])
    ax.set_xticks(y_pos); ax.set_xticklabels(order)
    ax.set_ylabel('loadings/score')
    LOG.append("b: " + ", ".join(f"{c} {v:+.4f}" for c, v in zip(order, vals)) +
               f"; all one sign: {all(v < 0 for v in vals) or all(v > 0 for v in vals)}; largest |.|: "
               f"{order[int(np.argmax(np.abs(vals)))]}")
    save_cell(fig, "S11b_C177_cell_type_cell")
    return vals


# ================================================================ c-d: loadings by chromosome (ros3b l.319-517)
MS = 4.0          # '.' marker size (pt); ros3b rc 9 (talk) x 0.469 (304.8 -> 142.9 mm), rounded
LEAD_LW = 0.35    # adjustText arrowprops lw 0.5 x 0.469 (rounded up)
Y_HEADROOM = {1: 1.9}   # j -> upper y limit instead of ros3b's, only where the ros3b limit leaves no room above
                        # the top points for their labels (c: ros3b -0.09, the top point at -0.10; the original's
                        # adjustText labels sat on the frame). Data, labels and lower limit unchanged.


def _seg_hits_box(p, q, box, pad=0.5):
    """Does segment p-q pass through box (x0, y0, x1, y1) (display units)? Sampled test, adequate here."""
    for t in np.linspace(0.08, 0.92, 24):
        x = p[0] + t * (q[0] - p[0]); y = p[1] + t * (q[1] - p[1])
        if box[0] - pad < x < box[2] + pad and box[1] - pad < y < box[3] + pad:
            return True
    return False


def _segs_cross(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, q1, q2) * ccw(p2, q1, q2) < 0) and (ccw(p1, p2, q1) * ccw(p1, p2, q2) < 0)


def _greedy(order, labels, cand, P, rad, axbb):
    """One greedy pass in the given order over precomputed candidates; returns (total cost, choices)."""
    placed = []; leaders = []; choice = {}; total = 0.0
    for i in order:
        p = cand[i]["p"]; best = None
        for (dd, box, anchor, ha, va) in cand[i]["c"]:
            if best is not None and best[0] <= dd:                         # nothing better can come from further out
                break
            if any(overlap(box, q, pad=1.0) for q in placed):
                continue
            cover = int(np.sum((P[:, 0] > box[0] - rad) & (P[:, 0] < box[2] + rad) &
                               (P[:, 1] > box[1] - rad) & (P[:, 1] < box[3] + rad)))
            cross = 0
            if dd > 2.0:
                qx = min(max(p[0], box[0]), box[2]); qy = min(max(p[1], box[1]), box[3])
                cross += sum(_seg_hits_box(p, (qx, qy), q) for q in placed)
                cross += sum(_segs_cross(p, (qx, qy), l1, l2) for l1, l2 in leaders)
                cross += sum(_seg_hits_box(l1, l2, box) for l1, l2 in leaders)
                cover += int(np.sum([_seg_hits_box(p, (qx, qy), (P[m, 0] - rad, P[m, 1] - rad, P[m, 0] + rad,
                                                                 P[m, 1] + rad), pad=0.0)
                                     for m in range(len(P)) if abs(P[m, 0] - p[0]) + abs(P[m, 1] - p[1]) > 0.5]))
            amb = 0.0
            if dd <= 2.0:                                                   # no leader: the nearest point must be its own
                dx_ = np.maximum(0, np.maximum(box[0] - P[:, 0], P[:, 0] - box[2]))
                dy_ = np.maximum(0, np.maximum(box[1] - P[:, 1], P[:, 1] - box[3]))
                dist = np.hypot(dx_, dy_)
                own = float(np.hypot(max(0, box[0] - p[0], p[0] - box[2]), max(0, box[1] - p[1], p[1] - box[3])))
                nearer = (dist < own - 0.3) & ((np.abs(P[:, 0] - p[0]) + np.abs(P[:, 1] - p[1])) > 0.5)
                amb = 25.0 * float(np.sum(nearer & LABPT)) + 4.0 * float(np.sum(nearer & ~LABPT))
            cost = dd + 10.0 * cover + 30.0 * cross + amb
            if best is None or cost < best[0]:
                best = (cost, dd, box, anchor, ha, va, cover, cross)
        if best is None:
            return np.inf, None
        total += best[0]; choice[i] = best; placed.append(best[2])
        if best[1] > 2.0:
            b = best[2]
            leaders.append((tuple(p), (min(max(p[0], b[0]), b[2]), min(max(p[1], b[1]), b[3]))))
    return total, choice


def place_labels(fig, ax, labels, pts_xy, fs=LAB, n_random=60):
    """Deterministic label placement (adjustText is not installed). Candidate positions around each point at
    increasing distances in 24 directions, inside the axes; a greedy pass keeps for each label the cheapest
    candidate that does not overlap an earlier label (cost = distance in pt + 10 per point covered by the label or
    crossed by its leader + 30 per crossing of an earlier label or leader; a label without a leader also pays 25 for
    each other labelled point and 4 for each unlabelled point that is nearer to it than its own point). The pass is run in several orders (black
    first then by |value|, by x, by -x, and 60 seeded shuffles) and the arrangement with the lowest total cost is
    drawn. A leader line (as adjustText's arrowprops: '-', black) joins the point and the label when the label is
    not right next to it."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    tr = ax.transData
    P = tr.transform(np.asarray(pts_xy, float))                          # display (pt, dpi 72)
    global LABPT                                                         # which points carry a label
    lp = tr.transform(np.asarray([(l[0], l[1]) for l in labels], float))
    LABPT = np.array([np.min(np.abs(lp - q).sum(1)) < 0.01 for q in P])
    rad = MS / 4 + 0.7                                                   # '.' circle radius + clearance, pt
    dists = [1.0, 2.0, 3.5, 5, 7, 9.5, 12, 15, 19, 24, 30, 37, 45, 55, 66, 80, 95]
    angs = np.deg2rad(np.arange(0, 360, 15))
    cand = []
    for (x, y, txt, col) in labels:
        t = ax.text(0, 0, txt, fontsize=fs, color=col, transform=None, ha='left', va='bottom')
        bb = t.get_window_extent(r); tw, th = bb.width, bb.height; t.remove()
        p = tr.transform((x, y)); cs = []
        for dd in dists:
            for a in angs:
                cx, cy = np.cos(a), np.sin(a)
                ha = 'left' if cx > 0.3 else ('right' if cx < -0.3 else 'center')
                va = 'bottom' if cy > 0.3 else ('top' if cy < -0.3 else 'center')
                X = p[0] + (dd + rad) * cx; Y = p[1] + (dd + rad) * cy
                x0 = X if ha == 'left' else (X - tw if ha == 'right' else X - tw / 2)
                y0 = Y if va == 'bottom' else (Y - th if va == 'top' else Y - th / 2)
                box = (x0, y0, x0 + tw, y0 + th)
                if (box[0] >= axbb.x0 + 1.5 and box[2] <= axbb.x1 - 1.5 and box[1] >= axbb.y0 + 1.5
                        and box[3] <= axbb.y1 - 1.5):
                    cs.append((dd, box, (X, Y), ha, va))
        cand.append(dict(p=p, c=cs))
    n = len(labels)
    orders = [sorted(range(n), key=lambda i: (labels[i][3] != "black", -abs(labels[i][1]))),
              sorted(range(n), key=lambda i: labels[i][0]), sorted(range(n), key=lambda i: -labels[i][0])]
    rng = np.random.default_rng(177)
    orders += [list(rng.permutation(n)) for _ in range(n_random)]
    best = (np.inf, None, None)
    for oi, o in enumerate(orders):
        tot, ch = _greedy(o, labels, cand, P, rad, axbb)
        if tot < best[0]:
            best = (tot, ch, oi)
    assert best[1] is not None, "no arrangement"
    LOG.append(f"   label arrangement: order {best[2]} of {len(orders)}, total cost {best[0]:.1f}")
    inv = tr.inverted(); texts = []
    for i in range(n):
        x, y, txt, col = labels[i]
        cost, dd, box, (X, Y), ha, va, cover, cross = best[1][i]
        dx, dy = inv.transform((X, Y))
        texts.append(ax.text(dx, dy, txt, fontsize=fs, color=col, ha=ha, va=va, zorder=4))
        if dd > 2.0:                                                        # leader from the point to the box edge
            p = cand[i]["p"]
            qx = min(max(p[0], box[0]), box[2]); qy = min(max(p[1], box[1]), box[3])
            ex, ey = inv.transform((qx, qy))
            ax.plot([x, ex], [y, ey], color='k', lw=LEAD_LW, zorder=3, solid_capstyle='butt')
        LOG.append(f"   label {txt} ({col}): distance {dd} pt, covers {cover} points, crossings {cross}")
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
    ylo, yhi = d["ylim"]
    if j in Y_HEADROOM:
        yhi = Y_HEADROOM[j]
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(d["ticks"]); ax.set_xticklabels(D.CHRS, rotation=90)
    ax.set_xlabel(D.XLAB[j])
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5 if h < 40 else 9, steps=[1, 2, 5, 10]))
    ylabel_left_of_ticks(fig, ax, 'loadings/score')
    texts = place_labels(fig, ax, d["labels"], list(zip(xs, ys)))
    LOG.append(f"{name} ({D.POS2TREAT[j]}): PIP > 0.5 {d['n_pip']}; plotted {len(xs)} (no annotation: "
               f"{len(d['unplaced'])}); labels {len(d['labels'])}; ylim ros3b {tuple(round(v, 4) for v in d['ylim'])}"
               + (f", drawn to {yhi}" if j in Y_HEADROOM else "") +
               f"; all loadings negative: {bool((ys < 0).all())}")
    save_cell(fig, {1: "S11c_C177_RNA_cell", 2: "S11d_C177_meth_cell"}[j], extra_texts=texts)
    return d


# ================================================================ e: p06 S11e at cell size
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


S_PT, RING_LW = 3.0, 0.6                    # p06 s 26 and ring s 26 x 3.2, lw 1.6 at its 15 in width, scaled


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
    lo_, hi_ = axb.get_xlim(); edge = lo_ if neg else hi_
    tk = mpl.ticker.MaxNLocator(4).tick_values(lo_, hi_)                  # as MaxNLocator(4), minus a tick that would
    tk = [t for t in tk if lo_ <= t <= hi_ and abs(t - edge) > 0.06 * (hi_ - lo_)]   # sit under the break marks
    axb.set_xticks(tk)
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
    return axes


def panel_e():
    k, g, ct = 177, "NDUFA13", "Mic"
    a = score(k); y = gene(k, g, ct); agg, n = top50_agg(k, ct)
    fig, w, h, _, bottom, right = new_cell("e")
    axh = h - TOP - bottom
    # two halves as p06's gridspec(1, 2): left half and right half of the cell, each with its own y label
    L1, L2 = AXL, 13.0                                                    # mm from each half's left edge
    halfw = (w - GUT_X) / 2
    r1 = (L1 / w, bottom / h, (halfw - L1 - 1.0) / w, axh / h)
    r2 = ((halfw + GUT_X + L2) / w, bottom / h, (halfw - L2 - right) / w, axh / h)
    ax1 = broken_x_scatter_mm(fig, r1, a, [(y, CTCOL[ct], S_PT, None)], f"Component {k} loadings/score", None)
    ax2 = broken_x_scatter_mm(fig, r2, a, [(agg, CTCOL[ct], S_PT, None)], f"Component {k} loadings/score", None)
    ylabel_left_of_ticks(fig, ax1[0], f"{g} in {ct} (CPM)")
    ylabel_left_of_ticks(fig, ax2[0], f"top-50 gene\naggregate in {ct}\nweighted mean z")
    r, p, nn = spearman(y, a)
    ra, pa, na = spearman(agg, a)
    LOG.append(f"e: NDUFA13 Mic n {nn}, Spearman {r:+.3f} (p {p:.2g}); aggregate n {na}, Spearman {ra:+.3f}; "
               f"aggregate genes {n}; dominant donor excluded by the snRNA flag: {bool(exR[most_extreme(a)])}; "
               f"donors at zero NDUFA13 in Mic: {int(np.sum(y[~np.isnan(y)] == 0))} of {nn}")
    assert nn == 274 and na == 274 and round(r, 2) == 0.28, (nn, r)
    save_cell(fig, "S11e_C177_NDUFA13_Mic_and_aggregate_cell")
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


def check_manhattan_points(j, d):
    """Every PIP > 0.5 point of the original: marker XObjects (series of >= 2 points) plus the inline dots that
    matplotlib writes for one-point series; same multiset of positions (one linear page map) and colours."""
    path = {1: "177_RNA.pdf", 2: "177_meth.pdf"}[j]
    mk = PM.markers(f"{CODING}/{path}") + PM.inline_dots(f"{CODING}/{path}")
    assert len(mk) == len(d["pts"]), (path, len(mk), len(d["pts"]))
    ours = sorted([(p[0], p[1], p[2]) for p in d["pts"]])
    theirs = sorted([(m[0], m[1], m[2]) for m in mk])
    res = fit_map([(o[0], o[1]) for o in ours], [(t[0], t[1]) for t in theirs])
    same_col = sum(tuple(round(v, 3) for v in mpl.colors.to_rgb(o[2])) == tuple(round(v, 3) for v in t[2])
                   for o, t in zip(ours, theirs))
    LOG.append(f"check {name_of(j)} vs {path}: {len(theirs)} points; max residual of the linear page map {res:.3f} pt;"
               f" colours equal {same_col}/{len(theirs)}")
    assert res < 0.05 and same_col == len(theirs), (res, same_col)


def name_of(j):
    return {1: "c", 2: "d"}[j]


def check_labels_in_original(j, d):
    """The original PDFs write each string as kerned TJ pieces; compare the glyph sequence with the label text."""
    path = {1: "177_RNA.pdf", 2: "177_meth.pdf"}[j]
    os.makedirs(f"{HERE}/orig_png", exist_ok=True)
    txt = f"{HERE}/orig_png/{path}.txt"
    os.system(f'pdftotext -raw "{CODING}/{path}" "{txt}"')
    words = open(txt).read().split()
    miss = [l[2] for l in d["labels"] if l[2] not in words]
    # and nothing labelled in the original that we do not label (words that look like features)
    ours = {l[2] for l in d["labels"]}
    extra = [wd for wd in words if (any(c.isalpha() for c in wd) and wd not in ours and wd not in
             ("Genes", "by", "Chromosome", "CpG", "Sites", "loadings/score", "X", "Y"))]
    LOG.append(f"check {path}: labels found in the original text {len(d['labels']) - len(miss)}/{len(d['labels'])}"
               + (f"; missing {miss}" if miss else "") + (f"; original words not drawn here {extra}" if extra else ""))
    assert not miss and not extra, (miss, extra)


def check_bars(vals):
    rr = [q for q in PM.rects(f"{PNEW}/g_ros/figs/outlier/177/177_cell_type.pdf") if q[4] is not None]
    assert len(rr) == 6, len(rr)
    hts = [q[3] for q in rr]
    k_ = np.polyfit(vals, hts, 1)
    res = float(np.abs(np.polyval(k_, vals) - np.array(hts)).max())
    cols_ok = all(np.allclose(q[4], mpl.colors.to_rgb(D.CT_COL[c]), atol=2e-3) for q, c in zip(rr, D.CT_ORDER))
    LOG.append(f"check b vs corrected 177_cell_type.pdf: 6 bars, max residual of a linear map {res:.4f} pt; "
               f"colours and order equal: {cols_ok}")
    assert res < 0.05 and cols_ok


def check_samples(A):
    """177_samples.pdf: brokenaxes drew the normal donors on both axes as marker XObjects (2 x 275 placements); the
    one-point outlier scatter is written inline by matplotlib (no XObject), so it is not among these marks."""
    mk = PM.markers(f"{CODING}/177_samples.pdf")
    n_norm = int(A["normal"].sum())
    LOG.append(f"check a vs 177_samples.pdf: {len(mk)} marker placements (expected 2 x {n_norm})")
    assert len(mk) == 2 * n_norm, len(mk)
    xs, ys = A["x"][A["normal"]], A["y"][A["normal"]]
    res = []
    for seg in (mk[:n_norm], mk[n_norm:2 * n_norm]):
        res.append(fit_map(np.c_[xs, ys], [(m[0], m[1]) for m in seg]))
    cm = {k: tuple(round(v, 3) for v in c) for k, c in A["cmap"].items()}
    col_ok = sum(tuple(round(v, 3) for v in m[2]) == cm[s] for m, s in zip(mk[:n_norm], A["sex"][A["normal"]]))
    LOG.append(f"   normal donors in sort order on both segments: max residuals {[round(r_, 4) for r_ in res]} pt; "
               f"sex colours equal {col_ok}/{n_norm}")
    assert max(res) < 0.05 and col_ok == n_norm


if __name__ == "__main__":
    A = panel_a()
    vals = panel_b()
    dd = {}
    for j, nm in ((1, "c"), (2, "d")):
        dd[j] = panel_manhattan(j, nm)
    E = panel_e()
    check_samples(A)
    check_bars(vals)
    for j in (1, 2):
        check_manhattan_points(j, dd[j])
        check_labels_in_original(j, dd[j])
    open(f"{HERE}/s11_panels.log", "w").write("\n".join(LOG) + "\n")
    print("\n".join(LOG))
