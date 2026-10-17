"""Figure 5 (Component 277), all five panels redrawn at their final printed size for the Python composite
(v23 main figures; assembled and lettered by composites/compose.py). Each PDF is exactly its
composite cell and is placed at scale 1 in the 180 mm wide figure, so the sizes set here are the printed sizes at
180 mm. Layout follows the Figure 5 first assembled in Illustrator: a b / c d / e (full width).

Sources and recipes (all read only; data loaders in f5_data.py):
  a  corrected 277_cogdx_samples.pdf and its relabelled replot
     (replot/sample_panels) -> qlabel_panels.py panel_cogdx: C277 score by final clinical diagnosis (cogdx 1/2/4),
     +/-5 SD outlier filter, seaborn-0.12 boxes (husl palette, desaturated 0.75; showfliers False), black donor
     points with the jitter recovered from the original PDF. v23 text: x title 'Clinical diagnosis', ticks 'NCI',
     'MCI', 'AD dementia'; label 'FDR = 0.0024' (547-family BH q 0.00242, two significant figures as in Figs 2a-4a; 2026-10-17), top-centre, y top raised
     only as far as needed for the label to clear every point (as the replot did).
  b  version-1 277_cell_type.pdf -> ros3b l.326-340: bars of the C277 cell-type loadings
     (cor_* = published; all six negative), order Ast Exc Inh Mic OPC Oli, ros3b colours.
  c  version-1 277_RNA.pdf -> ros3b l.319-517 (as supp S11c): 1,155 placed PIP > 0.5 genes at
     genomic position, colours cycled per chromosome, the 8 largest (black) and 8 smallest (grey) labelled (15 drawn:
     AC016831.7 has no annotation row, as in the original), x label 'Genes by Chromosome', full box.
  d  Fig4j_C277_PLCB1_Exc_vs_score.pdf -> panels/p05_neuronal.py scatter_view: PLCB1 in Exc (CPM, SDA
     input) against the C277 score, n 274 (snRNA-seq RNA-1/RNA-2 excluded), most extreme donor ringed, top/right
     spines off. Gene symbol italic in the axis title.
  e  version-1 enrichr/277_pathway_encrichment.pdf and its replot (replot/kegg_panels,
     Fig5e_C277_KEGG_v23.pdf): the same 10 terms, order, -log10(FDR) bar lengths and per-bar RGB,
     xlim 0..1.05 x max, integer ticks, x label '-log10(FDR)', full box; NO in-bar gene lists.

Print-size settings (presentation only; same words, colours, order): all text Arial 7 pt (tick labels, axis titles,
in-panel labels); axes lines 0.6 pt; markers, rings, box lines and leader lines scaled with the panel. Chromosome
tick labels of c are dropped where the rotated label would touch its left neighbour at this width (logged).
adjustText (used by ros3b) is not in the locked environment, so the c labels are placed by the deterministic search of
composites/S11 (place_labels); the labelled genes, their text and colours are those of ros3b.

Checks: every point/bar/label against the original PDFs (pdf_marks.py); a against the reference drawing of
qlabel_panels.panel_cogdx (same points, boxes, whiskers, caps, medians, colours in data units); d statistics
against the legend (n 274, rho 0.61). Logs aggregates only: no donor IDs, no per-donor values.
Writes only under <G_ROS_FIGURE_OUT>/Fig5/."""
import os, sys, re
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import f5_data as D                       # imports qlabel_panels lazily (it sets seaborn white/talk); rc reset below
import pdf_marks as PM
from scipy import stats as ss

from figure_paths import FIGOUT, V1_PANELS, V23_PANELS
HERE = f"{FIGOUT}/Fig5"
PAN = f"{HERE}/panels"; os.makedirs(PAN, exist_ok=True)
CODING = V1_PANELS
PNEW = V23_PANELS
MM = 1 / 25.4
FS = 7.0                                   # every text element, pt at 180 mm
LOG = []

Q = D.qp()                                 # loads qlabel_panels (sets sns white/talk + pdf.fonttype 42)
import matplotlib.pyplot as plt
import seaborn as sns
RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
      "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
      "legend.fontsize": FS, "axes.linewidth": 0.6, "lines.linewidth": 0.8, "patch.linewidth": 0.3,
      "xtick.major.pad": 1.8, "ytick.major.pad": 1.5, "xtick.major.size": 0, "ytick.major.size": 0,
      "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.labelpad": 2.0,
      "axes.edgecolor": "black", "text.color": "black", "axes.labelcolor": "black",
      "xtick.color": "black", "ytick.color": "black", "axes.unicode_minus": True,
      "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
      "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
      "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}


def set_rc():
    sns.set_theme(style="white", context="notebook")      # clears the talk scaling that qlabel_panels set
    mpl.rcParams.update(RC)


set_rc()

# ---------------------------------------------------------------- geometry (mm); the layout JSON uses the same numbers
W = 180.0
X2 = 108.0                         # left edge of the right-hand column (b, d)
WL, WR = 106.0, W - X2             # cell widths: a, c | b, d
TOP = 3.0                          # strip above the axes for the panel letter
AXL = 12.0                         # common left edge of the a and c axes (mm from the cell's left edge)
AXR = 12.0                         # common left edge of the b and d axes
ROW_H = (40.0, 46.0, 52.0)
ROW_GAP = 2.0
ROW_Y = (0.0, ROW_H[0] + ROW_GAP, ROW_H[0] + ROW_GAP + ROW_H[1] + ROW_GAP)
H_TOTAL = ROW_Y[2] + ROW_H[2]
GEOM = {  # name: (cell w, cell h, left, bottom, right) ; axes top = TOP
    "a": (WL, ROW_H[0], AXL, 7.2, 1.0),
    "b": (WR, ROW_H[0], AXR, 4.6, 1.0),
    "c": (WL, ROW_H[1], AXL, 9.4, 1.0),
    "d": (WR, ROW_H[1], AXR, 7.2, 1.0),
    "e": (W, ROW_H[2], None, 7.2, 1.0),       # e: left edge from the longest pathway name
}
CELL_XY = {"a": (0.0, ROW_Y[0]), "b": (X2, ROW_Y[0]), "c": (0.0, ROW_Y[1]), "d": (X2, ROW_Y[1]), "e": (0.0, ROW_Y[2])}
NAMES = {"a": "Fig5a_C277_cogdx_samples_cell", "b": "Fig5b_C277_cell_type_cell", "c": "Fig5c_C277_RNA_cell",
         "d": "Fig5d_C277_PLCB1_Exc_vs_score_cell", "e": "Fig5e_C277_KEGG_cell"}


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
    tick labels; in-panel labels clear of each other and of the tick labels; every text exactly FS pt."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(r); w, h = fig.get_size_inches()
    ext = [round(float(v) * 25.4, 2) for v in (tb.x0, tb.y0, tb.x1, tb.y1)]
    ok = tb.x0 >= -0.005 and tb.y0 >= -0.005 and tb.x1 <= w + 0.005 and tb.y1 <= h + 0.005
    LOG.append(f"{name}: drawn extent (mm, from bottom-left) {ext} within {w * 25.4:.2f} x {h * 25.4:.2f}: {ok}")
    assert ok, (name, ext)
    kw, kh = keep_clear_mm
    sizes = set()
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        sizes.add(round(t.get_fontsize(), 3))
        x0, y0, x1, y1 = bbox_mm(fig, t)
        assert not (x0 < kw and y0 < kh), (name, "text in the letter corner", t.get_text(), (x0, y0))
    assert sizes == {FS}, (name, sizes)
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
    LOG.append(f"{name}: text sizes {sorted(sizes)} pt")


def save_cell(fig, key, **kw):
    check_cell(fig, NAMES[key], **kw)
    fig.savefig(f"{PAN}/{NAMES[key]}.pdf", metadata={"CreationDate": None})     # exact cell size: no bbox_inches
    plt.close(fig); LOG.append(f"wrote panels/{NAMES[key]}.pdf")


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


# ================================================================ a: C277 score by clinical diagnosis
GROUP_LABELS = {1: "No cognitive \n impairment", 2: "Mild cognitive \n impairment", 4: "Alzheimer's\nDisease"}
ORDER_A = [GROUP_LABELS[g] for g in (1, 2, 4)]
XT_A = ["NCI", "MCI", "AD dementia"]
MS_A = 2.0          # stripplot marker diameter (pt): size 5 at the 648-pt original x 0.46 (106 mm cell), rounded
LW_BOX = 0.8        # box/whisker/cap/median lines: 2.25 (talk) x 0.46, rounded up
GAP_A_MM = 0.8      # label-to-point clearance


def prep_a():
    fs, qv, jit = D.cogdx_samples()
    codes = fs["cogdx"].copy()
    fs["cogdx"] = pd.Categorical(fs["cogdx"].map(GROUP_LABELS), categories=ORDER_A, ordered=True)
    return fs, qv, jit, codes


def draw_a(fs, jit, ytop=None):
    fig, w, h, left, bottom, right = new_cell("a")
    ax = add_axes_mm(fig, w, h, left, bottom, right)
    Q.boxplot_seaborn012("cogdx", D.K, fs, ax, order=ORDER_A, palette="husl", showfliers=False, linewidth=LW_BOX)
    Q.stripplot(jit, 0, x="cogdx", y=D.K, data=fs, ax=ax, jitter=True, color="black", size=MS_A, linewidth=0)
    ax.set_xlabel("Clinical diagnosis"); ax.set_ylabel("")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(XT_A)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    if ytop is not None:
        ax.set_ylim(top=ytop)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_edgecolor("black")
    return fig, ax, w, h


def label_clearance_mm(fig, ax, txt):
    """Minimum distance (mm) between the label box and any donor point edge or box/whisker/cap/median line."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bb = txt.get_window_extent(r); s = 25.4 / 72
    best = np.inf
    for c in ax.collections:
        if type(c).__name__ != "PathCollection":
            continue
        p = ax.transData.transform(np.asarray(c.get_offsets(), float))
        dx = np.maximum(np.maximum(bb.x0 - p[:, 0], 0), p[:, 0] - bb.x1)
        dy = np.maximum(np.maximum(bb.y0 - p[:, 1], 0), p[:, 1] - bb.y1)
        best = min(best, float((np.hypot(dx, dy) - MS_A / 2).min()) * s)
    for l in ax.lines:
        xy = ax.transData.transform(np.asarray(l.get_xydata(), float))
        for t in np.linspace(0, 1, 50):
            for k in range(len(xy) - 1):
                q = xy[k] + t * (xy[k + 1] - xy[k])
                dx = max(bb.x0 - q[0], 0, q[0] - bb.x1); dy = max(bb.y0 - q[1], 0, q[1] - bb.y1)
                best = min(best, float(np.hypot(dx, dy) - LW_BOX / 2) * s)
    return best


def panel_a():
    fs, qv, jit, codes = prep_a()
    label = f"FDR = {qv:.2g}"
    assert f"{qv:.4g}" == "0.00242" and label == "FDR = 0.0024", (qv, label)
    n = fs["cogdx"].value_counts().reindex(ORDER_A).tolist()
    assert n == [97, 65, 92] and codes.value_counts().reindex([1, 2, 4]).tolist() == [97, 65, 92]
    ytop = None
    for it in range(40):
        fig, ax, w, h = draw_a(fs, jit, ytop)
        y0, y1 = ax.get_ylim()
        axh_mm = (h - TOP - GEOM["a"][3])
        # label: top-centre, 0.7 mm below the frame (the original: axes (0.5, 0.95), centred)
        txt = ax.text(0.5, 1 - 0.7 / axh_mm, label, ha="center", va="top", transform=ax.transAxes, fontsize=FS)
        clr = label_clearance_mm(fig, ax, txt)
        if clr >= GAP_A_MM:
            break
        ytop = y1 + (y1 - y0) * 0.01
        plt.close(fig)
    assert clr >= GAP_A_MM, clr
    if ytop is None:
        ytop = ax.get_ylim()[1]
    # same drawing as qlabel_panels.panel_cogdx (the original's code path) in data units
    fig_r, ax_r, fs_r, lab_r = Q.panel_cogdx(Q.load_samples(), D.K, qv, jit, 0)
    a_off = [np.asarray(c.get_offsets(), float) for c in ax.collections if type(c).__name__ == "PathCollection"]
    r_off = [np.asarray(c.get_offsets(), float) for c in ax_r.collections if type(c).__name__ == "PathCollection"]
    assert len(a_off) == len(r_off) == 3 and all(np.array_equal(x, y) for x, y in zip(a_off, r_off)), "points differ"
    la = [np.asarray(l.get_xydata(), float) for l in ax.lines]; lb = [np.asarray(l.get_xydata(), float) for l in ax_r.lines]
    assert len(la) == len(lb) and all(np.array_equal(x, y) for x, y in zip(la, lb)), "box lines differ"
    pa = [np.asarray(p.get_path().vertices, float) for p in ax.patches]
    pb = [np.asarray(p.get_path().vertices, float) for p in ax_r.patches]
    assert len(pa) == len(pb) and all(np.array_equal(x, y) for x, y in zip(pa, pb)), "boxes differ"
    assert [p.get_facecolor() for p in ax.patches] == [p.get_facecolor() for p in ax_r.patches], "box colours differ"
    yr = ax_r.get_ylim()
    plt.close(fig_r); set_rc()
    LOG.append(f"a: groups NCI/MCI/AD dementia n {n} (total {sum(n)}); BH q (547) {qv:.6g} -> '{label}'; points, boxes, "
               f"whiskers, caps, medians and box colours identical (data units) to qlabel_panels.panel_cogdx; ylim "
               f"bottom {ax.get_ylim()[0]:.6g} (reference {yr[0]:.6g}), top raised {yr[1]:.6g} -> {ax.get_ylim()[1]:.6g} "
               f"for the label; label clearance {clr:.2f} mm")
    ylabel_left_of_ticks(fig, ax, "loadings/score")
    save_cell(fig, "a", extra_texts=[txt])
    return dict(n=n, label=label)


# ================================================================ b: cell-type bars (ros3b l.326-340)
def panel_b():
    order, vals = D.celltype()
    fig, w, h, left, bottom, right = new_cell("b")
    ax = add_axes_mm(fig, w, h, left, bottom, right)
    y_pos = np.arange(len(order))
    ax.bar(y_pos, vals, align="center", color=[D.CT_COL[c] for c in order])
    ax.set_xticks(y_pos); ax.set_xticklabels(order)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))      # the original's ticks 0, -2, ..., -12
    ylabel_left_of_ticks(fig, ax, "loadings/score")
    one_sign = all(v < 0 for v in vals) or all(v > 0 for v in vals)
    assert one_sign
    LOG.append("b: " + ", ".join(f"{c} {v:+.4f}" for c, v in zip(order, vals)) +
               f"; all one sign: {one_sign}; largest |.|: {order[int(np.argmax(np.abs(vals)))]}")
    save_cell(fig, "b")
    return vals


# ================================================================ c: gene loadings by chromosome (ros3b l.319-517)
MS = 2.6          # '.' marker size (pt); ros3b rc 9 (talk) x 0.33 (304.8 -> ~94 mm axes), rounded
LEAD_LW = 0.35
Y_HEADROOM_C = 195.0  # upper y limit instead of ros3b's 150.2: at this size the ros3b limit leaves no room above
                      # the top points (136.6) for their labels (as supp S11c). Data, labels, lower limit unchanged.


def _seg_hits_box(p, q, box, pad=0.5):
    for t in np.linspace(0.08, 0.92, 24):
        x = p[0] + t * (q[0] - p[0]); y = p[1] + t * (q[1] - p[1])
        if box[0] - pad < x < box[2] + pad and box[1] - pad < y < box[3] + pad:
            return True
    return False


def _segs_cross(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, q1, q2) * ccw(p2, q1, q2) < 0) and (ccw(p1, p2, q1) * ccw(p1, p2, q2) < 0)


def _greedy(order, labels, cand, P, rad, LABPT):
    """One greedy pass (supp S11 place_labels): cheapest non-overlapping candidate per label."""
    placed = []; leaders = []; choice = {}; total = 0.0
    for i in order:
        p = cand[i]["p"]; best = None
        for (dd, box, anchor, ha, va) in cand[i]["c"]:
            if best is not None and best[0] <= dd:
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
                near = np.where((np.abs(P[:, 0] - (p[0] + qx) / 2) < abs(qx - p[0]) / 2 + 3) &
                                (np.abs(P[:, 1] - (p[1] + qy) / 2) < abs(qy - p[1]) / 2 + 3))[0]
                cover += int(np.sum([_seg_hits_box(p, (qx, qy), (P[m, 0] - rad, P[m, 1] - rad, P[m, 0] + rad,
                                                                 P[m, 1] + rad), pad=0.0)
                                     for m in near if abs(P[m, 0] - p[0]) + abs(P[m, 1] - p[1]) > 0.5]))
            amb = 0.0
            dx_ = np.maximum(0, np.maximum(box[0] - P[:, 0], P[:, 0] - box[2]))
            dy_ = np.maximum(0, np.maximum(box[1] - P[:, 1], P[:, 1] - box[3]))
            dist = np.hypot(dx_, dy_)
            other = (np.abs(P[:, 0] - p[0]) + np.abs(P[:, 1] - p[1])) > 0.5
            if dd <= 2.0:
                own = float(np.hypot(max(0, box[0] - p[0], p[0] - box[2]), max(0, box[1] - p[1], p[1] - box[3])))
                nearer = (dist < own - 0.3) & other
                amb = 25.0 * float(np.sum(nearer & LABPT)) + 4.0 * float(np.sum(nearer & ~LABPT))
                # Fig 5 addition: another LABELLED point about as near as the own point reads as the owner
                amb += 25.0 * float(np.sum((dist < own + 4.0) & ~nearer & other & LABPT))
            else:
                # Fig 5 addition: a leader-joined label sitting right next to another labelled point
                amb += 25.0 * float(np.sum((dist < 4.0) & other & LABPT))
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


def place_labels(fig, ax, labels, pts_xy, fs=FS, n_random=60, seed=277):
    """Deterministic label placement of supp S11 (adjustText is not installed): candidates around each point at
    increasing distances in 24 directions, inside the axes; greedy passes in several orders, the cheapest kept;
    a leader line ('-', black, as adjustText's arrowprops) when the label is not next to its point."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    tr = ax.transData
    P = tr.transform(np.asarray(pts_xy, float))
    lp = tr.transform(np.asarray([(l[0], l[1]) for l in labels], float))
    LABPT = np.array([np.min(np.abs(lp - q).sum(1)) < 0.01 for q in P])
    rad = MS / 4 + 0.6
    dists = [1.0, 2.0, 3.5, 5, 7, 9.5, 12, 15, 19, 24, 30, 37, 45, 55, 66, 80, 95]
    angs = np.deg2rad(np.arange(0, 360, 15))
    cand = []
    for (x, y, txt, col) in labels:
        t = ax.text(0, 0, txt, fontsize=fs, color=col, transform=None, ha="left", va="bottom")
        bb = t.get_window_extent(r); tw, th = bb.width, bb.height; t.remove()
        p = tr.transform((x, y)); cs = []
        for dd in dists:
            for a in angs:
                cx, cy = np.cos(a), np.sin(a)
                ha = "left" if cx > 0.3 else ("right" if cx < -0.3 else "center")
                va = "bottom" if cy > 0.3 else ("top" if cy < -0.3 else "center")
                X = p[0] + (dd + rad) * cx; Y = p[1] + (dd + rad) * cy
                x0 = X if ha == "left" else (X - tw if ha == "right" else X - tw / 2)
                y0 = Y if va == "bottom" else (Y - th if va == "top" else Y - th / 2)
                box = (x0, y0, x0 + tw, y0 + th)
                if (box[0] >= axbb.x0 + 1.5 and box[2] <= axbb.x1 - 1.5 and box[1] >= axbb.y0 + 1.5
                        and box[3] <= axbb.y1 - 1.5):
                    cs.append((dd, box, (X, Y), ha, va))
        cand.append(dict(p=p, c=cs))
    n = len(labels)
    orders = [sorted(range(n), key=lambda i: (labels[i][3] != "black", -abs(labels[i][1]))),
              sorted(range(n), key=lambda i: labels[i][0]), sorted(range(n), key=lambda i: -labels[i][0])]
    rng = np.random.default_rng(seed)
    orders += [list(rng.permutation(n)) for _ in range(n_random)]
    best = (np.inf, None, None)
    for oi, o in enumerate(orders):
        tot, ch = _greedy(o, labels, cand, P, rad, LABPT)
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
        if dd > 2.0:
            p = cand[i]["p"]
            qx = min(max(p[0], box[0]), box[2]); qy = min(max(p[1], box[1]), box[3])
            ex, ey = inv.transform((qx, qy))
            ax.plot([x, ex], [y, ey], color="k", lw=LEAD_LW, zorder=3, solid_capstyle="butt")
        LOG.append(f"   label {txt} ({col}): distance {dd} pt, covers {cover} points, crossings {cross}")
    return texts


def thin_chrom_labels(fig, ax, min_gap_mm=0.3):
    """Blank a chromosome tick label when its rotated ink would come within min_gap_mm of the previous drawn one
    (left to right). Ink width of a 90-degree label = the digits' cap height (Arial 0.716 em)."""
    fig.canvas.draw()
    w = fig.get_size_inches()[0] * 25.4
    pos = ax.get_position(); x0d, x1d = ax.get_xlim()
    locs = ax.get_xticks(); labs = [t.get_text() for t in ax.get_xticklabels()]
    ink = 0.716 * FS * 25.4 / 72
    xm = [(pos.x0 + (l - x0d) / (x1d - x0d) * pos.width) * w for l in locs]
    keep, last = [], None
    for xx, lab in zip(xm, labs):
        if last is None or xx - last >= ink + min_gap_mm:
            keep.append(lab); last = xx
        else:
            keep.append("")
    ax.set_xticks(locs); ax.set_xticklabels(keep, rotation=90)
    dropped = [l for l, k in zip(labs, keep) if not k]
    spac = np.diff(xm)
    LOG.append(f"   chromosome tick spacing min {spac.min():.2f} mm (label ink {ink:.2f} mm + gap {min_gap_mm}); "
               f"labels not drawn: {dropped}")
    return dropped


def panel_c():
    d = D.manhattan()
    assert len(d["pts"]) == 1155 and len(d["labels"]) == 15
    fig, w, h, left, bottom, right = new_cell("c")
    ax = add_axes_mm(fig, w, h, left, bottom, right)
    xs = np.array([p[0] for p in d["pts"]]); ys = np.array([p[1] for p in d["pts"]])
    cols = [p[2] for p in d["pts"]]
    for colr in dict.fromkeys(cols):
        m = np.array([c == colr for c in cols])
        ax.plot(xs[m], ys[m], ".", color=colr, markersize=MS, markeredgewidth=0, zorder=2)
    allx = np.r_[xs, d["spacer_x"]]
    lo, hi = allx.min(), allx.max(); m5 = 0.05 * (hi - lo)
    ax.set_xlim(lo - m5, hi + m5)
    ylo, yhi = d["ylim"]
    if Y_HEADROOM_C is not None:
        yhi = Y_HEADROOM_C
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(d["ticks"]); ax.set_xticklabels(D.CHRS, rotation=90)
    ax.set_xlabel(D.XLAB_RNA)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
    dropped = thin_chrom_labels(fig, ax)
    ylabel_left_of_ticks(fig, ax, "loadings/score")
    texts = place_labels(fig, ax, d["labels"], list(zip(xs, ys)))
    LOG.append(f"c (RNA): PIP > 0.5 {d['n_pip']}; plotted {len(xs)} (no annotation: {len(d['unplaced'])}); labels "
               f"{len(d['labels'])} ({sum(l[3] == 'black' for l in d['labels'])} black, "
               f"{sum(l[3] == 'gray' for l in d['labels'])} grey); ylim ros3b {tuple(round(v, 3) for v in d['ylim'])}"
               + (f", drawn {Y_HEADROOM_C}" if Y_HEADROOM_C is not None else ""))
    save_cell(fig, "c", extra_texts=texts)
    return d


# ================================================================ d: PLCB1 in Exc vs the C277 score
S_PT, RING_LW = 3.0, 0.6        # p05 s 30 and ring s 110, lw 1.6 at its 8-in width, scaled (as supp S11e)
EXTRING = "#111111"


def panel_d():
    a, y, ct, ext = D.plcb1_view()
    assert ct == "Exc"
    fig, w, h, left, bottom, right = new_cell("d")
    ax = add_axes_mm(fig, w, h, left, bottom, right)
    ok = ~np.isnan(y)
    ax.scatter(a[ok], y[ok], s=S_PT, color=D.CT_COL[ct], alpha=0.8, linewidths=0, zorder=2)
    ax.scatter([a[ext]], [y[ext]], s=S_PT * 3.2, facecolors="none", edgecolors=EXTRING, linewidths=RING_LW, zorder=3)
    sns.despine(ax=ax)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(250))
    ax.set_xlabel(f"Component {D.K} loadings/score")
    ylabel_left_of_ticks(fig, ax, r"$\mathit{PLCB1}$ in Exc (CPM)")
    r, p = ss.spearmanr(a[ok], y[ok]); n = int(ok.sum())
    assert n == 274 and round(r, 2) == 0.61, (n, r)
    LOG.append(f"d: PLCB1 in {ct} n {n}, Spearman {r:+.3f} (p {p:.2g}); ring on the most extreme donor (robust z); "
               f"that donor has a PLCB1 value: {bool(ok[ext])}")
    save_cell(fig, "d")
    return dict(a=a, y=y, ext=ext, n=n, r=r)


# ================================================================ e: KEGG, no in-bar genes
BAR_FRAC = 0.4   # was 0.6 in the first build; 0.4 as the KEGG replot and the Fig 3e (0.42) / Fig 4e (0.4) builds


def panel_e():
    terms, vals, colors, xlabel = D.kegg()
    fig, w, h, _, bottom, right = new_cell("e")
    # left edge: the longest pathway name + tick pad, measured
    t = fig.text(0, 0, max(terms, key=len), fontsize=FS)
    fig.canvas.draw(); tw = bbox_mm(fig, t)[2] - bbox_mm(fig, t)[0]; t.remove()
    left = 0.4 + tw + 1.8 * 25.4 / 72 + 0.2
    ax = add_axes_mm(fig, w, h, left, bottom, right)
    ax.barh(np.arange(10), vals, height=BAR_FRAC, color=colors, linewidth=0, zorder=2)
    ax.set_ylim(9.5, -0.5)
    ax.set_xlim(0, 1.05 * vals.max())
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.set_yticks(np.arange(10)); ax.set_yticklabels(terms)
    ax.tick_params(axis="y", pad=1.8)
    ax.set_xlabel(xlabel)
    fig.canvas.draw()
    lx = min(bbox_mm(fig, tl)[0] for tl in drawn_ticklabels(ax.yaxis))
    LOG.append(f"e: terms {terms}; -log10(FDR) {[round(v, 3) for v in vals]}; bar = {BAR_FRAC} of the slot; "
               f"axes left {left:.1f} mm (longest name {tw:.1f} mm; leftmost label ink at {lx:.2f} mm)")
    save_cell(fig, "e", keep_clear_mm=(3.2, 3.0))
    return terms, vals, colors


# ================================================================ fidelity checks against the original panels
def fit_map(src, dst):
    src = np.asarray(src, float); dst = np.asarray(dst, float); res = 0.0
    for c in (0, 1):
        A = np.c_[src[:, c], np.ones(len(src))]
        coef, *_ = np.linalg.lstsq(A, dst[:, c], rcond=None)
        res = max(res, float(np.abs(A @ coef - dst[:, c]).max()))
    return res


def check_manhattan(d):
    path = f"{CODING}/outlier/277/277_RNA.pdf"
    mk = PM.markers(path) + PM.inline_dots(path)
    assert len(mk) == len(d["pts"]), (len(mk), len(d["pts"]))
    ours = sorted([(p[0], p[1], p[2]) for p in d["pts"]])
    theirs = sorted([(m[0], m[1], m[2]) for m in mk])
    res = fit_map([(o[0], o[1]) for o in ours], [(t[0], t[1]) for t in theirs])
    same_col = sum(tuple(round(v, 3) for v in mpl.colors.to_rgb(o[2])) == tuple(round(v, 3) for v in t[2])
                   for o, t in zip(ours, theirs))
    LOG.append(f"check c vs 277_RNA.pdf: {len(theirs)} points; max residual of the linear page map {res:.3f} pt; "
               f"colours equal {same_col}/{len(theirs)}")
    assert res < 0.05 and same_col == len(theirs), (res, same_col)
    os.makedirs(f"{HERE}/ref", exist_ok=True)
    txt = f"{HERE}/ref/277_RNA.txt"
    os.system(f'pdftotext -raw "{path}" "{txt}"')
    words = open(txt).read().split()
    miss = [l[2] for l in d["labels"] if l[2] not in words]
    ours_l = {l[2] for l in d["labels"]}
    extra = [wd for wd in words if (any(c.isalpha() for c in wd) and wd not in ours_l and wd not in
             ("Genes", "by", "Chromosome", "loadings/score", "X", "Y"))]
    LOG.append(f"check 277_RNA.pdf labels: found in the original text {len(d['labels']) - len(miss)}/{len(d['labels'])}"
               + (f"; missing {miss}" if miss else "") + (f"; original words not drawn here {extra}" if extra else ""))
    assert not miss and not extra, (miss, extra)


def check_bars(vals):
    rr = [q for q in PM.rects(f"{CODING}/outlier/277/277_cell_type.pdf") if q[4] is not None]
    rr = [q for q in rr if q[4] != (1.0, 1.0, 1.0)]
    assert len(rr) == 6, len(rr)
    hts = [q[3] for q in rr]
    k_ = np.polyfit(vals, hts, 1)
    res = float(np.abs(np.polyval(k_, vals) - np.array(hts)).max())
    cols_ok = all(np.allclose(q[4], mpl.colors.to_rgb(D.CT_COL[c]), atol=2e-3) for q, c in zip(rr, D.CT_ORDER))
    LOG.append(f"check b vs 277_cell_type.pdf: 6 bars, max residual of a linear map {res:.4f} pt; "
               f"colours and order equal: {cols_ok}")
    assert res < 0.05 and cols_ok


def stroked_rings(path, max_pt=20.0):
    """Unfilled closed curves written inline ('x y m' + Bezier 'c' segments + 'h' + 'S'): matplotlib writes a
    one-point scatter with facecolors 'none' this way (no XObject). Returns bbox centres (pt)."""
    c = PM.content(path); NUM = PM.NUM; out = []
    for m in re.finditer(rf"({NUM})\s+({NUM})\s+m\s+((?:(?:{NUM}\s+){{6}}c\s+)+)(?:{NUM}\s+{NUM}\s+l\s+)?h\s+S", c):
        v = [float(z) for z in re.findall(NUM, m.group(0))]
        xs = v[0::2]; ys = v[1::2]
        if max(xs) - min(xs) <= max_pt and max(ys) - min(ys) <= max_pt:
            out.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2))
    return out


def check_scatter(Dd):
    """Fig4j: 274 donor markers (XObjects, donor order, NaN dropped) and one inline stroked ring."""
    path = f"{PNEW}/new/Fig4/Fig4j_C277_PLCB1_Exc_vs_score.pdf"
    pts = PM.markers(path)
    ok = ~np.isnan(Dd["y"])
    xs, ys = Dd["a"][ok], Dd["y"][ok]
    LOG.append(f"check d vs Fig4j: {len(pts)} marker placements in the original (expected 274)")
    assert len(pts) == 274, len(pts)
    res = fit_map(np.c_[xs, ys], [(m[0], m[1]) for m in pts])
    A = np.c_[xs, np.ones(len(xs))]; cx, *_ = np.linalg.lstsq(A, [m[0] for m in pts], rcond=None)
    A2 = np.c_[ys, np.ones(len(ys))]; cy, *_ = np.linalg.lstsq(A2, [m[1] for m in pts], rcond=None)
    rings = stroked_rings(path)
    assert len(rings) == 1, len(rings)
    e = Dd["ext"]
    rres = max(abs(cx[0] * Dd["a"][e] + cx[1] - rings[0][0]), abs(cy[0] * Dd["y"][e] + cy[1] - rings[0][1]))
    col_ok = all(np.allclose(m[2], mpl.colors.to_rgb(D.CT_COL["Exc"]), atol=2e-3) for m in pts)
    LOG.append(f"   points in donor order: max residual {res:.4f} pt; colour Exc {col_ok}; the one ring sits on the "
               f"same donor as here: residual {rres:.3f} pt")
    assert res < 0.05 and rres < 0.05 and col_ok


def rects_rg_or_g(path):
    """pdf_marks.rects, with the fill also taken from the grey operator 'g' (matplotlib writes r = g = b fills as
    'v g', e.g. the grey GABAergic synapse bar)."""
    c = PM.content(path); NUM = PM.NUM
    out = []; fill = None
    pat = re.compile(rf'(?P<rg>({NUM})\s+({NUM})\s+({NUM})\s+rg)|(?P<g>(?<![\d.])({NUM})\s+g(?![\w]))|'
                     rf'(?P<box>({NUM})\s+({NUM})\s+m\s+({NUM})\s+({NUM})\s+l\s+'
                     rf'({NUM})\s+({NUM})\s+l\s+({NUM})\s+({NUM})\s+l\s+h\s+[Bf])')
    for m in pat.finditer(c):
        if m.group('rg'):
            fill = tuple(float(v) for v in m.group('rg').split()[:3])
        elif m.group('g'):
            v = float(m.group('g').split()[0]); fill = (v, v, v)
        else:
            v = [float(z) for z in re.findall(NUM, m.group('box'))]
            x0, y0, x1, y1b, x2, y2, x3, y3 = v
            if abs(y0 - y1b) < 1e-6 and abs(x1 - x2) < 1e-6 and abs(y2 - y3) < 1e-6 and abs(x3 - x0) < 1e-6:
                out.append((x0, y0, x1 - x0, y2 - y0, fill))
    return out


def check_kegg(vals, colors):
    path = f"{CODING}/enrichr/277_pathway_encrichment.pdf"
    rr = [q for q in rects_rg_or_g(path) if q[4] is not None and q[4] != (1.0, 1.0, 1.0)]
    bars = sorted([q for q in rr if abs(q[3]) > 30], key=lambda q: -(q[1] + q[3] / 2))   # 44.9-pt bars, top first
    assert len(bars) == 10, len(bars)
    widths = np.array([abs(q[2]) for q in bars])
    k_ = widths / vals
    col_ok = all(np.allclose(q[4], c, atol=2e-3) for q, c in zip(bars, colors))
    LOG.append(f"check e vs 277_pathway_encrichment.pdf: 10 bars top to bottom; width / -log10(FDR) "
               f"{k_.min():.4f}..{k_.max():.4f} pt per unit (one scale: {np.ptp(k_) < 0.01}); colours equal: {col_ok}")
    assert np.ptp(k_) < 0.01 and col_ok


if __name__ == "__main__":
    A = panel_a()
    set_rc()
    vals = panel_b()
    dC = panel_c()
    Dd = panel_d()
    terms, kv, kc = panel_e()
    check_bars(vals)
    check_manhattan(dC)
    check_scatter(Dd)
    check_kegg(kv, kc)
    open(f"{HERE}/f5_panels.log", "w").write("\n".join(LOG) + "\n")
    print("\n".join(LOG))
