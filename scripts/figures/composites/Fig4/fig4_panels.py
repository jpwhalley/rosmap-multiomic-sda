"""Figure 4 (Component 172), all five panels redrawn at their final printed size for the Python main composite
(v23 main text; assembled and lettered by composites/compose.py). Each PDF is exactly its
composite cell and is placed at scale 1 in the 180 mm wide figure (npj double column), so the sizes set here are the
printed sizes at 180 mm. Layout as the Figure 4 first assembled in Illustrator: a b / c d / e (full width).

Sources and recipes (all read only):
  a  172_ceradsc_samples(_v23labels).pdf -> the port of the v22 panel code
     (qlabel_panels/qlabel_panels.py, loaded as a module, unmodified: data loading,
     +/-5 SD outlier filter, seaborn-0.12 box shim with Set2, recovered stripplot jitter fed through seaborn's own code
     path) and the 547-family BH q (clinical_BH_547_family.csv), as replot/sample_panels/
     replot_v23labels.py. v23 text: x title 'CERAD neuritic plaque score', ticks 'CERAD 3–4' (n 106, left) and
     'CERAD 1–2' (n 170, right), bracket label 'FDR = 0.017'. The bracket is drawn here (statannotations' look:
     a 0.2-grey line with end ticks, label centred above) so that it is sized for print.
  b  version-1 172_cell_type.pdf (correct as published) -> ros3b l.326-340: bars of the corrected
     cell-type loadings (C172: no change) in the order Ast Exc Inh Mic OPC Oli, ros3b colours, 'loadings/score'.
  c  version-1 172_RNA.pdf -> ros3b l.319-422: PIP > 0.5 genes at their genomic position, colours
     cycled per chromosome, the 8 largest (black) and 8 smallest (grey) labelled, x label 'Genes by Chromosome'.
  d  Fig4i_C172_LRP8_Inh_vs_score.pdf -> panels/p05_neuronal.py scatter_view:
     LRP8 in Inh (CPM, SDA input; n05_views.npy) against the C172 score, n 274, Inh colour, most extreme donor ringed.
  e  Fig4e_C172_KEGG_v23.pdf (replot, no in-bar gene lists) and the original
     enrichr/172_pathway_encrichment.pdf -> replot/kegg_panels/03_replot.py: the same 10
     terms, order, -log10(FDR) and per-bar colours, no gene text, bars 0.4 of the slot (as the replot), xlim 0 to
     1.05 x max, integer ticks, x label '-log10(FDR)'.

Print-size settings (presentation only): all text Arial 7 pt (tick labels, axis titles, in-panel labels); axes lines
0.6 pt; markers, rings and leader lines scaled with the panel; text black. Gene symbols in axis titles are italic
(d). adjustText (ros3b) is not in the locked environment, so the c labels are placed by the deterministic search of
composites/S11/s11_panels_at_cell_size.py (place_labels, copied); labelled genes, text and colours are
those of ros3b. In c the chromosome labels 17, 19 and 21 are not drawn: at this width the rotated 7 pt
labels of chromosomes 16-22 would touch (axes 86 mm; 0.5 mm minimum gap between rotated labels) (the export shows them run together); their points keep the ros3b colours.

Checks: a, points and boxes identical (data coordinates) to the reference port at full size, group sizes, U and q;
b, c, d against the original PDFs (pdf_marks.py); e against the Enrichr table and the replot PDF.
Logs aggregates only: no donor IDs, no per-donor values. Writes only under <G_ROS_FIGURE_OUT>/Fig4/."""
import os, sys, json, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
FIGCODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/figures
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, f"{FIGCODE}/panels")

QDIR = f"{FIGCODE}/qlabel_panels"
_spec = importlib.util.spec_from_file_location("qlabel_panels", os.path.join(QDIR, "qlabel_panels.py"))
qp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(qp)     # sets Agg, sns white/talk, fonttype 42

from vp_common import (CTCOL, EXTRING, RA, most_extreme, spearman)          # also re-applies sns white/talk
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as ss
import fig4_data as D
import pdf_marks as PM

from figure_paths import FIGOUT, V1_PANELS, V23_PANELS
HERE = f"{FIGOUT}/Fig4"
PAN = f"{HERE}/panels"; os.makedirs(PAN, exist_ok=True); os.makedirs(f"{HERE}/orig_png", exist_ok=True)
CODING = V1_PANELS
PNEW = V23_PANELS
MM = 1 / 25.4
FS = 7.0                                     # every text item, pt
INK = "black"
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
                     "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "axes.linewidth": 0.6, "lines.linewidth": 0.8, "patch.linewidth": 0.3,
                     "xtick.major.pad": 1.8, "ytick.major.pad": 1.5, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.labelpad": 2.0,
                     "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.grid": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

# ---------------------------------------------------------------- geometry (mm); fig4_layout.py uses the same numbers
W = 180.0
LW, GUT_X = 100.0, 3.0                # left column (a, c) width; gutter; right column (b, d) = the rest
RX = LW + GUT_X; RW = W - RX
TOP = 3.2                             # strip above the axes for the panel letter
AL, BL = 12.6, 10.6                   # common axes left edge of the left (a, c) and right (b, d) columns
ROW_H = {1: 44.0, 2: 49.0, 3: 46.0}
ROW_GAP = 2.5
BOT = {1: 7.4, 2: 9.0, 3: 7.4}        # axes bottom per row (shared by the row's panels, so axes bottoms align)
GEOM = {  # name: (cell w, cell h, left, bottom, right); axes top = TOP
    "a": (LW, ROW_H[1], AL, BOT[1], 1.2),
    "b": (RW, ROW_H[1], BL, BOT[1], 1.2),
    "c": (LW, ROW_H[2], AL, BOT[2], 1.2),
    "d": (RW, ROW_H[2], BL, BOT[2], 1.6),
    "e": (W, ROW_H[3], None, BOT[3], 1.2),    # left computed from the pathway names
}
LOG = []


def new_cell(name, left=None):
    w, h, l, bottom, right = GEOM[name]
    l = l if left is None else left
    fig = plt.figure(figsize=(w * MM, h * MM), dpi=72)
    ax = fig.add_axes([l / w, bottom / h, (w - l - right) / w, (h - TOP - bottom) / h])
    return fig, ax


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


def check_cell(fig, name, keep_clear_mm=(3.4, 3.2), extra_texts=()):
    """Everything inside the cell; nothing in the letter corner; no overlapping tick labels; axis labels clear of
    tick labels; in-panel labels clear of each other and of the tick labels; every text item 7 pt."""
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
        assert abs(t.get_fontsize() - FS) < 1e-6, (name, t.get_text(), t.get_fontsize())
        x0, y0, x1, y1 = bbox_mm(fig, t)
        assert not (x0 < kw and y0 < kh), (name, "text in the letter corner", t.get_text(), (x0, y0))
    ticks_all = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            bbs = [bbox_mm(fig, t) for t in drawn_ticklabels(axis)]
            ticks_all += bbs
            # labels rotated 90 deg: the glyphs (digits, X, Y: no descenders) fill only the left 0.716 em of the
            # line box (cap height; the box also holds the descent), so test the glyph extent, not the line box
            ink = 0.716 * FS * 25.4 / 72
            bbs = [(b_[0], b_[1], b_[0] + ink, b_[3]) if abs(t.get_rotation() - 90) < 1e-6 else b_
                   for b_, t in zip(bbs, drawn_ticklabels(axis))]
            for i in range(len(bbs)):
                for j in range(i + 1, len(bbs)):
                    assert not overlap(bbs[i], bbs[j], pad=0.2), (name, "tick labels overlap", i, j)
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
    fig.savefig(f"{PAN}/{name}.pdf", metadata={"CreationDate": None})   # exact cell size: no bbox_inches
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


def full_box(ax):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6); sp.set_edgecolor(INK)


# ================================================================ a: sample scores by CERAD (qlabel port, v23 text)
XT_A = ["CERAD 3–4", "CERAD 1–2"]           # order = ["No", "Alzheimer's\nDisease"] (CERAD 3-4 = non-AD)
PT_A = 2.0                                   # stripplot size (pt); original size 5 at 9 in, scaled to the cell
BOX_LW = 0.6                                 # original 2.25 (talk lines.linewidth) at 9 in, scaled


def panel_a():
    k, feat = 172, "ceradsc"
    samples = qp.load_samples()
    q = float(pd.read_csv(qp.BH547, index_col=0).loc[k, feat])
    label = f"FDR = {q:.2g}"
    assert f"{q:.4g}" == "0.0168" and label == "FDR = 0.017", q
    jit = json.load(open(os.path.join(QDIR, "recovered_jitter.json")))["172_ceradsc_samples.pdf"]
    fs = qp.outlier_filter(samples, k)
    order = ["No", "Alzheimer's\nDisease"]
    fs = fs[fs[feat].isin(order)].copy()
    fs[feat] = pd.Categorical(fs[feat], categories=order, ordered=True)
    n = fs[feat].value_counts().reindex(order).tolist()
    assert n == [106, 170], n
    raw = pd.read_csv(qp.CLIN, index_col=0); raw.index = raw.index.astype(str)
    rc = raw.loc[fs.index, "ceradsc"]
    assert set(rc[fs[feat] == "No"].unique()) <= {3, 4} and set(rc[fs[feat] == order[1]].unique()) <= {1, 2}
    g_no, g_ad = fs.loc[fs[feat] == "No", k].values, fs.loc[fs[feat] == order[1], k].values
    U, p = ss.mannwhitneyu(g_ad, g_no, alternative="two-sided")
    U2 = len(g_ad) * len(g_no) - U
    assert 6726 in (round(U), round(U2)) and f"{p:.1e}" == "4.0e-04", (U, U2, p)
    fig, ax = new_cell("a")
    qp.boxplot_seaborn012(feat, k, fs, ax, order, "Set2", showfliers=False, linewidth=BOX_LW)
    qp.stripplot(jit, 0, x=feat, y=k, data=fs, ax=ax, jitter=True, color="black", size=PT_A, linewidth=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(XT_A)
    ax.set_xlabel("CERAD neuritic plaque score")
    ax.set_xlim(-0.5, 1.5)
    # y limits: 5% below the lowest point; above the highest point room for the bracket and its label
    pts = np.concatenate([np.asarray(c.get_offsets(), float)[:, 1] for c in ax.collections
                          if type(c).__name__ == "PathCollection"])
    dmin, dmax = pts.min(), pts.max()
    ylo = dmin - 0.05 * (dmax - dmin)
    axh = ROW_H[1] - TOP - BOT[1]
    GAP, TICK_H, TXT_GAP, TXT_H, MARG = 1.0, 0.7, 0.35, 2.6, 0.3     # mm
    need = GAP + TICK_H + TXT_GAP + TXT_H + MARG
    ytop = (dmax - ylo * need / axh) / (1 - need / axh)
    ax.set_ylim(ylo, ytop)
    upm = (ytop - ylo) / axh                                          # data units per mm
    yb = dmax + GAP * upm; yt = yb + TICK_H * upm
    ax.plot([0, 0, 1, 1], [yb, yt, yt, yb], color="0.2", lw=0.6, solid_capstyle="butt", clip_on=False, zorder=4)
    txt = ax.text(0.5, yt + TXT_GAP * upm, label, ha="center", va="bottom", fontsize=FS, color=INK, zorder=4)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))      # ScalarFormatter: Unicode minus
    full_box(ax)
    ylabel_left_of_ticks(fig, ax, "loadings/score")
    fig.canvas.draw()
    tb = bbox_mm(fig, txt); axb = bbox_mm(fig, ax)
    assert tb[1] >= axb[1] + 0.2, ("FDR label leaves the axes", tb, axb)
    # identical drawing (data coordinates) to the full-size reference port of the v22 panel
    fig_ref, ax_ref, _, lab_ref = qp.panel_ceradsc(samples, k, q, jit, 0)
    assert lab_ref == "p = 0.0168"
    off_new = [np.asarray(c.get_offsets(), float) for c in ax.collections if type(c).__name__ == "PathCollection"]
    off_ref = [np.asarray(c.get_offsets(), float) for c in ax_ref.collections if type(c).__name__ == "PathCollection"]
    assert len(off_new) == len(off_ref) == 2 and all(np.array_equal(a_, b_) for a_, b_ in zip(off_new, off_ref))
    box_new = [np.asarray(p_.get_path().vertices, float) for p_ in ax.patches]
    box_ref = [np.asarray(p_.get_path().vertices, float) for p_ in ax_ref.patches]
    assert len(box_new) == len(box_ref) == 2 and all(np.array_equal(a_, b_) for a_, b_ in zip(box_new, box_ref))
    assert [p_.get_facecolor() for p_ in ax.patches] == [p_.get_facecolor() for p_ in ax_ref.patches]
    stat_ref = [np.asarray(l.get_xydata(), float) for l in ax_ref.lines if len(l.get_xydata()) == 2]
    stat_new = [np.asarray(l.get_xydata(), float) for l in ax.lines if len(l.get_xydata()) == 2]
    assert all(np.array_equal(a_, b_) for a_, b_ in zip(stat_new, stat_ref[:len(stat_new)])) and len(stat_new) == 10
    plt.close(fig_ref)
    LOG.append(f"a: n {n[0]} (CERAD 3-4, left) and {n[1]} (CERAD 1-2, right); Mann-Whitney U {min(U, U2):.0f}/"
               f"{max(U, U2):.0f}, p {p:.2g}; BH q (547 family) {q:.4g} -> '{label}'; points ({sum(map(len, off_new))}), "
               f"boxes, whiskers, caps and medians identical to the reference port; ylim ({ylo:.4f}, {ytop:.4f})")
    save_cell(fig, "Fig4a_C172_ceradsc_samples_cell", extra_texts=[txt])
    return dict(n=n, q=q, U=(U, U2), p=p)


# ================================================================ b: cell-type bars (ros3b l.326-340)
def panel_b():
    order, vals = D.celltype()
    fig, ax = new_cell("b")
    y_pos = np.arange(len(order))
    ax.bar(y_pos, vals, align="center", color=[D.CT_COL[c] for c in order])
    ax.set_xticks(y_pos); ax.set_xticklabels(order)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    full_box(ax)
    ylabel_left_of_ticks(fig, ax, "loadings/score")
    LOG.append("b: " + ", ".join(f"{c} {v:+.4f}" for c, v in zip(order, vals)) +
               f"; all one sign: {all(v < 0 for v in vals) or all(v > 0 for v in vals)}; largest |.|: "
               f"{order[int(np.argmax(np.abs(vals)))]}; ylim {tuple(round(v, 3) for v in ax.get_ylim())}")
    save_cell(fig, "Fig4b_C172_cell_type_cell")
    return vals


# ================================================================ c: gene loadings by chromosome (ros3b l.319-422)
MS = 3.0          # '.' marker size (pt); ros3b rc 9 (talk) at 12 in, scaled to the 96 mm cell (x 0.315), rounded
LEAD_LW = 0.35    # adjustText arrowprops lw 0.5, scaled (rounded up)
LABPT = None


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


def _greedy(order, labels, cand, P, rad, axbb):
    """One greedy pass (s11_panels_at_cell_size.py, copied)."""
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
                cover += int(np.sum([_seg_hits_box(p, (qx, qy), (P[m, 0] - rad, P[m, 1] - rad, P[m, 0] + rad,
                                                                 P[m, 1] + rad), pad=0.0)
                                     for m in range(len(P)) if abs(P[m, 0] - p[0]) + abs(P[m, 1] - p[1]) > 0.5]))
            amb = 0.0
            if dd <= 2.0:
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


def place_labels(fig, ax, labels, pts_xy, fs=FS, n_random=60, seed=172):
    """Deterministic label placement (s11_panels_at_cell_size.py place_labels, copied; adjustText not installed)."""
    global LABPT
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    tr = ax.transData
    P = tr.transform(np.asarray(pts_xy, float))
    lp = tr.transform(np.asarray([(l[0], l[1]) for l in labels], float))
    LABPT = np.array([np.min(np.abs(lp - q).sum(1)) < 0.01 for q in P])
    rad = MS / 4 + 0.7
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
        if dd > 2.0:
            p = cand[i]["p"]
            qx = min(max(p[0], box[0]), box[2]); qy = min(max(p[1], box[1]), box[3])
            ex, ey = inv.transform((qx, qy))
            ax.plot([x, ex], [y, ey], color="k", lw=LEAD_LW, zorder=3, solid_capstyle="butt")
        LOG.append(f"   label {txt} ({col}): distance {dd} pt, covers {cover} points, crossings {cross}")
    return texts


def chrom_labels_without_overlap(fig, ax, min_gap_mm=0.5):
    """Rotated chromosome tick labels; from the left, a label whose glyphs would touch the previous drawn label is
    left blank (here 17, 19 and 21). Ink width of a rotated label = the digit height (0.716 em for Arial)."""
    fig.canvas.draw()
    tr = ax.transData; s = 25.4 / fig.dpi
    xs = [tr.transform((t, 0))[0] * s for t in ax.get_xticks()]
    ink = 0.716 * FS * 25.4 / 72
    keep, last = [], -1e9
    for x, ch in zip(xs, D.CHRS):
        if x - ink / 2 >= last + min_gap_mm or ch in ("X", "Y") and x - ink / 2 >= last + 0.05:
            keep.append(ch); last = x + ink / 2
        else:
            keep.append("")
    ax.set_xticklabels(keep, rotation=90)
    return [c for c, k in zip(D.CHRS, keep) if not k]


def panel_c():
    d = D.manhattan()
    fig, ax = new_cell("c")
    xs = np.array([p[0] for p in d["pts"]]); ys = np.array([p[1] for p in d["pts"]])
    cols = [p[2] for p in d["pts"]]
    for colr in dict.fromkeys(cols):                                   # one '.' series per colour, as ros3b
        m = np.array([c == colr for c in cols])
        ax.plot(xs[m], ys[m], ".", color=colr, markersize=MS, markeredgewidth=0, zorder=2)
    allx = np.r_[xs, d["spacer_x"]]
    lo, hi = allx.min(), allx.max(); m5 = 0.05 * (hi - lo)
    ax.set_xlim(lo - m5, hi + m5)
    ax.set_ylim(*d["ylim"])
    ax.set_xticks(d["ticks"])
    dropped = chrom_labels_without_overlap(fig, ax)
    ax.set_xlabel(D.XLAB_RNA)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    full_box(ax)
    ylabel_left_of_ticks(fig, ax, "loadings/score")
    texts = place_labels(fig, ax, d["labels"], list(zip(xs, ys)))
    LOG.append(f"c (RNA): PIP > 0.5 {d['n_pip']}; plotted {len(xs)} (no annotation: {len(d['unplaced'])}; "
               f"duplicated names {d['dup_names']}); labels {len(d['labels'])}; ylim ros3b "
               f"{tuple(round(v, 4) for v in d['ylim'])}; chromosome labels not drawn: {dropped}")
    save_cell(fig, "Fig4c_C172_RNA_cell", extra_texts=texts)
    return d


# ================================================================ d: LRP8 in Inh vs score (p05 scatter_view)
S_D, RING_S, RING_LW = 4.0, 14.7, 0.6       # p05 s 30, ring s 110 (x 3.67), lw 1.6 at 8 in wide, scaled


def panel_d():
    V = np.load(f"{RA}/neuronal_c21/cache/n05_views.npy", allow_pickle=True).item()
    c = V[172]; a = c["a"]; y = np.asarray(c["views"]["LRP8"], float); ct = c["ct"]; ext = most_extreme(a)
    assert ext == c["ext"] and len(a) == 274 and ct == "Inh"
    fig, ax = new_cell("d")
    ax.scatter(a, y, s=S_D, color=CTCOL[ct], alpha=0.8, linewidths=0, zorder=2)
    ax.scatter([a[ext]], [y[ext]], s=RING_S, facecolors="none", edgecolors=EXTRING, linewidths=RING_LW, zorder=3)
    ax.set_xlabel("Component 172 loadings/score")
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(40))
    sns.despine(ax=ax)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6); ax.spines[sp].set_edgecolor(INK)
    ylabel_left_of_ticks(fig, ax, r"$\mathit{LRP8}$ in Inh (CPM)")
    r, p, nn = spearman(y, a)
    m = np.ones(len(a), bool); m[ext] = False
    LOG.append(f"d: LRP8 in Inh n {nn}; Spearman {r:+.3f} (p {p:.2g}); without the most extreme donor "
               f"{ss.spearmanr(y[m], a[m])[0]:+.3f}; xlim {tuple(round(v, 4) for v in ax.get_xlim())}")
    assert nn == 274 and round(r, 2) == 0.79, (nn, r)
    save_cell(fig, "Fig4d_C172_LRP8_Inh_vs_score_cell")
    return dict(a=a, y=y, ext=ext, r=r)


# ================================================================ e: KEGG (replot/kegg_panels/03_replot.py, at print size)
BAR_FRAC = 0.4                               # bar thickness / slot, as the replot (0.405; original 0.80 with gene text)


def panel_e():
    terms, vals, colors, xlabel, qv = D.kegg()
    assert terms[1] == "MAPK signaling pathway" and (qv < 0.05).all()
    w, h, _, bottom, right = GEOM["e"]
    # left edge = widest pathway name + tick pad + 0.3 mm of white
    fig = plt.figure(figsize=(w * MM, h * MM), dpi=72)
    t = fig.text(0, 0, max(terms, key=len), fontsize=FS)
    fig.canvas.draw(); tw = bbox_mm(fig, t)[2] - bbox_mm(fig, t)[0]; t.remove(); plt.close(fig)
    left = 0.3 + tw + mpl.rcParams["ytick.major.pad"] * 25.4 / 72 + 0.05
    fig, ax = new_cell("e", left=left)
    ax.barh(np.arange(10), vals, height=BAR_FRAC, color=colors, edgecolor="none", zorder=2)
    ax.set_ylim(9.5, -0.5)
    ax.set_xlim(0, 1.05 * vals.max())
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.set_yticks(np.arange(10)); ax.set_yticklabels(terms)
    ax.set_xlabel(xlabel)
    full_box(ax)
    fig.canvas.draw()
    x0 = min(bbox_mm(fig, tl)[0] for tl in ax.get_yticklabels())
    LOG.append(f"e: {len(terms)} KEGG terms, -log10(FDR) {vals.round(3).tolist()}; max FDR {qv.max():.2g}; "
               f"axes left {left:.2f} mm; leftmost name at {x0:.2f} mm; bar {BAR_FRAC} of the slot")
    save_cell(fig, "Fig4e_C172_KEGG_cell", keep_clear_mm=(3.0, 2.9))
    return dict(terms=terms, vals=vals, colors=colors)


# ================================================================ fidelity checks against the original panels
def fit_map(src, dst):
    src = np.asarray(src, float); dst = np.asarray(dst, float); res = 0.0
    for c in (0, 1):
        A = np.c_[src[:, c], np.ones(len(src))]
        coef, *_ = np.linalg.lstsq(A, dst[:, c], rcond=None)
        res = max(res, float(np.abs(A @ coef - dst[:, c]).max()))
    return res


def check_manhattan(d):
    path = f"{CODING}/outlier/172/172_RNA.pdf"
    mk = PM.markers(path) + PM.inline_dots(path)
    assert len(mk) == len(d["pts"]), (len(mk), len(d["pts"]))
    ours = sorted([(p[0], p[1], p[2]) for p in d["pts"]])
    theirs = sorted([(m[0], m[1], m[2]) for m in mk])
    res = fit_map([(o[0], o[1]) for o in ours], [(t_[0], t_[1]) for t_ in theirs])
    same_col = sum(tuple(round(v, 3) for v in mpl.colors.to_rgb(o[2])) == tuple(round(v, 3) for v in t_[2])
                   for o, t_ in zip(ours, theirs))
    LOG.append(f"check c vs 172_RNA.pdf: {len(theirs)} points; max residual of the linear page map {res:.3f} pt; "
               f"colours equal {same_col}/{len(theirs)}")
    assert res < 0.05 and same_col == len(theirs), (res, same_col)
    txt = f"{HERE}/orig_png/172_RNA.txt"
    os.system(f'pdftotext -raw "{path}" "{txt}"')
    words = open(txt).read().split()
    miss = [l[2] for l in d["labels"] if l[2] not in words]
    ours_l = {l[2] for l in d["labels"]}
    extra = [wd for wd in words if (any(ch.isalpha() for ch in wd) and wd not in ours_l and wd not in
             ("Genes", "by", "Chromosome", "loadings/score", "X", "Y"))]
    LOG.append(f"check 172_RNA.pdf labels: found in the original text {len(d['labels']) - len(miss)}/{len(d['labels'])}"
               + (f"; missing {miss}" if miss else "") + (f"; original words not drawn here {extra}" if extra else ""))
    assert not miss and not extra, (miss, extra)


def check_bars(vals):
    rr = [q for q in PM.rects(f"{CODING}/outlier/172/172_cell_type.pdf") if q[4] is not None]
    assert len(rr) == 6, len(rr)
    hts = [q[3] for q in rr]
    k_ = np.polyfit(vals, hts, 1)
    res = float(np.abs(np.polyval(k_, vals) - np.array(hts)).max())
    cols_ok = all(np.allclose(q[4], mpl.colors.to_rgb(D.CT_COL[c]), atol=2e-3) for q, c in zip(rr, D.CT_ORDER))
    LOG.append(f"check b vs 172_cell_type.pdf: 6 bars, max residual of a linear map {res:.4f} pt; colours and order "
               f"equal: {cols_ok}")
    assert res < 0.05 and cols_ok


def check_scatter(Dd):
    mk = PM.markers(f"{PNEW}/new/Fig4/Fig4i_C172_LRP8_Inh_vs_score.pdf")
    n = len(Dd["a"])
    assert len(mk) >= n, len(mk)
    res = fit_map(np.c_[Dd["a"], Dd["y"]], [(m[0], m[1]) for m in mk[:n]])
    LOG.append(f"check d vs Fig4i_C172_LRP8_Inh_vs_score.pdf: {len(mk)} marker placements (the {n} donors in order; the one-point ring is written inline); "
               f"max residual of the linear page map {res:.4f} pt")
    assert res < 0.05


def check_kegg(E):
    for path in (f"{PNEW}/new/Fig4/Fig4e_C172_KEGG_v23.pdf", f"{CODING}/enrichr/172_pathway_encrichment.pdf"):
        rr = [q for q in PM.rects_gray(path) if q[4] is not None and min(q[4]) < 0.99 and q[2] > 1 and abs(q[3]) > 5]
        rr = sorted(rr, key=lambda q: -q[1])[:10]
        assert len(rr) == 10, (path, len(rr))
        widths = np.array([q[2] for q in rr])
        k_ = np.polyfit(E["vals"], widths, 1)
        res = float(np.abs(np.polyval(k_, E["vals"]) - widths).max())
        cols_ok = all(np.allclose(q[4], c, atol=2e-3) for q, c in zip(rr, E["colors"]))
        LOG.append(f"check e vs {os.path.basename(path)}: 10 bars top to bottom, bar length = {k_[0]:.3f} pt per unit "
                   f"(+{k_[1]:.3f}), max residual {res:.3f} pt; colours equal: {cols_ok}")
        assert res < 0.1 and abs(k_[1]) < 2.0 and cols_ok


if __name__ == "__main__":
    A = panel_a()
    vals = panel_b()
    d = panel_c()
    Dd = panel_d()
    E = panel_e()
    check_bars(vals)
    check_manhattan(d)
    check_scatter(Dd)
    check_kegg(E)
    LOG.append(f"geometry (mm): W {W}, left column {LW}, gutter {GUT_X}, right column x {RX} w {RW}, rows {ROW_H}, "
               f"row gap {ROW_GAP}, axes bottoms {BOT}, axes left a/c {AL}, b/d {BL}, top strip {TOP}")
    json.dump(dict(W=W, LW=LW, GUT_X=GUT_X, RX=RX, RW=RW, ROW_H=ROW_H, ROW_GAP=ROW_GAP, TOP=TOP),
              open(f"{HERE}/geometry.json", "w"), indent=1)
    open(f"{HERE}/fig4_panels.log", "w").write("\n".join(LOG) + "\n")
    print("\n".join(LOG))
