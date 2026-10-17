"""Figure 3 (Component 278), all six panels redrawn at their final printed size for the Python main composite
(v23). Each PDF is exactly its composite cell (mm) and is placed at scale 1 in the 180 mm wide figure
(npj double column), so the sizes set here are the printed sizes. Letters are added by compose.py (Arial Bold 8 pt).
Layout follows the Figure 3 first assembled in Illustrator: a b / c d / e f.

Sources and recipes (all read only; data loaders in f3_data.py):
  a  corrected 278_sex_samples.pdf = qlabel_panels.panel_sex: Set2 boxes
     drawn by its seaborn-0.12 shim (median, IQR, 1.5 x IQR whiskers, no fliers), black strip points with the
     recovered jitter, statannotations bracket; x 'Sex', y 'loadings/score'. CHANGED: bracket label
     'p = 0.00974' -> 'FDR = 0.0097' (the same BH q over 547 components, as the legend).
  b  corrected 278_cell_type.pdf: bars of the cor_* values (ROSMAP_cell_type_panels_values.csv) in the
     order Ast Exc Inh Mic OPC Oli with the ros3b colours, y 'loadings/score', full box.
  c  version-1 278_RNA.pdf (v22) = ros3b l.319-430: PIP > 0.5 genes at their genomic position, the
     8 largest (black) and 8 smallest (grey) labelled, ros3b y limits, chromosome ticks rotated 90 degrees,
     x 'Genes by Chromosome', y 'loadings/score', full box.
  d  version-1 278_dlpfc_correlation.pdf (v22) = synth1b l.103-145: sns.regplot look (deep blue
     points, alpha 0.8, OLS line over the data range, ci None), x 'ROSMAP', y 'SEA-AD', the two-line title
     'ROSMAP: 278 vs SEA-AD: 67' / 'Correlation (r) = 0.454', full box.
  e  version-1 enrichr/278_pathway_encrichment.pdf (v22) via the replot (replot/kegg_panels/03_replot.py,
     Fig3e_C278_KEGG_v23.pdf): the same 10 terms, order, -log10(FDR) bar lengths
     and per-bar RGB, 0.15-grey text and frame, white bar edges, xlim 0..1.05 x max, integer ticks, bars 0.42 of the
     row slot as in that replot; NO in-bar gene lists.
  f  Fig3f_C278_FKBP5_OPC_Oli_vs_score.pdf = panels/p04_c278.py (Fig 3f): FKBP5 CPM in OPC (orange) and
     Oli (green) against the score, n 274, most extreme donor ringed, legend OPC/Oli upper left, despined.
     CHANGED: the gene symbol in the y title is italic.

Print-size settings (presentation only; same words, data, colours and order): Arial 7 pt for every text (tick labels,
axis titles, keys, title, bracket label, pathway names; the 16 gene labels in c at FEAT_FS); axes lines 0.6 pt;
markers, rings and lines scaled to the panel. adjustText (used by ros3b) is not in the locked environment,
so the 16 gene labels in c are placed by the deterministic search of composites/S10 (copied below).
At 7 pt the rotated chromosome labels of the smallest chromosomes would overlap; overlapping ones are left unlabelled
(listed in the log) and the tick positions of the rest are unchanged.
Checks: every point/bar/label is compared with the original panel PDFs (pdf_marks.py); statistics are compared with
the legend. Logs aggregates only: no donor IDs, no per-donor values. Writes only under <G_ROS_FIGURE_OUT>/Fig3/."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # scripts/figures
from figure_paths import FIGOUT, V1_PANELS, V23_PANELS
HERE = f"{FIGOUT}/Fig3"
import json
import numpy as np, pandas as pd
from scipy import stats as ss
import f3_data as D

qp = D.qlabel_module()                          # sets Agg, pdf.fonttype 42, seaborn white/talk; statannotations on path
AK, FK, EXT, V = D.fkbp5()                      # imports vp_common (style() again)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import pdf_marks as PM
import pdf_points as PP

PAN = f"{HERE}/panels"; os.makedirs(PAN, exist_ok=True); os.makedirs(f"{HERE}/orig_png", exist_ok=True)
CODING = V1_PANELS
PNEW = V23_PANELS
MM = 1 / 25.4
TXT = 7.0                                       # every text (pt, as printed at 180 mm)
MINPT = 6.5
sns.set(style="white", context="paper")
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold", "mathtext.default": "regular", "axes.unicode_minus": True,
                     "font.size": TXT, "axes.labelsize": TXT, "axes.titlesize": TXT, "xtick.labelsize": TXT,
                     "ytick.labelsize": TXT, "legend.fontsize": TXT, "axes.linewidth": 0.6, "lines.linewidth": 0.8,
                     "patch.linewidth": 0.3, "xtick.major.pad": 1.8, "ytick.major.pad": 1.5,
                     "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
                     "axes.labelpad": 2.0, "axes.titlepad": 2.5, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white", "path.simplify": False})

# ---------------------------------------------------------------- geometry (mm); the layout JSON uses the same numbers
W = 180.0
WL, GUT = 96.0, 2.0
WR = W - WL - GUT                               # 82
XR = WL + GUT                                   # right column x
ROWS = {1: (0.0, 40.0), 2: (41.5, 52.0), 3: (95.0, 50.0)}   # y, height
H = ROWS[3][0] + ROWS[3][1]                     # 145
TOP = 3.5                                       # strip above the axes for the panel letter
LEFT = 11.0                                     # y-axis spine from the cell's left edge (a, b, c, d, f)
GEOM = {  # name: (cell x, cell y, cell w, cell h, left, bottom, right, top)
    "a": (0.0, ROWS[1][0], WL, ROWS[1][1], LEFT, 6.5, 1.5, TOP),
    "b": (XR, ROWS[1][0], WR, ROWS[1][1], LEFT, 6.5, 1.5, TOP),
    "c": (0.0, ROWS[2][0], WL, ROWS[2][1], LEFT, 9.0, 1.5, TOP),
    "d": (XR, ROWS[2][0], WR, ROWS[2][1], LEFT, 9.0, 1.5, 8.6),
    "e": (0.0, ROWS[3][0], WL, ROWS[3][1], None, 6.5, 1.5, TOP),
    "f": (XR, ROWS[3][0], WR, ROWS[3][1], LEFT, 6.5, 1.5, TOP),
}
NAMES = {"a": "Fig3a_C278_sex_samples_cell", "b": "Fig3b_C278_cell_type_cell", "c": "Fig3c_C278_RNA_cell",
         "d": "Fig3d_C278_SEAAD_67_cell", "e": "Fig3e_C278_KEGG_cell", "f": "Fig3f_C278_FKBP5_OPC_Oli_cell"}
LOG = []


def new_cell(name):
    x, y, w, h, left, bottom, right, top = GEOM[name]
    fig = plt.figure(figsize=(w * MM, h * MM), dpi=72)
    return fig, w, h, left, bottom, right, top


def add_axes_mm(fig, w, h, left, bottom, right, top):
    return fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - top - bottom) / h])


def bbox_mm(fig, artist):
    r = fig.canvas.get_renderer()
    bb = artist.get_window_extent(r); hh = fig.get_size_inches()[1] * fig.dpi; s = 25.4 / fig.dpi
    return (bb.x0 * s, (hh - bb.y1) * s, bb.x1 * s, (hh - bb.y0) * s)


def drawn_ticklabels(axis):
    lo, hi = sorted(axis.get_view_interval())
    eps = (hi - lo) * 1e-9
    return [t for loc, t in zip(axis.get_majorticklocs(), axis.get_ticklabels())
            if lo - eps <= loc <= hi + eps and t.get_visible() and t.get_text().strip()]


def overlap(a_, b_, pad=0.0):
    return a_[0] < b_[2] + pad and b_[0] < a_[2] + pad and a_[1] < b_[3] + pad and b_[1] < a_[3] + pad


def check_cell(fig, name, keep_clear_mm=(3.2, 3.0), extra_texts=()):
    """Everything inside the cell; nothing in the letter corner; every text >= MINPT; no overlapping tick labels; axis
    labels clear of tick labels; in-panel labels clear of each other and of the tick labels."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(r); w, h = fig.get_size_inches()
    ext = [round(v * 25.4, 2) for v in (tb.x0, tb.y0, tb.x1, tb.y1)]
    ok = tb.x0 >= -0.005 and tb.y0 >= -0.005 and tb.x1 <= w + 0.005 and tb.y1 <= h + 0.005
    LOG.append(f"{name}: drawn extent (mm, from bottom-left) {ext} within {w * 25.4:.2f} x {h * 25.4:.2f}: {ok}")
    assert ok, (name, ext)
    kw, kh = keep_clear_mm
    sizes = set()
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        sizes.add(round(t.get_fontsize(), 2))
        assert t.get_fontsize() >= MINPT - 1e-9, (name, t.get_text(), t.get_fontsize())
        x0, y0, x1, y1 = bbox_mm(fig, t)
        assert not (x0 < kw and y0 < kh), (name, "text in the letter corner", t.get_text(), (x0, y0))
    LOG.append(f"   font sizes used (pt): {sorted(sizes)}")
    ticks_all = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            bbs = [ink_box_mm(fig, t) for t in drawn_ticklabels(axis)]
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


def save_cell(fig, key, **kw):
    check_cell(fig, NAMES[key], **kw)
    fig.savefig(f"{PAN}/{NAMES[key]}.pdf", metadata={"CreationDate": None})          # exact cell size
    plt.close(fig); LOG.append(f"wrote panels/{NAMES[key]}.pdf")


# ================================================================ a: sample scores by sex (qlabel_panels.panel_sex)
A_SIZE = 2.2                       # strip marker diameter (pt); original size 5 at 9 in wide
A_LW = 0.8                         # box/whisker/median line width (pt); original 2.25 at 9 in wide


def panel_a():
    s, fs, q, jit = D.samples_sex(qp)
    assert len(fs) == 276 and (fs.Sex == "Male").sum() == 129 and (fs.Sex == "Female").sum() == 147
    label = f"FDR = {q:.2g}"
    assert label == "FDR = 0.0097", label
    fig, w, h, left, bottom, right, top = new_cell("a")
    ax = add_axes_mm(fig, w, h, left, bottom, right, top)
    qp.boxplot_seaborn012("Sex", D.K, fs, ax, ["Male", "Female"], "Set2", showfliers=False, linewidth=A_LW)
    ann = qp.Annotator(ax, [("Male", "Female")], data=fs, x="Sex", y=D.K, order=["Male", "Female"])
    ann.set_pvalues([q])
    ann.configure(text_format="simple", show_test_name=False, fontsize=TXT, line_width=0.6)
    ann.set_custom_annotations([label])
    ann.annotate()
    qp.stripplot(jit, 0, x="Sex", y=D.K, data=fs, ax=ax, jitter=True, color="black", size=A_SIZE)
    ax.set_xlabel("Sex"); ax.set_ylabel("loadings/score")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))                    # as the original (-0.15 ... 0.20)
    for sp in ax.spines.values():
        sp.set_visible(True)
    txt = [t for t in ax.texts if t.get_text() == label]
    assert len(txt) == 1
    # label clear of every point (pt)
    fig.canvas.draw(); r = fig.canvas.get_renderer(); bb = txt[0].get_window_extent(r)
    offs = [np.asarray(c.get_offsets(), float) for c in ax.collections if type(c).__name__ == "PathCollection"]
    P = ax.transData.transform(np.vstack(offs))
    dx = np.maximum(np.maximum(bb.x0 - P[:, 0], 0), P[:, 0] - bb.x1); dy = np.maximum(np.maximum(bb.y0 - P[:, 1], 0), P[:, 1] - bb.y1)
    clr = float((np.hypot(dx, dy) - A_SIZE / 2).min())
    LOG.append(f"a: n {len(fs)} (male {int((fs.Sex == 'Male').sum())}, female {int((fs.Sex == 'Female').sum())}); "
               f"q {q:.6g} -> '{label}'; ylim {tuple(round(v, 4) for v in ax.get_ylim())}; label-to-point clearance {clr:.1f} pt")
    assert clr > 1.0
    save_cell(fig, "a", extra_texts=txt)
    return dict(fs=fs, offs=offs, ax_ylim=ax.get_ylim())


# ================================================================ b: corrected cell-type bars
def panel_b():
    order, vals = D.celltype()
    assert np.sign(vals[4]) == np.sign(vals[5])           # OPC and Oli same sign (task, legend)
    fig, w, h, left, bottom, right, top = new_cell("b")
    ax = add_axes_mm(fig, w, h, left, bottom, right, top)
    pos = np.arange(len(order))
    ax.bar(pos, vals, align="center", color=[D.CT_COL[c] for c in order])
    ax.set_xticks(pos); ax.set_xticklabels(order)
    ax.set_ylabel("loadings/score")
    LOG.append("b: " + ", ".join(f"{c} {v:+.4f}" for c, v in zip(order, vals)) + f"; ylim {tuple(round(v, 3) for v in ax.get_ylim())}")
    save_cell(fig, "b")
    return vals


# ================================================================ c: loadings by chromosome (ros3b l.319-430)
MS = 3.4          # '.' marker size (pt); ros3b rc 9 (talk) on a 12 in wide panel
LEAD_LW = 0.35    # adjustText arrowprops lw 0.5, scaled
FEAT_FS = 7.0     # feature labels


def _seg_box(p0, p1, box, pad=0.0):
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
    a = np.asarray(a); b = np.asarray(b); ab = b - a; L2 = float(ab @ ab)
    if L2 == 0:
        return 0
    t = np.clip(((P - a) @ ab) / L2, 0, 1)
    proj = a + t[:, None] * ab
    dist = np.hypot(*(P - proj).T)
    return int(np.sum((dist < r) & (t > 0.02) & (t < 0.98)))


LEAD_MIN = 2.6
COVER_W, LEADPT_W = 45.0, 15.0
AMB_W = 6.0


def _greedy(order, labels, P, LP, sizes, axbb, rad):
    """composites/S10/s10_panels_at_cell_size.py _greedy, verbatim."""
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
                ib0 = box[1] + (0.0 if descend[i] else 0.20 * h); ib1 = box[1] + 1.06 * h
                cover = int(np.sum((P[:, 0] > box[0] - mrad) & (P[:, 0] < box[2] + mrad) &
                                   (P[:, 1] > ib0 - mrad) & (P[:, 1] < ib1 + mrad)))
                cost = dd + max(0.0, dd - 30.0) + COVER_W * cover
                q = None
                ddx = np.maximum(np.maximum(box[0] - P[:, 0], 0), P[:, 0] - box[2])
                ddy = np.maximum(np.maximum(box[1] - P[:, 1], 0), P[:, 1] - box[3])
                dist = np.hypot(ddx, ddy)
                own = np.hypot(max(box[0] - p[0], 0, p[0] - box[2]), max(box[1] - p[1], 0, p[1] - box[3]))
                amb = bool(np.sum(dist < own + 2.0) > 1)
                if amb:
                    cost += AMB_W
                if dd > LEAD_MIN or amb:
                    q = (min(max(p[0], box[0]), box[2]), min(max(p[1], box[1]), box[3]))
                    cost += LEADPT_W * _near_seg(P, p, q, rad + 0.6)
                    if any(_seg_box(p, q, bx, pad=0.8) for bx in boxes):
                        cost += 1e4
                    if any(_seg_seg(p, q, l0, l1) for l0, l1 in leaders):
                        cost += 1e4
                    for k_, lp in enumerate(LP):
                        if k_ != i and _near_seg(np.asarray([lp]), p, q, rad + 0.6):
                            cost += 1e4
                if any(_seg_box(l0, l1, box, pad=0.8) for l0, l1 in leaders):
                    cost += 1e4
                if any(overlap(box, (lp[0] - rad, lp[1] - rad, lp[0] + rad, lp[1] + rad)) for k_, lp in enumerate(LP) if k_ != i):
                    cost += 1e4
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


def place_labels(fig, ax, labels, pts_xy, fs):
    """composites/S10 place_labels (deterministic; adjustText is not in the environment)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    tr = ax.transData
    P = tr.transform(np.asarray(pts_xy, float))
    LP = [tuple(tr.transform((l[0], l[1]))) for l in labels]
    rad = MS / 2 * 0.5 + 0.4
    sizes = []
    for l in labels:
        t = ax.text(0, 0, l[2], fontsize=fs, ha='left', va='bottom', transform=None)
        bb = t.get_window_extent(r); t.remove(); sizes.append((bb.width, bb.height))
    base = sorted(range(len(labels)), key=lambda i: (labels[i][3] != "black", -abs(labels[i][1])))
    rng = np.random.default_rng(278)
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


CAP_EM = 0.716                     # Arial cap height / digit height (em)


def ink_box_mm(fig, t):
    """Text box in mm; for 90-degree text with no descenders (chromosome names) the horizontal extent is narrowed to
    the glyph ink (cap height), centred where matplotlib centres it, since the text box adds ascent/descent padding."""
    b = bbox_mm(fig, t)
    if abs(t.get_rotation() - 90) < 1e-6 and not any(ch in t.get_text() for ch in "gjpqy,()"):
        cx = 0.5 * (b[0] + b[2]); half = 0.5 * CAP_EM * t.get_fontsize() * 25.4 / 72
        return (cx - half, b[1], cx + half, b[3])
    return b


def thin_ticks(fig, ax, pos, labs, gap_mm=0.6, sex_gap_mm=0.25, keep_first=("X", "Y")):
    """Chromosome tick labels: X and Y first (kept if their glyphs are >= sex_gap_mm apart), then the autosomes left to
    right; an autosome label is left blank when its glyphs (ink box) would come within gap_mm of a label already kept.
    Tick positions are unchanged."""
    ax.set_xticks(pos); ax.set_xticklabels(labs, rotation=90)
    fig.canvas.draw()
    bbs = [ink_box_mm(fig, t) for t in ax.get_xticklabels()]
    order = [i for i, l in enumerate(labs) if l in keep_first] + [i for i, l in enumerate(labs) if l not in keep_first]
    keep = []
    for i in order:
        g = sex_gap_mm if labs[i] in keep_first else gap_mm      # X and Y (one letter each) may sit closer together
        if all(bbs[i][0] >= bbs[j][2] + g or bbs[j][0] >= bbs[i][2] + g for j in keep):
            keep.append(i)
    keep.sort()
    dropped = [labs[i] for i in range(len(labs)) if i not in keep]
    ax.set_xticks([pos[i] for i in keep]); ax.set_xticklabels([labs[i] for i in keep], rotation=90)
    return dropped


def panel_c():
    d = D.manhattan()
    fig, w, h, left, bottom, right, top = new_cell("c")
    ax = add_axes_mm(fig, w, h, left, bottom, right, top)
    xs = np.array([p[0] for p in d["pts"]]); ys = np.array([p[1] for p in d["pts"]])
    cols = [p[2] for p in d["pts"]]
    for colr in dict.fromkeys(cols):                                       # one '.' series per colour, as ros3b
        m = np.array([c == colr for c in cols])
        ax.plot(xs[m], ys[m], '.', color=colr, markersize=MS, markeredgewidth=0, zorder=2)
    allx = np.r_[xs, d["spacer_x"]]
    lo, hi = allx.min(), allx.max(); m5 = 0.05 * (hi - lo)
    ax.set_xlim(lo - m5, hi + m5)
    ax.set_ylim(*d["ylim"])
    dropped = thin_ticks(fig, ax, d["ticks"], D.CHRS)
    ax.set_xlabel("Genes by Chromosome"); ax.set_ylabel("loadings/score")
    ax.yaxis.set_major_locator(MultipleLocator(100))                     # as the original (200 ... -300)
    texts = place_labels(fig, ax, d["labels"], list(zip(xs, ys)), fs=FEAT_FS)
    LOG.append(f"c: PIP > 0.5 {d['n_pip']}; plotted {len(xs)} (no annotation: {len(d['unplaced'])}); labels "
               f"{len(d['labels'])}; ylim {tuple(round(v, 3) for v in d['ylim'])}; chromosome labels left blank "
               f"(would overlap at {TXT} pt): {dropped}")
    save_cell(fig, "c", extra_texts=texts)
    return d


# ================================================================ d: ROSMAP C278 vs SEA-AD C67 (synth1b regplot)
D_COL = sns.color_palette("deep")[0]
D_S = 11.0                          # marker area (pt^2): regplot default 81 (talk) on an 8 in wide panel, scaled
D_LW_LINE = 0.9


def panel_d():
    genes, x, y = D.seaad()
    r = float(np.corrcoef(x, y)[0, 1]); rho = float(ss.spearmanr(x, y)[0])
    assert len(genes) == 1050 and round(r, 3) == 0.454 and round(rho, 3) == 0.634, (len(genes), r, rho)
    fig, w, h, left, bottom, right, top = new_cell("d")
    ax = add_axes_mm(fig, w, h, left, bottom, right, top)
    ax.scatter(x, y, s=D_S, color=D_COL, alpha=0.8, linewidths=0.3, edgecolors="face", zorder=2)
    b1, b0 = np.polyfit(x, y, 1)                                           # regplot: OLS fit, ci None
    gx = np.linspace(x.min(), x.max(), 100)                                # truncate=True: data range
    ax.plot(gx, b0 + b1 * gx, color=D_COL, lw=D_LW_LINE, zorder=3)
    ax.set_xlabel("ROSMAP"); ax.set_ylabel("SEA-AD")
    ax.xaxis.set_major_locator(MultipleLocator(100)); ax.yaxis.set_major_locator(MultipleLocator(10))   # as the original
    ax.set_title(f"ROSMAP: {D.K} vs SEA-AD: 67\nCorrelation (r) = {r:.3f}", linespacing=1.15)
    LOG.append(f"d: genes {len(genes)}; Pearson {r:+.4f}; Spearman {rho:+.4f}; OLS slope {b1:.5f}, intercept {b0:.4f}; "
               f"xlim {tuple(round(v, 1) for v in ax.get_xlim())}; ylim {tuple(round(v, 2) for v in ax.get_ylim())}")
    save_cell(fig, "d")
    return dict(x=x, y=y, b=(b1, b0))


# ================================================================ e: KEGG (no in-bar gene lists)
INK = "0.15"
BAR_FRAC = 0.42                     # bar thickness / row slot, as the 10-16 replot (22.47 pt of a 53.2 pt slot)


def panel_e():
    terms, vals, colors, spec = D.kegg()
    fig, w, h, _, bottom, right, top = new_cell("e")
    # left margin from the widest pathway name (7 pt) + tick pad + 0.5 mm
    r = fig.canvas.get_renderer()
    wmax = 0.0
    for t in terms:
        tt = fig.text(0, 0, t, fontsize=TXT); wmax = max(wmax, tt.get_window_extent(r).width * 25.4 / 72); tt.remove()
    left = 0.5 + wmax + 1.5 * 25.4 / 72 + 0.1
    ax = add_axes_mm(fig, w, h, left, bottom, right, top)
    ax.barh(np.arange(10), vals, height=BAR_FRAC, color=colors, edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_ylim(9.5, -0.5)
    ax.set_xlim(0, 1.05 * vals.max())
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_yticks(np.arange(10)); ax.set_yticklabels(terms)
    ax.set_xlabel(spec["xlabel"])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(INK)
    LOG.append(f"e: terms {terms}; -log10(FDR) {[round(float(v), 4) for v in vals]}; label column {wmax:.1f} mm; "
               f"axes {w - left - right:.1f} x {h - top - bottom:.1f} mm; bar {BAR_FRAC} of the slot")
    save_cell(fig, "e")
    return dict(terms=terms, vals=vals, colors=colors)


# ================================================================ f: FKBP5 in OPC / Oli vs score (p04 Fig 3f)
F_S, F_RING_S, F_RING_LW = 4.8, 17.5, 0.6      # p04 s 30 / ring s 110, lw 1.6 on an 8 in wide panel, scaled


def panel_f():
    fig, w, h, left, bottom, right, top = new_cell("f")
    ax = add_axes_mm(fig, w, h, left, bottom, right, top)
    rhos = {}
    for ct in ("OPC", "Oli"):
        y = FK[ct]
        ax.scatter(AK, y, s=F_S, color=D.CT_COL[ct], alpha=0.8, linewidths=0, label=ct, zorder=2)
        ax.scatter([AK[EXT]], [y[EXT]], s=F_RING_S, facecolors="none", edgecolors=V.EXTRING, linewidths=F_RING_LW, zorder=3)
        r, p, n = V.spearman(y, AK)
        rhos[ct] = (r, n)
    ax.set_xlabel(f"Component {D.K} loadings/score"); ax.set_ylabel(r"$\mathit{FKBP5}$ (CPM)")
    ax.legend(frameon=False, loc="upper left", markerscale=1.3, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.3)
    sns.despine(ax=ax)
    assert all(n == 274 for _, n in rhos.values()) and round(rhos["OPC"][0], 2) == 0.89 and round(rhos["Oli"][0], 2) == 0.83
    LOG.append(f"f: n {rhos['OPC'][1]}; Spearman OPC {rhos['OPC'][0]:+.3f}, Oli {rhos['Oli'][0]:+.3f} (legend 0.89, 0.83); "
               f"xlim {tuple(round(v, 3) for v in ax.get_xlim())}; ylim {tuple(round(v, 1) for v in ax.get_ylim())}")
    save_cell(fig, "f", extra_texts=list(ax.get_legend().get_texts()))
    return rhos


# ================================================================ fidelity checks against the current panel PDFs
def fit_map(src, dst):
    src = np.asarray(src, float); dst = np.asarray(dst, float); res = 0.0
    for c in (0, 1):
        A = np.c_[src[:, c], np.ones(len(src))]
        coef, *_ = np.linalg.lstsq(A, dst[:, c], rcond=None)
        res = max(res, float(np.abs(A @ coef - dst[:, c]).max()))
    return res


def match_points(ours, theirs, it=12):
    """Fit separate linear x and y maps (data -> PDF page) by iterated nearest-neighbour least squares (start: the
    extremes), then return, for every replot point, the distance (pt) to the nearest original marker, and the number
    of original markers left unmatched."""
    from scipy.spatial import cKDTree
    ours = np.asarray(ours, float); theirs = np.asarray(theirs, float)
    ax_ = np.ptp(theirs[:, 0]) / np.ptp(ours[:, 0]); bx = theirs[:, 0].min() - ax_ * ours[:, 0].min()
    ay_ = np.ptp(theirs[:, 1]) / np.ptp(ours[:, 1]); by = theirs[:, 1].min() - ay_ * ours[:, 1].min()
    t = cKDTree(theirs)
    for _ in range(it):
        P = np.c_[ax_ * ours[:, 0] + bx, ay_ * ours[:, 1] + by]
        dd, ii = t.query(P)
        g = dd <= np.quantile(dd, 0.9) + 1e-9
        ax_, bx = np.polyfit(ours[g, 0], theirs[ii[g], 0], 1); ay_, by = np.polyfit(ours[g, 1], theirs[ii[g], 1], 1)
    P = np.c_[ax_ * ours[:, 0] + bx, ay_ * ours[:, 1] + by]
    dd, ii = t.query(P)
    return float(dd.max()), int(len(theirs) - len(np.unique(ii))), int(len(np.unique(ii)))


def check_a(A):
    """Strip points of the corrected 278_sex_samples.pdf vs the replot (jitter and scores)."""
    mk = [(m[0], m[1]) for m in PP.markers(f"{PNEW}/g_ros/figs/outlier/278/278_sex_samples.pdf")]
    ours = np.vstack(A["offs"])
    dmax, unm, nmatch = match_points(ours, mk)
    LOG.append(f"check a vs panels_new 278_sex_samples.pdf: {len(mk)} strip points, replot {len(ours)}; every replot point "
               f"within {dmax:.3f} pt of an original marker after the linear page map; originals unmatched {unm}")
    assert len(mk) == len(ours) and dmax < 0.05 and unm == 0


def check_b(vals):
    rr = [q for q in PM.rects(f"{PNEW}/g_ros/figs/outlier/278/278_cell_type.pdf") if q[4] is not None and q[4] != (1.0, 1.0, 1.0)]
    assert len(rr) == 6, len(rr)
    hts = [q[3] for q in rr]
    k_ = np.polyfit(vals, hts, 1)
    res = float(np.abs(np.polyval(k_, vals) - np.array(hts)).max())
    cols_ok = all(np.allclose(q[4], mpl.colors.to_rgb(D.CT_COL[c]), atol=2e-3) for q, c in zip(rr, D.CT_ORDER))
    LOG.append(f"check b vs panels_new 278_cell_type.pdf: 6 bars, max residual of a linear map {res:.4f} pt; colours and order equal: {cols_ok}")
    assert res < 0.05 and cols_ok


def check_c(d):
    mk = PP.markers(f"{CODING}/outlier/278/278_RNA.pdf")       # XObject stamps + inline circles (1-point series)
    assert len(mk) == len(d["pts"]), (len(mk), len(d["pts"]))
    ours = np.array([(p[0], p[1]) for p in d["pts"]])
    dmax, unm, _ = match_points(ours, [(m[0], m[1]) for m in mk])
    # colours: pair by the same nearest-neighbour map
    ours_s = sorted([(p[0], p[1], tuple(round(v, 3) for v in mpl.colors.to_rgb(p[2]))) for p in d["pts"]])
    th_s = sorted([(m[0], m[1], m[2]) for m in mk])
    same = sum(a_[2] == b_[2] for a_, b_ in zip(ours_s, th_s))
    txt = f"{HERE}/orig_png/278_RNA.txt"
    os.system(f'pdftotext -raw "{CODING}/outlier/278/278_RNA.pdf" "{txt}"')
    words = open(txt).read().split()
    miss = [l[2] for l in d["labels"] if l[2] not in words]
    LOG.append(f"check c vs 278_RNA.pdf: {len(mk)} markers; every replot point within {dmax:.3f} pt of an original marker "
               f"(originals unmatched {unm}); colours equal {same}/{len(mk)}; labels found in the original text "
               f"{len(d['labels']) - len(miss)}/{len(d['labels'])}")
    assert dmax < 0.05 and unm == 0 and same == len(mk) and not miss and {"SOX6", "FKBP5"} <= {l[2] for l in d["labels"]}


def check_d(Dd):
    mk = [m for m in PP.markers(f"{CODING}/outlier/278/278_dlpfc_correlation.pdf") if m[2] is not None]
    ours = np.c_[Dd["x"], Dd["y"]]
    dmax, unm, _ = match_points(ours, [(m[0], m[1]) for m in mk])
    cols = {m[2] for m in mk}
    LOG.append(f"check d vs 278_dlpfc_correlation.pdf: {len(mk)} filled markers; every replot point within {dmax:.3f} pt of an "
               f"original marker (originals unmatched {unm}); original fill {sorted(cols)}; replot {tuple(round(v, 3) for v in D_COL)}")
    assert len(mk) == len(ours) and dmax < 0.05 and unm == 0 and cols == {tuple(round(v, 3) for v in D_COL)}


def check_e(E):
    """The 10-16 replot (panels_new/new/Fig3/Fig3e_C278_KEGG_v23.pdf): bar lengths and colours, read from its stream."""
    rr = [q for q in PM.rects(f"{PNEW}/new/Fig3/Fig3e_C278_KEGG_v23.pdf") if q[4] is not None and q[4] != (1.0, 1.0, 1.0)]
    rr = sorted(rr, key=lambda q: -q[1])[:10]
    lens = np.array([q[2] for q in rr]); k_ = lens[0] / E["vals"][0]
    res = float(np.abs(lens / k_ - E["vals"]).max())
    cols_ok = all(np.allclose(q[4], c, atol=2e-3) for q, c in zip(rr, E["colors"]))
    LOG.append(f"check e vs panels_new Fig3e_C278_KEGG_v23.pdf: 10 bars; max |length difference| {res:.5f} units; colours equal: {cols_ok}")
    assert res < 1e-3 and cols_ok


def check_f():
    mk = PP.markers(f"{PNEW}/new/Fig3/Fig3f_C278_FKBP5_OPC_Oli_vs_score.pdf")
    res = {}
    for ct in ("OPC", "Oli"):
        col = tuple(round(v, 3) for v in mpl.colors.to_rgb(D.CT_COL[ct]))
        th = [(m[0], m[1]) for m in mk if m[2] == col]
        ours = np.c_[AK, FK[ct]]
        dmax, unm, _ = match_points(ours, th)
        res[ct] = (len(th), dmax, unm)
        assert dmax < 0.05, (ct, dmax)
    LOG.append("check f vs panels_new Fig3f: " + "; ".join(f"{ct}: {n} original markers of that colour (274 points + legend "
               f"handle), every replot point within {dm:.3f} pt after the linear page map, originals unmatched {u}"
               for ct, (n, dm, u) in res.items()))


if __name__ == "__main__":
    A = panel_a()
    vb = panel_b()
    dc = panel_c()
    Dd = panel_d()
    E = panel_e()
    F = panel_f()
    for fn, arg in ((check_a, A), (check_b, vb), (check_c, dc), (check_d, Dd), (check_e, E)):
        fn(arg)
    check_f()
    json.dump({k: dict(zip(("x_mm", "y_mm", "w_mm", "h_mm"), GEOM[k][:4]), file=f"{PAN}/{NAMES[k]}.pdf") for k in GEOM},
              open(f"{HERE}/cells_Fig3.json", "w"), indent=1)
    open(f"{HERE}/f3_panels.log", "w").write("\n".join(LOG) + "\n")
    print("\n".join(LOG))
