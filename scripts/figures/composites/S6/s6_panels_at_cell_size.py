"""Figure S6 (Components 172 and 277 source data), panels redrawn at their final printed size for the python
composite (v23 supplement; assembled and lettered by composites/compose.py).

Same data, exclusions, groupings, colours, jitter seed and label text as panels/p05_neuronal.py, sections
"S6 a-c, g-i scatters" (scatter_view) and "S6 d-f (CERAD), j-l (diagnosis)" (boxes), which wrote the
panels_new/new/FigS6 files. The drawing code is copied from p05 (p05 cannot be imported: it writes its own outputs
at import); only print-size settings change:
  - each figure is exactly its composite cell (placed at scale 1 in the 142.9 mm wide figure), axes placed in mm;
  - explicit Arial sizes as the S3 build: tick labels 6 pt, axis labels 6.5 pt; thinner lines, smaller markers;
  - tick marks off, as in p05 (seaborn "white" style draws none).
Presentation-only differences from p05 (same words, same order):
  - c, f, i, l: the two-line y label "top-50 gene aggregate in <ct> / weighted mean z" is broken into three lines
    ("top-50 gene / aggregate in <ct> / weighted mean z") so it fits the axes height;
  - j, k, l: the six diagnosis tick labels are written on one line each ("AD dementia (n = 92)" for p05's
    "AD / dementia / (n = 92)") and rotated 90 degrees: six horizontal labels do not fit a one-third-width panel
    ("dementia" alone is 8.7 mm at 6 pt; the group spacing is about 5 to 6 mm).
Nothing else changed: data (n05_views.npy, the deposited scores and the SDA input CPM behind it), the 274 donors
(two identity-flagged snRNA-seq profiles excluded), the most extreme donor ringed in the scatters, CERAD split
(AD = ceradsc <= 2), cogdx order and colours (Set2 NCI/MCI/AD dementia, grey for the three untested groups),
the dotted separator, box statistics (median, IQR, whiskers 1.5 x IQR, no fliers), jitter (default_rng(5)).

Writes only under <G_ROS_FIGURE_OUT>/S6/panels/. Logs aggregates only (no donor IDs, no per-donor values)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
from vp_common import *          # style(): seaborn white/talk, Arial, pdf.fonttype 42; RA, CTCOL, EXTRING, clinical ...
import numpy as np, pandas as pd
from scipy import stats as ss

HERE = f"{FIGOUT}/S6"
PAN = f"{HERE}/panels"; os.makedirs(PAN, exist_ok=True)
MM = 1 / 25.4
TICK, LAB = 6.0, 6.5
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "font.size": LAB, "axes.labelsize": LAB, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
                     "legend.fontsize": TICK, "axes.linewidth": 0.6, "lines.linewidth": 0.8,
                     "xtick.major.pad": 1.8, "ytick.major.pad": 1.5, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.labelpad": 2.0,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

# ---------------------------------------------------------------- geometry (mm); the layout JSON uses the same numbers
W = 142.9
GUT_X = 2.5                                   # gutter between columns
CW = (W - 2 * GUT_X) / 3                      # cell width
TOP = 3.0                                     # strip above the axes (panel letter)
AXH = 23.5                                    # axes height, all rows
RIGHT = 1.5
LEFT = {1: 8.8, 2: 8.8, 3: 12.3}            # per column; column 3 carries the three-line aggregate y label
BOTTOM = {"scatter": 6.2, "cerad": 8.2, "dx": 25.0}
CELL_H = {k: TOP + AXH + v for k, v in BOTTOM.items()}

LOG = []
V = np.load(f"{RA}/neuronal_c21/cache/n05_views.npy", allow_pickle=True).item()
clin = clinical()
SET2 = sns.color_palette("Set2"); GREYB = "#9e9e9e"            # as p05
DXCOL = {1: SET2[0], 2: SET2[2], 4: SET2[1]}


def ylabel_for(view, ct):
    # p05: f"{view} in {ct} (CPM)" or f"top-50 gene aggregate in {ct}\nweighted mean z" (line breaks only changed)
    return f"{view} in {ct} (CPM)" if view != "aggregate" else f"top-50 gene\naggregate in {ct}\nweighted mean z"


def new_cell(col, kind):
    w, h = CW, CELL_H[kind]
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([LEFT[col] / w, 1 - (TOP + AXH) / h, (w - LEFT[col] - RIGHT) / w, AXH / h])
    return fig, ax


def bbox_mm(fig, artist):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bb = artist.get_window_extent(r); h = fig.get_size_inches()[1] * fig.dpi; s = 25.4 / fig.dpi
    return (bb.x0 * s, (h - bb.y1) * s, bb.x1 * s, (h - bb.y0) * s)


def check_cell(fig, name, keep_clear_mm=(3.0, 2.8)):
    """Everything drawn inside the cell; no text in the letter corner; tick labels on one axis do not overlap."""
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
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            bbs = [bbox_mm(fig, t) for t in drawn_ticklabels(axis)]
            for i in range(len(bbs)):
                for j in range(i + 1, len(bbs)):
                    a_, b_ = bbs[i], bbs[j]
                    over = a_[0] < b_[2] and b_[0] < a_[2] and a_[1] < b_[3] and b_[1] < a_[3]
                    assert not over, (name, "tick labels overlap", i, j)
        # axis labels vs tick labels of the same axis
        yl = bbox_mm(fig, ax.yaxis.label)
        for t in drawn_ticklabels(ax.yaxis):
            b_ = bbox_mm(fig, t)
            assert yl[2] <= b_[0] + 0.01, (name, "y label overlaps tick labels", yl, b_, t.get_text())
        xl = bbox_mm(fig, ax.xaxis.label)
        for t in drawn_ticklabels(ax.xaxis):
            b_ = bbox_mm(fig, t)
            assert xl[1] >= b_[3] - 0.01, (name, "x label overlaps tick labels", xl, b_, t.get_text())


def drawn_ticklabels(axis):
    """Tick labels matplotlib actually draws: non-empty, at a tick inside the view interval."""
    lo, hi = sorted(axis.get_view_interval())
    eps = (hi - lo) * 1e-9
    out = []
    for loc, t in zip(axis.get_majorticklocs(), axis.get_ticklabels()):
        if lo - eps <= loc <= hi + eps and t.get_visible() and t.get_text().strip():
            out.append(t)
    return out


def save_cell(fig, name):
    check_cell(fig, name)
    fig.savefig(f"{PAN}/{name}.pdf")          # exact cell size: no bbox_inches
    plt.close(fig); LOG.append(f"wrote panels/{name}.pdf")


# ---------------------------------------------------------------- a-c, g-i scatters (p05 scatter_view)
S_PT, RING_S, RING_LW = 3.0, 11.0, 0.6        # p05 at its 177 mm size: s 30, ring s 110 (x 3.67), lw 1.6

def scatter_view(k, view, fname, col):
    c = V[k]; a = c["a"]; y = np.asarray(c["views"][view], float); ct = c["ct"]; ext = most_extreme(a)
    assert ext == c["ext"] and len(a) == 274
    fig, ax = new_cell(col, "scatter")
    ax.scatter(a, y, s=S_PT, color=CTCOL[ct], alpha=0.8, linewidths=0, zorder=2)
    ax.scatter([a[ext]], [y[ext]], s=RING_S, facecolors="none", edgecolors=EXTRING, linewidths=RING_LW, zorder=3)
    ax.set_xlabel(f"Component {k} loadings/score")
    ax.set_ylabel(ylabel_for(view, ct))
    sns.despine(ax=ax)
    save_cell(fig, fname)
    r, p, n = spearman(y, a); m = np.ones(len(a), bool); m[ext] = False
    LOG.append(f"{fname}: n {n}; rho {r:+.3f} (p {p:.2g}); without most extreme donor {ss.spearmanr(y[m], a[m])[0]:+.3f}")
    return r, n


RHO = {}
for col, (letter, view) in enumerate(zip("abc", ["LRP8", "SYN3", "aggregate"]), start=1):
    RHO[letter] = scatter_view(172, view, f"S6{letter}_C172_{view}_Inh_vs_score_cell", col)
for col, (letter, view) in enumerate(zip("ghi", ["PLCB1", "LRP1B", "aggregate"]), start=1):
    RHO[letter] = scatter_view(277, view, f"S6{letter}_C277_{view}_Exc_vs_score_cell", col)

# legend values (supp-template.tex, S6): a 0.79, b -0.61, c 0.94, g 0.61, h -0.53, i 0.76; n 274
for letter, want in zip("abcghi", [0.79, -0.61, 0.94, 0.61, -0.53, 0.76]):
    r, n = RHO[letter]
    assert n == 274 and abs(r - want) <= 0.0051, (letter, r, want)
LOG.append("scatter rho values match the legend to 2 dp, n 274 each")


# ---------------------------------------------------------------- d-f (CERAD), j-l (diagnosis) (p05 boxes)
PT_S = 1.5                                    # p05: s 12 at its 151-228 mm size

def boxes(ax, groups, labels, colours, ylab, rotate=False):
    data = [g for g in groups]
    bp = ax.boxplot(data, positions=range(len(data)), widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#333333", lw=0.8), whiskerprops=dict(color="#333333", lw=0.5),
                    capprops=dict(color="#333333", lw=0.5), boxprops=dict(edgecolor="#333333", lw=0.5))
    for patch, col in zip(bp["boxes"], colours): patch.set_facecolor(col)
    rng = np.random.default_rng(5)
    for i, g in enumerate(data):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g, s=PT_S, color="black", alpha=0.85, linewidths=0, zorder=3)
    ax.set_xticks(range(len(data)))
    if rotate:
        ax.set_xticklabels(labels, fontsize=TICK, rotation=90, ha="center", va="top")
    else:
        ax.set_xticklabels(labels, fontsize=TICK)
    ax.set_ylabel(ylab); sns.despine(ax=ax)


for k, letters in ((172, "def"), (277, "jkl")):
    c = V[k]; idx = c["donors_idx"]; cl = clin.iloc[idx]; ct = c["ct"]
    for col, (letter, view) in enumerate(zip(letters, list(c["views"])), start=1):
        y = np.asarray(c["views"][view], float)
        ylab = ylabel_for(view, ct)
        if k == 172:
            ad = cl.ceradsc.values <= 2
            g = [y[~ad], y[ad]]
            assert (int((~ad).sum()), int(ad.sum())) == (106, 168)
            fig, ax = new_cell(col, "cerad")
            boxes(ax, g, [f"non-AD\n(n = {int((~ad).sum())})", f"AD\n(n = {int(ad.sum())})"], [SET2[0], SET2[1]], ylab)
            ax.set_xlabel("CERAD score")
            LOG.append(f"S6{letter} C172 {view}: non-AD n {int((~ad).sum())}, AD n {int(ad.sum())}")
        else:
            # p05 labels "NCI", "MCI", "AD\ndementia", "MCI +\nother", "AD +\nother", "other\ndementia", each + "\n(n = ..)";
            # here the same words on one line each (rotated)
            order = [(1, "NCI"), (2, "MCI"), (4, "AD dementia"), (3, "MCI + other"), (5, "AD + other"), (6, "other dementia")]
            cg = cl.cogdx.values
            g = [y[cg == v] for v, _ in order]
            assert [len(x) for x in g] == [96, 64, 92, 3, 12, 7]
            fig, ax = new_cell(col, "dx")
            boxes(ax, g, [f"{lab} (n = {int((cg == v).sum())})" for v, lab in order], [DXCOL[1], DXCOL[2], DXCOL[4]] + [GREYB] * 3,
                  ylab, rotate=True)
            ax.axvline(2.5, color="grey", lw=0.5, ls=":")
            ax.set_xlabel("final clinical diagnosis")
            LOG.append(f"S6{letter} C277 {view}: " + ", ".join(f"{lab} n {int((cg == v).sum())}" for v, lab in order))
        save_cell(fig, f"S6{letter}_C{k}_{view}_{ct}_by_{'CERAD' if k == 172 else 'diagnosis'}_cell")

Pt = pd.read_csv(f"{RA}/neuronal_c21/out/n05_source_vs_phenotype.csv")
for _, r in Pt.iterrows():
    LOG.append(f"S6 phenotype (n05, verified VNC v04; not drawn, legend only): C{r.component} {r.view} {r.term} n {r.n}: "
               f"{r.beta_SD:+.2f} SD [{r.lo:+.2f}, {r.hi:+.2f}], p {r.p:.2g}")
LOG.append("geometry (mm): " + f"CW {CW:.3f}, gutter {GUT_X}, cell heights {CELL_H}, left {LEFT}, top {TOP}, axes h {AXH}")
open(f"{PAN}/s6_panels_at_cell_size.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
