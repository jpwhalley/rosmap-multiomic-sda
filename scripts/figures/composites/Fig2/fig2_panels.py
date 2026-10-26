"""Main Figure 2 (Component 74), panels a-g redrawn at their final printed size for the python composite
(v23 main text; assembled and lettered by composites/compose.py). Each PDF is exactly its
composite cell and is placed at scale 1 in the 180 mm wide figure, so the sizes set here are the printed sizes at 180 mm.
Layout as the Figure 2 first assembled in Illustrator: rows a b c / d e / f g.

Sources and recipes (all read only):
  a  corrected 74_sex_samples.pdf -> qlabel_panels/qlabel_panels.py
     (port of ros6a_cross_clin_discrete.py l.715-768): same data (samples_final.csv + ROSMAP_clinical.csv), +/-5 SD
     filter, seaborn-0.12 box shim (Set2, desat 0.75, quantised grey lines), statannotations bracket, black strip
     points with the recovered original jitter (recovered_jitter.json), q = BH over the 547-family
     (clinical_BH_547_family.csv). CHANGED: label 'p = 7.926e-44' -> 'FDR = 7.9 x 10^-44'
     (superscript exponent).
  b  corrected 74_cell_type.pdf -> ros3b l.326-340:
     ROSMAP_cell_type_panels_values.csv cor_* (C74; asserted equal to cell_type_loadings_corrected.csv),
     order Ast Exc Inh Mic OPC Oli, ros3b colours, y label 'loadings/score', full box. All six bars one sign.
  c  fig4c_labelled.pdf -> panels/round1/fig4c_labelled.py: fig4c_points.csv (G_ROS_ROUND1_DIR; 138
     genes), the same colours, highlighted five genes (XIST, TTTY14, NLGN4Y, UTY, USP9Y), y = -x dotted line, zero
     lines, symmetric limits, axis titles and the 'r = -0.982  n = 138' note. Labels italic (whole symbol; the original
     mathtext left the digits of TTTY14 upright), leader lines as the original; Arial instead of DejaVu (Type 3).
  d-f version-1 74_RNA.pdf, 74_chip.pdf, 74_meth.pdf -> ros3b_plot_outliers.py via fig2_data.py
     (copy of composites/S10/s10_data.py with K = 74): PIP > 0.5 features at genomic position, ros3b
     colours, 8 largest (black) and 8 smallest (grey) labelled, ros3b y limits, chromosome ticks rotated 90 degrees,
     full box. Labels placed by label_placer.py (from composites/S8; adjustText is not in the environment) with leader lines.
     Chromosome tick labels that would touch at this width are left blank (listed in the log).
  g  S4b_cpg_and_H3K9ac_vs_score.pdf content as drawn for Fig 2g by
     panels/p03b_fig2g_compact.py (same data, exclusions and aggregation as p03_c74.py): cg26921482 beta
     (n 216), H3K9ac weighted mean z at the 537 chrX and 120 autosomal C74 peaks (n 189), against the C74 score,
     coloured by recorded sex; x title on the middle plot, Male/Female key lower left of the first plot.
     Gene symbol italic in the axis title (AMDHD2).

Print-size settings (presentation only): Arial 7 pt for all text (Manhattan feature labels LABPT below), axes lines
0.6 pt, markers and lines scaled to the cell. Writes only under <G_ROS_FIGURE_OUT>/Fig2/panels/. Logs aggregates only
(no donor IDs, no per-donor values).
Final version: two fixes after a check of the first build, marked 'fix'. The final panels are drawn in three runs
(environment options as listed, see build_all.sh): 'b c f g' with no options; 'a d' with FIG2_NUDGE1="DDX3Y:-3:-0.4";
'e' with FIG2_EYMIN=-6.5 FIG2_EDGEB=1.8 FIG2_LCLEAR=1.5 FIG2_K=800 FIG2_SEED3=1 FIG2_SIDEPEN=100000.
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \\
     python fig2_panels.py [panel letters]
"""
import os, sys, json, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
from vp_common import *          # noqa: F401,F403 score, female, excl_mask, mod_input, comp_loadings, zagg, MALE, FEMALE, ...
import numpy as np, pandas as pd
from scipy import stats as ss
from matplotlib.lines import Line2D
import fig2_data as D
import label_placer

QDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "qlabel_panels")       # scripts/figures/qlabel_panels
_spec = importlib.util.spec_from_file_location("qlabel_panels", os.path.join(QDIR, "qlabel_panels.py"))
qp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(qp)   # Agg, fonttype 42, vendor/ statannotations

from figure_paths import ROUND1
PAN = f"{FIGOUT}/Fig2/panels"; os.makedirs(PAN, exist_ok=True)
MM = 1 / 25.4
PT_PER_MM = 72 / 25.4
FS = 7.0                    # all text
LABPT = float(os.environ.get("FIG2_LABPT", "7.0"))   # Manhattan feature labels
LOG = []


def style():
    sns.set(style="white", context="paper")          # v22 panels: seaborn 'white' (colours .15, full box)
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial", "mathtext.default": "regular",
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
        "axes.linewidth": 0.6, "lines.linewidth": 0.8, "patch.linewidth": 0.3,
        "xtick.major.pad": 1.6, "ytick.major.pad": 1.4, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.labelpad": 1.8,
        "legend.handletextpad": 0.3, "legend.borderpad": 0.3, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "figure.dpi": 72})


style()

# ------------------------------------------------------------------ geometry (mm); layout_Fig2.json uses the same numbers
W = 180.0
ROW_Y = [0.0, 48.0, 102.0]; ROW_H = [48.0, 54.0, 48.0]
H = ROW_Y[-1] + ROW_H[-1]                                     # 150 mm
TOP = 2.5                                                     # cell top -> axes top
BOT1 = 9.0                                                    # row 1: axes bottom -> cell bottom (tick labels + x title)
BOT = 8.6                                                     # rows 2-3
CELLS = dict(a=(0.0, 0, 48.0), b=(48.0, 0, 66.0), c=(114.0, 0, 66.0),
             d=(0.0, 1, 82.0), e=(82.0, 1, 98.0), f=(0.0, 2, 82.0), g=(82.0, 2, 98.0))   # x, row, width
AXL = 12.0                                                    # axes left edge inside the cell (a, d, e, f)
AXR_PAD = {"d": 2.5, "e": 1.2, "f": 2.5}                       # axes right edge -> cell right edge


def cell(letter):
    x, r, w = CELLS[letter]
    return x, ROW_Y[r], w, ROW_H[r]


def fig_mm(w, h):
    return plt.figure(figsize=(w * MM, h * MM), dpi=72)


def ax_mm(fig, x, y, w, h, **kw):
    """Axes with top-left corner at (x, y) mm inside the cell (y down), size w x h mm."""
    Wc, Hc = fig.get_size_inches() / MM
    return fig.add_axes([x / Wc, 1 - (y + h) / Hc, w / Wc, h / Hc], **kw)


def savep(fig, name):
    fig.savefig(f"{PAN}/{name}.pdf")          # exact cell size: no bbox_inches
    plt.close(fig); LOG.append(f"wrote panels/{name}.pdf")



EXP_PT, EXP_RAISE = 6.5, 3.0      # superscript exponent: 6.5 pt (the minimum), baseline raised 3 pt


def add_exponent(fig, ax, base, expo):
    """Append a superscript exponent to the statannotations label whose text is `base`: the label is shifted left by
    half the exponent width (so the whole label stays centred over the bracket) and the exponent is drawn at EXP_PT,
    its baseline EXP_RAISE pt above the label baseline. The y top is raised only if the exponent would touch the frame."""
    t = [a_ for a_ in ax.texts if a_.get_text() == base]
    assert len(t) == 1
    t = t[0]
    fig.canvas.draw(); R = fig.canvas.get_renderer()
    from matplotlib.font_manager import FontProperties
    we = R.get_text_width_height_descent(expo, FontProperties(family="Arial", size=EXP_PT), ismath=False)[0]
    gap = 0.4
    # fix: anchor the label on its baseline (it was va='bottom', whose baseline offset is the layout
    # descent, measured here only to whole 72-dpi pixels; the exponent landed 1.6 pt up instead of 3 pt). With both
    # texts va='baseline' at the same anchor, the exponent baseline is exactly EXP_RAISE above the label baseline.
    desc = t._get_layout(R)[2]
    t.set_va("baseline")
    t.xyann = (t.xyann[0] - (we + gap) / 2, t.xyann[1] + desc)
    fig.canvas.draw()
    bb = t.get_window_extent(R)
    A = ax.transData.transform(t.xy)
    e = ax.annotate(expo, xy=t.xy, xycoords="data", textcoords="offset points",
                    xytext=(bb.x1 - A[0] + gap, t.xyann[1] + EXP_RAISE), ha="left", va="baseline",
                    fontsize=EXP_PT, color=t.get_color(), clip_on=False, annotation_clip=False)
    fig.canvas.draw()
    eb = e.get_window_extent(R); axb = ax.get_window_extent(R)
    raised = False
    if eb.y1 > axb.y1 - 2.5:
        y0, y1 = ax.get_ylim()
        need = (eb.y1 - (axb.y1 - 2.5)) / axb.height * (y1 - y0)
        ax.set_ylim(y0, y1 + need * 1.05); raised = True
        fig.canvas.draw(); eb = e.get_window_extent(R); axb = ax.get_window_extent(R)
    return f"exponent top {axb.y1 - eb.y1:.1f} pt below the frame" + ("; y top raised" if raised else "")


# ================================================================== a: C74 score by sex (ros6a / qlabel port)
def panel_a():
    _, _, w, h = cell("a")
    k = 74
    samples = qp.load_samples()
    q = float(pd.read_csv(qp.BH547, index_col=0).loc[k, "Sex"])
    assert f"{q:.4g}" == "7.926e-44"
    mant, ex = f"{q:.1e}".split("e")
    base, expo = f"FDR = {mant} × 10", f"{int(ex)}".replace("-", "\u2212")
    label = f"{base}^{expo}"
    assert (base, expo) == ("FDR = 7.9 × 10", "\u221244"), label
    jit = json.load(open(os.path.join(QDIR, "recovered_jitter.json")))["74_sex_samples.pdf"]
    fs = qp.outlier_filter(samples, k)
    fig = fig_mm(w, h)
    ax = ax_mm(fig, AXL, TOP, w - AXL - 1.0, h - TOP - BOT1)
    lw = 0.7                                   # ros6a: rcParams lines.linewidth (talk 2.25) at about 0.3 x the size
    qp.boxplot_seaborn012("Sex", k, fs, ax, ["Male", "Female"], "Set2", showfliers=False, linewidth=lw)
    ann = qp.Annotator(ax, [("Male", "Female")], data=fs, x="Sex", y=k, order=["Male", "Female"])
    ann.set_pvalues([q])
    ann.configure(text_format="simple", show_test_name=False, fontsize=FS, line_width=0.6)
    ann.set_custom_annotations([base])
    ann.annotate()
    sup = add_exponent(fig, ax, base, expo)
    qp.stripplot(jit, 0, x="Sex", y=k, data=fs, ax=ax, jitter=True, color="black", size=1.9)
    ax.set_xlabel("Sex"); ax.set_ylabel("loadings/score")
    ax.tick_params(length=0)
    # fidelity: the strip offsets equal those of the 10-08 port (qp.panel_sex, same jitter), in data units
    fig_r, ax_r, fs_r, lab_r = qp.panel_sex(samples, k, q, jit, 0)
    off = [np.asarray(c.get_offsets(), float) for c in ax.collections if type(c).__name__ == "PathCollection"]
    off_r = [np.asarray(c.get_offsets(), float) for c in ax_r.collections if type(c).__name__ == "PathCollection"]
    assert len(off) == len(off_r) == 2 and all(np.array_equal(a_, b_) for a_, b_ in zip(off, off_r))
    box = [np.asarray(p.get_path().vertices, float) for p in ax.patches]
    box_r = [np.asarray(p.get_path().vertices, float) for p in ax_r.patches]
    assert all(np.allclose(a_, b_) for a_, b_ in zip(box, box_r)) and len(box) == len(box_r)
    assert [p.get_facecolor() for p in ax.patches] == [p.get_facecolor() for p in ax_r.patches]
    plt.close(fig_r)
    n = fs["Sex"].value_counts()
    savep(fig, "Fig2a_C74_sex_samples")
    LOG.append(f"a: n male {n['Male']}, female {n['Female']} (filter +/-5 SD kept {len(fs)} of {len(samples)}); "
               f"q (547 family) {q:.4g}; label '{label}' (exponent {EXP_PT} pt, raised {EXP_RAISE} pt; {sup}); "
               f"strip points and boxes identical to the 10-08 port; "
               f"ylim {tuple(round(v, 4) for v in ax.get_ylim())}")


# ================================================================== b: corrected cell-type loadings
def panel_b():
    _, _, w, h = cell("b")
    order, vals = D.celltype()
    fig = fig_mm(w, h)
    l = 11.0
    ax = ax_mm(fig, l, TOP, w - l - 1.5, h - TOP - BOT1)
    pos = np.arange(len(order))
    ax.bar(pos, vals, align="center", color=[D.CT_COL[c] for c in order], width=0.8)
    ax.set_xticks(pos); ax.set_xticklabels(order)
    ax.set_ylabel("loadings/score")
    ax.tick_params(length=0)
    assert all(v > 0 for v in vals) or all(v < 0 for v in vals)
    savep(fig, "Fig2b_C74_cell_type")
    LOG.append("b: corrected cell-type loadings " + ", ".join(f"{c} {v:.3f}" for c, v in zip(order, vals)) + " (all one sign)")


# ================================================================== c: ROSMAP vs SEA-AD loadings (fig4c_labelled)
def panel_c():
    _, _, w, h = cell("c")
    df = pd.read_csv(f"{ROUND1}/fig4c_points.csv", index_col=0)
    assert len(df) == 138
    r = np.corrcoef(df["ROSMAP"], df["SEA_AD"])[0, 1]
    assert f"{r:.3f}" == "-0.982"
    labels = ["XIST", "TTTY14", "NLGN4Y", "UTY", "USP9Y"]
    fig = fig_mm(w, h)
    l = 14.0
    aw, ah = w - l - 4.5, h - TOP - BOT1
    ax = ax_mm(fig, l, TOP, aw, ah)
    sc = (aw / 25.4) / 3.75                                  # original axes about 3.75 in wide
    ax.scatter(df["ROSMAP"], df["SEA_AD"], s=14 * sc ** 2 * 1.6, color="#4C72B0", alpha=0.55, edgecolor="none", zorder=2)
    for g in labels:
        ax.scatter([df.loc[g, "ROSMAP"]], [df.loc[g, "SEA_AD"]], s=32 * sc ** 2 * 1.6, color="#222222",
                   edgecolor="white", linewidth=0.3, zorder=4)
    # fig4c_labelled.py offsets (data units); NLGN4Y/UTY/TTTY14 nudged apart for 7 pt at this size
    # (original: XIST (100, -100), TTTY14 (80, -30), NLGN4Y (130, 30), UTY (130, 30), USP9Y (-140, 30); at 7 pt in
    # this axes the UTY/NLGN4Y/TTTY14 labels would overlap, so they are fanned into the empty area above the y = -x line,
    # and USP9Y moves left of the x = 0 line)
    offs = {"XIST": (100, -100), "UTY": (200, 520), "NLGN4Y": (300, 290), "TTTY14": (200, 120), "USP9Y": (-330, -200)}
    for g, (dx, dy) in offs.items():
        x, y = df.loc[g, "ROSMAP"], df.loc[g, "SEA_AD"]
        ax.annotate(g, xy=(x, y), xytext=(x + dx, y + dy), fontsize=FS, fontstyle="italic",
                    ha="left" if dx > 0 else "right", va="center", color="#111111",
                    arrowprops=dict(arrowstyle="-", color="#666", lw=0.4, shrinkA=1, shrinkB=1.5), zorder=5)
    lim = max(abs(df["ROSMAP"].min()), abs(df["ROSMAP"].max()), abs(df["SEA_AD"].min()), abs(df["SEA_AD"].max())) * 1.05
    ax.plot([-lim, lim], [lim, -lim], color="#bbb", linewidth=0.5, linestyle=":", zorder=1)
    ax.axhline(0, color="#999", linewidth=0.4, zorder=1); ax.axvline(0, color="#999", linewidth=0.4, zorder=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("ROSMAP Component 74 RNA loading"); ax.set_ylabel("SEA-AD sex-component loading")
    ax.text(0.03, 0.05, r"$\mathit{r}$ = −0.982   $\mathit{n}$ = 138", transform=ax.transAxes, va="bottom", ha="left",
            fontsize=FS, color="#333")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="both", length=2.0, width=0.6, pad=1.4, color=".15")
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1000)); ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(500))
    savep(fig, "Fig2c_C74_SEAAD_scatter")
    LOG.append(f"c: {len(df)} genes; Pearson r {r:.3f}; limits +/-{lim:.0f}; labelled {labels}")


# ================================================================== d-f: loadings by chromosome (ros3b)
MAN = {1: ("d", "Fig2d_C74_RNA"), 3: ("e", "Fig2e_C74_chip"), 2: ("f", "Fig2f_C74_meth")}


def blank_colliding_ticks(fig, ax, labels_txt, gap_pt=0.5):
    """Leave blank the chromosome tick labels whose ink would touch the previous kept label (rotated 90 degrees: the
    ink of a digit label is about 0.73 em wide across the axis). Greedy left to right; 1, X and Y are always kept (the
    autosome before them is blanked instead). Returns the blanked labels and the closest kept pair (pt)."""
    fig.canvas.draw()
    xs = ax.transData.transform(np.c_[ax.get_xticks(), np.zeros(len(ax.get_xticks()))])[:, 0]
    ink = 0.73 * mpl.rcParams["xtick.labelsize"]
    prot = ("1", "X", "Y")
    keep = [True] * len(xs); kept = []
    for i, t in enumerate(labels_txt):
        while kept and xs[i] - xs[kept[-1]] < ink + gap_pt:
            if t in prot and labels_txt[kept[-1]] not in prot:
                keep[kept[-1]] = False; kept.pop(); continue
            if t not in prot:
                keep[i] = False
            break
        if keep[i]:
            kept.append(i)
    ax.set_xticklabels([t if k_ else "" for t, k_ in zip(labels_txt, keep)], rotation=90)
    gaps = np.diff(xs[kept]) - ink
    # X and Y are both kept; if they nearly touch, nudge them apart symmetrically (presentation only)
    ix, iy = labels_txt.index("X"), labels_txt.index("Y")
    need = (ink + 1.2) - (xs[iy] - xs[ix])
    if need > 0:
        import matplotlib.transforms as mtrans
        tl = ax.get_xticklabels()
        for i_, sgn in ((ix, -1), (iy, 1)):
            tl[i_].set_transform(tl[i_].get_transform() + mtrans.ScaledTranslation(sgn * need / 2 / 72, 0, fig.dpi_scale_trans))
        gaps = np.where(np.isclose(gaps, xs[iy] - xs[ix] - ink), 1.2, gaps)
    return [t for t, k_ in zip(labels_txt, keep) if not k_], float(gaps.min())



def label_quality(fig, ax, pts, res):
    """Counts for the placed labels (display pt): leaders crossing another label, leader pairs crossing, plotted
    points under a label (other than its own), labels touching each other."""
    T = ax.transData; P = T.transform(np.asarray(pts, float))
    boxes = np.array([r_["box"] for r_ in res], float); n = len(res)
    lead = [(np.array(r_["point"]), np.array(r_["q"])) if r_["leader_pt"] > 0 else None for r_ in res]
    ll = lc = 0
    for i in range(n):
        if lead[i] is None:
            continue
        for j in range(n):
            if j != i and label_placer._seg_rect(lead[i][0], lead[i][1], boxes[j]):
                ll += 1; LOG.append(f"   leader of {res[i]['text']} passes through the box of {res[j]['text']}")
            if j > i and lead[j] is not None and label_placer._seg_cross(lead[i][0], lead[i][1], lead[j][0], lead[j][1]):
                lc += 1
    under = 0
    for i in range(n):
        b = boxes[i]; own = np.array(res[i]["point"])
        inside = (P[:, 0] > b[0] - 0.5) & (P[:, 0] < b[2] + 0.5) & (P[:, 1] > b[1] - 0.5) & (P[:, 1] < b[3] + 0.5)
        inside &= np.hypot(P[:, 0] - own[0], P[:, 1] - own[1]) > 0.5
        under += int(inside.sum())
    return f"leader-through-label {ll}, leader crossings {lc}, points under labels {under}"


def panel_manhattan(j):
    letter, name = MAN[j]
    _, _, w, h = cell(letter)
    d = D.manhattan(j)
    fig = fig_mm(w, h)
    aw, ah = w - AXL - AXR_PAD[letter], h - TOP - BOT
    ax = ax_mm(fig, AXL, TOP, aw, ah)
    scale = (aw * PT_PER_MM) / 750.75                     # ros3b axes width at 12 x 6 in (talk)
    ms = {1: 3.4, 3: 3.0, 2: 2.4}[j]                       # '.' markersize (ros3b: talk default 9 at 750 pt -> about 2.6)
    L = D.pip_features(j); info = D.annotation(j, L.index)
    shift = np.array([0.0])
    for ii, ch in enumerate(D.CHRS):                       # same draw order and colours as ros3b
        sub = info[info["chrom"] == ch]
        col = D.COLORS[j][ii % 4]
        if len(sub) == 0:
            xx = shift[-1] + np.array([D.CHR_MAX[ch]]); ax.plot(xx, [0], ".", color=col, markersize=0)
        else:
            xx = shift[-1] + sub["pos"].values.astype(float); yy = L.loc[sub["fid"].values].values
            ax.plot(xx, yy, ".", color=col, markersize=ms)
        shift = np.append(shift, np.max(xx))
    ax.set_xticks(d["ticks"]); ax.set_xticklabels(D.CHRS, rotation=90)
    ylim = d["ylim"]
    if j == 3 and os.environ.get("FIG2_EYMIN"):
        # fix: extra room under the chrX cluster so the grey labels clear the bottom spine
        ylim = (float(os.environ["FIG2_EYMIN"]), ylim[1])
    if j == 2 and os.environ.get("FIG2_FYMAX"):
        # option: room above the CpG band for a third row of labels
        ylim = (ylim[0], float(os.environ["FIG2_FYMAX"]))
    ax.set_ylim(*ylim)
    ax.tick_params(length=0)
    ax.set_xlabel(D.XLAB[j]); ax.set_ylabel("loadings/score")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    blank, mingap = blank_colliding_ticks(fig, ax, D.CHRS)
    pts = np.array([(p[0], p[1]) for p in d["pts"]])
    labs = d["labels"]
    labs = [labs[i] for i in np.argsort([-abs(l[1]) for l in labs])]
    common = dict(fontsize=LABPT, marker_r_pt=0.25 * ms + 0.2, step=float(os.environ.get('FIG2_STEP', '3.0')),
                  K=int(os.environ.get('FIG2_K', '550')), lead_lw=0.3,
                  pad_x=float(os.environ.get('FIG2_PADX', '2.5')), pad_y=float(os.environ.get('FIG2_PADY', '0.6')),
                  c_lead_label=2500.0, c_cross=1500.0, side_y=ax.transData.transform((0, 0))[1], side_pen=300.0,
                  w_pts=float(os.environ.get(f'FIG2_WPTS{j}', {1: 20.0, 3: 150.0, 2: 12.0}[j])),
                  seed=int(os.environ.get(f'FIG2_SEED{j}', '1')))
    if j == 3:
        # ChIP: the 8 largest peaks are one tight chrY cluster at the top right -> a deterministic sorted column;
        # the 8 smallest (one chrX cluster at the bottom right) are then placed by the annealer with the column, its
        # leaders and the plotted points as obstacles
        top = [l for l in labs if l[3] == "black"]; bot = [l for l in labs if l[3] == "gray"]
        res_t, boxes = label_placer.stack_place(fig, ax, pts, top, LABPT, marker_r_pt=0.25 * ms + 0.2, gap_pt=1.0)
        inv = ax.transData.inverted(); obst = []
        for b_, r_ in zip(boxes, res_t):
            gx, gy = np.meshgrid(np.arange(b_[0], b_[2], 2.0), np.arange(b_[1], b_[3], 2.0))
            obst += list(inv.transform(np.c_[gx.ravel(), gy.ravel()]))
            p_, q_ = np.array(r_["point"]), np.array(r_["q"])
            for t_ in np.linspace(0, 1, max(2, int(np.hypot(*(q_ - p_)) / 2))):
                obst.append(inv.transform(p_ + t_ * (q_ - p_)))
        pts_b = np.vstack([pts, np.array(obst)])
        common["side_pen"] = float(os.environ.get('FIG2_SIDEPEN', '2000'))
        common["c_lead_label"] = float(os.environ.get('FIG2_CLL', '20000'))
        common["c_cross"] = float(os.environ.get('FIG2_CX', '300'))
        # fix: keep the chrX labels off the axes frame (they sat on the bottom spine) and keep leaders
        # clear of other labels' text (they ran along the gaps between label rows)
        common["edge_pad"] = float(os.environ.get('FIG2_EDGE', '0.6'))
        common["edge_pad_bottom"] = float(os.environ.get('FIG2_EDGEB', '2.0'))
        common["lead_clear"] = float(os.environ.get('FIG2_LCLEAR', '0.0'))
        res_b, tot = label_placer.place(fig, ax, pts_b, bot, **common)
        res = res_t + res_b; labs = top + bot
    else:
        # fix: leaders kept clear of other labels' text (d: the KDM5D leader grazed DDX3Y; f: the
        # cg02530860 leader ran along the top of cg11240062); default 0 = the builder's placement
        common["lead_clear"] = float(os.environ.get(f'FIG2_LCLEAR{j}', '0'))
        common["K"] = int(os.environ.get(f'FIG2_K{j}', common["K"]))
        common["edge_pad"] = float(os.environ.get(f'FIG2_EDGE{j}', '0.6'))
        common["side_pen"] = float(os.environ.get(f'FIG2_SIDEPEN{j}', common["side_pen"]))
        if os.environ.get(f'FIG2_NUDGE{j}'):     # e.g. "DDX3Y:0:-0.8"
            common["nudge"] = {t: (float(a), float(b)) for t, a, b in
                               (n.split(":") for n in os.environ[f'FIG2_NUDGE{j}'].split(","))}
        res, tot = label_placer.place(fig, ax, pts, labs, **common)
    qual = label_quality(fig, ax, pts, res)
    savep(fig, name)
    LOG.append(f"{letter} ({name}): {qual}; {d['n_pip']} features PIP > 0.5, {len(d['pts'])} plotted, {len(labs)} labels "
               f"({sum(1 for l in labs if l[3] == 'black')} black, {sum(1 for l in labs if l[3] == 'gray')} grey); ylim "
               f"{tuple(round(v, 4) for v in ax.get_ylim())}; markersize {ms}; label size {LABPT} pt; longest leader "
               f"{max(r_['leader_pt'] for r_ in res):.1f} pt; label overlaps {sum(r_['overlaps_label'] for r_ in res)}; "
               f"blank chromosome tick labels {blank} (closest kept labels {mingap:.1f} pt apart)")
    return res


# ================================================================== g: cg26921482 and H3K9ac aggregates vs score (p03b)
def panel_g():
    _, _, w, h = cell("g")
    K = 74; a = score(K); fem = female(); sexlab = np.where(fem, "female", "male")
    exC = excl_mask("chip"); nomask = np.zeros(276, bool); assert exC.sum() == 4
    MI = mod_input("meth"); LM = comp_loadings(K, "meth"); cpgs = list(LM.index)
    LMp = LM[LM.P > 0.5]; top = LMp.X.abs().idxmax(); assert top == "cg26921482"
    ym = np.asarray(MI[:, cpgs.index(top)], float)
    CI = np.asarray(mod_input("chip"), float); LC = comp_loadings(K, "chip")
    Hd = pd.read_csv(f"{REFS}/H3K9acDomains.csv", sep="\t", index_col=0).loc[list(LC.index)]
    chrc = Hd.Chr.astype(str).values
    selX = (LC.P.values > 0.5) & (chrc == "X"); selA = (LC.P.values > 0.5) & ~np.isin(chrc, ["X", "Y"])
    aggX, nX = zagg(CI[:, selX], LC.X.values[selX]); aggA, nA = zagg(CI[:, selA], LC.X.values[selA])
    assert (nX, nA) == (537, 120)
    fig = fig_mm(w, h)
    lm, gapw = 12.3, 13.6                                   # first axes left; left margin of the next two
    aw = (w - lm - 2 * gapw - 1.0) / 3; ah = h - TOP - BOT
    lefts = [lm, lm + aw + gapw, lm + 2 * (aw + gapw)]
    axs = [ax_mm(fig, x, TOP, aw, ah) for x in lefts]
    S = 3.2

    def sex_scatter(ax, y, excl, ylab):
        ok = ~np.isnan(y) & ~excl
        for sx in ("male", "female"):
            m = ok & (sexlab == sx)
            ax.scatter(a[m], y[m], s=S, color=SEXCOL[sx], alpha=0.8, linewidths=0)
        ax.set_ylabel(ylab); sns.despine(ax=ax)
        ax.tick_params(length=0)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5, steps=[1, 2, 5, 10]))
        r, p = ss.spearmanr(y[ok], a[ok])
        return int(ok.sum()), float(r)

    n1, r1 = sex_scatter(axs[0], ym, nomask, "cg26921482 ($\\mathit{AMDHD2}$)\nmethylation (β)")
    n2, r2 = sex_scatter(axs[1], aggX, exC, f"H3K9ac, {nX} chrX peaks\nweighted mean z")
    n3, r3 = sex_scatter(axs[2], aggA, exC, f"H3K9ac, {nA} autosomal peaks\nweighted mean z")
    axs[1].set_xlabel("Component 74 loadings/score")
    hnd = [Line2D([], [], color=MALE, marker="o", ls="None", markersize=2.6, markeredgewidth=0, label="Male"),
           Line2D([], [], color=FEMALE, marker="o", ls="None", markersize=2.6, markeredgewidth=0, label="Female")]
    axs[0].legend(handles=hnd, loc="lower left", frameon=False, handletextpad=0.1, borderaxespad=0.2, borderpad=0.1,
                  labelspacing=0.2, handlelength=1.0)
    savep(fig, "Fig2g_C74_cpg_H3K9ac_vs_score")
    LOG.append(f"g: cg26921482 n {n1} (rho {r1:+.2f}); H3K9ac chrX {nX} peaks n {n2} (rho {r2:+.2f}); autosomal {nA} "
               f"peaks n {n3} (rho {r3:+.2f}); axes {aw:.1f} x {ah:.1f} mm")


if __name__ == "__main__":
    which = sys.argv[1:] or list("abcdefg")
    if "a" in which: panel_a()
    if "b" in which: panel_b()
    if "c" in which: panel_c()
    if "d" in which: panel_manhattan(1)
    if "e" in which: panel_manhattan(3)
    if "f" in which: panel_manhattan(2)
    if "g" in which: panel_g()
    geom = dict(W=W, H=H, ROW_Y=ROW_Y, ROW_H=ROW_H, CELLS=CELLS, TOP=TOP, BOT1=BOT1, BOT=BOT, AXL=AXL)
    json.dump(geom, open(f"{PAN}/geometry.json", "w"), indent=1)
    open(f"{PAN}/fig2_panels_{''.join(which)}.log", "w").write("\n".join(LOG) + "\n" + json.dumps(geom) + "\n")
    print("\n".join(LOG))
