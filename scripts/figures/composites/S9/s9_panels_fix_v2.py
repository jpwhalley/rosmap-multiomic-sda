"""Final version of the S9 panel script. Compared with the first build only thin_chr_labels changed: the minimum white gap between kept chromosome tick labels stays 0.35 mm, but is
0.25 mm where one of the two labels is X or Y, so that panel d no longer shows Y without X (its 22-X gap is 0.29 mm).
A global 0.25 mm was tried and rejected: it put rotated "20" against "22" in d (reads "2022") and "7" against "8" in e.
Everything else is unchanged.

Figure S9 (Component 2) panels redrawn at their final printed size (v23 supplement; composite by composites/compose.py).

Ground truth: the S9 legend in the supplement. Every panel is replotted from the same data with the same drawing logic; only the size,
the type sizes (explicit 6-7 pt Arial) and the placement of point labels change.

  a  C2 sample scores, x = age-at-death order, colour = sex, broken y axis (v22 S3a, CODING/figs/outlier/2/2_samples.pdf;
     drawing code ros3b_plot_outliers.py l.157-317). Data: <G_ROS_SOURCE_DATA_DIR>/controlled/FigS3a_C2_sample_scores_by_donor.csv
     (verified point-for-point against that panel; scores asserted equal to the deposited samples_final.csv row). The guide asks for a "loadings/score" y label, which v22 left blank.
     Layout-only: the Male/Female key moves from above the axes (v22 fig.legend, upper right) into the empty right part
     of the lower (outlier) pane.
  b  C2 corrected cell-type loadings (corrected 2_cell_type.pdf; ros3b l.324-340).
     Data: <G_ROS_SOURCE_DATA_DIR>/FigS9b_C2_cell_type_loadings.csv (asserted equal to vp_common.bcor(2)).
     Cell-type names at 6 pt so that "Mic" and "OPC" keep a gap at this width.
  c  C2 snRNA-seq gene loadings by chromosome, PIP > 0.5 (v22 Fig 3b, CODING/figs/outlier/2/2_RNA.pdf)
  d  C2 CpG loadings by chromosome (v22 Fig 3c, 2_meth.pdf)
  e  C2 H3K9ac peak loadings by chromosome (v22 Fig 3d, 2_chip.pdf)
     Data: <G_ROS_SOURCE_DATA_DIR>/FigS9c/S9d/S9e_C2_*.csv (the v22 Fig 3b-d files under their v23 names; plot_x, loading, label text and colour, drawn flag; verified
     against the v22 panels). Drawing logic = ros3b_plot_outliers.py l.341-620 (2026-08 snapshot, read only): the
     per-chromosome colour cycle, invisible placeholder points for chromosomes without features, x ticks at the
     midpoints of the running chromosome offsets (rotated 90), the y-limit rule, black (largest) / grey (smallest)
     labels joined to their points by plain lines, full axes box, no tick marks.
     Layout-only changes: (1) label positions (v22 used adjustText, which is not installed here; this script uses a
     deterministic placer with the same texts, colours and leader lines; the eight grey labels of c and the five chr22
     labels of e are placed by hand, see MAN_C / MAN_E); (2) chromosome tick labels whose glyphs would overlap at print
     size are left blank: labels of chromosomes that carry features are kept first, then the rest from the right
     (v22 printed them on top of one another, e.g. "2122"); every blank label is logged.
  f  H3K9ac log2(CPM + 1) at peak16865-peak16869 and their weighted mean z vs the C2 score, n 189 (v23 S9f,
     S9f_C2_chr22_peaks_and_aggregate_vs_score.pdf): the S9f section of panels/p06_c21_single.py re-run with the
     vp_common loaders and its broken_x_scatter logic, drawn in mm. Its logged statistics are asserted equal to
     <G_ROS_FIGURE_OUT>/panels/p06_c21_single.log (run p06 first).

Writes only under <G_ROS_FIGURE_OUT>/S9/ (panels/, layout_S9.json, s9_panels.log). Logs aggregates only: no donor IDs
and no per-donor values."""
import os, sys, re, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
from vp_common import *          # style(): seaborn white/talk, Arial, pdf.fonttype 42; score, comp_loadings, ...
import numpy as np, pandas as pd
from scipy import stats as ss

HERE = f"{FIGOUT}/S9"
PAN = f"{HERE}/panels"; os.makedirs(PAN, exist_ok=True)
from figure_paths import SOURCE_DATA as SD
MM = 1 / 25.4
FL, FT, FA = 7.0, 6.5, 6.0        # axis labels, tick labels, in-plot point labels (pt)
FC = 6.0                          # chromosome tick labels (pt)
LW = 0.6                          # spines, pt

# ---------------------------------------------------------------- page geometry (mm; origin top-left, y down)
W = 142.9
R1, R2, R3 = 0.0, 51.5, 95.0      # row tops
H1, H2, H3 = 48.0, 40.0, 69.0     # row heights -> total 164.0 (budget 166.3)
GEOM = {
    "a": dict(x=0.0, y=R1, w=29.0, h=H1),
    "b": dict(x=31.0, y=R1, w=37.0, h=H1),
    "c": dict(x=70.0, y=R1, w=W - 70.0, h=H1),
    "d": dict(x=0.0, y=R2, w=70.45, h=H2),
    "e": dict(x=72.45, y=R2, w=70.45, h=H2),
    "f": dict(x=0.0, y=R3, w=W, h=H3),
}
HTOT = R3 + H3

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": FL, "axes.labelsize": FL, "xtick.labelsize": FT, "ytick.labelsize": FT, "legend.fontsize": FT,
    "axes.linewidth": LW, "lines.linewidth": 0.8, "patch.linewidth": 0.0,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0,
    "xtick.major.width": LW, "ytick.major.width": LW, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
})
LOG = []


def ax_mm(fig, wmm, hmm, left, top, width, height, **kw):
    """Axes placed by its left/top edge and size in mm (origin top-left of the figure)."""
    return fig.add_axes([left / wmm, 1 - (top + height) / hmm, width / wmm, height / hmm], **kw)


def bbox_mm(fig, artist):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bb = artist.get_window_extent(r); h = fig.get_size_inches()[1] * fig.dpi
    s = 25.4 / fig.dpi
    return (bb.x0 * s, (h - bb.y1) * s, bb.x1 * s, (h - bb.y0) * s)


def check_inside(fig, name, keep_clear_mm=(3.6, 3.4)):
    """Everything drawn lies inside the cell, and the top-left corner (the letter box) is empty."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(r); w, h = fig.get_size_inches()
    ext = [round(v * 25.4, 2) for v in (tb.x0, tb.y0, tb.x1, tb.y1)]
    ok = tb.x0 >= -0.005 and tb.y0 >= -0.005 and tb.x1 <= w + 0.005 and tb.y1 <= h + 0.005
    LOG.append(f"{name}: drawn extent (mm, from bottom-left) {ext} within {round(w * 25.4, 2)} x {round(h * 25.4, 2)}: {ok}")
    assert ok, (name, ext)
    kw, kh = keep_clear_mm
    undrawn = set()                                   # tick labels of ticks outside the view limits are not drawn
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            drawn = {id(tk) for tk in axis._update_ticks()}
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if id(tk) not in drawn:
                    undrawn |= {id(tk.label1), id(tk.label2)}
    wmm, hmm_ = w * 25.4, h * 25.4
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in undrawn:
            continue
        x0, y0, x1, y1 = bbox_mm(fig, t)
        assert not (x0 < kw and y0 < kh), (name, t.get_text(), (x0, y0))
        # every drawn text inside the cell, with 0.3 mm to spare (the figure tight box clamps axis labels)
        assert x0 >= 0.3 and y0 >= 0.3 and x1 <= wmm - 0.3 and y1 <= hmm_ - 0.3, (name, t.get_text(), (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)))
    for ax in fig.axes:
        x0, y0, x1, y1 = bbox_mm(fig, ax)
        assert not (x0 < kw and y0 < kh), (name, "axes in letter corner")


def save_cell(fig, name):
    check_inside(fig, name)
    fig.savefig(f"{PAN}/{name}.pdf")          # exact cell size: no bbox_inches
    plt.close(fig); LOG.append(f"wrote panels/{name}.pdf")


def new_cell(letter):
    g = GEOM[letter]
    return plt.figure(figsize=(g["w"] * MM, g["h"] * MM)), g


# ================================================================ a  sample scores (ros3b l.157-317)
S = pd.read_csv(f"{SD}/controlled/FigS3a_C2_sample_scores_by_donor.csv")
assert len(S) == 276 and (S.component == 2).all() and S.projid.is_unique
assert np.allclose(np.sort(S.sda_sample_score.values), np.sort(score(2)))          # = deposited samples_final.csv row
assert sorted(S.x_position.unique().tolist()) == list(range(276))
y = S.sda_sample_score.values; x = S.x_position.values
q1, q3 = np.percentile(y, [25, 75])                                                  # pandas quantile = linear = np default
iqr = q3 - q1; lo_b, hi_b = q1 - 30 * iqr, q3 + 30 * iqr
out = (y < lo_b) | (y > hi_b)
assert (np.where(out, "outlier_segment", "main_segment") == S.broken_axis_segment.astype(str).values).all()   # same split as v22
LOG.append(f"a: n {len(S)}; donors beyond 30 x IQR {int(out.sum())}; broken_axis_segment values {sorted(S.broken_axis_segment.astype(str).unique())}")
nmin, nmax = y[~out].min(), y[~out].max()
nbuf = max((nmax - nmin) * 0.05, 0.02); normal_ylim = (nmin - nbuf, nmax + nbuf)
omin, omax = y[out].min(), y[out].max()
obuf = max(abs(omin) * 0.05, 0.01) if omin == omax else max((omax - omin) * 0.05, 0.01)
outlier_ylim = (omin - obuf, omax + obuf)
ylims = sorted([normal_ylim, outlier_ylim], key=lambda t: t[0])                     # brokenaxes: lowest range at the bottom
SEXC = {"Male": sns.color_palette("muted")[0], "Female": sns.color_palette("muted")[2]}
sexv = S.sex.astype(str).str.capitalize().values
assert set(sexv) <= {"Male", "Female"}

fig, g = new_cell("a")
L, T, B, Rr = 11.6, 4.0, 4.6, 0.6
aw, ah = g["w"] - L - Rr, g["h"] - T - B
gap = 1.4                                                                             # brokenaxes hspace 0.05 at print size
spans = [hi - lo for lo, hi in ylims[::-1]]                                          # top pane first
hs = [(ah - gap) * s_ / sum(spans) for s_ in spans]
axt = ax_mm(fig, g["w"], g["h"], L, T, aw, hs[0])
axb_ = ax_mm(fig, g["w"], g["h"], L, T + hs[0] + gap, aw, hs[1], sharex=axt)
for ax_, (lo, hi) in zip((axt, axb_), ylims[::-1]):
    m_n, m_o = ~out, out                                                              # v22: normal points first, then outliers
    for m in (m_n, m_o):
        ax_.scatter(x[m], y[m], c=[SEXC[s_] for s_ in sexv[m]], s=3.2, alpha=0.8, linewidths=0, edgecolors="k")
    ax_.set_ylim(lo, hi)
    ax_.set_xticks([])
    ax_.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax_.tick_params(axis="y", left=True, length=1.8, width=LW, pad=1.2)
    for sp in ("top", "right"):
        ax_.spines[sp].set_visible(False)
axt.spines["bottom"].set_visible(False); axb_.spines["top"].set_visible(False)
axt.tick_params(axis="x", bottom=False)
# brokenaxes-style diagonal marks on the left spine at the break
dmm = 0.9
for ax_, yy in ((axt, 0.0), (axb_, 1.0)):
    bb = ax_.get_position()
    hx = dmm / (aw); hy = dmm / (hs[0] if ax_ is axt else hs[1])
    ax_.plot((-hx, hx), (yy - hy, yy + hy), transform=ax_.transAxes, color="k", lw=LW, clip_on=False)
axb_.set_xlabel("Samples", fontsize=FL, labelpad=2.5)
fig.text((L - 9.6) / g["w"], 1 - (T + ah / 2) / g["h"], "loadings/score", rotation=90, ha="center", va="center", fontsize=FL)
from matplotlib.lines import Line2D
hdl = [Line2D([], [], color=SEXC["Male"], marker="o", ls="None", markersize=3.0, label="Male"),
       Line2D([], [], color=SEXC["Female"], marker="o", ls="None", markersize=3.0, label="Female")]
# v22: fig.legend at the upper right; at print size the key goes into the empty right part of the lower pane
leg = axb_.legend(handles=hdl, loc="lower right", ncol=1, frameon=False, fontsize=FT, borderaxespad=0.4, borderpad=0.0,
                  handlelength=0.8, handletextpad=0.3, labelspacing=0.35)
LOG.append(f"a: y panes {[tuple(round(v, 3) for v in t) for t in ylims[::-1]]} (top first), heights {[round(h_, 1) for h_ in hs]} mm")
save_cell(fig, "S9a_C2_sample_scores_cell")

# ================================================================ b  corrected cell-type loadings (ros3b l.324-340)
CTD = pd.read_csv(f"{SD}/FigS9b_C2_cell_type_loadings.csv").sort_values("bar_order")
assert list(CTD.cell_type) == CT_ORDER and (CTD.component == 2).all()
bc = bcor(2)
assert np.allclose(CTD.cell_type_loading.values, [bc[c] for c in CT_ORDER])
fig, g = new_cell("b")
L, T, B, Rr = 8.6, 4.0, 4.6, 0.5
ax = ax_mm(fig, g["w"], g["h"], L, T, g["w"] - L - Rr, g["h"] - T - B)
yp = np.arange(6)
ax.bar(yp, CTD.cell_type_loading.values, align="center", color=[CTCOL[c] for c in CT_ORDER])
ax.set_xticks(yp); ax.set_xticklabels(CT_ORDER, fontsize=FC)       # 6 pt: "Mic" and "OPC" keep a gap at this width
ax.set_ylabel("loadings/score")
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_linewidth(LW)
LOG.append("b: " + ", ".join(f"{c} {v:+.3f}" for c, v in zip(CT_ORDER, CTD.cell_type_loading.values)))
save_cell(fig, "S9b_C2_cell_type_loadings_corrected_cell")

# ================================================================ c, d, e  loadings by chromosome (ros3b l.341-620)
COLORS = {1: ['#6495ED', '#778899', '#008080', '#20B2AA'], 2: ['#ffffb2', '#fecc5c', '#fd8d3c', '#e31a1c'],
          3: ['#f1eef6', '#d7b5d8', '#df65b0', '#ce1256']}
CHR_MAX = {'1': 249230899.0, '2': 243064438.0, '3': 197955065.0, '4': 191012160.0, '5': 180794288.0, '6': 170886450.0,
           '7': 158820866.0, '8': 146277826.0, '9': 141069493.0, '10': 135497182.0, '11': 134248398.0, '12': 133758108.0,
           '13': 115079831.0, '14': 107282052.0, '15': 102501016.0, '16': 90244125.0, '17': 81037567.0, '18': 77915115.0,
           '19': 59086812.0, '20': 62958768.0, '21': 48110787.0, '22': 51205920.0, 'X': 155249967.0, 'Y': 59330252.0}
CHRS = [str(i) for i in range(1, 23)] + ['X', 'Y']


def ylim_rule(vals):
    vals = list(vals)
    if len(vals) == 1:
        v = vals[0]; m = abs(v * 0.1) if v != 0 else 0.1; return v - m, v + m
    mn, mx = min(vals), max(vals)
    mmin = abs(mn * 0.1) if mn != 0 else 0.1; mmax = abs(mx * 0.1) if mx != 0 else 0.1
    if mx > 0 and mn >= 0:
        return mn - mmin, mx + mmax
    if mx <= 0 and mn < 0:
        return mn - mmin, mx + mmax
    return mn * 1.1, mx * 1.1


def _rect_dist(P, bb):
    dx = np.maximum(np.maximum(bb.x0 - P[:, 0], 0), P[:, 0] - bb.x1)
    dy = np.maximum(np.maximum(bb.y0 - P[:, 1], 0), P[:, 1] - bb.y1)
    return np.hypot(dx, dy)


def _seg_hits_box(p, q, bb, n=24):
    t = np.linspace(0, 1, n)[:, None]; S_ = p[None, :] * (1 - t) + q[None, :] * t
    return bool(((S_[:, 0] > bb.x0) & (S_[:, 0] < bb.x1) & (S_[:, 1] > bb.y0) & (S_[:, 1] < bb.y1)).any())


def _seg_cross(p1, p2, q1, q2):
    def o(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return o(p1, p2, q1) * o(p1, p2, q2) < 0 and o(q1, q2, p1) * o(q1, q2, p2) < 0


def place_labels(fig, ax, pts, labels, manual=None, radii=(3, 4.5, 6, 8, 10, 12.5, 15, 18, 22, 27, 33), angles=None,
                 passes=12):
    """Deterministic label placer (stand-in for adjustText). pts: (n, 2) data coords of all drawn points; labels: list of
    dict(x, y, text, colour). Each label has candidate offsets on rings around its point. Unary cost: leaving the axes
    (prohibitive), text over other points, another point as near to the text as its own point (ambiguous), distance.
    Pair cost: overlapping texts (prohibitive), a leader line through another text, crossing leader lines.
    Greedy start, then coordinate descent until no label moves. manual: {text: (dx_pt, dy_pt, ha, va)} fixes a label.
    Each label is joined to its point by a plain line, as adjustText's arrowprops (arrowstyle '-', black)."""
    manual = manual or {}
    angles = angles if angles is not None else list(range(0, 360, 15))
    fig.canvas.draw(); rd = fig.canvas.get_renderer()
    P = ax.transData.transform(np.asarray(pts, float))
    axbb = ax.get_window_extent(rd)
    ptpx = fig.dpi / 72.0
    n = len(labels)
    C = []                                    # per label: list of (dx, dy, ha, va, bbox array, leader end, unary)
    for lab in labels:
        own = np.array(ax.transData.transform((lab["x"], lab["y"])))
        others = P[np.hypot(P[:, 0] - own[0], P[:, 1] - own[1]) > 0.5]
        if lab["text"] in manual:
            mv = manual[lab["text"]]
            if mv[0] == "data":                               # ("data", x, y, ha, va): text anchor in data coordinates
                tx, ty = ax.transData.transform((mv[1], mv[2])); dx, dy = (tx - own[0]) / ptpx, (ty - own[1]) / ptpx
                ha, va = mv[3], mv[4]
            else:
                dx, dy, ha, va = mv
            cands = [(np.hypot(dx, dy), dx, dy, ha, va)]
        else:
            cands = []
            for r in radii:
                for a in angles:
                    c, s_ = np.cos(np.radians(a)), np.sin(np.radians(a))
                    ha = "left" if c > 0.3 else ("right" if c < -0.3 else "center")
                    va = "bottom" if s_ > 0.3 else ("top" if s_ < -0.3 else "center")
                    cands.append((r, r * c, r * s_, ha, va))
        lst = []
        for (r, dx, dy, ha, va) in cands:
            t = ax.text(lab["x"], lab["y"], lab["text"], fontsize=FA, ha=ha, va=va,
                        transform=mpl.transforms.offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points"))
            bb = t.get_window_extent(rd); t.remove()
            b = np.array([bb.x0 - 0.6 * ptpx, bb.y0 - 0.8 * ptpx, bb.x1 + 0.6 * ptpx, bb.y1 + 0.8 * ptpx])   # ink + descender room
            u = 0.35 * r
            if b[0] < axbb.x0 + ptpx or b[2] > axbb.x1 - ptpx or b[1] < axbb.y0 + ptpx or b[3] > axbb.y1 - ptpx:
                u += 1e6
            dxo = np.maximum(np.maximum(b[0] - others[:, 0], 0), others[:, 0] - b[2])
            dyo = np.maximum(np.maximum(b[1] - others[:, 1], 0), others[:, 1] - b[3])
            d_o = np.hypot(dxo, dyo)
            u += 30 * int((d_o < 1.2 * ptpx).sum())
            end = np.array([np.clip(own[0], b[0], b[2]), np.clip(own[1], b[1], b[3])])
            d_own = float(np.hypot(*(own - end)))
            u += 8 * int((d_o < d_own + 2.0 * ptpx).sum())
            lst.append((dx, dy, ha, va, b, end, u, r))
        C.append((own, lst))

    def pair(i, ci, j, cj):
        bi, bj = C[i][1][ci][4], C[j][1][cj][4]
        cost = 0.0
        if min(bi[2], bj[2]) > max(bi[0], bj[0]) and min(bi[3], bj[3]) > max(bi[1], bj[1]):
            cost += 1e5
        oi, ei = C[i][0], C[i][1][ci][5]; oj, ej = C[j][0], C[j][1][cj][5]
        if C[i][1][ci][7] > 2.5 and _seg_hits_box(oi, ei, type("B", (), dict(x0=bj[0], y0=bj[1], x1=bj[2], y1=bj[3]))):
            cost += 2e3
        if C[j][1][cj][7] > 2.5 and _seg_hits_box(oj, ej, type("B", (), dict(x0=bi[0], y0=bi[1], x1=bi[2], y1=bi[3]))):
            cost += 2e3
        if _seg_cross(oi, ei, oj, ej):
            cost += 500
        return cost

    order = sorted(range(n), key=lambda i: -abs(labels[i]["y"]))
    choice = [None] * n
    for i in order:                                           # greedy start
        best = min(range(len(C[i][1])), key=lambda c: C[i][1][c][6] + sum(pair(i, c, j, choice[j]) for j in range(n) if choice[j] is not None))
        choice[i] = best
    for it in range(passes):                                  # coordinate descent
        moved = 0
        for i in order:
            cur = choice[i]
            best = min(range(len(C[i][1])), key=lambda c: (C[i][1][c][6] + sum(pair(i, c, j, choice[j]) for j in range(n) if j != i), c != cur))
            if best != cur:
                choice[i] = best; moved += 1
        if moved == 0:
            break
    anns = []
    for i, lab in enumerate(labels):
        dx, dy, ha, va, b, end, u, r = C[i][1][choice[i]]
        tot = u + sum(pair(i, choice[i], j, choice[j]) for j in range(n) if j != i)
        own = C[i][0]
        d_own = float(np.hypot(*(own - np.array([np.clip(own[0], b[0], b[2]), np.clip(own[1], b[1], b[3])]))))
        # a leader only where the text is not already touching its point (annotate clips the line at the text box
        # plus 2 pt, so a point inside that margin would get a line drawn through the text)
        arrow = dict(arrowstyle="-", color="k", lw=0.35, shrinkA=0.0, shrinkB=1.2) if d_own > 2.6 * ptpx else None
        an = ax.annotate(lab["text"], (lab["x"], lab["y"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                         fontsize=FA, color=lab["colour"], zorder=4, arrowprops=arrow)
        anns.append(an)
        LOG.append(f"  label {lab['text']}: offset ({dx:+.1f}, {dy:+.1f}) pt, ha {ha}, va {va}, cost {tot:.1f}")
    LOG.append(f"  placer: coordinate descent passes {it + 1}")
    return anns


def thin_chr_labels(fig, ax, tag, priority=()):
    """Blank chromosome tick labels whose glyphs would overlap at print size. Labels of chromosomes that carry
    features (priority) are kept first; the rest are then kept from the right (22, X, Y first) wherever they fit."""
    fig.canvas.draw(); rd = fig.canvas.get_renderer()
    tl = ax.get_xticklabels()
    iv = []
    for t in tl:
        bb = t.get_window_extent(rd)
        ink = bb.x1 - bb.x0                                  # Agg text boxes are the glyph ink extent (digit cap height)
        cx = (bb.x0 + bb.x1) / 2; iv.append((cx - ink / 2, cx + ink / 2))
    gap = 0.35 * fig.dpi / 25.4                             # at least 0.35 mm of white between kept labels
    gap_xy = 0.25 * fig.dpi / 25.4                          # final version: 0.25 mm next to X or Y (a letter beside digits stays legible)
    kept = []
    def fits(k):
        def g_(m):
            return gap_xy if {tl[k].get_text(), tl[m].get_text()} & {"X", "Y"} else gap
        return all(iv[k][1] + g_(m) <= iv[m][0] or iv[m][1] + g_(m) <= iv[k][0] for m in kept)
    order = [k for k in range(len(tl)) if tl[k].get_text() in priority][::-1] + \
            [k for k in range(len(tl)) if tl[k].get_text() not in priority][::-1]
    hidden = []
    for k in order:
        if fits(k):
            kept.append(k)
        else:
            tl[k].set_visible(False); hidden.append(tl[k].get_text())
    hidden = [t.get_text() for t in tl if t.get_text() in hidden]
    if os.environ.get("S9_DEBUG"):
        LOG.append(f"{tag} DEBUG ink intervals mm: " + " ".join(f"{t.get_text()}:{a_ * 25.4 / fig.dpi:.2f}-{b_ * 25.4 / fig.dpi:.2f}" for t, (a_, b_) in zip(tl, iv)))
    LOG.append(f"{tag}: chromosome tick labels left blank (overlap at print size): {hidden if hidden else 'none'}")


def chrom_panel(letter, csv, j, xlabel, name, manual=None, dims=None, order=None):
    D = pd.read_csv(f"{SD}/{csv}", keep_default_na=False, na_values=[''])
    assert (D.component == 2).all()
    D["chrom"] = D.chrom.astype("string")
    Dd = D[D.drawn.astype(str) == "True"]
    fig, g = new_cell(letter)
    L, T, B, Rr = dims
    ax = ax_mm(fig, g["w"], g["h"], L, T, g["w"] - L - Rr, g["h"] - T - B)
    shift = np.array([0.0]); pts = []; labs = []
    for ii, c in enumerate(CHRS):
        s = Dd[Dd.chrom == c]
        if len(s) == 0:
            xs = shift[-1] + np.array([CHR_MAX[c]]); ax.plot(xs, [0], '.', color=COLORS[j][ii % 4], markersize=0)
        else:
            assert np.allclose(s.plot_x.values - s.position_bp.values, shift[-1]), (letter, c)   # running offsets as v22
            xs = s.plot_x.values
            ax.plot(xs, s.loading.values, '.', color=COLORS[j][ii % 4], markersize=4.2, markeredgewidth=0.35)
            pts += list(zip(xs, s.loading.values))
            for _, r in s[s.label_text.notna()].iterrows():
                labs.append(dict(x=r.plot_x, y=r.loading, text=r.label_text, colour=r.label_colour))
        shift = np.append(shift, np.max(xs))
    mids = (shift[1:] + shift[:-1]) / 2.
    ax.set_xticks(mids); ax.set_xticklabels(CHRS, rotation=90, fontsize=FC)
    ax.set_ylim(*ylim_rule(D.loading.values))                # all PIP > 0.5 features, as v22
    ax.set_xlabel(xlabel); ax.set_ylabel("loadings/score")
    ax.tick_params(length=0)
    ax.tick_params(axis="x", pad=1.0)
    for sp in ax.spines.values():
        sp.set_linewidth(LW)
    thin_chr_labels(fig, ax, letter, priority=set(Dd.chrom.dropna().astype(str)))
    LOG.append(f"{letter}: {len(D)} features with PIP > 0.5 ({len(Dd)} drawn); labels {len(labs)} "
               f"({sum(l_['colour'] == 'black' for l_ in labs)} black, {sum(l_['colour'] == 'gray' for l_ in labs)} grey); "
               f"ylim {tuple(round(v, 4) for v in ax.get_ylim())}")
    place_labels(fig, ax, pts, labs, manual=manual)
    save_cell(fig, name)
    return D


# c: the eight grey (smallest-loading) labels sit in the sparse band below the point cloud; placed by hand in data
# coordinates (x = genome-wide plot position, y = loading) in three rows, each next to or just below its own point
MAN_C = {"AC093772.1": ("data", 7.66e8, 0.049, "right", "center"), "AC004014.1": ("data", 8.77e8, 0.283, "left", "center"),
         "AL391261.2": ("data", 2.194e9, 0.318, "left", "center"), "AL133338.1": ("data", 1.151e9, -0.215, "left", "center"),
         "AC009988.1": ("data", 1.747e9, -0.215, "left", "center"), "AC009501.1": ("data", 3.48e8, -0.60, "left", "center"),
         "AC025265.2": ("data", 1.929e9, -0.74, "right", "center"), "ZNF837": ("data", 2.608e9, -0.41, "left", "center")}
MAN_D = {}
# e: the five chr22 peaks (39.795-39.932 Mb) share one x position; their labels are stacked to the left in loading order
MAN_E = {"peak16869:Promoter": (-9.0, 0.0, "right", "center"), "peak16867:Enhancer": (-9.0, 4.0, "right", "center"),
         "peak16865:Enhancer": (-9.0, 4.6, "right", "center"), "peak16868:Promoter": (-9.0, -0.3, "right", "center"),
         "peak16866:Other": (-9.0, -0.8, "right", "center")}
Dc = chrom_panel("c", "FigS9c_C2_RNA_gene_loadings.csv", 1, "Genes by Chromosome", "S9c_C2_RNA_gene_loadings_cell",
                 manual=MAN_C, dims=(7.0, 4.0, 6.2, 0.5))
Dd_ = chrom_panel("d", "FigS9d_C2_CpG_loadings.csv", 2, "CpG Sites by Chromosome", "S9d_C2_CpG_loadings_cell",
                  manual=MAN_D, dims=(10.0, 4.0, 6.2, 0.5))
De = chrom_panel("e", "FigS9e_C2_H3K9ac_peak_loadings.csv", 3, "ChIP Peaks by Chromosome", "S9e_C2_H3K9ac_peak_loadings_cell",
                 manual=MAN_E, dims=(7.2, 4.0, 6.2, 0.5))
assert len(Dc) == 277 and len(Dd_) == 6 and len(De) == 7                  # legend counts: 277 genes, n = 6 CpGs, 7 peaks

# ================================================================ f  five chr22 peaks + aggregate (p06 S9f section)
a2 = score(2)
exC = excl_mask("chip")
LC = comp_loadings(2, "chip"); peaks = ["peak16865", "peak16866", "peak16867", "peak16868", "peak16869"]
CI = np.asarray(mod_input("chip"), float)
H = pd.read_csv(f"{REFS}/H3K9acDomains.csv", sep="\t", index_col=0)
jj = [list(LC.index).index(p) for p in peaks]
Mp = CI[:, jj].copy(); Mp[exC] = np.nan
okp = ~np.isnan(Mp).all(1)
ag, _ = zagg(Mp[okp], LC.X.values[jj]); aggC = np.full(276, np.nan); aggC[okp] = ag
CHIPCOL = "#ce1256"   # v22 ChIP colour (ros3b colors[3])
FLOG = []


def log_view(tag, yv, a):
    ok = ~np.isnan(yv); r, p, n = spearman(yv, a); ext = most_extreme(a[ok]); ai, yi = a[ok], yv[ok]
    m = np.ones(ok.sum(), bool); m[ext] = False
    r1 = ss.spearmanr(yi[m], ai[m])[0]; pr = ss.pearsonr(yi, ai)[0]; pr1 = ss.pearsonr(yi[m], ai[m])[0]
    FLOG.append(f"{tag}: n {n}; Spearman {r:+.3f} (p {p:.2g}); without most extreme donor {r1:+.3f}; Pearson {pr:+.3f} -> {pr1:+.3f}")
    return dict(n=n, rho=r, p=p, pr=pr, pr1=pr1)


def broken_x_scatter_mm(fig, g, box, a, series, xlabel, ylabel, ring=True):
    """vp_common.broken_x_scatter (same outlier rule, panes, limits, ticks, ring, labels), with the panes placed in mm.
    box = (left, top, width, height) of the pair of panes, mm."""
    left, top, width, height = box
    out = outliers_30iqr(a); ext = most_extreme(a)
    assert out.any()
    neg = a[out].mean() < np.median(a)
    wsp = 0.07                                               # subgridspec wspace (fraction of the mean pane width)
    unit = width / (5 + wsp * 2.5)                          # widths 1 : 4, gap = wsp * mean(1, 4) units
    wo, wb, gp = unit, 4 * unit, wsp * 2.5 * unit
    if neg:
        axo = ax_mm(fig, g["w"], g["h"], left, top, wo, height)
        axb = ax_mm(fig, g["w"], g["h"], left + wo + gp, top, wb, height, sharey=axo)
        axes = [axo, axb]
    else:
        axb = ax_mm(fig, g["w"], g["h"], left, top, wb, height)
        axo = ax_mm(fig, g["w"], g["h"], left + wb + gp, top, wo, height, sharey=axb)
        axes = [axb, axo]
    for (yv, col, sz, lab) in series:
        ok = ~np.isnan(yv)
        for ax_ in (axb, axo):
            ax_.scatter(a[ok], yv[ok], color=col, s=sz, alpha=0.8, linewidths=0, zorder=2)
        if ring and not np.isnan(yv[ext]):
            for ax_ in (axb, axo):
                ax_.scatter([a[ext]], [yv[ext]], s=sz * 3.2, facecolors="none", edgecolors=EXTRING, linewidths=0.6, zorder=3)
    for ax_ in axes:
        sns.despine(ax=ax_)
        ax_.tick_params(length=0)
        for sp in ("left", "bottom"):
            ax_.spines[sp].set_linewidth(LW)
    bulk = a[~out]; pad = (bulk.max() - bulk.min()) * 0.05
    axb.set_xlim(bulk.min() - pad, bulk.max() + pad)
    axb.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    o = a[out]; span = max(np.abs(o).max() * 0.04, 1e-3)
    axo.set_xlim(o.min() - span, o.max() + span)
    axo.set_xticks([float(f"{o.mean():.2f}")])
    lft, rgt = (axo, axb) if neg else (axb, axo)
    lft.spines["right"].set_visible(False); rgt.spines["left"].set_visible(False)
    rgt.tick_params(labelleft=False, left=False)
    # break marks: two parallel strokes across the gap (as vp_common), sized in mm
    dmm = 0.8
    for ax_, xx in ((lft, 1.0), (rgt, 0.0)):
        wpane = wo if ax_ is axo else wb
        hx, hy = dmm / wpane, dmm / height
        ax_.plot((xx - hx * 0.6, xx + hx * 0.6), (-hy, hy), transform=ax_.transAxes, color="k", lw=LW, clip_on=False)
    axb.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel, linespacing=1.05)
    return axes


fig, g = new_cell("f")
COLW = g["w"] / 3
FL_L, FL_R = 11.4, 4.0                       # per sub-panel: left margin (two-line y label + ticks), right gap (the x
                                             # label is centred on the bulk pane, as vp_common, and is wider than it)
ROWT = [5.2, 38.6]                         # pane tops (mm) in the two rows
PH = 24.0                                    # pane height (mm)
st = {}
for i, p in enumerate(peaks):
    r_, c_ = divmod(i, 3)
    box = (c_ * COLW + FL_L, ROWT[r_], COLW - FL_L - FL_R, PH)
    broken_x_scatter_mm(fig, g, box, a2, [(Mp[:, i], CHIPCOL, 4.0, None)], "Component 2 loadings/score",
                        f"{p}\nH3K9ac, log2(CPM + 1)")
    st[p] = log_view(f"S9f {p} (chr{H.loc[p, 'Chr']}:{H.loc[p, 'Start'] / 1e6:.3f}-{H.loc[p, 'End'] / 1e6:.3f} Mb, {H.loc[p, 'Classification']})", Mp[:, i], a2)
box = (2 * COLW + FL_L, ROWT[1], COLW - FL_L - FL_R, PH)
broken_x_scatter_mm(fig, g, box, a2, [(aggC, "#7a0177", 4.0, None)], "Component 2 loadings/score", "five-peak aggregate\nweighted mean z")
st["agg"] = log_view("S9f five-peak aggregate", aggC, a2)
dom = int(np.argmax(np.abs(robust_z(a2))))
FLOG.append(f"S9f n H3K9ac {int(okp.sum())}; dominant donor lowest at every peak: "
            f"{all(np.nanargmin(Mp[:, i]) == dom for i in range(5))}; dominant donor excluded: {bool(exC[dom])}")
save_cell(fig, "S9f_C2_chr22_peaks_and_aggregate_vs_score_cell")

# same statistics as panels/p06_c21_single.py (its log), line for line
p06 = [l.strip() for l in open(f"{OUT}/p06_c21_single.log") if l.startswith("S9f") and "wrote" not in l]
assert p06 == FLOG, ("p06 log differs", p06, FLOG)
LOG.append("f: statistics identical to <G_ROS_FIGURE_OUT>/panels/p06_c21_single.log (S9f lines):")
LOG += ["  " + l for l in FLOG]
LOG.append(f"f: aggregate Pearson with all donors {st['agg']['pr']:.4f}; without the dominant donor {st['agg']['pr1']:.4f}")

# ================================================================ layout for compose.py
layout = {"width_mm": W, "height_mm": HTOT, "cells": []}
names = {"a": "S9a_C2_sample_scores_cell", "b": "S9b_C2_cell_type_loadings_corrected_cell", "c": "S9c_C2_RNA_gene_loadings_cell",
         "d": "S9d_C2_CpG_loadings_cell", "e": "S9e_C2_H3K9ac_peak_loadings_cell", "f": "S9f_C2_chr22_peaks_and_aggregate_vs_score_cell"}
for k, g in GEOM.items():
    layout["cells"].append({"file": f"{PAN}/{names[k]}.pdf", "x_mm": g["x"], "y_mm": g["y"], "w_mm": round(g["w"], 3), "letter": k})
json.dump(layout, open(f"{HERE}/layout_S9.json", "w"), indent=1)
LOG.append(f"layout: {W} x {HTOT} mm (budget 166.3)")
open(f"{HERE}/s9_panels.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
