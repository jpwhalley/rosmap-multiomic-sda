"""Figure S4 (Component 278 source data) panels redrawn at their final printed size for the composite.

Same data, statistics, colours and label text as panels/p04_c278.py (the S4 part, copied below; Fig 3f
is not drawn here). Only the drawing changed: each panel is drawn at exactly its cell size in the S4 layout (mm),
with explicit Arial sizes (ticks 6 pt, axis labels 6.5 pt, keys 6 pt), thinner lines and smaller markers to suit
the size. Two-line breaks were added to some labels so they fit (text unchanged). The S4a sex key moved from inside
the FKBP5-in-OPC axes to the strip just above panel a (it no longer fits inside an axes at this size).

Reads (read-only) through vp_common: the SDA outputs and the analysis outputs. Writes only to
<G_ROS_FIGURE_OUT>/S4/panels/. Logs aggregate statistics only (no donor IDs, no per-donor values); the log is compared
with <G_ROS_FIGURE_OUT>/panels/p04_c278.log (panels/p04_c278.py, run first) by check_against_p04()."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
import json
import numpy as np, pandas as pd
from scipy import stats as ss
from matplotlib.lines import Line2D
from vp_common import *          # noqa: F401,F403  (loaders, palette, style; makes no new files)

HERE = os.path.dirname(os.path.abspath(__file__))
PAN = f"{FIGOUT}/S4/panels"
os.makedirs(PAN, exist_ok=True)
LAYOUT = json.load(open(f"{HERE}/cells_S4.json"))     # cell sizes (mm), shared with the compose layout

# ------------------------------------------------------------------ print style (sizes in points)
TICK, LAB, KEY = 6.0, 6.5, 6.0
sns.set(style="white", context="paper")
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "mathtext.default": "regular", "axes.unicode_minus": True,
    "font.size": TICK, "axes.labelsize": LAB, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "legend.fontsize": KEY, "axes.linewidth": 0.6, "lines.linewidth": 0.8,
    "xtick.major.pad": 1.6, "ytick.major.pad": 1.6, "axes.labelpad": 2.0,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
MM = 1 / 25.4
MS_SCAT = 2.6            # scatter marker area (pt^2)


def mmfig(cell):
    w, h = LAYOUT[cell]["w_mm"], LAYOUT[cell]["h_mm"]
    fig = plt.figure(figsize=(w * MM, h * MM))
    fig._mm = (w, h)
    return fig


def axmm(fig, x, y, w, h, **kw):
    """Axes with its data rectangle at (x, y) mm from the figure's top-left, size w x h mm."""
    W, H = fig._mm
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def grid(fig, lefts, top, w, h, rows, gap_y, gap_x_list=None):
    """rows x len(lefts) axes; lefts are the x (mm) of each column's data rectangle."""
    return np.array([[axmm(fig, lefts[c], top + r * (h + gap_y), w, h) for c in range(len(lefts))] for r in range(rows)])


def text_check(fig, name, letter_zone=True):
    """All text inside the figure; no text overlaps another text or another axes' data rectangle; the letter
    zone (top-left 3.2 x 3.0 mm) is free. Returns a list of problems (printed)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    W, H = fig._mm; px = fig.dpi * MM
    items = []
    for ax in fig.axes:
        ts = []
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tk in axis.get_major_ticks():
                t = tk.label1
                if t.get_visible() and t.get_text() and lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:
                    ts.append(t)
        ts += [t for t in (ax.xaxis.label, ax.yaxis.label) if t.get_text()]
        leg = ax.get_legend()
        if leg:
            ts += list(leg.get_texts())
        for t in ts:
            items.append((ax, t.get_text().replace("\n", "|"), t.get_window_extent(r)))
    for t in fig.texts:
        items.append((None, t.get_text(), t.get_window_extent(r)))
    for lg in fig.legends:
        for t in lg.get_texts():
            items.append((None, t.get_text(), t.get_window_extent(r)))
    probs = []
    for ax, s, b in items:
        x0, x1 = b.x0 / px, b.x1 / px; y0, y1 = H - b.y1 / px, H - b.y0 / px
        if x0 < -0.05 or y0 < -0.05 or x1 > W + 0.05 or y1 > H + 0.05:
            probs.append(f"{name}: '{s}' outside the cell ({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f}) mm")
        if letter_zone and x0 < 3.2 and y0 < 3.0:
            probs.append(f"{name}: '{s}' enters the letter zone ({x0:.1f},{y0:.1f})")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            bi, bj = items[i][2], items[j][2]
            if bi.overlaps(bj) and min(bi.x1, bj.x1) - max(bi.x0, bj.x0) > 0.5 and min(bi.y1, bj.y1) - max(bi.y0, bj.y0) > 0.5:
                probs.append(f"{name}: '{items[i][1]}' overlaps '{items[j][1]}'")
    for ax, s, b in items:
        for ax2 in fig.axes:
            if ax2 is ax:
                continue
            bb = ax2.get_window_extent(r)
            if b.overlaps(bb) and min(b.x1, bb.x1) - max(b.x0, bb.x0) > 0.5 and min(b.y1, bb.y1) - max(b.y0, bb.y0) > 0.5:
                probs.append(f"{name}: '{s}' overlaps another axes' data area")
    for p in probs:
        print("CHECK", p)
    return probs


def save(fig, name, letter_zone=True):
    probs = text_check(fig, name, letter_zone)
    fig.savefig(f"{PAN}/{name}.pdf", facecolor="white")     # exact cell size: no bbox_inches
    plt.close(fig)
    LOG.append(f"wrote panels/{name}.pdf ({fig._mm[0]:.1f} x {fig._mm[1]:.1f} mm; layout problems: {len(probs)})")
    return probs


# ------------------------------------------------------------------ data (verbatim from p04_c278.py)
LOG = []
K = 278; a = score(K); fem = female(); sexlab = np.where(fem, "female", "male")
exR = excl_mask("RNA"); keep = ~exR
RNA = rna_input(); LR = comp_loadings(K, "RNA"); genes = list(LR.index)
Bc = bcor(K)[CTS_BLOCK]
LEAD = ["FKBP5", "SGK1", "ZBTB16"]
XL = "Component 278 loadings/score"
XL2 = "Component 278\nloadings/score"          # same text, broken for the narrow columns
def cpm(g, ct): return np.asarray(RNA[CTS_BLOCK.index(ct), :, genes.index(g)], float)


def sex_scatter(ax, y, ylab, s=MS_SCAT):
    for sx in ("male", "female"):
        m = keep & (sexlab == sx)
        ax.scatter(a[m], y[m], s=s, color=SEXCOL[sx], alpha=0.8, linewidths=0, rasterized=False)
    ax.set_ylabel(ylab); sns.despine(ax=ax)


def sexkey(n_m=None, n_f=None):
    lm = "Male" if n_m is None else f"Male (n = {n_m})"
    lf = "Female" if n_f is None else f"Female (n = {n_f})"
    return [Line2D([], [], color=MALE, marker="o", ls="None", markersize=3, label=lm),
            Line2D([], [], color=FEMALE, marker="o", ls="None", markersize=3, label=lf)]


KEYKW = dict(frameon=False, handlelength=0.8, handletextpad=0.3, borderpad=0.2, labelspacing=0.3, borderaxespad=0.2)
PROBS = []

# ------------------------------------------------------------------ S4a: 2 x 3, OPC top, Oli bottom
G = LAYOUT["a"]["grid"]
fig = mmfig("a")
axs = grid(fig, G["lefts"], G["top"], G["w"], G["h"], 2, G["gap_y"])
for r_, ct in enumerate(("OPC", "Oli")):
    for c_, g in enumerate(LEAD):
        ax = axs[r_, c_]
        y = np.log2(cpm(g, ct) + 1)
        sex_scatter(ax, y, f"{g} in {ct}\nlog2(CPM + 1)")
        rr, p, n = spearman(y[keep], a[keep])
        LOG.append(f"S4a {g} {ct}: n {n}; rho {rr:+.3f}")
xlim = (np.nanmin(a[keep]), np.nanmax(a[keep])); pad = (xlim[1] - xlim[0]) * 0.05
for ax in axs.flat:
    ax.set_xlim(xlim[0] - pad, xlim[1] + pad); ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
for ax in axs[0]: ax.tick_params(labelbottom=False)
for ax in axs[1]: ax.set_xlabel(XL2)
# sex key: in the strip above the top-right axes (n as in p04: kept males / kept females)
axs[0, 1].legend(handles=sexkey(int((keep & ~fem).sum()), int((keep & fem).sum())), loc="lower center",
                 bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2, **KEYKW)
PROBS += save(fig, "S4a_leading_genes_OPC_Oli_vs_score")

# ------------------------------------------------------------------ S4b: 2 x 2, all PIP genes (left), without the three (right)
G = LAYOUT["b"]["grid"]
pip = LR.P.values > 0.5; nolead = pip & ~np.isin(genes, LEAD)
fig = mmfig("b")
axs = grid(fig, G["lefts"], G["top"], G["w"], G["h"], 2, G["gap_y"])
for r_, ct in enumerate(("OPC", "Oli")):
    Rc = np.log2(np.asarray(RNA[CTS_BLOCK.index(ct)], float) + 1); w = LR.X.values * Bc[ct]
    for c_, (sel, lab) in enumerate(((pip, f"{int(pip.sum()):,} genes"),
                                     (nolead, f"{int(nolead.sum()):,} genes,\nwithout FKBP5,\nSGK1, ZBTB16"))):
        ag, n_ = zagg(Rc[:, sel], w[sel])
        sex_scatter(axs[r_, c_], ag, f"{ct}, {lab}\nweighted mean z")
        rr, p, n = spearman(ag[keep], a[keep])
        LOG.append(f"S4b {ct} aggregate of {int(sel.sum())} PIP genes ({n_} with variance): n {n}; rho {rr:+.4f}")
for ax in axs.flat:
    ax.set_xlim(xlim[0] - pad, xlim[1] + pad); ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
for ax in axs[0]: ax.tick_params(labelbottom=False)
for ax in axs[1]: ax.set_xlabel(XL2)
axs[0, 0].legend(handles=sexkey(), loc="lower right", **KEYKW)
PROBS += save(fig, "S4b_aggregates_with_and_without_leading_genes")

# ------------------------------------------------------------------ S4c: female - male (verbatim statistics)
rows = []
for ct in ("OPC", "Oli"):
    for g in LEAD:
        y = np.log2(cpm(g, ct) + 1)
        yf, ym = y[keep & fem], y[keep & ~fem]
        rng = np.random.default_rng(24); bs = [rng.choice(yf, len(yf)).mean() - rng.choice(ym, len(ym)).mean() for _ in range(4000)]
        rows.append(dict(lab=f"{g}, {ct}", d=yf.mean() - ym.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         p=ss.mannwhitneyu(yf, ym).pvalue, nf=len(yf), nm=len(ym)))
D = pd.DataFrame(rows)
G = LAYOUT["c"]["ax"]
fig = mmfig("c"); ax = axmm(fig, G["x"], G["y"], G["w"], G["h"])
yy = np.arange(len(D))[::-1]
ax.errorbar(D.d, yy, xerr=[D.d - D.lo, D.hi - D.d], fmt="o", color="k", ms=3, lw=0.8, capsize=0)
ax.axvline(0, color="grey", lw=0.5, ls=":")
ax.set_yticks(yy); ax.set_yticklabels(D.lab); ax.set_ylim(-0.6, len(D) - 0.4)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.set_xlabel("female − male,\nmean log2(CPM + 1)"); sns.despine(ax=ax)
PROBS += save(fig, "S4c_female_minus_male_leading_genes")
for _, r in D.iterrows():
    LOG.append(f"S4c {r.lab}: {r.d:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]; MW p {r.p:.2g}; n {r.nf} female, {r.nm} male")

# ------------------------------------------------------------------ S4d1: SOX6 and BCAS1 (rows) in OPC and Oli (columns)
G = LAYOUT["d1"]["grid"]
fig = mmfig("d1")
axs = grid(fig, G["lefts"], G["top"], G["w"], G["h"], 2, G["gap_y"])
for r_, g in enumerate(("SOX6", "BCAS1")):
    for c_, ct in enumerate(("OPC", "Oli")):
        y = np.log2(cpm(g, ct) + 1); sex_scatter(axs[r_, c_], y, f"{g} in {ct}\nlog2(CPM + 1)")
        rr, p, n = spearman(y[keep], a[keep]); LOG.append(f"S4d1 {g} {ct}: n {n}; rho {rr:+.3f}")
for ax in axs.flat:
    ax.set_xlim(xlim[0] - pad, xlim[1] + pad); ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
for ax in axs[0]: ax.tick_params(labelbottom=False)
for ax in axs[1]: ax.set_xlabel(XL2)
axs[0, 0].legend(handles=sexkey(), loc="lower left", **KEYKW)
PROBS += save(fig, "S4d1_SOX6_BCAS1_vs_score")

# ------------------------------------------------------------------ S4d2: lineage markers (second-implementation v80 table)
T = pd.read_csv(f"{RA}/verify_c74_c278/out/v80_c278_markers.csv").dropna(subset=["rho_OPC"])
GRP = {"OPC": "OPC", "committed": "committed", "mature": "mature"}
G = LAYOUT["d2"]["ax"]
fig = mmfig("d2"); ax = axmm(fig, G["x"], G["y"], G["w"], G["h"])
yy = np.arange(len(T))[::-1]
for off, ct in ((-0.18, "OPC"), (0.18, "Oli")):
    ax.errorbar(T[f"rho_{ct}"], yy + off, xerr=[T[f"rho_{ct}"] - T[f"lo_{ct}"], T[f"hi_{ct}"] - T[f"rho_{ct}"]], fmt="o",
                ms=2.2, lw=0.7, color=CTCOL[ct], capsize=0, label=f"in {ct}")
ax.axvline(0, color="grey", lw=0.5, ls=":")
ax.set_yticks(yy); ax.set_yticklabels([f"{g} ({GRP[gr]})" for g, gr in zip(T.gene, T.group)])
ax.set_ylim(-0.7, len(T) - 0.3)
nb = T.group.value_counts()[["OPC", "committed"]].cumsum().values
for b in nb: ax.axhline(len(T) - b - 0.5, color="k", lw=0.5)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax.set_xlabel("Spearman ρ\nwith the score"); sns.despine(ax=ax)
# key in the strip above the axes (inside the axes it now covers data; was lower right in p04)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, columnspacing=1.0, handlelength=1.4,
          **{k: v for k, v in KEYKW.items() if k != "handlelength"})
PROBS += save(fig, "S4d2_lineage_markers_rho", letter_zone=False)
LOG.append(f"S4d2 markers drawn: {len(T)} (CLDN11 not in the SDA gene list); SOX6 OPC {T.set_index('gene').loc['SOX6','rho_OPC']:+.2f}, "
           f"Oli {T.set_index('gene').loc['SOX6','rho_Oli']:+.2f}; BCAS1 {T.set_index('gene').loc['BCAS1','rho_OPC']:+.2f} / {T.set_index('gene').loc['BCAS1','rho_Oli']:+.2f}")

# ------------------------------------------------------------------ S4d3: OPC share of OPC + Oli nuclei
qc = pd.read_csv(f"{R2}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str); qc = qc.loc[donors()]
share = (qc.nuclei_OPC / (qc.nuclei_OPC + qc.nuclei_Oli)).values
G = LAYOUT["d3"]["ax"]
fig = mmfig("d3"); ax = axmm(fig, G["x"], G["y"], G["w"], G["h"])
sex_scatter(ax, share, "OPC share of OPC + Oli nuclei"); ax.set_xlabel(XL)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
PROBS += save(fig, "S4d3_OPC_share_vs_score", letter_zone=False)
rr, p, n = spearman(share[keep], a[keep]); LOG.append(f"S4d3 OPC share: n {n}; rho {rr:+.3f}; p {p:.2g}")

# ------------------------------------------------------------------ S4d4: marker modules (second implementation, v80)
Mo = pd.read_csv(f"{RA}/verify_c74_c278/out/v80_c278_modules.csv")
Mo = Mo[Mo.ct != "composition"]
G = LAYOUT["d4"]["ax"]
fig = mmfig("d4"); ax = axmm(fig, G["x"], G["y"], G["w"], G["h"])
yy = np.arange(len(Mo))[::-1]
for (_, r), y_ in zip(Mo.iterrows(), yy):
    ax.errorbar(r.rho, y_, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt="o", color=CTCOL[r.ct], ms=3, lw=0.8, capsize=0)
ax.axvline(0, color="grey", lw=0.5, ls=":")
ax.set_yticks(yy); ax.set_yticklabels([f"{m} module, in {c}" for m, c in zip(Mo.module, Mo.ct)])
ax.set_ylim(-0.6, len(Mo) - 0.4); ax.set_xlim(-0.41, 0.06)       # auto limits were about (-0.39, 0.06)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.set_xlabel("Spearman ρ\nwith the score"); sns.despine(ax=ax)
PROBS += save(fig, "S4d4_marker_modules_rho", letter_zone=False)
for _, r in Mo.iterrows(): LOG.append(f"S4d4 {r.module} module in {r.ct}: rho {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]")


# ------------------------------------------------------------------ same statistics as the v23 panels?
def check_against_p04():
    ref = [l for l in open(f"{OUT}/p04_c278.log").read().splitlines() if l.startswith("S4")]
    mine = [l for l in LOG if l.startswith("S4")]
    miss = [l for l in ref if l not in mine]
    LOG.append(f"statistics lines vs p04_c278.log: {len(ref)} reference S4 lines, {len(ref) - len(miss)} identical"
               + ("" if not miss else "; DIFFERENT: " + " || ".join(miss)))


check_against_p04()
LOG.append(f"layout problems in all panels: {len(PROBS)}")
open(f"{PAN}/s4_panels.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
