"""Figure S7 panels f and g, final version.

Why: in the first S7 composite the x-axis titles of f and g ('Component 21 loadings/score') sit on the bottom page
edge; the descender of the 'g' in 'loadings' is cut by about 0.1 mm (pdftotext word box bottom 150.10 mm on a
150.00 mm page; visible at 2400 dpi).

Fix: identical to the f/g block of S7/scripts/s7_panels_final_size.py (same data, functions, colours, labels, ticks,
ring), except that the axes bottom is raised from 6.6 mm to 7.0 mm in the 40 x 50 mm cell (axes top unchanged), which
leaves about 0.3 mm below the descenders. Writes only to <G_ROS_FIGURE_OUT>/S7/panels/. Logs aggregates only.
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \\
     python s7fg_fix.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
from vp_common import *          # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
PAN = f"{FIGOUT}/S7/panels"
os.makedirs(PAN, exist_ok=True)
MM = 1 / 25.4
LOG = []

# style: verbatim from s7_panels_final_size.py
mpl.rcParams["pdf.fonttype"] = 42
sns.set(style="white", context="paper")
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "mathtext.default": "regular",
    "font.size": 6.5, "axes.labelsize": 7, "axes.titlesize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.labelpad": 2.0, "axes.titlepad": 2.0, "patch.linewidth": 0.5, "lines.linewidth": 0.8,
    "axes.edgecolor": "black", "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
})
TOP = 3.0
W_CELL, H_CELL = 40.0, 50.0
FB = 7.0          # was 6.6

RNA = rna_input()
exR = excl_mask("RNA"); keepR = ~exR


def top50_agg(k_, ct):
    L = comp_loadings(k_, "RNA"); X = L.X.values; P = L.P.values
    idx = np.where(P > 0.5)[0]; idx = idx[np.argsort(-np.abs(X[idx]))][:50]
    M = np.asarray(RNA[CTS_BLOCK.index(ct)][:, idx], float)
    w_ = X[idx] * bcor(k_)[ct]
    ag, n_ = zagg(M[keepR], w_)
    out = np.full(276, np.nan); out[keepR] = ag
    return out, n_


def gene(k_, g_, ct):
    j = list(comp_loadings(k_, "RNA").index).index(g_)
    y_ = np.asarray(RNA[CTS_BLOCK.index(ct), :, j], float).copy(); y_[exR] = np.nan
    return y_


a21 = score(21)
assert not outliers_30iqr(a21).any()
yf = gene(21, "OR2L3", "Exc"); yg, n_agg = top50_agg(21, "Exc")
ext = most_extreme(a21)
for nm, yy_, lab in (("S7f_C21_OR2L3_Exc_vs_score_final_v2", yf, "OR2L3 in Exc (CPM)"),
                     ("S7g_C21_aggregate_Exc_vs_score_final_v2", yg, "top-50 gene aggregate in Exc\nweighted mean z")):
    fig = plt.figure(figsize=(W_CELL * MM, H_CELL * MM))
    fL = 10.4
    ax = fig.add_axes([fL / W_CELL, FB / H_CELL, (W_CELL - fL - 3.6) / W_CELL, (H_CELL - TOP - FB) / H_CELL])
    ok = ~np.isnan(yy_)
    ax.scatter(a21[ok], yy_[ok], color=CTCOL["Exc"], s=3.0, alpha=0.8, linewidths=0, zorder=2)
    if not np.isnan(yy_[ext]):
        ax.scatter([a21[ext]], [yy_[ext]], s=3.0 * 3.2, facecolors="none", edgecolors=EXTRING, linewidths=0.5, zorder=3)
    sns.despine(ax=ax)
    ax.set_xlabel("Component 21 loadings/score"); ax.set_ylabel(lab)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10 if "OR2L3" in nm else 1))
    fig.savefig(f"{PAN}/{nm}.pdf", dpi=600)
    plt.close(fig)
    r_, p_, n_ = spearman(yy_, a21)
    m_ = ok.copy(); m_[ext] = False
    LOG.append(f"{nm}: n {n_}; Spearman {r_:+.3f}; without the most extreme donor "
               f"{ss.spearmanr(yy_[m_], a21[m_])[0]:+.3f}" + (f"; aggregate genes {n_agg}" if "aggregate" in nm else ""))

open(f"{PAN}/s7fg_fix.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
