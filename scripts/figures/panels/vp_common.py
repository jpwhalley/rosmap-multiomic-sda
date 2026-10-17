"""Shared style, paths and loaders for the v23 figure panels (npj Aging round 2, 2026-10-13).

Style = the v22 panels (ros3b_plot_outliers.py, ros6a_cross_clin_discrete.py): seaborn white/talk, pdf.fonttype 42,
Arial, the v22 cell-type palette and order (Ast Exc Inh Mic OPC Oli), muted blue = male / muted green = female,
'loadings/score' for component scores, no titles or in-panel statistics (the legends carry them).

Reads the SDA outputs, the analysis outputs (scripts/round2) and the catalogue tables; all locations come from
figure_paths.py (environment variables). Writes only under <G_ROS_FIGURE_OUT>/panels. Per-donor values are drawn as
points but never printed; logs hold aggregates only (no donor IDs, no per-donor clinical values)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats as ss

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # scripts/figures
from figure_paths import (RESULTS as R2, ROSMAP as ROS, SDA, TABLES as NEXT, PRIVATE, REFERENCES as REFS,
                          CLINICAL as CLIN, SEAAD, FIGOUT)
RA = f"{R2}/revision_analyses"
OUT = f"{FIGOUT}/panels"
DCACHE = f"{R2}/component_dossier/cache"
BCOR = f"{NEXT}/cell_type_loadings_corrected.csv"
os.makedirs(OUT, exist_ok=True)

CTS_BLOCK = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]          # rna.txt block order (cell_types.txt)
CT_ORDER = ["Ast", "Exc", "Inh", "Mic", "OPC", "Oli"]           # v22 plotting order
CTCOL = {"Ast": "#D85396", "Oli": "#8BC34A", "Inh": "#303F9F", "OPC": "#E5A43A", "Exc": "#D32F2F", "Mic": "#4DB6E7"}
CTNAME = {"Ast": "astrocytes", "Oli": "oligodendrocytes", "Inh": "inhibitory neurons", "OPC": "OPCs",
          "Exc": "excitatory neurons", "Mic": "microglia"}


def style():
    mpl.rcParams["pdf.fonttype"] = 42
    sns.set(style="white", context="talk")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["mathtext.default"] = "regular"


style()
MUTED = sns.color_palette("muted")
MALE, FEMALE = MUTED[0], MUTED[2]
SEXCOL = {"male": MALE, "female": FEMALE}
CLS = {"distributed": "#0072B2", "few-donor": "#E69F00", "concentrated": "#D55E00"}   # as the structure drafts
EXTRING = "#111111"


def donors():
    return [l.strip() for l in open(f"{SDA}/md_samples.txt") if l.strip()]


_SC = None
def score(k):
    """Deposited consensus sample score of component k (IDs used here are unchanged in the 547 catalogue)."""
    global _SC
    if _SC is None:
        _SC = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0); _SC.columns = _SC.columns.astype(str)
        assert list(_SC.columns) == donors()
    return _SC.loc[k].values.astype(float)


def clinical():
    c = pd.read_csv(CLIN); c.index = c.projid.astype(int).astype(str)
    return c.loc[donors()]


def female():
    return clinical().msex.values == 0


def flags():
    key = pd.read_csv(f"{PRIVATE}/donor_key.csv")
    d = np.array(donors()); out = {}
    for lab, pid in zip(key.label, key.projid.astype(str)):
        out[lab] = d == pid
        assert out[lab].sum() == 1, lab
    return out


def excl_mask(modality):
    """Identity-flagged profiles of the affected modality (identity_checks 17_identity_by_assay):
    snRNA-seq: RNA-1 (not own), RNA-2 (sex-discordant, unverifiable) -> n 274;
    H3K9ac: CHIP-1, CHIP-2 (sex-discordant) and CHIPID-1, CHIPID-2 (same-sex swaps) -> n 189 of 193;
    methylation: none."""
    F = flags(); m = np.zeros(276, bool)
    labs = {"RNA": ("RNA-1", "RNA-2"), "chip": ("CHIP-1", "CHIP-2", "CHIPID-1", "CHIPID-2"), "meth": ()}[modality]
    for l in labs:
        m |= F[l]
    return m


def rna_input():
    """SDA RNA input (unlogged CPM), shape (6 cell types [block order], 276 donors, 25,098 genes)."""
    return np.load(f"{DCACHE}/input_RNA.npy", mmap_mode="r").reshape(6, 276, -1)


def mod_input(m):
    return np.load(f"{DCACHE}/input_{m}.npy", mmap_mode="r")


_LD = None
def comp_loadings(k, m):
    global _LD
    if _LD is None:
        _LD = dict(np.load(f"{DCACHE}/loadings.npz", allow_pickle=True))
    ci = list(_LD["comps"]).index(k)
    return pd.DataFrame({"X": _LD[f"{m}_X"][ci], "P": _LD[f"{m}_P"][ci]}, index=list(_LD[f"{m}_feats"]))


def bcor(k):
    return pd.read_csv(BCOR, index_col=0).loc[k]


def rna_gene(g, ct):
    genes = list(comp_loadings(74, "RNA").index)
    j = genes.index(g)
    return np.asarray(rna_input()[CTS_BLOCK.index(ct), :, j], float)


def zagg(M, w):
    """Loading-weighted mean of column-standardised values (donors x features), over the rows given."""
    M = np.asarray(M, float)
    mu = np.nanmean(M, 0); sd = np.nanstd(M, 0); keep = sd > 0
    Z = (M[:, keep] - mu[keep]) / sd[keep]
    w = np.asarray(w, float)[keep]
    return Z @ w / np.abs(w).sum(), int(keep.sum())


def robust_z(x):
    x = np.asarray(x, float); med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / mad


def most_extreme(a):
    return int(np.argmax(np.abs(robust_z(a))))


def outliers_30iqr(a):
    q1, q3 = np.percentile(a, [25, 75]); iqr = q3 - q1
    return (a < q1 - 30 * iqr) | (a > q3 + 30 * iqr)


def spearman(x, y):
    ok = ~np.isnan(x) & ~np.isnan(y)
    r, p = ss.spearmanr(x[ok], y[ok]); return float(r), float(p), int(ok.sum())


def spearman_ci(x, y, B=2000, seed=1):
    ok = ~np.isnan(x) & ~np.isnan(y); x = np.asarray(x)[ok]; y = np.asarray(y)[ok]; n = len(x)
    rho, p = ss.spearmanr(x, y)
    rng = np.random.default_rng(seed); bs = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n); bs[b] = ss.spearmanr(x[i], y[i])[0]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(n=n, rho=float(rho), p=float(p), lo=float(lo), hi=float(hi))


def save(fig, folder, name, log=None):
    d = f"{OUT}/{folder}"; os.makedirs(d, exist_ok=True)
    fig.savefig(f"{d}/{name}.pdf", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    if log is not None:
        log.append(f"wrote {folder}/{name}.pdf")


def sexlegend(ax, n_m=None, n_f=None, loc="best", **kw):
    from matplotlib.lines import Line2D
    lm = "Male" if n_m is None else f"Male (n = {n_m})"
    lf = "Female" if n_f is None else f"Female (n = {n_f})"
    h = [Line2D([], [], color=MALE, marker="o", ls="None", markersize=10, label=lm),
         Line2D([], [], color=FEMALE, marker="o", ls="None", markersize=10, label=lf)]
    return ax.legend(handles=h, loc=loc, frameon=False, **kw)


def broken_x_scatter(fig, spec, a, series, xlabel, ylabel, ring=True, ylog=None, legend=None):
    """Scatter of y against the score a with a broken x-axis when donors lie beyond 30 x IQR (the paper's outlier
    rule). series: list of (y, colour, size, label). The most extreme donor (largest |robust z| of a) is ringed."""
    out = outliers_30iqr(a)
    ext = most_extreme(a)
    if out.any():
        neg = a[out].mean() < np.median(a)
        sub = spec.subgridspec(1, 2, width_ratios=[1, 4] if neg else [4, 1], wspace=0.07)
        axo = fig.add_subplot(sub[0, 0] if neg else sub[0, 1])
        axb = fig.add_subplot(sub[0, 1] if neg else sub[0, 0], sharey=axo)
        axes = [axo, axb] if neg else [axb, axo]
    else:
        axb = fig.add_subplot(spec); axo = None; axes = [axb]
    draw = [axb] if axo is None else [axb, axo]
    for (y, col, sz, lab) in series:
        ok = ~np.isnan(y)
        for ax in draw:
            ax.scatter(a[ok], y[ok], color=col, s=sz, alpha=0.8, linewidths=0, label=lab if ax is axb else None, zorder=2)
        if ring and not np.isnan(y[ext]):
            for ax in draw:
                ax.scatter([a[ext]], [y[ext]], s=sz * 3.2, facecolors="none", edgecolors=EXTRING, linewidths=1.6, zorder=3)
    for ax in axes:
        sns.despine(ax=ax)
        if ylog == "symlog":
            ax.set_yscale("symlog", linthresh=1); ax.set_ylim(bottom=-0.3)
    if axo is not None:
        bulk = a[~out]; pad = (bulk.max() - bulk.min()) * 0.05
        axb.set_xlim(bulk.min() - pad, bulk.max() + pad)
        axb.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        o = a[out]; span = max(np.abs(o).max() * 0.04, 1e-3)
        axo.set_xlim(o.min() - span, o.max() + span)
        axo.set_xticks([float(f"{o.mean():.2f}")])
        left, right = (axo, axb) if neg else (axb, axo)
        left.spines["right"].set_visible(False); right.spines["left"].set_visible(False)
        right.tick_params(labelleft=False, left=False)
        d = 0.015; kw = dict(color="k", clip_on=False, linewidth=1)
        dl, dr = (4 * d, d) if neg else (d, 4 * d)
        right.plot((-dr, +dr), (-d * 2, d * 2), transform=right.transAxes, **kw)
        left.plot((1 - dl, 1 + dl), (-d * 2, d * 2), transform=left.transAxes, **kw)
    axb.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    if legend:
        axb.legend(loc=legend, frameon=False, markerscale=1.3, fontsize=14)
    return axes
