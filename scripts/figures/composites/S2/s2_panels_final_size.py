"""Figure S2 (Component 21) panels a-g redrawn at their final printed size (v23 supplement, 2026-10-16).

Every panel is drawn at exactly its cell size in the composite (supplement \\textwidth = 142.9 mm, placed at 100 %),
with explicit Arial sizes (axis titles 7 pt, ticks/keys/titles 6.5 pt, in-plot gene/SNP labels 6 pt) and pdf.fonttype
42, the axes placed in millimetres (no bbox_inches='tight'), so that the composite prints every word at >= 6 pt.

Same data, same computations, same colours and labels as the published panels:
  a  21_samples.pdf    <- V22CODE ros3b_plot_outliers.py (samples block): C21 sample score vs sample rank by age at
                          death ('90+' as 90, pandas quicksort; the order is checked marker-for-marker against the PDF),
                          coloured by reported sex (muted blue/green, black edge), n 276.
  b  21_cell_type.pdf  <- ros3b (cell-type block): row 21 of the corrected table, cell_type_loadings_corrected.csv,
                          Ast Exc Inh Mic OPC Oli, v22 colours (ros3b read treatment_final.csv; its row 21 is
                          bit-for-bit the corrected row).
  c  21_RNA.pdf        <- ros3b (RNA block): PIP > 0.5 genes at their Ensembl GRCh38.100 start, chromosome blocks as
                          in ros3b (block width = largest plotted start; colours cycle #6495ED #778899 #008080 #20B2AA),
                          8 largest loadings labelled black, 8 smallest grey (marker positions checked against the PDF).
                          adjustText is not installed, so the 16 labels are placed by a deterministic 1-D spreader with
                          thin leader lines, and the upper y limit is raised to make room for the black labels inside
                          the frame (ros3b: min x 1.1 to max x 1.1; here the top is max x 1.1 plus one label row).
  d  21_Manhattan_Component.png <- V22CODE ros5_Manhattan_Plot_maf.py (assocplots.manhattan, not installed): replotted
                          as vector from the same GEMMA output (<G_ROS_ROSMAP_DIR>/oth/snps/gemma/output/results21.abbr.txt,
                          -lmm 4, p_score, n 228; checked: top SNP rs6664332, p 4.2e-16, the chromosome boundaries match
                          the PNG's grey lines to < 0.5 px). Same colours (#1b9e77 #d95f02 #7570b3 #e7298a by chromosome),
                          lines at -log10 p 10 (solid green 'g') and 15 (dashed red 'r'), light-grey boundaries, odd
                          chromosome labels, 'rs6664332' label. Points rasterised at 600 dpi (631,829 SNPs); text and
                          lines vector. y limit 0-18 (PNG 0-16) so that the SNP label sits inside the frame.
  e  21_OR2L3_rs6664332_trans_eQTL_tissues.pdf <- V22CODE ros8_eQTLs.py: log2(CPM + 1) OR2L3 per cell type by
                          rs6664332 genotype (PLINK bed read directly, = pysnptools count_A1=False), OLS F-test (anova_lm
                          typ 2); seaborn boxplot (Set2, saturation .75, whis 1.5, no fliers) + grey stripplot (alpha .5).
                          Titles are the v22 titles, broken after the cell type; p values and genotype n are recomputed
                          and asserted equal to the published panel/legend. Strip jitter is seeded here (random in v22).
  f, g  S2a/S2b panels <- panels/p06_c21_single.py (same functions and data; n 274).

Reads only (locations from figure_paths.py): the SDA outputs, the clinical table, the genotypes, pseudobulk and
GEMMA output under G_ROS_ROSMAP_DIR, the GRCh38.100 annotation, the analysis caches through vp_common, and the
version-1 panels for the checks. Prints and logs aggregates only (no donor
IDs, no per-donor values).
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \\
     python s2_panels_final_size.py
"""
import os, re, sys, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
sys.path.insert(0, HERE)
from vp_common import *          # noqa: F401,F403  (loaders, CTCOL, MALE/FEMALE; its talk style is overridden below)
import matplotlib.lines as mlines
from colorsys import rgb_to_hls
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pdf_streams import streams

from figure_paths import V1_PANELS
PAN = f"{FIGOUT}/S2/panels"
os.makedirs(PAN, exist_ok=True)
MM = 1 / 25.4
LOG = []
CODING = V1_PANELS

# ------------------------------------------------------------------ style (as the S1 rebuild: seaborn white, print sizes)
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
LABFS = 6.0      # in-plot gene / SNP labels

# ------------------------------------------------------------------ geometry (mm). Cells = composite cells.
W = 142.9
CELL = {  # name: (x, y_top, w, h) in the composite
    "a": (0.0, 0.0, 88.0, 26.0), "b": (91.0, 0.0, 51.9, 26.0),
    "c": (0.0, 29.5, W, 33.0),
    "d": (0.0, 66.0, W, 30.5),
    "e": (0.0, 100.0, 57.5, 50.0), "f": (60.5, 100.0, 40.0, 50.0), "g": (102.9, 100.0, 40.0, 50.0),
}
HEIGHT = 150.0
TOP = 3.0        # free band at the top of each cell (panel letter)
AX_L = 11.0      # common left edge of the axes of a, c, d


def fig_mm(name):
    x, y, w, h = CELL[name]
    return plt.figure(figsize=(w * MM, h * MM)), w, h


def ax_mm(fig, w, h, l, b, aw, ah):
    return fig.add_axes([l / w, b / h, aw / w, ah / h])


def savep(fig, name):
    p = f"{PAN}/{name}.pdf"
    fig.savefig(p, dpi=600)
    plt.close(fig)
    LOG.append(f"wrote panels/{name}.pdf")


def spread(centres, sizes, gap, lo=-np.inf, hi=np.inf):
    """1-D label spreader: positions as close as possible to the centres with neighbours >= (s_i + s_j)/2 + gap apart,
    kept within [lo, hi] (pool-adjacent-violators on the offsets). Input in any order; returns positions in that order."""
    c = np.asarray(centres, float); s = np.asarray(sizes, float); o = np.argsort(c)
    c, s = c[o], s[o]
    need = np.concatenate([[0], np.cumsum((s[1:] + s[:-1]) / 2 + gap)])   # minimal offsets from the first label
    t = c - need                                                           # free start positions
    blocks = []                                                            # isotonic (non-decreasing) fit of t
    for v in t:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2 = blocks.pop(); v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    start = np.concatenate([[b[0]] * b[1] for b in blocks])
    pos = start + need
    lo_s = lo + s[0] / 2; hi_s = hi - s[-1] / 2
    if pos[0] < lo_s:
        pos = np.maximum(pos, lo_s + need)
    if pos[-1] > hi_s:
        pos = np.minimum(pos, hi_s - (need[-1] - need))
    out = np.empty_like(pos); out[o] = pos
    return out


def text_size_mm(fig, s, fs, rotation=0):
    t = fig.text(0, 0, s, fontsize=fs, rotation=rotation)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi * 25.4, bb.height / fig.dpi * 25.4


# ================================================================== a: sample scores (ros3b samples block)
ROSD = ROS
samples = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0)
clin = pd.read_csv(CLIN, index_col=0)
samples.columns = samples.columns.astype(str); clin.index = clin.index.astype(str)
metadata = clin.loc[samples.columns.values][["Study", "msex", "age_death"]].reindex(samples.columns)
S = samples.T; S.index = S.index.astype(str)
S = S.merge(metadata[["msex", "age_death"]], left_index=True, right_index=True)
S["Sex"] = S["msex"].apply(lambda v: "Male" if v == 1.0 else "Female")
S["age_death_numeric"] = pd.to_numeric(S["age_death"], errors="coerce").fillna(90)
S = S.sort_values("age_death_numeric")          # ros3b: default kind (quicksort); verified below
S["x_pos"] = range(len(S))
k21 = [c for c in S.columns if str(c) == "21"][0]
color_map = {"Male": MALE, "Female": FEMALE}

# check against the published panel: marker y positions must be an exact linear function of the scores in this order
seg = streams(f"{CODING}/outlier/21/21_samples.pdf")[0].decode("latin1")
seg = seg[seg.find("/P0 Do") - 400:]
xx = yy = 0.0; pts = []; col = None
for tok in re.finditer(r"([\d.\-]+) ([\d.\-]+) ([\d.\-]+) rg|1 0 0 1 ([\d.\-]+) ([\d.\-]+) cm /P0 Do", seg):
    if tok.group(1):
        col = float(tok.group(3))
    else:
        xx += float(tok.group(4)); yy += float(tok.group(5)); pts.append((yy, col))
Ppdf = np.array([p[0] for p in pts]); sex_pdf = np.where(np.array([p[1] for p in pts]) > 0.7, "Male", "Female")
A_ = np.vstack([S[k21].values, np.ones(len(S))]).T
coef = np.linalg.lstsq(A_, Ppdf, rcond=None)[0]
res_a = float(np.abs(A_ @ coef - Ppdf).max())
assert len(Ppdf) == 276 and res_a < 0.01 and (sex_pdf == S.Sex.values).all(), "panel a order differs from 21_samples.pdf"
LOG.append(f"a: n {len(S)} (male {int((S.Sex == 'Male').sum())}, female {int((S.Sex == 'Female').sum())}); "
           f"order = 21_samples.pdf (max marker residual {res_a:.4f} pt, sex 276/276)")

fig, w, h = fig_mm("a")
ax = ax_mm(fig, w, h, AX_L, 4.6, 61.5, h - TOP - 4.6)
ax.scatter(S["x_pos"], S[k21], c=S["Sex"].map(color_map), edgecolors="k", alpha=0.8, linewidth=0.2, s=9)
ax.set_xticks([]); ax.set_xlabel("Samples", labelpad=2.5); ax.set_ylabel("loadings/score")
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.04))
ax.set_xlim(-276 * 0.05, 275 + 276 * 0.05)
h_ = [mlines.Line2D([], [], color=color_map[s_], marker="o", linestyle="None", markersize=3.5, label=s_)
      for s_ in ("Male", "Female")]
ax.legend(handles=h_, loc="upper left", bbox_to_anchor=(1.0, 1.04), frameon=False, handletextpad=0.2,
          borderaxespad=0.3, labelspacing=0.35)
savep(fig, "S2a_C21_samples_final")

# ================================================================== b: cell-type loadings (ros3b cell-type block)
tissues = pd.read_csv(BCOR, index_col=0)[["Ast", "Exc", "Inh", "Mic", "OPC", "Oli"]]
bc = bcor(21)
assert np.allclose(tissues.loc[21].values, bc[["Ast", "Exc", "Inh", "Mic", "OPC", "Oli"]].values)
LOG.append("b: " + ", ".join(f"{c_} {tissues.loc[21, c_]:+.3f}" for c_ in tissues.columns) +
           " (corrected table)")
fig, w, h = fig_mm("b")
ax = ax_mm(fig, w, h, 9.5, 4.0, w - 9.5 - 0.8, h - TOP - 4.0)
y_pos = np.arange(6)
ax.bar(y_pos, list(tissues.loc[21]), align="center", color=[CTCOL[x_] for x_ in tissues.columns])
ax.set_xticks(y_pos); ax.set_xticklabels(list(tissues.columns))
ax.set_ylabel("loadings/score")
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
savep(fig, "S2b_C21_cell_type_final")

# ================================================================== c: gene loadings by chromosome (ros3b RNA block)
genes1 = pd.read_csv(f"{SDA}/genes_final_RNA.csv", index_col=0)
pip1 = pd.read_csv(f"{SDA}/pip_final_RNA.csv", index_col=0)
expr_info = pd.read_csv(f"{REFS}/Homo_sapiens.GRCh38.100.txt",
                        index_col=0, sep=",")
expr_info["new_index"] = expr_info["gene_name"]; expr_info.set_index("new_index", inplace=True)
common_genes = expr_info.index.intersection(genes1.columns)
bulk_info = expr_info.loc[common_genes]
missing = sorted(set(genes1.columns) - set(common_genes))
missing_info = pd.DataFrame("N/A", index=missing, columns=["chrom", "chromStart", "chromEnd"])
bulk_info = pd.concat([bulk_info, missing_info])
bulk_info["chrom"] = bulk_info["chrom"].astype(str)
k = 21
sel = pip1.T[pip1.loc[k] > 0.5].index.values
pip_genes = genes1[sel]
pip_info = bulk_info[bulk_info.index.isin(sel)]
chrs_list = [str(i) for i in range(1, 23)] + ["X", "Y"]
if len(pip_genes.columns.values) > 16:
    large = pip_genes.loc[k].nlargest(n=8); small = pip_genes.loc[k].nsmallest(n=8)
else:
    large = pip_genes.loc[k][pip_genes.loc[k] > 0]; small = pip_genes.loc[k][pip_genes.loc[k] <= 0]
chr_max_pos = {'1': 249230899.0, '2': 243064438.0, '3': 197955065.0, '4': 191012160.0, '5': 180794288.0,
               '6': 170886450.0, '7': 158820866.0, '8': 146277826.0, '9': 141069493.0, '10': 135497182.0,
               '11': 134248398.0, '12': 133758108.0, '13': 115079831.0, '14': 107282052.0, '15': 102501016.0,
               '16': 90244125.0, '17': 81037567.0, '18': 77915115.0, '19': 59086812.0, '20': 62958768.0,
               '21': 48110787.0, '22': 51205920.0, 'X': 155249967.0, 'Y': 59330252.0}
COLS_RNA = ['#6495ED', '#778899', '#008080', '#20B2AA']
shift = np.array([0.0]); blocks = []; labs = []
for ii, i in enumerate(chrs_list):
    starts = list(pip_info["chromStart"][pip_info["chrom"] == i])
    if starts == []:
        x_ = shift[-1] + np.array([chr_max_pos[i]]); y_ = []; z_ = []
    else:
        x_ = shift[-1] + np.array(starts, float)
        y_ = list(pip_genes[pip_info["gene_name"][pip_info["chrom"] == i].values].loc[k])
        z_ = list(pip_info["gene_name"][pip_info["chrom"] == i])
    blocks.append((x_ if len(y_) else np.array([]), np.array(y_, float), COLS_RNA[ii % 4]))
    shift = np.append(shift, np.max(x_)) if len(x_) else np.append(shift, shift[-1])
    for hh, val in enumerate(z_):
        if val in large:
            labs.append((val, x_[hh], y_[hh], "black"))
        elif val in small:
            labs.append((val, x_[hh], y_[hh], "gray"))
ticks_c = (shift[1:] + shift[:-1]) / 2.
yv = list(pip_genes.loc[k]); ymin_c, ymax_c = min(yv) * 1.1, max(yv) * 1.1        # ros3b (mixed signs)
npts = sum(len(b_[1]) for b_ in blocks)

# check against the published panel: per chromosome block, the marker positions (PDF pt) must be one affine map of (x, y)
segc = streams(f"{CODING}/outlier/21/21_RNA.pdf")[0].decode("latin1")
pdf_blocks = []
for blk in re.findall(r"q\n(1 0 0 1 [^Q]*?)Q", segc):           # one q..Q group per chromosome (24, all non-empty)
    xs = ys = 0.0; cur = []
    for m in re.finditer(r"1 0 0 1 ([\d.\-e]+) ([\d.\-e]+) cm /M\d+ Do", blk):
        xs += float(m.group(1)); ys += float(m.group(2)); cur.append((xs, ys))
    pdf_blocks.append(np.array(cur))
assert len(pdf_blocks) == 24 and all(len(b_[1]) == len(pb) for b_, pb in zip(blocks, pdf_blocks)), \
    "panel c: per-chromosome marker counts differ from 21_RNA.pdf"
mine = np.vstack([np.c_[b_[0], b_[1]] for b_ in blocks]); th = np.vstack(pdf_blocks)
Ax = np.c_[mine[:, 0], np.ones(len(mine))]; Ay = np.c_[mine[:, 1], np.ones(len(mine))]
cx = np.linalg.lstsq(Ax, th[:, 0], rcond=None)[0]; cy = np.linalg.lstsq(Ay, th[:, 1], rcond=None)[0]
rx = np.abs(Ax @ cx - th[:, 0]).max(); ry = np.abs(Ay @ cy - th[:, 1]).max()
LOG.append(f"c: {npts} plotted gene markers (PIP > 0.5 genes: {len(sel)}); vs 21_RNA.pdf per-chromosome counts equal, "
           f"max residual x {rx:.4f} pt, y {ry:.4f} pt")
assert rx < 0.05 and ry < 0.05, "panel c differs from 21_RNA.pdf"
LOG.append("c: black (8 largest): " + ", ".join(large.index) + "; grey (8 smallest): " + ", ".join(small.index))

fig, w, h = fig_mm("c")
cb = 7.6; aw = w - AX_L - 1.2; ah = h - TOP - cb
ax = ax_mm(fig, w, h, AX_L, cb, aw, ah)
for x_, y_, c_ in blocks:
    if len(y_):
        ax.plot(x_, y_, ".", color=c_, markersize=3.2, markeredgewidth=0)
ax.set_xticks(ticks_c); ax.set_xticklabels(chrs_list, rotation=90, fontsize=6.0)   # 6 pt: chr 19-22 ticks are 1.9 mm apart
ax.set_xlabel("Genes by Chromosome", labelpad=1.5); ax.set_ylabel("loadings/score")
ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([-200, -150, -100, -50, 0]))
# label geometry in mm -> data units
lab_w = {n_: text_size_mm(fig, n_, LABFS)[0] for n_, *_ in labs}
lab_h = text_size_mm(fig, "OR2L3", LABFS)[1]
x0, x1 = ax.get_xlim()
top_c = ymax_c + (lab_h + 1.0) * (ymax_c - ymin_c) / (ah - lab_h - 1.0)       # one label row above max x 1.1
ax.set_ylim(ymin_c, top_c)
ux = (x1 - x0) / aw; uy = (top_c - ymin_c) / ah                                 # data units per mm
# black labels: one row just above the highest points, spread along x
blk = [l_ for l_ in labs if l_[3] == "black"]
cx_ = spread([l_[1] for l_ in blk], [lab_w[l_[0]] * ux for l_ in blk], 0.8 * ux, x0, x1)
yb = ymax_c + 0.35 * uy
for (n_, xp, yp, c_), xl in zip(blk, cx_):
    ax.text(xl, yb, n_, ha="center", va="bottom", fontsize=LABFS, color=c_)
    ax.plot([xp, xl], [yp + 0.25 * uy, yb], color="k", lw=0.3, zorder=1)
# grey labels: two columns either side of the chr1 OR cluster, spread along y below the bulk band
gry = sorted([l_ for l_ in labs if l_[3] == "gray"], key=lambda t: t[2])
xc = np.mean([l_[1] for l_ in gry])
allv = np.concatenate([b_[1] for b_ in blocks if len(b_[1])])
band_lo = allv[allv > max(l_[2] for l_ in gry)].min()          # lowest unlabelled gene = lower edge of the bulk band
LOG.append(f"c: lowest unlabelled loading {band_lo:.1f}; grey labels kept below {band_lo - 0.6 * lab_h * uy:.1f}")
cols = {"right": gry[0::2], "left": gry[1::2]}
for side, L_ in cols.items():
    ys_ = spread([l_[2] for l_ in L_], [lab_h * uy] * len(L_), 0.6 * uy, ymin_c + 0.2 * uy, band_lo - 0.6 * lab_h * uy)
    for (n_, xp, yp, c_), yl in zip(L_, ys_):
        dx = 1.6 * ux
        xl = xp + dx if side == "right" else xp - dx
        ax.text(xl, yl, n_, ha="left" if side == "right" else "right", va="center", fontsize=LABFS, color=c_)
        ax.plot([xp, xl - (0.25 * ux if side == "right" else -0.25 * ux)], [yp, yl], color="k", lw=0.3, zorder=1)
LOG.append(f"c: y limits {ymin_c:.1f} to {top_c:.1f} (ros3b {ymin_c:.1f} to {ymax_c:.1f}); min tick spacing "
           f"{np.diff(ticks_c).min() / ux:.2f} mm")
savep(fig, "S2c_C21_RNA_final")

# ================================================================== d: GWAS Manhattan (ros5 + assocplots.manhattan)
GW = pd.read_csv(f"{ROSD}/oth/snps/gemma/output/results21.abbr.txt", sep=" ", header=None,
                 names=["chr", "pos", "rs", "p"], dtype={"chr": str})
GCOL = ['#1b9e77', "#d95f02", '#7570b3', '#e7298a']
chrs_g = [str(i) for i in range(1, 23)]
chrs_names = np.array(chrs_g); chrs_names[1::2] = ''
fig, w, h = fig_mm("d")
db = 7.0; aw = w - AX_L - 1.2; ah = h - TOP - db
ax = ax_mm(fig, w, h, AX_L, db, aw, ah)
shift_g = np.array([0.0]); nplot = 0; top = None
for ii, i in enumerate(chrs_g):
    f_ = GW[GW.chr == i]
    x_ = shift_g[-1] + f_.pos.values.astype(float); y_ = -np.log10(f_.p.values)
    keep = y_ > 0                                                       # cut = 0
    ax.plot(x_[keep], y_[keep], ".", color=GCOL[ii % 4], markersize=2.6, markeredgewidth=0, rasterized=True,
            zorder=2)
    nplot += int(keep.sum())
    if y_.max() >= 8:                                                   # ros5: label the top SNP of the chromosome
        j_ = int(np.argmax(y_)); top = (x_[j_], y_[j_], f_.rs.values[j_])
    shift_g = np.append(shift_g, np.max(x_))
for s_ in shift_g[1:-1]:
    ax.axvline(s_, color="lightgrey", lw=0.4, zorder=1)
ax.axhline(10, color="#008000", ls="-", lw=0.7, zorder=3)          # matplotlib 'g' (seaborn remaps the code)
ax.axhline(15, color="#FF0000", ls="--", lw=0.7, zorder=3, dashes=(3.5, 2.0))   # matplotlib 'r'
ax.set_xlim(0, shift_g[-1]); ax.set_ylim(0, 18)
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
ax.set_xticks((shift_g[1:] + shift_g[:-1]) / 2.); ax.set_xticklabels(list(chrs_names))
ax.set_xlabel("chromosome", labelpad=1.5); ax.set_ylabel("-log10(p-value)")
ux = shift_g[-1] / aw; uy = 18 / ah
ax.text(top[0] + 0.7 * ux, top[1] + 0.15 * uy, top[2], ha="left", va="bottom", fontsize=LABFS, color="black", zorder=4)
# PNG check: boundaries (grey lines) of the published raster at x = 91.5 + 992 * shift / total
png = plt.imread(f"{CODING}/GWAS/21_Manhattan_Component.png")[100, :, :3]
grey_cols = np.where(np.all(np.abs(png * 255 - 211) < 2, axis=1))[0]
pred = 91.5 + 992 * shift_g[1:-1] / shift_g[-1]          # frame lines at px 91-92 and 1083-1084
dev = max(np.abs(grey_cols - p_).min() for p_ in pred)
LOG.append(f"d: {nplot} SNPs on chr 1-22 (GEMMA analysed {len(GW)}; n 228); top {top[2]} -log10 p {top[1]:.3f} "
           f"(p {10 ** -top[1]:.2g}); SNPs >= 8: {int((-np.log10(GW.p.values) >= 8).sum())}; "
           f"boundary match to PNG grey lines max {dev:.2f} px")
assert top[2] == "rs6664332" and abs(top[1] - 15.381) < 0.01 and dev <= 1.0
savep(fig, "S2d_C21_Manhattan_final")

# ================================================================== e: OR2L3 eQTL by rs6664332 genotype (ros8 RNA block)
SNP = f"{ROSD}/oth/snps/ROSMAP_arrayGenotype_maf"
bim = pd.read_csv(f"{SNP}.bim", sep="\t", header=None)
fam_iid = pd.Series([l.split()[1] for l in open(f"{SNP}.fam")])
snp_index = int(np.where(bim[1].values == "rs6664332")[0][0])
nS = len(fam_iid); nb = (nS + 3) // 4
with open(f"{SNP}.bed", "rb") as fh:
    magic = fh.read(3); assert magic == b"\x6c\x1b\x01"
    fh.seek(3 + snp_index * nb); raw = np.frombuffer(fh.read(nb), np.uint8)
codes = np.array([(raw[i // 4] >> (2 * (i % 4))) & 3 for i in range(nS)])
# PLINK: 00 hom A1 (bim col 5), 10 het, 11 hom A2 (bim col 6), 01 missing; pysnptools count_A1=False counts A2
gval = np.select([codes == 0, codes == 2, codes == 3], [0.0, 1.0, 2.0], np.nan)
snp_samples_clean = fam_iid.str.replace(r"^(ROS|MAP)", "", regex=True).astype(int)
a1 = bim.iloc[snp_index, 5]; a2 = bim.iloc[snp_index, 4]
genotype_order = [f"{a1}{a1}", f"{a1}{a2}", f"{a2}{a2}"]
PB = f"{ROSD}/mit/rna/pseudobulk"
E_ = {}
for ct in sorted(["Ast", "Exc", "Inh", "Mic", "OPC", "Oli"]):
    hdr = open(f"{PB}/{ct}_rpm_normalized.csv").readline().rstrip("\n").split(",")
    d_ = pd.read_csv(f"{PB}/{ct}_rpm_normalized.csv", index_col=0, usecols=[0, hdr.index("OR2L3")])
    d_ = np.log2(d_ + 1)
    shared = snp_samples_clean[snp_samples_clean.isin(d_.index)]
    ex = d_.loc[shared, "OR2L3"]
    gt = pd.Series(gval, index=snp_samples_clean).loc[shared]
    gt = gt.map({0: f"{a2}{a2}", 1: f"{a1}{a2}", 2: f"{a1}{a1}"})
    df = pd.DataFrame({"expression": ex.values, "genotype": gt.values}).dropna()
    model = ols("expression ~ C(genotype)", data=df).fit()
    pval = sm.stats.anova_lm(model, typ=2)["PR(>F)"].iloc[0]
    E_[ct] = (df, pval)
PUB_P = {"Ast": "8.17e-04", "Exc": "1.12e-17", "Inh": "1.44e-17", "Mic": "7.57e-01", "OPC": "4.97e-14", "Oli": "2.77e-13"}
for ct, (df, pval) in E_.items():
    assert f"{pval:.2e}" == PUB_P[ct], (ct, pval)
cnt = E_["Exc"][0].genotype.value_counts()
assert [int(cnt.get(g_, 0)) for g_ in genotype_order] == [92, 115, 21]
LOG.append(f"e: rs6664332 bim alleles {a2}/{a1}; genotype n " + ", ".join(f"{g_} {int(cnt.get(g_, 0))}" for g_ in genotype_order)
           + "; p (= published titles): " + ", ".join(f"{c_} {p_:.2e}" for c_, (_, p_) in E_.items()))
# old-seaborn box line grey: 0.6 x the lowest lightness of the desaturated palette (seaborn < 0.13 rule)
pal = sns.color_palette("Set2", 3, desat=0.75)
lum = min(rgb_to_hls(*c_)[1] for c_ in pal) * 0.6
GREY = mpl.colors.rgb2hex((lum, lum, lum))
fig, w, h = fig_mm("e")
eL = 7.2; sw = 14.8; gx = 2.4
tb = 5.9; bb = 7.0; gy = 1.0                         # title band, tick+xlabel band, inter-row gap (mm)
eh = (h - 2 * tb - 2 * bb - gy) / 2

axes = []
for i, ct in enumerate(sorted(E_)):
    r_, c_ = divmod(i, 3)
    bottom = h - tb - eh - r_ * (tb + eh + bb + gy)
    ax = fig.add_axes([(eL + c_ * (sw + gx)) / w, bottom / h, sw / w, eh / h], sharey=axes[0] if axes else None)
    df, pval = E_[ct]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.boxplot(x="genotype", y="expression", data=df, ax=ax, showfliers=False, hue="genotype", palette="Set2",
                    order=genotype_order, hue_order=genotype_order, legend=False, saturation=0.75, linewidth=0.6,
                    linecolor=GREY, width=0.8)
    np.random.seed(21 + i)
    sns.stripplot(x="genotype", y="expression", data=df, ax=ax, color="gray", alpha=0.5, jitter=True,
                  order=genotype_order, size=1.3, linewidth=0)
    ax.set_title(f"{ct}\n(p = {pval:.2e})", linespacing=1.1)
    ax.set_xlabel("Genotype", labelpad=1.0)
    ax.set_ylabel("Expression" if c_ == 0 else "")
    if c_ > 0:
        ax.tick_params(labelleft=False)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    axes.append(ax)
savep(fig, "S2e_C21_OR2L3_rs6664332_eQTL_final")

# ================================================================== f, g: p06_c21_single.py S2a / S2b at final size
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
assert not outliers_30iqr(a21).any()               # single x axis (no broken axis), as the S2a/S2b files
yf = gene(21, "OR2L3", "Exc"); yg, n_agg = top50_agg(21, "Exc")
ext = most_extreme(a21)
for nm, yy_, lab in (("S2f_C21_OR2L3_Exc_vs_score_final", yf, "OR2L3 in Exc (CPM)"),
                     ("S2g_C21_aggregate_Exc_vs_score_final", yg, "top-50 gene aggregate in Exc\nweighted mean z")):
    fig, w, h = fig_mm("f")
    fL = 10.4; fb = 6.6
    ax = ax_mm(fig, w, h, fL, fb, w - fL - 3.6, h - TOP - fb)
    ok = ~np.isnan(yy_)
    ax.scatter(a21[ok], yy_[ok], color=CTCOL["Exc"], s=3.0, alpha=0.8, linewidths=0, zorder=2)
    if not np.isnan(yy_[ext]):
        ax.scatter([a21[ext]], [yy_[ext]], s=3.0 * 3.2, facecolors="none", edgecolors=EXTRING, linewidths=0.5, zorder=3)
    sns.despine(ax=ax)
    ax.set_xlabel("Component 21 loadings/score"); ax.set_ylabel(lab)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10 if "OR2L3" in nm else 1))
    savep(fig, nm)
    r_, p_, n_ = spearman(yy_, a21)
    m_ = ok.copy(); m_[ext] = False
    LOG.append(f"{nm.split('_')[0]}: n {n_}; Spearman {r_:+.3f}; without the most extreme donor "
               f"{ss.spearmanr(yy_[m_], a21[m_])[0]:+.3f}" + (f"; aggregate genes {n_agg}" if "aggregate" in nm else ""))

open(f"{PAN}/s2_panels.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
