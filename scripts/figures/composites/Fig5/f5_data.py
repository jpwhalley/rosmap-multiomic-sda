"""Figure 5 (Component 277) panel data, read only everywhere. Nothing here prints donor IDs or per-donor values.

  a  C277 score by final clinical diagnosis (NCI / MCI / AD dementia): the port of the original panel code
     (qlabel_panels/qlabel_panels.py, loaded read-only as a module: data loading, +/-5 SD
     outlier filter, seaborn-0.12 box shim, husl palette, recovered stripplot jitter) and the 547-family BH q
     (clinical_BH_547_family.csv), i.e. the inputs of the corrected 277_cogdx_samples.pdf and of the
     relabelled replot (replot/sample_panels/replot_v23labels.py).
  b  cell-type loadings: ROSMAP_cell_type_panels_values.csv cor_* (row 'Fig S6f', comp 277; equal to
     the published values) = cell_type_loadings_corrected.csv, drawn into the version-1 277_cell_type.pdf.
  c  gene loadings by chromosome: the ros3b_plot_outliers.py recipe (as composites/S11/s11_data.py,
     K = 277, RNA only) from sda/genes_final_RNA.csv, sda/pip_final_RNA.csv and the GRCh38.100 annotation; drawn
     into the version-1 277_RNA.pdf.
  d  PLCB1 (CPM, SDA input) in Exc against the C277 score: revision_analyses/neuronal_c21/cache/n05_views.npy
     (G_ROS_RESULTS_DIR) as panels/p05_neuronal.py scatter_view (Fig4j_C277_PLCB1_Exc_vs_score.pdf); n 274.
  e  KEGG: replot/kegg_panels/orig_spec.json (terms, order, -log10(FDR), per-bar RGB read from the original
     enrichr/277_pathway_encrichment.pdf and verified there against the archived Enrichr table),
     re-checked here against that table (G_ROS_ENRICHR_DIR)."""
import os, sys, json, importlib.util
import numpy as np, pandas as pd

FIGCODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/figures
sys.path.insert(0, FIGCODE)
from figure_paths import RESULTS as R2, SDA, REFERENCES as REF, TABLES as NR, ENRICHR
K = 277

# ---------------------------------------------------------------- c: ros3b recipe (copied from s11_data.py)
COLORS = {1: ['#6495ED', '#778899', '#008080', '#20B2AA']}                     # ros3b l.149 (RNA)
CHR_MAX = {'1': 249230899.0, '2': 243064438.0, '3': 197955065.0, '4': 191012160.0, '5': 180794288.0, '6': 170886450.0,
           '7': 158820866.0, '8': 146277826.0, '9': 141069493.0, '10': 135497182.0, '11': 134248398.0,
           '12': 133758108.0, '13': 115079831.0, '14': 107282052.0, '15': 102501016.0, '16': 90244125.0,
           '17': 81037567.0, '18': 77915115.0, '19': 59086812.0, '20': 62958768.0, '21': 48110787.0,
           '22': 51205920.0, 'X': 155249967.0, 'Y': 59330252.0}
CHRS = [str(i) for i in range(1, 23)] + ['X', 'Y']
XLAB_RNA = "Genes by Chromosome"                                                  # ros3b l.422
CT_ORDER = ['Ast', 'Exc', 'Inh', 'Mic', 'OPC', 'Oli']                             # ros3b l.332
CT_COL = {'Ast': '#D85396', 'Oli': '#8BC34A', 'Inh': '#303F9F', 'OPC': '#E5A43A', 'Exc': '#D32F2F',
          'Mic': '#4DB6E7'}                                                       # ros3b l.213-220


def row(path, k=K):
    with open(path) as f:
        head = f.readline().rstrip("\n").split(",")[1:]
        key = f"{k},"
        for line in f:
            if line.startswith(key):
                vals = line.rstrip("\n").split(",")
                return pd.Series(np.asarray(vals[1:], float), index=head)
    raise KeyError(k)


def manhattan():
    g = row(f"{SDA}/genes_final_RNA.csv"); p = row(f"{SDA}/pip_final_RNA.csv")
    assert list(g.index) == list(p.index)
    L = g[p > 0.5]
    e = pd.read_csv(f"{REF}/Homo_sapiens.GRCh38.100.txt", index_col=0, sep=",")
    e["new_index"] = e["gene_name"]; e = e.set_index("new_index")
    info = e[e.index.isin(L.index)]
    info = pd.DataFrame({"chrom": info["chrom"].astype(str).values, "pos": info["chromStart"].values,
                         "fid": info["gene_name"].values, "lab": info["gene_name"].values})
    if len(L) > 16:
        large = L.nlargest(8); small = L.nsmallest(8)
    else:
        large = L[L > 0]; small = L[L <= 0]
    shift = np.array([0.0]); pts = []; labels = []; spacers = []
    for ii, ch in enumerate(CHRS):
        sub = info[info["chrom"] == ch]
        col = COLORS[1][ii % 4]
        if len(sub) == 0:
            x = shift[-1] + np.array([CHR_MAX[ch]]); spacers.append(float(x[0]))
        else:
            x = shift[-1] + sub["pos"].values.astype(float)
            y = L.loc[sub["fid"].values].values
            for xi, yi, fid, lab in zip(x, y, sub["fid"].values, sub["lab"].values):
                pts.append((float(xi), float(yi), col))
                if fid in large.index:
                    labels.append((float(xi), float(yi), lab, "black"))
                elif fid in small.index:
                    labels.append((float(xi), float(yi), lab, "gray"))
        shift = np.append(shift, np.max(x))
    ticks = (shift[1:] + shift[:-1]) / 2.0
    yv = L.values; mn, mx = yv.min(), yv.max()
    mmin = abs(mn * 0.1) if mn != 0 else 0.1; mmax = abs(mx * 0.1) if mx != 0 else 0.1
    if (mx > 0 and mn >= 0) or (mx <= 0 and mn < 0):
        ylim = (mn - mmin, mx + mmax)
    else:
        ylim = (mn * 1.1, mx * 1.1)
    return dict(n_pip=int(len(L)), n_annot_rows=int(len(info)), pts=pts, labels=labels, spacer_x=spacers,
                ticks=list(map(float, ticks)), xmax=float(shift[-1]), ylim=tuple(map(float, ylim)),
                dup_names=int(info["fid"].duplicated().sum()), unplaced=sorted(set(L.index) - set(info["fid"])),
                large=list(large.index), small=list(small.index))


# ---------------------------------------------------------------- b
def celltype():
    v = pd.read_csv(f"{NR}/ROSMAP_cell_type_panels_values.csv")
    r = v[v.comp == K].iloc[0]
    vals = [float(r[f"cor_{c}"]) for c in CT_ORDER]
    pub = [float(r[f"pub_{c}"]) for c in CT_ORDER]
    b = pd.read_csv(f"{NR}/cell_type_loadings_corrected.csv", index_col=0).loc[K]
    assert np.allclose(vals, [float(b[c]) for c in CT_ORDER]) and np.allclose(vals, pub)
    return CT_ORDER, vals


# ---------------------------------------------------------------- a
QDIR = f"{FIGCODE}/qlabel_panels"
_qp = None


def qp():
    global _qp
    if _qp is None:
        spec = importlib.util.spec_from_file_location("qlabel_panels", os.path.join(QDIR, "qlabel_panels.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        _qp = m
    return _qp


def cogdx_samples():
    """As qlabel_panels.panel_cogdx: outlier filter, groups 1/2/4, recovered jitter for 277_cogdx_samples.pdf."""
    q = qp()
    samples = q.load_samples()
    q547 = pd.read_csv(q.BH547, index_col=0)
    qv = float(q547.loc[K, "cogdx"])
    jit = json.load(open(os.path.join(QDIR, "recovered_jitter.json")))["277_cogdx_samples.pdf"]
    fs = q.outlier_filter(samples, K)
    fs = fs[fs["cogdx"].isin([1, 2, 4])].copy()
    return fs, qv, jit


# ---------------------------------------------------------------- d
def plcb1_view():
    V = np.load(f"{R2}/revision_analyses/neuronal_c21/cache/n05_views.npy", allow_pickle=True).item()
    c = V[K]
    a = np.asarray(c["a"], float); y = np.asarray(c["views"]["PLCB1"], float)
    return a, y, c["ct"], int(c["ext"])


# ---------------------------------------------------------------- e
SNAP_CSV = f"{ENRICHR}/277_pathway_encrichment.csv"


def kegg():
    s = json.load(open(f"{FIGCODE}/replot/kegg_panels/orig_spec.json"))[str(K)]
    df = pd.read_csv(SNAP_CSV)
    top = df.sort_values('Adjusted P-value', kind='mergesort').head(10).reset_index(drop=True)
    assert list(top['Term']) == s['terms'] and np.allclose(top['-log10(FDR)'].to_numpy(), s['values'])
    assert np.allclose(top['-log10(FDR)'], -np.log10(top['Adjusted P-value']))
    assert (top['Adjusted P-value'] < 0.05).all()
    return s['terms'], np.asarray(s['values'], float), [tuple(c) for c in s['colors']], s['xlabel']


if __name__ == "__main__":
    d = manhattan()
    print("RNA PIP>0.5:", d["n_pip"], "annotated rows:", d["n_annot_rows"], "points:", len(d["pts"]),
          "dup names:", d["dup_names"], "not placed:", len(d["unplaced"]), "labels:", len(d["labels"]),
          "ylim:", tuple(round(v, 3) for v in d["ylim"]))
    print("  labels:", [(l[2], l[3]) for l in d["labels"]])
    print("  large:", d["large"], "small:", d["small"])
    t = np.array(d["ticks"]); print("  tick spacing (Mb):", np.round(np.diff(t) / 1e6, 1).tolist(), "xmax", d["xmax"] / 1e6)
    print("cell types:", [(c, round(v, 4)) for c, v in zip(*celltype())])
    fs, qv, jit = cogdx_samples()
    print("a: q", qv, "groups", fs["cogdx"].value_counts().reindex([1, 2, 4]).tolist(), "jitter groups", [len(j) for j in jit])
    a, y, ct, ext = plcb1_view()
    from scipy import stats as ss
    ok = ~np.isnan(y)
    print("d:", ct, "n", int(ok.sum()), "rho", round(ss.spearmanr(a[ok], y[ok])[0], 3))
    print("e:", kegg()[0], kegg()[1].round(3).tolist())
