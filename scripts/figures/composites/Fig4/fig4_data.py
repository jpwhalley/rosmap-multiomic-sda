"""Fig 4 (Component 172) panel data for the Python main composite (v23). Read only everywhere.

  b, c  v22 recipe of ros3b_plot_outliers.py (the version-1 panel script, which wrote the version-1
        172_{cell_type,RNA}.pdf), from the same inputs: sda/genes_final_RNA.csv,
        sda/pip_final_RNA.csv, Homo_sapiens.GRCh38.100.txt (references folder) and, for the cell-type bars,
        ROSMAP_cell_type_panels_values.csv cor_* (C172: no change from the published values) checked
        against cell_type_loadings_corrected.csv (catalogue tables).
        Adapted from composites/S11/s11_data.py (same recipe) with K = 172 and the RNA layer only.
  e     archived Enrichr table 172_pathway_encrichment.csv (G_ROS_ENRICHR_DIR), top 10 by adjusted p,
        and the per-bar colours read from the original PDF stream (replot/kegg_panels/orig_spec.json), exactly
        as replot/kegg_panels/03_replot.py.
Nothing here prints donor IDs or per-donor values."""
import json, os, sys
import numpy as np, pandas as pd

FIGCODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/figures
sys.path.insert(0, FIGCODE)
from figure_paths import SDA, REFERENCES as REF, TABLES as NR, ENRICHR as ENRICHR_DIR
K = 172
COLORS_RNA = ['#6495ED', '#778899', '#008080', '#20B2AA']                                     # ros3b l.149, j = 1
CHR_MAX = {'1': 249230899.0, '2': 243064438.0, '3': 197955065.0, '4': 191012160.0, '5': 180794288.0, '6': 170886450.0,
           '7': 158820866.0, '8': 146277826.0, '9': 141069493.0, '10': 135497182.0, '11': 134248398.0,
           '12': 133758108.0, '13': 115079831.0, '14': 107282052.0, '15': 102501016.0, '16': 90244125.0,
           '17': 81037567.0, '18': 77915115.0, '19': 59086812.0, '20': 62958768.0, '21': 48110787.0,
           '22': 51205920.0, 'X': 155249967.0, 'Y': 59330252.0}                                # ros3b l.154
CHRS = [str(i) for i in range(1, 23)] + ['X', 'Y']
XLAB_RNA = "Genes by Chromosome"                                                               # ros3b l.422
CT_ORDER = ['Ast', 'Exc', 'Inh', 'Mic', 'OPC', 'Oli']                                           # ros3b l.332
CT_COL = {'Ast': '#D85396', 'Oli': '#8BC34A', 'Inh': '#303F9F', 'OPC': '#E5A43A', 'Exc': '#D32F2F',
          'Mic': '#4DB6E7'}                                                                     # ros3b l.213-220
ENRICHR = f"{ENRICHR_DIR}/172_pathway_encrichment.csv"
KEGG_SPEC = f"{FIGCODE}/replot/kegg_panels/orig_spec.json"


def row(path, k=K):
    """One component row of a component x feature CSV (header + the line whose index is k)."""
    with open(path) as f:
        head = f.readline().rstrip("\n").split(",")[1:]
        key = f"{k},"
        for line in f:
            if line.startswith(key):
                vals = line.rstrip("\n").split(",")
                assert vals[0] == str(k)
                return pd.Series(np.asarray(vals[1:], float), index=head)
    raise KeyError(k)


def pip_features():
    g = row(f"{SDA}/genes_final_RNA.csv"); p = row(f"{SDA}/pip_final_RNA.csv")
    assert list(g.index) == list(p.index)
    return g[p > 0.5]                                   # = pip_genes.loc[k] in ros3b l.322


def annotation(feats):
    """ros3b l.68-78: Ensembl table indexed by gene name (duplicated gene names keep every row)."""
    e = pd.read_csv(f"{REF}/Homo_sapiens.GRCh38.100.txt", index_col=0, sep=",")
    e["new_index"] = e["gene_name"]; e = e.set_index("new_index")
    info = e[e.index.isin(feats)]
    return pd.DataFrame({"chrom": info["chrom"].astype(str).values, "pos": info["chromStart"].values,
                         "fid": info["gene_name"].values, "lab": info["gene_name"].values})


def manhattan():
    """Points (x, y, colour), labels (x, y, text, colour), chromosome tick positions and y limits, as ros3b."""
    L = pip_features()
    info = annotation(L.index)
    if len(L) > 16:                                     # ros3b l.353-358
        large = L.nlargest(8); small = L.nsmallest(8)
    else:
        large = L[L > 0]; small = L[L <= 0]
    shift = np.array([0.0]); pts = []; labels = []; spacers = []
    for ii, ch in enumerate(CHRS):                       # ros3b l.363-384
        sub = info[info["chrom"] == ch]
        col = COLORS_RNA[ii % 4]
        if len(sub) == 0:
            x = shift[-1] + np.array([CHR_MAX[ch]])      # invisible spacer (markersize 0)
            spacers.append(float(x[0]))
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
    ticks = (shift[1:] + shift[:-1]) / 2.0               # ros3b l.388
    yv = L.values                                        # ros3b l.396-420
    mn, mx = yv.min(), yv.max()
    if len(yv) == 1:
        mg = abs(yv[0] * 0.1) if yv[0] != 0 else 0.1; ylim = (yv[0] - mg, yv[0] + mg)
    else:
        mmin = abs(mn * 0.1) if mn != 0 else 0.1; mmax = abs(mx * 0.1) if mx != 0 else 0.1
        if (mx > 0 and mn >= 0) or (mx <= 0 and mn < 0):
            ylim = (mn - mmin, mx + mmax)
        else:
            ylim = (mn * 1.1, mx * 1.1)
    return dict(n_pip=int(len(L)), n_annot_rows=int(len(info)), pts=pts, labels=labels, spacer_x=spacers,
                ticks=list(map(float, ticks)), xmax=float(shift[-1]), ylim=tuple(map(float, ylim)),
                dup_names=int(info["fid"].duplicated().sum()), unplaced=sorted(set(L.index) - set(info["fid"])))


def celltype():
    v = pd.read_csv(f"{NR}/ROSMAP_cell_type_panels_values.csv")
    r = v[v.comp == K].iloc[0]
    vals = [float(r[f"cor_{c}"]) for c in CT_ORDER]
    pub = [float(r[f"pub_{c}"]) for c in CT_ORDER]
    b = pd.read_csv(f"{NR}/cell_type_loadings_corrected.csv", index_col=0).loc[K]
    assert np.allclose(vals, [float(b[c]) for c in CT_ORDER]) and np.allclose(vals, pub)
    return CT_ORDER, vals


def kegg():
    s = json.load(open(KEGG_SPEC))[str(K)]
    df = pd.read_csv(ENRICHR)
    top = df.sort_values('Adjusted P-value', kind='mergesort').head(10).reset_index(drop=True)
    terms, vals = list(top['Term']), top['-log10(FDR)'].to_numpy()
    assert terms == s['terms'] and np.allclose(vals, s['values'])
    assert np.allclose(vals, -np.log10(top['Adjusted P-value'].to_numpy()))
    return terms, vals, [tuple(c) for c in s['colors']], s['xlabel'], top['Adjusted P-value'].to_numpy()


if __name__ == "__main__":
    d = manhattan()
    print("RNA PIP>0.5:", d["n_pip"], "annotated rows:", d["n_annot_rows"], "points:", len(d["pts"]),
          "dup names:", d["dup_names"], "not placed:", len(d["unplaced"]), "labels:", len(d["labels"]),
          "ylim:", tuple(round(v, 4) for v in d["ylim"]))
    print("  labels:", [(l[2], l[3]) for l in d["labels"]])
    print("cell types:", [(c, round(v, 4)) for c, v in zip(*celltype())])
    t, v, c, xl, q = kegg()
    print("KEGG:", [(a, round(b, 3)) for a, b in zip(t, v)], xl, "max FDR", q.max())
