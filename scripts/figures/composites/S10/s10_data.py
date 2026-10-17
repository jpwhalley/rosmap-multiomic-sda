"""S10 (Component 87) panel data for a-e, rebuilt with the v22 recipe of ros3b_plot_outliers.py
(the version-1 panel script, which wrote the version-1 87_{samples,RNA,meth,chip}.pdf), from the same inputs:
  sda/genes_final_{RNA,meth,chip}.csv, sda/pip_final_*.csv, sda/samples_final.csv, ROSMAP_clinical.csv,
  Homo_sapiens.GRCh38.100.txt, ROSMAP_arrayMethylation_metaData.tsv, H3K9acDomains.csv (references folder),
and, for the corrected cell-type bars (b), the cor_* columns of ROSMAP_cell_type_panels_values.csv (catalogue tables),
the values drawn into the corrected 87_cell_type.pdf.

Read only everywhere. Nothing here prints donor IDs or per-donor values."""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # scripts/figures
from figure_paths import SDA, REFERENCES as REF, TABLES as NR, CLINICAL
K = 87
POS2TREAT = {1: "RNA", 2: "meth", 3: "chip"}
# ros3b l.149 (chromosome colours, cycled by chromosome index)
COLORS = {2: ['#ffffb2', '#fecc5c', '#fd8d3c', '#e31a1c'], 3: ['#f1eef6', '#d7b5d8', '#df65b0', '#ce1256'],
          1: ['#6495ED', '#778899', '#008080', '#20B2AA']}
# ros3b l.154
CHR_MAX = {'1': 249230899.0, '2': 243064438.0, '3': 197955065.0, '4': 191012160.0, '5': 180794288.0, '6': 170886450.0,
           '7': 158820866.0, '8': 146277826.0, '9': 141069493.0, '10': 135497182.0, '11': 134248398.0,
           '12': 133758108.0, '13': 115079831.0, '14': 107282052.0, '15': 102501016.0, '16': 90244125.0,
           '17': 81037567.0, '18': 77915115.0, '19': 59086812.0, '20': 62958768.0, '21': 48110787.0,
           '22': 51205920.0, 'X': 155249967.0, 'Y': 59330252.0}
CHRS = [str(i) for i in range(1, 23)] + ['X', 'Y']
XLAB = {1: "Genes by Chromosome", 2: "CpG Sites by Chromosome", 3: "ChIP Peaks by Chromosome"}   # ros3b l.422/512/602
CT_ORDER = ['Ast', 'Exc', 'Inh', 'Mic', 'OPC', 'Oli']                                           # ros3b l.332
CT_COL = {'Ast': '#D85396', 'Oli': '#8BC34A', 'Inh': '#303F9F', 'OPC': '#E5A43A', 'Exc': '#D32F2F',
          'Mic': '#4DB6E7'}                                                                     # ros3b l.213-220


def row(path, k=K):
    """One component row of a component x feature CSV (header + the line whose index is k)."""
    with open(path) as f:
        head = f.readline().rstrip("\n").split(",")[1:]
        key = f"{k},"
        for line in f:
            if line.startswith(key):
                vals = line.rstrip("\n").split(",")
                return pd.Series(np.asarray(vals[1:], float), index=head)
    raise KeyError(k)


def pip_features(j):
    g = row(f"{SDA}/genes_final_{POS2TREAT[j]}.csv"); p = row(f"{SDA}/pip_final_{POS2TREAT[j]}.csv")
    assert list(g.index) == list(p.index)
    return g[p > 0.5]                                   # = pip_genes.loc[k] in ros3b l.322/432/522


def annotation(j, feats):
    """Rows of the ros3b annotation table for the PIP > 0.5 features: chrom, pos, feature id, label text."""
    if j == 1:                                          # ros3b l.68-78 (duplicated gene names keep every row)
        e = pd.read_csv(f"{REF}/Homo_sapiens.GRCh38.100.txt", index_col=0, sep=",")
        e["new_index"] = e["gene_name"]; e = e.set_index("new_index")
        info = e[e.index.isin(feats)]
        return pd.DataFrame({"chrom": info["chrom"].astype(str).values, "pos": info["chromStart"].values,
                             "fid": info["gene_name"].values, "lab": info["gene_name"].values})
    if j == 2:                                          # ros3b l.130-137
        m = pd.read_csv(f"{REF}/ROSMAP_arrayMethylation_metaData.tsv", index_col=0, sep="\t")
        m["CHR"] = m["CHR"].astype(str)
        m = m[m.index.isin(feats)]
        lab = [f"{i}:{r.split(';')[0]}" if pd.notna(r) else i for i, r in zip(m.index, m["RefGene"])]
        return pd.DataFrame({"chrom": m["CHR"].values, "pos": m["MAPINFO"].values, "fid": m.index.values, "lab": lab})
    c = pd.read_csv(f"{REF}/H3K9acDomains.csv", index_col=0, sep="\t")      # ros3b l.141-144
    c["Chr"] = c["Chr"].astype(str)
    c = c[c.index.isin(feats)]
    return pd.DataFrame({"chrom": c["Chr"].values, "pos": c["Start"].values, "fid": c.index.values,
                         "lab": [f"{i}:{cl}" for i, cl in zip(c.index, c["Classification"])]})


def manhattan(j):
    """Points (x, y, colour), labels (x, y, text, colour), chromosome tick positions and y limits, as ros3b."""
    L = pip_features(j)
    info = annotation(j, L.index)
    if len(L) > 16:                                     # ros3b l.353-358
        large = L.nlargest(8); small = L.nsmallest(8)
    else:
        large = L[L > 0]; small = L[L <= 0]
    shift = np.array([0.0]); pts = []; labels = []; spacers = []
    for ii, ch in enumerate(CHRS):                       # ros3b l.363-384
        sub = info[info["chrom"] == ch]
        col = COLORS[j][ii % 4]
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
    b = pd.read_csv(f"{NR}/cell_type_loadings_corrected.csv", index_col=0).loc[K]
    assert np.allclose(vals, [float(b[c]) for c in CT_ORDER])
    return CT_ORDER, vals


def samples():
    """ros3b l.158-176: samples merged with clinical (msex, age_death), sorted by age at death ('90+' as 90)."""
    s = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0); s.columns = s.columns.astype(str)
    clin = pd.read_csv(CLINICAL, index_col=0); clin.index = clin.index.astype(str)
    meta = clin.loc[s.columns.values][['Study', 'msex', 'age_death']].reindex(s.columns)
    st = s.T; st.index = st.index.astype(str)
    st = st.merge(meta[['msex', 'age_death']], left_index=True, right_index=True)
    st['Sex'] = st['msex'].apply(lambda x: 'Male' if x == 1.0 else 'Female')
    st['age_death_numeric'] = pd.to_numeric(st['age_death'], errors='coerce').fillna(90)
    st = st.sort_values('age_death_numeric')
    st['x_pos'] = range(len(st))
    k = K if K in st.columns else str(K)
    y = st[k].values.astype(float)
    q1, q3 = np.quantile(y, [0.25, 0.75]); iqr = q3 - q1                       # ros3b l.226-236 (pandas quantile)
    lo, hi = q1 - 30 * iqr, q3 + 30 * iqr
    out = (y < lo) | (y > hi)
    return st['x_pos'].values, y, st['Sex'].values, out


if __name__ == "__main__":
    for j in (1, 2, 3):
        d = manhattan(j)
        print(POS2TREAT[j], "PIP>0.5:", d["n_pip"], "annotated rows:", d["n_annot_rows"], "points:", len(d["pts"]),
              "dup names:", d["dup_names"], "not placed:", d["unplaced"][:10], len(d["unplaced"]),
              "labels:", len(d["labels"]), "ylim:", tuple(round(v, 4) for v in d["ylim"]))
        print("  labels:", [(l[2], l[3], round(l[1], 4)) for l in d["labels"]])
    print("cell types:", celltype())
    x, y, sex, out = samples()
    print("samples n:", len(y), "female n:", int((sex == 'Female').sum()), "beyond 30xIQR:", int(out.sum()),
          "sex of those:", list(sex[out]))
