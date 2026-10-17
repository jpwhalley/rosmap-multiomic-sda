"""Figure 3 (Component 278) panel data, from the same inputs and recipes as the current panels (all read only).

  a  278_sex_samples.pdf (corrected) = qlabel_panels.panel_sex (scripts/figures/qlabel_panels, the
     verified port of ros6a_cross_clin_discrete.py l.715-768): samples_final.csv x ROSMAP_clinical.csv msex, +/-5 SD
     filter, Set2 boxes (seaborn-0.12 shim), recovered stripplot jitter, BH q over the 547-component family
     (clinical_BH_547_family.csv; C278 Sex q = 0.00974 -> 'FDR = 0.0097').
  b  278_cell_type.pdf (corrected) = ros3b l.326-340:
     ROSMAP_cell_type_panels_values.csv cor_* for C278 (= cell_type_loadings_corrected.csv), order Ast Exc Inh Mic OPC Oli.
  c  version-1 278_RNA.pdf (v22) = ros3b_plot_outliers.py l.319-430 (RNA): genes_final_RNA.csv /
     pip_final_RNA.csv row 278, PIP > 0.5, GRCh38.100 annotation; same code as composites/S10/s10_data.py.
  d  version-1 278_dlpfc_correlation.pdf (v22) = synth1b_croos_clin_manuscript.py l.60-145: ROSMAP
     RNA loadings of the PIP > 0.5 genes that SEA-AD measured (gene names via gene_info feature_name) against SEA-AD
     component 67; the partner rule and r reproduced by the structure analysis (02_seaad.py).
  e  version-1 enrichr/278_pathway_encrichment.pdf (v22) = ros7ai_gene_pathways_kegg.py: top 10 rows by adjusted
     P of the archived Enrichr table (G_ROS_ENRICHR_DIR), bar colours from the original PDF stream
     (replot/kegg_panels/orig_spec.json, verified there against the PDF and the table).
  f  Fig3f_C278_FKBP5_OPC_Oli_vs_score.pdf = panels/p04_c278.py (Fig 3f block): FKBP5
     CPM of the SDA input in OPC and Oli against the deposited C278 score, snRNA-seq RNA-1/RNA-2 excluded (n 274), the
     most extreme donor (largest |robust z| of the score) ringed.

Nothing here prints donor IDs or per-donor values."""
import importlib.util, json, os, sys
import numpy as np, pandas as pd

FIGCODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/figures
sys.path.insert(0, FIGCODE)
from figure_paths import SDA, REFERENCES as REF, TABLES as NR, SEAAD as SEA, ENRICHR, FIGOUT
QDIR = f"{FIGCODE}/qlabel_panels"
K = 278

# ros3b l.149 / l.154 / l.332 / l.213-220 (as s10_data.py)
COLORS = {1: ['#6495ED', '#778899', '#008080', '#20B2AA']}
CHR_MAX = {'1': 249230899.0, '2': 243064438.0, '3': 197955065.0, '4': 191012160.0, '5': 180794288.0, '6': 170886450.0,
           '7': 158820866.0, '8': 146277826.0, '9': 141069493.0, '10': 135497182.0, '11': 134248398.0,
           '12': 133758108.0, '13': 115079831.0, '14': 107282052.0, '15': 102501016.0, '16': 90244125.0,
           '17': 81037567.0, '18': 77915115.0, '19': 59086812.0, '20': 62958768.0, '21': 48110787.0,
           '22': 51205920.0, 'X': 155249967.0, 'Y': 59330252.0}
CHRS = [str(i) for i in range(1, 23)] + ['X', 'Y']
CT_ORDER = ['Ast', 'Exc', 'Inh', 'Mic', 'OPC', 'Oli']
CT_COL = {'Ast': '#D85396', 'Oli': '#8BC34A', 'Inh': '#303F9F', 'OPC': '#E5A43A', 'Exc': '#D32F2F', 'Mic': '#4DB6E7'}


# ------------------------------------------------------------------------------------------------ a
def qlabel_module():
    spec = importlib.util.spec_from_file_location('qlabel_panels', os.path.join(QDIR, 'qlabel_panels.py'))
    qp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qp)          # sets Agg, pdf.fonttype 42, sns white/talk, appends vendor/ (statannotations)
    return qp


def samples_sex(qp):
    s = qp.load_samples()
    q = float(pd.read_csv(qp.BH547, index_col=0).loc[K, 'Sex'])
    jit = json.load(open(os.path.join(QDIR, 'recovered_jitter.json')))['278_sex_samples.pdf']
    fs = qp.outlier_filter(s, K)
    return s, fs, q, jit


# ------------------------------------------------------------------------------------------------ b
def celltype():
    v = pd.read_csv(f"{NR}/ROSMAP_cell_type_panels_values.csv")
    r = v[v.comp == K].iloc[0]
    vals = [float(r[f"cor_{c}"]) for c in CT_ORDER]
    b = pd.read_csv(f"{NR}/cell_type_loadings_corrected.csv", index_col=0).loc[K]
    assert np.allclose(vals, [float(b[c]) for c in CT_ORDER])
    return CT_ORDER, vals


# ------------------------------------------------------------------------------------------------ c
def row(path, k=K):
    with open(path) as f:
        head = f.readline().rstrip("\n").split(",")[1:]
        key = f"{k},"
        for line in f:
            if line.startswith(key):
                vals = line.rstrip("\n").split(",")
                return pd.Series(np.asarray(vals[1:], float), index=head)
    raise KeyError(k)


def rna_pip():
    g = row(f"{SDA}/genes_final_RNA.csv"); p = row(f"{SDA}/pip_final_RNA.csv")
    assert list(g.index) == list(p.index)
    return g, p


def manhattan():
    """ros3b RNA panel: points (x, y, colour), labels (x, y, text, colour), chromosome ticks, y limits."""
    g, p = rna_pip()
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
    ylim = (mn - mmin, mx + mmax) if ((mx > 0 and mn >= 0) or (mx <= 0 and mn < 0)) else (mn * 1.1, mx * 1.1)
    return dict(n_pip=int(len(L)), n_annot_rows=int(len(info)), pts=pts, labels=labels, spacer_x=spacers,
                ticks=list(map(float, ticks)), ylim=tuple(map(float, ylim)),
                dup_names=int(info["fid"].duplicated().sum()), unplaced=sorted(set(L.index) - set(info["fid"])))


# ------------------------------------------------------------------------------------------------ d
def seaad():
    """synth1b l.36-60 + l.103-118: genes = ROSMAP RNA genes with PIP > 0.5 that SEA-AD measured (by gene name);
    x = ROSMAP C278 loadings, y = SEA-AD component 67 loadings over the same genes (sorted gene order)."""
    g, p = rna_pip()
    fn = pd.read_csv(f"{SEA}/gene_info.csv", index_col=0)["feature_name"]
    sg = pd.read_csv(f"{SEA}/sda/genes_final.csv", index_col=0)
    sg.columns = [fn[c] for c in sg.columns]
    assert sg.columns.is_unique
    sg.index = sg.index.astype(int)
    common = sorted(set(sg.columns) & set(g.index))
    sel = [c for c in common if p[c] > 0.5]
    x = g[sel].values.astype(float); y = sg.loc[67, sel].values.astype(float)
    return sel, x, y


# ------------------------------------------------------------------------------------------------ e
def kegg():
    df = pd.read_csv(f"{ENRICHR}/278_pathway_encrichment.csv")
    assert df['Adjusted P-value'].is_monotonic_increasing
    top = df.sort_values('Adjusted P-value', kind='mergesort').head(10).reset_index(drop=True)
    assert np.allclose(top['-log10(FDR)'], -np.log10(top['Adjusted P-value']))
    spec = json.load(open(f"{FIGCODE}/replot/kegg_panels/orig_spec.json"))["278"]
    terms, vals = list(top['Term']), top['-log10(FDR)'].to_numpy()
    assert terms == spec['terms'] and np.allclose(vals, spec['values'])
    assert (top['Adjusted P-value'] < 0.05).all()
    return terms, vals, [tuple(c) for c in spec['colors']], spec


# ------------------------------------------------------------------------------------------------ f
def fkbp5():
    sys.path.insert(0, f"{FIGCODE}/panels")
    import vp_common as V
    a = V.score(K); keep = ~V.excl_mask("RNA")
    RNA = V.rna_input(); genes = list(V.comp_loadings(K, "RNA").index)
    out = {ct: np.asarray(RNA[V.CTS_BLOCK.index(ct), :, genes.index("FKBP5")], float)[keep] for ct in ("OPC", "Oli")}
    ak = a[keep]
    return ak, out, V.most_extreme(ak), V
