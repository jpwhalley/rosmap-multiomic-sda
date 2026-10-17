"""Corrected cell-type panels for relinking in the Illustrator layouts, in the ORIGINAL panel style.

Style sources (version-1 scripts):
  ROSMAP  : ros3b_plot_outliers.py (Oct 20 2025, the run that wrote the linked
            g_ros/figs/outlier/<k>/<k>_cell_type.pdf): l.23 rcParams['pdf.fonttype'] = 42,
            l.24 sns.set(style='white', context='talk'), l.213-220 colours, l.325-339 bar code
            (figsize (6, 6) for C74 else (9, 6); plt.tight_layout(); plt.savefig(pdf)).
  SEA-AD  : dlpfc3b_plot_separate.py l.14 sns.set(...), l.42-50 labels/order,
            l.86 colours, l.369-379 bar code (Type 3 fonts: no pdf.fonttype set in that script).

An earlier draft of these panels was written without pdf.fonttype = 42 (Arial as Type 3, where the linked
originals embed it as TrueType); this script restores the original settings. Values are read unchanged from the
catalogue tables (G_ROS_TABLES_DIR):
  ROSMAP_cell_type_panels_values.csv   (cor_* columns; pub_* only for the fidelity check)
  FigS2c_SEAAD_C21_cell_type_values.csv ('corrected' column; 'published_S2c' for the check)

Usage: python make_relink_panels.py OUTDIR    (e.g. $G_ROS_FIGURE_OUT/cell_type_bars)
Writes OUTDIR/{corrected,published_repro}/<mirrored path>/<k>_cell_type.pdf
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt, seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # scripts/figures
from figure_paths import TABLES as NR
OUT = sys.argv[1]

ROS_ORDER = ['Ast', 'Exc', 'Inh', 'Mic', 'OPC', 'Oli']                       # ros3b l.32 / l.332
ROS_COL = {'Ast': '#D85396', 'Oli': '#8BC34A', 'Inh': '#303F9F',
           'OPC': '#E5A43A', 'Exc': '#D32F2F', 'Mic': '#4DB6E7'}              # ros3b l.213-220
SEA_ORDER = ['glutamatergic\nneuron', 'pvalb', 'sst', 'vip', 'sncg', 'lamp5',
             'Oligo-\ndendrocyte', 'OPC', 'Astrocytes', 'Microglia']          # dlpfc3b l.50
SEA_COL = ['red', 'green', 'lightgreen', 'darkgreen', 'lime', 'yellowgreen',
           'darkorange', 'purple', 'blue', 'darkgray']                        # dlpfc3b l.86


def ros_panel(k, vals, path):
    # exactly ros3b_plot_outliers.py (g_ros_manu, Oct 20 2025) l.326-340
    mpl.rcdefaults()
    mpl.rcParams['pdf.fonttype'] = 42                                        # l.23
    sns.set(style='white', context='talk')                                   # l.24
    if k == 74:
        fig, ax3 = plt.subplots(figsize=(6, 6))
    else:
        fig, ax3 = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(ROS_ORDER))
    ax3.bar(y_pos, list(vals), align='center', color=[ROS_COL[x] for x in ROS_ORDER])
    ax3.set_xticks(y_pos)
    ax3.set_xticklabels(ROS_ORDER)
    plt.ylabel('loadings/score')
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close(fig)


def sea_panel(vals, path):
    # exactly dlpfc3b_plot_separate.py l.369-379 (no pdf.fonttype set -> Type 3, as the original)
    mpl.rcdefaults()
    sns.set(style='white', context='talk')                                   # l.14
    fig, ax3 = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(SEA_ORDER))
    ax3.bar(y_pos, list(vals), align='center', color=SEA_COL)
    ax3.set_xticks(y_pos)
    ax3.set_xticklabels(SEA_ORDER, fontsize=10)
    plt.ylabel('loadings/score')
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close(fig)


v = pd.read_csv(os.path.join(NR, "ROSMAP_cell_type_panels_values.csv"))
for k in (351, 2, 74, 87, 278, 177):
    row = v[v.comp == k].iloc[0]
    for kind, sub in (("cor", "corrected"), ("pub", "published_repro")):
        vals = [float(row[f"{kind}_{c}"]) for c in ROS_ORDER]
        ros_panel(k, vals, os.path.join(OUT, sub, "g_ros/figs/outlier", str(k), f"{k}_cell_type.pdf"))

s = pd.read_csv(os.path.join(NR, "FigS2c_SEAAD_C21_cell_type_values.csv"), index_col=0)
name = s["cell_type"].to_dict()
name["CL:0000128"] = "Oligo-\ndendrocyte"; name["CL:4023040"] = "glutamatergic\nneuron"   # dlpfc3b l.42-43
for col, sub in (("corrected", "corrected"), ("published_S2c", "published_repro")):
    ser = pd.Series(s[col].values.astype(float), index=[name[c] for c in s.index]).reindex(SEA_ORDER)
    assert ser.notna().all()
    sea_panel(ser.values, os.path.join(OUT, sub, "e_brain_td/figs/dlpfc/separate/21/21_cell_type.pdf"))

print("matplotlib", matplotlib.__version__, "seaborn", sns.__version__)
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        if f.endswith(".pdf"):
            print(os.path.join(root, f))
