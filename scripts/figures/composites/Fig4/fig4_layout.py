"""Write layout_fig4.json (next to this script) for composites/compose.py from the panel geometry
(<G_ROS_FIGURE_OUT>/Fig4/geometry.json, written by fig4_panels.py). Rows a b / c d / e (full width) as the Illustrator Figure 4; every panel is
placed at scale 1 (each panel PDF is exactly its cell), letters at each cell's top-left corner, so a/c/e share x = 0,
b/d share x = RX, and the letters of a row share y."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # scripts/figures
from figure_paths import FIGOUT
g = json.load(open(f"{FIGOUT}/Fig4/geometry.json"))
H = {int(k): v for k, v in g["ROW_H"].items()}
y1, y2, y3 = 0.0, H[1] + g["ROW_GAP"], H[1] + H[2] + 2 * g["ROW_GAP"]
P = "$G_ROS_FIGURE_OUT/Fig4/panels"
cells = [("a", "Fig4a_C172_ceradsc_samples_cell", 0.0, y1), ("b", "Fig4b_C172_cell_type_cell", g["RX"], y1),
         ("c", "Fig4c_C172_RNA_cell", 0.0, y2), ("d", "Fig4d_C172_LRP8_Inh_vs_score_cell", g["RX"], y2),
         ("e", "Fig4e_C172_KEGG_cell", 0.0, y3)]
lay = dict(width_mm=g["W"], height_mm=round(y3 + H[3], 3), letter_size_pt=8,
           cells=[dict(file=f"{P}/{f}.pdf", x_mm=round(x, 3), y_mm=round(y, 3), scale=1, letter=l)
                  for l, f, x, y in cells])
json.dump(lay, open(f"{HERE}/layout_fig4.json", "w"), indent=1)
print(json.dumps(lay, indent=1))
