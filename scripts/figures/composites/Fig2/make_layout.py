"""Write layout_Fig2.json (next to this script) for composites/compose.py from the cell geometry that fig2_panels.py
used (<G_ROS_FIGURE_OUT>/Fig2/panels/geometry.json). Every panel PDF is exactly its cell, placed at scale 1; letters at the cell's top-left corner
(Arial Bold 8 pt, lower case), so letters in a row share y and letters in a column share x."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # scripts/figures
from figure_paths import FIGOUT
g = json.load(open(f"{FIGOUT}/Fig2/panels/geometry.json"))
NAMES = dict(a="Fig2a_C74_sex_samples", b="Fig2b_C74_cell_type", c="Fig2c_C74_SEAAD_scatter", d="Fig2d_C74_RNA",
             e="Fig2e_C74_chip", f="Fig2f_C74_meth", g="Fig2g_C74_cpg_H3K9ac_vs_score")
cells = []
for L, (x, r, w) in g["CELLS"].items():
    cells.append({"file": f"$G_ROS_FIGURE_OUT/Fig2/panels/{NAMES[L]}.pdf", "x_mm": x, "y_mm": g["ROW_Y"][r], "scale": 1.0, "letter": L})
lay = {"width_mm": g["W"], "height_mm": g["H"], "letter_size_pt": 8, "cells": cells}
json.dump(lay, open(f"{HERE}/layout_Fig2.json", "w"), indent=1)
print(json.dumps(lay, indent=1))
