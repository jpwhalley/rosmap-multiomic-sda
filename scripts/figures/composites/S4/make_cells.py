"""Cell geometry for Figure S4 (mm, origin top-left of the composite). Writes cells_S4.json (used by s4_panels.py)
and layout_S4.json (used by composites/compose.py), both next to this script. Panel files are drawn at exactly these cell sizes, so they are
placed at scale 1 (w_mm = cell width)."""
import json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
W = 142.9
TOP = 4.2          # axes top inside a lettered cell (room for the letter / the S4a key strip)
BOT2 = 9.2         # bottom margin under a 2-line x label
BOT1 = 6.5         # bottom margin under a 1-line x label
GAPY = 2.4         # between the two rows of a 2-row grid (x tick labels hidden on the top row)
L2 = 11.2          # left margin for a 2-line y label + tick labels
L4 = 16.5          # left margin for a 4-line y label + tick labels (S4b right column)
R = 0.8

H1 = 57.0                  # band 1: a | b
Y2 = H1 + 2.0              # band 2 top
H = 160.0                  # total height (budget 162.3)

cells = {}
# a: 3 columns
wa = 82.4; pitch = wa / 3; dw = pitch - L2 - R
cells["a"] = dict(x_mm=0.0, y_mm=0.0, w_mm=wa, h_mm=H1,
                  grid=dict(lefts=[i * pitch + L2 for i in range(3)], top=TOP, w=dw, h=(H1 - TOP - GAPY - BOT2) / 2, gap_y=GAPY))
# b: 2 columns, the right one with the 4-line label
xb = wa + 1.0; wb = W - xb; dwb = (wb - L2 - L4 - 2 * R) / 2
cells["b"] = dict(x_mm=xb, y_mm=0.0, w_mm=wb, h_mm=H1,
                  grid=dict(lefts=[L2, L2 + dwb + R + L4], top=TOP, w=dwb, h=(H1 - TOP - GAPY - BOT2) / 2, gap_y=GAPY))
# c: forest, left column of band 2
wc = 38.6; hc = 54.0; Lc = 14.6
cells["c"] = dict(x_mm=0.0, y_mm=Y2, w_mm=wc, h_mm=hc, ax=dict(x=Lc, y=TOP, w=wc - Lc - 1.5, h=hc - TOP - BOT2))
# d block
xd = wc + 1.0
wd1 = 58.0; hd1 = 54.0; dwd = (wd1 - 2 * (L2 + R)) / 2
cells["d1"] = dict(x_mm=xd, y_mm=Y2, w_mm=wd1, h_mm=hd1,
                   grid=dict(lefts=[L2, L2 + dwd + R + L2], top=TOP, w=dwd, h=(hd1 - TOP - GAPY - BOT2) / 2, gap_y=GAPY))
xr = xd + wd1; wr = W - xr
hd2 = 75.0; Ld2 = 20.4
cells["d2"] = dict(x_mm=xr, y_mm=Y2, w_mm=wr, h_mm=hd2, ax=dict(x=Ld2, y=4.0, w=wr - Ld2 - 1.5, h=hd2 - 4.0 - BOT2))
yd3 = Y2 + hd1 + 1.5; hd3 = H - yd3; Ld3 = 7.4
cells["d3"] = dict(x_mm=xd, y_mm=yd3, w_mm=wd1, h_mm=hd3, ax=dict(x=Ld3, y=2.0, w=wd1 - Ld3 - 1.2, h=hd3 - 2.0 - BOT1))
yd4 = Y2 + hd2 + 1.5; hd4 = H - yd4; Ld4 = 26.0
cells["d4"] = dict(x_mm=xr, y_mm=yd4, w_mm=wr, h_mm=hd4, ax=dict(x=Ld4, y=1.0, w=wr - Ld4 - 1.5, h=hd4 - 1.0 - BOT2))
json.dump(cells, open(f"{HERE}/cells_S4.json", "w"), indent=1)

files = {"a": "S4a_leading_genes_OPC_Oli_vs_score", "b": "S4b_aggregates_with_and_without_leading_genes",
         "c": "S4c_female_minus_male_leading_genes", "d1": "S4d1_SOX6_BCAS1_vs_score",
         "d2": "S4d2_lineage_markers_rho", "d3": "S4d3_OPC_share_vs_score", "d4": "S4d4_marker_modules_rho"}
lay = dict(width_mm=W, height_mm=H, cells=[])
for k, f in files.items():
    c = cells[k]
    e = dict(file=f"$G_ROS_FIGURE_OUT/S4/panels/{f}.pdf", x_mm=round(c["x_mm"], 3), y_mm=round(c["y_mm"], 3), w_mm=round(c["w_mm"], 3))
    if k in ("a", "b", "c"):
        e["letter"] = k
    if k == "d1":
        e["letter"] = "d"
    lay["cells"].append(e)
json.dump(lay, open(f"{HERE}/layout_S4.json", "w"), indent=1)
for k, c in cells.items():
    print(k, {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in c.items() if kk not in ("grid", "ax")},
          {kk: (round(v, 2) if isinstance(v, float) else [round(x, 2) for x in v] if isinstance(v, list) else v)
           for kk, v in (c.get("grid") or c.get("ax")).items()})
