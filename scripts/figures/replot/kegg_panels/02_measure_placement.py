"""Step 2: measure the placed scale of each KEGG panel in the Illustrator exports of the main figures (read-only;
G_ROS_MAIN_FIGURES_DIR). Writes placement.json next to this script.
Scale = (text-matrix size of the first pathway label in the export) / (its size in the panel PDF, 16.5 pt);
cross-checked against the placed panel's white background rect vs the panel page size.
Print scale assumes the whole export is printed 180 mm wide."""
import json, os, re, zlib
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # scripts/figures
from figure_paths import MAIN_FIGURES as FIG
PRINT_W = 180 / 25.4 * 72  # 510.236 pt
spec = json.load(open(os.path.join(HERE, 'orig_spec.json')))
CASES = {'278': ('Fig3', 'comp278_sex .pdf', 'Ras signaling pathway'),
         '172': ('Fig4', 'comp_172.pdf', 'ranscriptional misregulation in cancer'),
         '277': ('Fig5', 'comp_277.pdf', 'Morphine addiction')}

def page_content(path):
    data = open(path, 'rb').read()
    pw, ph = [float(v) for v in re.search(rb'/MediaBox\[0\.0 0\.0 ([\d.]+) ([\d.]+)\]', data).groups()]
    page = re.search(rb'/Contents\[([\d 0R]+)\]', data).group(1)
    nums = re.findall(rb'(\d+) 0 R', page)
    out = []
    for n in nums:
        m = re.search(rb'(?<!\d)' + n + rb' 0 obj\s*<<(.*?)>>\s*stream\r?\n', data, re.S)
        L = int(re.search(rb'/Length (\d+)', m.group(1)).group(1))
        out.append(zlib.decompress(data[m.end():m.end() + L]).decode('latin1'))
    return (pw, ph), '\n'.join(out)

res = {}
for comp, (fig, fname, first) in CASES.items():
    (pw, ph), st = page_content(os.path.join(FIG, fname))
    k = st.index(first)
    tm = list(re.finditer(r'([\d.\-]+) ([\d.\-]+) ([\d.\-]+) ([\d.\-]+) ([\d.\-]+) ([\d.\-]+) Tm', st[:k]))[-1]
    a, b, c, d = (float(tm.group(i)) for i in range(1, 5))
    sx, sy = a / spec[comp]['tick_size'], d / spec[comp]['tick_size']
    W, H = spec[comp]['page']
    # placed panel background: matplotlib's full-page white rect, either direct or inside a cm
    bg = None
    for m in re.finditer(r'(?:([\d.\-]+) 0 0 ([\d.\-]+) 0 0 cm\n)?([\d.\-]+) ([\d.\-]+) ([\d.\-]+) ([\d.\-]+) re\nf', st[:k]):
        sxx = float(m.group(1)) if m.group(1) else 1.0
        syy = abs(float(m.group(2))) if m.group(2) else 1.0
        w, h = float(m.group(5)) * sxx, abs(float(m.group(6))) * syy
        if abs(w / W - sx) < 2e-3 and abs(h / H - sy) < 2e-3:
            bg = (w, h)
    f = PRINT_W / pw
    res[comp] = dict(figure=fig, export=fname, export_page_pt=[pw, ph], placed_scale_x=sx, placed_scale_y=sy,
                     placed_size_pt=[W * sx, H * sy], bg_rect_check=bg, print_factor_180mm=f,
                     print_scale_x=sx * f, print_scale_y=sy * f)
    print(f'C{comp} {fig} ({fname}): export {pw:.3f} x {ph:.3f} pt -> x{f:.4f} to print at 180 mm')
    print(f'   placed scale x {sx:.6f}, y {sy:.6f} (uniform: {abs(sx-sy)<1e-4}); placed size {W*sx:.2f} x {H*sy:.2f} pt; bg rect {bg}')
    print(f'   print scale x {sx*f:.4f}, y {sy*f:.4f}; original 16.5-pt labels print at {16.5*sy*f:.2f} pt (height), xlabel 18 pt -> {18*sy*f:.2f} pt')
json.dump(res, open(os.path.join(HERE, 'placement.json'), 'w'), indent=1)
