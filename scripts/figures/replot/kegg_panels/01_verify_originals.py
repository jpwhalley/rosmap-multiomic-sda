"""Step 1: confirm each original KEGG panel = top-10 rows (by adjusted P) of the archived Enrichr table.
Extracts terms/order/bar lengths/colours from the original PDF and compares; also samples bar colours
from a pdftoppm render. Writes orig_spec.json (geometry + colours; next to this script) used by the replot script.
Inputs: the version-1 KEGG panels (G_ROS_V1_PANELS_DIR/enrichr/) and the Enrichr tables (G_ROS_ENRICHR_DIR)."""
import json, os, subprocess, sys
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from pdfparse import page_stream, texts, filled_rects

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # scripts/figures
from figure_paths import V1_PANELS, ENRICHR, FIGOUT
CSV = ENRICHR + '/{}_pathway_encrichment.csv'
PNGDIR = os.path.join(FIGOUT, 'replot', 'kegg_panels', 'orig_png'); os.makedirs(PNGDIR, exist_ok=True)
ARIAL_DIGIT_W = 0.556  # Arial digit advance (em)

spec = {}
ok_all = True
for comp in ['278', '172', '277']:
    pdf = f'{V1_PANELS}/enrichr/{comp}_pathway_encrichment.pdf'
    media, st = page_stream(pdf)
    T = texts(st)
    R = filled_rects(st)
    W, H = media[2], media[3]
    # axes background = white rect not at origin; bars = coloured rects with 'B' (fill+stroke)
    bars = [r for r in R if r['op'] == 'B']
    bars.sort(key=lambda r: -r['y1'])
    ax_rect = [r for r in R if r['op'] == 'f' and r['x0'] > 0][0]
    ticks = [t for t in T if t['text'].strip().isdigit() and abs(t['y'] - min(tt['y'] for tt in T if tt['text'].strip().isdigit())) < 1e-6]
    tick_x = {int(t['text']): t['x'] + ARIAL_DIGIT_W * t['size'] / 2 for t in ticks}
    ks = sorted(tick_x)
    scale = np.polyfit(ks, [tick_x[k] for k in ks], 1)  # pt per unit, origin
    xlabel = [t for t in T if 'FDR' in t['text']][0]
    term_labels = [t for t in T if t['size'] == ticks[0]['size'] and not t['text'].strip().isdigit()]
    term_labels.sort(key=lambda t: -t['y'])
    pdf_terms = [t['text'] for t in term_labels]
    gene_texts = [t for t in T if t['size'] not in (ticks[0]['size'], xlabel['size'])]

    df = pd.read_csv(CSV.format(comp))
    assert df['Adjusted P-value'].is_monotonic_increasing, 'table not sorted by adj P'
    top = df.sort_values('Adjusted P-value', kind='mergesort').head(10).reset_index(drop=True)
    assert np.allclose(top['-log10(FDR)'], -np.log10(top['Adjusted P-value']))
    vals_pdf = [(b['x1'] - b['x0']) / scale[0] for b in bars]
    print(f'== C{comp}: page {W} x {H} pt; axes x {ax_rect["x0"]:.3f}-{ax_rect["x1"]:.3f}, y {ax_rect["y0"]:.3f}-{ax_rect["y1"]:.3f}')
    print(f'   tick fit: {scale[0]:.4f} pt/unit, origin {scale[1]:.3f} (axes x0 {ax_rect["x0"]:.3f}); xlim max = {(ax_rect["x1"]-scale[1])/scale[0]:.4f}, 1.05*max(val) = {1.05*top["-log10(FDR)"].max():.4f}')
    print(f'   font sizes: ticks/terms {ticks[0]["size"]}, xlabel {xlabel["size"]} ({xlabel["text"]!r}), gene text {sorted(set(t["size"] for t in gene_texts))} (n={len(gene_texts)})')
    print(f'   bars: n={len(bars)}, thickness {np.unique(np.round([b["y1"]-b["y0"] for b in bars],3))} pt, slot {np.round(np.diff([b["y1"] for b in bars]).mean(),3)} pt')
    rows = []
    for i in range(10):
        b = bars[i]; lab = term_labels[i]
        bar_mid = (b['y0'] + b['y1']) / 2
        rows.append(dict(i=i + 1, term_pdf=pdf_terms[i], term_csv=top.loc[i, 'Term'],
                         same=pdf_terms[i] == top.loc[i, 'Term'],
                         val_pdf=round(vals_pdf[i], 4), val_csv=round(top.loc[i, '-log10(FDR)'], 4),
                         dval=round(vals_pdf[i] - top.loc[i, '-log10(FDR)'], 5),
                         label_to_bar_dy=round(lab['y'] - bar_mid, 2),
                         rgb=tuple(round(c, 4) for c in b['color'])))
    rt = pd.DataFrame(rows)
    with pd.option_context('display.width', 250, 'display.max_colwidth', 60):
        print(rt.to_string(index=False))
    ok = rt['same'].all() and (rt['dval'].abs() < 1e-3).all() and np.ptp(rt['label_to_bar_dy']) < 0.5  # label baseline vs bar centre: constant offset (0.1-pt rounding)
    # sample colours from a render (72 dpi -> 1 px = 1 pt), top band of each bar, middle of its length
    png = os.path.join(PNGDIR, f'orig_{comp}_72')
    subprocess.run(['pdftoppm', '-png', '-r', '72', '-singlefile', pdf, png], check=True)
    im = np.asarray(Image.open(png + '.png').convert('RGB')).astype(float)
    samp = []
    for b in bars:
        x = int((b['x0'] + b['x1']) / 2); y = int(H - b['y1'] + 5)
        px = im[y - 1:y + 2, x - 3:x + 4].reshape(-1, 3).mean(0) / 255
        samp.append(px)
    diff = np.abs(np.array(samp) - np.array([b['color'] for b in bars])).max()
    print(f'   render-sampled vs stream colours: max |diff| = {diff:.4f} (8-bit = {diff*255:.1f})')
    print(f'   CHECK C{comp}: terms+order+lengths match top-10 = {ok}')
    if comp == '277':
        print(f'   C277 bar 8 = {pdf_terms[7]!r}; bar 10 = {pdf_terms[9]!r}')
        ok = ok and pdf_terms[7] == 'GABAergic synapse' and pdf_terms[9] == 'Calcium signaling pathway'
    ok_all &= ok
    # tie note: rows at the 10/11 boundary
    tie = df.loc[9, 'Adjusted P-value'] == df.loc[10, 'Adjusted P-value']
    if tie:
        print(f'   NOTE: row 10 ({df.loc[9,"Term"]!r}) ties row 11 ({df.loc[10,"Term"]!r}) on adj P; panel uses table order (row 10).')
    spec[comp] = dict(page=[W, H], axes=[ax_rect['x0'], ax_rect['y0'], ax_rect['x1'], ax_rect['y1']],
                      pt_per_unit=scale[0], bar_thickness=float(np.mean([b['y1'] - b['y0'] for b in bars])),
                      slot=float(np.diff([b['y1'] for b in bars]).mean() * -1),
                      tick_size=ticks[0]['size'], xlabel_size=xlabel['size'], xlabel=xlabel['text'],
                      terms=pdf_terms, values=[float(v) for v in top['-log10(FDR)']],
                      colors=[list(b['color']) for b in bars])
json.dump(spec, open(os.path.join(HERE, 'orig_spec.json'), 'w'), indent=1)
print('ALL ORIGINALS VERIFIED' if ok_all else 'MISMATCH FOUND')
