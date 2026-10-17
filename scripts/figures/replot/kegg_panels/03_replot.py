"""Step 3: replot the three KEGG panels (Fig 3e C278, Fig 4e C172, Fig 5e C277) without in-bar gene lists.
Same page size (pt), same 10 terms/order/-log10(FDR), same per-bar RGB (from the original PDF stream),
same style (seaborn white/talk look: 0.15-grey text and frame, 1.875-pt frame, no tick marks, white
1.5-pt bar edges, xlim 0..1.05*max, integer ticks, y slots -0.5..9.5), with
  * no gene text in the bars,
  * pathway names + tick labels sized to print at >= 7 pt at the panel's current placed scale
    (export printed 180 mm wide; Fig 3e is placed non-uniformly, so the smaller (vertical) scale is used),
    x-axis label keeping the original 18:16.5 ratio,
  * bars half their original absolute thickness (0.5 x 44.933 pt),
  * frame right/top edges at the original coordinates (so the panel lines up where it was placed).
Data: archived Enrichr tables (G_ROS_ENRICHR_DIR, read only), orig_spec.json and placement.json (next to this script).
Writes <G_ROS_FIGURE_OUT>/replot/kegg_panels/ (the three PDFs and replot_log.json)."""
import json, math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # scripts/figures
from figure_paths import ENRICHR, FIGOUT
OUT = os.path.join(FIGOUT, 'replot', 'kegg_panels'); os.makedirs(OUT, exist_ok=True)
CSV = ENRICHR + '/{}_pathway_encrichment.csv'
spec = json.load(open(os.path.join(HERE, 'orig_spec.json')))
place = json.load(open(os.path.join(HERE, 'placement.json')))
NAMES = {'278': 'Fig3e_C278_KEGG_v23.pdf', '172': 'Fig4e_C172_KEGG_v23.pdf', '277': 'Fig5e_C277_KEGG_v23.pdf'}
TARGET_PRINT_PT = 7.0
MARGIN_L, MARGIN_B = 1.0, 1.5      # pt of white between page edge and leftmost label / bottom of x label
PAD = 10.5                          # label-to-axis pad, as in the original (seaborn talk)
INK = '0.15'

plt.rcParams.update({
    'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.edgecolor': INK, 'axes.linewidth': 1.875, 'axes.facecolor': 'white', 'figure.facecolor': 'white',
    'xtick.major.size': 0, 'ytick.major.size': 0, 'xtick.minor.size': 0, 'ytick.minor.size': 0,
    'xtick.major.pad': PAD, 'ytick.major.pad': PAD, 'axes.labelpad': 4.0 * 1.5,
    'savefig.facecolor': 'white', 'savefig.transparent': False,
    'path.simplify': False,
})

log = {}
for comp, s in spec.items():
    W, H = s['page']
    ax_x1, ax_y1 = s['axes'][2], s['axes'][3]          # keep frame right/top where they were
    p = place[comp]
    scale_print = min(p['print_scale_x'], p['print_scale_y'])
    fs = math.ceil(TARGET_PRINT_PT / scale_print * 4) / 4   # round up to 0.25 pt
    fs_xlabel = round(fs * s['xlabel_size'] / s['tick_size'], 2)

    df = pd.read_csv(CSV.format(comp))
    top = df.sort_values('Adjusted P-value', kind='mergesort').head(10).reset_index(drop=True)
    terms, vals = list(top['Term']), top['-log10(FDR)'].to_numpy()
    assert terms == s['terms'] and np.allclose(vals, s['values'])
    colors = [tuple(c) for c in s['colors']]

    fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)   # 1 px = 1 pt for layout
    l, b = 0.3 * W, 60.0
    for it in range(6):
        fig.clf()
        ax = fig.add_axes([l / W, b / H, (ax_x1 - l) / W, (ax_y1 - b) / H])
        slot = (ax_y1 - b) / 10
        h = 0.5 * s['bar_thickness'] / slot                # half the original absolute thickness
        ax.barh(np.arange(10), vals, height=h, color=colors, edgecolor='white', linewidth=1.5, zorder=2)
        ax.set_ylim(9.5, -0.5)
        ax.set_xlim(0, 1.05 * vals.max())
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_yticks(np.arange(10), terms)
        ax.tick_params(axis='both', labelsize=fs)
        ax.set_xlabel(s['xlabel'], fontsize=fs_xlabel)
        for sp in ax.spines.values():
            sp.set_linewidth(1.875); sp.set_edgecolor(INK)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        ylab = [t.get_window_extent(r) for t in ax.get_yticklabels()]
        xlab_bb = ax.xaxis.label.get_window_extent(r)
        new_l = l + (MARGIN_L - min(e.x0 for e in ylab))
        new_b = b + (MARGIN_B - xlab_bb.y0)
        if abs(new_l - l) < 1e-3 and abs(new_b - b) < 1e-3:
            break
        l, b = new_l, new_b
    # final extents
    lo, hi = ax.get_xlim()
    xt = [t.get_window_extent(r) for t, loc in zip(ax.get_xticklabels(), ax.get_xticks()) if t.get_text() and lo - 1e-9 <= loc <= hi + 1e-9]
    drawn_xticks = [t.get_text() for t, loc in zip(ax.get_xticklabels(), ax.get_xticks()) if lo - 1e-9 <= loc <= hi + 1e-9]
    allbb = ylab + xt + [xlab_bb]
    ext = dict(x0=min(e.x0 for e in allbb), x1=max(e.x1 for e in allbb), y0=min(e.y0 for e in allbb), y1=max(e.y1 for e in allbb))
    out = os.path.join(OUT, NAMES[comp])
    fig.savefig(out, format='pdf', metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
    plt.close(fig)
    log[comp] = dict(file=out, page=[W, H], font_pt=fs, xlabel_pt=fs_xlabel, print_scale_used=scale_print,
                     print_pt_labels=fs * scale_print, print_pt_xlabel=fs_xlabel * scale_print,
                     print_pt_labels_xdir=fs * p['print_scale_x'],
                     axes_pt=[l, b, ax_x1, ax_y1], slot_pt=slot, bar_pt=h * slot, bar_frac=h,
                     orig_axes_pt=s['axes'], text_extent_pt=ext, xticks=drawn_xticks, layout_iters=it + 1)
    print(f'C{comp}: font {fs} pt (xlabel {fs_xlabel}) -> print {fs*scale_print:.2f} / {fs_xlabel*scale_print:.2f} pt; '
          f'axes x {l:.1f}-{ax_x1:.1f} (orig {s["axes"][0]:.1f}), y {b:.1f}-{ax_y1:.1f} (orig {s["axes"][1]:.1f}); '
          f'bar {h*slot:.2f} pt of slot {slot:.2f} (orig 44.93 / 56.17); xticks {drawn_xticks}; text extent ' + ', '.join(f'{k} {v:.1f}' for k, v in ext.items()))
json.dump(log, open(os.path.join(OUT, 'replot_log.json'), 'w'), indent=1)
