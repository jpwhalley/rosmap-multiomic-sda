"""Recover the stripplot jitter of the published panels from their PDF content streams.

matplotlib's PDF backend draws each stripplot PathCollection as one marker XObject (/M0, /M1,
...) placed with relative moves `1 0 0 1 dx dy cm /Mi Do`, dx/dy printed to 10 decimals and
restarted from (0,0) for each collection. Exact decimal cumulative sums give the absolute PDF
point of every dot. The data<->PDF-point affine of the axes is taken from the re-plot
(probe_v012_geometry.json); its non-marker content (axes frame, ticks, boxes, labels) was
verified token-identical to the original, so the axes geometry is the same.

data_x = (x_pt - pt_x_at_0) / pt_per_x ; jitter = data_x - group_position (0, 1, 2 ...).
Seaborn draws group i as the i-th PathCollection with jitter = np.random.uniform(-0.1, 0.1,
size=n_i) applied in data-row order, so the recovered arrays are fed back in that order.

Output: recovered_jitter.json  {file: [[offsets of group 0], [group 1], ...]}  (no donor IDs,
only per-dot horizontal offsets in draw order).
"""
import json
import os
import sys
from decimal import Decimal, getcontext

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_pdf import parse, markers_abs  # noqa: E402

getcontext().prec = 50
sys.path.insert(0, os.path.dirname(HERE))     # scripts/figures
from figure_paths import V1_PANELS  # noqa: E402
ORIG = f'{V1_PANELS}/outlier'
FILES = {'74_sex_samples.pdf': 74, '278_sex_samples.pdf': 278,
         '172_ceradsc_samples.pdf': 172, '277_cogdx_samples.pdf': 277}

geom = json.load(open(os.path.join(HERE, 'probe_v012_geometry.json')))
out, summary = {}, {}
for fname, k in FILES.items():
    p = parse(os.path.join(ORIG, str(k), fname))
    pts = markers_abs(p)
    g = geom[fname]
    a, c = Decimal(repr(g['pt_per_x'])), Decimal(repr(g['pt_x_at_0']))
    groups = []
    names = sorted(pts, key=lambda s: int(s[1:]))
    assert [len(pts[n]) for n in names] == g['n_by_group'], (fname, [len(pts[n]) for n in names])
    for i, name in enumerate(names):
        jit = [float((x - c) / a - i) for x, _ in pts[name]]
        assert all(abs(j) <= 0.1 + 1e-9 for j in jit), (fname, name, max(map(abs, jit)))
        groups.append(jit)
    out[fname] = groups
    flat = [j for grp in groups for j in grp]
    summary[fname] = dict(n_by_group=[len(x) for x in groups], min=min(flat), max=max(flat),
                          mean=sum(flat) / len(flat))
json.dump(out, open(os.path.join(HERE, 'recovered_jitter.json'), 'w'))
print(json.dumps(summary, indent=1))
