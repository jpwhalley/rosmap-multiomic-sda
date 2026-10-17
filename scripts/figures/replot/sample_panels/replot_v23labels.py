"""v23 relabelled sample-score panels: Fig 4a (C172 ceradsc) and Fig 5a (C277 cogdx).

Reuses, unmodified, the port of the original panel code
  scripts/figures/qlabel_panels/qlabel_panels.py
(loaded read-only as a module: data loading, +/-5 SD outlier filter, seaborn-0.12 box shim,
statannotations bracket, recovered stripplot jitter, Set2 / husl palettes, talk context,
pdf.fonttype 42) and the 547-family BH q values (clinical_BH_547_family.csv), i.e. exactly the
inputs that produced the corrected v23 panel files (reproduced byte-identically before relabelling).

Only the text changes:
  172: x title 'ceradsc' -> 'CERAD neuritic plaque score'; ticks 'No' -> 'Non-AD (CERAD 3–4)',
       "Alzheimer's Disease" -> 'AD (CERAD 1–2)'; bracket label 'p = 0.0168' -> 'FDR = 0.017'.
  277: x title 'cogdx' -> 'Clinical diagnosis'; ticks -> 'NCI', 'MCI', 'AD dementia';
       label 'Kruskal adj p = 0.00242' -> 'FDR = 0.0024'; y top raised only as far as needed so
       the label clears every point by >= GAP_PT.
Figure size (9 x 6 in = 648 x 432 pt) and plt.tight_layout() as in the original.

Prints only group sizes, q values, limits and clearances; no donor IDs or per-donor values.
Run: python replot_v23labels.py (writes under <G_ROS_FIGURE_OUT>/replot/sample_panels/)
"""
import importlib.util
import json
import os

import sys
FIGCODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/figures
sys.path.insert(0, FIGCODE)
from figure_paths import FIGOUT
QDIR = os.path.join(FIGCODE, 'qlabel_panels')
HERE = os.path.join(FIGOUT, 'replot', 'sample_panels')
OUT = os.path.join(HERE, 'out')

_spec = importlib.util.spec_from_file_location('qlabel_panels', os.path.join(QDIR, 'qlabel_panels.py'))
qp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qp)  # sets Agg, pdf.fonttype 42, sns talk/white, appends vendor/ to sys.path

plt, np, pd, mpl = qp.plt, qp.np, qp.pd, qp.mpl
assert mpl.rcParams['pdf.fonttype'] == 42

GAP_PT = 3.0          # required clearance between label box and marker edge (PDF points)
MARKER_R_PT = 2.5     # stripplot size=5 -> marker diameter 5 pt

XT_172 = ['Non-AD (CERAD 3–4)', 'AD (CERAD 1–2)']   # order = ["No", "Alzheimer's\nDisease"]
XT_277 = ['NCI', 'MCI', 'AD dementia']                     # order = cogdx 1, 2, 4


def fdr_label(q):
    return f'FDR = {q:.2g}'


def panel_ceradsc_v23(samples, k, q, jit):
    """qp.panel_ceradsc with the v23 text."""
    comp = k
    feature = 'ceradsc'
    fig, ax = plt.subplots(figsize=(9, 6))
    filtered_samples = qp.outlier_filter(samples, comp)
    group_order = ["No", "Alzheimer's\nDisease"]
    filtered_samples = filtered_samples[filtered_samples[feature].isin(group_order)]
    filtered_samples[feature] = pd.Categorical(filtered_samples[feature], categories=group_order, ordered=True)
    qp.boxplot(x=feature, y=comp, data=filtered_samples, ax=ax,
               showfliers=False, palette="Set2", order=["No", "Alzheimer's\nDisease"])
    label = fdr_label(q)
    pairs = [("No", "Alzheimer's\nDisease")]
    annotator = qp.Annotator(ax, pairs, data=filtered_samples, x=feature, y=comp,
                             order=["No", "Alzheimer's\nDisease"])
    annotator.set_pvalues([q])
    annotator.configure(text_format="simple", show_test_name=False)
    annotator.set_custom_annotations([label])
    annotator.annotate()
    qp.stripplot(jit, 0, x=feature, y=comp, data=filtered_samples, ax=ax,
                 jitter=True, color='black', size=5)
    ax.set_xlabel('CERAD neuritic plaque score')
    ax.set_ylabel('loadings/score')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(XT_172)
    plt.tight_layout()
    return fig, ax, filtered_samples, label


def panel_cogdx_v23(samples, k, q, jit, ytop=None):
    """qp.panel_cogdx with the v23 text (+ optional raised y top)."""
    comp = k
    feature = 'cogdx'
    fig, ax = plt.subplots(figsize=(9, 6))
    filtered_samples = qp.outlier_filter(samples, comp)
    group_order = [1, 2, 4]
    group_labels = {1: "No cognitive \n impairment", 2: "Mild cognitive \n impairment",
                    4: "Alzheimer's\nDisease"}
    filtered_samples = filtered_samples[filtered_samples[feature].isin(group_order)]
    codes = filtered_samples[feature].copy()
    filtered_samples[feature] = filtered_samples[feature].map(group_labels)
    filtered_samples[feature] = pd.Categorical(filtered_samples[feature],
                                               categories=[group_labels[g] for g in group_order],
                                               ordered=True)
    qp.boxplot(x=feature, y=comp, data=filtered_samples, ax=ax, showfliers=False, palette="husl",
               order=["No cognitive \n impairment", "Mild cognitive \n impairment", "Alzheimer's\nDisease"])
    label = fdr_label(q)
    txt = ax.text(0.5, 0.95, label, ha='center', va='center', transform=ax.transAxes)
    qp.stripplot(jit, 0, x=feature, y=comp, data=filtered_samples, ax=ax,
                 jitter=True, color='black', size=5)
    ax.set_xlabel('Clinical diagnosis')
    ax.set_ylabel('loadings/score')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(XT_277)
    if ytop is not None:
        ax.set_ylim(top=ytop)
    plt.tight_layout()
    return fig, ax, filtered_samples, label, txt, codes


def strip_offsets(ax):
    return [np.asarray(c.get_offsets(), float) for c in ax.collections
            if type(c).__name__ == 'PathCollection']


def artists_data(ax):
    """Box / whisker / cap / median / bracket geometry in data coordinates."""
    lines = [np.asarray(l.get_xydata(), float) for l in ax.lines]
    patches = [np.asarray(p.get_path().vertices, float) for p in ax.patches]
    return lines, patches


def same_drawing(ax_new, ax_ref, what):
    a, b = strip_offsets(ax_new), strip_offsets(ax_ref)
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b)), f'{what}: points differ'
    la, pa = artists_data(ax_new)
    lb, pb = artists_data(ax_ref)
    assert len(la) == len(lb) and all(np.array_equal(x, y) for x, y in zip(la, lb)), f'{what}: lines differ'
    assert len(pa) == len(pb) and all(np.array_equal(x, y) for x, y in zip(pa, pb)), f'{what}: boxes differ'
    fa = [p.get_facecolor() for p in ax_new.patches]
    fb = [p.get_facecolor() for p in ax_ref.patches]
    assert fa == fb, f'{what}: box colours differ'
    print(f'  {what}: points, boxes, whiskers, caps, medians, bracket and colours identical to the '
          f'v23 panel ({sum(len(x) for x in a)} points in {len(a)} strips)')


def label_clearance(fig, ax, txt):
    """Minimum distance (pt) between the label's box and any marker edge, at the PDF's 72 dpi."""
    dpi0 = fig.dpi
    fig.set_dpi(72)
    r = fig.canvas.get_renderer()
    bb = txt.get_window_extent(r)
    best = np.inf
    worst_needed_frac = []
    ax_bb = ax.get_window_extent(r).frozen()   # axes bbox is lazy: freeze at 72 dpi
    for off in strip_offsets(ax):
        p = ax.transData.transform(off)
        dx = np.maximum(np.maximum(bb.x0 - p[:, 0], 0), p[:, 0] - bb.x1)
        dy = np.maximum(np.maximum(bb.y0 - p[:, 1], 0), p[:, 1] - bb.y1)
        d = np.hypot(dx, dy) - MARKER_R_PT
        best = min(best, float(d.min()))
        # points horizontally under the label: data y they occupy
        under = dx < MARKER_R_PT + GAP_PT
        if under.any():
            worst_needed_frac.append(off[under, 1].max())
    f_lo = (bb.y0 - ax_bb.y0) / ax_bb.height      # label bottom, axes fraction
    fig.set_dpi(dpi0)
    return best, bb, f_lo, ax_bb.height, (max(worst_needed_frac) if worst_needed_frac else None)


def axes_pt(fig, ax):
    """Axes box (x0, y0, width, height) in PDF points."""
    dpi0 = fig.dpi
    fig.set_dpi(72)
    b = [round(float(v), 2) for v in ax.get_window_extent(fig.canvas.get_renderer()).frozen().bounds]
    fig.set_dpi(dpi0)
    return b


def lines_clear(fig, ax, txt):
    r = fig.canvas.get_renderer()
    bb = txt.get_window_extent(r)
    for l in ax.lines:
        if l.get_window_extent(r).overlaps(bb):
            return False
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = qp.load_samples()
    q547 = pd.read_csv(qp.BH547, index_col=0)
    jit_all = json.load(open(os.path.join(QDIR, 'recovered_jitter.json')))
    clin = pd.read_csv(qp.CLIN, index_col=0)
    clin.index = clin.index.astype(str)
    report = {}

    # ------------------------------------------------------------------ Fig 4a, C172 ceradsc
    k, fname = 172, '172_ceradsc_samples.pdf'
    q = float(q547.loc[k, 'ceradsc'])
    assert f'{q:.4g}' == '0.0168' and fdr_label(q) == 'FDR = 0.017', q
    print(f'C172 ceradsc: BH q (547 family) = {q:.6g} -> {fdr_label(q)!r}')
    fig_ref, ax_ref, _, lab_ref = qp.panel_ceradsc(samples, k, q, jit_all[fname], 0)
    assert lab_ref == 'p = 0.0168'
    fig, ax, fs, label = panel_ceradsc_v23(samples, k, q, jit_all[fname])
    n = fs['ceradsc'].value_counts().reindex(["No", "Alzheimer's\nDisease"])
    raw = clin.loc[fs.index, 'ceradsc']
    xt = pd.crosstab(fs['ceradsc'].astype(str).str.replace('\n', ' '), raw)
    print('  group sizes (left->right):', dict(zip(XT_172, n.tolist())))
    print('  raw CERAD score by plotted group (counts):\n' + xt.to_string())
    assert n.tolist() == [106, 170]
    assert set(raw[fs['ceradsc'] == 'No'].unique()) <= {3, 4}
    assert set(raw[fs['ceradsc'] == "Alzheimer's\nDisease"].unique()) <= {1, 2}
    assert [t.get_text() for t in ax.get_xticklabels()] == XT_172
    assert ax.get_xlim() == ax_ref.get_xlim() and ax.get_ylim() == ax_ref.get_ylim(), 'limits changed'
    same_drawing(ax, ax_ref, 'C172')
    out = os.path.join(OUT, '172', '172_ceradsc_samples_v23labels.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, metadata={'CreationDate': None})
    lab_txt = [t for t in ax.texts if t.get_text() == label]
    assert len(lab_txt) == 1
    clr, *_ = label_clearance(fig, ax, lab_txt[0])
    report['172'] = dict(label=label, ylim=list(ax.get_ylim()), min_clearance_pt=clr,
                         axes_pt_new=axes_pt(fig, ax), axes_pt_v23=axes_pt(fig_ref, ax_ref),
                         lines_clear=lines_clear(fig, ax, lab_txt[0]))
    print(f'  wrote {out}; ylim unchanged {ax.get_ylim()}; label-to-marker clearance {clr:.1f} pt')
    plt.close('all')

    # ------------------------------------------------------------------ Fig 5a, C277 cogdx
    k, fname = 277, '277_cogdx_samples.pdf'
    q = float(q547.loc[k, 'cogdx'])
    assert f'{q:.4g}' == '0.00242' and fdr_label(q) == 'FDR = 0.0024', q
    print(f'C277 cogdx: BH q (547 family) = {q:.6g} -> {fdr_label(q)!r}')
    fig_ref, ax_ref, _, lab_ref = qp.panel_cogdx(samples, k, q, jit_all[fname], 0)
    assert lab_ref == 'Kruskal adj p = 0.00242'
    ylim_v23 = ax_ref.get_ylim()
    ref_txt = [t for t in ax_ref.texts if t.get_text() == lab_ref][0]
    clr_v23, *_ = label_clearance(fig_ref, ax_ref, ref_txt)
    print(f'  v23 panel: label-to-marker clearance {clr_v23:.1f} pt (negative = overlap)')

    fig, ax, fs, label, txt, codes = panel_cogdx_v23(samples, k, q, jit_all[fname])
    clr, bb, f_lo, A, ymax_under = label_clearance(fig, ax, txt)
    y0, T = ax.get_ylim()
    print(f'  relabelled, original y limits: clearance {clr:.1f} pt')
    ytop = None
    if clr < GAP_PT:
        need = y0 + (ymax_under - y0) / (f_lo - (MARKER_R_PT + GAP_PT) / A)
        ytop = float(np.ceil(need * 1e4) / 1e4)
        plt.close(fig)
        for _ in range(5):   # tight_layout may move the axes a hair when ylim changes; re-check
            fig, ax, fs, label, txt, codes = panel_cogdx_v23(samples, k, q, jit_all[fname], ytop=ytop)
            clr, bb, f_lo, A, ymax_under = label_clearance(fig, ax, txt)
            if clr >= GAP_PT:
                break
            ytop = float(np.ceil((y0 + (ymax_under - y0) / (f_lo - (MARKER_R_PT + GAP_PT) / A) + 1e-4) * 1e4) / 1e4)
            plt.close(fig)
        assert clr >= GAP_PT, clr
        print(f'  y top raised {T:.6g} -> {ytop:.6g} (bottom unchanged {y0:.6g}); clearance {clr:.1f} pt')
    assert ax.get_ylim()[0] == ylim_v23[0] and ax.get_xlim() == ax_ref.get_xlim()
    n = fs['cogdx'].value_counts().reindex(["No cognitive \n impairment", "Mild cognitive \n impairment",
                                            "Alzheimer's\nDisease"])
    print('  group sizes (left->right):', dict(zip(XT_277, n.tolist())),
          '| cogdx codes:', codes.value_counts().reindex([1, 2, 4]).to_dict())
    assert n.tolist() == [97, 65, 92] and codes.value_counts().reindex([1, 2, 4]).tolist() == [97, 65, 92]
    assert [t.get_text() for t in ax.get_xticklabels()] == XT_277
    same_drawing(ax, ax_ref, 'C277')
    assert lines_clear(fig, ax, txt)
    out = os.path.join(OUT, '277', '277_cogdx_samples_v23labels.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, metadata={'CreationDate': None})
    report['277'] = dict(label=label, ylim=list(ax.get_ylim()), ylim_v23=list(ylim_v23),
                         ytop=ytop, min_clearance_pt=clr, min_clearance_pt_v23=clr_v23,
                         axes_pt_new=axes_pt(fig, ax), axes_pt_v23=axes_pt(fig_ref, ax_ref))
    print(f'  wrote {out}; ylim {ax.get_ylim()}')
    plt.close('all')
    json.dump(report, open(os.path.join(HERE, 'replot_v23labels_report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
