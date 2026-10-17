"""Re-plot the four clinical q-label panels of the ROSMAP SDA manuscript.

Faithful port of the panel blocks of the version-1 script
  ros6a_cross_clin_discrete.py  (l. 713-884)
  - {k}_sex_samples.pdf     (Mann-Whitney, statannotations bracket, label f"p = {q:.4g}")
  - {k}_cogdx_samples.pdf   (Kruskal, ax.text label f"Kruskal adj p = {q:.4g}")
  - {k}_ceradsc_samples.pdf (Mann-Whitney, statannotations bracket, label f"p = {q:.4g}")
with the data preparation of l. 25-36, 55-68, 159-183 and the BH families of l. 185-291.

Jitter: the original stripplot jitter was unseeded. Pass a JSON of per-group jitter
arrays (recovered from the original PDF content streams by recover_jitter.py) with
--jitter; they are fed back through seaborn's own code path by temporarily replacing
numpy.random.uniform (seaborn 0.13 plot_strips: partial(np.random.uniform, low=-jlim,
high=+jlim)(size=len(group)), one call per x-category with >1 point, in draw order).
Without --jitter a fixed seed is used.

Modes:
  --qfamily published : label = BH q over the 546-component family (recomputed here exactly
                        as the original script does; checked against the published labels)
  --qfamily corrected : label = BH q over the 547-component family from
                        clinical_BH_547_family.csv (G_ROS_TABLES_DIR)

Run with: python qlabel_panels.py ...
No donor-level table is written anywhere by this script.
"""
import argparse
import json
import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
# statannotations 0.7.2 (pure python, unmodified, MIT licence in vendor/statannotations/LICENSE); appended at
# the END of sys.path so every other package resolves from the locked environment.
sys.path.append(os.path.join(HERE, 'vendor'))

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from statannotations.Annotator import Annotator

sys.path.insert(0, os.path.dirname(HERE))     # scripts/figures
from figure_paths import SDA, CLINICAL, TABLES
SAMPLES = f'{SDA}/samples_final.csv'
GENES_RNA = f'{SDA}/genes_final_RNA.csv'
CLIN = CLINICAL
BH547 = f'{TABLES}/clinical_BH_547_family.csv'

PANELS = [  # (component, feature, original file)
    (74, 'Sex', '74_sex_samples.pdf'),
    (278, 'Sex', '278_sex_samples.pdf'),
    (172, 'ceradsc', '172_ceradsc_samples.pdf'),
    (277, 'cogdx', '277_cogdx_samples.pdf'),
]

# ---------------------------------------------------------------- style (original l. 25-29)
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='white', context='talk')
# (the original defines a `font` dict but never applies it)


def load_samples():
    """Original l. 32, 55-68, 159-183."""
    samples = pd.read_csv(SAMPLES, index_col=0)
    clin_data = pd.read_csv(CLIN, index_col=0)
    samples.columns = samples.columns.astype(str)
    clin_data.index = clin_data.index.astype(str)
    metadata = clin_data.loc[samples.columns.values]
    metadata = metadata[['Study', 'msex', 'ceradsc', 'cogdx']]
    metadata = metadata.reindex(samples.columns)

    samples = samples.T
    samples.index = samples.index.astype(str)
    metadata.index = metadata.index.astype(str)
    samples = samples.merge(metadata[['msex', 'ceradsc', 'cogdx']], left_index=True, right_index=True)
    samples['Sex'] = samples['msex'].apply(lambda x: 'Male' if x == 1.0 else 'Female')
    ceradsc_mapping = {1: "Alzheimer's\nDisease", 2: "Alzheimer's\nDisease", 3: "No", 4: "No"}
    samples['ceradsc'] = samples['ceradsc'].map(ceradsc_mapping)
    samples['x_pos'] = range(len(samples))
    return samples


def bh_546(samples, comps):
    """Original l. 185-291: BH over all components of genes_final_RNA (546)."""
    out = {}
    phenotypes = {'Sex': ('Male', 'Female'), 'ceradsc': ("Alzheimer's\nDisease", 'No')}
    for pheno, (grp1, grp2) in phenotypes.items():
        grp1_idx = samples[samples[pheno] == grp1].index
        grp2_idx = samples[samples[pheno] == grp2].index
        prop = len(grp1_idx) / (len(grp1_idx) + len(grp2_idx))
        assert 0.2 <= prop <= 0.8, (pheno, prop)
        pvals = [mannwhitneyu(samples.loc[grp1_idx, k], samples.loc[grp2_idx, k],
                              alternative='two-sided')[1] for k in comps]
        out[pheno] = pd.Series(multipletests(pvals, alpha=0.05, method='fdr_bh')[1], index=comps)
    filtered = samples[samples['cogdx'].isin([1, 2, 4])]
    raw = []
    for k in comps:
        groups = [filtered[filtered['cogdx'] == g][k].values for g in [1, 2, 4]]
        raw.append(kruskal(*groups, nan_policy='omit')[1])
    out['cogdx'] = pd.Series(multipletests(raw, alpha=0.05, method='fdr_bh')[1], index=comps)
    return pd.DataFrame(out)


class JitterFeeder:
    """Stand-in for numpy.random.uniform during sns.stripplot: returns recovered offsets."""

    def __init__(self, arrays):
        self.arrays = [np.asarray(a, dtype=float) for a in arrays]
        self.n_calls = 0

    def __call__(self, low=0.0, high=1.0, size=None):
        assert self.n_calls < len(self.arrays), 'more jitter calls than recovered groups'
        arr = self.arrays[self.n_calls]
        assert size == len(arr), (size, len(arr))
        assert np.isclose(low, -0.1) and np.isclose(high, 0.1), (low, high)
        assert np.all(np.abs(arr) <= 0.1 + 1e-9), 'recovered jitter outside +/-0.1'
        self.n_calls += 1
        return arr.copy()


def stripplot(jitter_arrays, seed, **kw):
    if jitter_arrays is None:
        np.random.seed(seed)
        return sns.stripplot(**kw)
    feeder = JitterFeeder(jitter_arrays)
    with mock.patch.object(np.random, 'uniform', feeder):
        out = sns.stripplot(**kw)
    assert feeder.n_calls == len(feeder.arrays), (feeder.n_calls, len(feeder.arrays))
    return out


def boxplot_seaborn012(x, y, data, ax, order, palette, showfliers=False,
                       saturation=.75, width=.8, linewidth=None, whis=1.5):
    """seaborn 0.12 `_BoxPlotter` (establish_colors + draw_boxplot + restyle_boxplot +
    annotate_axes) re-implemented on top of the locked environment.

    The published PDFs were rendered by a seaborn < 0.13 (box/whisker/cap/median line width
    = rcParams['lines.linewidth'] = 2.25 in the 'talk' context, line gray quantised through
    rgb2hex, boxes at zorder 0.9 so both boxes are drawn before any whisker, whiskers with the
    style's round caps). seaborn 0.13.2 (locked) draws 1.0-pt lines, an unquantised gray,
    box+whiskers per group and butt-capped whiskers. This shim reproduces the original look;
    box statistics come from matplotlib's ax.boxplot exactly as in seaborn 0.12.
    """
    import colorsys
    vals = data[y]
    grp = data[x]
    plot_data = [np.asarray(vals[grp == g].dropna()) for g in order]
    colors = sns.color_palette(palette, len(plot_data))
    if saturation < 1:
        colors = sns.color_palette(colors, desat=saturation)
    rgb_colors = sns.color_palette(colors)
    lum = min(colorsys.rgb_to_hls(*c)[1] for c in rgb_colors) * .6
    gray = mpl.colors.rgb2hex((lum, lum, lum))
    lw = mpl.rcParams["lines.linewidth"] if linewidth is None else linewidth
    for i, box_data in enumerate(plot_data):
        if box_data.size == 0:
            continue
        artist_dict = ax.boxplot(box_data, orientation='vertical', patch_artist=True,
                                 positions=[i], widths=width, showfliers=showfliers, whis=whis)
        color = rgb_colors[i]
        for box in artist_dict["boxes"]:
            box.update(dict(facecolor=color, zorder=.9, edgecolor=gray, linewidth=lw))
        for whisk in artist_dict["whiskers"]:
            whisk.update(dict(color=gray, linewidth=lw, linestyle="-"))
        for cap in artist_dict["caps"]:
            cap.update(dict(color=gray, linewidth=lw))
        for med in artist_dict["medians"]:
            med.update(dict(color=gray, linewidth=lw))
        for fly in artist_dict["fliers"]:
            fly.update(dict(markerfacecolor=gray, marker="d", markeredgecolor=gray,
                            markersize=5))  # fliersize default 5 (not drawn: showfliers=False)
    # annotate_axes (orient 'v')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xticks(np.arange(len(plot_data)))
    ax.set_xticklabels(order)
    ax.xaxis.grid(False)
    ax.set_xlim(-.5, len(plot_data) - .5, auto=None)
    return ax


BOXPLOT_IMPL = {'v012': True}
# Original code: f"{adj_pval:.4g}". Python's 'g' drops trailing zeros (0.009740 -> '0.00974');
# '#.4g' keeps them so every label shows 4 significant figures ('0.009740'). Both give the
# identical strings for the published 546-family values (none ends in 0).
LABEL_FMT = ['.4g']


def boxplot(**kw):
    """Dispatch: seaborn-0.12-equivalent box drawing (default) or plain seaborn 0.13.2."""
    if BOXPLOT_IMPL['v012']:
        return boxplot_seaborn012(kw['x'], kw['y'], kw['data'], kw['ax'], kw['order'], kw['palette'],
                                  showfliers=kw['showfliers'])
    return sns.boxplot(**kw)


def outlier_filter(samples, comp):
    mean_value = samples[comp].mean()
    std_value = samples[comp].std()
    lower_bound = mean_value - 5 * std_value
    upper_bound = mean_value + 5 * std_value
    return samples[(samples[comp] > lower_bound) & (samples[comp] < upper_bound)]


def panel_sex(samples, k, adj_pval, jitter_arrays, seed):
    """Original l. 715-768."""
    comp = k
    feature = 'Sex'
    if k != 74:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
    filtered_samples = outlier_filter(samples, comp)
    boxplot(x=feature, y=comp, data=filtered_samples, ax=ax,
                showfliers=False, palette="Set2", order=['Male', 'Female'])
    label = f"p = {adj_pval:{LABEL_FMT[0]}}"
    pairs = [('Male', 'Female')]
    annotator = Annotator(ax, pairs, data=filtered_samples, x=feature, y=comp, order=['Male', 'Female'])
    annotator.set_pvalues([adj_pval])
    annotator.configure(text_format="simple", show_test_name=False)
    annotator.set_custom_annotations([label])
    annotator.annotate()
    stripplot(jitter_arrays, seed, x=feature, y=comp, data=filtered_samples, ax=ax,
              jitter=True, color='black', size=5)
    ax.set_xlabel(feature)
    ax.set_ylabel('loadings/score')
    plt.tight_layout()
    return fig, ax, filtered_samples, label


def panel_cogdx(samples, k, adj_pval, jitter_arrays, seed):
    """Original l. 770-830."""
    comp = k
    feature = 'cogdx'
    fig, ax = plt.subplots(figsize=(9, 6))
    filtered_samples = outlier_filter(samples, comp)
    group_order = [1, 2, 4]
    group_labels = {1: "No cognitive \n impairment", 2: "Mild cognitive \n impairment",
                    4: "Alzheimer's\nDisease"}
    filtered_samples = filtered_samples[filtered_samples[feature].isin(group_order)]
    filtered_samples[feature] = filtered_samples[feature].map(group_labels)
    filtered_samples[feature] = pd.Categorical(filtered_samples[feature],
                                               categories=[group_labels[g] for g in group_order],
                                               ordered=True)
    boxplot(x=feature, y=comp, data=filtered_samples, ax=ax, showfliers=False, palette="husl",
                order=["No cognitive \n impairment", "Mild cognitive \n impairment", "Alzheimer's\nDisease"])
    label = f"Kruskal adj p = {adj_pval:{LABEL_FMT[0]}}"
    ax.text(0.5, 0.95, label, ha='center', va='center', transform=ax.transAxes)
    stripplot(jitter_arrays, seed, x=feature, y=comp, data=filtered_samples, ax=ax,
              jitter=True, color='black', size=5)
    ax.set_xlabel(feature)
    ax.set_ylabel('loadings/score')
    plt.tight_layout()
    return fig, ax, filtered_samples, label


def panel_ceradsc(samples, k, adj_pval, jitter_arrays, seed):
    """Original l. 832-884."""
    comp = k
    feature = 'ceradsc'
    fig, ax = plt.subplots(figsize=(9, 6))
    filtered_samples = outlier_filter(samples, comp)
    group_order = ["No", "Alzheimer's\nDisease"]
    filtered_samples = filtered_samples[filtered_samples[feature].isin(group_order)]
    filtered_samples[feature] = pd.Categorical(filtered_samples[feature], categories=group_order, ordered=True)
    boxplot(x=feature, y=comp, data=filtered_samples, ax=ax,
                showfliers=False, palette="Set2", order=["No", "Alzheimer's\nDisease"])
    label = f"p = {adj_pval:{LABEL_FMT[0]}}"
    pairs = [("No", "Alzheimer's\nDisease")]
    annotator = Annotator(ax, pairs, data=filtered_samples, x=feature, y=comp,
                          order=["No", "Alzheimer's\nDisease"])
    annotator.set_pvalues([adj_pval])
    annotator.configure(text_format="simple", show_test_name=False)
    annotator.set_custom_annotations([label])
    annotator.annotate()
    stripplot(jitter_arrays, seed, x=feature, y=comp, data=filtered_samples, ax=ax,
              jitter=True, color='black', size=5)
    ax.set_xlabel(feature)
    ax.set_ylabel('loadings/score')
    plt.tight_layout()
    return fig, ax, filtered_samples, label


PANEL_FUNCS = {'Sex': panel_sex, 'cogdx': panel_cogdx, 'ceradsc': panel_ceradsc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qfamily', choices=['published', 'corrected'], required=True)
    ap.add_argument('--outdir', required=True, help='root; files go to <outdir>/<k>/<name>')
    ap.add_argument('--jitter', default=None, help='JSON {file: [[group0 offsets], ...]}')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label-format', choices=['.4g', '#.4g'], default='.4g')
    ap.add_argument('--boxplot', choices=['v012', 'seaborn013'], default='v012',
                    help='v012: seaborn-0.12 box styling (matches the published PDFs); '
                         'seaborn013: plain sns.boxplot of the locked seaborn 0.13.2')
    ap.add_argument('--geometry-out', default=None,
                    help='write axes geometry (PDF points) per panel to this JSON')
    args = ap.parse_args()

    BOXPLOT_IMPL['v012'] = args.boxplot == 'v012'
    LABEL_FMT[0] = args.label_format
    samples = load_samples()
    comps = pd.read_csv(GENES_RNA, index_col=0, usecols=[0]).index.values
    assert len(comps) == 546 and list(comps) == list(range(1, 547)), comps[:5]

    q546 = bh_546(samples, comps)
    q547 = pd.read_csv(BH547, index_col=0)
    jit = json.load(open(args.jitter)) if args.jitter else {}

    geometry = {}
    for k, feature, fname in PANELS:
        q = float(q546.loc[k, feature]) if args.qfamily == 'published' else float(q547.loc[k, feature])
        print(f'C{k} {feature}: q546={q546.loc[k, feature]:.6g} q547={q547.loc[k, feature]:.6g} -> '
              f'using {args.qfamily} q={q:.4g}')
        fig, ax, fs, label = PANEL_FUNCS[feature](samples, k, q, jit.get(fname), args.seed)
        out = os.path.join(args.outdir, str(k), fname)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, metadata={'CreationDate': None})
        # PDF backend renders at 72 dpi; record the exact data<->PDF-point transform it used.
        fig.set_dpi(72)
        inv = ax.transData.inverted()
        x0, y0 = ax.transData.transform((0.0, 0.0))
        x1, y1 = ax.transData.transform((1.0, 1.0))
        geometry[fname] = {
            'label': label, 'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim()),
            'pt_per_x': x1 - x0, 'pt_x_at_0': x0, 'pt_per_y': y1 - y0, 'pt_y_at_0': y0,
            'n_by_group': [int(len(c.get_offsets())) for c in ax.collections
                           if type(c).__name__ == 'PathCollection'],
        }
        print(f'   wrote {out}  label={label!r}  ylim={ax.get_ylim()}')
        plt.close(fig)
    if args.geometry_out:
        json.dump(geometry, open(args.geometry_out, 'w'), indent=1)


if __name__ == '__main__':
    main()
