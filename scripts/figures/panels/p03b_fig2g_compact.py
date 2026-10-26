"""Fig 2g (10-16): the S4b content (cg26921482 methylation; H3K9ac chrX and autosomal C74 peak aggregates against the
C74 score, coloured by recorded sex) redrawn to fit the slot beside Fig 2f at the same type scale as the reused v22 panels
(74_meth.pdf is 864 x 432 pt and is placed at about 0.26). Same data, exclusions and aggregation as p03_c74.py (S4b)."""
import numpy as np, pandas as pd
from vp_common import *

LOG = []
K = 74; a = score(K); fem = female(); sexlab = np.where(fem, "female", "male")
exC = excl_mask("chip"); nomask = np.zeros(276, bool); assert exC.sum() == 4

def sex_scatter(ax, y, excl, ylab):
    ok = ~np.isnan(y) & ~excl
    for sx in ("male", "female"):
        m = ok & (sexlab == sx)
        ax.scatter(a[m], y[m], s=30, color=SEXCOL[sx], alpha=0.8, linewidths=0)
    ax.set_ylabel(ylab, fontsize=17); ax.tick_params(labelsize=14); sns.despine(ax=ax)
    return int(ok.sum())

MI = mod_input("meth"); LM = comp_loadings(K, "meth"); cpgs = list(LM.index)
LMp = LM[LM.P > 0.5]; top = LMp.X.abs().idxmax(); assert top == "cg26921482"
ym = np.asarray(MI[:, cpgs.index(top)], float)
CI = np.asarray(mod_input("chip"), float); LC = comp_loadings(K, "chip")
H = pd.read_csv(f"{REFS}/H3K9acDomains.csv", sep="\t", index_col=0).loc[list(LC.index)]
chrc = H.Chr.astype(str).values
selX = (LC.P.values > 0.5) & (chrc == "X"); selA = (LC.P.values > 0.5) & ~np.isin(chrc, ["X", "Y"])
aggX, nX = zagg(CI[:, selX], LC.X.values[selX]); aggA, nA = zagg(CI[:, selA], LC.X.values[selA])
assert (nX, nA) == (537, 120)
fig, axs = plt.subplots(1, 3, figsize=(11, 6))
n1 = sex_scatter(axs[0], ym, nomask, "cg26921482 (AMDHD2)\nmethylation (β)")
n2 = sex_scatter(axs[1], aggX, exC, f"H3K9ac, {nX} chrX peaks\nweighted mean z")
n3 = sex_scatter(axs[2], aggA, exC, f"H3K9ac, {nA} autosomal peaks\nweighted mean z")
axs[1].set_xlabel("Component 74 loadings/score", fontsize=17)
sexlegend(axs[0], loc="lower left", fontsize=13)
fig.tight_layout(w_pad=1.2); save(fig, "Fig2", "Fig2g_C74_cpg_H3K9ac_vs_score_compact", LOG)
LOG.append(f"Fig2g compact: n {n1} (CpG), {n2} (chrX), {n3} (autosomal)")
print("\n".join(LOG))
