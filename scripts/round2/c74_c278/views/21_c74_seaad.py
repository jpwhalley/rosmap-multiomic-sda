"""Task 2: C74 vs SEA-AD partner 106 (corrected 141-component SEA-AD catalogue = deposited genes_final minus retired
IDs 9 and 128; partner and gene set as in Data S1 / 12_seaad_validation.py: sorted common genes with ROSMAP C74
PIP > 0.5). Concordance scoped by chromosome class (GRCh38.100 gene table): sex-chromosome genes (chrX + chrY) vs
autosomal genes, with Pearson, Spearman (bootstrap CIs over genes), and sign concordance with an exact binomial test.
The pair's overall orientation is negative (r -0.98; component signs are arbitrary, only products are interpretable),
so a 'concordant' gene is one with opposite signs in the two loading vectors.
Outputs: out/21_c74_seaad_concordance.csv (gene-level aggregates only; no donor data), panels/C74_h_seaad_scoped.*"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats as ss
from cc_common import *

mpl.rcParams["pdf.fonttype"] = 42
sns.set(style="white", context="talk")
mpl.rcParams["font.family"] = "sans-serif"; mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
fn = pd.read_csv(f"{SEA}/gene_info.csv", index_col=0)["feature_name"]
sg = pd.read_csv(f"{SEA}/sda/genes_final.csv", index_col=0); sg.columns = [fn[c] for c in sg.columns]; sg.index = sg.index.astype(int)
assert sg.columns.is_unique and len(sg) == 143
sg = sg.drop(index=[9, 128])                                   # corrected 141-component catalogue
LR = comp_loadings(74, "RNA"); x = LR.X; p = LR.P
genes = [g for g in sorted(set(sg.columns) & set(LR.index)) if p[g] > 0.5]
# re-derive the partner on the corrected catalogue (must be 106, r -0.982 as in Data S1)
r_all = sg[genes].apply(lambda row: row.corr(x[genes]), axis=1)
j = r_all.idxmax() if abs(r_all.max()) >= abs(r_all.min()) else r_all.idxmin()
print(f"C74 PIP genes in SEA-AD: {len(genes)}; partner {j} r {r_all[j]:+.4f} (runner-up |r| {r_all.abs().sort_values().iloc[-2]:.3f})")
assert j == 106
y = sg.loc[106, genes].astype(float); xx = x[genes].astype(float)
gc = gene_chrom(); ch = pd.Series({g: gc.get(g, "NA") for g in genes})
cls = ch.map(lambda c: c if c in ("X", "Y", "NA") else "auto")
orient = np.sign(ss.pearsonr(xx, y)[0])
rng = np.random.default_rng(20261013)


def conc(sel, label):
    a, b = xx[sel].values, y[sel].values; n = len(a)
    pr, pp = ss.pearsonr(a, b); sr, sp = ss.spearmanr(a, b)
    bs_p, bs_s = [], []
    for _ in range(5000):
        i = rng.integers(0, n, n)
        if np.std(a[i]) > 0 and np.std(b[i]) > 0:
            bs_p.append(np.corrcoef(a[i], b[i])[0, 1]); bs_s.append(ss.spearmanr(a[i], b[i])[0])
    k = int((np.sign(a) * np.sign(b) == orient).sum())
    bt = ss.binomtest(k, n, 0.5)
    return dict(gene_class=label, n_genes=n, pearson_r=pr, pearson_lo=np.percentile(bs_p, 2.5), pearson_hi=np.percentile(bs_p, 97.5),
                pearson_p=pp, spearman_rho=sr, spearman_lo=np.percentile(bs_s, 2.5), spearman_hi=np.percentile(bs_s, 97.5),
                spearman_p=sp, sign_concordant=k, sign_concordant_frac=k / n, binom_p_two_sided=bt.pvalue,
                binom_ci_lo=bt.proportion_ci().low, binom_ci_hi=bt.proportion_ci().high,
                share_of_C74_sq_loading=float((xx[sel] ** 2).sum() / (xx ** 2).sum()))


rows = [conc(cls.index, "all (Data S1 gene set)"), conc(cls.index[cls.isin(["X", "Y"])], "sex chromosomes (X+Y)"),
        conc(cls.index[cls == "X"], "chrX"), conc(cls.index[cls == "Y"], "chrY"), conc(cls.index[cls == "auto"], "autosomal")]
auto = cls.index[cls == "auto"]
o = xx[auto].abs().sort_values(ascending=False).index
for d in (5, 10):
    rr = conc(o[d:], f"autosomal minus top-{d} |C74 loading|"); rows.append(rr)
# autosomal genes that are also PIP > 0.5 in SEA-AD 106
spip = pd.read_csv(f"{SEA}/sda/pip_final.csv", index_col=0); spip.columns = [fn[c] for c in spip.columns]; spip.index = spip.index.astype(int)
both = [g for g in auto if spip.loc[106, g] > 0.5]
rows.append(conc(pd.Index(both), "autosomal, PIP > 0.5 in both"))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/21_c74_seaad_concordance.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
print(T.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("unannotated genes excluded from the class rows:", list(ch.index[ch == "NA"]))

# ---- panel ----
fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
col = {"X": "#7a0177", "Y": "#41ab5d", "auto": "#636363", "NA": "#bdbdbd"}
for ax, sel, ttl in ((axs[0], cls.index[cls.isin(["X", "Y"])], "Sex-chromosome genes"), (axs[1], auto, "Autosomal genes")):
    for c in ("X", "Y", "auto"):
        s = [g for g in sel if cls[g] == c]
        if s: ax.scatter(xx[s], y[s], s=24, color=col[c], alpha=0.8, linewidths=0, label={"X": "chrX", "Y": "chrY", "auto": "autosomal"}[c] + f" (n = {len(s)})")
    r = T.set_index("gene_class").loc["sex chromosomes (X+Y)" if ttl.startswith("Sex") else "autosomal"]
    ax.axhline(0, color="grey", lw=0.6, ls=":"); ax.axvline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("ROSMAP Component 74 gene loadings", fontsize=12); ax.set_ylabel("SEA-AD Component 106 gene loadings", fontsize=12)
    ax.set_title(ttl, fontsize=13, loc="left"); ax.tick_params(labelsize=10); sns.despine(ax=ax)
    ax.text(0.98, 0.98, f"n = {int(r.n_genes)} genes\nPearson r = {r.pearson_r:+.2f} [{r.pearson_lo:+.2f}, {r.pearson_hi:+.2f}]\n"
            f"Spearman ρ = {r.spearman_rho:+.2f} [{r.spearman_lo:+.2f}, {r.spearman_hi:+.2f}]\n"
            f"sign-concordant {int(r.sign_concordant)}/{int(r.n_genes)} (binomial p = {fmt_p(r.binom_p_two_sided)})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
fig.suptitle("Component 74 vs SEA-AD 106: concordance by chromosome class (138 C74 PIP > 0.5 genes measured in SEA-AD; 1 unannotated gene omitted)", fontsize=11.5, y=1.02)
fig.tight_layout(); fig.savefig(f"{PAN}/C74_h_seaad_scoped.pdf", bbox_inches="tight"); fig.savefig(f"{PAN}/C74_h_seaad_scoped.png", dpi=200, bbox_inches="tight")
