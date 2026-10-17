"""
Reviewer 2 (npj Aging, Jul 2026): Figure 4c outlier check.

R2 wrote: "In Figure 4c, the vast majority of the samples appear to cluster close
to zero, while the reported correlation seems to be driven largely by approximately
five outlier samples... examine whether these samples exhibit abnormalities in the
sex chromosomes. The reported correlation should also be reassessed after excluding
these outliers."

PRE-REGISTERED QUESTIONS (locked before inspecting output):
 Q1. What are the points in Fig 4c: samples or genes?
 Q2. Identify the extreme points. Are they sex-chromosome genes?
 Q3. Recompute r after excluding them, at several definitions of "exclude":
     (a) drop top-5 by |ROSMAP loading|
     (b) drop ALL chrX/chrY genes  <-- the decisive test
     (c) leave-one-out jackknife (max |delta r| from dropping any single gene)
 Q4. Rank correlation (Spearman) as an outlier-insensitive comparator.

Reproduces panel c of figs/comp74_sex.pdf from
old/g_ros_manu/synth1b_croos_clin_manuscript.py (lines 32-145), which builds
df_plot_d indexed by GENE (r_pip_genes.columns), not by sample.

Read-only on its inputs (SEA-AD and ROSMAP consensus loadings). Writes only to $G_ROS_RESULTS_DIR/revision_r2/.
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import pandas as pd, numpy as np
from scipy import stats
from pathlib import Path

SEA  = Path(cfg.SEAAD)                          # SEA-AD DLPFC: gene_info.csv, sda/genes_final.csv
SDA  = Path(cfg.SDA)                            # ROSMAP version-1 consensus loadings
OUT  = Path(cfg.out_dir(cfg.WORK, "revision_r2"))
K    = 74

# --- SEA-AD (dlpfc) ---
d_feature_names = pd.read_csv(SEA/'gene_info.csv', index_col=0)
d_genes = pd.read_csv(SEA/'sda'/'genes_final.csv', index_col=0)
d_genes.rename(columns=d_feature_names['feature_name'].to_dict(), inplace=True)

# --- ROSMAP ---
r_genes = pd.read_csv(SDA/'genes_final_RNA.csv', index_col=0)
r_pip   = pd.read_csv(SDA/'pip_final_RNA.csv',   index_col=0)

common = sorted(set(d_genes.columns) & set(r_genes.columns))
d_f, r_f, p_f = d_genes[common], r_genes[common], r_pip[common]

# PIP>0.5 gene set for component K (exactly as in the manuscript script)
sel = p_f.T[p_f.loc[K] > 0.5].index.values
r_pip_genes, d_pip_genes = r_f[sel], d_f[sel]

# partner SEA-AD component = max |corr| (as in manuscript script)
corrs = d_pip_genes.apply(lambda row: row.corr(r_pip_genes.loc[K]), axis=1)
partner = corrs.abs().idxmax()

df = pd.DataFrame({'ROSMAP': r_pip_genes.loc[K].values,
                   'SEA_AD': d_pip_genes.loc[partner].values}, index=sel)

# annotate chromosome
chrom_col = next((c for c in ('chromosome','chrom','chr','seqname','seqnames')
                  if c in d_feature_names.columns), None)
if chrom_col:
    name2chr = dict(zip(d_feature_names['feature_name'], d_feature_names[chrom_col].astype(str)))
    df['chrom'] = [name2chr.get(g, 'NA') for g in df.index]
else:
    df['chrom'] = 'NA'
df['sex_chr'] = df['chrom'].str.replace('chr','',regex=False).isin(['X','Y'])

def rep(tag, d):
    if len(d) < 3: return dict(subset=tag, n=len(d), pearson_r=np.nan, p=np.nan, spearman_r=np.nan)
    pr, pp = stats.pearsonr(d.ROSMAP, d.SEA_AD)
    sr, _  = stats.spearmanr(d.ROSMAP, d.SEA_AD)
    return dict(subset=tag, n=len(d), pearson_r=round(pr,4), p=f"{pp:.3g}", spearman_r=round(sr,4))

rows = [rep('ALL (as published, Fig 4c)', df)]

# (a) drop top-5 by |ROSMAP|
top5 = df.reindex(df.ROSMAP.abs().sort_values(ascending=False).index)[:5]
rows.append(rep('drop top-5 by |ROSMAP loading|', df.drop(top5.index)))

# (b) drop all sex-chromosome genes  <-- decisive
rows.append(rep('drop ALL chrX/chrY genes', df[~df.sex_chr]))
rows.append(rep('ONLY chrX/chrY genes',      df[ df.sex_chr]))

# (c) jackknife
jk = []
for g in df.index:
    d2 = df.drop(g)
    if len(d2) > 2: jk.append((g, stats.pearsonr(d2.ROSMAP, d2.SEA_AD)[0]))
jk = pd.DataFrame(jk, columns=['gene','r_without']).set_index('gene')
r_all = stats.pearsonr(df.ROSMAP, df.SEA_AD)[0]
jk['delta_r'] = jk.r_without - r_all
jk = jk.reindex(jk.delta_r.abs().sort_values(ascending=False).index)

res = pd.DataFrame(rows)
print(f"\nComponent {K}  |  SEA-AD partner component: {partner}")
print(f"Points in Fig 4c: {len(df)}  -> these are GENES (PIP>0.5), not samples.")
print(f"Sex-chromosome genes among them: {int(df.sex_chr.sum())}\n")
print(res.to_string(index=False))
print("\n--- Top 8 points by |ROSMAP loading| (the 'outliers' R2 saw) ---")
print(top5.join(df[['chrom']], rsuffix='_x')[['ROSMAP','SEA_AD','chrom','sex_chr']].to_string())
print(df.reindex(df.ROSMAP.abs().sort_values(ascending=False).index)[:8][['ROSMAP','SEA_AD','chrom','sex_chr']].to_string())
print("\n--- Jackknife: single genes with largest influence on r ---")
print(jk.head(8).round(4).to_string())

df.to_csv(OUT/'fig4c_points.csv')
res.to_csv(OUT/'fig4c_sensitivity.csv', index=False)
jk.to_csv(OUT/'fig4c_jackknife.csv')
