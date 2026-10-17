"""Genotype-free duplicate scan of snRNA pseudobulk: for each donor, its most-correlated other donor per cell type
(gene-centred log2 CPM, genes with median CPM >= 10, top 10 PCs removed). Pairs that are mutual nearest neighbours in
>= 4 of 6 cell types are flagged as possible duplicate source material. Aggregates/pseudonymous labels only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from qtl_common import residualize
ROS = cfg.ROS; SDA = f"{ROS}/sda"; P = cfg.out_dir(cfg.PRIVATE)
samps = [int(l) for l in open(f"{SDA}/md_samples.txt") if l.strip()]
key = pd.read_csv(f"{P}/donor_key.csv"); lab = dict(zip(key.projid, key.label))
R = pd.read_csv(f"{SDA}/rna.txt", sep=" ", header=None, dtype=np.float32).values
NN = np.zeros((276, 6), int); TOP = np.zeros((276, 6))
for c in range(6):
    X = R[c * 276:(c + 1) * 276]; keep = np.median(X, 0) >= 10
    Z = residualize(np.log2(X[:, keep] + 1).astype(np.float64))
    C = np.corrcoef(Z); np.fill_diagonal(C, -1)
    NN[:, c] = C.argmax(1); TOP[:, c] = C.max(1)
cnt = {}
for i in range(276):
    for c in range(6):
        j = NN[i, c]
        if NN[j, c] == i: cnt[tuple(sorted((i, j)))] = cnt.get(tuple(sorted((i, j))), 0) + 1
pairs = {k: v // 2 for k, v in cnt.items()}   # each mutual pair counted twice per cell type
hist = pd.Series(list(pairs.values())).value_counts().sort_index().to_dict()
print("mutual-nearest-neighbour pairs by number of cell types (of 6):", hist)
rna = pd.read_csv(f"{P}/10_rna_identity.csv", index_col=0)
for (i, j), v in sorted(pairs.items(), key=lambda kv: -kv[1]):
    if v < 4: continue
    a, b = samps[i], samps[j]
    print(f"pair in {v}/6 cell types: labels {lab.get(a, 'unlabelled')} & {lab.get(b, 'unlabelled')}; residual r per type {np.round([np.corrcoef(0,0)[0,0]] if False else TOP[i], 2).tolist()}; "
          f"genotype identity: {['own' if rna.loc[p,'own_rank']==1 else ('not own' if pd.notna(rna.loc[p,'own_rank']) else 'not genotyped') for p in (a, b)]}")
print("typical nearest-neighbour residual r (median over donors and types):", round(float(np.median(TOP)), 2))
for lb in ["RNA-2", "RNA-1", "RNAMIX-1"]:
    pj = key[key.label == lb].projid.iloc[0]; i = samps.index(pj)
    print(f"{lb}: nearest-neighbour residual r per type {np.round(TOP[i], 2).tolist()}; same neighbour across types: {len(set(NN[i]))==1}; percentile of its NN r among all donors (mean over types) {np.mean([(TOP[:, c] < TOP[i, c]).mean() for c in range(6)]):.2f}")
