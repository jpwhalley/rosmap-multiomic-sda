"""Genotype-based identity of every snRNA-seq pseudobulk profile (SDA input rna.txt), via cis/trans-agnostic eQTLs
discovered with 5-fold cross-fitting among genotyped SDA donors (a donor is never in its own discovery fold).
For each SDA donor: identity score of its RNA against all 1708 genotyped people; own rank and z.
Per-donor values -> $G_ROS_PRIVATE_DIR; aggregates and pseudonymous labels printed."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, time, numpy as np, pandas as pd
from qtl_common import *
P = cfg.out_dir(cfg.PRIVATE); ROS = cfg.ROS; SDA = f"{ROS}/sda"
t0 = time.time()
G, gproj, bim = load_geno(); af = np.nanmean(np.where(G < 0, np.nan, G.astype(np.float32)), 0) / 2
print("genotypes", G.shape, f"{time.time()-t0:.0f}s")
samps = np.array([int(l) for l in open(f"{SDA}/md_samples.txt") if l.strip()])
CTS = [l.strip() for l in open(f"{SDA}/cell_types.txt") if l.strip()]
genes = np.array([l.strip() for l in open(f"{SDA}/genes.txt")])
R = pd.read_csv(f"{SDA}/rna.txt", sep=" ", header=None, dtype=np.float32).values
blocks = []
for c, ct in enumerate(CTS):
    E = np.log2(R[c * 276:(c + 1) * 276] + 1)
    keep = np.median(R[c * 276:(c + 1) * 276], 0) >= 10
    blocks.append(residualize(E[:, keep].astype(np.float64)))
    print(ct, "genes kept", int(keep.sum()))
Z = np.hstack(blocks).astype(np.float32); print("phenotypes", Z.shape, f"{time.time()-t0:.0f}s")
row = {p: i for i, p in enumerate(gproj)}
gi = np.array([row.get(p, -1) for p in samps]); typed = np.where(gi >= 0)[0]
print("SDA donors genotyped:", len(typed))
rng = np.random.default_rng(1); fold = np.full(276, -1); fold[typed] = rng.permutation(np.arange(len(typed)) % 5)
res = pd.DataFrame(index=samps, columns=["own_score", "own_rank", "own_z", "best_other_score", "best_other_z", "best_other_proj", "n_pairs"], dtype=float)
for k in range(5):
    d = typed[fold[typed] != k]
    Gd = std_geno(G[gi[d]])
    pairs, _ = discover(Z[d], Gd, 0.5, bim)
    print(f"fold {k}: discovery n {len(d)}; eQTL pairs kept {len(pairs)} (|r|>=0.5, SNPs >=2 Mb apart) {time.time()-t0:.0f}s", flush=True)
    test = [i for i in range(276) if fold[i] == k or (fold[i] == -1 and k == 0)]
    for i in test:
        sc = identity_scores(Z[i], pairs, G, af)
        mu, sd = np.nanmean(sc), np.nanstd(sc)
        if gi[i] >= 0:
            own = sc[gi[i]]; oth = np.delete(sc, gi[i]); othp = np.delete(gproj, gi[i])
            res.loc[samps[i], ["own_score", "own_rank", "own_z"]] = [own, 1 + np.sum(oth > own), (own - mu) / sd]
        else:
            oth, othp = sc, gproj
        b = int(np.nanargmax(oth))
        res.loc[samps[i], ["best_other_score", "best_other_z", "best_other_proj", "n_pairs"]] = [oth[b], (oth[b] - mu) / sd, othp[b], len(pairs)]
res.to_csv(f"{P}/10_rna_identity.csv")
t = res[res.own_rank.notna()]
print(f"RNA identity, genotyped SDA donors {len(t)}: own genotype ranks 1st of 1708 for {int((t.own_rank == 1).sum())}; own z median {t.own_z.median():.1f} (min {t.own_z.min():.1f}); best-other z median {t.best_other_z.median():.1f} (max {t.best_other_z.max():.1f})")
bad = t[(t.own_rank > 1) | (t.own_z < 5)]
print(f"  donors whose RNA does not match their own genotype best (rank>1 or own z<5): {len(bad)}")
for pj, r in bad.iterrows():
    print(f"   rank {int(r.own_rank)}, own z {r.own_z:.1f}, best-other z {r.best_other_z:.1f}; best-matching person is an SDA donor: {int(r.best_other_proj) in set(samps)}")
nt = res[res.own_rank.isna()]
print(f"non-genotyped SDA donors {len(nt)}: best-matching genotyped person z median {nt.best_other_z.median():.1f}, max {nt.best_other_z.max():.1f}; with z>=8: {int((nt.best_other_z >= 8).sum())}")
key = pd.read_csv(f"{P}/donor_key.csv")
for pj, lb in key.values:
    r = res.loc[pj]
    if pd.notna(r.own_rank):
        print(f"{lb}: RNA vs own genotype: rank {int(r.own_rank)} of 1708, own z {r.own_z:.1f}; best other z {r.best_other_z:.1f} (best other is SDA donor: {int(r.best_other_proj) in set(samps)}); pairs {int(r.n_pairs)}")
    else:
        bo = int(r.best_other_proj)
        print(f"{lb}: not genotyped; best-matching genotyped person z {r.best_other_z:.1f} (is an SDA donor: {bo in set(samps)}; is another flagged donor: {bo in set(key.projid)})")
print(f"done {time.time()-t0:.0f}s")
