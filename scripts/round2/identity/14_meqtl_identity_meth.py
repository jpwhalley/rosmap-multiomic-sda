"""Genotype-based identity of every 450K sample via meQTLs (imputed autosomal matrix; 20,000 most variable CpGs),
5-fold cross-fitted over samples whose recorded person is genotyped. Per-sample values -> $G_ROS_PRIVATE_DIR; aggregates printed."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, time, numpy as np, pandas as pd
from qtl_common import *
P = cfg.out_dir(cfg.PRIVATE); ROS = cfg.ROS; O = f"{ROS}/oth"; t0 = time.time()
keep = None
hdr = pd.read_csv(f"{ROS}/meth/ROSMAP_arrayMethylation_imputed.tsv.gz", sep="\t", index_col=0, nrows=0).columns
for ch in pd.read_csv(f"{ROS}/meth/ROSMAP_arrayMethylation_imputed.tsv.gz", sep="\t", index_col=0, chunksize=40000, dtype={c: np.float32 for c in hdr}):
    sd = ch.std(axis=1); ch = ch.assign(_sd=sd)
    keep = ch if keep is None else pd.concat([keep, ch]).nlargest(20000, "_sd")
    keep = keep.nlargest(20000, "_sd")
cols = keep.columns[:-1]; E = keep[cols].values.T.astype(np.float64)
print("CpGs kept", E.shape[1], "samples", E.shape[0], f"{time.time()-t0:.0f}s", flush=True)
bio = pd.read_csv(f"{O}/meta/ROSMAP_biospecimen_metadata.csv", low_memory=False); s2i = bio.drop_duplicates("specimenID").set_index("specimenID").individualID
clin = pd.read_csv(f"{O}/meta/ROSMAP_clinical.csv"); i2p = dict(zip(clin.individualID, clin.projid))
proj = [i2p.get(s2i.get(c)) for c in cols]
samps = set(int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip())
Z = residualize(E).astype(np.float32)
G, gproj, bim = load_geno(); af = np.nanmean(np.where(G < 0, np.nan, G.astype(np.float32)), 0) / 2
row = {p: i for i, p in enumerate(gproj)}
gi = np.array([row.get(p, -1) if p is not None and not pd.isna(p) else -1 for p in proj]); typed = np.where(gi >= 0)[0]
print("samples with genotyped recorded person", len(typed), flush=True)
rng = np.random.default_rng(3); people = np.array(sorted(set(gi[typed]))); pf = dict(zip(people, rng.permutation(np.arange(len(people)) % 5)))
fold = np.array([pf.get(g, -1) for g in gi])
res = pd.DataFrame(index=cols, columns=["projid", "own_score", "own_rank", "own_z", "best_other_score", "best_other_z", "best_other_proj", "n_pairs"], dtype=float); res["projid"] = proj
for k in range(5):
    d = typed[fold[typed] != k]; _, first = np.unique(gi[d], return_index=True); d = d[first]
    pairs, _ = discover(Z[d], std_geno(G[gi[d]]), 0.45, bim)
    print(f"fold {k}: discovery n {len(d)}; meQTL pairs kept {len(pairs)} {time.time()-t0:.0f}s", flush=True)
    for i in [i for i in range(len(cols)) if fold[i] == k or (fold[i] == -1 and k == 0)]:
        sc = identity_scores(Z[i], pairs, G, af); mu, s_ = np.nanmean(sc), np.nanstd(sc)
        if gi[i] >= 0:
            own = sc[gi[i]]; oth = np.delete(sc, gi[i]); othp = np.delete(gproj, gi[i])
            res.loc[cols[i], ["own_score", "own_rank", "own_z"]] = [own, 1 + np.sum(oth > own), (own - mu) / s_]
        else: oth, othp = sc, gproj
        b = int(np.nanargmax(oth)); res.loc[cols[i], ["best_other_score", "best_other_z", "best_other_proj", "n_pairs"]] = [oth[b], (oth[b] - mu) / s_, othp[b], len(pairs)]
res.to_csv(f"{P}/14_meth_identity.csv")
t = res[res.own_rank.notna()]
print(f"meth: samples with genotyped recorded person {len(t)}; own rank 1: {int((t.own_rank == 1).sum())}; own z median {t.own_z.median():.1f}; best-other z median {t.best_other_z.median():.1f} (99th pct {t.best_other_z.quantile(0.99):.1f})")
bad = t[t.own_rank > 1]
print(f"  mismatches: {len(bad)}; recorded SDA donor: {int(bad.projid.isin(samps).sum())}")
for s, r in bad.iterrows():
    print(f"   rank {int(r.own_rank)}, own z {r.own_z:.1f}, best other z {r.best_other_z:.1f}; recorded SDA: {r.projid in samps}; best match SDA: {int(r.best_other_proj) in samps}")
ts = t[t.projid.isin(samps)]; print(f"  SDA donors with genotype: {len(ts)}; own rank 1: {int((ts.own_rank == 1).sum())}")
nt = res[res.own_rank.isna()]; print(f"  samples without genotyped recorded person {len(nt)}: best-match z >= 8: {int((nt.best_other_z >= 8).sum())} (recorded SDA: {int((nt.projid.isin(samps) & (nt.best_other_z >= 8)).sum())})")
key = pd.read_csv(f"{P}/donor_key.csv")
for pj, lb in key.values:
    for s, r in res[res.projid == pj].iterrows():
        print(f"{lb}: 450K vs own genotype: " + (f"rank {int(r.own_rank)}, own z {r.own_z:.1f}; best other z {r.best_other_z:.1f}" if pd.notna(r.own_rank) else f"not genotyped; best match z {r.best_other_z:.1f}"))
print(f"done {time.time()-t0:.0f}s")
