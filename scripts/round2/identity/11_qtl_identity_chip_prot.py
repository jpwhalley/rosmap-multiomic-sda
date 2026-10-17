"""Genotype-based identity of every H3K9ac sample (full ReadCounts, ros1b ID chain) and every TMT column
(DLPFC-only mapping), via QTLs discovered with 5-fold cross-fitting (a sample never in its own discovery fold).
Usage: python 11_qtl_identity_chip_prot.py chip|prot. Per-sample values -> $G_ROS_PRIVATE_DIR; aggregates printed."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, time, numpy as np, pandas as pd
from qtl_common import *
P = cfg.out_dir(cfg.PRIVATE); ROS = cfg.ROS; O = f"{ROS}/oth"
mod = sys.argv[1]; t0 = time.time()
clin = pd.read_csv(f"{O}/meta/ROSMAP_clinical.csv"); i2p = dict(zip(clin.individualID, clin.projid))
samps = set(int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip())
if mod == "chip":
    rc = pd.read_csv(f"{O}/chip/ReadCounts.csv", sep="\t", index_col=0)
    dom = pd.read_csv(f"{O}/chip/H3K9acDomains.csv", sep="\t", index_col=0); dom["Chr"] = dom.Chr.astype(str)
    rc = rc.loc[rc.index.intersection(dom.index)]; rc = rc[~dom.loc[rc.index, "Chr"].isin(["X", "Y"])]
    bm = pd.read_csv(f"{O}/chip/bam_id_mapping.csv"); rg2bam = dict(zip(bm.RG_ID.astype(str), bm.BAM_File))
    sid = [str(c) for c in rc.columns]; proj = [i2p.get(rg2bam.get(c, rg2bam.get(c.lstrip("0")))) for c in sid]
    cpm = rc.values / rc.values.sum(0) * 1e6
    keep = np.median(rc.values, 1) >= 20
    E = np.log2(cpm[keep] + 1).T; rthr = 0.4
else:
    T = f"{ROS}/prot/tmt"
    tmt = pd.read_csv(f"{T}/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv", index_col=0)
    meta = pd.read_csv(f"{T}/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv", index_col=0)
    dl = meta[meta.index.str.startswith("ROSMAP.DLPFC") & (meta.isAssayControl.astype(str).str.upper() == "FALSE")]
    m = dict(zip(dl.index.str.split(".").str[-3] + "." + dl.index.str.split(".").str[-2], dl.index.str.split(".").str[-1]))
    sid = list(tmt.columns); proj = [i2p.get(m.get(c)) for c in sid]
    V = tmt.values; keep = np.isnan(V).mean(1) <= 0.05
    E = V[keep].T; E = np.where(np.isnan(E), np.nanmean(E, 0), E); rthr = 0.45
Z = residualize(E.astype(np.float64)).astype(np.float32)
print(mod, "samples", Z.shape[0], "features", Z.shape[1], "| samples mapped to a projid", sum(p is not None and not pd.isna(p) for p in proj), f"{time.time()-t0:.0f}s", flush=True)
G, gproj, bim = load_geno(); af = np.nanmean(np.where(G < 0, np.nan, G.astype(np.float32)), 0) / 2
row = {p: i for i, p in enumerate(gproj)}
gi = np.array([row.get(p, -1) if p is not None and not pd.isna(p) else -1 for p in proj]); typed = np.where(gi >= 0)[0]
# a person with >1 sample: keep all for testing, but only one per person in any discovery set
print("samples whose recorded person is genotyped:", len(typed), "| distinct people", len(set(gi[typed])), flush=True)
rng = np.random.default_rng(2); people = np.array(sorted(set(gi[typed]))); pf = dict(zip(people, rng.permutation(np.arange(len(people)) % 5)))
fold = np.array([pf.get(g, -1) for g in gi])
res = pd.DataFrame(index=sid, columns=["projid", "own_score", "own_rank", "own_z", "best_other_score", "best_other_z", "best_other_proj", "n_pairs"], dtype=float)
res["projid"] = proj
for k in range(5):
    d = typed[fold[typed] != k]; _, first = np.unique(gi[d], return_index=True); d = d[first]
    pairs, _ = discover(Z[d], std_geno(G[gi[d]]), rthr, bim)
    print(f"fold {k}: discovery n {len(d)}; QTL pairs kept {len(pairs)} (|r|>={rthr}) {time.time()-t0:.0f}s", flush=True)
    for i in [i for i in range(len(sid)) if fold[i] == k or (fold[i] == -1 and k == 0)]:
        sc = identity_scores(Z[i], pairs, G, af); mu, sd = np.nanmean(sc), np.nanstd(sc)
        if gi[i] >= 0:
            own = sc[gi[i]]; oth = np.delete(sc, gi[i]); othp = np.delete(gproj, gi[i])
            res.loc[sid[i], ["own_score", "own_rank", "own_z"]] = [own, 1 + np.sum(oth > own), (own - mu) / sd]
        else: oth, othp = sc, gproj
        b = int(np.nanargmax(oth))
        res.loc[sid[i], ["best_other_score", "best_other_z", "best_other_proj", "n_pairs"]] = [oth[b], (oth[b] - mu) / sd, othp[b], len(pairs)]
res.to_csv(f"{P}/11_{mod}_identity.csv")
t = res[res.own_rank.notna()]
print(f"{mod}: samples with genotyped recorded person {len(t)}; own genotype ranks 1st for {int((t.own_rank == 1).sum())}; own z median {t.own_z.median():.1f}; best-other z median {t.best_other_z.median():.1f} (99th pct {t.best_other_z.quantile(0.99):.1f})")
bad = t[(t.own_rank > 1)]
print(f"  samples where someone else matches better than the recorded person: {len(bad)} (SDA donors among recorded: {int(bad.projid.isin(samps).sum())})")
for s_, r in bad.iterrows():
    print(f"   rank {int(r.own_rank)}, own z {r.own_z:.1f}, best-other z {r.best_other_z:.1f}; recorded person is SDA donor: {r.projid in samps}; best match is SDA donor: {int(r.best_other_proj) in samps}")
ts = t[t.projid.isin(samps)]
print(f"  SDA donors: {ts.projid.nunique()} with genotype; own rank 1: {int((ts.own_rank == 1).sum())}")
nt = res[res.own_rank.isna()]
print(f"  samples without a genotyped recorded person {len(nt)}: best-match z median {nt.best_other_z.median():.1f}; z>=8: {int((nt.best_other_z >= 8).sum())}")
key = pd.read_csv(f"{P}/donor_key.csv")
for pj, lb in key.values:
    for s_, r in res[res.projid == pj].iterrows():
        if pd.notna(r.own_rank):
            print(f"{lb}: {mod} sample vs own genotype: rank {int(r.own_rank)} of 1708, own z {r.own_z:.1f}; best other z {r.best_other_z:.1f} (best other is SDA donor: {int(r.best_other_proj) in samps}; is another flagged donor: {int(r.best_other_proj) in set(key.projid)})")
        else:
            print(f"{lb}: {mod} sample; donor not genotyped; best-matching genotyped person z {r.best_other_z:.1f} (SDA donor: {int(r.best_other_proj) in samps})")
print(f"done {time.time()-t0:.0f}s")
