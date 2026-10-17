"""Genotype-free cross-modal identity: does a donor's snRNA profile come from the same person as its 450K sample?
SNPs that are both strong eQTLs (snRNA) and strong meQTLs (450K) are found in genotyped donors (held-out donors excluded);
then, for a held-out RNA profile, the eQTL-implied allele dosages are correlated with the meQTL-implied dosages of every
450K sample. Validated on 40 random held-out genotyped donors; applied to the RNA-discordant donors (RNA-2 is not genotyped).
Aggregates/pseudonymous labels only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, time, numpy as np, pandas as pd
from qtl_common import *
P = cfg.out_dir(cfg.PRIVATE); ROS = cfg.ROS; SDA = f"{ROS}/sda"; O = f"{ROS}/oth"; t0 = time.time()
samps = np.array([int(l) for l in open(f"{SDA}/md_samples.txt") if l.strip()])
key = pd.read_csv(f"{P}/donor_key.csv"); inv = dict(zip(key.label, key.projid))
G, gproj, bim = load_geno(); row = {p: i for i, p in enumerate(gproj)}
R = pd.read_csv(f"{SDA}/rna.txt", sep=" ", header=None, dtype=np.float32).values
blocks, gname = [], []
for c in range(6):
    X = R[c * 276:(c + 1) * 276]; keep = np.median(X, 0) >= 10
    blocks.append(residualize(np.log2(X[:, keep] + 1).astype(np.float64)))
Zr = np.hstack(blocks).astype(np.float32)
hdr = pd.read_csv(f"{ROS}/meth/ROSMAP_arrayMethylation_imputed.tsv.gz", sep="\t", index_col=0, nrows=0).columns
keep = None
for ch in pd.read_csv(f"{ROS}/meth/ROSMAP_arrayMethylation_imputed.tsv.gz", sep="\t", index_col=0, chunksize=40000, dtype={c: np.float32 for c in hdr}):
    ch = ch.copy(); ch["_sd"] = ch.std(axis=1); keep = (ch if keep is None else pd.concat([keep, ch])).nlargest(20000, "_sd")
Zm = residualize(keep[hdr].values.T.astype(np.float64)).astype(np.float32)
bio = pd.read_csv(f"{O}/meta/ROSMAP_biospecimen_metadata.csv", low_memory=False); s2i = bio.drop_duplicates("specimenID").set_index("specimenID").individualID
clin = pd.read_csv(f"{O}/meta/ROSMAP_clinical.csv"); i2p = dict(zip(clin.individualID, clin.projid))
mproj = np.array([i2p.get(s2i.get(c), -1) for c in hdr], dtype=float)
print("loaded", f"{time.time()-t0:.0f}s", flush=True)
rng = np.random.default_rng(4)
typed_rna = [i for i, p in enumerate(samps) if p in row and p in set(mproj)]
hold = set(rng.choice(typed_rna, 40, replace=False).tolist()) | {int(np.where(samps == inv[l])[0][0]) for l in ["RNA-1", "RNA-2", "RNAMIX-1"]}
hold_proj = set(samps[list(hold)])
d = [i for i, p in enumerate(samps) if p in row and i not in hold]
pairs, _ = discover(Zr[d], std_geno(G[[row[samps[i]] for i in d]]), 0.5, bim)
print(f"eQTL pairs {len(pairs)} (discovery n {len(d)}) {time.time()-t0:.0f}s", flush=True)
js = np.array([j for _, j, _ in pairs])
dm = [k for k, p in enumerate(mproj) if p in row and p not in hold_proj]
Gs = std_geno(G[[row[int(mproj[k])] for k in dm]][:, js])
Mz = (Zm[dm] - Zm[dm].mean(0)) / (Zm[dm].std(0) + 1e-9)
Cm = Gs.T @ Mz / len(dm)                                   # snps x cpgs
bc = np.abs(Cm).argmax(1); br = Cm[np.arange(len(js)), bc]
ok = np.abs(br) >= 0.5
print(f"of these eQTL SNPs, also strong meQTLs (|r|>=0.5, discovery n {len(dm)}): {int(ok.sum())}", flush=True)
ge = np.array([p for p, _, _ in pairs])[ok]; se = np.sign([r for _, _, r in pairs])[ok]; cg = bc[ok]; sm = np.sign(br[ok])
def score(i):
    u = Zr[i, ge] * se; V = Zm[:, cg] * sm
    u = (u - u.mean()) / u.std(); V = (V - V.mean(1, keepdims=True)) / V.std(1, keepdims=True)
    return V @ u / len(u)
res = []
for i in sorted(hold):
    sc = score(i); own = np.where(mproj == samps[i])[0]
    z = (sc - sc.mean()) / sc.std()
    if len(own):
        o = own[0]; res.append(dict(projid=samps[i], has_meth=True, own_rank=int(1 + np.sum(np.delete(sc, o) > sc[o])), own_z=z[o], best_other_z=np.max(np.delete(z, o))))
    else: res.append(dict(projid=samps[i], has_meth=False, own_rank=np.nan, own_z=np.nan, best_other_z=z.max()))
T = pd.DataFrame(res).set_index("projid"); T.to_csv(f"{P}/16_cross_modal_rna_meth.csv")
val = T.drop([inv["RNA-1"], inv["RNA-2"], inv["RNAMIX-1"]], errors="ignore"); val = val[val.has_meth]
print(f"validation (held-out genotyped donors with RNA and 450K): n {len(val)}; own 450K sample ranks 1st of {len(hdr)}: {int((val.own_rank == 1).sum())}; own z median {val.own_z.median():.1f} (min {val.own_z.min():.1f}); best other z median {val.best_other_z.median():.1f} (max {val.best_other_z.max():.1f})")
for l in ["RNA-1", "RNA-2", "RNAMIX-1"]:
    x = T.loc[inv[l]]
    print(f"{l}: snRNA vs own 450K sample: rank {int(x.own_rank)} of {len(hdr)}, own z {x.own_z:.1f}; best other 450K z {x.best_other_z:.1f}")
print(f"done {time.time()-t0:.0f}s")
