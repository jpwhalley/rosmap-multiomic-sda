"""For every SDA 450K sample: best-matching genotyped person by the 15 autosomal rs probes (all samples, including
donors who are not genotyped), with the number of SNPs compared and mismatches. Then cross-assay identity summary for
the flagged donors: which genotyped person each assay's sample matches (pseudonymous: own / same person as other assay)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE); C = cfg.out_dir(cfg.WORK, "identity_checks", "cache")
ROS = cfg.ROS; O = f"{ROS}/oth"
d = np.load(f"{C}/meth_raw_nonauto.npz"); ids = d["ids"]; B = d["AVG_Beta"].astype(float)
S = pd.read_csv(f"{P}/03_meth_samples.csv", index_col=0)
rsm = np.char.startswith(ids, "rs"); rsid = ids[rsm]; Brs = B[rsm]
G = np.where(np.isnan(Brs), -1, np.where(Brs < 0.2, 0, np.where(Brs > 0.8, 2, 1)))
Gf = f"{O}/snps/ROSMAP_arrayGenotype"
fam = pd.read_csv(f"{Gf}.fam", sep=" ", header=None); bim = pd.read_csv(f"{Gf}.bim", sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
gproj = fam[1].str[3:].astype(int).values; nb = (len(fam) + 3) // 4; bed = np.memmap(f"{Gf}.bed", dtype=np.uint8, mode="r")
def geno(i):
    raw = bed[3 + i * nb: 3 + (i + 1) * nb]; out = np.empty(nb * 4, np.int8)
    for k in range(4):
        v = (raw >> (2 * k)) & 3; out[k::4] = np.select([v == 0, v == 1, v == 2, v == 3], [2, -1, 1, 0])
    return out[:len(fam)]
shared = [r for r in rsid if r in set(bim.SNP[bim.CHR.between(1, 22)])]
A = np.array([geno(int(np.where(bim.SNP == r)[0][0])) for r in shared])
Mg = G[[list(rsid).index(r) for r in shared]]
pos = {p: i for i, p in enumerate(gproj)}
pairs = [(si, pos[p]) for si, p in enumerate(S.projid.values) if pd.notna(p) and p in pos]
flip = []
for k in range(len(shared)):
    a = np.array([Mg[k, si] for si, gi in pairs]); b = np.array([A[k, gi] for si, gi in pairs]); ok = (a >= 0) & (b >= 0)
    flip.append(np.mean(a[ok] == b[ok]) < np.mean(a[ok] == 2 - b[ok]))
Mo = np.array([np.where(Mg[k] < 0, -1, 2 - Mg[k]) if flip[k] else Mg[k] for k in range(len(shared))])
af = np.array([np.mean(A[k][A[k] >= 0]) / 2 for k in range(len(shared))])
def llr(si, gi):
    """log10 likelihood ratio: same person (error 1%) vs random person (HWE) over SNPs called in both"""
    a, b = Mo[:, si], A[:, gi]; ok = (a >= 0) & (b >= 0); tot = 0.0
    for k in np.where(ok)[0]:
        p = af[k]; hw = [(1 - p) ** 2, 2 * p * (1 - p), p ** 2][a[k]]
        tot += np.log10((0.99 if a[k] == b[k] else 0.005) / hw)
    return tot, int(ok.sum()), int((a[ok] != b[ok]).sum())
sda = S[S.sda_meth]
rows = []
for nm, x in sda.iterrows():
    si = list(S.index).index(nm)
    sc = [llr(si, g) for g in range(len(fam))]
    l = np.array([s[0] for s in sc]); b = int(np.argmax(l))
    own = pos.get(int(x.projid))
    rows.append(dict(projid=int(x.projid), best_proj=int(gproj[b]), best_llr=l[b], best_n=sc[b][1], best_mm=sc[b][2],
                     own_llr=l[own] if own is not None else np.nan, own_n=sc[own][1] if own is not None else np.nan, own_mm=sc[own][2] if own is not None else np.nan,
                     second_llr=np.sort(l)[-2]))
R = pd.DataFrame(rows).set_index("projid"); R.to_csv(f"{P}/03f_meth_best_match.csv")
g = R[R.own_llr.notna()]
print(f"SDA 450K samples {len(R)}; donor genotyped {len(g)}: best match is own genotype {int((g.best_proj == g.index).sum())}; own LLR median {g.own_llr.median():.1f} (min {g.own_llr.min():.1f}); 2nd-best LLR median {g.second_llr.median():.1f}")
ng = R[R.own_llr.isna()]
print(f"donor not genotyped {len(ng)}: best-match LLR median {ng.best_llr.median():.1f}, max {ng.best_llr.max():.1f}; with LLR >= 4 (10^4:1) {int((ng.best_llr >= 4).sum())}")
key = pd.read_csv(f"{P}/donor_key.csv"); lab = dict(zip(key.projid, key.label))
rna = pd.read_csv(f"{P}/10_rna_identity.csv", index_col=0); chip = pd.read_csv(f"{P}/11_chip_identity.csv", index_col=0)
for pj, lb in key.values:
    out = [lb + ":"]
    rz = rna.loc[pj]; rna_src = int(rz.best_other_proj) if (pd.isna(rz.own_rank) or rz.own_rank > 1) and rz.best_other_z >= 8 else (pj if pd.notna(rz.own_rank) and rz.own_rank == 1 else None)
    out.append(f"RNA source {'own' if rna_src == pj else ('person P' if rna_src else 'unresolved')}")
    cz = chip[chip.projid == pj]
    chip_src = None
    if len(cz):
        cz = cz.iloc[0]; chip_src = int(cz.best_other_proj) if (pd.isna(cz.own_rank) or cz.own_rank > 1) and cz.best_other_z >= 8 else (pj if pd.notna(cz.own_rank) and cz.own_rank == 1 else None)
        out.append(f"ChIP source {'own' if chip_src == pj else ('person P' if chip_src and chip_src == rna_src else ('person Q' if chip_src else 'unresolved'))}")
    if pj in R.index:
        m = R.loc[pj]; ms = int(m.best_proj) if m.best_llr >= 4 else None
        tag = 'own' if ms == pj else ('person P' if ms and ms == rna_src else ('person Q' if ms and ms == chip_src else ('another person' if ms else 'unresolved')))
        out.append(f"450K source {tag} (best LLR {m.best_llr:.1f} over {int(m.best_n)} SNPs, {int(m.best_mm)} mismatches; own LLR {m.own_llr if pd.notna(m.own_llr) else float('nan'):.1f})")
    print(" ".join(out))
