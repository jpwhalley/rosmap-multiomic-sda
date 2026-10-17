"""Genotype-based sex and identity for all 1708 genotyped samples, focused on the 276 SDA donors.
X heterozygosity and Y call counts decoded directly from the PLINK .bed (fam sex ignored).
Relatedness: KING kin0 (228 GWAS donors, archived) + PLINK --genome over all 1708 (cache/plink).
Per-sample values -> $G_ROS_PRIVATE_DIR; aggregates printed."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE); C = cfg.out_dir(cfg.WORK, "identity_checks", "cache"); RS = cfg.out_dir(cfg.WORK, "identity_checks", "results")
ROS = cfg.ROS; G = f"{ROS}/oth/snps/ROSMAP_arrayGenotype"
fam = pd.read_csv(f"{G}.fam", sep=" ", header=None, names=["FID", "IID", "P", "M", "SEX", "PH"])
bim = pd.read_csv(f"{G}.bim", sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
n = len(fam); nb = (n + 3) // 4
bed = np.memmap(f"{G}.bed", dtype=np.uint8, mode="r")
assert bytes(bed[:3]) == b"\x6c\x1b\x01"
def geno(idx):
    """returns (len(idx), n) int8 matrix: 0,1,2 = count of A1? (plink code), -1 missing"""
    raw = np.stack([bed[3 + i * nb: 3 + (i + 1) * nb] for i in idx])
    out = np.empty((len(idx), nb * 4), np.int8)
    for k in range(4):
        v = (raw >> (2 * k)) & 3
        out[:, k::4] = np.select([v == 0, v == 1, v == 2, v == 3], [2, -1, 1, 0])
    return out[:, :n]
X = geno(np.where(bim.CHR == 23)[0]); Y = geno(np.where(bim.CHR == 24)[0])
s = pd.DataFrame({"IID": fam.IID, "famsex": fam.SEX, "batch": np.where(fam.FID.str.startswith("KronosII"), "KronosII", "other")})
s["projid"] = s.IID.str[3:].astype(int)
s["x_called"] = (X >= 0).sum(0); s["x_het"] = (X == 1).sum(0)
s["x_hetrate"] = s.x_het / s.x_called.replace(0, np.nan)
s["y_called"] = (Y >= 0).sum(0); s["y_het"] = (Y == 1).sum(0)
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv").set_index("projid")
s["clin_male"] = clin.msex.reindex(s.projid).values
# genetic sex: informative only when X calls exist
inf = s.x_called > 1000
s["geno_call"] = np.where(~inf, "no X/Y calls", np.where((s.x_hetrate > 0.05) & (s.y_called < 10), "F",
                   np.where((s.x_hetrate < 0.005) & (s.y_called > 100), "M", "ambiguous")))
print("genotyped samples", n, "| with X/Y calls:", int(inf.sum()), "| by batch:", s.groupby("batch").apply(lambda d: int((d.x_called > 1000).sum())).to_dict())
print("genetic call counts:", s.geno_call.value_counts().to_dict())
for g, d in s[inf].groupby("geno_call"):
    print(f"  {g}: X het rate q01/50/99 {np.round(np.nanpercentile(d.x_hetrate, [1, 50, 99]), 4)}; Y called q01/50/99 {np.percentile(d.y_called, [1, 50, 99])}")
cc = s[s.geno_call.isin(["M", "F"]) & s.clin_male.notna()]
print("genetic vs clinical sex (all genotyped, informative):", len(cc), "discordant:", int(((cc.geno_call == "M") != (cc.clin_male == 1)).sum()))
print("fam sex vs clinical sex (all 1708): discordant", int(((s.famsex == 1) != (s.clin_male == 1))[s.clin_male.notna()].sum()), "| not in clinical", int(s.clin_male.isna().sum()))
print("fam sex vs genetic call (informative):", int(((s.famsex == 1) != (s.geno_call == "M"))[s.geno_call.isin(["M", "F"])].sum()))
# SDA donors
samps = np.array([int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip()])
gw = set(int(l.split()[1][3:]) for l in open(f"{ROS}/oth/snps/ROSMAP_arrayGenotype_maf.fam"))
sd = s[s.projid.isin(samps)].copy()
print(f"SDA donors genotyped: {sd.projid.nunique()} of 276 (rows {len(sd)}); in GWAS .fam (maf subset): {len(gw & set(samps))}; GWAS set == SDA∩genotyped: {gw == set(sd.projid)}")
print("SDA genotyped with X/Y calls:", int((sd.x_called > 1000).sum()), "| KronosII (no X/Y calls):", int((sd.batch == "KronosII").sum()))
ci = sd[sd.geno_call.isin(["M", "F"])]
print("SDA genetic vs clinical sex: n", len(ci), "discordant", int(((ci.geno_call == "M") != (ci.clin_male == 1)).sum()))
print("SDA fam sex vs clinical: discordant", int(((sd.famsex == 1) != (sd.clin_male == 1)).sum()))
s.to_csv(f"{P}/02_genotype_sex_all.csv", index=False)
# relatedness
k = pd.read_csv(f"{ROS}/oth/snps/gemma/relatedness.kin0", sep="\t")
print(f"KING kin0 (GWAS 228): pairs {len(k)}; max kinship {k.Kinship.max():.4f}; pairs >0.354 (dup/MZ) {int((k.Kinship > 0.354).sum())}, 0.177-0.354 (1st deg) {int(((k.Kinship > 0.177) & (k.Kinship <= 0.354)).sum())}, 0.0884-0.177 (2nd) {int(((k.Kinship > 0.0884) & (k.Kinship <= 0.177)).sum())}; min IBS0 {k.IBS0.min():.4f}")
g = pd.read_csv(f"{C}/plink/genome_full.genome", sep=r"\s+")
print("PLINK --genome over all 1708 (86,696 LD-pruned autosomal SNPs), pairs with PI_HAT>0.1:", len(g))
# fam-vs-clinical discordant sample: is it an SDA donor?
fd = s[((s.famsex == 1) != (s.clin_male == 1)) & s.clin_male.notna()]
print("fam/clinical-discordant samples:", len(fd), "| SDA donors among them:", int(fd.projid.isin(samps).sum()), "| their genetic call agrees with fam sex:", [ (r.geno_call == "M") == (r.famsex == 1) if r.geno_call in ("M","F") else r.geno_call for r in fd.itertuples()])
# discordant-donor labels
key = pd.read_csv(f"{P}/donor_key.csv")
for pj, lab in key.values:
    r = s[s.projid == pj]
    if len(r) == 0: print(lab, ": not genotyped; GWAS set: False"); continue
    r = r.iloc[0]
    agree = None if r.geno_call not in ("M", "F") else ((r.geno_call == "M") == (r.clin_male == 1))
    print(f"{lab}: genotyped (batch {r.batch}); GWAS set {pj in gw}; X calls {r.x_called}, X het rate {r.x_hetrate:.4f}, Y calls {r.y_called}; genetic call {'n/a' if agree is None else ('AGREES' if agree else 'DISAGREES')} with clinical; fam sex {'agrees' if (r.famsex==1)==(r.clin_male==1) else 'disagrees'} with clinical")
