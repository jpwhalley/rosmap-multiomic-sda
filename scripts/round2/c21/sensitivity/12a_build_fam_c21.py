"""C21 GEMMA sensitivity: phenotype columns = C21 as archived (byte-identical column from the validated geno547.fam),
C21 with the RNA-discordant genotyped donor set missing (NA), and C21 with all flagged donors in the GWAS set missing."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.PRIVATE; OUT = cfg.out_dir(cfg.WORK, "identity_checks", "gwas_c21")
SNP = cfg.SNP; AUD = f"{cfg.AUD}/gwas547"   # geno547.fam from c21/gwas547/01_build_fam.py
for f in ["bed", "bim"]:
    if not os.path.exists(f"{OUT}/geno.{f}"): os.symlink(f"{SNP}/ROSMAP_arrayGenotype_maf.{f}", f"{OUT}/geno.{f}")
if not os.path.exists(f"{OUT}/relatedness_matrix.txt"): os.symlink(f"{SNP}/gemma/relatedness_matrix.txt", f"{OUT}/relatedness_matrix.txt")
key = pd.read_csv(f"{P}/donor_key.csv"); lab = dict(zip(key.projid, key.label))
rna_bad = {p for p, l in lab.items() if l.startswith("RNA-")}
all_bad = {p for p, l in lab.items() if l.startswith(("RNA-", "CHIP-", "RNAMIX-"))}
out = []; nA = nB = 0
for line in open(f"{AUD}/geno547.fam"):
    f = line.rstrip("\n").split(" "); pj = int(f[1][3:]); c21 = f[5]
    a = "NA" if pj in rna_bad else c21; b = "NA" if pj in all_bad else c21
    nA += a == "NA"; nB += b == "NA"
    out.append(" ".join(f[:5] + [c21, a, b]))
open(f"{OUT}/geno.fam", "w").write("\n".join(out) + "\n")
print("rows", len(out), "| set missing: RNA-discordant", nA, "| all sex-flagged", nB)
