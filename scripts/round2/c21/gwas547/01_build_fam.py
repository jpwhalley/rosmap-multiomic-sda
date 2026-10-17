"""Build a fresh GEMMA .fam for C21, C74, C351 (validation) + C547, C548 (new).

Replicates ros4_GWAS_association_study_maf.py: line[:-3] of the *_before.fam,
then ' ' + str(value) per phenotype, value looked up by fam IID minus its
3-letter study prefix (= projid). Archive files are read-only; nothing edited.
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os
import numpy as np
import pandas as pd

SNP = cfg.SNP
SDA = cfg.SDA
C547 = f"{cfg.AUD}/c547"
OUT = cfg.out_dir(cfg.AUD, "gwas547")   # GEMMA working folder (run_gemma.sh)
VAL = [21, 74, 351]

md = open(f"{SDA}/md_samples.txt").read().splitlines()
samples = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0)
cols = [str(c) for c in samples.columns]
print("md_samples n =", len(md), "| samples_final columns n =", len(cols))
print("md_samples order == samples_final column order:", md == cols)
assert md == cols
print("samples_final rows (components):", samples.shape[0], "index head:", list(samples.index[:3]))
samples = samples.T
samples.index = samples.index.astype(str)
labels = [int(c) for c in samples.columns]
print("component labels are 1..546 in order:", labels == list(range(1, 547)))
assert labels == list(range(1, 547))

a547 = np.load(f"{C547}/C547_A.npy")
a548 = np.load(f"{C547}/C548_A.npy")
print("C547_A shape", a547.shape, "C548_A shape", a548.shape)
assert a547.shape == (276,) and a548.shape == (276,)
new = {547: dict(zip(md, a547)), 548: dict(zip(md, a548))}

lines = open(f"{SNP}/ROSMAP_arrayGenotype_maf_before.fam").read().splitlines()
arch = open(f"{SNP}/ROSMAP_arrayGenotype_maf.fam").read().splitlines()
assert len(lines) == len(arch) == 228
assert all(l.endswith(" -9") for l in lines)

missing = []
out_lines = []
n_byte_mismatch = {k: 0 for k in VAL}
n_prefix_mismatch = 0
for line, aline in zip(lines, arch):
    temp = line.split(" ")
    pid = temp[1][3:]
    if pid not in samples.index or pid not in new[547]:
        missing.append(temp[1])
        continue
    # original iterated `for comp in samples.loc[pid]` -> str(comp); pick C21/C74/C351 by label
    row = list(samples.loc[pid])
    vals = [str(row[labels.index(k)]) for k in VAL]
    vals += [str(new[547][pid]), str(new[548][pid])]
    s = line[:-3] + "".join(" " + v for v in vals)
    out_lines.append(s)
    # byte check against archived .fam
    af = aline.split(" ")
    if " ".join(af[:5]) != line[:-3]:
        n_prefix_mismatch += 1
    for j, k in enumerate(VAL):
        if af[5 + (k - 1)] != vals[j]:
            n_byte_mismatch[k] += 1

print("fam IIDs without a projid value:", len(missing) if missing else "none")   # count only
print("first-five-field mismatches vs archived fam:", n_prefix_mismatch)
print("validation column byte mismatches vs archived fam (C21, C74, C351):", n_byte_mismatch)

with open(f"{OUT}/geno547.fam", "w") as f:
    for s in out_lines:
        f.write(s + "\n")

# Extra: byte-identity of each validation column as a whole column string
chk = open(f"{OUT}/geno547.fam").read().splitlines()
for j, k in enumerate(VAL):
    mine = "\n".join(l.split(" ")[5 + j] for l in chk)
    theirs = "\n".join(l.split(" ")[5 + k - 1] for l in arch)
    print(f"C{k}: column byte-identical = {mine == theirs}")
print("rows written:", len(chk), "fields per row:", sorted(set(len(l.split(' ')) for l in chk)))
# finite / missing check on new columns
for j, k in enumerate([547, 548]):
    v = np.array([float(l.split(" ")[8 + j]) for l in chk])
    print(f"C{k}: n={v.size} finite={np.isfinite(v).sum()} mean={v.mean():.4g} sd={v.std():.4g} min={v.min():.4g} max={v.max():.4g}")
