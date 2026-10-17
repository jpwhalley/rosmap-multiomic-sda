"""Fig S1e reproduced independently (plink dosage, rna.txt OR2L3 column, log2(RPM+1), 3-group OLS F-test), with and
without the genotyped donor whose snRNA profile is another person's (identity_checks RNA-1). Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import subprocess, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "neuronal", "check"))
from v_common import *   # neuronal/check/v_common.py
SNP = cfg.SNP
open(f"{CACHE}/v09_snp.txt", "w").write("rs6664332\n")
subprocess.run(["plink", "--bfile", f"{SNP}/ROSMAP_arrayGenotype_maf", "--extract", f"{CACHE}/v09_snp.txt", "--recode", "A", "--threads", cfg.THREADS,
                "--out", f"{CACHE}/v09_geno"], check=True, capture_output=True)
g = pd.read_csv(f"{CACHE}/v09_geno.raw", sep=r"\s+"); g.index = g.IID.astype(str).str[3:]
col = [c for c in g.columns if c.startswith("rs6664332_")][0]; print("dosage column (counted allele):", col)
don = donors(); F = pd.read_csv(f"{CACHE}/frame.csv"); F.index = don
shared = [d for d in don if d in g.index and not np.isnan(g.loc[d, col])]
print("genotyped SDA donors:", len(shared), "| of which RNA-mismatched:", int(F.loc[shared, "rna_mismatch"].sum()))
genes = [l.strip() for l in open(f"{ROS}/sda/genes.txt") if l.strip()]; gi = genes.index("OR2L3")
X = pd.read_csv(f"{ROS}/sda/rna.txt", sep=" ", header=None, usecols=[gi]).values[:, 0]
rows = []
for lab, keep in (("all genotyped", shared), ("excl. RNA-mismatched", [d for d in shared if F.loc[d, "rna_mismatch"] == 0])):
    pos = [don.index(d) for d in keep]; gg = g.loc[keep, col].values
    cnt = pd.Series(gg).value_counts().sort_index().to_dict()
    for c, ct in enumerate(CT_ORDER):
        y = np.log2(X[c * 276:(c + 1) * 276][pos] + 1)
        df = pd.DataFrame({"y": y, "g": gg}); df["gc"] = df.g.astype(int).astype(str)
        f3 = smf.ols("y ~ C(gc)", df).fit(); fa = smf.ols("y ~ g", df).fit()
        rows.append(dict(sample=lab, n=len(keep), geno_counts=str({int(k): int(v) for k, v in cnt.items()}), cell_type=ct, p_F3=f3.f_pvalue, beta_per_A=fa.params["g"],
                         frac_zero=float((y == 0).mean())))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/v09_figS1e.csv", index=False, float_format="%.3g")
pd.set_option("display.width", 200); print(R.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
