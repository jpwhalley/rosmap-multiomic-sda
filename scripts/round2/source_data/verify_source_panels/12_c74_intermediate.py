"""C74 donors with intermediate scores: are they cross-modality sex-discordant samples (possible sample mix-ups)?
Counts only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
C = f"{cfg.WORK}/verify_source_panels/cache"
A = np.load(f"{C}/A.npy"); cov = pd.read_csv(f"{C}/covariates_verify.csv", index_col=0)
rna = np.load(f"{C}/RNA_input.npy"); genes = [str(g) for g in np.load(f"{C}/RNA_feats.npy")]
chip = np.load(f"{C}/chip_input.npy"); cf = np.load(f"{C}/chip_feats.npy")
dom = pd.read_csv(f"{cfg.ROS}/oth/chip/H3K9acDomains.csv", sep="\t", index_col=0); dom["Chr"] = dom.Chr.astype(str)
a = A[73]; sex = cov.msex.values
inter = np.abs(a) < 0.08
print("C74 donors with |score| < 0.08:", int(inter.sum()), "(clinical males:", int((sex[inter] == 1).sum()), ")")
# RNA sex per cell type from XIST and Y genes
xi = genes.index("XIST"); ys = [genes.index(g) for g in ["UTY", "DDX3Y", "USP9Y", "KDM5D", "RPS4Y1"] if g in genes]
calls = []
for c in range(6):
    blk = rna[c * 276:(c + 1) * 276]
    ysum = blk[:, ys].sum(1); x = blk[:, xi]
    calls.append(np.where(ysum > x, 1, 0))
calls = np.array(calls)          # 6 x 276
rna_call = (calls.mean(0) > 0.5).astype(int); split = (calls.min(0) != calls.max(0))
print("RNA-called sex discordant with clinical:", int((rna_call != sex).sum()), "; donors with cell types disagreeing:", int(split.sum()))
ym = np.nanmean(chip[:, dom.loc[cf, "Chr"].values == "Y"], 1); has = ~np.isnan(ym)
v = np.sort(ym[has]); g = np.argmax(np.diff(v)); thr = (v[g] + v[g + 1]) / 2
chip_call = np.where(has, (ym > thr).astype(float), np.nan)
print("ChIP-called sex discordant with clinical:", int(np.nansum(chip_call != sex) - np.isnan(chip_call).sum() * 0), "(of", int(has.sum()), ")")
dc = has & (chip_call != sex)
print("ChIP-discordant n:", int(dc.sum()))
for i in np.where(inter)[0]:
    print(f"  intermediate donor: clinical {int(sex[i])}, RNA call {int(rna_call[i])}, RNA split {bool(split[i])}, ChIP call {chip_call[i]}, score sign {np.sign(a[i]):+.0f}")
print("RNA-discordant donors that are intermediate on C74:", int(((rna_call != sex) & inter).sum()), "; ChIP-discordant that are intermediate:", int((dc & inter).sum()))
