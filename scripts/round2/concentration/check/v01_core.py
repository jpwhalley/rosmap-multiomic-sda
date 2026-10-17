"""VERIFY structure analysis, part 1 (independent code; reads archive files directly, not the structure/ caches).
Concentration (centred top-donor share, PR), PIP>0.5 counts per modality from pip_final_*.csv, donor-modality mask from
the SDA input matrices, Gaussian / heavy-tail reference simulation (own seed), class counts and relations.
Writes $G_ROS_RESULTS_DIR/revision_analyses/verify_structure/private/v01_core.csv (per-component, includes top-donor index -> private) and out/v01_core.log."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, numpy as np, pandas as pd
from scipy import stats as ss

ROS = cfg.SDA
AUD = cfg.AUD
DS1 = f"{cfg.TABLES}/Data_S1_corrected_547.csv"
V = cfg.out_dir(cfg.WORK, "revision_analyses", "verify_structure"); cfg.out_dir(V, "out")
cfg.out_dir(V, "private")

don = [l.strip() for l in open(f"{ROS}/md_samples.txt") if l.strip()]
S = pd.read_csv(f"{ROS}/samples_final.csv", index_col=0)
S.columns = S.columns.astype(str)
assert list(S.columns) == don
S = S.drop(index=520)
for k in (547, 548):
    S.loc[k] = np.load(f"{AUD}/c547/C{k}_A.npy")
comps = list(S.index); assert len(comps) == 547

A = S.values.astype(float)
D = A - A.mean(1, keepdims=True)
D2 = D ** 2
tot = D2.sum(1)
srt = np.sort(D2, 1)[:, ::-1]
R = pd.DataFrame(index=pd.Index(comps, name="component"))
R["top_share"] = srt[:, 0] / tot
R["second_share"] = srt[:, 1] / tot
R["top3_share"] = srt[:, :3].sum(1) / tot
R["top5_share"] = srt[:, :5].sum(1) / tot
R["PR"] = tot ** 2 / (D2 ** 2).sum(1)
R["top_idx"] = D2.argmax(1)
# alternative: uncentred share (sensitivity of the centring choice)
R["top_share_uncentred"] = (A ** 2).max(1) / (A ** 2).sum(1)
# alternative: share after removing the top donor (does a second donor dominate the rest?)
R["share_excl_top"] = srt[:, 1] / (tot - srt[:, 0])

# ---- PIP > 0.5 counts per modality, read straight from the archive csvs
MODS = {"RNA": "pip_final_RNA.csv", "meth": "pip_final_meth.csv", "chip": "pip_final_chip.csv", "proteins": "pip_final_proteins.csv"}
for j, (m, f) in enumerate(MODS.items(), 1):
    cnt = {}
    for chunk in pd.read_csv(f"{ROS}/{f}", index_col=0, chunksize=50):
        for k, row in chunk.iterrows():
            cnt[int(k)] = int((row.values.astype(float) > 0.5).sum())
    for k in (547, 548):
        cnt[k] = int((np.load(f"{AUD}/c547/C{k}_P{j}.npy") > 0.5).sum())
    R[m] = [cnt[k] for k in comps]
    print(f"read {f}", flush=True)
R["n_mod"] = (R[list(MODS)] > 0).sum(1)
ds1 = pd.read_csv(DS1, index_col=0)
print("n_mod == Data S1 omics_count:", int((R.n_mod.values == ds1.loc[comps, "omics_count"].values).sum()), "/ 547")

# ---- donor-modality mask from the SDA inputs
mask = np.zeros((276, 4), bool)
rna = pd.read_csv(f"{ROS}/rna.txt", sep=" ", header=None, na_values="NA", engine="c", dtype=np.float32)
assert rna.shape[0] == 6 * 276
rv = rna.values.reshape(6, 276, -1)
mask[:, 0] = (~np.isnan(rv).all(2)).any(0)
del rna, rv
for i, f in enumerate(["meth.txt", "chip.txt", "prot.txt"], 1):
    x = pd.read_csv(f"{ROS}/{f}", sep=" ", header=None, na_values="NA", engine="c", dtype=np.float32).values
    mask[:, i] = ~np.isnan(x).all(1)
    del x
print("donors with data per modality (RNA, meth, chip, prot):", mask.sum(0).tolist())
np.save(f"{V}/private/v01_donor_modality_mask.npy", mask)
R.to_csv(f"{V}/private/v01_core.csv")
print("saved")
