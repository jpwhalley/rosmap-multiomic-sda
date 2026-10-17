"""Build the proteomics input as ros1d would have with the ROSMAP.DLPFC-only mapping (same
complete-case rule), in md_prot.txt feature order (7 features absent -> NaN). Cache only.
Report whether each headline component's extreme donor is among the mis-attributed rows."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from common import *
ROS = cfg.ROS
S, R, Aall, B, T = load()
samps = [int(s) for s in S["samples"]]
all_clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv", index_col=0); sc_clin = all_clin.loc[samps]
tmt = pd.read_csv(f"{ROS}/prot/tmt/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv", index_col=0)
meta0 = pd.read_csv(f"{ROS}/prot/tmt/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv", index_col=0)
m = meta0[meta0.index.str.startswith("ROSMAP.DLPFC") & (meta0.isAssayControl.astype(str).str.upper() == "FALSE")].copy()
m["R"] = m.index.str.split(".").str[-1]; m["SID"] = m.index.str.split(".").str[-3] + "." + m.index.str.split(".").str[-2]
ren = tmt[m.SID.values].rename(columns=dict(zip(m.SID, m.R)))
ids = set(sc_clin.individualID).intersection(ren.columns)
f = ren[list(ids)].dropna(how="any").reindex(columns=sc_clin.individualID.values)
f.columns = samps
feats = [l.rstrip("\n") for l in open(f"{SDA}/md_prot.txt")]
Pc = f.reindex(feats).T.values  # 276 x 5230
np.save(f"{CACHE}/prot_corrected_full.npy", Pc)
Po = pd.read_csv(f"{SDA}/prot.txt", sep=" ", header=None, na_values="NA").values
has_o = ~np.isnan(Po).all(1); has_c = ~np.isnan(Pc).all(1)
wrong = np.array([has_o[i] and (not has_c[i] or not np.allclose(Po[i][~np.isnan(Pc[i])], Pc[i][~np.isnan(Pc[i])])) for i in range(276)])
np.save(f"{CACHE}/prot_rows_misattributed.npy", wrong)
print("donors with prot.txt data:", has_o.sum(), " of which not their own sample:", wrong.sum(), " corrected-mapping donors with data:", has_c.sum())
for k in [351, 177, 2, 74, 87, 278, 21, 172, 277]:
    e = score_stats(Aall[k - 1].astype(float))["idx_extreme"]
    print(f"C{k}: extreme donor has prot.txt data {has_o[e]}, row mis-attributed {wrong[e]}, has own TMT sample {has_c[e]}")
