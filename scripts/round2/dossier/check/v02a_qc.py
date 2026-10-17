"""V02a: rebuild the donor QC proxies (nuclei, UMIs per nucleus, class proportions) from the sources and compare with
covariates/rna_qc_donor.csv used by the dossier. Writes cache/qc_donor.csv (per-donor, local only); prints aggregates."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from v00_common import *

don = donors()
dc = pd.read_csv(f"{cfg.WORK}/revision_r2/donor_celltype_counts_from_h5ad.csv", index_col=0)
dc.index = dc.index.astype(str); dc = dc.loc[don, CTS]
umi = {}
for c in CTS:
    x = pd.read_csv(f"{ROS}/mit/rna/pseudobulk/{c}_raw_counts.csv", index_col=0)
    x.index = x.index.astype(str)
    if not set(don) <= set(x.index):          # orientation check
        x = x.T; x.index = x.index.astype(str)
    umi[c] = x.loc[don].sum(axis=1)
umi = pd.DataFrame(umi)
q = pd.DataFrame(index=don)
q["nuclei"] = dc.sum(1); q["umis"] = umi.sum(1); q["upn"] = q.umis / q.nuclei
for c in CTS:
    q[f"prop_{c}"] = dc[c] / q.nuclei
q.to_csv(f"{CACHE}/qc_donor.csv")
ref = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); ref.index = ref.index.astype(str); ref = ref.loc[don]
print("max rel diff nuclei", float(np.max(np.abs(q.nuclei - ref.nuclei) / ref.nuclei)))
print("max rel diff umis/nucleus", float(np.max(np.abs(q.upn - ref.umis_per_nucleus) / ref.umis_per_nucleus)))
print("proportions summary (median):", q[[f"prop_{c}" for c in CTS]].median().round(3).to_dict())
