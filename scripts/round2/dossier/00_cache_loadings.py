"""Cache the published (deposited) feature loadings and PIPs of the dossier components (all four modalities)
plus the corrected cell-type loadings. Read-only on the archive; cache under component_dossier/cache."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import time, numpy as np, pandas as pd
from dc_common import *
t0 = time.time()
out = {}
for m, fn in MODS.items():
    feats = [l.rstrip("\n") for l in open(f"{SDA}/{fn}")]
    X = read_rows(f"{SDA}/genes_final_{m}.csv", COMPS).loc[COMPS]
    P = read_rows(f"{SDA}/pip_final_{m}.csv", COMPS).loc[COMPS]
    assert list(X.columns) == feats and list(P.columns) == feats
    out[f"{m}_X"] = X.values.astype(np.float64); out[f"{m}_P"] = P.values.astype(np.float64); out[f"{m}_feats"] = np.array(feats)
    print(m, X.shape, f"{time.time()-t0:.0f}s", flush=True)
B = pd.read_csv(BCOR, index_col=0).loc[COMPS, CTS]
Bp = pd.read_csv(f"{SDA}/treatment_final.csv", index_col=0)
Bp.index = Bp.index.astype(int)
out["B_cor"] = B.values; out["B_pub"] = Bp.loc[COMPS].values; out["B_pub_cols"] = np.array(list(Bp.columns))
out["comps"] = np.array(COMPS)
np.savez_compressed(f"{CACHE}/loadings.npz", **out)
print("saved", f"{time.time()-t0:.0f}s")
