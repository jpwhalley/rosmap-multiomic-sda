"""Parse the four SDA input files into npy (cache only) and describe their missingness. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np
from vp_common import *

for k in ("rna", "meth", "chip", "prot"):
    M = read_txt(f"{SDA}/{k}.txt")
    np.save(f"{CACHE}/in_{k}.npy", M)
    rowall = np.isnan(M).all(1); rowany = np.isnan(M).any(1)
    log(k, M.shape, "all-NA rows", int(rowall.sum()), "partial-NA rows", int((rowany & ~rowall).sum()),
        "min", float(np.nanmin(M)), "max", float(np.nanmax(M)), "mean", float(np.nanmean(M)))
R = np.load(f"{CACHE}/in_rna.npy")
assert R.shape[0] == 6 * N
for t in range(6):
    B = R[t * N:(t + 1) * N]
    log(" rna block", t, "all-NA rows", int(np.isnan(B).all(1).sum()), "zero-variance genes", int((np.nanstd(B, 0) == 0).sum()))
