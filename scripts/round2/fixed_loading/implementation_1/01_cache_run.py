"""Convert one run's it1500 SDA text outputs to float32 .npy in $G_ROS_RESULTS_DIR/protein_influence/cache/run{r}/.
Usage: 01_cache_run.py r [--with-S]"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, os, numpy as np
from pi_common import *

r = int(sys.argv[1]); withS = "--with-S" in sys.argv
out = f"{CACHE}/run{r}"; os.makedirs(out, exist_ok=True)
which = ["A", "B1", "X1", "X2", "X3", "X4"] + (["S1", "S2", "S3", "S4"] if withS else [])
for w in which:
    f = f"{out}/{w}.npy"
    if os.path.exists(f):
        continue
    M = load_run(r, which=(w,), dtype=np.float32 if w not in ("A", "B1") else np.float64)[w]
    np.save(f, M)
    tlog(f"run{r} {w} {M.shape} {M.dtype} NaN={int(np.isnan(M).sum())}")
