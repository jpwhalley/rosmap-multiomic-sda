"""Parse one SDA run (it1500) into float64 npy in $G_ROS_RESULTS_DIR/verify_protein_influence/cache/run{r}/ and print basic structure (aggregates only)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, numpy as np
from vp_common import *
r = int(sys.argv[1])
for w in ("A", "B1", "X1", "S1", "X2", "S2", "X3", "S3", "X4", "S4"):
    M = run_mat(r, w)
    log(f"run{r} {w}", M.shape, "nan", int(np.isnan(M).sum()))
    del M
A = run_mat(r, "A"); B1 = run_mat(r, "B1")
log("A col mean(a^2): quantiles", np.round(np.quantile((A ** 2).mean(0), [0, .1, .5, .9, 1]), 3))
log("B1 |b| max per comp quantiles", np.round(np.quantile(np.abs(B1).max(0), [0, .1, .5, .9, 1]), 3))
for d in (1, 4):
    X = run_mat(r, f"X{d}"); S = run_mat(r, f"S{d}")
    for lo, hi in ((0, 1e-3), (1e-3, .05), (.05, .2), (.2, .5), (.5, .9), (.9, 1.01)):
        m = (S >= lo) & (S < hi)
        log(f" X{d}: S in [{lo},{hi}) n={int(m.sum())} median|X|={np.median(np.abs(X[m])) if m.any() else np.nan:.3g} median|X|/S={np.median(np.abs(X[m]) / S[m]) if m.any() else np.nan:.3g}")
    del X, S
