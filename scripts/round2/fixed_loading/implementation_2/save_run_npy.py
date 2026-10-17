"""Save one run's it1500 matrices as float32 npy in $G_ROS_RESULTS_DIR/verify_protein_influence/cache/run{r}/ (so repeated variant fits do not re-parse the text)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, numpy as np
from vp_common import *
import vp_common
r = int(sys.argv[1])
for w in ("A", "B1", "X1", "S1", "X2", "S2", "X3", "S3", "X4", "S4"):
    M = run_mat(r, w, np.float32 if w not in ("A", "B1") else np.float64)
    np.save(f"{CACHE}/run{r}/{w}.npy", M); log(r, w, M.shape, M.dtype); del M; vp_common._MEM.clear()
