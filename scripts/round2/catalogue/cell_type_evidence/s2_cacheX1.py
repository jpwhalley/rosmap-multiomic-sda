"""Parse archived X1 (2000 x 25098, one component per line) and B1 (6 x 2000) per run into npy ($G_ROS_AUDIT_DIR/ds1verify_cosB)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import time, numpy as np, pandas as pd
BASE = cfg.SDA
OUT = cfg.out_dir(cfg.AUD, "ds1verify_cosB")
t0 = time.time()
for i in range(1, 11):
    X = pd.read_csv(f"{BASE}/results{i}/it1500/X1", sep=" ", header=None, dtype=np.float64).values
    B = pd.read_csv(f"{BASE}/results{i}/it1500/B1", sep=" ", header=None, dtype=np.float64).values
    assert X.shape == (2000, 25098), X.shape
    assert B.shape == (6, 2000), B.shape
    np.save(f"{OUT}/X1_run{i}.npy", X); np.save(f"{OUT}/B1_run{i}.npy", B)
    print(i, X.shape, B.shape, f"{time.time()-t0:.0f}s", flush=True)
