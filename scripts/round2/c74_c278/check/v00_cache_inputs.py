"""Independent cache of the SDA inputs (read from the raw text files, not from the dossier caches).
Writes cache/*.npy to $G_ROS_RESULTS_DIR/revision_analyses/verify_c74_c278/cache (per donor)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd, time
os.environ["OMP_NUM_THREADS"] = cfg.THREADS
SDA = cfg.SDA
OUT = cfg.out_dir(cfg.WORK, "revision_analyses", "verify_c74_c278", "cache")
t = time.time()
for m in ("rna", "meth", "chip"):
    X = pd.read_csv(f"{SDA}/{m}.txt", sep=" ", header=None, na_values=["NA"], dtype=np.float32, engine="c")
    X = X.dropna(axis=1, how="all")  # trailing separator column
    np.save(f"{OUT}/{m}.npy", X.values.astype(np.float32))
    print(m, X.shape, f"{time.time()-t:.0f}s", flush=True)
