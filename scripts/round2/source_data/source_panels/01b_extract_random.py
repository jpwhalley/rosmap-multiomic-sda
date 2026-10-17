"""Random background features (not PIP>0.5 in any of the nine components) from the SDA
input matrices, for a null calibration of the loading-weighted aggregate. Read-only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
BASE = cfg.SDA
OUT = cfg.out_dir(cfg.WORK, "source_panels", "cache")
S = np.load(f"{OUT}/slices.npz", allow_pickle=True)
rng = np.random.default_rng(20261013)
NR = {"RNA": 2000, "meth": 2000, "chip": 2000, "proteins": 1000}
INP = {"RNA": "rna.txt", "meth": "meth.txt", "chip": "chip.txt", "proteins": "prot.txt"}
out = {}
for m, n in NR.items():
    nf = len(S[f"{m}_feats"]); excl = set(S[f"{m}_sel"].tolist())
    pool = np.array([i for i in range(nf) if i not in excl])
    pick = np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))
    V = pd.read_csv(f"{BASE}/{INP[m]}", sep=" ", header=None, usecols=list(pick),
                    dtype=np.float32 if m == "RNA" else np.float64, na_values="NA").values
    out[f"{m}_rsel"] = pick; out[f"{m}_rV"] = V
    print(m, V.shape)
np.savez_compressed(f"{OUT}/random_bg.npz", **out)
