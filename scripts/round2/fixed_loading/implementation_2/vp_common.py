"""Shared paths/loaders for the independent verification of the protein-influence analysis.
Read-only on the SDA archive and other outputs; writes only under $G_ROS_RESULTS_DIR/verify_protein_influence/cache
(per-donor values) and $G_ROS_RESULTS_DIR/verify_protein_influence/{results,logs} (aggregates only, no donor IDs)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, time, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(cfg.WORK, "verify_protein_influence")
CACHE = cfg.out_dir(cfg.WORK, "verify_protein_influence", "cache")
RES = cfg.out_dir(ROOT, "results")
LOGS = cfg.out_dir(ROOT, "logs")
SDA = cfg.SDA
ROS = cfg.ROS
REL = cfg.CATALOGUE   # outputs of scripts/07_cluster_components.py
N, K = 276, 2000
HL = [351, 2, 74, 87, 278, 177, 21, 172, 277, 545]

_t0 = time.time()
def log(*a):
    print(f"[{time.time() - _t0:7.1f}s]", *a, flush=True)


def read_txt(path, dtype=np.float64):
    M = pd.read_csv(path, sep=" ", header=None, na_values="NA", dtype=dtype, engine="c").values
    if M.shape[1] > 1 and np.isnan(M[:, -1]).all():      # trailing space -> empty last column
        M = M[:, :-1]
    return M


_MEM = {}
def run_mat(r, w, dtype=np.float64):
    """results{r}/it1500/{w}. Uses CACHE/run{r}/{w}.npy if present (run 3 only); otherwise parses the archive text
    once per process and keeps it in memory (X/S as float32: the text carries 6 significant digits)."""
    f = f"{CACHE}/run{r}/{w}.npy"
    if os.path.exists(f):
        return np.load(f).astype(dtype, copy=False)
    if (r, w) not in _MEM:
        path = f"{SDA}/results{r}/it1500/{w}"
        if w in ("A", "B1"):
            _MEM[(r, w)] = read_txt(path)
        else:                                            # chunked parse straight into float32 (keeps peak memory low)
            parts = []
            for ch in pd.read_csv(path, sep=" ", header=None, na_values="NA", dtype=np.float32, engine="c", chunksize=250):
                v = ch.values
                if v.shape[1] > 1 and np.isnan(v[:, -1]).all(): v = v[:, :-1]
                parts.append(np.ascontiguousarray(v))
            _MEM[(r, w)] = np.vstack(parts); del parts
    return _MEM[(r, w)].astype(dtype, copy=True)


def inputs():
    return {k: np.load(f"{CACHE}/in_{k}.npy") for k in ("rna", "meth", "chip", "prot")}
