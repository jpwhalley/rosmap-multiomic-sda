"""Shared paths and loaders for the protein-influence (fixed-loadings score re-estimation) analysis.
Read-only on the SDA archive and on other outputs; writes only under $G_ROS_RESULTS_DIR/protein_influence/cache (per-donor
values) and $G_ROS_RESULTS_DIR/protein_influence/{results,logs} (aggregates only, no donor IDs)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, time, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(cfg.WORK, "protein_influence")
CACHE = cfg.out_dir(cfg.WORK, "protein_influence", "cache")
RES = cfg.out_dir(ROOT, "results")
LOGS = cfg.out_dir(ROOT, "logs")
SDA = cfg.SDA
ROS = cfg.ROS
VCACHE = f"{cfg.WORK}/verify_source_panels/cache"
AUDIT = cfg.AUD
REL = cfg.CATALOGUE   # outputs of scripts/07_cluster_components.py
N = 276
K = 2000
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
MODS = ["RNA", "meth", "chip", "proteins"]
HIGHLIGHT = [351, 2, 74, 87, 278, 177, 21, 172, 277, 545]


def tlog(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def read_mat(path, dtype=np.float64):
    """space-separated SDA output / input matrix (NA -> NaN)."""
    return pd.read_csv(path, sep=" ", header=None, na_values="NA", dtype=dtype, engine="c").values


def load_run(r, it=1500, which=("A", "B1", "X1", "X2", "X3", "X4"), dtype=np.float64):
    d = {}
    for w in which:
        p = f"{SDA}/results{r}/it{it}/{w}"
        M = read_mat(p, dtype)
        # some SDA files end lines with a trailing space -> an all-NaN last column
        if M.shape[1] > 1 and np.isnan(M[:, -1]).all():
            M = M[:, :-1]
        d[w] = M
    return d


def get_mat(r, w, dtype=np.float32):
    """run r it1500 matrix w (A, B1, X1..4, S1..4): cached .npy if present (run 1), else parsed from the archive text."""
    f = f"{CACHE}/run{r}/{w}.npy"
    if os.path.exists(f):
        return np.load(f)
    return load_run(r, which=(w,), dtype=np.float64 if w in ("A", "B1") else dtype)[w]


def load_inputs():
    """inputs exactly as fed (verified caches of rna/meth/chip/prot.txt; see 00_check_inputs.py)."""
    Y = {m: np.load(f"{VCACHE}/{m}_input.npy", mmap_mode="r") for m in MODS}
    return Y
