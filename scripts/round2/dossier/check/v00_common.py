"""Independent verification of the component dossier (2026-10-13 round 2).
Own loaders; does not import the dossier's dc_common. Read-only on $G_ROS_ROSMAP_DIR, the corrected tables and the other
outputs. Writes only under $G_ROS_RESULTS_DIR/verify_dossier/out (aggregates only, no donor IDs) and
$G_ROS_RESULTS_DIR/verify_dossier/cache (per donor)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats as ss

ROS = cfg.ROS
SDA = cfg.SDA
AUD = cfg.AUD
R2 = cfg.WORK
VD = f"{R2}/verify_dossier"
OUT = cfg.out_dir(VD, "out"); CACHE = cfg.out_dir(cfg.WORK, "verify_dossier", "cache")
NEXT = cfg.TABLES
CTS = [l.strip() for l in open(f"{SDA}/cell_types.txt") if l.strip()]   # rna.txt block order
COMPS = [351, 2, 74, 87, 278, 177, 21, 172, 277, 545]


def donors():
    return [l.strip() for l in open(f"{SDA}/md_samples.txt") if l.strip()]


def scores():
    """Deposited consensus scores, C520 retired, C547/C548 from the audit (547 rows)."""
    S = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0)
    S.columns = S.columns.astype(str)
    assert list(S.columns) == donors()
    S.index = S.index.astype(int)
    S = S.drop(index=520)
    for k in (547, 548):
        S.loc[k] = np.load(f"{AUD}/c547/C{k}_A.npy")
    return S


def clin():
    c = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv")
    c.index = c.projid.astype(int).astype(str)
    return c.loc[donors()]


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(adj, 1); return out


def rint(x):
    x = np.asarray(x, float); n = len(x)
    return ss.norm.ppf((ss.rankdata(x) - 0.5) / n)        # Blom-type offset differs from the dossier's (0.375) on purpose


def read_rows(path, rows):
    keep = set(rows)
    return pd.read_csv(path, index_col=0, skiprows=lambda i: i != 0 and i not in keep)
