"""Shared definitions for the decomposition-structure analysis (npj Aging round 2, 2026-10-13).

Read-only on: the SDA archive ($G_ROS_ROSMAP_DIR/sda), SEA-AD ($G_ROS_SEAAD_DIR/sda), the 2026-10-08 audit and round-2
outputs, and the corrected tables ($G_ROS_TABLES_DIR). Writes only under $G_ROS_RESULTS_DIR/revision_analyses/structure/. Per-donor values stay in cache/ and private/;
out/ holds aggregates only (no donor IDs).

CONCENTRATION (one definition, used throughout). For component k with sample scores s_1..s_n over the n = 276 donors,
let d_i = s_i - mean(s) (centred scores).
    top-donor share  = max_i d_i^2 / sum_i d_i^2          (1/n = perfectly even; 1 = one donor carries everything)
    participation ratio PR = (sum_i d_i^2)^2 / sum_i d_i^4  (effective number of donors; 1 = one donor, n = all equal)
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats as ss

ROS = cfg.ROS
SDA = cfg.SDA
SEA = f"{cfg.SEAAD}/sda"
AUD = cfg.AUD
R2 = cfg.WORK
NEXT = cfg.TABLES
DS1 = f"{NEXT}/Data_S1_corrected_547.csv"
BCOR = f"{NEXT}/cell_type_loadings_corrected.csv"
GI = f"{cfg.SEAAD}/gene_info.csv"  # read-only gene names
HERE = os.path.dirname(os.path.abspath(__file__))
ST = f"{R2}/revision_analyses/structure"
OUT, CACHE, PANELS, PRIV = (f"{ST}/{d}" for d in ("out", "cache", "panels", "private"))
for d in (OUT, CACHE, PANELS, PRIV):
    cfg.out_dir(d)
FOCUS = [74, 278, 172, 277, 21]                 # retained findings
OUTGOING = [351, 2, 87, 177]                    # leave the main argument
OTHER_NAMED = [545]
OLD_PRIORITISED = [2, 74, 76, 87, 99, 130, 177, 291, 345, 351, 416]
MODS = ["RNA", "meth", "chip", "proteins"]
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]   # rna.txt block order (cell_types.txt)


def samples():
    return [l.strip() for l in open(f"{SDA}/md_samples.txt") if l.strip()]


def scores547():
    """547-component sample scores (C520 retired; C547/C548 appended from the 2026-10-08 audit). index = component."""
    dep = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0)
    dep.columns = dep.columns.astype(str)
    assert list(dep.columns) == samples() and list(dep.index) == list(range(1, 547))
    S = dep.drop(index=520).copy()
    for k in (547, 548):
        S.loc[k] = np.load(f"{AUD}/c547/C{k}_A.npy")
    assert S.shape == (547, 276)
    return S


def concentration(a):
    """Centred top-donor share, participation ratio, top-3 share, robust z of the top two donors, top index."""
    a = np.asarray(a, float)
    d = a - a.mean(); d2 = d ** 2; tot = d2.sum()
    o = np.argsort(-d2)
    med = np.median(a); mad = 1.4826 * np.median(np.abs(a - med))
    rz = np.abs(a - med) / mad
    return dict(top_share=d2[o[0]] / tot, second_share=d2[o[1]] / tot, top3_share=d2[o[:3]].sum() / tot,
                PR=tot ** 2 / (d2 ** 2).sum(), top_rz=rz[o[0]], second_rz=rz[o[1]],
                excess_kurtosis=float(ss.kurtosis(a)), top_idx=int(o[0]))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o]
    adj = np.minimum.accumulate((r * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(adj, 1); return out


def read_rows(path, rows):
    """Read selected 1-based data rows (plus header) of a large csv."""
    keep = set(rows)
    return pd.read_csv(path, index_col=0, skiprows=lambda i: i != 0 and i not in keep)
