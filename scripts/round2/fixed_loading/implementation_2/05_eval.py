"""Calibration of a fit (Ahat scenario a vs published A) on the catalogue members of run r; aggregates only.
usage: 05_eval.py r fitname"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, numpy as np, pandas as pd
from vp_common import *
r = int(sys.argv[1]); fn = sys.argv[2]
A = run_mat(r, "A"); F = np.load(f"{CACHE}/run{r}/{fn}.npz")
m = pd.read_csv(f"{REL}/cluster_members.csv"); m = m[m.run == r]
idx = m.sda_index.values


def cc(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    return (P * Q).sum(0) / np.sqrt((P * P).sum(0) * (Q * Q).sum(0))


Aa = F["Aa"]
rc = cc(Aa[:, idx], A[:, idx]); rel = np.linalg.norm(Aa[:, idx] - A[:, idx], axis=0) / np.linalg.norm(A[:, idx], axis=0)
hl = m.component_id.isin(HL).values
print(f"run {r} {fn}: members {len(idx)}  r median {np.median(rc):.4f} 1st {np.quantile(rc, .01):.4f} 5th {np.quantile(rc, .05):.4f} min {rc.min():.4f} | "
      f"rel median {np.median(rel):.3f} 95th {np.quantile(rel, .95):.3f} | highlighted ({hl.sum()}): r min {rc[hl].min():.4f} rel max {rel[hl].max():.3f}")
for k in HL:
    s = m.component_id.values == k
    if s.any():
        print(f"   C{k}: r {np.round(rc[s], 4).tolist()} rel {np.round(rel[s], 3).tolist()}")
