"""Minor 1 check: it1500 -> it2000 sample-loading direction stability (runs 8, 10), three denominators,
to reconcile FINDINGS §1 ("93-94 % of components have |r| > 0.99") with critique2_ckpt_c278/a_summary.txt."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
B = cfg.SDA; AUD = cfg.AUD
cl = np.load(f"{AUD}/clusters_it1500.npy"); ro = np.load(f"{AUD}/run_of_pos_it1500.npy")
mem = pd.read_csv(f"{AUD}/members_buggy_max_it1500.csv")
stable_cl = set(mem.cluster) - {int(mem.loc[mem.buggy_max_id == 520, "cluster"].iloc[0])}
corr = pd.read_csv(f"{AUD}/members_corrected_max1_it1500.csv"); stable_cl |= set(corr.cluster)   # adds 1465, 716
print("stable clusters (547 catalogue):", len(stable_cl))
for run in (8, 10):
    A0 = np.loadtxt(f"{B}/results{run}/it1500/A").T; A1 = np.loadtxt(f"{B}/results{run}/it2000/A").T
    nz0 = (A0 != 0).any(1); nz1 = (A1 != 0).any(1)
    keep = nz0 & nz1
    def r(i):
        a, b = A0[i] - A0[i].mean(), A1[i] - A1[i].mean(); return abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
    rr = np.full(2000, np.nan); rr[keep] = [r(i) for i in np.nonzero(keep)[0]]
    # members of stable clusters in this run: pooled positions -> slot index (pooled = non-null comps at it1500, in order)
    slots0 = np.nonzero(nz0)[0]; pos = np.nonzero(ro == run)[0]
    member_slots = slots0[[j for j, p in enumerate(pos) if cl[p] in stable_cl]]
    print(f"run {run}: non-null at 1500 {nz0.sum()}, at 2000 {nz1.sum()}, both {keep.sum()}")
    print(f"  |r|>0.99 of all 2000 slots: {(rr > 0.99).sum()}/2000 = {(rr > 0.99).mean():.1%}; of non-null: {(rr[keep] > 0.99).mean():.1%}; median {np.nanmedian(rr):.4f}")
    m = rr[member_slots]
    print(f"  members of the 547 stable clusters: n={len(member_slots)}, |r|>0.99 {np.mean(m > 0.99):.1%}, median {np.nanmedian(m):.4f}, min {np.nanmin(m):.3f}")
