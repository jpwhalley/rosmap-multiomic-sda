"""Re-derive the ROSMAP SDA stable-component set from the archived it1500 sample
loadings (A only), under the pre-Dec-2024 index formula (range(max, max+n)) and
the corrected one (range(max+1, max+1+n)). Read-only on the archive."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, time, json, hashlib
from collections import Counter
import numpy as np, pandas as pd
import scipy.cluster.hierarchy as hac
from scipy import stats
from scipy.spatial.distance import squareform

BASE = cfg.SDA
OUT = cfg.out_dir(cfg.AUD)
IT = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
t0 = time.time()

blocks, run_of_pos, kept_per_run = [], [], []
for i in range(1, 11):
    A = np.loadtxt(f"{BASE}/results{i}/it{IT}/A")          # 276 x 2000
    S = A.T                                                # comps x samples
    keep = np.where((S != 0).any(axis=1))[0]
    blocks.append(S[keep]); kept_per_run.append(keep); run_of_pos += [i] * len(keep)
X = np.vstack(blocks); run_of_pos = np.array(run_of_pos); N = len(X)
print(f"it{IT}: pooled {N} non-zero comps; per run {[len(k) for k in kept_per_run]}", flush=True)

# 1 - |Pearson| distance, vectorised
Z = X - X.mean(1, keepdims=True); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
C = Z @ Z.T; np.abs(C, out=C); D = 1.0 - C; del C
np.fill_diagonal(D, 0.0); np.maximum(D, 0.0, out=D)
d = squareform(D, checks=False); del D
print(f"distances done {time.time()-t0:.0f}s", flush=True)
L = hac.linkage(d, method="complete"); del d
cl = hac.fcluster(L, 0.4, criterion="distance")
np.save(f"{OUT}/linkage_it{IT}.npy", L); np.save(f"{OUT}/clusters_it{IT}.npy", cl); np.save(f"{OUT}/run_of_pos_it{IT}.npy", run_of_pos)
print(f"linkage done {time.time()-t0:.0f}s; {len(set(cl))} flat clusters", flush=True)

# label lists per run, both formulas
def labels(buggy):
    out, mx = [], None
    for i, k in enumerate(kept_per_run):
        n = len(k)
        if i == 0: lab = list(range(n))
        else: lab = list(range(mx, mx + n)) if buggy else list(range(mx + 1, mx + 1 + n))
        out.append(lab); mx = max(lab)
    return out

def build(buggy):
    inds = labels(buggy); idx = np.concatenate([np.array(x) for x in inds])
    df = pd.DataFrame(X, index=idx)
    size = Counter(cl); rows, meta = [], []
    for c in size:                                    # Counter insertion order, as original
        if size[c] < 5: continue
        pos = set(np.nonzero(cl == c)[0])            # positions (original line 163)
        counted = sum(1 for ind in inds if pos.intersection(ind))
        true_runs = len(set(run_of_pos[cl == c]))
        if counted < 5: continue
        t = df[pd.Series(cl == c, index=df.index.values).values].copy()
        lab = t.index.values
        dup = len(lab) != len(set(lab))
        vals = t.values.copy()
        for r in range(len(vals)):
            if stats.pearsonr(vals[0], vals[r])[0] < 0: vals[r] *= -1
        rows.append(vals.mean(0))
        meta.append(dict(cluster=int(c), size=int(size[c]), counted_runs=counted, true_runs=true_runs, dup_label_in_cluster=dup))
    return np.array(rows), pd.DataFrame(meta)

dep = pd.read_csv(f"{BASE}/samples_final.csv", index_col=0)
res = {}
for name, buggy in (("buggy_max", True), ("corrected_max1", False)):
    R, M = build(buggy); M.index = range(1, len(M) + 1); M.index.name = f"{name}_id"
    M.to_csv(f"{OUT}/members_{name}_it{IT}.csv")
    entry = dict(n=len(M))
    if len(R) == len(dep):
        diff = np.abs(R - dep.values); entry["max_abs_diff_vs_deposit"] = float(diff.max())
        entry["rows_mismatch_gt_1e-9"] = int((diff.max(1) > 1e-9).sum())
    res[name] = entry
    print(name, entry, flush=True)
json.dump(res, open(f"{OUT}/summary_it{IT}.json", "w"), indent=1)
print(f"done {time.time()-t0:.0f}s")
