"""Independent rebuild of ROSMAP SDA cluster membership (it1500), read-only on the archive.
Pools all non-zero sample vectors (A.T rows) of results1..10, complete linkage on 1-|Pearson r|,
fcluster 0.4, Counter (insertion) order, size>=5 and >=5 runs counted with the original
off-by-one labels (run1: 0..n-1; run i: range(max, max+n)). Sample flips by pearsonr<0 vs first member.
Checks the rebuilt sample consensus against samples_final.csv. Writes members.csv (component,pos,run,slot,fA)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import time
from collections import Counter
import numpy as np, pandas as pd
import scipy.cluster.hierarchy as hac
from scipy import stats
from scipy.spatial.distance import squareform

BASE = cfg.SDA
OUT = cfg.out_dir(cfg.AUD, "ds1verify_cosB")
t0 = time.time()
vecs, run_of, slot_of, nper = [], [], [], []
for i in range(1, 11):
    A = pd.read_csv(f"{BASE}/results{i}/it1500/A", sep=" ", header=None).values  # 276 x 2000
    S = A.T
    keep = np.where((S != 0).any(axis=1))[0]
    vecs.append(S[keep]); run_of += [i] * len(keep); slot_of += list(keep); nper.append(len(keep))
S = np.vstack(vecs); run_of = np.array(run_of); slot_of = np.array(slot_of)
np.save(f"{OUT}/pooled_A.npy", S); np.save(f"{OUT}/run_of.npy", run_of); np.save(f"{OUT}/slot_of.npy", slot_of)
print("pooled", S.shape, nper, flush=True)

Zs = S - S.mean(1, keepdims=True)
Zs /= np.linalg.norm(Zs, axis=1, keepdims=True)
D = Zs @ Zs.T
np.abs(D, out=D); np.subtract(1.0, D, out=D)
np.fill_diagonal(D, 0.0); np.clip(D, 0.0, None, out=D)
d = squareform(D, checks=False); del D
print(f"dist {time.time()-t0:.0f}s", flush=True)
L = hac.linkage(d, method="complete"); del d
cl = hac.fcluster(L, 0.4, criterion="distance")
np.save(f"{OUT}/clusters.npy", cl)
print(f"linkage {time.time()-t0:.0f}s; {len(set(cl))} flat clusters", flush=True)

# off-by-one labels per run (pre-Dec-2024 formula)
labs, mx = [], None
for i, n in enumerate(nper):
    lab = list(range(n)) if i == 0 else list(range(mx, mx + n))
    labs.append(set(lab)); mx = max(lab)

size = Counter(cl)
rows, recs = [], []
comp = 0
for c in size:
    if size[c] < 5: continue
    pos = np.nonzero(cl == c)[0]
    counted = sum(1 for lab in labs if set(pos).intersection(lab))
    if counted < 5: continue
    comp += 1
    V = S[pos].copy()
    fA = np.ones(len(pos), int)
    for r in range(len(pos)):
        if stats.pearsonr(V[0], V[r])[0] < 0: fA[r] = -1
    rows.append((V * fA[:, None]).mean(0))
    for p, f in zip(pos, fA):
        recs.append(dict(component=comp, cluster=int(c), pos=int(p), run=int(run_of[p]), slot=int(slot_of[p]), fA=int(f)))
R = np.array(rows)
pub = pd.read_csv(f"{BASE}/samples_final.csv", index_col=0)
print("components", len(R), "published", pub.shape, "index", list(pub.index[:3]), "...", list(pub.index[-3:]))
diff = np.abs(R - pub.values)
print("max abs diff vs samples_final:", diff.max(), "rows > 1e-9:", int((diff.max(1) > 1e-9).sum()))
pd.DataFrame(recs).to_csv(f"{OUT}/members.csv", index=False)
print(f"done {time.time()-t0:.0f}s; members {len(recs)}")
