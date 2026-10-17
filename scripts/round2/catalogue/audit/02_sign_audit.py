"""Sign-alignment audit. The historical step aligned each block (A samples, B1 cell
types, X1..X4 features) to the cluster's first member independently, by the sign of
Pearson r. A member's reconstruction products (A*B1*X1 for RNA; A*Xj for 2-D omics)
are preserved only if flips are consistent: fA*fB*fX1 = +1 and fA*fXj = +1.
Pass 1 collects reference rows; pass 2 streams member rows, records flips, and
accumulates cluster sums under the historical scheme and a product-consistent one
(anchor A; Xj forced to fA for 2-D; for RNA anchor A and X1, force fB = fA*fX1).
Read-only on the archive; outputs to $G_ROS_AUDIT_DIR."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
BASE = cfg.SDA; OUT = cfg.out_dir(cfg.AUD); IT = 1500
cl = np.load(f"{OUT}/clusters_it{IT}.npy"); run_of_pos = np.load(f"{OUT}/run_of_pos_it{IT}.npy")
hist = pd.read_csv(f"{OUT}/members_buggy_max_it{IT}.csv", index_col=0); corr_ = pd.read_csv(f"{OUT}/members_corrected_max1_it{IT}.csv", index_col=0)
clusters = list(dict.fromkeys(list(hist.cluster) + list(corr_.cluster)))       # 548, historical order first
cidx = {c: k for k, c in enumerate(clusters)}
pos_in = {}                                                                     # position -> cluster slot
for c in clusters:
    for p in np.nonzero(cl == c)[0]: pos_in[int(p)] = cidx[c]
ref_pos = {cidx[c]: int(np.nonzero(cl == c)[0][0]) for c in clusters}         # first member = iloc[0]
starts = {i: int(np.nonzero(run_of_pos == i)[0][0]) for i in range(1, 11)}      # 2000 comps/run, none dropped

def rows_of(path, wanted):  # wanted: {line_no: key}; yields (key, vector)
    with open(path) as f:
        for ln, line in enumerate(f):
            if ln in wanted: yield wanted[ln], np.array(line.split(), dtype=np.float64)

def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.nan if na == 0 or nb == 0 else float(a @ b / (na * nb))

def collect_refs(args):
    j, i = args; s = starts[i]
    wanted = {p - s: k for k, p in ref_pos.items() if run_of_pos[p] == i}
    return dict(rows_of(f"{BASE}/results{i}/it{IT}/X{j}", wanted))

def member_pass(args):
    j, i, refs = args; s = starts[i]
    wanted = {p - s: p for p in pos_in if run_of_pos[p] == i}
    out = {}
    for p, v in rows_of(f"{BASE}/results{i}/it{IT}/X{j}", wanted):
        k = pos_in[p]; r = corr(refs[k], v); out[p] = (k, r, v)
    return out

if __name__ == "__main__":
    t0 = time.time()
    A = {i: np.loadtxt(f"{BASE}/results{i}/it{IT}/A").T for i in range(1, 11)}
    Bm = {i: np.loadtxt(f"{BASE}/results{i}/it{IT}/B1").T for i in range(1, 11)}
    row = lambda M, p: M[run_of_pos[p]][p - starts[run_of_pos[p]]]
    fA, fB = {}, {}
    for p, k in pos_in.items():
        ra = corr(row(A, ref_pos[k]), row(A, p)); rb = corr(row(Bm, ref_pos[k]), row(Bm, p))
        fA[p] = -1 if (ra == ra and ra < 0) else 1; fB[p] = (-1 if (rb == rb and rb < 0) else 1, rb)
    recs = {p: dict(pos=p, run=int(run_of_pos[p]), slot=k, cluster=int(clusters[k]), fA=fA[p], fB=fB[p][0], rB=fB[p][1]) for p, k in pos_in.items()}
    nF = {}; Bsum_h = np.zeros((len(clusters), 6)); Bsum_c = np.zeros((len(clusters), 6))
    sums = {}
    for j in (1, 2, 3, 4):
        with Pool(int(cfg.THREADS)) as pool:
            refs = {}
            for d in pool.map(collect_refs, [(j, i) for i in range(1, 11)]): refs.update(d)
            F = len(next(iter(refs.values()))); nF[j] = F
            Sh = np.zeros((len(clusters), F)); Sc = np.zeros((len(clusters), F))
            for d in pool.imap_unordered(member_pass, [(j, i, refs) for i in range(1, 11)]):
                for p, (k, r, v) in d.items():
                    fX = -1 if (r == r and r < 0) else 1
                    recs[p][f"fX{j}"] = fX; recs[p][f"rX{j}"] = r; recs[p][f"normX{j}"] = float(np.linalg.norm(v))
                    Sh[k] += fX * v
                    if j == 1:
                        Sc[k] += fX * v                               # RNA: anchor X1 own sign
                        fBc = fA[p] * fX                              # force B so A*B*X1 preserved
                        b = row(Bm, p); Bsum_h[k] += recs[p]["fB"] * b; Bsum_c[k] += fBc * b
                    else:
                        Sc[k] += fA[p] * v                            # 2-D: force X to A's sign
        cnt = np.bincount([pos_in[p] for p in pos_in], minlength=len(clusters))[:, None]
        sums[j] = (Sh / cnt, Sc / cnt)
        np.save(f"{OUT}/X{j}_mean_hist.npy", sums[j][0]); np.save(f"{OUT}/X{j}_mean_cons.npy", sums[j][1])
        print(f"X{j} done {time.time()-t0:.0f}s", flush=True)
    cnt = np.bincount([pos_in[p] for p in pos_in], minlength=len(clusters))[:, None]
    np.save(f"{OUT}/B1_mean_hist.npy", Bsum_h / cnt); np.save(f"{OUT}/B1_mean_cons.npy", Bsum_c / cnt)
    pd.DataFrame(recs.values()).to_csv(f"{OUT}/member_flips.csv", index=False)
    json.dump(dict(clusters=[int(c) for c in clusters], n_hist=len(hist), nF=nF), open(f"{OUT}/sign_audit_meta.json", "w"))
    print(f"all done {time.time()-t0:.0f}s")
