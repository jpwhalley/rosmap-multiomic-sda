"""Independent re-derivation of the SEA-AD DLPFC SDA consensus exactly as
dlpfc2_cluster.py does it (it3000 A/B1/X1/S1 of results1..10), reproduction of the
deposited samples/treatment/genes/pip_final, then the corrected run count.
Read-only on the archive; writes only to $G_ROS_AUDIT_DIR/seaad141."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, json, time
from collections import Counter
import numpy as np, pandas as pd
import scipy.cluster.hierarchy as hac
from scipy import stats
from scipy.spatial.distance import squareform

D = f"{cfg.SEAAD}/sda"
OUT = cfg.out_dir(cfg.AUD, "seaad141")
t0 = time.time()
samps = open(f"{D}/md_samples.txt").read().splitlines()
cts = open(f"{D}/cell_types.txt").read().splitlines()
rna_genes = open(f"{D}/genes.txt").read().splitlines()

# ---- pool A (and B1) exactly as the script: drop all-zero A rows (comps), keep order
S_blocks, T_blocks, keep_per_run, run_of_pos, slot_of_pos = [], [], [], [], []
for i in range(1, 11):
    A = pd.read_csv(f"{D}/results{i}/it3000/A", sep=" ", header=None).T
    A.columns = samps
    keep = A.index.values[(A.T != 0).any().values]
    B = pd.read_csv(f"{D}/results{i}/it3000/B1", sep=" ", header=None).T
    B.columns = cts
    keepB = B.index.values[(B.T != 0).any().values]
    assert np.array_equal(keep, keepB), i   # script relies on equal row counts
    S_blocks.append(A.loc[keep].values); T_blocks.append(B.loc[keepB].values)
    keep_per_run.append(keep); run_of_pos += [i] * len(keep); slot_of_pos += list(keep)
    print(f"run {i}: {A.shape[0]} comps, {len(keep)} non-zero", flush=True)
S = np.vstack(S_blocks); T = np.vstack(T_blocks)
run_of_pos = np.array(run_of_pos); slot_of_pos = np.array(slot_of_pos); N = len(S)

# ---- historical (off-by-one) labels: run1 range(n1); run k range(max, max+n_k)
hist_labels, mx = [], None
for k, keep in enumerate(keep_per_run):
    n = len(keep)
    lab = list(range(n)) if k == 0 else list(range(mx, mx + n))
    hist_labels.append(lab); mx = max(lab)
label_of_pos = np.concatenate([np.array(x) for x in hist_labels])

# ---- 1-|r| complete linkage, cut at 0.4
Zs = S - S.mean(1, keepdims=True); Zs /= np.linalg.norm(Zs, axis=1, keepdims=True)
C = Zs @ Zs.T; np.abs(C, out=C); Dm = 1.0 - C; del C
np.fill_diagonal(Dm, 0.0); np.maximum(Dm, 0.0, out=Dm)
L = hac.linkage(squareform(Dm, checks=False), method="complete"); del Dm
cl = hac.fcluster(L, 0.4, criterion="distance")
h = L[:, 2]
print(f"pooled N={N}; {len(set(cl))} flat clusters; closest merge height to 0.4: "
      f"{h[np.argmin(np.abs(h-0.4))]!r} (|diff|={np.min(np.abs(h-0.4)):.3e}) [{time.time()-t0:.0f}s]", flush=True)
np.save(f"{OUT}/clusters.npy", cl); np.save(f"{OUT}/run_of_pos.npy", run_of_pos)
np.save(f"{OUT}/slot_of_pos.npy", slot_of_pos); np.save(f"{OUT}/linkage.npy", L)

# ---- keep rule, Counter insertion order, historical sequential IDs
size = Counter(cl)
meta, hist_id = [], 0
for c in size:
    pos = np.nonzero(cl == c)[0]
    true_runs = len(set(run_of_pos[pos]))
    if size[c] < 10 // 2:
        if true_runs >= 5: print("impossible", c)
        continue
    counted = sum(1 for ind in hist_labels if len(set(pos).intersection(ind)) > 0)
    hist_pass = counted >= 10 // 2
    if hist_pass: hist_id += 1
    labs = label_of_pos[pos]
    meta.append(dict(cluster=int(c), size=int(size[c]), counted_runs_hist=counted, true_runs=true_runs,
                     hist_pass=hist_pass, corrected_pass=true_runs >= 5,
                     historical_id=hist_id if hist_pass else None,
                     runs=" ".join(map(str, sorted(set(run_of_pos[pos])))),
                     dup_label_in_cluster=len(labs) != len(set(labs)),
                     positions=" ".join(map(str, pos))))
M = pd.DataFrame(meta)
M.to_csv(f"{OUT}/clusters_size_ge5.csv", index=False)
H = M[M.hist_pass].set_index("historical_id")
print(f"historical keep: {len(H)}; corrected keep: {int(M.corrected_pass.sum())}", flush=True)
print("hist pass but corrected fail:\n", M[M.hist_pass & ~M.corrected_pass].drop(columns="positions").to_string(), flush=True)
print("corrected pass but hist fail:\n", M[~M.hist_pass & M.corrected_pass].drop(columns="positions").to_string(), flush=True)
print("dup labels within a kept cluster:", int(H.dup_label_in_cluster.sum()), flush=True)

# ---- reproduce samples_final / treatment_final with the original flip/mean logic
def flip_mean(V):
    V = V.copy()
    for r in range(len(V)):
        if stats.pearsonr(V[0], V[r])[0] < 0: V[r] *= -1
    return V.mean(0)

members = {hid: np.array(list(map(int, H.loc[hid, "positions"].split()))) for hid in H.index}
sf = np.array([flip_mean(S[members[h_]]) for h_ in H.index])
tf = np.array([flip_mean(T[members[h_]]) for h_ in H.index])
res = {}
for name, R in (("samples_final", sf), ("treatment_final", tf)):
    dep = pd.read_csv(f"{D}/{name}.csv", index_col=0)
    assert list(dep.index) == list(H.index)
    d = np.abs(R - dep.values)
    res[name] = dict(n=len(dep), max_abs_diff=float(d.max()), rows_gt_1e9=int((d.max(1) > 1e-9).sum()))
    print(name, res[name], flush=True)

# ---- genes (X1, sign-aligned mean) and pip (S1, median), gathered run by run
need = {}  # pos -> (run, slot)
allpos = np.concatenate(list(members.values()))
Xrows, Prows = {}, {}
for i in range(1, 11):
    sel = [p for p in allpos if run_of_pos[p] == i]
    slots = slot_of_pos[sel]
    X = pd.read_csv(f"{D}/results{i}/it3000/X1", sep=" ", header=None)
    assert X.shape[1] == len(rna_genes)
    Xv = X.values[slots]; del X
    P = pd.read_csv(f"{D}/results{i}/it3000/S1", sep=" ", header=None)
    Pv = P.values[slots]; del P
    for j, p in enumerate(sel): Xrows[p] = Xv[j]; Prows[p] = Pv[j]
    print(f"X1/S1 run {i} gathered {len(sel)} rows [{time.time()-t0:.0f}s]", flush=True)
gf = np.array([flip_mean(np.array([Xrows[p] for p in members[h_]])) for h_ in H.index])
pf = np.array([np.median(np.array([Prows[p] for p in members[h_]]), axis=0) for h_ in H.index])
np.save(f"{OUT}/genes_final_repro.npy", gf); np.save(f"{OUT}/pip_final_repro.npy", pf)
for name, R in (("genes_final", gf), ("pip_final", pf)):
    dep = pd.read_csv(f"{D}/{name}.csv", index_col=0)
    assert list(dep.index) == list(H.index) and list(dep.columns) == rna_genes
    d = np.abs(R - dep.values)
    res[name] = dict(n=len(dep), max_abs_diff=float(d.max()), rows_gt_1e9=int((d.max(1) > 1e-9).sum()))
    print(name, res[name], flush=True)

# ---- corrected catalogue: historical IDs kept, failing ones retired
fail = H[~H.corrected_pass]
res["pooled_N"] = int(N); res["n_flat_clusters"] = int(len(set(cl)))
res["closest_merge_to_0.4"] = float(np.min(np.abs(h - 0.4)))
res["historical_n"] = int(len(H)); res["corrected_n_true_runs"] = int(M.corrected_pass.sum())
res["retired_ids"] = {int(k): dict(size=int(v.size), counted_runs_hist=int(v.counted_runs_hist),
                                   true_runs=int(v.true_runs), runs=v.runs) for k, v in fail.iterrows()}
res["new_clusters_qualifying"] = M[~M.hist_pass & M.corrected_pass].drop(columns="positions").to_dict("records")
res["hist_pass_counted_vs_true"] = H[["size", "counted_runs_hist", "true_runs"]].reset_index().to_dict("records")
json.dump(res, open(f"{OUT}/recluster_summary.json", "w"), indent=1, default=int)
print(f"done {time.time()-t0:.0f}s")
