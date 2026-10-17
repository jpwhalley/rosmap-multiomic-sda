"""VERIFY part 2: compare own concentration/PIP numbers with structure/out, re-derive class counts, distribution,
reference simulation (own seed), modality/recurrence relations (run counts recounted from the pooled cluster labels)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss

V = cfg.out_dir(cfg.WORK, "revision_analyses", "verify_structure"); cfg.out_dir(V, "out")
cfg.out_dir(V, "private")
ST = f"{cfg.WORK}/revision_analyses/structure"
AUD = cfg.AUD
R = pd.read_csv(f"{V}/private/v01_core.csv", index_col=0)
M = pd.read_csv(f"{ST}/out/01_master_547.csv", index_col=0)
MODS = ["RNA", "meth", "chip", "proteins"]
print("max |top_share diff| vs structure:", float((R.top_share - M.top_share).abs().max()),
      "| max rel PR diff:", float(((R.PR - M.PR) / M.PR).abs().max()),
      "| PIP counts identical:", bool((R[MODS].values == M[MODS].values).all()))
top_st = np.load(f"{cfg.WORK}/revision_analyses/structure/private/01_top_donor_index.npy")
print("top donor index identical:", bool((R.top_idx.values == top_st).all()))

cls = np.where(R.top_share < 0.10, "distributed", np.where(R.top_share < 0.50, "few-donor", "concentrated"))
R["cls"] = cls
ORDER = ["distributed", "few-donor", "concentrated"]
print("\nclass counts:", {c: int((R.cls == c).sum()) for c in ORDER})
print(">0.8:", int((R.top_share > 0.8).sum()), "| >0.95:", int((R.top_share > 0.95).sum()), "| 0.5-0.8:", int(R.top_share.between(0.5, 0.8).sum()))
for nm, sel in [(">=0.5", R.top_share >= 0.5), (">0.8", R.top_share > 0.8)]:
    t = R.top_idx[sel]
    print(f"{nm}: n {int(sel.sum())}, distinct top donors {t.nunique()}, max comps per donor {t.value_counts().max()}")
for c in ORDER:
    s = R[R.cls == c]
    print(f"PR {c}: median {s.PR.median():.1f} [{s.PR.min():.1f}, {s.PR.max():.1f}]")
bins = [0, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 0.90, 0.95, 1.0]
print(pd.cut(R.top_share, bins).value_counts().sort_index().to_string())
for k in [74, 278, 172, 277, 21, 545, 351, 2, 87, 177, 83, 239, 484]:
    print(f"  C{k}: top {R.top_share[k]:.4f} second {R.second_share[k]:.3f} top3 {R.top3_share[k]:.3f} PR {R.PR[k]:.1f} share_excl_top {R.share_excl_top[k]:.3f}")

# ---- reference simulation (own seed)
rng = np.random.default_rng(777)
n, B = 276, 20000
def sim(A):
    D = A - A.mean(1, keepdims=True); D2 = D ** 2; t = D2.sum(1)
    return D2.max(1) / t, t ** 2 / (D2 ** 2).sum(1)
print("\nreference (n=276, B=20000, seed 777):")
for nm, A in [("Normal", rng.standard_normal((B, n))), ("t5", rng.standard_t(5, (B, n))), ("t3", rng.standard_t(3, (B, n))),
              ("Laplace", rng.laplace(size=(B, n))), ("lognormal(0,1)", rng.lognormal(0, 1, (B, n))), ("t10", rng.standard_t(10, (B, n)))]:
    ts, pr = sim(A)
    print(f"  {nm:15s} share q50 {np.median(ts):.4f} q99.9 {np.quantile(ts, .999):.4f} P(>=0.10) {np.mean(ts >= 0.10):.5f} "
          f"P(>=0.5) {np.mean(ts >= 0.5):.5f} PR median {np.median(pr):.1f} mean {pr.mean():.1f}")
print("  theory Gaussian PR ~ n/3 =", n / 3)

# where does the catalogue's broad body sit vs Gaussian? (distributed class only)
d = R[R.cls == "distributed"]
print(f"\ndistributed: median top share {d.top_share.median():.4f}, frac > Gaussian q99.9 (0.072): {np.mean(d.top_share > 0.072):.3f}")

# ---- modalities
print("\nn_mod by class:")
print(pd.crosstab(R.cls, R.n_mod).loc[ORDER].to_string())
ge2 = R.n_mod >= 2
print(">=2 modalities:", int(ge2.sum()), "| concentrated", int((ge2 & (R.cls == 'concentrated')).sum()),
      "| distributed", list(R.index[ge2 & (R.cls == 'distributed')]), "| few", list(R.index[ge2 & (R.cls == 'few-donor')]))
ge3 = R.n_mod >= 3
print(">=3 modalities:", int(ge3.sum()), "| concentrated", int((ge3 & (R.cls == 'concentrated')).sum()), "| non-conc", list(R.index[ge3 & (R.cls != 'concentrated')]))
dist = R[R.cls == "distributed"]
print("distributed with exactly 1 modality:", int((dist.n_mod == 1).sum()), "; of which RNA:", int(((dist.n_mod == 1) & (dist.RNA > 0)).sum()),
      "; distributed with RNA at all:", int((dist.RNA > 0).sum()))
print("Spearman(log share, n_mod):", round(ss.spearmanr(np.log10(R.top_share), R.n_mod)[0], 3))
mask = np.load(f"{V}/private/v01_donor_modality_mask.npy")
conc = R[R.cls == "concentrated"]
act = conc[MODS].values > 0; dh = mask[conc.top_idx.values]
print(f"concentrated active slots {int(act.sum())}, top donor has data {int((act & dh).sum())}; per modality",
      [(m, int(act[:, i].sum()), int((act[:, i] & dh[:, i]).sum())) for i, m in enumerate(MODS)])
# does the top donor of a concentrated component have more assays than average?
nass = mask.sum(1)
print(f"assays per donor: all donors mean {nass.mean():.2f}; top donors of concentrated comps mean {nass[conc.top_idx.values].mean():.2f}; "
      f"of >=2-modality concentrated comps {nass[conc.top_idx.values[conc.n_mod.values >= 2]].mean():.2f}")
# n_mod vs number of assays the top donor has
print("crosstab n_mod (rows) vs top-donor n assays (cols), concentrated:")
print(pd.crosstab(conc.n_mod.values, nass[conc.top_idx.values]).to_string())

# ---- run recurrence recount from pooled cluster labels
clu = np.load(f"{AUD}/clusters_it1500.npy"); run = np.load(f"{AUD}/run_of_pos_it1500.npy")
mb = pd.read_csv(f"{AUD}/members_buggy_max_it1500.csv").set_index("buggy_max_id")
mc = pd.read_csv(f"{AUD}/members_corrected_max1_it1500.csv").set_index("cluster")
cid = {k: int(mb.loc[k, "cluster"]) for k in R.index if k <= 546}
cid[547], cid[548] = 1465, 716
R["runs"] = [len(set(run[clu == cid[k]])) for k in R.index]
R["members"] = [int((clu == cid[k]).sum()) for k in R.index]
rc = pd.read_csv(f"{cfg.WORK}/r1_numbers/out/run_counts_547.csv", index_col=0)
print("\nruns recount == r1_numbers:", bool((R.runs.values == rc.loc[R.index, "true_runs"].values).all()),
      "; members == size:", bool((R.members.values == rc.loc[R.index, "size"].values).all()))
print(pd.crosstab(R.cls, R.runs).loc[ORDER].to_string())
for c in ORDER:
    s = R[R.cls == c]
    print(f"  {c}: all-10 {int((s.runs == 10).sum())}/{len(s)} ({np.mean(s.runs == 10):.2f}); median runs {s.runs.median()}; median members {s.members.median()}")
print("total in all 10 runs:", int((R.runs == 10).sum()))
print("Spearman(log share, runs):", round(ss.spearmanr(np.log10(R.top_share), R.runs)[0], 3),
      "; (log share, members):", round(ss.spearmanr(np.log10(R.top_share), R.members)[0], 3))
# within single-modality components only (removes the modality confound)
one = R.n_mod == 1
print("single-modality comps: all-10 runs by class", {c: f"{int(((R.cls == c) & one & (R.runs == 10)).sum())}/{int(((R.cls == c) & one).sum())}" for c in ORDER})
R.to_csv(f"{V}/private/v02_core_plus.csv")
