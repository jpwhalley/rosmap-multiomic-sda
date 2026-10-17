"""Extract, read-only, the slices needed for the source-data panels:
component sample scores, loadings and PIPs for the nine headline components, and the
SDA INPUT values (rna/meth/chip/prot .txt) for the union of their PIP>0.5 features.
Writes only to $G_ROS_RESULTS_DIR/source_panels/cache (per-donor values)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd

BASE = cfg.SDA
OUT = cfg.out_dir(cfg.WORK, "source_panels", "cache")
COMPS = [351, 2, 74, 87, 278, 177, 21, 172, 277]
MODS = {"RNA": ("genes.txt", "rna.txt"), "meth": ("md_cpg.txt", "meth.txt"),
        "chip": ("md_chip.txt", "chip.txt"), "proteins": ("md_prot.txt", "prot.txt")}

samples = [l.strip() for l in open(f"{BASE}/md_samples.txt") if l.strip()]
A = pd.read_csv(f"{BASE}/samples_final.csv", index_col=0)
assert list(A.columns.astype(str)) == samples, "samples_final columns != md_samples order"
assert list(A.index) == list(range(1, 547))
np.save(f"{OUT}/A_all.npy", A.values)  # 546 x 276, row k-1 = component k
out = {"samples": np.array(samples), "comps": np.array(COMPS)}

def read_rows(path, rows):
    keep = set(rows)
    df = pd.read_csv(path, index_col=0, skiprows=lambda i: i != 0 and i not in keep)
    return df

for m, (fn, inp) in MODS.items():
    feats = [l.rstrip("\n") for l in open(f"{BASE}/{fn}")]
    X = read_rows(f"{BASE}/genes_final_{m}.csv", COMPS)
    P = read_rows(f"{BASE}/pip_final_{m}.csv", COMPS)
    assert list(X.columns) == feats, f"{m}: loading columns != {fn}"
    assert list(P.columns) == feats
    X = X.loc[COMPS]; P = P.loc[COMPS]
    sel = np.where((P.values > 0.5).any(axis=0))[0]
    print(m, "features", len(feats), "union PIP>0.5", len(sel),
          {c: int((P.loc[c].values > 0.5).sum()) for c in COMPS})
    if m == "RNA":
        V = pd.read_csv(f"{BASE}/{inp}", sep=" ", header=None, usecols=list(sel),
                        dtype=np.float32, na_values="NA").values  # 1656 x |sel|
        assert V.shape[0] == 6 * 276
    else:
        V = pd.read_csv(f"{BASE}/{inp}", sep=" ", header=None, usecols=list(sel),
                        dtype=np.float64, na_values="NA").values  # 276 x |sel|
        assert V.shape[0] == 276
        miss = np.isnan(V).all(axis=1)
        partial = np.isnan(V).any(axis=1) & ~miss
        print(f"   {m}: donors with all-NA rows {miss.sum()}, donors with partial NA {partial.sum()}, n with data {(~miss).sum()}")
    out[f"{m}_feats"] = np.array(feats)
    out[f"{m}_X"] = X.values
    out[f"{m}_P"] = P.values
    out[f"{m}_sel"] = sel
    out[f"{m}_V"] = V

np.savez_compressed(f"{OUT}/slices.npz", **out)
print("saved")
