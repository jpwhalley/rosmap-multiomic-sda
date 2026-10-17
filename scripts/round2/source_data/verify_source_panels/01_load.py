"""Independent loader (verification of source_panels). Read-only on the SDA archive.
Caches full input matrices and full loading/PIP matrices as .npy under $G_ROS_RESULTS_DIR/verify_source_panels/cache."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, time, numpy as np, pandas as pd

SDA = cfg.SDA
OUT = cfg.out_dir(cfg.WORK, "verify_source_panels", "cache")

t0 = time.time()
samples = [l.strip() for l in open(f"{SDA}/md_samples.txt") if l.strip()]
A = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0)
print("samples_final", A.shape, "cols==md_samples:", list(A.columns.astype(str)) == samples,
      "index 1..546:", list(A.index) == list(range(1, 547)))
np.save(f"{OUT}/A.npy", A.values.astype(np.float64))
np.save(f"{OUT}/samples.npy", np.array(samples))
Bp = pd.read_csv(f"{SDA}/treatment_final.csv", index_col=0)
print("treatment_final", Bp.shape, list(Bp.columns))
np.save(f"{OUT}/B_published.npy", Bp.values)

files = {"RNA": ("genes.txt", "rna.txt"), "meth": ("md_cpg.txt", "meth.txt"),
         "chip": ("md_chip.txt", "chip.txt"), "proteins": ("md_prot.txt", "prot.txt")}
for m, (ff, inp) in files.items():
    feats = [l.rstrip("\n") for l in open(f"{SDA}/{ff}")]
    np.save(f"{OUT}/{m}_feats.npy", np.array(feats))
    V = pd.read_csv(f"{SDA}/{inp}", sep=" ", header=None, na_values="NA", dtype=np.float64, engine="c").values
    print(m, "input", V.shape, "n feats file", len(feats), "NaN frac", np.isnan(V).mean().round(4), f"{time.time()-t0:.0f}s")
    np.save(f"{OUT}/{m}_input.npy", V)
    X = pd.read_csv(f"{SDA}/genes_final_{m}.csv", index_col=0)
    P = pd.read_csv(f"{SDA}/pip_final_{m}.csv", index_col=0)
    print("  loadings", X.shape, "cols==feats", list(X.columns) == feats, "pip cols==feats", list(P.columns) == feats,
          "rows", X.index[:3].tolist(), X.index[-1], f"{time.time()-t0:.0f}s")
    assert list(X.index) == list(range(1, 547)) and list(P.index) == list(range(1, 547))
    np.save(f"{OUT}/{m}_X.npy", X.values.astype(np.float64))
    np.save(f"{OUT}/{m}_P.npy", P.values.astype(np.float32))
print("done", f"{time.time()-t0:.0f}s")
