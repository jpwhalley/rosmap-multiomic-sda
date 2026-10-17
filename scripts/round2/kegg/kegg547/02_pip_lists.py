"""Reconstruct PIP>0.5 RNA gene lists for components 1..546 from the deposit, exactly as ros7ai did
(genes[1][pip[1].T[pip[1].loc[k] > 0.5].index.values]); background = genes_final_RNA columns."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import json, time
import numpy as np, pandas as pd
S = cfg.SDA
D = cfg.out_dir(cfg.AUD, "kegg547")
t = time.time()
pip = pd.read_csv(f"{S}/pip_final_RNA.csv", index_col=0)
# only the header of genes_final_RNA is needed for the background (all_proteins = genes[1].columns)
gcols = pd.read_csv(f"{S}/genes_final_RNA.csv", index_col=0, nrows=2)
txt = [l.rstrip("\n") for l in open(f"{S}/genes.txt")]
print("pip shape", pip.shape, "genes_final cols", gcols.shape[1], "genes.txt", len(txt), f"{time.time()-t:.1f}s")
print("pip cols == genes_final cols:", list(pip.columns) == list(gcols.columns))
print("pip cols == genes.txt:", list(pip.columns) == txt)
raw_hdr = open(f"{S}/pip_final_RNA.csv").readline().rstrip("\n").split(",")[1:]
print("raw header == genes.txt:", raw_hdr == txt, "| n unique raw names:", len(set(raw_hdr)),
      "| pandas-mangled names differing from raw:", sum(a != b for a, b in zip(raw_hdr, pip.columns)))
print("row index:", pip.index.min(), pip.index.max(), len(pip.index), pip.index.dtype)
lists = {}
for k in pip.index:
    row = pip.loc[k]
    lists[int(k)] = list(pip.columns[(row > 0.5).values])
n = {k: len(v) for k, v in lists.items()}
print("components with 0 PIP>0.5 genes:", [k for k, v in n.items() if v == 0])
print("list size quantiles:", np.percentile(list(n.values()), [0, 25, 50, 75, 100]))
json.dump({"background": list(gcols.columns), "lists": {str(k): v for k, v in lists.items()}},
          open(f"{D}/pip05_lists_RNA.json", "w"))
print("done", f"{time.time()-t:.1f}s")
