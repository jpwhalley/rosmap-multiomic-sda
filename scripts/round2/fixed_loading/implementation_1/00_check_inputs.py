"""Check that the verify_source_panels input caches equal the SDA input text files exactly (NaN pattern and values)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np
from pi_common import *

files = {"RNA": "rna.txt", "meth": "meth.txt", "chip": "chip.txt", "proteins": "prot.txt"}
Y = load_inputs()
for m, f in files.items():
    V = read_mat(f"{SDA}/{f}")
    C = np.asarray(Y[m])
    print(m, V.shape, C.shape, "NaN pattern equal", bool(np.array_equal(np.isnan(V), np.isnan(C))),
          "values equal", bool(np.array_equal(V[~np.isnan(V)], C[~np.isnan(C)])), flush=True)
    del V
