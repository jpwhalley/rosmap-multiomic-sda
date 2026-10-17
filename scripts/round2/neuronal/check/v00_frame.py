"""Write cache/frame.csv (per donor, md_samples order, no IDs): v_common.frame() plus the deposited scores of C172, C277,
C21 and C74. This step was run inline in the round-2 analysis; this script reproduces that file byte for byte."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
from v_common import *

F = frame(); A = scores()
for k in (172, 277, 21, 74):
    F[f"s{k}"] = A.loc[k].values
F.to_csv(f"{CACHE}/frame.csv", index=False)
print("frame.csv:", F.shape)
