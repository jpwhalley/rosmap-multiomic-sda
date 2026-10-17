"""R1 Major 3: donors with data per modality, from the SDA INPUT matrices (read-only archive).

Orientation (checked): rna.txt = 1,656 rows (6 cell-type blocks x 276 donors, block order cell_types.txt)
x 25,098 genes; meth.txt / chip.txt / prot.txt = 276 donor rows x features; space separated; missing = "NA".
Donor order = md_samples.txt. Outputs aggregates only (no donor IDs).
Writes: $G_ROS_RESULTS_DIR/r1_numbers/out/modality_coverage.json and donor_modality_mask.npy (276 x 4 bool, row order = md_samples;
kept local, no IDs inside).
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import json, os
import numpy as np, pandas as pd

S = cfg.SDA
OUT = cfg.out_dir(cfg.WORK, "r1_numbers", "out")

donors = [l.strip() for l in open(f"{S}/md_samples.txt") if l.strip()]
assert len(donors) == 276 and len(set(donors)) == 276
res = {"n_donors_md_samples": len(donors)}
mask = {}
for name, fn, nfeat_file in [("meth", "meth.txt", "md_cpg.txt"), ("chip", "chip.txt", "md_chip.txt"),
                             ("prot", "prot.txt", "md_prot.txt")]:
    X = pd.read_csv(f"{S}/{fn}", sep=" ", header=None, na_values=["NA"], dtype=np.float64).values
    nfeat = sum(1 for l in open(f"{S}/{nfeat_file}") if l.strip())
    assert X.shape == (276, nfeat), (name, X.shape, nfeat)
    nn = np.isfinite(X)
    per_donor = nn.sum(1)
    full = per_donor == nfeat; empty = per_donor == 0; partial = ~full & ~empty
    mask[name] = ~empty
    res[name] = {"shape_rows_x_features": list(X.shape), "donors_with_any_data": int((~empty).sum()),
                 "donors_complete": int(full.sum()), "donors_all_NA": int(empty.sum()),
                 "donors_partial": int(partial.sum()),
                 "missing_cells_in_present_donors": int((~nn[~empty]).sum()),
                 "features_with_any_NA_among_present_donors": int((~nn[~empty]).any(0).sum())}
    del X
# RNA: 6 blocks x 276 donors
R = pd.read_csv(f"{S}/rna.txt", sep=" ", header=None, na_values=["NA"], dtype=np.float64).values
assert R.shape == (1656, 25098)
blk = R.reshape(6, 276, 25098)
ct = [l.strip() for l in open(f"{S}/cell_types.txt") if l.strip()]
nn = np.isfinite(blk)
rowsum = np.nansum(blk, axis=2)
res["rna"] = {"shape": list(R.shape), "cell_type_block_order": ct,
              "any_NA": int((~nn).sum()),
              "donor_x_celltype_rows_all_zero": int((rowsum == 0).sum()),
              "row_sums_min_max": [float(rowsum.min()), float(rowsum.max())],
              "donors_with_data_all_6_blocks": int((rowsum > 0).all(0).sum())}
mask["rna"] = (rowsum > 0).all(0)
del R, blk
M = np.column_stack([mask["rna"], mask["meth"], mask["chip"], mask["prot"]])
np.save(os.path.join(OUT, "donor_modality_mask.npy"), M)
k = M.sum(1)
res["n_modalities_per_donor"] = {str(i): int((k == i).sum()) for i in range(1, 5)}
combos = {}
for row in M:
    key = "+".join(n for n, b in zip(["RNA", "meth", "chip", "prot"], row) if b)
    combos[key] = combos.get(key, 0) + 1
res["modality_combinations"] = dict(sorted(combos.items(), key=lambda kv: -kv[1]))
res["pairwise_overlap"] = {f"{a}&{b}": int((M[:, i] & M[:, j]).sum())
                           for i, a in enumerate(["RNA", "meth", "chip", "prot"])
                           for j, b in enumerate(["RNA", "meth", "chip", "prot"]) if j > i}
res["triple_overlap"] = {"RNA&meth&chip": int((M[:, 0] & M[:, 1] & M[:, 2]).sum()),
                         "RNA&meth&prot": int((M[:, 0] & M[:, 1] & M[:, 3]).sum()),
                         "RNA&chip&prot": int((M[:, 0] & M[:, 2] & M[:, 3]).sum())}
json.dump(res, open(os.path.join(OUT, "modality_coverage.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
