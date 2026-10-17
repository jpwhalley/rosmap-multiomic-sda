"""R1 Major 1: components passing each prioritisation stage (Methods l.199-201), on the corrected 547 catalogue.

Inputs (read-only): Data_S1_corrected_547.csv ($G_ROS_TABLES_DIR), pip_final_{RNA,meth,chip,proteins}.csv (archive; historical
rows 1..546, row 520 retired), C547/C548 consensus PIPs ($G_ROS_AUDIT_DIR/c547/). omics_count is recomputed from
the PIPs and checked against Data S1. Also writes per-component PIP>0.5 feature counts per modality (all 547) for
later steps: $G_ROS_RESULTS_DIR/r1_numbers/out/pip_counts_547.csv.
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import json, os
import numpy as np, pandas as pd

S = cfg.SDA
C547 = f"{cfg.AUD}/c547"
DS1 = f"{cfg.TABLES}/Data_S1_corrected_547.csv"
OUT = cfg.out_dir(cfg.WORK, "r1_numbers", "out")

ds1 = pd.read_csv(DS1, index_col=0)
assert len(ds1) == 547 and 520 not in ds1.index and {547, 548} <= set(ds1.index)
mods = [("RNA", "RNA", "P1"), ("meth", "meth", "P2"), ("chip", "chip", "P3"), ("proteins", "proteins", "P4")]
counts = pd.DataFrame(index=ds1.index, columns=[m[0] for m in mods], dtype=float)
for name, fn, pk in mods:
    P = pd.read_csv(f"{S}/pip_final_{fn}.csv", index_col=0)
    assert list(P.index) == list(range(1, 547)), name
    n = (P > 0.5).sum(1)
    for k in ds1.index:
        if k <= 546:
            counts.loc[k, name] = n.loc[k]
    for k in (547, 548):
        counts.loc[k, name] = int((np.load(f"{C547}/C{k}_{pk}.npy") > 0.5).sum())
    del P
counts = counts.astype(int)
counts["omics_count_recomputed"] = (counts[[m[0] for m in mods]] > 0).sum(1)
counts.to_csv(os.path.join(OUT, "pip_counts_547.csv"))
mism = (counts["omics_count_recomputed"] != ds1["omics_count"]).sum()

oc = ds1["omics_count"]
r = ds1["SEA-AD_corr"].abs()
has_partner = ds1["SEA-AD_comp"].notna()
stage1 = oc >= 3
stage2_all = r > 0.4
both = stage1 & stage2_all
paper_main = [351, 2, 74, 87, 177]          # main-text sections with a figure
paper_other = [278, 172, 277, 21]           # discussed with Supp figures / text
res = {
    "n_components": int(len(ds1)),
    "omics_count_recomputed_vs_DataS1_mismatches": int(mism),
    "omics_count_distribution": {str(i): int((oc == i).sum()) for i in range(0, 5)},
    "omics_count_ge2": int((oc >= 2).sum()), "omics_count_ge3": int(stage1.sum()),
    "modality_active_counts (>=1 PIP>0.5 feature)": {m[0]: int((counts[m[0]] > 0).sum()) for m in mods},
    "components_with_any_RNA_PIP_gene": int((counts["RNA"] > 0).sum()),
    "components_with_a_SEAAD_partner (non-blank)": int(has_partner.sum()),
    "abs_r_gt_0.4_all_547": int(stage2_all.sum()),
    "abs_r_gt_0.4_among_omics_ge3": int(both.sum()),
    "abs_r_gt_0.4_among_omics_lt3": int((stage2_all & ~stage1).sum()),
    "both_criteria_ids": [int(i) for i in ds1.index[both]],
    "expected_11": [2, 74, 76, 87, 99, 130, 177, 291, 345, 351, 416],
    "abs_r distribution among omics>=3": {
        "n": int(stage1.sum()), "n_with_partner": int((stage1 & has_partner).sum()),
        ">0.3": int((stage1 & (r > 0.3)).sum()), ">0.4": int(both.sum()), ">0.5": int((stage1 & (r > 0.5)).sum()),
        ">0.6": int((stage1 & (r > 0.6)).sum()), "median": float(r[stage1].median())},
    "abs_r distribution all 547": {">0.3": int((r > 0.3).sum()), ">0.5": int((r > 0.5).sum()),
                                    ">0.6": int((r > 0.6).sum()), "median": float(r.median())},
}
res["both_equals_expected_11"] = res["both_criteria_ids"] == res["expected_11"]
# components near the |r| = 0.4 boundary among omics>=3 (margin view)
near = ds1.loc[stage1 & (r > 0.35) & (r <= 0.45), ["omics_count", "SEA-AD_comp", "SEA-AD_corr"]]
res["omics_ge3_with_abs_r_in_(0.35,0.45]"] = {int(k): [int(v.omics_count), float(v["SEA-AD_comp"]), round(float(v["SEA-AD_corr"]), 3)]
                                               for k, v in near.iterrows()}
cols = ["omics_count", "SEA-AD_comp", "SEA-AD_corr", "Sex", "ceradsc", "cogdx", "Top_positive_cell_type",
        "Top_negative_cell_type", "GeneExpression_Pathway", "CpG_Site_Pathway", "Protein_Pathway"]
tab = ds1.loc[sorted(set(res["both_criteria_ids"]) | set(paper_main) | set(paper_other)), cols].copy()
tab = tab.join(counts[["RNA", "meth", "chip", "proteins"]])
tab["in_both_criteria"] = tab.index.isin(res["both_criteria_ids"])
tab["discussed_in_paper"] = tab.index.map(lambda k: "main text + figure" if k in paper_main else
                                          ("supp figure / text" if k in paper_other else "no"))
tab.to_csv(os.path.join(OUT, "prioritised_and_discussed_components.csv"))
json.dump(res, open(os.path.join(OUT, "filter_stages.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
print(tab.to_string())
