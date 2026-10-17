"""Pseudonymous per-label summary (no IDs, no clinical values): sex concordance and genotype identity per assay,
dominance, C74 position, GWAS membership. -> $G_ROS_RESULTS_DIR/identity_checks/results/15_flagged_donors_summary.csv (one row per flagged label)"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
P = cfg.out_dir(cfg.PRIVATE); ROS = cfg.ROS; SDA = f"{ROS}/sda"; AUD = cfg.AUD
RSP = cfg.out_dir(cfg.WORK, "identity_checks", "results")
samps = [int(l) for l in open(f"{SDA}/md_samples.txt") if l.strip()]; ss = set(samps)
key = pd.read_csv(f"{P}/donor_key.csv"); lab = dict(zip(key.projid, key.label))
T = pd.read_csv(f"{P}/13_all_donor_sex_table.csv", index_col=0)
def ident(file, idxcol=None):
    r = pd.read_csv(f"{P}/{file}", index_col=0)
    if idxcol: r = r.dropna(subset=[idxcol]).assign(**{idxcol: lambda d: d[idxcol].astype(int)}).drop_duplicates(idxcol).set_index(idxcol)
    return r
idt = {"RNA": ident("10_rna_identity.csv"), "ChIP": ident("11_chip_identity.csv", "projid"), "450K": ident("14_meth_identity.csv", "projid"), "TMT": ident("11_prot_identity.csv", "projid")}
def status(r, pj):
    if pj not in r.index: return "no sample"
    x = r.loc[pj]
    if pd.notna(x.own_rank):
        if x.own_rank == 1: return f"own genotype (z {x.own_z:.0f})"
        if x.best_other_z >= 8: return f"NOT own (z {x.own_z:.1f}); matches {'another SDA donor' if int(x.best_other_proj) in ss else 'a non-SDA person'} (z {x.best_other_z:.0f})"
        return f"NOT own (z {x.own_z:.1f}); source not among genotyped people (best z {x.best_other_z:.1f})"
    if x.best_other_z >= 8: return f"donor not genotyped; matches {'another SDA donor' if int(x.best_other_proj) in ss else 'a genotyped non-SDA person'} (z {x.best_other_z:.0f})"
    return "donor not genotyped; unverifiable"
gw = set(int(l.split()[1][3:]) for l in open(f"{ROS}/oth/snps/ROSMAP_arrayGenotype_maf.fam"))
A = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0).values.astype(float)
D = A - A.mean(1, keepdims=True); share = D ** 2 / (D ** 2).sum(1, keepdims=True)
rows = []
for pj, lb in key.values:
    i = samps.index(pj); dom = [k + 1 for k in range(546) if share[k].argmax() == i and share[k, i] > 0.8]
    row = dict(label=lb)
    for a, c in [("RNA", "RNA"), ("ChIP", "ChIP_fed"), ("450K", "meth_450K"), ("TMT", "TMT_own_DLPFC"), ("genotype", "genotype"), ("snATAC", "snATAC")]:
        v = T.loc[pj, c]; row[f"sex_{a}"] = "no data" if pd.isna(v) or v == "" else v
    for a in ["RNA", "ChIP", "450K", "TMT"]: row[f"identity_{a}"] = status(idt[a], pj)
    row["dominates"] = ";".join(f"C{k}" for k in dom); row["in_GWAS_set"] = pj in gw
    rows.append(row)
R = pd.DataFrame(rows); R.to_csv(f"{RSP}/15_flagged_donors_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
for _, r in R.iterrows():
    print(r.label); [print(f"   {k}: {v}") for k, v in r.items() if k != "label"]
