"""Aggregate genotype-identity results per assay (SDA donors and all samples). -> $G_ROS_RESULTS_DIR/identity_checks/results/17_identity_by_assay.csv"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import pandas as pd, numpy as np
P = cfg.out_dir(cfg.PRIVATE); ROS = cfg.ROS; RS = cfg.out_dir(cfg.WORK, "identity_checks", "results")
samps = set(int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip())
key = pd.read_csv(f"{P}/donor_key.csv"); lab = dict(zip(key.projid, key.label))
rows = []
for a, f, method in [("snRNA (rna.txt)", "10_rna_identity.csv", "eQTL, 5-fold cross-fitted, ~600 pairs"), ("H3K9ac (ReadCounts)", "11_chip_identity.csv", "haQTL, cross-fitted, ~330 pairs"),
                     ("450K (imputed autosomal)", "14_meth_identity.csv", "meQTL, cross-fitted, ~640 pairs"), ("TMT (DLPFC-only mapping)", "11_prot_identity.csv", "pQTL, cross-fitted, ~210 pairs")]:
    r = pd.read_csv(f"{P}/{f}", index_col=0)
    if "projid" not in r.columns: r["projid"] = r.index
    for scope, d in [("SDA donors", r[r.projid.isin(samps)]), ("all samples", r)]:
        t = d[d.own_rank.notna()]; nt = d[d.own_rank.isna()]
        bad = t[t.own_rank > 1]
        rows.append(dict(assay=a, method=method, scope=scope, samples=len(d), recorded_person_genotyped=len(t), own_genotype_best=int((t.own_rank == 1).sum()),
                         own_z_median=round(t.own_z.median(), 1), not_own=len(bad), not_own_source_identified=int((bad.best_other_z >= 8).sum()),
                         not_own_labels=";".join(sorted(lab.get(int(p), "") for p in bad.projid if int(p) in lab)) if scope == "SDA donors" else "",
                         not_genotyped=len(nt), not_genotyped_strong_match=int((nt.best_other_z >= 8).sum())))
R = pd.DataFrame(rows); R.to_csv(f"{RS}/17_identity_by_assay.csv", index=False); print(R.drop(columns=["method"]).to_string(index=False))
