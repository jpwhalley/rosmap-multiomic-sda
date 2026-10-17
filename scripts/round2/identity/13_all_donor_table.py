"""Per-donor sex concordance across all assays for the 276 SDA donors (private table) and aggregate counts per assay.
Assays: snRNA (rna.txt), H3K9ac (chip.txt as fed; and full ReadCounts), 450K (raw X/Y), TMT (DLPFC-only own column,
and the row as fed), genotype (X het / Y calls), snATAC. Identity (QTL/rs-probe) results joined where available."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE); RS = cfg.out_dir(cfg.WORK, "identity_checks", "results")
ROS = cfg.ROS
samps = [int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip()]
T = pd.read_csv(f"{P}/01_rna_chip_calls.csv", index_col=0)[["clin_male", "rna_call", "chip_fed_call"]]
cm = T.clin_male.astype(bool)
def agree(call):  # call in M/F/other -> 'agree'/'DISAGREE'/'ambiguous'/'' (no data)
    return np.where(call.isna() | (call == ""), "", np.where(~call.isin(["M", "F"]), "ambiguous", np.where((call == "M") == cm.reindex(call.index), "agree", "DISAGREE")))
T["RNA"] = agree(T.rna_call); T["ChIP_fed"] = agree(T.chip_fed_call.fillna(""))
ch = pd.read_csv(f"{P}/05_chip_all_samples.csv", index_col=0); ch = ch[ch.proj_ros1b.isin(samps)].drop_duplicates("proj_ros1b").set_index("proj_ros1b")
T["ChIP_full"] = agree(pd.Series(np.where(ch.call_male, "M", "F"), index=ch.index).reindex(T.index))
me = pd.read_csv(f"{P}/03_meth_samples.csv", index_col=0); me = me[me.sda_meth].set_index("projid")
T["meth_450K"] = agree(me.meth_call.reindex(T.index))
pr = pd.read_csv(f"{P}/04_prot_columns.csv", index_col=0)
own = pr[pr.dl_proj.isin(samps)].dropna(subset=["score"]).drop_duplicates("dl_proj").set_index("dl_proj")
T["TMT_own_DLPFC"] = agree(pd.Series(np.where(own.call_male, "M", "F"), index=own.index).reindex(T.index))
T["TMT_own_margin_sd"] = own.margin_sd.reindex(T.index).abs()
ge = pd.read_csv(f"{P}/02_genotype_sex_all.csv"); ge = ge[ge.projid.isin(samps)].set_index("projid")
T["genotype"] = agree(ge.geno_call.where(ge.geno_call.isin(["M", "F"]), "").reindex(T.index).fillna(""))
T.loc[T.index.isin(ge.index) & (T.genotype == ""), "genotype"] = "no X/Y calls"
at = pd.read_csv(f"{P}/07_atac_libraries.csv"); at = at[at.projid.isin(samps)].drop_duplicates("projid").set_index("projid")
T["snATAC"] = agree(pd.Series(np.where(at.call_male, "M", "F"), index=at.index).reindex(T.index))
key = pd.read_csv(f"{P}/donor_key.csv"); T["label"] = pd.Series(dict(zip(key.projid, key.label))).reindex(T.index)
T.to_csv(f"{P}/13_all_donor_sex_table.csv")
rows = []
for a in ["RNA", "ChIP_fed", "ChIP_full", "meth_450K", "TMT_own_DLPFC", "genotype", "snATAC"]:
    v = T[a]; n = int(v.isin(["agree", "DISAGREE", "ambiguous"]).sum())
    dis = T[v == "DISAGREE"]
    rows.append(dict(assay=a, donors_with_call=n, agree=int((v == "agree").sum()), disagree=int((v == "DISAGREE").sum()), ambiguous=int((v == "ambiguous").sum()),
                     disagree_labels=";".join(sorted(dis.label.dropna())), disagree_unlabelled=int(dis.label.isna().sum()),
                     note={"genotype": f"{int((T.genotype == 'no X/Y calls').sum())} genotyped donors have no X/Y calls (KronosII batch)",
                           "TMT_own_DLPFC": f"strong (|margin|>1 sd) disagreements: {int(((T.TMT_own_DLPFC == 'DISAGREE') & (T.TMT_own_margin_sd > 1)).sum())}"}.get(a, "")))
R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(f"{RS}/13_sex_concordance_by_assay.csv", index=False)
# donors with >=1 disagreement: how many other assays agree with clinical
cols = ["RNA", "ChIP_fed", "meth_450K", "TMT_own_DLPFC", "genotype", "snATAC"]
anyd = T[(T[cols] == "DISAGREE").any(axis=1)]
print(f"donors with any disagreement (excluding the duplicate ChIP_full column): {len(anyd)}")
for pj, r in anyd.iterrows():
    print(f"  {r.label if pd.notna(r.label) else 'unlabelled'}: disagree in {[c for c in cols if r[c] == 'DISAGREE']}; agree in {[c for c in cols if r[c] == 'agree']}")
