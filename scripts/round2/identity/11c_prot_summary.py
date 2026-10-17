"""TMT identity under the DLPFC-only mapping: columns where ros1d and DLPFC-only mappings differ; overlap of identity
mismatches with Y-protein sex disagreements; batch of mismatches. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
ROS = cfg.ROS; P = cfg.out_dir(cfg.PRIVATE)
r = pd.read_csv(f"{P}/11_prot_identity.csv", index_col=0)
c = pd.read_csv(f"{P}/04_prot_columns.csv", index_col=0)
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv"); pmale = dict(zip(clin.projid, clin.msex == 1))
samps = set(int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip())
diff = c.index[c.dl_proj != c.fed_proj]
t = r[r.own_rank.notna()]
td = t[t.index.isin(diff)]
print(f"columns whose donor differs between ros1d and DLPFC-only mappings: {len(diff)}; with genotyped DLPFC-only person {len(td)}; own (DLPFC-only) genotype ranks 1st: {int((td.own_rank == 1).sum())}")
# does the ros1d person match those columns? best other == fed person?
fedp = c.fed_proj.reindex(td.index)
print(f"  of these, ros1d-assigned person is genotyped: {int(fedp.isin(r.projid.dropna().astype(int)).sum() if False else sum(int(p) in set(pd.read_csv(f'{P}/02_genotype_sex_all.csv').projid) for p in fedp.dropna()))}")
mis = t[t.own_rank > 1]
c["sex_disagree"] = c.call_male != c.dl_male.astype("boolean")
for s, x in mis.iterrows():
    print(f"  mismatch: SDA {x.projid in samps}; own z {x.own_z:.1f}; best other z {x.best_other_z:.1f}; Y-protein sex call disagrees with record: {bool(c.loc[s, 'sex_disagree'])} (margin {abs(c.loc[s, 'margin_sd']):.1f} sd); n Y proteins {int(c.loc[s, 'n_y'])}; batch {s.split('.')[0]} (other mismatches in batch: {int(sum(m.split('.')[0] == s.split('.')[0] for m in mis.index)) - 1})")
sd = c[c.dl_proj.isin(samps) & c.sex_disagree.fillna(False).astype(bool)]
print(f"SDA TMT columns with Y-protein sex disagreement: {len(sd)}; identity checked (genotyped): {int(sd.index.isin(t.index).sum())}; of these own genotype rank 1: {int((t.reindex(sd.index).own_rank == 1).sum())}")
print("own z of genotyped columns vs number of Y proteins observed: corr", round(t.own_z.corr(c.n_y.reindex(t.index)), 2))
