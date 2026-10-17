"""Structure of the snRNA identity mismatches (pseudonymous): is the matched person another SDA donor, does that donor's
own RNA match them (i.e. two SDA labels carry the same person's nuclei?), sex of matched person, pseudobulk duplication."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE); RS = cfg.out_dir(cfg.WORK, "identity_checks", "results")
ROS = cfg.ROS; SDA = f"{ROS}/sda"
r = pd.read_csv(f"{P}/10_rna_identity.csv", index_col=0)
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv"); pmale = dict(zip(clin.projid, clin.msex == 1))
samps = [int(l) for l in open(f"{SDA}/md_samples.txt") if l.strip()]
key = pd.read_csv(f"{P}/donor_key.csv")
mis = r[((r.own_rank > 1) | (r.own_z < 5)) & r.own_rank.notna()]
nt = r[r.own_rank.isna() & (r.best_other_z >= 8)]
add = [(int(p), f"RNAID-{j+1}") for j, p in enumerate(nt.index) if p not in set(key.projid)]
if add:
    key = pd.concat([key, pd.DataFrame(add, columns=["projid", "label"])]).drop_duplicates(); key.to_csv(f"{P}/donor_key.csv", index=False)
lab = dict(zip(key.projid, key.label))
R = pd.read_csv(f"{SDA}/rna.txt", sep=" ", header=None, dtype=np.float32).values
L = np.log2(R + 1)
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
for pj, x in pd.concat([mis, nt]).iterrows():
    bo = int(x.best_other_proj); i = samps.index(pj)
    insda = bo in samps
    line = f"{lab.get(pj, 'unlabelled')}: matched genotype z {x.best_other_z:.1f}; matched person is SDA donor: {insda}; matched person's clinical sex opposite to recorded: {pmale.get(bo) != pmale.get(pj)}"
    if insda:
        j = samps.index(bo); rj = r.loc[bo]
        line += f"; matched donor's own RNA matches them: rank {int(rj.own_rank) if pd.notna(rj.own_rank) else 'n/a'}, z {rj.own_z:.1f}; matched donor flagged as {lab.get(bo, 'none')}"
        cors = []
        for c in range(6):
            B = L[c * 276:(c + 1) * 276]; B = B - B.mean(0)
            cc = np.corrcoef(B[i], B[j])[0, 1]
            allc = np.corrcoef(B[i], np.delete(B, i, 0))[0, 1:] if False else np.array([np.corrcoef(B[i], B[k])[0, 1] for k in range(276) if k != i])
            cors.append((round(cc, 2), int(1 + np.sum(allc > cc))))
        line += f"; residual pseudobulk correlation with the matched donor per cell type (r, rank among 275): {cors}"
    print(line)
print("labels now:", key.label.tolist())
