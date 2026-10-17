"""Add secondary donors of interest to the private key: SDA donors with a consortium sex-related exclusion flag
(WGS gender mismatch; Low-XIST female) and the SDA donor whose 450K X and Y calls conflict."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE)
ROS = cfg.ROS; O = f"{ROS}/oth/meta"
clin = pd.read_csv(f"{O}/ROSMAP_clinical.csv"); i2p = dict(zip(clin.individualID, clin.projid))
samps = set(int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip())
b = pd.read_csv(f"{O}/ROSMAP_biospecimen_metadata.csv", low_memory=False)
key = pd.read_csv(f"{P}/donor_key.csv"); key = key[~key.label.str.startswith(("FLAG", "METHXY"))]
f = b[(b.exclude == True) & b.excludeReason.str.contains("gender", case=False, na=False)]
add = [(p, f"FLAGWGS-{j+1}") for j, p in enumerate(sorted(i2p[i] for i in set(f.individualID) if i2p.get(i) in samps))]
f = b[(b.exclude == True) & b.excludeReason.str.contains("XIST", case=False, na=False)]
add += [(p, f"FLAGXIST-{j+1}") for j, p in enumerate(sorted(i2p[i] for i in set(f.individualID) if i2p.get(i) in samps))]
m = pd.read_csv(f"{P}/03_meth_samples.csv", index_col=0)
add += [(int(p), f"METHXY-{j+1}") for j, p in enumerate(m[m.sda_meth & (m.meth_call == "X/Y conflict")].projid)]
key = pd.concat([key, pd.DataFrame(add, columns=["projid", "label"])]).drop_duplicates()
key.to_csv(f"{P}/donor_key.csv", index=False)
print("labels:", key.label.tolist(), "| distinct donors:", key.projid.nunique(), "| overlaps:", key[key.projid.duplicated(keep=False)].label.tolist())
