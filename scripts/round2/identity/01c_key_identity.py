"""Add SDA donors found by the genotype-identity checks (same-sex mismatches) to the private key."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import pandas as pd
P = cfg.out_dir(cfg.PRIVATE); ROS = cfg.ROS
samps = set(int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip())
key = pd.read_csv(f"{P}/donor_key.csv"); key = key[~key.label.str.startswith(("CHIPID", "PROTID"))]; have = set(key.projid)
add = []
ch = pd.read_csv(f"{P}/11_chip_identity.csv", index_col=0)
x = ch[(ch.own_rank > 1) & ch.projid.isin(samps) & ~ch.projid.isin(have)]
add += [(int(p), f"CHIPID-{j+1}") for j, p in enumerate(x.projid)]
pr = pd.read_csv(f"{P}/11_prot_identity.csv", index_col=0)
x = pr[(pr.own_rank > 1) & pr.projid.isin(samps) & ~pr.projid.isin(have)]
add += [(int(p), f"PROTID-{j+1}") for j, p in enumerate(x.projid)]
key = pd.concat([key, pd.DataFrame(add, columns=["projid", "label"])]).drop_duplicates("projid"); key.to_csv(f"{P}/donor_key.csv", index=False)
print("labels:", key.label.tolist())
