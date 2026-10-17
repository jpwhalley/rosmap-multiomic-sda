"""For each flagged donor: components it dominates (share of sum of squared scores), whether any is a highlighted
component, its position on C74 (sex axis), and GWAS-set membership. Aggregates/pseudonymous labels only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE)
ROS = cfg.ROS; SDA = f"{ROS}/sda"; AUD = cfg.AUD
samps = np.array([int(l) for l in open(f"{SDA}/md_samples.txt") if l.strip()])
A = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0); assert [int(c) for c in A.columns] == list(samps)
A = A.values.astype(float)                                   # 546 x 276, row k-1 = archived component k
A = np.vstack([A, np.load(f"{AUD}/c547/C547_A.npy")[None], np.load(f"{AUD}/c547/C548_A.npy")[None]])   # rows 547, 548
labels = list(range(1, 547)) + [547, 548]
D = A - A.mean(1, keepdims=True); share = D ** 2 / (D ** 2).sum(1, keepdims=True)
top = share.argmax(1); topshare = share.max(1)
HL = {"main/supp figures": [351, 2, 87, 177, 74, 278, 21, 172, 277, 545], "prioritised 11": [2, 74, 76, 87, 99, 130, 177, 291, 345, 351, 416], "new": [547, 548]}
hl = sorted(set(sum(HL.values(), [])))
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv").set_index("projid"); male = clin.loc[samps, "msex"].values == 1
a74 = A[73]
mu_m, mu_f = np.median(a74[male]), np.median(a74[~male]); sdp = np.sqrt((np.var(a74[male]) + np.var(a74[~male])) / 2)
# position on the C74 axis: 0 = recorded-sex cluster centre, 1 = opposite cluster centre
pos = np.where(male, (a74 - mu_m) / (mu_f - mu_m), (a74 - mu_f) / (mu_m - mu_f))
inter = np.where((pos > 0.25) & (pos < 0.75))[0]
print(f"C74: donors between the sex clusters (25-75% of the way to the other cluster): {len(inter)}")
gw = set(int(l.split()[1][3:]) for l in open(f"{ROS}/oth/snps/ROSMAP_arrayGenotype_maf.fam"))
key = pd.read_csv(f"{P}/donor_key.csv")
lab_of = dict(zip(key.projid, key.label))
print("  flagged donors among them:", sorted(lab_of[p] for p in samps[inter] if p in lab_of), "| unflagged:", sum(p not in lab_of for p in samps[inter]))
for pj, lb in key.values:
    i = int(np.where(samps == pj)[0][0])
    dom = [labels[k] for k in range(len(labels)) if top[k] == i and topshare[k] > 0.8]
    topk = [labels[k] for k in range(len(labels)) if top[k] == i]
    hl_top = [c for c in topk if c in hl]
    rank_in_hl = {c: int((np.argsort(-share[labels.index(c)]) == i).nonzero()[0][0]) + 1 for c in hl}
    near = {c: r for c, r in rank_in_hl.items() if r <= 3}
    print(f"{lb}: dominant (>80% share) donor of {len(dom)} component(s) {dom}; top donor (any share) of {topk}; highlighted components where it is top donor: {hl_top or 'none'}; "
          f"highlighted components where it ranks top-3 by share: {near or 'none'}; C74 position {pos[i]:.2f} (0 = own recorded-sex cluster, 1 = opposite); in GWAS set: {pj in gw}")
