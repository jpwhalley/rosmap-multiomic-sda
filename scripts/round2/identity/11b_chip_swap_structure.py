"""Structure of H3K9ac identity mismatches (aggregate/pseudonymous): reciprocal swaps, batch, sex of the matched person,
QC of mismatched samples, and whether the matched person's own ChIP sample exists/matches."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE); RS = cfg.out_dir(cfg.WORK, "identity_checks", "results")
ROS = cfg.ROS; O = f"{ROS}/oth"
r = pd.read_csv(f"{P}/11_chip_identity.csv", index_col=0); r.index = r.index.astype(str)
ch = pd.read_csv(f"{P}/05_chip_all_samples.csv", index_col=0); ch.index = ch.index.astype(str)
qc = pd.read_csv(f"{O}/chip/QualityControl.csv", sep="\t", dtype={"SampleID": str}).set_index("SampleID"); qc.index = qc.index.astype(str)
clin = pd.read_csv(f"{O}/meta/ROSMAP_clinical.csv"); pmale = dict(zip(clin.projid, clin.msex == 1))
samps = set(int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip())
key = pd.read_csv(f"{P}/donor_key.csv"); lab = dict(zip(key.projid, key.label))
def q(s, col):
    s2 = s if s in qc.index else s.zfill(8)
    return float((qc[col] < qc.loc[s2, col]).mean()) if s2 in qc.index else np.nan
def bat(s):
    s2 = s if s in qc.index else s.zfill(8); return qc.loc[s2, "Batch"] if s2 in qc.index else None
r["frip_pct"] = [q(s, "FractionOfReadsInPeaks") for s in r.index]; r["corr_pct"] = [q(s, "CorrelationWithMedianSample") for s in r.index]
t = r[r.own_rank.notna()]
print(f"own z vs QC: Spearman with FRiP percentile {t.own_z.corr(t.frip_pct, method='spearman'):.2f}, with CorrWithMedian percentile {t.own_z.corr(t.corr_pct, method='spearman'):.2f}")
lowq = t[(t.corr_pct < 0.05)]
print(f"genotyped samples in the bottom 5% of CorrelationWithMedianSample: {len(lowq)}; of these matching own genotype (rank 1): {int((lowq.own_rank == 1).sum())}; own z median {lowq.own_z.median():.1f}")
mis = t[t.own_rank > 1].copy()
by_proj = {}
for s, x in r.iterrows():
    if pd.notna(x.projid): by_proj.setdefault(int(x.projid), []).append(s)
rows = []
for s, x in mis.iterrows():
    rec, bo = int(x.projid), int(x.best_other_proj)
    strong = x.best_other_z >= 8
    partner_samples = by_proj.get(bo, [])
    recip = any(int(r.loc[ps, "best_other_proj"]) == rec and r.loc[ps, "best_other_z"] >= 8 for ps in partner_samples if pd.notna(r.loc[ps, "best_other_proj"]))
    rows.append(dict(label=lab.get(rec, "SDA" if rec in samps else "non-SDA"), own_z=round(x.own_z, 1), best_z=round(x.best_other_z, 1), source_identified=strong,
                     source_is_SDA=bo in samps if strong else None, source_has_own_chip_sample=len(partner_samples) > 0 if strong else None,
                     reciprocal_swap=recip if strong else None, same_batch=(bat(s) == bat(partner_samples[0])) if (strong and partner_samples) else None,
                     source_opposite_sex=(pmale.get(bo) != pmale.get(rec)) if strong else None,
                     chrY_call_vs_record=("agrees" if ch.loc[s, "call_male"] == pmale.get(rec) else "DISAGREES") if s in ch.index else "",
                     chrY_call_vs_source=("agrees" if ch.loc[s, "call_male"] == pmale.get(bo) else "DISAGREES") if (strong and s in ch.index) else "",
                     frip_pct=round(x.frip_pct, 2), corr_pct=round(x.corr_pct, 2)))
M = pd.DataFrame(rows); print(M.to_string(index=False)); M.to_csv(f"{RS}/11_chip_identity_mismatches.csv", index=False)
nt = r[r.own_rank.isna() & (r.best_other_z >= 8)]
print(f"non-genotyped recorded person but strong match to someone: {len(nt)}; recorded SDA donor: {int(nt.projid.isin(samps).sum())}; matched person is SDA donor: {int(nt.best_other_proj.astype(int).isin(samps).sum())}")
for s, x in nt.iterrows():
    bo = int(x.best_other_proj); ps = by_proj.get(bo, [])
    print(f"   matched person has own ChIP sample: {len(ps) > 0}; that sample matches them: {[bool(r.loc[p, 'own_rank'] == 1) for p in ps]}; sex of matched vs recorded opposite: {pmale.get(bo) != pmale.get(int(x.projid))}; chrY call vs matched person: {'agrees' if ch.loc[s, 'call_male'] == pmale.get(bo) else 'DISAGREES'}")
