"""H3K9ac: ID chain (ReadCounts RG_ID -> bam_id_mapping BAM_File=individualID -> projid) and chrY/chrX read
fractions from the full ReadCounts (all ChIP samples, not just SDA). Per-sample values -> $G_ROS_PRIVATE_DIR."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE)
ROS = cfg.ROS; O = f"{ROS}/oth"
clin = pd.read_csv(f"{O}/meta/ROSMAP_clinical.csv")
i2p = dict(zip(clin.individualID, clin.projid)); pmale = dict(zip(clin.projid, clin.msex == 1))
samps = [int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip()]
bm = pd.read_csv(f"{O}/chip/bam_id_mapping.csv")
print("bam_id_mapping rows", len(bm), "| unique BAM", bm.BAM_File.nunique(), "| unique RG", bm.RG_ID.nunique())
bm["bam_proj"] = bm.BAM_File.map(i2p)
bm["rg_num"] = pd.to_numeric(bm.RG_ID, errors="coerce"); bm["rg_eq_proj"] = bm.rg_num == bm.bam_proj; print("non-numeric RG IDs:", int(bm.rg_num.isna().sum()))
print("RG_ID equals the projid of the BAM's individualID:", int(bm.rg_eq_proj.sum()), "of", len(bm), "| BAM individual not in clinical:", int(bm.bam_proj.isna().sum()))
rc = pd.read_csv(f"{O}/chip/ReadCounts.csv", sep="\t", index_col=0)
dom = pd.read_csv(f"{O}/chip/H3K9acDomains.csv", sep="\t", index_col=0); dom["Chr"] = dom.Chr.astype(str)
rc = rc.loc[rc.index.intersection(dom.index)]
print("ReadCounts", rc.shape, "| chrY domains", int((dom.loc[rc.index, "Chr"] == "Y").sum()), "| chrX", int((dom.loc[rc.index, "Chr"] == "X").sum()))
tot = rc.sum(0); yf = rc[dom.loc[rc.index, "Chr"] == "Y"].sum(0) / tot; xf = rc[dom.loc[rc.index, "Chr"] == "X"].sum(0) / tot
S = pd.DataFrame({"y_frac": yf, "x_frac": xf, "total": tot}); S.index = S.index.astype(str)
# mapping as used by ros1b: RG -> BAM (last wins) -> projid
rg2bam = dict(zip(bm.RG_ID.astype(str), bm.BAM_File))
S["bam"] = [rg2bam.get(c, rg2bam.get(c.lstrip("0"))) for c in S.index]
S["proj_ros1b"] = S.bam.map(i2p)
S["proj_rg"] = pd.to_numeric(pd.Series(S.index, index=S.index), errors="coerce")
S["male_ros1b"] = S.proj_ros1b.map(pmale); S["male_rg"] = S.proj_rg.map(pmale)
ly = np.log10(S.y_frac + 1e-7)
s = np.sort(ly.values); g = np.argmax(np.diff(s)); thr = (s[g] + s[g + 1]) / 2
S["call_male"] = ly > thr
print(f"all ChIP samples {len(S)}: log10 chrY read fraction largest gap {s[g+1]-s[g]:.2f} at {thr:.2f}; split {g+1}/{len(s)-g-1}")
for k in ["male_ros1b", "male_rg"]:
    d = S[S[k].notna()]
    au = mannwhitneyu(ly[d.index][d[k] == True], ly[d.index][d[k] == False]).statistic / ((d[k] == True).sum() * (d[k] == False).sum())
    print(f"  sex via {k}: n {len(d)}; AUC {au:.3f}; discordant {int((d.call_male != d[k].astype(bool)).sum())}")
d = S[S.male_ros1b.notna()]; ax = mannwhitneyu(d.x_frac[d.male_ros1b == True], d.x_frac[d.male_ros1b == False]).statistic / ((d.male_ros1b == True).sum() * (d.male_ros1b == False).sum())
print(f"  chrX read fraction AUC(male>female) {ax:.3f} (weak secondary marker)")
sd = S[S.proj_ros1b.isin(samps)]
print(f"SDA ChIP donors (ros1b chain): {sd.proj_ros1b.nunique()}; discordant (full chrY counts) {int((sd.call_male != sd.male_ros1b.astype(bool)).sum())}; RG_ID==projid for {int((sd.proj_rg == sd.proj_ros1b).sum())}")
S.to_csv(f"{P}/05_chip_all_samples.csv")
# QC / batch / assay metadata for context
qc = pd.read_csv(f"{O}/chip/QualityControl.csv", sep="\t", dtype={"SampleID": str}).set_index("SampleID")
qc.index = qc.index.astype(str)
key = pd.read_csv(f"{P}/donor_key.csv")
lab_rows = []
for pj, lb in key.values:
    r = S[S.proj_ros1b == pj]
    if len(r) == 0: print(f"{lb}: no ChIP sample"); continue
    for c, x in r.iterrows():
        dup = int((bm.RG_ID.astype(str) == c.lstrip("0")).sum() + (bm.RG_ID.astype(str) == c).sum())
        q = qc.loc[c] if c in qc.index else (qc.loc[c.zfill(8)] if c.zfill(8) in qc.index else None)
        pct = lambda col: None if q is None else float((qc[col] < q[col]).mean())
        bat = None if q is None else q["Batch"]
        nb = None if q is None else int((qc.Batch == bat).sum())
        # batch-mates' discordance
        mates = [str(i) for i in qc.index[qc.Batch == bat]] if q is not None else []
        mm_ = S.loc[S.index.intersection(mates)]; mm_ = mm_[mm_.male_ros1b.notna()]
        bdisc = int((mm_.call_male != mm_.male_ros1b.astype(bool)).sum())
        print(f"{lb}: ChIP sample; full-count chrY call {'AGREES' if x.call_male == x.male_ros1b else 'DISAGREES'} with clinical (log10 yfrac {np.log10(x.y_frac+1e-7):.2f} vs threshold {thr:.2f}); "
              f"RG_ID equals donor projid: {x.proj_rg == pj}; RG rows in mapping: {dup}; QC batch size {nb}, discordant batch-mates {bdisc} (excl. self {bdisc - (x.call_male != x.male_ros1b)}); "
              f"percentile UniquelyMappedUniqueReads {pct('UniquelyMappedUniqueReads'):.2f}, FRiP {pct('FractionOfReadsInPeaks'):.2f}, CorrWithMedian {pct('CorrelationWithMedianSample'):.2f}")
