"""MIT snATAC (independent nuclei prep from the MIT multiomics set): chrY and chrX fragment fractions per library;
library -> specimen -> individual -> projid; sex call vs clinical for all libraries; flagged donors by label."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, glob, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE); C = cfg.out_dir(cfg.WORK, "identity_checks", "cache", "atac")
ROS = cfg.ROS; M = f"{ROS}/mit/meta"
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv"); i2p = dict(zip(clin.individualID, clin.projid)); pmale = dict(zip(clin.projid, clin.msex == 1))
am = pd.read_csv(f"{M}/MIT_ROSMAP_Multiomics_assay_snATACseq_metadata.csv"); mb = pd.read_csv(f"{M}/MIT_ROSMAP_Multiomics_biospecimen_metadata.csv")
s2i = mb.drop_duplicates("specimenID").set_index("specimenID").individualID
lib2proj = {l: i2p.get(s2i.get(s)) for l, s in zip(am.libraryID, am.specimenID)}
rows = []
for f in sorted(glob.glob(f"{C}/*.chr.txt")):
    lib = os.path.basename(f).replace(".chr.txt", "")
    cnt = {}
    for line in open(f):
        n, ch = line.split(); cnt[ch] = cnt.get(ch, 0) + int(n)
    tot = sum(cnt.values()); rows.append(dict(lib=lib, total=tot, y=cnt.get("chrY", 0) / tot, x=cnt.get("chrX", 0) / tot, projid=lib2proj.get(lib)))
S = pd.DataFrame(rows); S["clin_male"] = S.projid.map(pmale)
print("libraries counted", len(S), "| mapped to a projid", int(S.projid.notna().sum()), "| distinct donors", S.projid.nunique())
ly = np.log10(S.y + 1e-7); s = np.sort(ly); g = np.argmax(np.diff(s)); thr = (s[g] + s[g + 1]) / 2
S["call_male"] = ly > thr
d = S[S.clin_male.notna()]
au = mannwhitneyu(ly[d.index][d.clin_male == True], ly[d.index][d.clin_male == False]).statistic / ((d.clin_male == True).sum() * (d.clin_male == False).sum())
ax = mannwhitneyu(d.x[d.clin_male == False], d.x[d.clin_male == True]).statistic / ((d.clin_male == True).sum() * (d.clin_male == False).sum())
print(f"chrY fraction: largest gap {s[g+1]-s[g]:.2f} log10 at {thr:.2f}; split {g+1}/{len(s)-g-1}; AUC vs clinical {au:.3f}; chrX fraction AUC(female>male) {ax:.3f}")
print(f"libraries discordant with clinical sex: {int((d.call_male != d.clin_male.astype(bool)).sum())} of {len(d)}")
S.to_csv(f"{P}/07_atac_libraries.csv", index=False)
samps = set(int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip())
sd = d[d.projid.isin(samps)]
print(f"SDA donors with snATAC: {sd.projid.nunique()}; discordant {int((sd.call_male != sd.clin_male.astype(bool)).sum())}")
key = pd.read_csv(f"{P}/donor_key.csv")
for pj, lb in key.values:
    r = S[S.projid == pj]
    for _, x in r.iterrows():
        print(f"{lb}: snATAC library; chrY call {'AGREES' if x.call_male == x.clin_male else 'DISAGREES'} with clinical (log10 chrY frac {np.log10(x.y+1e-7):.2f}, threshold {thr:.2f}); chrX fraction percentile among recorded-same-sex libraries {float((d[d.clin_male == x.clin_male].x < x.x).mean()):.2f}")
