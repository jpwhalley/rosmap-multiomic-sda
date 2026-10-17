"""Recompute RNA (snRNA pseudobulk, as fed to SDA) and H3K9ac (as fed + full ReadCounts) sex calls for every SDA donor.
Writes per-donor calls to $G_ROS_PRIVATE_DIR. Prints aggregates and pseudonymous labels only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE); RS = cfg.out_dir(cfg.WORK, "identity_checks", "results")
ROS = cfg.ROS; SDA = f"{ROS}/sda"
CTS = [l.strip() for l in open(f"{SDA}/cell_types.txt") if l.strip()]
samps = np.array([int(l) for l in open(f"{SDA}/md_samples.txt") if l.strip()])
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv").set_index("projid")
male = clin.loc[samps, "msex"].values == 1
print("CTS", CTS, "n", len(samps), "clinical males", male.sum())
genes = [l.strip() for l in open(f"{SDA}/genes.txt")]
YG = ["RPS4Y1", "DDX3Y", "UTY", "KDM5D", "USP9Y", "EIF1AY", "ZFY", "NLGN4Y"]
XG = ["XIST", "TSIX"]
gi = [genes.index(g) for g in XG + YG]
R = pd.read_csv(f"{SDA}/rna.txt", sep=" ", header=None, usecols=gi, dtype=np.float64)
R.columns = [genes[i] for i in sorted(gi)]
out = pd.DataFrame(index=samps)
out["clin_male"] = male
for c, ct in enumerate(CTS):
    blk = R.iloc[c * 276:(c + 1) * 276]
    out[f"rna_{ct}_XIST"] = np.log1p(blk["XIST"].values)
    out[f"rna_{ct}_Y"] = np.log1p(blk[YG].values).mean(1)
out["rna_XIST"] = out[[f"rna_{ct}_XIST" for ct in CTS]].mean(1)
out["rna_Y"] = out[[f"rna_{ct}_Y" for ct in CTS]].mean(1)
# thresholds without labels: largest gap in each 1-D marker
def gap_thr(v):
    s = np.sort(v[~np.isnan(v)]); g = np.argmax(np.diff(s)); return (s[g] + s[g + 1]) / 2, s[g + 1] - s[g]
tX, gX = gap_thr(out.rna_XIST.values); tY, gY = gap_thr(out.rna_Y.values)
print(f"RNA XIST(mean log1p CPM over 6 types): largest gap at {tX:.2f} (gap {gX:.2f}); Y-genes: at {tY:.2f} (gap {gY:.2f})")
for k in ["rna_XIST", "rna_Y"]:
    print(k, "female q05/50/95", np.round(np.percentile(out.loc[~male, k], [5, 50, 95]), 2),
          "male q05/50/95", np.round(np.percentile(out.loc[male, k], [5, 50, 95]), 2))
xh = out.rna_XIST > tX; yh = out.rna_Y > tY
out["rna_call"] = np.where(yh & ~xh, "M", np.where(~yh & xh, "F", np.where(yh & xh, "mixed(XIST+Y)", "neither")))
cm = out.rna_call.map({"M": True, "F": False})
out["rna_disc"] = cm.notna() & (cm != out.clin_male)
print("RNA calls:", out.rna_call.value_counts().to_dict(), " discordant with clinical:", int(out.rna_disc.sum()),
      " non-binary calls:", int(cm.isna().sum()))
# per-cell-type consistency of the discordant donors
# ChIP as fed (chip.txt chrY peaks)
Hd = pd.read_csv(f"{ROS}/oth/chip/H3K9acDomains.csv", sep="\t", index_col=0); Hd["Chr"] = Hd.Chr.astype(str)
peaks = [l.strip() for l in open(f"{SDA}/md_chip.txt")]
ych = np.where(Hd.loc[peaks, "Chr"].values == "Y")[0]
C = pd.read_csv(f"{SDA}/chip.txt", sep=" ", header=None, usecols=list(ych), na_values="NA").values
out["chip_fed_Y"] = np.nanmean(C, 1)
has = ~np.isnan(out.chip_fed_Y.values)
t, g = gap_thr(out.chip_fed_Y.values)
print(f"ChIP as fed: {len(ych)} chrY peaks; donors {has.sum()}; largest gap {t:.2f} (gap {g:.2f})")
out["chip_fed_call"] = np.where(~has, "", np.where(out.chip_fed_Y > t, "M", "F"))
out["chip_fed_disc"] = has & ((out.chip_fed_Y > t) != out.clin_male)
print("ChIP-as-fed discordant:", int(out.chip_fed_disc.sum()))
# reproduce the earlier best-threshold rule (05e)
v = out.chip_fed_Y.values; ok = np.where(has)[0]; o = ok[np.argsort(v[ok])]; m = male[o]
ti = np.argmin([np.sum(m[:i]) + np.sum(~m[i:]) for i in range(len(o) + 1)])
pm = np.zeros(276, bool); pm[o[ti:]] = True
print("ChIP best-threshold discordant (05e rule):", int(sum(pm[i] != male[i] for i in ok)),
      " same set as largest-gap:", set(i for i in ok if pm[i] != male[i]) == set(np.where(out.chip_fed_disc)[0]))
out.to_csv(f"{P}/01_rna_chip_calls.csv")
# pseudonymous key
key = []
for j, i in enumerate(np.where(out.rna_disc)[0]): key.append((samps[i], f"RNA-{j+1}"))
for j, i in enumerate(np.where(out.chip_fed_disc)[0]): key.append((samps[i], f"CHIP-{j+1}"))
for j, i in enumerate(np.where(~out.rna_call.isin(["M", "F"]))[0]): key.append((samps[i], f"RNAMIX-{j+1}"))
key = pd.DataFrame(key, columns=["projid", "label"]); key.to_csv(f"{P}/donor_key.csv", index=False)
print("labels:", key.label.tolist(), " distinct donors:", key.projid.nunique())
# describe discordant donors without sex values: pattern relative to the clinical record
for pj, lab in key.values:
    r = out.loc[pj]
    print(lab, "RNA call", "agrees" if (r.rna_call in ("M","F") and (r.rna_call=="M")==r.clin_male) else f"DISAGREES/{'non-binary' if r.rna_call not in ('M','F') else 'opposite'}",
          "| RNA pattern:", r.rna_call if r.rna_call not in ("M","F") else "clean single-sex",
          "| ChIP as fed:", "no data" if r.chip_fed_call == "" else ("agrees" if (r.chip_fed_call=="M")==r.clin_male else "opposite"))
