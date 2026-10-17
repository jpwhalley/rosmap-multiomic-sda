"""Side check: does the archived proteomics block assign each TMT data column to the donor the ROSMAP (400) metadata names?
The original ros1d mapping keys metadata on 'bXX.CHANNEL', which collides between the ROSMAP and ROSMAP210 studies."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import pandas as pd, numpy as np
B = cfg.ROS
clin = pd.read_csv(f"{B}/oth/meta/ROSMAP_clinical.csv", dtype={"projid": str}); i2p = dict(zip(clin.individualID, clin.projid.astype(int).astype(str)))
tm = pd.read_csv(f"{B}/prot/tmt/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv")
sp = tm.specimenID.str.split(".")
tm["study"] = sp.str[0]; tm["col"] = sp.str[-3] + "." + sp.str[-2]; tm["R_ID"] = sp.str[-1]
d = pd.read_csv(f"{B}/prot/tmt/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv", index_col=0)
x = pd.read_csv(f"{B}/prot/tmt/prot_crossed_with_sc.csv", index_col=0)
x = x.loc[:, x.notna().any()]
print("archived crossed proteomics: proteins", x.shape[0], "donor columns with data", x.shape[1])
dd = d.loc[x.index]
match = {}
for c in x.columns:
    diff = (dd.sub(x[c], axis=0)).abs().max()
    j = diff.idxmin(); match[c] = (j, float(diff.min()))
m = pd.DataFrame(match, index=["data_col", "maxabs"]).T
print("exact column matches (max|diff| < 1e-9):", int((m.maxabs.astype(float) < 1e-9).sum()), "of", len(m))
r400 = tm[tm.study == "ROSMAP"].set_index("col").R_ID
r210 = tm[tm.study == "ROSMAP210"].drop_duplicates("col").set_index("col").R_ID
m["donor_by_ROSMAP400_meta"] = m.data_col.map(r400).map(i2p)
m["donor_by_ROSMAP210_meta"] = m.data_col.map(r210).map(i2p)
m["assigned"] = m.index.astype(str)
ok400 = (m.assigned == m.donor_by_ROSMAP400_meta); ok210 = (m.assigned == m.donor_by_ROSMAP210_meta)
print("assigned donor == ROSMAP(400) metadata donor for that column:", int(ok400.sum()), "| == ROSMAP210 donor for a colliding label:", int(ok210.sum()), "| neither:", int((~ok400 & ~ok210).sum()))
# SDA block check: prot.txt rows (donors) vs crossed matrix
don = open(f"{B}/sda/md_samples.txt").read().split(); feats = open(f"{B}/sda/md_prot.txt").read().split()
print("SDA prot features all in crossed matrix:", set(feats) <= set(x.index))

# ---- corroboration: study fields, and sex concordance using Y-linked proteins ----
print("\nplatform by study:", tm.groupby("study").platform.agg(lambda v: sorted(set(map(str, v)))).to_dict())
print("metadata batch labels by study (first 5):", tm.groupby("study").batch.agg(lambda v: sorted(set(map(str, v)))[:5]).to_dict())
mis = m[ok210]
in400 = mis.donor_by_ROSMAP210_meta.isin(set(tm[tm.study == "ROSMAP"].R_ID.map(i2p).dropna()))
print("misassigned (ROSMAP210 label) donors that also have their own column in the ROSMAP(400) matrix:", int(in400.sum()), "of", len(mis))
ylink = [i for i in d.index if i.split("|")[0] in {"DDX3Y", "EIF1AY", "KDM5D", "UTY", "USP9Y", "RPS4Y1", "ZFY", "NLGN4Y", "PCDH11Y", "TMSB4Y"}]
print("Y-linked proteins in TMT matrix:", ylink)
if ylink:
    sex = dict(zip(clin.projid.astype(int).astype(str), clin.msex))
    ysc = d.loc[ylink].mean()
    for lab, sel in [("assigned == ROSMAP400 donor", ok400), ("assigned == ROSMAP210 donor (collision)", ok210)]:
        mm = m[sel]
        ys = ysc.loc[mm.data_col.values].values
        s_assigned = mm.assigned.map(sex).values
        s_meta400 = mm.donor_by_ROSMAP400_meta.map(sex).values
        # threshold Y score at the midpoint between sexes among columns with known 400-metadata sex
        print(f"  {lab}: n {len(mm)}; mean Y-protein score by ASSIGNED donor sex  M {np.nanmean(ys[s_assigned == 1]):.2f} / F {np.nanmean(ys[s_assigned == 0]):.2f}"
              f" | by ROSMAP400-metadata donor sex  M {np.nanmean(ys[s_meta400 == 1]):.2f} / F {np.nanmean(ys[s_meta400 == 0]):.2f}")
    allm = pd.DataFrame({"y": ysc.values}, index=ysc.index)
    allm["sex400"] = pd.Series(tm[tm.study == "ROSMAP"].set_index("col").R_ID).reindex(allm.index).map(i2p).map(sex)
    thr = (allm[allm.sex400 == 1].y.median() + allm[allm.sex400 == 0].y.median()) / 2
    pred = (allm.y > thr).astype(float)
    print(f"  all 400 columns: Y-score sex call agrees with ROSMAP400-metadata sex for {int((pred == allm.sex400)[allm.sex400.notna()].sum())}/{int(allm.sex400.notna().sum())}")
    for lab, sel in [("ok400", ok400), ("collision", ok210)]:
        mm = m[sel]; pr = (ysc.loc[mm.data_col.values].values > thr).astype(float)
        print(f"  {lab}: Y-score sex call agrees with ASSIGNED donor sex {int((pr == mm.assigned.map(sex).values).sum())}/{len(mm)}; with ROSMAP400-metadata donor sex {int((pr == mm.donor_by_ROSMAP400_meta.map(sex).values).sum())}/{int(mm.donor_by_ROSMAP400_meta.notna().sum())}")

# ---- does the SDA input block (prot.txt) carry the crossed matrix as is? ----
P = pd.read_csv(f"{B}/sda/prot.txt", sep=r"\s+", header=None, na_values="NA").values   # 276 x 5230
P = pd.DataFrame(P, index=don, columns=feats)
have = P.index[P.notna().any(axis=1)]
cors = [np.corrcoef(P.loc[c].values, x.loc[feats, c].values)[0, 1] for c in have]
print("\nSDA prot.txt donors with data:", len(have), "| same donor set as crossed matrix:", set(have) == set(x.columns.astype(str)),
      "| per-donor Pearson(prot.txt row, crossed column) min", round(float(np.min(cors)), 6))
print("so the 29 collision assignments are in the SDA input:", int(m[ok210].assigned.isin(have).sum()), "of 29")
