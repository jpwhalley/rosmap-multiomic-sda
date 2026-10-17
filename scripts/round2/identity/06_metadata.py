"""Metadata evidence for the flagged donors (no IDs printed): MIT individual sex record, assays/specimens on record,
consortium exclusion flags (ROSMAP biospecimen 'exclude'/'excludeReason') for any of the donor's specimens, assay batches.
Also aggregate: SDA donors with any consortium exclusion flag, by assay and reason."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE)
ROS = cfg.ROS; O = f"{ROS}/oth/meta"; M = f"{ROS}/mit/meta"
clin = pd.read_csv(f"{O}/ROSMAP_clinical.csv"); p2i = dict(zip(clin.projid, clin.individualID)); pmale = dict(zip(clin.projid, clin.msex == 1))
samps = [int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip()]
b = pd.read_csv(f"{O}/ROSMAP_biospecimen_metadata.csv", low_memory=False)
mi0 = pd.read_csv(f"{M}/MIT_ROSMAP_Multiomics_individual_metadata.csv")
print("MIT individual rows", len(mi0), "duplicated individualIDs", int(mi0.individualID.duplicated().sum()), "of which with conflicting sex", int((mi0.groupby("individualID").sex.nunique() > 1).sum()))
mi = mi0.drop_duplicates("individualID").set_index("individualID")
mb = pd.read_csv(f"{M}/MIT_ROSMAP_Multiomics_biospecimen_metadata.csv")
atac = pd.read_csv(f"{M}/MIT_ROSMAP_Multiomics_assay_snATACseq_metadata.csv"); mult = pd.read_csv(f"{M}/MIT_ROSMAP_Multiomics_assay_multiome_metadata.csv")
cells = pd.read_csv(f"{ROS}/mit/rna/mit_sample_cell_counts.csv", index_col=0).iloc[:, 0]
# aggregate: MIT sex record vs clinical for SDA donors
sda_ind = [p2i[p] for p in samps]
mm = mi.reindex(sda_ind).sex
cm = pd.Series([pmale[p] for p in samps], index=sda_ind)
okm = mm.notna()
print(f"SDA donors in MIT individual metadata: {int(okm.sum())}/276; MIT sex disagrees with ROSMAP clinical: {int(((mm[okm] == 'male') != cm[okm]).sum())}")
# aggregate: consortium exclusion flags among SDA donors
bs = b[b.individualID.isin(sda_ind)]
ex = bs[bs.exclude == True]
print(f"SDA donors with any specimen flagged exclude=True: {ex.individualID.nunique()}; by assay: {ex.groupby('assay').individualID.nunique().to_dict()}")
print("  reasons (specimen counts):", ex.excludeReason.value_counts().to_dict())
sexflag = ex[ex.excludeReason.str.contains("XIST|gender|sex", case=False, na=False)]
print(f"  SDA donors with a sex-related flag (Low-XIST / gender mismatch): {sexflag.individualID.nunique()} ({sexflag.groupby('assay').individualID.nunique().to_dict()})")
key = pd.read_csv(f"{P}/donor_key.csv")
for pj, lb in key.values:
    ind = p2i[pj]
    recs = set(mi0.loc[mi0.individualID == ind, "sex"].dropna()); rec = None if not recs else (list(recs)[0] if len(recs) == 1 else "conflict")
    mitsex = "not in MIT individual table" if rec is None or pd.isna(rec) else (("CONFLICTING duplicate records" if rec == "conflict" else ("agrees" if (rec == "male") == pmale[pj] else "DISAGREES")))
    spec = mb[mb.individualID == ind]
    has_atac = spec[spec.assay == "snATACSeq"].specimenID.isin(atac.specimenID).sum(); has_mult = spec[spec.assay == "snMultiome"].specimenID.isin(mult.specimenID).sum()
    mine = b[b.individualID == ind]
    flags = mine[mine.exclude == True]
    nc = cells.get(pj)
    print(f"{lb}: MIT sex record {mitsex} with ROSMAP clinical; MIT specimens by assay {spec.assay.value_counts().to_dict()}; snATAC libraries {has_atac}, multiome {has_mult}; "
          f"snRNA nuclei (MIT count) percentile {float((cells < nc).mean()) if nc is not None else float('nan'):.2f}; "
          f"ROSMAP specimens by assay {mine.assay.value_counts().to_dict()}; exclusion flags: {flags[['assay','excludeReason']].values.tolist() if len(flags) else 'none'}")
