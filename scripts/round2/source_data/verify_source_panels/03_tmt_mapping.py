"""Independent check of the proteomics donor mapping (ros1d) claimed by the first implementation.
Rebuilds prot input under (i) the ros1d logic as written and (ii) a ROSMAP.DLPFC-only mapping;
compares with prot.txt; checks Y-linked proteins in the raw 400-sample TMT matrix vs clinical sex.
Prints aggregates only (counts, AUCs); no donor IDs."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats
C = cfg.out_dir(cfg.WORK, "verify_source_panels", "cache")
R = cfg.out_dir(cfg.WORK, "verify_source_panels", "results")
T = f"{cfg.ROS}/prot/tmt"
tmt = pd.read_csv(f"{T}/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv", index_col=0)
meta = pd.read_csv(f"{T}/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv", index_col=0)
clin = pd.read_csv(f"{cfg.ROS}/oth/meta/ROSMAP_clinical.csv", index_col=0)
samples = np.load(f"{C}/samples.npy").astype(int)
print("tmt", tmt.shape, "meta rows", meta.shape[0])

# metadata ordering
study = meta.index.str.split(".").str[0]
print("study counts", pd.Series(study).value_counts().to_dict())
print("first ROSMAP210 row position", int(np.where(study == "ROSMAP210")[0].min()), "last DLPFC row", int(np.where(study == "ROSMAP")[0].max()) if (study == "ROSMAP").any() else None)

def build(meta_sub):
    rid = meta_sub.index.str.split(".").str[-1]
    sid = meta_sub.index.str.split(".").str[-3] + "." + meta_sub.index.str.split(".").str[-2]
    s = pd.Series(rid, index=sid)
    common = tmt.columns.intersection(s.index)
    return s.loc[common].to_dict(), common          # .to_dict keeps last duplicate (ros1d behaviour)

map_ros1d, common = build(meta)
dl = meta[meta.index.str.startswith("ROSMAP.DLPFC")]
map_dl, common_dl = build(dl)
print("columns mapped: ros1d", len(map_ros1d), " DLPFC-only", len(map_dl))
diff = [c for c in tmt.columns if map_ros1d.get(c) != map_dl.get(c)]
print("columns whose individual differs between mappings:", len(diff), " batches:", sorted(set(c.split(".")[0] for c in diff))[:3], "...", sorted(set(c.split(".")[0] for c in diff))[-1:])

sc_ind = clin.loc[samples, "individualID"]
ind2proj = dict(zip(sc_ind.values, sc_ind.index))
def to_input(mp):
    ren = tmt[list(mp.keys())].rename(columns=mp)
    dup = ren.columns[ren.columns.duplicated()].unique()
    common_ind = [i for i in sc_ind.values if i in set(ren.columns)]
    sub = ren.loc[:, ~ren.columns.duplicated(keep="last")][common_ind] if len(dup) else ren[common_ind]
    sub = sub.dropna(how="any")
    fin = sub.reindex(columns=sc_ind.values).rename(columns=ind2proj)
    return fin, len(dup)

P1, d1 = to_input(map_ros1d); P2, d2 = to_input(map_dl)
print("ros1d-rebuilt:", P1.shape, "donors with data", int((~P1.isna().all()).sum()), "dup individuals", d1)
print("DLPFC-only   :", P2.shape, "donors with data", int((~P2.isna().all()).sum()), "dup individuals", d2)
prot = np.load(f"{C}/proteins_input.npy"); pf = np.load(f"{C}/proteins_feats.npy")
same_feats = list(P1.index) == list(pf)
print("ros1d-rebuilt features == md_prot:", same_feats)
if same_feats:
    V1 = P1[samples].T.values
    both = ~np.isnan(V1) & ~np.isnan(prot)
    print("ros1d-rebuilt vs prot.txt: NaN pattern identical", bool((np.isnan(V1) == np.isnan(prot)).all()),
          " max |diff|", float(np.nanmax(np.abs(V1 - prot))))
has1 = set(P1.columns[~P1.isna().all()]); has2 = set(P2.columns[~P2.isna().all()])
print("donor sets: shared", len(has1 & has2), " only ros1d", len(has1 - has2), " only DLPFC", len(has2 - has1))
# Per-donor: does the row in prot.txt belong to that donor?
mis = []
for pj in has1:
    ind = clin.loc[pj, "individualID"]
    cols1 = [c for c, i in map_ros1d.items() if i == ind]
    cols2 = [c for c, i in map_dl.items() if i == ind]
    mis.append(dict(own=sorted(cols1) == sorted(cols2) and len(cols2) > 0, has_own_sample=len(cols2) > 0))
mis = pd.DataFrame(mis)
print("SDA protein donors whose row is their own sample:", int(mis.own.sum()), "/", len(mis), "; mis-attributed:", int((~mis.own).sum()),
      "; of those with no own sample in the 400:", int((~mis.has_own_sample).sum()))

# Y-linked proteins in the raw 400-column matrix
yl = [r for r in tmt.index if r.split("|")[0] in ("DDX3Y", "EIF1AY", "RPS4Y1", "KDM5D", "UTY", "USP9Y", "NLGN4Y", "ZFY")]
print("Y-linked rows in raw TMT:", yl)
sexmap = clin.set_index("individualID")["msex"].to_dict()
rows = []
for nm, mp in [("ros1d", map_ros1d), ("DLPFC-only", map_dl)]:
    for r in yl:
        x = tmt.loc[r]; y = np.array([sexmap.get(mp.get(c), np.nan) for c in tmt.columns], float)
        ok = ~np.isnan(x.values) & ~np.isnan(y)
        for subset, cols in [("all400", np.ones(400, bool)), ("112_differing", tmt.columns.isin(diff))]:
            o = ok & cols
            xx, yy = x.values[o], y[o]
            if (yy == 1).sum() < 3 or (yy == 0).sum() < 3: continue
            au = stats.mannwhitneyu(xx[yy == 1], xx[yy == 0]).statistic / ((yy == 1).sum() * (yy == 0).sum())
            rows.append(dict(mapping=nm, protein=r, subset=subset, n=int(o.sum()), auc_male_gt_female=round(au, 3)))
Y = pd.DataFrame(rows); print(Y.to_string(index=False))
Y.to_csv(f"{R}/03_tmt_sex_auc.csv", index=False)
# save corrected protein input aligned to samples for later use (cache only)
np.save(f"{C}/prot_dlpfc_only.npy", P2[samples].T.values); np.save(f"{C}/prot_dlpfc_only_feats.npy", np.array(P2.index))
np.save(f"{C}/prot_ros1d_rebuilt.npy", P1[samples].T.values)
