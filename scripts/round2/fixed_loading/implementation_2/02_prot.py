"""Independent rebuild of the SDA proteomics input.
(i) ros1d logic (read from the backup copy of ros1d_prepare_data_imputed_prot_tmt.py) with the full metadata:
    must reproduce prot.txt (and the archived prot_crossed_with_sc.csv);
(ii) the same logic with the metadata restricted to the ROSMAP.DLPFC study (the study the 400-column matrix belongs to).
Both restricted to md_prot.txt features, donor order md_samples.txt. Writes cache/prot_{orig,corr}.npy and
cache/prot_cat.npy (per-donor category); prints aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from vp_common import *

clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv", index_col=0)
samps = [int(x) for x in open(f"{SDA}/md_samples.txt").read().split()]
assert len(samps) == N
sc_clin = clin.loc[samps]
T = f"{cfg.ROS}/prot/tmt"
tmt = pd.read_csv(f"{T}/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv", index_col=0)
meta = pd.read_csv(f"{T}/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv", index_col=0)
feats = open(f"{SDA}/md_prot.txt").read().split("\n"); feats = [f for f in feats if f]
log("tmt", tmt.shape, "meta", meta.shape, "md_prot", len(feats))


def build(mt):
    rid = mt.index.str.split(".").str[-1]
    sid = mt.index.str.split(".").str[-3] + "." + mt.index.str.split(".").str[-2]
    s = pd.Series(np.asarray(rid), index=np.asarray(sid))
    common = tmt.columns.intersection(s.index)
    mp = s.loc[common].to_dict()                         # duplicate SampleIDs: last occurrence wins (as in ros1d)
    ren = tmt[common].rename(columns=mp)
    ids = set(sc_clin.individualID.values).intersection(ren.columns.values)
    f = ren[list(ids)].dropna(how="any")
    f = f.reindex(columns=sc_clin.individualID.values)
    f.columns = sc_clin.index.values
    return f, mp


F0, mp0 = build(meta)
F1, mp1 = build(meta[meta.index.str.startswith("ROSMAP.DLPFC.")])
log("orig build", F0.shape, "donors", int((~F0.isna().all()).sum()), "| DLPFC build", F1.shape, "donors", int((~F1.isna().all()).sum()))
log("md_prot features in orig build", int(pd.Index(feats).isin(F0.index).sum()), "in DLPFC build", int(pd.Index(feats).isin(F1.index).sum()),
    "| orig build features == md_prot (as set)", set(F0.index) == set(feats))

P0 = F0.reindex(feats)[samps].T.values
P = np.load(f"{CACHE}/in_prot.npy")
log("orig rebuild vs prot.txt: NaN identical", bool(np.array_equal(np.isnan(P0), np.isnan(P))),
    "max|diff|", float(np.nanmax(np.abs(P0 - P))), "exactly equal", bool(np.array_equal(P0[~np.isnan(P)], P[~np.isnan(P)])))
arc = pd.read_csv(f"{T}/prot_crossed_with_sc.csv", index_col=0)
arc.columns = arc.columns.astype(int)
log("archived prot_crossed_with_sc.csv", arc.shape, "vs orig build: same features", list(arc.index) == list(F0.index),
    "max|diff|", float(np.nanmax(np.abs(arc[samps].values - F0[samps].values))))

P1 = F1.reindex(feats)[samps].T.values
h0 = ~np.isnan(P0).all(1); h1 = ~np.isnan(P1).all(1)
ind = sc_clin.individualID.values
# per donor: is the fed row the donor's own sample? (own = the DLPFC-study channel(s) labelled with this individual)
own_cols = {i: sorted(c for c, v in mp1.items() if v == i) for i in ind}
fed_cols = {i: sorted(c for c, v in mp0.items() if v == i) for i in ind}
cat = np.zeros(N, int)
for n, i in enumerate(ind):
    if h0[n] and fed_cols[i] == own_cols[i]: cat[n] = 1
    elif h0[n] and not h1[n]: cat[n] = 2
    elif h0[n] and h1[n]: cat[n] = 3
    elif (not h0[n]) and h1[n]: cat[n] = 4
log("categories 0 none / 1 own / 2 other, no own / 3 other, own exists / 4 own dropped:", {c: int((cat == c).sum()) for c in range(5)})
# cross-check the category by value: category-1 rows identical to the corrected rows on shared features
ok = ~np.isnan(P1).all(0)
same = np.array([h0[n] and h1[n] and np.allclose(P0[n, ok], P1[n, ok]) for n in range(N)])
log("value check: rows identical orig vs corrected", int(same.sum()), "== category 1", bool((same == (cat == 1)).all()))
for n in np.where(cat == 3)[0]:
    log("  category-3 donor r(fed, own) =", round(float(np.corrcoef(P0[n, ok], P1[n, ok])[0, 1]), 3))
# columns whose individual changed and where the fed proteomes come from
chg = [c for c in tmt.columns if mp0.get(c) != mp1.get(c)]
log("TMT columns whose label differs between mappings", len(chg), "batches", sorted({c.split('.')[0] for c in chg})[0], "..", sorted({c.split('.')[0] for c in chg})[-1])
# donors fed another donor's proteome: does the fed TMT column belong (in DLPFC labels) to another SDA donor?
inv1 = {v: c for c, v in mp1.items()}
src_in_sda = 0
for n in np.where(np.isin(cat, [2, 3]))[0]:
    c = fed_cols[ind[n]][0]; true_ind = mp1.get(c)
    src_in_sda += int(true_ind in set(ind))
log("mis-paired rows whose source proteome belongs to another SDA donor:", src_in_sda, "of", int(np.isin(cat, [2, 3]).sum()))
log("corrected: md_prot features absent (all-NaN)", int((~ok).sum()), "; partial-NA rows", int((np.isnan(P1[h1][:, ok])).any(1).sum()))
np.save(f"{CACHE}/prot_orig.npy", P0); np.save(f"{CACHE}/prot_corr.npy", P1); np.save(f"{CACHE}/prot_cat.npy", cat)
