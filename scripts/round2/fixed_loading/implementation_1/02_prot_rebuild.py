"""Rebuild the SDA proteomics input exactly as ros1d_prepare_data_imputed_prot_tmt.py does
(identical code path), (i) with the original metadata (must reproduce prot.txt exactly) and
(ii) with the metadata filtered to the ROSMAP.DLPFC study first (the correction).
Both restricted to the SDA protein features (md_prot.txt), donor order md_samples.txt.
Writes cache/prot_fed.npy, cache/prot_corrected.npy, cache/prot_group.npy (per-donor category code);
prints aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from pi_common import *

all_clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv", index_col=0)
samps = [int(l) for l in open(f"{SDA}/md_samples.txt") if l.strip()]
sc_clin = all_clin.loc[samps]
tmt = pd.read_csv(f"{ROS}/prot/tmt/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv", index_col=0)
meta0 = pd.read_csv(f"{ROS}/prot/tmt/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv", index_col=0)
feats = [l.rstrip("\n") for l in open(f"{SDA}/md_prot.txt")]


def ros1d(tmt_meta):
    """verbatim logic of ros1d (steps 1-5 + complete-case + reindex + projid rename)."""
    tmt_meta = tmt_meta.copy()
    tmt_meta['R_ID'] = tmt_meta.index.str.split('.').str[-1]
    tmt_meta['SampleID'] = tmt_meta.index.str.split('.').str[-3] + '.' + tmt_meta.index.str.split('.').str[-2]
    sampleid_rid_df = tmt_meta[['R_ID']].set_index(tmt_meta['SampleID'])
    common_sample_ids = tmt.columns.intersection(sampleid_rid_df.index)
    tmt_filtered = tmt[common_sample_ids]
    sampleid_to_rid = sampleid_rid_df.loc[common_sample_ids, 'R_ID'].to_dict()
    tmt_renamed = tmt_filtered.rename(columns=sampleid_to_rid)
    common_individual_ids = set(sc_clin['individualID'].values).intersection(tmt_renamed.columns.values)
    tf = tmt_renamed[list(common_individual_ids)]
    tf = tf.dropna(how='any')
    tmt_final = tf.reindex(columns=sc_clin['individualID'].values, fill_value=np.nan)
    individual_to_projid = dict(zip(sc_clin['individualID'], sc_clin.index))
    tmt_final = tmt_final.rename(columns=individual_to_projid)
    return tmt_final, sampleid_to_rid


fed_df, map_fed = ros1d(meta0)
print("original-mapping build:", fed_df.shape, "; md_prot features present:", int(pd.Index(feats).isin(fed_df.index).sum()), "/", len(feats))
Pfed = fed_df.reindex(feats)[samps].T.values
P = read_mat(f"{SDA}/prot.txt")
same_na = bool(np.array_equal(np.isnan(Pfed), np.isnan(P)))
print("rebuild == prot.txt: NaN pattern identical", same_na, "; max |diff|", float(np.nanmax(np.abs(Pfed - P))),
      "; exact equality on observed", bool(np.array_equal(Pfed[~np.isnan(P)], P[~np.isnan(P)])))
assert same_na and np.nanmax(np.abs(Pfed - P)) < 1e-12

meta_dl = meta0[meta0.index.str.startswith("ROSMAP.DLPFC")]
cor_df, map_dl = ros1d(meta_dl)
print("DLPFC-only build:", cor_df.shape, "; md_prot features present:", int(pd.Index(feats).isin(cor_df.index).sum()), "/", len(feats))
Pcor = cor_df.reindex(feats)[samps].T.values
# compare with the earlier corrected caches (independent implementations)
for nm, f in [("source_panels prot_corrected_full", f"{cfg.WORK}/source_panels/cache/prot_corrected_full.npy")]:
    try:
        Q = np.load(f)
        print(f"  vs {nm}: NaN pattern identical", bool(np.array_equal(np.isnan(Q), np.isnan(Pcor))), "max|diff|", float(np.nanmax(np.abs(Q - Pcor))))
    except Exception as e:
        print("  could not compare", nm, e)

has_f = ~np.isnan(Pfed).all(1); has_c = ~np.isnan(Pcor).all(1)
same = np.array([has_f[i] and has_c[i] and np.allclose(Pfed[i][~np.isnan(Pcor[i])], Pcor[i][~np.isnan(Pcor[i])]) for i in range(N)])
grp = np.zeros(N, int)            # 0 no protein in either
grp[has_f & same] = 1             # own proteome fed (correct)
grp[has_f & ~same & ~has_c] = 2   # fed another donor's proteome; no own sample -> no protein after correction
grp[has_f & ~same & has_c] = 3    # fed another donor's proteome; own sample exists -> replaced
grp[~has_f & has_c] = 4           # own proteome dropped from the input; restored by the correction
print("donor categories:", {k: int((grp == k).sum()) for k in range(5)})
print("fed donors", int(has_f.sum()), " mis-paired (2+3)", int(np.isin(grp, [2, 3]).sum()), " corrected donors", int(has_c.sum()))
partial = np.isnan(Pcor[has_c]).any(1).sum()
allnan_feat = np.isnan(Pcor[has_c]).all(0)
print("corrected: donors with partial-NA rows", int(partial), "; md_prot features all-NaN among corrected donors", int(allnan_feat.sum()))
# mis-paired rows: correlation between the fed (wrong) proteome and the donor's own proteome when it exists
i3 = np.where(grp == 3)[0]
for i in i3:
    ok = ~np.isnan(Pcor[i])
    print("  category-3 donor: r(fed row, own row) =", round(float(np.corrcoef(Pfed[i][ok], Pcor[i][ok])[0, 1]), 3))
np.save(f"{CACHE}/prot_fed.npy", Pfed); np.save(f"{CACHE}/prot_corrected.npy", Pcor); np.save(f"{CACHE}/prot_group.npy", grp)
np.save(f"{CACHE}/prot_corrected_allnan_feat.npy", allnan_feat)
