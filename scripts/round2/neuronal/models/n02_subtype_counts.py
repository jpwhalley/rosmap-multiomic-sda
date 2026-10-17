"""Per-donor nucleus counts by major class and high-resolution subtype, from the obs of the labelled snRNA-seq object
(mit_all_doublets_labelled.h5ad), with the filters of ros0_prepare_data.py: two failed projids removed ($G_ROS_QC_EXCLUDE_PATH), predicted
doublets removed, six major classes, donors with >= 100 nuclei in every class. Only obs is read (h5py).
Writes cache/subtype_counts.csv (donor x subtype; local, controlled data). Prints aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import h5py, numpy as np, pandas as pd
from nc_common import *

FN = f"{ROS}/mit/rna/mit_all_doublets_labelled.h5ad"
f = h5py.File(FN, "r"); o = f["obs"]
def cat(k):
    g = o[k]; cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][()]], dtype=object)
    return cats[g["codes"][()]]
pj = o["projid"][()]; maj = cat("major_cell_type"); hr = cat("cell_type_high_resolution"); dbl = cat("predicted_doublets")
print("nuclei in object:", len(pj), "| doublet labels:", sorted(set(dbl)))
keep = (~np.isin(pj, cfg.qc_failed_projids())) & (np.array([str(x) for x in dbl]) == "False") & np.isin(maj, CTS)
df = pd.DataFrame({"projid": pj[keep].astype(str), "major": maj[keep], "sub": hr[keep]})
cnt = df.groupby(["projid", "major"]).size().unstack(fill_value=0)
valid = cnt.index[(cnt[CTS] >= 100).all(1)]
don = list(scores547().columns)
print("donors passing >=100 per class:", len(valid), "| SDA donors among them:", len(set(don) & set(valid)))
# check against the counts the QC covariates used
qc = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str)
diff = (cnt.loc[don, CTS].values - qc.loc[don, [f"nuclei_{c}" for c in CTS]].values)
print("max |difference| vs covariates/rna_qc_donor.csv class counts:", int(np.abs(diff).max()))
sub = df[df.projid.isin(don)].groupby(["projid", "sub"]).size().unstack(fill_value=0).loc[don]
maj_of = df.drop_duplicates("sub").set_index("sub").major.to_dict()
sub.columns = [f"{maj_of[c]}|{c}" for c in sub.columns]
sub.to_csv(f"{CACHE}/subtype_counts.csv")
for c in ["Exc", "Inh"]:
    cols = [x for x in sub.columns if x.startswith(c + "|")]
    print(f"{c}: {len(cols)} subtypes; median nuclei per donor {int(sub[cols].sum(1).median())}")
    print("   ", ", ".join(x.split("|")[1] for x in cols))
