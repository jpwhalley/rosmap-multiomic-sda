"""Own per-donor nucleus counts (class and high-resolution subtype) from the labelled snRNA-seq object obs, with the
ros0a filters (failed projids from $G_ROS_QC_EXCLUDE_PATH, predicted doublets, six classes, >=100 nuclei per class). Compared with the
neuronal_c21 cache and the covariates QC table (aggregate equality only)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import h5py, numpy as np, pandas as pd
from v_common import *
f = h5py.File(f"{ROS}/mit/rna/mit_all_doublets_labelled.h5ad", "r"); o = f["obs"]
print("obs keys:", list(o.keys())[:40])
def col(k):
    g = o[k]
    if isinstance(g, h5py.Group):
        cats = g["categories"][()]; cats = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in cats], dtype=object)
        return cats[g["codes"][()]]
    v = g[()]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v], dtype=object) if v.dtype.kind in "OS" else v
pj = col("projid"); maj = col("major_cell_type"); sub = col("cell_type_high_resolution"); dbl = col("predicted_doublets")
print("doublet values:", pd.Series(dbl).astype(str).value_counts().to_dict())
pj = pd.Series(pj).astype(float).astype(int).astype(str).values
keep = (~np.isin(pj, [str(x) for x in cfg.qc_failed_projids()])) & (pd.Series(dbl).astype(str).values == "False") & np.isin(maj, ["Exc", "Oli", "Inh", "Ast", "OPC", "Mic"])
df = pd.DataFrame({"pj": pj[keep], "maj": maj[keep], "sub": sub[keep]})
cls = df.groupby(["pj", "maj"]).size().unstack(fill_value=0)
valid = cls.index[(cls >= 100).all(1)]
don = donors(); print("valid donors", len(valid), "SDA donors covered", len(set(don) & set(valid)))
cls = cls.loc[don]
sb = df[df.pj.isin(don)].groupby(["pj", "sub"]).size().unstack(fill_value=0).loc[don]
mo = df.drop_duplicates("sub").set_index("sub").maj
sb.columns = [f"{mo[c]}|{c}" for c in sb.columns]
theirs = pd.read_csv(f"{cfg.WORK}/revision_analyses/neuronal_c21/cache/subtype_counts.csv", index_col=0); theirs.index = theirs.index.astype(str)
common = sorted(set(theirs.columns) & set(sb.columns))
print("subtype columns mine/theirs/common:", sb.shape[1], theirs.shape[1], len(common), "| max |diff|:", int(np.abs(sb.loc[don, common].values - theirs.loc[don, common].values).max()))
qc = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str)
print("class counts max |diff| vs covariates QC:", int(np.abs(cls[["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]].values - qc.loc[don, [f"nuclei_{c}" for c in ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]]].values).max()))
out = pd.concat([cls.add_prefix("cls|"), sb], axis=1).reset_index(drop=True)
out.to_csv(f"{CACHE}/counts.csv", index=False)
print("Exc subtypes:", sum(c.startswith("Exc|") for c in sb.columns), "Inh subtypes:", sum(c.startswith("Inh|") for c in sb.columns))
