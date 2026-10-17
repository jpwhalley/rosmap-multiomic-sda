"""Independent verification helpers (verify_neuronal_c21, 2026-10-13). Own loaders written from the raw archive files;
nothing imported from the neuronal_c21 or component_dossier code. Read-only on the archive and other outputs.
Writes only under $G_ROS_RESULTS_DIR/revision_analyses/verify_neuronal_c21/out; per-donor arrays go to
$G_ROS_RESULTS_DIR/revision_analyses/verify_neuronal_c21/cache."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats as ss
ROS = cfg.ROS
R2 = cfg.WORK
ME = f"{R2}/revision_analyses/verify_neuronal_c21"
OUT, CACHE = cfg.out_dir(ME, "out"), cfg.out_dir(cfg.WORK, "revision_analyses", "verify_neuronal_c21", "cache")
CT_ORDER = [l.strip() for l in open(f"{ROS}/sda/cell_types.txt") if l.strip()]

def donors():
    return [l.strip() for l in open(f"{ROS}/sda/md_samples.txt") if l.strip()]

def scores():
    A = pd.read_csv(f"{ROS}/sda/samples_final.csv", index_col=0); A.columns = A.columns.astype(str)
    assert list(A.columns) == donors(); return A

def frame():
    don = donors(); c = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv")
    c.index = c.projid.astype(int).astype(str); c = c.loc[don]
    F = pd.DataFrame(index=don)
    a = c.age_death.astype(str)
    F["age"] = np.minimum(pd.to_numeric(a.str.replace("+", "", regex=False)), 90.0)
    F["pmi"] = c.pmi.astype(float); F["ros"] = (c.Study == "ROS").astype(float); F["male"] = c.msex.astype(float)
    F["ceradsc"] = c.ceradsc.astype(float); F["cerad_ad"] = c.ceradsc.isin([1, 2]).astype(float)
    F["cerad_burden"] = 5 - c.ceradsc.astype(float)
    F["braak"] = c.braaksc.astype(float); F["cogdx"] = c.cogdx.astype(float); F["mmse"] = c.cts_mmse30_lv.astype(float)
    F["mci"] = (c.cogdx == 2).astype(float); F["addem"] = (c.cogdx == 4).astype(float)
    key = pd.read_csv(f"{cfg.PRIVATE}/donor_key.csv"); key["projid"] = key.projid.astype(str)
    F["rna_mismatch"] = F.index.isin(key.loc[key.label.isin(["RNA-1", "RNA-2"]), "projid"]).astype(float)
    return F

def rint(x):
    x = np.asarray(x, float); return ss.norm.ppf((ss.rankdata(x) - 0.5) / len(x))   # Blom-free variant (differs from 0.375 offset on purpose)
