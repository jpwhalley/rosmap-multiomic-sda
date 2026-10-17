"""Shared loaders for the component dossier (2026-10-13 round 2).
Read-only on the archive ($G_ROS_ROSMAP_DIR), the audit/round-2 outputs and the corrected tables.
Writes only under $G_ROS_RESULTS_DIR/component_dossier/out (aggregates) and $G_ROS_RESULTS_DIR/component_dossier/cache (per donor)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats as ss

ROS = cfg.ROS
SDA = cfg.SDA
AUD = cfg.AUD
R2 = cfg.WORK
HERE = os.path.dirname(os.path.abspath(__file__))
DOS = f"{R2}/component_dossier"
OUT = f"{DOS}/out"
CACHE = f"{cfg.WORK}/component_dossier/cache"
cfg.out_dir(OUT); cfg.out_dir(CACHE)
NEXT = cfg.TABLES
BCOR = f"{NEXT}/cell_type_loadings_corrected.csv"
REF547 = f"{NEXT}/clinical_BH_547_family.csv"
COMPS = [351, 2, 74, 87, 278, 177, 21, 172, 277, 545]
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]   # rna.txt block order
MODS = {"RNA": "genes.txt", "meth": "md_cpg.txt", "chip": "md_chip.txt", "proteins": "md_prot.txt"}
INPUT = {"RNA": "rna.txt", "meth": "meth.txt", "chip": "chip.txt", "proteins": "prot.txt"}


def samples():
    return [l.strip() for l in open(f"{SDA}/md_samples.txt") if l.strip()]


def scores547():
    """547-component sample scores (C520 retired, C547/C548 appended), index = component, columns = projid (str)."""
    dep = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0); dep.columns = dep.columns.astype(str)
    assert list(dep.columns) == samples()
    S = dep.drop(index=520).copy()
    for k in (547, 548):
        S.loc[k] = np.load(f"{AUD}/c547/C{k}_A.npy")
    return S


def clinical():
    clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv")
    clin.index = clin.projid.astype(int).astype(str)
    return clin.loc[samples()]


def design_frame():
    """Covariates as in covariates/_common.py (age 90+ -> 90, PMI, cohort, QC proxies) plus composition."""
    clin = clinical(); don = clin.index
    qc = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str); qc = qc.loc[don]
    D = pd.DataFrame(index=don)
    D["sex"] = clin.msex.astype(float)                       # 1 = male
    D["cerad_ad"] = clin.ceradsc.map({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0})
    D["ceradsc"] = clin.ceradsc.astype(float)
    D["cogdx"] = clin.cogdx.astype(float)
    age = clin.age_death.astype(str)
    D["age90p"] = (age == "90+").astype(float)
    D["age"] = pd.to_numeric(age.replace("90+", "90")).clip(upper=90.0)
    D["pmi"] = clin.pmi.astype(float); D["logpmi"] = np.log(D.pmi)
    D["ros"] = (clin.Study == "ROS").astype(float)
    D["lognuc"] = np.log(qc.nuclei); D["logupn"] = np.log(qc.umis_per_nucleus)
    tot = qc[[f"nuclei_{c}" for c in CTS]].sum(1)
    for c in CTS:
        D[f"prop_{c}"] = qc[f"nuclei_{c}"] / tot
    return D


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o]
    adj = np.minimum.accumulate((r * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(adj, 1); return out


def rint(x):
    x = np.asarray(x, float); n = len(x)
    return ss.norm.ppf((ss.rankdata(x) - 0.375) / (n + 0.25))


def read_rows(path, rows):
    keep = set(rows)
    return pd.read_csv(path, index_col=0, skiprows=lambda i: i != 0 and i not in keep)


def robust_z(x):
    med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / mad
