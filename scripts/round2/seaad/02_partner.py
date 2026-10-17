"""Data S1 SEA-AD partner: validate on all 547 rows of Data_S1_corrected_547.csv with
the deposited 143-component SEA-AD genes_final, then re-match against the corrected
catalogue (deposited genes_final minus retired IDs). Read-only on all inputs."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, json
import numpy as np, pandas as pd

OUT = cfg.out_dir(cfg.AUD, "seaad141")
SEA = f"{cfg.SEAAD}/sda"
ROS = cfg.SDA
GI = f"{cfg.SEAAD}/gene_info.csv"
C547 = f"{cfg.AUD}/c547"
DS1 = f"{cfg.TABLES}/Data_S1_corrected_547.csv"
RETIRE = [int(x) for x in sys.argv[1:]] or [9, 128]

ds = pd.read_csv(DS1, dtype=str, keep_default_na=False).set_index("Component")
fn = pd.read_csv(GI, index_col=0)["feature_name"]
sg = pd.read_csv(f"{SEA}/genes_final.csv", index_col=0)
sg.columns = [fn[c] for c in sg.columns]
assert sg.columns.is_unique and len(sg) == 143
ros = open(f"{ROS}/genes.txt").read().splitlines()
X = pd.read_csv(f"{ROS}/genes_final_RNA.csv", index_col=0)
P = pd.read_csv(f"{ROS}/pip_final_RNA.csv", index_col=0)
assert list(X.columns) == ros and list(P.columns) == ros and list(X.index) == list(range(1, 547))
rows = {int(k): (X.loc[k], P.loc[k]) for k in X.index}
for k in (547, 548):
    rows[k] = (pd.Series(np.load(f"{C547}/C{k}_X1.npy"), index=ros), pd.Series(np.load(f"{C547}/C{k}_P1.npy"), index=ros))
sgset = set(sg.columns)

COMMON = sorted(sgset & set(ros))   # 12_seaad_validation.py: sorted common genes

def partner(x, p, cat, order="orig"):
    """12_seaad_validation.py: genes = sorted common genes with ROSMAP PIP > 0.5;
    r = SEA-AD row.corr(ROSMAP vector); larger-|r| of idxmax/idxmin (ties -> max)."""
    if order == "orig":
        genes = [g for g in COMMON if p[g] > 0.5]
    else:  # sensitivity: ROSMAP feature order, ROSMAP.corr(SEA-AD)
        genes = [g for g in ros if p[g] > 0.5 and g in sgset]
    if len(genes) <= 1: return None
    xs = x[genes]; sub = cat[genes]
    if order == "orig":
        r = sub.apply(lambda row: row.corr(xs), axis=1)
    else:
        r = pd.Series({j: xs.corr(sub.loc[j]) for j in sub.index})
    imax, imin = r.idxmax(), r.idxmin()
    j = imax if abs(r.max()) >= abs(r.min()) else imin
    a = r.abs().sort_values(ascending=False, kind="stable")
    return dict(comp=int(j), r=float(r[j]), n_genes=len(genes), runner_up=int(a.index[1]), runner_up_abs=float(a.iloc[1]))

out, mism = [], []
cat_full = sg; cat_corr = sg.drop(index=RETIRE)
for k in ds.index.astype(int):
    x, p = rows[k]
    a = partner(x, p, cat_full)
    pc, pr = ds.loc[str(k), "SEA-AD_comp"], ds.loc[str(k), "SEA-AD_corr"]
    rec = dict(rosmap=k, ds1_comp=pc, ds1_corr=pr, omics_count=int(ds.loc[str(k), "omics_count"]))
    if a is None:
        rec.update(full_comp="", full_corr="", n_genes=0, comp_ok=pc == "", corr_repr_ok=pr == "")
    else:
        rec.update(full_comp=repr(float(a["comp"])), full_corr=repr(a["r"]), n_genes=a["n_genes"],
                   comp_ok=repr(float(a["comp"])) == pc, corr_repr_ok=repr(a["r"]) == pr,
                   corr_absdiff=abs(a["r"] - float(pr)) if pr else np.nan)
    b = partner(x, p, cat_corr)
    rec.update(new_comp="" if b is None else repr(float(b["comp"])), new_corr="" if b is None else repr(b["r"]),
               new_runner_up="" if b is None else b["runner_up"], new_runner_up_abs="" if b is None else b["runner_up_abs"])
    rec["changed"] = (rec["new_comp"], rec["new_corr"]) != (rec["full_comp"], rec["full_corr"])  # same-environment full-143 vs corrected-141
    out.append(rec)
R = pd.DataFrame(out)
R.to_csv(f"{OUT}/partner_full_vs_corrected.csv", index=False)
print("validation vs Data S1 (full 143): partner id match", int(R.comp_ok.sum()), "/", len(R),
      "; corr repr identical", int(R.corr_repr_ok.sum()), "/", len(R),
      "; max |r diff|", R.corr_absdiff.max(), flush=True)
bad = R[~R.comp_ok | ~R.corr_repr_ok]
print("non-identical rows:\n", bad[["rosmap", "ds1_comp", "ds1_corr", "full_comp", "full_corr", "n_genes"]].to_string(), flush=True)
# rows whose repr differs: try SEA-AD column order to see if that is summation order
for k in bad.rosmap:
    x, p = rows[k]; a = partner(x, p, cat_full, order="ros")
    print(k, "ros-order:", None if a is None else (a["comp"], repr(a["r"])))
ch = R[R.changed]
print(f"\nrows changing when {RETIRE} retired: {len(ch)}", flush=True)
print(ch[["rosmap", "omics_count", "ds1_comp", "ds1_corr", "new_comp", "new_corr", "n_genes", "new_runner_up", "new_runner_up_abs"]].to_string(), flush=True)
named = R[R.ds1_comp.isin([f"{float(x)!r}" for x in RETIRE])].rosmap.tolist()
print("rows naming retired IDs:", named, "; changed rows == those rows:", sorted(ch.rosmap.tolist()) == sorted(named))
# l.201 prioritised set: |r| > 0.4 and omics_count >= 3
def prio(comp, corr):
    return set(R.rosmap[(R[corr].replace("", np.nan).astype(float).abs() > 0.4) & (R.omics_count >= 3)])
before = prio("ds1_comp", "ds1_corr"); after = prio("new_comp", "new_corr")
print("prioritised set before", len(before), sorted(before)); print("after", len(after), "; identical:", before == after)
json.dump(dict(prio_before=sorted(map(int, before)), prio_after=sorted(map(int, after))), open(f"{OUT}/prioritised_set.json", "w"))
