"""(1) Reproduce the published unadjusted tests (546 family = Data S1 as published; 547 family = 10-08 table).
(2) Own proportional-odds (cumulative-logit) LRT models for all 547 components x 3 phenotypes.
(3) Own RINT-OLS F-test models for all 547 x 3.
Writes vc_unadj.csv, vc_po_547.csv, vc_rint_547.csv (component-level only)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import time, numpy as np, pandas as pd
from scipy import stats as ss
from vc_common import *

s = load_scores(); don = s.columns; cl = load_clin(don); qc = load_qc(don)
print("donors", len(don), "| components", len(s))
print("missing per covariate among 276:", {c: int(cl[c].isna().sum()) for c in ["msex", "age", "pmi", "Study", "ceradsc", "cogdx"]},
      "| QC missing", int(qc[["nuclei", "umis"]].isna().sum().sum()), "| ages coded 90 from '90+':", int(cl.age90.sum()))

# ---------------- (1) published tests ----------------
raw = pd.DataFrame([published_tests(s.loc[k].values, cl) for k in s.index], index=s.index, columns=["Sex", "ceradsc", "cogdx"])
q547 = raw.apply(bh)
ref = pd.read_csv(f"{cfg.TABLES}/clinical_BH_547_family.csv", index_col=0)
ref = ref.loc[q547.index]
for c in raw:
    print(f"547 family {c}: max|q - 10-08 table| {float((q547[c] - ref[c]).abs().max()):.1e}; calls {int((q547[c] < .05).sum())} vs table {int((ref[c] < .05).sum())}; same set {set(q547.index[q547[c] < .05]) == set(ref.index[ref[c] < .05])}")
# 546 family (published Data S1)
s546 = pd.read_csv(f"{B}/sda/samples_final.csv", index_col=0); s546.columns = s546.columns.astype(str)
raw546 = pd.DataFrame([published_tests(s546.loc[k].values, cl) for k in s546.index], index=s546.index, columns=raw.columns)
q546 = raw546.apply(bh)
ds1 = pd.read_csv(cfg.DS1_V22, index_col=0)
for c in raw:
    pub = ds1[c].dropna()
    print(f"546 family {c}: published calls {len(pub)}, reproduced calls {int((q546[c] < .05).sum())}, same set {set(pub.index) == set(q546.index[q546[c] < .05])}, max|q diff| {float((q546.loc[pub.index, c] - pub).abs().max()):.1e}")
calls546 = {c: set(ds1[c].dropna().index) for c in raw}
calls547 = {c: set(q547.index[q547[c] < .05]) for c in raw}
print("total calls: 546 family", sum(len(v) for v in calls546.values()), "| 547 family", sum(len(v) for v in calls547.values()))
print("547 vs 546 call differences:", {c: (sorted(calls547[c] - calls546[c]), sorted(calls546[c] - calls547[c])) for c in raw})
out = raw.add_prefix("p_").join(q547.add_prefix("q_"))
out.to_csv("vc_unadj.csv")

# ---------------- covariate matrices ----------------
age, pmi, mp, male = cl.age.values, cl.pmi.values, cl.is_map.values, cl.male.values
lnuc, lupn = np.log(qc.nuclei.values), np.log(qc.upn.values)
a90, lpmi = cl.age90.values, np.log(cl.pmi.values)
cog = cl.cogdx.values; sub = np.isin(cog, [1, 2, 4])
PH = {"Sex": (np.ones(276, bool), male[:, None]),
      "ceradsc": (np.ones(276, bool), cl.cerad_ad.values[:, None]),
      "cogdx": (sub, np.column_stack([(cog == 2), (cog == 4)]).astype(float))}
def covs(model, ph):
    base = {"M0": [], "M1": [age, pmi, mp], "M2": [age, pmi, mp, lnuc, lupn], "M3": [age, a90, lpmi, mp],
            "Age": [age], "PMI": [pmi], "Cohort": [mp], "SexOnly": [], "QC": [lnuc, lupn]}[model]
    if ph != "Sex" and model in ("M1", "M2", "M3", "SexOnly"):
        base = base + [male]
    return np.column_stack(base) if base else np.zeros((276, 0))
MODELS = ["M0", "M1", "M2", "M3", "Age", "PMI", "Cohort", "SexOnly", "QC"]

# ---------------- (2) proportional odds ----------------
t0 = time.time(); rows = []
for ph, (m, Xp) in PH.items():
    for model in MODELS:
        if model == "SexOnly" and ph == "Sex":
            continue
        Xc = covs(model, ph)[m]
        for k in s.index:
            y = s.loc[k].values[m]
            r = po_lrt(y, Xc, Xp[m])
            rows.append(dict(comp=k, pheno=ph, model=model, p=r["p"], beta=r["beta"][-1], beta_mci=r["beta"][0] if ph == "cogdx" else np.nan, conv=r["conv"]))
    print(f"PO {ph} done ({time.time() - t0:.0f}s)", flush=True)
po = pd.DataFrame(rows)
po["q"] = po.groupby(["pheno", "model"]).p.transform(bh)
po.to_csv("vc_po_547.csv", index=False)
print("non-converged PO fits:", int((~po.conv).sum()), po.loc[~po.conv, ["comp", "pheno", "model", "p"]].to_string() if (~po.conv).any() else "")

# ---------------- (3) RINT-OLS ----------------
R = np.column_stack([rint(s.loc[k].values) for k in s.index])   # RINT over all 276 donors
rows = []
for ph, (m, Xp) in PH.items():
    for model in ["M0", "M1", "M2", "M3"]:
        p, b = ols_F(R[m], covs(model, ph)[m], Xp[m])
        for j, k in enumerate(s.index):
            rows.append(dict(comp=k, pheno=ph, model=model, p=p[j], beta=b[-1, j]))
ri = pd.DataFrame(rows)
ri["q"] = ri.groupby(["pheno", "model"]).p.transform(bh)
ri.to_csv("vc_rint_547.csv", index=False)
print("done", f"{time.time() - t0:.0f}s")
