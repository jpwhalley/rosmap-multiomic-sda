"""Verify C278 task 7: lineage markers vs the C278 score (OPC and Oli nuclei), marker modules (own definition: mean of
within-gene rank-normalised expression), OPC share of OL-lineage nuclei (nuclei counts from covariates/rna_qc_donor.csv,
read-only), and sex difference of that share. Outputs out/v80_* (aggregates)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from vc_common import *
a = score(278); keep = ~excl("RNA"); fem = female(); R = rna(); L = loadings(278, "RNA"); fk = np.sign(L.loc["FKBP5", "X"])
MK = {"OPC": ["PDGFRA", "CSPG4", "PTPRZ1", "VCAN", "OLIG1", "OLIG2", "SOX6", "PCDH15", "LHFPL3", "MEGF11", "TNR"],
      "committed": ["GPR17", "BCAS1", "ENPP6", "TCF7L2", "ITPR2"],
      "mature": ["MOG", "MBP", "PLP1", "MOBP", "OPALIN", "MAG", "CNP", "ST18", "CLDN11", "ERMN", "KLK6"]}
rows = []
for grp, gl in MK.items():
    for g in gl:
        if g not in GENES: rows.append(dict(group=grp, gene=g, note="not in SDA gene list")); continue
        d = dict(group=grp, gene=g, PIP=L.loc[g, "P"], model_dir="rises" if np.sign(L.loc[g, "X"]) == fk else "falls")
        for ct in ("OPC", "Oli"):
            y = np.asarray(R[CTS.index(ct), :, GENES.index(g)], float); r = rho_ci(y[keep], a[keep], B=1000, seed=3)
            d[f"rho_{ct}"] = r["rho"]; d[f"lo_{ct}"] = r["lo"]; d[f"hi_{ct}"] = r["hi"]
        rows.append(d)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/v80_c278_markers.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250); print(T.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
mods = []
for grp, gl in MK.items():
    idx = [GENES.index(g) for g in gl if g in GENES]
    for ct in ("OPC", "Oli"):
        E = np.asarray(R[CTS.index(ct)], float)[:, idx]; Rk = np.apply_along_axis(ss.rankdata, 0, E); Rk = (Rk - Rk.mean(0)) / Rk.std(0)
        m = Rk.mean(1); r = rho_ci(m[keep], a[keep], B=1000, seed=4); mods.append(dict(module=grp, ct=ct, n_genes=len(idx), rho=r["rho"], lo=r["lo"], hi=r["hi"]))
qc = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str); qc = qc.loc[DON]
share = (qc.nuclei_OPC / (qc.nuclei_OPC + qc.nuclei_Oli)).values
r = rho_ci(share[keep], a[keep], B=1000, seed=5)
mods.append(dict(module="OPC/(OPC+Oli) nuclei", ct="composition", n_genes=np.nan, rho=r["rho"], lo=r["lo"], hi=r["hi"]))
print(pd.DataFrame(mods).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
print(f"OPC share median F {np.median(share[keep & fem]):.4f} vs M {np.median(share[keep & ~fem]):.4f}; MW p {ss.mannwhitneyu(share[keep & fem], share[keep & ~fem]).pvalue:.3f}")
pd.DataFrame(mods).to_csv(f"{OUT}/v80_c278_modules.csv", index=False, float_format="%.4g")
