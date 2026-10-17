"""VERIFY part 11: do the phenotype associations of the prioritised components hold under three alternative tests?
Results: "the associations of the four phenotype-associated components held under three alternative tests (Methods)".

The 34 prioritised components with a covariate-adjusted phenotype association (support A1: PO1 q < 0.05 for sex,
CERAD or cognitive diagnosis) are re-tested with
  alt      the alternative phenotype model (phenotype as outcome; v05_pheno.py),
  PO2      the PO1 model with the RNA QC proxies added (covariates/po_tests_547.csv),
  jointBH  PO1 p-values with Benjamini-Hochberg across all 3 x 547 tests jointly instead of per phenotype,
and a component "loses A1" under a test when none of its three q-values is below 0.05.

Inputs and the definitions of A1, A2 and the prioritised set are those of v07_prioritise_verify.py (run v01 to v05
first). In the original analysis this check was run inline after v07 (record log out/v11_tierA_fragility.log, whose
three lines this script reproduces); the per-association table at the end is added here. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
V = cfg.out_dir(cfg.WORK, "revision_analyses", "verify_structure"); cfg.out_dir(V, "out")
ST = f"{V}/../structure"
C = pd.read_csv(f"{V}/private/v02_core_plus.csv", index_col=0)
G = pd.read_csv(f"{V}/out/v03_gwas_547.csv", index_col=0)
SE = pd.read_csv(f"{V}/out/v04_seaad_547.csv", index_col=0)
PH = pd.read_csv(f"{V}/out/v05_pheno_547.csv", index_col=0)
po = pd.read_csv(f"{cfg.WORK}/covariates/po_tests_547.csv", header=[0, 1], index_col=0)
SM = pd.read_csv(f"{ST}/out/05_master_prioritisation.csv", index_col=0)
M = C.join(G).join(SE.rename(columns={"top3_share": "seaad_top3_share"}))
P3 = ["Sex", "ceradsc", "cogdx"]
for ph in P3:
    M[f"q_{ph}"] = po[(ph, "PO1_q")].values; M[f"qqc_{ph}"] = po[(ph, "PO2_q")].values
    M[f"qalt_{ph}"] = PH[f"alt_{ph}_q"].values

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum(1, np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]); return q
qj = bh(np.concatenate([po[(p, "PO1_p")].values for p in P3])).reshape(3, -1).T
for i, ph in enumerate(P3):
    M[f"qj_{ph}"] = qj[:, i]

# supports and prioritisation as in v07
A1 = (M[[f"q_{p}" for p in P3]] < .05).any(axis=1)
A2 = (M.min_p_auto < 5e-8 / 547) & M.lam.between(0.95, 1.05)
B = M.B.fillna(False).astype(bool)
Cx = SM.support_C_crosslayer.astype(bool).reindex(M.index)
pop = M.top_share < 0.10
pri = pop & (A1 | A2 | B | Cx)
tierA = pri & (A1 | A2)
report = SM.index[SM.prioritised.astype(bool) & (SM.support_A1_phenotype_adj.astype(bool) | SM.support_A2_gwas.astype(bool))]
print("Tier A ids == report list:", sorted(M.index[tierA]) == sorted(report))

TESTS = {"alt": "qalt", "PO2": "qqc", "jointBH": "qj"}
holds = {t: (M[[f"{pre}_{p}" for p in P3]] < .05).any(axis=1) for t, pre in TESTS.items()}
ph_comps = list(M.index[pri & A1])
lose = {k: [t for t in TESTS if not holds[t][k]] for k in ph_comps}
print("Tier A phenotype comps losing A1 under:", {k: v for k, v in lose.items() if v})
robust = [k for k in ph_comps if not lose[k]]
print("robust under all three:", robust, "n", len(robust))

# added for the release: the specific association described for each of the four components
FOCUS = {74: "Sex", 278: "Sex", 172: "ceradsc", 277: "cogdx"}
print("\nper association (q): PO1 | alt | PO2 | jointBH")
for k, ph in FOCUS.items():
    qs = [M.loc[k, f"{pre}_{ph}"] for pre in ("q", "qalt", "qqc", "qj")]
    print(f"  C{k} {ph:8s} " + " | ".join(f"{q:.2g}" for q in qs) + ("   holds under all three" if max(qs) < .05 else "   DOES NOT HOLD"))
