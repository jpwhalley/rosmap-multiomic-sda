"""Point 4 re-derivation: the replacement prioritisation with within-cohort source-data (cross-layer) support treated as
descriptive only. Supports kept: A1 phenotype (adjusted PO1 q < 0.05, sex / CERAD / cogdx), A2 genome-wide association
(autosomal min p < 5e-8/547, lambda 0.95-1.05), B SEA-AD RNA concordance (|r|, |rho| >= 0.4 same sign, null p < 0.01,
partner top share < 0.5). Dropped: C cross-layer.

Independent of the structure builder where possible: concentration recomputed from the deposited scores; A1 from the
verifier's PO refit (verify_covariates/vc_po_547.csv); A2 from verify_structure v03; B from verify_structure v04.
C can only come from structure/out/03_crosslayer.csv (the one computation). Prints aggregates and component IDs only.
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd

R2 = cfg.WORK
SDA = cfg.SDA
AUD = cfg.AUD
DS1 = cfg.DS1_V23

# ---- concentration from the scores (no per-donor output)
dep = pd.read_csv(f"{SDA}/samples_final.csv", index_col=0)
S = dep.drop(index=520).copy()
for k in (547, 548):
    S.loc[k] = np.load(f"{AUD}/c547/C{k}_A.npy")
X = S.values.astype(float)
D = X - X.mean(1, keepdims=True)
D2 = D ** 2
top = pd.Series(D2.max(1) / D2.sum(1), index=S.index)
PR = pd.Series(D2.sum(1) ** 2 / (D2 ** 2).sum(1), index=S.index)
ids = list(S.index)

ds1 = pd.read_csv(DS1, index_col=0)
assert list(ds1.index) == ids
print("top-donor share vs Data S1 (4 dp): max |diff|", float((top.round(4) - ds1.top_donor_share).abs().max()))
step1 = top < 0.10
print("step 1 (share < 0.10):", int(step1.sum()))

# ---- A1 from the verifier's PO refit, model M1
vpo = pd.read_csv(f"{R2}/verify_covariates/vc_po_547.csv")
m1 = vpo[vpo.model == "M1"].pivot(index="comp", columns="pheno", values="q").reindex(ids)
names = {"Sex": "sex", "ceradsc": "cerad", "cogdx": "cogdx"}
A1 = (m1 < 0.05).any(axis=1)
A1lab = {k: ";".join(names[p] for p in ("Sex", "ceradsc", "cogdx") if m1.loc[k, p] < 0.05) for k in ids}
# cross-check with the builder's PO1 q
po = pd.read_csv(f"{R2}/covariates/po_tests_547.csv", header=[0, 1], index_col=0)
po.index = po.index.astype(int)
b1 = pd.DataFrame({p: po[(p, "PO1_q")] for p in ("Sex", "ceradsc", "cogdx")}).reindex(ids)
print("A1 builder vs verifier call disagreements:", int(((b1 < .05).any(axis=1) != A1).sum()))

# ---- A2 from verify_structure v03
g = pd.read_csv(f"{R2}/revision_analyses/verify_structure/out/v03_gwas_547.csv", index_col=0).reindex(ids)
A2 = (g.min_p_auto < 5e-8 / 547) & g.lam.between(0.95, 1.05)

# ---- B from verify_structure v04
v = pd.read_csv(f"{R2}/revision_analyses/verify_structure/out/v04_seaad_547.csv", index_col=0).reindex(ids)
B = ((v.r.abs() >= .4) & (v.rho.abs() >= .4) & (np.sign(v.r) == np.sign(v.rho)) & (v.null_p < .01)
     & (v.partner_top_share < .5)).fillna(False)
print("B recomputed vs verifier's own B column disagreements:", int((B != v.B.astype(bool)).sum()))

# ---- C from structure 03 (as in 05_prioritise.py)
M = pd.read_csv(f"{R2}/revision_analyses/structure/out/05_master_prioritisation.csv", index_col=0).reindex(ids)
C = M.support_C_crosslayer.astype(bool)
print("components with C support (all classes):", [k for k in ids if C[k]], "| distributed:", [k for k in ids if C[k] and step1[k]])

old = step1 & (A1 | A2 | B | C)
new = step1 & (A1 | A2 | B)
print(f"\nprioritised with C: {int(old.sum())}; without C: {int(new.sum())}; lost: {[k for k in ids if old[k] and not new[k]]}")
print("matches Data S1 prioritised_2026:", bool(((ds1.prioritised_2026 == 'yes') == old).all()))
tierA_old = old & (A1 | A2 | C)
tierA_new = new & (A1 | A2)
print(f"Tier A (non-SEA-AD support) with C: {int(tierA_old.sum())}; phenotype or GWAS: {int(tierA_new.sum())}; "
      f"SEA-AD only: {int((new & B & ~A1 & ~A2).sum())}")
print("Tier A phenotype vs GWAS:", {"phenotype": int((new & A1).sum()), "GWAS": int((new & A2).sum()),
                                   "both": int((new & A1 & A2).sum())})

def lab(k, withC):
    s = (["phenotype(" + A1lab[k] + ")"] if A1[k] else []) + (["GWAS"] if A2[k] else []) + (["SEA-AD"] if B[k] else []) \
        + (["cross-layer"] if (withC and C[k]) else [])
    return "; ".join(s)
newsup = pd.Series({k: lab(k, False) if new[k] else "" for k in ids})
oldsup = pd.Series({k: lab(k, True) if old[k] else "" for k in ids})
print("old support strings reproduce Data S1:", bool((oldsup == ds1.prioritisation_support.fillna("")).all()))
chg = [k for k in ids if newsup[k] != ds1.prioritisation_support.fillna("")[k]]
print("rows whose prioritisation_support would change:", {k: (ds1.prioritisation_support[k], newsup[k]) for k in chg})
print("combinations without C:", newsup[new].str.replace(r"\(.*?\)", "", regex=True).value_counts().to_dict())

print("\nretained and named components (criteria met):")
for k in (74, 278, 172, 277, 21, 545):
    print(f"  C{k}: share {top[k]:.4f} PR {PR[k]:.1f} | A1 {A1lab[k] or '-'} | A2 {bool(A2[k])} "
          f"(auto min p {g.min_p_auto[k]:.2g}, lambda {g.lam[k]:.3f}) | B {bool(B[k])} (r {v.r[k]:+.3f}, rho {v.rho[k]:+.3f}, "
          f"null p {v.null_p[k]:.2g}, partner share {v.partner_top_share[k]:.2f}) | C {bool(C[k])} -> {'yes' if new[k] else 'NO'}")
# robustness rows that include C in 05_robustness: does dropping C change any threshold variant?
print("\nthreshold variants with and without C:")
for t in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30, 0.50):
    a = int(((top < t) & (A1 | A2 | B | C)).sum()); b = int(((top < t) & (A1 | A2 | B)).sum())
    five = all(((top < t) & (A1 | A2 | B))[k] for k in (74, 278, 172, 277, 21))
    print(f"  share < {t}: with C {a}, without C {b}; five retained captured without C: {five}")
for pr in (10, 20, 30, 50):
    a = int(((PR >= pr) & (A1 | A2 | B | C)).sum()); b = int(((PR >= pr) & (A1 | A2 | B)).sum())
    print(f"  PR >= {pr}: with C {a}, without C {b}")
print("drop B (phenotype or GWAS only), without C:", int((step1 & (A1 | A2)).sum()),
      "| drop A2, without C:", int((step1 & (A1 | B)).sum()))
# the 'support from outside the fit' counts over all 547 before step 1
print("support counts over 547 without C: any", int((A1 | A2 | B).sum()), "| with C", int((A1 | A2 | B | C).sum()),
      "| excluded by step 1 without C:", int((~step1 & (A1 | A2 | B)).sum()), "with C:", int((~step1 & (A1 | A2 | B | C)).sum()))
