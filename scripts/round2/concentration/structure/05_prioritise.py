"""Part 2: a replacement prioritisation that selects population-level signal.

Step 1 (population level): centred top-donor share < T, T = 0.10 (primary; justified in 04 from the reference simulation and
        the catalogue distribution). Robustness: T in {0.05 ... 0.50} and participation-ratio cut-offs.
Step 2 (support: at least one of, each measured independently of the decomposition's own fit):
  A  donor-trait association
     A1 clinical phenotype, adjusted: proportional-odds LRT with age at death (90+ -> 90), PMI, cohort [+ sex], BH over the
        547 components per phenotype, q < 0.05 (sex, CERAD AD vs non-AD, cogdx 3-group) [covariates/po_tests_547.csv, PO1]
     A2 genotype: study-wide genome-wide significance on autosomal SNPs, GEMMA min p (chr 1-22) < 5e-8 / 547 = 9.1e-11,
        genomic-control lambda 0.95-1.05 (unplaced / X / Y probes set aside: C74's hits are all unplaced sex-linked probes)
  B  cross-cohort RNA signature: matched SEA-AD component with |Pearson| >= 0.4 AND |Spearman| >= 0.4 over the matched genes,
     same sign, beating the gene-permutation null (null A, B = 2000) at p < 0.01, and the SEA-AD partner itself not
     concentrated (top-sample share < 0.5)
  C  cross-layer support in measured data: >= 2 layers, each with >= 1 feature tracking the score at layer-wide Bonferroni
     significance in the loading's direction (mislabelled samples and top 1% removed), and the two layers' aggregates
     correlated with each other (Spearman rho >= 0.3, p < 0.001; protein layer taken from the corrected mapping)
Writes out/05_prioritised.csv, out/05_robustness.csv, out/05_named_components.csv, out/05_prioritise.log (via tee)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss
from st_common import *

M = pd.read_csv(f"{OUT}/04_master_with_seaad_tech.csv", index_col=0)
XL = pd.read_csv(f"{OUT}/03_crosslayer.csv")
ds1 = pd.read_csv(DS1, index_col=0)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60)

# ---------------------------------------------------------------- support flags
def supportA1(model="adj"):
    return (M[f"{model}_sex_q"] < .05) | (M[f"{model}_cerad_q"] < .05) | (M[f"{model}_cogdx_q"] < .05)
def supportA2(th=5e-8 / 547, col="gwas_min_p_autosomal"):
    return (M[col] < th) & M.gwas_lambda.between(0.95, 1.05)
def supportB(r=0.4, p=0.01, partner_conc=0.5):
    return ((M.seaad_pearson.abs() >= r) & (M.seaad_spearman.abs() >= r) & (np.sign(M.seaad_pearson) == np.sign(M.seaad_spearman))
            & (M.nullA_p_J < p) & (M.seaad_partner_top_share < partner_conc)).fillna(False)
def supportC(rho=0.3):
    out = pd.Series(False, index=M.index)
    for k, g in XL.groupby("component"):
        lay = g[g.variant.isin(["fed", "corrected"]) & ~((g.layer == "proteins") & (g.variant == "fed"))]
        good = set(lay.layer[lay.n_bonf_pass_predicted_sign >= 1])
        cr = g[(g.variant == "cross-layer") & ~g.layer.str.contains(r"proteins(?!_corrected)", regex=True)]
        ok = False
        for _, row in cr.iterrows():
            a, b = [x.replace("_corrected", "") for x in row.layer.split(" vs ")]
            if a in good and b in good and row.cross_rho >= rho and row.cross_p < 1e-3: ok = True
        out.loc[k] = ok
    return out

A1, A2, B, C = supportA1(), supportA2(), supportB(), supportC()
M["support_A1_phenotype_adj"] = A1; M["support_A2_gwas"] = A2; M["support_B_seaad"] = B; M["support_C_crosslayer"] = C
M["supported"] = A1 | A2 | B | C
def support_str(k):
    s = []
    if A1[k]: s.append("phenotype(" + M.loc[k, "adj_calls"] + ")")
    if A2[k]: s.append("GWAS")
    if B[k]: s.append("SEA-AD")
    if C[k]: s.append("cross-layer")
    return "; ".join(s)
M["support"] = [support_str(k) for k in M.index]
T = 0.10
M["population_level"] = M.top_share < T
M["prioritised"] = M.population_level & M.supported

print("=== Support criteria over all 547 (before step 1), by concentration class")
cls = ["distributed", "few-donor", "concentrated"]
tab = pd.DataFrame({nm: {c: int(v[M.conc_class == c].sum()) for c in cls} for nm, v in
                    [("A1 phenotype (adj)", A1), ("A2 GWAS study-wide", A2), ("B SEA-AD (r & rho >= 0.4, null p < 0.01)", B),
                     ("C cross-layer", C), ("any support", M.supported)]}).T
tab["all"] = tab.sum(1); print(tab.to_string())
print("\nautosomal GWAS hits below the study-wide threshold but failing the lambda condition:",
      len(M.index[(M.gwas_min_p_autosomal < 5e-8 / 547) & ~M.gwas_lambda.between(0.95, 1.05)]), "components, by class:",
      M.conc_class[(M.gwas_min_p_autosomal < 5e-8 / 547) & ~M.gwas_lambda.between(0.95, 1.05)].value_counts().to_dict())
print("study-wide GWAS hits that are not autosomal:", list(M.index[(M.gwas_min_p < 5e-8 / 547) & ~(M.gwas_min_p_autosomal < 5e-8 / 547)]))

P = M[M.prioritised].copy()
print(f"\n=== Prioritised (top-donor share < {T} AND >= 1 support): {len(P)} components")
cols = ["top_share", "PR", "n_modalities", "true_runs", "support", "adj_sex_beta", "adj_cerad_beta", "adj_cogdx_beta",
        "seaad_partner", "seaad_pearson", "seaad_spearman", "nullA_p_J", "gwas_min_p_autosomal", "rho_log nuclei", "max_abs_rho_composition"]
print(P[cols].to_string(float_format=lambda v: f"{v:.3g}"))
out = P[cols].join(ds1[["Top_positive_cell_type", "Top_negative_cell_type", "Top_Positive_RNA", "Top_Negative_RNA",
                        "GeneExpression_Pathway"]])
out.to_csv(f"{OUT}/05_prioritised.csv")
print("\nsupport combinations:", P.support.str.replace(r"\(.*?\)", "", regex=True).value_counts().to_dict())
print("focus five captured:", {k: bool(M.loc[k, "prioritised"]) for k in FOCUS})

# ---------------------------------------------------------------- distributed but unsupported / supported-but-concentrated
print(f"\npopulation-level (step 1) components: {int(M.population_level.sum())}; of these supported: {len(P)}; unsupported: {int((M.population_level & ~M.supported).sum())}")
ex = M[~M.population_level & M.supported]
print(f"components with a support flag but excluded by step 1: {len(ex)} "
      f"(few-donor {int((ex.conc_class == 'few-donor').sum())}, concentrated {int((ex.conc_class == 'concentrated').sum())}); by support:",
      {nm: int(v[~M.population_level].sum()) for nm, v in [("A1", A1), ("A2", A2), ("B", B), ("C", C)]})

# ---------------------------------------------------------------- named components: why in / why out
NAMED = FOCUS + OTHER_NAMED + OUTGOING + [k for k in OLD_PRIORITISED if k not in OUTGOING + FOCUS]
def why(k):
    r = M.loc[k]; s = []
    s.append(f"top-donor share {r.top_share:.3f} (PR {r.PR:.1f}) -> {'passes' if r.population_level else 'fails'} step 1")
    s.append("A1 " + ("yes: " + r.adj_calls if A1[k] else "no adjusted phenotype call"))
    s.append("A2 " + (f"yes (autosomal min p {r.gwas_min_p_autosomal:.1e})" if A2[k] else f"no (autosomal min p {r.gwas_min_p_autosomal:.1e}{', lambda ' + format(r.gwas_lambda, '.2f') if r.gwas_min_p_autosomal < 5e-8/547 else ''}{'; overall min p ' + format(r.gwas_min_p, '.1e') + ' on unplaced/sex-chromosome probes' if r.gwas_min_p < r.gwas_min_p_autosomal else ''})"))
    if pd.notna(r.seaad_J):
        s.append("B " + ("yes" if B[k] else "no") + f" (r {r.seaad_pearson:+.2f}, rho {r.seaad_spearman:+.2f}, null-A p {r.nullA_p_J:.3g}, partner top share {r.seaad_partner_top_share:.2f})")
    else:
        s.append("B n/a (no matched RNA genes)")
    s.append("C " + ("yes" if C[k] else ("no" if r.n_modalities >= 2 and r.top_share < 0.5 else "n/a (single layer or concentrated)")))
    return "; ".join(s)
NM = pd.DataFrame({"role": ["retained" if k in FOCUS else "named (C545)" if k in OTHER_NAMED else "leaves main argument" if k in OUTGOING
                            else "old prioritised, never discussed" for k in NAMED],
                   "prioritised": [bool(M.loc[k, "prioritised"]) for k in NAMED],
                   "reason": [why(k) for k in NAMED]}, index=pd.Index(NAMED, name="component"))
NM.to_csv(f"{OUT}/05_named_components.csv")
print("\n=== Named components")
for k, r in NM.iterrows():
    print(f"C{k} [{r.role}] prioritised={r.prioritised}: {r.reason}")

# ---------------------------------------------------------------- robustness
rows = []
base = set(P.index)
def record(label, pop, sup):
    s = set(M.index[pop & sup])
    rows.append(dict(variant=label, n_population_level=int(pop.sum()), n_prioritised=len(s),
                     focus_captured=",".join(f"C{k}" for k in FOCUS if k in s), focus_missing=",".join(f"C{k}" for k in FOCUS if k not in s),
                     gained_vs_primary=",".join(f"C{k}" for k in sorted(s - base)), lost_vs_primary=",".join(f"C{k}" for k in sorted(base - s))))
sup = M.supported
for t in [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30, 0.50]:
    record(f"top-donor share < {t}", M.top_share < t, sup)
for pr in [10, 20, 30, 50]:
    record(f"PR >= {pr} (instead of top share)", M.PR >= pr, sup)
record("top share < 0.10 AND PR >= 30", (M.top_share < .1) & (M.PR >= 30), sup)
pop = M.population_level
record("A1 with QC proxies (PO2) instead of PO1", pop, supportA1("adjqc") | A2 | B | C)
record("A1 unadjusted (Data S1 calls)", pop, supportA1("unadj") | A2 | B | C)
record("A2 at 5e-8", pop, A1 | supportA2(5e-8) | B | C)
record("A2 at 1e-15 (paper screen)", pop, A1 | supportA2(1e-15) | B | C)
record("A2 counting all SNPs (incl. unplaced/X/Y)", pop, A1 | supportA2(col="gwas_min_p") | B | C)
record("B at |r|,|rho| >= 0.3", pop, A1 | A2 | supportB(r=0.3) | C)
record("B at |r|,|rho| >= 0.5", pop, A1 | A2 | supportB(r=0.5) | C)
record("B null p < 0.05", pop, A1 | A2 | supportB(p=0.05) | C)
record("B without partner-concentration condition", pop, A1 | A2 | supportB(partner_conc=1.01) | C)
record("B Pearson only (|r| > 0.4, old rule)", pop, A1 | A2 | (M.seaad_pearson.abs() > 0.4) | C)
record("C at rho >= 0.5", pop, A1 | A2 | B | supportC(0.5))
record("drop A2 (no genotype support)", pop, A1 | B | C)
record("drop B (no SEA-AD support)", pop, A1 | A2 | C)
record("old rule: >= 3 modalities AND |r| > 0.4", pd.Series(True, index=M.index), (M.n_modalities >= 3) & (M.seaad_pearson.abs() > 0.4))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/05_robustness.csv", index=False)
print("\n=== Robustness"); print(R.to_string(index=False))
M.to_csv(f"{OUT}/05_master_prioritisation.csv")
