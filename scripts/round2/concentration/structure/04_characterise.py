"""Part 1: characterise the 547 corrected components by donor concentration (definition in st_common.py) and relate it to
the number of modalities, run recurrence, SEA-AD concordance (Pearson and Spearman), phenotype calls (Data S1 unadjusted;
adjusted PO1), GWAS, technical covariates and top-donor data-quality flags.

Reference distributions for the threshold: centred top-donor share and PR of n = 276 i.i.d. draws from Normal, Student t(5),
t(3) and Laplace (20,000 simulations each, seed 20261013): what a population-level score, including a heavy-tailed one,
looks like when no single donor is special.
Classes used throughout: distributed (top-donor share < 0.10), few-donor (0.10-0.50), concentrated (>= 0.50).
Writes out/04_*.csv and out/04_characterise.log (via tee)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss
from st_common import *

rng = np.random.default_rng(20261013)
M = pd.read_csv(f"{OUT}/01_master_547.csv", index_col=0)
SE = pd.read_csv(f"{OUT}/02_seaad_547.csv", index_col=0)
M = M.join(SE.drop(columns=["seaad_partner", "seaad_pearson"]))
M = M.join(pd.read_csv(f"{OUT}/01b_gwas_autosomal.csv", index_col=0))
T1, T2 = 0.10, 0.50
M["conc_class"] = np.where(M.top_share < T1, "distributed", np.where(M.top_share < T2, "few-donor", "concentrated"))
ORDER = ["distributed", "few-donor", "concentrated"]
pd.set_option("display.width", 250)

# ---------------------------------------------------------------- reference simulation
n = 276; B = 20000
def sim(draw):
    A = draw((B, n)); D = A - A.mean(1, keepdims=True); D2 = D ** 2; tot = D2.sum(1)
    return D2.max(1) / tot, tot ** 2 / (D2 ** 2).sum(1)
refs = {"Normal": lambda s: rng.standard_normal(s), "t(5)": lambda s: rng.standard_t(5, s),
        "t(3)": lambda s: rng.standard_t(3, s), "Laplace": lambda s: rng.laplace(size=s)}
rr = []
for nm, f in refs.items():
    ts, pr = sim(f)
    rr.append(dict(reference=nm, top_share_q50=np.quantile(ts, .5), top_share_q95=np.quantile(ts, .95), top_share_q99=np.quantile(ts, .99),
                   top_share_q999=np.quantile(ts, .999), frac_top_share_ge_0p10=float(np.mean(ts >= 0.10)),
                   PR_q50=np.quantile(pr, .5), PR_q01=np.quantile(pr, .01)))
REF = pd.DataFrame(rr).set_index("reference"); REF.to_csv(f"{OUT}/04_reference_simulation.csv")
print("=== Reference: centred top-donor share and PR for n = 276 i.i.d. scores (20,000 simulations each)")
print(REF.to_string(float_format=lambda v: f"{v:.4g}"))

# ---------------------------------------------------------------- distribution
print("\n=== Distribution of the centred top-donor share over the 547 components")
bins = [0, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 0.90, 0.95, 1.0]
print(pd.cut(M.top_share, bins).value_counts().sort_index().to_string())
print(f"median {M.top_share.median():.3f}; < 0.10: {int((M.top_share < .1).sum())}; 0.10-0.50: {int(M.top_share.between(.1, .5, inclusive='left').sum())}; "
      f">= 0.50: {int((M.top_share >= .5).sum())}; > 0.80 (the earlier 'single-donor' count): {int((M.top_share > .8).sum())}; > 0.95: {int((M.top_share > .95).sum())}")
lg = np.log10(M.top_share); h, e = np.histogram(lg, bins=np.arange(-2.4, 0.01, 0.1))
print("log10(top share) histogram (bin start: count):", " ".join(f"{e[i]:.1f}:{h[i]}" for i in range(len(h))))
print("PR by class (median [min, max]):", {c: f"{M.PR[M.conc_class == c].median():.1f} [{M.PR[M.conc_class == c].min():.1f}, {M.PR[M.conc_class == c].max():.1f}]" for c in ORDER})
print("top-3 share by class (median):", {c: round(float(M.top3_share[M.conc_class == c].median()), 3) for c in ORDER})
print(f"Spearman(top share, PR) = {ss.spearmanr(M.top_share, M.PR)[0]:.3f}; distributed class PR >= 20 in "
      f"{int((M.PR[M.conc_class == 'distributed'] >= 20).sum())}/{int((M.conc_class == 'distributed').sum())}")
top = np.load(f"{PRIV}/01_top_donor_index.npy")
for nm, sel in [("concentrated (>= 0.5)", (M.top_share >= 0.5).values), ("single-donor (> 0.8)", (M.top_share > 0.8).values)]:
    t = top[sel]
    print(f"{nm}: {int(sel.sum())} components; distinct top donors {len(set(t))}; "
          f"max components sharing one top donor {int(pd.Series(t).value_counts().max())}")

# ---------------------------------------------------------------- relations
def by_class(f):
    return {c: f(M[M.conc_class == c]) for c in ORDER}
rows = []
def add(name, fn): rows.append(dict(measure=name, **by_class(fn)))
add("n components", len)
for k in range(5):
    add(f"n_modalities = {k}", lambda d, k=k: int((d.n_modalities == k).sum()))
add("n_modalities >= 3 (old filter stage 1)", lambda d: int((d.n_modalities >= 3).sum()))
for m in MODS:
    add(f"active in {m} (>=1 PIP>0.5 feature)", lambda d, m=m: int((d[m] > 0).sum()))
for r in (5, 6, 7, 8, 9, 10):
    add(f"recurs in {r}/10 runs", lambda d, r=r: int((d.true_runs == r).sum()))
add("median runs", lambda d: float(d.true_runs.median()))
add("median members", lambda d: float(d.n_members.median()))
add("SEA-AD partner (>1 matched gene)", lambda d: int(d.seaad_J.notna().sum()))
add("median |Pearson|", lambda d: round(float(d.seaad_pearson.abs().median()), 3))
add("median |Spearman| (same partner)", lambda d: round(float(d.seaad_spearman.abs().median()), 3))
add("|Pearson| > 0.4 (old stage 2)", lambda d: int((d.seaad_pearson.abs() > 0.4).sum()))
add("|Pearson| > 0.4 and |Spearman| > 0.4, same sign", lambda d: int((d.seaad_J > 0.4).sum()))
add("|Pearson| > 0.4 but |Spearman| < 0.2", lambda d: int(((d.seaad_pearson.abs() > 0.4) & (d.seaad_spearman.abs() < 0.2)).sum()))
add("median top-3 genes' share of Pearson numerator", lambda d: round(float(d.seaad_top3_gene_share_of_numerator.median()), 3))
add("median SEA-AD partner top-sample share", lambda d: round(float(d.seaad_partner_top_share.median()), 3))
add("SEA-AD partner itself concentrated (top share >= 0.5)", lambda d: int((d.seaad_partner_top_share >= 0.5).sum()))
add("Data S1 calls (unadjusted q<0.05): sex", lambda d: int(d.ds1_sex_q.notna().sum()))
add("Data S1 calls: CERAD", lambda d: int(d.ds1_cerad_q.notna().sum()))
add("Data S1 calls: cogdx", lambda d: int(d.ds1_cogdx_q.notna().sum()))
add("adjusted calls (PO1 q<0.05): sex", lambda d: int((d.adj_sex_q < .05).sum()))
add("adjusted calls: CERAD", lambda d: int((d.adj_cerad_q < .05).sum()))
add("adjusted calls: cogdx", lambda d: int((d.adj_cogdx_q < .05).sum()))
add("components with >=1 adjusted call", lambda d: int((d.n_adj_calls > 0).sum()))
add("adjusted + QC proxies (PO2) >=1 call", lambda d: int(((d.adjqc_sex_q < .05) | (d.adjqc_cerad_q < .05) | (d.adjqc_cogdx_q < .05)).sum()))
add("GWAS min p < 1e-15 (paper screen)", lambda d: int((d.gwas_min_p < 1e-15).sum()))
add("GWAS min p < 5e-8", lambda d: int((d.gwas_min_p < 5e-8).sum()))
add("GWAS min p < 5e-8/547 (study-wide)", lambda d: int((d.gwas_min_p < 5e-8 / 547).sum()))
add("GWAS autosomal min p < 1e-15", lambda d: int((d.gwas_min_p_autosomal < 1e-15).sum()))
add("GWAS autosomal min p < 5e-8/547", lambda d: int((d.gwas_min_p_autosomal < 5e-8 / 547).sum()))
add("median GWAS lambda", lambda d: round(float(d.gwas_lambda.median()), 3))
add("top donor = identity-flagged sample (meaningful only when concentrated)", lambda d: int((d.top_donor_identity_flag.fillna("") != "").sum()) if d.conc_class.iloc[0] == "concentrated" else np.nan)
add("top donor fed another person's proteome (meaningful only when concentrated)", lambda d: int(d.top_donor_misfed_proteome.sum()) if d.conc_class.iloc[0] == "concentrated" else np.nan)
add("protein PIP>0.5 features present", lambda d: int((d.proteins > 0).sum()))
BC = pd.DataFrame(rows).set_index("measure"); BC.to_csv(f"{OUT}/04_by_class.csv")
print("\n=== Relations by concentration class")
print(BC.to_string())

# identity-flagged top donors (labels only)
fl = M[(M.top_donor_identity_flag.fillna("") != "") & (M.conc_class == "concentrated")]
print("\nconcentrated components whose dominant donor is an identity-flagged sample (label from identity_checks):")
print(fl[["top_donor_identity_flag", "top_share", "conc_class", "n_modalities", "gwas_min_p"]].to_string(float_format=lambda v: f"{v:.3g}"))

# ---------------------------------------------------------------- rank correlations across the catalogue
lt = np.log10(M.top_share)
print("\n=== Spearman correlations with log10 top-donor share (547 components; GWAS on 547 incl. local C547/C548)")
for nm, v in [("n_modalities", M.n_modalities), ("true_runs", M.true_runs), ("n_members", M.n_members),
              ("|SEA-AD Pearson|", M.seaad_pearson.abs()), ("|SEA-AD Spearman|", M.seaad_spearman.abs()),
              ("|Pearson| - |Spearman|", M.seaad_pearson.abs() - M.seaad_spearman.abs()),
              ("-log10 GWAS min p", -np.log10(M.gwas_min_p)), ("-log10 GWAS autosomal min p", -np.log10(M.gwas_min_p_autosomal)), ("SEA-AD partner top share", M.seaad_partner_top_share)]:
    ok = v.notna(); r, p = ss.spearmanr(lt[ok], v[ok]); print(f"  {nm:28s} rho {r:+.3f} (p {p:.1e}, n {int(ok.sum())})")

# 'does the dominant donor have data where the component is active' (concentrated)
mask = np.load(f"{R2}/r1_numbers/out/donor_modality_mask.npy")         # 276 x 4 (RNA, meth, chip, prot), md_samples order
act = (M[MODS].values > 0); conc = (M.conc_class == "concentrated").values
slots = act[conc]; dh = mask[top[conc]]
print(f"\nconcentrated components: active modality slots {int(slots.sum())}; top donor has data in that modality in {int((slots & dh).sum())}")
for i, m in enumerate(MODS):
    print(f"  {m}: active {int(slots[:, i].sum())}, top donor has data {int((slots[:, i] & dh[:, i]).sum())}")
dist = (M.conc_class == "distributed").values
print(f"concentrated components with >=2 modalities: {int((M.n_modalities[conc] >= 2).sum())}/{int(conc.sum())}; "
      f"distributed: {int((M.n_modalities[dist] >= 2).sum())}/{int(dist.sum())}")

# ---------------------------------------------------------------- technical covariates (RNA QC proxies, composition, age, PMI)
qc = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str)
don = samples(); qc = qc.loc[don]
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv"); clin.index = clin.projid.astype(int).astype(str); clin = clin.loc[don]
S = scores547()
tot = qc[[f"nuclei_{c}" for c in CTS]].sum(1)
COV = {"log nuclei": np.log(qc.nuclei), "log UMIs/nucleus": np.log(qc.umis_per_nucleus),
       "age": pd.to_numeric(clin.age_death.astype(str).replace("90+", "90")), "PMI": clin.pmi.astype(float)}
for c in CTS: COV[f"prop {c}"] = qc[f"nuclei_{c}"] / tot
TR = pd.DataFrame(index=M.index)
for nm, v in COV.items():
    v = np.asarray(v, float); ok = ~np.isnan(v)
    rp = [ss.spearmanr(S.loc[k].values[ok], v[ok]) for k in M.index]
    TR[f"rho_{nm}"] = [r[0] for r in rp]; TR[f"q_{nm}"] = bh([r[1] for r in rp])
TR["max_abs_rho_composition"] = TR[[f"rho_prop {c}" for c in CTS]].abs().max(1)
TR.to_csv(f"{OUT}/04_technical_rho_547.csv")
print("\n=== Technical / composition associations (Spearman, BH over 547 per covariate; q < 0.05 counts by class)")
tab = {}
for nm in COV:
    tab[nm] = {c: int((TR[f"q_{nm}"][M.conc_class == c] < 0.05).sum()) for c in ORDER}
tab["|rho| >= 0.3 with any composition"] = {c: int((TR.max_abs_rho_composition[M.conc_class == c] >= 0.3).sum()) for c in ORDER}
tab["|rho| >= 0.3 with log nuclei or log UMIs/nucleus"] = {c: int(((TR["rho_log nuclei"].abs() >= 0.3) | (TR["rho_log UMIs/nucleus"].abs() >= 0.3))[M.conc_class == c].sum()) for c in ORDER}
print(pd.DataFrame(tab).T.to_string())
M.join(TR).to_csv(f"{OUT}/04_master_with_seaad_tech.csv")
