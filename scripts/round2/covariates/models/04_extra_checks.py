"""Extra checks for the central five (aggregates only):
(a) ordinal/categorical phenotype codings with family-wide BH over 547 (RINT-OLS, M1 and M2 covariates):
    CERAD 1-4 linear trend; cogdx all six categories (n=276, 5-df F); NCI<MCI<AD linear trend (n=254).
(b) modality batch: are the central scores associated with H3K9ac batch, methylation batch/plate or TMT batch
    (within the donors who have that modality), and does adding batch fixed effects change the phenotype test
    in that subset? Batch levels with < 2 donors in the subset are pooled into one 'rare' level.
Outputs: extra_ordinal_family.csv, extra_modality_batch.csv, 04_extra_checks.log."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, warnings, numpy as np, pandas as pd
from scipy import stats as ss
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = cfg.out_dir(cfg.WORK, "covariates")
exec(open(f"{HERE}/_common.py").read())
warnings.filterwarnings("ignore")
CENTRAL = [(74, "Sex"), (278, "Sex"), (545, "Sex"), (172, "ceradsc"), (277, "cogdx")]
base = lambda ph: ["age", "pmi", "ros"] + ([] if ph == "Sex" else ["sex"])
M2c = lambda ph: base(ph) + ["lognuc", "logupn"]

# (a) ordinal / categorical codings, BH across 547 ------------------------------------------------------------------
D3 = D.copy(); D3["cog3"] = D3.cogdx.map({1.0: 0.0, 2.0: 1.0, 4.0: 2.0})
rows = []
for name, term, sub, covph in [("CERAD 1-4 linear", ["ceradsc"], list(don), "ceradsc"),
                               ("cogdx 6 categories", ["C(cogdx6)"], list(don), "cogdx"),
                               ("NCI<MCI<AD trend", ["cog3"], SUB["cogdx"], "cogdx")]:
    for mn, cv in [("M1", base(covph)), ("M2", M2c(covph))]:
        p, _, _ = nested_F(term, cv, sub, Dx=D3); q = pd.Series(bh(p), index=S.index); p = pd.Series(p, index=S.index)
        r = dict(test=name, model=mn, n_sig_547=int((q < 0.05).sum()))
        for k, _ in CENTRAL: r[f"C{k}_p"] = p[k]; r[f"C{k}_q"] = q[k]
        rows.append(r)
od = pd.DataFrame(rows); od.to_csv(f"{OUT}/extra_ordinal_family.csv", index=False)
print("(a) Ordinal / categorical codings, family-wide BH over 547 (RINT-OLS):")
for _, r in od.iterrows():
    print(f"  {r.test:20s} {r.model}: components q<0.05 = {r.n_sig_547:3d} | " + "; ".join(f"C{k} q {r[f'C{k}_q']:.2g}" for k, _ in CENTRAL))

# (b) modality batch ------------------------------------------------------------------------------------------------
bio = pd.read_csv(f"{B}/oth/meta/ROSMAP_biospecimen_metadata.csv")[["specimenID", "individualID"]]
ind2pid = pd.Series(clin.index, index=clin.individualID)
chip = pd.read_csv(f"{B}/oth/chip/QualityControl.csv", sep="\t", dtype={"SampleID": str})
chip["projid"] = chip.SampleID.astype(int).astype(str); chip = chip[chip.projid.isin(don)].drop_duplicates("projid").set_index("projid")
meth = pd.read_csv(f"{B}/oth/meta/ROSMAP_assay_methylationArray_metadata.csv").merge(bio, on="specimenID", how="left")
meth["projid"] = meth.individualID.map(ind2pid); meth = meth[meth.projid.isin(don)].drop_duplicates("projid").set_index("projid")
tmt = pd.read_csv(f"{B}/prot/tmt/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv").merge(bio, on="specimenID", how="left")
tmt["projid"] = tmt.individualID.map(ind2pid); tmt = tmt[tmt.projid.isin(don)].drop_duplicates("projid").set_index("projid")
BAT = {"H3K9ac batch": chip.Batch.astype(str), "methylation batch": meth["batch"].astype(str),
       "methylation plate": meth.Sample_Plate.astype(str), "TMT batch": tmt["batch"].astype(str)}
rows = []
print("\n(b) Modality batch (within donors with that modality):")
for bn, bat in BAT.items():
    vc = bat.value_counts(); bat2 = bat.where(bat.map(vc) >= 2, "rare")
    print(f"  {bn}: {len(bat)} donors, {bat.nunique()} levels ({int((vc >= 2).sum())} with >=2 donors)")
    # does batch differ by phenotype? (aggregate)
    for k, ph in CENTRAL:
        idx = [x for x in bat2.index if x in set(SUB[ph])]
        sc = S.loc[k, idx]; g = [sc[bat2.loc[idx] == lv].values for lv in bat2.loc[idx].unique()]
        g = [x for x in g if len(x) >= 2]
        p_kw = ss.kruskal(*g).pvalue if len(g) > 1 else np.nan
        # phenotype test in the subset: RINT recomputed in the subset; M1 covariates; +/- batch dummies
        Dx = D.loc[idx].copy()
        levs = sorted(bat2.loc[idx].unique())[1:]
        bcols = []
        for j, lv in enumerate(levs):
            Dx[f"b{j}"] = (bat2.loc[idx] == lv).astype(float).values; bcols.append(f"b{j}")
        y = pd.DataFrame({k: rint(sc.values)}, index=idx)
        p0, _, _ = nested_F(PHTERM[ph], base(ph), idx, Yd=y, Dx=Dx)
        p1, _, dff = nested_F(PHTERM[ph], base(ph) + bcols, idx, Yd=y, Dx=Dx)
        rows.append(dict(modality_batch=bn, component=k, phenotype=ph, n=len(idx), n_levels=len(levs) + 1, score_vs_batch_KW_p=p_kw,
                         pheno_p_subset_M1=p0[0], pheno_p_subset_M1_plus_batch=p1[0], resid_df=dff))
        print(f"    C{k} {ph}: n {len(idx)}; score ~ batch KW p {p_kw:.2g}; phenotype p in subset (M1) {p0[0]:.2g} -> + batch {p1[0]:.2g} (resid df {dff})")
pd.DataFrame(rows).to_csv(f"{OUT}/extra_modality_batch.csv", index=False)
