"""(3) Batch/technical associations of the nine component scores, built independently from the metadata.
ChIP: QualityControl.csv via ReadCounts RG_ID -> bam_id_mapping -> individualID -> projid.
Methylation: assay metadata (Sample_Plate, Sentrix_ID, batch) via specimenID -> individualID.
TMT: batch of the column actually fed to SDA for each donor (ros1d mapping) and of the donor's own column.
snRNA: per-donor x cell-type library size inferred from CPM granularity (1e6 / smallest nonzero CPM).
Clinical: Study, msex, age_death, pmi. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
H = os.path.dirname(os.path.abspath(__file__)); C = cfg.out_dir(cfg.WORK, "verify_source_panels", "cache"); R = cfg.out_dir(cfg.WORK, "verify_source_panels", "results")
O = cfg.ROS
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
COMPS = [351, 2, 74, 87, 278, 177, 21, 172, 277]
A = np.load(f"{C}/A.npy"); samples = np.load(f"{C}/samples.npy").astype(int)
clin = pd.read_csv(f"{O}/oth/meta/ROSMAP_clinical.csv", index_col=0)
cov = pd.DataFrame(index=samples)
cov["Study"] = clin.loc[samples, "Study"].values
cov["msex"] = clin.loc[samples, "msex"].values
cov["age_death"] = pd.to_numeric(clin.loc[samples, "age_death"].astype(str).str.replace("+", "", regex=False), errors="coerce").values
cov["pmi"] = clin.loc[samples, "pmi"].values
ind = clin.loc[samples, "individualID"]
# ChIP
qc = pd.read_csv(f"{O}/oth/chip/QualityControl.csv", sep="\t", dtype={"SampleID": str}).set_index("SampleID")
bm = pd.read_csv(f"{O}/oth/chip/bam_id_mapping.csv").drop_duplicates("RG_ID", keep="last")
rg2ind = {str(r).lstrip("0"): b for r, b in zip(bm.RG_ID, bm.BAM_File)}
qc["individualID"] = [rg2ind.get(str(s).lstrip("0")) for s in qc.index]
qci = qc.dropna(subset=["individualID"]).drop_duplicates("individualID").set_index("individualID")
chip = np.load(f"{C}/chip_input.npy"); has_chip = ~np.isnan(chip).all(1)
for col in ["Batch", "UniquelyMappedUniqueReads", "NonRedundantFraction", "GCBias", "CrossCorrelation", "FractionOfReadsInPeaks", "CorrelationWithMedianSample"]:
    cov["chip_" + col] = [qci.loc[i, col] if (i in qci.index and h) else np.nan for i, h in zip(ind.values, has_chip)]
print("ChIP donors with QC row:", int(cov.chip_Batch.notna().sum()), "/", int(has_chip.sum()))
# methylation
mm = pd.read_csv(f"{O}/oth/meta/ROSMAP_assay_methylationArray_metadata.csv")
bio = pd.read_csv(f"{O}/oth/meta/ROSMAP_biospecimen_metadata.csv", low_memory=False)
mm = mm.merge(bio[["specimenID", "individualID"]].drop_duplicates(), on="specimenID", how="left").drop_duplicates("individualID").set_index("individualID")
meth = np.load(f"{C}/meth_input.npy"); has_meth = ~np.isnan(meth).all(1)
for col in ["Sample_Plate", "Sentrix_ID", "batch"]:
    cov["meth_" + col] = [mm.loc[i, col] if (i in mm.index and h) else np.nan for i, h in zip(ind.values, has_meth)]
print("meth donors with assay row:", int(cov.meth_Sample_Plate.notna().sum()), "/", int(has_meth.sum()))
# TMT batch of the column fed to SDA (ros1d) -- rebuild mapping
T = f"{O}/prot/tmt"
meta = pd.read_csv(f"{T}/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv", index_col=0)
cols = pd.read_csv(f"{T}/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv", index_col=0, nrows=1).columns
def mapping(md):
    rid = md.index.str.split(".").str[-1]; sid = md.index.str.split(".").str[-3] + "." + md.index.str.split(".").str[-2]
    s = pd.Series(rid, index=sid); return s.loc[cols.intersection(s.index)].to_dict()
m1 = mapping(meta); m2 = mapping(meta[meta.index.str.startswith("ROSMAP.DLPFC")])
inv1 = {}
for c_, i_ in m1.items(): inv1[i_] = c_      # last wins, as rename+selection in ros1d
prot = np.load(f"{C}/proteins_input.npy"); has_prot = ~np.isnan(prot).all(1)
cov["tmt_batch_fed"] = [inv1[i].split(".")[0] if (h and i in inv1) else np.nan for i, h in zip(ind.values, has_prot)]
# snRNA library size from CPM granularity
rna = np.load(f"{C}/RNA_input.npy")
for c, ct in enumerate(CTS):
    blk = rna[c * 276:(c + 1) * 276]
    mn = np.where(blk > 0, blk, np.inf).min(1)
    cov[f"rna_libsize_{ct}"] = 1e6 / mn
cov["rna_libsize_total6"] = cov[[f"rna_libsize_{ct}" for ct in CTS]].sum(1)
# compare with the first implementation's covariates (consistency, not reused)
fa = pd.read_csv(f"{cfg.WORK}/source_panels/cache/donor_covariates.csv", index_col=0)
for ct in CTS:
    x = fa.loc[samples, f"rna_libsize_{ct}"].values; y = cov[f"rna_libsize_{ct}"].values
    print(f"libsize {ct}: max rel diff vs first implementation {np.nanmax(np.abs(x - y) / y):.2e}")
print("ChIP batch agrees with first implementation:", bool((fa.loc[samples, "chip_Batch"].fillna(-1).values == cov.chip_Batch.fillna(-1).values).all()))
cov.to_csv(f"{C}/covariates_verify.csv")

def rz(x):
    med = np.median(x); return (x - med) / (np.median(np.abs(x - med)) * 1.4826)
cat = ["Study", "msex", "chip_Batch", "meth_Sample_Plate", "meth_Sentrix_ID", "meth_batch", "tmt_batch_fed"]
cont = [c for c in cov.columns if c not in cat]
rows = []
for k in COMPS:
    a = A[k - 1]; e = int(np.argmax(np.abs(rz(a))))
    for drop in [False, True]:
        for v in cat + cont:
            x = cov[v].values; ok = pd.notna(x)
            if drop: ok = ok & (np.arange(276) != e)
            if v in cat:
                g = pd.Series(a[ok]).groupby(pd.Series(x[ok]).astype(str).values)
                groups = [s.values for _, s in g if len(s) >= 2]
                if len(groups) < 2: continue
                st, p = stats.kruskal(*groups); eff_ = np.nan; nlev = len(groups)
            else:
                st, p = stats.spearmanr(x[ok].astype(float), a[ok]); eff_ = st; nlev = np.nan
            rows.append(dict(component=k, drop_extreme=drop, variable=v, n=int(ok.sum()), levels=nlev, rho=eff_, p=p))
D = pd.DataFrame(rows)
D["q_within_comp"] = D.groupby(["component", "drop_extreme"]).p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
D.to_csv(f"{R}/08_batch_associations.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(D[(D.q_within_comp < 0.1) | ((D.p < 0.05) & D.variable.isin(cat))].round(4).to_string(index=False))
# batch-mates of the extreme donors
bm_rows = []
for k in [351, 2, 87, 177]:
    a = A[k - 1]; z = rz(a); e = int(np.argmax(np.abs(z)))
    for v in ["chip_Batch", "meth_Sample_Plate", "meth_Sentrix_ID", "tmt_batch_fed", "Study"]:
        b = cov[v].values[e]
        if pd.isna(b): bm_rows.append(dict(component=k, variable=v, n_mates=0)); continue
        mates = np.where((cov[v].astype(str).values == str(b)) & (np.arange(276) != e))[0]
        bm_rows.append(dict(component=k, variable=v, n_mates=len(mates), median_rz_mates=float(np.median(z[mates])) if len(mates) else np.nan,
                            max_abs_rz_mates=float(np.max(np.abs(z[mates]))) if len(mates) else np.nan))
    for v in ["rna_libsize_total6", "rna_libsize_Mic", "rna_libsize_Ast", "chip_UniquelyMappedUniqueReads", "chip_FractionOfReadsInPeaks", "chip_CorrelationWithMedianSample", "pmi", "age_death"]:
        x = cov[v].values.astype(float)
        if np.isnan(x[e]): continue
        bm_rows.append(dict(component=k, variable=v, extreme_donor_percentile=float(stats.percentileofscore(x[~np.isnan(x)], x[e]))))
B = pd.DataFrame(bm_rows); B.to_csv(f"{R}/09_extreme_donor_batchmates.csv", index=False)
print(B.round(3).to_string(index=False))
