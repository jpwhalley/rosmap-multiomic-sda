"""Per-donor clinical/technical covariates aligned to the 276 SDA donors (cache only;
controlled values never printed). Prints aggregate coverage only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, gzip, numpy as np, pandas as pd

ROS = cfg.ROS
OUT = cfg.out_dir(cfg.WORK, "source_panels", "cache")
S = np.load(f"{OUT}/slices.npz", allow_pickle=True)
samples = [int(s) for s in S["samples"]]
T = pd.DataFrame(index=pd.Index(samples, name="projid"))

clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv", dtype={"individualID": str})
clin = clin.set_index("projid")
c = clin.loc[samples]
T["Study"] = c["Study"].values
T["msex"] = c["msex"].values
T["age_death"] = pd.to_numeric(c["age_death"].replace("90+", 90), errors="coerce").values
T["pmi"] = c["pmi"].values
T["individualID"] = c["individualID"].values
ind2proj = dict(zip(T.individualID, T.index))

# ---- H3K9ac ChIP QC (QualityControl.csv SampleID == read-group ID; bam_id_mapping BAM_File == individualID)
qc = pd.read_csv(f"{ROS}/oth/chip/QualityControl.csv", sep="\t", dtype={"SampleID": str})
bm = pd.read_csv(f"{ROS}/oth/chip/bam_id_mapping.csv", dtype=str)
rg2ind = dict(zip(bm.RG_ID.astype(str), bm.BAM_File))
qc["individualID"] = qc.SampleID.map(rg2ind)
qc = qc[qc.individualID.isin(ind2proj)]
print("ChIP QC rows matched to SDA donors:", len(qc), "unique donors", qc.individualID.nunique())
qc = qc.drop_duplicates("individualID").set_index("individualID")
for col in ["Batch", "UniquelyMappedUniqueReads", "NonRedundantFraction", "GCBias", "CrossCorrelation",
            "FragmentSize", "NumberOfPeaks", "MeanFoldEnrichment", "FractionOfReadsInPeaks",
            "CorrelationWithMedianSample"]:
    T["chip_" + col] = T.individualID.map(qc[col]).values

# ---- TMT batch (specimenID ROSMAP.DLPFC.bNN.channel.<individualID>)
tm = pd.read_csv(f"{ROS}/prot/tmt/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv")
tm = tm[tm.isAssayControl.astype(str).str.upper() == "FALSE"].copy()
tm["individualID"] = tm.specimenID.str.split(".").str[-1]
tmd = tm[tm.individualID.isin(ind2proj)]
dup = tmd.individualID.value_counts()
print("TMT rows matched:", len(tmd), "donors", tmd.individualID.nunique(), "donors with >1 TMT row", int((dup > 1).sum()))
T["tmt_batch"] = T.individualID.map(tmd.drop_duplicates("individualID").set_index("individualID")["batch"]).values

# ---- methylation array: specimens actually in the imputed matrix, then array metadata
with gzip.open(f"{ROS}/oth/meth/ROSMAP_arrayMethylation_imputed.tsv.gz", "rt") as f:
    spec = f.readline().rstrip("\n").split("\t")[1:]
bio = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_biospecimen_metadata.csv", dtype=str)
s2i = dict(zip(bio.specimenID, bio.individualID))
mm = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_assay_methylationArray_metadata.csv", dtype=str)
mm = mm[mm.specimenID.isin(spec)].copy()
mm["individualID"] = mm.specimenID.map(s2i)
mm = mm[mm.individualID.isin(ind2proj)]
print("meth specimens in matrix matched to SDA donors:", len(mm), "donors", mm.individualID.nunique())
mm = mm.drop_duplicates("individualID").set_index("individualID")
for col in ["Sample_Plate", "batch", "Sentrix_ID"]:
    T["meth_" + col] = T.individualID.map(mm[col]).values

# ---- snRNA-seq technical proxies (no library/sequencing batch fields are populated in the MIT metadata)
cc = pd.read_csv(f"{cfg.WORK}/revision_r2/donor_celltype_counts_from_h5ad.csv", index_col=0)
for ct in ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]:
    T["rna_ncells_" + ct] = cc.reindex(samples)[ct].values
T["rna_ncells_total6"] = cc.reindex(samples)[["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]].sum(axis=1).values
for ct in ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]:
    raw = pd.read_csv(f"{ROS}/mit/rna/pseudobulk/{ct}_raw_counts.csv", index_col=0)
    T["rna_libsize_" + ct] = raw.sum(axis=1).reindex(samples).values
T = T.drop(columns=["individualID"])
T.to_csv(f"{OUT}/donor_covariates.csv")
print(T.notna().sum().to_string())
