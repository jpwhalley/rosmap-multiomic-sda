"""Chromosome of each RNA feature, joined through the atlas's stable Ensembl gene IDs (snRNA-seq h5ad var/gene_ids)
to the Ensembl release-100 GTF gene records. The deposit sidecar's chrom (a symbol join to the same release) is kept
where present; 669 of the 764 symbols it could not place are filled from their stable ID; the other 95 (IDs absent from
release 100) stay missing. Returns a str Series indexed like RNA_features.csv (NaN = unresolved)."""
import gzip, re
import h5py, pandas as pd


def rna_chrom(cfg, rna):
    with h5py.File(f"{cfg.ROS}/mit/rna/mit_all_doublets_labelled.h5ad", "r") as f:
        sid = pd.Series(f["var/gene_ids"].asstr()[()], index=f["var/index"].asstr()[()])
    gtf = {}; kid = re.compile(r'gene_id "([^"]+)"')
    with gzip.open(f"{cfg.SEQ}/Homo_sapiens.GRCh38.100.gtf.gz", "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            t = line.split("\t", 9)
            if t[2] == "gene": gtf[kid.search(t[8]).group(1)] = t[0]
    by_id = sid.reindex(rna.index).map(gtf)
    side = rna.chrom.where(rna.chrom.notna()).astype(object).map(lambda c: c if pd.isna(c) else str(c).replace("chr", ""))
    both = side.notna() & by_id.notna()
    assert (side[both] == by_id[both]).all(), "sidecar and stable-ID chromosomes disagree"
    return side.fillna(by_id)


AUTOSOMES = [str(i) for i in range(1, 23)]
