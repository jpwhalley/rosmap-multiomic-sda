"""Gene -> chromosome from the Ensembl GRCh38.100 GTF (parsed here from the .gtf.gz; independent of the .txt table the
first implementation used). Writes cache/gene_chrom_gtf100.csv (gene_name, gene_id, chrom)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import gzip, re, pandas as pd
rows = []
with gzip.open(f"{cfg.SEQ}/Homo_sapiens.GRCh38.100.gtf.gz", "rt") as f:
    for l in f:
        if l.startswith("#"): continue
        p = l.split("\t", 9)
        if p[2] != "gene": continue
        gid = re.search(r'gene_id "([^"]+)"', p[8]).group(1); gn = re.search(r'gene_name "([^"]+)"', p[8])
        rows.append((gn.group(1) if gn else gid, gid, p[0], int(p[3]), int(p[4])))
D = pd.DataFrame(rows, columns=["gene_name", "gene_id", "chrom", "start", "end"])
D.to_csv(f"{cfg.out_dir(cfg.WORK, 'revision_analyses', 'verify_c74_c278', 'cache')}/gene_chrom_gtf100.csv", index=False)
print(len(D), D.chrom.value_counts().head(26).to_dict())
