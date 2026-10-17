"""Fetch the public Enrichr KEGG_2021_Human library once via gseapy (approved network use)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import hashlib, json, datetime, sys
import gseapy as gp
cfg.out_dir(os.path.dirname(cfg.KEGG_GMT)); OUT = cfg.KEGG_GMT
# max_size raised so gseapy's default 2000-gene size filter cannot drop any term
gs = gp.get_library(name="KEGG_2021_Human", organism="Human", min_size=0, max_size=10**7, save=OUT)
md5 = hashlib.md5(open(OUT, "rb").read()).hexdigest()
genes = set(g for v in gs.values() for g in v)
sizes = sorted(len(v) for v in gs.values())
meta = dict(file=OUT, md5=md5, n_terms=len(gs), n_unique_genes=len(genes),
            min_size=sizes[0], max_size=sizes[-1], gseapy=gp.__version__,
            fetched=datetime.datetime.now().isoformat(timespec="seconds"),
            source="https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human (via gseapy.get_library)")
json.dump(meta, open(OUT.replace(".gmt", ".meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
