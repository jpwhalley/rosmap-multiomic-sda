"""R1 Minor 3: robustness of the reported KEGG results to the PIP threshold (0.45, 0.5, 0.55, 0.6, 0.75). OFFLINE.

Methods, as validated on 2026-10-08 (audit kegg547; release_v2 check_kegg_backgrounds.py, PATCH_NOTES §11-12):
  RNA (Figs S2a, S4e, S6d, S6h; Data S1 GeneExpression_Pathway): Enrichr BACKGROUND mode, background = all 25,098
      RNA genes, server case (upper-cased), via kegg_local.enrich (kegg/kegg547/kegg_local.py, md5 ee12e655... before the path edit).
  CpG genes, Fig 4g (C74): STANDARD mode; gene list = every RefGene symbol (split on ; or :) of the PIP > thr CpGs.
  CpG genes, Data S1 CpG_Site_Pathway for C351/C2/C87/C177: BACKGROUND mode against the 19,872 RefGene genes of the
      whole 450K manifest (how 180/181 archived cpg tables were made; PATCH_NOTES §12 residual).
Step 1 validates each emulation at PIP > 0.5 against the archived result tables (term set, P, adjusted P).
Figure labels are those of version 22 (v23: S4e -> Fig 3e, S6d -> Fig 4e, S6h -> Fig 5e; S2a and 4g are not v23
figures). The original analysis also re-tested the version-22 protein enrichment (Fig 2d and the C351 signed lists);
that result is withdrawn with the proteomics results, and its sets are removed here (2026-10-17). The C277 claims now
include Calcium signaling pathway, named in version 23 (follow-up to v2.0.0; five added rows). The other rows are
unchanged.
Library: KEGG_2021_Human.gmt (audit kegg547, md5 87cf90b6...). Network is blocked (kegg_local + here).
Outputs: $G_ROS_RESULTS_DIR/r1_numbers/out/kegg_threshold_long.csv, kegg_threshold_claims.csv, kegg_validation.json
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "kegg547"))
from kegg_local import enrich, ds1_rule, GMT          # noqa: E402  (blocks requests)
import numpy as np, pandas as pd                       # noqa: E402
from scipy.stats import hypergeom                      # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

S = cfg.SDA
ARCH = cfg.ARCHIVE
REFS = cfg.REFS
OUT = cfg.out_dir(cfg.WORK, "r1_numbers", "out")
THR = [0.45, 0.5, 0.55, 0.6, 0.75]

gmt = {l.split("\t")[0]: [g for g in l.rstrip("\n").split("\t")[2:] if g] for l in open(GMT)}


def standard(gl):
    """Enrichr standard (default-background) test as reproduced in check_kegg_backgrounds.py: user genes upper-cased,
    library symbols as published (3 mixed-case symbols never match), N = 20,000, m = full term size."""
    gl = {g.upper() for g in gl}; k = len(gl); rows = []
    for t, genes in gmt.items():
        ov = sorted(set(genes) & gl)
        if not ov:
            continue
        m = len(set(genes))
        rows.append((t, f"{len(ov)}/{m}", len(ov), m, float(hypergeom.sf(len(ov) - 1, 20000, m, k)), ";".join(ov)))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["Term", "Overlap", "n_overlap", "term_size", "P-value", "Genes"])
    df["Adjusted P-value"] = multipletests(df["P-value"], method="fdr_bh")[1]
    return df.sort_values(["P-value", "Term"], kind="mergesort").reset_index(drop=True)


def read_rows(path, ids):
    want = {str(i) for i in ids}; out = {}
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split(",")[1:]
        for line in f:
            key, rest = line.split(",", 1)
            if key in want:
                out[int(key)] = np.array(rest.rstrip("\n").split(","), dtype=float)
                if len(out) == len(want):
                    break
    return hdr, out


# ---------------- feature names, annotations ----------------
genes_txt = [l.rstrip("\n") for l in open(f"{S}/genes.txt")]
cpg_ids = [l.rstrip("\n") for l in open(f"{S}/md_cpg.txt")]
ens = pd.read_csv(f"{REFS}/Homo_sapiens.GRCh38.100.txt", index_col=0, usecols=[0, 11])
ens_names = set(ens["gene_name"].dropna().astype(str))
man = pd.read_csv(f"{REFS}/ROSMAP_arrayMethylation_metaData.tsv", sep="\t", index_col=0, usecols=["TargetID", "RefGene"])
bg_refgene = sorted(set(man["RefGene"].dropna().astype(str).str.split(r"[:;]").explode()))
refgene = man["RefGene"]

COMPS_RNA = [351, 278, 172, 277, 21, 2, 74, 87, 177]
hdr, PIP_R = read_rows(f"{S}/pip_final_RNA.csv", COMPS_RNA)
assert hdr == genes_txt
hdr, PIP_M = read_rows(f"{S}/pip_final_meth.csv", [74, 351, 2, 87, 177]); assert hdr == cpg_ids


def rna_list(k, thr):
    return [g for g, p in zip(genes_txt, PIP_R[k]) if p > thr]


def cpg_list(k, thr):
    sel = [cpg_ids[i] for i, p in enumerate(PIP_M[k]) if p > thr]
    rg = refgene.reindex(sel).dropna().astype(str)
    return sorted(set(rg.str.split(r"[:;]").explode())), len(sel)


# ---------------- step 1: validation at PIP > 0.5 against the archive ----------------
def compare(arch_csv, res):
    a = pd.read_csv(arch_csv, index_col=0)
    out = {"archived_terms": int(len(a)), "recomputed_terms": int(len(res)),
           "same_term_set": set(a.Term) == set(res.Term)}
    if set(a.Term) <= set(res.Term):
        r = res.set_index("Term").loc[a.Term]
        out["max_rel_diff_P"] = float(np.max(np.abs(r["P-value"].values - a["P-value"].values) / a["P-value"].values))
        out["max_rel_diff_adjP"] = float(np.max(np.abs(r["Adjusted P-value"].values - a["Adjusted P-value"].values) / a["Adjusted P-value"].values))
        sig_a = set(a.Term[a["Adjusted P-value"] < 0.05]); sig_r = set(res.Term[res["Adjusted P-value"] < 0.05])
        out["same_significant_set"] = sig_a == sig_r
    return out


val = {}
for k, f in [(351, "enrichr/all_kegg/351"), (278, "enrichr/278"), (172, "enrichr/172"), (277, "enrichr/277")]:
    val[f"RNA C{k}"] = compare(f"{ARCH}/{f}_pathway_encrichment.csv", enrich(rna_list(k, 0.5), genes_txt))
gl, n = cpg_list(74, 0.5); val["CpG C74 (Fig 4g, standard)"] = {"n_cpgs": n, "n_genes": len(gl), **compare(f"{ARCH}/cpg_enrichr/74_pathway_encrichment.csv", standard(gl))}
for k in (351, 2, 87, 177):
    gl, n = cpg_list(k, 0.5)
    p = f"{ARCH}/cpg_enrichr/{k}_pathway_encrichment.csv"
    val[f"CpG C{k} (Data S1, background)"] = {"n_cpgs": n, "n_genes": len(gl), "background": len(bg_refgene),
                                              **(compare(p, enrich(gl, bg_refgene)) if os.path.exists(p) else {"archived": "none"})}
val["background_RefGene_size"] = len(bg_refgene)
json.dump(val, open(os.path.join(OUT, "kegg_validation.json"), "w"), indent=1)
print(json.dumps(val, indent=1), flush=True)

# ---------------- step 2: thresholds ----------------
SETS = [  # (label, layer, mode, component, sign, claimed terms (as reported in figure/caption/text/Data S1))
    ("Fig S2a C351 RNA", "RNA", "background", 351, 0, ["Cytokine-cytokine receptor interaction"]),
    ("Fig S4e C278 RNA", "RNA", "background", 278, 0, ["Ras signaling pathway", "PI3K-Akt signaling pathway", "MAPK signaling pathway"]),
    ("Fig S6d C172 RNA", "RNA", "background", 172, 0, ["MAPK signaling pathway", "Transcriptional misregulation in cancer"]),
    ("Fig S6h C277 RNA", "RNA", "background", 277, 0, ["GABAergic synapse", "Morphine addiction", "Calcium signaling pathway"]),
    ("Data S1 C21 RNA", "RNA", "background", 21, 0, ["Olfactory transduction"]),
    ("Data S1 C2 RNA (blank)", "RNA", "background", 2, 0, []),
    ("Data S1 C74 RNA (blank)", "RNA", "background", 74, 0, []),
    ("Data S1 C87 RNA (blank)", "RNA", "background", 87, 0, []),
    ("Data S1 C177 RNA (blank); text: mitochondrial", "RNA", "background", 177, 0, ["Oxidative phosphorylation"]),
    ("Fig 4g C74 CpG genes", "meth", "standard", 74, 0, ["Signaling pathways regulating pluripotency of stem cells", "Adrenergic signaling in cardiomyocytes"]),
    ("Data S1 C351 CpG genes", "meth", "background", 351, 0, ["Endometrial cancer"]),
    ("Data S1 C2 CpG genes", "meth", "background", 2, 0, ["Pantothenate and CoA biosynthesis"]),
    ("Data S1 C87 CpG genes", "meth", "background", 87, 0, ["RNA polymerase"]),
    ("Data S1 C177 CpG genes (blank)", "meth", "background", 177, 0, []),
]
long_rows, claim_rows = [], []
for label, layer, mode, k, sign, claims in SETS:
    for thr in THR:
        if layer == "RNA":
            gl = rna_list(k, thr); nfeat = len(gl); res = enrich(gl, genes_txt)
        else:
            gl, nfeat = cpg_list(k, thr)
            res = standard(gl) if mode == "standard" else enrich(gl, bg_refgene)
        if len(res):
            res = res.copy()
            res["rank"] = res.sort_values(["Adjusted P-value", "P-value", "Term"], kind="mergesort").index.argsort() + 1
            if "n_overlap" not in res:
                res["n_overlap"] = res["Overlap"].str.split("/").str[0].astype(int)
        top, topadj, tied = ds1_rule(res)
        nsig = int((res["Adjusted P-value"] < 0.05).sum()) if len(res) else 0
        long_rows.append({"set": label, "layer": layer, "mode": mode, "component": k, "pip_gt": thr,
                          "n_features": nfeat, "n_genes_sent": len(gl), "n_terms_FDR<0.05": nsig,
                          "top_term_DataS1_rule": top, "top_adjP": topadj, "n_tied_top": len(tied),
                          "top3_by_adjP": "; ".join(res.sort_values(["Adjusted P-value", "P-value"]).Term[:3]) if len(res) else ""})
        for c in claims:
            if len(res) and c in set(res.Term):
                r = res[res.Term == c].iloc[0]
                claim_rows.append({"set": label, "claim": c, "pip_gt": thr, "rank": int(r["rank"]),
                                   "adjP": float(r["Adjusted P-value"]), "overlap": int(r["n_overlap"]),
                                   "FDR<0.05": bool(r["Adjusted P-value"] < 0.05)})
            else:
                claim_rows.append({"set": label, "claim": c, "pip_gt": thr, "rank": None, "adjP": None, "overlap": 0, "FDR<0.05": False})
L = pd.DataFrame(long_rows); C = pd.DataFrame(claim_rows)
L.to_csv(os.path.join(OUT, "kegg_threshold_long.csv"), index=False)
C.to_csv(os.path.join(OUT, "kegg_threshold_claims.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(L[["set", "pip_gt", "n_features", "n_genes_sent", "n_terms_FDR<0.05", "top_term_DataS1_rule", "top_adjP"]].to_string())
print(C.to_string())
