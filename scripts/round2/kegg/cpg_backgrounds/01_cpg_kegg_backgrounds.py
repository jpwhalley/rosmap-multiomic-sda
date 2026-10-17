"""Point 5(a): which background did each archived CpG KEGG table (the Data S1 CpG_Site_Pathway source) use?

Independent re-check (own hypergeometric code, no gseapy, no network) of all 181 archived tables in
  $G_ROS_ARCHIVE_DIR/cpg_enrichr/<k>_pathway_encrichment.csv
(made by ros7bi_cpg_gene_pathways.py; version-1 results, read only). For each table k:
  gene list = every RefGene symbol (split on ; or :) of the PIP > 0.5 CpGs of row k of pip_final_meth.csv
  (the SDA file is byte-identical to the snapshot results/pip_final_meth.csv, md5 b08dea62...).
  mode BG  = Enrichr background mode (Speedrichr): query and background upper-cased, background = the 19,872 RefGene
             symbols of the whole 420,132-probe manifest, N = |background|, term genes and query intersected with it.
  mode STD = Enrichr standard mode: N = 20,000, m = full library term size, k = |query| (upper-cased).
The P-value column of the archive is compared with both modes over the archived terms (max relative difference).
Also: header (an `Overlap` column is written by standard mode only), file date, the Data S1 cell for that
component, and the ros7bi rule that skips a component whose table already exists.
Writes $G_ROS_RESULTS_DIR/v23_review/point_5_analysis/out/01_cpg_kegg_backgrounds.csv (per-table: k, mode match, max rel diffs, header type, date; no donor data).
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, glob, re, datetime, json
import numpy as np, pandas as pd
from scipy.stats import hypergeom

ARCH = f"{cfg.ARCHIVE}/cpg_enrichr"
SDA = cfg.SDA
MAN = f"{cfg.REFS}/ROSMAP_arrayMethylation_metaData.tsv"
GMT = cfg.KEGG_GMT
DS1 = cfg.DS1_V23
OUT = cfg.out_dir(cfg.WORK, "v23_review", "point_5_analysis", "out")

gmt = {l.split("\t")[0]: [g for g in l.rstrip("\n").split("\t")[2:] if g] for l in open(GMT)}
man = pd.read_csv(MAN, sep="\t", index_col=0, usecols=["TargetID", "RefGene"])
bg = sorted({g.upper() for g in man.RefGene.dropna().astype(str).str.split(r"[:;]").explode() if g})
bgset = set(bg)
print(f"manifest probes {len(man)}; RefGene background symbols (upper-cased) {len(bg)}")

files = sorted(glob.glob(f"{ARCH}/*_pathway_encrichment.csv"), key=lambda p: int(os.path.basename(p).split("_")[0]))
ks = [int(os.path.basename(p).split("_")[0]) for p in files]
print(f"archived CpG tables: {len(files)}")

cpgs = open(f"{SDA}/md_cpg.txt").read().splitlines()
PIP = {}
with open(f"{SDA}/pip_final_meth.csv") as f:
    hdr = f.readline().rstrip("\n").split(",")[1:]
    assert hdr == cpgs
    want = {str(k) for k in ks}
    for line in f:
        key, rest = line.split(",", 1)
        if key in want:
            PIP[int(key)] = np.array(rest.rstrip("\n").split(","), dtype=float)
assert set(PIP) == set(ks)
refg = man.RefGene.reindex(cpgs)


def glist(k):
    sel = [c for c, p in zip(cpgs, PIP[k]) if p > 0.5]
    rg = refg.reindex(sel).dropna().astype(str)
    return sorted({g for g in rg.str.split(r"[:;]").explode() if g}), len(sel)


def p_bg(genes):
    q = {g.upper() for g in genes} & bgset; N = len(bgset); out = {}
    for t, tg in gmt.items():
        m = set(tg) & bgset; x = len(m & q)
        if x: out[t] = hypergeom.sf(x - 1, N, len(m), len(q))
    return out


def p_std(genes):
    q = {g.upper() for g in genes}; out = {}
    for t, tg in gmt.items():
        x = len(set(tg) & q)
        if x: out[t] = hypergeom.sf(x - 1, 20000, len(set(tg)), len(q))
    return out


def reldiff(arch, rec):
    common = [t for t in arch.index if t in rec]
    if len(common) == 0: return np.nan, 0
    a = arch.loc[common].values.astype(float); r = np.array([rec[t] for t in common])
    return float(np.max(np.abs(a - r) / np.maximum(r, 1e-300))), len(common)


ds1 = pd.read_csv(DS1)
rows = []
for k, fp in zip(ks, files):
    a = pd.read_csv(fp, index_col=0)
    has_overlap = "Overlap" in a.columns
    date = datetime.datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d")
    gl, ncpg = glist(k)
    arch = a.set_index("Term")["P-value"]
    rb, nb = reldiff(arch, p_bg(gl)); rs, ns = reldiff(arch, p_std(gl))
    mode = "background" if rb < 1e-3 and not rs < 1e-3 else ("standard" if rs < 1e-3 and not rb < 1e-3 else "ambiguous/none")
    top = a.sort_values("-log10(FDR)", ascending=False).iloc[0]
    rows.append(dict(table_k=k, n_cpgs=ncpg, n_genes=len(gl), n_terms=len(a), overlap_column=has_overlap, file_date=date,
                     maxrel_P_background19872=rb, maxrel_P_standard20000=rs, mode_reproduced=mode,
                     top_term=top.Term, top_adjP=float(top["Adjusted P-value"])))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/01_cpg_kegg_backgrounds.csv", index=False)
print(R.groupby(["mode_reproduced", "overlap_column", "file_date"]).size().to_string())
print("background-mode tables: max rel diff P (worst table)", R.loc[R.mode_reproduced == "background", "maxrel_P_background19872"].max())
odd = R[R.mode_reproduced != "background"]
print("non-background tables:\n", odd[["table_k", "n_cpgs", "n_genes", "n_terms", "overlap_column", "file_date",
                                      "maxrel_P_background19872", "maxrel_P_standard20000", "top_term", "top_adjP"]].to_string())
# Data S1: C74 row (C74 keeps its ID in the 547 catalogue) vs table 74's top term
r74 = ds1.loc[ds1.Component == 74].iloc[0]
print("Data S1 C74 CpG_Site_Pathway:", r74.CpG_Site_Pathway, "| table 74 top term:", R.loc[R.table_k == 74, "top_term"].item(),
      "adjP", R.loc[R.table_k == 74, "top_adjP"].item())
print("Data S1 CpG_Site_Pathway filled cells:", int(ds1.CpG_Site_Pathway.notna().sum()), "of", len(ds1))
# what the C74 call gives under the background ros7bi passed (the stated Methods background), for the record
gl74, _ = glist(74); pb = p_bg(gl74)
t = pd.Series(pb).sort_values(); from statsmodels.stats.multitest import multipletests
q = pd.Series(multipletests(t.values, method="fdr_bh")[1], index=t.index).sort_values()
print(f"C74 under the stated 19,872-gene background: {int((q < 0.05).sum())} KEGG terms at q < 0.05; top: {q.index[0]} q {q.iloc[0]:.3g};"
      f" 'Adrenergic signaling in cardiomyocytes' q {q.get('Adrenergic signaling in cardiomyocytes', np.nan):.3g};"
      f" 'Signaling pathways regulating pluripotency of stem cells' q {q.get('Signaling pathways regulating pluripotency of stem cells', np.nan):.3g}")
json.dump({"n_tables": len(R), "by_mode": R.mode_reproduced.value_counts().to_dict(),
           "c74_bg19872_terms_q05": int((q < 0.05).sum()), "c74_bg19872_top": q.index[0], "c74_bg19872_top_q": float(q.iloc[0])},
          open(f"{OUT}/01_summary.json", "w"), indent=1)
