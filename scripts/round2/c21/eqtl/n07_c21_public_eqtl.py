"""(4) Is a cis-eQTL for the OR2L/OR2AK2 genes at rs6664332 already reported? Summarise the read-only public look-ups
saved under $G_ROS_RESULTS_DIR/revision_analyses/neuronal_c21/net/ (raw responses kept as fetched; see public_queries.txt) into
out/n07_*.csv. Downloaded files are read as data only, never imported or executed.
Sources (queried 2026-10-13):
  GTEx portal API v2 (gtexportal.org/api/v2): variant lookup; significant single-tissue eQTLs for the variant and for
    OR2L3/OR2L13/OR2L2/OR2AK2 (gtex_v8 and gtex_v10); conditionally independent eQTLs; LD (r2) in the gene windows;
    on-the-fly eQTL (dyneqtl) for frontal cortex / cortex.
  eQTL Catalogue: REST API retired (HTTP 410, 'This API is no longer available'); fine-mapped credible sets queried
    through the ELIXIR-Estonia eQTL Catalogue browser backend (elixir.ut.ee/eqtl/api/subset?field=rsid&q=rs6664332).
  MetaBrain (de Klein et al. 2023): cis-eQTL browser backend (eqtlgen.molgeniscloud.org/api/v2/metabrain_cis_eqtl_v2).
  Bryois et al. 2022 (Nat Neurosci) brain cell-type cis-eQTL summary statistics, Zenodo 7276971: chr1 files streamed,
    only rows for the OR2L/OR2M cluster genes kept (n01_bryois_stream.sh).
  Local LD: PLINK 1.9 --r2 around rs6664332 in the ROSMAP array genotypes (n = 1,708), cache/ld_rs6664332_rosmap.ld (n07a_ld_rosmap.sh)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import json, os, glob, numpy as np, pandas as pd

ROOT = f"{cfg.WORK}/revision_analyses/neuronal_c21"
NET = f"{ROOT}/net"; OUT = cfg.out_dir(ROOT, "out")
V = "chr1_247965627_G_A_b38"; RS = "rs6664332"
GENES = ["OR2L3", "OR2L13", "OR2L2", "OR2AK2"]
lines = []
def say(s=""):
    print(s); lines.append(s)

v = json.load(open(f"{NET}/gtex_variant_rs6664332.json"))["data"][0]
say(f"rs6664332: GRCh38 {v['variantId']} | GRCh37 {v['b37VariantId']} | ref {v['ref']} alt {v['alt']} (alt = A = the ROSMAP A1 allele)")
gg = {r["geneSymbol"]: r for r in json.load(open(f"{NET}/gtex_genes.json"))["data"]}
for g in ["OR2L8", "OR2AK2", "OR2L5", "OR2L2", "OR2L3", "OR2L13", "OR2M4"]:
    r = gg[g]; d = 0 if r["start"] <= v["pos"] <= r["end"] else min(abs(v["pos"] - r["start"]), abs(v["pos"] - r["end"]))
    say(f"  {g:7s} GRCh38 {r['start']}-{r['end']} ({r['gencodeId']}, GENCODE v26): distance {d/1000:.1f} kb" + (" (SNP inside the gene)" if d == 0 else ""))

# ---- GTEx
ld = {}
for g in ["OR2L3", "OR2L13", "OR2L2"]:
    for pair, r2 in json.load(open(f"{NET}/gtex_ld_{g}_v8.json"))["data"]:
        a, b = pair.split(",")
        if V in (a, b):
            o = b if a == V else a; ld[o] = max(ld.get(o, 0), r2)
rows = []
for g in GENES:
    for ds in ("gtex_v8", "gtex_v10"):
        d = json.load(open(f"{NET}/gtex_sig_{g}_{ds}.json"))["data"]
        for t in sorted(set(r["tissueSiteDetailId"] for r in d)):
            rr = sorted([r for r in d if r["tissueSiteDetailId"] == t], key=lambda r: r["pValue"])
            ours = [i for i, r in enumerate(rr) if r["variantId"] == V]; lead = rr[0]
            rows.append(dict(source="GTEx", dataset=ds, gene=g, tissue=t, n_sig_variants=len(rr), lead_variant=lead.get("snpId"), lead_p=lead["pValue"],
                             lead_r2_with_rs6664332=1.0 if lead["variantId"] == V else ld.get(lead["variantId"], np.nan),
                             rs6664332_significant=bool(ours), rs6664332_rank=(ours[0] + 1) if ours else np.nan,
                             rs6664332_p=rr[ours[0]]["pValue"] if ours else np.nan, rs6664332_NES_alt_A=rr[ours[0]]["nes"] if ours else np.nan))
G = pd.DataFrame(rows); G.to_csv(f"{OUT}/n07_gtex_by_gene_tissue.csv", index=False, float_format="%.4g")
say("\nGTEx: tissues in which rs6664332 is a significant cis-eQTL (gene-level FDR), by gene and release")
for g in GENES:
    for ds in ("gtex_v8", "gtex_v10"):
        t = G[(G.gene == g) & (G.dataset == ds)]
        sig = t[t.rs6664332_significant]; br = sig[sig.tissue.str.startswith("Brain")]
        lead1 = sig[sig.rs6664332_rank == 1]
        say(f"  {g:6s} {ds:8s}: eGene in {len(t)} tissues; rs6664332 significant in {len(sig)} ({len(br)} brain), top variant in {len(lead1)}; "
            f"NES(A) {sig.rs6664332_NES_alt_A.min():+.2f} to {sig.rs6664332_NES_alt_A.max():+.2f}" if len(sig) else f"  {g:6s} {ds:8s}: eGene in {len(t)} tissues; rs6664332 not significant")
        if len(sig):
            say("      brain: " + "; ".join(f"{r.tissue.replace('Brain_', '')} p {r.rs6664332_p:.1e}{' (lead)' if r.rs6664332_rank == 1 else f' (lead {r.lead_variant}, r2 {r.lead_r2_with_rs6664332:.2f})'}" for r in br.itertuples()))
for f in sorted(glob.glob(f"{NET}/gtex_dyn/*.json")):
    d = json.load(open(f)); nm = os.path.basename(f)[:-5]
    say(f"  GTEx v8 on-the-fly {nm}: " + (f"NES(A) {d['nes']:+.2f}, p {d['pValue']:.1e}, genotypes {d['homoRefCount']}/{d['hetCount']}/{d['homoAltCount']}" if "nes" in d else d.get("detail", "")))
ldr = pd.read_csv(f"{ROOT}/cache/ld_rs6664332_rosmap.ld", sep=r"\s+")
k4478 = [x for x in ld if x.startswith("chr1_247965938_")]   # rs4478844: hg18 246195863 -> GRCh38 247965938 (same offset as rs6664332)
say(f"  LD check: GTEx r2(rs6664332, rs4478844) {ld[k4478[0]] if k4478 else 'n/a'}; ROSMAP array (n = 1,708) r2 "
    f"{float(ldr.loc[ldr.SNP_B == 'rs4478844', 'R2'].iloc[0]):.2f}; GTEx gene-window LD partners with r2 >= 0.8: {sum(x >= 0.8 for x in ld.values())}")

# ---- eQTL Catalogue credible sets
e = json.load(open(f"{NET}/elixir/subset_rs6664332.json"))
E = pd.DataFrame(e["page"]); E.to_csv(f"{OUT}/n07_eqtlcatalogue_credible_sets.csv", index=False, float_format="%.4g")
say(f"\neQTL Catalogue fine-mapped (SuSiE) credible sets containing rs6664332: {e['n']} (all quantification types)")
ge = E[E.quantification_method == "ge"].sort_values("pvalue")
for r in ge.itertuples():
    say(f"  {r.gene_name:16s} {r.dataset:12s} {r.dataset_label:34s} credible set size {r.credible_set_size:3d}, PIP {r.pip:.2f}, p {r.pvalue:.1e}, beta(A) {r.beta:+.2f}")

# ---- MetaBrain
rows = []
for q in GENES + [RS]:
    for r in json.load(open(f"{NET}/metabrain/mb_{q}.json")).get("items", []):
        rows.append(dict(query=q, region=r["Brainregion"], gene=r["GeneSymbol"], snp=r["SNP"].split(":")[2], effect_allele=r["SNPEffectAllele"], rank=r["QTLRank"],
                         meta_p=float(r["MetaP"]), meta_beta=float(r["MetaBeta"]), n=r.get("MetaPN", r.get("MetaN")), n_datasets=r["NrDatasets"]))
M = pd.DataFrame(rows).drop_duplicates(["region", "gene", "snp", "rank"]); M.to_csv(f"{OUT}/n07_metabrain.csv", index=False, float_format="%.4g")
say("\nMetaBrain cis-eQTL (top/conditional effects as served): rs6664332 rows")
for r in M[M.snp == RS].sort_values("meta_p").itertuples():
    say(f"  {r.region:16s} {r.gene:8s} rank {r.rank}, effect allele {r.effect_allele}, beta {r.meta_beta:+.2f}, p {r.meta_p:.1e}, N {r.n}, {r.n_datasets} datasets")
for g in GENES:
    t = M[(M.gene == g) & (M["rank"] == 1)]
    say(f"  {g}: rank-1 variant by region: " + "; ".join(f"{r.region} {r.snp}" for r in t.itertuples()))

# ---- Bryois et al. 2022 cell-type eQTL (chr1 rows for the cluster genes)
bdir = f"{NET}/bryois"; sp = f"{bdir}/snp_pos.chr1_247-249Mb.txt"
if os.path.exists(sp):
    pos = pd.read_csv(sp, sep="\t").set_index("SNP")   # columns SNP, SNP_id_hg38, SNP_id_hg19, effect_allele, other_allele, MAF
    rows = []
    for f in sorted(glob.glob(f"{bdir}/*.chr1.OR.txt")):
        ct = os.path.basename(f).split(".chr1")[0]
        if os.path.getsize(f) == 0:
            rows.append(dict(cell_type=ct, gene="(no OR2L/OR2M-cluster gene tested)", n_tested=0)); continue
        B = pd.read_csv(f, sep=r"\s+", header=None, names=["gene_id", "snp", "dist", "p", "beta"])
        B["gene"] = B.gene_id.str.split("_").str[0]
        for g, t in B.groupby("gene"):
            t = t.sort_values("p"); ours = t[t.snp == RS]
            rk = int(np.where(t.snp.values == RS)[0][0]) + 1 if len(ours) else np.nan
            ea = pos.loc[RS, "effect_allele"] if RS in pos.index else "?"
            rows.append(dict(cell_type=ct, gene=g, n_tested=len(t), lead_snp=t.snp.iloc[0], lead_p=t.p.iloc[0], rs6664332_p=ours.p.iloc[0] if len(ours) else np.nan,
                             rs6664332_beta=ours.beta.iloc[0] if len(ours) else np.nan, effect_allele=ea, rs6664332_rank=rk))
    Bt = pd.DataFrame(rows); Bt.to_csv(f"{OUT}/n07_bryois2022_celltype.csv", index=False, float_format="%.4g")
    say(f"\nBryois et al. 2022 brain cell-type cis-eQTL (nominal; effect allele of rs6664332 in snp_pos: {pos.loc[RS, 'effect_allele'] if RS in pos.index else '?'}, other {pos.loc[RS, 'other_allele'] if RS in pos.index else '?'})")
    say(Bt.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
open(f"{OUT}/n07_c21_public_eqtl_summary.txt", "w").write("\n".join(lines) + "\n")
