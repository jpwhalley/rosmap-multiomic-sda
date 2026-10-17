"""C21 / Fig S1 checks (aggregates only; per-donor genotypes and expression stay on this host).
1. Genome build of the genotype .bim (sentinel SNPs vs dbSNP-135 hg18; RSPOS = GRCh37) and the true position of rs6664332
   relative to the OR2L/OR2M cluster in GRCh37 and GRCh38 (OR2L3 GRCh37 locus found by exact cDNA match in GRCh37.p13).
2. Fig S1e eQTL reproduced from measured expression (rna.txt RPM blocks, log2(RPM+1)) by rs6664332 genotype:
   3-group OLS F-test per cell type (as 11_eqtls.py), additive per-allele effect with 95% CI, covariate-adjusted
   (age, sex, PMI, cohort, log nuclei), and drop-most-influential donor. Other cluster genes in Exc.
3. C21 score vs rs6664332 dosage: OLS unadjusted / adjusted (sex, age, PMI, cohort, 5 genotype PCs) / rank-INT score.
4. C21 score vs sex, CERAD, cogdx (Data S1 547 family q)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import gzip, re, numpy as np, pandas as pd
from scipy import stats as ss
import statsmodels.formula.api as smf
from dc_common import *
SNP = cfg.SNP
bim = pd.read_csv(f"{SNP}/ROSMAP_arrayGenotype_maf.bim", sep=r"\s+", header=None, names=["chr", "rs", "cm", "pos", "a1", "a2"])
fam = pd.read_csv(f"{SNP}/ROSMAP_arrayGenotype_maf.fam", sep=r"\s+", header=None)
iid = fam[1].astype(str).str[3:]
N = len(fam); nb = (N + 3) // 4
def geno(idx):
    """A1 dosage (bim col 5) for SNP rows idx -> array len(idx) x N"""
    out = np.empty((len(idx), N))
    lut = np.array([2.0, np.nan, 1.0, 0.0])
    with open(f"{SNP}/ROSMAP_arrayGenotype_maf.bed", "rb") as f:
        assert f.read(3) == b"\x6c\x1b\x01"
        for j, i in enumerate(idx):
            f.seek(3 + int(i) * nb); by = np.frombuffer(f.read(nb), dtype=np.uint8)
            codes = np.stack([(by >> s) & 3 for s in (0, 2, 4, 6)], 1).ravel()[:N]
            out[j] = lut[codes]
    return out

print("=== 1. genome build ===")
SENT = {"rs9939609": (52378028, 53820527), "rs1800562": (26201120, 26093141), "rs1801282": (12368125, 12393125),
        "rs12913832": (26039213, 28365618)}   # (hg18, GRCh37) positions
for rs, (h18, h19) in SENT.items():
    p = bim.loc[bim.rs == rs, "pos"]
    if len(p): print(f"  {rs}: bim {int(p.iloc[0])} | hg18 {h18} | GRCh37 {h19} -> {'hg18' if int(p.iloc[0]) == h18 else ('GRCh37' if int(p.iloc[0]) == h19 else '?')}")
r = bim[bim.rs == "rs6664332"].iloc[0]; print(f"  rs6664332 bim chr{r.chr}:{r.pos} A1 {r.a1} A2 {r.a2}")
# dbSNP 135 hg18 line (RSPOS = GRCh37 position)
line = None
with open(f"{cfg.SEQ}/dbsnp_135.hg18.sort.vcf") as f:
    for l in f:
        if "\trs6664332\t" in l: line = l.rstrip("\n").split("\t"); break
info = dict(kv.split("=", 1) for kv in line[7].split(";") if "=" in kv)
h18pos = int(line[1]); g37 = int(info["RSPOS"])
print(f"  dbSNP135 hg18: chr1:{h18pos} REF {line[3]} ALT {line[4]} GMAF {info.get('GMAF')} GENEINFO {info.get('GENEINFO')} RSPOS(GRCh37) {g37}")
# GRCh37 sequence around the locus
fai = pd.read_csv(f"{cfg.SEQ}/GRCh37.p13.genome.fa.fai", sep="\t", header=None, index_col=0)
def g37seq(chrom, start, end):          # 1-based inclusive
    L, off, lb, lbytes = fai.loc[chrom, 1], fai.loc[chrom, 2], fai.loc[chrom, 3], fai.loc[chrom, 4]
    with open(f"{cfg.SEQ}/GRCh37.p13.genome.fa", "rb") as f:
        s0 = off + (start - 1) // lb * lbytes + (start - 1) % lb
        f.seek(s0); raw = f.read((end - start + 1) + (end - start) // lb + 2).replace(b"\n", b"")
    return raw[:end - start + 1].decode().upper()
chrom = "chr1" if "chr1" in fai.index else "1"
print(f"  GRCh37 base at RSPOS {g37}: {g37seq(chrom, g37, g37)} (dbSNP REF {line[3]})")
W0, W1 = 247_700_000, 248_800_000; reg = g37seq(chrom, W0, W1)
# GRCh38 cDNA of cluster genes -> exact 60-mer match in GRCh37 region
cd = {}; cur = None
with gzip.open(f"{cfg.SEQ}/Homo_sapiens.GRCh38.cdna.all.fa.gz", "rt") as f:
    for l in f:
        if l.startswith(">"):
            m = re.search(r"gene_symbol:(\S+)", l); g = m.group(1) if m else None
            cur = g if g in {"OR2L3", "OR2L2", "OR2L5", "OR2L13", "OR2AK2", "OR2M4", "OR2L8"} else None
            if cur and cur not in cd: cd[cur] = []
            elif cur: cd[cur].append("|")
        elif cur: cd[cur].append(l.strip())
gtf38 = {}
with gzip.open(f"{cfg.SEQ}/Homo_sapiens.GRCh38.100.gtf.gz", "rt") as f:
    for l in f:
        if l.startswith("1\t") and "\tgene\t" in l:
            m = re.search(r'gene_name "([^"]+)"', l)
            if m and m.group(1) in cd: t = l.split("\t"); gtf38[m.group(1)] = (int(t[3]), int(t[4]))
offs = {}
for g, parts in cd.items():
    seq = "".join(parts).split("|")[0]
    hits = []
    for k0 in range(0, max(1, len(seq) - 60), 100):
        kmer = seq[k0:k0 + 60]; h = reg.find(kmer)
        if h >= 0: hits.append(W0 + h - k0)       # implied GRCh37 transcript-start coordinate
    if hits:
        g37start = int(np.median(hits)); offs[g] = (gtf38[g][0] - g37start, g37start) if g in gtf38 else None
        print(f"  {g}: GRCh38 gene {gtf38.get(g)}; GRCh37 cDNA-implied start ~{g37start}; offset GRCh38-GRCh37 at gene start ~{offs[g][0]}")
near = min((v for v in offs.values() if v is not None), key=lambda t: abs(t[1] - g37))
off = int(near[0])   # offset of the gene nearest the SNP (assemblies differ by indels along the cluster)
g38 = g37 + off
print(f"  rs6664332: hg18 {h18pos} | GRCh37 {g37} | GRCh38 ~{g38} (offset of nearest gene {off}; median over cluster genes {int(np.median([v[0] for v in offs.values() if v]))})")
for g, (s, e) in sorted(gtf38.items(), key=lambda t: t[1][0]):
    d = 0 if s <= g38 <= e else min(abs(g38 - s), abs(g38 - e))
    print(f"    distance to {g} (GRCh38 {s}-{e}): {d/1e3:.1f} kb {'(inside)' if d == 0 else ''}")
print(f"  pipeline classify_qtl compares bim (hg18) {h18pos} with GRCh38 OR2L3 start {gtf38['OR2L3'][0]}: |d| = {abs(gtf38['OR2L3'][0]-h18pos)/1e6:.2f} Mb -> "
      f"{'trans' if abs(gtf38['OR2L3'][0]-h18pos) > 1e6 else 'cis'} (1 Mb window); in one build |d| = {abs(gtf38['OR2L3'][0]-g38)/1e3:.0f} kb -> cis")

print("\n=== 2. Fig S1e eQTL from measured expression ===")
i_snp = int(bim.index[bim.rs == "rs6664332"][0]); i2 = int(bim.index[bim.rs == "rs4478844"][0])
G = geno([i_snp, i2]); gs = pd.Series(G[0], index=iid.values)
don = samples(); D = design_frame()
shared = [d for d in don if d in set(iid)]
print(f"  donors with genotype and RNA: {len(shared)}; A1 = {r.a1} (dbSNP ALT {line[4]}); A1 freq {np.nanmean(gs.loc[shared])/2:.3f}; genotype counts "
      f"{dict(pd.Series(gs.loc[shared]).value_counts().sort_index())}")
genes = [l.strip() for l in open(f"{SDA}/genes.txt")]
RNA = np.load(f"{CACHE}/input_RNA.npy").reshape(6, 276, -1)
rows = []
FIGP = {"Ast": 8.17e-04, "Exc": 1.12e-17, "Inh": 1.44e-17, "Mic": 7.57e-01, "OPC": 4.97e-14, "Oli": 2.77e-13}
di = [don.index(d) for d in shared]
for gname in ["OR2L3", "OR2L2", "OR2L5", "OR2L13", "OR2M4", "OR2M3", "OR2M2", "OR2L8", "OR2AK2"]:
    if gname not in genes: continue
    gi = genes.index(gname)
    for c, ct in enumerate(CTS):
        if gname != "OR2L3" and ct != "Exc": continue
        y = np.log2(RNA[c][di, gi].astype(float) + 1)
        df = pd.DataFrame({"y": y, "g": gs.loc[shared].values}, index=shared).join(D[["age", "sex", "pmi", "ros", "lognuc"]]).dropna()
        df["gc"] = df.g.map({0.0: "BB", 1.0: "AB", 2.0: "AA"})
        f3 = smf.ols("y ~ C(gc)", df).fit(); pF = f3.f_pvalue
        fa = smf.ols("y ~ g", df).fit(); b, (lo, hi) = fa.params["g"], fa.conf_int().loc["g"]
        fc = smf.ols("y ~ g + age + sex + pmi + ros + lognuc", df).fit()
        infl = fa.get_influence().cooks_distance[0]; j = int(np.argmax(infl))
        fd = smf.ols("y ~ g", df.drop(df.index[j])).fit()
        rec = dict(gene=gname, cell_type=ct, n=len(df), mean_log2RPM=df.y.mean(), frac_zero=float((y == 0).mean()), p_3group_F=pF,
                   fig_S1e_p=FIGP.get(ct) if gname == "OR2L3" else np.nan, beta_per_A=b, lo=lo, hi=hi, R2_additive=fa.rsquared,
                   p_additive=fa.pvalues["g"], p_adjusted=fc.pvalues["g"], beta_adjusted=fc.params["g"], p_drop_most_influential=fd.pvalues["g"])
        rows.append(rec)
E = pd.DataFrame(rows); E.to_csv(f"{OUT}/04_c21_eqtl.csv", index=False, float_format="%.4g")
print(E.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\n=== 3. C21 score vs rs6664332 ===")
S = scores547(); a = S.loc[21]
# genotype PCs from a thinned autosomal SNP set
auto = bim.index[(bim.chr.between(1, 22))].values[::60]
Gm = geno(auto); keep = (np.nanmean(Gm, 1) / 2 > 0.05) & (np.nanmean(Gm, 1) / 2 < 0.95) & (np.isnan(Gm).mean(1) < 0.05)
Gm = Gm[keep]; Gm = np.where(np.isnan(Gm), np.nanmean(Gm, 1, keepdims=True), Gm)
Z = (Gm - Gm.mean(1, keepdims=True)) / Gm.std(1, keepdims=True); U, s_, Vt = np.linalg.svd(Z.T, full_matrices=False)
pcs = pd.DataFrame(U[:, :5] * s_[:5], index=iid.values, columns=[f"PC{i}" for i in range(1, 6)])
df = pd.DataFrame({"a": a.loc[shared].values, "g": gs.loc[shared].values}, index=shared).join(D[["age", "sex", "pmi", "ros"]]).join(pcs).dropna()
df["ar"] = rint(df.a.values)
for lab, fml in [("unadjusted", "a ~ g"), ("adj sex+age+PMI+cohort+5PCs", "a ~ g + sex + age + pmi + ros + PC1 + PC2 + PC3 + PC4 + PC5"),
                 ("rank-INT unadjusted", "ar ~ g"), ("rank-INT adjusted", "ar ~ g + sex + age + pmi + ros + PC1 + PC2 + PC3 + PC4 + PC5")]:
    f_ = smf.ols(fml, df).fit(); lo, hi = f_.conf_int().loc["g"]
    print(f"  {lab}: n {int(f_.nobs)} beta/A-allele {f_.params['g']:.4g} [{lo:.4g}, {hi:.4g}] p {f_.pvalues['g']:.3g}; partial R2 "
          f"{(f_.tvalues['g']**2/(f_.tvalues['g']**2+f_.df_resid)):.3f}")
sd = df.a.std(); f_ = smf.ols("a ~ g", df).fit(); print(f"  per-A-allele shift = {f_.params['g']/sd:.2f} SD of the C21 score")
infl = f_.get_influence().cooks_distance[0]; print(f"  max Cook's D {infl.max():.3f}; drop most influential: p {smf.ols('a ~ g', df.drop(df.index[int(np.argmax(infl))])).fit().pvalues['g']:.3g}")
print(f"  genotype PC check: C21 score vs PC1..5 (Spearman p): " + ", ".join(f"{ss.spearmanr(df.a, df[c])[1]:.2g}" for c in pcs.columns))

print("\n=== 4. C21 phenotype tests (547 family) ===")
ref = pd.read_csv(REF547, index_col=0); print("  q:", ref.loc[21].to_dict())
sx = D.sex; print(f"  C21 male vs female Mann-Whitney p {ss.mannwhitneyu(a[sx == 1], a[sx == 0]).pvalue:.3g}")
