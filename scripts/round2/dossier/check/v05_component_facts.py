"""V05: feature-level and source-data facts the dossier quotes, recomputed. Aggregates only (no donor IDs)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd, warnings, json
from scipy import stats as ss
from v00_common import *
warnings.filterwarnings("ignore")
REFS = cfg.REFS
don = donors(); S = scores(); c = clin(); male = (c.msex == 1).values
genes = [l.rstrip("\n") for l in open(f"{SDA}/genes.txt")]
cpgs = [l.rstrip("\n") for l in open(f"{SDA}/md_cpg.txt")]
peaks = [l.rstrip("\n") for l in open(f"{SDA}/md_chip.txt")]
prots = [l.rstrip("\n") for l in open(f"{SDA}/md_prot.txt")]
C = COMPS
def rows(fn, cols):
    df = read_rows(f"{SDA}/{fn}", C); df.index = df.index.astype(int); df = df.loc[C]; assert list(df.columns) == cols; return df
XR, PR = rows("genes_final_RNA.csv", genes), rows("pip_final_RNA.csv", genes)
XM, PM = rows("genes_final_meth.csv", cpgs), rows("pip_final_meth.csv", cpgs)
XC, PC = rows("genes_final_chip.csv", peaks), rows("pip_final_chip.csv", peaks)
XP, PP = rows("genes_final_proteins.csv", prots), rows("pip_final_proteins.csv", prots)
print("PIP>0.5 counts (RNA, meth, chip, prot):")
for k in C:
    print(f"  C{k}: {int((PR.loc[k]>0.5).sum())}, {int((PM.loc[k]>0.5).sum())}, {int((PC.loc[k]>0.5).sum())}, {int((PP.loc[k]>0.5).sum())}")
ens = pd.read_csv(f"{REFS}/Homo_sapiens.GRCh38.100.txt", usecols=["chrom", "gene_name"]).drop_duplicates("gene_name").set_index("gene_name").chrom.astype(str)
chrom = pd.Series(genes, index=genes).map(ens)
print("RNA genes without a GRCh38 chromosome:", int(chrom.isna().sum()), "of", len(genes))

def rank_report(k, glist, by_pip=True):
    x = XR.loc[k]; p = PR.loc[k]; sel = p > 0.5
    order = x[sel].abs().sort_values(ascending=False).index
    out = []
    for g in glist:
        if g not in x.index: out.append(f"{g}: absent"); continue
        rk = (list(order).index(g) + 1) if g in order else None
        out.append(f"{g}: PIP {p[g]:.2f}, loading {x[g]:+.3g}, rank among PIP genes {rk}")
    return out

print("\nTop PIP genes by |loading|:")
for k in C:
    x = XR.loc[k]; p = PR.loc[k]; sel = p > 0.5
    print(f"  C{k}:", ", ".join(x[sel].abs().sort_values(ascending=False).index[:12]))
print("\nC351 markers:", rank_report(351, ["IGKC", "IGHG1", "CD68", "HLA-DRB1"]))
print("C74:", rank_report(74, ["XIST", "UTY"]))
GR = ["FKBP5", "SGK1", "ZBTB16", "KLF9", "PER1", "DDIT4", "KLF15", "PDK4", "FOXO3", "NFKBIA", "DUSP1", "SESN1", "CEBPD", "BCL6", "MT2A", "TSC22D3"]
print("C278 GR genes:"); [print("   ", s) for s in rank_report(278, GR + ["SOX6", "PDGFRA", "PTPRZ1", "BCAS1", "OPALIN", "MOG"])]
print("C87:", rank_report(87, ["VTCN1"]))
print("C177:", rank_report(177, ["NDUFA13", "COX6A1"]))
x = XP.loc[177]; p = PP.loc[177]; print("C177 top protein (PIP>0.5):", list(x[p > 0.5].abs().sort_values(ascending=False).index[:3]))

# X/Y share of squared PIP-gene loading
print("\nX/Y share of squared PIP>0.5 RNA loading:")
for k in C:
    x = XR.loc[k]; sel = PR.loc[k] > 0.5; ch = chrom[sel]
    e = x[sel] ** 2; xy = ch.isin(["X", "Y"])
    print(f"  C{k}: n PIP {int(sel.sum())}, X {int((ch=='X').sum())}, Y {int((ch=='Y').sum())}, share XY {e[xy].sum()/e.sum():.4f}; XIST in PIP set: {bool(sel.get('XIST', False))}")

# C74 peaks: sex-chromosome share (Fig 4e caption 'concentrated on the sex chromosomes')
dom = pd.read_csv(f"{REFS}/H3K9acDomains.csv", sep="\t").set_index("DomainID")
pc = dom.loc[peaks]
for k in (74, 351):
    sel = (PC.loc[k] > 0.5).values; ch = pc.Chr.astype(str).values[sel]; e = XC.loc[k].values[sel] ** 2
    xy = np.isin(ch, ["X", "Y", "23", "24"])
    prom = (pc.Classification.values[sel] == "Promoter")
    print(f"C{k} PIP peaks {sel.sum()}: on X {np.sum(ch=='X')}, Y {np.sum(ch=='Y')}; X/Y share of count {xy.mean():.3f}, of squared loading {e[xy].sum()/e.sum():.3f};"
          f" promoter peaks {prom.sum()} (X/Y promoter {np.sum(prom & xy)}); all-peak X/Y share {np.isin(pc.Chr.astype(str), ['X','Y']).mean():.4f}")
    top = np.argsort(-e)[:20]; print(f"   top-20 |loading| peaks on X/Y: {int(xy[top].sum())}")

# C74: SEA-AD Fig 4c (partner 106 in the deposited 143-component SEA-AD catalogue)
SEA = cfg.SEAAD
fn = pd.read_csv(f"{SEA}/gene_info.csv", index_col=0)["feature_name"]
sg = pd.read_csv(f"{SEA}/sda/genes_final.csv", index_col=0); sg.columns = [fn[q] for q in sg.columns]; sg.index = sg.index.astype(int)
spip = pd.read_csv(f"{SEA}/sda/pip_final.csv", index_col=0); spip.columns = [fn[q] for q in spip.columns]; spip.index = spip.index.astype(int)
common = sorted(set(sg.columns) & set(genes))
def seaad(k, j, auto_only=False):
    gl = [g for g in common if PR.loc[k, g] > 0.5]
    if auto_only: gl = [g for g in gl if chrom.get(g) not in ("X", "Y")]
    a, b = XR.loc[k, gl].astype(float), sg.loc[j, gl].astype(float)
    return len(gl), ss.pearsonr(a, b), ss.spearmanr(a, b)
for k, j in ((74, 106), (351, 21), (2, 21), (278, 67), (21, 69), (172, 64), (277, 136), (545, 123)):
    n, pr, sr = seaad(k, j)
    print(f"C{k} vs SEA-AD {j}: n {n}, Pearson {pr[0]:+.3f} (p {pr[1]:.2g}), Spearman {sr[0]:+.3f} (p {sr[1]:.2g})")
n, pr, sr = seaad(74, 106, auto_only=True)
print(f"C74 vs SEA-AD 106, autosomal genes only: n {n}, Pearson {pr[0]:+.3f} (p {pr[1]:.2g}), Spearman {sr[0]:+.3f} (p {sr[1]:.2g})")
gl = [g for g in common if PR.loc[74, g] > 0.5 and chrom.get(g) not in ("X", "Y")]
a, b = XR.loc[74, gl].astype(float), sg.loc[106, gl].astype(float)
print(f"   autosomal sign agreement (opposite sign = concordant given r<0): {int((np.sign(a)!=np.sign(b)).sum())}/{len(gl)}; binomial p {ss.binomtest(int((np.sign(a)!=np.sign(b)).sum()), len(gl)).pvalue:.2g}")
# S2d / S3d gene sets
g351 = [g for g in common if PR.loc[351, g] > 0.5]; print(f"C351 PIP genes {int((PR.loc[351]>0.5).sum())}; in SEA-AD common {len(g351)}; of these C2 PIP {sum(PR.loc[2, g]>0.5 for g in g351)}; C351∩C2 PIP overall {int(((PR.loc[351]>0.5)&(PR.loc[2]>0.5)).sum())}")
print(f"SEA-AD 21: IGKC PIP {spip.loc[21,'IGKC']:.3f}, HLA-DRB1 PIP {spip.loc[21,'HLA-DRB1']:.3f}; C351-gene set passing SEA-AD PIP>0.5: {int((spip.loc[21, g351]>0.5).sum())}")

# composition correlations
q = pd.read_csv(f"{CACHE}/qc_donor.csv", index_col=0); q.index = q.index.astype(str); q = q.loc[don]
print("\nScore vs class proportion (Pearson r / Spearman rho):")
for k in (277, 545, 172, 278, 21, 74):
    s = S.loc[k].values
    print(f"  C{k}: " + "; ".join(f"{ct} {ss.pearsonr(s, q[f'prop_{ct}'])[0]:+.2f}/{ss.spearmanr(s, q[f'prop_{ct}'])[0]:+.2f}" for ct in CTS))
cog = c.cogdx.values
print("Inh proportion by cogdx 1/2/4 KW p:", f"{ss.kruskal(*[q.prop_Inh.values[cog==l] for l in (1,2,4)]).pvalue:.3g}",
      "| Exc:", f"{ss.kruskal(*[q.prop_Exc.values[cog==l] for l in (1,2,4)]).pvalue:.3g}", "| Oli:", f"{ss.kruskal(*[q.prop_Oli.values[cog==l] for l in (1,2,4)]).pvalue:.3g}")

# source-data checks
R = np.load(f"{cfg.WORK}/component_dossier/cache/input_RNA.npy").reshape(6, 276, -1); gi = {g: i for i, g in enumerate(genes)}
def rho(k, ct, g, sub=None):
    v = R[CTS.index(ct), :, gi[g]]; s = S.loc[k].values
    if sub is not None: v, s = v[sub], s[sub]
    return ss.spearmanr(v, s)[0]
print("\nsource checks (Spearman with score):")
print("  C278 FKBP5 OPC/Oli/Ast:", [round(rho(278, ct, "FKBP5"), 2) for ct in ("OPC", "Oli", "Ast")])
print("  C21 OR2L3 Exc all/male/female:", round(rho(21, "Exc", "OR2L3"), 2), round(rho(21, "Exc", "OR2L3", male), 2), round(rho(21, "Exc", "OR2L3", ~male), 2))
print("  C172 LRP8 Inh:", round(rho(172, "Inh", "LRP8"), 2), "| C277 PLCB1 Exc:", round(rho(277, "Exc", "PLCB1"), 2))
print("  C74 XIST by cell type:", [round(rho(74, ct, "XIST"), 2) for ct in CTS])
s351 = S.loc[351].values; top = int(np.argmax((s351 - s351.mean()) ** 2))
rk = {ct: int(ss.rankdata(-R[CTS.index(ct), :, gi["IGKC"]])[top]) for ct in CTS}
print("  C351 IGKC: rho Mic", round(rho(351, "Mic", "IGKC"), 2), "| dominant donor's IGKC rank (1 = highest) per cell type:", rk)
print("  C278 vs PMI rho", round(ss.spearmanr(S.loc[278], c.pmi)[0], 2), "| vs age(capped) rho",
      round(ss.spearmanr(S.loc[278], pd.to_numeric(c.age_death.astype(str).str.replace('+','',regex=False)))[0], 2),
      "| vs CERAD (4-level) rho", round(ss.spearmanr(S.loc[278], c.ceradsc)[0], 2))
# C74 intermediate donors (positions between sex-cluster medians, no IDs)
s = S.loc[74].values; mm, mf = np.median(s[male]), np.median(s[~male])
pos = np.where(male, (s - mm) / (mf - mm), (s - mf) / (mm - mf))     # 0 = own-sex median, 1 = other-sex median
print(f"\nC74: donors with position in [0.25,0.75] toward the other sex: {int(((pos>0.25)&(pos<0.75)).sum())}; >0.75: {int((pos>=0.75).sum())};"
      f" in [0.15,0.85]: {int(((pos>0.15)&(pos<0.85)).sum())}; positions of the intermediate donors: {sorted(np.round(pos[(pos>0.15)&(pos<0.85)],2))}")
gap = np.sort(np.abs(pos))[-8:]; print("  eight largest positions:", np.round(gap, 2))
