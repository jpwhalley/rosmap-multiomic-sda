"""450K methylation: sex and identity of every raw-array sample, from probes the SDA input does not contain.
- The local annotation (metaData.tsv) and the imputed matrix are autosomal-only (420,132 probes), so sex-chromosome probes
  are taken from the raw GenomeStudio report (485,577 probes; 65,445 non-annotated kept by 03a).
- Two label-free sex readouts per sample: (Y) log2 total-intensity score on the probes loading most on the intensity sex axis;
  (X) beta score on always-detected CpG probes loading most on the beta sex axis (X-inactivation). Cluster names from majority.
- Identity: 65 rs SNP probes -> genotype calls; 15 autosomal ones also on the Affymetrix array: concordance with every genotyped person.
Per-sample values -> $G_ROS_PRIVATE_DIR; aggregates printed."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE); C = cfg.out_dir(cfg.WORK, "identity_checks", "cache")
ROS = cfg.ROS; O = f"{ROS}/oth"
d = np.load(f"{C}/meth_raw_nonauto.npz")
ids = d["ids"]; samples = d["samples"]; Pv = d["Detection_Pval"]; B = d["AVG_Beta"].astype(np.float64)
I = np.log2(d["Signal_A"].astype(np.float64) + d["Signal_B"] + 1)
mm = pd.read_csv(f"{O}/meta/ROSMAP_assay_methylationArray_metadata.csv")
bio = pd.read_csv(f"{O}/meta/ROSMAP_biospecimen_metadata.csv", low_memory=False)
clin = pd.read_csv(f"{O}/meta/ROSMAP_clinical.csv")
s2i = bio.drop_duplicates("specimenID").set_index("specimenID").individualID
i2p = dict(zip(clin.individualID, clin.projid)); pmale = dict(zip(clin.projid, clin.msex == 1))
S = pd.DataFrame(index=samples)
S["individualID"] = [s2i.get(s) for s in samples]; S["projid"] = S.individualID.map(i2p)
S["clin_male"] = S.projid.map(pmale)
imp_cols = pd.read_csv(f"{ROS}/meth/ROSMAP_arrayMethylation_imputed.tsv.gz", sep="\t", nrows=1, index_col=0).columns
S["in_imputed"] = S.index.isin(imp_cols)
sdas = [int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip()]
S["sda_meth"] = S.in_imputed & S.projid.isin(sdas)
print(f"raw samples {len(S)}; mapped to a projid {S.projid.notna().sum()}; in imputed matrix {S.in_imputed.sum()} (imputed has {len(imp_cols)}); SDA meth donors among imputed {S[S.sda_meth].projid.nunique()} (rows {S.sda_meth.sum()})")
print("projids with >1 raw sample:", int((S.projid.value_counts() > 1).sum()))
ff = (Pv > 0.01).mean(1); cg = np.char.startswith(ids, "cg"); rsm = np.char.startswith(ids, "rs")
def pc_scores(M, k=2):
    mu = M.mean(1, keepdims=True); sd = M.std(1, keepdims=True) + 1e-9; Z = (M - mu) / sd
    u, s, vt = np.linalg.svd(Z.T, full_matrices=False); return u[:, :k] * s[:k], vt[:k], Z
lab = S.clin_male.values
def auc(v):
    ok = pd.notna(lab) & np.isfinite(v); l = lab[ok].astype(bool); x = v[ok]
    return mannwhitneyu(x[l], x[~l]).statistic / (l.sum() * (~l).sum())
def kmeans_thr(v):
    s = np.sort(v[np.isfinite(v)]); a, b = np.percentile(s, 20), np.percentile(s, 80)
    for _ in range(200):
        t = (a + b) / 2; a, b = s[s <= t].mean(), s[s > t].mean()
    return t, a, b, np.sqrt((s[s <= t].var() + s[s > t].var()) / 2)
res = {}
for nm, M in [("Y_intensity", I[~rsm] - np.median(I[~rsm], 0)), ("X_beta", np.nan_to_num(B[cg & (ff < 0.001)], nan=0.5))]:
    sc, vt, Z = pc_scores(M)
    k = int(np.argmax([abs(auc(sc[:, j]) - 0.5) for j in range(2)]))      # the sex axis among PC1-2 (PC1 is a single-sample outlier axis)
    load = vt[k]; top = np.argsort(-np.abs(load))[:500]
    score = (Z[top] * np.sign(load[top])[:, None]).mean(0)
    if auc(score) < 0.5: score = -score       # orient: higher = male-like cluster (majority naming only)
    t, a, b, sd = kmeans_thr(score)
    S[nm] = score; S[nm + "_call_male"] = score > t; S[nm + "_margin_sd"] = (score - t) / sd
    print(f"{nm}: sex axis = PC{k+1}; 500 top-loading probes; AUC vs clinical {auc(score):.4f}; 2-means clusters {int((score<=t).sum())}/{int((score>t).sum())}; cluster separation {(b-a)/sd:.1f} pooled sd")
    res[nm] = top
agree_xy = (S.X_beta_call_male == S.Y_intensity_call_male)
print(f"X-call vs Y-call agree in {int(agree_xy.sum())} of {len(S)} samples")
S["meth_call"] = np.where(agree_xy, np.where(S.X_beta_call_male, "M", "F"), "X/Y conflict")
c = S[S.clin_male.notna() & (S.meth_call != "X/Y conflict")]
print(f"all mapped raw samples with clinical sex and a consistent call: {len(c)}; discordant with clinical: {int(((c.meth_call == 'M') != c.clin_male.astype(bool)).sum())}")
cs = c[c.sda_meth]
print(f"SDA meth rows: {int(S.sda_meth.sum())}; consistent calls {len(cs)}; discordant with clinical {int(((cs.meth_call == 'M') != cs.clin_male.astype(bool)).sum())}; X/Y-conflict {int((S.sda_meth & (S.meth_call == 'X/Y conflict')).sum())}")
# --- rs SNP probes -> genotype calls
Brs = B[rsm]; rsid = ids[rsm]
G = np.where(np.isnan(Brs), -1, np.where(Brs < 0.2, 0, np.where(Brs > 0.8, 2, 1)))
print(f"rs probes {len(rsid)}; call fractions hom-low/het/hom-high {np.round([(G==0).mean(), (G==1).mean(), (G==2).mean()], 3)}")
# meth-vs-meth duplicate check (all 65 rs probes): pairs with identical/near-identical calls
Gm = G.T.astype(float)
conc = np.zeros((len(S), len(S)))
for g in (0, 1, 2): conc += (Gm == g).astype(float) @ (Gm == g).astype(float).T
conc /= len(rsid); np.fill_diagonal(conc, np.nan)
iu = np.triu_indices(len(S), 1); cv = conc[iu]
print(f"meth-meth rs concordance over 65 SNPs: median {np.nanmedian(cv):.2f}, 99.9th pct {np.nanpercentile(cv, 99.9):.2f}; pairs >=0.95: {int((cv >= 0.95).sum())}")
dp = [(samples[i], samples[j]) for i, j, v in zip(*iu, cv) if v >= 0.95]
same_ind = sum(S.loc[a, "projid"] == S.loc[b, "projid"] for a, b in dp)
print(f"  of those pairs, same projid on record: {same_ind}; different/unknown projid: {len(dp) - same_ind}; involving an SDA meth row: {sum(S.loc[a,'sda_meth'] or S.loc[b,'sda_meth'] for a,b in dp)}")
pd.DataFrame(dp, columns=["a", "b"]).to_csv(f"{P}/03_meth_rs_duplicate_pairs.csv", index=False)
# vs array genotypes
Gf = f"{O}/snps/ROSMAP_arrayGenotype"
fam = pd.read_csv(f"{Gf}.fam", sep=" ", header=None); bim = pd.read_csv(f"{Gf}.bim", sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
nb = (len(fam) + 3) // 4; bed = np.memmap(f"{Gf}.bed", dtype=np.uint8, mode="r")
def geno(i):
    raw = bed[3 + i * nb: 3 + (i + 1) * nb]; out = np.empty(nb * 4, np.int8)
    for k in range(4):
        v = (raw >> (2 * k)) & 3; out[k::4] = np.select([v == 0, v == 1, v == 2, v == 3], [2, -1, 1, 0])
    return out[:len(fam)]
shared = [r for r in rsid if r in set(bim.SNP[bim.CHR.between(1, 22)])]
A = np.array([geno(int(np.where(bim.SNP == r)[0][0])) for r in shared])     # snps x people (plink A1 dosage; -1 missing)
Mg = G[[list(rsid).index(r) for r in shared]]                                   # snps x meth samples
gproj = fam[1].str[3:].astype(int).values
# orient each SNP: meth call 0/2 vs array dosage; choose flip maximising population concordance on record-matched pairs
pos = {p: i for i, p in enumerate(gproj)}
pairs = [(si, pos[p]) for si, p in enumerate(S.projid.values) if pd.notna(p) and p in pos]
flip = []
for k in range(len(shared)):
    a = np.array([Mg[k, si] for si, gi in pairs]); b = np.array([A[k, gi] for si, gi in pairs]); ok = (a >= 0) & (b >= 0)
    flip.append(np.mean(a[ok] == b[ok]) < np.mean(a[ok] == 2 - b[ok]))
Mo = np.array([np.where(Mg[k] < 0, -1, 2 - Mg[k]) if flip[k] else Mg[k] for k in range(len(shared))])
def concord(si, gi):
    a, b = Mo[:, si], A[:, gi]; ok = (a >= 0) & (b >= 0); return (a[ok] == b[ok]).mean() if ok.sum() >= 8 else np.nan, int(ok.sum())
own = []; best_other = []; rank = []
for si, gi in pairs:
    cself, n = concord(si, gi)
    allc = np.array([concord(si, g)[0] for g in range(len(fam))])
    oth = np.delete(allc, gi)
    own.append(cself); best_other.append(np.nanmax(oth)); rank.append(int(np.sum(oth >= cself)))
S.loc[S.index[[p[0] for p in pairs]], "geno_conc_own"] = own
S.loc[S.index[[p[0] for p in pairs]], "geno_conc_best_other"] = best_other
S.loc[S.index[[p[0] for p in pairs]], "geno_n_others_ge_own"] = rank
gg = S[S.geno_conc_own.notna()]
print(f"meth samples whose recorded donor is genotyped: {len(gg)} (SDA meth rows {int(gg.sda_meth.sum())}); {len(shared)} autosomal rs SNPs shared")
print(f"  concordance with own genotype: median {gg.geno_conc_own.median():.2f}, min {gg.geno_conc_own.min():.2f}; best other person: median {gg.geno_conc_best_other.median():.2f}")
print(f"  samples where own concordance < 0.8: {int((gg.geno_conc_own < 0.8).sum())} (SDA {int(((gg.geno_conc_own < 0.8) & gg.sda_meth).sum())}); where someone else matches at least as well: {int((gg.geno_n_others_ge_own > 0).sum())}")
null = []
rng = np.random.default_rng(0)
for _ in range(2000):
    si, _g = pairs[rng.integers(len(pairs))]; null.append(concord(si, rng.integers(len(fam)))[0])
print(f"  null (random person) concordance: median {np.nanmedian(null):.2f}, 99th pct {np.nanpercentile(null, 99):.2f}, max {np.nanmax(null):.2f}")
S.to_csv(f"{P}/03_meth_samples.csv")
key = pd.read_csv(f"{P}/donor_key.csv")
for pj, lb in key.values:
    r = S[S.projid == pj]
    if len(r) == 0: print(f"{lb}: no 450K sample"); continue
    for nm, x in r.iterrows():
        print(f"{lb}: 450K sample (SDA row {x.sda_meth}); Y-intensity call {'AGREES' if x.Y_intensity_call_male == x.clin_male else 'DISAGREES'} (margin {abs(x.Y_intensity_margin_sd):.1f} sd), "
              f"X-beta call {'AGREES' if x.X_beta_call_male == x.clin_male else 'DISAGREES'} (margin {abs(x.X_beta_margin_sd):.1f} sd); "
              + ("genotype check: own concordance %.2f over the shared SNPs, best other person %.2f, people matching at least as well %d" % (x.geno_conc_own, x.geno_conc_best_other, x.geno_n_others_ge_own) if pd.notna(x.get("geno_conc_own")) else "genotype check: donor not genotyped / no shared calls"))
