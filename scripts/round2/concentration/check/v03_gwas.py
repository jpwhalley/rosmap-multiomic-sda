"""VERIFY part 3: GWAS per component, re-scanned with own code from the archived GEMMA 0.96 abbr files (1..546, 520 kept
in the scan but dropped from the 547) and the local C547/C548 runs. Min p overall / autosomal, counts, lambda_GC from p_score.
Checks that fam phenotype column k equals the deposited score of component k. Tests whether C74's chr-0 hit SNPs are
sex-linked (genotype vs fam sex). Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats as ss

V = cfg.out_dir(cfg.WORK, "revision_analyses", "verify_structure"); cfg.out_dir(V, "out")
cfg.out_dir(V, "private")
SN = cfg.SNP
ARC = f"{SN}/gemma/output"
AUD = cfg.AUD
ROS = cfg.SDA
MED1 = ss.chi2.ppf(0.5, 1)

def one(k):
    f = f"{ARC}/results{k}.abbr.txt" if k <= 546 else f"{cfg.AUD}/gwas547/output/local_C{k}.abbr.txt"
    a = pd.read_csv(f, sep=" ", header=None, usecols=[0, 3], names=["chr", "p"], dtype={"chr": np.int16, "p": np.float64}, engine="c")
    chrv = a.chr.values; p = a.p.values
    au = (chrv >= 1) & (chrv <= 22)
    return dict(component=k, n_snps=len(p), min_p=p.min(), top_chr=int(chrv[p.argmin()]), min_p_auto=p[au].min(),
                n5e8_auto=int((p[au] < 5e-8).sum()), n5e8_nonauto=int((p[~au] < 5e-8).sum()), n1e15=int((p < 1e-15).sum()),
                lam=float(np.median(ss.chi2.isf(p, 1)) / MED1), lam_auto=float(np.median(ss.chi2.isf(p[au], 1)) / MED1))

if __name__ == "__main__":
    comps = [k for k in range(1, 547) if k != 520] + [547, 548]
    with Pool(int(cfg.THREADS)) as pool:
        rows = pool.map(one, comps, chunksize=8)
    G = pd.DataFrame(rows).set_index("component")
    G.to_csv(f"{V}/out/v03_gwas_547.csv")
    print("snp counts:", G.n_snps.unique())
    st = pd.read_csv(f"{V}/../structure/out/01_master_547.csv", index_col=0)
    sb = pd.read_csv(f"{V}/../structure/out/01b_gwas_autosomal.csv", index_col=0)
    print("min p rel diff vs structure:", float((G.min_p / st.gwas_min_p - 1).abs().max()),
          "| autosomal:", float((G.min_p_auto / sb.gwas_min_p_autosomal - 1).abs().max()),
          "| lambda max abs diff vs catalogue:", float((G.lam - st.gwas_lambda).abs().max()))
    print("n autosomal <5e-8 identical:", bool((G.n5e8_auto == sb.gwas_n_below_5e8_autosomal).all()))
    # ---- fam phenotype column == deposited score?
    fam = pd.read_csv(f"{SN}/ROSMAP_arrayGenotype_maf.fam", sep=r"\s+", header=None)
    ids = fam[1].str.replace(r"^[A-Z]+", "", regex=True).values
    S = pd.read_csv(f"{ROS}/samples_final.csv", index_col=0); S.columns = S.columns.astype(str)
    dif = [np.abs(fam[5 + k].values - S.loc[k, ids].values).max() for k in range(1, 547)]
    print(f"fam column k vs samples_final row k (228 genotyped donors): max abs diff {max(dif):.2e}; n genotyped {len(ids)}")
    # ---- C74 chr-0 hits: sex-linked?
    bim = pd.read_csv(f"{SN}/ROSMAP_arrayGenotype_maf.bim", sep=r"\s+", header=None, names=["chr", "rs", "cm", "bp", "a1", "a2"])
    a74 = pd.read_csv(f"{ARC}/results74.abbr.txt", sep=" ", header=None, names=["chr", "ps", "rs", "p"])
    hits = a74[(a74.chr == 0) & (a74.p < 5e-8)].rs.tolist()
    chr0 = a74[a74.chr == 0].rs.tolist()
    n = len(fam); nb = (n + 3) // 4
    raw = np.fromfile(f"{SN}/ROSMAP_arrayGenotype_maf.bed", dtype=np.uint8)
    assert raw[:3].tolist() == [0x6c, 0x1b, 0x01] and len(raw) == 3 + nb * len(bim)
    pos = {r: i for i, r in enumerate(bim.rs)}
    sex = fam[4].values  # 1 male, 2 female
    def geno(r):
        i = pos[r]; b = raw[3 + i * nb: 3 + (i + 1) * nb]
        g = np.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], 1).ravel()[:n]
        out = np.full(n, np.nan); out[g == 0] = 2; out[g == 2] = 1; out[g == 3] = 0   # 01 = missing
        return out
    res = []
    for r in chr0:
        g = geno(r)
        okm = ~np.isnan(g)
        het_m = np.mean(g[(sex == 1) & okm] == 1); het_f = np.mean(g[(sex == 2) & okm] == 1)
        miss_m = np.mean(np.isnan(g[sex == 1])); miss_f = np.mean(np.isnan(g[sex == 2]))
        rr = abs(ss.spearmanr(g[okm], sex[okm])[0]) if np.std(g[okm]) > 0 else 0
        res.append(dict(rs=r, c74_hit=r in hits, het_male=het_m, het_female=het_f, miss_male=miss_m, miss_female=miss_f, abs_rho_geno_sex=rr))
    T = pd.DataFrame(res)
    print(f"\nC74: chr-0 SNPs {len(chr0)}, of which p<5e-8: {len(hits)}")
    print(T.groupby("c74_hit")[["het_male", "het_female", "miss_male", "miss_female", "abs_rho_geno_sex"]].median().to_string())
    print("C74 hits with |rho(genotype, sex)| > 0.5:", int((T[T.c74_hit].abs_rho_geno_sex > 0.5).sum()), "/", len(hits))
    xs = a74[a74.chr.isin([23, 24, 25, 26])]
    print("C74 min p on chr 23/24/25/26:", {c: float(xs[xs.chr == c].p.min()) for c in (23, 24, 25, 26)})
