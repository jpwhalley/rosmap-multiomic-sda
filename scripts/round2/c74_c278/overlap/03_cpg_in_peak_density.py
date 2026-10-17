"""Point 5(b-ii): does the C74 CpG-in-C74-H3K9ac-peak enrichment (published 54 CpGs, 8.2-fold) need a probe-density or
chromosome correction, or a correction for CpG clustering within peaks?

Step 0 reproduces the published Q3 (round-1 revision script 04_c74_cpg_followthrough.py): deposit sidecars
($G_ROS_DEPOSIT_DIR/{meth,chip}_features.csv; GRCh37) + SDA PIP rows of component 74.
  CpG level: universe = autosomal SDA CpGs inside any autosomal peak (17,435); successes = CpGs inside a C74 PIP > 0.5
  autosomal peak (100); draws = C74 PIP > 0.5 CpGs inside any peak (1,148); k = 54 (expected 6.58).
  Peak level: universe = autosomal peaks holding >= 1 SDA CpG (8,782); 910 hold a C74 CpG; 48 are C74 peaks; k = 22.
The CpG-level test treats the 54 CpGs as independent, but they sit in 22 peaks. The dependent unit is the peak, so the
nulls below permute the C74 label over whole peaks (each peak keeps its own CpGs and their C74 status):
  S1 = number of C74 CpGs inside C74-labelled peaks (the published 54);  S2 = C74-labelled peaks holding >= 1 C74 CpG (22).
Strata (the C74 label is permuted within each stratum, preserving the per-stratum count of C74 peaks):
  P0 none; P1 SDA-CpG count of the peak (density); P1b 450K-probe count of the peak (array density, deciles);
  P2 chromosome; P3 chromosome x density; P4 density x peak class (Promoter/Enhancer/Other) x island (any SDA CpG in a
  CpG island); P5 chromosome x density x class; P6 chromosome x density x class x island.
S2 is exact (convolution of per-stratum hypergeometrics); S1 is sampled (B = 100,000, seed below).
Calibration: size match is exact by construction; frac of draws reaching the observation and null max are printed;
positive control: spike s extra C74-CpG-holding peaks into the C74 label (replacing same-stratum peaks without one) and
record detection at p < 0.05 under P5 (S2).
Outputs: out/03_* (aggregates only; no donor data)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, json, sys
os.environ.setdefault("OMP_NUM_THREADS", cfg.THREADS); os.environ.setdefault("OPENBLAS_NUM_THREADS", cfg.THREADS); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", cfg.THREADS)
import numpy as np, pandas as pd
from scipy import stats as ss

SDA = cfg.SDA
SYN = cfg.DEPOSIT   # deposit feature sidecars (meth/chip identical in V1 and V2)
MAN = f"{cfg.REFS}/ROSMAP_arrayMethylation_metaData.tsv"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = cfg.out_dir(cfg.WORK, "v23_review", "point_5_analysis", "out")
SEED = 20261014; rng = np.random.default_rng(SEED); K = 74


def pip_row(fname, k=K):
    with open(f"{SDA}/{fname}") as f:
        hdr = f.readline().rstrip("\n").split(",")[1:]
        for line in f:
            key, rest = line.split(",", 1)
            if key == str(k):
                return pd.Series(np.array(rest.rstrip("\n").split(","), dtype=float), index=hdr)


meth = pd.read_csv(f"{SYN}/meth_features.csv", index_col=0)
chip = pd.read_csv(f"{SYN}/chip_features.csv", index_col=0)
pm = pip_row("pip_final_meth.csv"); pc = pip_row("pip_final_chip.csv")
assert list(pm.index) == list(meth.index) and list(pc.index) == list(chip.index)
meth["chr"] = meth.CHR.astype(str); chip["chr"] = chip.Chr.astype(str)
c74cpg = (pm > 0.5).values; c74peak = (pc > 0.5).values
auto_pk = ~chip.chr.isin(["X", "Y"]).values
print(f"C74 PIP>0.5 CpGs {c74cpg.sum()} (all autosomal: {not meth.chr.isin(['X','Y']).any()}); C74 PIP>0.5 peaks {c74peak.sum()}, autosomal {(c74peak & auto_pk).sum()}")

# CpG -> covering autosomal peaks (closed intervals, as the original)
P = chip[auto_pk]; pk_ids = P.index.values; pk_c74 = c74peak[auto_pk]
cpg_pos = meth.MAPINFO.values.astype(np.int64); cpg_chr = meth.chr.values
pairs = []                                   # (cpg index, peak index within P)
for ch, grp in P.groupby("chr"):
    m = np.where(cpg_chr == ch)[0]
    if len(m) == 0: continue
    pos = cpg_pos[m]; o = np.argsort(pos); pos_s = pos[o]
    for j, (s, e) in zip(np.where(P.chr.values == ch)[0], zip(grp.Start.values, grp.End.values)):
        lo, hi = np.searchsorted(pos_s, s, "left"), np.searchsorted(pos_s, e, "right")
        for t in range(lo, hi): pairs.append((m[o[t]], j))
pairs = np.array(pairs)
in_any = np.zeros(len(meth), bool); in_any[pairs[:, 0]] = True
in_c74pk = np.zeros(len(meth), bool); in_c74pk[pairs[pk_c74[pairs[:, 1]], 0]] = True
M_h, n_h, N_h = in_any.sum(), in_c74pk.sum(), (c74cpg & in_any).sum(); k_h = (c74cpg & in_c74pk).sum()
exp_h = n_h * N_h / M_h; p_h = ss.hypergeom.sf(k_h - 1, M_h, n_h, N_h)
print(f"[0] CpG level: universe {M_h}, in C74 peaks {n_h}, C74 CpGs in any peak {N_h}, k {k_h}, expected {exp_h:.2f}, fold {k_h/exp_h:.2f}, p {p_h:.3g}"
      f"   (published 17435/100/1148/54, 6.58, 8.2, 1.9e-37)")
alt_exp = c74cpg.sum() * n_h / len(meth)
print(f"    all-CpG universe variant (verifier): expected {alt_exp:.2f}, fold {k_h/alt_exp:.2f}, p {ss.hypergeom.sf(k_h-1, len(meth), n_h, c74cpg.sum()):.3g}")
print(f"    C74 CpGs inside any peak {N_h}/{c74cpg.sum()} = {N_h/c74cpg.sum():.3f} vs all SDA CpGs {M_h}/{len(meth)} = {M_h/len(meth):.3f}")

# peak-level table
npk = len(P)
n_sda = np.bincount(pairs[:, 1], minlength=npk)
n_c74 = np.bincount(pairs[:, 1], weights=c74cpg[pairs[:, 0]], minlength=npk).astype(int)
isl = np.bincount(pairs[:, 1], weights=(meth.Island_Feature.values[pairs[:, 0]] == "Island"), minlength=npk) > 0
univ = n_sda > 0
k2 = int(((n_c74 > 0) & pk_c74 & univ).sum()); M2, n2, N2 = univ.sum(), ((n_c74 > 0) & univ).sum(), (pk_c74 & univ).sum()
e2 = n2 * N2 / M2
print(f"[0] peak level: universe {M2}, peaks with a C74 CpG {n2}, C74 peaks in universe {N2}, k {k2}, expected {e2:.2f}, fold {k2/e2:.2f}, "
      f"p {ss.hypergeom.sf(k2-1, M2, n2, N2):.3g}   (published 8782/910/48/22, 4.97, 4.42, 3.4e-10)")
S1obs = int(n_c74[pk_c74].sum()); assert S1obs == k_h
# 450K array probe count per peak (whole manifest; GRCh37)
man = pd.read_csv(MAN, sep="\t", usecols=["TargetID", "CHR", "MAPINFO"]).dropna()
man["CHR"] = man.CHR.astype(str); n450 = np.zeros(npk, int)
for ch, grp in P.groupby("chr"):
    pos = np.sort(man.loc[man.CHR == ch, "MAPINFO"].values.astype(np.int64))
    idx = np.where(P.chr.values == ch)[0]
    n450[idx] = np.searchsorted(pos, grp.End.values, "right") - np.searchsorted(pos, grp.Start.values, "left")

# ---- descriptives (universe peaks) ----
U = np.where(univ)[0]
d = pd.DataFrame({"chr": P.chr.values[U], "cls": P.Classification.values[U], "isl": isl[U], "n_sda": n_sda[U], "n450": n450[U],
                  "width": P.End.values[U] - P.Start.values[U] + 1, "c74pk": pk_c74[U], "n_c74": n_c74[U]})
d["has_c74"] = d.n_c74 > 0
print("\n[desc] universe peaks: C74 vs other")
for col in ["n_sda", "n450", "width"]:
    print(f"   {col}: C74 mean {d.loc[d.c74pk, col].mean():.2f} (median {d.loc[d.c74pk, col].median():.0f}) vs other {d.loc[~d.c74pk, col].mean():.2f} "
          f"(median {d.loc[~d.c74pk, col].median():.0f}); MW p {ss.mannwhitneyu(d.loc[d.c74pk, col], d.loc[~d.c74pk, col]).pvalue:.3g}")
print(f"   promoter share: C74 {np.mean(d.loc[d.c74pk, 'cls'] == 'Promoter'):.2f} vs other {np.mean(d.loc[~d.c74pk, 'cls'] == 'Promoter'):.2f}; "
      f"island share: C74 {d.loc[d.c74pk, 'isl'].mean():.2f} vs other {d.loc[~d.c74pk, 'isl'].mean():.2f}")
print(f"   P(peak holds a C74 CpG) by n_sda: " + ", ".join(f"{a}: {b:.3f}" for a, b in d.groupby(np.minimum(d.n_sda, 8)).has_c74.mean().items()))
# chromosome: C74 CpG share and C74 peak share
cc = pd.DataFrame({"SDA_CpGs": meth.chr.value_counts(), "C74_CpGs": meth.chr[c74cpg].value_counts(),
                   "univ_peaks": d.chr.value_counts(), "C74_univ_peaks": d.chr[d.c74pk].value_counts(),
                   "C74_univ_peaks_with_C74_CpG": d.chr[d.c74pk & d.has_c74].value_counts(),
                   "in_peak_C74_CpGs_in_C74_peaks": d[d.c74pk].groupby("chr").n_c74.sum()}).fillna(0).astype(int)
cc["C74_CpG_rate"] = cc.C74_CpGs / cc.SDA_CpGs
cc = cc.loc[sorted(cc.index, key=lambda x: int(x))]
print("[desc] by chromosome:\n" + cc.to_string()); cc.to_csv(f"{OUT}/03_by_chromosome.csv")
chi = ss.chi2_contingency(np.vstack([cc.C74_CpGs, cc.SDA_CpGs - cc.C74_CpGs]))
print(f"   C74 CpG rate heterogeneity across chromosomes: chi2 {chi[0]:.1f}, df {chi[2]}, p {chi[1]:.3g}")

# ---- stratified peak-label permutations ----
dens_bin = np.digitize(d.n_sda.values, [1, 2, 3, 4, 5, 7, 11]) - 1
coarse = np.digitize(d.n_sda.values, [1, 2, 4]) - 1          # 1 / 2-3 / 4+
q = np.unique(np.quantile(d.n450.values, np.linspace(0, 1, 11)))
b450 = np.clip(np.searchsorted(q, d.n450.values, side="right") - 1, 0, len(q) - 2)
STRATA = {
    "P0 none": np.zeros(len(d), int),
    "P1 density (SDA CpGs per peak: 1,2,3,4,5-6,7-10,11+)": dens_bin,
    "P1b array density (450K probes per peak, deciles)": b450,
    "P2 chromosome": pd.factorize(d.chr)[0],
    "P3 chromosome x density (1 / 2-3 / 4+)": pd.factorize(d.chr + "_" + coarse.astype(str))[0],
    "P4 density x class x island": pd.factorize(pd.Series(dens_bin).astype(str).values + "_" + d.cls.values + "_" + d.isl.astype(str).values)[0],
    "P5 chromosome x density (1/2-3/4+) x class": pd.factorize(d.chr + "_" + coarse.astype(str) + "_" + d.cls.values)[0],
    "P6 chromosome x density (1/2-3/4+) x class x island": pd.factorize(d.chr + "_" + coarse.astype(str) + "_" + d.cls.values + "_" + d.isl.astype(str).values)[0],
}
lab = d.c74pk.values; has = d.has_c74.values; cnt = d.n_c74.values
NB = 100_000


def exact_S2(st, lab_, k):
    dist = np.array([1.0])
    for s in np.unique(st):
        m = st == s; u, a, n = m.sum(), (has & m).sum(), (lab_ & m).sum()
        if a == 0 or n == 0: continue
        x = np.arange(0, min(a, n) + 1); dist = np.convolve(dist, ss.hypergeom.pmf(x, u, a, n))
    x = np.arange(len(dist)); cdf = np.cumsum(dist)
    return dict(mean=float((x * dist).sum()), p=float(dist[k:].sum()) if k < len(dist) else 0.0,
                q95=int(np.searchsorted(cdf, 0.95)), q999=int(np.searchsorted(cdf, 0.999)))


def sample_S1(st, ndraw=NB, chunk=5000):
    out = np.zeros(ndraw)
    for s in np.unique(st):
        idx = np.where(st == s)[0]; n = int(lab[idx].sum())
        if n == 0: continue
        c = cnt[idx]
        if c.sum() == 0: continue
        for a in range(0, ndraw, chunk):
            m = min(chunk, ndraw - a)
            keys = rng.random((m, len(idx)))
            sel = np.argpartition(keys, n - 1, axis=1)[:, :n] if n < len(idx) else np.tile(np.arange(len(idx)), (m, 1))
            out[a:a + m] += c[sel].sum(1)
    return out


rows = []; feasible = {}
for name, st in STRATA.items():
    # strata holding a C74 peak but no other peak make the label fixed there (no freedom): report it
    fixed = sum(1 for s in np.unique(st) if (lab & (st == s)).sum() == (st == s).sum() and (lab & (st == s)).sum() > 0)
    e = exact_S2(st, lab, k2); s1 = sample_S1(st)
    p1 = (1 + (s1 >= S1obs).sum()) / (NB + 1)
    rows.append(dict(null=name, n_strata=len(np.unique(st)), strata_all_C74=fixed,
                     S2_obs=k2, S2_mean=e["mean"], S2_fold=k2 / e["mean"], S2_p_exact=e["p"], S2_q95=e["q95"], S2_q999=e["q999"],
                     S1_obs=S1obs, S1_mean=s1.mean(), S1_fold=S1obs / s1.mean(), S1_p=p1, S1_p_floor=1 / (NB + 1),
                     S1_null_max=s1.max(), S1_q95=np.quantile(s1, 0.95), S1_q999=np.quantile(s1, 0.999),
                     S1_frac_draws_ge_half_obs=float(np.mean(s1 >= S1obs / 2))))
R = pd.DataFrame(rows).set_index("null")
pd.set_option("display.width", 250)
print(f"\n[nulls] peak-label permutations; S1 sampled B = {NB:,}, seed {SEED}; S2 exact")
print(R.to_string(float_format=lambda x: f"{x:.4g}"))
R.to_csv(f"{OUT}/03_peak_nulls.csv")

# ---- positive control for S2 under P5 (and P0) ----
pcr = []
for name in ["P0 none", "P5 chromosome x density (1/2-3/4+) x class", "P6 chromosome x density (1/2-3/4+) x class x island"]:
    st = STRATA[name]
    for s in [1, 2, 3, 4, 5, 6, 8]:
        det = 0; R_ = 300
        for _ in range(R_):
            labn = np.zeros(len(d), bool)
            for g in np.unique(st):                     # null label with the observed per-stratum counts
                idx = np.where(st == g)[0]; n = int(lab[idx].sum())
                if n: labn[rng.choice(idx, n, replace=False)] = True
            for _ in range(s):                          # swap: a labelled peak without a C74 CpG -> same-stratum peak with one
                cand = np.where(labn & ~has)[0]
                if not len(cand): break
                i = rng.choice(cand); pool = np.where((st == st[i]) & has & ~labn)[0]
                if len(pool): labn[i] = False; labn[rng.choice(pool)] = True
            kk = int((labn & has).sum()); det += exact_S2(st, labn, kk)["p"] < 0.05
        pcr.append(dict(null=name, spiked_peaks=s, detection_rate=det / R_))
pcr = pd.DataFrame(pcr); print("\n[positive control] S2 detection at p < 0.05:\n" + pcr.to_string(index=False))
pcr.to_csv(f"{OUT}/03_positive_control.csv", index=False)
json.dump(dict(cpg_level=dict(M=int(M_h), n=int(n_h), N=int(N_h), k=int(k_h), expected=exp_h, fold=k_h / exp_h, p=p_h),
               peak_level=dict(M=int(M2), n=int(n2), N=int(N2), k=k2, expected=e2, fold=k2 / e2), seed=SEED, B=NB),
          open(f"{OUT}/03_summary.json", "w"), indent=1, default=float)
