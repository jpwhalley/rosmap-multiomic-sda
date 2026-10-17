"""Cross-cohort RNA concordance (SEA-AD) for all 547 components: Pearson (published rule), Spearman over the same matched
genes, permutation nulls, and the SEA-AD partner's own donor concentration.

Published rule (12_seaad_validation.py; re-derived in 2026-10-08 seaad141/): genes = sorted ROSMAP RNA genes with PIP > 0.5
that are also SEA-AD genes; r = Pearson between the ROSMAP loading vector and each of the 141 corrected SEA-AD components'
loadings over those genes; partner = larger |r| (max if |max| >= |min|). Reproduced here and asserted against Data S1.

Added: rho = Spearman between the same two vectors at the same partner; best-|rho| partner; leverage (r after dropping the
single gene contributing most to the Pearson numerator); SEA-AD partner concentration (centred top-sample share, PR over 82).

Joint concordance statistic J = min(|r|, |rho|) at the Pearson partner if sign(r) == sign(rho), else 0.
Null A (primary; keeps the gene set and every SEA-AD loading, breaks only the gene-to-weight correspondence):
    permute the ROSMAP loading values across the matched genes, redo the partner search over all 141 SEA-AD components,
    recompute J. B = 2000, seed 20261013. p = (1 + #{J_null >= J_obs}) / (B + 1); floor 1/2001.
Null B (secondary; random gene sets of the same size from the common genes; keeps ROSMAP values) for components with
    top-donor share < 0.5 plus the named components; B = 500.
Calibration diagnostics per component: null draw size (= observed by construction), fraction of null draws with
|r| >= 0.4 and |rho| >= 0.4 (can the null produce a 'hit'), null 95/99% quantiles of J (detection threshold in J units).
Positive control: (i) reference panel = positives C74, C278, C172, C277 (SEA-AD RNA signature confirmed, assessment 10-13),
negatives C351, C2 (match rests on one donor / a few genes); (ii) spike-in: synthetic ROSMAP vectors = c * z(SEA-AD partner
loadings) + sqrt(1-c^2) * z(permuted own loadings) for c in a grid, on five distributed components spanning the gene-count range.
Writes out/02_seaad_547.csv, out/02_seaad_positive_control.csv, out/02_seaad.log (via tee)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import time, numpy as np, pandas as pd
from scipy import stats as ss
from st_common import *

t0 = time.time()
rng = np.random.default_rng(20261013)
BA, BB = 2000, 500
M = pd.read_csv(f"{OUT}/01_master_547.csv", index_col=0)
ds1 = pd.read_csv(DS1, index_col=0)
fn = pd.read_csv(GI, index_col=0)["feature_name"]
sg = pd.read_csv(f"{SEA}/genes_final.csv", index_col=0)
sg.columns = [fn[c] for c in sg.columns]
assert sg.columns.is_unique and len(sg) == 143
sg = sg.drop(index=[9, 128]); assert len(sg) == 141
seaids = sg.index.values.astype(int)
VSP = f"{cfg.WORK}/verify_source_panels/cache"
ros = list(np.load(f"{VSP}/RNA_feats.npy"))
assert ros == open(f"{SDA}/genes.txt").read().splitlines()
X = np.load(f"{VSP}/RNA_X.npy"); P = np.load(f"{VSP}/RNA_P.npy")
common = sorted(set(sg.columns) & set(ros))
cpos = np.array([ros.index(g) for g in common]); SG = sg[common].values.astype(float)       # 141 x common
print(f"common genes {len(common)}; SEA-AD components {len(sg)} [{time.time()-t0:.0f}s]", flush=True)

def xp(k):
    if k <= 546: return X[k - 1], P[k - 1]
    return np.load(f"{AUD}/c547/C{k}_X1.npy"), np.load(f"{AUD}/c547/C{k}_P1.npy")

def zrows(A):
    A = A - A.mean(-1, keepdims=True); return A / np.sqrt((A ** 2).sum(-1, keepdims=True))

def rankrows(A):
    """row-wise ranks with ties averaged (scipy), as float"""
    return ss.rankdata(A, axis=-1)

def partner_stats(x, Sz, SRz):
    """x: ROSMAP vector over genes; Sz/SRz: z-scored (unit-norm) SEA-AD rows and rank rows. Returns r, rho vectors."""
    xz = zrows(x[None, :])[0]; xr = zrows(ss.rankdata(x)[None, :])[0]
    return Sz @ xz, SRz @ xr

def pick(r):
    return int(np.argmax(r)) if abs(r.max()) >= abs(r.min()) else int(np.argmin(r))

def J(r, rho):
    return min(abs(r), abs(rho)) if np.sign(r) == np.sign(rho) else 0.0

def null_A(x, Sz, SRz, B):
    n = len(x)
    perm = np.argsort(rng.random((B, n)), axis=1)
    xz = zrows(x[None, :])[0]; xr = zrows(ss.rankdata(x)[None, :])[0]
    R = xz[perm] @ Sz.T; RH = xr[perm] @ SRz.T                       # B x 141
    jj = np.where(np.abs(R.max(1)) >= np.abs(R.min(1)), R.argmax(1), R.argmin(1))
    rp = R[np.arange(B), jj]; hp = RH[np.arange(B), jj]
    Jn = np.where(np.sign(rp) == np.sign(hp), np.minimum(np.abs(rp), np.abs(hp)), 0.0)
    return Jn, np.abs(R).max(1), np.abs(RH).max(1), rp, hp

def null_B(x, B):
    n = len(x); xz = zrows(x[None, :])[0]; xr = zrows(ss.rankdata(x)[None, :])[0]
    Jn = np.empty(B); rmax = np.empty(B); rp = np.empty(B); hp = np.empty(B)
    for b in range(B):
        idx = rng.choice(len(common), size=n, replace=False)
        Ss = SG[:, idx]; Sz = zrows(Ss); SRz = zrows(rankrows(Ss))
        r = Sz @ xz; h = SRz @ xr; j = pick(r)
        rp[b], hp[b], rmax[b] = r[j], h[j], np.abs(r).max(); Jn[b] = J(r[j], h[j])
    return Jn, rmax, rp, hp

SAF = pd.read_csv(f"{SEA}/samples_final.csv", index_col=0)
seaconc = {int(k): concentration(SAF.loc[k].values) for k in SAF.index}

need_B = set(M.index[M.top_share < 0.5]) | set(FOCUS + OUTGOING + OTHER_NAMED)
rows = []
for k in M.index:
    x, p = xp(k)
    sel = p[cpos] > 0.5; genes_idx = np.where(sel)[0]; n = len(genes_idx)
    rec = dict(component=k, seaad_n_genes=n)
    if n <= 1:
        rows.append(rec); continue
    xv = x[cpos][sel]; Ss = SG[:, sel]; Sz = zrows(Ss); SRz = zrows(rankrows(Ss))
    r, rho = partner_stats(xv, Sz, SRz)
    j = pick(r); jr = int(np.argmax(np.abs(rho)))
    # published-rule reproduction
    assert int(seaids[j]) == int(ds1.loc[k, "SEA-AD_comp"]), (k, seaids[j], ds1.loc[k, "SEA-AD_comp"])
    assert abs(r[j] - ds1.loc[k, "SEA-AD_corr"]) < 1e-9, k
    a = np.abs(r); o = np.argsort(-a)
    # leverage: drop the gene with the largest contribution to the Pearson numerator, recompute r at the same partner
    xz = xv - xv.mean(); sz = Ss[j] - Ss[j].mean(); contrib = xz * sz * np.sign(r[j])
    g1 = int(np.argmax(contrib)); keep = np.ones(n, bool); keep[g1] = False
    r_drop1 = np.corrcoef(xv[keep], Ss[j][keep])[0, 1] if n > 3 else np.nan
    top3 = np.sort(contrib)[-3:].sum() / contrib.sum() if contrib.sum() > 0 else np.nan
    sc = seaconc[int(seaids[j])]
    rec.update(seaad_partner=int(seaids[j]), seaad_pearson=float(r[j]), seaad_spearman=float(rho[j]),
               seaad_runnerup_abs_pearson=float(a[o[1]]), seaad_best_spearman_partner=int(seaids[jr]),
               seaad_best_abs_spearman=float(abs(rho[jr])), seaad_pearson_drop_top_gene=float(r_drop1),
               seaad_top3_gene_share_of_numerator=float(top3),
               seaad_J=J(r[j], rho[j]), seaad_partner_top_share=float(sc["top_share"]), seaad_partner_PR=float(sc["PR"]))
    Jn, rmaxA, hmaxA, rpA, hpA = null_A(xv, Sz, SRz, BA)
    rec.update(nullA_p_J=float((1 + np.sum(Jn >= rec["seaad_J"])) / (BA + 1)),
               nullA_p_pearson=float((1 + np.sum(rmaxA >= abs(r[j]))) / (BA + 1)),
               nullA_J_q95=float(np.quantile(Jn, 0.95)), nullA_J_q99=float(np.quantile(Jn, 0.99)),
               nullA_maxabs_r_median=float(np.median(rmaxA)),
               nullA_frac_draws_hit04=float(np.mean((np.abs(rpA) >= 0.4) & (np.abs(hpA) >= 0.4) & (np.sign(rpA) == np.sign(hpA)))))
    if k in need_B:
        JnB, rmaxB, rpB, hpB = null_B(xv, BB)
        rec.update(nullB_p_J=float((1 + np.sum(JnB >= rec["seaad_J"])) / (BB + 1)),
                   nullB_p_pearson=float((1 + np.sum(rmaxB >= abs(r[j]))) / (BB + 1)),
                   nullB_J_q95=float(np.quantile(JnB, 0.95)), nullB_J_q99=float(np.quantile(JnB, 0.99)),
                   nullB_frac_draws_hit04=float(np.mean((np.abs(rpB) >= 0.4) & (np.abs(hpB) >= 0.4) & (np.sign(rpB) == np.sign(hpB)))))
    rows.append(rec)
    if k % 50 == 0: print(f"  C{k} done [{time.time()-t0:.0f}s]", flush=True)
R = pd.DataFrame(rows).set_index("component")
R.to_csv(f"{OUT}/02_seaad_547.csv")
print(f"published partner and Pearson reproduced for all {int(R.seaad_partner.notna().sum())} components with >1 matched gene "
      f"({int(R.seaad_partner.isna().sum())} blank, as in Data S1) [{time.time()-t0:.0f}s]")

# ---------------------------------------------------------------- calibration summary
ok = R.seaad_partner.notna()
print(f"\nNull A (B={BA}): floor {1/(BA+1):.2e}; draw size = observed size (by construction).")
print(f"  fraction of null draws giving |r|>=0.4 & |rho|>=0.4 same sign: median {R.loc[ok,'nullA_frac_draws_hit04'].median():.4f}; "
      f"components where >5% of draws hit: {int((R.loc[ok,'nullA_frac_draws_hit04'] > 0.05).sum())}; none hit: {int((R.loc[ok,'nullA_frac_draws_hit04'] == 0).sum())}")
print(f"  null J 99% quantile vs n genes: n<50 median {R.loc[ok & (R.seaad_n_genes<50),'nullA_J_q99'].median():.3f}; "
      f"50-200 {R.loc[ok & R.seaad_n_genes.between(50,200),'nullA_J_q99'].median():.3f}; >200 {R.loc[ok & (R.seaad_n_genes>200),'nullA_J_q99'].median():.3f}")
okB = R.nullB_p_J.notna()
print(f"Null B (B={BB}) on {int(okB.sum())} components: frac draws hit04 median {R.loc[okB,'nullB_frac_draws_hit04'].median():.4f}; "
      f"Spearman(p_A, p_B) {ss.spearmanr(R.loc[okB,'nullA_p_J'], R.loc[okB,'nullB_p_J'])[0]:.3f}")
panel = FOCUS + OUTGOING + OTHER_NAMED
print("\nReference panel and named components:")
print(R.loc[panel, ["seaad_n_genes", "seaad_partner", "seaad_pearson", "seaad_spearman", "seaad_runnerup_abs_pearson",
                    "seaad_pearson_drop_top_gene", "seaad_top3_gene_share_of_numerator", "seaad_J", "nullA_J_q99", "nullA_p_J",
                    "nullB_J_q99", "nullB_p_J", "seaad_partner_top_share"]].to_string(float_format=lambda v: f"{v:.3g}"))

# ---------------------------------------------------------------- spike-in positive control
dist = R[ok & (M.top_share < 0.1)].copy()
targets = [50, 150, 400, 900, 1600]
pick_k = [int((dist.seaad_n_genes - t).abs().idxmin()) for t in targets]
cs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
pc = []
for k in pick_k:
    x, p = xp(k); sel = p[cpos] > 0.5; xv = x[cpos][sel]; Ss = SG[:, sel]; Sz = zrows(Ss); SRz = zrows(rankrows(Ss))
    j = int(np.where(seaids == R.loc[k, "seaad_partner"])[0][0])
    for c in cs:
        det = []; Jobs = []
        for rep in range(20):
            noise = rng.permutation(xv)
            xs = c * (Ss[j] - Ss[j].mean()) / Ss[j].std() + np.sqrt(1 - c ** 2) * (noise - noise.mean()) / noise.std()
            r, rho = partner_stats(xs, Sz, SRz); jj = pick(r); Jo = J(r[jj], rho[jj])
            Jn, *_ = null_A(xs, Sz, SRz, 500)
            det.append((1 + np.sum(Jn >= Jo)) / 501 < 0.01); Jobs.append(Jo)
        pc.append(dict(component=k, n_genes=int(sel.sum()), c=c, detect_rate_p01=float(np.mean(det)), median_J=float(np.median(Jobs))))
PC = pd.DataFrame(pc); PC.to_csv(f"{OUT}/02_seaad_positive_control.csv", index=False)
print("\nSpike-in (detection at null-A p < 0.01, 20 repeats, B=500): rows = component (n genes), columns = mixing weight c")
print(PC.pivot_table(index=["component", "n_genes"], columns="c", values="detect_rate_p01").to_string(float_format=lambda v: f"{v:.2f}"))
print(f"done [{time.time()-t0:.0f}s]")
