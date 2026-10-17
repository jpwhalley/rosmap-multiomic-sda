"""Shared helpers for QTL-based identity checks (genotype decoding, cross-fitted discovery, identity scoring)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd, time
H = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
C = cfg.out_dir(cfg.WORK, "identity_checks", "cache")
def load_geno():
    """people x SNPs int8 (0/1/2 = PLINK A1 count? we only need a consistent coding; -1 missing), projid per row, bim"""
    fam = pd.read_csv(f"{C}/plink/auto_qc.fam", sep=" ", header=None)
    bim = pd.read_csv(f"{C}/plink/auto_qc.bim", sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    n = len(fam); nb = (n + 3) // 4
    raw = np.fromfile(f"{C}/plink/auto_qc.bed", dtype=np.uint8)[3:].reshape(len(bim), nb)
    lut = np.zeros((256, 4), np.int8)
    code = np.array([2, -1, 1, 0], np.int8)
    for b in range(256):
        for k in range(4): lut[b, k] = code[(b >> (2 * k)) & 3]
    G = lut[raw].reshape(len(bim), nb * 4)[:, :n].T.copy()   # people x snps
    proj = fam[1].str[3:].astype(int).values
    return G, proj, bim
def std_geno(Gs):
    X = Gs.astype(np.float32); X[Gs < 0] = np.nan
    mu = np.nanmean(X, 0); X = np.where(np.isnan(X), mu, X); sd = X.std(0); sd[sd == 0] = 1
    return (X - mu) / sd
def residualize(E, k=10):
    """donors x phenotypes; remove top-k PCs; standardize"""
    Z = (E - E.mean(0)) / (E.std(0) + 1e-9)
    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    Z = Z - (u[:, :k] * s[:k]) @ vt[:k]
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
def discover(Ed, Gd, rthr, bim, chunk=256, min_dist=2_000_000):
    """best SNP per phenotype; keep |r|>=rthr; greedy unique SNPs >= min_dist apart on a chromosome"""
    n = Ed.shape[0]
    Ez = (Ed - Ed.mean(0)) / (Ed.std(0) + 1e-9); Gz = (Gd - Gd.mean(0)) / (Gd.std(0) + 1e-9)
    best_j = np.empty(Ez.shape[1], np.int64); best_r = np.empty(Ez.shape[1], np.float32)
    GzT = np.ascontiguousarray(Gz.T.astype(np.float32))
    for a in range(0, Ez.shape[1], chunk):
        R = (GzT @ Ez[:, a:a + chunk].astype(np.float32)) / n         # snps x chunk
        j = np.argmax(np.abs(R), 0); best_j[a:a + chunk] = j; best_r[a:a + chunk] = R[j, np.arange(R.shape[1])]
    keep = np.where(np.abs(best_r) >= rthr)[0]; keep = keep[np.argsort(-np.abs(best_r[keep]))]
    chosen = []; used = {}
    for p in keep:
        j = best_j[p]; c, bp = bim.CHR.values[j], bim.BP.values[j]
        if any(abs(bp - b2) < min_dist for b2 in used.get(c, [])): continue
        used.setdefault(c, []).append(bp); chosen.append((p, j, best_r[p]))
    return chosen, best_r
def identity_scores(z, pairs, G, af):
    """z: test phenotype vector (all phenotypes); pairs: [(p, j, r)]; G: people x snps int8; returns score per person"""
    ps = np.array([p for p, _, _ in pairs]); js = np.array([j for _, j, _ in pairs]); rs = np.array([r for _, _, r in pairs])
    y = z[ps] * np.sign(rs)
    X = G[:, js].astype(np.float32); X[G[:, js] < 0] = np.nan
    Xc = (X - 2 * af[js]) * 1.0
    sc = np.full(G.shape[0], np.nan)
    for i in range(G.shape[0]):
        ok = ~np.isnan(Xc[i]) & ~np.isnan(y)
        if ok.sum() >= 10: sc[i] = np.corrcoef(y[ok], Xc[i, ok])[0, 1]
    return sc
