"""Point 5(b-ii) follow-up: why does matching on peak class and CpG-island content lower the CpG-in-peak enrichment?
Reports (aggregates only): CpG-island share of C74 CpGs vs all SDA CpGs (overall and within peaks); the rate at which a
universe peak holds a C74 CpG by class x island; and a CpG-level exact test conditioned on island status
(per-stratum hypergeometric convolution over Island_Feature x peak class of the covering peak)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os
import numpy as np, pandas as pd
from scipy import stats as ss
SDA = cfg.SDA; SYN = cfg.DEPOSIT   # deposit feature sidecars


def pip_row(fname, k=74):
    with open(f"{SDA}/{fname}") as f:
        hdr = f.readline().rstrip("\n").split(",")[1:]
        for line in f:
            key, rest = line.split(",", 1)
            if key == str(k): return pd.Series(np.array(rest.rstrip("\n").split(","), dtype=float), index=hdr)


meth = pd.read_csv(f"{SYN}/meth_features.csv", index_col=0); chip = pd.read_csv(f"{SYN}/chip_features.csv", index_col=0)
c74cpg = (pip_row("pip_final_meth.csv") > 0.5).values; c74pk = (pip_row("pip_final_chip.csv") > 0.5).values
isl = meth.Island_Feature.fillna("OpenSea").values
print("Island_Feature share, C74 CpGs vs all SDA CpGs:")
t = pd.DataFrame({"all": pd.Series(isl).value_counts(normalize=True), "C74": pd.Series(isl[c74cpg]).value_counts(normalize=True)})
print(t.round(3).to_string())
auto = ~chip.Chr.astype(str).isin(["X", "Y"]).values; P = chip[auto]; pk74 = c74pk[auto]
pos = meth.MAPINFO.values.astype(np.int64); chr_ = meth.CHR.astype(str).values
cover_cls = np.array([""] * len(meth), dtype=object); in74 = np.zeros(len(meth), bool)
for ch, grp in P.assign(c74=pk74).groupby(P.Chr.astype(str)):
    m = np.where(chr_ == ch)[0]
    for s, e, c, f in zip(grp.Start.values, grp.End.values, grp.Classification.values, grp.c74.values):
        h = m[(pos[m] >= s) & (pos[m] <= e)]
        cover_cls[h] = np.where(cover_cls[h] == "", c, cover_cls[h]); in74[h] |= f
inany = cover_cls != ""
print(f"\nwithin peaks: Island share C74 {np.mean(isl[c74cpg & inany] == 'Island'):.3f} vs all {np.mean(isl[inany] == 'Island'):.3f}; "
      f"share of in-peak CpGs that are in a C74 peak: Island {in74[inany & (isl == 'Island')].mean():.4f} vs non-island {in74[inany & (isl != 'Island')].mean():.4f}")
# CpG-level exact test stratified by island status x class of the covering peak
strata = pd.Series(isl[inany]).where(lambda s: s == "Island", "nonIsland").values + "_" + cover_cls[inany].astype(str)
a = c74cpg[inany]; b = in74[inany]; k = int((a & b).sum()); dist = np.array([1.0]); E = 0
for s in np.unique(strata):
    m = strata == s; u, nb, na = m.sum(), int(b[m].sum()), int(a[m].sum())
    E += na * nb / u
    if nb and na: dist = np.convolve(dist, ss.hypergeom.pmf(np.arange(min(na, nb) + 1), u, nb, na))
print(f"CpG-level, stratified by island status x peak class: k {k}, expected {E:.2f}, fold {k/E:.2f}, exact p {dist[k:].sum():.3g} "
      f"(unstratified: 54 vs 6.58, 8.2-fold, p 1.9e-37; still treats CpGs in one peak as independent)")
