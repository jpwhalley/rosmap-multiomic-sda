"""Cross-layer support in the measured (SDA input) data, for every component that is not single-donor (top-donor share < 0.5)
and has PIP > 0.5 features in >= 2 modalities.

For each such component and each active layer m:
  * feature-level: Spearman(feature value, score) over donors with data in m, after removing (i) the mislabelled samples of
    that modality (2 snRNA, 2 H3K9ac; identity_checks) and (ii) the top 1% of donors by |robust z| of the score. A feature
    counts if p < 0.05 / (all features in that layer) [Bonferroni over the whole layer, so selection of the best of many
    features cannot produce a pass] and its sign matches the loading (RNA: loading x corrected cell-type loading).
  * layer aggregate: loading-weighted mean of column-standardised PIP > 0.5 features (RNA: each cell type separately; the
    RNA aggregate is the mean of the six cell-type aggregates weighted by |corrected cell-type loading|).
  * cross-layer: Spearman between the aggregates of two different layers over donors with both (same exclusions). Two
    different measurements do not share the score's projection; reference = the same quantity in single-donor components
    with the dominant donor removed (verify_source_panels/07_circularity_null.csv).
  * protein layer evaluated twice: as fed to SDA (29/105 proteomes mis-mapped) and with the DLPFC-only (corrected) mapping.
Writes out/03_crosslayer.csv (aggregates only)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss
from st_common import *

M = pd.read_csv(f"{OUT}/01_master_547.csv", index_col=0)
todo = list(M.index[(M.top_share < 0.5) & (M.n_modalities >= 2)])
print("components evaluated:", todo)
VSP = f"{cfg.WORK}/verify_source_panels/cache"
S = scores547(); don = samples()
key = pd.read_csv(f"{cfg.PRIVATE}/donor_key.csv"); lab = dict(zip(key.projid.astype(str), key.label))
bad = {"RNA": [i for i, d in enumerate(don) if lab.get(d) in ("RNA-1", "RNA-2")],
       "chip": [i for i, d in enumerate(don) if lab.get(d) in ("CHIP-1", "CHIP-2")]}
assert len(bad["RNA"]) == 2 and len(bad["chip"]) == 2
Bc = pd.read_csv(BCOR, index_col=0)
IN = {m: np.load(f"{VSP}/{m}_input.npy") for m in ["meth", "chip", "proteins"]}
RNA = np.load(f"{VSP}/RNA_input.npy").reshape(6, 276, -1)
pfeat = list(np.load(f"{VSP}/proteins_feats.npy", allow_pickle=True))
pc = np.load(f"{VSP}/prot_dlpfc_only.npy"); pcf = [str(x) for x in np.load(f"{VSP}/prot_dlpfc_only_feats.npy", allow_pickle=True)]
pcmap = {f: i for i, f in enumerate(pcf)}
PROTC = np.full((276, len(pfeat)), np.nan)
for j, f in enumerate(pfeat):
    if f in pcmap: PROTC[:, j] = pc[:, pcmap[f]]
for m in bad:
    if m == "RNA": RNA[:, bad[m], :] = np.nan
    else: IN[m][bad[m], :] = np.nan
LX = {m: np.load(f"{VSP}/{m}_X.npy", mmap_mode="r") for m in MODS}
LP = {m: np.load(f"{VSP}/{m}_P.npy", mmap_mode="r") for m in MODS}
NFEAT = {"RNA": RNA.shape[2] * 6, "meth": IN["meth"].shape[1], "chip": IN["chip"].shape[1], "proteins": IN["proteins"].shape[1]}

def loadings(k, m):
    j = {"RNA": 1, "meth": 2, "chip": 3, "proteins": 4}[m]
    if k <= 546: return np.asarray(LX[m][k - 1]), np.asarray(LP[m][k - 1])
    return np.load(f"{AUD}/c547/C{k}_X{j}.npy"), np.load(f"{AUD}/c547/C{k}_P{j}.npy")

def rz(a):
    med = np.median(a); return np.abs(a - med) / (1.4826 * np.median(np.abs(a - med)))

def zagg(V, w):
    mu = np.nanmean(V, 0); sd = np.nanstd(V, 0); keep = sd > 0
    Z = (V[:, keep] - mu[keep]) / sd[keep]; w = w[keep]
    out = np.full(V.shape[0], np.nan); has = ~np.isnan(Z).all(1)
    out[has] = np.nansum(Z[has] * w, 1) / np.sum(np.abs(w)); return out

def keepmask(a, has):
    idx = np.where(has)[0]; k1 = int(np.ceil(0.01 * len(idx)))
    drop = idx[np.argsort(-rz(a)[idx])[:k1]]; m = has.copy(); m[drop] = False; return m

rows, aggs_all = [], {}
for k in todo:
    a = S.loc[k].values.astype(float); aggs = {}
    for m in MODS:
        X, P = loadings(k, m); sel = np.where(P > 0.5)[0]
        if len(sel) == 0: continue
        variants = [("fed", IN["proteins"]), ("corrected", PROTC)] if m == "proteins" else [("fed", None)]
        for tag, V in variants:
            if m == "RNA":
                bk = Bc.loc[k, CTS].values.astype(float); per = []; best = []
                npass = 0; ntest = 0
                for c in range(6):
                    Vc = RNA[c][:, sel]; has = ~np.isnan(Vc).all(1); keep = keepmask(a, has)
                    expr = np.nanmean(Vc[has], 0) >= 1
                    w = X[sel] * bk[c]
                    ag = zagg(Vc[:, expr], w[expr]); per.append((abs(bk[c]), ag))
                    for i in np.where(expr)[0]:
                        rho, p = ss.spearmanr(Vc[keep, i], a[keep]); ntest += 1
                        if p < 0.05 / NFEAT[m] and np.sign(rho) == np.sign(w[i]): npass += 1
                        best.append(abs(rho) if np.sign(rho) == np.sign(w[i]) else 0)
                wsum = sum(x for x, _ in per)
                agg = sum(x * g for x, g in per) / wsum
                has = ~np.isnan(agg); keep = keepmask(a, has)
                rho_agg = ss.spearmanr(agg[keep], a[keep])[0]
                rows.append(dict(component=k, layer=m, variant=tag, n_pip=len(sel), n_tested=ntest, n_donors=int(has.sum()),
                                 n_bonf_pass_predicted_sign=npass, best_abs_rho_predicted_sign=float(max(best)) if best else np.nan,
                                 bonf_threshold_p=0.05 / NFEAT[m], agg_rho_with_score=float(rho_agg)))
                aggs["RNA"] = agg
            else:
                Vm = V[:, sel] if V is not None else IN[m][:, sel]
                okc = ~np.isnan(Vm).all(0); Vm = Vm[:, okc]; w = X[sel][okc]
                has = ~np.isnan(Vm).all(1); keep = keepmask(a, has)
                npass = 0; best = []
                for i in range(Vm.shape[1]):
                    v = Vm[keep, i]; ok = ~np.isnan(v)
                    if ok.sum() < 20 or np.std(v[ok]) == 0: continue
                    rho, p = ss.spearmanr(v[ok], a[keep][ok])
                    if p < 0.05 / NFEAT[m] and np.sign(rho) == np.sign(w[i]): npass += 1
                    best.append(abs(rho) if np.sign(rho) == np.sign(w[i]) else 0)
                agg = zagg(Vm, w); rho_agg = ss.spearmanr(agg[keep], a[keep])[0]
                rows.append(dict(component=k, layer=m, variant=tag, n_pip=len(sel), n_tested=int(Vm.shape[1]), n_donors=int(has.sum()),
                                 n_bonf_pass_predicted_sign=npass, best_abs_rho_predicted_sign=float(max(best)) if best else np.nan,
                                 bonf_threshold_p=0.05 / NFEAT[m], agg_rho_with_score=float(rho_agg)))
                aggs[m if tag == "fed" else f"{m}_corrected"] = agg
    names = list(aggs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[i].split("_")[0] == names[j].split("_")[0]: continue
            x, y = aggs[names[i]], aggs[names[j]]; has = ~np.isnan(x) & ~np.isnan(y); keep = keepmask(a, has)
            rho, p = ss.spearmanr(x[keep], y[keep])
            rows.append(dict(component=k, layer=f"{names[i]} vs {names[j]}", variant="cross-layer", n_donors=int(keep.sum()),
                             cross_rho=float(rho), cross_p=float(p)))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/03_crosslayer.csv", index=False)
pd.set_option("display.width", 250)
print(T.to_string(float_format=lambda v: f"{v:.3g}"))
ref = pd.read_csv(f"{R2}/verify_source_panels/results/07_circularity_null.csv")
ref = ref[(ref.eff_donors < 1.5) & (ref.kind == "cross")]
print(f"\nreference (single-donor components, dominant donor removed): cross-layer rho n={len(ref)}, "
      f"median {ref.rho_drop_ext.median():.3f}, 95% {ref.rho_drop_ext.quantile(.95):.3f}, max |rho| {ref.rho_drop_ext.abs().max():.3f}")
