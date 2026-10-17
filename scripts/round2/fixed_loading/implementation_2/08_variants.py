"""Method challenge: calibration and scenario-(d) results across weighting / diagonal / ridge choices, run-level,
for the runs given (catalogue members only). Aggregates only.  usage: 08_variants.py run [run ...]"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, os, numpy as np, pandas as pd
from vp_common import *
runs = [int(x) for x in sys.argv[1:]]
cat = np.load(f"{CACHE}/prot_cat.npy"); mis = np.isin(cat, [2, 3]); res = cat == 4
mem = pd.read_csv(f"{REL}/cluster_members.csv")
V = [("fit_plain_r1", "plain: residual weights, no 2nd-moment diagonal (= V0)"),
     ("fit_ss_r1", "spike-slab diagonal only (no slab variance)"),
     ("fit_full_r1", "analytic, A variance = diag(M^-1)"),
     ("fit_full_r1_none", "analytic, A variance = 0"),
     ("fit_full_r1_mf", "PRIMARY analytic, A variance = 1/diag(M)"),
     ("fit_full_r1_mf_emp", "primary + empirical NNLS per-layer diagonal (V3-like)"),
     ("fit_pooled_r1_mf", "RNA weight per gene pooled over cell types"),
     ("fit_rate_r1_mf", "noise prior read as Gamma rate 1e6"),
     ("fit_flat_r1_mf", "one weight per layer x cell type (homoscedastic)"),
     ("fit_full_r0_mf", "no ridge (prior on A dropped)"),
     ("fit_full_r10_mf", "ridge 10 (prior variance 0.1)")]


def cc(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    den = np.sqrt((P * P).sum(0) * (Q * Q).sum(0))
    return np.divide((P * Q).sum(0), den, out=np.full(P.shape[1], np.nan), where=den > 0)


rows = []; per = {}
for r in runs:
    A = read_txt(f"{SDA}/results{r}/it1500/A"); S4 = read_txt(f"{SDA}/results{r}/it1500/S4")
    m = mem[mem.run == r]; idx = m.sda_index.values; hl = m.component_id.isin(HL).values
    npp = (S4[idx] > 0.5).sum(1)
    Ap = A[:, idx]; sd = Ap.std(0)
    for f, desc in V:
        p = f"{CACHE}/run{r}/{f}.npz"
        if not os.path.exists(p): continue
        F = np.load(p); Aa = F["Aa"][:, idx]; Ad = F["Ad"][:, idx]; Ab = F["Ab"][:, idx]
        cr = cc(Aa, Ap); rel = np.linalg.norm(Aa - Ap, axis=0) / np.linalg.norm(Ap, axis=0)
        dr = cc(Ad, Aa); br = cc(Ab, Aa); dz = np.abs(Ad - Aa) / sd
        mz = dz[mis].mean(0)
        per[(r, f)] = pd.DataFrame(dict(component_id=m.component_id.values, sda_index=idx, npp=npp, d_r=dr, mz=mz))
        rows.append(dict(run=r, fit=f, desc=desc, cal_r_med=np.median(cr), cal_r_q01=np.quantile(cr, .01), cal_rel_med=np.median(rel),
                         cal_rel_q95=np.quantile(rel, .95), hl_cal_r_min=cr[hl].min(), hl_cal_rel_max=rel[hl].max(),
                         d_r_min=np.nanmin(dr), n_d_r_lt099=int((dr < .99).sum()), n_d_r_lt099_noprot=int(((dr < .99) & (npp == 0)).sum()),
                         b_r_min=np.nanmin(br), hl_d_r_min=dr[hl].min(),
                         hl_mis29_dz_med=np.median(mz[hl]), hl_mis29_dz_max=dz[mis][:, hl].max(),
                         hl_noprot_mis29_dz_med=np.median(mz[hl & (npp == 0)]),
                         C351prot_d_r=dr[(m.component_id.values == 351) & (npp > 0)].min() if ((m.component_id.values == 351) & (npp > 0)).any() else np.nan,
                         noprot_mis29_dz_med=np.median(mz[npp == 0]), prot_mis29_dz_med=np.median(mz[npp > 0])))
T = pd.DataFrame(rows)
T.to_csv(f"{RES}/variants_runlevel.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.drop(columns="desc").round(4).to_string(index=False))
# agreement of the most-affected members with the primary fit
for r in runs:
    if (r, "fit_full_r1_mf") not in per: continue
    P = per[(r, "fit_full_r1_mf")]
    worst = set(P.sort_values("d_r").head(10).sda_index)
    for f, _ in V:
        if (r, f) not in per or f == "fit_full_r1_mf": continue
        Q = per[(r, f)]
        rho = pd.Series(1 - P.d_r.values).corr(pd.Series(1 - Q.d_r.values), method="spearman")
        ov = len(worst & set(Q.sort_values("d_r").head(10).sda_index))
        print(f"run {r} {f:22s}: Spearman(1-d_r) vs primary {rho:.3f}; top-10 most-affected overlap {ov}/10; "
              f"top-10 have protein features: {int((Q.sort_values('d_r').head(10).npp > 0).sum())}/10")
