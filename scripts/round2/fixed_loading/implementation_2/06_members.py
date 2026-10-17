"""Run-level scenario metrics for every catalogue member of the given runs (aggregates per member; no donor IDs).
usage: 06_members.py fitname run [run ...]  -> results/members_{fitname}.csv (appends runs)"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, os, numpy as np, pandas as pd
from vp_common import *
fn = sys.argv[1]; runs = [int(x) for x in sys.argv[2:]]
cat = np.load(f"{CACHE}/prot_cat.npy")
grp = {"mis29": np.isin(cat, [2, 3]), "restored28": cat == 4, "correct76": cat == 1, "noprot143": cat == 0}
mem = pd.read_csv(f"{REL}/cluster_members.csv")


def cc(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    den = np.sqrt((P * P).sum(0) * (Q * Q).sum(0))
    return np.divide((P * Q).sum(0), den, out=np.full(P.shape[1], np.nan), where=den > 0)


rows = []
for r in runs:
    F = np.load(f"{CACHE}/run{r}/{fn}.npz")
    A = read_txt(f"{SDA}/results{r}/it1500/A")
    S4 = read_txt(f"{SDA}/results{r}/it1500/S4")
    m = mem[mem.run == r]
    idx = m.sda_index.values
    Ap = A[:, idx]; Aa = F["Aa"][:, idx]
    sd = Ap.std(0)
    out = pd.DataFrame(dict(component_id=m.component_id.values, run=r, sda_index=idx, flip=m.flip_samples.values,
                            n_prot_pip05=(S4[idx] > 0.5).sum(1),
                            cal_r=cc(Aa, Ap), cal_rel=np.linalg.norm(Aa - Ap, axis=0) / np.linalg.norm(Ap, axis=0)))
    dom = np.abs(Ap).argmax(0)
    out["dom_cat"] = cat[dom]
    out["dom_share_base"] = (Ap ** 2).max(0) / (Ap ** 2).sum(0)
    for d in (1, 2, 3, 4):
        Cd = F[f"Ca{d}"][:, idx]
        out[f"dom_contrib_L{d}"] = Cd[dom, np.arange(len(idx))] / Aa[dom, np.arange(len(idx))]
    for s in "bcd":
        As = F[f"A{s}"][:, idx]; Dl = As - Aa; Apn = Ap + Dl
        out[f"{s}_r"] = cc(As, Aa)
        out[f"{s}_r_pub"] = cc(Apn, Ap)
        dz = np.abs(Dl) / sd
        for g, mk in grp.items():
            out[f"{s}_mean_dz_{g}"] = dz[mk].mean(0); out[f"{s}_max_dz_{g}"] = dz[mk].max(0)
        out[f"{s}_dom_same"] = np.abs(Apn).argmax(0) == dom
        out[f"{s}_dom_share"] = (Apn ** 2).max(0) / (Apn ** 2).sum(0)
    rows.append(out)
    print(f"run {r}: members {len(out)}; cal r median {out.cal_r.median():.4f}; d_r min {out.d_r.min():.4f} 1st {out.d_r.quantile(.01):.4f}; "
          f"members with d_r<0.99: {int((out.d_r < 0.99).sum())}", flush=True)
R = pd.concat(rows)
f = f"{RES}/members_{fn}.csv"
if os.path.exists(f):
    old = pd.read_csv(f); R = pd.concat([old[~old.run.isin(runs)], R])
R.sort_values(["run", "component_id", "sda_index"]).to_csv(f, index=False)
H = R[R.component_id.isin(HL)]
for k in HL:
    h = H[H.component_id == k]
    if len(h) == 0: continue
    print(f"C{k}: n {len(h)} runs {sorted(h.run.unique().tolist())} | cal r min {h.cal_r.min():.4f} rel max {h.cal_rel.max():.3f} | "
          f"r_b min {h.b_r.min():.4f} r_c min {h.c_r.min():.4f} r_d min {h.d_r.min():.4f} med {h.d_r.median():.4f} | "
          f"mis29 dz med (d) {h.d_mean_dz_mis29.median():.3f} max {h.d_max_dz_mis29.max():.3f} | restored dz med {h.d_mean_dz_restored28.median():.3f} | "
          f"dom cat {sorted(h.dom_cat.unique().tolist())} share_base min {h.dom_share_base.min():.3f} | dom kept all {bool(h[['b_dom_same','c_dom_same','d_dom_same']].all().all())} | "
          f"prot contrib at dom max {h.dom_contrib_L4.abs().max():.3f} RNA min {h.dom_contrib_L1.min():.3f} | n_prot_pip05 {h.n_prot_pip05.min()}-{h.n_prot_pip05.max()}")
