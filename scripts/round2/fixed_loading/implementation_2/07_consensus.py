"""Consensus-level (547 catalogue components) scenario summary from the per-run fixed-loadings fits.
consensus = mean over members of flip_samples * member scores (release cluster_members.csv); checked against the
release samples_final.csv. Scenario effect added to the published consensus: cons_pub + mean(flip * (Ahat_s - Ahat_a)).
Clinical tests as the 547-family table (sex MWU, CERAD binarised MWU, cogdx KW over 1/2/4; BH per variable).
usage: 07_consensus.py fitname  -> results/consensus_{fitname}.csv, results/clinical_{fitname}.csv (aggregates only)"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, numpy as np, pandas as pd
from scipy import stats as ss
from vp_common import *
fn = sys.argv[1]
cat = np.load(f"{CACHE}/prot_cat.npy")
grp = {"mis29": np.isin(cat, [2, 3]), "restored28": cat == 4, "correct76": cat == 1, "noprot143": cat == 0}
mem = pd.read_csv(f"{REL}/cluster_members.csv")
samps = [int(x) for x in open(f"{SDA}/md_samples.txt").read().split()]
pub = pd.read_csv(f"{REL}/samples_final.csv", index_col=0)
assert [int(c) for c in pub.columns] == samps
pipP = pd.read_csv(f"{REL}/pip_final_proteins.csv", index_col=0)
nprot = (pipP.values > 0.5).sum(1); nprot = dict(zip(pipP.index.astype(int), nprot))
IDS = [int(i) for i in pub.index]                       # 1..548 without 520
assert len(IDS) == 547 and sorted(mem.component_id.unique()) == sorted(IDS)
row = {k: i for i, k in enumerate(IDS)}

Cp = np.zeros((547, N)); Cs = {s: np.zeros((547, N)) for s in "abcd"}; Dl = {s: np.zeros((547, N)) for s in "bcd"}
cnt = np.zeros(547)
for r in range(1, 11):
    F = np.load(f"{CACHE}/run{r}/{fn}.npz"); A = read_txt(f"{SDA}/results{r}/it1500/A")
    m = mem[mem.run == r]
    for k, j, f in zip(m.component_id.values, m.sda_index.values, m.flip_samples.values):
        i = row[int(k)]
        Cp[i] += f * A[:, j]; cnt[i] += 1
        for s in "abcd": Cs[s][i] += f * F[f"A{s}"][:, j]
        for s in "bcd": Dl[s][i] += f * (F[f"A{s}"][:, j] - F["Aa"][:, j])
Cp /= cnt[:, None]
for s in Cs: Cs[s] /= cnt[:, None]
for s in Dl: Dl[s] /= cnt[:, None]
log("consensus rebuild vs release samples_final: max |diff|", float(np.abs(Cp - pub.values).max()))


def cc(P, Q):   # row-wise correlation
    P = P - P.mean(1, keepdims=True); Q = Q - Q.mean(1, keepdims=True)
    return (P * Q).sum(1) / np.sqrt((P * P).sum(1) * (Q * Q).sum(1))


def robz(a):
    med = np.median(a); mad = np.median(np.abs(a - med)) * 1.4826
    return np.abs(a - med) / mad if mad > 0 else np.full_like(a, np.nan)


T = pd.DataFrame(dict(component_id=IDS, n_members=cnt.astype(int), n_prot_pip05=[nprot[k] for k in IDS]))
T["cal_r"] = cc(Cs["a"], Cp); T["cal_rel"] = np.linalg.norm(Cs["a"] - Cp, axis=1) / np.linalg.norm(Cp, axis=1)
sd = Cp.std(1)
dom = np.abs(Cp).argmax(1); T["dom_cat"] = cat[dom]
T["dom_share_base"] = (Cp ** 2).max(1) / (Cp ** 2).sum(1)
T["dom_robz_base"] = [robz(Cp[i])[dom[i]] for i in range(547)]
for s in "bcd":
    Pn = Cp + Dl[s]
    T[f"{s}_r"] = cc(Cs[s], Cs["a"]); T[f"{s}_r_pub"] = cc(Pn, Cp)
    dz = np.abs(Dl[s]) / sd[:, None]
    for g, mk in grp.items():
        T[f"{s}_mean_dz_{g}"] = dz[:, mk].mean(1); T[f"{s}_max_dz_{g}"] = dz[:, mk].max(1)
    T[f"{s}_dom_same"] = np.abs(Pn).argmax(1) == dom
    T[f"{s}_dom_share"] = (Pn ** 2).max(1) / (Pn ** 2).sum(1)
    T[f"{s}_dom_robz"] = [robz(Pn[i])[dom[i]] for i in range(547)]
T.to_csv(f"{RES}/consensus_{fn}.csv", index=False)

# ---- clinical ----
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv"); clin.index = clin.projid.astype(int)
clin = clin.loc[samps]
CER = {1: "AD", 2: "AD", 3: "No", 4: "No"}
msex = clin.msex.values; cb = clin.ceradsc.map(CER).values; cg = clin.cogdx.values


def tests(v):
    return (ss.mannwhitneyu(v[msex == 1], v[msex == 0], alternative="two-sided").pvalue,
            ss.mannwhitneyu(v[cb == "AD"], v[cb == "No"], alternative="two-sided").pvalue,
            ss.kruskal(*[v[cg == k] for k in (1, 2, 4)]).pvalue)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o]
    adj = np.minimum.accumulate((r * n / np.arange(1, n + 1))[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(adj, 1); return out


cols = ["Sex", "ceradsc", "cogdx"]
ref = pd.read_csv(f"{cfg.AUD}/c547/clinical_bh_547.csv", index_col=0)
out = {}
for nm, M in [("pub", Cp)] + [(f"pub+{s}", Cp + Dl[s]) for s in "bcd"] + [(f"re_{s}", Cs[s]) for s in "abcd"]:
    raw = np.array([tests(M[i]) for i in range(547)])
    q = np.column_stack([bh(raw[:, j]) for j in range(3)])
    for j, c in enumerate(cols):
        out[f"{nm}_p_{c}"] = raw[:, j]; out[f"{nm}_q_{c}"] = q[:, j]
Q = pd.DataFrame(out, index=IDS); Q.index.name = "component_id"
Q.to_csv(f"{RES}/clinical_{fn}.csv")
log("reproduction of clinical_bh_547.csv: max |q diff|", float(np.abs(Q[[f"pub_q_{c}" for c in cols]].values - ref.loc[IDS, cols].values).max()))
for nm in ["pub+b", "pub+c", "pub+d", "re_a", "re_d"]:
    for c in cols:
        s0 = set(Q.index[Q[f"pub_q_{c}"] < 0.05]); s1 = set(Q.index[Q[f"{nm}_q_{c}"] < 0.05])
        lost = sorted(s0 - s1); gained = sorted(s1 - s0)
        det = "; ".join(f"C{k} q {Q.loc[k, f'pub_q_{c}']:.3g}->{Q.loc[k, f'{nm}_q_{c}']:.3g}" for k in lost + gained)
        print(f"  {nm:6s} {c:7s}: published calls {len(s0)}, lost {len(lost)}, gained {len(gained)} {det}")

# ---- summaries ----
for s in "bcd":
    r = T[f"{s}_r"]
    print(f"scenario {s}: r vs re-estimated baseline min {r.min():.4f} (C{int(T.component_id[r.idxmin()])}) 1st {r.quantile(.01):.4f} 5th {r.quantile(.05):.4f} "
          f"median {r.median():.5f}; n<0.99 {int((r < .99).sum())} n<0.95 {int((r < .95).sum())} n<0.90 {int((r < .9).sum())} | "
          f"r_pub min {T[f'{s}_r_pub'].min():.4f}")
print("components with protein PIP>0.5 features:", int((T.n_prot_pip05 > 0).sum()), "; median d_r with / without",
      round(T.d_r[T.n_prot_pip05 > 0].median(), 5), round(T.d_r[T.n_prot_pip05 == 0].median(), 5))
print("calibration (consensus): r median", round(T.cal_r.median(), 5), "1st", round(T.cal_r.quantile(.01), 4), "min", round(T.cal_r.min(), 4),
      "| rel median", round(T.cal_rel.median(), 3), "95th", round(T.cal_rel.quantile(.95), 3), "| comps cal_r<0.95:", T.component_id[T.cal_r < .95].tolist())
w = T.sort_values("d_r").head(12)
print(w[["component_id", "n_members", "n_prot_pip05", "cal_r", "d_r", "b_r", "c_r", "d_mean_dz_mis29", "d_max_dz_mis29", "d_mean_dz_restored28",
         "d_max_dz_restored28", "dom_cat", "dom_share_base", "d_dom_share", "d_dom_same"]].round(4).to_string(index=False))
sdm = T.dom_share_base > 0.5
print(f"single-donor comps (share>0.5): {int(sdm.sum())}; keep dominant donor in all scenarios: {int(T.loc[sdm, ['b_dom_same','c_dom_same','d_dom_same']].all(1).sum())}")
drop = (T.dom_share_base - T[["b_dom_share", "c_dom_share", "d_dom_share"]].min(1))[sdm]
i = drop.idxmax(); print(f"  largest share drop C{int(T.component_id[i])}: {T.dom_share_base[i]:.3f} -> {T.loc[i, ['b_dom_share','c_dom_share','d_dom_share']].min():.3f}")
mp = sdm & T.dom_cat.isin([2, 3])
print(f"  with mis-paired dominant donor: {int(mp.sum())} (cat2 {int((sdm & (T.dom_cat == 2)).sum())}, cat3 {int((sdm & (T.dom_cat == 3)).sum())}); "
      f"of those with protein features: {[(int(a), int(b)) for a, b in T.loc[mp & (T.n_prot_pip05 > 0), ['component_id', 'n_prot_pip05']].values]}")
print("  their d_r:", T.loc[mp & (T.n_prot_pip05 > 0), ["component_id", "d_r", "d_mean_dz_mis29", "d_max_dz_mis29"]].round(4).values.tolist())
print("highlighted:")
for k in HL:
    t = T[T.component_id == k].iloc[0]
    ph = "; ".join(f"{c} p {Q.loc[k, f'pub_p_{c}']:.3g}->{Q.loc[k, f'pub+d_p_{c}']:.3g} q {Q.loc[k, f'pub_q_{c}']:.3g}->{Q.loc[k, f'pub+d_q_{c}']:.3g}"
                   for c in cols if Q.loc[k, f"pub_q_{c}"] < 0.05)
    print(f"  C{k}: n {t.n_members} prot {t.n_prot_pip05} cal_r {t.cal_r:.4f} | d_r {t.d_r:.5f} b_r {t.b_r:.5f} c_r {t.c_r:.5f} | mis29 {t.d_mean_dz_mis29:.4f} (max {t.d_max_dz_mis29:.3f}) "
          f"restored {t.d_mean_dz_restored28:.4f} (max {t.d_max_dz_restored28:.3f}) | dom cat {t.dom_cat} share {t.dom_share_base:.3f}->{t.d_dom_share:.3f} kept {t.d_dom_same} robz {t.dom_robz_base:.0f}->{t.d_dom_robz:.0f} | {ph}")
