"""Point 1: what the fixed-loading re-estimation r values compare, per implementation, for the retained components.
Recomputes from the two implementations' per-run caches (read-only):
  impl 1 = $G_ROS_RESULTS_DIR/protein_influence/cache/run*/scen.npz (V3 = scenario machinery; A3cv = cross-fitted as-fed calibration)
  impl 2 = $G_ROS_RESULTS_DIR/verify_protein_influence/cache/run*/fit_full_r1_mf.npz (Aa..Ad)
Consensus = mean over release members of flip_samples * member scores (as both pipelines).
For each component: r(as-fed re-estimated, published) [baseline reproduction]; r(scenario, as-fed re-estimated);
r(scenario re-estimated, published); r(published + (scenario - as-fed), published).
Also checks that Data S1 score_r_corrected_protein equals impl-1 r(V3_d, V3_a) for all 547 rows.
Aggregates only: no donor IDs, no per-donor values written."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = cfg.THREADS
import numpy as np, pandas as pd

RA = cfg.WORK
REL = cfg.CATALOGUE   # outputs of scripts/07_cluster_components.py
DS1 = cfg.DS1_V23
OUT = cfg.out_dir(RA, "v23_review", "points_1_3_6", "out")
RET = [74, 278, 172, 277, 21, 545]
HL = RET + [351, 2, 87, 177]

mem = pd.read_csv(f"{REL}/cluster_members.csv")
pub = pd.read_csv(f"{REL}/samples_final.csv", index_col=0)
pub = pub.loc[pub.index != 520]
IDS = [int(i) for i in pub.index]
row = {k: i for i, k in enumerate(IDS)}
P = pub.values.astype(np.float64)
N = P.shape[1]


def build(loader, keys):
    C = {k: np.zeros((547, N)) for k in keys}; cnt = np.zeros(547)
    for r in range(1, 11):
        Z = loader(r)
        m = mem[mem.run == r]
        for k, j, f in zip(m.component_id.values, m.sda_index.values, m.flip_samples.values):
            i = row[int(k)]; cnt[i] += 1
            for kk in keys:
                C[kk][i] += f * Z[kk][:, j].astype(np.float64)
    for kk in keys:
        C[kk] /= cnt[:, None]
    return C


def cc(X, Y):
    X = X - X.mean(1, keepdims=True); Y = Y - Y.mean(1, keepdims=True)
    return (X * Y).sum(1) / np.sqrt((X * X).sum(1) * (Y * Y).sum(1))


def l1(r):
    z = np.load(f"{cfg.WORK}/protein_influence/cache/run{r}/scen.npz")
    return {"a": z["V3_a"], "b": z["V3_b"], "c": z["V3_c"], "d": z["V3_d"], "acv": z["A3cv"]}


def l2(r):
    z = np.load(f"{cfg.WORK}/verify_protein_influence/cache/run{r}/fit_full_r1_mf.npz")
    return {"a": z["Aa"], "b": z["Ab"], "c": z["Ac"], "d": z["Ad"]}


C1 = build(l1, ["a", "b", "c", "d", "acv"])
C2 = build(l2, ["a", "b", "c", "d"])

rows = []
for impl, C in [("impl1", C1), ("impl2", C2)]:
    T = pd.DataFrame(index=IDS); T.index.name = "component_id"
    T["asfed_vs_pub"] = cc(C["a"], P)
    if "acv" in C:
        T["asfedcv_vs_pub"] = cc(C["acv"], P)
    for s in "bcd":
        T[f"{s}_vs_asfed"] = cc(C[s], C["a"])
        T[f"{s}_vs_pub"] = cc(C[s], P)
        T[f"pubplus{s}_vs_pub"] = cc(P + C[s] - C["a"], P)
    T.columns = [f"{impl}_{c}" for c in T.columns]
    rows.append(T)
T = pd.concat(rows, axis=1)
T.to_csv(f"{OUT}/p1_reproduction_547.csv", float_format="%.6f")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("== retained and example components (Pearson r over 276 donors; consensus level)")
for impl in ("impl1", "impl2"):
    cols = [c for c in T.columns if c.startswith(impl)]
    print(f"\n-- {impl}")
    print(T.loc[HL, cols].rename(columns=lambda c: c[len(impl) + 1:]).round(5).to_string())

print("\n== min over the six retained components")
for c in T.columns:
    print(f"  {c:28s} min {T.loc[RET, c].min():.5f} (C{T.loc[RET, c].idxmin()})")

# cross-check against the archived tables
a1 = pd.read_csv(f"{RA}/protein_influence/results/consensus_547_scenarios.csv", index_col=0).loc[IDS]
a2 = pd.read_csv(f"{RA}/verify_protein_influence/results/consensus_fit_full_r1_mf.csv", index_col=0).loc[IDS]
print("\n== cross-checks against archived tables (max |diff| over 547)")
print("  impl1 V3_d_r       vs recomputed d_vs_asfed :", float((a1.V3_d_r - T.impl1_d_vs_asfed).abs().max()))
print("  impl1 cal_r_V3     vs recomputed asfed_vs_pub:", float((a1.cal_r_V3 - T.impl1_asfed_vs_pub).abs().max()))
print("  impl1 cal_r_V3cv   vs recomputed asfedcv     :", float((a1.cal_r_V3cv - T.impl1_asfedcv_vs_pub).abs().max()))
print("  impl2 cal_r        vs recomputed asfed_vs_pub:", float((a2.cal_r - T.impl2_asfed_vs_pub).abs().max()))
print("  impl2 d_r          vs recomputed d_vs_asfed  :", float((a2.d_r - T.impl2_d_vs_asfed).abs().max()))
print("  impl2 d_r_pub      vs recomputed pubplusd    :", float((a2.d_r_pub - T.impl2_pubplusd_vs_pub).abs().max()))

# Data S1 column
ds = pd.read_csv(DS1)
ds["cid"] = ds.Component.astype(str).str.replace("C", "", regex=False).astype(int)
ds = ds.set_index("cid").loc[IDS]
col = ds.score_r_corrected_protein.astype(float)
print("\n== Data S1 score_r_corrected_protein (547 rows; non-missing", int(col.notna().sum()), ")")
for c in ["impl1_d_vs_asfed", "impl1_d_vs_pub", "impl1_pubplusd_vs_pub", "impl2_d_vs_asfed", "impl2_d_vs_pub", "impl2_pubplusd_vs_pub"]:
    d = (col - T[c]).abs()
    print(f"  vs {c:24s}: max |diff| {d.max():.3e}; rows with |diff| > 1e-6: {int((d > 1e-6).sum())}; > 1e-3: {int((d > 1e-3).sum())}")
print("  Data S1 value for retained:", {k: col[k] for k in RET})
print("  Data S1 summary: min {:.4f} (C{}), n < 0.99 {}, n < 0.95 {}".format(col.min(), col.idxmin(), int((col < 0.99).sum()), int((col < 0.95).sum())))
for c in ["impl1_d_vs_pub", "impl1_pubplusd_vs_pub"]:
    x = T[c]
    print(f"  if replaced by {c}: min {x.min():.4f} (C{x.idxmin()}), n < 0.99 {int((x < 0.99).sum())}, n < 0.95 {int((x < 0.95).sum())}; retained min {x.loc[RET].min():.5f}")
