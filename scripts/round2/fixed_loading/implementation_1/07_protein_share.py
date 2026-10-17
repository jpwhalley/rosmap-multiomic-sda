"""Why the scores move (or not): for each highlighted run-level member, (i) the share of the score precision
that the protein layer supplies (diagonal of the Grams, + fitted diagonal), and (ii) the additive decomposition of the
baseline (V3, as fed) score into layer contributions, a_n = P_n^-1 (h_RNA + h_meth + h_chip + h_prot),
reported for the dominant donor and averaged over the 29 mis-paired donors. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy.linalg import cho_factor, cho_solve
from pi_common import *
from pi_solve import load_stats

grp = np.load(f"{CACHE}/prot_group.npy"); mis = np.isin(grp, [2, 3])
mem = pd.read_csv(f"{REL}/cluster_members.csv")
rows = []
for r in range(1, 11):
    st, A, B1 = load_stats(r)
    z = np.load(f"{CACHE}/run{r}/scen.npz")
    D = {j: z[f"D3_{j}"] for j in (1, 2, 3, 4)}
    a = z["V3_a"].astype(float)
    has = {2: st["has2"], 3: st["has3"], 4: st["has4"]}
    H = {1: st["H1"], 2: st["H2"], 3: st["H3"], 4: st["H4"]}
    contrib = {j: np.zeros((N, K)) for j in (1, 2, 3, 4)}
    pats = {}
    for n in range(N):
        pats.setdefault((bool(has[2][n]), bool(has[3][n]), bool(has[4][n])), []).append(n)
    for (m2, m3, m4), idx in pats.items():
        P = st["G1"].copy(); d = 1.0 + D[1]
        for j, mj in ((2, m2), (3, m3), (4, m4)):
            if mj: P += st[f"G{j}"]; d = d + D[j]
        P[np.diag_indices(K)] += d
        cf = cho_factor(P, check_finite=False)
        for j, mj in ((1, True), (2, m2), (3, m3), (4, m4)):
            if mj: contrib[j][idx] = cho_solve(cf, H[j][idx].T, check_finite=False).T
    tot = sum(contrib.values())
    assert np.allclose(tot, a, atol=1e-6 * np.abs(a).max() + 1e-9)
    hl = mem[(mem.run == r) & mem.component_id.isin(HIGHLIGHT)]
    for _, m in hl.iterrows():
        k = int(m.sda_index); ak = a[:, k]; sd = ak.std()
        med = np.median(ak); mad = np.median(np.abs(ak - med)) * 1.4826; i0 = int(np.argmax(np.abs((ak - med) / mad)))
        tot_diag = 1 + st["G1"][k, k] + st["G2"][k, k] + st["G3"][k, k] + st["G4"][k, k]
        rows.append(dict(component_id=int(m.component_id), run=r, sda_index=k,
                         prot_share_precision_G=float(st["G4"][k, k] / tot_diag),
                         prot_share_precision_GD=float((st["G4"][k, k] + D[4][k]) / (tot_diag + D[1][k] + D[2][k] + D[3][k] + D[4][k])),
                         dom_cat=int(grp[i0]),
                         dom_frac_from_RNA=float(contrib[1][i0, k] / ak[i0]), dom_frac_from_meth=float(contrib[2][i0, k] / ak[i0]),
                         dom_frac_from_chip=float(contrib[3][i0, k] / ak[i0]), dom_frac_from_prot=float(contrib[4][i0, k] / ak[i0]),
                         mis29_mean_abs_prot_contrib_over_sd=float(np.mean(np.abs(contrib[4][mis, k])) / sd),
                         correct76_mean_abs_prot_contrib_over_sd=float(np.mean(np.abs(contrib[4][grp == 1, k])) / sd)))
    tlog(f"run {r} done")
T = pd.DataFrame(rows).sort_values(["component_id", "run", "sda_index"])
T.to_csv(f"{RES}/highlighted_members_layer_contributions.csv", index=False)
pd.set_option("display.width", 250)
g = T.groupby("component_id").agg(n=("run", "size"), prot_share_G_med=("prot_share_precision_G", "median"), prot_share_G_max=("prot_share_precision_G", "max"),
                                  prot_share_GD_med=("prot_share_precision_GD", "median"),
                                  dom_frac_prot_med=("dom_frac_from_prot", "median"), dom_frac_prot_maxabs=("dom_frac_from_prot", lambda x: x.abs().max()),
                                  dom_frac_RNA_med=("dom_frac_from_RNA", "median"),
                                  mis29_prot_contrib_med=("mis29_mean_abs_prot_contrib_over_sd", "median"),
                                  correct76_prot_contrib_med=("correct76_mean_abs_prot_contrib_over_sd", "median"),
                                  dom_cats=("dom_cat", lambda x: sorted(set(x))))
print(g.round(4).to_string())
