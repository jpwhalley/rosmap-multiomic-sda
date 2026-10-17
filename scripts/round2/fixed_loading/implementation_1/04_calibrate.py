"""Calibration on one run: re-estimate A with the inputs exactly as fed and compare with the published A.
Variants: V0 data Grams + unit ridge only; V2 + spike-slab second-moment term from the PIPs (slab variance ignored);
V1 + per-component, per-layer diagonal fitted by NNLS to the fixed-point residual (2-fold cross-fitted by donor,
stratified by layer pattern, so the reproduction is scored on donors not used to fit the diagonal).
Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, json, numpy as np, pandas as pd
from pi_common import *
from pi_solve import *

r = int(sys.argv[1])
WMODE = next((a.split("=")[1] for a in sys.argv[2:] if a.startswith("--wmode=")), "resid")
st, A, B1 = load_stats(r, WMODE)
print("weight mode:", WMODE)
H = {1: st["H1"], 2: st["H2"], 3: st["H3"], 4: st["H4"]}
has = {2: st["has2"], 3: st["has3"], 4: st["has4"]}
mem = pd.read_csv(f"{REL}/cluster_members.csv")
mr = mem[mem.run == r]
hl = mr[mr.component_id.isin(HIGHLIGHT)]
cat_idx = np.unique(mr.sda_index.values)
tlog(f"run{r}: catalogue members {len(cat_idx)}; highlighted members {len(hl)}")


def summarise(name, Ahat):
    rr, rel, sd = comp_metrics(Ahat, A)
    def q(x, idx=None):
        x = x if idx is None else x[idx]
        x = x[~np.isnan(x)]
        return [round(float(v), 4) for v in np.percentile(x, [1, 5, 25, 50, 75, 95])]
    row = dict(variant=name,
               r_q01_05_25_50_75_95_all=q(rr), rel_q_all=q(rel),
               r_q_catalogue=q(rr, cat_idx), rel_q_catalogue=q(rel, cat_idx),
               r_highlight_min=round(float(np.nanmin(rr[hl.sda_index.values])), 4),
               rel_highlight_max=round(float(np.nanmax(rel[hl.sda_index.values])), 4),
               frac_all_r_gt_0_99=round(float(np.nanmean(rr > 0.99)), 4),
               frac_cat_r_gt_0_99=round(float(np.nanmean(rr[cat_idx] > 0.99)), 4))
    print(json.dumps(row), flush=True)
    return rr, rel, sd


out = {}
A0 = solve(st, H, has); out["V0"] = summarise("V0 Gram+ridge", A0)
D2 = {1: st["Q1"], 2: st["Q2"], 3: st["Q3"], 4: st["Q4"]}
A2 = solve(st, H, has, D=D2); out["V2"] = summarise("V2 +PIP second-moment", A2)
E = fixed_point_residual(st, A, H, has)
# 2-fold cross-fit, stratified by pattern
rng = np.random.default_rng(1)
pat = has[2].astype(int) * 4 + has[3].astype(int) * 2 + has[4].astype(int)
fold = np.zeros(N, int)
for p in np.unique(pat):
    idx = np.where(pat == p)[0]; rng.shuffle(idx); fold[idx[::2]] = 1
A1cv = np.zeros((N, K))
for f in (0, 1):
    D = fit_diag(A, E, has, rows=np.where(fold != f)[0])
    A1cv[fold == f] = solve(st, H, has, D=D)[fold == f]
out["V1cv"] = summarise("V1 fitted diagonal, cross-fitted", A1cv)
D1 = fit_diag(A, E, has)
A1 = solve(st, H, has, D=D1); out["V1"] = summarise("V1 fitted diagonal, all donors", A1)
np.savez(f"{CACHE}/run{r}/calib_{WMODE}.npz", D1_1=D1[1], D1_2=D1[2], D1_3=D1[3], D1_4=D1[4], A0=A0, A2=A2, A1cv=A1cv, A1=A1)
# highlighted members table
rows = []
for _, m in hl.iterrows():
    k = int(m.sda_index)
    rows.append(dict(run=r, component_id=int(m.component_id), sda_index=k, sd_A=float(A[:, k].std()),
                     **{f"r_{v}": float(out[v][0][k]) for v in out}, **{f"rel_{v}": float(out[v][1][k]) for v in out},
                     D_RNA=float(D1[1][k]), D_meth=float(D1[2][k]), D_chip=float(D1[3][k]), D_prot=float(D1[4][k]),
                     Gdiag_RNA=float(st["G1"][k, k]), Gdiag_meth=float(st["G2"][k, k]), Gdiag_chip=float(st["G3"][k, k]), Gdiag_prot=float(st["G4"][k, k])))
T = pd.DataFrame(rows).sort_values(["component_id", "sda_index"])
pd.set_option("display.width", 250)
print(T[["component_id", "sda_index", "sd_A"] + [f"r_{v}" for v in out] + [f"rel_{v}" for v in out]].round(4).to_string(index=False))
print(T[["component_id", "sda_index", "D_RNA", "D_meth", "D_chip", "D_prot", "Gdiag_RNA", "Gdiag_meth", "Gdiag_chip", "Gdiag_prot"]].round(3).to_string(index=False))
T.to_csv(f"{RES}/calibration_run{r}_{WMODE}_highlighted_members.csv", index=False)
