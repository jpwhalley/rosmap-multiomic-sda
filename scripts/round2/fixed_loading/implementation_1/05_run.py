"""One run: calibration (inputs as fed) and the four protein scenarios with the fitted loadings held fixed.
Machinery: pi_solve.solve with per-feature residual precisions (stats.npz from 03_suffstats.py) and
  V3 (primary): + PIP second-moment diagonal (Q) + per-component, per-layer diagonal fitted by NNLS to the remaining
                published fixed-point residual, each donor's equation scaled by its pattern's diagonal precision
  V1 (sensitivity): + unweighted NNLS diagonal only (first calibration)
  V0 (sensitivity): Gram + unit ridge only.
Scenarios: a as fed | b protein removed for all | c protein removed for the 29 mis-paired donors |
           d corrected protein rows (105 donors with their own proteome; 28 wrongly-fed donors without protein).
Writes cache/run{r}/scen.npz (per-donor scores; on host only) and results/calibration_run{r}.json (aggregates)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, json, numpy as np, pandas as pd
from pi_common import *
from pi_solve import *

r = int(sys.argv[1])
WMODE = next((a.split("=")[1] for a in sys.argv[2:] if a.startswith("--wmode=")), "resid")
SUF = "" if WMODE == "resid" else f"_{WMODE}"
st, A, B1 = load_stats(r, WMODE)
grp = np.load(f"{CACHE}/prot_group.npy")
mis = np.isin(grp, [2, 3])
H = {1: st["H1"], 2: st["H2"], 3: st["H3"], 4: st["H4"]}
has = {2: st["has2"], 3: st["has3"], 4: st["has4"]}
assert np.array_equal(has[4], np.isin(grp, [1, 2, 3])) and np.array_equal(st["has4cor"], np.isin(grp, [1, 3, 4]))
mem = pd.read_csv(f"{REL}/cluster_members.csv"); mr = mem[mem.run == r]
cat_idx = np.unique(mr.sda_index.values); hl = mr[mr.component_id.isin(HIGHLIGHT)]

# ---------------- calibration
E = fixed_point_residual(st, A, H, has)
rng = np.random.default_rng(1)
pat = has[2].astype(int) * 4 + has[3].astype(int) * 2 + has[4].astype(int)
fold = np.zeros(N, int)
for p in np.unique(pat):
    idx = np.where(pat == p)[0]; rng.shuffle(idx); fold[idx[::2]] = 1
A1cv = np.zeros((N, K))
for f in (0, 1):
    Df = fit_diag(A, E, has, rows=np.where(fold != f)[0])
    A1cv[fold == f] = solve(st, H, has, D=Df)[fold == f]
D1 = fit_diag(A, E, has)
Q = {1: st["Q1"], 2: st["Q2"], 3: st["Q3"], 4: st["Q4"]}
EQ = E - (Q[1][None, :] + has[2][:, None] * Q[2][None, :] + has[3][:, None] * Q[3][None, :] + has[4][:, None] * Q[4][None, :]) * A
PdQ = pattern_diag(st, has, D=Q)
A3cv = np.zeros((N, K))
for f in (0, 1):
    Df = fit_diag_w(A, EQ, has, PdQ, rows=np.where(fold != f)[0], base=Q)
    A3cv[fold == f] = solve(st, H, has, D=Df)[fold == f]
D3 = fit_diag_w(A, EQ, has, PdQ, base=Q)
A3 = solve(st, H, has, D=D3)
A0 = solve(st, H, has)
A1 = solve(st, H, has, D=D1)
D2 = {1: st["Q1"], 2: st["Q2"], 3: st["Q3"], 4: st["Q4"]}
A2 = solve(st, H, has, D=D2)
cal = {}
for nm, Ah in [("V0", A0), ("V2", A2), ("V1cv", A1cv), ("V1", A1), ("V3cv", A3cv), ("V3", A3)]:
    rr, rel, sd = comp_metrics(Ah, A)
    cal[nm] = dict(r=rr, rel=rel)
summ = dict(run=r, n_catalogue_members=int(len(cat_idx)), n_highlight_members=int(len(hl)))
for nm in cal:
    for sub, idx in [("all", np.arange(K)), ("catalogue", cat_idx), ("highlight", hl.sda_index.values)]:
        x = cal[nm]["r"][idx]; y = cal[nm]["rel"][idx]
        summ[f"{nm}_{sub}_r_q01_05_50"] = [float(np.nanpercentile(x, q)) for q in (1, 5, 50)]
        summ[f"{nm}_{sub}_r_min"] = float(np.nanmin(x))
        summ[f"{nm}_{sub}_rel_q50_95_max"] = [float(np.nanpercentile(y, q)) for q in (50, 95)] + [float(np.nanmax(y))]
json.dump(summ, open(f"{RES}/calibration_run{r}{SUF}.json", "w"), indent=1)
tlog(f"run{r} [{WMODE}] calibration: catalogue V3cv r q01/q05/q50 {np.round(summ['V3cv_catalogue_r_q01_05_50'], 4)}, rel q50/q95/max {np.round(summ['V3cv_catalogue_rel_q50_95_max'], 3)}; "
     f"highlight V3cv r min {summ['V3cv_highlight_r_min']:.4f}, rel max {summ['V3cv_highlight_rel_q50_95_max'][2]:.3f}")

# ---------------- scenarios
hasb = dict(has); hasb[4] = np.zeros(N, bool)
hasc = dict(has); hasc[4] = has[4] & ~mis
Hd = dict(H); Hd[4] = st["H4cor"]; hasd = dict(has); hasd[4] = st["has4cor"]
out = {}
for vn, D in [("V3", D3), ("V1", D1), ("V0", None)]:
    out[f"{vn}_a"] = {"V3": A3, "V1": A1, "V0": A0}[vn]
    out[f"{vn}_b"] = solve(st, H, hasb, D=D)
    out[f"{vn}_c"] = solve(st, H, hasc, D=D)
    Dd = None if D is None else dict(D)
    if vn == "V3": Dd[4] = D[4] - Q[4] + st["Q4cor"]      # PIP term over the protein features present in the corrected build
    out[f"{vn}_d"] = solve(st, Hd, hasd, D=Dd, G4key="G4cor")
np.savez(f"{CACHE}/run{r}/scen{SUF}.npz", **{k: v.astype(np.float32) for k, v in out.items()},
         **{f"cal_r_{v}": cal[v]["r"] for v in cal}, **{f"cal_rel_{v}": cal[v]["rel"] for v in cal},
         D1_1=D1[1], D1_2=D1[2], D1_3=D1[3], D1_4=D1[4], D3_1=D3[1], D3_2=D3[2], D3_3=D3[3], D3_4=D3[4],
         A1cv=A1cv.astype(np.float32), A3cv=A3cv.astype(np.float32))
tlog(f"run{r} scenarios saved")
