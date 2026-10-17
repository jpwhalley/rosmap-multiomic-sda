"""Calibration variants for the fitted diagonal: V1 (NNLS, unweighted), V1w (NNLS with each donor's equation scaled by
its pattern's diagonal precision, i.e. errors in score units), V2V1w (PIP second-moment term + V1w on the remainder).
2-fold cross-fitted by donor. Aggregates only. Usage: 04c_calib_weighted.py r [k_extra ...]"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, numpy as np, pandas as pd
from pi_common import *
from pi_solve import *

r = int(sys.argv[1]); extra = [int(x) for x in sys.argv[2:]]
st, A, B1 = load_stats(r)
H = {1: st["H1"], 2: st["H2"], 3: st["H3"], 4: st["H4"]}; has = {2: st["has2"], 3: st["has3"], 4: st["has4"]}
mem = pd.read_csv(f"{REL}/cluster_members.csv"); mr = mem[mem.run == r]
cat_idx = np.unique(mr.sda_index.values); hl = mr[mr.component_id.isin(HIGHLIGHT)].sda_index.values
E = fixed_point_residual(st, A, H, has)
Q = {1: st["Q1"], 2: st["Q2"], 3: st["Q3"], 4: st["Q4"]}
EQ = E - (Q[1][None, :] + has[2][:, None] * Q[2][None, :] + has[3][:, None] * Q[3][None, :] + has[4][:, None] * Q[4][None, :]) * A
Pd = pattern_diag(st, has)
PdQ = pattern_diag(st, has, D=Q)
rng = np.random.default_rng(1)
pat = has[2].astype(int) * 4 + has[3].astype(int) * 2 + has[4].astype(int)
fold = np.zeros(N, int)
for p in np.unique(pat):
    idx = np.where(pat == p)[0]; rng.shuffle(idx); fold[idx[::2]] = 1
fits = {"V1": lambda rows: fit_diag(A, E, has, rows),
        "V1w": lambda rows: fit_diag_w(A, E, has, Pd, rows),
        "V2V1w": lambda rows: fit_diag_w(A, EQ, has, PdQ, rows, base=Q)}
for nm, fn in fits.items():
    Acv = np.zeros((N, K))
    for f in (0, 1):
        Acv[fold == f] = solve(st, H, has, D=fn(np.where(fold != f)[0]))[fold == f]
    rr, rel, _ = comp_metrics(Acv, A)
    msg = (f"run{r} {nm:6s} CV: catalogue r q01/q05/q50 {np.round(np.nanpercentile(rr[cat_idx], [1, 5, 50]), 4)} min {np.nanmin(rr[cat_idx]):.3f}; "
           f"rel q50/q95 {np.round(np.nanpercentile(rel[cat_idx], [50, 95]), 3)}; highlight r min {np.nanmin(rr[hl]):.4f} rel max {np.nanmax(rel[hl]):.3f}; "
           f"all r q05 {np.nanpercentile(rr, 5):.4f}")
    msg += "".join(f"; k{k}: r {rr[k]:.3f} rel {rel[k]:.3f}" for k in extra)
    print(msg, flush=True)
