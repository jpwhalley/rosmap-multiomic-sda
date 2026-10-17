"""Alignment-free direction test for every cell-type label change in the corrected Data S1.
For component k and cell type c:
  P_c = B_pub[c] * outer(A_pub, X_pub)            (published consensus vectors)
  M_c = sum_m B_m[c] * outer(A_m, X1_m)            (each run-level member's own product)
  cos_c = <P_c, M_c> / (|P_c| |M_c|), via inner products of rank-1 terms (276 samples x 25098 genes).
Also re-derives the published B/X consensus and the corrected cell-type matrix from the archive."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
BASE = cfg.SDA
OUT = cfg.out_dir(cfg.AUD, "ds1verify_cosB")
NR = cfg.TABLES
CT = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]

mem = pd.read_csv(f"{OUT}/members.csv")
S = np.load(f"{OUT}/pooled_A.npy")
X1 = {i: np.load(f"{OUT}/X1_run{i}.npy", mmap_mode="r") for i in range(1, 11)}
B1 = {i: np.load(f"{OUT}/B1_run{i}.npy") for i in range(1, 11)}

A_pub = pd.read_csv(f"{BASE}/samples_final.csv", index_col=0)
B_pub = pd.read_csv(f"{BASE}/treatment_final.csv", index_col=0)
X_pub = pd.read_csv(f"{BASE}/genes_final_RNA.csv", index_col=0)
B_cor = pd.read_csv(f"{NR}/cell_type_loadings_corrected.csv", index_col=0)
assert list(B_pub.columns) == CT and list(B_cor.columns) == CT
assert X_pub.shape == (546, 25098) and A_pub.shape == (546, 276) and B_cor.shape == (546, 6)
genes = open(f"{BASE}/genes.txt").read().splitlines()
assert list(X_pub.columns) == genes
assert (A_pub.index == B_pub.index).all() and (X_pub.index == B_pub.index).all() and (B_cor.index == B_pub.index).all()

def flips(V):
    f = np.ones(len(V), int)
    for r in range(len(V)):
        if stats.pearsonr(V[0], V[r])[0] < 0: f[r] = -1
    return f

recs, chk = [], []
for k, g in mem.groupby("component", sort=True):
    g = g.sort_values("pos")                       # positional order = original row order
    Am = S[g.pos.values]                            # n x 276
    Bm = np.array([B1[r][:, s] for r, s in zip(g.run, g.slot)])   # n x 6
    Xm = np.array([X1[r][s] for r, s in zip(g.run, g.slot)])      # n x 25098
    fA = g.fA.values; fB = flips(Bm); fX = flips(Xm)
    assert (flips(Am) == fA).all()
    # reproduction checks
    bcons = (Bm * fB[:, None]).mean(0); xcons = (Xm * fX[:, None]).mean(0)
    bcorr = (Bm * (fA * fX)[:, None]).mean(0)
    chk.append(dict(component=k, n=len(g), dB_pub=np.abs(bcons - B_pub.loc[k].values).max(),
                    dX_pub=np.abs(xcons - X_pub.loc[k].values).max(),
                    dB_cor=np.abs(bcorr - B_cor.loc[k].values).max(),
                    n_fB_ne_fAfX=int((fB != fA * fX).sum())))
    a = A_pub.loc[k].values; x = X_pub.loc[k].values
    aA = Am @ a; xX = Xm @ x                         # n
    H = (Am @ Am.T) * (Xm @ Xm.T)                    # n x n
    na, nx = np.linalg.norm(a), np.linalg.norm(x)
    for ci, c in enumerate(CT):
        b = Bm[:, ci]; bp = B_pub.loc[k, c]
        num = bp * np.sum(b * aA * xX)
        nM = np.sqrt(max(b @ H @ b, 0.0)); nP = abs(bp) * na * nx
        cos = num / (nP * nM) if nP > 0 and nM > 0 else np.nan
        recs.append(dict(component=k, celltype=c, B_pub=bp, B_cor=B_cor.loc[k, c], cos=cos))
cosdf = pd.DataFrame(recs); chkdf = pd.DataFrame(chk)
cosdf.to_csv(f"{OUT}/cos_all.csv", index=False); chkdf.to_csv(f"{OUT}/reproduction_check.csv", index=False)
print("max |B consensus - treatment_final|:", chkdf.dB_pub.max())
print("max |X consensus - genes_final_RNA|:", chkdf.dX_pub.max())
print("max |B corrected - cell_type_loadings_corrected|:", chkdf.dB_cor.max())
print("components where fB != fA*fX1 for >=1 member:", int((chkdf.n_fB_ne_fAfX > 0).sum()))
