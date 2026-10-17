"""Protein share of the score precision and of the dominant donor's score, highlighted members (all runs), primary fit.
precision share = (G4_kk + D4_k) / (ridge + sum over the donor's layers of (G_kk + D_k)) for the dominant donor's pattern."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from vp_common import *
fn = "fit_full_r1_mf"
M = pd.read_csv(f"{RES}/members_{fn}.csv")
Y4 = np.load(f"{CACHE}/in_prot.npy"); Y2 = np.load(f"{CACHE}/in_meth.npy"); Y3 = np.load(f"{CACHE}/in_chip.npy")
has = {2: ~np.isnan(Y2).all(1), 3: ~np.isnan(Y3).all(1), 4: ~np.isnan(Y4).all(1)}
out = []
for r in sorted(M.run.unique()):
    F = np.load(f"{CACHE}/run{r}/{fn}.npz"); A = read_txt(f"{SDA}/results{r}/it1500/A")
    for _, t in M[(M.run == r) & M.component_id.isin(HL)].iterrows():
        k = int(t.sda_index); n = int(np.abs(A[:, k]).argmax())
        tot = 1 + F["G1diag"][k] + F["D1"][k] + sum((F[f"G{d}diag"][k] + F[f"D{d}"][k]) for d in (2, 3, 4) if has[d][n])
        out.append(dict(component_id=t.component_id, run=r, sda_index=k, n_prot_pip05=t.n_prot_pip05, dom_cat=t.dom_cat,
                        dom_has_prot=bool(has[4][n]), prot_prec_share=(F["G4diag"][k] + F["D4"][k]) / tot if has[4][n] else 0.0,
                        dom_contrib_RNA=t.dom_contrib_L1, dom_contrib_prot=t.dom_contrib_L4))
T = pd.DataFrame(out); T.to_csv(f"{RES}/highlighted_layer_share_{fn}.csv", index=False)
for k in HL:
    s = T[T.component_id == k]
    if len(s) == 0: continue
    sp = s[s.n_prot_pip05 > 0]
    print(f"C{k}: members {len(s)} (with protein features {len(sp)}); dominant donor has protein in {int(s.dom_has_prot.sum())}; "
          f"protein precision share median {s.prot_prec_share.median():.4f} max {s.prot_prec_share.max():.4f}; "
          f"dominant-donor score from RNA median {s.dom_contrib_RNA.median():.3f}, from protein median {s.dom_contrib_prot.median():.3f} max {s.dom_contrib_prot.abs().max():.3f}"
          + (f" | protein-carrying members: prec share {sp.prot_prec_share.round(3).tolist()} RNA {sp.dom_contrib_RNA.round(3).tolist()} prot {sp.dom_contrib_prot.round(3).tolist()}" if len(sp) else ""))
