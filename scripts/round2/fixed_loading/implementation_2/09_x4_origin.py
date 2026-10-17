"""Whose proteome do the protein loadings of the protein-carrying components reflect?
For each catalogue component with protein PIP>0.5 features (release consensus genes_final_proteins / pip_final_proteins):
correlate its protein loadings (PIP>0.5 features) with every fed protein row (prot.txt, centred per feature over the 105
fed donors) and with the corrected rows; report the best-matching donor's category and |r|, and the dominant donor's.
Aggregates only (categories, correlations; no donor IDs)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from vp_common import *
cat = np.load(f"{CACHE}/prot_cat.npy")
P0 = np.load(f"{CACHE}/prot_orig.npy"); P1 = np.load(f"{CACHE}/prot_corr.npy")
X = pd.read_csv(f"{REL}/genes_final_proteins.csv", index_col=0); S = pd.read_csv(f"{REL}/pip_final_proteins.csv", index_col=0)
feats = [f for f in open(f"{SDA}/md_prot.txt").read().split("\n") if f]
assert list(X.columns) == feats and list(S.columns) == feats
cons = pd.read_csv(f"{REL}/samples_final.csv", index_col=0).values
h0 = ~np.isnan(P0).all(1); h1 = ~np.isnan(P1).all(1)
C0 = P0 - np.nanmean(P0[h0], 0); C1 = P1 - np.nanmean(P1[h1], 0)
rows = []
for k in X.index.astype(int):
    sel = S.loc[k].values > 0.5
    if sel.sum() < 5: continue
    x = X.loc[k].values[sel]
    def rr(C, h):
        out = np.full(N, np.nan)
        for n in np.where(h)[0]:
            v = C[n, sel]; ok = ~np.isnan(v)
            out[n] = np.corrcoef(x[ok], v[ok])[0, 1]
        return out
    r0 = rr(C0, h0); r1 = rr(C1, h1)
    b0 = int(np.nanargmax(np.abs(r0))); dom = int(np.abs(cons[k - 1]).argmax())
    share = float((cons[k - 1] ** 2).max() / (cons[k - 1] ** 2).sum())
    rows.append(dict(component_id=k, n_prot_pip05=int(sel.sum()), dom_share=share, dom_cat=int(cat[dom]),
                     dom_fed_r=float(r0[dom]) if h0[dom] else np.nan, dom_own_r=float(r1[dom]) if h1[dom] else np.nan,
                     best_fed_cat=int(cat[b0]), best_fed_absr=float(abs(r0[b0])), best_fed_is_dom=b0 == dom,
                     second_fed_absr=float(np.sort(np.abs(r0[h0]))[-2]),
                     best_corr_absr=float(np.nanmax(np.abs(r1))), best_corr_cat=int(cat[int(np.nanargmax(np.abs(r1)))])))
T = pd.DataFrame(rows).sort_values("n_prot_pip05", ascending=False)
T.to_csv(f"{RES}/protein_loadings_origin.csv", index=False)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))
