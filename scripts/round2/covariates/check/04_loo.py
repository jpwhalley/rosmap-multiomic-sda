"""Leave-one-donor-out worst case for the central five, with the WHOLE 547 family refitted for every dropped donor
(exact BH), for PO M1/M2 and RINT-OLS M1/M2 (RINT recomputed over the retained donors). Also records the
'fixed-others' approximation (other components' p held at full-data values). Aggregates only; no donor IDs."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, time, numpy as np, pandas as pd
from scipy import stats as ss
from vc_common import *

s = load_scores(); don = s.columns; cl = load_clin(don); qc = load_qc(don)
FIVE = {"Sex": [74, 278, 545], "ceradsc": [172], "cogdx": [277]}
age, pmi, mp, male = cl.age.values, cl.pmi.values, cl.is_map.values, cl.male.values
lnuc, lupn = np.log(qc.nuclei.values), np.log(qc.upn.values)
cog = cl.cogdx.values; sub = np.isin(cog, [1, 2, 4])
Y = s.values  # 547 x 276
idx = list(s.index)

def design(ph, model, keep):
    base = {"M1": [age, pmi, mp], "M2": [age, pmi, mp, lnuc, lupn]}[model]
    if ph != "Sex": base = base + [male]
    Xc = np.column_stack(base)[keep]
    if ph == "Sex": Xp = male[keep][:, None]
    elif ph == "ceradsc": Xp = cl.cerad_ad.values[keep][:, None]
    else: Xp = np.column_stack([cog[keep] == 2, cog[keep] == 4]).astype(float)
    return Xc, Xp

full = pd.read_csv("vc_po_547.csv"); fullr = pd.read_csv("vc_rint_547.csv")
which = sys.argv[1:] if len(sys.argv) > 1 else ["Sex", "ceradsc", "cogdx"]
rows = []
t0 = time.time()
for ph in which:
    base_keep = sub if ph == "cogdx" else np.ones(276, bool)
    drops = np.nonzero(base_keep)[0]
    for model in ["M1", "M2"]:
        pf = full[(full.pheno == ph) & (full.model == model)].set_index("comp").p.loc[idx].values
        pr = fullr[(fullr.pheno == ph) & (fullr.model == model)].set_index("comp").p.loc[idx].values
        for j in drops:
            keep = base_keep.copy(); keep[j] = False
            Xc, Xp = design(ph, model, keep)
            p = np.array([po_lrt(Y[i, keep], Xc, Xp)["p"] for i in range(len(idx))])
            q = bh(p)
            # RINT over all retained donors (276-1), then phenotype subset
            k_all = np.ones(276, bool); k_all[j] = False
            R = np.column_stack([rint(Y[i, k_all]) for i in range(len(idx))])
            m = base_keep[k_all]
            prr, _ = ols_F(R[m], Xc, Xp); qr = bh(prr)
            for c in FIVE[ph]:
                i = idx.index(c)
                pa = pf.copy(); pa[i] = p[i]; qa = bh(pa)[i]
                pra = pr.copy(); pra[i] = prr[i]; qra = bh(pra)[i]
                rows.append(dict(pheno=ph, model=model, comp=c, drop_pos=int(j), p_po=p[i], q_po=q[i], q_po_fixedothers=qa,
                                 p_rint=prr[i], q_rint=qr[i], q_rint_fixedothers=qra))
        print(f"{ph} {model} done ({time.time() - t0:.0f}s)", flush=True)
d = pd.DataFrame(rows)
d.to_csv(f"vc_loo_{'_'.join(which)}.csv", index=False)   # drop_pos is a column position only (no donor IDs)
g = d.groupby(["comp", "pheno", "model"]).agg(worst_q_po=("q_po", "max"), worst_q_po_fixed=("q_po_fixedothers", "max"),
                                              worst_q_rint=("q_rint", "max"), worst_q_rint_fixed=("q_rint_fixedothers", "max"))
print(g.to_string(float_format=lambda x: f"{x:.3g}"))
