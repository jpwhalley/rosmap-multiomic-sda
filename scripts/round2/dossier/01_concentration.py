"""Score concentration for the dossier components on the 547 catalogue (aggregates only).
top_share = largest squared deviation from the mean / sum of squared deviations (as in grounding §1);
PR = participation ratio of centred scores, (sum d^2)^2 / sum d^4 (1 = one donor, 276 = all equal);
robust z = (x - median) / (1.4826 MAD). Also per-modality participation ratio over donors with that modality."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss
from dc_common import *
S = scores547(); don = S.columns
has = {}
for m in ["meth", "chip", "proteins"]:
    v = pd.read_csv(f"{SDA}/{INPUT[m]}", sep=" ", header=None, usecols=[0, 1, 2], na_values="NA").values
    has[m] = ~np.isnan(v).all(1)
mis = np.load(f"{cfg.WORK}/source_panels/cache/prot_rows_misattributed.npy")

def conc(a):
    d = a - a.mean(); d2 = d ** 2
    rz = robust_z(a); o = np.argsort(-np.abs(rz))
    return dict(top_share=d2.max() / d2.sum(), top3_share=np.sort(d2)[-3:].sum() / d2.sum(),
                PR=d2.sum() ** 2 / (d2 ** 2).sum(), max_rz=abs(rz[o[0]]), second_rz=abs(rz[o[1]]),
                excess_kurtosis=ss.kurtosis(a), idx=int(np.argmax(d2)))
rows = []
for k in COMPS:
    a = S.loc[k].values.astype(float); c = conc(a); i = c.pop("idx")
    r = dict(component=k, **c)
    for m in ["meth", "chip", "proteins"]:
        am = a[has[m]]; dm = am - am.mean()
        r[f"PR_{m}"] = (dm ** 2).sum() ** 2 / (dm ** 4).sum()
        r[f"top_has_{m}"] = bool(has[m][i])
    r["top_prot_row_misattributed"] = bool(mis[i])
    rows.append(r)
T = pd.DataFrame(rows).set_index("component")
T.to_csv(f"{OUT}/01_concentration.csv", float_format="%.4g")
print(T.to_string(float_format=lambda x: f"{x:.3g}"))
# catalogue level on 547
allc = pd.DataFrame([conc(S.loc[k].values.astype(float)) for k in S.index], index=S.index)
print(f"\n547 catalogue: top_share > 0.8: {int((allc.top_share > 0.8).sum())}; > 0.5: {int((allc.top_share > 0.5).sum())};"
      f" PR < 2: {int((allc.PR < 2).sum())}; distinct top donors among top_share>0.8: {allc[allc.top_share > 0.8].idx.nunique()}")
allc[["top_share", "PR", "max_rz", "excess_kurtosis"]].to_csv(f"{OUT}/01_concentration_547.csv", float_format="%.4g")
