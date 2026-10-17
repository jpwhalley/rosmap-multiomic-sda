"""Spearman rho of component scores with donor cell-class proportions (nuclei per class / total, six classes) and QC proxies;
class proportions by cogdx (1/2/4) and by sex. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss
from dc_common import *
S=scores547(); D=design_frame(); rows=[]
for k in [277,172,278,545,21,74]:
    a=S.loc[k].values; r={c:ss.spearmanr(a,D[f"prop_{c}"])[0] for c in CTS}
    r["lognuc"]=ss.spearmanr(a,D.lognuc)[0]; r["logupn"]=ss.spearmanr(a,D.logupn)[0]; rows.append(pd.Series(r,name=k))
print(pd.DataFrame(rows).round(2).to_string())
sub=D.cogdx.isin([1,2,4])
for c in ["Exc","Inh","Oli"]:
    print(c,"prop by cogdx medians",D[sub].groupby("cogdx")[f"prop_{c}"].median().round(3).to_dict(),"KW p",round(ss.kruskal(*[D[sub&(D.cogdx==g)][f"prop_{c}"] for g in (1,2,4)]).pvalue,4))
for c in CTS:
    print(c,"F-M median",round(D[D.sex==0][f"prop_{c}"].median()-D[D.sex==1][f"prop_{c}"].median(),3),"MW p",round(ss.mannwhitneyu(D[D.sex==0][f"prop_{c}"],D[D.sex==1][f"prop_{c}"]).pvalue,4))
