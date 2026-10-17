"""Build Data S1 rows for the two components added by the corrected run count (cluster 1465 -> C547,
cluster 716 -> C548) and the 547-family BH clinical values, with the original pipeline's conventions:
A and X_j: per-block Pearson alignment to the first member (ros2_cluster.py); PIPs: median; cell types:
corrected (member flip = fA*fX1) plus the runs-evidence check; clinical tests as notebook 03/ros6a
(sex MWU; CERAD binarised MWU; cogdx KW over 1,2,4); BH per variable across the catalogue; SEA-AD partner
as 12_seaad_validation.py (Pearson over ROSMAP PIP>0.5 genes shared with SEA-AD; larger |r| extreme).
Note: the SEA-AD partner here is matched against all 143 deposited SEA-AD components. C548's (128) is replaced in
make_DataS1_547.py step 5 by its best partner among the corrected 141 (SEA-AD 125).
Each step is first validated against the published Data S1. Read-only on its inputs; writes to
$G_ROS_AUDIT_DIR/c547/ (clinical_bh_547.csv -> $G_ROS_TABLES_DIR/clinical_BH_547_family.csv,
new_rows.json -> $G_ROS_TABLES_DIR/DataS1_new_rows_C547_C548.json, both copied as they are)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, json, numpy as np, pandas as pd
from scipy import stats as ss
B=cfg.SDA; AUD=cfg.AUD; OUT=cfg.out_dir(AUD, "c547")   # intermediates; the two files the tables step uses are copied to $G_ROS_TABLES_DIR as clinical_BH_547_family.csv and DataS1_new_rows_C547_C548.json
SEA=f"{cfg.SEAAD}/sda"; GI=f"{cfg.SEAAD}/gene_info.csv"
DS1=cfg.DS1_V22; CLIN=cfg.CLINICAL
CT=["Ast","Oli","Inh","OPC","Exc","Mic"]; OM={1:"RNA",2:"meth",3:"chip",4:"proteins"}
feat={1:open(f"{B}/genes.txt").read().split(),2:open(f"{B}/md_cpg.txt").read().split(),3:open(f"{B}/md_chip.txt").read().split(),4:open(f"{B}/md_prot.txt").read().split()}
cl=np.load(f"{AUD}/clusters_it1500.npy"); rop=np.load(f"{AUD}/run_of_pos_it1500.npy"); st={i:int(np.nonzero(rop==i)[0][0]) for i in range(1,11)}
ds1=pd.read_csv(DS1,index_col=0); dep=pd.read_csv(f"{B}/samples_final.csv",index_col=0); dep.columns=dep.columns.astype(str)
def pr(a,b):
    r=ss.pearsonr(a,b)[0]; return r
def flip(r): return -1 if (r==r and r<0) else 1       # NaN -> no flip, as `nan < 0` is False in ros2_cluster.py
def line(path,ln):
    with open(path) as f:
        for i,l in enumerate(f):
            if i==ln: return np.array(l.split(),dtype=np.float64)
A={i:np.loadtxt(f"{B}/results{i}/it1500/A").T for i in range(1,11)}; B1={i:np.loadtxt(f"{B}/results{i}/it1500/B1").T for i in range(1,11)}
def consensus(c):
    pos=np.nonzero(cl==c)[0]; mem=[(int(rop[p]),int(p-st[rop[p]])) for p in pos]
    Am=np.array([A[r][j] for r,j in mem]); Bm=np.array([B1[r][j] for r,j in mem])
    X={k:np.array([line(f"{B}/results{r}/it1500/X{k}",j) for r,j in mem]) for k in OM}
    S={k:np.array([line(f"{B}/results{r}/it1500/S{k}",j) for r,j in mem]) for k in OM}
    fA=np.array([flip(pr(Am[0],a)) for a in Am]); fB=np.array([flip(pr(Bm[0],b)) for b in Bm]); fX={k:np.array([flip(pr(X[k][0],x)) for x in X[k]]) for k in OM}
    out=dict(members=mem,A=(fA[:,None]*Am).mean(0),B_hist=(fB[:,None]*Bm).mean(0),B_cor=((fA*fX[1])[:,None]*Bm).mean(0),
             X={k:(fX[k][:,None]*X[k]).mean(0) for k in OM},P={k:np.median(S[k],0) for k in OM},Am=Am,Bm=Bm,X1m=X[1])
    return out
# sanity: rebuilding historical C520 (cluster 1970) must reproduce the deposit row
c520=consensus(1970); print("C520 rebuild vs deposit: samples", float(np.abs(c520["A"]-dep.loc[520].values).max()),
      "| cell types", float(np.abs(c520["B_hist"]-pd.read_csv(f"{B}/treatment_final.csv",index_col=0).loc[520].values).max()))
new={547:consensus(1465),548:consensus(716)}
# ---- clinical tests and BH ----------------------------------------------------------------
clin=pd.read_csv(CLIN); clin.index=clin.projid.astype(int).astype(str); clin=clin.loc[dep.columns]
CER={1:"AD",2:"AD",3:"No",4:"No"}
def tests(v):
    s=pd.Series(v,index=dep.columns); m=clin.msex; cb=clin.ceradsc.map(CER)
    return (ss.mannwhitneyu(s[m==1],s[m==0],alternative="two-sided").pvalue, ss.mannwhitneyu(s[cb=="AD"],s[cb=="No"],alternative="two-sided").pvalue,
            ss.kruskal(*[s[clin.cogdx==k] for k in (1,2,4)]).pvalue)
def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); r=p[o]; adj=np.minimum.accumulate((r*n/np.arange(1,n+1))[::-1])[::-1]; out=np.empty(n); out[o]=np.minimum(adj,1); return out
raw=pd.DataFrame([tests(dep.loc[i].values) for i in dep.index],index=dep.index,columns=["Sex","ceradsc","cogdx"])
adj546=raw.apply(bh)
for c in raw:   # validation: published cells reproduced, same significant set
    pub=ds1[c].dropna(); sig=adj546.index[adj546[c]<0.05]
    assert set(sig)==set(pub.index), c; print(f"{c}: published {len(pub)} significant cells reproduced; max |diff| {float((adj546.loc[pub.index,c]-pub).abs().max()):.1e}")
raw547=raw.drop(index=520).copy()
for k,d in new.items(): raw547.loc[k]=tests(d["A"])
adj547=raw547.apply(bh)
for c in raw: print(f"{c}: significant 546-family {int((adj546[c]<0.05).sum())} -> 547-family {int((adj547[c]<0.05).sum())}; lost {sorted(set(adj546.index[adj546[c]<0.05])-set(adj547.index[adj547[c]<0.05]))}; gained {sorted(set(adj547.index[adj547[c]<0.05])-set(adj546.index[adj546[c]<0.05]))}")
adj547.to_csv(f"{OUT}/clinical_bh_547.csv")
# ---- SEA-AD partner ----------------------------------------------------------------------------
sg=pd.read_csv(f"{SEA}/genes_final.csv",index_col=0); fn=pd.read_csv(GI,index_col=0)["feature_name"].to_dict(); sg=sg.rename(columns=fn)
rg=pd.read_csv(f"{B}/genes_final_RNA.csv",index_col=0); rp=pd.read_csv(f"{B}/pip_final_RNA.csv",index_col=0)
common=sorted(set(sg.columns)&set(rg.columns)); sg=sg[common]
def partner(vec, pip):
    v=pd.Series(vec,index=feat[1])[common]; p=pd.Series(pip,index=feat[1])[common]; g=p.index[p>0.5]
    if len(g)<=1: return None,None
    cor=sg[g].apply(lambda row: row.corr(v[g]),axis=1); mx,mn=cor.idxmax(),cor.idxmin()
    return (mx,cor[mx]) if abs(cor[mx])>=abs(cor[mn]) else (mn,cor[mn])
chk=[k for k in ds1.index if pd.notna(ds1.loc[k,"SEA-AD_comp"])]; bad=0; worst=0
for k in chk:
    pid,pc=partner(rg.loc[k].values,rp.loc[k].values)
    if pid is None or int(pid)!=int(ds1.loc[k,"SEA-AD_comp"]): bad+=1
    else: worst=max(worst,abs(pc-ds1.loc[k,"SEA-AD_corr"]))
print(f"SEA-AD partner validation: {len(chk)-bad}/{len(chk)} published partners reproduced (max |r| diff {worst:.1e})")
# ---- cell-type labels with runs-evidence check --------------------------------------------------
def labels(d):
    b=d["B_cor"]; mx,mn=b.max(),b.min(); p=CT[int(b.argmax())] if mx>0 else ""; n=CT[int(b.argmin())] if mn<0 else ""
    a,x=d["A"],d["X"][1]; pa=d["Am"]@a; px=d["X1m"]@x
    dirn={CT[c]: np.sign((d["Bm"][:,c]*pa*px).sum()) for c in range(6)}      # runs' direction relative to the consensus A (x) X1
    if p and np.sign(b[CT.index(p)])!=dirn[p]: p=""
    if n and np.sign(b[CT.index(n)])!=dirn[n]: n=""
    return p,n,dirn
rows={}
for k,d in new.items():
    p,n,dirn=labels(d); om=sum(int((d["P"][j]>0.5).any()) for j in OM)
    r=dict(Component=k,omics_count=om,Top_positive_cell_type=p,Top_negative_cell_type=n)
    for c in ["Sex","ceradsc","cogdx"]: r[c]=adj547.loc[k,c] if adj547.loc[k,c]<0.05 else ""
    pid,pc=partner(d["X"][1],d["P"][1]); r["SEA-AD_comp"]=float(pid) if pid is not None else ""; r["SEA-AD_corr"]=pc if pc is not None else ""
    for j,o in OM.items():
        s=pd.Series(d["X"][j],index=feat[j])[d["P"][j]>0.5]
        r[f"Top_Positive_{o}"]=s[s>0].idxmax() if (s>0).any() else ""; r[f"Top_Negative_{o}"]=s[s<0].idxmin() if (s<0).any() else ""
    r["GeneExpression_Pathway"]="PENDING (KEGG_2021_Human enrichment not run)"; r["CpG_Site_Pathway"]=""; r["Protein_Pathway"]=""
    rows[k]=r
    print(f"C{k}: {len(d['members'])} members from runs {sorted(set(m[0] for m in d['members']))}; omics_count {om}; cell types corrected {dict(zip(CT,np.round(d['B_cor'],3)))} -> labels {p or '-'}/{n or '-'} (runs direction {dirn})")
    print(f"      clinical 547-BH: Sex {adj547.loc[k,'Sex']:.3g}, CERAD {adj547.loc[k,'ceradsc']:.3g}, cogdx {adj547.loc[k,'cogdx']:.3g}; SEA-AD partner {pid} r={pc:.3f}; PIP>0.5 RNA genes {int((d['P'][1]>0.5).sum())}; top +{r['Top_Positive_RNA']} / -{r['Top_Negative_RNA']}")
json.dump({str(k):{kk:(vv if not isinstance(vv,(np.floating,)) else float(vv)) for kk,vv in v.items()} for k,v in rows.items()},open(f"{OUT}/new_rows.json","w"),indent=1,default=str)
for k,d in new.items():
    np.save(f"{OUT}/C{k}_A.npy",d["A"]); np.save(f"{OUT}/C{k}_Bcor.npy",d["B_cor"])
    for j in OM: np.save(f"{OUT}/C{k}_X{j}.npy",d["X"][j]); np.save(f"{OUT}/C{k}_P{j}.npy",d["P"][j])
    pd.Series(d["X"][1],index=feat[1])[d["P"][1]>0.5].to_csv(f"{OUT}/C{k}_PIPgenes_RNA.csv",header=["loading"])
