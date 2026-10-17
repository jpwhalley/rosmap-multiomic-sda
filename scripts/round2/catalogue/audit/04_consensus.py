"""Classify each RNA-active historical component: do run-level members agree on the
sign-invariant cell-type x gene product (sample-aligned), and does the published
B_final x X1_final product match the member consensus on the dominant cell types?
Consensus = rank-1 SVD of the mean of unit-norm member products. Read-only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from multiprocessing import Pool
BASE=cfg.SDA; OUT=cfg.out_dir(cfg.AUD); IT=1500
m=pd.read_csv(f"{OUT}/member_flips.csv"); m=m[m.slot<546]
ro=np.load(f"{OUT}/run_of_pos_it{IT}.npy"); st={i:int(np.nonzero(ro==i)[0][0]) for i in range(1,11)}
pip=pd.read_csv(f"{BASE}/pip_final_RNA.csv",index_col=0).values>0.5
Bf=pd.read_csv(f"{BASE}/treatment_final.csv",index_col=0).values; Xf=pd.read_csv(f"{BASE}/genes_final_RNA.csv",index_col=0).values
Bm={i:np.loadtxt(f"{BASE}/results{i}/it{IT}/B1").T for i in range(1,11)}
def work(i):
    sub=m[m.run==i]; want={int(p-st[i]):(int(k),int(fa)) for p,k,fa in zip(sub.pos,sub.slot,sub.fA)}; out=[]
    with open(f"{BASE}/results{i}/it{IT}/X1") as f:
        for ln,line in enumerate(f):
            if ln in want:
                k,fa=want[ln]
                if pip[k].sum()==0: continue
                P=fa*np.outer(Bm[i][ln],np.array(line.split(),dtype=np.float64)[pip[k]]); n=np.linalg.norm(P)
                if n>0: out.append((k,P/n))
    return out
if __name__=="__main__":
    mem={}
    with Pool(int(cfg.THREADS)) as pool:
        for res in pool.imap_unordered(work,range(1,11)):
            for k,P in res: mem.setdefault(k,[]).append(P)
    rows=[]
    for k,Ps in mem.items():
        C=sum(Ps); U,s,Vt=np.linalg.svd(C,full_matrices=False); b=U[:,0]*s[0]; x=Vt[0]
        g=Xf[k,pip[k]]; 
        if np.dot(x,g)<0: b,x=-b,-x                                    # orient consensus genes like the published genes
        rm=np.array([np.corrcoef(P.ravel(),C.ravel())[0,1] for P in Ps])
        dom=np.abs(b)>=0.25*np.abs(b).max()
        pub_b=Bf[k]
        rows.append(dict(id=k+1,n_members=len(Ps),frac_members_agree=float((rm>0).mean()),median_r_member=float(np.median(rm)),
                         rank1_share=float(s[0]**2/np.sum(s**2)),r_genes_pub_vs_cons=float(np.corrcoef(x,g)[0,1]) if len(g)>1 else 1.0,
                         n_dom_ct=int(dom.sum()),dom_ct_sign_disagree=int((np.sign(b[dom])!=np.sign(pub_b[dom])).sum()),
                         cons_b=np.round(b/np.abs(b).max(),2).tolist(),pub_b=np.round(pub_b/np.abs(pub_b).max(),2).tolist()))
    R=pd.DataFrame(rows).sort_values("id"); R.to_csv(f"{OUT}/consensus_rna.csv",index=False)
    stable=R.frac_members_agree>=0.8
    print(f"RNA-active comps {len(R)}; members agree (>=80%) in {int(stable.sum())}")
    print(f"  stable & published dominant cell-type sign disagrees with consensus: {int((stable&(R.dom_ct_sign_disagree>0)).sum())}")
    print(f"  stable & agrees: {int((stable&(R.dom_ct_sign_disagree==0)).sum())}; unstable (<80% agree): {int((~stable).sum())}")
    print(f"  gene-side: published vs consensus gene loadings r<0.9 in {int((R.r_genes_pub_vs_cons<0.9).sum())}")
    for k in (2,74,87,177,278,351,158,140,231,277,172,83,311):
        r=R[R.id==k]
        if len(r): r=r.iloc[0]; print(f"  C{k}: members {r.n_members}, agree {r.frac_members_agree:.0%}, rank1 {r.rank1_share:.2f}, dom-ct disagree {r.dom_ct_sign_disagree}/{r.n_dom_ct} | consensus {r.cons_b} | published {r.pub_b}")
