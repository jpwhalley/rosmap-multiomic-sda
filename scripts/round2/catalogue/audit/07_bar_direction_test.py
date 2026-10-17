"""Alignment-free test of each published cell-type bar (rule: fix a bar only if
it points the wrong way, and only if certain). For component k and cell type c:
  published slice  P_c = B_pub[c] * (A_pub outer X_pub)                (what the panels show)
  fitted slice     M_c = sum over run-level members m of B_m[c] * (A_m outer X_m)
Each member's own product is invariant to its pair flips, so M_c needs no sign alignment.
cos_c = <P_c, M_c> / (|P_c| |M_c|), computed exactly via inner products (rank-1 terms).
  cos_c ~ +1 : the bar points the same way as the fitted runs  -> correct
  cos_c ~ -1 : the bar points the opposite way                 -> wrong direction
  |cos_c| small: the runs' contribution for c is not carried by the published sample/gene
                 vectors -> cannot judge (leave as published)
Also reports, per member loading materially on c, the sign of its projection onto P_c.
Read-only on the archive."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
BASE=cfg.SDA; OUT=cfg.out_dir(cfg.AUD); IT=1500
ct=["Ast","Oli","Inh","OPC","Exc","Mic"]
COMPS=[2,74,87,177,278,351,21,140,158,172,231,277]
m=pd.read_csv(f"{OUT}/member_flips.csv"); ro=np.load(f"{OUT}/run_of_pos_it{IT}.npy"); st={i:int(np.nonzero(ro==i)[0][0]) for i in range(1,11)}
Ap=pd.read_csv(f"{BASE}/samples_final.csv",index_col=0); Bp=pd.read_csv(f"{BASE}/treatment_final.csv",index_col=0)
Xp=pd.read_csv(f"{BASE}/genes_final_RNA.csv",index_col=0)
A={i:np.loadtxt(f"{BASE}/results{i}/it{IT}/A").T for i in range(1,11)}; Bm={i:np.loadtxt(f"{BASE}/results{i}/it{IT}/B1").T for i in range(1,11)}
mem={k:m[m.slot==k-1].sort_values("pos") for k in COMPS}
need={}
for k,d in mem.items():
    for r in d.itertuples(): need.setdefault(int(r.run),{})[int(r.pos-st[r.run])]=None
for i,want in need.items():
    with open(f"{BASE}/results{i}/it{IT}/X1") as f:
        for ln,line in enumerate(f):
            if ln in want: want[ln]=np.array(line.split(),dtype=np.float64)
rows=[]
for k in COMPS:
    a=Ap.loc[k].values; x=Xp.loc[k].values; b=Bp.loc[k].values
    M=[(A[r.run][r.pos-st[r.run]], Bm[r.run][r.pos-st[r.run]], need[r.run][r.pos-st[r.run]], r.run) for r in mem[k].itertuples()]
    aa=a@a; xx=x@x
    G_A=np.array([[mi[0]@mj[0] for mj in M] for mi in M]); G_X=np.array([[mi[2]@mj[2] for mj in M] for mi in M])
    pa=np.array([a@mi[0] for mi in M]); px=np.array([x@mi[2] for mi in M])
    for c in range(6):
        if abs(b[c]) < 0.10*np.abs(b).max(): continue
        w=np.array([mi[1][c] for mi in M])
        normM=np.sqrt(max(w@(G_A*G_X)@w,0)); proj=np.sign(b[c])*(w*pa*px).sum()
        cos=proj/(np.sqrt(aa*xx)*normM) if normM>0 else np.nan
        mat=np.array([abs(mi[1][c])>=0.10*np.abs(mi[1]).max() for mi in M])
        msign=np.sign(b[c]*w*pa*px)[mat]; mweight=np.abs(w*pa*px)[mat]
        rows.append(dict(comp=k,ct=ct[c],B_pub=round(b[c],3),cos=round(float(cos),3),
                         members_material=int(mat.sum()),members_agree=int((msign>0).sum()),members_oppose=int((msign<0).sum()),
                         weight_share_opposing=round(float(mweight[msign<0].sum()/mweight.sum()),3) if mweight.sum()>0 else np.nan))
R=pd.DataFrame(rows)
def verdict(cs): return "WRONG direction" if cs<=-0.5 else ("correct" if cs>=0.5 else "cannot judge")
R["verdict"]=R.cos.apply(verdict); R.to_csv(f"{OUT}/bar_direction_test.csv",index=False)
pd.set_option("display.width",200); print(R.to_string(index=False))
