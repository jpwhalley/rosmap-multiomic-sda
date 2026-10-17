"""VERIFY part 4: SEA-AD concordance re-derived with own code from the archive csvs (ROSMAP genes_final_RNA / pip_final_RNA;
SEA-AD genes_final minus retired 9, 128; SEA-AD samples_final for partner concentration).
Pearson partner rule (Data S1), Spearman at the same partner, best-Spearman partner, top-3 gene share of the Pearson
numerator, robustness to dropping the top-5% contributing genes, own permutation null (seed 99, B = 1000, partner
search over all 141 inside each draw), and a spike-in that applies the FULL criterion B (|r|,|rho| >= 0.4 same sign,
null p < 0.01), not the null alone."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import time, numpy as np, pandas as pd
from scipy import stats as ss

t0 = time.time()
V = cfg.out_dir(cfg.WORK, "revision_analyses", "verify_structure"); cfg.out_dir(V, "out")
cfg.out_dir(V, "private")
ST = f"{V}/../structure"
ROS = cfg.SDA
SEA = f"{cfg.SEAAD}/sda"
AUD = cfg.AUD
GI = f"{cfg.SEAAD}/gene_info.csv"
DS1 = f"{cfg.TABLES}/Data_S1_corrected_547.csv"
rng = np.random.default_rng(99)

X = pd.read_csv(f"{ROS}/genes_final_RNA.csv", index_col=0)
P = pd.read_csv(f"{ROS}/pip_final_RNA.csv", index_col=0)
genes = list(X.columns); assert list(P.columns) == genes
X = X.drop(index=520); P = P.drop(index=520)
for k in (547, 548):
    X.loc[k] = np.load(f"{AUD}/c547/C{k}_X1.npy"); P.loc[k] = np.load(f"{AUD}/c547/C{k}_P1.npy")
fn = pd.read_csv(GI, index_col=0)["feature_name"]
sg = pd.read_csv(f"{SEA}/genes_final.csv", index_col=0)
sg.columns = [fn[c] for c in sg.columns]
sg = sg.drop(index=[9, 128]); seaids = sg.index.values.astype(int)
common = sorted(set(sg.columns) & set(genes))
Xc = X[common].values; Pc = P[common].values; SG = sg[common].values.astype(float)
SAF = pd.read_csv(f"{SEA}/samples_final.csv", index_col=0)
def topshare(a):
    d = a - a.mean(); d2 = d ** 2; return d2.max() / d2.sum()
sea_ts = {int(k): topshare(SAF.loc[k].values.astype(float)) for k in SAF.index}
print(f"common genes {len(common)}; SEA-AD comps {len(seaids)}; SEA-AD samples {SAF.shape[1]} [{time.time()-t0:.0f}s]", flush=True)

def unit(A):
    A = A - A.mean(-1, keepdims=True); return A / np.linalg.norm(A, axis=-1, keepdims=True)

ds1 = pd.read_csv(DS1, index_col=0)
rows = []; cache = {}
for i, k in enumerate(X.index):
    sel = Pc[i] > 0.5; n = int(sel.sum())
    rec = dict(component=k, n_genes=n)
    if n > 1:
        x = Xc[i, sel]; S_ = SG[:, sel]
        r = unit(S_) @ unit(x)
        rho = unit(np.apply_along_axis(ss.rankdata, 1, S_)) @ unit(ss.rankdata(x))
        j = int(np.argmax(r)) if abs(r.max()) >= abs(r.min()) else int(np.argmin(r))
        jr = int(np.argmax(np.abs(rho)))
        xc = x - x.mean(); sc = S_[j] - S_[j].mean(); con = xc * sc * np.sign(r[j])
        top3 = np.sort(con)[-3:].sum() / con.sum()
        drop = np.argsort(-con)[:max(1, int(np.ceil(0.05 * n)))]; keep = np.ones(n, bool); keep[drop] = False
        r_d = np.corrcoef(x[keep], S_[j][keep])[0, 1] if keep.sum() > 2 else np.nan
        rho_d = ss.spearmanr(x[keep], S_[j][keep])[0] if keep.sum() > 2 else np.nan
        rec.update(partner=int(seaids[j]), r=float(r[j]), rho=float(rho[j]), partner_top_share=sea_ts[int(seaids[j])],
                   top3_share=float(top3), r_drop5pct=float(r_d), rho_drop5pct=float(rho_d),
                   best_rho_partner=int(seaids[jr]), best_rho=float(rho[jr]), r_at_best_rho=float(r[jr]),
                   best_rho_partner_top_share=sea_ts[int(seaids[jr])])
        cache[k] = (x, S_)
    rows.append(rec)
R = pd.DataFrame(rows).set_index("component")
ok = R.partner.notna()
dsok = ds1["SEA-AD_comp"].notna()
print("components with partner:", int(ok.sum()), "| Data S1 non-blank:", int(dsok.sum()))
print("partner == Data S1:", int((R.partner[ok].astype(int) == ds1.loc[R.index[ok], "SEA-AD_comp"].astype(int)).sum()),
      "| max |r - DS1|:", float((R.r[ok] - ds1.loc[R.index[ok], "SEA-AD_corr"]).abs().max()))
st = pd.read_csv(f"{ST}/out/02_seaad_547.csv", index_col=0)
print("max |rho - structure|:", float((R.rho[ok] - st.loc[R.index[ok], "seaad_spearman"]).abs().max()),
      "| partner top share max diff:", float((R.partner_top_share[ok] - st.loc[R.index[ok], "seaad_partner_top_share"]).abs().max()),
      "| top3 max diff:", float((R.top3_share[ok] - st.loc[R.index[ok], "seaad_top3_gene_share_of_numerator"]).abs().max()))

# ---- own permutation null for every component with |r| >= 0.3 and |rho| >= 0.3 same sign, plus named
J = np.where(np.sign(R.r) == np.sign(R.rho), np.minimum(R.r.abs(), R.rho.abs()), 0.0)
R["J"] = np.where(ok, J, np.nan)
NAMED = [74, 278, 172, 277, 21, 545, 351, 2, 87, 177]
todo = [k for k in R.index[ok] if R.J[k] >= 0.3 or k in NAMED]
Bn = 1000
def null_p(x, S_, Jobs, B):
    Sz = unit(S_); SRz = unit(np.apply_along_axis(ss.rankdata, 1, S_))
    xz = unit(x); xr = unit(ss.rankdata(x))
    perm = np.argsort(rng.random((B, len(x))), 1)
    Rr = xz[perm] @ Sz.T; Rh = xr[perm] @ SRz.T
    jj = np.where(np.abs(Rr.max(1)) >= np.abs(Rr.min(1)), Rr.argmax(1), Rr.argmin(1))
    a = Rr[np.arange(B), jj]; b = Rh[np.arange(B), jj]
    Jn = np.where(np.sign(a) == np.sign(b), np.minimum(np.abs(a), np.abs(b)), 0)
    return (1 + np.sum(Jn >= Jobs)) / (B + 1), np.mean(Jn >= 0.4), np.quantile(Jn, 0.99)
for k in todo:
    x, S_ = cache[k]
    p, fh, q99 = null_p(x, S_, R.J[k], Bn)
    R.loc[k, "null_p"] = p; R.loc[k, "null_frac_hit04"] = fh; R.loc[k, "null_J_q99"] = q99
print(f"own null run on {len(todo)} components [{time.time()-t0:.0f}s]", flush=True)

def critB(df, thr=0.4, pthr=0.01, conc=0.5, r="r", rho="rho", pt="partner_top_share"):
    return ((df[r].abs() >= thr) & (df[rho].abs() >= thr) & (np.sign(df[r]) == np.sign(df[rho])) &
            (df.null_p < pthr) & (df[pt] < conc)).fillna(False)
R["B"] = critB(R)
sm = pd.read_csv(f"{ST}/out/05_master_prioritisation.csv", index_col=0)
bst = sm.support_B_seaad.astype(bool)
print("criterion B (own null): n pass", int(R.B.sum()), "| structure:", int(bst.sum()),
      "| disagreements:", sorted(set(R.index[R.B]) ^ set(sm.index[bst])))
for k in sorted(set(R.index[R.B]) ^ set(sm.index[bst])):
    print(f"   C{k}: r {R.r[k]:.3f} rho {R.rho[k]:.3f} own p {R.null_p[k]:.4f} struct p {st.nullA_p_J[k]:.4f}")
# null calibration diagnostics where it matters
tested = R.loc[todo]
print("null frac draws with J >= 0.4: median", float(tested.null_frac_hit04.median()), "| comps with > 5%:", int((tested.null_frac_hit04 > 0.05).sum()),
      "| of B-passers, max frac:", float(R.loc[R.B, "null_frac_hit04"].max()))
print("among B-passers: n genes min", int(R.loc[R.B, "n_genes"].min()), "| null J q99 max", round(float(R.loc[R.B, "null_J_q99"].max()), 3))

# ---- alternative partner choice: best |rho| partner (rank-first rule)
R["Jalt"] = np.where(np.sign(R.best_rho) == np.sign(R.r_at_best_rho), np.minimum(R.best_rho.abs(), R.r_at_best_rho.abs()), 0)
alt = ((R.best_rho.abs() >= 0.4) & (R.r_at_best_rho.abs() >= 0.4) & (np.sign(R.best_rho) == np.sign(R.r_at_best_rho)) &
       (R.best_rho_partner_top_share < 0.5)).fillna(False)
print("alt rule (partner by best |rho|, same thresholds, no null):", int(alt.sum()), "| main rule without null:",
      int(((R.r.abs() >= 0.4) & (R.rho.abs() >= 0.4) & (np.sign(R.r) == np.sign(R.rho)) & (R.partner_top_share < 0.5)).sum()))
R["B_alt_partner"] = alt
# ---- robustness of B-passers to dropping top-5% contributing genes
bp = R[R.B]
print(f"B-passers: after dropping top-5% genes, |r|>=0.3 & |rho|>=0.3 in {int(((bp.r_drop5pct.abs() >= 0.3) & (bp.rho_drop5pct.abs() >= 0.3)).sum())}/{len(bp)}; "
      f"|r|>=0.4 & |rho|>=0.4 in {int(((bp.r_drop5pct.abs() >= 0.4) & (bp.rho_drop5pct.abs() >= 0.4)).sum())}/{len(bp)}")

# ---- class relations (match structure claims)
C = pd.read_csv(f"{V}/private/v02_core_plus.csv", index_col=0)
R = R.join(C[["top_share", "cls"]])
for c in ["distributed", "few-donor", "concentrated"]:
    s = R[(R.cls == c) & ok]
    print(f"{c}: n {len(s)} med|r| {s.r.abs().median():.3f} med|rho| {s.rho.abs().median():.3f} top3 {s.top3_share.median():.3f} "
          f"|r|>0.4 {int((s.r.abs() > 0.4).sum())}; of those |rho|<0.2 {int(((s.r.abs() > 0.4) & (s.rho.abs() < 0.2)).sum())}; partner conc>=0.5 {int((s.partner_top_share >= 0.5).sum())}")
print("Spearman(log share, |rho|):", round(ss.spearmanr(np.log10(R.top_share[ok]), R.rho[ok].abs())[0], 3),
      "| (log share, |r|):", round(ss.spearmanr(np.log10(R.top_share[ok]), R.r[ok].abs())[0], 3))
print("SEA-AD 21 top share:", round(sea_ts[21], 3), "| SEA-AD 69:", round(sea_ts[69], 3))
print(R.loc[NAMED, ["n_genes", "partner", "r", "rho", "partner_top_share", "top3_share", "null_p", "B"]].to_string(float_format=lambda v: f"{v:.3g}"))

# ---- spike-in with the FULL criterion B
dist = R[(R.cls == "distributed") & ok]
targets = [60, 150, 400, 900, 1600]
pick = [int((dist.n_genes - t).abs().idxmin()) for t in targets]
out = []
for k in pick:
    x, S_ = cache[k]; j = int(np.where(seaids == R.partner[k])[0][0])
    sz = (S_[j] - S_[j].mean()) / S_[j].std()
    for c in [0.2, 0.3, 0.4, 0.5, 0.6]:
        passB = []; passN = []
        for rep in range(20):
            nz = rng.permutation(x); nz = (nz - nz.mean()) / nz.std()
            xs = c * sz + np.sqrt(1 - c * c) * nz
            r = unit(S_) @ unit(xs); rho = unit(np.apply_along_axis(ss.rankdata, 1, S_)) @ unit(ss.rankdata(xs))
            jj = int(np.argmax(r)) if abs(r.max()) >= abs(r.min()) else int(np.argmin(r))
            Jo = min(abs(r[jj]), abs(rho[jj])) if np.sign(r[jj]) == np.sign(rho[jj]) else 0
            p, _, _ = null_p(xs, S_, Jo, 300)
            passN.append(p < 0.01); passB.append(p < 0.01 and Jo >= 0.4 and jj == j)
        out.append(dict(component=k, n_genes=int(R.n_genes[k]), c=c, detect_null_only=np.mean(passN), detect_full_B=np.mean(passB)))
PC = pd.DataFrame(out); PC.to_csv(f"{V}/out/v04_spikein_fullB.csv", index=False)
print("\nspike-in: detection by the null alone vs by the full criterion B (20 reps)")
print(PC.pivot_table(index=["component", "n_genes"], columns="c", values=["detect_null_only", "detect_full_B"]).to_string(float_format=lambda v: f"{v:.2f}"))
R.drop(columns=["top_share", "cls"]).to_csv(f"{V}/out/v04_seaad_547.csv")
print(f"done [{time.time()-t0:.0f}s]")
