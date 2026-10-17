"""VERIFY part 7: recompute the prioritisation from the verify outputs (own concentration, own GWAS scan, own SEA-AD + null),
the PO1 adjusted q (checked in v05) and the structure cross-layer flags (C74 re-checked in v06). Compare every count in
the report; robustness incl. alternative phenotype model and alternative concentration definitions; t(nu) spill-over
estimate for the 0.10 threshold; technical covariates (own code); Data S1 additions check. Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss
V = cfg.out_dir(cfg.WORK, "revision_analyses", "verify_structure"); cfg.out_dir(V, "out")
cfg.out_dir(V, "private")
ST = f"{V}/../structure"
R2 = cfg.WORK
ROS = cfg.ROS
DS1 = f"{cfg.TABLES}/Data_S1_corrected_547.csv"
pd.set_option("display.width", 250)
C = pd.read_csv(f"{V}/private/v02_core_plus.csv", index_col=0)
G = pd.read_csv(f"{V}/out/v03_gwas_547.csv", index_col=0)
SE = pd.read_csv(f"{V}/out/v04_seaad_547.csv", index_col=0)
PH = pd.read_csv(f"{V}/out/v05_pheno_547.csv", index_col=0)
po = pd.read_csv(f"{R2}/covariates/po_tests_547.csv", header=[0, 1], index_col=0)
XL = pd.read_csv(f"{ST}/out/03_crosslayer.csv")
SM = pd.read_csv(f"{ST}/out/05_master_prioritisation.csv", index_col=0)
M = C.join(G).join(SE.rename(columns={"top3_share": "seaad_top3_share"}))
ORDER = ["distributed", "few-donor", "concentrated"]
P3 = ["Sex", "ceradsc", "cogdx"]
for ph in P3:
    M[f"q_{ph}"] = po[(ph, "PO1_q")].values; M[f"qqc_{ph}"] = po[(ph, "PO2_q")].values; M[f"qu_{ph}"] = po[(ph, "U_q")].values
    M[f"qalt_{ph}"] = PH[f"alt_{ph}_q"].values

# ---------------- relations claimed in the report
print("=== unadjusted Data S1 calls by class (calls):", {c: int((M.loc[M.cls == c, [f'qu_{p}' for p in P3]] < .05).values.sum()) for c in ORDER})
adj = (M[[f"q_{p}" for p in P3]] < .05)
print("=== adjusted (PO1) calls by class: calls", {c: int(adj[M.cls == c].values.sum()) for c in ORDER},
      "| components", {c: int(adj[M.cls == c].any(axis=1).sum()) for c in ORDER},
      "| concentrated with call:", list(M.index[(M.cls == 'concentrated') & adj.any(axis=1)]),
      "| which phenotype:", {k: [p for p in P3 if adj.loc[k, f'q_{p}']] for k in M.index[(M.cls == 'concentrated') & adj.any(axis=1)]})
un = (M[[f"qu_{p}" for p in P3]] < .05)
print("adjusted calls total", int(adj.values.sum()), "| unadjusted", int(un.values.sum()), "| kept", int((adj.values & un.values).sum()),
      "| new", int((adj.values & ~un.values).sum()), "by phenotype new:", {p: int((adj[f'q_{p}'] & ~un[f'qu_{p}']).sum()) for p in P3})
print("=== GWAS by class:")
for c in ORDER:
    s = M[M.cls == c]
    print(f"  {c}: min_p<1e-15 {int((s.min_p < 1e-15).sum())}; auto<1e-15 {int((s.min_p_auto < 1e-15).sum())}; min_p<5e-8/547 {int((s.min_p < 5e-8/547).sum())}; "
          f"auto<5e-8/547 {int((s.min_p_auto < 5e-8/547).sum())}; auto<5e-8 {int((s.min_p_auto < 5e-8).sum())}; median lambda {s.lam.median():.3f}")
print("  distributed with autosomal min p < 5e-8:", {k: f"{M.min_p_auto[k]:.1e} (lambda {M.lam[k]:.3f})" for k in M.index[(M.cls == 'distributed') & (M.min_p_auto < 5e-8)]})
print("  Spearman(log share, -log10 auto p):", round(ss.spearmanr(np.log10(M.top_share), -np.log10(M.min_p_auto))[0], 3))
print("  C74: overall min", f"{M.min_p[74]:.2e} chr {int(M.top_chr[74])}; autosomal {M.min_p_auto[74]:.2e}; non-autosomal <5e-8: {int(M.n5e8_nonauto[74])}")

# ---------------- supports
A1 = (M[[f"q_{p}" for p in P3]] < .05).any(axis=1)
A2 = (M.min_p_auto < 5e-8 / 547) & M.lam.between(0.95, 1.05)
B = M.B.fillna(False).astype(bool)
Cx = SM.support_C_crosslayer.astype(bool).reindex(M.index)
sup = A1 | A2 | B | Cx
pop = M.top_share < 0.10
pri = pop & sup
mine = set(M.index[pri]); theirs = set(pd.read_csv(f"{ST}/out/05_prioritised.csv", index_col=0).index)
print(f"\n=== prioritised: mine {len(mine)}, structure {len(theirs)}, sym diff {sorted(mine ^ theirs)}")
tierA = pri & (A1 | A2)
print("Tier A", int(tierA.sum()), "| B-only", int((pri & ~(A1 | A2)).sum()), "| support counts by class:",
      {nm: {c: int((v & (M.cls == c)).sum()) for c in ORDER} for nm, v in [("A1", A1), ("A2", A2), ("B", B), ("C", Cx), ("any", sup)]})
pa = M[pri & A1]
print("Tier A phenotype breakdown:", {"sex": int((pa.q_Sex < .05).sum()), "cerad only": int(((pa.q_ceradsc < .05) & ~(pa.q_cogdx < .05)).sum()),
      "cogdx only": int(((pa.q_cogdx < .05) & ~(pa.q_ceradsc < .05)).sum()), "both": int(((pa.q_cogdx < .05) & (pa.q_ceradsc < .05)).sum())})
ex = ~pop & sup
print("supported but excluded by step 1:", int(ex.sum()), {c: int((ex & (M.cls == c)).sum()) for c in ORDER},
      "| concentrated via A2:", int((ex & (M.cls == 'concentrated') & A2).sum()))
print("prioritised with zero PIP>0.5 features:", list(M.index[pri & (M.n_mod == 0)]), "| with RNA PIP>0.5 genes < 20:",
      {k: int(M.RNA[k]) for k in M.index[pri & (M.RNA < 20)]})
print("B-passers in the 113 with < 30 matched genes:", {k: (int(M.n_genes[k]), round(M.r[k], 2), round(M.rho[k], 2), float(M.null_p[k])) for k in M.index[pri & B & (M.n_genes < 30)]})

# ---------------- robustness
rows = []
def rec(lbl, p, s):
    st = set(M.index[p & s]); foc = [k for k in [74, 278, 172, 277, 21] if k not in st]
    rows.append((lbl, int((p & s).sum()), ",".join(f"C{k}" for k in foc)))
for t in [0.05, 0.075, 0.10, 0.125, 0.15, 0.2, 0.3, 0.5]:
    rec(f"share<{t}", M.top_share < t, sup)
for pr in [10, 20, 30, 50]:
    rec(f"PR>={pr}", M.PR >= pr, sup)
rec("top3 share<0.25", M.top3_share < 0.25, sup)
rec("uncentred share<0.10", M.top_share_uncentred < 0.10, sup)
rec("A1 alt model (phenotype-as-outcome)", pop, (M[[f"qalt_{p}" for p in P3]] < .05).any(axis=1) | A2 | B | Cx)
rec("A1 PO2 (QC proxies)", pop, (M[[f"qqc_{p}" for p in P3]] < .05).any(axis=1) | A2 | B | Cx)
rec("A1 unadjusted", pop, (M[[f"qu_{p}" for p in P3]] < .05).any(axis=1) | A2 | B | Cx)
rec("A1 BH across 3x547 jointly", pop, pd.Series(False, index=M.index) | A2 | B | Cx)  # placeholder replaced below
qall = np.concatenate([po[(p, "PO1_p")].values for p in P3])
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum(1, np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]); return q
qj = bh(qall).reshape(3, -1).T
A1j = pd.Series((qj < .05).any(1), index=M.index)
rows[-1] = ("A1 BH across 3x547 jointly", int((pop & (A1j | A2 | B | Cx)).sum()), ",".join(f"C{k}" for k in [74, 278, 172, 277, 21] if not (pop & (A1j | A2 | B | Cx))[k]))
rec("B Pearson-only |r|>0.4", pop, A1 | A2 | (M.r.abs() > 0.4) | Cx)
rec("drop B", pop, A1 | A2 | Cx)
rec("drop A2", pop, A1 | B | Cx)
rec("A2 at 5e-8", pop, A1 | ((M.min_p_auto < 5e-8) & M.lam.between(0.95, 1.05)) | B | Cx)
rec("require >=1 PIP>0.5 feature", pop & (M.n_mod > 0), sup)
print("\n=== robustness (n prioritised, focus missing)")
for r in rows: print(f"  {r[0]:40s} {r[1]:4d}  {r[2]}")
oldB = pop & (A1 | A2 | (M.r.abs() > 0.4) | Cx) & ~pri
o = M[oldB]
print("Pearson-only rule adds", len(o), "; of these |rho|<0.4:", int((o.rho.abs() < 0.4).sum()), "; |rho|<0.2:", int((o.rho.abs() < 0.2).sum()),
      "; failing only on null/partner/sign:", list(o.index[o.rho.abs() >= 0.4]))
print("A1 unadjusted vs PO1: gained", sorted(set(M.index[pop & ((M[[f'qu_{p}' for p in P3]] < .05).any(axis=1) | A2 | B | Cx)]) - mine),
      "lost", sorted(mine - set(M.index[pop & ((M[[f'qu_{p}' for p in P3]] < .05).any(axis=1) | A2 | B | Cx)])))
print("Tier A with alt phenotype model:", int((pop & ((M[[f'qalt_{p}' for p in P3]] < .05).any(axis=1) | A2)).sum()),
      "| Tier A with joint BH:", int((pop & (A1j | A2)).sum()), "| Tier A with PO2:", int((pop & ((M[[f'qqc_{p}' for p in P3]] < .05).any(axis=1) | A2)).sum()))

# ---------------- t(nu) spill-over estimate for the 0.10 threshold
rng = np.random.default_rng(31)
obs = M.top_share[M.top_share < 0.075].values
print(f"\n=== t(nu) body fit on the {len(obs)} components with share < 0.075 (truncated likelihood, histogram density)")
bins = np.logspace(np.log10(0.004), np.log10(0.075), 25)
best = None
for nu in [4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 1000]:
    A = rng.standard_t(nu, (40000, 276)) if nu < 1000 else rng.standard_normal((40000, 276))
    D = A - A.mean(1, keepdims=True); D2 = D ** 2; ts = D2.max(1) / D2.sum(1)
    pin = np.mean(ts < 0.075); h, _ = np.histogram(ts[ts < 0.075], bins=bins, density=True)
    idx = np.clip(np.digitize(obs, bins) - 1, 0, len(h) - 1); ll = np.sum(np.log(np.maximum(h[idx], 1e-6)))
    nbody = len(obs) / pin; spill = nbody * np.mean(ts >= 0.10); spill5 = nbody * np.mean(ts >= 0.5)
    print(f"  nu {nu:5d}: loglik {ll:8.1f}; P(<0.075) {pin:.3f}; implied body {nbody:.0f}; expected body comps >= 0.10: {spill:.1f}; >= 0.5: {spill5:.2f}")
    if best is None or ll > best[1]: best = (nu, ll, spill)
print("  best nu", best[0], "-> expected population-level components above 0.10:", round(best[2], 1), "of the 76 few-donor")

# ---------------- technical covariates (own code)
don = list(C.columns) if False else [l.strip() for l in open(f"{ROS}/sda/md_samples.txt") if l.strip()]
qc = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str); qc = qc.loc[don]
S = pd.read_csv(f"{ROS}/sda/samples_final.csv", index_col=0); S.columns = S.columns.astype(str); S = S.drop(index=520)
for k in (547, 548): S.loc[k] = np.load(f"{cfg.AUD}/c547/C{k}_A.npy")
cts = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
tot = qc[[f"nuclei_{c}" for c in cts]].sum(1)
covs = {"lognuc": np.log(qc.nuclei.values)} | {c: (qc[f"nuclei_{c}"] / tot).values for c in cts}
T = pd.DataFrame(index=M.index)
for nm, v in covs.items():
    rr = [ss.spearmanr(S.loc[k].values, v) for k in M.index]
    T[f"rho_{nm}"] = [x[0] for x in rr]; T[f"q_{nm}"] = bh([x[1] for x in rr])
T["maxcomp"] = T[[f"rho_{c}" for c in cts]].abs().max(1)
d = M.cls == "distributed"
print(f"\n=== technical: distributed with log-nuclei q<0.05: {int((T.q_lognuc[d] < .05).sum())}/{int(d.sum())}; composition |rho|>=0.3: {int((T.maxcomp[d] >= .3).sum())}")
tr = (T.maxcomp >= .3) | (T.rho_lognuc.abs() >= .3)
print(f"prioritised tracking nuclei or composition |rho|>=0.3: {int((tr & pri).sum())}/{int(pri.sum())}; Tier A composition |rho|>=0.3: {int(((T.maxcomp >= .3) & tierA).sum())}; "
      f"C277 rho Inh {T.rho_Inh[277]:+.2f} Exc {T.rho_Exc[277]:+.2f} Oli {T.rho_Oli[277]:+.2f}")
print(f"B-only tracking nuclei/composition |rho|>=0.3: {int((tr & pri & ~(A1 | A2)).sum())}/{int((pri & ~(A1 | A2)).sum())}")

# ---------------- identity / proteome flags of top donors (labels only)
key = pd.read_csv(f"{cfg.PRIVATE}/donor_key.csv"); lab = dict(zip(key.projid.astype(str), key.label))
cat = np.load(f"{cfg.WORK}/verify_protein_influence/cache/prot_cat.npy")
conc = M.cls == "concentrated"
fl = {k: lab.get(don[int(M.top_idx[k])], "") for k in M.index[conc]}
print("\n=== concentrated with identity-flagged top donor:", sum(1 for v in fl.values() if v), "| mislabelled:", {k: v for k, v in fl.items() if v in ("RNA-1", "RNA-2", "CHIP-1", "CHIP-2")})
print("concentrated with top donor fed another person's proteome (cat 2/3):", int(np.isin(cat[M.top_idx[conc].astype(int).values], [2, 3]).sum()),
      "| misfed donors total:", int(np.isin(cat, [2, 3]).sum()))

# ---------------- Data S1 additions
add = pd.read_csv(f"{ST}/out/DataS1_additions_draft.csv", index_col=0, dtype=str, keep_default_na=False)
add.index = add.index.astype(int)   # pandas >= 3 applies dtype=str to the index too; the record ran pandas 2 (integer index)
chk = {}
chk["top_donor_share"] = np.allclose(add.top_donor_share.astype(float), M.top_share.round(4), atol=6e-5)
chk["participation_ratio"] = np.allclose(add.participation_ratio.astype(float), M.PR, atol=0.06)
chk["class"] = (add.concentration_class.values == M.cls.values).all()
sp = add["SEA-AD_spearman"].replace("", np.nan).astype(float)
chk["spearman"] = bool(((sp - M.rho).abs() < 6e-5)[M.rho.notna()].all() and sp.isna().equals(M.rho.isna()))
for col, ph in [("Sex_adj_q", "Sex"), ("ceradsc_adj_q", "ceradsc"), ("cogdx_adj_q", "cogdx")]:
    v = add[col].replace("", np.nan).astype(float)
    chk[col] = bool(v.notna().equals(M[f"q_{ph}"] < .05) and np.allclose(v.dropna(), M.loc[v.notna(), f"q_{ph}"], rtol=1e-9))
chk["prioritised"] = set(add.index[add.prioritised_2026 == "yes"]) == mine
pf = pd.read_csv(f"{ST}/out/01_master_547.csv", index_col=0)
chk["protein_PIP05"] = (add.protein_PIP05_features.astype(int).values == M.proteins.values).all()
print("\n=== Data S1 additions vs verify values:", chk)
ds = pd.read_csv(DS1, index_col=0)
pcols = ds[["Top_Positive_proteins", "Top_Negative_proteins", "Protein_Pathway"]].notna().any(axis=1)
print("protein PIP05>0 rows == rows with protein columns filled:", set(M.index[M.proteins > 0]) == set(ds.index[pcols]), int(pcols.sum()))
src = open(DS1, encoding="utf-8").read().split("\n"); full = open(f"{ST}/out/DataS1_with_additions_draft.csv", encoding="utf-8").read().split("\n")
print("original lines are exact prefixes of the full draft:", all(f.startswith(s + ",") for s, f in zip(src[:-1], full[:-1])), len(src), len(full))
sr = add.score_r_corrected_protein.astype(float)
print("score_r_corrected_protein: min", sr.min(), "at C", sr.idxmin(), "| rows < 0.99:", sorted(sr.index[sr < 0.99]))
flags = add.protein_flag
for f in ["P1", "P2", "P3", "P4"]:
    print(f"  {f}: {sorted(flags.index[flags.str.contains(f)])}")
pi = pd.read_csv(f"{R2}/protein_influence/results/consensus_547_scenarios.csv").set_index("component_id")
vp = pd.read_csv(f"{R2}/verify_protein_influence/results/consensus_fit_full_r1_mf.csv").set_index("component_id")
p1 = sorted(flags.index[flags.str.contains("P1")])
print("  P1 score r (corrected) range impl1:", round(pi.loc[p1, "V3_d_r"].min(), 4), "-", round(pi.loc[p1, "V3_d_r"].max(), 4),
      "| impl2:", round(vp.loc[p1, "d_r"].min(), 4))
p2 = sorted(flags.index[flags.str.contains("P2")])
print("  P2 r impl1:", pi.loc[p2, "V3_d_r"].round(3).to_dict(), "| impl2:", vp.loc[p2, "d_r"].round(3).to_dict())
print("  C544/C506 impl1 vs impl2:", {k: (round(pi.loc[k, 'V3_d_r'], 3), round(vp.loc[k, 'd_r'], 3), round(vp.loc[k, 'cal_r'], 3)) for k in (544, 506)})
M.join(T).to_csv(f"{V}/private/v07_master.csv")
