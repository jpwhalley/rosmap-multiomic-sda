"""Ranking of the five retained components (C74, C278, C172, C277, C21) against the 547-component catalogue.

Read-only on every input. Writes aggregates only (component-level q/p, loadings, labels); no donor IDs or per-donor values.

Sources (independent where possible):
  DS1  = Data S1 v23 ($G_ROS_DATA_S1); q columns blank when q >= 0.05
  PO   = covariates/po_tests_547.csv (builder; U = published MWU/KW, PO1 = adjusted proportional odds)
  VPO  = verify_covariates/vc_po_547.csv + vc_unadj.csv (verifier, own code)
  OWN  = unadjusted MWU/KW recomputed here from the deposited scores + clinical file (third derivation)
  ST   = revision_analyses/structure/out/05_master_prioritisation.csv (builder: class, SEA-AD, GWAS, support)
  VST  = revision_analyses/verify_structure/out/{v04_seaad_547,v03_gwas_547}.csv (verifier)
  CT   = deposit V2 cell_type.csv ($G_ROS_DEPOSIT_DIR; 547 x 6, corrected cell-type loadings)
  CONS = $G_ROS_AUDIT_DIR/consensus_rna.csv (equal-member-weighting consensus cell-type vector; robustness)
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, json
os.environ.update(OMP_NUM_THREADS=cfg.THREADS, OPENBLAS_NUM_THREADS=cfg.THREADS, VECLIB_MAXIMUM_THREADS=cfg.THREADS, MKL_NUM_THREADS=cfg.THREADS)
import numpy as np, pandas as pd
from scipy import stats as ss

OUT = cfg.out_dir(cfg.WORK, "selection_review", "ranking_of_the_five")
RA = cfg.WORK
B = cfg.ROS
AUD = cfg.AUD
DS1P = cfg.DS1_V23
CTP = f"{cfg.DEPOSIT}/cell_type.csv"
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
NEU = {"Exc", "Inh"}
PH = ["Sex", "ceradsc", "cogdx"]
FIVE = [74, 278, 172, 277, 21]
log = open(f"{OUT}/01_rank.log", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(r, 1.0); return out

# ------------------------------------------------------------------ load
ds1 = pd.read_csv(DS1P, index_col=0)
P("DS1 rows", len(ds1), "| columns", list(ds1.columns)[:18])
po = pd.read_csv(f"{RA}/covariates/po_tests_547.csv", header=[0, 1], index_col=0)
po.index = po.index.astype(int)
vpo = pd.read_csv(f"{RA}/verify_covariates/vc_po_547.csv")
vun = pd.read_csv(f"{RA}/verify_covariates/vc_unadj.csv", index_col=0)
st = pd.read_csv(f"{RA}/revision_analyses/structure/out/05_master_prioritisation.csv", index_col=0)
vsea = pd.read_csv(f"{RA}/revision_analyses/verify_structure/out/v04_seaad_547.csv", index_col=0)
vgw = pd.read_csv(f"{RA}/revision_analyses/verify_structure/out/v03_gwas_547.csv", index_col=0)
ct = pd.read_csv(CTP, index_col=0); ct.index = ct.index.astype(int)
cons = pd.read_csv(f"{AUD}/consensus_rna.csv", index_col=0)
idx = ds1.index
for name, d in [("PO", po), ("ST", st), ("VSEA", vsea), ("VGWAS", vgw), ("CT", ct), ("VUNADJ", vun)]:
    P(f"{name}: n {len(d)}; same 547 IDs as DS1: {set(d.index) == set(idx)}")

# ------------------------------------------------------------------ OWN unadjusted tests (third derivation)
S = pd.read_csv(f"{B}/sda/samples_final.csv", index_col=0); S.columns = S.columns.astype(str)
S = S.drop(index=520)
for k in (547, 548):
    S.loc[k] = np.load(f"{AUD}/c547/C{k}_A.npy")
clin = pd.read_csv(f"{B}/oth/meta/ROSMAP_clinical.csv", dtype={"projid": str})
clin.index = clin.projid.astype(float).astype(int).astype(str); clin = clin.loc[S.columns]
male = clin.msex.values == 1; cer = clin.ceradsc.map({1: 1, 2: 1, 3: 0, 4: 0}).values; cog = clin.cogdx.values
own_p = {}
for k in S.index:
    v = S.loc[k].values
    own_p[k] = (ss.mannwhitneyu(v[male], v[~male], alternative="two-sided").pvalue,
                ss.mannwhitneyu(v[cer == 1], v[cer == 0], alternative="two-sided").pvalue,
                ss.kruskal(*[v[cog == g] for g in (1, 2, 4)]).pvalue)
own = pd.DataFrame(own_p, index=["p_Sex", "p_ceradsc", "p_cogdx"]).T
for ph in PH:
    own[f"q_{ph}"] = bh(own[f"p_{ph}"].values)
P("OWN unadjusted: donors", S.shape[1], "| n male", int(male.sum()), "| CERAD AD", int((cer == 1).sum()), "| cogdx 1/2/4", [int((cog == g).sum()) for g in (1, 2, 4)])

# ------------------------------------------------------------------ assemble one table
T = pd.DataFrame(index=idx)
for ph in PH:
    T[f"{ph}_unadj_p_PO"] = po[(ph, "U_p")].reindex(idx)
    T[f"{ph}_unadj_q_PO"] = po[(ph, "U_q")].reindex(idx)
    T[f"{ph}_unadj_p_OWN"] = own[f"p_{ph}"].reindex(idx)
    T[f"{ph}_unadj_q_OWN"] = own[f"q_{ph}"].reindex(idx)
    T[f"{ph}_unadj_q_VC"] = vun[f"q_{ph}"].reindex(idx)
    T[f"{ph}_unadj_q_DS1"] = ds1[ph]
    T[f"{ph}_adj_p_PO"] = po[(ph, "PO1_p")].reindex(idx)
    T[f"{ph}_adj_q_PO"] = po[(ph, "PO1_q")].reindex(idx)
    vv = vpo[(vpo.pheno == ph) & (vpo.model == "M1")].set_index("comp")
    T[f"{ph}_adj_p_VC"] = vv.p.reindex(idx)
    T[f"{ph}_adj_q_VC"] = vv.q.reindex(idx)
    T[f"{ph}_adj_q_DS1"] = ds1[f"{ph}_adj_q"]
    bcol = [c for c in po[ph].columns if c.startswith("PO1_beta")][0]
    T[f"{ph}_adj_beta_PO"] = po[(ph, bcol)].reindex(idx)

# cross-source agreement
P("\n=== Cross-source agreement")
for ph in PH:
    for tag, a, b in [("unadj q PO vs OWN", "unadj_q_PO", "unadj_q_OWN"), ("unadj q PO vs VC", "unadj_q_PO", "unadj_q_VC"),
                      ("adj q PO vs VC", "adj_q_PO", "adj_q_VC"), ("adj p PO vs VC", "adj_p_PO", "adj_p_VC")]:
        d = (T[f"{ph}_{a}"] - T[f"{ph}_{b}"]).abs()
        rel = (np.log10(T[f"{ph}_{a}"]) - np.log10(T[f"{ph}_{b}"])).abs()
        P(f"  {ph:8s} {tag:20s} max|diff| {d.max():.2e}; max|log10 ratio| {rel.max():.2e}")
    for kind in ("unadj", "adj"):
        shown = T[f"{ph}_{kind}_q_DS1"].dropna()
        src = T[f"{ph}_{kind}_q_PO"]
        calls = set(src.index[src < 0.05])
        P(f"  {ph:8s} DS1 {kind} q shown {len(shown)}; PO calls {len(calls)}; same set {set(shown.index) == calls}; "
          f"max|DS1 - PO| {float((shown - src.loc[shown.index]).abs().max()):.2e}")

# ------------------------------------------------------------------ structure / SEA-AD / GWAS
T["top_donor_share_DS1"] = ds1.top_donor_share
T["top_share_ST"] = st.top_share.reindex(idx)
T["class_DS1"] = ds1.concentration_class
T["class_ST"] = st.conc_class.reindex(idx)
T["prioritised_DS1"] = ds1.prioritised_2026.eq("yes")
T["prioritised_ST"] = st.prioritised.reindex(idx).astype(bool)
T["support_DS1"] = ds1.prioritisation_support
T["omics_count"] = ds1.omics_count
T["seaad_partner"] = ds1["SEA-AD_comp"]
T["seaad_r_DS1"] = ds1["SEA-AD_corr"]; T["seaad_rho_DS1"] = ds1["SEA-AD_spearman"]
T["seaad_r_VST"] = vsea.r.reindex(idx); T["seaad_rho_VST"] = vsea.rho.reindex(idx)
T["seaad_nullp_ST"] = st.nullA_p_J.reindex(idx); T["seaad_nullp_VST"] = vsea.null_p.reindex(idx)
T["seaad_partner_top_share_ST"] = st.seaad_partner_top_share.reindex(idx)
T["seaad_partner_top_share_VST"] = vsea.partner_top_share.reindex(idx)
T["seaadB_ST"] = st.support_B_seaad.reindex(idx).astype(bool)
T["seaadB_VST"] = vsea.B.reindex(idx).fillna(False).astype(bool)
r, rho = T.seaad_r_DS1, T.seaad_rho_DS1
T["seaad_corr_part_DS1"] = (r.abs() >= 0.4) & (rho.abs() >= 0.4) & (np.sign(r) == np.sign(rho))
# recomputed full rule from builder columns
T["seaadB_recomputed"] = (T.seaad_corr_part_DS1 & (T.seaad_nullp_ST < 0.01) & (T.seaad_partner_top_share_ST < 0.5))
T["gwas_min_p_auto_ST"] = st.gwas_min_p_autosomal.reindex(idx)
T["gwas_min_p_auto_VST"] = vgw.min_p_auto.reindex(idx)
T["gwas_min_p_all_VST"] = vgw.min_p.reindex(idx)
T["gwas_top_chr_VST"] = vgw.top_chr.reindex(idx)
T["gwas_lambda_VST"] = vgw.lam.reindex(idx)
T["gwas_lambda_ST"] = st.gwas_lambda.reindex(idx)
T["gwasA2_ST"] = st.support_A2_gwas.reindex(idx).astype(bool)
P("\n=== Structure / SEA-AD / GWAS agreement")
P("  class DS1 == ST:", int((T.class_DS1 == T.class_ST).sum()), "/ 547; counts", T.class_DS1.value_counts().to_dict())
P("  top share |DS1 - ST| max (DS1 at 4 dp):", float((T.top_donor_share_DS1 - T.top_share_ST).abs().max()))
P("  prioritised DS1 == ST:", int((T.prioritised_DS1 == T.prioritised_ST).sum()), "; DS1 n", int(T.prioritised_DS1.sum()))
P("  SEA-AD r |DS1 - VST| max:", float((T.seaad_r_DS1 - T.seaad_r_VST).abs().max()), "; rho:", float((T.seaad_rho_DS1 - T.seaad_rho_VST).abs().max()))
P("  SEA-AD B: ST n", int(T.seaadB_ST.sum()), "; VST n", int(T.seaadB_VST.sum()), "; recomputed n", int(T.seaadB_recomputed.sum()),
  "; ST==VST", int((T.seaadB_ST == T.seaadB_VST).sum()), "; ST==recomputed", int((T.seaadB_ST == T.seaadB_recomputed).sum()))
dd = T.index[T.seaadB_ST != T.seaadB_VST]
if len(dd):
    P("  B disagreements ST vs VST:", T.loc[dd, ["seaad_r_DS1", "seaad_rho_DS1", "seaad_nullp_ST", "seaad_nullp_VST", "seaad_partner_top_share_ST", "seaadB_ST", "seaadB_VST"]].to_string())
P("  B pass with only the |r|,|rho|>=0.4 same-sign part (DS1 columns):", int(T.seaad_corr_part_DS1.sum()))
P("  GWAS autosomal min p |log10 ST - VST| max:", float((np.log10(T.gwas_min_p_auto_ST) - np.log10(T.gwas_min_p_auto_VST)).abs().max()))

# ------------------------------------------------------------------ cell types
ct = ct.reindex(idx)
A = ct[CTS].abs()
tot2 = (ct[CTS] ** 2).sum(axis=1)
T["ct_Ast"], T["ct_Oli"], T["ct_Inh"], T["ct_OPC"], T["ct_Exc"], T["ct_Mic"] = [ct[c] for c in CTS]
T["ct_max_abs"] = A.max(axis=1)
T["dom_ct"] = A.idxmax(axis=1).where(T.ct_max_abs > 1e-12)
srt = np.sort(A.values, 1)
T["second_ct"] = pd.Series([CTS[j] for j in np.argsort(A.values, 1)[:, -2]], index=idx).where(T.ct_max_abs > 1e-12)
T["dom_over_second"] = srt[:, -1] / np.where(srt[:, -2] > 0, srt[:, -2], np.nan)
T["neuron_sq_share"] = ((ct.Exc ** 2 + ct.Inh ** 2) / tot2).where(tot2 > 0)
T["dom_sq_share"] = (A.max(axis=1) ** 2 / tot2).where(tot2 > 0)
T["all_same_sign"] = (np.sign(ct[CTS]).replace(0, np.nan).nunique(axis=1) == 1)
# Data S1 labels, recomputed with the published rule (argmax if any > 0, argmin if any < 0)
T["top_pos_rule"] = ct[CTS].idxmax(axis=1).where((ct[CTS] > 0).any(axis=1) & (T.ct_max_abs > 1e-12))
T["top_neg_rule"] = ct[CTS].idxmin(axis=1).where((ct[CTS] < 0).any(axis=1) & (T.ct_max_abs > 1e-12))
T["Top_pos_DS1"] = ds1.Top_positive_cell_type; T["Top_neg_DS1"] = ds1.Top_negative_cell_type
def ds1_dom(row):
    labs = [x for x in (row.Top_pos_DS1, row.Top_neg_DS1) if isinstance(x, str)]
    if not labs: return np.nan
    return max(labs, key=lambda c: abs(row[f"ct_{c}"]))
T["dom_ct_DS1labels"] = T.apply(ds1_dom, axis=1)
# equal-member-weighting consensus (historical IDs = 547 IDs except C520 retired; C547/C548 absent)
cb = cons.cons_b.apply(json.loads)
T["dom_ct_cons"] = pd.Series({k: CTS[int(np.argmax(np.abs(v)))] for k, v in cb.items()}).reindex(idx)
T["neuron_sq_share_cons"] = pd.Series({k: (v[2] ** 2 + v[4] ** 2) / max(sum(x * x for x in v), 1e-300) for k, v in cb.items()}).reindex(idx)
# definitions
T["neuron_led_absmax"] = T.dom_ct.isin(NEU)
T["neuron_led_sq50"] = T.neuron_sq_share > 0.5
T["neuron_led_DS1labels"] = T.dom_ct_DS1labels.isin(NEU)
T["neuron_led_cons"] = T.dom_ct_cons.isin(NEU)
P("\n=== Cell-type labels vs corrected loadings (release_v2 cell_type.csv)")
for side, a, b in [("pos", "top_pos_rule", "Top_pos_DS1"), ("neg", "top_neg_rule", "Top_neg_DS1")]:
    shown = T[b].notna()
    P(f"  Top_{side}: DS1 shown {int(shown.sum())}; equal to rule on shown {int((T.loc[shown, a] == T.loc[shown, b]).sum())}; "
      f"rule non-blank but DS1 blank {int((T[a].notna() & ~shown).sum())}")
for d in ["neuron_led_absmax", "neuron_led_sq50", "neuron_led_DS1labels", "neuron_led_cons"]:
    P(f"  {d}: all {int(T[d].sum())}; distributed {int((T[d] & (T.class_DS1 == 'distributed')).sum())}; prioritised {int((T[d] & T.prioritised_DS1).sum())}")
cross = pd.crosstab(T.neuron_led_absmax, T.neuron_led_sq50)
P("  absmax vs sq50 crosstab:\n" + cross.to_string())
P("  absmax vs DS1-label rule disagree:", int((T.neuron_led_absmax != T.neuron_led_DS1labels).sum()),
  "| absmax vs equal-weight consensus disagree (where consensus exists):",
  int(((T.neuron_led_absmax != T.neuron_led_cons) & T.dom_ct_cons.notna()).sum()))
P("  dominant cell type counts (absmax):", T.dom_ct.value_counts().to_dict())

T.to_csv(f"{OUT}/component_table_547.csv")

# ------------------------------------------------------------------ rankings
SHOW = ["class_DS1", "prioritised_DS1", "support_DS1", "omics_count", "dom_ct", "second_ct", "dom_over_second", "neuron_sq_share",
        "Top_pos_DS1", "Top_neg_DS1", "seaad_partner", "seaad_r_DS1", "seaad_rho_DS1", "seaad_nullp_ST", "seaad_partner_top_share_ST", "seaadB_ST"]
sets = {"all547": T.index, "distributed193": T.index[T.class_DS1 == "distributed"], "prioritised113": T.index[T.prioritised_DS1]}
P("\n=== set sizes", {k: len(v) for k, v in sets.items()})
rows = []
for ph in PH:
    for kind in ("unadj", "adj"):
        pc, qc = f"{ph}_{kind}_p_PO", f"{ph}_{kind}_q_PO"
        for sname, sidx in sets.items():
            sub = T.loc[sidx].sort_values(pc).head(15)
            for rank, (k, row) in enumerate(sub.iterrows(), 1):
                d = dict(phenotype=ph, test=kind, set=sname, rank=rank, component=k, p=row[pc], q=row[qc],
                         q_DS1=row[f"{ph}_{kind}_q_DS1"], q_other=row[f"{ph}_{kind}_q_OWN" if kind == "unadj" else f"{ph}_adj_q_VC"],
                         adj_beta=row[f"{ph}_adj_beta_PO"])
                d.update({c: row[c] for c in SHOW})
                rows.append(d)
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/top15_rankings.csv", index=False)

def fmt(x):
    if isinstance(x, (float, np.floating)):
        if np.isnan(x): return ""
        return f"{x:.3g}" if (abs(x) < 1e-3 or abs(x) >= 1e4) else f"{x:.4g}"
    return str(x)
for ph in PH:
    for kind in ("unadj", "adj"):
        for sname in sets:
            sub = R[(R.phenotype == ph) & (R.test == kind) & (R.set == sname)]
            P(f"\n--- {ph} | {kind} | {sname}")
            P(sub[["rank", "component", "q", "q_DS1", "q_other", "class_DS1", "prioritised_DS1", "dom_ct", "second_ct", "neuron_sq_share",
                   "Top_pos_DS1", "Top_neg_DS1", "seaad_r_DS1", "seaad_rho_DS1", "seaadB_ST"]].map(fmt).to_string(index=False))

# ------------------------------------------------------------------ neuron-led sub-rankings
P("\n\n=== Neuron-led rankings (definition in column header); SEA-AD = criterion B (ST, == VST)")
nrows = []
for ph in ("ceradsc", "cogdx", "Sex"):
    for kind in ("unadj", "adj"):
        pc, qc = f"{ph}_{kind}_p_PO", f"{ph}_{kind}_q_PO"
        for defn in ("neuron_led_absmax", "neuron_led_sq50", "neuron_led_DS1labels", "neuron_led_cons"):
            for scope, mask in [("all", T[defn]), ("all+SEA-AD", T[defn] & T.seaadB_ST),
                                ("distributed", T[defn] & (T.class_DS1 == "distributed")),
                                ("distributed+SEA-AD", T[defn] & (T.class_DS1 == "distributed") & T.seaadB_ST),
                                ("prioritised", T[defn] & T.prioritised_DS1),
                                ("prioritised+SEA-AD", T[defn] & T.prioritised_DS1 & T.seaadB_ST)]:
                sub = T.loc[mask].sort_values(pc).head(8)
                for rank, (k, row) in enumerate(sub.iterrows(), 1):
                    nrows.append(dict(phenotype=ph, test=kind, definition=defn, scope=scope, rank=rank, component=k, p=row[pc], q=row[qc],
                                      ratio_to_rank1=row[qc] / sub[qc].iloc[0], class_=row.class_DS1, prioritised=row.prioritised_DS1,
                                      dom_ct=row.dom_ct, second_ct=row.second_ct, neuron_sq_share=row.neuron_sq_share,
                                      ct_Exc=row.ct_Exc, ct_Inh=row.ct_Inh, Top_pos_DS1=row.Top_pos_DS1, Top_neg_DS1=row.Top_neg_DS1,
                                      seaad_r=row.seaad_r_DS1, seaad_rho=row.seaad_rho_DS1, seaadB=row.seaadB_ST, adj_beta=row[f"{ph}_adj_beta_PO"]))
N = pd.DataFrame(nrows); N.to_csv(f"{OUT}/neuron_led_rankings.csv", index=False)
for ph in ("cogdx", "ceradsc"):
    for kind in ("unadj", "adj"):
        for scope in ("all", "all+SEA-AD", "distributed", "distributed+SEA-AD", "prioritised+SEA-AD"):
            sub = N[(N.phenotype == ph) & (N.test == kind) & (N.definition == "neuron_led_absmax") & (N.scope == scope)].head(6)
            P(f"\n--- neuron-led (absmax) | {ph} | {kind} | {scope}")
            P(sub[["rank", "component", "p", "q", "ratio_to_rank1", "class_", "prioritised", "dom_ct", "second_ct", "neuron_sq_share",
                   "seaad_r", "seaad_rho", "seaadB"]].map(fmt).to_string(index=False))

# top neuron-led by each definition (rank 1 and runner-up) in compact form
P("\n=== Rank-1 and runner-up under each neuron-led definition")
for ph in ("cogdx", "ceradsc"):
    for kind in ("unadj", "adj"):
        for defn in ("neuron_led_absmax", "neuron_led_sq50", "neuron_led_DS1labels", "neuron_led_cons"):
            for scope in ("all", "all+SEA-AD", "prioritised+SEA-AD"):
                sub = N[(N.phenotype == ph) & (N.test == kind) & (N.definition == defn) & (N.scope == scope)].head(3)
                P(f"  {ph:7s} {kind:5s} {defn:22s} {scope:20s} " + " | ".join(f"C{int(c)} q={q:.4g}" for c, q in zip(sub.component, sub.q)))

# ------------------------------------------------------------------ named components
P("\n=== Named components")
NAMED = [74, 278, 545, 172, 134, 542, 277, 21, 31, 32, 48, 169]
cols = ["class_DS1", "top_donor_share_DS1", "prioritised_DS1", "support_DS1", "omics_count", "dom_ct", "second_ct", "dom_over_second",
        "neuron_sq_share", "dom_ct_cons", "Top_pos_DS1", "Top_neg_DS1", "ct_Ast", "ct_Oli", "ct_Inh", "ct_OPC", "ct_Exc", "ct_Mic",
        "seaad_partner", "seaad_r_DS1", "seaad_rho_DS1", "seaad_nullp_ST", "seaad_partner_top_share_ST", "seaadB_ST", "seaadB_VST",
        "gwas_min_p_auto_ST", "gwas_lambda_ST"]
for ph in PH:
    cols += [f"{ph}_unadj_q_PO", f"{ph}_unadj_q_OWN", f"{ph}_unadj_q_DS1", f"{ph}_adj_q_PO", f"{ph}_adj_q_VC", f"{ph}_adj_q_DS1", f"{ph}_adj_beta_PO"]
Nm = T.loc[[k for k in NAMED if k in T.index], cols]
Nm.to_csv(f"{OUT}/named_components.csv")
P(Nm.T.to_string())

# ------------------------------------------------------------------ sex
P("\n=== Sex: all components ranked (top 6), with SEA-AD")
for kind in ("unadj", "adj"):
    sub = T.sort_values(f"Sex_{kind}_p_PO").head(6)
    P(f"  {kind}: " + " | ".join(f"C{k} q={row[f'Sex_{kind}_q_PO']:.4g} ({row.class_DS1}; B={row.seaadB_ST})" for k, row in sub.iterrows()))
    sub = T[T.seaadB_ST].sort_values(f"Sex_{kind}_p_PO").head(4)
    P(f"  {kind} SEA-AD B only: " + " | ".join(f"C{k} q={row[f'Sex_{kind}_q_PO']:.4g}" for k, row in sub.iterrows()))

# ------------------------------------------------------------------ GWAS
P("\n=== GWAS among distributed components")
dist = T[T.class_DS1 == "distributed"]
thr = 5e-8 / 547
for lab, col in [("ST autosomal", "gwas_min_p_auto_ST"), ("VST autosomal", "gwas_min_p_auto_VST"), ("VST any chromosome", "gwas_min_p_all_VST")]:
    s = dist[col].sort_values()
    P(f"  {lab}: < {thr:.3g} (study-wide): {list(s.index[s < thr])}; < 5e-8: {list(s.index[s < 5e-8])}; < 1e-15: {list(s.index[s < 1e-15])}")
    P("    lowest 6: " + " | ".join(f"C{k} {v:.3g}" for k, v in s.head(6).items()))
P("  A2 (study-wide + lambda 0.95-1.05) among distributed:", list(dist.index[dist.gwasA2_ST]))
for cls in ("few-donor", "concentrated"):
    sub = T[T.class_DS1 == cls]
    P(f"  A2 in {cls}: {int(sub.gwasA2_ST.sum())}; autosomal < study-wide: {int((sub.gwas_min_p_auto_ST < thr).sum())}")
P("  C21 lambda ST", T.loc[21, "gwas_lambda_ST"], "VST", T.loc[21, "gwas_lambda_VST"], "; C21 min p auto", T.loc[21, "gwas_min_p_auto_ST"])
P("  C74 min p any chr", T.loc[74, "gwas_min_p_all_VST"], "top chr", T.loc[74, "gwas_top_chr_VST"], "; autosomal", T.loc[74, "gwas_min_p_auto_VST"])
log.close()
