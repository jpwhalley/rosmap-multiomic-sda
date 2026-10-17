"""Near-tie detail for the CERAD, cogdx and sex candidates; GWAS detail for distributed components.
Read-only; aggregates only. Uses component_table_547.csv written by 01_rank.py plus the builder's PO2/PO3, RINT and ST columns."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os
os.environ.update(OMP_NUM_THREADS=cfg.THREADS, OPENBLAS_NUM_THREADS=cfg.THREADS, VECLIB_MAXIMUM_THREADS=cfg.THREADS)
import numpy as np, pandas as pd
OUT = cfg.out_dir(cfg.WORK, "selection_review", "ranking_of_the_five"); RA = cfg.WORK
T = pd.read_csv(f"{OUT}/component_table_547.csv", index_col=0)
po = pd.read_csv(f"{RA}/covariates/po_tests_547.csv", header=[0, 1], index_col=0); po.index = po.index.astype(int)
ri = pd.read_csv(f"{RA}/covariates/all_tests_547.csv", header=[0, 1], index_col=0); ri.index = ri.index.astype(int)
st = pd.read_csv(f"{RA}/revision_analyses/structure/out/05_master_prioritisation.csv", index_col=0)
vgw = pd.read_csv(f"{RA}/revision_analyses/verify_structure/out/v03_gwas_547.csv", index_col=0)
log = open(f"{OUT}/02_ties.log", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]

def block(ph, comps, title):
    rows = []
    for k in comps:
        r = dict(component=k, U_q=po.loc[k, (ph, "U_q")], PO1_q=po.loc[k, (ph, "PO1_q")], PO2_q=po.loc[k, (ph, "PO2_q")],
                 PO3_q=po.loc[k, (ph, "PO3_q")], RINT_M1_q=ri.loc[k, (ph, "M1_q")], RINT_M2_q=ri.loc[k, (ph, "M2_q")])
        other = [x for x in ("Sex", "ceradsc", "cogdx") if x != ph]
        for o in other:
            r[f"{o}_PO1_q"] = po.loc[k, (o, "PO1_q")]
        r.update(dict(dom_ct=T.loc[k, "dom_ct"], second=T.loc[k, "second_ct"], dom_over_second=T.loc[k, "dom_over_second"],
                      neuron_sq_share=T.loc[k, "neuron_sq_share"], Top_pos=T.loc[k, "Top_pos_DS1"], Top_neg=T.loc[k, "Top_neg_DS1"],
                      top_share=st.loc[k, "top_share"], runs=st.loc[k, "true_runs"], members=st.loc[k, "n_members"], RNA_PIP=st.loc[k, "RNA"],
                      seaad_partner=T.loc[k, "seaad_partner"], seaad_r=T.loc[k, "seaad_r_DS1"], seaad_rho=T.loc[k, "seaad_rho_DS1"],
                      seaadB=T.loc[k, "seaadB_ST"], rho_lognuc=st.loc[k, "rho_log nuclei"], max_rho_comp=st.loc[k, "max_abs_rho_composition"],
                      support=T.loc[k, "support_DS1"]))
        rows.append(r)
    D = pd.DataFrame(rows).set_index("component")
    P(f"\n=== {title}")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        P(D.T.to_string())
    D.to_csv(f"{OUT}/ties_{ph}.csv")
    return D

# CERAD: neuron-led + SEA-AD candidates within ~2x of the best, plus C83 (best neuron-led overall, few-donor)
block("ceradsc", [169, 134, 172, 519, 542, 285, 83], "CERAD near-tie: neuron-led (largest |cell-type loading| Exc/Inh) with SEA-AD criterion B, plus C83")
# cogdx: C277 vs other neuron-led and vs the top of the catalogue
block("cogdx", [277, 481, 454, 285, 519, 162, 48, 515, 500], "cogdx: C277 vs neuron-led runners-up and catalogue leaders")
block("Sex", [74, 545, 278], "Sex: the three calls")

# cell-type loadings of the candidates (full precision)
P("\n=== Corrected cell-type loadings (release_v2 cell_type.csv), candidates")
C = T.loc[[74, 278, 545, 172, 169, 134, 542, 519, 285, 162, 277, 481, 454, 21, 48, 515, 500, 83], [f"ct_{c}" for c in CTS] + ["dom_ct", "neuron_sq_share", "dom_ct_cons", "neuron_sq_share_cons"]]
with pd.option_context("display.width", 250, "display.max_columns", 50, "display.precision", 6):
    P(C.to_string())

# how many neuron-led components (absmax) in each set, and with SEA-AD
for nm, m in [("all", T.index == T.index), ("distributed", T.class_DS1 == "distributed"), ("prioritised", T.prioritised_DS1)]:
    P(f"neuron-led {nm}: {int((T.neuron_led_absmax & m).sum())}; +SEA-AD B: {int((T.neuron_led_absmax & m & T.seaadB_ST).sum())}; "
      f"Exc-led {int(((T.dom_ct == 'Exc') & m).sum())}, Inh-led {int(((T.dom_ct == 'Inh') & m).sum())}")
# neuron-led components with any significant cogdx/CERAD call
for ph in ("ceradsc", "cogdx"):
    for kind in ("unadj", "adj"):
        q = T[f"{ph}_{kind}_q_PO"]
        m = T.neuron_led_absmax & (q < 0.05)
        P(f"{ph} {kind} calls among neuron-led: {int(m.sum())} (distributed {int((m & (T.class_DS1 == 'distributed')).sum())}; +SEA-AD {int((m & T.seaadB_ST).sum())})")

# GWAS: distributed components with conventional genome-wide hits
P("\n=== GWAS, distributed components with autosomal min p < 5e-8")
d = T[(T.class_DS1 == "distributed")]
g = vgw.loc[d.index]
sel = g[g.min_p_auto < 5e-8].sort_values("min_p_auto")
P(sel[["min_p_auto", "n5e8_auto", "n1e15", "lam", "lam_auto", "top_chr"]].to_string())
P("expected number of distributed components with a min p < 5e-8 if each GWAS had family-wise 5% at that threshold: 193 x 0.05 =", 193 * 0.05)
P("all 547 with autosomal min p < 5e-8:", int((vgw.min_p_auto < 5e-8).sum()), "; by class:",
  T.loc[vgw.index[vgw.min_p_auto < 5e-8], "class_DS1"].value_counts().to_dict())
P("distributed lambda range (VST):", float(g.lam.min()), float(g.lam.max()), "; C21 n SNPs < 5e-8 auto:", int(g.loc[21, "n5e8_auto"]))
log.close()
