"""Draft Data S1 additions (547 rows, joined by Component ID; the existing Data S1 text is kept byte-for-byte).

New columns
  top_donor_share, participation_ratio, concentration_class      (concentration; st_common definition)
  SEA-AD_spearman                                                 (rank-based concordance at the published partner)
  Sex_adj_q, ceradsc_adj_q, cogdx_adj_q                           (adjusted PO1 BH q, shown when < 0.05, as Data S1 does)
  prioritised_2026, prioritisation_support                        (05_prioritise.py)
  protein_PIP05_features, protein_flag, score_r_corrected_protein (protein reliability; fixed-loading re-estimation)
  provenance                                                       (row-level provenance)
Protein flags (from the two independent protein-influence implementations, 2026-10-13):
  P1 'dominant donor fed another person's proteome': run-level members with protein PIP > 0.5 features whose dominant donor
     (top share > 0.5) was fed another person's proteome (categories 2/3) - the 23 components of the assessment;
  P2 'protein-dominated': consensus protein PIP > 0.5 features and the score changes when the protein rows are corrected
     (fixed-loading r < 0.95 in both implementations) - the 7 components of the assessment;
  P3 'dominant-donor magnitude depends on the protein row': concentrated, mis-fed dominant donor, dominant-donor score ratio
     (corrected / fed) < 0.8 in both implementations - C56, C125, C316, C318;
  P4 'protein features from the mis-mapped input': any other row with consensus protein PIP > 0.5 features.
Writes out/DataS1_additions_draft.csv (new columns only) and out/DataS1_with_additions_draft.csv (all columns)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import csv, io, numpy as np, pandas as pd
from st_common import *

M = pd.read_csv(f"{OUT}/05_master_prioritisation.csv", index_col=0)
comps = list(M.index)
# ---- protein flags
mem = pd.read_csv(f"{R2}/verify_protein_influence/results/members_fit_full_r1_mf.csv",
                  usecols=["component_id", "n_prot_pip05", "dom_cat", "dom_share_base"])
p1 = sorted(mem[(mem.n_prot_pip05 > 0) & (mem.dom_share_base > 0.5) & mem.dom_cat.isin([2, 3])].component_id.unique())
EXPECT23 = [19, 37, 38, 56, 81, 94, 109, 125, 132, 155, 176, 201, 244, 279, 288, 296, 316, 318, 324, 338, 351, 380, 452]
print("P1 (member-level, recomputed):", len(p1), "== assessment list:", p1 == EXPECT23)
pi = pd.read_csv(f"{R2}/protein_influence/results/consensus_547_scenarios.csv").set_index("component_id").loc[comps]
vp = pd.read_csv(f"{R2}/verify_protein_influence/results/consensus_fit_full_r1_mf.csv").set_index("component_id").loc[comps]
p2 = sorted(M.index[(M.proteins > 0) & (pi.V3_d_r < 0.95) & (vp.d_r < 0.95)])
print("P2:", p2, "== assessment's seven:", set(p2) == {356, 425, 434, 533, 490, 183, 521})
ratio_v = vp.d_dom_robz / vp.dom_robz_base
p3 = sorted(M.index[(M.top_share >= 0.5) & M.top_donor_misfed_proteome & (pi.V3_d_dom_score_ratio < 0.8) & (ratio_v < 0.8)])
print("P3:", p3, "== {56,125,316,318}:", set(p3) == {56, 125, 316, 318})
p4 = sorted(set(M.index[M.proteins > 0]) - set(p1) - set(p2) - set(p3))
print("P4 (other rows with protein features):", p4)
FL = {k: [] for k in comps}
for k in p1: FL[k].append("P1 dominant donor fed another person's proteome")
for k in p2: FL[k].append("P2 protein-dominated")
for k in p3: FL[k].append("P3 dominant-donor magnitude depends on the protein row")
for k in p4: FL[k].append("P4 protein features from the mis-mapped input")
print("rows with protein columns filled in Data S1:",
      int(pd.read_csv(DS1, index_col=0)[["Top_Positive_proteins", "Top_Negative_proteins", "Protein_Pathway"]].notna().any(axis=1).sum()),
      "; rows with consensus protein PIP>0.5 features:", int((M.proteins > 0).sum()))

def f3(v, nd=3):
    return "" if pd.isna(v) else f"{v:.{nd}f}"
def qfmt(v):
    return repr(float(v)) if pd.notna(v) and v < 0.05 else ""
add = pd.DataFrame(index=pd.Index(comps, name="Component"))
add["top_donor_share"] = [f3(v, 4) for v in M.top_share]
add["participation_ratio"] = [f3(v, 1) for v in M.PR]
add["concentration_class"] = M.conc_class
add["SEA-AD_spearman"] = [f3(v, 4) for v in M.seaad_spearman]
add["Sex_adj_q"] = [qfmt(v) for v in M.adj_sex_q]
add["ceradsc_adj_q"] = [qfmt(v) for v in M.adj_cerad_q]
add["cogdx_adj_q"] = [qfmt(v) for v in M.adj_cogdx_q]
add["prioritised_2026"] = np.where(M.prioritised, "yes", "")
add["prioritisation_support"] = np.where(M.prioritised, M.support, "")
add["protein_PIP05_features"] = M.proteins.astype(int)
add["protein_flag"] = ["; ".join(FL[k]) for k in comps]
add["score_r_corrected_protein"] = [f3(v, 4) for v in pi.V3_d_r]
add["provenance"] = ["D+A" if k <= 546 else "D+A; new ID (qualifies only under the corrected run count; cluster " + ("1465" if k == 547 else "716") + ")" for k in comps]
add.to_csv(f"{OUT}/DataS1_additions_draft.csv")

# full table: original text kept exactly, new columns appended
src = open(DS1, "rb").read().decode("utf-8").split("\n"); assert src[-1] == ""; src = src[:-1]
hdr = next(csv.reader([src[0]])); newcols = list(add.columns)
buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
lines = [src[0] + "," + ",".join(newcols)]
for ln in src[1:]:
    k = int(next(csv.reader([ln]))[0])
    b = io.StringIO(); csv.writer(b, lineterminator="").writerow([add.loc[k, c] for c in newcols])
    lines.append(ln + "," + b.getvalue())
open(f"{OUT}/DataS1_with_additions_draft.csv", "w", encoding="utf-8").write("\n".join(lines) + "\n")
chk = pd.read_csv(f"{OUT}/DataS1_with_additions_draft.csv", index_col=0, dtype=str, keep_default_na=False)
orig = pd.read_csv(DS1, index_col=0, dtype=str, keep_default_na=False)
assert (chk[orig.columns].values == orig.values).all() and len(chk) == 547
print("full draft written; original 19 columns identical to Data_S1_corrected_547.csv:", True)
print("\ncounts:", add.concentration_class.value_counts().to_dict(), "| prioritised:", int((add.prioritised_2026 == "yes").sum()),
      "| protein_flag non-empty:", int((add.protein_flag != "").sum()),
      {f"P{i}": len(x) for i, x in enumerate([p1, p2, p3, p4], 1)})
print(add.loc[FOCUS + OUTGOING + OTHER_NAMED + p2 + p3].to_string())
