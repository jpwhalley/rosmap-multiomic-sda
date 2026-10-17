"""Data S1 for the corrected 547-component catalogue.

Starts from Data_S1_cell_types_corrected.csv (cell-type columns already corrected) and:
  1. removes row 520 (it spans 4 of the 10 runs; the original run count credited one of its components
     to the wrong run);
  2. sets Sex / ceradsc / cogdx to the BH values over the 547-component family (clinical_BH_547_family.csv;
     only values < 0.05 are shown, as before);
  3. appends C547 (cluster 1465) and C548 (cluster 716), built by build_DataS1_new_rows_C547_C548.py with
     the original pipeline's conventions (DataS1_new_rows_C547_C548.json).
  4. fills GeneExpression_Pathway for C547/C548 with the published rule (ros9b l.13-36: the Term with the smallest
     Adjusted P-value among terms with Adjusted P-value < 0.05, else blank; ties go to Enrichr rank 1, the smallest
     raw P, as in all 34 tied published rows) from KEGG_2021_Human_C547.csv / _C548.csv. Those tables are
     KEGG_2021_Human enrichments of each component's PIP>0.5 RNA genes against all 25,098 RNA genes, computed
     offline with gseapy 1.2.1 on the Enrichr library and validated against all 507 usable archived enrichments
     and all 546 published pathway cells (kegg/kegg547/; outputs in $G_ROS_AUDIT_DIR/kegg547/).
  5. re-matches SEA-AD_comp / SEA-AD_corr against the corrected SEA-AD catalogue. Counted correctly, the >=5-run rule
     retires SEA-AD 9 and 128 (4 runs each) and adds nothing (143 -> 141, IDs kept). The 15 rows whose partner was 9 or
     128 take their best partner among the remaining 141 (SEAAD_partner_rematch_141.csv: same Pearson / larger-|r|
     method, verified by two independent re-derivations; seaad/02_partner.py, outputs in $G_ROS_AUDIT_DIR/seaad141/).
Every other field keeps its exact text. Run: PYTHONDONTWRITEBYTECODE=1 python3 make_DataS1_547.py
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import csv, io, json, os
HERE = cfg.out_dir(cfg.TABLES)                                                  # tables folder (inputs and outputs)
src = open(os.path.join(HERE, "Data_S1_cell_types_corrected.csv"), "rb").read().decode("utf-8")
lines = src.split("\n"); assert lines[-1] == ""; lines = lines[:-1]
hdr = next(csv.reader([lines[0]])); ix = {h: i for i, h in enumerate(hdr)}
bh = {}
with open(os.path.join(HERE, "clinical_BH_547_family.csv")) as f:
    for r in csv.DictReader(f):
        bh[int(float(r[""] if "" in r else r["Component"]))] = r
new = json.load(open(os.path.join(HERE, "DataS1_new_rows_C547_C548.json")))
def kegg_term(k):
    with open(os.path.join(HERE, f"KEGG_2021_Human_C{k}.csv")) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["Rank"]))
    sig = [r for r in rows if float(r["Adjusted P-value"]) < 0.05]
    if not sig: return ""
    best = min(float(r["Adjusted P-value"]) for r in sig)
    return next(r["Term"] for r in sig if float(r["Adjusted P-value"]) == best)   # first in Enrichr rank order
def fmt_bh(v): v = float(v); return repr(v) if v < 0.05 else ""
def row_text(fields): buf = io.StringIO(); csv.writer(buf, lineterminator="").writerow(fields); return buf.getvalue()
SEAAD_RETIRED = {9, 128}
rematch = {}
with open(os.path.join(HERE, "SEAAD_partner_rematch_141.csv")) as f:
    for r in csv.DictReader(f): rematch[r["rosmap"]] = (r["new_comp"], r["new_corr"])
def seaad_fix(g):
    if g[ix["SEA-AD_comp"]] and int(float(g[ix["SEA-AD_comp"]])) in SEAAD_RETIRED:
        g[ix["SEA-AD_comp"]], g[ix["SEA-AD_corr"]] = rematch[g[0]]
    return g
out = [lines[0]]
for ln in lines[1:]:
    f = next(csv.reader([ln])); k = int(f[0])
    if k == 520: continue
    g = list(f)
    for c in ("Sex", "ceradsc", "cogdx"): g[ix[c]] = fmt_bh(bh[k][c])
    g = seaad_fix(g)
    out.append(ln if g == f else row_text(g))
for k in ("547", "548"):
    r = new[k]; g = [""] * len(hdr); g[0] = k
    for h in hdr[1:]:
        v = r.get(h, "")
        if h in ("Sex", "ceradsc", "cogdx"): v = fmt_bh(bh[int(k)][h])
        elif h == "SEA-AD_corr" and v != "": v = repr(float(v))
        elif h == "SEA-AD_comp" and v != "": v = repr(float(v))
        elif h == "omics_count": v = str(int(v))
        elif h == "GeneExpression_Pathway": v = kegg_term(k)
        g[ix[h]] = "" if v is None else str(v)
    out.append(row_text(seaad_fix(g)))
assert not any(next(csv.reader([ln]))[ix["SEA-AD_comp"]] in ("9.0", "128.0") for ln in out[1:]), "retired SEA-AD partner left"
dst = os.path.join(HERE, "Data_S1_corrected_547.csv")
open(dst, "wb").write(("\n".join(out) + "\n").encode("utf-8"))
print(f"wrote {os.path.basename(dst)}: {len(out)-1} component rows")
