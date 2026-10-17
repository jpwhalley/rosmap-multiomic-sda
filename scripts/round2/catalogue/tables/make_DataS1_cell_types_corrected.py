"""Corrected Data S1: Top_positive_cell_type / Top_negative_cell_type.

1. Rule (unchanged; ros9b_summarize_manuscript.py l.130-150, github/scripts/13_assemble_table_s1.py):
   top positive = argmax of the component's cell-type loadings if any loading > 0, else blank;
   top negative = argmin if any loading < 0, else blank.
   It is applied to the corrected cell-type loadings (cell_type_loadings_corrected.csv), the matrix behind
   the remade panels: each run-level component's cell-type flip is the product of its sample and gene
   flips, so each run's sample x cell type x gene product keeps its sign.
2. Evidence check: a label is shown only where the individual SDA runs back its direction. For component k
   and cell type c, cell_type_evidence_from_runs.csv holds cos = <B_pub[c] A_pub (x) X_pub, M_c> / norms, where
   M_c is the sum of each run's own product (no sign alignment needed). The runs' direction for c, relative
   to the published sample and gene loadings, is sign(cos)*sign(B_pub[c]). A positive label needs that
   direction > 0 and a negative label needs it < 0; otherwise the field is left blank. Blanking never
   contradicts the matrix; it only withholds a label the runs do not support.
3. Rows whose loadings are numerically zero (|loading| < 1e-300 everywhere) get blank labels, like the
   24 exactly-zero rows.

First the script checks that the rule reproduces the published columns from the published loadings. Only
the two columns change; every other byte of every row is kept.
Run: PYTHONDONTWRITEBYTECODE=1 python3 make_DataS1_cell_types_corrected.py
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import csv, io, os, math
HERE = cfg.out_dir(cfg.TABLES)                                                  # tables folder (inputs and outputs)
SRC = cfg.DS1_V22
PUB = f"{cfg.SDA}/treatment_final.csv"       # published loadings (archive)
COR = os.path.join(HERE, "cell_type_loadings_corrected.csv")
EVI = os.path.join(HERE, "cell_type_evidence_from_runs.csv")
CT = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f); cols = next(r)[1:]
        return {int(float(row[0])): dict(zip(cols, map(float, row[1:]))) for row in r}

def labels(v):
    vals = [v[c] for c in CT]; mx, mn = max(vals), min(vals)
    return (CT[vals.index(mx)] if mx > 0 else "", CT[vals.index(mn)] if mn < 0 else "")

pub, cor = load(PUB), load(COR)
runs_dir = {}
for r in csv.DictReader(open(EVI)):
    k, c, b, cs = int(r["component"]), r["celltype"], float(r["B_pub"]), r["cos"]
    if cs not in ("", "nan") and b != 0 and not math.isnan(float(cs)):
        runs_dir[(k, c)] = math.copysign(1, float(cs)) * math.copysign(1, b)

def final(k):
    if max(abs(x) for x in cor[k].values()) < 1e-300 and max(abs(x) for x in pub[k].values()) < 1e-300:
        return ("", ""), "numerically zero"
    p, n = labels(cor[k]); why = []
    if p and runs_dir.get((k, p), 0) <= 0: why.append(f"positive {p} not backed by runs"); p = ""
    if n and runs_dir.get((k, n), 0) >= 0: why.append(f"negative {n} not backed by runs"); n = ""
    return (p, n), "; ".join(why)

text = open(SRC, "rb").read().decode("utf-8"); nl = "\r\n" if "\r\n" in text else "\n"
lines = text.split(nl); trailing = lines[-1] == ""; lines = lines[:-1] if trailing else lines
hdr = next(csv.reader([lines[0]])); ip, ineg = hdr.index("Top_positive_cell_type"), hdr.index("Top_negative_cell_type")
assert all((lambda f: (f[ip], f[ineg]) == labels(pub[int(f[0])]))(next(csv.reader([ln]))) for ln in lines[1:]), "rule does not reproduce published"
print(f"rule check: reproduces all {len(lines)-1} published rows from the published loadings")

out = [lines[0]]; log = []
for ln in lines[1:]:
    f = next(csv.reader([ln])); k = int(f[0]); (p, n), why = final(k)
    if (f[ip], f[ineg]) != (p, n): log.append([k, f[ip], f[ineg], p, n, why])
    g = list(f); g[ip], g[ineg] = p, n
    buf = io.StringIO(); csv.writer(buf, lineterminator="").writerow(g); out.append(buf.getvalue())
open(os.path.join(HERE, "Data_S1_cell_types_corrected.csv"), "wb").write((nl.join(out) + (nl if trailing else "")).encode("utf-8"))
with open(os.path.join(HERE, "Data_S1_cell_type_changes.csv"), "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["Component", "published_top_positive", "published_top_negative", "corrected_top_positive", "corrected_top_negative", "note"])
    w.writerows(log)
print(f"rows changed: {len(log)} of {len(lines)-1}")
