"""Validation (b): recompute the Data S1 GeneExpression_Pathway rule for ALL published rows from the
deposit PIP>0.5 lists (fresh local enrichment, no archive input) and compare with the published value."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from kegg_local import enrich, ds1_rule
D = cfg.out_dir(cfg.AUD, "kegg547")
A = f"{cfg.ARCHIVE}/enrichr/all_kegg"
DS1 = cfg.DS1_V22
J = json.load(open(f"{D}/pip05_lists_RNA.json")); BG = J["background"]; L = {int(k): v for k, v in J["lists"].items()}
ds = pd.read_csv(DS1)
t0 = time.time()
rows = []
for comp, pub in zip(ds.Component, ds.GeneExpression_Pathway):
    pub = "" if pd.isna(pub) else str(pub)
    q = L[int(comp)]
    res = enrich(q, BG, server_case=True)
    term, adjp, tied = ds1_rule(res)
    res_cs = enrich(q, BG, server_case=False)
    term_cs, adjp_cs, _ = ds1_rule(res_cs)
    # what the (stale-bug) archive file would have given
    f = f"{A}/{comp}_pathway_encrichment.csv"
    if os.path.exists(f):
        a = pd.read_csv(f, index_col=0); fa = a[a["Adjusted P-value"] < 0.05]
        arch = fa.loc[fa["Adjusted P-value"].idxmin(), "Term"] if len(fa) else ""
    else:
        arch = "<no archive file>"
    rows.append(dict(Component=comp, n_pip05=len(q), published=pub, recomputed=term, recomputed_adjp=adjp,
                     n_tied_at_min=len(tied), tied_terms=";".join(tied), recomputed_case_sensitive=term_cs,
                     archive_rule=arch, agree=pub == term))
el = time.time() - t0
df = pd.DataFrame(rows)
def reason(r):
    if r.agree:
        return "agree (both blank)" if r.published == "" else "agree"
    if r.n_pip05 == 0 and r.published != "":
        return "published value from stale archive file (component has no PIP>0.5 RNA gene)"
    if r.archive_rule == "<no archive file>":
        return "no archive file for this component"
    if r.published == "" and r.archive_rule == r.recomputed:
        return "published blank but archive file + recomputation give a term"
    if r.published == r.archive_rule:
        return "published = archive file, recomputation differs"
    return "other"
df["reason"] = df.apply(reason, axis=1)
df.to_csv(f"{D}/validation_b_dataS1_vs_recomputed.csv", index=False)
print(f"recomputed {len(df)} Data S1 rows (x2 case modes) in {el:.1f}s")
print("published non-blank:", int((df.published != "").sum()), "| recomputed non-blank:", int((df.recomputed != "").sum()))
print(df.reason.value_counts().to_string())
print("disagreements:")
print(df.loc[~df.agree, ["Component", "n_pip05", "published", "recomputed", "archive_rule", "reason"]].to_string(index=False))
print("rows where min Adj P is shared by >1 significant term:", int((df.n_tied_at_min > 1).sum()),
      "| of those, published == our tie-break (smallest raw P):", int(((df.n_tied_at_min > 1) & df.agree).sum()))
print("case-sensitive variant changes the recomputed term in rows:", df.loc[df.recomputed != df.recomputed_case_sensitive, "Component"].tolist())
blank_pub_zero = df[(df.n_pip05 == 0)]
print("components with 0 PIP>0.5 RNA genes:", len(blank_pub_zero), "| published blank:", int((blank_pub_zero.published == "").sum()),
      "| stale archive rule non-blank:", int((blank_pub_zero.archive_rule.isin(['', '<no archive file>']) == False).sum()))
