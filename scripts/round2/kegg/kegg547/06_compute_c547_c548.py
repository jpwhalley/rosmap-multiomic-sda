"""Compute KEGG_2021_Human enrichment for the two new components with the validated call."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from kegg_local import enrich, ds1_rule
D = cfg.out_dir(cfg.AUD, "kegg547")
C = f"{cfg.AUD}/c547"
S = cfg.SDA
genes = open(f"{S}/genes.txt").read().split()
BG = json.load(open(f"{D}/pip05_lists_RNA.json"))["background"]
assert genes == BG and len(BG) == 25098
summary = {}
for c in ["C547", "C548"]:
    P = np.load(f"{C}/{c}_P1.npy"); X = np.load(f"{C}/{c}_X1.npy")
    q = [g for g, p in zip(genes, P) if p > 0.5]
    ref = pd.read_csv(f"{C}/{c}_PIPgenes_RNA.csv", index_col=0)
    assert set(q) == set(ref.index) and len(q) == len(ref), "PIP list mismatch with C54x_PIPgenes_RNA.csv"
    near = int(((P > 0.45) & (P < 0.55)).sum())
    res = enrich(q, BG, server_case=True)
    res_cs = enrich(q, BG, server_case=False)
    res.insert(0, "Rank", np.arange(1, len(res) + 1))
    res["Gene_set"] = "KEGG_2021_Human"
    res["Combined Score (Enrichr)"] = -np.log(res["P-value"]) * res["Odds Ratio (Enrichr)"]
    res = res.rename(columns={"Odds Ratio": "Odds Ratio (gseapy, +0.5 corrected)", "Combined Score": "Combined Score (gseapy)"})
    res["-log10(FDR)"] = -np.log10(res["Adjusted P-value"])
    cols = ["Rank", "Gene_set", "Term", "Overlap", "n_overlap", "term_size_in_bg", "P-value", "Adjusted P-value", "-log10(FDR)",
            "Odds Ratio (Enrichr)", "Combined Score (Enrichr)", "Odds Ratio (gseapy, +0.5 corrected)", "Combined Score (gseapy)", "Genes"]
    res = res[cols]
    res.to_csv(f"{D}/{c}_kegg.csv", index=False)
    term, adjp, tied = ds1_rule(res)
    term_cs, adjp_cs, tied_cs = ds1_rule(res_cs)
    sig = res[res["Adjusted P-value"] < 0.05]
    summary[c] = dict(n_pip05=len(q), n_pip_0p45_0p55=near, n_query_in_KEGG=len(set(g.upper() for g in q) & set(g for l in open(f"{D}/KEGG_2021_Human.gmt") for g in l.rstrip("\n").split("\t")[2:])),
                      n_terms_with_overlap=len(res), n_sig_adj05=len(sig), DataS1_value=term, top_adjp=adjp, tied_at_min=tied,
                      case_sensitive_value=term_cs, case_sensitive_adjp=adjp_cs,
                      top10=res.head(10)[["Rank", "Term", "Overlap", "P-value", "Adjusted P-value"]].to_dict("records"))
    print(f"\n{c}: PIP>0.5 RNA genes {len(q)} (matches {c}_PIPgenes_RNA.csv); genes with PIP in (0.45,0.55): {near}")
    print(f"  terms with >=1 hit: {len(res)}; Adj P < 0.05: {len(sig)}")
    print(f"  Data S1 GeneExpression_Pathway = {term!r} (Adj P {adjp:.4g}); tied at min Adj P: {tied}")
    print(f"  case-sensitive sensitivity: {term_cs!r} (Adj P {adjp_cs:.4g})")
    print(res.head(12)[["Rank", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]].to_string(index=False, max_colwidth=70))
json.dump(summary, open(f"{D}/C547_C548_summary.json", "w"), indent=1, default=float)
