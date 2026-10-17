"""Local (offline) re-implementation of the original ros7ai KEGG call.

Original call (Nov 2024): gp.enrichr(gene_list=PIP>0.5 RNA genes, background=all 25,098 RNA genes,
gene_sets='KEGG_2021_Human', organism='human', outdir=None, no_plot=True, cutoff=0.05).
With a *string* library and a background list, gseapy sends the list + background to the Enrichr
Speedrichr server (/speedrichr/api/backgroundenrich). Here we run the same gseapy.enrichr with the
downloaded KEGG_2021_Human.gmt so gseapy uses its local hypergeometric path (calc_pvalues):
N = |background|, term genes intersected with background, query intersected with background,
one-sided hypergeometric P, BH over terms with >= 1 overlapping gene.
A network guard makes any HTTP attempt raise, so no gene list ever leaves the machine.
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
import requests

def _blocked(*a, **k):
    raise RuntimeError("network blocked in validation/compute (only the library fetch was approved)")
requests.Session.request = _blocked
requests.get = requests.post = _blocked

import gseapy as gp

GMT = cfg.KEGG_GMT

def enrich(gene_list, background, gmt=GMT, cutoff=0.05, server_case=True):
    """Exact original arguments, but gene_sets = local GMT. Returns DataFrame in Enrichr rank order
    (ascending P-value; ties broken by Term name), with Enrichr-style odds ratio added.

    server_case=True reproduces the Enrichr/Speedrichr server: user genes and background are
    upper-cased, so the 3 mixed-case library symbols (C9orf72, RPL17-C18orf32, C8orf44-SGK3) never
    match (found in validation: archived term sizes are 1 smaller for every term carrying them).
    server_case=False = gseapy's case-sensitive local matching (sensitivity check)."""
    if len(gene_list) == 0:
        return pd.DataFrame()
    if server_case:
        gene_list = [str(g).upper() for g in gene_list]
        background = [str(g).upper() for g in background]
    enr = gp.enrichr(gene_list=list(gene_list), background=list(background), gene_sets=gmt,
                     organism="human", outdir=None, no_plot=True, cutoff=cutoff, verbose=False)
    res = enr.results
    if res is None or len(res) == 0:
        return pd.DataFrame()
    res = res.copy()
    res["n_overlap"] = res["Overlap"].str.split("/").str[0].astype(int)
    res["term_size_in_bg"] = res["Overlap"].str.split("/").str[1].astype(int)
    # Enrichr's (speedrichr's) odds ratio, no Haldane-Anscombe correction (gseapy uses +0.5)
    N = len(set(background)); k = len(set(gene_list) & set(background))
    x = res.n_overlap; m = res.term_size_in_bg
    res["Odds Ratio (Enrichr)"] = (x * (N - m - k + x)) / np.maximum((m - x) * (k - x), 1)
    res = res.sort_values(["P-value", "Term"], kind="mergesort").reset_index(drop=True)
    return res

def ds1_rule(res):
    """ros9b rule: Term with smallest Adjusted P-value among Adj P < 0.05 (else blank).
    Returns (term, adjp, tied_terms) where tied_terms are all terms sharing that minimum Adj P.
    Deterministic tie-break here = smallest raw P (Enrichr rank 1), then Term name."""
    if res is None or len(res) == 0:
        return "", np.nan, []
    f = res[res["Adjusted P-value"] < 0.05]
    if f.empty:
        return "", np.nan, []
    mn = f["Adjusted P-value"].min()
    tied = f[np.isclose(f["Adjusted P-value"], mn, rtol=1e-12, atol=0)]
    return tied.iloc[0]["Term"], float(mn), list(tied["Term"])
