"""RNA sex-marker detail for flagged donors: per cell type, XIST and Y-gene CPM expressed as a fraction of the
recorded-female median (XIST) and recorded-male median (Y genes). A clean opposite-sex profile gives ~1 on the
opposite-sex marker and ~0 on the own-sex marker; a mixture gives intermediate values on both."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE)
d = pd.read_csv(f"{P}/01_rna_chip_calls.csv", index_col=0)
CTS = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
key = pd.read_csv(f"{P}/donor_key.csv")
male = d.clin_male.astype(bool); ok = ~d.rna_disc & d.rna_call.isin(["M", "F"])
for pj, lb in key.values:
    r = d.loc[pj]; own, opp = [], []
    for ct in CTS:
        xi = np.expm1(r[f"rna_{ct}_XIST"]) / np.median(np.expm1(d.loc[ok & ~male, f"rna_{ct}_XIST"]))
        yi = np.expm1(r[f"rna_{ct}_Y"]) / np.median(np.expm1(d.loc[ok & male, f"rna_{ct}_Y"]))
        own.append(yi if r.clin_male else xi); opp.append(xi if r.clin_male else yi)
    print(f"{lb}: own-sex marker as fraction of typical, per cell type {np.round(own, 2).tolist()}; opposite-sex marker {np.round(opp, 2).tolist()}")
# all donors: how many show any mixed signal (both markers > 0.2 of typical in the 6-type mean)?
mx = []
for pj, r in d.iterrows():
    xi = np.mean([np.expm1(r[f"rna_{ct}_XIST"]) / np.median(np.expm1(d.loc[ok & ~male, f"rna_{ct}_XIST"])) for ct in CTS])
    yi = np.mean([np.expm1(r[f"rna_{ct}_Y"]) / np.median(np.expm1(d.loc[ok & male, f"rna_{ct}_Y"])) for ct in CTS])
    mx.append(min(xi, yi))
mx = np.array(mx)
print(f"all 276 donors: min(XIST fraction, Y fraction) > 0.1 in {int((mx > 0.1).sum())}, > 0.2 in {int((mx > 0.2).sum())}")
