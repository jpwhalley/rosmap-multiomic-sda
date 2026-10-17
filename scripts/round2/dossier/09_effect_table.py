"""Orient the 02_effects_ci.csv estimates to the contrasts the manuscript names (aggregates only).
Sex: female vs male (OR>1 = higher scores in females; the score's sign is arbitrary, so the deposited sign is kept);
CERAD: AD vs non-AD; cogdx: MCI vs NCI and AD-dementia vs NCI; trends per step. RINT effects in SD units, same contrast."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from dc_common import *
T = pd.read_csv(f"{OUT}/02_effects_ci.csv")
flip = T.term.eq("sex")
o = T.copy()
o.loc[flip, ["OR", "OR_lo", "OR_hi"]] = np.column_stack([1 / T.loc[flip, "OR"], 1 / T.loc[flip, "OR_hi"], 1 / T.loc[flip, "OR_lo"]])
o.loc[flip, ["beta_rint_SD", "rint_lo", "rint_hi"]] = np.column_stack([-T.loc[flip, "beta_rint_SD"], -T.loc[flip, "rint_hi"], -T.loc[flip, "rint_lo"]])
o["contrast"] = o.term.map({"sex": "female vs male", "cerad_ad": "CERAD AD vs non-AD", "cogdx2": "MCI vs NCI", "cogdx4": "AD dementia vs NCI",
                            "cerad4_rev": "per CERAD step (more plaques)", "cogdx_trend": "per step NCI->MCI->AD"})
f = lambda r: f"{r.OR:.3g} [{r.OR_lo:.3g}, {r.OR_hi:.3g}]"
g = lambda r: f"{r.beta_rint_SD:+.2f} [{r.rint_lo:+.2f}, {r.rint_hi:+.2f}]"
o["PO_OR_95CI"] = o.apply(f, axis=1); o["RINT_SD_95CI"] = o.apply(g, axis=1)
cols = ["component", "contrast", "model", "n", "PO_OR_95CI", "p_PO_LRT", "q_PO_family", "RINT_SD_95CI", "p_rint_F", "q_rint_family"]
o[cols].to_csv(f"{OUT}/09_effect_table_oriented.csv", index=False, float_format="%.3g")
pd.set_option("display.width", 250)
print(o[cols].to_string(index=False, float_format=lambda v: f"{v:.2g}"))
