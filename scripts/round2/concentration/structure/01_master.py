"""Per-component master table for the 547 corrected components (aggregates only; no donor IDs in out/).

Columns: concentration (centred top-donor share, PR; st_common docstring), PIP>0.5 feature counts per modality (re-derived
from the archive loadings caches and checked against r1_numbers/pip_counts_547.csv and Data S1 omics_count), run recurrence
(r1_numbers/run_counts_547.csv), SEA-AD partner and Pearson r (Data S1), phenotype calls (Data S1 = unadjusted BH < 0.05
over 547; adjusted = proportional-odds LRT with age, PMI, cohort [+ sex], BH over 547 per phenotype, covariates/po_tests_547.csv),
GWAS (archived GEMMA 0.96 runs; gwas547_verify catalogue + local C547/C548), protein-influence measures
(protein_influence consensus scenarios, verify_protein_influence fit), top-donor flags (identity checks; proteome category).
Writes out/01_master_547.csv; private/01_top_donor_index.npy (local only)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import json, numpy as np, pandas as pd
from st_common import *

S = scores547(); comps = list(S.index); don = samples()
C = pd.DataFrame([concentration(S.loc[k].values) for k in comps], index=comps)
C.index.name = "component"
np.save(f"{PRIV}/01_top_donor_index.npy", C.top_idx.values)

# ---- PIP > 0.5 counts per modality, re-derived (546 archive caches + C547/C548) and cross-checked
VSP = f"{cfg.WORK}/verify_source_panels/cache"
pip = pd.DataFrame(index=comps, columns=MODS, dtype=float)
for m, j in zip(MODS, (1, 2, 3, 4)):
    P = np.load(f"{VSP}/{m}_P.npy", mmap_mode="r")
    assert P.shape[0] == 546
    n = (np.asarray(P) > 0.5).sum(1)
    for k in comps:
        pip.loc[k, m] = n[k - 1] if k <= 546 else int((np.load(f"{AUD}/c547/C{k}_P{j}.npy") > 0.5).sum())
pip = pip.astype(int)
ref = pd.read_csv(f"{R2}/r1_numbers/out/pip_counts_547.csv", index_col=0)
assert (ref.loc[comps, MODS].values == pip.values).all(), "PIP counts differ from r1_numbers"
ds1 = pd.read_csv(DS1, index_col=0)
assert list(ds1.index) == comps
pip["n_modalities"] = (pip[MODS] > 0).sum(1)
assert (pip.n_modalities == ds1.omics_count).all()
print("PIP>0.5 counts re-derived: identical to r1_numbers and to Data S1 omics_count (547/547)")

# ---- run recurrence
rc = pd.read_csv(f"{R2}/r1_numbers/out/run_counts_547.csv").set_index("buggy_max_id")
runs = rc.loc[comps, ["true_runs", "size"]].rename(columns={"size": "n_members"})

# ---- SEA-AD (Data S1) and phenotype calls
M = C.drop(columns="top_idx").join(pip).join(runs)
M["seaad_partner"] = ds1["SEA-AD_comp"]; M["seaad_pearson"] = ds1["SEA-AD_corr"]
for ph, nm in (("Sex", "sex"), ("ceradsc", "cerad"), ("cogdx", "cogdx")):
    M[f"ds1_{nm}_q"] = ds1[ph]                                  # shown only when < 0.05
po = pd.read_csv(f"{R2}/covariates/po_tests_547.csv", header=[0, 1], index_col=0)
assert list(po.index) == comps
for ph, nm in (("Sex", "sex"), ("ceradsc", "cerad"), ("cogdx", "cogdx")):
    M[f"unadj_{nm}_q"] = po[(ph, "U_q")].values
    M[f"adj_{nm}_q"] = po[(ph, "PO1_q")].values
    M[f"adjqc_{nm}_q"] = po[(ph, "PO2_q")].values
    bcol = [c for c in po[ph].columns if c.startswith("PO1_beta")][0]
    M[f"adj_{nm}_beta"] = po[(ph, bcol)].values
# Data S1 calls must equal unadjusted q < 0.05
for nm, ph in (("sex", "Sex"), ("cerad", "ceradsc"), ("cogdx", "cogdx")):
    a = M[f"ds1_{nm}_q"].notna(); b = M[f"unadj_{nm}_q"] < 0.05
    assert (a == b).all(), nm
M["n_ds1_calls"] = sum(M[f"ds1_{nm}_q"].notna().astype(int) for nm in ("sex", "cerad", "cogdx"))
M["n_adj_calls"] = sum((M[f"adj_{nm}_q"] < 0.05).astype(int) for nm in ("sex", "cerad", "cogdx"))
M["adj_calls"] = [";".join(nm for nm in ("sex", "cerad", "cogdx") if M.loc[k, f"adj_{nm}_q"] < 0.05) for k in comps]
M["ds1_calls"] = [";".join(nm for nm in ("sex", "cerad", "cogdx") if pd.notna(M.loc[k, f"ds1_{nm}_q"])) for k in comps]

# ---- GWAS (archived GEMMA runs for 1..546 incl. retired 520; local runs for C547/C548)
G = pd.read_csv(f"{AUD}/gwas547_verify/04_catalogue_546.csv").set_index("component")
G2 = pd.read_csv(f"{AUD}/gwas547/new_components_summary.csv").set_index("component")
gw = pd.DataFrame(index=comps, columns=["gwas_min_p", "gwas_n_below_5e8", "gwas_n_below_1e15", "gwas_lambda"], dtype=float)
for k in comps:
    if k <= 546:
        gw.loc[k] = [G.loc[k, "min_p"], G.loc[k, "n_below_5e-08"], G.loc[k, "n_below_1e-15"], G.loc[k, "lambda_gc"]]
    else:
        gw.loc[k] = [G2.loc[k, "min_p_score"], G2.loc[k, "n_below_5e8"], G2.loc[k, "n_below_1e15"], G2.loc[k, "lambda_gc"]]
M = M.join(gw)

# ---- protein influence (fixed-loading re-estimation; two independent implementations)
pi = pd.read_csv(f"{R2}/protein_influence/results/consensus_547_scenarios.csv").set_index("component_id")
vp = pd.read_csv(f"{R2}/verify_protein_influence/results/consensus_fit_full_r1_mf.csv").set_index("component_id")
assert set(pi.index) == set(comps) and set(vp.index) == set(comps)
M["n_prot_pip05_check"] = pi.loc[comps, "n_prot_pip05"].values
assert (M.n_prot_pip05_check == M.proteins).all(); M = M.drop(columns="n_prot_pip05_check")
M["prot_fixedload_r_corrected"] = pi.loc[comps, "V3_d_r"].values          # scenario d: corrected protein rows
M["prot_fixedload_r_removed"] = pi.loc[comps, "V3_b_r"].values            # scenario b: protein layer removed
M["prot_fixedload_r_corrected_verify"] = vp.loc[comps, "d_r"].values
cat = np.load(f"{cfg.WORK}/verify_protein_influence/cache/prot_cat.npy")
assert len(cat) == 276 and {c: int((cat == c).sum()) for c in range(5)} == {0: 143, 1: 76, 2: 28, 3: 1, 4: 28}
has_prot = ~np.isnan(pd.read_csv(f"{SDA}/prot.txt", sep=" ", header=None, usecols=[0, 1, 2], na_values="NA").values).all(1)
assert (has_prot == np.isin(cat, [1, 2, 3])).all(), "prot_cat order != md_samples"
CATNAME = {0: "none fed", 1: "own proteome", 2: "another person's proteome (no own)", 3: "another person's proteome (own exists)",
           4: "none fed (own sample dropped)"}
M["top_donor_protein_input"] = [CATNAME[int(cat[i])] for i in C.top_idx]
M["top_donor_misfed_proteome"] = [bool(cat[i] in (2, 3)) for i in C.top_idx]

# ---- identity-check flags for the top donor (labels only)
key = pd.read_csv(f"{cfg.PRIVATE}/donor_key.csv")
lab = dict(zip(key.projid.astype(str), key.label))
M["top_donor_identity_flag"] = [lab.get(don[i], "") for i in C.top_idx]

# ---- top donor shared across components? (count of components whose top donor is the same donor)
td = pd.Series(C.top_idx.values, index=comps)
M["top_donor_n_components_topped"] = td.map(td.value_counts()).values

M.index.name = "component"
M.to_csv(f"{OUT}/01_master_547.csv")
print(M.loc[FOCUS + OUTGOING + OTHER_NAMED, ["top_share", "PR", "n_modalities", "true_runs", "seaad_pearson", "adj_calls",
                                              "gwas_min_p", "prot_fixedload_r_corrected"]].to_string(float_format=lambda x: f"{x:.3g}"))
print("written", f"{OUT}/01_master_547.csv", M.shape)
