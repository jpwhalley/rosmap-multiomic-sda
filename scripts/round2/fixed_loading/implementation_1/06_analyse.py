"""Aggregate the fixed-loadings protein scenarios across runs: highlighted run-level members, consensus
scores (mean of members with the published sample-flip signs) and the 547-component catalogue.
Aggregates only: no donor IDs; donors are referred to by protein-mapping category.
Categories: 0 no protein either way (143) | 1 own proteome fed (76) | 2 fed another donor's proteome, no own
sample (28) | 3 fed another donor's proteome, own sample exists (1) | 4 own proteome dropped, restored (28).
Mis-paired = categories 2+3 (29)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, json, numpy as np, pandas as pd
from scipy import stats as ss
from pi_common import *

VARS = ["V3", "V1", "V0"]   # V3 primary; V1, V0 sensitivity (see 05_run.py)
SCEN = {"b": "protein removed (all donors)", "c": "protein removed (29 mis-paired)", "d": "corrected protein rows"}
runs = [r for r in range(1, 11) if os.path.exists(f"{CACHE}/run{r}/scen.npz")]
print("runs available:", runs)
grp = np.load(f"{CACHE}/prot_group.npy")
mis = np.isin(grp, [2, 3])
GROUPS = {"mispaired29": mis, "restored28": grp == 4, "correct76": grp == 1, "noprot143": grp == 0, "rest247": ~mis}
mem = pd.read_csv(f"{REL}/cluster_members.csv")
pubC = pd.read_csv(f"{REL}/samples_final.csv", index_col=0)
samps = [l.strip() for l in open(f"{SDA}/md_samples.txt") if l.strip()]
assert list(pubC.columns.astype(str)) == samps
pubC = pubC.loc[pubC.index != 520]          # 547 active components
assert len(pubC) == 547 and set(pubC.index) == set(mem.component_id)
pipP = pd.read_csv(f"{REL}/pip_final_proteins.csv", index_col=0).loc[pubC.index]
nprot = (pipP.values > 0.5).sum(1)
nprot = pd.Series(nprot, index=pubC.index)

# ---- clinical (unadjusted tests as in Data S1: sex MWU; CERAD binarised MWU; cogdx KW over 1,2,4)
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv", index_col=0).loc[[int(s) for s in samps]]
msex = clin.msex.values; cer = clin.ceradsc.map({1: "AD", 2: "AD", 3: "No", 4: "No"}).values; cog = clin.cogdx.values


def tests(v):
    v = np.asarray(v, float)
    u = ss.mannwhitneyu(v[msex == 1], v[msex == 0], alternative="two-sided")
    uc = ss.mannwhitneyu(v[cer == "AD"], v[cer == "No"], alternative="two-sided")
    kw = ss.kruskal(*[v[cog == k] for k in (1, 2, 4)])
    return dict(sex_p=u.pvalue, sex_auc_male=u.statistic / ((msex == 1).sum() * (msex == 0).sum()),
                cerad_p=uc.pvalue, cerad_auc_AD=uc.statistic / ((cer == "AD").sum() * (cer == "No").sum()),
                cogdx_p=kw.pvalue, cogdx_H=kw.statistic)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); rr = p[o]
    adj = np.minimum.accumulate((rr * n / np.arange(1, n + 1))[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(adj, 1); return out


PHENO = {74: "sex", 278: "sex", 545: "sex", 172: "cerad", 277: "cogdx"}
# validate the test implementation against the 547-family BH values
bhref = pd.read_csv(f"{AUDIT}/c547/clinical_bh_547.csv", index_col=0)
raw_pub = pd.DataFrame([tests(pubC.loc[k].values) for k in pubC.index], index=pubC.index)
adj_pub = pd.DataFrame({c: bh(raw_pub[f"{c2}_p"]) for c, c2 in [("Sex", "sex"), ("ceradsc", "cerad"), ("cogdx", "cogdx")]}, index=pubC.index)
dif = (adj_pub - bhref.loc[adj_pub.index, adj_pub.columns]).abs().max()
print("clinical tests reproduce clinical_bh_547 (published consensus): max |diff| per variable", dif.round(12).to_dict())


def robust_z(x):
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / mad


def dom(a):
    rz = robust_z(a); i = int(np.argmax(np.abs(rz))); d = a - a.mean()
    return i, float(np.abs(rz[i])), float(d[i] ** 2 / np.sum(d ** 2))


def compare(a, b):
    """a baseline, b scenario (276 donors)."""
    sd = a.std(); dlt = (b - a) / sd
    i0, z0, s0 = dom(a); i1, z1, s1 = dom(b)
    o = dict(r=float(np.corrcoef(a, b)[0, 1]), rho=float(ss.spearmanr(a, b)[0]),
             dom_cat=int(grp[i0]), dom_absz_base=z0, dom_share_base=s0,
             dom_same=bool(i0 == i1), dom_absz_scen_same_donor=float(np.abs(robust_z(b)[i0])), dom_share_scen=s1,
             dom_score_ratio=float(b[i0] / a[i0]))
    for g, m in GROUPS.items():
        o[f"mean_abs_dz_{g}"] = float(np.mean(np.abs(dlt[m])))
        o[f"max_abs_dz_{g}"] = float(np.max(np.abs(dlt[m])))
    return o


# ---- load per-run scenario scores for the member positions only (memory)
need = mem[mem.run.isin(runs)]
S = {}
cal = {}
for r in runs:
    z = np.load(f"{CACHE}/run{r}/scen.npz")
    idx = np.unique(need[need.run == r].sda_index.values)
    for v in VARS:
        for s in "abcd":
            S[(r, v, s)] = pd.DataFrame(z[f"{v}_{s}"][:, idx].astype(np.float64), columns=idx)
    S[(r, "pub", "a")] = pd.DataFrame(np.load(f"{CACHE}/run{r}/A.npy")[:, idx], columns=idx)
    S[(r, "V1cv", "a")] = pd.DataFrame(z["A1cv"][:, idx].astype(np.float64), columns=idx)
    S[(r, "V3cv", "a")] = pd.DataFrame(z["A3cv"][:, idx].astype(np.float64), columns=idx)
    cal[r] = json.load(open(f"{RES}/calibration_run{r}.json"))

# ---- (1) calibration across runs
crow = []
for r in runs:
    c = cal[r]
    crow.append(dict(run=r, n_cat=c["n_catalogue_members"], n_hl=c["n_highlight_members"],
                     **{f"{v}_cat_r_q01": c[f"{v}_catalogue_r_q01_05_50"][0] for v in ["V0", "V2", "V1cv", "V1", "V3cv", "V3"]},
                     **{f"{v}_cat_r_q50": c[f"{v}_catalogue_r_q01_05_50"][2] for v in ["V0", "V2", "V1cv", "V1", "V3cv", "V3"]},
                     **{f"{v}_cat_rel_q50": c[f"{v}_catalogue_rel_q50_95_max"][0] for v in ["V0", "V2", "V1cv", "V1", "V3cv", "V3"]},
                     **{f"{v}_cat_rel_q95": c[f"{v}_catalogue_rel_q50_95_max"][1] for v in ["V0", "V2", "V1cv", "V1", "V3cv", "V3"]},
                     **{f"{v}_hl_r_min": c[f"{v}_highlight_r_min"] for v in ["V0", "V2", "V1cv", "V1", "V3cv", "V3"]},
                     **{f"{v}_hl_rel_max": c[f"{v}_highlight_rel_q50_95_max"][2] for v in ["V0", "V2", "V1cv", "V1", "V3cv", "V3"]},
                     **{f"{v}_all_r_q05": c[f"{v}_all_r_q01_05_50"][1] for v in ["V0", "V2", "V1cv", "V1", "V3cv", "V3"]}))
CAL = pd.DataFrame(crow); CAL.to_csv(f"{RES}/calibration_by_run.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print("\n== calibration by run (catalogue members; r = per-component Pearson vs published A; rel = ||Ahat-A||/||A||)")
print(CAL[["run", "n_cat", "n_hl"] + [c for c in CAL.columns if c.startswith("V3cv") or c.startswith("V1cv") or c.startswith("V0_")]].round(4).to_string(index=False))

# per highlighted member calibration (r, rel vs published)
hlm = need[need.component_id.isin(HIGHLIGHT)].copy()
rows = []
for _, m in hlm.iterrows():
    r, k = int(m.run), int(m.sda_index)
    a_pub = S[(r, "pub", "a")][k].values
    row = dict(component_id=int(m.component_id), run=r, sda_index=k, flip=int(m.flip_samples))
    for v, key in [("V3cv", (r, "V3cv", "a")), ("V3", (r, "V3", "a")), ("V1cv", (r, "V1cv", "a")), ("V1", (r, "V1", "a")), ("V0", (r, "V0", "a"))]:
        b = S[key][k].values
        row[f"cal_r_{v}"] = float(np.corrcoef(a_pub, b)[0, 1]); row[f"cal_rel_{v}"] = float(np.linalg.norm(b - a_pub) / np.linalg.norm(a_pub))
    for v in VARS:
        a = S[(r, v, "a")][k].values
        for s in "bcd":
            o = compare(a, S[(r, v, s)][k].values)
            for kk, vv in o.items(): row[f"{v}_{s}_{kk}"] = vv
        ph = PHENO.get(int(m.component_id))
        if ph:
            t0 = tests(a)
            row[f"{v}_pheno"] = ph; row[f"{v}_a_p"] = t0[f"{ph}_p"]
            for s in "bcd":
                row[f"{v}_{s}_p"] = tests(S[(r, v, s)][k].values)[f"{ph}_p"]
    rows.append(row)
M = pd.DataFrame(rows).sort_values(["component_id", "run", "sda_index"])
M.to_csv(f"{RES}/highlighted_members_scenarios.csv", index=False)

# ---- (2) consensus scores
def consensus(v, s, comps):
    out = {}
    for c in comps:
        mm = need[need.component_id == c]
        vals = [m.flip_samples * S[(int(m.run), v, s)][int(m.sda_index)].values for _, m in mm.iterrows()]
        out[c] = np.mean(vals, 0)
    return pd.DataFrame(out).T


allc = list(pubC.index)
CONS = {(v, s): consensus(v, s, allc) for v in VARS + ["pub"] for s in ("abcd" if v != "pub" else "a")}
CONS[("V1cv", "a")] = consensus("V1cv", "a", allc)
CONS[("V3cv", "a")] = consensus("V3cv", "a", allc)
full = len(runs) == 10
if full:
    d = (CONS[("pub", "a")] - pubC.loc[allc].values).abs().values.max()
    print(f"\nmean of flipped published members reproduces the release consensus: max |diff| {d:.2e}")
# consensus calibration: re-estimated baseline vs published consensus
cc = []
for c in allc:
    p = pubC.loc[c].values
    row = dict(component_id=c)
    for v in ["V3cv", "V3", "V1cv", "V1", "V0"]:
        b = CONS[(v, "a")].loc[c].values
        row[f"cal_r_{v}"] = float(np.corrcoef(p, b)[0, 1]); row[f"cal_rel_{v}"] = float(np.linalg.norm(b - p) / np.linalg.norm(p))
        dz = np.abs(b - p) / p.std()
        row[f"cal_mean_abs_dz_mis29_{v}"] = float(dz[mis].mean()); row[f"cal_mean_abs_dz_rest_{v}"] = float(dz[~mis].mean())
        i0, _, _ = dom(p); row[f"cal_dom_ratio_{v}"] = float(b[i0] / p[i0])
    cc.append(row)
CC = pd.DataFrame(cc).set_index("component_id")
print("\n== consensus calibration over 547 components (re-estimated baseline vs published consensus)")
for v in ["V3cv", "V3", "V1cv", "V1", "V0"]:
    print(f"  {v}: r q01/q05/q50 {np.round(np.percentile(CC[f'cal_r_{v}'], [1, 5, 50]), 4)}, min {CC[f'cal_r_{v}'].min():.4f}; "
          f"rel q50/q95/max {np.round(np.percentile(CC[f'cal_rel_{v}'], [50, 95, 100]), 3)}; highlighted r min {CC.loc[HIGHLIGHT, f'cal_r_{v}'].min():.4f}, rel max {CC.loc[HIGHLIGHT, f'cal_rel_{v}'].max():.3f}")

# consensus scenario comparisons (all 547)
crow = []
for c in allc:
    row = dict(component_id=c, n_members=int((need.component_id == c).sum()), n_prot_pip05=int(nprot[c]))
    for v in VARS:
        a = CONS[(v, "a")].loc[c].values
        for s in "bcd":
            o = compare(a, CONS[(v, s)].loc[c].values)
            for kk, vv in o.items(): row[f"{v}_{s}_{kk}"] = vv
    crow.append(row)
CS = pd.DataFrame(crow).set_index("component_id").join(CC)
# clinical: raw + BH (547 family) per scenario
for v in VARS:
    for s in "abcd":
        T = pd.DataFrame([tests(CONS[(v, s)].loc[c].values) for c in allc], index=allc)
        for ph in ["sex", "cerad", "cogdx"]:
            CS[f"{v}_{s}_{ph}_p"] = T[f"{ph}_p"].values; CS[f"{v}_{s}_{ph}_q"] = bh(T[f"{ph}_p"].values)
        CS[f"{v}_{s}_sex_auc_male"] = T["sex_auc_male"].values; CS[f"{v}_{s}_cerad_auc_AD"] = T["cerad_auc_AD"].values; CS[f"{v}_{s}_cogdx_H"] = T["cogdx_H"].values
# delta form: published consensus + (scenario - baseline) change from the same machinery (cancels calibration error)
for s_ in "bcd":
    T = pd.DataFrame([tests(pubC.loc[c].values + CONS[("V3", s_)].loc[c].values - CONS[("V3", "a")].loc[c].values) for c in allc], index=allc)
    for ph in ["sex", "cerad", "cogdx"]:
        CS[f"pubdelta_{s_}_{ph}_p"] = T[f"{ph}_p"].values; CS[f"pubdelta_{s_}_{ph}_q"] = bh(T[f"{ph}_p"].values)
for ph in ["sex", "cerad", "cogdx"]:
    CS[f"pub_{ph}_p"] = raw_pub[f"{ph}_p"].values
    CS[f"pub_{ph}_q"] = adj_pub[{"sex": "Sex", "cerad": "ceradsc", "cogdx": "cogdx"}[ph]].values
CS = CS.copy()
CS.to_csv(f"{RES}/consensus_547_scenarios.csv")

print("\n== highlighted consensus components (V3 primary; V0 in brackets)")
for c in HIGHLIGHT:
    x = CS.loc[c]
    line = (f"C{c} ({int(x.n_members)} members; protein PIP>0.5 features {int(x.n_prot_pip05)}; calibration vs published consensus: V3cv r {x.cal_r_V3cv:.4f} rel {x.cal_rel_V3cv:.3f}, "
            f"mean|dz| mis29 {x.cal_mean_abs_dz_mis29_V3cv:.3f} rest {x.cal_mean_abs_dz_rest_V3cv:.3f}, dominant-donor ratio {x.cal_dom_ratio_V3cv:.3f}; V3 r {x.cal_r_V3:.4f} rel {x.cal_rel_V3:.3f}; V0 r {x.cal_r_V0:.4f} rel {x.cal_rel_V0:.3f})")
    print(line)
    for s in "bcd":
        print(f"   {s} {SCEN[s]:<32} r={x[f'V3_{s}_r']:.4f} [{x[f'V0_{s}_r']:.4f}]  mean|dz| mis29={x[f'V3_{s}_mean_abs_dz_mispaired29']:.4f} rest={x[f'V3_{s}_mean_abs_dz_rest247']:.4f} "
              f"restored28={x[f'V3_{s}_mean_abs_dz_restored28']:.4f} max|dz| mis29={x[f'V3_{s}_max_abs_dz_mispaired29']:.3f} rest={x[f'V3_{s}_max_abs_dz_rest247']:.3f} | "
              f"dominant donor cat {int(x[f'V3_{s}_dom_cat'])}: same={bool(x[f'V3_{s}_dom_same'])} |z| {x[f'V3_{s}_dom_absz_base']:.1f}->{x[f'V3_{s}_dom_absz_scen_same_donor']:.1f} share {x[f'V3_{s}_dom_share_base']:.3f}->{x[f'V3_{s}_dom_share_scen']:.3f}")
    if c in PHENO:
        ph = PHENO[c]
        print(f"   phenotype {ph}: published p={raw_pub.loc[c, ph + '_p']:.3g} q={CS.loc[c, 'pub_' + ph + '_q']:.3g} | baseline p={x[f'V3_a_{ph}_p']:.3g} q={x[f'V3_a_{ph}_q']:.3g} | "
              + " | ".join(f"{s}: p={x[f'V3_{s}_{ph}_p']:.3g} q={x[f'V3_{s}_{ph}_q']:.3g}" for s in "bcd")
              + " || published+change: " + " | ".join(f"{s}: p={x[f'pubdelta_{s}_{ph}_p']:.3g} q={x[f'pubdelta_{s}_{ph}_q']:.3g}" for s in "bcd"))

# ---- (4) catalogue-wide distribution
print("\n== catalogue-wide (547 consensus): Pearson r of scenario vs baseline consensus scores (V3 [V0])")
for s in "bcd":
    q = np.percentile(CS[f"V3_{s}_r"], [0, 1, 5, 25, 50]); q0 = np.percentile(CS[f"V0_{s}_r"], [0, 1, 5, 25, 50])
    print(f"   {s}: min/q01/q05/q25/q50 {np.round(q, 4)} [{np.round(q0, 4)}]; n r<0.99 {int((CS[f'V3_{s}_r'] < 0.99).sum())} [{int((CS[f'V0_{s}_r'] < 0.99).sum())}], "
          f"n r<0.95 {int((CS[f'V3_{s}_r'] < 0.95).sum())} [{int((CS[f'V0_{s}_r'] < 0.95).sum())}], n r<0.9 {int((CS[f'V3_{s}_r'] < 0.9).sum())} [{int((CS[f'V0_{s}_r'] < 0.9).sum())}]")
hasP = CS.n_prot_pip05 > 0
print(f"   components with protein PIP>0.5 features: {int(hasP.sum())}; median r_d with / without: {CS.loc[hasP, 'V3_d_r'].median():.4f} / {CS.loc[~hasP, 'V3_d_r'].median():.4f}; "
      f"q05 with / without: {CS.loc[hasP, 'V3_d_r'].quantile(0.05):.4f} / {CS.loc[~hasP, 'V3_d_r'].quantile(0.05):.4f}")
worst = CS.sort_values("V3_d_r").head(25)
cols = ["n_members", "n_prot_pip05", "V3_d_r", "V1_d_r", "V0_d_r", "V3_b_r", "V3_c_r", "V3_d_mean_abs_dz_mispaired29", "V3_d_mean_abs_dz_rest247", "V3_d_max_abs_dz_mispaired29",
        "V3_d_max_abs_dz_restored28", "V3_d_dom_cat", "V3_d_dom_share_base", "V3_d_dom_same", "V3_d_dom_share_scen", "cal_r_V3cv", "cal_rel_V3cv"]
print("\n   25 most affected (lowest scenario-d r):")
print(worst[cols].round(4).to_string())
worst[cols].to_csv(f"{RES}/catalogue_most_affected_d.csv")
# clinical significance changes across the catalogue (BH q<0.05)
print("\n== catalogue clinical calls (BH q<0.05, 547 family): published | re-estimated baseline | b | c | d  (V3); then published+change form")
for ph in ["sex", "cerad", "cogdx"]:
    sets = {"pub": set(CS.index[CS[f"pub_{ph}_q"] < 0.05])}
    for s in "abcd": sets[s] = set(CS.index[CS[f"V3_{s}_{ph}_q"] < 0.05])
    for s in "bcd": sets["pd" + s] = set(CS.index[CS[f"pubdelta_{s}_{ph}_q"] < 0.05])
    print(f"   {ph}: " + " | ".join(f"{k} {len(v)}" for k, v in sets.items())
          + f" ; baseline vs published lost {sorted(sets['pub'] - sets['a'])} gained {sorted(sets['a'] - sets['pub'])}"
          + "".join(f" ; {s} vs baseline lost {sorted(sets['a'] - sets[s])} gained {sorted(sets[s] - sets['a'])}" for s in "bcd")
          + "".join(f" || published+change {s}: lost {sorted(sets['pub'] - sets['pd' + s])} gained {sorted(sets['pd' + s] - sets['pub'])}" for s in "bcd"))
    ch = sorted((sets["pub"] ^ sets["pdd"]) | (sets["a"] ^ sets["d"]))
    for c in ch:
        print(f"      C{c}: published p={CS.loc[c, f'pub_{ph}_p']:.3g} q={CS.loc[c, f'pub_{ph}_q']:.3g}; published+change(d) q={CS.loc[c, f'pubdelta_d_{ph}_q']:.3g}; "
              f"re-est baseline q={CS.loc[c, f'V3_a_{ph}_q']:.3g}, d q={CS.loc[c, f'V3_d_{ph}_q']:.3g}; protein PIP>0.5 features {int(CS.loc[c, 'n_prot_pip05'])}; r_d {CS.loc[c, 'V3_d_r']:.4f}")
json.dump(dict(runs=runs, full=full), open(f"{RES}/analysis_meta.json", "w"))
