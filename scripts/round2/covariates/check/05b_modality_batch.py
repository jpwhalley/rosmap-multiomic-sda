"""Modality-batch checks for the central five, with the TMT batch taken from the data column actually used for each
SDA proteomics donor (original ros1d R_ID route; batch = 'bXX' prefix of the TMT data column). Two variants:
 (A) all batch levels as dummies, RINT over all 276 donors;  (B) the covariates/ run's variant: levels with <2 donors
     pooled into 'rare', RINT recomputed within the subset, KW over levels with >=2 donors in the phenotype subset.
Aggregates only."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np, pandas as pd
from scipy import stats as ss
from vc_common import *

s = load_scores(); don = list(s.columns); cl = load_clin(don)
age, pmi, mp, male, cad, cog = cl.age.values, cl.pmi.values, cl.is_map.values, cl.male.values, cl.cerad_ad.values, cl.cogdx.values
pos_of = pd.Series(np.arange(276), index=don)
def present(f):
    return pd.Series([l.split(maxsplit=1)[0] != "NA" for l in open(f"{B}/sda/{f}")], index=open(f"{B}/sda/md_samples.txt").read().split())
clin_all = pd.read_csv(f"{B}/oth/meta/ROSMAP_clinical.csv", dtype={"projid": str})
i2p = dict(zip(clin_all.individualID, clin_all.projid.astype(int).astype(str)))
rbio = pd.read_csv(f"{B}/oth/meta/ROSMAP_biospecimen_metadata.csv")
s2p = {sp: i2p.get(i) for sp, i in zip(rbio.specimenID, rbio.individualID)}
qcc = pd.read_csv(f"{B}/oth/chip/QualityControl.csv", sep="\t", dtype={"SampleID": str}); qcc["pid"] = qcc.SampleID.astype(int).astype(str)
mm = pd.read_csv(f"{B}/oth/meta/ROSMAP_assay_methylationArray_metadata.csv"); mm["pid"] = mm.specimenID.map(s2p)
# TMT: map each archived proteomics donor to the data column it carries (exact match), batch = column prefix
d = pd.read_csv(f"{B}/prot/tmt/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv", index_col=0)
x = pd.read_csv(f"{B}/prot/tmt/prot_crossed_with_sc.csv", index_col=0); x = x.loc[:, x.notna().any()]; dd = d.loc[x.index]
tcol = {str(c): (dd.sub(x[c], axis=0)).abs().max().idxmin() for c in x.columns}
BAT = {"chip": qcc.drop_duplicates("pid").set_index("pid").Batch.astype(str),
       "meth": mm.dropna(subset=["pid"]).drop_duplicates("pid").set_index("pid").batch.astype(str),
       "meth_plate": mm.dropna(subset=["pid"]).drop_duplicates("pid").set_index("pid").Sample_Plate.astype(str),
       "tmt": pd.Series({k: v.split(".")[0] for k, v in tcol.items()})}
blk = {"chip": present("chip.txt"), "meth": present("meth.txt"), "meth_plate": present("meth.txt"), "tmt": present("prot.txt")}
FIVE = [(74, "Sex"), (278, "Sex"), (545, "Sex"), (172, "ceradsc"), (277, "cogdx")]
rows = []
for m, bat in BAT.items():
    dn = [p for p in blk[m].index[blk[m]]]
    b = bat.reindex(dn)
    print(f"{m}: SDA-block donors {len(dn)}; with batch {int(b.notna().sum())}; levels {b.nunique()}; levels with >=2 donors {int((b.value_counts() >= 2).sum())}")
    for k, ph in FIVE:
        keep = np.array([True] * len(dn)) if ph != "cogdx" else np.isin(cog[pos_of[dn].values], [1, 2, 4])
        dk = list(np.array(dn)[keep]); pos = pos_of[dk].values; bb = b.loc[dk].values
        y = s.loc[k].values[pos]
        Xc = np.column_stack([age[pos], pmi[pos], mp[pos]] + ([male[pos]] if ph != "Sex" else []))
        Xp = (male[pos][:, None] if ph == "Sex" else cad[pos][:, None] if ph == "ceradsc" else np.column_stack([cog[pos] == 2, cog[pos] == 4]).astype(float))
        # (A) all levels, RINT over 276
        DA = pd.get_dummies(bb, drop_first=True).values.astype(float)
        RA = rint(s.loc[k].values)[pos][:, None]
        kwA = ss.kruskal(*[y[bb == g] for g in np.unique(bb)]).pvalue
        pA0 = ols_F(RA, Xc, Xp)[0][0]; pA1 = ols_F(RA, np.column_stack([Xc, DA]), Xp)[0][0]
        poA0 = po_lrt(y, Xc, Xp)["p"]; poA1 = po_lrt(y, np.column_stack([Xc, DA]), Xp)["p"]
        # (B) pooled rare levels, RINT within subset
        vc = pd.Series(bb).value_counts(); bp = np.where(pd.Series(bb).map(vc).values >= 2, bb, "rare")
        DB = pd.get_dummies(bp, drop_first=True).values.astype(float)
        RB = rint(y)[:, None]
        grp = [y[bp == g] for g in np.unique(bp) if (bp == g).sum() >= 2 and g != "rare"] + ([y[bp == "rare"]] if (bp == "rare").sum() >= 2 else [])
        kwB = ss.kruskal(*grp).pvalue
        pB0 = ols_F(RB, Xc, Xp)[0][0]; pB1 = ols_F(RB, np.column_stack([Xc, DB]), Xp)[0][0]
        rows.append(dict(modality=m, comp=k, pheno=ph, n=len(dk), dummiesA=DA.shape[1], dummiesB=DB.shape[1], kwA=kwA, kwB=kwB,
                         rintA_p=pA0, rintA_p_batch=pA1, poA_p=poA0, poA_p_batch=poA1, rintB_p=pB0, rintB_p_batch=pB1))
r = pd.DataFrame(rows); r.to_csv("vc_modality_batch_v2.csv", index=False)
print(r.to_string(float_format=lambda v: f"{v:.3g}"))
for m in BAT:
    x_ = r[r.modality == m]
    print(f"{m}: KW range (A) {x_.kwA.min():.2g}-{x_.kwA.max():.2g}; (B) {x_.kwB.min():.2g}-{x_.kwB.max():.2g}")
