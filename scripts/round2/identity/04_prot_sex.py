"""Proteomics sex check under the DLPFC-only TMT mapping (ROSMAP.DLPFC rows, isAssayControl FALSE).
Score per TMT column = mean z of Y-linked proteins minus mean z of X-escape proteins (z within TMT batch).
Also contrasts the as-fed (ros1d, last-row-wins) mapping. Per-column/donor values -> $G_ROS_PRIVATE_DIR; aggregates printed."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
H = os.path.dirname(os.path.abspath(__file__)); P = cfg.out_dir(cfg.PRIVATE)
ROS = cfg.ROS; T = f"{ROS}/prot/tmt"
tmt = pd.read_csv(f"{T}/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv", index_col=0)
meta = pd.read_csv(f"{T}/ROSMAP_assay_proteomics_TMTquantitation_metadata.csv", index_col=0)
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv")
ind2proj = dict(zip(clin.individualID, clin.projid)); projmale = dict(zip(clin.projid, clin.msex == 1))
samps = np.array([int(l) for l in open(f"{ROS}/sda/md_samples.txt") if l.strip()])
def mapping(md):
    sid = md.index.str.split(".").str[-3] + "." + md.index.str.split(".").str[-2]
    rid = md.index.str.split(".").str[-1]
    s = pd.Series(rid, index=sid); return {c: s.loc[c] if not isinstance(s.loc[c], pd.Series) else s.loc[c].iloc[-1] for c in tmt.columns if c in s.index}
dl_meta = meta[meta.index.str.startswith("ROSMAP.DLPFC") & (meta.isAssayControl.astype(str).str.upper() == "FALSE")]
dl = mapping(dl_meta); fed = mapping(meta)
print("DLPFC-only rows", len(dl_meta), "| unique batch.channel", dl_meta.index.str.split(".").str[-3:-1].str.join(".").nunique(), "| columns mapped", len(dl), "of", tmt.shape[1])
dupc = dl_meta.index.str.split(".").str[-3:-1].str.join(".").duplicated().sum(); print("duplicated batch.channel within DLPFC-only:", int(dupc))
Yp = [r for r in tmt.index if r.split("|")[0] in ("RPS4Y1", "DDX3Y", "EIF1AY", "KDM5D", "UTY", "USP9Y", "NLGN4Y", "ZFY")]
Xp = [r for r in tmt.index if r.split("|")[0] in ("DDX3X", "KDM5C", "EIF1AX", "KDM6A", "ZFX", "USP9X", "RPS4X")]
print("Y proteins:", [p.split("|")[0] for p in Yp], "| X-escape proteins:", [p.split("|")[0] for p in Xp])
batch = pd.Series(tmt.columns.str.split(".").str[0], index=tmt.columns)
def zb(v):  # z within batch
    return v.groupby(batch).transform(lambda x: (x - x.mean()) / x.std() if x.notna().sum() > 2 and x.std() > 0 else x * np.nan)
RAW = tmt.loc[Yp + Xp].T; Z = (RAW - RAW.mean()) / RAW.std()   # global z: ratios share the GIS-pool reference across batches
col = pd.DataFrame(index=tmt.columns)
col["y"] = Z[Yp].mean(1); col["x"] = Z[Xp].mean(1); col["n_y"] = Z[Yp].notna().sum(1)
col["dl_proj"] = [ind2proj.get(dl.get(c)) for c in col.index]; col["dl_male"] = col.dl_proj.map(projmale)
# Y-protein ratios have batch-specific scale (per-batch female medians vary), so reference each column to the
# leave-one-out median of the recorded-female columns in its own TMT batch (the tested column never sets its own reference).
fem = col.dl_male == False
R = pd.DataFrame(index=tmt.columns)
for p_ in Yp:
    v = tmt.loc[p_]; ref = pd.Series(np.nan, index=tmt.columns)
    for c in tmt.columns:
        mates = [m for m in tmt.columns[batch == batch[c]] if m != c and fem.get(m, False) and pd.notna(v[m])]
        ref[c] = np.median(v[mates]) if len(mates) >= 2 else np.nanmedian(v[fem.reindex(tmt.columns).fillna(False).values])
    r = v - ref; fr = r[fem.reindex(tmt.columns).fillna(False).values].dropna()
    R[p_] = r / (1.4826 * np.median(np.abs(fr - fr.median())))
col["score"] = R.mean(1); col["n_y"] = R.notna().sum(1)
col["dl_proj"] = [ind2proj.get(dl.get(c)) for c in col.index]; col["fed_proj"] = [ind2proj.get(fed.get(c)) for c in col.index]
col["dl_male"] = col.dl_proj.map(projmale); col["fed_male"] = col.fed_proj.map(projmale)
ok = col.dl_male.notna() & col.score.notna()
lab = col.loc[ok, "dl_male"].astype(bool).values; v = col.loc[ok, "score"].values
au = mannwhitneyu(v[lab], v[~lab]).statistic / (lab.sum() * (~lab).sum())
print(f"DLPFC-only mapping: columns with a score and clinical sex {ok.sum()}; AUC(score, clinical male) {au:.3f}")
for k in ["y", "x"]:
    vv = col.loc[ok, k].values; o2 = ~np.isnan(vv)
    print(f"  component {k}: AUC {mannwhitneyu(vv[o2 & lab], vv[o2 & ~lab]).statistic / ((o2 & lab).sum() * (o2 & ~lab).sum()):.3f}")
# label-free threshold: 2-means on the score
s = np.sort(col.score.dropna().values); c1, c2 = np.percentile(s, 25), np.percentile(s, 75)
for _ in range(100):
    thr = (c1 + c2) / 2; c1, c2 = s[s <= thr].mean(), s[s > thr].mean()
print(f"2-means threshold {thr:.2f}; clusters {int((s <= thr).sum())}/{int((s > thr).sum())}")
col["call_male"] = col.score > thr
# margin: distance to threshold in units of pooled within-cluster sd
sd = np.sqrt((np.var(s[s <= thr]) + np.var(s[s > thr])) / 2); col["margin_sd"] = (col.score - thr) / sd
d = col[ok]
disc = d.call_male != d.dl_male.astype(bool)
print(f"DLPFC-only: discordant columns {int(disc.sum())} of {len(d)}; of those with |margin|>1 sd: {int((disc & (d.margin_sd.abs() > 1)).sum())}")
dd = col[col.fed_male.notna() & col.score.notna()]
print(f"as-fed mapping: discordant columns {int((dd.call_male != dd.fed_male.astype(bool)).sum())} of {len(dd)}")
# SDA donors: own columns under DLPFC-only
sda = col[col.dl_proj.isin(samps)]
print(f"SDA donors with own TMT column(s) (DLPFC-only): {sda.dl_proj.nunique()} (columns {len(sda)}); discordant donors {int(sda.groupby('dl_proj').apply(lambda g: (g.call_male != g.dl_male.astype(bool)).any()).sum())}; strong (|margin|>1sd) {int(sda.groupby('dl_proj').apply(lambda g: ((g.call_male != g.dl_male.astype(bool)) & (g.margin_sd.abs()>1)).any()).sum())}")
fsda = col[col.fed_proj.isin(samps)]
# the as-fed row of each SDA donor: ros1d keeps the last column per individual
print(f"SDA donors with a protein row as fed: {fsda.fed_proj.nunique()}")
fed_last = {}
for c, r in fed.items():
    pj = ind2proj.get(r)
    if pj in set(samps): fed_last[pj] = c
fl = col.loc[list(fed_last.values())]
fl_disc = fl.call_male != fl.fed_male.astype(bool)
print(f"  as-fed rows sex-discordant with the donor's clinical sex: {int(fl_disc.sum())} of {len(fl)}; among rows not the donor's own sample: {int((fl_disc & (fl.fed_proj.values != fl.dl_proj.values)).sum())} of {int((fl.fed_proj.values != fl.dl_proj.values).sum())}")
col.to_csv(f"{P}/04_prot_columns.csv")
key = pd.read_csv(f"{P}/donor_key.csv")
for pj, lb in key.values:
    g = col[col.dl_proj == pj]; gf = pj in fed_last
    if len(g) == 0:
        print(f"{lb}: no own TMT sample under DLPFC-only mapping; has an as-fed SDA protein row: {gf}" + (f" (that row is {'its own' if fed_last[pj] in g.index else 'ANOTHER individual'}'s)" if gf else "")); continue
    for c, r in g.iterrows():
        print(f"{lb}: own TMT column; Y proteins observed {r.n_y}; call {'AGREES' if r.call_male == r.dl_male else 'DISAGREES'} with clinical; margin {abs(r.margin_sd):.1f} sd; batch-mates {int((batch == batch[c]).sum())}; as-fed row is own: {fed_last.get(pj) == c}")
