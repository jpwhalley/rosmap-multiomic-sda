"""Attribution of the changes seen in 01_: which covariate moves which calls, and what the nuclei covariate is.
Aggregates only. Reuses 01_'s data loading by import-free copy (kept short)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, numpy as np, pandas as pd
from scipy import stats as ss
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = cfg.out_dir(cfg.WORK, "covariates")
exec(open(f"{HERE}/_common.py").read())

PH = ["Sex", "ceradsc", "cogdx"]
pub = {c: set(ref.index[ref[c] < 0.05]) for c in PH}
base = {"Sex": ["age", "pmi", "ros"], "ceradsc": ["age", "pmi", "ros", "sex"], "cogdx": ["age", "pmi", "ros", "sex"]}

print("Published q distribution of calls (how close to 0.05):")
for ph in PH:
    q = ref.loc[sorted(pub[ph]), ph]
    print(f"  {ph}: n={len(q)}; q<0.01: {(q<0.01).sum()}, 0.01-0.03: {((q>=0.01)&(q<0.03)).sum()}, 0.03-0.05: {(q>=0.03).sum()}")

print("\nOne covariate at a time (RINT ~ phenotype + covariate), calls significant / published kept:")
for ph in PH:
    line = []
    for cov in [[], ["age"], ["pmi"], ["ros"], ["sex"], ["lognuc"], ["logupn"], ["age90p", "age"], base[ph], base[ph] + ["lognuc"], base[ph] + ["logupn"], base[ph] + ["lognuc", "logupn"]]:
        if ph == "Sex" and "sex" in cov: continue
        p, _, _ = nested_F(PHTERM[ph], cov, SUB[ph]); q = bh(p); sig = set(S.index[q < 0.05])
        line.append(f"[{'+'.join(cov) or 'none'}] {len(sig)} (kept {len(sig & pub[ph])})")
    print(f"  {ph}: " + "; ".join(line))

# what is the nuclei count? relation to disease per class (aggregates)
qc = pd.read_csv(f"{cfg.WORK}/covariates/rna_qc_donor.csv", index_col=0); qc.index = qc.index.astype(str); qc = qc.loc[don]
CT = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
prop = qc[[f"nuclei_{c}" for c in CT]].div(qc.nuclei, axis=0)
print("\nNuclei by CERAD group (median nuclei per class; median class proportion) and Mann-Whitney p:")
for c in CT:
    a = qc.loc[D.cerad_ad == 1, f"nuclei_{c}"]; b = qc.loc[D.cerad_ad == 0, f"nuclei_{c}"]
    pa = prop.loc[D.cerad_ad == 1, f"nuclei_{c}"]; pb = prop.loc[D.cerad_ad == 0, f"nuclei_{c}"]
    print(f"  {c}: AD {a.median():.0f} vs non-AD {b.median():.0f} (p={ss.mannwhitneyu(a,b).pvalue:.2g}); proportion AD {pa.median():.3f} vs {pb.median():.3f} (p={ss.mannwhitneyu(pa,pb).pvalue:.2g})")
a = qc.loc[D.cerad_ad == 1, "nuclei"]; b = qc.loc[D.cerad_ad == 0, "nuclei"]
print(f"  total: AD {a.median():.0f} vs non-AD {b.median():.0f} (p={ss.mannwhitneyu(a,b).pvalue:.2g})")
print("Spearman of log nuclei with: age %.2f (p=%.2g), pmi %.2f (p=%.2g), ceradsc(1-4) %.2f (p=%.2g), braak %.2f (p=%.2g)" % (
    *ss.spearmanr(D.lognuc, D.age), *ss.spearmanr(D.lognuc, D.pmi), *ss.spearmanr(D.lognuc, D.ceradsc), *ss.spearmanr(D.lognuc, clin.braaksc)))
# does nuclei count track the Exc (neuronal) proportion?
print("Spearman log nuclei vs Exc proportion %.2f; vs Oli proportion %.2f" % (ss.spearmanr(D.lognuc, prop.nuclei_Exc)[0], ss.spearmanr(D.lognuc, prop.nuclei_Oli)[0]))
# overlap: of components associated with log nuclei, how many are CERAD/cogdx calls?
X = np.column_stack([np.ones(len(D)), D.lognuc.values]); bb, *_ = np.linalg.lstsq(X, Y, rcond=None)
r = Y - X @ bb; rss = (r**2).sum(0); tss = ((Y - Y.mean(0))**2).sum(0); F = (tss - rss) / (rss / (len(D) - 2)); pn = ss.f.sf(F, 1, len(D) - 2)
nucsig = set(S.index[bh(pn) < 0.05])
for ph in PH:
    print(f"  {ph}: published calls also associated with log nuclei (q<0.05): {len(pub[ph] & nucsig)}/{len(pub[ph])}")
