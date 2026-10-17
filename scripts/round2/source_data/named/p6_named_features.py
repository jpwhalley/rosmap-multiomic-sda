"""Point 6: the named single-donor features (Results l.209; Figs S8f, S9f, S10f) with and without the dominant donor.
Independent of v23_panels: RNA from verify_protein_influence/cache/in_rna.npy (verifier's parse of rna.txt),
H3K9ac from in_chip.npy, gene/peak names from the SDA archive, scores from the release samples_final.csv,
identity flags from identity_checks (RNA-1/RNA-2 for snRNA; NOT-own H3K9ac rows). Layout of rna.txt is
established here from XIST vs recorded sex, not assumed.
Prints aggregates only (rho, p, n, ranks); no donor IDs or per-donor values."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = cfg.THREADS
import numpy as np, pandas as pd
from scipy import stats as ss

RA = cfg.WORK
SDA = cfg.SDA
ROS = cfg.ROS
REL = cfg.CATALOGUE   # outputs of scripts/07_cluster_components.py
VC = f"{cfg.WORK}/verify_protein_influence/cache"
samps = [int(x) for x in open(f"{SDA}/md_samples.txt").read().split()]
N = len(samps); pos = {s: i for i, s in enumerate(samps)}
genes = [l.strip() for l in open(f"{SDA}/genes.txt") if l.strip()]
peaks = [l.strip() for l in open(f"{SDA}/md_chip.txt") if l.strip()]
cts = [l.strip() for l in open(f"{SDA}/cell_types.txt") if l.strip()]
G = len(genes)

R = np.load(f"{VC}/in_rna.npy", mmap_mode="r")
assert R.shape == (len(cts) * N, G)
clin = pd.read_csv(f"{ROS}/oth/meta/ROSMAP_clinical.csv"); clin.index = clin.projid.astype(int)
male = clin.loc[samps].msex.values == 1
jx = genes.index("XIST")
for name, L in [("celltype-major (6,276,G)", np.asarray(R[:, jx]).reshape(len(cts), N)),
                ("donor-major (276,6,G)", np.asarray(R[:, jx]).reshape(N, len(cts)).T)]:
    auc = [ss.mannwhitneyu(L[c][~male], L[c][male]).statistic / ((~male).sum() * male.sum()) for c in range(len(cts))]
    print(f"layout {name}: XIST female>male AUC per block {np.round(auc, 3)}")
RNA = np.asarray(R).reshape(len(cts), N, G)          # confirmed above if AUCs ~1 for every block
CH = np.load(f"{VC}/in_chip.npy")

pub = pd.read_csv(f"{REL}/samples_final.csv", index_col=0)
assert [int(c) for c in pub.columns] == samps

key = pd.read_csv(f"{cfg.PRIVATE}/donor_key.csv")
lab2pos = {l: pos[int(p)] for p, l in zip(key.projid, key.label) if int(p) in pos}
fl = pd.read_csv(f"{cfg.WORK}/identity_checks/results/15_flagged_donors_summary.csv")
exR = np.zeros(N, bool); exC = np.zeros(N, bool)
for _, r in fl.iterrows():
    if r.label in ("RNA-1", "RNA-2"): exR[lab2pos[r.label]] = True
    if str(r.identity_ChIP).startswith("NOT own"): exC[lab2pos[r.label]] = True
print(f"excluded: snRNA {exR.sum()}, H3K9ac {exC.sum()}")


def dominant(a):
    d = a - a.mean(); return int(np.argmax(d ** 2)), float(np.max(d ** 2) / np.sum(d ** 2))


def robz_top(a):
    med = np.median(a); mad = np.median(np.abs(a - med)) * 1.4826; return int(np.argmax(np.abs((a - med) / mad)))


rows = []
def view(comp, feat, y, keep):
    a = pub.loc[comp].values.astype(float)
    dom, share = dominant(a)
    ok = keep & ~np.isnan(y)
    assert ok[dom], "dominant donor is excluded"
    ext_in_set = np.where(ok)[0][robz_top(a[ok])]
    r_all, p_all = ss.spearmanr(y[ok], a[ok])
    ok2 = ok.copy(); ok2[dom] = False
    r_wo, p_wo = ss.spearmanr(y[ok2], a[ok2])
    # full donor set (no identity exclusion), for reference
    okf = ~np.isnan(y); r_f, p_f = ss.spearmanr(y[okf], a[okf]); okf2 = okf.copy(); okf2[dom] = False
    r_f2, p_f2 = ss.spearmanr(y[okf2], a[okf2])
    rank = int(np.sum(y[ok] > y[dom])) + 1   # 1 = highest
    rank_low = int(np.sum(y[ok] < y[dom])) + 1
    rows.append(dict(component=comp, feature=feat, n=int(ok.sum()), rho=r_all, p=p_all, n_wo_dom=int(ok2.sum()), rho_wo_dom=r_wo, p_wo_dom=p_wo,
                     dom_rank_high=rank, dom_rank_low=rank_low, dom_is_most_extreme=bool(ext_in_set == dom), dom_share=share,
                     n_full=int(okf.sum()), rho_full=r_f, p_full=p_f, rho_full_wo_dom=r_f2, p_full_wo_dom=p_f2))


for ct in ["Ast", "Exc", "Inh", "Mic", "OPC", "Oli"]:
    view(351, f"IGKC {ct}", RNA[cts.index(ct), :, genes.index("IGKC")].astype(float), ~exR)
view(87, "VTCN1 Ast", RNA[cts.index("Ast"), :, genes.index("VTCN1")].astype(float), ~exR)
for pk in ["peak16865", "peak16866", "peak16867", "peak16868", "peak16869"]:
    view(2, pk, CH[:, peaks.index(pk)].astype(float), ~exC)
T = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
T.to_csv(f"{cfg.out_dir(RA, 'v23_review', 'points_1_3_6', 'out')}/p6_named_features.csv", index=False)
nm = T[T.component == 351]
nonmic = nm[nm.feature != "IGKC Mic"]
print(f"\nC351 IGKC, five non-microglial classes: rho (n {nonmic.n.iloc[0]}) {nonmic.rho.min():+.3f} to {nonmic.rho.max():+.3f}; "
      f"without dominant (n {nonmic.n_wo_dom.iloc[0]}) {nonmic.rho_wo_dom.min():+.3f} to {nonmic.rho_wo_dom.max():+.3f}; max |rho| wo {nonmic.rho_wo_dom.abs().max():.3f}; "
      f"p wo dom min {nonmic.p_wo_dom.min():.2g}; dominant ranks {sorted(nonmic.dom_rank_high.tolist())}")
c2 = T[T.component == 2]
print(f"C2 peaks: rho (n {c2.n.iloc[0]}) {c2.rho.min():+.3f} to {c2.rho.max():+.3f}; without dominant {c2.rho_wo_dom.min():+.3f} to {c2.rho_wo_dom.max():+.3f}; dominant lowest at every peak: {bool((c2.dom_rank_low == 1).all())}")
