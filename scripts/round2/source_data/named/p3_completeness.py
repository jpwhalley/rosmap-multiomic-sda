"""Point 3: donors with all four modalities, as fed, under the corrected proteomics mapping, and after the
identity exclusions (2 snRNA, 4 H3K9ac, 3 TMT under the corrected mapping; Methods / ledger 5.1).
Inputs (read-only): verify_protein_influence/cache/in_{meth,chip,prot}.npy (as fed), prot_cat.npy and
protein_influence/cache/prot_group.npy (categories 0 none / 1 own / 2 other, no own / 3 other, own exists /
4 own dropped), identity_checks/results/15_flagged_donors_summary.csv + private/donor_key.csv (label -> donor).
Prints counts only: no donor IDs, labels are mapped to SDA row positions in memory."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = cfg.THREADS
import numpy as np, pandas as pd

RA = cfg.WORK
SDA = cfg.SDA
VC = f"{cfg.WORK}/verify_protein_influence/cache"
samps = [int(x) for x in open(f"{SDA}/md_samples.txt").read().split()]
N = len(samps); pos = {s: i for i, s in enumerate(samps)}

meth = ~np.isnan(np.load(f"{VC}/in_meth.npy")).all(1)
chip = ~np.isnan(np.load(f"{VC}/in_chip.npy")).all(1)
prot = ~np.isnan(np.load(f"{VC}/in_prot.npy")).all(1)
rna = np.ones(N, bool)
cat = np.load(f"{VC}/prot_cat.npy")
cat1 = np.load(f"{cfg.WORK}/protein_influence/cache/prot_group.npy")
assert np.array_equal(cat, cat1), "the two implementations' protein categories differ"
pcorr_arr = np.load(f"{VC}/prot_corr.npy")
prot_corr = ~np.isnan(pcorr_arr).all(1)
assert np.array_equal(prot_corr, np.isin(cat, [1, 3, 4])), "corrected-mapping presence != categories 1/3/4"
assert np.array_equal(prot, np.isin(cat, [1, 2, 3])), "as-fed presence != categories 1/2/3"
own_fed = cat == 1        # as-fed row is the donor's own proteome

print(f"donors {N}; RNA {rna.sum()} meth {meth.sum()} H3K9ac {chip.sum()} prot as fed {prot.sum()} prot corrected {prot_corr.sum()}")
print("categories:", {int(k): int((cat == k).sum()) for k in np.unique(cat)})


def combos(R, M, C, P, tag):
    allfour = R & M & C & P
    nmod = R.astype(int) + M + C + P
    print(f"\n-- {tag}: all four {int(allfour.sum())}; RNA+meth+H3K9ac (any prot) {int((R & M & C).sum())}; "
          f"per number of modalities " + str({int(k): int((nmod == k).sum()) for k in range(0, 5)}))
    lab = np.array(["".join(x) for x in zip(np.where(R, "R", "-"), np.where(M, "M", "-"), np.where(C, "C", "-"), np.where(P, "P", "-"))])
    print("   combinations:", dict(pd.Series(lab).value_counts().sort_index()))
    return allfour


A = combos(rna, meth, chip, prot, "(1) as fed (original mapping)")
B = combos(rna, meth, chip, prot_corr, "(2) corrected proteomics mapping")
print(f"\n   of the {int(A.sum())} as-fed complete donors: own proteome (cat 1) {int((A & (cat == 1)).sum())}, "
      f"another donor's proteome with no own sample (cat 2) {int((A & (cat == 2)).sum())}, other with own elsewhere (cat 3) {int((A & (cat == 3)).sum())}")
print(f"   corrected complete = {int(B.sum())}: kept from the 65 {int((A & B).sum())}, lost {int((A & ~B).sum())}, gained (own proteome restored, cat 4) {int((B & ~A).sum())}")
print(f"   as-fed complete with correctly paired proteomics (own = cat 1, or cat 3 whose fed row is not own): {int((A & own_fed).sum())}")

# identity exclusions
key = pd.read_csv(f"{cfg.PRIVATE}/donor_key.csv")
lab2pos = {l: pos[int(p)] for p, l in zip(key.projid, key.label) if int(p) in pos}
fl = pd.read_csv(f"{cfg.WORK}/identity_checks/results/15_flagged_donors_summary.csv")
print(f"\nidentity table rows {len(fl)}; labels mapped to SDA donors {sum(l in lab2pos for l in fl.label)}")
bad_rna = np.zeros(N, bool); bad_chip = np.zeros(N, bool); bad_prot = np.zeros(N, bool)
# per ledger 5.1 / Methods l.321: 2 snRNA (RNA-1 genotype-confirmed; RNA-2 sex-discordant, unverifiable),
# 4 H3K9ac (identity NOT own), 3 TMT (identity NOT own under the corrected mapping)
for _, r in fl.iterrows():
    i = lab2pos[r.label]
    if r.label in ("RNA-1", "RNA-2"):
        bad_rna[i] = True
    if str(r.identity_ChIP).startswith("NOT own"):
        bad_chip[i] = True
    if str(r.identity_TMT).startswith("NOT own"):
        bad_prot[i] = True
print(f"excluded profiles: snRNA {int(bad_rna.sum())}, H3K9ac {int(bad_chip.sum())}, TMT (corrected) {int(bad_prot.sum())}")
assert bad_rna.sum() == 2 and bad_chip.sum() == 4 and bad_prot.sum() == 3
assert not (bad_chip & ~chip).any() and not (bad_prot & ~prot_corr).any()
R2, C2, P2 = rna & ~bad_rna, chip & ~bad_chip, prot_corr & ~bad_prot
D = combos(R2, meth, C2, P2, "(3) corrected mapping, identity-flagged profiles removed")
print(f"   lost from (2) by identity exclusion: {int((B & ~D).sum())} (via snRNA {int((B & bad_rna).sum())}, H3K9ac {int((B & bad_chip).sum())}, TMT {int((B & bad_prot).sum())})")
print(f"   per-modality n after exclusion: RNA {int(R2.sum())} meth {int(meth.sum())} H3K9ac {int(C2.sum())} TMT {int(P2.sum())}")
print(f"   donors complete in both (1) and (3): {int((A & D).sum())}")
# as-fed complete AND own as-fed proteome AND no identity flag in any assay used
A_ok = A & own_fed & ~bad_rna & ~bad_chip & ~bad_prot
print(f"   of the 65 as-fed complete donors, those whose four as-fed profiles are all the donor's own (cat 1 and no identity flag): {int(A_ok.sum())}")
