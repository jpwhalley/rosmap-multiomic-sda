"""Build Data S1 for the npj Aging round-2 revision (v23): $G_ROS_RESULTS_DIR/v23_ds1/Data_S1.csv (the repository's Data_S1.csv).

Starts from Data_S1_corrected_547.csv ($G_ROS_TABLES_DIR; the 2026-10-08 corrected catalogue) and makes the file match the
Data S1 legend of the Supplementary Information:
  1. drops the three columns the legend withdraws (Top_Positive_proteins, Top_Negative_proteins, Protein_Pathway). The
     protein feature count of the 10-13 draft (protein_PIP05_features) is not carried: the legend does not describe it.
  2. adds the 12 columns the legend describes. Values come from the 2026-10-13 structure outputs, which
     revision_analyses/verify_structure re-derived with its own code (VST v07: all values agree):
       top_donor_share, participation_ratio, concentration_class   structure/out/05_master_prioritisation.csv
       Sex_adj_q, ceradsc_adj_q, cogdx_adj_q                        same (PO1 BH q over the 547 components), shown when < 0.05
       SEA-AD_spearman                                              same (same partner and genes as SEA-AD_corr)
       prioritised_2026, prioritisation_support                     same (05_prioritise.py rule; support listed when prioritised).
                                                                    2026-10-14 (v23 review, point 4): criterion C (cross-layer)
                                                                    is no longer a support; prioritised = distributed and
                                                                    (A1 or A2 or B), which leaves the 113 unchanged, and the
                                                                    "cross-layer" token is dropped from the support string
                                                                    (changes C74 only). v23_review/points_2_4_small_items
       protein_flag                                                 P1-P4, the rule of structure/scripts/07_dataS1.py, as codes
       score_r_corrected_protein                                    protein_influence consensus_547_scenarios.csv, V3_d_r
                                                                    (implementation 1, the calibrated one; scenario d)
       provenance                                                   D+A; C547 and C548 also carry the run-count note
     Each new value is checked against the verified draft (structure/out/DataS1_additions_draft.csv) before writing.
  3. keeps every retained field byte for byte (raw field text, quotes included), the row order (C1-C519, C521-C548),
     LF line endings, no BOM and a final newline.
Column order follows the legend; retained columns keep their v22 order relative to each other, so each pathway column
still follows its modality's feature columns.
omics_count is unchanged. It counts the four modalities, proteomics included, as the legend states and as the
decomposition-structure numbers in the text assume (61 components with >= 3 modalities; 539 of 540 slots). The 20 rows
whose count includes proteomics all carry a protein_flag.
Number formats: top_donor_share 4 dp and participation_ratio 1 dp (stated in the legend). The other new numbers follow
the corrected file's convention for floats (shortest repr of the float64 value, as in SEA-AD_corr and Sex/ceradsc/cogdx);
q columns are blank when q >= 0.05, as in the unadjusted columns.
No per-donor value is read into the output. Run (writes Data_S1.csv and build.log to $G_ROS_RESULTS_DIR/v23_ds1):
  python build_Data_S1.py
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, cfg.THREADS)
import csv, io, re, sys
import numpy as np, pandas as pd

SRC = f"{cfg.TABLES}/Data_S1_corrected_547.csv"
R2 = cfg.WORK
DST = f"{cfg.out_dir(R2, 'v23_ds1')}/Data_S1.csv"
ST = f"{R2}/revision_analyses/structure/out"
LOG = f"{R2}/v23_ds1/build.log"

RETAINED, NEW = "retained", "new"
COLUMNS = [  # legend order
    ("Component", RETAINED), ("omics_count", RETAINED),
    ("Top_positive_cell_type", RETAINED), ("Top_negative_cell_type", RETAINED),
    ("top_donor_share", NEW), ("participation_ratio", NEW), ("concentration_class", NEW),
    ("Sex", RETAINED), ("ceradsc", RETAINED), ("cogdx", RETAINED),
    ("Sex_adj_q", NEW), ("ceradsc_adj_q", NEW), ("cogdx_adj_q", NEW),
    ("SEA-AD_comp", RETAINED), ("SEA-AD_corr", RETAINED), ("SEA-AD_spearman", NEW),
    ("prioritised_2026", NEW), ("prioritisation_support", NEW),
    ("Top_Positive_RNA", RETAINED), ("Top_Negative_RNA", RETAINED), ("GeneExpression_Pathway", RETAINED),
    ("Top_Positive_meth", RETAINED), ("Top_Negative_meth", RETAINED), ("CpG_Site_Pathway", RETAINED),
    ("Top_Positive_chip", RETAINED), ("Top_Negative_chip", RETAINED),
    ("protein_flag", NEW), ("score_r_corrected_protein", NEW), ("provenance", NEW),
]
WITHDRAWN = ["Top_Positive_proteins", "Top_Negative_proteins", "Protein_Pathway"]
PROV_KEPT = "D+A"
PROV_NEW = "D+A; new ID (qualifies only under the corrected run count)"   # C547 (cluster 1465), C548 (cluster 716)
EXPECT = {  # facts ledger 14.3 (V2)
    "P1": [19, 37, 38, 56, 81, 94, 109, 125, 132, 155, 176, 201, 244, 279, 288, 296, 316, 318, 324, 338, 351, 380, 452],
    "P2": [183, 356, 425, 434, 490, 521, 533],
    "P3": [56, 125, 316, 318],
    "P4": [167, 177, 228, 281, 373, 403, 431, 459],
}


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s); self.f.write(s + "\n"); self.f.flush()


def raw_fields(line):
    """Split one CSV line into raw field texts (quotes kept), so retained fields can be copied byte for byte."""
    out, i, n = [], 0, len(line)
    while True:
        if i < n and line[i] == '"':
            j = i + 1
            while True:
                j = line.index('"', j)
                if j + 1 < n and line[j + 1] == '"':
                    j += 2; continue
                break
            out.append(line[i:j + 1]); i = j + 1
        else:
            j = line.find(",", i); j = n if j < 0 else j
            out.append(line[i:j]); i = j
        if i >= n:
            return out
        assert line[i] == ",", "malformed CSV line"
        i += 1
        if i == n:
            out.append(""); return out


def unquote(f):
    return f[1:-1].replace('""', '"') if f.startswith('"') else f


def quote(s):
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\r\n') else s


def fnum(v):
    """The corrected file's float convention: shortest repr of the float64 value."""
    return repr(float(v))


def main():
    log = Tee(LOG)
    log("build_Data_S1.py", pd.Timestamp.now().isoformat(timespec="seconds"), "| pandas", pd.__version__, "| numpy", np.__version__)

    # ---- source file: format facts and raw fields
    b = open(SRC, "rb").read()
    assert not b.startswith(b"\xef\xbb\xbf") and b"\r" not in b and b.endswith(b"\n"), "source is not LF, no-BOM, final newline"
    lines = b.decode("utf-8").split("\n")[:-1]
    parsed = list(csv.reader(io.StringIO(b.decode("utf-8"))))
    raw = [raw_fields(ln) for ln in lines]
    assert [[unquote(f) for f in r] for r in raw] == parsed, "raw field split disagrees with csv.reader"
    hdr = parsed[0]
    retained = [c for c, k in COLUMNS if k == RETAINED]
    assert sorted(hdr) == sorted(retained + WITHDRAWN) and len(hdr) == 20, "source columns differ from retained + withdrawn"
    ids = [int(r[0]) for r in parsed[1:]]
    assert ids == list(range(1, 520)) + list(range(521, 549)), "source IDs or order changed"
    log(f"source: {os.path.basename(SRC)} | {len(b)} bytes | {len(ids)} rows | {len(hdr)} columns")

    # ---- structure outputs (verified)
    M = pd.read_csv(f"{ST}/05_master_prioritisation.csv", index_col=0)
    assert list(M.index) == ids
    pi = pd.read_csv(f"{R2}/protein_influence/results/consensus_547_scenarios.csv").set_index("component_id").loc[ids]
    vp = pd.read_csv(f"{R2}/verify_protein_influence/results/consensus_fit_full_r1_mf.csv").set_index("component_id").loc[ids]
    mem = pd.read_csv(f"{R2}/verify_protein_influence/results/members_fit_full_r1_mf.csv",
                      usecols=["component_id", "n_prot_pip05", "dom_cat", "dom_share_base"])

    # protein flags: the rule of structure/scripts/07_dataS1.py
    p1 = sorted(int(k) for k in mem[(mem.n_prot_pip05 > 0) & (mem.dom_share_base > 0.5) & mem.dom_cat.isin([2, 3])].component_id.unique())
    p2 = sorted(int(k) for k in M.index[(M.proteins > 0) & (pi.V3_d_r < 0.95) & (vp.d_r < 0.95)])
    ratio_v = vp.d_dom_robz / vp.dom_robz_base
    p3 = sorted(int(k) for k in M.index[(M.top_share >= 0.5) & M.top_donor_misfed_proteome
                                        & (pi.V3_d_dom_score_ratio < 0.8) & (ratio_v < 0.8)])
    p4 = sorted(set(int(k) for k in M.index[M.proteins > 0]) - set(p1) - set(p2) - set(p3))
    got = {"P1": p1, "P2": p2, "P3": p3, "P4": p4}
    for f in EXPECT:
        assert got[f] == EXPECT[f], f"{f} differs from ledger 14.3"
    flag = {k: "; ".join(f for f in ("P1", "P2", "P3", "P4") if k in got[f]) for k in ids}
    log("protein flags reproduce ledger 14.3:", {f: len(v) for f, v in got.items()}, "| rows flagged:", sum(1 for v in flag.values() if v))

    # concentration class from the threshold rule, as a guard on the stored labels
    cls_rule = np.where(M.top_share < 0.10, "distributed", np.where(M.top_share < 0.50, "few-donor", "concentrated"))
    assert (M.conc_class.astype(str).values == cls_rule).all()

    # prioritisation without criterion C (cross-layer; v23 review point 4, 2026-10-14): distributed and (A1 or A2 or B)
    pri_noC = (M.top_share < 0.10) & (M.support_A1_phenotype_adj.astype(bool) | M.support_A2_gwas.astype(bool)
                                      | M.support_B_seaad.astype(bool))
    assert (pri_noC.values == M.prioritised.astype(bool).values).all(), "dropping criterion C changes the prioritised set"
    support_noC = {k: "; ".join(t for t in str(M.loc[k, "support"]).split("; ") if t != "cross-layer") for k in ids}
    log("prioritised without criterion C:", int(pri_noC.sum()), "| support strings changed by dropping 'cross-layer':",
        [k for k in ids if bool(pri_noC[k]) and support_noC[k] != str(M.loc[k, "support"])])

    new = {}
    for k in ids:
        r = M.loc[k]
        sp = r.seaad_spearman
        new[k] = {
            "top_donor_share": f"{r.top_share:.4f}",
            "participation_ratio": f"{r.PR:.1f}",
            "concentration_class": str(r.conc_class),
            "Sex_adj_q": fnum(r.adj_sex_q) if r.adj_sex_q < 0.05 else "",
            "ceradsc_adj_q": fnum(r.adj_cerad_q) if r.adj_cerad_q < 0.05 else "",
            "cogdx_adj_q": fnum(r.adj_cogdx_q) if r.adj_cogdx_q < 0.05 else "",
            "SEA-AD_spearman": "" if pd.isna(sp) else fnum(sp),
            "prioritised_2026": "yes" if bool(pri_noC[k]) else "",
            "prioritisation_support": support_noC[k] if bool(pri_noC[k]) else "",
            "protein_flag": flag[k],
            "score_r_corrected_protein": fnum(pi.loc[k, "V3_d_r"]),
            "provenance": PROV_NEW if k in (547, 548) else PROV_KEPT,
        }
    assert set(new[1]) == {c for c, kd in COLUMNS if kd == NEW}

    # ---- cross-check every new value against the verified draft (structure/out/DataS1_additions_draft.csv)
    D = pd.read_csv(f"{ST}/DataS1_additions_draft.csv", index_col=0, dtype=str, keep_default_na=False)
    assert [int(x) for x in D.index] == ids
    D.index = ids
    # the draft predates the 10-14 removal of criterion C: compare its support strings with the cross-layer token removed
    n_draft_C = sum(1 for k in ids if "cross-layer" in D.loc[k, "prioritisation_support"].split("; "))
    D["prioritisation_support"] = ["; ".join(t for t in s.split("; ") if t != "cross-layer") for s in D["prioritisation_support"]]
    log(f"  verified draft: rows whose support string listed 'cross-layer' (removed for the comparison): {n_draft_C}")
    same_text = ["top_donor_share", "participation_ratio", "concentration_class", "Sex_adj_q", "ceradsc_adj_q", "cogdx_adj_q",
                 "prioritised_2026", "prioritisation_support"]
    bad = {c: [k for k in ids if new[k][c] != D.loc[k, c]] for c in same_text}
    for c in ("SEA-AD_spearman", "score_r_corrected_protein"):   # the draft rounded these to 4 dp
        bad[c] = [k for k in ids if (f"{float(new[k][c]):.4f}" if new[k][c] else "") != D.loc[k, c]]
    bad["protein_flag"] = [k for k in ids if new[k]["protein_flag"] != "; ".join(s.split(" ")[0] for s in D.loc[k, "protein_flag"].split("; ") if s)]
    bad["provenance"] = [k for k in ids if new[k]["provenance"] != re.sub(r"; cluster \d+\)$", ")", D.loc[k, "provenance"])]
    for c, v in bad.items():
        log(f"  vs verified draft: {c:28s} {'identical in all 547 rows' if not v else 'DIFFERS ' + str(v[:10])}")
    unflagged = [k for k in ids if int(D.loc[k, "protein_PIP05_features"]) > 0 and not flag[k]]
    log(f"  rows with protein PIP > 0.5 features (draft count, not carried): {sum(int(x) > 0 for x in D.protein_PIP05_features)}"
        f"; without a protein_flag: {len(unflagged)}")
    assert not any(bad.values()) and not unflagged, "new values differ from the verified draft"

    # ---- assemble
    src_ix = {c: i for i, c in enumerate(hdr)}
    out = [",".join(quote(c) for c, _ in COLUMNS)]
    for rf, k in zip(raw[1:], ids):
        out.append(",".join(rf[src_ix[c]] if kind == RETAINED else quote(new[k][c]) for c, kind in COLUMNS))
    text = "\n".join(out) + "\n"
    chk = list(csv.reader(io.StringIO(text)))
    assert len(chk) == 548 and all(len(r) == len(COLUMNS) for r in chk)
    assert [int(r[0]) for r in chk[1:]] == ids
    for c, kind in COLUMNS:
        j = [cc for cc, _ in COLUMNS].index(c)
        if kind == RETAINED:
            assert [r[j] for r in chk] == [r[src_ix[c]] for r in parsed], c
    data = text.encode("utf-8")
    prev = open(DST, "rb").read() if os.path.exists(DST) else None
    open(DST, "wb").write(data)
    log(f"wrote {DST} | {len(data)} bytes | {len(chk) - 1} rows | {len(COLUMNS)} columns"
        + ("" if prev is None else (" | unchanged from the previous build" if prev == data else " | replaced a different earlier build")))
    log("dropped:", ", ".join(WITHDRAWN))
    log("added:", ", ".join(c for c, k in COLUMNS if k == NEW))
    n = lambda c: sum(1 for k in ids if new[k][c])
    log("non-blank counts:", {c: n(c) for c in ("Sex_adj_q", "ceradsc_adj_q", "cogdx_adj_q", "SEA-AD_spearman", "prioritised_2026",
                                               "prioritisation_support", "protein_flag")},
        "| classes:", pd.Series([new[k]["concentration_class"] for k in ids]).value_counts().to_dict())


if __name__ == "__main__":
    sys.exit(main())
