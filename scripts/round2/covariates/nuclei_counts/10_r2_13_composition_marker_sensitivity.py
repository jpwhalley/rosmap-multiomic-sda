#!/usr/bin/env python3
"""
Reviewer 2.13: composition-vs-activity and single-marker sensitivity.

Reviewer 2 asked whether component interpretations could reflect cell-type
composition rather than cell-state/activity, and whether single canonical
markers such as PLP1 could drive a component's interpretation.

This script applies two bounded checks to the seven headline components:

1. Component sample scores vs donor cell counts / cell-type proportions.
   If h5py is available, exact donor x broad-cell-type counts are read from
   the AnnData metadata (`projid`, `major_cell_type`, `predicted_doublets`) and
   filtered using the same rules as the original pseudobulk preparation:
   failed projids removed, predicted doublets removed, six broad cell types
   retained, and donors required to have >=100 cells in every broad type.

   If h5py is not available, the script still runs the total-cell-count
   correlation from the cached `mit_sample_cell_counts.csv` file, but skips the
   exact cell-type-proportion table.

2. Single-marker sensitivity for canonical marker genes. This is not a refit
   of SDA. It asks a narrower, auditable question: what fraction of a
   component's RNA loading strength (sum |loading| over PIP > 0.5 genes) is
   explained by the strongest single gene and by the strongest canonical
   broad-cell-type marker gene? If that fraction is small, the textual
   interpretation is not being carried by one canonical marker.

Read-only inputs: the deposit ($G_ROS_DEPOSIT_DIR) and the controlled ROSMAP
data ($G_ROS_ROSMAP_DIR; version-1 consensus loadings in sda/). Outputs
(per donor) are written to $G_ROS_RESULTS_DIR/revision_r2/.
"""

from __future__ import annotations
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)

from pathlib import Path
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


OUT = Path(cfg.out_dir(cfg.WORK, "revision_r2"))
SYNAPSE = Path(cfg.DEPOSIT)
ARCHIVE = Path(cfg.SDA)       # version-1 consensus loadings (identical to the version-1 results folder copies)
CLUSTER_ROS = Path(cfg.ROS)

H5AD = CLUSTER_ROS / "mit" / "rna" / "mit_all_doublets_labelled.h5ad"
TOTAL_CELL_COUNTS = CLUSTER_ROS / "mit" / "rna" / "mit_sample_cell_counts.csv"

HEADLINE_COMPONENTS = [2, 21, 74, 87, 177, 278, 351]
CELL_TYPES = ["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]
FAILED_PROJIDS = set(cfg.qc_failed_projids())   # the two QC-failed projids ($G_ROS_QC_EXCLUDE_PATH)
MIN_CELLS_PER_TYPE = 100
PIP_CUT = 0.5

CANONICAL_MARKERS: dict[str, list[str]] = {
    "Ast": ["AQP4", "GFAP", "ALDH1L1", "SLC1A3", "SLC1A2", "GJA1"],
    "Oli": ["PLP1", "MBP", "MOG", "MOBP", "MAG", "CNP"],
    "OPC": ["PDGFRA", "VCAN", "CSPG4", "OLIG1", "OLIG2", "SOX10", "BCAN"],
    "Exc": ["SLC17A7", "CAMK2A", "SATB2", "RBFOX3", "NEUROD6", "GRIN2B"],
    "Inh": ["GAD1", "GAD2", "SLC6A1", "PVALB", "SST", "VIP", "LHX6"],
    "Mic": [
        "P2RY12", "TMEM119", "CX3CR1", "CSF1R", "AIF1", "C1QA", "C1QB",
        "C1QC", "CD68", "HLA-DRA", "HLA-DRB1", "TREM2", "APOE",
    ],
}


def decode_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O"}:
        return np.array([
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            for x in arr
        ])
    return arr


def read_obs_column(obs: Any, name: str) -> np.ndarray:
    obj = obs[name]
    # AnnData categorical columns are stored as a group with integer codes and
    # categories. Plain numeric/string columns are stored as datasets.
    if hasattr(obj, "keys"):
        if "codes" in obj and "categories" in obj:
            codes = np.asarray(obj["codes"][:])
            cats = decode_array(obj["categories"][:])
            out = np.empty(len(codes), dtype=object)
            out[:] = None
            valid = codes >= 0
            out[valid] = cats[codes[valid]]
            return out
        if "values" in obj:
            return decode_array(obj["values"][:])
        raise KeyError(f"Do not know how to read obs column group {name!r}: {list(obj.keys())}")
    return decode_array(obj[:])


def to_bool(values: np.ndarray) -> np.ndarray:
    return np.array([
        bool(x) if isinstance(x, (bool, np.bool_)) else str(x).lower() == "true"
        for x in values
    ])


def derive_celltype_counts_from_h5ad() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    cached_counts = OUT / "donor_celltype_counts_from_h5ad.csv"
    cached_proportions = OUT / "donor_celltype_proportions_from_h5ad.csv"
    if os.environ.get("R2_13_FORCE_H5AD", "0") != "1" and cached_counts.exists() and cached_proportions.exists():
        counts = pd.read_csv(cached_counts, index_col=0)
        proportions = pd.read_csv(cached_proportions, index_col=0)
        counts.index = counts.index.astype(str)
        proportions.index = proportions.index.astype(str)
        return counts, proportions, "loaded_cached_h5ad_counts"

    try:
        import h5py  # type: ignore
    except ImportError:
        return None, None, "h5py_not_available"

    if not H5AD.exists():
        return None, None, f"h5ad_missing:{H5AD}"

    with h5py.File(H5AD, "r") as f:
        obs = f["obs"]
        required = ["projid", "major_cell_type", "predicted_doublets"]
        missing = [x for x in required if x not in obs]
        if missing:
            return None, None, f"missing_obs_columns:{','.join(missing)}"

        projid = pd.Series(read_obs_column(obs, "projid")).astype(int)
        major = pd.Series(read_obs_column(obs, "major_cell_type")).astype(str)
        doublet = pd.Series(to_bool(read_obs_column(obs, "predicted_doublets")))

    keep = (
        ~projid.isin(FAILED_PROJIDS)
        & ~doublet
        & major.isin(CELL_TYPES)
    )
    counts = (
        pd.crosstab(projid[keep], major[keep])
        .reindex(columns=CELL_TYPES, fill_value=0)
        .astype(int)
    )
    valid = counts.ge(MIN_CELLS_PER_TYPE).all(axis=1)
    counts = counts.loc[valid].copy()
    counts.index = counts.index.astype(str)
    counts.index.name = "projid"

    proportions = counts.div(counts.sum(axis=1), axis=0)
    proportions.index.name = "projid"

    counts.to_csv(OUT / "donor_celltype_counts_from_h5ad.csv")
    proportions.to_csv(OUT / "donor_celltype_proportions_from_h5ad.csv")
    return counts, proportions, "derived_from_h5ad"


def load_total_cell_counts() -> pd.Series:
    total = pd.read_csv(TOTAL_CELL_COUNTS, index_col=0)
    # File has one column named `projid`, but values are total cells per donor.
    s = total.iloc[:, 0].copy()
    s.index = s.index.astype(str)
    s.name = "total_cells"
    return s


def corr_pair(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    common = x.dropna().index.intersection(y.dropna().index)
    x2 = x.loc[common].astype(float)
    y2 = y.loc[common].astype(float)
    if len(common) < 3 or x2.nunique() < 2 or y2.nunique() < 2:
        return {
            "n": int(len(common)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }
    pr, pp = stats.pearsonr(x2, y2)
    sr, sp = stats.spearmanr(x2, y2)
    return {
        "n": int(len(common)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def add_bh(df: pd.DataFrame, p_col: str, q_col: str) -> pd.DataFrame:
    out = df.copy()
    mask = out[p_col].notna()
    out[q_col] = np.nan
    if mask.any():
        out.loc[mask, q_col] = multipletests(out.loc[mask, p_col], method="fdr_bh")[1]
    return out


def composition_correlations(samples: pd.DataFrame, proportions: pd.DataFrame | None) -> None:
    total_cells = load_total_cell_counts()

    total_rows = []
    for comp in HEADLINE_COMPONENTS:
        score = samples.loc[comp]
        result = corr_pair(score, total_cells)
        total_rows.append({"component": comp, "covariate": "total_cells", **result})
    total_df = add_bh(pd.DataFrame(total_rows), "pearson_p", "pearson_q_bh")
    total_df = add_bh(total_df, "spearman_p", "spearman_q_bh")
    total_df.to_csv(OUT / "headline_components_total_cell_count_correlations.csv", index=False)

    if proportions is None:
        return

    prop_rows = []
    for comp in HEADLINE_COMPONENTS:
        score = samples.loc[comp]
        for cell_type in CELL_TYPES:
            result = corr_pair(score, proportions[cell_type])
            prop_rows.append({"component": comp, "cell_type": cell_type, **result})
    prop_df = add_bh(pd.DataFrame(prop_rows), "pearson_p", "pearson_q_bh")
    prop_df = add_bh(prop_df, "spearman_p", "spearman_q_bh")
    prop_df["abs_pearson_r"] = prop_df["pearson_r"].abs()
    prop_df.to_csv(OUT / "headline_components_celltype_proportion_correlations.csv", index=False)

    summary = (
        prop_df.sort_values(["component", "abs_pearson_r"], ascending=[True, False])
        .groupby("component", as_index=False)
        .first()
    )
    summary = summary[[
        "component", "cell_type", "n", "pearson_r", "pearson_p",
        "pearson_q_bh", "spearman_r", "spearman_p", "spearman_q_bh",
    ]]
    summary.to_csv(OUT / "headline_components_celltype_proportion_top_correlations.csv", index=False)


def marker_sensitivity() -> None:
    load = pd.read_csv(ARCHIVE / "genes_final_RNA.csv", index_col=0)
    pip = pd.read_csv(ARCHIVE / "pip_final_RNA.csv", index_col=0)
    load.index = load.index.astype(int)
    pip.index = pip.index.astype(int)

    marker_rows = []
    summary_rows = []
    marker_to_type = {
        marker: cell_type
        for cell_type, markers in CANONICAL_MARKERS.items()
        for marker in markers
    }
    all_markers = sorted(marker_to_type)

    for comp in HEADLINE_COMPONENTS:
        comp_load = load.loc[comp]
        comp_pip = pip.loc[comp]
        called = comp_pip > PIP_CUT
        called_load = comp_load[called].dropna()
        total_abs = float(called_load.abs().sum())

        if len(called_load) > 0 and total_abs > 0:
            top_gene = called_load.abs().idxmax()
            top_gene_loading = float(comp_load[top_gene])
            top_gene_fraction = abs(top_gene_loading) / total_abs
        else:
            top_gene = None
            top_gene_loading = np.nan
            top_gene_fraction = np.nan

        top_marker = None
        top_marker_loading = np.nan
        top_marker_fraction = 0.0

        for marker in all_markers:
            in_universe = marker in comp_load.index
            in_called = bool(called.get(marker, False)) if in_universe else False
            loading = float(comp_load[marker]) if in_universe else np.nan
            marker_pip = float(comp_pip[marker]) if in_universe else np.nan
            fraction = abs(loading) / total_abs if in_called and total_abs > 0 else 0.0

            if in_called and fraction > top_marker_fraction:
                top_marker = marker
                top_marker_loading = loading
                top_marker_fraction = fraction

            marker_rows.append({
                "component": comp,
                "marker": marker,
                "canonical_cell_type": marker_to_type[marker],
                "in_RNA_universe": in_universe,
                "in_component_PIP_gt_0_5": in_called,
                "loading": loading,
                "PIP": marker_pip,
                "abs_loading_fraction_of_component_RNA_L1": fraction,
            })

        summary_rows.append({
            "component": comp,
            "n_RNA_features_PIP_gt_0_5": int(called.sum()),
            "component_RNA_L1_over_PIP_gt_0_5": total_abs,
            "top_gene": top_gene,
            "top_gene_loading": top_gene_loading,
            "top_gene_abs_loading_fraction": top_gene_fraction,
            "top_canonical_marker": top_marker,
            "top_canonical_marker_loading": top_marker_loading,
            "top_canonical_marker_abs_loading_fraction": top_marker_fraction,
            "PLP1_in_RNA_universe": "PLP1" in comp_load.index,
            "PLP1_in_component_PIP_gt_0_5": bool(called.get("PLP1", False)) if "PLP1" in comp_load.index else False,
            "PLP1_loading": float(comp_load["PLP1"]) if "PLP1" in comp_load.index else np.nan,
            "PLP1_PIP": float(comp_pip["PLP1"]) if "PLP1" in comp_load.index else np.nan,
            "PLP1_abs_loading_fraction": (
                abs(float(comp_load["PLP1"])) / total_abs
                if "PLP1" in comp_load.index and bool(called.get("PLP1", False)) and total_abs > 0
                else 0.0
            ),
        })

    marker_df = pd.DataFrame(marker_rows)
    summary_df = pd.DataFrame(summary_rows)
    marker_df.to_csv(OUT / "headline_components_canonical_marker_sensitivity.csv", index=False)
    summary_df.to_csv(OUT / "headline_components_canonical_marker_sensitivity_summary.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(SYNAPSE / "samples_final.csv", index_col=0)
    samples.index = samples.index.astype(int)
    samples.columns = samples.columns.astype(str)

    counts, proportions, status = derive_celltype_counts_from_h5ad()
    pd.DataFrame([{
        "h5ad_path": str(H5AD),
        "celltype_proportion_status": status,
        "n_donors_with_exact_celltype_counts": 0 if counts is None else counts.shape[0],
    }]).to_csv(OUT / "r2_13_composition_extraction_status.csv", index=False)
    print(f"Cell-type proportion extraction: {status}")

    composition_correlations(samples, proportions)
    marker_sensitivity()

    print("Wrote R2.13 outputs to", OUT)
    for name in [
        "r2_13_composition_extraction_status.csv",
        "headline_components_total_cell_count_correlations.csv",
        "headline_components_celltype_proportion_correlations.csv",
        "headline_components_celltype_proportion_top_correlations.csv",
        "headline_components_canonical_marker_sensitivity.csv",
        "headline_components_canonical_marker_sensitivity_summary.csv",
    ]:
        p = OUT / name
        if p.exists():
            print(" ", p.name, pd.read_csv(p).shape)


if __name__ == "__main__":
    main()
