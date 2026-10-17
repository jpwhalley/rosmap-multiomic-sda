"""diff_against_cluster.py — validate scripts 02-05 outputs against the
cluster reference archive.

Run from the validation work directory:
    cd /path/to/work
    python /path/to/diff_against_cluster.py

Compares each generated output against the cluster's copy, classifying as:
    OK exact      — bit-for-bit identical
    OK ~          — numerically equivalent within float tolerance
    DIFF          — values differ beyond tolerance
    SHAPE MISMATCH — different dimensions
    MISSING       — file not present in one or both locations

Override paths via env vars if your layout differs:
    REF=/path/to/cluster/combat/ros WORK=/path/to/work python diff_against_cluster.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

WORK = os.environ.get("WORK", os.getcwd())
REF = os.environ.get("REF", os.environ.get("G_ROS_ROSMAP_DIR", "rosmap"))  # reference archive (layout of scripts/round2's G_ROS_ROSMAP_DIR)

# Each entry: (label, relative_path, kind)
#   kind: "csv"     — pandas read_csv, index col 0
#         "matrix"  — whitespace-separated numeric (SDA input format)
#         "text"    — line-by-line identifier files (md_*, genes, cell_types)
PAIRS = [
    ("chip_crossed", "oth/chip/chip_crossed_with_sc.csv", "csv"),
    ("meth_crossed", "oth/meth/meth_crossed_with_sc.csv", "csv"),
    ("prot_crossed", "prot/tmt/prot_crossed_with_sc.csv", "csv"),
    ("sda/rna.txt",  "sda/rna.txt",  "matrix"),
    ("sda/meth.txt", "sda/meth.txt", "matrix"),
    ("sda/chip.txt", "sda/chip.txt", "matrix"),
    ("sda/prot.txt", "sda/prot.txt", "matrix"),
    ("sda/md_samples", "sda/md_samples.txt", "text"),
    ("sda/md_cpg",     "sda/md_cpg.txt",     "text"),
    ("sda/md_chip",    "sda/md_chip.txt",    "text"),
    ("sda/md_prot",    "sda/md_prot.txt",    "text"),
    ("sda/genes",      "sda/genes.txt",      "text"),
    ("sda/cell_types", "sda/cell_types.txt", "text"),
]

RTOL = 1e-6
ATOL = 1e-9


def diff_csv(p_work: str, p_ref: str) -> str:
    a = pd.read_csv(p_work, index_col=0)
    b = pd.read_csv(p_ref, index_col=0)
    if a.shape != b.shape:
        return f"SHAPE MISMATCH  work={a.shape} ref={b.shape}"
    if not a.index.equals(b.index):
        return f"DIFF (index order or values differ)"
    if not a.columns.equals(b.columns):
        return f"DIFF (columns order or values differ)"
    av, bv = a.values, b.values
    if np.array_equal(av, bv, equal_nan=True):
        return f"OK exact         shape={a.shape}"
    if np.allclose(av, bv, rtol=RTOL, atol=ATOL, equal_nan=True):
        max_d = np.nanmax(np.abs(av - bv))
        return f"OK ~ (max|Δ|={max_d:.2e}) shape={a.shape}"
    diff_mask = ~(av == bv)
    diff_mask &= ~(np.isnan(av) & np.isnan(bv))
    n_diff = int(diff_mask.sum())
    max_d = np.nanmax(np.abs(av - bv))
    return f"DIFF             {n_diff} cells differ, max|Δ|={max_d:.3e}"


def diff_matrix(p_work: str, p_ref: str) -> str:
    # SDA inputs encode whole-sample missingness as the literal string "NA"
    # (consumed by SDA's `--ignore_missing true`); np.loadtxt chokes on that,
    # so use pandas with na_values="NA" instead.
    a = pd.read_csv(p_work, sep=r"\s+", header=None, na_values="NA").to_numpy()
    b = pd.read_csv(p_ref,  sep=r"\s+", header=None, na_values="NA").to_numpy()
    if a.shape != b.shape:
        return f"SHAPE MISMATCH  work={a.shape} ref={b.shape}"
    if np.array_equal(a, b, equal_nan=True):
        return f"OK exact         shape={a.shape}"
    if np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
        max_d = np.nanmax(np.abs(a - b))
        return f"OK ~ (max|Δ|={max_d:.2e}) shape={a.shape}"
    n_diff = int(np.sum(~(a == b) & ~(np.isnan(a) & np.isnan(b))))
    max_d = np.nanmax(np.abs(a - b))
    return f"DIFF             {n_diff} cells differ, max|Δ|={max_d:.3e}"


def diff_text(p_work: str, p_ref: str) -> str:
    with open(p_work) as f:
        a = f.read().splitlines()
    with open(p_ref) as f:
        b = f.read().splitlines()
    if a == b:
        return f"OK exact         {len(a)} lines"
    if len(a) != len(b):
        return f"LINE COUNT       work={len(a)} ref={len(b)}"
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    sample = ", ".join(str(i) for i in diffs[:5])
    return f"DIFF             {len(diffs)} lines differ (first: {sample})"


def main() -> int:
    print(f"WORK = {WORK}")
    print(f"REF  = {REF}")
    print()
    print(f"{'label':<22} {'kind':<8} {'result'}")
    print("-" * 90)

    bad = 0
    for label, rel, kind in PAIRS:
        p_work = os.path.join(WORK, rel)
        p_ref = os.path.join(REF, rel)
        if not os.path.exists(p_work) or not os.path.exists(p_ref):
            missing = []
            if not os.path.exists(p_work):
                missing.append("work")
            if not os.path.exists(p_ref):
                missing.append("ref")
            print(f"{label:<22} {kind:<8} MISSING in {','.join(missing)}")
            bad += 1
            continue

        try:
            if kind == "csv":
                msg = diff_csv(p_work, p_ref)
            elif kind == "matrix":
                msg = diff_matrix(p_work, p_ref)
            else:
                msg = diff_text(p_work, p_ref)
        except Exception as exc:
            msg = f"ERROR: {type(exc).__name__}: {exc}"
            bad += 1
        else:
            if not msg.startswith("OK"):
                bad += 1

        print(f"{label:<22} {kind:<8} {msg}")

    print()
    if bad == 0:
        print("All checks passed.")
        return 0
    print(f"{bad} mismatch(es) — investigate above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
