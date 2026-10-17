"""
07_cluster_components.py
========================

Pool the components from the ten SDA runs and apply stability-based
clustering to identify the reproducible components: 547 components, with
component IDs 1-519 and 521-548 (see "Component IDs" below). Produces the
component-loading matrices that go into the Synapse deposit.

Manuscript: Whalley J.P. et al., *Multi-omic Analysis of Sex and Alzheimer's
Disease in the Aged Human Cortex* (under review at npj Aging).
Methods §"Tensor and matrix decomposition".

Original filename in the working tree: `ros2_cluster.py`. That script read
every run's matrices into memory at once (tens of GB) and took ~4 hours on a
cluster node. This version computes the same quantities in two passes
(sample and cell-type loadings first; then one omic at a time, reading only
the rows of components that belong to a retained cluster) and computes the
pairwise distances in one vectorised step, so it runs on a workstation
(13 minutes and 8.6 GB peak memory on a 12-core machine).

Pipeline:
1. For each of the `NUM_RUNS` SDA runs (10), read the `it{ITERATION}`
   checkpoint: A (sample loadings), B1 (cell-type loadings of the RNA
   tensor), X1..X4 (feature loadings) and S1..S4 (PIPs) for RNA /
   methylation / ChIP / proteomics. The runs were launched with
   `--max_iter 3000 --save_freq 500` and ended at the cluster's 10-day time
   limit after 1,803-2,006 iterations; it1500 is the last checkpoint common
   to all ten runs, and the components are taken from it.
2. Drop components whose sample loadings are all zero; give the surviving
   components a global position across runs (run 1 first).
3. Cluster the pooled components by complete-linkage hierarchical
   clustering on 1 - |Pearson r| of their sample loadings, and cut the tree
   at `COPHENETIC_THRESHOLD` (0.4).
4. Keep a cluster only if its members come from at least
   `MIN_RUNS_PER_CLUSTER` (5) of the 10 runs, counting each member's own run.
5. Within each retained cluster, align each member to the cluster's first
   member and average:
   - sample flip fA: -1 if the Pearson r between the member's and the first
     member's sample loadings is negative, else +1;
   - feature flips fX1..fX4: the same rule, for each feature block;
   - cell-type flip fB = fA x fX1, so that each member's
     sample x cell-type x gene product (A.B1.X1) keeps its sign.
   A NaN correlation (constant vector) counts as no flip, as in the
   original. The representative loadings are the unweighted means of the
   aligned members; PIPs are the median across members, without alignment.
6. Assign component IDs from `component_id_map.csv` and write the outputs
   in component-ID order.

Corrections relative to the deposit of record (Synapse syn75275226,
version 1, and Data S1 as submitted on 2026-08-19), made 2026-10-08:
- Run count (step 4). The version-1 deposit was built with an earlier
  form of step 2 whose position labels overlapped by one between
  consecutive runs, so step 4 miscounted the runs of 14 of the 1,104
  clusters with at least 5 members; for three of them this changed the
  outcome: it kept one cluster drawn from 4 runs (historical C520) and
  missed two clusters drawn from 5 runs (now C547 and C548). The other 545
  components have the same members, and the same sample and feature
  loadings, as in version 1.
- Cell-type flips (step 5). The version-1 deposit decided each member's
  cell-type flip from the Pearson r of its own 6-value cell-type vector,
  separately from fA and fX1. Pearson r centres the vector and so ignores
  its overall sign; where that decision differed from fA x fX1, the member
  entered the average with its sample x cell-type x gene product reversed.
  Here fB = fA x fX1. Sample, feature and PIP aggregation are unchanged.

Component IDs:
The published component IDs are kept. Historical C520 is retired, and the
two clusters that qualify under the correct run count are appended as C547
and C548, in their order of first appearance among the retained clusters.
Component ID is therefore not the row number: rows are written in ID order
(1-519, 521-548). `component_id_map.csv` (next to this script) is the
explicit map. It keys each component on its cluster's first member (the
run, and the 0-based index of the component in that run's SDA output) and
records the expected numbers of members and runs; the script stops if the
clustering does not reproduce it. Join every matrix and table by
component ID, never by position.

Outputs (in `OUTPUT_DIR`, rows indexed by component ID):
- `samples_final.csv` - 547 components x 276 samples
- `treatment_final.csv` - 547 components x 6 cell types (deposited as
  `cell_type.csv`)
- `genes_final_RNA.csv`, `genes_final_meth.csv`, `genes_final_chip.csv`,
  `genes_final_proteins.csv` - component x feature loadings (unmasked)
- `pip_final_RNA.csv`, `pip_final_meth.csv`, `pip_final_chip.csv`,
  `pip_final_proteins.csv` - component x feature PIPs (median over members)
- `component_catalogue.csv` - one row per component: cluster, numbers of
  members and runs, first member
- `cluster_members.csv` - one row per member: run, SDA index, the flips
  applied, and (for reference) the cell-type flip the version-1 deposit used

The PIP > 0.5 mask used in the deposit's `*.txt` files is applied **after**
this script: it is the post-clustering mask that produces `RNA.txt`,
`meth.txt` and `chip.txt` (and, in version 1 of the deposit, `proteins.txt`)
from the unmasked `genes_final_*.csv` and the `pip_final_*.csv`.

Diagnostic plots (the dendrogram + cluster-size histogram from the original
script) are produced only when `MAKE_DIAGNOSTIC_PLOTS = True`. They are
useful for QC but do not appear in the manuscript.

Usage:
    python 07_cluster_components.py [--sda-results-base PREFIX]
        [--sda-input-dir DIR] [--output-dir DIR] [--id-map CSV]
"""
from __future__ import annotations
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
from scipy import stats

# ----------------------------------------------------------------------------
# Paths — adapt to your local layout (or pass them on the command line). SDA
# outputs come from the SLURM array submitted by `05_export_sda_inputs.py`.
# Each run writes into `SDA_RESULTS_BASE{i}/it{ITERATION}/{A,B1,X1..X4,S1..S4}`
# for `i = 1..NUM_RUNS`.
# ----------------------------------------------------------------------------
SDA_RESULTS_BASE = os.path.join(os.environ.get("G_ROS_SDA_DIR", "sda"), "results")  # run i is at f"{SDA_RESULTS_BASE}{i}/"
SDA_INPUT_DIR = "sda/"            # output dir of 05_export_sda_inputs.py; carries the metadata sidecars
OUTPUT_DIR = "results/"           # downstream scripts expect clustered outputs here
COMPONENT_ID_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "component_id_map.csv")

# ----------------------------------------------------------------------------
# Run / clustering parameters — match the manuscript Methods.
# ----------------------------------------------------------------------------
NUM_RUNS = 10                       # SDA array size
ITERATION = 1500                    # checkpoint read: the last one common to all ten runs (launched with --max_iter 3000; stopped by the time limit after 1,803-2,006 iterations)
COPHENETIC_THRESHOLD = 0.4          # flat-cluster cut on (1 − |Pearson|) dissimilarity
MIN_RUNS_PER_CLUSTER = 5            # keep cluster only if it draws from ≥ 5 of the 10 runs
MAKE_DIAGNOSTIC_PLOTS = False       # toggle to write dendrogram + cluster-size histogram

OMIC_NAMES = {1: "RNA", 2: "meth", 3: "chip", 4: "proteins"}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def correlation_distance(x, y) -> float:
    """1 − |Pearson correlation| as a hierarchical-clustering metric (the
    original script's metric, evaluated one pair at a time)."""
    return 1.0 - abs(stats.pearsonr(x, y)[0])


def condensed_correlation_distances(matrix, block_rows: int = 2048) -> np.ndarray:
    """All pairwise 1 − |Pearson r| between the rows of `matrix`, in scipy's
    condensed (`pdist`) order.

    Equal to `scipy.spatial.distance.pdist(matrix, correlation_distance)` up
    to floating-point rounding, computed with one matrix product per block of
    rows instead of one `pearsonr` call per pair (~2 x 10^8 pairs here).
    """
    z = np.asarray(matrix, dtype=np.float64)
    z = z - z.mean(axis=1, keepdims=True)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    n = z.shape[0]
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    pos = 0
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        r = z[start:stop] @ z.T
        for k, i in enumerate(range(start, stop)):
            seg = r[k, i + 1:]
            out[pos:pos + seg.size] = 1.0 - np.abs(seg)
            pos += seg.size
    np.clip(out, 0.0, 1.0, out=out)   # |r| can exceed 1 by rounding; pearsonr clips r the same way
    return out


def read_sda_matrix(path: str, rows=None) -> pd.DataFrame:
    """Read a space-separated SDA output matrix (no header). If `rows` is
    given, keep only those 0-based file rows; the result is indexed by them."""
    if rows is None:
        return pd.read_csv(path, sep=" ", header=None)
    wanted = {int(r) for r in rows}
    df = pd.read_csv(path, sep=" ", header=None, skiprows=lambda r: r not in wanted)
    df.index = sorted(wanted)
    return df


def load_run_samples_and_cell_types(run_dir: str, sample_labels, cell_type_labels):
    """Read one run's A (sample loadings) and B1 (cell-type loadings) and drop
    components whose sample loadings are all zero.

    Returns (samples, treatment, keep): component x sample and component x
    cell-type DataFrames indexed 0..n_kept-1, and `keep`, the SDA component
    index (row of X/S, column of A and B1) of each kept component.
    """
    # A: SDA writes samples x components; transpose to components x samples
    samples = read_sda_matrix(os.path.join(run_dir, "A")).T
    samples.columns = sample_labels
    keep = samples[(samples.T != 0).any()].index.values  # non-zero components
    samples = samples.loc[keep].reset_index(drop=True)

    # B1: cell-type loadings (only the RNA tensor has a B matrix: B is the
    # third dimension of the snRNA-seq tensor)
    treatment = read_sda_matrix(os.path.join(run_dir, "B1")).T
    treatment.columns = cell_type_labels
    treatment = treatment.loc[keep].reset_index(drop=True)
    return samples, treatment, np.asarray(keep)


def stable_clusters(samples: pd.DataFrame, run_of: np.ndarray):
    """Steps 3-4. Cluster the pooled components and keep clusters drawn from
    ≥ MIN_RUNS_PER_CLUSTER runs.

    `samples` holds the pooled sample loadings (row position = global
    position); `run_of[p]` is the run of the component at position p.
    Returns (Z, cluster_assignments, retained, rejected): `retained` and
    `rejected` are lists of dicts {cluster, members (positions, pooled
    order), n_runs}, in order of first appearance; `rejected` lists the
    clusters with ≥ MIN_RUNS_PER_CLUSTER members that span fewer runs.
    """
    Z = hac.linkage(condensed_correlation_distances(samples.values), method="complete")
    cluster_assignments = hac.fcluster(Z, t=COPHENETIC_THRESHOLD, criterion="distance")
    cluster_sizes = Counter(cluster_assignments)     # insertion order = order of first appearance
    retained, rejected = [], []
    for cluster_id, size in cluster_sizes.items():
        # Cheap pre-filter: a cluster cannot draw from more runs than its size
        if size < MIN_RUNS_PER_CLUSTER:
            continue
        members = np.nonzero(cluster_assignments == cluster_id)[0]
        n_runs = len(set(run_of[members].tolist()))       # each member's own run
        entry = {"cluster": int(cluster_id), "members": members, "n_runs": n_runs}
        (retained if n_runs >= MIN_RUNS_PER_CLUSTER else rejected).append(entry)
    return Z, cluster_assignments, retained, rejected


def alignment_flips(block: np.ndarray) -> np.ndarray:
    """+1/−1 for each row of `block`: −1 where the row's Pearson r with the
    first row is negative. A NaN r (constant vector) counts as +1, as
    `nan < 0` is False in the original."""
    flips = np.ones(len(block))
    for k in range(1, len(block)):
        if stats.pearsonr(block[0], block[k])[0] < 0:
            flips[k] = -1.0
    return flips


def aligned_mean(block: np.ndarray, flips: np.ndarray) -> np.ndarray:
    """Unweighted mean of the rows of `block` after multiplying each by its flip."""
    return (block * flips[:, None]).mean(axis=0)


def load_component_id_map(path: str) -> pd.DataFrame:
    """Read `component_id_map.csv` (see the module docstring)."""
    id_map = pd.read_csv(path)
    needed = {"component_id", "status", "first_member_run", "first_member_sda_index",
              "n_members", "n_runs"}
    missing = needed - set(id_map.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    if not id_map["component_id"].is_unique:
        raise ValueError(f"{path}: component IDs are not unique")
    return id_map


def assign_component_ids(retained, rejected, run_of, sda_index_of, id_map: pd.DataFrame) -> None:
    """Step 6. Add 'component_id' to each retained cluster from the explicit
    map, checking members and runs; stop if the map is not reproduced."""
    def key(entry):
        first = entry["members"][0]
        return int(run_of[first]), int(sda_index_of[first])

    active = id_map[id_map["status"] == "active"]
    by_key = {(int(r.first_member_run), int(r.first_member_sda_index)): r for r in active.itertuples()}
    seen = set()
    for entry in retained:
        row = by_key.get(key(entry))
        if row is None:
            raise RuntimeError(
                f"Retained cluster {entry['cluster']} (first member: run {key(entry)[0]}, "
                f"SDA index {key(entry)[1]}; {len(entry['members'])} members, {entry['n_runs']} runs) "
                f"is not in the component-ID map: the clustering does not reproduce the catalogue.")
        if (len(entry["members"]), entry["n_runs"]) != (int(row.n_members), int(row.n_runs)):
            raise RuntimeError(
                f"Component {row.component_id}: expected {row.n_members} members from {row.n_runs} runs, "
                f"found {len(entry['members'])} from {entry['n_runs']}.")
        entry["component_id"] = int(row.component_id)
        seen.add(int(row.component_id))
    missing = sorted(set(active["component_id"].astype(int)) - seen)
    if missing:
        raise RuntimeError(f"Component IDs in the map but not retained: {missing}")

    # Report the retired IDs (their clusters must fail the run criterion)
    rejected_by_key = {key(e): e for e in rejected}
    for r in id_map[id_map["status"] == "retired"].itertuples():
        e = rejected_by_key.get((int(r.first_member_run), int(r.first_member_sda_index)))
        if e is None:
            print(f"  retired C{r.component_id}: its cluster is not among the rejected clusters")
        else:
            print(f"  retired C{r.component_id}: cluster {e['cluster']}, {len(e['members'])} members "
                  f"from {e['n_runs']} runs (< {MIN_RUNS_PER_CLUSTER})")


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster the SDA runs into the stable component catalogue.")
    parser.add_argument("--sda-results-base", default=SDA_RESULTS_BASE,
                        help="prefix of the run directories; run i is at <prefix><i>/")
    parser.add_argument("--sda-input-dir", default=SDA_INPUT_DIR,
                        help="directory with md_samples.txt, cell_types.txt, genes.txt, md_cpg.txt, md_chip.txt, md_prot.txt")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--id-map", default=COMPONENT_ID_MAP)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    def run_dir(i: int) -> str:
        return f"{args.sda_results_base}{i}/it{ITERATION}"

    # ------------------------------------------------------------------------
    # 1. Load metadata sidecars produced by 05_export_sda_inputs
    # ------------------------------------------------------------------------
    def lines(name):
        return open(os.path.join(args.sda_input_dir, name)).read().splitlines()

    samps_in_common = lines("md_samples.txt")
    cell_types = lines("cell_types.txt")
    feature_labels = {1: lines("genes.txt"), 2: lines("md_cpg.txt"),
                      3: lines("md_chip.txt"), 4: lines("md_prot.txt")}
    id_map = load_component_id_map(args.id_map)

    # ------------------------------------------------------------------------
    # 2. Pool the sample and cell-type loadings of all runs. Global position
    #    p runs over run 1's kept components, then run 2's, and so on;
    #    run_of[p] and sda_index_of[p] record where each came from.
    # ------------------------------------------------------------------------
    samples_parts, treatment_parts, run_of, sda_index_of = [], [], [], []
    for i in range(1, NUM_RUNS + 1):
        samples_i, treatment_i, keep = load_run_samples_and_cell_types(run_dir(i), samps_in_common, cell_types)
        samples_parts.append(samples_i)
        treatment_parts.append(treatment_i)
        run_of += [i] * len(keep)
        sda_index_of += keep.tolist()
        print(f"  run {i:>2}: {len(keep)} non-zero components retained")
    samples = pd.concat(samples_parts, axis=0, ignore_index=True)
    treatment = pd.concat(treatment_parts, axis=0, ignore_index=True)
    run_of = np.asarray(run_of)
    sda_index_of = np.asarray(sda_index_of)
    print(f"  pooled: {len(samples)} components across {NUM_RUNS} runs")

    # ------------------------------------------------------------------------
    # 3-4. Hierarchical clustering on (1 − |Pearson|) of sample loadings, then
    #      keep clusters drawing from ≥ MIN_RUNS_PER_CLUSTER distinct runs
    # ------------------------------------------------------------------------
    Z, cluster_assignments, retained, rejected = stable_clusters(samples, run_of)
    print(f"  {len(set(cluster_assignments))} flat clusters at cophenetic ≤ {COPHENETIC_THRESHOLD}")
    print(f"  retained {len(retained)} stable clusters (≥ {MIN_RUNS_PER_CLUSTER}/{NUM_RUNS} runs each); "
          f"{len(rejected)} clusters with ≥ {MIN_RUNS_PER_CLUSTER} members span fewer runs")

    # ------------------------------------------------------------------------
    # 6 (before averaging). Component IDs from the explicit map
    # ------------------------------------------------------------------------
    assign_component_ids(retained, rejected, run_of, sda_index_of, id_map)
    retained.sort(key=lambda e: e["component_id"])
    ids = [e["component_id"] for e in retained]
    unused = sorted(set(range(1, ids[-1] + 1)) - set(ids))
    print(f"  component IDs {ids[0]}..{ids[-1]} ({len(ids)} components; not used: {unused})")

    # ------------------------------------------------------------------------
    # 5. Sign-align and average. Samples first (fA); then one omic at a time,
    #    reading only the member rows of X_j and S_j (fX_j, PIP medians);
    #    finally the cell types, with fB = fA x fX1.
    # ------------------------------------------------------------------------
    A = samples.values
    B = treatment.values
    flips_A = {}
    flips_X = {j: {} for j in OMIC_NAMES}
    samples_rows, treatment_rows = [], []
    genes_rows = {j: [] for j in OMIC_NAMES}
    pip_rows = {j: [] for j in OMIC_NAMES}
    for e in retained:
        fA = alignment_flips(A[e["members"]])
        flips_A[e["component_id"]] = fA
        samples_rows.append(aligned_mean(A[e["members"]], fA))

    all_members = np.sort(np.concatenate([e["members"] for e in retained]))
    row_of = {int(p): k for k, p in enumerate(all_members)}   # global position -> row of the member block
    by_run = {i: all_members[run_of[all_members] == i] for i in range(1, NUM_RUNS + 1)}

    def member_block(prefix: str, j: int) -> np.ndarray:
        """Rows of X_j or S_j for every member of a retained cluster."""
        block = np.empty((len(all_members), len(feature_labels[j])), dtype=np.float64)
        for i, positions in by_run.items():
            if len(positions) == 0:
                continue
            df = read_sda_matrix(os.path.join(run_dir(i), f"{prefix}{j}"), rows=sda_index_of[positions])
            if df.shape[1] != len(feature_labels[j]):
                raise ValueError(f"run {i} {prefix}{j}: {df.shape[1]} columns, expected {len(feature_labels[j])}")
            block[[row_of[int(p)] for p in positions]] = df.to_numpy(dtype=np.float64)
        return block

    for j, name in OMIC_NAMES.items():
        X = member_block("X", j)
        for e in retained:
            rows = [row_of[int(p)] for p in e["members"]]
            fX = alignment_flips(X[rows])
            flips_X[j][e["component_id"]] = fX
            genes_rows[j].append(aligned_mean(X[rows], fX))
        del X
        S = member_block("S", j)
        for e in retained:
            rows = [row_of[int(p)] for p in e["members"]]
            pip_rows[j].append(np.median(S[rows], axis=0))
        del S
        print(f"  {name}: aligned and averaged ({len(feature_labels[j])} features)")

    member_records = []
    for e in retained:
        cid = e["component_id"]
        fA = flips_A[cid]
        fB = fA * flips_X[1][cid]                         # cell-type flip = sample flip x gene flip
        treatment_rows.append(aligned_mean(B[e["members"]], fB))
        fB_v1 = alignment_flips(B[e["members"]])          # rule used for the version-1 deposit, recorded for reference only
        for k, p in enumerate(e["members"]):
            rec = {"component_id": cid, "run": int(run_of[p]), "sda_index": int(sda_index_of[p]),
                   "flip_samples": int(fA[k])}
            for j, name in OMIC_NAMES.items():
                rec[f"flip_{name}"] = int(flips_X[j][cid][k])
            rec["flip_cell_types"] = int(fB[k])
            rec["flip_cell_types_v1_deposit"] = int(fB_v1[k])
            member_records.append(rec)
    members_df = pd.DataFrame(member_records)
    n_diff = int((members_df["flip_cell_types"] != members_df["flip_cell_types_v1_deposit"]).sum())
    print(f"  cell-type flips: {n_diff} of {len(members_df)} members differ from the version-1 rule")

    # ------------------------------------------------------------------------
    # 7. Write outputs (rows indexed by component ID)
    # ------------------------------------------------------------------------
    out = args.output_dir
    pd.DataFrame(np.vstack(samples_rows), index=ids, columns=samples.columns).to_csv(
        os.path.join(out, "samples_final.csv"))
    pd.DataFrame(np.vstack(treatment_rows), index=ids, columns=treatment.columns).to_csv(
        os.path.join(out, "treatment_final.csv"))
    for j, name in OMIC_NAMES.items():
        pd.DataFrame(np.vstack(genes_rows[j]), index=ids, columns=feature_labels[j]).to_csv(
            os.path.join(out, f"genes_final_{name}.csv"))
        pd.DataFrame(np.vstack(pip_rows[j]), index=ids, columns=feature_labels[j]).to_csv(
            os.path.join(out, f"pip_final_{name}.csv"))
    catalogue = pd.DataFrame([{
        "component_id": e["component_id"], "cluster": e["cluster"], "n_members": len(e["members"]),
        "n_runs": e["n_runs"], "runs": " ".join(str(r) for r in sorted(set(run_of[e["members"]].tolist()))),
        "first_member_run": int(run_of[e["members"][0]]),
        "first_member_sda_index": int(sda_index_of[e["members"][0]]),
    } for e in retained])
    catalogue.to_csv(os.path.join(out, "component_catalogue.csv"), index=False)
    members_df.to_csv(os.path.join(out, "cluster_members.csv"), index=False)
    print(f"  wrote samples_final / treatment_final / genes_final_* / pip_final_* / "
          f"component_catalogue / cluster_members to {out}")

    # ------------------------------------------------------------------------
    # 8. Optional diagnostic plots
    # ------------------------------------------------------------------------
    if MAKE_DIAGNOSTIC_PLOTS:
        import matplotlib.pyplot as plt
        os.makedirs("figs/td/all/", exist_ok=True)

        plt.figure(figsize=(25, 10))
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("sample index")
        plt.ylabel("distance")
        plt.yticks(np.arange(0, 2.01, 0.2))
        hac.dendrogram(Z, leaf_rotation=90.0, leaf_font_size=8.0)
        plt.axhline(y=COPHENETIC_THRESHOLD, c="k")
        plt.savefig("figs/td/all/sample_components_cluster.pdf")
        plt.close()

        sizes = pd.Series(list(Counter(cluster_assignments).values()))
        vc = sizes.value_counts().sort_index()
        vc.plot(kind="bar", colormap="ocean")
        plt.xlabel("Cluster size")
        plt.ylabel("Frequency")
        plt.savefig("figs/td/all/no_of_clusters.pdf")
        plt.close()
        print("  wrote diagnostic plots in figs/td/all/")


if __name__ == "__main__":
    main()
