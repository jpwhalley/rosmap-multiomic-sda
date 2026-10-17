"""Diagnostics for permutation / resampling nulls.

Loaded by skill({skill: "null-calibration"}). See SKILL.md for the three checks
and the reporting rules.
"""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)

import math
import statistics

SIZE_TOL_DEFAULT = 2.0
DEGENERATE_HIT_FRACTION = 0.01


def empirical_p(obs_stat, null_stats, larger_is_extreme=True):
    """Add-one empirical p-value. Never returns 0; floor is 1/(B+1)."""
    b = len(null_stats)
    if b == 0:
        raise ValueError("null_stats is empty")
    if larger_is_extreme:
        n_extreme = sum(1 for x in null_stats if x >= obs_stat)
    else:
        n_extreme = sum(1 for x in null_stats if x <= obs_stat)
    return (1.0 + n_extreme) / (b + 1.0)


def null_calibration_report(obs_stat, null_stats, obs_size=None, null_sizes=None,
                            larger_is_extreme=True, size_tol=None,
                            hit_threshold=None, label=""):
    """Check whether a permutation null can support a p-value.

    obs_stat / null_stats : the observed statistic and its value in each draw.
    obs_size / null_sizes : optional foreground size, observed and per draw.
                            Supplying these enables the size-match check.
    hit_threshold         : value above which a draw counts as producing a hit.
                            Defaults to the smallest positive outcome (>0), which
                            suits count statistics such as 'number of significant
                            terms'. Pass explicitly for continuous statistics.

    Returns a dict; read ["verdict"] and ["problems"]. See SKILL.md.
    """
    if size_tol is None:
        size_tol = SIZE_TOL_DEFAULT
    b = len(null_stats)
    if b == 0:
        raise ValueError("null_stats is empty")

    p = empirical_p(obs_stat, null_stats, larger_is_extreme=larger_is_extreme)
    p_floor = 1.0 / (b + 1.0)

    thr = hit_threshold
    if thr is None:
        thr = 0
    n_hit = sum(1 for x in null_stats if x > thr)
    frac_hit = n_hit / float(b)

    rep = {
        "label": label,
        "n_draws": b,
        "obs_stat": obs_stat,
        "null_median": statistics.median(null_stats),
        "null_max": max(null_stats),
        "p_empirical": p,
        "p_floor": p_floor,
        "p_at_floor": abs(p - p_floor) < 1e-12,
        "hit_threshold": thr,
        "frac_draws_with_hit": frac_hit,
        "size_ratio": None,
        "size_calibrated": None,
        "problems": [],
        "verdict": "calibrated",
    }

    if obs_size is not None and null_sizes:
        med_size = statistics.median(null_sizes)
        ratio = med_size / float(obs_size) if obs_size else float("inf")
        rep["obs_size"] = obs_size
        rep["null_median_size"] = med_size
        rep["size_ratio"] = ratio
        rep["size_calibrated"] = bool((1.0 / size_tol) <= ratio <= size_tol)
        if not rep["size_calibrated"]:
            rep["problems"].append(
                "SIZE MISMATCH: median null foreground is %.2fx the observed one "
                "(tolerance %.1fx). The null is not comparing like with like."
                % (ratio, size_tol))

    if frac_hit <= DEGENERATE_HIT_FRACTION:
        rep["problems"].append(
            "NULL CANNOT FAIL: only %.2f%% of %d draws produce any hit above %s, "
            "so p is pinned near its floor %.4g for ANY positive observation. "
            "The p-value is a property of the null, not of the data."
            % (100.0 * frac_hit, b, thr, p_floor))

    if frac_hit >= 1.0 - DEGENERATE_HIT_FRACTION and larger_is_extreme:
        rep["problems"].append(
            "NULL ALWAYS HITS: %.2f%% of draws exceed the threshold, so no "
            "observation could look extreme. A null result here is uninformative."
            % (100.0 * frac_hit))

    if rep["problems"]:
        rep["verdict"] = "degenerate"
    elif obs_size is not None and not null_sizes:
        rep["verdict"] = "check_manually"
        rep["problems"].append(
            "obs_size supplied without null_sizes: the size-match check did not run.")
    return rep


def format_calibration_report(rep):
    """Human-readable summary of null_calibration_report()."""
    lines = []
    head = "null calibration"
    if rep.get("label"):
        head = head + " [" + str(rep["label"]) + "]"
    lines.append(head + ": " + rep["verdict"].upper())
    lines.append("  draws            %d   (p floor %.4g)" % (rep["n_draws"], rep["p_floor"]))
    lines.append("  observed stat    %s   null median %s   null max %s"
                 % (rep["obs_stat"], rep["null_median"], rep["null_max"]))
    lines.append("  p empirical      %.4g%s"
                 % (rep["p_empirical"], "   <-- AT THE FLOOR" if rep["p_at_floor"] else ""))
    lines.append("  draws with hit   %.2f%%   (threshold %s)"
                 % (100.0 * rep["frac_draws_with_hit"], rep["hit_threshold"]))
    if rep.get("size_ratio") is not None:
        lines.append("  size ratio       %.2fx   calibrated: %s"
                     % (rep["size_ratio"], rep["size_calibrated"]))
    for prob in rep["problems"]:
        lines.append("  ! " + prob)
    if rep["verdict"] == "degenerate":
        lines.append("  -> Do not report this p-value. Fix the null first.")
    return "\n".join(lines)


def blocked_permutation_indices(block_sizes, pool_size, rng):
    """Place blocks of the given sizes at random non-overlapping offsets.

    Returns a sorted list of indices into range(pool_size), preserving both the
    NUMBER of blocks and the multiset of block sizes -- the structure-preserving
    null of SKILL.md check 3. rng is a random.Random or numpy Generator with
    .randrange or .integers.
    """
    total = sum(block_sizes)
    if total > pool_size:
        raise ValueError("blocks (%d) exceed pool (%d)" % (total, pool_size))
    taken = set()
    for size in sorted(block_sizes, reverse=True):
        for _attempt in range(1000):
            if hasattr(rng, "randrange"):
                start = rng.randrange(0, pool_size - size + 1)
            else:
                start = int(rng.integers(0, pool_size - size + 1))
            span = range(start, start + size)
            if taken.isdisjoint(span):
                taken.update(span)
                break
        else:
            raise RuntimeError("could not place a block of size %d without overlap" % size)
    return sorted(taken)
