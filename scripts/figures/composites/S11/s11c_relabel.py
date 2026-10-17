"""Final S11c: re-place the 15 feature labels so that no label sits nearer to another point than
to its own one (in the builder's panel, ARL6IP1 and PSPC1-AS2 each had an unlabelled point nearer to the label than
the labelled point, and no leader line). Everything else is the builder's code, imported read-only from
s11_panels_at_cell_size.py: same points, colours, labels, label colours, axes, limits.

The only change: in the label search, a leader-less label must have its own point clearly nearest (any other point
within own distance + 1 pt costs 25, labelled or not; the builder allowed others up to 0.3 pt nearer and charged 4
for unlabelled ones), so the search picks another adjacent spot or a leader line, as adjustText did in the original.
Writes only under <G_ROS_FIGURE_OUT>/S11/ (panels_fix/ and the log). Aggregates only in the log."""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import s11_panels_at_cell_size as M

OUT = f"{M.FIGOUT}/S11/panels_fix"
os.makedirs(OUT, exist_ok=True)
M.PAN = OUT
M.HERE = f"{M.FIGOUT}/S11"   # check_labels_in_original dumps text under HERE/orig_png
UNLAB_W = 25.0
AMB_MARGIN = 1.0   # another point counts as competing unless it is >= 1 pt farther from the label than its own point (builder: own - 0.3)


def _greedy(order, labels, cand, P, rad, axbb):
    """M._greedy, except: a point other than the label's own counts as competing when it is less than AMB_MARGIN pt
    farther from the leader-less label than the own point (builder: when it is > 0.3 pt nearer), and an unlabelled
    competing point costs UNLAB_W (builder: 4). Otherwise identical."""
    placed = []; leaders = []; choice = {}; total = 0.0
    for i in order:
        p = cand[i]["p"]; best = None
        for (dd, box, anchor, ha, va) in cand[i]["c"]:
            if best is not None and best[0] <= dd:
                break
            if any(M.overlap(box, q, pad=1.0) for q in placed):
                continue
            cover = int(np.sum((P[:, 0] > box[0] - rad) & (P[:, 0] < box[2] + rad) &
                               (P[:, 1] > box[1] - rad) & (P[:, 1] < box[3] + rad)))
            cross = 0
            if dd > 2.0:
                qx = min(max(p[0], box[0]), box[2]); qy = min(max(p[1], box[1]), box[3])
                cross += sum(M._seg_hits_box(p, (qx, qy), q) for q in placed)
                cross += sum(M._segs_cross(p, (qx, qy), l1, l2) for l1, l2 in leaders)
                cross += sum(M._seg_hits_box(l1, l2, box) for l1, l2 in leaders)
                cover += int(np.sum([M._seg_hits_box(p, (qx, qy), (P[m, 0] - rad, P[m, 1] - rad, P[m, 0] + rad,
                                                                   P[m, 1] + rad), pad=0.0)
                                     for m in range(len(P)) if abs(P[m, 0] - p[0]) + abs(P[m, 1] - p[1]) > 0.5]))
            amb = 0.0
            if dd <= 2.0:
                dx_ = np.maximum(0, np.maximum(box[0] - P[:, 0], P[:, 0] - box[2]))
                dy_ = np.maximum(0, np.maximum(box[1] - P[:, 1], P[:, 1] - box[3]))
                dist = np.hypot(dx_, dy_)
                own = float(np.hypot(max(0, box[0] - p[0], p[0] - box[2]), max(0, box[1] - p[1], p[1] - box[3])))
                nearer = (dist < own + AMB_MARGIN) & ((np.abs(P[:, 0] - p[0]) + np.abs(P[:, 1] - p[1])) > 0.5)
                amb = 25.0 * float(np.sum(nearer & M.LABPT)) + UNLAB_W * float(np.sum(nearer & ~M.LABPT))
            cost = dd + 10.0 * cover + 30.0 * cross + amb
            if best is None or cost < best[0]:
                best = (cost, dd, box, anchor, ha, va, cover, cross)
        if best is None:
            return np.inf, None
        total += best[0]; choice[i] = best; placed.append(best[2])
        if best[1] > 2.0:
            b = best[2]
            leaders.append((tuple(p), (min(max(p[0], b[0]), b[2]), min(max(p[1], b[1]), b[3]))))
    return total, choice


M._greedy = _greedy
M.LOG.clear()
d = M.panel_manhattan(1, "c")
M.check_manhattan_points(1, d)
M.check_labels_in_original(1, d)
open(f"{OUT}/../s11c_relabel.log", "w").write("\n".join(M.LOG) + "\n")
print("\n".join(M.LOG))
