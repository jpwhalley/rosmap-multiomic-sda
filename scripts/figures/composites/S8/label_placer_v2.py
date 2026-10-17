"""Deterministic text-label placement with leader lines (stands in for adjustText, which is not installed).

Each label gets K candidate boxes inside the axes (a grid search, pruned to the cheapest by its own cost); a seeded
simulated annealing then picks one candidate per label to minimise the total cost:
  own cost   leader length (+ a small quadratic term), the label covering its own point, plotted points under the
             label, and another labelled point under or right next to the label (which would read as its owner);
  pair cost  two labels overlapping (prohibitive), a leader running through another label, two leaders crossing.
Works in display units of a 72-dpi figure (= PostScript points). Only presentation: the label set, text and colour
come from the caller unchanged.

v2 (copy of label_placer.py, which is unchanged). Presentation-only changes:
  1. own cost: a leader that leaves the label's left or right side horizontally and is shorter than 8 pt reads as a
     hyphen or dash after the name ("DOCK9-AS1-", "cg13516541:AKT1-"); such candidates cost +25.
  2. own cost: a leader that runs over another labelled point reads as that point's leader (the KCNS2 leader crossed
     the AC068189.1 point); such candidates cost +200.
  4. own cost: a label with no leader that is as close to another plotted point as to its own costs +60.
  3. place() freezes the autoscaled axis limits before placing (leader lines drawn later cannot re-autoscale x), and
     nudge_ticks() separates rotated chromosome tick labels that collide (minimal least-squares shift, PAVA)."""
import numpy as np
LAST_TICK_MOVES = []
V2_HYPHEN = True; V2_LEADER_OVER = True; V2_AMBIG = True      # switches for the v2 cost terms 1, 2, 4
HYPHEN_COST = 25.0


def _pava_nondecreasing(v):
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    blocks = []                                    # [mean, weight, count]
    for x in v:
        blocks.append([float(x), 1.0, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop(); m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out = []
    for m_, w_, c_ in blocks:
        out += [m_] * c_
    return np.array(out)


def nudge_ticks(fig, ax, min_gap_pt):
    """Shift x tick positions (tick marks are length 0 here, so only the labels move) by the least-squares minimum
    so that neighbouring label centres are at least min_gap_pt apart. Returns (labels, shift in pt) for moved ones."""
    fig.canvas.draw()
    ticks = np.asarray(ax.get_xticks(), float)
    labs = [t.get_text() for t in ax.get_xticklabels()]
    T = ax.transData
    px = np.array([T.transform((t, 0))[0] for t in ticks])
    order = np.argsort(px); p = px[order]
    i = np.arange(len(p))
    y = _pava_nondecreasing(p - i * min_gap_pt)
    x = y + i * min_gap_pt
    new_px = np.empty_like(px); new_px[order] = x
    inv = T.inverted()
    new = np.array([inv.transform((v, 0))[0] for v in new_px])
    ax.set_xticks(new); ax.set_xticklabels(labs, rotation=90)
    moved = [(labs[k], float(new_px[k] - px[k])) for k in range(len(labs)) if abs(new_px[k] - px[k]) > 0.01]
    return moved


def _pt_seg_dist(P, A, B):
    """distance from points P (n, 2) to segments A->B (k, 2): (k, n)."""
    AB = B - A; L2 = (AB ** 2).sum(1)[:, None]
    AP = P[None, :, :] - A[:, None, :]
    t = np.clip((AP * AB[:, None, :]).sum(2) / np.where(L2 > 0, L2, 1.0), 0, 1)
    C = A[:, None, :] + t[..., None] * AB[:, None, :]
    return np.hypot(*(P[None, :, :] - C).transpose(2, 0, 1))


def _seg_rect(p, q, R):
    """Does segment p->q (arrays (..., 2)) intersect rectangles R (..., 4: x0, y0, x1, y1)? Liang-Barsky."""
    px, py = p[..., 0], p[..., 1]; dx = q[..., 0] - px; dy = q[..., 1] - py
    t0 = np.zeros(np.broadcast(px, R[..., 0]).shape); t1 = np.ones_like(t0); ok = np.ones_like(t0, bool)
    for pk, qk in ((-dx, px - R[..., 0]), (dx, R[..., 2] - px), (-dy, py - R[..., 1]), (dy, R[..., 3] - py)):
        pk = np.broadcast_to(pk, t0.shape); qk = np.broadcast_to(qk, t0.shape)
        par = np.abs(pk) < 1e-12
        ok &= ~(par & (qk < 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(par, 0.0, qk / np.where(par, 1.0, pk))
        neg = (pk < 0) & ~par; pos = (pk > 0) & ~par
        t0 = np.where(neg, np.maximum(t0, r), t0); t1 = np.where(pos, np.minimum(t1, r), t1)
    return ok & (t0 <= t1)


def _seg_cross(a1, a2, b1, b2):
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    d1 = orient(b1, b2, a1); d2 = orient(b1, b2, a2); d3 = orient(a1, a2, b1); d4 = orient(a1, a2, b2)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def place(fig, ax, pts_xy, labels, fontsize, marker_r_pt=1.0, step=1.5, K=400, w_pts=6.0, near_pt=3.0,
          lead_min=2.0, lead_lw=0.3, pad=0.8, iters=40000, seed=1, top_overflow=0.0, restarts=8, tick_gap_pt=6.1):
    fig.canvas.draw()
    ax.set_xlim(*ax.get_xlim()); ax.set_ylim(*ax.get_ylim())      # v2 (3): freeze the autoscaled limits
    if tick_gap_pt:
        global LAST_TICK_MOVES
        LAST_TICK_MOVES = nudge_ticks(fig, ax, tick_gap_pt)
    fig.canvas.draw()
    Rnd = fig.canvas.get_renderer()
    T = ax.transData
    bb = ax.get_window_extent(Rnd)
    X0, Y0, X1, Y1 = bb.x0 + 0.6, bb.y0 + 0.6, bb.x1 - 0.6, bb.y1 - 0.6 + top_overflow   # labels may rise into the top margin
    P = T.transform(np.asarray(pts_xy, float))
    n = len(labels)
    meas = []
    for (x, y, s, c) in labels:
        t = ax.text(0, 0, s, fontsize=fontsize, ha="left", va="baseline", transform=None)
        e = t.get_window_extent(Rnd); t.remove()
        meas.append((e.width, e.height, e.y0))
    LP = np.array([T.transform((l[0], l[1])) for l in labels])
    m = marker_r_pt
    cand_box, cand_q, cand_d, base = [], [], [], []
    for i in range(n):
        w, h, _ = meas[i]; px, py = LP[i]
        xs = np.arange(X0, X1 - w + 1e-6, step); ys = np.arange(Y0, Y1 - h + 1e-6, step)
        GX, GY = np.meshgrid(xs, ys); b = np.stack([GX.ravel(), GY.ravel(), GX.ravel() + w, GY.ravel() + h], 1)
        nx = np.clip(px, b[:, 0], b[:, 2]); ny = np.clip(py, b[:, 1], b[:, 3])
        d = np.hypot(nx - px, ny - py)
        c = d + 0.015 * d ** 2 + 40.0 * (d < m + 0.4)
        others = np.ones(len(P), bool)
        others[np.argmin(np.hypot(P[:, 0] - px, P[:, 1] - py))] = False
        Q = P[others]
        cov = ((Q[None, :, 0] > b[:, None, 0] - m) & (Q[None, :, 0] < b[:, None, 2] + m) &
               (Q[None, :, 1] > b[:, None, 1] - m) & (Q[None, :, 1] < b[:, None, 3] + m)).sum(1)
        c = c + w_pts * cov
        for j in range(n):
            if j == i:
                continue
            qx, qy = LP[j]; mq = m + near_pt
            c = c + 200.0 * ((qx > b[:, 0] - mq) & (qx < b[:, 2] + mq) & (qy > b[:, 1] - mq) & (qy < b[:, 3] + mq))
        # v2 (1): short horizontal side leader reads as a hyphen / dash
        side = ((px < b[:, 0]) | (px > b[:, 2])) & (py >= b[:, 1]) & (py <= b[:, 3])
        c = c + HYPHEN_COST * V2_HYPHEN * (side & (d >= lead_min) & (d < 8.0))
        # v2 (2): leader running over another labelled point
        if n > 1:
            others_lab = np.array([LP[j] for j in range(n) if j != i])
            A = np.broadcast_to(LP[i], (len(b), 2)); Bq = np.stack([nx, ny], 1)
            dd_ = _pt_seg_dist(others_lab, A, Bq).min(1)
            c = c + 200.0 * V2_LEADER_OVER * ((d >= lead_min) & (dd_ < m + 0.8))
        # v2 (4): a label with no leader whose box is as close to another plotted point as to its own reads as
        # that point's label
        if len(Q):
            gx = np.maximum(np.maximum(b[:, None, 0] - Q[None, :, 0], 0), Q[None, :, 0] - b[:, None, 2])
            gy = np.maximum(np.maximum(b[:, None, 1] - Q[None, :, 1], 0), Q[None, :, 1] - b[:, None, 3])
            dq = np.hypot(gx, gy).min(1)
            c = c + 60.0 * V2_AMBIG * ((d < lead_min) & (dq <= d + 1.0))
        keep = np.argsort(c, kind="stable")[:K]
        cand_box.append(b[keep]); cand_q.append(np.stack([nx[keep], ny[keep]], 1)); cand_d.append(d[keep])
        base.append(c[keep])
    # pairwise matrices
    M = {}
    for i in range(n):
        for j in range(i + 1, n):
            Bi = cand_box[i][:, None, :]; Bj = cand_box[j][None, :, :]
            ov = ((np.minimum(Bi[..., 2], Bj[..., 2]) + pad > np.maximum(Bi[..., 0], Bj[..., 0]) - pad) &
                  (np.minimum(Bi[..., 3], Bj[..., 3]) + pad > np.maximum(Bi[..., 1], Bj[..., 1]) - pad))
            cost = 1e4 * ov
            li = cand_d[i][:, None] >= lead_min; lj = cand_d[j][None, :] >= lead_min
            pi = np.broadcast_to(LP[i], (len(cand_box[i]), 1, 2)); qi = cand_q[i][:, None, :]
            pj = np.broadcast_to(LP[j], (1, len(cand_box[j]), 2)); qj = cand_q[j][None, :, :]
            cost = cost + 600.0 * (li & _seg_rect(pi, qi, Bj))
            cost = cost + 600.0 * (lj & _seg_rect(pj, qj, Bi))
            cost = cost + 400.0 * (li & lj & _seg_cross(np.broadcast_to(pi, cost.shape + (2,)),
                                                       np.broadcast_to(qi, cost.shape + (2,)),
                                                       np.broadcast_to(pj, cost.shape + (2,)),
                                                       np.broadcast_to(qj, cost.shape + (2,))))
            M[(i, j)] = cost; M[(j, i)] = cost.T

    def pair(i, a, cur):
        return sum(M[(i, j)][a, cur[j]] for j in range(n) if j != i)

    def total_cost(c):
        return (sum(base[i][c[i]] for i in range(n)) +
                sum(M[(i, j)][c[i], c[j]] for i in range(n) for j in range(i + 1, n)))

    # greedy start (most extreme first, as passed), then seeded annealing runs (restarts), each ending with an ICM
    # sweep; the cheapest result is kept
    start = np.zeros(n, int)
    for i in range(n):
        tot = base[i] + sum(M[(i, j)][:, start[j]] for j in range(i))
        start[i] = int(np.argmin(tot))
    best_c, best_t = start.copy(), total_cost(start)
    for sd in range(seed, seed + restarts):
        cur = start.copy()
        rng = np.random.default_rng(sd)
        Tmax, Tmin = 60.0, 0.3
        for it in range(iters):
            Tt = Tmax * (Tmin / Tmax) ** (it / iters)
            i = int(rng.integers(n)); a = int(rng.integers(len(base[i])))
            old = cur[i]
            dl = base[i][a] - base[i][old] + sum(M[(i, j)][a, cur[j]] - M[(i, j)][old, cur[j]]
                                                 for j in range(n) if j != i)
            if dl < 0 or rng.random() < np.exp(-dl / Tt):
                cur[i] = a
        for _ in range(3):
            for i in range(n):
                tot = base[i] + sum(M[(i, j)][:, cur[j]] for j in range(n) if j != i)
                cur[i] = int(np.argmin(tot))
        t = total_cost(cur)
        if t < best_t:
            best_c, best_t = cur.copy(), t
    cur = best_c; total = best_t
    # draw
    inv = ax.transData.inverted()
    out = []
    for i, (x, y, s, col) in enumerate(labels):
        b = cand_box[i][cur[i]]; dy0 = meas[i][2]
        tx, ty = inv.transform((b[0], b[1] - dy0))
        ax.text(tx, ty, s, color=col, fontsize=fontsize, ha="left", va="baseline", zorder=5, clip_on=False)
        d = cand_d[i][cur[i]]
        if d >= lead_min:
            p = LP[i]; q = cand_q[i][cur[i]]; v = q - p; u = v / np.hypot(*v)
            A = inv.transform(p + u * (m + 0.3)); B = inv.transform(q - u * 0.4)
            ax.plot([A[0], B[0]], [A[1], B[1]], "-", color="k", lw=lead_lw, zorder=4, clip_on=False)
        overl = any(M[(i, j)][cur[i], cur[j]] >= 1e4 for j in range(n) if j != i)
        out.append(dict(text=s, leader_pt=float(d if d >= lead_min else 0.0), overlaps_label=bool(overl),
                        point=[round(float(v), 1) for v in LP[i]], box=[round(float(v), 1) for v in b],
                        q=[round(float(v), 1) for v in cand_q[i][cur[i]]]))
    return out, float(total)
