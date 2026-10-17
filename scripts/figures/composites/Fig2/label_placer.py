"""Deterministic text-label placement with leader lines (stands in for adjustText, which is not in the environment).
Copied from composites/S8/label_placer.py; Fig 2 change: separate horizontal and vertical
clearance between labels (pad_x, pad_y; default = pad as before) so that labels side by side cannot read as one line;
the leader-through-label and leader-crossing costs are parameters (c_lead_label, c_cross; defaults as before); optional
side_y (display y of the zero line) adds side_pen when a label sits on the other side of zero from its point.

Each label gets K candidate boxes inside the axes (a grid search, pruned to the cheapest by its own cost); a seeded
simulated annealing then picks one candidate per label to minimise the total cost:
  own cost   leader length (+ a small quadratic term), the label covering its own point, plotted points under the
             label, and another labelled point under or right next to the label (which would read as its owner);
  pair cost  two labels overlapping (prohibitive), a leader running through another label, two leaders crossing.
Works in display units of a 72-dpi figure (= PostScript points). Only presentation: the label set, text and colour
come from the caller unchanged."""
import numpy as np


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
          lead_min=2.0, lead_lw=0.3, pad=0.8, iters=40000, seed=1, top_overflow=0.0, restarts=8, pad_x=None, pad_y=None,
          c_lead_label=600.0, c_cross=400.0, side_y=None, side_pen=0.0, edge_pad=0.6, lead_clear=0.0, edge_pad_bottom=None, nudge=None):
    """edge_pad: clearance (pt) between a label box and the axes frame (addition; default as before).
    lead_clear: a leader counts as running through another label when it passes within this many pt of its box
    (addition; default 0 = as before)."""
    pad_x = pad if pad_x is None else pad_x; pad_y = pad if pad_y is None else pad_y
    fig.canvas.draw()
    Rnd = fig.canvas.get_renderer()
    T = ax.transData
    bb = ax.get_window_extent(Rnd)
    epb = edge_pad if edge_pad_bottom is None else edge_pad_bottom
    X0, Y0, X1, Y1 = bb.x0 + edge_pad, bb.y0 + epb, bb.x1 - edge_pad, bb.y1 - edge_pad + top_overflow   # labels may rise into the top margin
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
        if side_y is not None and side_pen:
            cy = (b[:, 1] + b[:, 3]) / 2
            c = c + side_pen * (np.sign(cy - side_y) != np.sign(py - side_y))
        for j in range(n):
            if j == i:
                continue
            qx, qy = LP[j]; mq = m + near_pt
            c = c + 200.0 * ((qx > b[:, 0] - mq) & (qx < b[:, 2] + mq) & (qy > b[:, 1] - mq) & (qy < b[:, 3] + mq))
        keep = np.argsort(c, kind="stable")[:K]
        cand_box.append(b[keep]); cand_q.append(np.stack([nx[keep], ny[keep]], 1)); cand_d.append(d[keep])
        base.append(c[keep])
    # pairwise matrices
    M = {}
    for i in range(n):
        for j in range(i + 1, n):
            Bi = cand_box[i][:, None, :]; Bj = cand_box[j][None, :, :]
            ov = ((np.minimum(Bi[..., 2], Bj[..., 2]) + pad_x > np.maximum(Bi[..., 0], Bj[..., 0]) - pad_x) &
                  (np.minimum(Bi[..., 3], Bj[..., 3]) + pad_y > np.maximum(Bi[..., 1], Bj[..., 1]) - pad_y))
            cost = 1e4 * ov
            li = cand_d[i][:, None] >= lead_min; lj = cand_d[j][None, :] >= lead_min
            pi = np.broadcast_to(LP[i], (len(cand_box[i]), 1, 2)); qi = cand_q[i][:, None, :]
            pj = np.broadcast_to(LP[j], (1, len(cand_box[j]), 2)); qj = cand_q[j][None, :, :]
            infl = np.array([-lead_clear, -lead_clear, lead_clear, lead_clear])
            cost = cost + c_lead_label * (li & _seg_rect(pi, qi, Bj + infl))
            cost = cost + c_lead_label * (lj & _seg_rect(pj, qj, Bi + infl))
            cost = cost + c_cross * (li & lj & _seg_cross(np.broadcast_to(pi, cost.shape + (2,)),
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
    nudge = nudge or {}
    for i, (x, y, s, col) in enumerate(labels):
        b = cand_box[i][cur[i]]; dy0 = meas[i][2]
        if s in nudge:                       # addition: shift one chosen label box (pt), leader recomputed
            dx_, dy_ = nudge[s]
            b = b + np.array([dx_, dy_, dx_, dy_])
            px_, py_ = LP[i]
            qn = np.array([np.clip(px_, b[0], b[2]), np.clip(py_, b[1], b[3])])
            cand_q[i] = cand_q[i].copy(); cand_q[i][cur[i]] = qn
            cand_d[i] = cand_d[i].copy(); cand_d[i][cur[i]] = float(np.hypot(*(qn - LP[i])))
            cand_box[i] = cand_box[i].copy(); cand_box[i][cur[i]] = b
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


def stack_place(fig, ax, pts_xy, labels, fontsize, marker_r_pt=1.0, gap_pt=1.0, edge_pad=0.8, clear_pt=1.2, lead_dx=14.0, align=True,
                lead_min=2.0, lead_lw=0.3, from_top=True):
    """Fig 2 addition (10-16). Deterministic column for a tight cluster of labelled points: labels sorted by their
    point's height, stacked from the top of the axes (from_top) or the bottom, each right-aligned as close to the
    cluster as possible (just left of the leftmost labelled point) and pushed left only as far as needed to cover no
    plotted point; straight leader from the label's right end to its point. Sorted rows give leaders that do not
    cross. Returns the same records as place() and the label boxes (display pt)."""
    fig.canvas.draw()
    R = fig.canvas.get_renderer(); T = ax.transData
    bb = ax.get_window_extent(R)
    P = T.transform(np.asarray(pts_xy, float))
    LP = np.array([T.transform((l[0], l[1])) for l in labels])
    meas = []
    for (x, y, s, c) in labels:
        t = ax.text(0, 0, s, fontsize=fontsize, ha="left", va="baseline", transform=None)
        e = t.get_window_extent(R); t.remove()
        meas.append((e.width, e.height, e.y0))
    order = np.argsort(-LP[:, 1]) if from_top else np.argsort(LP[:, 1])
    m = marker_r_pt
    r0 = LP[:, 0].min() - m - lead_dx                  # column right edge: lead_dx pt left of the cluster
    y_cur = (bb.y1 - edge_pad) if from_top else (bb.y0 + edge_pad)
    out = [None] * len(labels); boxes = [None] * len(labels)
    inv = ax.transData.inverted()
    rows = {}
    for i in order:                                 # pass 1: rows and the right edge each row would need
        w, h, dy0 = meas[i]
        y0, y1 = (y_cur - h, y_cur) if from_top else (y_cur, y_cur + h)
        r = r0
        while r - w > bb.x0 + edge_pad:
            cov = ((P[:, 0] > r - w - m - clear_pt) & (P[:, 0] < r + m + clear_pt) &
                   (P[:, 1] > y0 - m - clear_pt) & (P[:, 1] < y1 + m + clear_pt))
            if not cov.any():
                break
            r -= 0.5
        rows[i] = (y0, y1, r)
        y_cur = (y0 - gap_pt) if from_top else (y1 + gap_pt)
    r_col = min(v[2] for v in rows.values()) if align else None
    for i in order:                                 # pass 2: one common right edge (align) so no leader meets a label
        w, h, dy0 = meas[i]
        y0, y1, r = rows[i]
        if align:
            r = r_col
        b = np.array([r - w, y0, r, y1]); boxes[i] = b
        tx, ty = inv.transform((b[0], b[1] - dy0))
        s, col = labels[i][2], labels[i][3]
        ax.text(tx, ty, s, color=col, fontsize=fontsize, ha="left", va="baseline", zorder=5, clip_on=False)
        q = np.array([b[2] + 0.4, (b[1] + b[3]) / 2]); p = LP[i]
        d = float(np.hypot(*(q - p)))
        if d >= lead_min:
            v = q - p; u = v / np.hypot(*v)
            A = inv.transform(p + u * (m + 0.3)); B = inv.transform(q)
            ax.plot([A[0], B[0]], [A[1], B[1]], "-", color="k", lw=lead_lw, zorder=4, clip_on=False)
        out[i] = dict(text=s, leader_pt=d if d >= lead_min else 0.0, overlaps_label=False,
                      point=[round(float(v), 1) for v in p], box=[round(float(v), 1) for v in b],
                      q=[round(float(v), 1) for v in q])
    return out, np.array(boxes)
