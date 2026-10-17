"""Final S10c: label fix. The builder's panel-c recipe (s10_panels_at_cell_size.py) is imported,
not edited, and run unchanged, except that after its label placement each label named in argv[3] (default AL121890.2, whose glyphs
('8', '9') overlap a data point in the builder's panel; AL031666.2's leader grazes another point) is moved, in turn, to the position with the shortest leader for which
  * its glyphs keep >= marker radius + GL pt from every data point,
  * it keeps >= 0.8 pt from every other label and leader,
  * its leader keeps >= marker radius + LC pt from every other data point and crosses no other leader or label,
  * it stays >= 1 pt inside the axes.
Data, points, colours, label texts and sizes, every other label and leader, axes and geometry are the builder's.
Writes only <G_ROS_FIGURE_OUT>/S10/panels_fix/S10c_C87_RNA_cell.pdf. Prints aggregates only (no donor data are involved).
The published panel was made with: python fix_c_al121.py 1.5 0.3 AL121890.2,AL031666.2"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s10_panels_at_cell_size as B

B.PAN = f"{B.FIGOUT}/S10/panels_fix"; os.makedirs(B.PAN, exist_ok=True)
GL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
LC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
MOVES = [m for m in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["AL121890.2"])]
_orig = B.place_labels


def _seg_pt(a, b, P):
    ab = b - a; L2 = float(ab @ ab)
    t = np.clip(((P - a) @ ab) / L2, 0, 1) if L2 > 0 else np.zeros(len(P))
    return np.hypot(*(P - (a + t[:, None] * ab)).T)


def _seg_seg(a, b, c, d):
    o = lambda p, q, r: (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


def _seg_box(a, b, bx, pad):
    s = a + np.linspace(0, 1, 300)[:, None] * (b - a)
    return bool(np.any((s[:, 0] > bx[0] - pad) & (s[:, 0] < bx[2] + pad) & (s[:, 1] > bx[1] - pad) & (s[:, 1] < bx[3] + pad)))


def _box_pt(bx, P):
    return np.hypot(np.maximum(np.maximum(bx[0] - P[:, 0], 0), P[:, 0] - bx[2]),
                    np.maximum(np.maximum(bx[1] - P[:, 1], 0), P[:, 1] - bx[3]))


def _apart(b, o, pad):
    return b[2] + pad < o[0] or o[2] + pad < b[0] or b[3] + pad < o[1] or o[3] + pad < b[1]


def wrapped(fig, ax, labels, pts_xy, fs=B.LAB):
    texts = _orig(fig, ax, labels, pts_xy, fs=fs)
    for MOVE in MOVES:
        move_one(fig, ax, labels, pts_xy, texts, MOVE)
    return texts


def move_one(fig, ax, labels, pts_xy, texts, MOVE):
    names = [l[2] for l in labels]
    if MOVE not in names:
        return texts
    i = names.index(MOVE)
    fig.canvas.draw(); r = fig.canvas.get_renderer(); tr = ax.transData
    P = tr.transform(np.asarray(pts_xy, float))
    tgt = np.asarray(tr.transform((labels[i][0], labels[i][1])))
    others = P[np.hypot(*(P - tgt).T) > 1e-6]
    R = B.MS / 2 * 0.5
    # leaders: 2-point black lines; own leader starts at the label's data point
    own, leads = None, []
    for ln in ax.lines:
        xd, yd = ln.get_xdata(), ln.get_ydata()
        if len(xd) == 2 and ln.get_color() in ('k', 'black', (0, 0, 0, 1)):
            seg = tr.transform(np.c_[xd, yd])
            if np.allclose([xd[0], yd[0]], [labels[i][0], labels[i][1]]):
                own = ln
            else:
                leads.append(seg)

    def ink(bb):                 # the builder's text-box model: glyphs from 20% to 106% of the box height
        return (bb.x0, bb.y0 + 0.20 * bb.height, bb.x1, bb.y0 + 1.06 * bb.height)
    obox = [ink(t.get_window_extent(r)) for k, t in enumerate(texts) if k != i]
    full = [t.get_window_extent(r) for k, t in enumerate(texts) if k != i]
    fullb = [(b.x0, b.y0, b.x1, b.y1) for b in full]
    me = texts[i].get_window_extent(r); Wd, Hb = me.width, me.height
    old_ink = ink(me)
    old_lead = (_seg_pt(tgt, tr.transform(np.c_[own.get_xdata(), own.get_ydata()])[1], others).min() - R) if own is not None else float('nan')
    LOGX = [f"   {MOVE}: builder position glyph-to-point clearance {(_box_pt(old_ink, others).min() - R):+.2f} pt; "
            f"leader-to-point clearance {old_lead:+.2f} pt"]
    axb = ax.get_window_extent(r)
    best = None
    for x0 in np.arange(axb.x0 + 1, axb.x1 - 1 - Wd, 0.25):
        for y0 in np.arange(axb.y0 + 1, axb.y1 - 1 - Hb, 0.25):
            fb = (x0, y0, x0 + Wd, y0 + Hb); ib = (x0, y0 + 0.20 * Hb, x0 + Wd, y0 + 1.06 * Hb)
            if _box_pt(ib, others).min() < R + GL or _box_pt(ib, tgt[None])[0] < R + 0.3:
                continue
            if not all(_apart(ib, o, 0.8) for o in obox):
                continue
            if any(_seg_box(L[0], L[1], fb, 0.8) for L in leads):
                continue
            q = np.array([min(max(tgt[0], fb[0]), fb[2]), min(max(tgt[1], fb[1]), fb[3])])
            d = float(np.hypot(*(q - tgt)))
            if best is not None and d >= best[0]:
                continue
            if _seg_pt(tgt, q, others).min() < R + LC:
                continue
            if any(_seg_seg(tgt, q, L[0], L[1]) for L in leads):
                continue
            if any(_seg_box(tgt, q, o, 0.5) for o in fullb):
                continue
            best = (d, x0, y0, q)
    assert best is not None, "no clean position"
    d, x0, y0, q = best
    inv = tr.inverted()
    texts[i].set_ha('left'); texts[i].set_va('bottom'); texts[i].set_position(inv.transform((x0, y0)))
    qd = inv.transform(q)
    if own is not None:
        own.set_data([labels[i][0], qd[0]], [labels[i][1], qd[1]])
    else:
        ax.plot([labels[i][0], qd[0]], [labels[i][1], qd[1]], color='k', lw=B.LEAD_LW, zorder=3, solid_capstyle='butt')
    fig.canvas.draw()
    nb = ink(texts[i].get_window_extent(fig.canvas.get_renderer()))
    LOGX.append(f"   {MOVE}: moved; leader {d:.1f} pt; "
                f"glyph-to-point clearance {(_box_pt(nb, others).min() - R):+.2f} pt; leader-to-point clearance "
                f"{(_seg_pt(tgt, q, others).min() - R):+.2f} pt (GL {GL}, LC {LC})")
    B.LOG.extend(LOGX)
    return texts


B.place_labels = wrapped
d = B.panel_manhattan(1, "c")
B.check_rna_points(d)
B.check_labels_in_original(1, d)
print("\n".join(B.LOG))
