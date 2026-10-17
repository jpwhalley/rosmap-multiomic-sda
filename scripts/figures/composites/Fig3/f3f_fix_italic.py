"""Final Fig 3f: the y title '$\\mathit{FKBP5}$ (CPM)' set the digit '5' upright (matplotlib mathtext draws
non-letters of \\mathit in the roman font), so the gene symbol was only partly italic. Redraw panel f exactly as
f3_panels.py panel_f (same data, sizes, colours, ring, legend, limits) but with the y title built from two
text runs on one baseline: 'FKBP5' in Arial Italic and ' (CPM)' in Arial, placed where matplotlib puts the plain
one-run title (same baseline x and vertical centre). Writes only under <G_ROS_FIGURE_OUT>/Fig3/ (panels_fix/). No donor IDs are printed."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f3_panels as F3                      # module import only: draws nothing, writes nothing at import
VF = F3.HERE
import numpy as np
import matplotlib as mpl
import seaborn as sns

F3.PAN = f"{VF}/panels_fix"; os.makedirs(F3.PAN, exist_ok=True)
F3.NAMES["f"] = "Fig3f_C278_FKBP5_OPC_Oli_cell_v2"


def panel_f_v2():
    D, V, AK, FK, EXT = F3.D, F3.V, F3.AK, F3.FK, F3.EXT
    fig, w, h, left, bottom, right, top = F3.new_cell("f")
    ax = F3.add_axes_mm(fig, w, h, left, bottom, right, top)
    rhos = {}
    for ct in ("OPC", "Oli"):
        y = FK[ct]
        ax.scatter(AK, y, s=F3.F_S, color=D.CT_COL[ct], alpha=0.8, linewidths=0, label=ct, zorder=2)
        ax.scatter([AK[EXT]], [y[EXT]], s=F3.F_RING_S, facecolors="none", edgecolors=V.EXTRING,
                   linewidths=F3.F_RING_LW, zorder=3)
        r, p, n = V.spearman(y, AK)
        rhos[ct] = (r, n)
    ax.set_xlabel(f"Component {D.K} loadings/score")
    ax.legend(frameon=False, loc="upper left", markerscale=1.3, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.3)
    sns.despine(ax=ax)
    assert all(n == 274 for _, n in rhos.values()) and round(rhos["OPC"][0], 2) == 0.89 and round(rhos["Oli"][0], 2) == 0.83
    # reference: where matplotlib puts the one-run title (all upright) -> baseline x (right edge - descent) and centre y
    ax.set_ylabel("FKBP5 (CPM)")
    fig.canvas.draw(); r_ = fig.canvas.get_renderer()
    ref = ax.yaxis.label.get_window_extent(r_)
    ax.set_ylabel("")
    kw = dict(fontsize=F3.TXT, rotation=90, rotation_mode="anchor", ha="left", va="baseline", transform=None,
              color=mpl.rcParams["axes.labelcolor"])
    t1 = fig.text(0, 0, "FKBP5", fontstyle="italic", **kw)
    t2 = fig.text(0, 0, " (CPM)", **kw)
    # lengths along the baseline (rotated 90: vertical extents); include the leading space of t2 via an 'x' probe
    fig.canvas.draw()
    L1 = t1.get_window_extent(r_).height
    probe_a = fig.text(0, 0, "x (CPM)", fontsize=F3.TXT); probe_b = fig.text(0, 0, "x", fontsize=F3.TXT)
    fig.canvas.draw()
    L2 = probe_a.get_window_extent(r_).width - probe_b.get_window_extent(r_).width
    probe_a.remove(); probe_b.remove()
    cy = 0.5 * (ref.y0 + ref.y1)
    y0 = cy - 0.5 * (L1 + L2)
    t2.set_text("(CPM)")
    sp_a = fig.text(0, 0, "x (", fontsize=F3.TXT); sp_b = fig.text(0, 0, "x(", fontsize=F3.TXT)
    fig.canvas.draw()
    space = sp_a.get_window_extent(r_).width - sp_b.get_window_extent(r_).width
    sp_a.remove(); sp_b.remove()
    # baseline x: matched so the union box right edge equals the reference box right edge (descender side)
    t1.set_position((0, y0)); t2.set_position((0, y0 + L1 + space))
    fig.canvas.draw()
    u1 = t2.get_window_extent(r_)                  # '(CPM)' has the parentheses' descent, as the reference title
    bx = ref.x1 - u1.x1
    t1.set_position((bx, y0)); t2.set_position((bx, y0 + L1 + space))
    fig.canvas.draw()
    b1, b2 = t1.get_window_extent(r_), t2.get_window_extent(r_)
    # checks: on one line, clear of the y tick labels, inside the cell
    tl = [t.get_window_extent(r_) for t in F3.drawn_ticklabels(ax.yaxis)]
    assert max(b1.x1, b2.x1) <= min(t.x0 for t in tl) + 0.01, "y title overlaps tick labels"
    assert b1.y1 <= b2.y0 + 0.01, "runs overlap"
    assert min(b1.x0, b2.x0) >= 0 and min(b1.y0, b2.y0) >= 0
    F3.LOG.append(f"f v2: y title runs 'FKBP5' (italic) + '(CPM)'; centre {0.5 * (b1.y0 + b2.y1):.2f} px vs reference "
                  f"{cy:.2f}; right edge {b2.x1:.2f} vs {ref.x1:.2f}; gap {b2.y0 - b1.y1:.2f} pt (space advance {space:.2f})")
    F3.save_cell(fig, "f", extra_texts=list(ax.get_legend().get_texts()) + [t1, t2])
    return rhos


if __name__ == "__main__":
    rh = panel_f_v2()
    F3.check_f()                                   # same point check against the panels_new Fig3f as the builder ran
    open(f"{VF}/f3f_fix_italic.log", "w").write("\n".join(F3.LOG) + "\n")
    print("\n".join(F3.LOG))
