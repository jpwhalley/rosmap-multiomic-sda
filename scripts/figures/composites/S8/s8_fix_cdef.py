"""S8 final panels c, d, e and f: re-drawn with s8_panels.py (imported, not modified) and three presentation-only
changes. a and b are used as s8_panels.py draws them.
  c-e  label_placer_v2 (copy of the builder's placer): no dash-like short side leaders, no leader across another
       labelled point, no leaderless label that sits as close to another point as to its own, axis limits frozen
       before placing, colliding rotated chromosome tick labels separated by the least-squares minimum shift
       (centres >= 6.1 pt apart; tick marks have length 0, so only labels move). Per panel: c and d with the
       builder's w_pts 6 and hyphen cost 25; e with w_pts 15 (labels avoid covering points) and hyphen cost 60,
       chosen from several variants (not included) as the one with the fewest ambiguous labels.
  f    x tick-label pad 1.6 -> 2.4 pt so the axis-break marks no longer touch the '-0.008' tick labels.
Data, colours, label sets and text are untouched (s8_panels / s8_data do all of that). Writes to
<G_ROS_FIGURE_OUT>/S8/panels_fix/.
Prints aggregates only (no donor IDs, no per-donor values)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label_placer_v2
sys.modules["label_placer"] = label_placer_v2          # s8_panels' `import label_placer` now gets v2
import s8_panels as S
import matplotlib as mpl

OUT = f"{S.FIGOUT}/S8/panels_fix"
os.makedirs(OUT, exist_ok=True)
S.PAN = OUT

which = sys.argv[1:] or ["c", "d", "e", "f"]
moves = {}
for k, (j, name, row, wp, hc) in {"c": (1, "S8c_C351_RNA", "c", 6.0, 25.0), "d": (2, "S8d_C351_meth", "d", 6.0, 25.0),
                                  "e": (3, "S8e_C351_chip", "e", 15.0, 60.0)}.items():
    if k in which:
        label_placer_v2.LAST_TICK_MOVES = []
        label_placer_v2.HYPHEN_COST = hc
        S.panel_manhattan(j, name, S.ROWS[row], w_pts=wp)
        moves[k] = label_placer_v2.LAST_TICK_MOVES
if "f" in which:
    mpl.rcParams["xtick.major.pad"] = 2.4
    S.panel_f()
    mpl.rcParams["xtick.major.pad"] = 1.6
log = list(S.LOG)
for k, mv in moves.items():
    log.append(f"{k}: chromosome tick labels moved (pt): " + ", ".join(f"{l} {d:+.2f}" for l, d in mv))
open(os.path.join(OUT, f"s8_fix_{'_'.join(which)}.log"), "w").write("\n".join(log) + "\n")
print("\n".join(log))
