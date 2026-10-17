"""
Fig 4c of the first revision (v23: Figure 2c). Component 74 ROSMAP vs SEA-AD RNA-loading scatter, with the five
canonical sex-chromosome markers labelled directly on the panel.

n = 138 genes with PIP > 0.5 in C74. Overall Pearson r = -0.982.
Extremes labelled per R2.8: XIST (chrX); TTTY14, NLGN4Y, UTY, USP9Y (chrY).
"""
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
})

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # scripts/figures
from figure_paths import ROUND1, FIGOUT
OUT = os.path.join(FIGOUT, "panels", "round1"); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(f"{ROUND1}/fig4c_points.csv", index_col=0)
labels = ["XIST", "TTTY14", "NLGN4Y", "UTY", "USP9Y"]

fig, ax = plt.subplots(figsize=(4.6, 4.2))

# All 138 points
ax.scatter(df["ROSMAP"], df["SEA_AD"],
           s=14, color="#4C72B0", alpha=0.55, edgecolor="none", zorder=2)

# Highlight the five extreme genes with a slightly larger, saturated dot
for g in labels:
    x, y = df.loc[g, "ROSMAP"], df.loc[g, "SEA_AD"]
    ax.scatter([x], [y], s=32, color="#222222", edgecolor="white",
               linewidth=0.5, zorder=4)

# Label placement: offset each label so they don't collide with axes or each other
label_offsets = {
    "XIST":   ( 100, -100),
    "TTTY14": ( 80,  -30),
    "NLGN4Y": ( 130,  30),
    "UTY":    ( 130,  30),
    "USP9Y":  (-140,  30),
}
for g, (dx, dy) in label_offsets.items():
    x, y = df.loc[g, "ROSMAP"], df.loc[g, "SEA_AD"]
    ax.annotate(
        f"$\\it{{{g}}}$",
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=10,
        ha="left" if dx > 0 else "right",
        va="center",
        color="#111111",
        arrowprops=dict(arrowstyle="-", color="#666", lw=0.6, shrinkA=2, shrinkB=3),
        zorder=5,
    )

# Reference: y = -x line (perfect anti-correlation would follow it)
lim = max(abs(df["ROSMAP"].min()), abs(df["ROSMAP"].max()),
          abs(df["SEA_AD"].min()), abs(df["SEA_AD"].max())) * 1.05
ax.plot([-lim, lim], [lim, -lim], color="#bbb", linewidth=0.7, linestyle=":", zorder=1)

ax.axhline(0, color="#999", linewidth=0.6, zorder=1)
ax.axvline(0, color="#999", linewidth=0.6, zorder=1)

ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("ROSMAP Component 74 RNA loading", fontsize=10)
ax.set_ylabel("SEA-AD sex-component loading", fontsize=10)

# r annotation in bottom-left (out of the way of the labelled extremes)
ax.text(0.03, 0.05, f"$r = -0.982$   $n = 138$",
        transform=ax.transAxes, va="bottom", ha="left",
        fontsize=10, color="#333")

ax.tick_params(axis="both", length=4, pad=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4c_labelled.pdf"), bbox_inches="tight", dpi=300)
plt.savefig(os.path.join(OUT, "fig4c_labelled.png"), bbox_inches="tight", dpi=180)
print("Saved fig4c_labelled.{pdf,png}")
