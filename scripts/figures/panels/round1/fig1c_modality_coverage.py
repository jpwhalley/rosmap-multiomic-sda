"""
Fig 1c of the first revision (v23: Figure 1b). Aggregate modality availability across the 276-donor SDA anchor.
The proteomics bar is credited to Johnson et al. 2020, the AD Knowledge Portal reference for the ROSMAP round-1 TMT data
(the first revision said Ping; corrected 2026-10-17).
No donor-level structure; four aggregate bars.
"""
import os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

ANCHOR = 276
# Palette matched to Fig 1A schematic:
#   snRNA-seq  → black
#   Methylation → orange
#   H3K9ac ChIP → green
#   MassSpec   → purple
modalities = [
    ("snRNA-seq\n(Mathys)", 276, "#222222"),
    ("DNA methylation\n(De Jager)", 216, "#E67E22"),
    ("H3K9ac ChIP-seq\n(Klein)", 193, "#4CAF50"),
    ("TMT proteomics\n(Johnson)", 105, "#8E44AD"),
]
labels = [m[0] for m in modalities]
counts = [m[1] for m in modalities]
colors = [m[2] for m in modalities]

fig, ax = plt.subplots(figsize=(4.2, 2.6))

# Background bars at anchor (light grey)
ax.barh(range(len(modalities)), [ANCHOR]*len(modalities),
        color="#e8e8e8", edgecolor="none", height=0.7)

# Foreground coverage bars
bars = ax.barh(range(len(modalities)), counts,
               color=colors, edgecolor="none", height=0.7)

# Value labels
for i, (n, c) in enumerate(zip(counts, colors)):
    pct = 100 * n / ANCHOR
    ax.text(n + 5, i, f"{n}/{ANCHOR}  ({pct:.0f}%)",
            va="center", ha="left", fontsize=8.5, color="#333")

# 276-anchor reference tick
ax.set_xlim(0, ANCHOR + 60)
ax.set_xticks([0, 100, 200, ANCHOR])
ax.set_xticklabels(["0", "100", "200", str(ANCHOR)])
ax.set_yticks(range(len(modalities)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Donors (of 276-donor SDA anchor)", fontsize=9)
ax.tick_params(axis="both", length=3, pad=2)

plt.tight_layout()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # scripts/figures
from figure_paths import FIGOUT
OUT = os.path.join(FIGOUT, "panels", "round1"); os.makedirs(OUT, exist_ok=True)
plt.savefig(os.path.join(OUT, "fig1c_modality_coverage.pdf"), bbox_inches="tight", dpi=300)
plt.savefig(os.path.join(OUT, "fig1c_modality_coverage.png"), bbox_inches="tight", dpi=180)
print("Saved fig1c_modality_coverage.{pdf,png}")
