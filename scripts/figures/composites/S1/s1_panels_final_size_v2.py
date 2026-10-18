"""Figure S1 panels a-d redrawn at their final printed size (v23 supplement, 2026-10-16).
Final version (v2): the S1d C21 and C87 label positions were moved after a label-overlap check (see LPOS below).

Same data, same computations and same colours/labels as panels/p01_structure.py (S1a-d blocks, copied
here line for line); only the figure size, the explicit type sizes (6.5-7 pt Arial), line widths, marker sizes and
legend placement change, so that each panel prints at 100 % in the composite (supplement \\textwidth = 142.9 mm).
Label text is unchanged; two long legend entries in (a) are broken onto two lines.

Each panel is saved at exactly its cell size (no bbox_inches='tight'), with the axes placed in millimetres, so that
(a) and (b) share the same x-axis position and width (both are log top-donor share, same limits).

Reads: RA/structure/out/05_master_prioritisation.csv, RA/verify_structure/private/v02_core_plus.csv (as p01).
Writes: <G_ROS_FIGURE_OUT>/S1/panels/*.pdf and s1_panels.log there (aggregates only; one point per component, no donor data).
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \\
     python s1_panels_final_size_v2.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panels"))   # scripts/figures/panels
from vp_common import *          # noqa: F401,F403  (RA, CLS, loaders; sets the v22 style which is overridden below)
from matplotlib.lines import Line2D

PAN = f"{FIGOUT}/S1/panels"
os.makedirs(PAN, exist_ok=True)
MM = 1 / 25.4

# ------------------------------------------------------------------ style: v22 seaborn 'white', sizes at print size
mpl.rcParams["pdf.fonttype"] = 42
sns.set(style="white", context="paper")
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "mathtext.default": "regular",
    "font.size": 6.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "legend.title_fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.5,
    "patch.linewidth": 0.5, "lines.linewidth": 0.8,
    "legend.handlelength": 1.8, "legend.handletextpad": 0.5, "legend.labelspacing": 0.4,
    "legend.borderaxespad": 0.3, "legend.borderpad": 0.2,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
FS = 6.5          # annotation size (C-labels, class names, truncation note)


def panel(w_mm, h_mm, l_mm, b_mm, aw_mm, ah_mm):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    ax = fig.add_axes([l_mm / w_mm, b_mm / h_mm, aw_mm / w_mm, ah_mm / h_mm])
    return fig, ax


def savep(fig, name, LOG):
    fig.savefig(f"{PAN}/{name}.pdf")
    plt.close(fig)
    LOG.append(f"wrote panels/{name}.pdf")


# ------------------------------------------------------------------ geometry (mm); full width = 142.9
W = 142.9
AX_L = 10.0                     # common left edge of the axes in a, b, c (y titles and tick labels to its left)
AB_W = 92.0                     # common axes width of a and b (keys to the right)
A_H, B_H, CD_H = 40.0, 54.0, 55.0
D_W = 56.0                      # cell width of d; c gets the rest of the row minus a 4 mm gutter
GUT = 4.0
C_W = W - D_W - GUT
BOT = 8.0                       # room for x tick labels + x title
GEOM = dict(W=W, A_H=A_H, B_H=B_H, CD_H=CD_H, D_W=D_W, C_W=C_W, GUT=GUT)

# ================================================================== p01_structure.py, S1a-d (data and computations)
M = pd.read_csv(f"{RA}/structure/out/05_master_prioritisation.csv", index_col=0)
V = pd.read_csv(f"{RA}/verify_structure/private/v02_core_plus.csv", index_col=0)
assert (V.cls == M.conc_class).all() and np.allclose(V.top_share, M.top_share)
T1, T2 = 0.10, 0.50
n = M.conc_class.value_counts()
LOG = []
XT = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
XLO, XHI = 10 ** -2.4, 1.25     # shared x range of a and b (a's first bin edge is 10^-2.4; p01 b used 0.004)


def logx(ax, lo=XLO, hi=XHI):
    ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_xticks(XT); ax.set_xticklabels([str(t) for t in XT])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


XLAB = "top-donor share of score variance"

# ---------------------------------------------------------------- a  share histogram vs simulated references
fig, ax = panel(W, A_H, AX_L, BOT, AB_W, A_H - BOT - 2.0)
bins = np.arange(-2.4, 0.05, 0.1)
lg = np.log10(V.top_share)
ax.hist([lg[V.cls == c] for c in CLS], bins=bins, stacked=True, color=list(CLS.values()),
        label=[f"{c} (n = {int((V.cls == c).sum())})" for c in CLS])
rng = np.random.default_rng(31)
NSIM = 20000
for nm, draw, ls in [("Gaussian", lambda: rng.standard_normal((NSIM, 276)), "-"),
                     ("t, 5 df", lambda: rng.standard_t(5, (NSIM, 276)), ":"),
                     ("t, 12 df (best fit)", lambda: rng.standard_t(12, (NSIM, 276)), "--")]:  # df 12: best of a df grid by truncated histogram likelihood (scripts/round2/concentration/check/v07_prioritise_verify.py)
    A = draw(); D = A - A.mean(1, keepdims=True); D2 = D ** 2; ts = D2.max(1) / D2.sum(1)
    nbody = (V.top_share < 0.075).sum() / np.mean(ts < 0.075)
    h, _ = np.histogram(np.log10(ts), bins=bins)
    exp10 = nbody * np.mean(ts >= 0.10)
    # p01 label text, unchanged; the two long entries are broken onto two lines to fit the key column
    lab = f"{nm} reference" if nm.startswith("t, 5") else f"{nm} reference\n(expected ≥ 0.10: {exp10:.0f})"
    ax.step(bins[:-1], h / len(ts) * nbody, where="post", color="k", ls=ls, lw=0.8, label=lab)
    LOG.append(f"S1a reference {nm}: implied body {nbody:.1f}; expected >= 0.10 {exp10:.2f}; median share {np.median(ts):.4f}; "
               f"P(>=0.10) {np.mean(ts >= 0.10):.5f}")
    del A, D, D2
for t in (T1, T2):
    ax.axvline(np.log10(t), color="grey", ls=(0, (4, 3)), lw=0.6)
ax.set_xticks(np.log10(XT)); ax.set_xticklabels([str(t) for t in XT])
ax.set_xlim(np.log10(XLO), np.log10(XHI))
ax.set_xlabel(XLAB + " (log scale)"); ax.set_ylabel("components")
ax.set_ylim(0, 80)
n08 = int((V.top_share > 0.8).sum())
ax.text(np.log10(0.46), 77, f"{n08} above 0.8\n(bar truncated)", ha="right", va="top", fontsize=FS)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0)); sns.despine(ax=ax)
savep(fig, "S1a_top_share_vs_references_final", LOG)
LOG.append(f"S1a components > 0.8: {n08}")

# ---------------------------------------------------------------- b  overview
GREY = "#BBBBBB"
PH = {"none": (GREY, "o"), "sex": ("#0072B2", "D"), "CERAD": ("#E69F00", "^"), "diagnosis": ("#009E73", "s"),
      "CERAD and diagnosis": ("#D55E00", "P")}


def ph_cat(s):
    s = "" if pd.isna(s) else s
    if s == "": return "none"
    if s == "sex": return "sex"
    if s == "cerad": return "CERAD"
    if s == "cogdx": return "diagnosis"
    if set(s.split(";")) == {"cerad", "cogdx"}: return "CERAD and diagnosis"
    return s


M["ph"] = M.adj_calls.map(ph_cat); assert set(M.ph) <= set(PH)
rng = np.random.default_rng(1)
fig, ax = panel(W, B_H, AX_L, BOT, AB_W, B_H - BOT - 1.5)
y = M.n_modalities + rng.uniform(-0.36, 0.36, len(M))
ax.axvspan(XLO, T1, color=CLS["distributed"], alpha=0.06, lw=0); ax.axvspan(T2, XHI, color=CLS["concentrated"], alpha=0.06, lw=0)
pr = M.prioritised
S_NONE, S_PH, LW_OUT = 4.0, 12.0, 0.45     # marker areas (pt^2) and outline width at print size
for c in PH:
    col, mk = PH[c]
    for pp in (False, True):
        s_ = (M.ph == c) & (pr == pp)
        ax.scatter(M.top_share[s_], y[s_], c=col, marker=mk, s=(S_NONE if c == "none" else S_PH), alpha=0.85 if c == "none" else 1.0,
                   edgecolors="k" if pp else "none", linewidths=LW_OUT if pp else 0, zorder=2 if c == "none" else 3)
OFF = {74: (0.012, 3.55), 278: (0.010, 2.25), 172: (0.020, 2.6), 277: (0.045, 2.3), 21: (0.075, 0.2),
       351: (0.42, 4.5), 2: (0.3, 3.75), 87: (0.42, 3.3), 177: (0.3, 2.75)}
OFF[351] = (0.39, 4.5); OFF[87] = (0.39, 3.3)   # print size: nudged left so the text clears the 0.50 line (leaders unchanged)
for k, (tx, ty) in OFF.items():
    ax.annotate(f"C{k}", (M.top_share[k], y[k]), xytext=(tx, ty), fontsize=FS, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color="k", lw=0.4, shrinkA=1, shrinkB=1.5), zorder=5)
for t in (T1, T2): ax.axvline(t, color="k", ls="--", lw=0.6)
logx(ax); ax.set_ylim(-0.6, 5.1); ax.set_yticks(range(5))
ax.set_xlabel(XLAB + " (log scale)"); ax.set_ylabel("modalities with PIP > 0.5 features")
for x, c in [(0.02, "distributed"), (np.sqrt(T1 * T2), "few-donor"), (np.sqrt(T2 * 1.25), "concentrated")]:
    ax.text(x, 4.85, f"{c}\n(n = {n[c]})", ha="center", va="center", fontsize=FS)
h = [Line2D([], [], marker=PH[c][1], color=PH[c][0], ls="", markersize=4, label=c) for c in PH]
h.append(Line2D([], [], marker="o", mfc=GREY, mec="k", mew=LW_OUT, ls="", markersize=3.5, label=f"prioritised (n = {int(pr.sum())})"))
ax.legend(handles=h, title="adjusted association\n(FDR < 0.05)", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
ax.get_legend()._legend_box.align = "left"
sns.despine(ax=ax); savep(fig, "S1b_share_vs_modalities_final", LOG)
LOG.append(f"S1b classes {n.to_dict()}; prioritised {int(pr.sum())}; adjusted-call categories {M.ph.value_counts().to_dict()}")
LOG.append(f"S1b >=3 modalities: {int((M.n_modalities >= 3).sum())}, of them concentrated {int(((M.n_modalities >= 3) & (M.conc_class == 'concentrated')).sum())}")

# ---------------------------------------------------------------- c  run recurrence
fig, ax = panel(C_W, CD_H, AX_L, BOT, C_W - AX_L - 1.0, CD_H - BOT - 2.5)
ct = pd.crosstab(M.true_runs, M.conc_class)[list(CLS)]
frac = ct / ct.sum(0)
x = np.arange(len(ct.index)); w = 0.27
for i, c in enumerate(frac.columns):
    ax.bar(x + (i - 1) * w, frac[c], width=w - 0.02, color=CLS[c], label=f"{c} (n = {int(ct[c].sum())})")
ax.set_xticks(x); ax.set_xticklabels(ct.index); ax.set_xlabel("runs (of 10) in which the component recurs")
ax.set_ylabel("fraction of class"); ax.legend(frameon=False, loc="upper left"); sns.despine(ax=ax)
savep(fig, "S1c_run_recurrence_final", LOG)
LOG.append("S1c fraction in all 10 runs: " + ", ".join(f"{c} {frac.loc[10, c]:.2f} ({ct.loc[10, c]}/{ct[c].sum()})" for c in CLS))

# ---------------------------------------------------------------- d  SEA-AD Pearson vs Spearman
D_AX = CD_H - BOT - 2.5                      # square axes
fig, ax = panel(D_W, CD_H, D_W - D_AX - 1.0, BOT, D_AX, D_AX)
ok = M.seaad_J.notna()
for c in ["concentrated", "few-donor", "distributed"]:
    s = ok & (M.conc_class == c)
    ax.scatter(M.seaad_pearson.abs()[s], M.seaad_spearman.abs()[s], c=CLS[c], s=3.0, alpha=0.8, lw=0)
bs = M.support_B_seaad.astype(bool)
ax.scatter(M.seaad_pearson.abs()[bs], M.seaad_spearman.abs()[bs], facecolors="none", edgecolors="k", s=10, lw=0.4)
LPOS = {74: (0.93, 0.80), 278: (0.30, 0.66), 172: (0.84, 0.45), 277: (0.70, 0.33), 21: (0.24, 0.02), 2: (0.52, 0.13),
        87: (0.44, 0.27), 177: (0.75, 0.24), 351: (0.80, 0.06)}
# at print size the type is ~1.7x larger relative to the axes than in p01: two labels moved (same leaders to the same
# points): C278 below the key (was (0.30, 0.66), under the key's last entry), C21 off the x-axis line (was y 0.02)
LPOS[278] = (0.25, 0.575); LPOS[21] = (0.24, 0.045)
# at print size C21's label box (white, alpha 0.85) hid ~22 component markers in the dense
# lower-left cloud and C87's straddled the |r| = 0.4 line; moved to clear space (same text, same leaders to the same
# points): C21 to the empty area above the diagonal, C87 right of the 0.4 line. Markers hidden by labels: 39 -> 13.
LPOS[21] = (0.13, 0.285); LPOS[87] = (0.51, 0.24)
for k, (tx, ty) in LPOS.items():
    if pd.notna(M.seaad_J[k]):
        ax.annotate(f"C{k}", (abs(M.seaad_pearson[k]), abs(M.seaad_spearman[k])), xytext=(tx, ty), textcoords="data",
                    fontsize=FS, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.4, shrinkA=0, shrinkB=1), zorder=6)
ax.axvline(0.4, color="k", ls="--", lw=0.6); ax.axhline(0.4, color="k", ls="--", lw=0.6)
ax.plot([0, 1], [0, 1], color="grey", lw=0.5)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_xlabel("|Pearson r| with SEA-AD partner"); ax.set_ylabel("|Spearman ρ| with SEA-AD partner")
h = [Line2D([], [], marker="o", color=CLS[c], ls="", markersize=3, label=c) for c in CLS]
h.append(Line2D([], [], marker="o", mfc="none", mec="k", mew=0.4, ls="", markersize=3.6, label="cross-cohort\nsupport"))
# key kept left of the |r| = 0.4 line, in the empty corner (no component has |r| < 0.4 and |rho| > 0.53)
ax.legend(handles=h, frameon=False, loc="upper left", handlelength=0.8, handletextpad=0.3, labelspacing=0.3,
          borderaxespad=0.1, borderpad=0.1)
sns.despine(ax=ax)
savep(fig, "S1d_seaad_pearson_vs_spearman_final", LOG)
LOG.append(f"S1d components with a partner: {int(ok.sum())}; meeting the SEA-AD criterion: {int(bs.sum())}")

LOG.append(f"geometry_mm {GEOM}")
open(f"{PAN}/s1_panels.log", "w").write("\n".join(LOG) + "\n")
print("\n".join(LOG))
