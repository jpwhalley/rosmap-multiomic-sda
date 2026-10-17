"""Shared helpers for the neuronal (C172, C277) and C21 revision analyses (2026-10-13, round 2).
Read-only on the SDA archive, clinical metadata, snRNA-seq objects and the earlier round-2 outputs; writes only under
$G_ROS_RESULTS_DIR/revision_analyses/neuronal_c21/ (aggregates) and $G_ROS_RESULTS_DIR/revision_analyses/neuronal_c21/cache
(per donor). Reuses the validated loaders of dossier/dc_common.py (scores547, design_frame, rint, bh)."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, sys, numpy as np, pandas as pd
from scipy import stats as ss

R2 = cfg.WORK
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dossier"))   # dc_common.py
import dc_common as dc                                     # noqa: E402  (validated loaders)
from dc_common import scores547, clinical, design_frame, rint, bh, CTS, SDA, ROS   # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = f"{R2}/revision_analyses/neuronal_c21"
OUT = f"{ROOT}/out"; CACHE = f"{cfg.WORK}/revision_analyses/neuronal_c21/cache"; PANELS = f"{ROOT}/panels"
for d in (OUT, CACHE, PANELS):
    cfg.out_dir(d)
KEY = f"{cfg.PRIVATE}/donor_key.csv"
RNA_MISMATCH_LABELS = ["RNA-1", "RNA-2"]                    # snRNA profile is another person's (identity_checks)
CHIP_MISMATCH_LABELS = ["CHIP-1", "CHIP-2"]


def rna_mismatch_donors():
    """projids (str) of the two snRNA samples that are not the donor's own; never printed."""
    k = pd.read_csv(KEY)
    return [str(p) for p in k.loc[k.label.isin(RNA_MISMATCH_LABELS), "projid"]]


def full_frame():
    """dossier design frame + Braak, raw CERAD, last-visit MMSE; derived phenotype codings."""
    D = design_frame()
    clin = clinical()
    D["braak"] = clin.braaksc.astype(float)
    D["mmse_lv"] = pd.to_numeric(clin.cts_mmse30_lv, errors="coerce")
    D["cerad_burden"] = 5 - D.ceradsc                       # 1 = no AD ... 4 = definite AD (higher = more neuritic plaques)
    D["cogdx_mci"] = (D.cogdx == 2).astype(float)
    D["cogdx_ad"] = (D.cogdx == 4).astype(float)
    D["dementia_any"] = D.cogdx.isin([4, 5, 6]).astype(float)   # AD dementia, AD + other, other dementia
    D["braak_hi"] = (D.braak >= 5).astype(float)
    return D


def ols(y, X, names):
    """OLS with t-based 95% CI. X without intercept. Returns DataFrame of coefficients (excluding intercept), rss, df."""
    Xo = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(Xo, y, rcond=None)
    res = y - Xo @ b; dfr = len(y) - Xo.shape[1]; s2 = res @ res / dfr
    cov = s2 * np.linalg.pinv(Xo.T @ Xo); se = np.sqrt(np.diag(cov)); tq = ss.t.ppf(0.975, dfr)
    T = pd.DataFrame({"beta": b, "se": se, "lo": b - tq * se, "hi": b + tq * se,
                      "p": 2 * ss.t.sf(np.abs(b / se), dfr)}, index=["const"] + list(names))
    return T.drop(index="const"), float(res @ res), dfr


def nested_F(y, Xfull, Xred):
    """F test of the columns in Xfull not in Xred (both without intercept)."""
    def rss(X):
        Xo = np.column_stack([np.ones(len(y))] + ([X] if X.shape[1] else []))
        b, *_ = np.linalg.lstsq(Xo, y, rcond=None); r = y - Xo @ b; return r @ r, len(y) - Xo.shape[1]
    rf, dff = rss(Xfull); rr, _ = rss(Xred); q = Xfull.shape[1] - Xred.shape[1]
    F = ((rr - rf) / q) / (rf / dff); return float(F), float(ss.f.sf(F, q, dff)), float(1 - rf / rr)


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
