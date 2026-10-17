"""Read SDA-input RNA values (as fed to SDA: CPM, rna.txt; blocks of 276 donors in cell_types.txt order) from the
component_dossier cache, and the loadings/PIPs of the dossier components. Cross-checked against source_panels slices."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import numpy as np
from nc_common import R2, CTS, SDA

_RNA = None; _FE = None; _L = None


def _load():
    global _RNA, _FE, _L
    if _RNA is None:
        _RNA = np.load(f"{cfg.WORK}/component_dossier/cache/input_RNA.npy", mmap_mode="r")
        _L = dict(np.load(f"{cfg.WORK}/component_dossier/cache/loadings.npz", allow_pickle=True))
        _FE = {g: i for i, g in enumerate(_L["RNA_feats"])}
        assert _RNA.shape == (6 * 276, len(_FE))
        assert [l.strip() for l in open(f"{SDA}/cell_types.txt") if l.strip()] == CTS


def feature_values_by_name(gene, ct):
    _load(); c = CTS.index(ct)
    return np.asarray(_RNA[c * 276:(c + 1) * 276, _FE[gene]], dtype=float)


def rna_block(ct, idx):
    _load(); c = CTS.index(ct)
    return np.asarray(_RNA[c * 276:(c + 1) * 276][:, idx], dtype=float)


def loadings(k):
    """gene loadings X, PIP P, gene names, corrected cell-type loadings (dict) for component k."""
    _load(); r = int(np.where(_L["comps"] == k)[0][0])
    return _L["RNA_X"][r], _L["RNA_P"][r], _L["RNA_feats"], dict(zip(CTS, _L["B_cor"][r]))


def crosscheck():
    """compare with source_panels/cache/slices.npz for the LRP8 / PLCB1 columns."""
    S = dict(np.load(f"{cfg.WORK}/source_panels/cache/slices.npz", allow_pickle=True))
    out = {}
    for g, ct in (("LRP8", "Inh"), ("PLCB1", "Exc"), ("OR2L3", "Exc")):
        j = list(S["RNA_feats"]).index(g); pos = int(np.where(S["RNA_sel"] == j)[0][0]); c = CTS.index(ct)
        v = S["RNA_V"][c * 276:(c + 1) * 276, pos].astype(float)
        out[(g, ct)] = float(np.max(np.abs(v - feature_values_by_name(g, ct))))
    return out
