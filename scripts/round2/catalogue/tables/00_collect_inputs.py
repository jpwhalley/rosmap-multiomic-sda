"""Collect the inputs of make_DataS1_cell_types_corrected.py and make_DataS1_547.py into $G_ROS_TABLES_DIR.
In the 10-08 analysis these files were copied or written by hand; this script repeats those steps.
  cell_type_evidence_from_runs.csv   = $G_ROS_AUDIT_DIR/ds1verify_cosB/cos_all.csv (catalogue/cell_type_evidence/s3_cos.py)
  clinical_BH_547_family.csv         = $G_ROS_AUDIT_DIR/c547/clinical_bh_547.csv (build_DataS1_new_rows_C547_C548.py)
  DataS1_new_rows_C547_C548.json     = $G_ROS_AUDIT_DIR/c547/new_rows.json (same)
  KEGG_2021_Human_C547.csv, _C548    = $G_ROS_AUDIT_DIR/kegg547/C547_kegg.csv, C548_kegg.csv (kegg/kegg547/06_compute_c547_c548.py)
  SEAAD_partner_rematch_141.csv      = the rows of $G_ROS_AUDIT_DIR/seaad141/partner_full_vs_corrected.csv with changed == True
                                       (seaad/02_partner.py); byte-identical to the file used
  cell_type_loadings_corrected.csv   = rows 1-546 of $G_ROS_AUDIT_DIR/B1_mean_cons.npy (catalogue/audit/02_sign_audit.py).
                                       The file used was written by a step that is not in the record; it agrees with this
                                       one to 1.8e-15 and gives byte-identical Data S1 cell-type columns.
The first four are copied as they are."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import shutil
import numpy as np, pandas as pd

A, T = cfg.AUD, cfg.out_dir(cfg.TABLES)
for src, dst in [(f"{A}/ds1verify_cosB/cos_all.csv", "cell_type_evidence_from_runs.csv"),
                 (f"{A}/c547/clinical_bh_547.csv", "clinical_BH_547_family.csv"),
                 (f"{A}/c547/new_rows.json", "DataS1_new_rows_C547_C548.json"),
                 (f"{A}/kegg547/C547_kegg.csv", "KEGG_2021_Human_C547.csv"),
                 (f"{A}/kegg547/C548_kegg.csv", "KEGG_2021_Human_C548.csv")]:
    shutil.copyfile(src, f"{T}/{dst}")
R = pd.read_csv(f"{A}/seaad141/partner_full_vs_corrected.csv", dtype=str, keep_default_na=False)
cols = ["rosmap", "omics_count", "ds1_comp", "ds1_corr", "full_comp", "full_corr", "new_comp", "new_corr", "n_genes",
        "new_runner_up", "new_runner_up_abs"]
R[R.changed == "True"][cols].to_csv(f"{T}/SEAAD_partner_rematch_141.csv", index=False)
B = np.load(f"{A}/B1_mean_cons.npy")[:546]
pd.DataFrame(B, index=pd.Index(range(1, 547), name="Component"),
             columns=["Ast", "Oli", "Inh", "OPC", "Exc", "Mic"]).to_csv(f"{T}/cell_type_loadings_corrected.csv")
print("collected 7 files into", T)
