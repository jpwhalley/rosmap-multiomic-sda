"""Stream the raw 450K GenomeStudio report once; keep probes that are absent from the autosomal annotation
(= chrX, chrY, rs SNP probes and any other excluded probes). Saves AVG_Beta, Signal_A, Signal_B, Detection Pval
for every sample to $G_ROS_RESULTS_DIR/identity_checks/cache."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import os, time, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); C = cfg.out_dir(cfg.WORK, "identity_checks", "cache")
ROS = cfg.ROS
auto = set(l.split("\t", 1)[0] for l in open(f"{ROS}/oth/meth/ROSMAP_arrayMethylation_metaData.tsv"))
t0 = time.time()
f = open(f"{ROS}/oth/meth/ROSMAP_arrayMethylation_raw", "r", buffering=1 << 24)
for line in f:
    if line.startswith("TargetID"):
        hdr = line.rstrip("\n").split("\t"); break
print("header fields", len(hdr), "last fields", hdr[-2:])
want = ["AVG_Beta", "Signal_A", "Signal_B", "Detection Pval"]
samples = []
for h in hdr[3:]:
    if "." in h:
        s, fld = h.rsplit(".", 1)
        if s not in samples: samples.append(s)
cols = {w: [hdr.index(f"{s}.{w}") for s in samples] for w in want}
print("samples", len(samples), {w: len(v) for w, v in cols.items()})
ids, rows = [], {w: [] for w in want}; n = 0
for line in f:
    n += 1
    tid = line[:line.index("\t")] if "\t" in line else line.strip()
    if not tid or tid.startswith("["): continue
    if tid in auto: continue
    p = line.rstrip("\n").split("\t")
    ids.append(tid)
    for w in want:
        rows[w].append(np.array([p[i] if p[i] not in ("", "NaN") else "nan" for i in cols[w]], dtype=np.float32))
    if len(ids) % 10000 == 0: print(len(ids), "kept of", n, f"{time.time()-t0:.0f}s", flush=True)
print("rows scanned", n, "kept", len(ids), f"{time.time()-t0:.0f}s")
np.savez(f"{C}/meth_raw_nonauto.npz", ids=np.array(ids), samples=np.array(samples),
         **{w.replace(" ", "_"): np.vstack(rows[w]) for w in want})
print("saved", f"{time.time()-t0:.0f}s")
