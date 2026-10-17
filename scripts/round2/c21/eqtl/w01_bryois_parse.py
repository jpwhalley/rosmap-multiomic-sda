"""Independent parse of cached Bryois et al. 2022 Zenodo nominal cis-eQTL files (read-only).
Full chr1 files for Exc, Ast, Mic (downloaded from Zenodo 7276971 into verify_neuronal_c21/net/bryois); OR-filtered files for
the rest (neuronal_c21/net/bryois, n01_bryois_stream.sh; grep filter
OR2L3|OR2L13|OR2AK2|OR2L2|OR2L5|OR2L8|OR2M4|OR2M3|OR2W3|OR2T8|CLK3P2|OR2AJ1). Columns: gene, SNP, dist, p, beta."""
import os, sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/round2
import config as cfg  # noqa: E402  (paths and thread cap; see scripts/round2/config.py)
import gzip, collections, os, sys
R2 = cfg.WORK
V = f"{R2}/revision_analyses/verify_neuronal_c21/net/bryois"; N = f"{R2}/revision_analyses/neuronal_c21/net/bryois"
FILTER = set("OR2L3 OR2L13 OR2AK2 OR2L2 OR2L5 OR2L8 OR2M4 OR2M3 OR2W3 OR2T8 CLK3P2 OR2AJ1".split())
OR15 = "OR2L3 OR2L5 OR2L2 OR2M4 OR2L13 OR2M3 OR2T6 OR2M2 OR2L8 OR2AK2 OR2T12 OR2T1 OR2G6 OR2T2 OR2T27".split()
SNP = "rs6664332"
def scan(path, opener):
    genes = collections.defaultdict(list)
    with opener(path, "rt") as fh:
        for line in fh:
            g, s, d, p, b = line.split()
            genes[g].append((s, int(d), float(p), float(b)))
    return genes
src = {"Exc": (f"{V}/Excitatory.neurons.1.gz", gzip.open, "full"), "Ast": (f"{V}/Astrocytes.1.gz", gzip.open, "full"),
       "Mic": (f"{V}/Microglia.1.gz", gzip.open, "full"),
       "Inh": (f"{N}/Inhibitory.neurons.chr1.OR.txt", open, "filtered"), "Oli": (f"{N}/Oligodendrocytes.chr1.OR.txt", open, "filtered"),
       "OPC": (f"{N}/OPCs...COPs.chr1.OR.txt", open, "filtered"), "Endo": (f"{N}/Endothelial.cells.chr1.OR.txt", open, "filtered"),
       "Peri": (f"{N}/Pericytes.chr1.OR.txt", open, "filtered"), "pb": (f"{N}/pb.chr1.OR.txt", open, "filtered")}
out = open(f"{cfg.out_dir(R2, 'selection_review', 'verify_eQTL_extension', 'out')}/w01_bryois_parse.tsv", "w")
out.write("cell_type\tfile_kind\tgene\ttested\tn_snps\tp_rs6664332\tbeta_rs6664332\trank_rs6664332\tlead_snp\tlead_p\tn_tied_at_lead\n")
for ct, (path, op, kind) in src.items():
    G = scan(path, op)
    # all genes where rs6664332 tested (cis window) in this file
    cis = sorted(g for g, rows in G.items() if any(r[0] == SNP for r in rows))
    print(f"== {ct} ({kind}); genes in file: {len(G)}; genes with rs6664332 in cis window: {len(cis)}")
    for g in cis:
        rows = G[g]; sym = g.split("_")[0]
        rows_s = sorted(rows, key=lambda r: r[2]); lead = rows_s[0]
        me = [r for r in rows if r[0] == SNP][0]
        rank = 1 + sum(1 for r in rows if r[2] < me[2]); ntie = sum(1 for r in rows if r[2] == lead[2])
        print(f"   {g:32s} n={len(rows):5d} p={me[2]!r} beta={me[3]!r} rank={rank} lead={lead[0]} p_lead={lead[2]!r} ties_at_lead={ntie}")
    for sym in OR15:
        hits = [g for g in G if g.split("_")[0] == sym]
        if not hits:
            st = "not_tested" if (kind == "full" or sym in FILTER) else "not_cached"
            out.write(f"{ct}\t{kind}\t{sym}\t{st}\t\t\t\t\t\t\t\n"); continue
        g = hits[0]; rows = G[g]; rows_s = sorted(rows, key=lambda r: r[2]); lead = rows_s[0]
        me = [r for r in rows if r[0] == SNP]
        if me:
            me = me[0]; rank = 1 + sum(1 for r in rows if r[2] < me[2])
            out.write(f"{ct}\t{kind}\t{sym}\ttested\t{len(rows)}\t{me[2]!r}\t{me[3]!r}\t{rank}\t{lead[0]}\t{lead[2]!r}\t{sum(1 for r in rows if r[2]==lead[2])}\n")
        else:
            out.write(f"{ct}\t{kind}\t{sym}\ttested_snp_absent\t{len(rows)}\t\t\t\t{lead[0]}\t{lead[2]!r}\t\n")
out.close()
