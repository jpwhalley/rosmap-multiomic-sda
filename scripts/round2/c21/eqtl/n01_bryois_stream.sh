#!/bin/bash
# Stream Bryois et al. 2022 (Zenodo 7276971) chr1 cell-type cis-eQTL summary statistics and keep only rows for the
# OR2L/OR2M cluster genes, plus the snp_pos rows for the variants kept. Read-only; nothing else stored.
cd "$(dirname "$0")" || exit 1
OUT=$(${PYTHON:-python} -c "import sys; sys.path.insert(0, '../..'); import config; print(config.out_dir(config.WORK, 'revision_analyses', 'neuronal_c21', 'net', 'bryois'))") || exit 1
BASE="https://zenodo.org/api/records/7276971/files"
for ct in Excitatory.neurons Inhibitory.neurons Oligodendrocytes OPCs...COPs Astrocytes Microglia Endothelial.cells Pericytes pb; do
  curl -s -m 1800 -L "$BASE/${ct}.1.gz/content" | gzip -dc | grep -E "^(OR2L3|OR2L13|OR2AK2|OR2L2|OR2L5|OR2L8|OR2M4|OR2M3|OR2W3|OR2T8|CLK3P2|OR2AJ1)_" > "$OUT/${ct}.chr1.OR.txt"
  echo "$ct $(wc -l < "$OUT/${ct}.chr1.OR.txt") rows $(date)"
  sleep 2
done
# snp_pos.txt.gz columns: SNP, SNP_id_hg38 (chr:pos), SNP_id_hg19, effect_allele, other_allele, MAF (differs from the
# uncompressed snp_pos.txt); the first run (14:22) used the uncompressed layout and kept only the header, so this line
# was re-run alone at 14:23 with the filter below: header + GRCh38 chr1:247,000,000-248,999,999.
curl -s -m 1800 -L "$BASE/snp_pos.txt.gz/content" | gzip -dc | awk -F"\t" 'NR==1 || $2 ~ /^chr1:24[78][0-9][0-9][0-9][0-9][0-9][0-9]$/' > "$OUT/snp_pos.chr1_247-249Mb.txt"
echo "snp_pos $(wc -l < "$OUT/snp_pos.chr1_247-249Mb.txt") rows $(date)"
