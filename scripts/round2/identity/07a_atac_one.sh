#!/bin/bash
# per-chromosome fragment counts for one MIT snATAC library (read-only input; counts only). Called by 07a_atac_chrcounts.sh.
OUT=${ATAC_OUT:?run 07a_atac_chrcounts.sh}
b=$(basename "$1" .fragments.tsv.gz)
[ -s "$OUT/$b.chr.txt" ] && exit 0
gzip -cd "$1" | grep -v "^#" | cut -f1 | uniq -c > "$OUT/$b.chr.tmp" && mv "$OUT/$b.chr.tmp" "$OUT/$b.chr.txt"
