#!/usr/bin/env bash
# run_prep.sh — run the data-prep stage (scripts 01-05) end to end.
#
# Each script's stdout+stderr is teed to NN.log in the working directory.
# Bails on the first failing script (set -e + pipefail).
#
# Usage (from your validation work dir, with the synced repo at ../repo):
#   cd /path/to/work
#   bash ../repo/scripts/run_prep.sh
#
# Override the repo location if it lives elsewhere:
#   REPO=/path/to/repo bash run_prep.sh
#
# Resume from a given step (skips earlier ones):
#   FROM=03 bash ../repo/scripts/run_prep.sh

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
SCRIPTS_DIR="$REPO/scripts"
FROM="${FROM:-01}"

run() {
    local n="$1" name="$2"
    if [[ "$n" < "$FROM" ]]; then
        echo "--- skipping ${name} (FROM=${FROM}) ---"
        return 0
    fi
    local log="${n}.log"
    echo
    echo "=== [$(date '+%H:%M:%S')] running ${name} ==="
    if python "$SCRIPTS_DIR/$name" 2>&1 | tee "$log"; then
        echo "=== [$(date '+%H:%M:%S')] ${name} ok (log: $log) ==="
    else
        echo "!!! ${name} failed — see $log" >&2
        exit 1
    fi
}

run 01 01_prepare_snrna.py
run 02 02_prepare_chip.py
run 03 03_prepare_meth.py
run 04 04_prepare_proteomics.py
run 05 05_export_sda_inputs.py

echo
echo "All five prep scripts completed."
echo "Next: diff outputs against the reference archive (REF=... python \"$SCRIPTS_DIR/diff_against_cluster.py\")"
