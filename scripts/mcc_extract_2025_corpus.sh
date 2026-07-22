#!/usr/bin/env bash
# Extract the May-2025-vintage MCC corpus subset into a clean root.
#
# Use this to materialize the property XMLs (CTL/LTL/Reachability/UpperBounds/...)
# that match the consensus CSV (raw-result-analysis.csv) produced from the
# May-2025 MCC inputs. The pre-existing /private/tmp/mcc-models-root/ contains
# 2024-vintage XMLs whose formula IDs do not match the 2025 CSV — running
# ty-mcc-csv-compare against that root yields spurious wrong-answer rows.
#
# Usage:
#   scripts/mcc_extract_2025_corpus.sh <subset-file> <dest-root>
# Example:
#   scripts/mcc_extract_2025_corpus.sh \
#     /tmp/mcc-13exam-v2-subset.txt \
#     /private/tmp/mcc-2025-corpus

set -euo pipefail

SUBSET="${1:?subset file required}"
DEST="${2:?destination root required}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/mcc-benchmarks/2025/inputs/INPUTS-2025}"

if [[ ! -d "${ARCHIVE_ROOT}" ]]; then
    echo "FATAL: archive root not found: ${ARCHIVE_ROOT}" >&2
    exit 1
fi

mkdir -p "${DEST}"

extracted=0
missing=0
while IFS= read -r model; do
    [[ -z "${model}" ]] && continue
    tgz="${ARCHIVE_ROOT}/${model}.tgz"
    if [[ ! -f "${tgz}" ]]; then
        echo "MISSING archive: ${model}" >&2
        missing=$((missing+1))
        continue
    fi
    # Re-extract every time so stale files cannot linger.
    rm -rf "${DEST:?}/${model}"
    tar xzf "${tgz}" -C "${DEST}"
    extracted=$((extracted+1))
done < "${SUBSET}"

echo "Extracted ${extracted} model(s) into ${DEST}"
if [[ "${missing}" -gt 0 ]]; then
    echo "Missing: ${missing}" >&2
    exit 2
fi
