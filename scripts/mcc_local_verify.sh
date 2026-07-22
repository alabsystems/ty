#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Local MCC verification harness: run ty-mcc over local models x examinations,
# tabulate outcomes (ANSWER / CANNOT_COMPUTE / TIMEOUT / partial '?'), and diff
# against a saved baseline. Used to measure CC->answer conversion from fixes
# without introducing wrong answers.
#
# Usage:
#   scripts/mcc_local_verify.sh [--timeout SECS] [--out FILE] [--exams "E1 E2"] [--models "dir1 dir2"]
# Defaults: timeout=60, all examinations, all tmp_benchmark_models + tmp_mcc dirs.
set -u

BIN="${TY_MCC_BIN:-./target/debug/ty-mcc}"
TIMEOUT=60
OUT="/tmp/mcc_local_results.tsv"
EXAMS="ReachabilityDeadlock OneSafe QuasiLiveness StableMarking Liveness StateSpace UpperBounds ReachabilityCardinality ReachabilityFireability CTLCardinality CTLFireability LTLCardinality LTLFireability"
MODELS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --timeout) TIMEOUT="$2"; shift 2;;
        --out) OUT="$2"; shift 2;;
        --exams) EXAMS="$2"; shift 2;;
        --models) MODELS="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

if [ -z "$MODELS" ]; then
    MODELS="$(ls -d tmp_benchmark_models/*/ tmp_mcc/*/ 2>/dev/null)"
fi

: > "$OUT"
classify() {
    # $1 = first result line from ty-mcc stdout
    case "$1" in
        *CANNOT_COMPUTE*) printf 'CC';;
        *DO_NOT_COMPETE*) printf 'DNC';;
        "") printf 'EMPTY';;
        *:\?*) printf 'PARTIAL';;
        FORMULA*|STATE_SPACE*) printf 'ANSWER';;
        *) printf 'OTHER';;
    esac
}

for M in $MODELS; do
    name="$(basename "$M")"
    for E in $EXAMS; do
        sdir="/tmp/ty-verify/${name}-${E}"
        rm -rf "$sdir" 2>/dev/null; mkdir -p "$sdir"
        start=$(date +%s)
        out="$(TY_MCC_REQUIRE_BACKEND_EVIDENCE=0 "$BIN" "$M" --examination "$E" \
                 --threads 4 --memory-fraction 0.5 --storage auto \
                 --storage-dir "$sdir" --timeout "$TIMEOUT" 2>/dev/null)"
        rc=$?
        end=$(date +%s)
        first="$(printf '%s\n' "$out" | grep -E '^(FORMULA|STATE_SPACE|CANNOT_COMPUTE|DO_NOT_COMPETE)' | head -1)"
        if [ "$rc" -ne 0 ] && [ -z "$first" ]; then cls="CRASH/rc$rc"; else cls="$(classify "$first")"; fi
        printf '%s\t%s\t%s\t%ss\t%s\n' "$name" "$E" "$cls" "$((end-start))" "$first" >> "$OUT"
        printf '%-32s %-24s %-8s %ss\n' "$name" "$E" "$cls" "$((end-start))"
    done
done

echo "=== SUMMARY ($OUT) ==="
awk -F'\t' '{c[$3]++} END{for(k in c) printf "  %-12s %d\n", k, c[k]}' "$OUT" | sort -t' ' -k2 -rn
