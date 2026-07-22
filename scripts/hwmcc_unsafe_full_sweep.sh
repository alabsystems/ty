#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# HWMCC /unsafe/ full soundness canary sweep for #4311.
#
# The 34 HWMCC'25 benchmarks under:
#   ~/hwmcc/benchmarks/bitlevel/safety/2019/mann/data-integrity/unsafe/
# have SAT consensus and no canonical UNSAT verdicts per the R22 audit
# (`reports/2026-04-20-r22-hwmcc-soundness-audit.md`). Any `unsat` result
# from ty on this set is therefore a P0 false-UNSAT signal.
#
# Usage:
#   ./scripts/hwmcc_unsafe_full_sweep.sh [BINARY] [RESULTS_CSV] [INNER_TIMEOUT] [OUTER_TIMEOUT]
#
# Defaults:
#   BINARY        target/release/ty
#   RESULTS_CSV   /tmp/hwmcc_unsafe_full_sweep.csv
#   INNER_TIMEOUT 60
#   OUTER_TIMEOUT 75
#
# Environment:
#   HWMCC_SAFETY_ROOT            override benchmark root
#   HWMCC_UNSAFE_EXPECTED_COUNT  expected .aig count, default 34
#   HWMCC_UNSAFE_STRICT=1        also fail on unknown/timeout verdicts
#   TIMEOUT                      timeout command, default `timeout`

set -u

BINARY=${1:-${TY_BINARY:-target/release/ty}}
RESULTS_CSV=${2:-/tmp/hwmcc_unsafe_full_sweep.csv}
INNER_TIMEOUT=${3:-${HWMCC_UNSAFE_INNER_TIMEOUT:-60}}
OUTER_TIMEOUT=${4:-${HWMCC_UNSAFE_OUTER_TIMEOUT:-75}}
SAFETY_ROOT=${HWMCC_SAFETY_ROOT:-$HOME/hwmcc/benchmarks/bitlevel/safety}
EXPECTED_COUNT=${HWMCC_UNSAFE_EXPECTED_COUNT:-34}
STRICT=${HWMCC_UNSAFE_STRICT:-0}
UNSAFE_DIR="$SAFETY_ROOT/2019/mann/data-integrity/unsafe"
TIMEOUT_BIN=${TIMEOUT:-timeout}

if [ ! -x "$BINARY" ]; then
    echo "ERROR: binary not found or not executable: $BINARY" >&2
    echo "Build with: cargo build --release --bin ty --features ay" >&2
    exit 1
fi

if [ ! -d "$UNSAFE_DIR" ]; then
    echo "ERROR: HWMCC /unsafe/ benchmark dir not found: $UNSAFE_DIR" >&2
    echo "Set HWMCC_SAFETY_ROOT or install fixtures under ~/hwmcc/benchmarks/bitlevel/safety." >&2
    exit 1
fi

TMP_LIST=$(mktemp)
trap 'rm -f "$TMP_LIST"' EXIT

find "$UNSAFE_DIR" -name "*.aig" -type f | sort > "$TMP_LIST"
benchmark_count=$(wc -l < "$TMP_LIST" | tr -d '[:space:]')

if [ "$benchmark_count" != "$EXPECTED_COUNT" ]; then
    echo "ERROR: expected $EXPECTED_COUNT /unsafe/ AIGER benchmarks, found $benchmark_count in $UNSAFE_DIR" >&2
    echo "Refusing to call this a full #4311 sweep until the fixture set matches the R22 audit." >&2
    exit 1
fi

if command -v "$TIMEOUT_BIN" >/dev/null 2>&1; then
    HAVE_TIMEOUT=1
else
    HAVE_TIMEOUT=0
    echo "WARN: timeout command not found; relying only on ty --timeout $INNER_TIMEOUT" >&2
fi

echo "benchmark,result,exit_code,time" > "$RESULTS_CSV"

total=0
sat=0
unsat=0
unknown=0
timeouts=0
errors=0

echo "HWMCC /unsafe/ full sweep (#4311)"
echo "Binary: $BINARY"
echo "Benchmark dir: $UNSAFE_DIR"
echo "Benchmarks: $benchmark_count"
echo "Timeouts: inner=${INNER_TIMEOUT}s outer=${OUTER_TIMEOUT}s"
echo "Results CSV: $RESULTS_CSV"
echo ""

while IFS= read -r f; do
    total=$((total + 1))
    rel=${f#"$SAFETY_ROOT/"}

    start=$(python3 -c "import time; print(int(time.time()*1e9))")
    if [ "$HAVE_TIMEOUT" -eq 1 ]; then
        output=$("$TIMEOUT_BIN" "$OUTER_TIMEOUT" "$BINARY" aiger "$f" --timeout "$INNER_TIMEOUT" 2>&1)
        exit_code=$?
    else
        output=$("$BINARY" aiger "$f" --timeout "$INNER_TIMEOUT" 2>&1)
        exit_code=$?
    fi
    end=$(python3 -c "import time; print(int(time.time()*1e9))")
    elapsed=$(python3 -c "print(f'{($end - $start) / 1e9:.3f}')" 2>/dev/null || echo "?")

    verdict=$(printf '%s\n' "$output" | awk '
        /^[[:space:]]*[Ss][Aa][Tt][[:space:]]*$/ { print "sat"; exit }
        /^[[:space:]]*[Uu][Nn][Ss][Aa][Tt][[:space:]]*$/ { print "unsat"; exit }
        /^[[:space:]]*[Uu][Nn][Kk][Nn][Oo][Ww][Nn][[:space:]]*$/ { print "unknown"; exit }
    ')

    if [ "$exit_code" -eq 124 ]; then
        result="timeout"
        timeouts=$((timeouts + 1))
    elif [ "$verdict" = "sat" ]; then
        result="sat"
        sat=$((sat + 1))
    elif [ "$verdict" = "unsat" ]; then
        result="unsat"
        unsat=$((unsat + 1))
        echo "P0 SOUNDNESS: ty returned unsat on SAT-consensus /unsafe/ benchmark: $rel" >&2
    elif [ "$verdict" = "unknown" ]; then
        result="unknown"
        unknown=$((unknown + 1))
    else
        result="error"
        errors=$((errors + 1))
        echo "ERROR: no parseable verdict for $rel (exit $exit_code)" >&2
        printf '%s\n' "$output" >&2
    fi

    echo "$rel,$result,$exit_code,$elapsed" >> "$RESULTS_CSV"
    echo "[$total/$benchmark_count] $rel = $result (${elapsed}s)"
done < "$TMP_LIST"

echo ""
echo "=== HWMCC /unsafe/ FULL SWEEP RESULTS ==="
echo "Total:   $total"
echo "SAT:     $sat"
echo "UNSAT:   $unsat"
echo "UNKNOWN: $unknown"
echo "TIMEOUT: $timeouts"
echo "ERROR:   $errors"
echo "Results saved to: $RESULTS_CSV"

if [ "$unsat" -ne 0 ]; then
    echo "FAILED: $unsat false-UNSAT candidate(s) on SAT-consensus /unsafe/ benchmarks." >&2
    exit 2
fi

if [ "$errors" -ne 0 ]; then
    echo "FAILED: $errors benchmark execution/parsing error(s)." >&2
    exit 3
fi

if [ "$STRICT" = "1" ] && [ $((unknown + timeouts)) -ne 0 ]; then
    echo "FAILED: strict mode saw $unknown unknown and $timeouts timeout verdict(s)." >&2
    exit 4
fi

exit 0
