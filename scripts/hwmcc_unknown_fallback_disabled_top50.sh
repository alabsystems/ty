#!/bin/bash
# Copyright 2026 Andrew Yates
# Licensed under the Apache License, Version 2.0

# HWMCC #4233 top-50 sweep with IC3 consecutive-Unknown fallback disabled.
#
# Usage:
#   ./scripts/hwmcc_unknown_fallback_disabled_top50.sh [BINARY] [RESULTS_CSV]
#
# Build example:
#   CARGO_TARGET_DIR=/tmp/ty-hwmcc-no-unknown-fallback \
#     cargo build --release --bin ty --features ay
#
# The benchmark corpus is external and expected at:
#   ${HWMCC_BENCH_DIR:-$HOME/hwmcc/benchmarks/bitlevel/safety}

set -u

INNER_TIMEOUT=${INNER_TIMEOUT:-100}
OUTER_TIMEOUT=${OUTER_TIMEOUT:-120}
BINARY=${1:-/tmp/ty-hwmcc-no-unknown-fallback/release/ty}
RESULTS_CSV=${2:-/tmp/hwmcc_unknown_fallback_disabled_top50_results.csv}
BENCH_DIR=${HWMCC_BENCH_DIR:-$HOME/hwmcc/benchmarks/bitlevel/safety}
DISABLE_ENV=TY_AIGER_DISABLE_IC3_UNKNOWN_FALLBACK

if ! command -v timeout >/dev/null 2>&1; then
    echo "ERROR: timeout(1) not found; install coreutils or run on a host with timeout" >&2
    exit 1
fi

if [ ! -x "$BINARY" ]; then
    echo "ERROR: binary not found or not executable: $BINARY" >&2
    echo "Build with: CARGO_TARGET_DIR=/tmp/ty-hwmcc-no-unknown-fallback cargo build --release --bin ty --features ay" >&2
    exit 1
fi

if [ ! -d "$BENCH_DIR" ]; then
    echo "ERROR: benchmark dir not found: $BENCH_DIR" >&2
    echo "Set HWMCC_BENCH_DIR to a directory containing HWMCC safety .aig files." >&2
    exit 1
fi

TMP_LIST=$(mktemp)
trap 'rm -f "$TMP_LIST"' EXIT

find "$BENCH_DIR" -name "*.aig" -type f -exec stat -f '%z %N' {} \; 2>/dev/null \
    | sort -n | head -n 50 | awk '{$1=""; sub(/^ /, ""); print}' > "$TMP_LIST"

if [ ! -s "$TMP_LIST" ]; then
    find "$BENCH_DIR" -name "*.aig" -type f -printf '%s %p\n' 2>/dev/null \
        | sort -n | head -n 50 | awk '{$1=""; sub(/^ /, ""); print}' > "$TMP_LIST"
fi

files=()
while IFS= read -r line; do
    files+=("$line")
done < "$TMP_LIST"

if [ "${#files[@]}" -lt 50 ]; then
    echo "ERROR: expected at least 50 .aig files under $BENCH_DIR, found ${#files[@]}" >&2
    exit 1
fi

echo "benchmark,result,time,size_bytes" > "$RESULTS_CSV"

echo "HWMCC #4233 unknown-fallback-disabled top-50 sweep"
echo "Binary: $BINARY"
echo "Benchmarks: $BENCH_DIR"
echo "Results: $RESULTS_CSV"
echo "Timeouts: inner=${INNER_TIMEOUT}s outer=${OUTER_TIMEOUT}s"
echo "$DISABLE_ENV=1"
echo ""

total=0
sat=0
unsat=0
unknown=0
errors=0

for f in "${files[@]}"; do
    total=$((total + 1))
    rel=${f#"$BENCH_DIR/"}
    size=$(stat -f '%z' "$f" 2>/dev/null || stat -c '%s' "$f" 2>/dev/null || echo 0)

    start=$(python3 -c "import time; print(int(time.time()*1e9))")
    output=$(env "$DISABLE_ENV=1" timeout "$OUTER_TIMEOUT" "$BINARY" aiger "$f" --engine sat --timeout "$INNER_TIMEOUT" 2>/dev/null)
    exit_code=$?
    end=$(python3 -c "import time; print(int(time.time()*1e9))")
    elapsed=$(python3 -c "print(f'{($end - $start) / 1e9:.3f}')" 2>/dev/null || echo "?")

    result_line=$(echo "$output" | head -1 | tr -d '[:space:]')

    if [ $exit_code -eq 124 ]; then
        result_line="timeout"
        unknown=$((unknown + 1))
    elif [ -z "$result_line" ]; then
        result_line="error"
        errors=$((errors + 1))
    elif [ "$result_line" = "sat" ]; then
        sat=$((sat + 1))
    elif [ "$result_line" = "unsat" ]; then
        unsat=$((unsat + 1))
    elif [ "$result_line" = "unknown" ]; then
        unknown=$((unknown + 1))
    else
        errors=$((errors + 1))
    fi

    echo "$rel,$result_line,$elapsed,$size" >> "$RESULTS_CSV"
    echo "[$total/${#files[@]}] $rel = $result_line (${elapsed}s, ${size}B)"
done

echo ""
echo "=== #4233 UNKNOWN-FALLBACK-DISABLED TOP-50 RESULTS ==="
echo "Total:   $total"
echo "SAT:     $sat"
echo "UNSAT:   $unsat"
echo "UNKNOWN: $unknown"
echo "ERROR:   $errors"
echo "SOLVED:  $((sat + unsat)) / $total"
echo ""
echo "Results saved to: $RESULTS_CSV"
