#!/bin/bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# PERFORMANCE BENCHMARK SCRIPT
# Run this BEFORE and AFTER performance changes
# Records timing for comparison

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
if [[ "$TARGET_DIR" != /* ]]; then
    TARGET_DIR="$REPO_ROOT/$TARGET_DIR"
fi
TY_FAIR_ENV=(env)
while IFS='=' read -r name _; do
    if [[ "$name" == TY_* ]]; then
        TY_FAIR_ENV+=( -u "$name" )
    fi
done < <(env)

echo "=== TY Performance Benchmark ==="
echo "Date: $(date)"
echo "Git: $(git rev-parse --short HEAD)"
echo ""

# Build the maximum-runtime-perf binary (full/fat LTO). The default `release`
# profile is now thin LTO for fast builds; `release-perf` carries the fat LTO this
# benchmark measures. See Cargo.toml `[profile.release-perf]`.
echo "Building release-perf binary..."
cargo build --profile release-perf -p tla-cli 2>/dev/null

TY="$TARGET_DIR/release-perf/tla"

run_benchmark() {
    local name="$1"
    local spec="$2"
    local config="${3:-}"

    if [ ! -f "$spec" ]; then
        echo "| $name | SKIP | - | - |"
        return
    fi

    # Run with timing
    start=$(python3 -c 'import time; print(time.time())')
    if [ -n "$config" ] && [ -f "$config" ]; then
        output=$("${TY_FAIR_ENV[@]}" "$TY" check "$spec" --config "$config" --workers 1 --force 2>&1) || true
    else
        output=$("${TY_FAIR_ENV[@]}" "$TY" check "$spec" --workers 1 --force 2>&1) || true
    fi
    end=$(python3 -c 'import time; print(time.time())')

    # Calculate time
    time=$(python3 -c "print(f'{$end - $start:.3f}')")

    # Extract state count
    states=$(echo "$output" | grep -oE "States found: [0-9,]+" | tr -d ',' | grep -oE "[0-9]+" || echo "0")

    # Calculate rate
    if [ "$states" != "0" ]; then
        rate=$(python3 -c "print(int($states / ($end - $start)))")
    else
        rate="N/A"
    fi

    echo "| $name | $states | ${time}s | $rate |"
}

echo "| Spec | States | Time | States/sec |"
echo "|------|--------|------|------------|"

run_benchmark "DieHard" "$HOME/tlaplus-examples/specifications/DieHard/DieHard.tla"
run_benchmark "DiningPhilosophers" "$HOME/tlaplus-examples/specifications/DiningPhilosophers/DiningPhilosophers.tla"
run_benchmark "bcastFolklore" "$HOME/tlaplus-examples/specifications/bcastFolklore/bcastFolklore.tla" "$HOME/tlaplus-examples/specifications/bcastFolklore/bcastFolklore.cfg"

# Issue #284 benchmark: Disruptor_SPMC - key spec for LET caching performance
# Baseline: TY=190s, TLC=1.2s (160x gap)
# Target after fix: <20s (10x improvement minimum)
run_benchmark "Disruptor_SPMC (#284)" "examples/test/disruptor/Disruptor_SPMC.tla" "examples/test/disruptor/Disruptor_SPMC.cfg"

echo ""
echo "Benchmark complete. Compare with previous runs to verify improvement."
