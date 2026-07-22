#!/usr/bin/env bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# determinism_gate.sh - run-to-run determinism gate (the "TwoPhase x3" gate).
#
# Runs one spec RUNS times (default 3) under identical, TY_*-scrubbed
# conditions and FAILS unless every run reports the same (distinct-state
# count, verdict) pair. This codifies the validation gate run after dep
# bumps / upstream alignment: TY exploration must be run-to-run deterministic.
#
# ay determinism: `ty check` can hand subproblems to the ay solver stack
# (e.g. SMT Init enumeration in InitMode::Auto), and ay-sat schedules its
# inprocessing with wall-clock deadlines by default, so host-load jitter can
# leak nondeterminism into a TY run from below. This gate therefore exports
# AY_AB_DETERMINISTIC_INPROC=1, switching ay-sat to deterministic work-count
# budgets (see ay crates/ay-sat/src/determinism.rs). The knob costs solve
# performance, so it is set ONLY in determinism-validation contexts like this
# gate — never globally. "0" remains ay's kill switch.
#
# Usage:
#   scripts/determinism_gate.sh                      # TwoPhase x3
#   scripts/determinism_gate.sh Spec.tla [Spec.cfg]
#
# Environment:
#   TY_BIN            ty binary (default: target/release/tla)
#   RUNS              repetitions (default: 3)
#   TLAPLUS_EXAMPLES  corpus root (default: ~/tlaplus-examples)
#
# Exit codes: 0 deterministic, 1 nondeterministic, 2 setup error.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
if [[ "$TARGET_DIR" != /* ]]; then
    TARGET_DIR="$REPO_ROOT/$TARGET_DIR"
fi
TY_BIN="${TY_BIN:-$TARGET_DIR/release/tla}"
RUNS="${RUNS:-3}"
EXAMPLES_DIR="${TLAPLUS_EXAMPLES:-$HOME/tlaplus-examples}/specifications"

export AY_AB_DETERMINISTIC_INPROC="${AY_AB_DETERMINISTIC_INPROC:-1}"

SPEC="${1:-$EXAMPLES_DIR/transaction_commit/TwoPhase.tla}"
CONFIG="${2:-}"
if [[ -z "$CONFIG" ]]; then
    default_cfg="${SPEC%.tla}.cfg"
    if [[ -f "$default_cfg" ]]; then
        CONFIG="$default_cfg"
    fi
fi

if [[ ! -x "$TY_BIN" ]]; then
    echo "ERROR: ty binary not found/executable: $TY_BIN (build: cargo build --release -p tla-cli)" >&2
    exit 2
fi
if [[ ! -f "$SPEC" ]]; then
    echo "ERROR: spec not found: $SPEC (the default spec needs the corpus: ty corpus fetch)" >&2
    exit 2
fi
if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 2 ]]; then
    echo "ERROR: RUNS must be an integer >= 2, got: $RUNS" >&2
    exit 2
fi

# Hermetic child env: scrub ambient TY_* (same policy as the
# verify_correctness runners) so only this script's settings shape the runs.
TY_RUN_ENV=(env)
while IFS='=' read -r name _; do
    if [[ "$name" == TY_* ]]; then
        TY_RUN_ENV+=( -u "$name" )
    fi
done < <(env)

verdict_of() {
    local output="$1"
    if grep -q "Error: Invariant" <<<"$output"; then
        echo "invariant"
    elif grep -q "Error: Deadlock" <<<"$output"; then
        echo "deadlock"
    elif grep -qE "Error:.*(liveness|temporal|stuttering)" <<<"$output"; then
        echo "liveness"
    elif grep -q "Error:" <<<"$output"; then
        echo "error"
    else
        echo "ok"
    fi
}

echo "=== TY determinism gate ==="
echo "spec:   $SPEC"
echo "config: ${CONFIG:-<none>}"
echo "runs:   $RUNS  (AY_AB_DETERMINISTIC_INPROC=$AY_AB_DETERMINISTIC_INPROC)"

results=()
for ((i = 1; i <= RUNS; i++)); do
    if [[ -n "$CONFIG" ]]; then
        output="$("${TY_RUN_ENV[@]}" "$TY_BIN" check "$SPEC" --config "$CONFIG" --workers 1 --force 2>&1)" || true
    else
        output="$("${TY_RUN_ENV[@]}" "$TY_BIN" check "$SPEC" --workers 1 --force 2>&1)" || true
    fi
    states="$(grep -oE "States found: [0-9,]+" <<<"$output" | tr -d ',' | grep -oE "[0-9]+" | head -n 1 || echo "0")"
    verdict="$(verdict_of "$output")"
    results+=("${states:-0}/$verdict")
    echo "run $i: states=${states:-0} verdict=$verdict"
done

first="${results[0]}"
for r in "${results[@]}"; do
    if [[ "$r" != "$first" ]]; then
        echo "[ FAIL ] nondeterministic: runs disagree (${results[*]})"
        exit 1
    fi
done
echo "[ PASS ] $RUNS/$RUNS runs identical (states/verdict = $first)"
