#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# check_perf_anchor_gate.sh — fast pre-push anchor gate for the TY-vs-TLC
# comparison INSTRUMENT (plan task 7.3; complements check_verification_gates.sh,
# which deliberately scopes benchmark tooling out).
#
# Default mode runs `ty supremacy compare` on three tiny verified-match anchors
# at PARITY policy with paired repetitions — it catches instrument breakage,
# engine-provenance loss, corpus/TLC env rot, and verdict/count drift in about
# a minute. It does NOT make performance claims: strict both-axis gating needs
# an idle box and belongs to the burndown's enforced collection, not a pre-push
# hook. `--strict` opts into parity-and-speed-and-memory with the burndown's
# noise margins — only meaningful on an idle machine (the script warns).
#
# Prereqs: corpus (`ty corpus verify`), TLC jar at ~/tlaplus/tytools.jar, a
# modern JDK on PATH (Temurin 21 per docs/perf plan §2), release ty binary.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TY="${TY:-$REPO_ROOT/target/release/ty}"
# Tiny, verified-match, check-mode baseline rows (same trio as
# `ty supremacy reproduce`): DiningPhilosophers 67 states, EWD840 302 states,
# Prisoners 214 states.
ANCHORS=(DiningPhilosophers EWD840 Prisoners)
RUNS="${RUNS:-2}"
MODE=parity
if [ "${1:-}" = "--strict" ]; then
    MODE=strict
    echo "WARNING: --strict gates speed+memory with noise margins; results are" >&2
    echo "meaningless unless this box is otherwise idle (plan §4.7)." >&2
fi

if [ ! -x "$TY" ]; then
    echo "FATAL: release binary missing at $TY (cargo build -p tla-cli --release)" >&2
    exit 2
fi
"$TY" corpus verify >/dev/null 2>&1 || {
    echo "FATAL: corpus not present (run: ty corpus fetch)" >&2
    exit 2
}

OUT="${OUT:-$REPO_ROOT/reports/perf/anchor-gate-$(date -u +%Y%m%dT%H%M%SZ)}"
if [ "$MODE" = parity ]; then
    POLICY_ARGS=(--policy parity)
else
    POLICY_ARGS=(--policy parity-and-speed-and-memory --min-speedup 1.05 --max-memory-ratio 0.95)
fi

# Sound-baseline arm: interpreter backend (compare's default) is the oracle
# path; provenance lands on every row (engine_tier), proving what ran.
"$TY" supremacy compare \
    --spec "${ANCHORS[@]}" \
    --backend interpreter \
    --runs "$RUNS" \
    "${POLICY_ARGS[@]}" \
    --output-dir "$OUT"
rc=$?
echo "---"
if [ "$rc" -eq 0 ]; then
    echo "check_perf_anchor_gate: clean ($MODE mode, ${#ANCHORS[@]} anchors, runs=$RUNS; artifacts: $OUT)"
else
    echo "check_perf_anchor_gate: FAILED (see $OUT/compare.json)"
fi
exit "$rc"
