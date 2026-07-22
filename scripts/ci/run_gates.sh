#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# run_gates.sh — the aggregate CI gate runner.
#
# The repo has many individual gate scripts (scripts/check_*_gate.sh, the
# soundness watchdog, spec_regression) but, as ci-soundness-watchdog.sh notes,
# nothing invoked them all — there was no single "run everything" entrypoint.
# This IS that entrypoint: one command that runs the fast, self-contained gates
# in order, captures each result, and exits non-zero if ANY gate fails.
#
# It is the pushable core of the CI runner: a `.github/workflows/*.yml` (blocked
# on a PAT lacking `workflow` scope) would just be `run: scripts/ci/run_gates.sh`.
# Until that lands, a developer runs this before pushing.
#
# Usage:
#   scripts/ci/run_gates.sh              # fast gates (build + quality + canaries)
#   scripts/ci/run_gates.sh --full       # also run the workspace test suite
#
# Memory posture: this host is memory-tight, so builds/tests are pinned to 2
# jobs / 2 test threads (matching CLAUDE.md), and tla-petri — whose tests spawn
# parallel explorers — is run single-process to avoid the concurrent-explorer OOM.

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 2
REPO_ROOT="$(pwd)"

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export RUST_TEST_THREADS="${RUST_TEST_THREADS:-2}"

FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

# --- Gate bookkeeping --------------------------------------------------------
declare -a GATE_NAMES=()
declare -a GATE_STATUS=()
FAILED=0

run_gate() {
  local name="$1"; shift
  echo "──────────────────────────────────────────────────────────────────────"
  echo "▶ gate: $name"
  echo "  \$ $*"
  local start; start=$SECONDS
  if "$@"; then
    local dur=$((SECONDS - start))
    echo "✔ $name (${dur}s)"
    GATE_NAMES+=("$name"); GATE_STATUS+=("PASS")
  else
    local rc=$? dur=$((SECONDS - start))
    echo "✗ $name FAILED (rc=$rc, ${dur}s)"
    GATE_NAMES+=("$name"); GATE_STATUS+=("FAIL")
    FAILED=1
  fi
}

# A gate script is only run if it exists + is executable; otherwise it is
# recorded as SKIP (missing) rather than silently ignored.
run_gate_script() {
  local name="$1" script="$2"
  if [[ -x "$script" ]]; then
    run_gate "$name" "$script"
  elif [[ -f "$script" ]]; then
    run_gate "$name" bash "$script"
  else
    echo "▶ gate: $name — SKIP (missing $script)"
    GATE_NAMES+=("$name"); GATE_STATUS+=("SKIP")
  fi
}

# --- 1. Workspace build (the precondition for every other gate) --------------
run_gate "build:workspace" cargo build --workspace --quiet

# --- 2. Self-contained gate scripts -----------------------------------------
run_gate_script "quality:code"        "$REPO_ROOT/scripts/check_code_quality_gate.sh"
run_gate_script "api:canary"          "$REPO_ROOT/scripts/check_api_canary_gate.sh"
run_gate_script "verification:gates"  "$REPO_ROOT/scripts/check_verification_gates.sh"
run_gate_script "silent-error:coerce" "$REPO_ROOT/scripts/check_silent_error_coercion.sh"
run_gate_script "file-size:regress"   "$REPO_ROOT/scripts/check_file_size_regressions.sh"
run_gate_script "span:sync"           "$REPO_ROOT/scripts/ci/span_sync_check.sh"

# --- 3. Test suite (opt-in via --full; heavy on this host) -------------------
if [[ "$FULL" == 1 ]]; then
  # tla-petri first, single-process (parallel explorers OOM if concurrent).
  run_gate "test:tla-petri" cargo test -p tla-petri --lib -- --test-threads=1
  # The rest of the workspace under the pinned thread budget.
  run_gate "test:workspace" cargo test --workspace --exclude tla-petri
fi

# --- Summary -----------------------------------------------------------------
echo "══════════════════════════════════════════════════════════════════════"
echo "gate summary:"
for i in "${!GATE_NAMES[@]}"; do
  printf "  %-6s %s\n" "${GATE_STATUS[$i]}" "${GATE_NAMES[$i]}"
done
if [[ "$FAILED" == 1 ]]; then
  echo "RESULT: FAIL — one or more gates failed."
  exit 1
fi
echo "RESULT: PASS — all run gates green."
exit 0
