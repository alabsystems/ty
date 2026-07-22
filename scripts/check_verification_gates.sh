#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# check_verification_gates.sh — one-command pre-push gate for TY's VERIFICATION
# surface (the part this repo's correctness rests on).
#
# There is no automated CI runner in this repo (the build needs sibling-repo
# provisioning: `../trust-cg` / `../trust-ir` path patches and the `ay` git dep),
# so this is the manual equivalent — run it before pushing changes that touch the
# checker, the symbolic engines, or cfg(feature="ay") code. Each gate prints
# PASS/FAIL; the exit code is the number of failing gates (0 = all green).
#
# Scope is deliberately the VERIFICATION-CORRECTNESS surface, NOT the benchmark/
# governance/canary tooling (cmd_supremacy, cmd_canary_gate, cmd_rust_function_span_scan),
# which carries pre-existing content/baseline drift unrelated to verification — see
# docs/trust-hardening-report-2026-06-23.md.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FAIL=0
gate() {
    local label="$1"; shift
    echo "=== $label ==="
    if "$@" >/dev/null 2>&1; then
        echo "PASS: $label"
    else
        echo "FAIL: $label  (re-run to see output: $*)"
        FAIL=$((FAIL + 1))
    fi
}

# 0. Vendored Git test dependencies are immutable and Cargo.lock resolves the
#    same full commits. This stays fast enough to catch drift before builds.
gate "reproducible Git dependency pins" \
    python3 "$SCRIPT_DIR/check_reproducible_git_pins.py"
gate "reproducible Git dependency pin regressions" \
    python3 "$SCRIPT_DIR/test_check_reproducible_git_pins.py"

# 1. Both build configurations compile (the all-verification build is not in CI;
#    a clippy/auto-fix sweep already broke it once — d4ae86e1).
gate "build gate (default + --features ay + workspace)" bash "$SCRIPT_DIR/check_ay_build_gate.sh"

# 2. The model checker + symbolic engines + certifying-verification test suite.
#    This is the authority for soundness/termination (4395 tests at time of writing).
gate "tla-check verification tests (--features ay)" \
    cargo test -q -p tla-check --features ay --lib

# 2b. The clean-cic (Clean CIC kernel) build — the `Certified` trust-base tier. Unguarded
#     until now; it depends on an out-of-workspace path (../../../clean) and can silently
#     rot the same way the ay build did (see "ay verification build not in CI").
gate "clean-cic build (--features clean-cic + ay,clean-cic)" \
    bash -c 'cargo build -q -p tla-check --features clean-cic && cargo build -q -p tla-check --features ay,clean-cic'

# 2c. The seven `testing`-gated parity/soundness integration tests. They require
#     --features testing, which NO default `cargo test` enables, so they were never run.
gate "tla-check testing-gated parity/soundness tests (--features ay,testing)" \
    cargo test -q -p tla-check --features ay,testing \
        --test parallel_state_count_parity --test parallel_tlcget_level_context \
        --test liveness_disk_backend_parity --test liveness_disk_bitmask_backend \
        --test liveness_disk_successor_backend --test trust_cg_state_graph_parity \
        --test choose_soundness

# 2d. The compiled-BFS / JIT hot-loop static source guard (integration test, so the `--lib`
#     gate above skips it — it had silently rotted on a directory-module refactor).
gate "compiled-BFS hot-loop static guard" \
    cargo test -q -p tla-check --test compiled_bfs_hot_loop_static_guard

# 3. The cross-repo dependency drift guard (ay / trust-ir / trust-cg pin agreement).
gate "cross-repo dep drift guard (unit tests)" \
    cargo test -q -p tla-petri --lib drift_guard

# 3b. The REAL cross-repo scan: runs the ty-mcc-drift-guard binary over the
#     live sibling workspaces' `cargo metadata`. The --lib gate above only
#     exercises the guard's synthetic fixtures and stays green while actual
#     pin drift exists between ~/root/{ty,trust-ir,trust-cg,ay}; this one
#     fails on live drift (it self-skips when no sibling repos are present).
gate "cross-repo dep drift guard (live sibling scan)" \
    cargo test -q -p tla-petri --test cargo_dep_drift_guard

echo "---"
if [ "$FAIL" -eq 0 ]; then
    echo "check_verification_gates: clean (all verification gates pass)."
else
    echo "check_verification_gates: $FAIL gate(s) FAILED."
fi
exit "$FAIL"
