#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# check_ay_build_gate.sh — assert TY compiles in BOTH feature configurations.
#
# Why this exists
# ---------------
# The `--features ay` build — the "all verification enabled" build that turns on
# the AY-backed symbolic engines (BMC/PDR/k-induction), `ty certify` /
# `certify-liveness` / `certify-all-n`, and the certifying-verification legs — is
# NOT covered by the default `cargo build` / `cargo test` (which is non-ay). So an
# ay-only break can land on main unnoticed.
#
# This is exactly what happened on 2026-06-23: a clippy "unused variable" autofix
# renamed `source` -> `_source` in cmd_certify.rs / cmd_liveness.rs considering only
# the non-ay build; the `cfg(feature = "ay")` arms still used `source`, so the
# verification build failed to compile (E0425) while the default build stayed green.
# See docs/trust-hardening-report-2026-06-23.md.
#
# Like the other gates in this directory this is a MANUAL gate: there is no
# automated CI runner in this repo. Run it before pushing changes that touch
# cfg(feature = "ay") code (or after any clippy/auto-fix sweep). Exit code is the
# number of failing build configurations (0 = all green).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FAIL=0

check_build() {
    local label="$1"; shift
    echo "=== building: $label ==="
    if cargo build "$@" >/dev/null 2>&1; then
        echo "PASS: $label"
    else
        echo "FAIL: $label — re-run without redirection to see the error:"
        echo "      cargo build $*"
        FAIL=$((FAIL + 1))
    fi
}

# The default (non-ay) build — the one the existing test suite exercises.
check_build "tla-cli (default / non-ay)"    -p tla-cli
# The verification build — gated, NOT exercised by the default suite.
check_build "tla-cli (--features ay)"       -p tla-cli --features ay
# The whole workspace with the ay verification path on.
check_build "workspace (--features tla-cli/ay)" --workspace --features tla-cli/ay

echo "---"
if [ "$FAIL" -eq 0 ]; then
    echo "check_ay_build_gate: clean (all build configurations compile)."
else
    echo "check_ay_build_gate: $FAIL build configuration(s) FAILED."
fi
exit "$FAIL"
