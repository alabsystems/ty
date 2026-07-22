#!/usr/bin/env bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# Thin compatibility wrapper for the silent eval-error coercion gate.
#
# Source of truth:
#   ty canary-gate --kind silent-error
#
# Usage:
#   scripts/check_silent_error_coercion.sh [--mode warn|enforce]
#   scripts/check_silent_error_coercion.sh --mode enforce  # blocking gate
#   scripts/check_silent_error_coercion.sh                  # default: warn mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/canary_gate_common.sh"

TARGET_DIR="$(resolve_canary_target_dir "$REPO_ROOT")"

(
    cd "$REPO_ROOT"
    CARGO_TARGET_DIR="$TARGET_DIR" cargo run --profile release-canary --bin ty -- \
        canary-gate --kind silent-error "$@"
)
