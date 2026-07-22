#!/usr/bin/env bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# Thin compatibility wrapper for the API consumer compatibility canary gate.
#
# Source of truth:
#   ty canary-gate --kind api
#
# Usage:
#   scripts/check_api_canary_gate.sh [--mode warn|enforce] [--verbose]
#   scripts/check_api_canary_gate.sh --mode enforce  # blocking gate
#
# The wrapper defaults to enforce mode to preserve the historical script
# behavior; pass --mode warn for advisory local runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/canary_gate_common.sh"

TARGET_DIR="$(resolve_canary_target_dir "$REPO_ROOT")"
MODE_ARGS=(--mode enforce)
for arg in "$@"; do
    if [[ "$arg" == "--mode" || "$arg" == --mode=* ]]; then
        MODE_ARGS=()
        break
    fi
done

(
    cd "$REPO_ROOT"
    CARGO_TARGET_DIR="$TARGET_DIR" cargo run --profile release-canary --bin ty -- \
        canary-gate --kind api "${MODE_ARGS[@]}" "$@"
)
