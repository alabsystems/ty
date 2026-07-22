#!/usr/bin/env bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# Thin compatibility wrapper for the enumerate canary gate.
#
# Source of truth:
#   ty canary-gate --kind enumerate
#
# Usage:
#   scripts/check_enumerate_canaries.sh [--mode warn|enforce] [--staged] [--changed-files FILE...]
#   scripts/check_enumerate_canaries.sh --mode enforce  # blocking gate
#   scripts/check_enumerate_canaries.sh                  # default: warn mode
#
# In pre-commit context, pass --staged and let Rust read git diff --cached.
# Without --changed-files, the Rust gate checks git diff --name-only HEAD.
#
# Environment:
#   TY_ENUMERATE_CANARY_SKIP=1   Skip the Rust gate after the CLI launches
#   TY_ENUMERATE_CANARY_WARN=1   Downgrade failures to advisory warnings
#   TY_ENUMERATE_CANARY_TIMEOUT  Per-spec timeout in seconds (default: 30)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/canary_gate_common.sh"

TARGET_DIR="$(resolve_canary_target_dir "$REPO_ROOT")"

(
    cd "$REPO_ROOT"
    CARGO_TARGET_DIR="$TARGET_DIR" cargo run --profile release-canary --bin ty -- \
        canary-gate --kind enumerate "$@"
)
