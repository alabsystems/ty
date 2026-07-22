#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# Part of #4372. Runs the bounded MCL-sized native fused BFS canary through the
# Rust `ty supremacy smoke` compatibility surface.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT/scripts"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-240}"
source "$SCRIPT_DIR/canary_gate_common.sh"
TARGET_DIR="$(resolve_canary_target_dir "$ROOT")"

if [[ -n "${TIMEOUT_BIN:-}" ]]; then
    timeout_bin="$TIMEOUT_BIN"
elif [[ -x /opt/homebrew/bin/timeout ]]; then
    timeout_bin=/opt/homebrew/bin/timeout
elif command -v timeout >/dev/null 2>&1; then
    timeout_bin="$(command -v timeout)"
else
    echo "error: set TIMEOUT_BIN or install timeout" >&2
    exit 2
fi

cd "$ROOT"
exec env CARGO_NET_GIT_FETCH_WITH_CLI="${CARGO_NET_GIT_FETCH_WITH_CLI:-true}" \
    CARGO_TARGET_DIR="$TARGET_DIR" \
    "$timeout_bin" "$TIMEOUT_SECONDS" \
    cargo run --profile release-canary -p tla-cli \
    --bin ty -- \
    supremacy smoke \
    --target-dir "$TARGET_DIR" \
    --cargo-profile release-canary \
    --timeout "$TIMEOUT_SECONDS" \
    --specs MCLamportMutex \
    "$@"
