#!/bin/bash
# Copyright 2026 Andrew Yates. All rights reserved.
# Licensed under Apache 2.0.
#
# test_safe.sh - Run cargo test with OOM protection via lock serialization
#
# Usage:
#   ./scripts/test_safe.sh                       # Full suite (serialized)
#   ./scripts/test_safe.sh -p tla-check          # Package tests (serialized)
#   ./scripts/test_safe.sh -p tla-check test_name  # Specific test (serialized)
#   ./scripts/test_safe.sh --status              # Check lock status
#
# ALL cargo test invocations are serialized because even targeted tests
# trigger cargo compilation, and concurrent compilations cause OOM.
#
# Works on both Linux (flock) and macOS (mkdir-based lock).
#
# Related: #331

set -eo pipefail

LOCK_DIR="/tmp/ty-test.lock"
LOCK_TIMEOUT=1800  # 30 minutes max wait
STUCK_THRESHOLD=600  # 10 minutes = potentially stuck

# Show lock status
show_status() {
    if [[ ! -d "$LOCK_DIR" ]]; then
        echo "[test_safe] Lock status: FREE"
        echo "[test_safe] No test currently running"
        return 0
    fi

    local holder_pid=""
    local holder_role=""
    local holder_cmd=""
    local lock_age=0

    if [[ -f "$LOCK_DIR/pid" ]]; then
        holder_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null)
    fi
    if [[ -f "$LOCK_DIR/role" ]]; then
        holder_role=$(cat "$LOCK_DIR/role" 2>/dev/null)
    fi
    if [[ -f "$LOCK_DIR/cmd" ]]; then
        holder_cmd=$(cat "$LOCK_DIR/cmd" 2>/dev/null)
    fi

    # Calculate lock age
    if [[ -f "$LOCK_DIR/pid" ]]; then
        local lock_time=$(stat -f %m "$LOCK_DIR/pid" 2>/dev/null || stat -c %Y "$LOCK_DIR/pid" 2>/dev/null)
        local now=$(date +%s)
        lock_age=$((now - lock_time))
    fi

    echo "[test_safe] Lock status: HELD"
    echo "[test_safe] Holder PID: ${holder_pid:-unknown}"
    echo "[test_safe] Holder role: ${holder_role:-unknown}"
    echo "[test_safe] Command: ${holder_cmd:-unknown}"
    echo "[test_safe] Duration: ${lock_age}s ($(( lock_age / 60 ))m)"

    # Check if holder is still alive
    if [[ -n "$holder_pid" ]]; then
        if kill -0 "$holder_pid" 2>/dev/null; then
            echo "[test_safe] Process: RUNNING"
        else
            echo "[test_safe] Process: DEAD (stale lock)"
            echo "[test_safe] Action: Lock will be auto-cleared on next test run"
            return 1
        fi
    fi

    # Check if potentially stuck
    if [[ $lock_age -ge $STUCK_THRESHOLD ]]; then
        echo ""
        echo "[test_safe] WARNING: Test running for ${lock_age}s (>10min)"
        echo "[test_safe] This may indicate a stuck test or infinite loop"
        echo "[test_safe] To investigate: ps aux | grep $holder_pid"
        echo "[test_safe] To force-clear: rm -rf $LOCK_DIR"
        return 2
    fi

    return 0
}

# Handle --status flag
if [[ "$1" == "--status" ]]; then
    show_status
    exit $?
fi

# Cross-platform lock acquisition using mkdir (atomic operation)
acquire_lock() {
    local start_time=$(date +%s)
    local pid=$$

    while true; do
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            # Got the lock, write metadata for status reporting
            echo $pid > "$LOCK_DIR/pid"
            echo "ty" > "$LOCK_DIR/role"
            echo "cargo test $*" > "$LOCK_DIR/cmd"
            return 0
        fi

        # Check if lock is stale (holder process died)
        if [[ -f "$LOCK_DIR/pid" ]]; then
            local holder_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null)
            if [[ -n "$holder_pid" ]] && ! kill -0 "$holder_pid" 2>/dev/null; then
                echo "[test_safe] Removing stale lock from dead process $holder_pid"
                rm -rf "$LOCK_DIR"
                continue
            fi
        fi

        # Check timeout
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        if [[ $elapsed -ge $LOCK_TIMEOUT ]]; then
            echo "[test_safe] ERROR: Lock timeout after ${LOCK_TIMEOUT}s"
            return 1
        fi

        echo "[test_safe] Waiting for lock (held by PID $(cat "$LOCK_DIR/pid" 2>/dev/null || echo '?'))..."
        sleep 5
    done
}

# Release lock
release_lock() {
    rm -rf "$LOCK_DIR"
}

# Cleanup on exit
cleanup() {
    release_lock
}

# Main - ALL tests are serialized to prevent OOM from concurrent compilations
echo "[test_safe] Acquiring lock for cargo test..."
if acquire_lock "$@"; then
    trap cleanup EXIT
    echo "[test_safe] Lock acquired, running: cargo test $*"
    # Set flag to prevent cargo wrapper from intercepting recursively
    export TY_TEST_SAFE_ACTIVE=1
    cargo test "$@"
else
    exit 1
fi
