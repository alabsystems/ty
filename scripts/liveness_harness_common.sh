#!/bin/bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# Shared verdict classification for liveness harness scripts.

classify_tlc_status() {
    local output="${1:-}"
    local rc="${2:-0}"

    if [ "$rc" -eq 124 ] || echo "$output" | grep -q "\[TIMEOUT\]"; then
        echo "timeout"
        return
    fi

    if echo "$output" | grep -qi "Invariant[[:space:]].*violated"; then
        echo "invariant"
    elif echo "$output" | grep -qi "Deadlock reached"; then
        echo "deadlock"
    elif echo "$output" | grep -qi "Temporal properties were violated\|Error:[[:space:]].*liveness\|Error:[[:space:]].*temporal"; then
        echo "liveness"
    elif [ "$rc" -ne 0 ] || echo "$output" | grep -qi "Error:\|Exception\|FAILED"; then
        echo "error"
    elif echo "$output" | grep -qi "No error has been found"; then
        echo "success"
    else
        echo "unknown"
    fi
}

classify_ty_status() {
    local output="${1:-}"
    local rc="${2:-0}"

    if [ "$rc" -eq 124 ] || echo "$output" | grep -q "\[TIMEOUT\]"; then
        echo "timeout"
        return
    fi

    if echo "$output" | grep -qi "Error:[[:space:]]*Invariant"; then
        echo "invariant"
    elif echo "$output" | grep -qi "Error:[[:space:]]*Deadlock"; then
        echo "deadlock"
    elif echo "$output" | grep -qi "Temporal properties were violated\|Error:[[:space:]].*liveness\|Error:[[:space:]].*temporal"; then
        echo "liveness"
    elif [ "$rc" -ne 0 ] || echo "$output" | grep -qi "Error:\|panic\|exceeded\|overflow"; then
        echo "error"
    elif echo "$output" | grep -q "No errors found"; then
        echo "success"
    else
        echo "unknown"
    fi
}

state_match_flag() {
    local tlc_states="${1:-}"
    local ty_states="${2:-}"

    if [ -z "$tlc_states" ] || [ -z "$ty_states" ]; then
        echo "unknown"
    elif [ "$tlc_states" = "$ty_states" ]; then
        echo "yes"
    else
        echo "no"
    fi
}

evaluate_liveness_gate() {
    local tlc_status="${1:-unknown}"
    local ty_status="${2:-unknown}"
    local tlc_states="${3:-}"
    local ty_states="${4:-}"

    local verdict_match="no"
    if [ "$tlc_status" = "$ty_status" ]; then
        verdict_match="yes"
    fi

    local state_match
    state_match="$(state_match_flag "$tlc_states" "$ty_states")"

    local overall=""
    if [ "$tlc_status" = "timeout" ] || [ "$tlc_status" = "unknown" ]; then
        overall="SKIP"
    elif [ "$ty_status" = "timeout" ] || [ "$ty_status" = "unknown" ]; then
        overall="ERROR"
    elif [ "$verdict_match" != "yes" ]; then
        overall="FAIL"
    elif [ "$tlc_status" = "success" ]; then
        if [ "$state_match" = "yes" ]; then
            overall="PASS"
        else
            overall="FAIL"
        fi
    else
        overall="PASS"
    fi

    printf 'overall=%s|verdict_match=%s|state_match=%s\n' "$overall" "$verdict_match" "$state_match"
}
