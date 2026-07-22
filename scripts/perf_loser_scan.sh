#!/bin/bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0
#
# perf_loser_scan.sh — ty-vs-TLC timing scan that ALSO records the ty execution
# tier (interpreter / per-action callout / native-fused) via TY_ENGINE_TIER.
#
# Purpose (Part of #5): find specs where ty is SLOWER than single-thread TLC
# while stuck on the per-action callout tier (the tier that does NOT beat TLC),
# so we can identify the dominant "perf_loser" layout class to widen
# native-fused admission for.
#
# Unlike mass_tlc_compare.sh this does NOT strip TY_ENGINE_TIER (it only strips
# the other TY_* levers for fairness), so the tier label is captured.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TY="${TY:-$REPO_ROOT/target/release/ty}"
TLC_JAR="${TLC_JAR:-$HOME/tlaplus/tytools.jar}"
JAVA="${JAVA:-java}"
command -v "$JAVA" >/dev/null 2>&1 && JAVA="$(command -v "$JAVA")"
EXAMPLES_DIR="${EXAMPLES_DIR:-${TLAPLUS_EXAMPLES:-$HOME/tlaplus-examples/specifications}}"
TIMEOUT="${TIMEOUT:-25}"
TIMEOUT_BIN="${TIMEOUT_BIN:-$(command -v gtimeout || command -v timeout)}"
OUT_CSV="${OUT_CSV:-$REPO_ROOT/reports/perf_loser_scan.csv}"

mkdir -p "$(dirname "$OUT_CSV")"

# Strip TY_* levers for fairness EXCEPT TY_ENGINE_TIER (we want the tier line).
TY_FAIR=(env)
while IFS='=' read -r name _; do
    case "$name" in
        TY_ENGINE_TIER) ;;                       # keep
        TY_*) TY_FAIR+=( -u "$name" ) ;;
    esac
done < <(env)

TLC_ENV=(env -u JAVA_TOOL_OPTIONS -u JDK_JAVA_OPTIONS -u _JAVA_OPTIONS)
TLC_ARGS=(-XX:ActiveProcessorCount=1 -XX:+UseSerialGC -Xms64m -Xmx4g)

echo "spec,cfg,ty_states,tlc_states,ty_time,tlc_time,ratio,tier,parity,note" > "$OUT_CSV"

run_one() {
    local tla="$1" cfg="$2"
    local name; name="$(basename "$tla" .tla)"
    local rel="${tla#"$EXAMPLES_DIR"/}"

    # --- ty ---
    local ty_start ty_end ty_out ty_rc
    ty_start=$(date +%s.%N)
    ty_out=$(TY_ENGINE_TIER=1 "$TIMEOUT_BIN" "${TIMEOUT}s" "${TY_FAIR[@]}" TY_ENGINE_TIER=1 \
        "$TY" check "$tla" --config "$cfg" --workers 1 --force 2>&1) && ty_rc=0 || ty_rc=$?
    ty_end=$(date +%s.%N)

    local note="" tier="-" ty_states="-" ty_time="-"
    if [ "$ty_rc" -eq 124 ]; then note="ty_timeout";
    elif [ "$ty_rc" -ne 0 ]; then note="ty_exit_$ty_rc";
    else
        ty_states=$(echo "$ty_out" | grep -oE "States found: [0-9,]+" | tr -d ',' | grep -oE "[0-9]+" | head -1)
        ty_time=$(echo "$ty_end - $ty_start" | bc)
    fi
    tier=$(echo "$ty_out" | grep -oE "execution tier: .*" | sed 's/execution tier: //' | head -1)
    [ -z "$tier" ] && tier="-"

    # --- TLC (only if ty produced states) ---
    local tlc_states="-" tlc_time="-" ratio="-"
    if [ "$ty_states" != "-" ] && [ -n "$ty_states" ]; then
        local meta; meta=$(mktemp -d)
        local tlc_start tlc_end tlc_out tlc_rc
        tlc_start=$(date +%s.%N)
        tlc_out=$("$TIMEOUT_BIN" "${TIMEOUT}s" "${TLC_ENV[@]}" "$JAVA" "${TLC_ARGS[@]}" -cp "$TLC_JAR" tlc2.TLC \
            -metadir "$meta" -teSpecOutDir "$meta" -deadlock -config "$cfg" -workers 1 "$tla" 2>&1) && tlc_rc=0 || tlc_rc=$?
        tlc_end=$(date +%s.%N)
        rm -rf "$meta"
        if [ "$tlc_rc" -eq 124 ]; then note="${note:+$note;}tlc_timeout";
        else
            tlc_states=$(echo "$tlc_out" | grep -oE "[0-9,]+ distinct states found" | tail -1 | tr -d ',' | grep -oE "^[0-9]+")
            [ -z "$tlc_states" ] && tlc_states="-"
            tlc_time=$(echo "$tlc_end - $tlc_start" | bc)
        fi
    fi

    local parity="-"
    if [ "$ty_states" != "-" ] && [ "$tlc_states" != "-" ] && [ -n "$tlc_states" ]; then
        [ "$ty_states" = "$tlc_states" ] && parity="YES" || parity="NO"
    fi
    if [ "$ty_time" != "-" ] && [ "$tlc_time" != "-" ]; then
        ratio=$(echo "scale=2; $ty_time / $tlc_time" | bc)
    fi

    echo "$name,$rel,$ty_states,$tlc_states,$ty_time,$tlc_time,$ratio,$tier,$parity,$note" >> "$OUT_CSV"
    printf '%-40s ty=%-7s tlc=%-7s ratio=%-6s %-34s %s\n' "$name" "$ty_states" "$tlc_states" "$ratio" "$tier" "$note"
}

# Spec list: passed as args (cfg paths), else a curated small/fast set.
if [ "$#" -gt 0 ]; then
    for cfg in "$@"; do
        tla="${cfg%.cfg}.tla"
        [ -f "$tla" ] && run_one "$tla" "$cfg"
    done
else
    while IFS= read -r cfg; do
        tla="${cfg%.cfg}.tla"
        [ -f "$tla" ] && run_one "$tla" "$cfg"
    done < <(find "$EXAMPLES_DIR" -name "*.cfg" -type f | sort)
fi

echo ""
echo "Wrote $OUT_CSV"
