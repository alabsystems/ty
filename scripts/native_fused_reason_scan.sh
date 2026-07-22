#!/bin/bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0
#
# native_fused_reason_scan.sh — for each spec, run ty just long enough to learn
# WHICH execution tier it lands on and WHY native-fused was (or was not) used.
#
# Part of #5: categorize perf_losers by their native-fused blocker so we can tell
# admission-gate rejections (fixable in tla-check) apart from trust-cg codegen
# failures (fixable only in the sibling trust-cg repo) and cost-threshold routing.
#
# Uses a SHORT timeout (default 6s) and caps address space so big specs cannot
# OOM the machine — we only need the setup-phase tier decision, not full
# exploration.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TY="${TY:-$REPO_ROOT/target/release/ty}"
EXAMPLES_DIR="${EXAMPLES_DIR:-${TLAPLUS_EXAMPLES:-$HOME/tlaplus-examples/specifications}}"
TIMEOUT="${TIMEOUT:-6}"
TIMEOUT_BIN="${TIMEOUT_BIN:-$(command -v gtimeout || command -v timeout)}"
OUT="${OUT:-$REPO_ROOT/reports/native_fused_reason.csv}"
# Address-space cap (KB) so a runaway interpreter run cannot eat all RAM.
VMEM_KB="${VMEM_KB:-8000000}"   # ~8 GB

mkdir -p "$(dirname "$OUT")"
echo "spec,cfg,flat_primary,constraints,tier,category,detail" > "$OUT"

classify() {
    local tla="$1" cfg="$2"
    local name; name="$(basename "$tla" .tla)"
    local rel="${cfg#"$EXAMPLES_DIR"/}"

    local out
    out=$( (ulimit -v "$VMEM_KB" 2>/dev/null; \
            TY_ENGINE_TIER=1 "$TIMEOUT_BIN" "${TIMEOUT}s" \
            "$TY" check "$tla" --config "$cfg" --workers 1 --force) 2>&1 )

    local tier
    tier=$(echo "$out" | grep -oE "execution tier: .*" | sed 's/.*execution tier: //' | head -1)
    [ -z "$tier" ] && tier="(none/timeout-before-tier)"

    local flat_primary="?" constraints="?"
    echo "$out" | grep -q "flat_state_primary=true" && flat_primary="yes"
    echo "$out" | grep -q "flat_state_primary=false" && flat_primary="no"

    # category + detail: first matching native-fused blocker reason
    local category="other" detail=""
    if echo "$out" | grep -qi "code generation failed\|unsupported trust_ir\|trust-codegen adapter failed"; then
        category="codegen_fail"
        detail=$(echo "$out" | grep -oiE "unsupported trust_ir instruction: [^;]*" | head -1)
        [ -z "$detail" ] && detail=$(echo "$out" | grep -i "code generation failed" | head -1 | cut -c1-120)
    elif echo "$out" | grep -qi "strict native-fused mode is disabled"; then
        category="needs_strict"
    elif echo "$out" | grep -qi "blocked until #4433\|parent-loop successor parity"; then
        category="blocked_4433_nonprimary"
    elif echo "$out" | grep -qi "native fused flat frontier\|flat layout is not admitted\|not admitted for strict"; then
        category="admission_reject"
        detail=$(echo "$out" | grep -oiE "rejected: [^]]*" | head -1 | cut -c1-100)
    elif [ "$tier" = "trust-cg native-fused (compiled BFS)" ]; then
        category="native_fused_OK"
    elif [ "$tier" = "interpreter" ]; then
        category="interpreter"
    elif echo "$tier" | grep -qi "per-action callout"; then
        category="per_action_callout"
    fi

    detail=$(echo "$detail" | tr ',' ';' | tr -d '\n')
    echo "$name,$rel,$flat_primary,$constraints,$tier,$category,$detail" >> "$OUT"
    printf '%-34s flat=%-3s %-34s %-26s %s\n' "$name" "$flat_primary" "$tier" "$category" "$detail"
}

if [ "$#" -gt 0 ]; then
    for cfg in "$@"; do
        tla="${cfg%.cfg}.tla"; [ -f "$tla" ] && classify "$tla" "$cfg"
    done
else
    while IFS= read -r cfg; do
        tla="${cfg%.cfg}.tla"; [ -f "$tla" ] && classify "$tla" "$cfg"
    done < <(find "$EXAMPLES_DIR" -name "*.cfg" -type f | sort)
fi
echo ""; echo "Wrote $OUT"
