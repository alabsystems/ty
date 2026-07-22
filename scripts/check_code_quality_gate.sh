#!/bin/bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# Code quality gate for Epic #1261 (Phase 2 structural cleanup)
# Supersedes #1192 gate. Run to check acceptance criteria.
#
# Machine-readable output: each check prints PASS/FAIL with offender details.
# Exit code = number of failing checks (0 = all pass).
set -euo pipefail

PASS=0
FAIL=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/canary_gate_common.sh"
CANARY_TARGET_DIR="$(resolve_canary_target_dir "$REPO_ROOT")"

run_canary_gate() {
    (
        cd "$REPO_ROOT"
        CARGO_TARGET_DIR="$CANARY_TARGET_DIR" cargo run --profile release-canary --bin ty -- \
            canary-gate "$@"
    )
}

run_rust_function_span_scan() {
    (
        cd "$REPO_ROOT"
        CARGO_TARGET_DIR="$CANARY_TARGET_DIR" cargo run --profile release-canary --bin ty -- \
            rust-function-span-scan "$@"
    )
}

run_system_health_gate() {
    (
        cd "$REPO_ROOT"
        CARGO_TARGET_DIR="$CANARY_TARGET_DIR" cargo run --profile release-canary --bin ty -- \
            system-health-gate "$@"
    )
}

# Scope: crates actively targeted by Epic #1261 structural cleanup.
# tla-ay and tla-codegen excluded per #1327 — decomposition tracked separately
# in #1328 and #1329 as post-epic follow-up.
GATE_SCOPE=(
    crates/tla-cert/src
    crates/tla-check/src
    crates/tla-cli/src
    crates/tla-core/src
    crates/tla-lsp/src
    crates/tla-prove/src
    crates/tla-runtime/src
    crates/tla-smt/src
    crates/tla-wire/src
    crates/tla-zenon/src
)
echo "=== Code Quality Gate (#1261 — scoped to listed crates only) ==="
echo "Scope: ${GATE_SCOPE[*]}"
echo ""

# ─── 1. No production file >3,000 lines ──────────────────────────────────────
# Exemptions: kani_harnesses.rs (proof harness collection), test files
echo "--- Check 1: No production file exceeds 3,000 lines ---"
oversized=$(find "${GATE_SCOPE[@]}" -name "*.rs" \
    -not -name "kani_harnesses.rs" \
    -not -name "*_tests.rs" \
    -not -name "tests.rs" \
    -not -path "*/tests/*" \
    -exec wc -l {} \; | awk '$1 > 3000 {print}' | sort -rn)
if [ -z "$oversized" ]; then
    echo "PASS: No production files exceed 3,000 lines"
    PASS=$((PASS+1))
else
    echo "FAIL: Production files over 3,000 lines:"
    echo "$oversized"
    FAIL=$((FAIL+1))
fi
echo ""

# ─── 1b. Warning zone: files within 15% of 3,000-line gate ──────────────────
# Informational only — no PASS/FAIL. Tracks files at risk of becoming gate
# failures so decomposition can be planned proactively. (Part of #1360)
WARNING_THRESHOLD=2550  # 85% of 3,000
echo "--- Info: Files in warning zone (${WARNING_THRESHOLD}-3,000 lines) ---"
warning_zone=$(find "${GATE_SCOPE[@]}" -name "*.rs" \
    -not -name "kani_harnesses.rs" \
    -not -name "*_tests.rs" \
    -not -name "tests.rs" \
    -not -path "*/tests/*" \
    -exec wc -l {} \; | awk -v lo="$WARNING_THRESHOLD" '$1 >= lo && $1 <= 3000 {print}' | sort -rn)
if [ -z "$warning_zone" ]; then
    echo "OK: No files in warning zone"
else
    wz_count=$(echo "$warning_zone" | wc -l | tr -d ' ')
    echo "WARNING: $wz_count files within 15% of gate-1 threshold:"
    echo "$warning_zone"
fi
echo ""

# ─── 1c. Oversized test files (informational) ────────────────────────────────
# Non-blocking informational check (Part of #1371). Tracks test file size debt
# so decomposition can be planned. Scans all crates, not just GATE_SCOPE.
TEST_FILE_THRESHOLD=1000
echo "--- Info: Oversized test files (>${TEST_FILE_THRESHOLD} lines) ---"
oversized_tests=$(find crates/ \( -name "*_tests.rs" -o -name "tests.rs" -o -path "*/tests/*.rs" \) \
    -exec wc -l {} \; | awk -v limit="$TEST_FILE_THRESHOLD" '$1 > limit {print}' | sort -rn)
if [ -z "$oversized_tests" ]; then
    echo "OK: No test files exceed ${TEST_FILE_THRESHOLD} lines"
else
    ot_count=$(echo "$oversized_tests" | wc -l | tr -d ' ')
    ot_total=$(echo "$oversized_tests" | awk '{s+=$1} END {print s}')
    echo "INFO: $ot_count test files exceed ${TEST_FILE_THRESHOLD} lines (${ot_total} total lines):"
    echo "$oversized_tests"
fi
echo ""

# ─── 2. No function exceeds 500 lines ────────────────────────────────────────
# Uses a Rust-aware scanner that ignores comments and string literals while
# tracking braces.
echo "--- Check 2: No function exceeds 500 lines ---"
FUNC_LIMIT=500
rust_scan_files=()
while IFS= read -r file; do
    rust_scan_files+=("$file")
done < <(find "${GATE_SCOPE[@]}" -name "*.rs" \
    -not -name "kani_harnesses.rs" \
    -not -name "*_tests.rs" \
    -not -name "tests.rs" \
    -not -path "*/tests/*" 2>/dev/null)
long_funcs_output=""
if [ "${#rust_scan_files[@]}" -gt 0 ]; then
    # The Rust scanner exits 0 even when it reports offenders; stdout is the
    # failure signal so this gate can aggregate every check result.
    long_funcs_output=$(run_rust_function_span_scan \
        --limit "$FUNC_LIMIT" \
        "${rust_scan_files[@]}")
fi
long_funcs_output=$(echo "$long_funcs_output" | sed '/^$/d')
if [ -z "$long_funcs_output" ]; then
    echo "PASS: No functions exceed $FUNC_LIMIT lines"
    PASS=$((PASS+1))
else
    count=$(echo "$long_funcs_output" | wc -l | tr -d ' ')
    echo "FAIL: $count functions exceed $FUNC_LIMIT lines:"
    echo "$long_funcs_output"
    FAIL=$((FAIL+1))
fi
echo ""

# ─── 2b. Oversized test functions (informational) ────────────────────────────
# Non-blocking informational check (Part of #1371). Tracks test function size
# debt separately from production function size (Check 2). Scans all crates.
TEST_FUNC_LIMIT=200
echo "--- Info: Oversized test functions (>${TEST_FUNC_LIMIT} lines) ---"
test_scan_files=()
while IFS= read -r file; do
    test_scan_files+=("$file")
done < <(find crates/ \( -name "*_tests.rs" -o -name "tests.rs" -o -path "*/tests/*.rs" \) 2>/dev/null)
long_test_funcs=""
if [ "${#test_scan_files[@]}" -gt 0 ]; then
    long_test_funcs=$(run_rust_function_span_scan \
        --limit "$TEST_FUNC_LIMIT" \
        "${test_scan_files[@]}" 2>/dev/null) || long_test_funcs=""
fi
long_test_funcs=$(echo "$long_test_funcs" | sed '/^$/d')
if [ -z "$long_test_funcs" ]; then
    echo "OK: No test functions exceed ${TEST_FUNC_LIMIT} lines"
else
    tf_count=$(echo "$long_test_funcs" | wc -l | tr -d ' ')
    echo "INFO: $tf_count test functions exceed ${TEST_FUNC_LIMIT} lines:"
    echo "$long_test_funcs"
fi
echo ""

# ─── 3. No #[allow(dead_code)] in non-test code ──────────────────────────────
echo "--- Check 3: No #[allow(dead_code)] annotations ---"
dead_count=$({ grep -r '#\[allow(dead_code)\]' "${GATE_SCOPE[@]}" 2>/dev/null || true; } | { grep -v "tests.rs" || true; } | { grep -v '#\[cfg(test)\]' || true; } | wc -l | tr -d ' ')
if [ "$dead_count" -eq 0 ]; then
    echo "PASS: No #[allow(dead_code)] annotations"
    PASS=$((PASS+1))
else
    echo "FAIL: $dead_count #[allow(dead_code)] annotations found:"
    { grep -rn '#\[allow(dead_code)\]' "${GATE_SCOPE[@]}" 2>/dev/null || true; } | { grep -v "tests.rs" || true; } | { grep -v '#\[cfg(test)\]' || true; }
    FAIL=$((FAIL+1))
fi
echo ""

# ─── 4. No uncached std::env::var in hot paths ───────────────────────────────
echo "--- Check 4: No uncached std::env::var() calls ---"
uncached=""
while IFS= read -r file; do
    while IFS= read -r match_line; do
        lineno="${match_line%%:*}"
        context_start=$((lineno > 30 ? lineno - 30 : 1))
        context_end=$((lineno + 30))
        near_cached=$(sed -n "${context_start},${context_end}p" "$file" | grep -cE "OnceLock|get_or_init|LazyLock") || near_cached=0
        if [ "$near_cached" -eq 0 ]; then
            uncached="${uncached}${file}:${match_line}
"
        fi
    done < <(grep -n "std::env::var(" "$file" 2>/dev/null | grep -v "tests.rs\|_tests.rs\|#\[cfg(test)\]\|//.*std::env::var\|gate:env-var-ok" || true)
done < <(grep -rl "std::env::var(" "${GATE_SCOPE[@]}" 2>/dev/null | grep -v "tests.rs\|_tests.rs" || true)
uncached=$(echo "$uncached" | sed '/^$/d')
if [ -z "$uncached" ]; then
    echo "PASS: No uncached std::env::var() in crates"
    PASS=$((PASS+1))
else
    echo "FAIL: Uncached std::env::var() calls found:"
    echo "$uncached"
    FAIL=$((FAIL+1))
fi
echo ""

# ─── 5. eval_builtin decomposition ───────────────────────────────────────────
echo "--- Check 5: eval_builtin decomposition ---"
eval_dir="crates/tla-check/src/eval"
submodule_count=0
for submod in builtin_sequences builtin_bags eval_arith eval_debug; do
    if [ -f "$eval_dir/${submod}.rs" ]; then
        submodule_count=$((submodule_count+1))
    fi
done
if [ "$submodule_count" -ge 3 ]; then
    echo "PASS: eval_builtin split into $submodule_count submodules"
    PASS=$((PASS+1))
else
    echo "FAIL: eval_builtin needs decomposition ($submodule_count submodules found, need >=3)"
    FAIL=$((FAIL+1))
fi
echo ""

# ─── 6. Generic AST visitor migration ────────────────────────────────────────
# Supports both single-file (expr_visitor.rs) and directory (expr_visitor/) layouts.
echo "--- Check 6: Generic AST visitor migration ---"
visitor_file="crates/tla-check/src/expr_visitor.rs"
visitor_dir="crates/tla-check/src/expr_visitor"
if [ -f "$visitor_file" ]; then
    visitor_impls=$(grep -c "impl ExprVisitor for" "$visitor_file" 2>/dev/null) || visitor_impls=0
    if [ "$visitor_impls" -ge 10 ]; then
        echo "PASS: expr_visitor.rs has $visitor_impls ExprVisitor implementations"
        PASS=$((PASS+1))
    else
        echo "FAIL: expr_visitor.rs has only $visitor_impls ExprVisitor implementations (need >=10)"
        FAIL=$((FAIL+1))
    fi
elif [ -d "$visitor_dir" ]; then
    visitor_impls=$(grep -rc "impl ExprVisitor for" "$visitor_dir" 2>/dev/null | awk -F: '{s+=$2} END {print s+0}') || visitor_impls=0
    if [ "$visitor_impls" -ge 10 ]; then
        echo "PASS: expr_visitor/ has $visitor_impls ExprVisitor implementations across submodules"
        PASS=$((PASS+1))
    else
        echo "FAIL: expr_visitor/ has only $visitor_impls ExprVisitor implementations (need >=10)"
        FAIL=$((FAIL+1))
    fi
else
    echo "FAIL: expr_visitor module does not exist (neither .rs file nor directory)"
    FAIL=$((FAIL+1))
fi
echo ""

# ─── 7. No production wildcard imports ────────────────────────────────────────
# Check for `use super::*` in non-test code. Test modules (within 20 lines of
# #[cfg(test)] or `mod tests`) are excluded since `use super::*` is idiomatic there.
echo "--- Check 7: No production wildcard imports (use super::*) ---"
wildcard_hits=""
while IFS= read -r match; do
    file="${match%%:*}"
    rest="${match#*:}"
    lineno="${rest%%:*}"
    # Skip test files entirely
    case "$file" in *_tests.rs|*tests.rs|*/tests/*) continue ;; esac
    # Check if within 20 lines of #[cfg(test)] or mod tests
    context_start=$((lineno > 20 ? lineno - 20 : 1))
    in_test=$(sed -n "${context_start},${lineno}p" "$file" | grep -cE '#\[cfg\(test\)\]|mod tests') || in_test=0
    if [ "$in_test" -eq 0 ]; then
        wildcard_hits="${wildcard_hits}${match}
"
    fi
done < <(grep -rn '^[[:space:]]*use super::\*' "${GATE_SCOPE[@]}" 2>/dev/null || true)
wildcard_hits=$(echo "$wildcard_hits" | sed '/^$/d')
wildcard_count=$(echo "$wildcard_hits" | sed '/^$/d' | wc -l | tr -d ' ')
if [ -z "$wildcard_hits" ]; then
    wildcard_count=0
fi
if [ "$wildcard_count" -eq 0 ]; then
    echo "PASS: No production wildcard imports"
    PASS=$((PASS+1))
else
    echo "FAIL: $wildcard_count production wildcard imports found:"
    echo "$wildcard_hits"
    FAIL=$((FAIL+1))
fi
echo ""

# ─── 8. Build check ──────────────────────────────────────────────────────────
echo "--- Check 8: Build gate ---"
check8_log=$(mktemp -t check_code_quality_gate.XXXXXX)
check8_guard_pattern="DIRTY-SCOPE|Tracked files changed during verification"
set +e
cargo check -p tla-check >"$check8_log" 2>&1
check8_status=$?
set -e

# Read guard diagnostics from the log file directly to avoid pipefail false
# negatives on large outputs (echo|grep can return 141 via SIGPIPE).
if [ "$check8_status" -eq 0 ]; then
    echo "PASS: cargo check passes"
    PASS=$((PASS+1))
elif [ "$check8_status" -eq 86 ] && grep -qE "$check8_guard_pattern" "$check8_log"; then
    echo "FAIL: nondeterministic/shared-tree guard abort (cargo check exit 86)"
    echo "Action: rerun from a clean/stable worktree after workers checkpoint."
    drift_detail=$(grep -m1 -E "$check8_guard_pattern" "$check8_log" || true)
    if [ -n "$drift_detail" ]; then
        echo "Detail: $drift_detail"
    fi
    FAIL=$((FAIL+1))
else
    echo "FAIL: deterministic build failure (cargo check exit $check8_status)"
    echo "Diagnostic excerpt (last 20 lines):"
    tail -n 20 "$check8_log"
    FAIL=$((FAIL+1))
fi
rm -f "$check8_log"
echo ""

# ─── 9. API consumer compatibility canaries (Part of #1325) ─────────────────
# Runs the Rust CLI gate that compile-checks canary crates importing stable
# public API paths.
# Failure here means a refactor broke a downstream API contract.
echo "--- Check 9: API consumer compatibility canaries ---"
set +e
run_canary_gate --kind api --mode enforce
api_canary_status=$?
set -e
if [ "$api_canary_status" -eq 0 ]; then
    echo "PASS: API canary gate passed"
    PASS=$((PASS+1))
else
    echo "FAIL: API canary gate failed (exit $api_canary_status)"
    FAIL=$((FAIL+1))
fi
echo ""

# ─── 10. No silent error coercion in model checker (Part of #1433) ─────────
echo "--- Check 10: No silent error coercion in model checker ---"
if run_canary_gate --kind silent-error --mode enforce; then
    PASS=$((PASS+1))
else
    echo "FAIL: Silent error coercion detected — see output above"
    FAIL=$((FAIL+1))
fi
echo ""

# ─── Info: Rust system-health gate facade ───────────────────────────────────
# Transitional, non-blocking call: the legacy Python health facade currently
# reports environment drift/staleness on developer machines, so this quality
# gate exercises the Rust facade without adding a new blocking criterion.
echo "--- Info: System health gate (Rust CLI, warn mode) ---"
run_system_health_gate --mode warn
echo ""

echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -eq 0 ]; then
    echo "ALL #1261-SCOPED GATES PASSED"
    echo "Note: This gate covers #1261 crates only. tla-ay and tla-codegen are"
    echo "tracked separately (#1328, #1329). Check those before lifting FEATURE_FREEZE."
else
    echo "GATES FAILING — do NOT lift FEATURE_FREEZE"
fi
exit $FAIL
