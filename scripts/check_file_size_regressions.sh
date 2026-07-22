#!/bin/bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# check_file_size_regressions.sh - enforce the production Rust file-size ceiling.
#
# The configuration lives in .file_size_baselines.json:
#   production_rust.line_ceiling              default maximum line count
#   production_rust.production_roots          first-party roots to scan
#   production_rust.excluded_path_patterns    tests/generated-code exclusions
#   production_rust.waivers                   explicit large-file waivers
#
# Usage:
#   scripts/check_file_size_regressions.sh [--threshold N] [--baseline] [--flags] [--quiet]
#
# --threshold N   Override the configured line ceiling for this run.
# --baseline      Accepted for compatibility; waiver growth is always checked.
# --flags         Write results to .flags/large_file_regressions.
# --quiet         Suppress informational output, only show violations.

set -euo pipefail

THRESHOLD_OVERRIDE=""
CHECK_WAIVER_GROWTH=true
WRITE_FLAGS=false
QUIET=false
EXIT_CODE=0
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    echo "Usage: $0 [--threshold N] [--baseline] [--flags] [--quiet]"
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

is_positive_integer() {
    case "$1" in
        ""|*[!0-9]*) return 1 ;;
        *) [ "$1" -gt 0 ] ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --threshold)
            [ $# -ge 2 ] || die "--threshold requires a number"
            THRESHOLD_OVERRIDE="$2"
            shift 2
            ;;
        --threshold=*)
            THRESHOLD_OVERRIDE="${1#*=}"
            shift
            ;;
        --baseline)
            CHECK_WAIVER_GROWTH=true
            shift
            ;;
        --flags)
            WRITE_FLAGS=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

[ -z "$THRESHOLD_OVERRIDE" ] || is_positive_integer "$THRESHOLD_OVERRIDE" || die "--threshold must be a positive integer"

BASELINE_FILE="$REPO_ROOT/.file_size_baselines.json"
FLAGS_DIR="$REPO_ROOT/.flags"
FLAGS_OUTPUT="$FLAGS_DIR/large_file_regressions"

[ -f "$BASELINE_FILE" ] || die "missing $BASELINE_FILE"
command -v jq >/dev/null 2>&1 || die "jq is required to parse $BASELINE_FILE"

if ! jq -e '
    .schema_version == 2
    and ((.production_rust.line_ceiling | type) == "number")
    and ((.production_rust.production_roots | type) == "array")
    and ((.production_rust.waivers | type) == "array")
    and all(.production_rust.production_roots[]?; (type == "string") and (length > 0))
    and all(.production_rust.waivers[]?; ((.path | type) == "string") and (.path | length > 0) and ((.lines | type) == "number") and (.lines > 0))
' "$BASELINE_FILE" >/dev/null; then
    die "$BASELINE_FILE does not match schema_version 2 production_rust shape"
fi

if ! jq -e '
    ([.production_rust.waivers[]?.path] | length)
    ==
    ([.production_rust.waivers[]?.path] | unique | length)
' "$BASELINE_FILE" >/dev/null; then
    die "$BASELINE_FILE contains duplicate waiver paths"
fi

CONFIGURED_CEILING="$(jq -r '.production_rust.line_ceiling' "$BASELINE_FILE")"
is_positive_integer "$CONFIGURED_CEILING" || die "production_rust.line_ceiling must be a positive integer"

if [ -n "$THRESHOLD_OVERRIDE" ]; then
    THRESHOLD="$THRESHOLD_OVERRIDE"
else
    THRESHOLD="$CONFIGURED_CEILING"
fi

DEFAULT_WAIVER_MULTIPLIER="$(jq -r '.production_rust.default_waiver_multiplier // 1.0' "$BASELINE_FILE")"
if ! jq -e '.production_rust.default_waiver_multiplier? // 1.0 | type == "number" and . > 0' "$BASELINE_FILE" >/dev/null; then
    die "production_rust.default_waiver_multiplier must be a positive number"
fi

exclude_patterns=()
while IFS= read -r pattern; do
    [ -n "$pattern" ] && exclude_patterns+=("$pattern")
done < <(jq -r '.production_rust.excluded_path_patterns[]?' "$BASELINE_FILE")

missing_roots=()
while IFS= read -r root; do
    if [ ! -d "$REPO_ROOT/$root" ]; then
        missing_roots+=("$root")
    fi
done < <(jq -r '.production_rust.production_roots[]' "$BASELINE_FILE")

if [ ${#missing_roots[@]} -gt 0 ]; then
    echo "ERROR: $BASELINE_FILE contains missing production roots:" >&2
    for root in "${missing_roots[@]}"; do
        echo "  $root" >&2
    done
    exit 2
fi

unwaived_violations=()
waiver_regressions=()

is_excluded() {
    local path="$1"
    local pattern

    for pattern in "${exclude_patterns[@]}"; do
        case "$path" in
            $pattern) return 0 ;;
        esac
    done

    return 1
}

is_waived() {
    local path="$1"
    jq -e --arg path "$path" '.production_rust.waivers[]? | select(.path == $path)' "$BASELINE_FILE" >/dev/null
}

production_files() {
    local root
    local full_root

    while IFS= read -r root; do
        full_root="$REPO_ROOT/$root"
        find "$full_root" -name '*.rs' -type f 2>/dev/null
    done < <(jq -r '.production_rust.production_roots[]' "$BASELINE_FILE")
}

# --- Ceiling scan ---
while IFS= read -r file; do
    rel="${file#$REPO_ROOT/}"
    if is_excluded "$rel"; then
        continue
    fi

    lines="$(wc -l < "$file")"
    if [ "$lines" -gt "$THRESHOLD" ] && ! is_waived "$rel"; then
        unwaived_violations+=("OVER ($lines > $THRESHOLD): $rel")
        EXIT_CODE=1
    fi
done < <(production_files | sort -u)

if [ ${#unwaived_violations[@]} -gt 0 ]; then
    echo "=== Unwaived production Rust files exceeding ${THRESHOLD}-line ceiling ==="
    for v in "${unwaived_violations[@]}"; do
        echo "  $v"
    done
    echo ""
fi

# --- Waiver growth check ---
if [ "$CHECK_WAIVER_GROWTH" = true ]; then
    while IFS=$'\t' read -r filepath waived_lines multiplier allowed_lines waiver_issue; do
        full_path="$REPO_ROOT/$filepath"
        issue_suffix=""
        if [ -n "$waiver_issue" ] && [ "$waiver_issue" != "null" ]; then
            issue_suffix=" (waiver: #$waiver_issue)"
        fi

        if [ ! -f "$full_path" ]; then
            waiver_regressions+=("MISSING WAIVER TARGET: $filepath$issue_suffix")
            EXIT_CODE=1
            continue
        fi

        current_lines="$(wc -l < "$full_path")"
        if [ "$current_lines" -gt "$allowed_lines" ]; then
            waiver_regressions+=("WAIVER GROWTH ($current_lines > ${allowed_lines}, ${multiplier}x $waived_lines): $filepath$issue_suffix")
            EXIT_CODE=1
        fi
    done < <(jq -r --argjson default_multiplier "$DEFAULT_WAIVER_MULTIPLIER" '
        .production_rust.waivers[]
        | (.allowed_multiplier // $default_multiplier) as $multiplier
        | [
            .path,
            (.lines | tostring),
            ($multiplier | tostring),
            ((.lines * $multiplier) | ceil | tostring),
            (.waiver_issue // "")
          ]
        | @tsv
    ' "$BASELINE_FILE")
fi

if [ ${#waiver_regressions[@]} -gt 0 ]; then
    echo "=== Waived production Rust files exceeding allowed growth ==="
    for r in "${waiver_regressions[@]}"; do
        echo "  $r"
    done
    echo ""
fi

# --- Write flags output ---
if [ "$WRITE_FLAGS" = true ]; then
    mkdir -p "$FLAGS_DIR"
    {
        date -u +"%Y-%m-%dT%H:%M:%SZ"
        echo "Production Rust ceiling: ${THRESHOLD} lines"
        echo "Default waiver multiplier: ${DEFAULT_WAIVER_MULTIPLIER}"
        if [ ${#unwaived_violations[@]} -gt 0 ]; then
            echo "Unwaived files above ceiling:"
            for v in "${unwaived_violations[@]}"; do
                echo "  $v"
            done
        fi
        if [ ${#waiver_regressions[@]} -gt 0 ]; then
            echo "Waiver growth regressions:"
            for r in "${waiver_regressions[@]}"; do
                echo "  $r"
            done
        fi
        if [ ${#unwaived_violations[@]} -eq 0 ] && [ ${#waiver_regressions[@]} -eq 0 ]; then
            echo "No regressions detected."
        fi
    } > "$FLAGS_OUTPUT"
fi

# --- Summary ---
if [ "$QUIET" = false ]; then
    total=$((${#unwaived_violations[@]} + ${#waiver_regressions[@]}))
    waiver_count="$(jq -r '.production_rust.waivers | length' "$BASELINE_FILE")"
    if [ "$total" -eq 0 ]; then
        echo "No file size regressions detected (ceiling: $THRESHOLD, waivers: $waiver_count)."
    else
        echo "Found $total file size issue(s)."
    fi
fi

exit $EXIT_CODE
