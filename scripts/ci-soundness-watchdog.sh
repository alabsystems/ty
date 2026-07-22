#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# ci-soundness-watchdog.sh — wrong-row regression watchdog.
#
# NOTE: this is a MANUAL gate. There is no automated CI runner in this repo
# (no .github/, no git hooks, no scheduler), so nothing invokes this script
# for you — it runs only when a developer runs it. The "ci-" prefix in the
# filename is historical.
#
# Runs a curated, fast-converging multi-tool-gold-truth sweep on ~10 MCC
# models across all 13 examinations and asserts wrong_units == 0 against the
# official MCC 2025 raw-result-analysis.csv. Exits non-zero (intended for a
# developer to catch before push, or for a future CI step) if any new wrong
# answer appears.
#
# Why this exists
# ---------------
# `cargo test` covers a handful of in-process benchmarks
# (crates/tla-petri/tests/mcc_benchmarks.rs) and exercises ty-mcc-internal
# code paths only. It cannot catch regressions that:
#
#   * surface only through the `ty-mcc` binary's stdout-parsing surface,
#   * depend on actual MCC corpus property XML files,
#   * manifest as silent soundness regressions in CTL / LTL examinations,
#   * are masked by a "score win" on one model that flips another to wrong.
#
# This watchdog is the cheapest gate that closes that gap. It runs the same
# pipeline a score-affecting change would run, but on a small, fast subset.
#
# Subset (~10 models)
# -------------------
# Chosen to (a) touch every examination class, (b) span PT and COL nets, and
# (c) converge inside the per-case timeout on developer hardware:
#
#   Anderson-PT-04, TokenRing-PT-005, Philosophers-PT-000010,
#   Sudoku-PT-BN01, Sudoku-COL-AN01, GlobalResAllocation-COL-03,
#   BridgeAndVehicles-COL-V04P05N02, CSRepetitions-COL-02,
#   LamportFastMutEx-PT-2, AirplaneLD-PT-0010
#
# At threads=4, timeout=5s/case, the full sweep finishes in ~1m35s wall on a
# 2024-class developer Mac. That budget keeps the gate cheap enough to run by
# hand before every score-affecting push (and well inside a <5 min target if
# this is ever wired into an automated runner).
#
# Usage
# -----
#   scripts/ci-soundness-watchdog.sh
#
# Environment overrides (all optional):
#   TY_CORPUS_VERSION   default 2025
#   TIMEOUT             default 5  (seconds per case)
#   THREADS             default 4
#   OUT_DIR             default <repo>/target/ci-soundness-watchdog
#
# Exit codes
# ----------
#   0  every selected row has wrong_units == 0  (PASS)
#   1  at least one wrong row                   (FAIL — regression)
#   2  harness error / missing binary / corpus  (cannot adjudicate)
#
# Extending
# ---------
# To extend the subset, add model names to the heredoc-built SUBSET_FILE
# below. Keep the per-case wall budget compatible with TIMEOUT or bump
# TIMEOUT. Verify every added model exists in the corpus with:
#   target/release/ty-corpus list --version 2025 | grep '^<NAME>$'
#
# Do NOT add models that routinely time out at TIMEOUT — they pollute the
# `timeout` row category and slow the gate without adding signal.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TY_MCC_BIN="${TY_MCC_BIN:-$REPO_DIR/target/release/ty-mcc}"
CMP_BIN="${CMP_BIN:-$REPO_DIR/target/release/ty-mcc-csv-compare}"
CORPUS_BIN="${TY_CORPUS_BIN:-$REPO_DIR/target/release/ty-corpus}"
CORPUS_VERSION="${TY_CORPUS_VERSION:-2025}"
THREADS="${THREADS:-4}"
TIMEOUT="${TIMEOUT:-5}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/target/ci-soundness-watchdog}"

# --- preflight: required binaries ------------------------------------------
for path in "$TY_MCC_BIN" "$CMP_BIN" "$CORPUS_BIN"; do
    if [ ! -x "$path" ]; then
        echo "FAIL: binary not found or not executable: $path" >&2
        echo "      build first: cargo build --release -p tla-petri" >&2
        exit 2
    fi
done

mkdir -p "$OUT_DIR"

# --- resolve corpus via ty-corpus (no hardcoded paths) ---------------------
if ! MODELS_ROOT="$("$CORPUS_BIN" ensure --version "$CORPUS_VERSION")"; then
    echo "FAIL: ty-corpus ensure --version $CORPUS_VERSION failed" >&2
    exit 2
fi
if [ ! -d "$MODELS_ROOT" ]; then
    echo "FAIL: resolved MODELS_ROOT is not a directory: $MODELS_ROOT" >&2
    exit 2
fi

if ! CSV_PATH="$("$CORPUS_BIN" csv-path --version "$CORPUS_VERSION")"; then
    echo "FAIL: ty-corpus csv-path --version $CORPUS_VERSION failed" >&2
    exit 2
fi
if [ ! -e "$CSV_PATH" ]; then
    echo "FAIL: reference CSV does not exist: $CSV_PATH" >&2
    exit 2
fi

# --- curated subset (~10 models, all 13 exams in ~1m35s wall) --------------
SUBSET_FILE="$OUT_DIR/subset.txt"
cat >"$SUBSET_FILE" <<'EOF'
Anderson-PT-04
TokenRing-PT-005
Philosophers-PT-000010
Sudoku-PT-BN01
Sudoku-COL-AN01
GlobalResAllocation-COL-03
BridgeAndVehicles-COL-V04P05N02
CSRepetitions-COL-02
LamportFastMutEx-PT-2
AirplaneLD-PT-0010
EOF

# Verify every subset model is in the resolved corpus so we fail loudly
# (exit 2 = harness error) rather than silently shrinking the gate.
CORPUS_LIST="$OUT_DIR/corpus.list"
"$CORPUS_BIN" list --version "$CORPUS_VERSION" >"$CORPUS_LIST"
MISSING=()
while IFS= read -r name; do
    [ -z "$name" ] && continue
    if ! grep -Fxq "$name" "$CORPUS_LIST"; then
        MISSING+=("$name")
    fi
done <"$SUBSET_FILE"
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "FAIL: subset models missing from corpus $CORPUS_VERSION: ${MISSING[*]}" >&2
    echo "      corpus list: $CORPUS_LIST" >&2
    exit 2
fi

SUBSET="$(tr '\n' ',' <"$SUBSET_FILE" | sed 's/,$//;s/,,*/,/g')"
N_MODELS="$(printf '%s\n' "$SUBSET" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' ')"

# All 13 examinations (mirrors Examination::ALL and measure-all-exams.sh).
EXAMS="ReachabilityDeadlock,ReachabilityCardinality,ReachabilityFireability,\
CTLCardinality,CTLFireability,LTLCardinality,LTLFireability,\
StateSpace,UpperBounds,OneSafe,QuasiLiveness,StableMarking,Liveness"

RESULTS_TSV="$OUT_DIR/results.tsv"
SUMMARY_JSON="$OUT_DIR/summary.json"
STDOUT_LOG="$OUT_DIR/run.stdout"
STDERR_LOG="$OUT_DIR/run.stderr"

echo "ci-soundness-watchdog.sh"
echo "  corpus       : $CORPUS_VERSION"
echo "  models_root  : $MODELS_ROOT"
echo "  csv          : $CSV_PATH"
echo "  binary       : $TY_MCC_BIN"
echo "  compare      : $CMP_BIN"
echo "  subset       : $N_MODELS models (see $SUBSET_FILE)"
echo "  exams        : 13 (all of Examination::ALL)"
echo "  threads      : $THREADS"
echo "  timeout_s    : $TIMEOUT"
echo "  out_dir      : $OUT_DIR"
echo

START=$(date +%s)
set +e
"$CMP_BIN" \
    --csv-path "$CSV_PATH" \
    --models-root "$MODELS_ROOT" \
    --binary "$TY_MCC_BIN" \
    --subset "$SUBSET" \
    --exams "$EXAMS" \
    --threads "$THREADS" \
    --timeout "$TIMEOUT" \
    --results-tsv "$RESULTS_TSV" \
    --summary-json "$SUMMARY_JSON" \
    >"$STDOUT_LOG" 2>"$STDERR_LOG"
CMP_RC=$?
set -e
ELAPSED=$(( $(date +%s) - START ))

if [ "$CMP_RC" -eq 2 ]; then
    echo "FAIL: ty-mcc-csv-compare harness error (exit 2); see $STDERR_LOG" >&2
    exit 2
fi
if [ ! -s "$SUMMARY_JSON" ]; then
    echo "FAIL: summary JSON missing or empty: $SUMMARY_JSON" >&2
    exit 2
fi

# Parse wrong_units. Prefer jq when available; otherwise fall back to a
# tolerant grep+awk that handles the pretty-printed `"wrong_units": N,` form
# emitted by serde_json::to_string_pretty.
if command -v jq >/dev/null 2>&1; then
    WRONG="$(jq -r '.wrong_units // empty' "$SUMMARY_JSON")"
    ROWS="$(jq -r '.rows // empty' "$SUMMARY_JSON")"
    EXACT="$(jq -r '.exact_units // empty' "$SUMMARY_JSON")"
else
    WRONG="$(grep -E '"wrong_units"' "$SUMMARY_JSON" | head -1 \
            | awk -F'[:,]' '{gsub(/[[:space:]]/, "", $2); print $2}')"
    ROWS="$(grep -E '"rows"' "$SUMMARY_JSON" | head -1 \
            | awk -F'[:,]' '{gsub(/[[:space:]]/, "", $2); print $2}')"
    EXACT="$(grep -E '"exact_units"' "$SUMMARY_JSON" | head -1 \
            | awk -F'[:,]' '{gsub(/[[:space:]]/, "", $2); print $2}')"
fi

if [ -z "${WRONG:-}" ] || ! [[ "$WRONG" =~ ^[0-9]+$ ]]; then
    echo "FAIL: could not parse wrong_units from $SUMMARY_JSON" >&2
    exit 2
fi

echo
echo "summary:"
echo "  elapsed_s    : $ELAPSED"
echo "  rows         : ${ROWS:-?}"
echo "  exact_units  : ${EXACT:-?}"
echo "  wrong_units  : $WRONG"
echo "  cmp_exit     : $CMP_RC"
echo "  summary_json : $SUMMARY_JSON"
echo "  results_tsv  : $RESULTS_TSV"
echo

if [ "$WRONG" -gt 0 ]; then
    echo "FAIL ci-soundness-watchdog: wrong_units=$WRONG over $N_MODELS models x 13 exams in ${ELAPSED}s (regression — inspect $RESULTS_TSV)"
    exit 1
fi

echo "PASS ci-soundness-watchdog: wrong_units=0 over $N_MODELS models x 13 exams in ${ELAPSED}s"
exit 0
