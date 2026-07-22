#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# Wrap ty-mcc-csv-compare across all 13 MCC examinations.
#
# Models and reference CSV are resolved through the versioned `ty-corpus`
# CLI, not hardcoded snapshot paths. This eliminates the recurring failure
# mode where an ad-hoc /private/tmp/mcc-models-root snapshot drifts from the
# official MCC tarballs and produces spurious "wrong" rows.
#
# Background: prior broad measurements (v3 / v4 / v5) ran only the four
# examinations that don't need property XML files (StateSpace,
# ReachabilityDeadlock, OneSafe, QuasiLiveness), which made the per-tool
# podium estimator report 0 on CTL / LTL / Reachability(Fireability,
# Cardinality) / UpperBounds — see
# docs/mcc-2026/ctl-ltl-reachability-formula-diagnosis-2026-05-24.md. This
# script closes that gap by passing all 13 examinations enumerated in
# crates/tla-petri/src/examination_kind.rs (Examination::ALL).
#
# Usage:
#   scripts/measure-all-exams.sh <subset_file> <output_prefix> [timeout_s]
#
# Args:
#   subset_file    Path to newline-separated list of model dir names.
#                  Each name must exist under the corpus cache resolved by
#                  `ty-corpus ensure`. Use the literal string `ALL` to use
#                  every model in the corpus.
#   output_prefix  Prefix for outputs: <prefix>-results.tsv,
#                  <prefix>-summary.json, <prefix>.stdout, <prefix>.stderr.
#   timeout_s      Optional per-case wall budget (default 30 s).
#
# Environment overrides (all optional — defaults come from ty-corpus):
#   TY_CORPUS_VERSION       Corpus year, default `2025`
#   TY_CORPUS_ARCHIVES_DIR  Override the .tgz source dir (see `ty-corpus --help`)
#   TY_CORPUS_CACHE_DIR     Override the extraction cache root
#   MODELS_ROOT             Bypass ty-corpus entirely; must match the extracted layout
#   CSV_PATH                Bypass ty-corpus csv-path resolution
#   TY_MCC_BIN              default <repo>/target/release/ty-mcc
#   CMP_BIN                 default <repo>/target/release/ty-mcc-csv-compare
#   TY_CORPUS_BIN           default <repo>/target/release/ty-corpus
#   THREADS                 default 4

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    cat >&2 <<USAGE
Usage: scripts/measure-all-exams.sh <subset_file> <output_prefix> [timeout_s]

  subset_file    Path to newline-separated model dir names, OR the literal
                 string "ALL" to use every model in the resolved corpus.
  output_prefix  Prefix for the output TSV/JSON/logs.
  timeout_s      Per-case wall budget (default 30 s).
USAGE
    exit 2
fi

SUBSET_FILE="$1"
OUT_PREFIX="$2"
TIMEOUT="${3:-30}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TY_MCC_BIN="${TY_MCC_BIN:-$REPO_DIR/target/release/ty-mcc}"
CMP_BIN="${CMP_BIN:-$REPO_DIR/target/release/ty-mcc-csv-compare}"
CORPUS_BIN="${TY_CORPUS_BIN:-$REPO_DIR/target/release/ty-corpus}"
CORPUS_VERSION="${TY_CORPUS_VERSION:-2025}"
THREADS="${THREADS:-4}"

for path in "$TY_MCC_BIN" "$CMP_BIN" "$CORPUS_BIN"; do
    if [ ! -x "$path" ]; then
        echo "ERROR: binary not found or not executable: $path" >&2
        echo "       run: cargo build --release -p tla-petri" >&2
        exit 2
    fi
done

# Resolve MODELS_ROOT via ty-corpus unless the caller pinned it explicitly.
if [ -z "${MODELS_ROOT:-}" ]; then
    if ! MODELS_ROOT="$("$CORPUS_BIN" ensure --version "$CORPUS_VERSION")"; then
        echo "ERROR: ty-corpus ensure --version $CORPUS_VERSION failed; aborting" >&2
        exit 2
    fi
fi
if [ ! -d "$MODELS_ROOT" ]; then
    echo "ERROR: resolved MODELS_ROOT is not a directory: $MODELS_ROOT" >&2
    exit 2
fi

# Resolve CSV_PATH via ty-corpus unless pinned.
if [ -z "${CSV_PATH:-}" ]; then
    if ! CSV_PATH="$("$CORPUS_BIN" csv-path --version "$CORPUS_VERSION")"; then
        echo "ERROR: ty-corpus csv-path --version $CORPUS_VERSION failed; aborting" >&2
        exit 2
    fi
fi
if [ ! -e "$CSV_PATH" ]; then
    echo "ERROR: reference CSV does not exist: $CSV_PATH" >&2
    exit 2
fi

# Resolve the subset. `ALL` means every model in the resolved corpus.
if [ "$SUBSET_FILE" = "ALL" ]; then
    SUBSET="$("$CORPUS_BIN" list --version "$CORPUS_VERSION" | tr '\n' ',' | sed 's/,$//;s/,,*/,/g')"
else
    if [ ! -e "$SUBSET_FILE" ]; then
        echo "ERROR: subset file not found: $SUBSET_FILE" >&2
        exit 2
    fi
    SUBSET="$(tr '\n' ',' <"$SUBSET_FILE" | sed 's/,$//;s/,,*/,/g')"
fi
if [ -z "$SUBSET" ]; then
    echo "ERROR: resolved subset is empty (file: $SUBSET_FILE)" >&2
    exit 2
fi
N_MODELS="$(printf '%s\n' "$SUBSET" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' ')"

# The 13 MCC examinations, mirroring tla_petri::Examination::ALL. Kept
# explicit here (rather than synthesized from the binary) so the wrapper
# documents intent even when the enum grows or shrinks.
EXAMS="ReachabilityDeadlock,ReachabilityCardinality,ReachabilityFireability,\
CTLCardinality,CTLFireability,LTLCardinality,LTLFireability,\
StateSpace,UpperBounds,OneSafe,QuasiLiveness,StableMarking,Liveness"

RESULTS_TSV="${OUT_PREFIX}-results.tsv"
SUMMARY_JSON="${OUT_PREFIX}-summary.json"
STDOUT_LOG="${OUT_PREFIX}.stdout"
STDERR_LOG="${OUT_PREFIX}.stderr"

echo "measure-all-exams.sh"
echo "  corpus       : $CORPUS_VERSION"
echo "  models_root  : $MODELS_ROOT"
echo "  csv          : $CSV_PATH"
echo "  binary       : $TY_MCC_BIN"
echo "  compare      : $CMP_BIN"
echo "  subset_file  : $SUBSET_FILE ($N_MODELS models)"
echo "  exams        : $EXAMS"
echo "  threads      : $THREADS"
echo "  timeout_s    : $TIMEOUT"
echo "  results      : $RESULTS_TSV"
echo "  summary_json : $SUMMARY_JSON"
echo

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
rc=$?
set -e

echo "ty-mcc-csv-compare exit: $rc"
echo "  stdout: $STDOUT_LOG"
echo "  stderr: $STDERR_LOG"
echo "  rows  : $(wc -l <"$RESULTS_TSV" 2>/dev/null | tr -d ' ')"

# Exit-code semantics from ty-mcc-csv-compare:
#   0 -> all rows have wrong_units == 0
#   1 -> at least one wrong unit (still a valid measurement)
#   2 -> harness error (propagate failure)
if [ "$rc" -eq 2 ]; then
    echo "ERROR: harness failure; see $STDERR_LOG" >&2
    exit 2
fi
exit 0
