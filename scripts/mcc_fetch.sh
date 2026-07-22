#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# Fetch MCC benchmark inputs and consensus-result history for local sweeps.
#
# Default layout:
#   ~/mcc-benchmarks/2024/
#     archives/INPUTS-2024.tar.gz
#     archives/raw-result-analysis.csv.zip
#     inputs/INPUTS-2024/*.tgz
#     results/extracted/raw-result-analysis.csv

set -euo pipefail

YEAR="${MCC_YEAR:-2024}"
ROOT="${MCC_ROOT:-$HOME/mcc-benchmarks/$YEAR}"
BASE_URL="${MCC_BASE_URL:-https://mcc.lip6.fr/$YEAR/archives}"
EXTRACT_INPUTS=1
FETCH_INPUTS=1
FORCE=0

usage() {
    cat <<'USAGE'
Usage: scripts/mcc_fetch.sh [options]

Options:
  --year YEAR            MCC year to fetch (default: 2024)
  --root DIR             Destination root (default: ~/mcc-benchmarks/YEAR)
  --base-url URL         Archive base URL (default: https://mcc.lip6.fr/YEAR/archives)
  --answer-key-only      Fetch/extract raw-result-analysis.csv.zip only
  --no-extract-inputs    Download INPUTS-YEAR.tar.gz but do not extract it
  --force                Redownload existing archives and re-extract results
  -h, --help             Show this help

Environment:
  MCC_YEAR, MCC_ROOT, MCC_BASE_URL provide the same defaults as the flags.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --year)
            YEAR="$2"
            shift 2
            ;;
        --root)
            ROOT="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --answer-key-only)
            FETCH_INPUTS=0
            shift
            ;;
        --no-extract-inputs)
            EXTRACT_INPUTS=0
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

ARCHIVE_DIR="$ROOT/archives"
INPUTS_DIR="$ROOT/inputs"
RESULTS_DIR="$ROOT/results/extracted"
INPUTS_ARCHIVE="$ARCHIVE_DIR/INPUTS-$YEAR.tar.gz"
# MCC ships the analysis CSVs as .tar.gz (the old `.csv.zip` name 302s to
# error404.php — that was the long-standing bug that made fetches silently fail).
RESULTS_TGZ="$ARCHIVE_DIR/raw-result-analysis.csv.tar.gz"
SUMMARY_TGZ="$ARCHIVE_DIR/GlobalSummary.csv.tar.gz"

# download URL DEST: returns 0 on success, 1 if the server has no such file
# (curl -f exits 22 on HTTP>=400, e.g. inputs not published yet). Fail-soft so
# the caller can keep going and grab whatever IS available.
download() {
    local url="$1"
    local dest="$2"
    local tmp="$dest.tmp"

    mkdir -p "$(dirname "$dest")"
    if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then
        echo "skip existing: $dest"
        return 0
    fi

    rm -f "$tmp"
    echo "download: $url"
    if curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$url"; then
        mv "$tmp" "$dest"
        return 0
    fi
    rm -f "$tmp"
    echo "NOT AVAILABLE: $url" >&2
    return 1
}

extract_results() {
    mkdir -p "$RESULTS_DIR"
    echo "extract: $RESULTS_TGZ -> $RESULTS_DIR"
    tar -xzf "$RESULTS_TGZ" -C "$RESULTS_DIR"
    if [ -s "$SUMMARY_TGZ" ]; then
        echo "extract: $SUMMARY_TGZ -> $RESULTS_DIR"
        tar -xzf "$SUMMARY_TGZ" -C "$RESULTS_DIR"
    fi
}

extract_inputs() {
    local extracted_root="$INPUTS_DIR/INPUTS-$YEAR"
    mkdir -p "$INPUTS_DIR"
    if [ "$FORCE" -eq 0 ] && [ -d "$extracted_root" ]; then
        if find "$extracted_root" -maxdepth 1 -type f -name '*.tgz' -print -quit | grep -q .; then
            echo "skip existing: $extracted_root"
            return
        fi
    fi

    echo "extract: $INPUTS_ARCHIVE -> $INPUTS_DIR"
    tar -xzf "$INPUTS_ARCHIVE" -C "$INPUTS_DIR"
}

INPUTS_NOTE="inputs:     $INPUTS_DIR/INPUTS-$YEAR"

if download "$BASE_URL/raw-result-analysis.csv.tar.gz" "$RESULTS_TGZ"; then
    download "$BASE_URL/GlobalSummary.csv.tar.gz" "$SUMMARY_TGZ" || true
    extract_results
else
    echo "WARN: result CSVs not published yet for $YEAR" >&2
fi

if [ "$FETCH_INPUTS" -eq 1 ]; then
    if download "$BASE_URL/INPUTS-$YEAR.tar.gz" "$INPUTS_ARCHIVE"; then
        if [ "$EXTRACT_INPUTS" -eq 1 ]; then
            extract_inputs
        fi
    else
        INPUTS_NOTE="inputs:     NOT PUBLISHED YET for $YEAR (use the prior year's corpus as a proxy: --year $((YEAR-1)))"
    fi
fi

cat <<EOF
MCC $YEAR fetch complete.
root:       $ROOT
$INPUTS_NOTE
answer key: $RESULTS_DIR/raw-result-analysis.csv
summary:    $RESULTS_DIR/GlobalSummary.csv
EOF
