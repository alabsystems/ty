#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# Ensure every model dir under the MCC models-root has all property XML
# files alongside model.pnml. The v4 broad measurement corpus only had
# model.pnml per directory (no property XMLs), which made the 7 property
# examinations (CTL/LTL/Reachability cardinality + fireability + UpperBounds)
# physically unmeasurable on that corpus. See
# docs/mcc-2026/ctl-ltl-reachability-formula-diagnosis-2026-05-24.md.
#
# Idempotent: re-running is safe. Existing XMLs are not re-extracted.
#
# Default layout:
#   MODELS_ROOT  = /private/tmp/mcc-models-root        (the symlink corpus)
#   ARCHIVES_DIR = ~/mcc-benchmarks/2024/inputs/INPUTS (.tgz per model name)
#
# For each model dir M under MODELS_ROOT (which may be a symlink), we look
# for ARCHIVES_DIR/${M}.tgz, then extract only its *.xml entries into the
# resolved model dir (so symlinks stay valid and the XML lives next to the
# existing model.pnml).

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/private/tmp/mcc-models-root}"
ARCHIVES_DIR="${ARCHIVES_DIR:-$HOME/mcc-benchmarks/2024/inputs/INPUTS}"

# Property XMLs the 7 property examinations dispatch on. Listed in canonical
# MCC order so logs read consistently against the diagnosis doc.
PROPERTY_XMLS=(
    ReachabilityCardinality.xml
    ReachabilityFireability.xml
    CTLCardinality.xml
    CTLFireability.xml
    LTLCardinality.xml
    LTLFireability.xml
    UpperBounds.xml
)

usage() {
    cat <<'USAGE'
Usage: scripts/setup-property-xmls.sh [--models-root DIR] [--archives DIR]

Ensures every model dir under MODELS_ROOT has the 7 MCC property XML files
alongside model.pnml. Extracts only *.xml entries from the matching .tgz in
ARCHIVES_DIR. Idempotent.

Environment / flags:
  --models-root DIR   Root of the model-dir symlinks (default: /private/tmp/mcc-models-root)
  --archives DIR      Source .tgz archives (default: ~/mcc-benchmarks/2024/inputs/INPUTS)
  -h, --help          Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --models-root) MODELS_ROOT="$2"; shift 2 ;;
        --archives)    ARCHIVES_DIR="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ ! -d "$MODELS_ROOT" ]; then
    echo "ERROR: models root not found: $MODELS_ROOT" >&2
    exit 2
fi
if [ ! -d "$ARCHIVES_DIR" ]; then
    echo "ERROR: archives dir not found: $ARCHIVES_DIR" >&2
    exit 2
fi

TOTAL=0
ALREADY=0
EXTRACTED=0
MISSING_ARCHIVE=0
FULL=0
PARTIAL=0
EMPTY=0

# We iterate the immediate children of MODELS_ROOT, which may be symlinks
# into one or more extraction trees. We resolve each child to its target
# directory before extracting, so the extracted XMLs land next to the
# already-present model.pnml.
for entry in "$MODELS_ROOT"/*; do
    [ -e "$entry" ] || continue
    name="$(basename "$entry")"
    target="$(cd "$entry" 2>/dev/null && pwd -P)" || continue
    if [ ! -f "$target/model.pnml" ]; then
        continue
    fi
    TOTAL=$((TOTAL + 1))

    archive="$ARCHIVES_DIR/${name}.tgz"
    have_count=0
    for xml in "${PROPERTY_XMLS[@]}"; do
        if [ -f "$target/$xml" ]; then
            have_count=$((have_count + 1))
        fi
    done

    if [ "$have_count" -eq "${#PROPERTY_XMLS[@]}" ]; then
        ALREADY=$((ALREADY + 1))
        FULL=$((FULL + 1))
        continue
    fi

    if [ ! -f "$archive" ]; then
        MISSING_ARCHIVE=$((MISSING_ARCHIVE + 1))
        if [ "$have_count" -gt 0 ]; then
            PARTIAL=$((PARTIAL + 1))
        else
            EMPTY=$((EMPTY + 1))
        fi
        continue
    fi

    # Extract only the *.xml entries for this model, strip the leading
    # "<name>/" component so files land directly in $target.
    if tar -xzf "$archive" \
            --strip-components=1 \
            -C "$target" \
            --include="${name}/*.xml" 2>/dev/null; then
        EXTRACTED=$((EXTRACTED + 1))
    else
        # Some BSD tars want patterns without --include; fall back to a
        # filter list using GNU-style globbing through stdin.
        tmpdir="$(mktemp -d)"
        tar -xzf "$archive" -C "$tmpdir"
        if [ -d "$tmpdir/$name" ]; then
            cp -n "$tmpdir/$name"/*.xml "$target"/ 2>/dev/null || true
            EXTRACTED=$((EXTRACTED + 1))
        fi
        rm -rf "$tmpdir"
    fi

    have_count=0
    for xml in "${PROPERTY_XMLS[@]}"; do
        if [ -f "$target/$xml" ]; then
            have_count=$((have_count + 1))
        fi
    done
    if [ "$have_count" -eq "${#PROPERTY_XMLS[@]}" ]; then
        FULL=$((FULL + 1))
    elif [ "$have_count" -gt 0 ]; then
        PARTIAL=$((PARTIAL + 1))
    else
        EMPTY=$((EMPTY + 1))
    fi
done

# Final verification pass: count from scratch so the totals are an audit,
# not an accumulator that could lie if the loop exited early.
FINAL_TOTAL=0
FINAL_FULL=0
FINAL_PARTIAL=0
FINAL_EMPTY=0
for entry in "$MODELS_ROOT"/*; do
    [ -e "$entry" ] || continue
    target="$(cd "$entry" 2>/dev/null && pwd -P)" || continue
    [ -f "$target/model.pnml" ] || continue
    FINAL_TOTAL=$((FINAL_TOTAL + 1))
    have=0
    for xml in "${PROPERTY_XMLS[@]}"; do
        [ -f "$target/$xml" ] && have=$((have + 1))
    done
    if [ "$have" -eq "${#PROPERTY_XMLS[@]}" ]; then
        FINAL_FULL=$((FINAL_FULL + 1))
    elif [ "$have" -gt 0 ]; then
        FINAL_PARTIAL=$((FINAL_PARTIAL + 1))
    else
        FINAL_EMPTY=$((FINAL_EMPTY + 1))
    fi
done

cat <<EOF
setup-property-xmls.sh complete
  models_root        : $MODELS_ROOT
  archives_dir       : $ARCHIVES_DIR
  walked             : $TOTAL
  already_complete   : $ALREADY
  newly_extracted    : $EXTRACTED
  missing_archive    : $MISSING_ARCHIVE
  ----
  final_total        : $FINAL_TOTAL
  final_full_xmls    : $FINAL_FULL
  final_partial_xmls : $FINAL_PARTIAL
  final_no_xmls      : $FINAL_EMPTY
EOF

if [ "$FINAL_FULL" -eq 0 ]; then
    echo "WARNING: no model dir ended up with full property XMLs" >&2
    exit 1
fi
