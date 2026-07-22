#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# MCC examination-name parity guard (structural fence).
#
# The 13 MCC examination names are the canonical protocol vocabulary.
# Rust has `tla_petri::examination::Examination` (enum with `as_str()`
# and `ALL: [Examination; 13]`) as the source of truth.
#
# All former Python sites that duplicated this list have been replaced by
# in-tree Rust binaries (`ty-mcc-history`, `ty-mcc-sweep`,
# `ty-mcc-validate`) that route every examination name through the
# `Examination` enum. There is no remaining Python `EXAMS` parity site.
#
# This guard now serves two purposes:
#   1. Confirm the Rust enum still matches the canonical 13-name list.
#   2. Fail closed if any Python `EXAMS = [...]` tuple is reintroduced
#      under crates/, scripts/, mcc/, or tests/ — that would be a
#      regression of the cross-language drift class that produced the
#      qualification-1 bug.
#
# Run locally: ./scripts/mcc_examination_parity.sh
# Wired in: pre-commit (see .pre-commit-config.yaml) and via the Rust
# integration test `crates/tla-petri/tests/mcc_examination_parity.rs`.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CANONICAL=(
    ReachabilityDeadlock
    ReachabilityCardinality
    ReachabilityFireability
    CTLCardinality
    CTLFireability
    LTLCardinality
    LTLFireability
    StateSpace
    OneSafe
    QuasiLiveness
    StableMarking
    UpperBounds
    Liveness
)
CANONICAL_SORTED="$(printf '%s\n' "${CANONICAL[@]}" | LC_ALL=C sort)"

if [ "${#CANONICAL[@]}" -ne 13 ]; then
    printf 'mcc_examination_parity: canonical list size is %d, MCC has 13.\n' "${#CANONICAL[@]}" >&2
    exit 1
fi

# Extract a quoted python-string EXAMS tuple from a file.
extract_python_exams() {
    local path="$1"
    awk '
        BEGIN { in_block = 0 }
        /^EXAMS *= *[\[(]/ { in_block = 1; next }
        in_block && /^[)\]]/ { in_block = 0 }
        in_block {
            line = $0
            while (match(line, /"[A-Za-z]+"/)) {
                token = substr(line, RSTART + 1, RLENGTH - 2)
                print token
                line = substr(line, RSTART + RLENGTH)
            }
        }
    ' "$path" | LC_ALL=C sort
}

# Check Rust source of truth — fetch every `"<Examination>"` arm from
# the `as_str` body. This is a string-match audit; the canonical proof
# is the enum's `ALL` array, but a textual match also catches stale
# entries in `from_name`.
extract_rust_exams() {
    awk '
        /pub fn as_str\(self\)/ { in_block = 1; next }
        in_block && /^[[:space:]]*}/ { in_block = 0 }
        in_block {
            line = $0
            while (match(line, /"[A-Za-z]+"/)) {
                token = substr(line, RSTART + 1, RLENGTH - 2)
                print token
                line = substr(line, RSTART + RLENGTH)
            }
        }
    ' crates/tla-petri/src/examination_kind.rs | LC_ALL=C sort -u
}

failures=0

# 1) Rust as_str arms must match the canonical list verbatim.
rust_actual="$(extract_rust_exams)"
if [ "$rust_actual" != "$CANONICAL_SORTED" ]; then
    {
        echo "mcc_examination_parity: Rust as_str arms drift from canonical list."
        echo "Expected:"
        printf '  %s\n' "${CANONICAL_SORTED}"
        echo "Got:"
        printf '  %s\n' "$rust_actual"
    } >&2
    failures=1
fi

# 2) No Python `EXAMS = [...]` re-introduction. The three former Python
#    drift sites are gone; any reintroduction is a structural regression
#    (the qualification-1 drift class). Scan production roots and fail
#    if any Python file declares an EXAMS list with MCC-shaped names.
PYTHON_ROOTS=(scripts mcc crates tests)
python_offenders=""
for root in "${PYTHON_ROOTS[@]}"; do
    [ -d "$root" ] || continue
    while IFS= read -r py; do
        [ -f "$py" ] || continue
        py_actual="$(extract_python_exams "$py")"
        if [ -n "$py_actual" ]; then
            python_offenders+="$py"$'\n'
        fi
    done < <(find "$root" -type f -name '*.py' 2>/dev/null)
done
if [ -n "$python_offenders" ]; then
    {
        echo "mcc_examination_parity: Python EXAMS list reintroduced."
        echo "All MCC examination-name routing must go through the Rust"
        echo "Examination enum (crates/tla-petri/src/examination_kind.rs)."
        echo "Offending file(s):"
        printf '%s' "$python_offenders" | sed 's/^/  /'
    } >&2
    failures=1
fi

if [ "$failures" -eq 1 ]; then
    exit 1
fi
echo "mcc_examination_parity: clean (13/13 names match Rust enum; no Python EXAMS sites)."
