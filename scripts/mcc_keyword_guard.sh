#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# MCC keyword regression guard.
#
# The MCC 2026 answer parser requires every protocol keyword to use the
# UPPERCASE_WITH_UNDERSCORES form: CANNOT_COMPUTE, DO_NOT_COMPETE,
# STATE_SPACE, MAX_TOKEN_IN_PLACE, MAX_TOKEN_PER_MARKING. The May 2026
# qualification-1 rejection was caused by emitting the spaced variants.
# Narrative below names the spaced form for context:
# mcc-keyword-guard: allow-spaced-mention
# (`CANNOT COMPUTE`, `STATE SPACE`, ...). See
# docs/mcc-2026/qualification-1/analysis.md.
#
# This script greps every production source file in the repo and fails
# if any spaced variant appears. Files where the spaced form is
# legitimate (negative test assertions, legacy parsers that accept old
# data, this script itself, the keyword module's doc-comment narrative)
# must opt in via the directive:
#
#     mcc-keyword-guard: allow-spaced-mention
#
# on its own line anywhere in the file.
#
# Run locally: ./scripts/mcc_keyword_guard.sh

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Build the forbidden-patterns list at runtime so this script is not
# self-flagging.
# mcc-keyword-guard: allow-spaced-mention
# (no `CANNOT COMPUTE` etc. as a literal in the source).
SP=' '
FORBIDDEN=(
    "CANNOT${SP}COMPUTE"
    "DO${SP}NOT${SP}COMPETE"
    "STATE${SP}SPACE"
    "MAX${SP}TOKEN${SP}IN${SP}PLACE"
    "MAX${SP}TOKEN${SP}PER${SP}MARKING"
)

# File roots to scan. Production sources only — docs/, reports/, vendored
# trees, and build outputs are excluded.
ROOTS=(crates scripts mcc tests)
EXTENSIONS=(rs sh py)

ALLOW_DIRECTIVE='mcc-keyword-guard: allow-spaced-mention'

# Collect candidate files.
files_tmp=$(mktemp -t mcc-kg-files.XXXXXX)
trap 'rm -f "$files_tmp"' EXIT
for root in "${ROOTS[@]}"; do
    [ -d "$root" ] || continue
    find "$root" \
        \( -name target -o -name .git -o -name third_party \
            -o -name tla_baseline_corpus -o -name node_modules \) -prune \
        -o -type f \( -name '*.rs' -o -name '*.sh' -o -name '*.py' \) -print \
        >>"$files_tmp"
done

python3 - "$files_tmp" "$ALLOW_DIRECTIVE" <<'PY'
import pathlib
import sys

files_list = pathlib.Path(sys.argv[1])
allow = sys.argv[2]
sp = " "
patterns = [
    "CANNOT" + sp + "COMPUTE",
    "DO" + sp + "NOT" + sp + "COMPETE",
    "STATE" + sp + "SPACE",
    "MAX" + sp + "TOKEN" + sp + "IN" + sp + "PLACE",
    "MAX" + sp + "TOKEN" + sp + "PER" + sp + "MARKING",
]

hits: dict[str, list[str]] = {}
for raw_path in files_list.read_text(encoding="utf-8").splitlines():
    path = pathlib.Path(raw_path)
    if not path.is_file():
        continue
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        continue
    marker_lines = {idx for idx, line in enumerate(lines, start=1) if allow in line}
    for idx, line in enumerate(lines, start=1):
        if not any(pattern in line for pattern in patterns):
            continue
        if any((idx - offset) in marker_lines for offset in range(4)):
            continue
        hits.setdefault(raw_path, []).append(f"{idx}:{line}")

if hits:
    print("MCC keyword regression guard: forbidden spaced keyword(s) found.", file=sys.stderr)
    print("", file=sys.stderr)
    print("All MCC protocol keywords must use underscores:", file=sys.stderr)
    print("  CANNOT_COMPUTE  DO_NOT_COMPETE  STATE_SPACE", file=sys.stderr)
    print("  MAX_TOKEN_IN_PLACE  MAX_TOKEN_PER_MARKING", file=sys.stderr)
    print("", file=sys.stderr)
    print("Fix: route every emit site through crates/tla-petri/src/mcc_keywords.rs", file=sys.stderr)
    print(f"  or add the directive '{allow}'", file=sys.stderr)
    print("  on the same line or the line immediately above the literal.", file=sys.stderr)
    print("  (File-wide opt-out is no longer supported - too coarse.)", file=sys.stderr)
    print("", file=sys.stderr)
    for path, path_hits in hits.items():
        print(f"  {path}:", file=sys.stderr)
        for hit in path_hits:
            print(f"    {hit}", file=sys.stderr)
    sys.exit(1)
PY
echo "MCC keyword guard: clean (no spaced MCC keywords in production sources)."
