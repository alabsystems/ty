#!/usr/bin/env bash
# End-to-end MCC test: run our BenchKit wrapper against the mutex fixture
# for one examination, capture stdout, validate it parses as canonical MCC
# output (underscored keywords, no spaced variants).
#
# Usage: mcc_e2e_test.sh <path-to-ty-mcc-binary> <path-to-input-dir> <examination>

set -eu

BINARY="${1:?usage: $0 <binary> <input-dir> <examination>}"
INPUT="${2:?usage: $0 <binary> <input-dir> <examination>}"
EXAM="${3:?usage: $0 <binary> <input-dir> <examination>}"

if [ ! -x "$BINARY" ]; then
    echo "FAIL: binary $BINARY missing or not executable" >&2
    exit 1
fi
if [ ! -d "$INPUT" ] && [ ! -f "$INPUT/model.pnml" ]; then
    echo "FAIL: input dir $INPUT missing or no model.pnml" >&2
    exit 1
fi

echo "=== running $(basename "$BINARY") on $INPUT for examination=$EXAM ==="
set +e
out_file=$(mktemp -t mcc-stdout.XXXXXX)
err_file=$(mktemp -t mcc-stderr.XXXXXX)
"$BINARY" "$INPUT" --examination "$EXAM" --timeout 60 \
    > "$out_file" 2> "$err_file"
status=$?
set -e

echo ""
echo "=== STDOUT (MCC answer to BenchKit) ==="
cat "$out_file"
echo ""
echo "=== exit status: $status ==="
echo ""

# Validate canonical underscored MCC keywords are present and no
# spaced variants. Build the forbidden tokens at runtime so a
# textual rewriter can't accidentally make this a tautology.
SP=' '
BAD_CANNOT="CANNOT${SP}COMPUTE"
BAD_STATE="STATE${SP}SPACE"
BAD_DONT="DO${SP}NOT${SP}COMPETE"
fail=0
if grep -q -E "(${BAD_CANNOT}|${BAD_STATE}|${BAD_DONT})" "$out_file"; then
    echo "FAIL: stdout contains spaced MCC keyword (qual-1 regression)"
    fail=1
fi
# Verify at least one canonical token shape appears (FORMULA, STATE_SPACE, or alone-line keyword)
if ! grep -qE '^(FORMULA |STATE_SPACE |CANNOT_COMPUTE$|DO_NOT_COMPETE$)' "$out_file"; then
    echo "FAIL: stdout has no recognisable canonical MCC line"
    fail=1
fi

# Strict protocol + verdict check via the Rust validator (replaces the
# previous scripts/mcc_validate.py Python cross-check). Only invoked
# when an expected.json fixture sits alongside model.pnml in the input
# directory — older smoke fixtures without one keep the loose shape
# check above.
if [ -f "$INPUT/expected.json" ]; then
    VALIDATOR_DIR="$(dirname "$BINARY")"
    VALIDATOR="${TY_MCC_VALIDATE_BIN:-$VALIDATOR_DIR/ty-mcc-validate}"
    if [ ! -x "$VALIDATOR" ]; then
        echo "FAIL: ty-mcc-validate binary $VALIDATOR missing or not executable" >&2
        fail=1
    elif ! "$VALIDATOR" "$out_file" "$INPUT/expected.json" "$EXAM"; then
        echo "FAIL: ty-mcc-validate rejected stdout for $EXAM"
        fail=1
    fi
fi

# Treat a non-zero binary exit as a hard failure even if the stdout
# happens to contain a recognizable line. A crashing binary that
# prints one good line before dying must not return success — that's
# exactly the class of false-positive that lets qual-1-style bugs
# survive a "smoke passed" claim.
if [ "$status" -ne 0 ]; then
    echo "FAIL: $EXAM binary exited with status $status"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "PASS: $EXAM produced canonical MCC output"
fi

rm -f "$out_file" "$err_file"
exit "$fail"
