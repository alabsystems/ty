#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Full-MCC-benchmark harness: run the improved ty-mcc binary over an entire MCC
# input set at CONTEST budget (default 3600s, the MCC time confinement), in the
# exact contest configuration, collect results into a contest-shaped CSV, and
# (optionally) diff against the official summary_TY.csv to produce the full
# recovery/regression report that certifies improvement at scale.
#
# This is the off-machine run requirement for evaluating "does TY win MCC":
# point --inputs at the full MCC-2026 INPUTS directory (1953 model dirs).
#
# Usage:
#   scripts/mcc_full_benchmark.sh --inputs /path/to/MCC2026/INPUTS \
#       [--bin ./target/release/ty-mcc] [--timeout 3600] [--jobs 1] \
#       [--out /tmp/ty_full_results.csv] [--baseline summary_TY.csv] \
#       [--exams "ReachabilityDeadlock OneSafe ..."] [--models "Anderson-PT-04 ..."]
#
# Each (model, examination) is run through mcc/BenchKit_head.sh — the SAME entry
# point the contest uses — so the timeout/memory/threads/backend-evidence
# semantics match the competition exactly.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/target/release/ty-mcc"
INPUTS=""
TIMEOUT=3600
MEMORY=16384            # MB, MCC default confinement
JOBS=1                  # parallel models; each examination already uses 4 threads
OUT="/tmp/ty_full_results.csv"
BASELINE=""             # contest summary_TY.csv to diff against (optional)
EXAMS="ReachabilityDeadlock OneSafe QuasiLiveness StableMarking Liveness StateSpace UpperBounds ReachabilityCardinality ReachabilityFireability CTLCardinality CTLFireability LTLCardinality LTLFireability"
MODELS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --inputs)   INPUTS="$2"; shift 2;;
        --bin)      BIN="$2"; shift 2;;
        --timeout)  TIMEOUT="$2"; shift 2;;
        --memory)   MEMORY="$2"; shift 2;;
        --jobs)     JOBS="$2"; shift 2;;
        --out)      OUT="$2"; shift 2;;
        --baseline) BASELINE="$2"; shift 2;;
        --exams)    EXAMS="$2"; shift 2;;
        --models)   MODELS="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

[ -n "$INPUTS" ] || { echo "ERROR: --inputs <MCC INPUTS dir> is required" >&2; exit 2; }
[ -x "$BIN" ]    || { echo "ERROR: binary not found/executable: $BIN" >&2; exit 2; }
HEAD="${ROOT}/mcc/BenchKit_head.sh"
[ -x "$HEAD" ]   || { echo "ERROR: $HEAD missing" >&2; exit 2; }

# Optional outer hard-kill guard (belt-and-suspenders beyond the head script's
# own --timeout). GNU coreutils `timeout` exists on the Linux MCC VM; on macOS
# it may be absent (or present as `gtimeout`). When neither exists, rely on the
# inner --timeout (BK_TIME_CONFINEMENT -> ty-mcc --timeout) alone.
TIMEOUT_CMD=""
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD="timeout $((TIMEOUT + 120))"
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD="gtimeout $((TIMEOUT + 120))"
fi

if [ -z "$MODELS" ]; then
    # Portable directory discovery (BSD find lacks -printf).
    MODELS="$(cd "$INPUTS" && for d in */; do [ -d "$d" ] && printf '%s\n' "${d%/}"; done | sort)"
fi
nmodels=$(printf '%s' "$MODELS" | wc -w | tr -d ' ')
echo "ty-mcc full MCC benchmark"
echo "  binary:   $BIN"
echo "  inputs:   $INPUTS  ($nmodels models)"
echo "  timeout:  ${TIMEOUT}s   memory: ${MEMORY}MB   jobs: $JOBS"
echo "  out:      $OUT"
echo "### tool,Input,Examination,results,wall_ms,status" > "$OUT"

# Portable millisecond clock: GNU `date +%s%3N` works on the Linux MCC VM but
# not on BSD/macOS, so fall back to python3 (already required by the comparison)
# then to whole-second resolution.
now_ms() {
    local t
    t="$(date +%s%3N 2>/dev/null)"
    case "$t" in
        ''|*[!0-9]*) python3 -c 'import time;print(int(time.time()*1000))' 2>/dev/null \
                        || echo $(( $(date +%s) * 1000 ));;
        *) printf '%s' "$t";;
    esac
}

run_cell() {
    local model="$1"
    local exam="$2"
    local dir="$INPUTS/$model"
    [ -d "$dir" ] || { echo "MISSING model dir: $dir" >&2; return; }
    local t0 t1 out first status
    t0=$(now_ms)
    out="$(cd "$dir" && \
        BK_EXAMINATION="$exam" BK_TOOL=TY BK_INPUT="$model" \
        BK_TIME_CONFINEMENT="$TIMEOUT" BK_MEMORY_CONFINEMENT="$MEMORY" \
        TY_MCC_BIN="$BIN" TY_MCC_FPSET_BACKEND=cas \
        TY_MCC_REQUIRE_BACKEND_EVIDENCE="${TY_MCC_REQUIRE_BACKEND_EVIDENCE:-0}" \
        $TIMEOUT_CMD bash "$HEAD" 2>/dev/null)"
    t1=$(now_ms)
    first="$(printf '%s\n' "$out" | grep -E '^(FORMULA|STATE_SPACE|CANNOT_COMPUTE|DO_NOT_COMPETE)' | head -1 | tr ',' ' ')"
    case "$first" in
        *CANNOT_COMPUTE*) status=cc;;
        *DO_NOT_COMPETE*) status=dnc;;
        "") status=empty;;
        *) status=ok;;
    esac
    printf 'TY,%s,%s,%s,%s,%s\n' "$model" "$exam" "${first:-NONE}" "$((t1-t0))" "$status" >> "$OUT"
    printf '%-34s %-24s %s\n' "$model" "$exam" "$status"
}
# Fan out per (model, examination) with a portable background-job batch limiter
# (works on macOS bash 3.2 BSD and Linux bash GNU alike — no `xargs -d`/`wait -n`).
# Single-line `>> "$OUT"` appends are < PIPE_BUF, so they stay atomic under
# concurrency; progress lines on stdout may interleave harmlessly.
i=0
for m in $MODELS; do
    for e in $EXAMS; do
        if [ "$JOBS" -le 1 ]; then
            run_cell "$m" "$e"
        else
            run_cell "$m" "$e" &
            i=$((i + 1))
            [ $((i % JOBS)) -eq 0 ] && wait
        fi
    done
done
wait

echo "=== outcome summary ($OUT) ==="
awk -F',' 'NR>1{c[$6]++} END{for(k in c) printf "  %-8s %d\n", k, c[k]}' "$OUT"

if [ -n "$BASELINE" ] && [ -f "$BASELINE" ]; then
    echo "=== recovery vs baseline ($BASELINE) ==="
    python3 "${ROOT}/scripts/mcc_compare_runs.py" "$BASELINE" "$OUT"
fi
