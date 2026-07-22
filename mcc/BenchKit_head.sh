#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

set -u

echoerr() {
    printf '%s\n' "$*" >&2
}

# The MCC answer parser requires the literal keyword `DO_NOT_COMPETE`
# (with underscores) alone on a single line. The qualification-1
# rejection (May 2026) was caused by the previous spaced variant. See
# docs/mcc-2026/qualification-1/analysis.md.
do_not_compete() {
    echoerr "$1"
    printf 'DO_NOT_COMPETE\n'
    exit 0
}

is_positive_int() {
    [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

truthy() {
    case "$(trim "${1:-}")" in
        1 | true | TRUE | yes | YES | on | ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

trim() {
    local value="${1:-}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

known_examination() {
    case "$1" in
        ReachabilityDeadlock | OneSafe | QuasiLiveness | StableMarking | Liveness | StateSpace | \
            UpperBounds | ReachabilityCardinality | ReachabilityFireability | \
            CTLCardinality | CTLFireability | LTLCardinality | LTLFireability)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

select_tool_from_dir() {
    local dir="${1%/}"
    if [ -x "$dir/ty-mcc" ]; then
        printf '%s\n' "$dir/ty-mcc"
    elif [ -x "$dir/pnml-tools" ]; then
        printf '%s\n' "$dir/pnml-tools"
    elif [ -x "$dir/ty" ]; then
        printf '%s\n' "$dir/ty"
    fi
}

direct_mcc_tool() {
    case "$(basename "$1")" in
        ty-mcc | pnml-tools)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Per-formula / per-StateSpace `CANNOT_COMPUTE` line.
#
# Both `STATE_SPACE` and `CANNOT_COMPUTE` MUST be underscored — the MCC
# answer parser tokenises spaced keywords and silently misclassifies
# crashes as participation. See docs/mcc-2026/qualification-1/analysis.md.
#
# For pure tool-level crashes (no useful per-formula result available)
# emit `CANNOT_COMPUTE` alone on a line — see `tool_level_cannot_compute`.
cannot_compute() {
    local exam="$2"
    if [ "$exam" = "StateSpace" ]; then
        printf 'STATE_SPACE CANNOT_COMPUTE TECHNIQUES EXPLICIT\n'
    else
        printf 'FORMULA %s CANNOT_COMPUTE TECHNIQUES EXPLICIT\n' "$exam"
    fi
}

# Tool-level `CANNOT_COMPUTE` — alone on a single line, no `FORMULA`
# prefix, no `TECHNIQUES` suffix. This is what the parser expects when
# the tool detects it cannot run the examination at all.
tool_level_cannot_compute() {
    printf 'CANNOT_COMPUTE\n'
}

# Packaged-ay-revision freshness check now lives inside
# `ty-mcc-backend-evidence-validate --require-ay-rev <REV>`. The Python
# heredoc was the last cross-language MCC entry point in ty; the Rust
# validator walks `report.evidence` rows the same way and rejects when
# all observed `current_ay_rev=` tokens fail to match the expected rev.
# Kept as a no-op stub so that any external callers still naming this
# function get a clear error rather than a silent shell function-not-found.
validate_packaged_ay_revision() {
    echoerr "validate_packaged_ay_revision: use ty-mcc-backend-evidence-validate --require-ay-rev <REV>"
    return 1
}

validate_backend_evidence_sidecar() {
    local path="$1"
    local validator
    local checks
    local args=()
    local check
    local output
    local expected_ay_rev

    if [ ! -s "$path" ]; then
        echoerr "backend evidence sidecar is missing or empty: $path"
        return 1
    fi

    # Resolution order:
    #   1. TY_MCC_BACKEND_EVIDENCE_VALIDATOR env var (must be a Rust
    #      binary — the Python validator was removed in #4509).
    #   2. The in-tree Rust binary ty-mcc-backend-evidence-validate on
    #      PATH.
    #   3. /usr/local/bin/ty-mcc-backend-evidence-validate (the path
    #      packaged inside the official MCC VM image).
    validator="$(trim "${TY_MCC_BACKEND_EVIDENCE_VALIDATOR:-}")"
    if [ -z "$validator" ] && command -v ty-mcc-backend-evidence-validate >/dev/null 2>&1; then
        validator="$(command -v ty-mcc-backend-evidence-validate)"
    fi
    if [ -z "$validator" ] && [ -x /usr/local/bin/ty-mcc-backend-evidence-validate ]; then
        validator=/usr/local/bin/ty-mcc-backend-evidence-validate
    fi
    if [ -z "$validator" ]; then
        echoerr "ty-mcc-backend-evidence-validate not found (set TY_MCC_BACKEND_EVIDENCE_VALIDATOR)"
        return 1
    fi
    if [ ! -x "$validator" ] && ! command -v "$validator" >/dev/null 2>&1; then
        echoerr "backend evidence validator is not executable: $validator"
        return 1
    fi

    checks="$(trim "${TY_MCC_BACKEND_EVIDENCE_REQUIRED_CHECKS:-}")"
    for check in $checks; do
        args+=(--require "$check")
    done

    # Fold the packaged-ay-rev freshness check into the same validator
    # invocation. The Rust binary tokenises `report.evidence` rows and
    # rejects when no `current_ay_rev=<value>` matches.
    expected_ay_rev="$(trim "${TY_MCC_PACKAGED_AY_REV:-}")"
    if [ -n "$expected_ay_rev" ]; then
        if [[ ! "$expected_ay_rev" =~ ^[0-9a-f]{40}$ ]]; then
            echoerr "invalid packaged AY revision: $expected_ay_rev"
            return 1
        fi
        args+=(--require-ay-rev "$expected_ay_rev")
    fi

    output="$("$validator" "${args[@]}" "$path" 2>&1)" || {
        echoerr "$output"
        return 1
    }
    echoerr "$output"
}

exam="$(trim "${BK_EXAMINATION:-}")"
[ -n "$exam" ] || do_not_compete "BK_EXAMINATION is not set"

if ! known_examination "$exam"; then
    do_not_compete "unsupported MCC examination: $exam"
fi

# Capability gate: refuse to run if the host architecture has no working
# trust-cg codegen backend AND the build was wired to require native. The
# default MCC binary uses the interpreter path so this only fires when an
# operator explicitly opts in to native via TY_MCC_REQUIRE_NATIVE=1.
# Without this gate, an unsupported arch would silently produce wrong
# native code (the `unknown-host` sentinel in tla-trust-cg — see
# docs/mcc-2026/qualification-1/analysis.md).
require_native="$(trim "${TY_MCC_REQUIRE_NATIVE:-0}")"
if truthy "$require_native"; then
    host_arch="$(uname -m 2>/dev/null || printf unknown)"
    case "$host_arch" in
        arm64 | aarch64) ;; # trust-cg backend present
        *)
            echoerr "TY_MCC_REQUIRE_NATIVE=1 but trust-cg has no codegen backend for host arch '$host_arch' (only aarch64 supported as of 2026-05-17). Emitting CANNOT_COMPUTE."
            printf 'CANNOT_COMPUTE\n'
            exit 0
            ;;
    esac
fi

bk_tool="$(trim "${BK_TOOL:-TY}")"
if [ -z "$bk_tool" ]; then
    bk_tool="TY"
fi

if [ -n "${BK_LOG_FILE:-}" ]; then
    mkdir -p "$(dirname "$BK_LOG_FILE")" 2>/dev/null || true
    {
        printf 'BK_TOOL=%s\n' "$bk_tool"
        printf 'BK_EXAMINATION=%s\n' "$exam"
        printf 'BK_TIME_CONFINEMENT=%s\n' "${BK_TIME_CONFINEMENT:-}"
        printf 'BK_MEMORY_CONFINEMENT=%s\n' "${BK_MEMORY_CONFINEMENT:-}"
        printf 'BK_BIN_PATH=%s\n' "${BK_BIN_PATH:-}"
    } > "$BK_LOG_FILE" 2>/dev/null || true
fi

if [ -n "${BK_BIN_PATH:-}" ]; then
    PATH="${BK_BIN_PATH%/}:$PATH"
    export PATH
fi

: "${TY_MCC_PACKAGED_AY_REV:=0adeaab4d66b1414a95ab5cee4ec64078c9dbd97}"
: "${TY_MCC_BACKEND_EVIDENCE_REQUIRED_CHECKS:=mcc_ay_symbolic_execution native_jit_fail_closed_gate trust_ir_transport_identity trust_cg_native_admission ay_solve_decision_profile hardware_proof_replay_boundary hardware_replay_decision trust_cg_compile_artifact_cache_telemetry trust_cg_host_jit_pgo_provenance trust_cg_call_packet_contract_descriptor portfolio_route ay_solver_capability_descriptor ay_symbolic_execution_contract_manifest trust_ir_native_evidence_artifact_resolution trust_ir_native_semantic_bridge_proof_identity petri_trust_mc_model_acceptance}"
export TY_MCC_PACKAGED_AY_REV
export TY_MCC_BACKEND_EVIDENCE_REQUIRED_CHECKS

for skipped in ${TY_MCC_SKIP_EXAMINATIONS:-}; do
    if [ "$exam" = "$skipped" ]; then
        do_not_compete "explicitly skipped MCC examination: $exam"
    fi
done

raw_input="$(trim "${BK_INPUT:-}")"
if [ -n "$raw_input" ] && [ -e "$raw_input" ]; then
    input="$raw_input"
elif [ -f model.pnml ]; then
    input="$(pwd)"
elif [ -n "$raw_input" ]; then
    input="$raw_input"
else
    input="$(pwd)"
fi

tool="${TY_MCC_BIN:-}"
if [ -z "$tool" ] && [ -n "${BK_BIN_PATH:-}" ]; then
    tool="$(select_tool_from_dir "$BK_BIN_PATH")"
fi
if [ -z "$tool" ]; then
    tool="$(select_tool_from_dir "/usr/local/bin")"
fi

if [ ! -x "$tool" ]; then
    if command -v ty-mcc >/dev/null 2>&1; then
        tool="$(command -v ty-mcc)"
    elif command -v pnml-tools >/dev/null 2>&1; then
        tool="$(command -v pnml-tools)"
    elif command -v ty >/dev/null 2>&1; then
        tool="$(command -v ty)"
    else
        # Tool-level failure: missing binary means we cannot run the
        # examination at all. Per the MCC protocol (Fabrice's qual-1
        # feedback) this must emit `CANNOT_COMPUTE` alone on a line —
        # NOT a per-formula `FORMULA … CANNOT_COMPUTE …` wrapper line,
        # which is exactly the "inside a result line" form he flagged.
        echoerr "TY MCC binary not found: $tool"
        tool_level_cannot_compute
        exit 0
    fi
fi

# Default to host core count so we don't waste cores on 8/16-core competition
# VMs. Per-examination overrides remain available via TY_MCC_THREADS (highest
# precedence) and BK_CORES (set by MCC's BenchKit when present).
host_cores=""
if command -v nproc >/dev/null 2>&1; then
    host_cores="$(nproc 2>/dev/null || true)"
elif command -v sysctl >/dev/null 2>&1; then
    host_cores="$(sysctl -n hw.ncpu 2>/dev/null || true)"
fi
is_positive_int "$host_cores" || host_cores=4
threads="${TY_MCC_THREADS:-${BK_CORES:-$host_cores}}"
is_positive_int "$threads" || threads="$host_cores"

memory_fraction="${TY_MCC_MEMORY_FRACTION:-0.70}"
storage_dir="${TY_MCC_STORAGE_DIR:-${TMPDIR:-/tmp}/ty-mcc-storage}"
mkdir -p "$storage_dir" 2>/dev/null || true
fpset_backend="$(trim "${TY_MCC_FPSET_BACKEND:-cas}")"
fpset_backend="$(printf '%s' "$fpset_backend" | tr '[:upper:]' '[:lower:]')"
case "$fpset_backend" in
    cas | sharded)
        ;;
    *)
        echoerr "unsupported TY_MCC_FPSET_BACKEND: $fpset_backend"
        cannot_compute "$input" "$exam"
        exit 0
        ;;
esac
export TY_MCC_FPSET_BACKEND="$fpset_backend"
backend_evidence_jsonl="$(trim "${TY_MCC_BACKEND_EVIDENCE_JSONL:-}")"
if [ -z "$backend_evidence_jsonl" ]; then
    backend_evidence_jsonl="$(trim "${MCC_BACKEND_EVIDENCE_JSONL:-}")"
fi
if [ -z "$backend_evidence_jsonl" ]; then
    backend_evidence_jsonl="${storage_dir%/}/backend-capability.jsonl"
fi
mkdir -p "$(dirname "$backend_evidence_jsonl")" 2>/dev/null || true
export TY_MCC_BACKEND_EVIDENCE_JSONL="$backend_evidence_jsonl"
export MCC_BACKEND_EVIDENCE_JSONL="$backend_evidence_jsonl"

require_backend_evidence_raw="$(trim "${TY_MCC_REQUIRE_BACKEND_EVIDENCE:-${MCC_REQUIRE_BACKEND_EVIDENCE:-1}}")"
case "$require_backend_evidence_raw" in
    0 | false | FALSE | no | NO | off | OFF)
        require_backend_evidence=0
        ;;
    *)
        require_backend_evidence=1
        ;;
esac
if [ "$require_backend_evidence" -eq 1 ]; then
    require_backend_evidence=1
    if [ -e "$backend_evidence_jsonl" ] && ! rm -f "$backend_evidence_jsonl" 2>/dev/null; then
        echoerr "failed to remove stale backend evidence sidecar: $backend_evidence_jsonl"
        tool_level_cannot_compute
        exit 0
    fi
fi

: "${TY_MCC_ENABLE_BMC_DEPTH1_CHUNKING:=1}"
: "${TY_MCC_BMC_DEPTH1_CHUNK_SIZE:=4}"
export TY_MCC_ENABLE_BMC_DEPTH1_CHUNKING
export TY_MCC_BMC_DEPTH1_CHUNK_SIZE

ulimit_kb="${TY_MCC_ULIMIT_KB:-15000000}"
if is_positive_int "$ulimit_kb"; then
    ulimit -v "$ulimit_kb" 2>/dev/null || true
fi

timeout_args=()
total_time="${BK_TIME_CONFINEMENT:-}"
if is_positive_int "$total_time"; then
    # The Rust MCC runtime applies its own safety margin before the deadline.
    timeout_args=(--timeout "$total_time")
fi

if direct_mcc_tool "$tool"; then
    cmd=(
        "$tool" "$input"
        --examination "$exam"
        --threads "$threads"
        --memory-fraction "$memory_fraction"
        --storage auto
        --storage-dir "$storage_dir"
    )
else
    cmd=(
        "$tool" mcc "$input"
        --examination "$exam"
        --threads "$threads"
        --memory-fraction "$memory_fraction"
        --storage auto
        --storage-dir "$storage_dir"
    )
fi
if [ "${#timeout_args[@]}" -gt 0 ]; then
    cmd+=("${timeout_args[@]}")
fi

echoerr "TY MCC"
echoerr "Host: $(hostname 2>/dev/null || printf unknown)"
echoerr "BK_TOOL: $bk_tool"
echoerr "Input: $input"
echoerr "Examination: $exam"
echoerr "Threads: $threads"
echoerr "Storage: $storage_dir"
echoerr "Fingerprint set backend: $fpset_backend"
echoerr "Backend evidence: $backend_evidence_jsonl"

child_pid=""
stdout_capture=""
terminate() {
    if [ -n "$child_pid" ]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    if [ -n "$stdout_capture" ]; then
        rm -f "$stdout_capture" 2>/dev/null || true
    fi
    tool_level_cannot_compute
    exit 0
}
trap terminate TERM INT HUP

stdout_capture="$(mktemp "${storage_dir%/}/ty-mcc-stdout.XXXXXX")" || {
    echoerr "failed to create stdout capture file"
    tool_level_cannot_compute
    exit 0
}
"${cmd[@]}" > "$stdout_capture" &
child_pid=$!
wait "$child_pid"
status=$?
child_pid=""

if [ "$status" -ne 0 ]; then
    # Tool-level failure (crashed mid-examination): emit `CANNOT_COMPUTE`
    # alone on a line per the qual-1 protocol clarification from Fabrice
    # ("must be in a single line, not inside a result line"). Per-formula
    # `FORMULA … CANNOT_COMPUTE …` would be the result-line shape he
    # rejected — that was the original qual-1 bug class.
    echoerr "$(basename "$tool") exited with status $status"
    rm -f "$stdout_capture" 2>/dev/null || true
    tool_level_cannot_compute
    exit 0
fi

if [ "$require_backend_evidence" -eq 1 ]; then
    if ! validate_backend_evidence_sidecar "$backend_evidence_jsonl"; then
        # Tool-level failure: backend-evidence sidecar invariant broken
        # means we cannot trust the run. Same protocol as the crash path.
        echoerr "backend evidence validation failed; failing closed"
        rm -f "$stdout_capture" 2>/dev/null || true
        tool_level_cannot_compute
        exit 0
    fi
fi

cat "$stdout_capture"
rm -f "$stdout_capture" 2>/dev/null || true

exit 0
