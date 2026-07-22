#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

set -euo pipefail

# Official MCC 2026 final guard.
#
# The official submission contract is VM/VMDK based: submit a bootable disk image
# named <toolName>-2026.vmdk, with the unmodified mcc2026-input.vmdk mounted by
# the organizers in read-only mode. Docker is intentionally not part of this
# preflight; it is only a local build/smoke convenience.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TOOL_NAME="${MCC_TOOL_NAME:-TY}"
TOOL_KIND="${MCC_TOOL_KIND:-parallel}"
KIT_DIR="${MCC_SUBMISSION_KIT_DIR:-/tmp/mcc2026/work/SubmissionKit-2026}"
OFFICIAL_VMDK="${MCC_PREFLIGHT_OFFICIAL_VMDK:-${KIT_DIR}/${TOOL_NAME}-2026.vmdk}"
INPUT_VMDK="${MCC_PREFLIGHT_INPUT_VMDK:-${KIT_DIR}/mcc2026-input.vmdk}"
SIDECAR="${MCC_PREFLIGHT_SHA256_SIDECAR:-${OFFICIAL_VMDK}.sha256}"
BENCHKIT_START="${MCC_PREFLIGHT_BENCHKIT_START:-${KIT_DIR}/BenchKitStart.sh}"
SSH_KEY="${MCC_PREFLIGHT_SSH_KEY:-${KIT_DIR}/bk-private_key}"
SUBMISSION_MANUAL="${MCC_SUBMISSION_MANUAL:-/tmp/mcc2026/kit-docs/SubmissionKit-2026/MCC2026-SubmissionManual.pdf}"
EXPECTED_INPUT_SHA256="${MCC_PREFLIGHT_EXPECTED_INPUT_SHA256:-}"
EXPECTED_OFFICIAL_SHA256="${MCC_PREFLIGHT_EXPECTED_OFFICIAL_SHA256:-}"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

ok() {
    printf 'OK: %s\n' "$*"
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

require_file() {
    local path="$1"
    [ -f "$path" ] || fail "missing file: $path"
    [ -r "$path" ] || fail "file is not readable: $path"
}

sha256_file() {
    local path="$1"
    local output

    output="$(shasum -a 256 "$path")" || fail "failed to hash $path"
    printf '%s\n' "${output%% *}"
}

require_digest() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local actual

    [ -n "$expected" ] || return 0
    actual="$(sha256_file "$path")"
    [ "$actual" = "$expected" ] || fail "$label SHA-256 mismatch: expected $expected, got $actual"
    ok "$label SHA-256 matches $expected"
}

check_tool_identity() {
    case "$TOOL_NAME" in
        ""|*"/"*|*".."*|*" "*)
            fail "invalid MCC tool name: ${TOOL_NAME:-<empty>}"
            ;;
    esac
    case "$TOOL_KIND" in
        sequential|parallel)
            ;;
        *)
            fail "MCC_TOOL_KIND must be sequential or parallel, got: $TOOL_KIND"
            ;;
    esac

    local expected_name="${TOOL_NAME}-2026.vmdk"
    local actual_name
    actual_name="$(basename "$OFFICIAL_VMDK")"
    [ "$actual_name" = "$expected_name" ] \
        || fail "official VMDK must be named ${expected_name}, got ${actual_name}"

    ok "tool identity is ${TOOL_NAME}; BK_TOOL must be ${TOOL_NAME}; tool kind is ${TOOL_KIND}"
}

check_official_files() {
    require_file "$OFFICIAL_VMDK"
    require_file "$INPUT_VMDK"
    require_file "$SIDECAR"
    require_file "$BENCHKIT_START"
    require_file "$SSH_KEY"
    if [ -e "$SUBMISSION_MANUAL" ]; then
        require_file "$SUBMISSION_MANUAL"
        ok "submission manual is present: $SUBMISSION_MANUAL"
    fi
    [ "$(basename "$INPUT_VMDK")" = "mcc2026-input.vmdk" ] \
        || fail "input VMDK must be named mcc2026-input.vmdk"
}

check_no_qemu_processes() {
    local ps_output
    local matches=""
    local line
    local pid
    local rest

    # This preflight, the ty-mccctl uploader that spawned it, and the invoking
    # shell legitimately carry the VMDK path in their argv. Collect our own
    # process ancestry so the guard does not trip on its own invocation; a real
    # hypervisor attached to the VMDK is never in this ancestry.
    local skip_pids=" "
    local p=$$
    while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
        skip_pids+="$p "
        p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d '[:space:]')"
    done

    ps_output="$(ps -axo pid=,command=)" || fail "failed to inspect process list"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        read -r pid rest <<< "$line"
        case " $skip_pids " in
            *" $pid "*) continue ;;
        esac
        case "$rest" in
            *qemu-system*|*qemu-kvm*|*"${OFFICIAL_VMDK}"*|*"${INPUT_VMDK}"*)
                matches+="${line}"$'\n'
                ;;
        esac
    done <<< "$ps_output"

    if [ -n "$matches" ]; then
        printf 'Matching process list:\n%s' "$matches" >&2
        fail "QEMU or VMDK-attached process is running"
    fi

    ok "no qemu-system or VMDK-attached process found"
}

check_sidecar() {
    local sidecar_hash=""
    local sidecar_path=""
    local sidecar_extra=""
    local actual

    read -r sidecar_hash sidecar_path sidecar_extra < "$SIDECAR" \
        || fail "failed to read sidecar: $SIDECAR"

    actual="$(sha256_file "$OFFICIAL_VMDK")"
    [ "$sidecar_hash" = "$actual" ] \
        || fail "sidecar hash mismatch: expected current VMDK hash $actual, got ${sidecar_hash:-<empty>}"
    [ "$sidecar_path" = "$OFFICIAL_VMDK" ] \
        || fail "sidecar target mismatch: expected $OFFICIAL_VMDK, got ${sidecar_path:-<empty>}"
    [ -z "$sidecar_extra" ] || fail "sidecar has unexpected extra fields: $SIDECAR"

    shasum -c "$SIDECAR" || fail "sidecar verification failed"
    ok "sidecar verifies $OFFICIAL_VMDK"
}

check_qemu_img() {
    local output

    output="$(qemu-img check "$OFFICIAL_VMDK" 2>&1)" \
        || fail "qemu-img check failed for $OFFICIAL_VMDK: $output"
    printf '%s\n' "$output"
    ok "qemu-img check passed for $OFFICIAL_VMDK"

    output="$(qemu-img check "$INPUT_VMDK" 2>&1)" \
        || fail "qemu-img check failed for $INPUT_VMDK: $output"
    printf '%s\n' "$output"
    ok "qemu-img check passed for $INPUT_VMDK"
}

main() {
    need_cmd basename
    need_cmd ps
    need_cmd qemu-img
    need_cmd shasum

    printf 'MCC 2026 official VM/VMDK preflight\n'
    printf 'Tool/BK_TOOL: %s\n' "$TOOL_NAME"
    printf 'Tool kind: %s\n' "$TOOL_KIND"
    printf 'Official VMDK: %s\n' "$OFFICIAL_VMDK"
    printf 'Input VMDK: %s\n' "$INPUT_VMDK"
    printf 'VMDK SHA-256 sidecar: %s\n' "$SIDECAR"
    printf 'BenchKitStart: %s\n' "$BENCHKIT_START"

    check_tool_identity
    check_official_files
    check_no_qemu_processes
    check_sidecar
    require_digest "official VMDK" "$OFFICIAL_VMDK" "$EXPECTED_OFFICIAL_SHA256"
    require_digest "input VMDK" "$INPUT_VMDK" "$EXPECTED_INPUT_SHA256"
    check_qemu_img

    ok "official MCC 2026 VMDK artifact passed preflight"
}

main "$@"
