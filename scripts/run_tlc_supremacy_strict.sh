#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# Launch a strict TY-vs-TLC evidence command in a transient delegated systemd
# user service.  The Python helper performs all cgroup qualification and fails
# closed before the benchmark command starts.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_tlc_supremacy_strict.sh [LAUNCHER_OPTIONS] -- \
    /absolute/path/to/ty supremacy COMMAND [OPTIONS]

Launcher options:
  --cpu N|auto                Logical CPU to reserve and confine (default: auto).
  --wall-timeout-seconds N    Required positive outer wall-clock cap for the
                              entire TY invocation.
  --run-dir PATH              New persistent launcher artifact directory.
                              Default: target/supremacy-strict/<UTC>-<PID>
  --provenance PATH           New machine-provenance JSON path.
                              Default: <run-dir>/machine-provenance.json
  -h, --help                  Show this help.

Strict evidence commands are limited to:
  * supremacy matrix-segment --mode enforce
      --campaign-plan /absolute/path --segment-id segment-NNNN
      --runtime-output-dir /absolute/path
  * supremacy matrix-merge-inventory --mode enforce
      --campaign-plan /absolute/path
      --segment-report /absolute/path [...]
      --runtime-output-dir /absolute/path
  * supremacy matrix-merge --mode enforce
      --campaign-plan /absolute/path
      --segment-report /absolute/path [...]
      --runtime-output-dir /absolute/path

The command is not started unless delegated cgroup v2, zero descendant swap,
one isolated CPU, unlimited CPU quota, and stable runtime directories are all
verified. The helper exports the qualified empty parent as
TY_SUPREMACY_CGROUP_PARENT and predeclares the final receipt path at
TY_SUPREMACY_FINAL_RECEIPT. A segment additionally receives a root-owned,
mode-0444, FS_IMMUTABLE_FL TY_SUPREMACY_OBSERVATION_STORAGE_CAPABILITY. Its
plan-bound evidence/payload ext4 project quotas are leased, assigned, and
re-attested through the installed root-owned
/usr/local/libexec/ty-strict-storage-attestor. Project quota must already be
enabled; the launcher never changes filesystem-global quota state. Production
sudoers must authorize only the SHA-256-pinned helper's exact
attest-observation-storage command shape.
After the unit exits, success seals the retained tree and commits an immutable
fixed-size release slot plus the durable root ledger transition. Failure and
signals invoke the root-ledger-authoritative abort path; a live stale cgroup or
unrecoverable journal refuses the next launch. The retained immutable
capability and release require an explicit privileged immutable-flag clear
before their run directory can ever be removed; this launcher never clears or
deletes them.
The receipt is exclusively created and write-once by protocol; it is not an
OS-immutable file. The explicit command output directory must be new;
the helper creates output-owned TMPDIR/TMP/TEMP, XDG roots, and TY cache
storage there and records the deterministic TLC-metadir/TY-cache contract.
Each measured command then receives its own runner-confined scratch tree and
bounded sampled disk high-water record. The wrapper preserves HOME and PATH
plus only the narrow optional TLA+ input variables admitted by the helper.
The outer wall cap is distinct from the campaign plan's per-observation
runtime timeout. See docs/perf/strict-supremacy-linux-launcher.md.
EOF
}

fail() {
    echo "strict supremacy launcher: $*" >&2
    exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "$script_dir/.." && pwd -P)"
helper="$script_dir/strict_supremacy_linux.py"
working_directory="$(pwd -P)"
cpu_request="auto"
wall_timeout_seconds=""
run_dir=""
provenance=""

while (($#)); do
    case "$1" in
        --cpu)
            (($# >= 2)) || fail "--cpu requires N or auto"
            cpu_request="$2"
            shift 2
            ;;
        --run-dir)
            (($# >= 2)) || fail "--run-dir requires a path"
            run_dir="$2"
            shift 2
            ;;
        --wall-timeout-seconds)
            (($# >= 2)) || fail "--wall-timeout-seconds requires a positive integer"
            wall_timeout_seconds="$2"
            shift 2
            ;;
        --provenance)
            (($# >= 2)) || fail "--provenance requires a path"
            provenance="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            fail "unknown launcher argument: $1"
            ;;
    esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
    fail "Linux is required"
fi

for required in bash env python3 sudo systemd-run systemctl realpath printenv stat; do
    command -v "$required" >/dev/null 2>&1 ||
        fail "missing required command: $required"
done
env_bin="$(realpath -e -- "$(command -v env)")"
python_bin="$(realpath -e -- "$(command -v python3)")"
sudo_bin="$(realpath -e -- "$(command -v sudo)")"
storage_attestor="/usr/local/libexec/ty-strict-storage-attestor"
[[ -f "$storage_attestor" && -x "$storage_attestor" && ! -L "$storage_attestor" ]] ||
    fail "missing installed root-owned storage attestor: $storage_attestor"

[[ "$wall_timeout_seconds" =~ ^[1-9][0-9]*$ ]] ||
    fail "--wall-timeout-seconds is required and must be a positive integer"
[[ -n "${HOME+x}" && -n "$HOME" ]] ||
    fail "caller HOME must be explicitly set and nonempty"
[[ -n "${PATH+x}" && -n "$PATH" ]] ||
    fail "caller PATH must be explicitly set and nonempty"
expected_runtime_directory="/run/user/$(id -u)"
[[ -d "$expected_runtime_directory" ]] ||
    fail "canonical systemd user runtime directory is absent: $expected_runtime_directory"
runtime_directory="$(realpath -e -- "$expected_runtime_directory")"
[[ "$runtime_directory" == "$expected_runtime_directory" ]] ||
    fail "systemd user runtime directory is not canonical: $expected_runtime_directory"
runtime_directory_owner="$(stat -Lc '%u' -- "$runtime_directory")" ||
    fail "cannot inspect systemd user runtime directory owner"
runtime_directory_mode="$(stat -Lc '%a' -- "$runtime_directory")" ||
    fail "cannot inspect systemd user runtime directory mode"
[[ "$runtime_directory_owner" == "$(id -u)" ]] ||
    fail "systemd user runtime directory is not owned by the caller"
[[ "$runtime_directory_mode" == "700" ]] ||
    fail "systemd user runtime directory must have mode 0700"

caller_environment=(
    "HOME=$HOME"
    "PATH=$PATH"
    "XDG_RUNTIME_DIR=$runtime_directory"
)
for environment_name in \
    TLAPLUS_EXAMPLES TLC_JAR TYTOOLS_JAR COMMUNITY_MODULES \
    TLA_LIBRARY TLA_PLUS_LIBRARY
do
    if environment_value="$(printenv "$environment_name")"; then
        caller_environment+=(
            "${environment_name}=${environment_value}"
        )
    fi
done

(($# >= 3)) ||
    fail "expected -- /absolute/path/to/ty supremacy COMMAND [OPTIONS]"

if [[ "$cpu_request" == "auto" ]]; then
    selected_cpu="$(
        "$env_bin" -i LANG=C LC_ALL=C TZ=UTC \
            "${caller_environment[@]}" \
            "$python_bin" "$helper" select-cpu
    )" ||
        fail "automatic CPU selection failed"
elif [[ "$cpu_request" =~ ^(0|[1-9][0-9]*)$ ]]; then
    selected_cpu="$cpu_request"
else
    fail "--cpu must be auto or a non-negative integer"
fi

command_argv=("$@")
command_path="${command_argv[0]}"
if [[ "$command_path" == */* ]]; then
    command_path="$(realpath -e -- "$command_path")" ||
        fail "cannot resolve TY executable: ${command_argv[0]}"
else
    command_path="$(command -v -- "$command_path")" ||
        fail "cannot find TY executable on PATH: ${command_argv[0]}"
    command_path="$(realpath -e -- "$command_path")"
fi
[[ -x "$command_path" ]] || fail "TY executable is not executable: $command_path"
[[ "$(basename -- "$command_path")" == "ty" ]] ||
    fail "command executable must be named ty"
[[ "${command_argv[1]}" == "supremacy" ]] ||
    fail "command must begin with ty supremacy"
command_argv[0]="$command_path"

utc_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$run_dir" ]]; then
    run_dir="$project_root/target/supremacy-strict/${utc_stamp}-$$"
fi
run_dir="$(realpath -m -- "$run_dir")"
[[ ! -e "$run_dir" ]] || fail "run directory already exists: $run_dir"
mkdir -m 0700 -p -- "$run_dir"
for child in tmp xdg-cache xdg-config xdg-state; do
    mkdir -m 0700 -- "$run_dir/$child"
done
receipt="$run_dir/strict-evidence-receipt.json"
[[ ! -e "$receipt" && ! -L "$receipt" ]] ||
    fail "final receipt output already exists: $receipt"

if [[ -z "$provenance" ]]; then
    provenance="$run_dir/machine-provenance.json"
else
    provenance="$(realpath -m -- "$provenance")"
fi
[[ -d "$(dirname -- "$provenance")" ]] ||
    fail "provenance parent directory does not exist: $(dirname -- "$provenance")"
[[ ! -e "$provenance" && ! -L "$provenance" ]] ||
    fail "provenance output already exists: $provenance"

unit="ty-supremacy-$(id -u)-$$-${utc_stamp}.service"

echo "[strict-supremacy] unit=$unit cpu=$selected_cpu" >&2
echo "[strict-supremacy] run_dir=$run_dir" >&2
echo "[strict-supremacy] provenance=$provenance" >&2
echo "[strict-supremacy] final_receipt=$receipt" >&2

run_storage_abort() {
    "$env_bin" -i \
        LANG=C \
        LC_ALL=C \
        TZ=UTC \
        HOME="$HOME" \
        PATH="$PATH" \
        "$python_bin" "$storage_attestor" abort-after-run \
            --provenance "$provenance" \
            --unit "$unit" \
            --storage-attestor "$storage_attestor" \
            --sudo "$sudo_bin"
}

cleanup_needed=1
storage_cleanup_complete=0
cleanup_on_exit() {
    local requested_status="$1"
    trap - EXIT INT TERM
    set +e
    if ((cleanup_needed != 0)); then
        systemctl --user stop "$unit" >/dev/null 2>&1
        run_storage_abort
        local cleanup_status=$?
        if ((cleanup_status != 0)); then
            echo "[strict-supremacy] emergency lease abort failed with status $cleanup_status" >&2
        fi
    fi
    exit "$requested_status"
}
trap 'cleanup_on_exit $?' EXIT
trap 'cleanup_on_exit 130' INT
trap 'cleanup_on_exit 143' TERM

"$env_bin" -i \
    LANG=C \
    LC_ALL=C \
    TZ=UTC \
    HOME="$HOME" \
    PATH="$PATH" \
    "$python_bin" "$storage_attestor" recover-storage-before-run \
        --storage-attestor "$storage_attestor" \
        --sudo "$sudo_bin"

set +e
# These private run-directory paths bootstrap the Python supervisor only. The
# helper replaces them with exclusively-created output-owned paths in the
# `ty supremacy` command environment after qualification.
systemd-run \
    --user \
    --wait \
    --collect \
    --pipe \
    --quiet \
    --unit="$unit" \
    --service-type=exec \
    --property=Delegate=yes \
    --property=AllowedCPUs="$selected_cpu" \
    --property=MemorySwapMax=0 \
    --property=CPUAccounting=yes \
    --property=MemoryAccounting=yes \
    --property=IOAccounting=yes \
    --property=TasksMax=infinity \
    --property=KillMode=control-group \
    --property=TimeoutStopSec=30s \
    --property=RuntimeMaxSec="${wall_timeout_seconds}s" \
    --property=LimitCORE=0 \
    -- \
    "$env_bin" -i \
        LANG=C \
        LC_ALL=C \
        TZ=UTC \
        TMPDIR="$run_dir/tmp" \
        TMP="$run_dir/tmp" \
        TEMP="$run_dir/tmp" \
        XDG_CACHE_HOME="$run_dir/xdg-cache" \
        XDG_CONFIG_HOME="$run_dir/xdg-config" \
        XDG_STATE_HOME="$run_dir/xdg-state" \
        "${caller_environment[@]}" \
        "$python_bin" "$helper" prepare-and-run \
        --unit "$unit" \
        --cpu "$selected_cpu" \
        --wall-timeout-seconds "$wall_timeout_seconds" \
        --run-dir "$run_dir" \
        --provenance "$provenance" \
        --working-directory "$working_directory" \
        --storage-attestor "$storage_attestor" \
        --sudo "$sudo_bin" \
        -- "${command_argv[@]}"
status=$?
if ((status == 0)); then
    "$env_bin" -i \
        LANG=C \
        LC_ALL=C \
        TZ=UTC \
        HOME="$HOME" \
        PATH="$PATH" \
        "$python_bin" "$storage_attestor" release-after-run \
            --provenance "$provenance" \
            --receipt "$receipt" \
            --storage-attestor "$storage_attestor" \
            --sudo "$sudo_bin"
    release_status=$?
    if ((release_status != 0)); then
        echo "[strict-supremacy] release failed; retrying the idempotent root-journal transition" >&2
        "$env_bin" -i \
            LANG=C \
            LC_ALL=C \
            TZ=UTC \
            HOME="$HOME" \
            PATH="$PATH" \
            "$python_bin" "$storage_attestor" release-after-run \
                --provenance "$provenance" \
                --receipt "$receipt" \
                --storage-attestor "$storage_attestor" \
                --sudo "$sudo_bin"
        release_retry_status=$?
        if ((release_retry_status != 0)); then
            run_storage_abort
            abort_status=$?
            if ((abort_status != 0)); then
                status=$abort_status
            else
                status=$release_retry_status
                storage_cleanup_complete=1
            fi
        else
            storage_cleanup_complete=1
        fi
    else
        storage_cleanup_complete=1
    fi
else
    run_storage_abort
    abort_status=$?
    if ((abort_status != 0)); then
        status=$abort_status
    else
        storage_cleanup_complete=1
    fi
fi
if ((storage_cleanup_complete != 0)); then
    cleanup_needed=0
fi
set -e

if [[ -f "$provenance" ]]; then
    echo "[strict-supremacy] machine provenance: $provenance" >&2
else
    echo "[strict-supremacy] no provenance was created; systemd rejected the unit before helper startup" >&2
fi
if [[ -f "$receipt" ]]; then
    echo "[strict-supremacy] receipt artifact (qualifies only with the matching command_passed machine-provenance link): $receipt" >&2
else
    echo "[strict-supremacy] no qualifying strict evidence receipt was created" >&2
fi
exit "$status"
