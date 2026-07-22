#!/usr/bin/env bash
# clean_build_artifacts.sh - Comprehensive build artifact cleanup for ty
#
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Addresses #4111: target/ directory accumulated to 191 GB.
# Three cleanup targets:
#   1. target/*/incremental/ — stale incremental compilation cache
#   2. target/*/deps/ entries older than 7 days — stale dependency artifacts
#   3. /tmp/ty-agent-* directories older than 1 day — ephemeral agent builds
#
# Usage:
#   ./scripts/clean_build_artifacts.sh              # Dry-run (default)
#   ./scripts/clean_build_artifacts.sh --execute    # Actually delete
#   ./scripts/clean_build_artifacts.sh --threshold 50  # Only if target/ > 50 GB

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: clean_build_artifacts.sh [OPTIONS]

Periodic cleanup of build artifacts that accumulate over time.

Options:
  --execute          Actually delete files (default is dry-run)
  --threshold N      Only proceed if target/ exceeds N GB (default: 0, always run)
  --deps-age N       Remove deps entries older than N days (default: 7)
  --agent-age N      Remove /tmp/ty-agent-* older than N days (default: 1)
  -h, --help         Show this help message

Safety:
  Dry-run is always allowed. --execute refuses to delete while active TY
  cargo/rustc/ty/test processes are running, or when lsof reports open
  files under cleanup candidate roots.

Targets:
  1. target/*/incremental/   Stale incremental compilation caches (always removed)
  2. target/*/deps/          Dependency artifacts older than --deps-age days
  3. /tmp/ty-agent-*       Agent build dirs older than --agent-age days

Examples:
  ./scripts/clean_build_artifacts.sh                     # Preview what would be cleaned
  ./scripts/clean_build_artifacts.sh --execute           # Execute cleanup
  ./scripts/clean_build_artifacts.sh --threshold 50      # Only if target/ > 50 GB
  ./scripts/clean_build_artifacts.sh --execute --deps-age 14  # Keep deps for 14 days
EOF
    exit "${1:-0}"
}

# --- Helpers ---

format_bytes() {
    local bytes="$1"
    if (( bytes >= 1073741824 )); then
        awk "BEGIN { printf \"%.1f GB\", $bytes / 1073741824 }"
    elif (( bytes >= 1048576 )); then
        awk "BEGIN { printf \"%.1f MB\", $bytes / 1048576 }"
    elif (( bytes >= 1024 )); then
        awk "BEGIN { printf \"%.1f KB\", $bytes / 1024 }"
    else
        printf '%d B' "$bytes"
    fi
}

# Get size of a path in bytes (macOS + Linux compatible)
get_size_bytes() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo 0
        return
    fi
    # du -sk gives KiB; convert to bytes
    local kib
    kib=$(du -sk "$path" 2>/dev/null | awk '{print $1}')
    echo $(( kib * 1024 ))
}

# Get modification time as epoch seconds (macOS + Linux)
get_mtime_epoch() {
    local path="$1"
    case "$(uname -s)" in
        Darwin) stat -f %m "$path" 2>/dev/null ;;
        *)      stat -c %Y "$path" 2>/dev/null ;;
    esac
}

absolute_path() {
    local path="$1"
    if [[ -d "$path" ]]; then
        (cd "$path" && pwd -P)
    else
        local dir base
        dir="$(dirname "$path")"
        base="$(basename "$path")"
        (cd "$dir" && printf '%s/%s\n' "$(pwd -P)" "$base")
    fi
}

indent_text() {
    local text="$1"
    while IFS= read -r line; do
        printf '  %s\n' "$line"
    done <<<"$text"
}

command_references_any_root() {
    local command="$1"
    shift

    local root
    for root in "$@"; do
        if [[ "$command" == *"$root"* ]]; then
            return 0
        fi
    done
    return 1
}

lsof_process_references_under() {
    local pid="$1"
    local root="$2"
    [[ -e "$root" ]] || return 1

    local abs_root
    abs_root="$(absolute_path "$root")" || return 1

    lsof -nP -p "$pid" 2>/dev/null | awk -v root="$abs_root" '
        index($0, root) != 0 {
            found = 1
            exit 0
        }
        END {
            exit(found ? 0 : 1)
        }
    '
}

process_references_any_root() {
    local pid="$1"
    shift

    local root
    for root in "$@"; do
        if lsof_process_references_under "$pid" "$root"; then
            return 0
        fi
    done
    return 1
}

find_active_build_processes() {
    local -a active_roots=("$@")
    local self_pid="$$"
    local line pid ppid cmd

    while IFS=$'\t' read -r pid ppid cmd; do
        [[ -n "$pid" ]] || continue
        if command_references_any_root "$cmd" "${active_roots[@]}" ||
           process_references_any_root "$pid" "${active_roots[@]}"; then
            printf '%s %s %s\n' "$pid" "$ppid" "$cmd"
        fi
    done < <(ps -axo pid=,ppid=,command= | awk -v self="$self_pid" '
        function matches_build_process(cmd) {
            return cmd ~ /(^|[\/[:space:]])cargo([[:space:]]|$)/ ||
                   cmd ~ /(^|[\/[:space:]])rustc([[:space:]]|$)/ ||
                   cmd ~ /(^|[\/[:space:]])ty([[:space:]]|$)/ ||
                   cmd ~ /(^|[\/[:space:]])test([[:space:]]|$)/ ||
                   cmd ~ /\/target\/debug\/deps\// ||
                   cmd ~ /\/target\/agent\//
        }
        {
            pid = $1
            ppid = $2
            $1 = ""
            $2 = ""
            sub(/^[ \t]+/, "", $0)
            cmd = $0
            if (pid == self) {
                next
            }
            if (cmd ~ /clean_build_artifacts\.sh/) {
                next
            }
            if (cmd ~ /\/opt\/homebrew\/bin\/gh / || cmd ~ / git-upload-pack /) {
                next
            }
            if (matches_build_process(cmd)) {
                print pid "\t" ppid "\t" cmd
            }
        }
    ')
}

lsof_references_under() {
    local root="$1"
    [[ -e "$root" ]] || return 1

    local abs_root
    abs_root="$(absolute_path "$root")" || return 1

    lsof -nP 2>/dev/null | awk -v root="$abs_root" '
        index($0, root) == 0 {
            next
        }
        {
            print
            count += 1
            if (count >= 25) {
                exit 0
            }
        }
        END {
            exit(count > 0 ? 0 : 1)
        }
    '
}

is_stale_by_mtime_days() {
    local path="$1"
    local max_age_days="$2"
    local now="$3"
    local mtime
    mtime=$(get_mtime_epoch "$path")
    if [[ -z "$mtime" ]]; then
        return 1
    fi
    local age=$((now - mtime))
    local max_age_sec=$((max_age_days * 86400))
    (( age > max_age_sec ))
}

enforce_execute_safety() {
    local repo_root="$1"
    local target_dir="$2"
    local agent_age_days="$3"
    local now="$4"

    echo "--- Safety preflight (--execute) ---"

    if ! command -v lsof >/dev/null 2>&1; then
        echo "ERROR: refusing --execute because lsof is unavailable for open-file safety checks." >&2
        return 1
    fi

    local -a active_roots=("$repo_root")
    if [[ -d "$target_dir" ]]; then
        active_roots+=("$target_dir")
    fi

    # Watch all TY temp run roots for active processes without adding them to
    # the cleanup set; cleanup still only considers stale /tmp/ty-agent-* dirs.
    local d
    for d in /tmp/ty-*; do
        [[ -d "$d" ]] || continue
        active_roots+=("$d")
    done

    local active_processes
    active_processes="$(find_active_build_processes "${active_roots[@]}" || true)"
    if [[ -n "$active_processes" ]]; then
        echo "ERROR: refusing --execute while active TY cargo/rustc/ty/test processes are running." >&2
        echo "Active processes:" >&2
        indent_text "$active_processes" >&2
        echo "Dry-run remains safe; rerun --execute after these processes finish." >&2
        return 1
    fi

    local -a candidate_roots=()
    if [[ -d "$target_dir" ]]; then
        candidate_roots+=("$target_dir")
    fi

    for d in /tmp/ty-agent-*; do
        [[ -d "$d" ]] || continue
        if is_stale_by_mtime_days "$d" "$agent_age_days" "$now"; then
            candidate_roots+=("$d")
        fi
    done

    local root refs
    for root in "${candidate_roots[@]}"; do
        refs="$(lsof_references_under "$root" || true)"
        if [[ -n "$refs" ]]; then
            echo "ERROR: refusing --execute; open files reference cleanup candidate root: $root" >&2
            indent_text "$refs" >&2
            echo "Dry-run remains safe; rerun --execute after references close." >&2
            return 1
        fi
    done

    echo "Safety preflight passed: no active build/test processes or open candidate references found."
    echo
}

# --- Main ---

main() {
    local execute=false
    local threshold_gb=0
    local deps_age_days=7
    local agent_age_days=1

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --execute)
                execute=true
                shift
                ;;
            --threshold)
                threshold_gb="${2:?--threshold requires a number}"
                shift 2
                ;;
            --threshold=*)
                threshold_gb="${1#*=}"
                shift
                ;;
            --deps-age)
                deps_age_days="${2:?--deps-age requires a number}"
                shift 2
                ;;
            --deps-age=*)
                deps_age_days="${1#*=}"
                shift
                ;;
            --agent-age)
                agent_age_days="${2:?--agent-age requires a number}"
                shift 2
                ;;
            --agent-age=*)
                agent_age_days="${1#*=}"
                shift
                ;;
            -h|--help)
                usage 0
                ;;
            *)
                echo "Error: Unknown option: $1" >&2
                usage 1
                ;;
        esac
    done

    # Locate repo root
    local script_dir repo_root target_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "$script_dir/.." && pwd)"
    target_dir="$repo_root/target"

    local mode
    if $execute; then
        mode="EXECUTE"
    else
        mode="DRY-RUN"
    fi

    echo "=== Build Artifact Cleanup [$mode] ==="
    echo "Repo:        $repo_root"
    echo "Target dir:  $target_dir"
    echo "Deps age:    ${deps_age_days} days"
    echo "Agent age:   ${agent_age_days} days"
    echo

    # --- Size before ---
    local target_size_before=0
    if [[ -d "$target_dir" ]]; then
        target_size_before=$(get_size_bytes "$target_dir")
        echo "target/ size before: $(format_bytes "$target_size_before")"
    else
        echo "target/ directory not found — nothing to clean."
    fi

    local agent_dirs_size_before=0
    local agent_dir_count=0
    for d in /tmp/ty-agent-*; do
        if [[ -d "$d" ]]; then
            local sz
            sz=$(get_size_bytes "$d")
            agent_dirs_size_before=$((agent_dirs_size_before + sz))
            agent_dir_count=$((agent_dir_count + 1))
        fi
    done
    if (( agent_dir_count > 0 )); then
        echo "/tmp/ty-agent-* size: $(format_bytes "$agent_dirs_size_before") ($agent_dir_count dirs)"
    else
        echo "/tmp/ty-agent-*: none found"
    fi
    echo

    # --- Threshold check ---
    if (( threshold_gb > 0 )); then
        local threshold_bytes=$(( threshold_gb * 1073741824 ))
        if (( target_size_before < threshold_bytes )); then
            echo "target/ ($(format_bytes "$target_size_before")) is below ${threshold_gb} GB threshold — skipping."
            exit 0
        fi
        echo "target/ exceeds ${threshold_gb} GB threshold — proceeding with cleanup."
        echo
    fi

    local now
    now=$(date +%s)

    if $execute; then
        enforce_execute_safety "$repo_root" "$target_dir" "$agent_age_days" "$now"
    fi

    # --- Phase 1: Incremental compilation caches ---
    echo "--- Phase 1: Incremental compilation caches (target/*/incremental/) ---"
    local incremental_total=0
    local incremental_count=0
    if [[ -d "$target_dir" ]]; then
        for inc_dir in "$target_dir"/*/incremental; do
            if [[ -d "$inc_dir" ]]; then
                local sz
                sz=$(get_size_bytes "$inc_dir")
                incremental_total=$((incremental_total + sz))
                incremental_count=$((incremental_count + 1))
                local parent
                parent=$(basename "$(dirname "$inc_dir")")
                echo "  $parent/incremental/: $(format_bytes "$sz")"
                if $execute; then
                    rm -rf "$inc_dir"
                fi
            fi
        done
    fi
    if (( incremental_count == 0 )); then
        echo "  (none found)"
    else
        echo "  Total: $(format_bytes "$incremental_total") across $incremental_count profile(s)"
        if ! $execute; then
            echo "  [dry-run] Would remove $incremental_count incremental/ directories"
        else
            echo "  Removed $incremental_count incremental/ directories"
        fi
    fi
    echo

    # --- Phase 2: Old deps entries ---
    echo "--- Phase 2: Stale deps entries (target/*/deps/, older than ${deps_age_days} days) ---"
    local deps_total=0
    local deps_file_count=0
    local max_age_sec=$((deps_age_days * 86400))

    if [[ -d "$target_dir" ]]; then
        for deps_dir in "$target_dir"/*/deps; do
            if [[ -d "$deps_dir" ]]; then
                local parent
                parent=$(basename "$(dirname "$deps_dir")")
                local dir_stale_bytes=0
                local dir_stale_count=0

                # Check files directly in the deps directory
                for entry in "$deps_dir"/*; do
                    [[ -e "$entry" ]] || continue
                    local mtime
                    mtime=$(get_mtime_epoch "$entry")
                    if [[ -z "$mtime" ]]; then
                        continue
                    fi
                    local age=$((now - mtime))
                    if (( age > max_age_sec )); then
                        local entry_size
                        entry_size=$(get_size_bytes "$entry")
                        dir_stale_bytes=$((dir_stale_bytes + entry_size))
                        dir_stale_count=$((dir_stale_count + 1))
                        if $execute; then
                            rm -rf "$entry"
                        fi
                    fi
                done

                if (( dir_stale_count > 0 )); then
                    echo "  $parent/deps/: $dir_stale_count stale entries ($(format_bytes "$dir_stale_bytes"))"
                    deps_total=$((deps_total + dir_stale_bytes))
                    deps_file_count=$((deps_file_count + dir_stale_count))
                fi
            fi
        done
    fi
    if (( deps_file_count == 0 )); then
        echo "  (no stale deps entries found)"
    else
        echo "  Total: $(format_bytes "$deps_total") across $deps_file_count entries"
        if ! $execute; then
            echo "  [dry-run] Would remove $deps_file_count stale deps entries"
        else
            echo "  Removed $deps_file_count stale deps entries"
        fi
    fi
    echo

    # --- Phase 3: /tmp/ty-agent-* directories ---
    echo "--- Phase 3: Agent temp dirs (/tmp/ty-agent-*, older than ${agent_age_days} day(s)) ---"
    local agent_total=0
    local agent_removed_count=0
    local agent_max_age_sec=$((agent_age_days * 86400))

    for d in /tmp/ty-agent-*; do
        if [[ -d "$d" ]]; then
            local mtime
            mtime=$(get_mtime_epoch "$d")
            if [[ -z "$mtime" ]]; then
                continue
            fi
            local age=$((now - mtime))
            if (( age > agent_max_age_sec )); then
                local sz
                sz=$(get_size_bytes "$d")
                agent_total=$((agent_total + sz))
                agent_removed_count=$((agent_removed_count + 1))
                echo "  $(basename "$d"): $(format_bytes "$sz") ($(( age / 86400 )) days old)"
                if $execute; then
                    rm -rf "$d"
                fi
            fi
        fi
    done
    if (( agent_removed_count == 0 )); then
        echo "  (no stale agent dirs found)"
    else
        echo "  Total: $(format_bytes "$agent_total") across $agent_removed_count dir(s)"
        if ! $execute; then
            echo "  [dry-run] Would remove $agent_removed_count agent dir(s)"
        else
            echo "  Removed $agent_removed_count agent dir(s)"
        fi
    fi
    echo

    # --- Summary ---
    local total_reclaimable=$((incremental_total + deps_total + agent_total))
    echo "=== Summary ==="
    echo "  Incremental caches: $(format_bytes "$incremental_total")"
    echo "  Stale deps:         $(format_bytes "$deps_total")"
    echo "  Agent temp dirs:    $(format_bytes "$agent_total")"
    echo "  ─────────────────────────────────"
    if $execute; then
        echo "  Total recovered:    $(format_bytes "$total_reclaimable")"
        # Size after
        if [[ -d "$target_dir" ]]; then
            local target_size_after
            target_size_after=$(get_size_bytes "$target_dir")
            echo
            echo "  target/ size after:  $(format_bytes "$target_size_after")"
            echo "  Reduction:           $(format_bytes $((target_size_before - target_size_after)))"
        fi
    else
        echo "  Total reclaimable:  $(format_bytes "$total_reclaimable")"
        echo
        echo "  Run with --execute to perform cleanup."
    fi
}

main "$@"
