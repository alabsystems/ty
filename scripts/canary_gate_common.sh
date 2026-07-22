#!/usr/bin/env bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# Shared helpers for canary gate build/runtime artifact selection.

resolve_canary_target_dir() {
    local repo_root="${1:?repo_root is required}"
    local configured_target="${CARGO_TARGET_DIR:-}"

    if [[ -n "$configured_target" ]]; then
        if [[ "$configured_target" = /* ]]; then
            printf '%s\n' "$configured_target"
        else
            printf '%s\n' "$repo_root/$configured_target"
        fi
        return 0
    fi

    printf '%s\n' "$repo_root/target/user"
}
