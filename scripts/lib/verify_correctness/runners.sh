# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

config_has_property() {
    local cfg="${1:-}"
    [ -n "$cfg" ] && [ -f "$cfg" ] && grep -Eq '^[[:space:]]*PROPERTY([[:space:]]|$)' "$cfg"
}

TLC_JAVA_SINGLE_THREAD_ARGS=(
    -XX:ActiveProcessorCount=1
    -XX:+UseSerialGC
    -Xms64m
    -Xmx4g
)
TLC_JAVA_ENV=(env -u JAVA_TOOL_OPTIONS -u JDK_JAVA_OPTIONS -u _JAVA_OPTIONS)

build_ty_fair_env() {
    local skip_liveness="${1:-0}"
    TY_RUN_ENV=(env)
    while IFS='=' read -r name _; do
        if [[ "$name" == TY_* ]]; then
            TY_RUN_ENV+=( -u "$name" )
        fi
    done < <(env)
    if [ "$skip_liveness" = "1" ]; then
        TY_RUN_ENV+=( TY_SKIP_LIVENESS=1 )
    fi
}

# W10: Run a negative test with trace comparison between TY and TLC
# Verifies both tools find the same violation with matching trace signatures
run_negative() {
    local name="$1"
    local spec="$2"
    local config="${3:-}"
    local expected_error="${4:-invariant}"  # invariant, deadlock, or liveness
    local extra_args="${5:-}"

    if [ ! -f "$spec" ]; then
        echo "[ SKIP ] $name (negative) - spec not found"
        SKIP=$((SKIP + 1))
        return
    fi

    # Run TY
    local ty_args="--workers 1 --force"
    if [ -n "$config" ] && [ -f "$config" ]; then
        ty_args="$ty_args --config $config"
    fi
    local ty_output=""
    build_ty_fair_env 0
    ty_output="$("${TY_RUN_ENV[@]}" "$TY" check "$spec" $ty_args $extra_args 2>&1)" || true

    # Run TLC
    local tlc_jar="$HOME/tlaplus/tytools.jar"
    if [ ! -f "$tlc_jar" ]; then
        echo "[ SKIP ] $name (negative) - TLC jar not found"
        SKIP=$((SKIP + 1))
        return
    fi

    # Build classpath with CommunityModules if available
    local community_modules="${COMMUNITY_MODULES:-$HOME/tlaplus/CommunityModules.jar}"
    local tlc_cp="$tlc_jar"
    if [ -f "$community_modules" ]; then
        tlc_cp="$tlc_jar:$community_modules"
    fi

    local spec_dir=""
    spec_dir="$(cd "$(dirname "$spec")" && pwd)"
    local states_dir="$spec_dir/states"

    local -a tlc_args=( -workers 1 )
    if [ -n "$config" ] && [ -f "$config" ]; then
        tlc_args+=( -config "$config" )
    fi
    if [ "$expected_error" = "deadlock" ]; then
        tlc_args+=( -deadlock )
    fi

    local tlc_metadir=""
    if [ "${TY_KEEP_STATES:-0}" != "1" ]; then
        local tlc_metadir_root="${TY_TLC_METADIR_ROOT:-$REPO_ROOT/target/tlc_metadir}"
        mkdir -p "$tlc_metadir_root"
        tlc_metadir="$(mktemp -d "$tlc_metadir_root/tlc_XXXXXX")"
        tlc_args+=( -metadir "$tlc_metadir" )
    fi

    local tlc_output=""
    tlc_output="$(cd "$spec_dir" && "${TLC_JAVA_ENV[@]}" java "${TLC_JAVA_SINGLE_THREAD_ARGS[@]}" -cp "$tlc_cp" tlc2.TLC "${tlc_args[@]}" "$(basename "$spec")" 2>&1)" || true

    if [ -n "$tlc_metadir" ]; then
        rm -rf "$tlc_metadir"
    fi
    if [ "${TY_KEEP_STATES:-0}" != "1" ] && [ "${TY_PRESERVE_STATES_DIR:-0}" != "1" ] && [ -d "$states_dir" ]; then
        # Safety: never delete `/states` if someone points a spec at `/`.
        if [ -z "${spec_dir:-}" ] || [ "$spec_dir" = "/" ] || [ "$states_dir" = "/states" ] || [ "$(basename -- "$states_dir")" != "states" ]; then
            echo "[ WARN ] refusing to delete suspicious states dir: spec_dir=$spec_dir states_dir=$states_dir" >&2
        else
            rm -rf "$states_dir"
        fi
    fi

    # Extract TY trace info (use extended regex for multi-digit state numbers)
    local ty_trace_len=""
    ty_trace_len="$(echo "$ty_output" | grep -cE "^State [0-9]+[ :]" || true)"
    local ty_trace_lines=""
    ty_trace_lines="$(printf '%s\n' "$ty_output" | extract_ty_trace_lines)"
    local ty_final_state_block=""
    ty_final_state_block="$(printf '%s\n' "$ty_output" | extract_ty_final_state_block)"
    local ty_final_state_sig=""
    if [ -n "$ty_final_state_block" ]; then
        ty_final_state_sig="$(printf '%s' "$ty_final_state_block" | sha256_hex)"
    fi
    local ty_trace_sig=""
    if [ "$ty_trace_len" != "0" ] && [ -n "$ty_trace_lines" ]; then
        ty_trace_sig="$(printf '%s\n' "$ty_trace_lines" | trace_signature)"
    fi

    # Extract TY error type
    local ty_error=""
    if echo "$ty_output" | grep -q "Error: Invariant"; then
        ty_error="invariant"
    elif echo "$ty_output" | grep -q "Error: Deadlock"; then
        ty_error="deadlock"
    elif echo "$ty_output" | grep -q "Error:.*liveness\|Error:.*temporal"; then
        ty_error="liveness"
    fi

    # Extract TLC trace info (use extended regex for multi-digit state numbers)
    local tlc_trace_len=""
    tlc_trace_len="$(echo "$tlc_output" | grep -cE "^State [0-9]+:" || true)"
    local tlc_trace_lines=""
    tlc_trace_lines="$(printf '%s\n' "$tlc_output" | extract_tlc_trace_lines)"
    local tlc_final_state_block=""
    tlc_final_state_block="$(printf '%s\n' "$tlc_output" | extract_tlc_final_state_block)"
    local tlc_final_state_sig=""
    if [ -n "$tlc_final_state_block" ]; then
        tlc_final_state_sig="$(printf '%s' "$tlc_final_state_block" | sha256_hex)"
    fi
    local tlc_trace_sig=""
    if [ "$tlc_trace_len" != "0" ] && [ -n "$tlc_trace_lines" ]; then
        tlc_trace_sig="$(printf '%s\n' "$tlc_trace_lines" | trace_signature)"
    fi

    # Extract TLC error type
    local tlc_error=""
    if echo "$tlc_output" | grep -q "Invariant.*violated"; then
        tlc_error="invariant"
    elif echo "$tlc_output" | grep -q "Deadlock reached"; then
        tlc_error="deadlock"
    elif echo "$tlc_output" | grep -q "liveness\|temporal"; then
        tlc_error="liveness"
    fi

    # Compare results
    local passed=true
    local failures=""

    # Check both found errors
    if [ -z "$ty_error" ]; then
        passed=false
        failures="$failures TY found no error;"
    fi
    if [ -z "$tlc_error" ]; then
        passed=false
        failures="$failures TLC found no error;"
    fi

    # Check error types match
    if [ -n "$ty_error" ] && [ -n "$tlc_error" ] && [ "$ty_error" != "$tlc_error" ]; then
        passed=false
        failures="$failures error type mismatch (TY:$ty_error vs TLC:$tlc_error);"
    fi

    # Check error type matches expectation
    if [ -n "$expected_error" ] && [ -n "$ty_error" ] && [ "$ty_error" != "$expected_error" ]; then
        passed=false
        failures="$failures unexpected TY error type (expected:$expected_error got:$ty_error);"
    fi
    if [ -n "$expected_error" ] && [ -n "$tlc_error" ] && [ "$tlc_error" != "$expected_error" ]; then
        passed=false
        failures="$failures unexpected TLC error type (expected:$expected_error got:$tlc_error);"
    fi

    # Check trace lengths match
    if [ "$ty_trace_len" != "$tlc_trace_len" ]; then
        passed=false
        failures="$failures trace length mismatch (TY:$ty_trace_len vs TLC:$tlc_trace_len);"
    fi

    # Check full trace signature matches (normalized per-state lines).
    if [ "$ty_trace_len" != "0" ] && [ "$tlc_trace_len" != "0" ]; then
        if [ -z "$ty_trace_sig" ] || [ -z "$tlc_trace_sig" ]; then
            passed=false
            failures="$failures could not extract trace signature;"
        elif [ "$ty_trace_sig" != "$tlc_trace_sig" ]; then
            passed=false
            failures="$failures trace mismatch;"
        fi
    fi

    # Check trace signature matches (normalized final state).
    if [ "$ty_trace_len" != "0" ] && [ "$tlc_trace_len" != "0" ]; then
        if [ -z "$ty_final_state_sig" ] || [ -z "$tlc_final_state_sig" ]; then
            passed=false
            failures="$failures could not extract final state signature;"
        elif [ "$ty_final_state_sig" != "$tlc_final_state_sig" ]; then
            passed=false
            failures="$failures final state mismatch;"
        fi
    fi

    if [ "$passed" = "true" ]; then
        echo "[ PASS ] $name (negative): $ty_error, trace=$ty_trace_len states"
        PASS=$((PASS + 1))
    else
        echo "[ FAIL ] $name (negative):$failures"
        if [ -n "$ty_trace_lines" ] && [ -n "$tlc_trace_lines" ] && [ "$ty_trace_sig" != "$tlc_trace_sig" ]; then
            echo "TY trace signature lines (normalized):" >&2
            printf '%s\n' "$ty_trace_lines" | LC_ALL=C sort -t$'\t' -k1,1n -k2,2 | head -50 >&2
            echo "---" >&2
            echo "TLC trace signature lines (normalized):" >&2
            printf '%s\n' "$tlc_trace_lines" | LC_ALL=C sort -t$'\t' -k1,1n -k2,2 | head -50 >&2
        fi
        if [ -n "$ty_final_state_block" ] && [ -n "$tlc_final_state_block" ] && [ "$ty_final_state_sig" != "$tlc_final_state_sig" ]; then
            echo "TY final state (normalized):" >&2
            printf '%s\n' "$ty_final_state_block" | head -50 >&2
            echo "---" >&2
            echo "TLC final state (normalized):" >&2
            printf '%s\n' "$tlc_final_state_block" | head -50 >&2
        fi
        echo "TY output:" >&2
        printf '%s\n' "$ty_output" | head -50 >&2
        echo "---"
        echo "TLC output:" >&2
        printf '%s\n' "$tlc_output" | head -50 >&2
        FAIL=$((FAIL + 1))
    fi
}

run_check() {
    local name="$1"
    local spec="$2"
    local expected="$3"
    local config="${4:-}"
    local skip_liveness="${5:-0}"
    local extra_args="${6:-}"
    local expected_error="${7:-}"  # W1: Optional expected error type (invariant/deadlock/liveness)

    if [ ! -f "$spec" ]; then
        echo "[ SKIP ] $name - spec not found"
        SKIP=$((SKIP + 1))
        return
    fi

    # W5: If TLC config has PROPERTY, liveness must be enabled.
    local effective_skip_liveness="$skip_liveness"
    if config_has_property "$config"; then
        effective_skip_liveness="0"
    fi

    # Run TY with only explicit per-test liveness settings; ambient TY_* is scrubbed.
    build_ty_fair_env "$effective_skip_liveness"
    local output=""
    if [ -n "$config" ] && [ -f "$config" ]; then
        if [ "$skip_liveness" = "1" ] && [ "$effective_skip_liveness" = "0" ]; then
            echo "[ INFO ] $name: enabling liveness (PROPERTY in config)"
        fi
        output="$("${TY_RUN_ENV[@]}" "$TY" check "$spec" --config "$config" --workers 1 --force $extra_args 2>&1)" || true
    else
        output="$("${TY_RUN_ENV[@]}" "$TY" check "$spec" --workers 1 --force $extra_args 2>&1)" || true
    fi

    # Extract state count
    local states=""
    states="$(echo "$output" | grep -oE "States found: [0-9,]+" | tr -d ',' | grep -oE "[0-9]+" || echo "0")"

    # W1: Detect errors in output
    local error_found=""
    if echo "$output" | grep -q "Error: Invariant"; then
        error_found="invariant"
    elif echo "$output" | grep -q "Error: Deadlock"; then
        error_found="deadlock"
    elif echo "$output" | grep -q "Error:.*liveness\|Error:.*temporal\|Error:.*stuttering"; then
        error_found="liveness"
    elif echo "$output" | grep -q "Error:"; then
        error_found="other"
    fi

    # W1: Verify error detection matches expectation
    local error_ok=true
    local error_msg=""

    if [ -n "$expected_error" ]; then
        # Error expected - verify it was found
        if [ -z "$error_found" ]; then
            error_ok=false
            error_msg="Expected $expected_error error, but TY found no error"
        elif [ "$error_found" != "$expected_error" ]; then
            # Allow some type flexibility (invariant=safety, liveness=temporal)
            case "$expected_error-$error_found" in
                invariant-safety|safety-invariant|liveness-temporal|temporal-liveness)
                    error_ok=true  # Acceptable mismatch
                    ;;
                *)
                    error_ok=false
                    error_msg="Expected $expected_error error, but TY found $error_found"
                    ;;
            esac
        fi
    else
        # No error expected - verify none was found
        if [ -n "$error_found" ]; then
            error_ok=false
            error_msg="TY found unexpected $error_found error"
        fi
    fi

    # Final pass/fail decision
    if [ "$states" = "$expected" ] && [ "$error_ok" = "true" ]; then
        echo "[ PASS ] $name: $states states (expected $expected)"
        PASS=$((PASS + 1))
    else
        if [ "$states" != "$expected" ]; then
            echo "[ FAIL ] $name: $states states (expected $expected)"
        else
            echo "[ FAIL ] $name: $error_msg"
        fi
        echo "Output: $output"
        FAIL=$((FAIL + 1))
    fi
}

# run_eval: Run an evaluator-only test (1 state, no transitions)
# These test expression evaluation via ASSUME/invariants, NOT model checking.
# Output uses [EVAL ] prefix to distinguish from model checking tests.
run_eval() {
    local name="$1"
    local spec="$2"
    local expected="$3"
    local config="${4:-}"
    local extra_args="${5:-}"

    if [ ! -f "$spec" ]; then
        echo "[ SKIP ] $name - spec not found"
        SKIP=$((SKIP + 1))
        return
    fi

    # Evaluator tests always skip liveness (no transitions), but scrub ambient TY_*.
    build_ty_fair_env 1
    local output=""
    if [ -n "$config" ] && [ -f "$config" ]; then
        output="$("${TY_RUN_ENV[@]}" "$TY" check "$spec" --config "$config" --workers 1 --force $extra_args 2>&1)" || true
    else
        output="$("${TY_RUN_ENV[@]}" "$TY" check "$spec" --workers 1 --force $extra_args 2>&1)" || true
    fi

    # Extract state count
    local states=""
    states="$(echo "$output" | grep -oE "States found: [0-9,]+" | tr -d ',' | grep -oE "[0-9]+" || echo "0")"

    # Check for unexpected errors
    if echo "$output" | grep -q "Error:"; then
        echo "[ FAIL ] $name (eval): unexpected error in evaluator test"
        echo "Output: $output"
        FAIL=$((FAIL + 1))
        return
    fi

    if [ "$states" = "$expected" ]; then
        echo "[ EVAL ] $name: $states states (evaluator-only)"
        EVAL=$((EVAL + 1))
    else
        echo "[ FAIL ] $name (eval): $states states (expected $expected)"
        echo "Output: $output"
        FAIL=$((FAIL + 1))
    fi
}
