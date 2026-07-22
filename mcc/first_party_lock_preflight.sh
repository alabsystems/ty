#!/usr/bin/env sh
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
#
# Validate authority-bearing first-party Cargo.lock sources before Cargo is
# allowed to compile the richer Rust provenance validator (or any dependency
# code). This script intentionally uses only POSIX sh plus standard grep/awk so
# a fresh Docker build can run it before fetching/building the workspace graph.

set -eu

if [ "$#" -ne 4 ]; then
    echo "usage: $0 CARGO_LOCK AY_REV TRUST_IR_REV TRUST_CG_REV" >&2
    exit 2
fi

lock_path=$1
ay_rev=$2
trust_ir_rev=$3
trust_cg_rev=$4

for revision in "$ay_rev" "$trust_ir_rev" "$trust_cg_rev"; do
    if ! printf '%s\n' "$revision" | grep -Eq '^[0-9a-f]{40}$'; then
        echo "first-party lock preflight: revision is not lowercase 40-hex: $revision" >&2
        exit 1
    fi
done

if [ ! -f "$lock_path" ]; then
    echo "first-party lock preflight: lockfile does not exist: $lock_path" >&2
    exit 1
fi

check_family() {
    family=$1
    name_pattern=$2
    expected_source=$3
    allowed_path_packages=${4-}
    allowed_path_version=${5-}

    awk \
        -v family="$family" \
        -v name_pattern="$name_pattern" \
        -v expected_source="$expected_source" \
        -v allowed_path_packages="$allowed_path_packages" \
        -v allowed_path_version="$allowed_path_version" '
        BEGIN {
            RS = ""
            found = 0
            found_expected_source = 0
            bad = 0
            allowed_count = split(allowed_path_packages, allowed_names, ",")
            for (allowed_index = 1; allowed_index <= allowed_count; allowed_index++) {
                if (allowed_names[allowed_index] != "") {
                    allowed_path[allowed_names[allowed_index]] = 1
                }
            }
        }
        {
            package_name = ""
            package_version = ""
            package_source = ""
            line_count = split($0, lines, "\n")
            for (line_index = 1; line_index <= line_count; line_index++) {
                line = lines[line_index]
                if (line ~ /^name = "[^"]+"$/) {
                    sub(/^name = "/, "", line)
                    sub(/"$/, "", line)
                    package_name = line
                } else if (line ~ /^version = "[^"]+"$/) {
                    sub(/^version = "/, "", line)
                    sub(/"$/, "", line)
                    package_version = line
                } else if (line ~ /^source = "[^"]+"$/) {
                    sub(/^source = "/, "", line)
                    sub(/"$/, "", line)
                    package_source = line
                }
            }
            if (package_name ~ name_pattern) {
                found++
                if (package_source == expected_source) {
                    found_expected_source++
                } else if (package_source == "" && (package_name in allowed_path)) {
                    observed_path[package_name]++
                    if (allowed_path_version != "" && package_version != allowed_path_version) {
                        printf "first-party lock preflight: audited %s cycle-boundary path package %s has version %s; expected %s\n", family, package_name, package_version, allowed_path_version > "/dev/stderr"
                        bad = 1
                    }
                } else {
                    observed = package_source == "" ? "<path/no-source>" : package_source
                    expected = expected_source == "" ? "<path/no-source>" : expected_source
                    printf "first-party lock preflight: %s package %s has source %s; expected %s\n", family, package_name, observed, expected > "/dev/stderr"
                    bad = 1
                }
            }
        }
        END {
            if (found == 0) {
                printf("first-party lock preflight: no %s packages found\n", family) > "/dev/stderr"
                bad = 1
            }
            if (expected_source != "" && found_expected_source == 0) {
                printf("first-party lock preflight: no %s packages use the expected exact Git source\n", family) > "/dev/stderr"
                bad = 1
            }
            for (allowed_name in allowed_path) {
                if (!(allowed_name in observed_path)) {
                    printf("first-party lock preflight: audited %s cycle-boundary path package %s is missing\n", family, allowed_name) > "/dev/stderr"
                    bad = 1
                } else if (observed_path[allowed_name] != 1) {
                    printf("first-party lock preflight: audited %s cycle-boundary path package %s occurs %d times; expected exactly once\n", family, allowed_name, observed_path[allowed_name]) > "/dev/stderr"
                    bad = 1
                }
            }
            exit bad
        }
        ' "$lock_path"
}

# The canonical Clean checkout resolves `clean-auto` through its audited
# sibling checkout.
# That crate's deliberate `../ay` cycle boundary creates a second Cargo source
# identity for exactly this closed AY package set. Docker subsequently clones
# `/ay` at AY_REV before Cargo runs, so these source-less entries resolve from
# the same immutable checkout. Keep this list exact: an added, missing, or
# differently sourced AY package fails before dependency code is compiled.
audited_clean_ay_path_packages='ay,ay-allsat,ay-arrays,ay-bv,ay-chc,ay-core,ay-count,ay-diff-logic,ay-dispatch,ay-dpll,ay-drat-check,ay-dt,ay-euf,ay-fp,ay-frontend,ay-intsat,ay-jit,ay-lia,ay-lra,ay-map,ay-milp,ay-model-check,ay-multiset,ay-nia,ay-nonlinear-common,ay-nra,ay-prefetch,ay-proof,ay-proof-common,ay-sat,ay-sat-congruence-core,ay-seq,ay-set,ay-strings,ay-sys,ay-translate'

check_family \
    AY \
    '^ay($|-)' \
    "git+https://github.com/alabsystems/ay.git?rev=$ay_rev#$ay_rev" \
    "$audited_clean_ay_path_packages" \
    '0.1.0'
check_family \
    TrustIR \
    '^trust-ir($|-)' \
    "git+https://github.com/alabsystems/trust-ir.git?rev=$trust_ir_rev#$trust_ir_rev"
check_family \
    trust-cg \
    '^trust-cg($|-)' \
    "git+https://github.com/alabsystems/trust-cg.git?rev=$trust_cg_rev#$trust_cg_rev"

# TY-root builds intentionally patch the complete Clean family to the audited
# sibling checkout. Any Clean Git/registry source in this lock is a mixed
# universe and must fail before compilation.
check_family Clean '^clean($|-)' ''
