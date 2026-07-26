#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# Validate and optionally expand the strict TY-vs-TLC corpus manifest without
# checking out or modifying the user's tlaplus/Examples worktree.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/check_tlc_corpus_manifest.sh [OPTIONS]

Options:
  --examples-repo PATH  Git repository containing the pinned Examples object
                        (default: $TLAPLUS_EXAMPLES_REPO or ~/tlaplus-examples)
  --emit PATH           Write the expanded, normalized JSON manifest to PATH;
                        use - for stdout
  -h, --help            Show this help
EOF
}

fail() {
    echo "corpus manifest validation failed: $*" >&2
    exit 1
}

for required in git jq awk comm sort; do
    command -v "$required" >/dev/null 2>&1 || fail "missing required command: $required"
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
manifest="$project_root/tests/tlc_comparison/strict_corpus_manifest.json"
baseline="$project_root/tests/tlc_comparison/spec_baseline.json"
examples_repo="${TLAPLUS_EXAMPLES_REPO:-${HOME}/tlaplus-examples}"
emit_path=""

while (($#)); do
    case "$1" in
        --examples-repo)
            (($# >= 2)) || fail "--examples-repo requires a path"
            examples_repo="$2"
            shift 2
            ;;
        --emit)
            (($# >= 2)) || fail "--emit requires a path or -"
            emit_path="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[[ -f "$manifest" ]] || fail "missing manifest: $manifest"
[[ -f "$baseline" ]] || fail "missing baseline: $baseline"
git -C "$examples_repo" rev-parse --git-dir >/dev/null 2>&1 ||
    fail "not a Git repository: $examples_repo"

pin="$(jq -er '.source.commit' "$manifest")"
source_root="$(jq -er '.source.root' "$manifest")"
expected_cfg_count="$(jq -er '.source.expected_cfg_count' "$manifest")"
expected_non_corpus_count="$(jq -er '.baseline_non_corpus.expected_count' "$manifest")"
work_equivalence_rule_id="$(
    jq -er '.work_equivalence_policy.default_eligible_rule_id' "$manifest"
)"
jq -e '
    .work_equivalence_policy == {
      schema_version: 1,
      default_eligible_rule_id: "exhaustive_generated_work_parity_v1",
      rules: {
        exhaustive_generated_work_parity_v1: {
          kind: "exhaustive_state_space",
          required_verdict: "holds",
          require_complete_exploration: true,
          distinct_state_parity: "exact",
          raw_initial_state_generation_parity: "exact",
          raw_successor_generation_parity: "exact",
          total_state_generation_parity: "exact",
          count_arm: "bfs_no_reduction_single_worker"
        }
      },
      outcome_dispositions: {
        expected_violation: "exclude_unless_predeclared_typed_rule",
        deadlock: "exclude_unless_predeclared_typed_rule",
        simulation: "exclude",
        randomized_external_operator: "exclude",
        external_io: "exclude",
        timeout: "missing_or_stale"
      }
    }
' "$manifest" >/dev/null ||
    fail "work-equivalence policy does not match the schema-v1 exhaustive contract"
git -C "$examples_repo" cat-file -e "${pin}^{commit}" 2>/dev/null ||
    fail "Examples repository does not contain pinned commit $pin"

audit_tmp="$(mktemp -d "${TMPDIR:-/tmp}/ty-corpus-manifest.XXXXXX")"
trap 'rm -rf -- "${audit_tmp:?}"' EXIT

cfgs="$audit_tmp/cfgs.txt"
rows="$audit_tmp/rows.tsv"
overrides="$audit_tmp/overrides.txt"
exclusions="$audit_tmp/exclusions.txt"
non_corpus_names="$audit_tmp/non-corpus-names.txt"
baseline_rows="$audit_tmp/baseline.tsv"
baseline_external="$audit_tmp/baseline-external.tsv"
baseline_external_cfgs="$audit_tmp/baseline-external-cfgs.txt"
baseline_gaps="$audit_tmp/baseline-gaps.txt"
actual_gaps="$audit_tmp/actual-gaps.txt"
manifest_rows="$audit_tmp/manifest-rows.tsv"
manifest_names="$audit_tmp/manifest-names.txt"

git -C "$examples_repo" ls-tree -r --name-only "$pin" -- "$source_root" |
    awk -v prefix="$source_root/" '
        index($0, prefix) == 1 && $0 ~ /\.cfg$/ {
            print substr($0, length(prefix) + 1)
        }
    ' |
    sort -u >"$cfgs"

actual_cfg_count="$(wc -l <"$cfgs" | tr -d ' ')"
[[ "$actual_cfg_count" == "$expected_cfg_count" ]] ||
    fail "pin has $actual_cfg_count cfg files; manifest expects $expected_cfg_count"

jq -r '.tla_path_overrides | keys[]' "$manifest" | sort -u >"$overrides"
jq -r '.eligibility.exclusions | keys[]' "$manifest" | sort -u >"$exclusions"

unknown_overrides="$(comm -23 "$overrides" "$cfgs")"
[[ -z "$unknown_overrides" ]] ||
    fail "TLA overrides name cfg files absent from the pin: $unknown_overrides"
unknown_exclusions="$(comm -23 "$exclusions" "$cfgs")"
[[ -z "$unknown_exclusions" ]] ||
    fail "exclusions name cfg files absent from the pin: $unknown_exclusions"

while IFS= read -r cfg_path; do
    tla_path="$(jq -r --arg cfg "$cfg_path" '.tla_path_overrides[$cfg] // empty' "$manifest")"
    if [[ -z "$tla_path" ]]; then
        tla_path="${cfg_path%.cfg}.tla"
    fi
    git -C "$examples_repo" cat-file -e "${pin}:${source_root}/${tla_path}" 2>/dev/null ||
        fail "no pinned TLA module for $cfg_path (mapped to $tla_path)"

    reason_code="$(jq -r --arg cfg "$cfg_path" \
        '.eligibility.exclusions[$cfg].reason_code // empty' "$manifest")"
    detail="$(jq -r --arg cfg "$cfg_path" \
        '.eligibility.exclusions[$cfg].detail // empty' "$manifest")"
    if [[ -n "$reason_code" ]]; then
        case "$reason_code" in
            deadlock_first_found_noncomparable|expected_violation_first_found_noncomparable|external_io_dependency|external_io_side_effect|nested_tool_driver|randomized_external_operator|semantic_assertion_only|simulation_only)
                ;;
            *)
                fail "exclusion for $cfg_path uses unsupported reason code: $reason_code"
                ;;
        esac
        [[ -n "${detail//[[:space:]]/}" ]] ||
            fail "exclusion for $cfg_path must include a nonempty detail"
        eligibility="excluded"
    else
        eligibility="eligible"
    fi
    [[ "$detail" != *$'\t'* && "$detail" != *$'\n'* ]] ||
        fail "exclusion detail for $cfg_path contains a tab or newline"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$cfg_path" "$tla_path" "$eligibility" "$reason_code" "$detail" >>"$rows"
done <"$cfgs"

jq -r '.rows[] | [.cfg_path, .tla_path, .name] | @tsv' "$manifest" >"$manifest_rows"
manifest_row_count="$(wc -l <"$manifest_rows" | tr -d ' ')"
[[ "$manifest_row_count" == "$expected_cfg_count" ]] ||
    fail "manifest has $manifest_row_count explicit rows; expected $expected_cfg_count"

awk -F $'\t' '
    NR == FNR { mapped[$1] = $2; next }
    {
        if (!($1 in mapped)) {
            print "explicit manifest row absent from resolved pin: " $1 >"/dev/stderr"
            bad = 1
        } else if (mapped[$1] != $2) {
            print "explicit manifest TLA mismatch for " $1 ": row=" $2 \
                  ", resolved=" mapped[$1] >"/dev/stderr"
            bad = 1
        }
    }
    END { exit bad }
' "$rows" "$manifest_rows" ||
    fail "explicit rows do not match pinned cfg enumeration and override rules"

cut -f3 "$manifest_rows" | sort >"$manifest_names"
duplicate_names="$(uniq -d "$manifest_names")"
[[ -z "$duplicate_names" ]] ||
    fail "explicit manifest row names are not unique: $duplicate_names"
if ! sort -c -t $'\t' -k1,1 "$manifest_rows" 2>/dev/null; then
    fail "explicit manifest rows must be sorted by cfg_path"
fi

jq -r '
    .baseline_non_corpus.apalache_removed.names[],
    (.baseline_non_corpus.ty_repository_replacements.entries | keys[])
' "$manifest" | sort -u >"$non_corpus_names"

actual_non_corpus_count="$(wc -l <"$non_corpus_names" | tr -d ' ')"
[[ "$actual_non_corpus_count" == "$expected_non_corpus_count" ]] ||
    fail "manifest classifies $actual_non_corpus_count non-corpus baseline rows; expected $expected_non_corpus_count"

jq -r '
    .specs | to_entries[] |
    [
      .key,
      (.value.category // ""),
      .value.source.tla_path,
      .value.source.cfg_path
    ] | @tsv
' "$baseline" >"$baseline_rows"

awk -F $'\t' '
    NR == FNR { excluded[$1] = 1; next }
    !($1 in excluded) { print $4 "\t" $3 "\t" $1 }
' "$non_corpus_names" "$baseline_rows" | sort >"$baseline_external"

awk -F $'\t' '
    NR == FNR { mapped[$1] = $2; next }
    {
        if (!($1 in mapped)) {
            print "baseline cfg absent from pin: " $1 " (row " $3 ")" >"/dev/stderr"
            bad = 1
        } else if (mapped[$1] != $2) {
            print "baseline TLA mapping mismatch for " $1 ": baseline=" $2 \
                  ", manifest=" mapped[$1] >"/dev/stderr"
            bad = 1
        }
    }
    END { exit bad }
' "$rows" "$baseline_external" ||
    fail "legacy baseline does not agree with normalized external-corpus mappings"

cut -f1 "$baseline_external" | sort -u >"$baseline_external_cfgs"
comm -23 "$cfgs" "$baseline_external_cfgs" >"$actual_gaps"
jq -r '.baseline_gaps | keys[]' "$manifest" | sort -u >"$baseline_gaps"
cmp -s "$actual_gaps" "$baseline_gaps" ||
    fail "baseline omissions differ from manifest (actual: $(tr '\n' ' ' <"$actual_gaps"))"

apalache_tla_prefix="$(jq -er '.baseline_non_corpus.apalache_removed.baseline_tla_prefix' "$manifest")"
apalache_cfg_prefix="$(jq -er '.baseline_non_corpus.apalache_removed.baseline_cfg_prefix' "$manifest")"
while IFS= read -r name; do
    read -r category tla_path cfg_path < <(
        jq -r --arg name "$name" '
            .specs[$name] |
            [(.category // ""), .source.tla_path, .source.cfg_path] | @tsv
        ' "$baseline"
    )
    [[ "$category" == "apalache" ]] ||
        fail "removed Apalache row $name has category $category"
    [[ "$tla_path" == "${apalache_tla_prefix}${name}.tla" ]] ||
        fail "unexpected stale TLA path for $name: $tla_path"
    [[ "$cfg_path" == "${apalache_cfg_prefix}${name}.cfg" ]] ||
        fail "unexpected stale cfg path for $name: $cfg_path"
    [[ ! -e "$project_root/$tla_path" && ! -e "$project_root/$cfg_path" ]] ||
        fail "removed Apalache fixture unexpectedly exists for $name"
done < <(jq -r '.baseline_non_corpus.apalache_removed.names[]' "$manifest")

historical_source_commit="$(jq -er \
    '.baseline_non_corpus.ty_repository_replacements.historical_source_commit' "$manifest")"
while IFS=$'\t' read -r name old_tla old_cfg replacement_tla replacement_cfg evidence; do
    read -r baseline_tla baseline_cfg < <(
        jq -r --arg name "$name" '
            .specs[$name] | [.source.tla_path, .source.cfg_path] | @tsv
        ' "$baseline"
    )
    [[ "$baseline_tla" == "$old_tla" && "$baseline_cfg" == "$old_cfg" ]] ||
        fail "baseline source paths changed for replacement row $name"
    [[ -f "$project_root/$replacement_tla" && -f "$project_root/$replacement_cfg" ]] ||
        fail "replacement files are missing for $name"

    if [[ "$evidence" == "byte_identical" ]]; then
        historical_tla="${old_tla#../../ty/}"
        historical_cfg="${old_cfg#../../ty/}"
        old_tla_blob="$(git -C "$project_root" rev-parse \
            "${historical_source_commit}:${historical_tla}")"
        old_cfg_blob="$(git -C "$project_root" rev-parse \
            "${historical_source_commit}:${historical_cfg}")"
        new_tla_blob="$(git -C "$project_root" hash-object "$replacement_tla")"
        new_cfg_blob="$(git -C "$project_root" hash-object "$replacement_cfg")"
        [[ "$old_tla_blob" == "$new_tla_blob" && "$old_cfg_blob" == "$new_cfg_blob" ]] ||
            fail "replacement for $name is not byte-identical to its historical source"
    fi
done < <(
    jq -r '
        .baseline_non_corpus.ty_repository_replacements.entries |
        to_entries[] |
        [
          .key,
          .value.baseline_tla_path,
          .value.baseline_cfg_path,
          .value.replacement_tla_path,
          .value.replacement_cfg_path,
          (.value.replacement_evidence // "")
        ] | @tsv
    ' "$manifest"
)

eligible_count="$(awk -F $'\t' '$3 == "eligible" { n += 1 } END { print n + 0 }' "$rows")"
excluded_count="$(awk -F $'\t' '$3 == "excluded" { n += 1 } END { print n + 0 }' "$rows")"
baseline_external_count="$(wc -l <"$baseline_external" | tr -d ' ')"
gap_count="$(wc -l <"$actual_gaps" | tr -d ' ')"

if [[ -n "$emit_path" ]]; then
    if command -v shasum >/dev/null 2>&1; then
        manifest_sha256="$(shasum -a 256 "$manifest" | awk '{print $1}')"
    else
        manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
    fi
    expanded="$audit_tmp/expanded.json"
    jq -Rn \
        --arg manifest "tests/tlc_comparison/strict_corpus_manifest.json" \
        --arg manifest_sha256 "$manifest_sha256" \
        --arg repository "$(jq -er '.source.repository' "$manifest")" \
        --arg commit "$pin" \
        --arg root "$source_root" \
        --arg work_equivalence_rule_id "$work_equivalence_rule_id" \
        '
        [
          inputs |
          split("\t") |
          {
            row_id: .[0],
            cfg_path: .[0],
            tla_path: .[1],
            eligibility: .[2],
            reason_code: (if .[3] == "" then null else .[3] end),
            detail: (if .[4] == "" then null else .[4] end),
            work_equivalence: (
              if .[2] == "eligible"
              then {
                schema_version: 1,
                rule_id: $work_equivalence_rule_id
              }
              else null
              end
            )
          }
        ] as $rows |
        {
          schema_version: 1,
          generated_from: {
            manifest: $manifest,
            manifest_sha256: $manifest_sha256
          },
          source: {
            repository: $repository,
            commit: $commit,
            root: $root
          },
          work_equivalence: {
            schema_version: 1,
            default_eligible_rule_id: $work_equivalence_rule_id
          },
          summary: {
            total: ($rows | length),
            eligible: ([$rows[] | select(.eligibility == "eligible")] | length),
            excluded: ([$rows[] | select(.eligibility == "excluded")] | length)
          },
          rows: $rows
        }
        ' <"$rows" >"$expanded"
    if [[ "$emit_path" == "-" ]]; then
        cat "$expanded"
    else
        mkdir -p "$(dirname "$emit_path")"
        mv "$expanded" "$emit_path"
    fi
fi

echo "corpus manifest OK: ${actual_cfg_count} pinned rows (${eligible_count} eligible, ${excluded_count} excluded); baseline covers ${baseline_external_count} and omitted ${gap_count}; ${actual_non_corpus_count} legacy non-corpus rows classified" >&2
