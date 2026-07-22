// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Static source guard for compiled BFS/JIT hot-loop materialization.
//!
//! The production fused loop must stay on flat successor buffers. Legacy owned
//! APIs still exist in `tla-jit-abi` for compatibility, but any new nested
//! materialization in the model-checker hot path should be a deliberate,
//! allowlisted exception.

use std::fs;
use std::path::PathBuf;

const GUARDED_FILES: &[&str] = &[
    "crates/tla-check/src/check/model_checker/bfs/compiled_bfs_loop.rs",
    "crates/tla-check/src/check/model_checker/bfs/compiled_step_trait.rs",
    "crates/tla-check/src/check/model_checker/bfs/full_state_successors.rs",
    // trust_cg_dispatch and run_helpers were refactored from single files into directory
    // modules; guard every (non-test) submodule so the hot-loop coverage is not silently lost.
    "crates/tla-check/src/check/model_checker/trust_cg_dispatch/mod.rs",
    "crates/tla-check/src/check/model_checker/trust_cg_dispatch/abi.rs",
    "crates/tla-check/src/check/model_checker/trust_cg_dispatch/admission.rs",
    "crates/tla-check/src/check/model_checker/trust_cg_dispatch/config.rs",
    "crates/tla-check/src/check/model_checker/run_helpers/mod.rs",
    "crates/tla-check/src/check/model_checker/run_helpers/jit_successors.rs",
    "crates/tla-check/src/check/model_checker/run_helpers/jit_tuning.rs",
    "crates/tla-check/src/check/model_checker/run_helpers/bfs_profile.rs",
    "crates/tla-jit-abi/src/bfs_output.rs",
    "crates/tla-trust-cg/src/bfs_level.rs",
    "crates/tla-trust-cg/src/native_bfs.rs",
];

const FORBIDDEN_PATTERNS: &[ForbiddenPattern] = &[
    ForbiddenPattern {
        needle: "Vec<Vec<i64>>",
        description: "nested owned flat-state materialization",
    },
    ForbiddenPattern {
        needle: "BfsStepOutput",
        description: "legacy owned compiled-BFS step output",
    },
    ForbiddenPattern {
        needle: "succ.to_vec()",
        description: "per-successor slice clone",
    },
    ForbiddenPattern {
        needle: "successor.to_vec()",
        description: "per-successor slice clone",
    },
];

const ALLOWLIST: &[AllowlistedRegion] = &[
    AllowlistedRegion {
        file: "crates/tla-jit-abi/src/bfs_output.rs",
        needle: "BfsStepOutput",
        start_marker: "pub struct BfsStepOutput",
        end_marker: "pub struct FlatBfsStepOutput",
        reason: "legacy owned output type definition; hot loops must not consume it",
    },
    AllowlistedRegion {
        file: "crates/tla-jit-abi/src/bfs_output.rs",
        needle: "Vec<Vec<i64>>",
        start_marker: "pub struct BfsStepOutput",
        end_marker: "pub struct FlatBfsStepOutput",
        reason: "legacy owned output type definition; hot loops must not consume it",
    },
    AllowlistedRegion {
        file: "crates/tla-jit-abi/src/bfs_output.rs",
        needle: "Vec<Vec<i64>>",
        start_marker: "pub struct BfsBatchResult",
        end_marker: "pub enum BfsStepError",
        reason: "legacy batch result type definition; hot loops must not consume it",
    },
    AllowlistedRegion {
        file: "crates/tla-check/src/check/model_checker/trust_cg_dispatch/mod.rs",
        needle: "Vec<Vec<i64>>",
        start_marker: "enum TrustCgInnerExistsExpansionProofKind",
        end_marker: "struct RuntimeTaggedScalarOrSetTypeProof",
        reason: "inner-EXISTS binding proof metadata; not successor materialization",
    },
    AllowlistedRegion {
        file: "crates/tla-check/src/check/model_checker/bfs/compiled_bfs_loop.rs",
        needle: "succ.to_vec()",
        start_marker: "collect owned native successors",
        end_marker: "Attribution claimed complete but an index is missing",
        reason: "verification crosscheck peek-run collects owned native successors to compare \
                 against the interpreter; not the production hot loop",
    },
];

#[derive(Clone, Copy)]
struct ForbiddenPattern {
    needle: &'static str,
    description: &'static str,
}

#[derive(Clone, Copy)]
struct AllowlistedRegion {
    file: &'static str,
    needle: &'static str,
    start_marker: &'static str,
    end_marker: &'static str,
    reason: &'static str,
}

#[test]
fn compiled_bfs_jit_hot_loop_sources_reject_legacy_materialization() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut violations = Vec::new();

    for file in GUARDED_FILES {
        let path = repo_root.join(file);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        let production_source = source_before_cfg_test_mod(&source);
        let production_source = strip_cfg_test_items_preserving_lines(production_source);
        let lines: Vec<&str> = production_source.lines().collect();
        let stripped_lines = strip_comments_preserving_lines(&production_source);

        for (line_idx, line) in stripped_lines.iter().enumerate() {
            for pattern in FORBIDDEN_PATTERNS {
                if !line_contains_pattern(line, pattern.needle) {
                    continue;
                }
                if is_allowlisted(file, pattern.needle, line_idx, &lines) {
                    continue;
                }
                violations.push(format!(
                    "{}:{} contains forbidden {} `{}`: {}",
                    file,
                    line_idx + 1,
                    pattern.description,
                    pattern.needle,
                    lines[line_idx].trim()
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "compiled BFS/JIT hot-loop legacy materialization guard failed:\n{}",
        violations.join("\n")
    );
}

#[test]
fn flat_primary_streaming_prefilter_hashes_before_flat_materialization() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let helpers_path =
        repo_root.join("crates/tla-check/src/check/model_checker/run_helpers/mod.rs");
    let helpers = fs::read_to_string(&helpers_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", helpers_path.display()));

    let gate = slice_between(
        &helpers,
        "fn flat_successor_prefilter_streaming_candidate",
        "fn compiled_bfs_step_width_matches_flat_frontier",
    );
    for required_gate in [
        "!self.flat_state_primary",
        "cache_for_liveness",
        "!self.config.constraints.is_empty()",
        "!self.config.action_constraints.is_empty()",
        "self.por.independence.is_some()",
        "self.coverage.collect",
        "!self.config.trace_invariants.is_empty()",
        "!self.symmetry.perms.is_empty()",
        "self.compiled.cached_view_name.is_some()",
        "self.inline_liveness_active()",
    ] {
        assert!(
            gate.contains(required_gate),
            "streaming flat prefilter gate must contain `{required_gate}`"
        );
    }

    let prefilter_helper = slice_between(
        &helpers,
        "fn push_prefiltered_flat_successor_from_scratch",
        "/// Check if state exploration limit has been reached.",
    );
    assert_markers_in_order(
        prefilter_helper,
        &[
            "fingerprint_flat_compiled",
            "is_state_seen_checked",
            "FlatState::from_buffer",
        ],
    );

    let full_path =
        repo_root.join("crates/tla-check/src/check/model_checker/bfs/full_state_successors.rs");
    let full = fs::read_to_string(&full_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", full_path.display()));
    let flat_primary = slice_between(
        &full,
        "fn process_flat_state_primary_successors",
        "fn process_flat_state_primary_prefiltered_successors",
    );
    assert_markers_in_order(
        flat_primary,
        &[
            "generate_successors_filtered_flat_prefiltered",
            "generate_successors_filtered_flat(&parent_flat)",
        ],
    );
}

#[test]
fn compiled_bfs_builders_reject_eval_implied_actions_before_artifact_construction() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let helpers_path =
        repo_root.join("crates/tla-check/src/check/model_checker/run_helpers/mod.rs");
    let helpers = fs::read_to_string(&helpers_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", helpers_path.display()));

    // `reject_marker` is the source predicate each builder uses to decline a
    // configuration whose implied actions must be evaluated by the interpreter
    // before any compiled artifact is built. The trust-cg LEVEL builder declines
    // via the more precise `implied_actions_require_interpreter_eval()` predicate
    // (commit f6287aca, "Fail closed unsafe Trust-CG implied-action paths"):
    // native-capable implied actions are evaluated in compiled code, so the
    // raw `eval_implied_actions.is_empty()` check would over-reject. The other
    // three builders still gate on the raw non-empty check.
    for (function, end_marker, construction_marker, return_marker, reject_marker) in [
        (
            "fn try_build_compiled_bfs_step",
            "/// Attempt to build a fused `CompiledBfsLevel`",
            "let meta = match self.compiled.split_action_meta.as_ref()",
            "return;",
            "!self.compiled.eval_implied_actions.is_empty()",
        ),
        (
            "fn try_build_compiled_bfs_level",
            "/// Check whether the compiled BFS path should be used.",
            "let meta = match self.compiled.split_action_meta.as_ref()",
            "return;",
            "!self.compiled.eval_implied_actions.is_empty()",
        ),
        (
            "fn try_build_trust_cg_compiled_bfs_step",
            "fn try_build_trust_cg_compiled_bfs_level",
            "let meta = match self.compiled.split_action_meta.as_ref()",
            "return None;",
            "!self.compiled.eval_implied_actions.is_empty()",
        ),
        (
            "fn try_build_trust_cg_compiled_bfs_level",
            "pub(in crate::check) fn initialize_trust_cg_cache",
            "let meta = match self.compiled.split_action_meta.as_ref()",
            "return None;",
            "self.implied_actions_require_interpreter_eval()",
        ),
    ] {
        let body = slice_between(&helpers, function, end_marker);
        let pre_construction = slice_between(body, reject_marker, construction_marker);
        assert!(
            pre_construction.contains(return_marker),
            "`{function}` must return with `{return_marker}` after rejecting implied actions"
        );
        assert_markers_in_order(body, &[reject_marker, return_marker, construction_marker]);
    }
}

fn source_before_cfg_test_mod(source: &str) -> &str {
    if let Some(idx) = source.find("\n#[cfg(test)]\nmod tests") {
        &source[..idx]
    } else {
        source
    }
}

fn strip_cfg_test_items_preserving_lines(source: &str) -> String {
    let mut lines: Vec<String> = source.lines().map(str::to_owned).collect();
    let mut idx = 0;

    while idx < lines.len() {
        if lines[idx].trim() != "#[cfg(test)]" {
            idx += 1;
            continue;
        }

        lines[idx].clear();
        idx += 1;
        while idx < lines.len() && lines[idx].trim_start().starts_with("#[") {
            lines[idx].clear();
            idx += 1;
        }

        let mut brace_depth = 0usize;
        let mut saw_brace = false;
        while idx < lines.len() {
            let original = std::mem::take(&mut lines[idx]);
            saw_brace |= update_brace_depth(&original, &mut brace_depth);
            let ends_semicolon_item = !saw_brace && original.trim_end().ends_with(';');
            idx += 1;
            if ends_semicolon_item || (saw_brace && brace_depth == 0) {
                break;
            }
        }
    }

    lines.join("\n")
}

fn update_brace_depth(line: &str, brace_depth: &mut usize) -> bool {
    let mut saw_brace = false;
    for ch in line.chars() {
        match ch {
            '{' => {
                *brace_depth += 1;
                saw_brace = true;
            }
            '}' => {
                *brace_depth = brace_depth.saturating_sub(1);
                saw_brace = true;
            }
            _ => {}
        }
    }
    saw_brace
}

fn strip_comments_preserving_lines(source: &str) -> Vec<String> {
    let mut stripped = Vec::new();
    let mut in_block_comment = false;

    for line in source.lines() {
        let mut out = String::new();
        let mut rest = line;

        while !rest.is_empty() {
            if in_block_comment {
                if let Some(end) = rest.find("*/") {
                    rest = &rest[end + 2..];
                    in_block_comment = false;
                } else {
                    rest = "";
                }
                continue;
            }

            let line_comment = rest.find("//");
            let block_comment = rest.find("/*");
            match (line_comment, block_comment) {
                (Some(line_idx), Some(block_idx)) if line_idx < block_idx => {
                    out.push_str(&rest[..line_idx]);
                    break;
                }
                (Some(_), Some(block_idx)) | (None, Some(block_idx)) => {
                    out.push_str(&rest[..block_idx]);
                    rest = &rest[block_idx + 2..];
                    in_block_comment = true;
                }
                (Some(line_idx), None) => {
                    out.push_str(&rest[..line_idx]);
                    break;
                }
                (None, None) => {
                    out.push_str(rest);
                    break;
                }
            }
        }

        stripped.push(out);
    }

    stripped
}

fn line_contains_pattern(line: &str, needle: &str) -> bool {
    if needle == "BfsStepOutput" {
        contains_rust_ident(line, needle)
    } else {
        line.contains(needle)
    }
}

fn contains_rust_ident(line: &str, needle: &str) -> bool {
    line.match_indices(needle).any(|(idx, _)| {
        let before = line[..idx].chars().next_back();
        let after = line[idx + needle.len()..].chars().next();
        !before.is_some_and(is_rust_ident_char) && !after.is_some_and(is_rust_ident_char)
    })
}

fn is_rust_ident_char(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

fn is_allowlisted(file: &str, needle: &str, line_idx: usize, lines: &[&str]) -> bool {
    ALLOWLIST
        .iter()
        .filter(|region| region.file == file && region.needle == needle)
        .any(|region| {
            let (start, end) = region
                .line_range(lines)
                .unwrap_or_else(|err| panic!("invalid allowlist: {err}"));
            start <= line_idx && line_idx < end
        })
}

impl AllowlistedRegion {
    fn line_range(&self, lines: &[&str]) -> Result<(usize, usize), String> {
        let start = line_containing(lines, self.start_marker).ok_or_else(|| {
            format!(
                "{} allowlist for `{}` missing start marker `{}` ({})",
                self.file, self.needle, self.start_marker, self.reason
            )
        })?;
        let end = line_containing_from(lines, self.end_marker, start + 1).ok_or_else(|| {
            format!(
                "{} allowlist for `{}` missing end marker `{}` ({})",
                self.file, self.needle, self.end_marker, self.reason
            )
        })?;
        if start >= end {
            return Err(format!(
                "{} allowlist for `{}` has empty range `{}`..`{}` ({})",
                self.file, self.needle, self.start_marker, self.end_marker, self.reason
            ));
        }
        Ok((start, end))
    }
}

fn line_containing(lines: &[&str], marker: &str) -> Option<usize> {
    line_containing_from(lines, marker, 0)
}

fn line_containing_from(lines: &[&str], marker: &str, start: usize) -> Option<usize> {
    lines
        .iter()
        .enumerate()
        .skip(start)
        .find_map(|(idx, line)| line.contains(marker).then_some(idx))
}

fn slice_between<'a>(source: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
    let start = source
        .find(start_marker)
        .unwrap_or_else(|| panic!("missing start marker `{start_marker}`"));
    let rel_end = source[start..]
        .find(end_marker)
        .unwrap_or_else(|| panic!("missing end marker `{end_marker}` after `{start_marker}`"));
    &source[start..start + rel_end]
}

fn assert_markers_in_order(source: &str, markers: &[&str]) {
    let mut cursor = 0usize;
    for marker in markers {
        let rel = source[cursor..]
            .find(marker)
            .unwrap_or_else(|| panic!("missing ordered marker `{marker}`"));
        cursor += rel + marker.len();
    }
}
