// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Liveness verdict comparison support.
//!
//! Ported from `scripts/liveness_verdict_lib.py`. Provides the
//! TLC↔TY verdict-classification primitives, state-count parsers,
//! trace-info extraction, and TLA path manipulation used by the
//! liveness parity matrix (`scripts/liveness_verdict_matrix.py`,
//! `scripts/test_all_liveness.sh`).
//!
//! All helpers are pure functions over captured tool output so the
//! liveness matrix binary can wire them to subprocess transcripts.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Discovered temporal-spec target (TLC↔TY parity).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecTarget {
    /// Spec name (module / fixture identifier).
    pub name: String,
    /// Where the target was discovered.
    pub source: SpecSource,
    /// Path to the `.tla` specification.
    pub spec_path: PathBuf,
    /// Path to the `.cfg` model configuration.
    pub cfg_path: PathBuf,
    /// Temporal markers found (see [`temporal_markers`]): `"PROPERTY"`,
    /// `"WF_/SF_"`, both, or empty.
    pub temporal_markers: Vec<String>,
}

/// Where a spec target was discovered. Matches the Python script's
/// `source` field (`"baseline"` or `"tests"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpecSource {
    /// Discovered from `tests/tlc_comparison/spec_baseline.json`.
    Baseline,
    /// Discovered from local `test_specs/` configs.
    Tests,
}

impl SpecSource {
    /// The source's wire name (`"baseline"` or `"tests"`), matching the
    /// Python script's `source` field.
    pub fn as_str(self) -> &'static str {
        match self {
            SpecSource::Baseline => "baseline",
            SpecSource::Tests => "tests",
        }
    }
}

/// Trace structure summary produced by [`parse_trace_info`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TraceInfo {
    /// Number of `State N:` headers found in the transcript.
    pub state_count: usize,
    /// Whether the transcript reports a stuttering step.
    pub has_stuttering: bool,
    /// `Some(hex)` SHA-256 of the canonicalised assignment list when at
    /// least one assignment was parsed; otherwise `None`.
    pub signature: Option<String>,
}

/// Tool being classified. Differs in error-message phrasing and trace
/// shape between TLC (Java) and TY (Rust).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    /// The reference TLC model checker (Java).
    Tlc,
    /// This toolchain, `ty` (Rust).
    Ty,
}

/// Classified verdict for a single run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VerdictStatus {
    /// Model checking completed with no error found.
    Success,
    /// A temporal (liveness) property was violated.
    Liveness,
    /// An invariant (safety property) was violated.
    Invariant,
    /// A deadlock was reached.
    Deadlock,
    /// The run failed with an error (non-zero exit or an `error:` line).
    Error,
    /// The run timed out (exit code 124).
    Timeout,
    /// The output could not be classified into any of the above.
    Unknown,
}

impl VerdictStatus {
    /// The verdict's lowercase wire name (e.g. `"success"`, `"liveness"`).
    pub fn as_str(self) -> &'static str {
        match self {
            VerdictStatus::Success => "success",
            VerdictStatus::Liveness => "liveness",
            VerdictStatus::Invariant => "invariant",
            VerdictStatus::Deadlock => "deadlock",
            VerdictStatus::Error => "error",
            VerdictStatus::Timeout => "timeout",
            VerdictStatus::Unknown => "unknown",
        }
    }
}

/// Prepend `extra_path` to a `TLA_PATH`-style separator list.
///
/// `existing` is the current `TLA_PATH` value (use `None` if unset).
/// Returns the new value. Matches `prepend_to_tla_path` in
/// `liveness_verdict_lib.py`: deduplicates `extra_path` if it already
/// appears anywhere in the list, and uses the platform path separator.
pub fn prepend_to_tla_path(existing: Option<&str>, extra_path: &Path) -> String {
    let extra = extra_path.to_string_lossy().to_string();
    if extra.is_empty() {
        return existing.unwrap_or("").to_string();
    }
    let Some(existing) = existing.filter(|s| !s.is_empty()) else {
        return extra;
    };
    let sep = path_separator();
    let entries: Vec<&str> = existing.split(sep).filter(|e| !e.is_empty()).collect();
    let mut out = vec![extra.as_str()];
    for entry in &entries {
        if *entry == extra {
            continue;
        }
        out.push(entry);
    }
    out.join(&sep.to_string())
}

fn path_separator() -> char {
    if cfg!(windows) {
        ';'
    } else {
        ':'
    }
}

/// `true` iff the `.cfg` body contains a top-level `PROPERTY` directive.
///
/// Mirrors `config_has_property` in `liveness_verdict_lib.py`. Reads the
/// file as UTF-8 with lossy decoding for resilience against stray
/// non-UTF-8 bytes in legacy configs.
pub fn config_has_property(cfg_text: &str) -> bool {
    cfg_text.lines().any(|line| {
        let trimmed = line.trim_start();
        let rest = match trimmed.strip_prefix("PROPERTY") {
            Some(r) => r,
            None => return false,
        };
        rest.is_empty() || rest.starts_with(|c: char| c.is_whitespace())
    })
}

/// `true` iff the spec body mentions a TLC fairness operator (`WF_` or
/// `SF_`). The check is whole-word: `\bWF_` / `\bSF_`.
pub fn module_has_fairness(spec_text: &str) -> bool {
    contains_word_prefix(spec_text, "WF_") || contains_word_prefix(spec_text, "SF_")
}

fn contains_word_prefix(text: &str, prefix: &str) -> bool {
    let bytes = text.as_bytes();
    let pbytes = prefix.as_bytes();
    let mut start = 0;
    while start + pbytes.len() <= bytes.len() {
        let window = &bytes[start..start + pbytes.len()];
        if window == pbytes {
            let prev_ok = start == 0 || !is_word_byte(bytes[start - 1]);
            if prev_ok {
                return true;
            }
        }
        start += 1;
    }
    false
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Compute the temporal markers list. The Python script returns
/// `("PROPERTY", "WF_/SF_")`, `("PROPERTY",)`, `("WF_/SF_",)`, or `()`.
pub fn temporal_markers(spec_text: &str, cfg_text: &str) -> Vec<String> {
    let mut markers = Vec::new();
    if config_has_property(cfg_text) {
        markers.push("PROPERTY".to_string());
    }
    if module_has_fairness(spec_text) {
        markers.push("WF_/SF_".to_string());
    }
    markers
}

/// Parse the last `N distinct states found` count from TLC stdout.
pub fn parse_tlc_states(output: &str) -> Option<u64> {
    parse_last_count_before(output, "distinct states found")
}

fn parse_last_count_before(output: &str, marker: &str) -> Option<u64> {
    let mut found = None;
    let mut search_start = 0;
    while let Some(idx) = output[search_start..].find(marker) {
        let absolute = search_start + idx;
        let head = &output[..absolute];
        let trimmed = head.trim_end();
        let digit_start = trimmed
            .char_indices()
            .rev()
            .find(|(_, c)| !(c.is_ascii_digit() || *c == ','))
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        let token = &trimmed[digit_start..];
        let cleaned: String = token.chars().filter(|c| *c != ',').collect();
        if let Ok(value) = cleaned.parse::<u64>() {
            found = Some(value);
        }
        search_start = absolute + marker.len();
    }
    found
}

/// Parse `States found: <N>` from TY stdout. Returns the first match.
pub fn parse_ty_states(output: &str) -> Option<u64> {
    for line in output.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("States found:") else {
            continue;
        };
        let token = rest.split_whitespace().next()?;
        let cleaned: String = token.chars().filter(|c| *c != ',').collect();
        return cleaned.parse::<u64>().ok();
    }
    // Also support an inline `States found: N` substring (matches the
    // Python `re.search`).
    let needle = "States found:";
    let idx = output.find(needle)?;
    let tail = &output[idx + needle.len()..];
    let token = tail.split_whitespace().next()?;
    let cleaned: String = token.chars().filter(|c| *c != ',').collect();
    cleaned.parse::<u64>().ok()
}

/// Classify TLC stdout + return code into a [`VerdictStatus`].
pub fn classify_tlc_status(output: &str, return_code: i32) -> VerdictStatus {
    if return_code == 124 {
        return VerdictStatus::Timeout;
    }
    let lower = output.to_ascii_lowercase();
    if lower.contains("temporal properties were violated") {
        return VerdictStatus::Liveness;
    }
    if lower.contains("invariant") && lower.contains("violated") {
        return VerdictStatus::Invariant;
    }
    if lower.contains("deadlock reached") {
        return VerdictStatus::Deadlock;
    }
    if lower.contains("model checking completed. no error has been found") {
        return VerdictStatus::Success;
    }
    if parse_tlc_states(output).is_some() && !lower.contains("error:") && return_code == 0 {
        return VerdictStatus::Success;
    }
    if return_code != 0 {
        return VerdictStatus::Error;
    }
    if lower.contains("error:") {
        return VerdictStatus::Error;
    }
    VerdictStatus::Unknown
}

/// Classify TY stdout + return code into a [`VerdictStatus`].
pub fn classify_ty_status(output: &str, return_code: i32) -> VerdictStatus {
    if return_code == 124 {
        return VerdictStatus::Timeout;
    }
    let lower = output.to_ascii_lowercase();
    if lower.contains("liveness") && lower.contains("violated") {
        return VerdictStatus::Liveness;
    }
    if lower.contains("invariant") && lower.contains("violated") {
        return VerdictStatus::Invariant;
    }
    if lower.contains("deadlock") {
        return VerdictStatus::Deadlock;
    }
    if lower.contains("model checking complete")
        && lower.contains("no errors found")
        && return_code == 0
    {
        return VerdictStatus::Success;
    }
    if return_code != 0 {
        return VerdictStatus::Error;
    }
    if lower.contains("error:") {
        return VerdictStatus::Error;
    }
    VerdictStatus::Unknown
}

/// Normalize one assignment line (drop the leading `/\` connective).
pub fn normalize_assignment_line(line: &str) -> String {
    let trimmed = line.trim();
    trimmed
        .strip_prefix("/\\")
        .map(|rest| rest.trim().to_string())
        .unwrap_or_else(|| trimmed.to_string())
}

/// Parse trace structure from a TLC or TY transcript.
///
/// Matches the Python implementation: counts `State N:` headers, picks
/// the assignment-line predicate per tool (TLC accepts `/\` *or* space
/// continuations; TY uses space-indented bodies only), and emits a
/// SHA-256 over the sorted assignments.
pub fn parse_trace_info(output: &str, tool: Tool) -> TraceInfo {
    let mut states: Vec<u64> = Vec::new();
    let mut assignments: Vec<(u64, String)> = Vec::new();
    let mut current: Option<u64> = None;

    for raw in output.lines() {
        if let Some(state_id) = parse_state_header(raw, tool) {
            current = Some(state_id);
            states.push(state_id);
            continue;
        }
        if current.is_none() {
            continue;
        }
        if is_assignment_line(raw, tool) {
            let cleaned = normalize_assignment_line(raw);
            if !cleaned.is_empty() {
                assignments.push((current.unwrap(), cleaned));
            }
            continue;
        }
        if raw.trim().is_empty() {
            continue;
        }
        // Anything else terminates the current state block.
        current = None;
    }

    let signature = if assignments.is_empty() {
        None
    } else {
        let mut sorted: BTreeSet<(u64, String)> = BTreeSet::new();
        for entry in &assignments {
            sorted.insert(entry.clone());
        }
        let payload: String = sorted
            .iter()
            .map(|(idx, text)| format!("{idx}\t{text}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut hasher = Sha256::new();
        hasher.update(payload.as_bytes());
        Some(format!("{:x}", hasher.finalize()))
    };

    TraceInfo {
        state_count: states.len(),
        has_stuttering: output.to_ascii_lowercase().contains("stuttering"),
        signature,
    }
}

fn parse_state_header(line: &str, tool: Tool) -> Option<u64> {
    let rest = line.strip_prefix("State")?;
    let rest = rest.strip_prefix(' ')?;
    // Find the run of digits.
    let mut end = 0;
    for (i, c) in rest.char_indices() {
        if c.is_ascii_digit() {
            end = i + c.len_utf8();
            continue;
        }
        break;
    }
    if end == 0 {
        return None;
    }
    let id = rest[..end].parse::<u64>().ok()?;
    let separator = rest[end..].chars().next()?;
    match tool {
        Tool::Tlc => {
            if separator == ':' {
                Some(id)
            } else {
                None
            }
        }
        Tool::Ty => {
            // TY accepts ': ' or ' ' after the digits.
            if separator == ':' || separator == ' ' {
                Some(id)
            } else {
                None
            }
        }
    }
}

fn is_assignment_line(line: &str, tool: Tool) -> bool {
    match tool {
        Tool::Tlc => line.starts_with("/\\") || line.starts_with(' '),
        Tool::Ty => line.starts_with(' '),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepend_inserts_extra_first() {
        let new = prepend_to_tla_path(Some("/foo"), Path::new("/bar"));
        assert!(new.starts_with("/bar"));
        assert!(new.contains("/foo"));
    }

    #[test]
    fn prepend_dedupes_existing() {
        let existing = format!("/foo{sep}/bar", sep = path_separator());
        let new = prepend_to_tla_path(Some(&existing), Path::new("/bar"));
        let entries: Vec<&str> = new.split(path_separator()).collect();
        assert_eq!(entries.first().copied(), Some("/bar"));
        assert_eq!(entries.iter().filter(|e| **e == "/bar").count(), 1);
    }

    #[test]
    fn prepend_empty_existing_returns_extra() {
        assert_eq!(prepend_to_tla_path(None, Path::new("/foo")), "/foo");
        assert_eq!(prepend_to_tla_path(Some(""), Path::new("/foo")), "/foo");
    }

    #[test]
    fn config_property_detection() {
        assert!(config_has_property("SPECIFICATION Next\nPROPERTY Live\n"));
        assert!(config_has_property("PROPERTY"));
        // Substring within an identifier doesn't count.
        assert!(!config_has_property("XYZPROPERTYABC\n"));
        assert!(!config_has_property(""));
    }

    #[test]
    fn module_fairness_detection() {
        assert!(module_has_fairness(
            "Spec == Init /\\ [][Next]_v /\\ WF_v(Next)"
        ));
        assert!(module_has_fairness("Spec /\\ SF_v(Step)"));
        assert!(!module_has_fairness("ConstantWF_butNoUnderscoreInDef"));
        assert!(!module_has_fairness("nothing here"));
    }

    #[test]
    fn temporal_markers_combines_signals() {
        let markers = temporal_markers(
            "Spec == Init /\\ [][Next]_v /\\ WF_v(Next)",
            "PROPERTY Live\n",
        );
        assert_eq!(markers, vec!["PROPERTY".to_string(), "WF_/SF_".to_string()]);
        assert!(temporal_markers("only spec", "nothing").is_empty());
    }

    #[test]
    fn parses_tlc_states_takes_last_match() {
        let stdout = "\
            Progress(7) at 2024-01-01 12:00:00: 4 states generated, 3 distinct states found\n\
            1234 distinct states found in some intermediate line\n\
            Model checking completed.\n\
            Finished in 0s at (2024-01-01 12:00:00)\n\
            1,234,567 distinct states found, 0 left on queue.\n\
        ";
        assert_eq!(parse_tlc_states(stdout), Some(1_234_567));
    }

    #[test]
    fn parses_ty_states_with_commas() {
        let stdout = "States found: 1,234,567 (queue empty)\n";
        assert_eq!(parse_ty_states(stdout), Some(1_234_567));
    }

    #[test]
    fn classify_tlc_success() {
        let stdout = "\
            Model checking completed. No error has been found.\n\
            42 distinct states found.\n\
        ";
        assert_eq!(classify_tlc_status(stdout, 0), VerdictStatus::Success);
    }

    #[test]
    fn classify_tlc_liveness() {
        let stdout = "Temporal properties were violated";
        assert_eq!(classify_tlc_status(stdout, 1), VerdictStatus::Liveness);
    }

    #[test]
    fn classify_tlc_timeout() {
        assert_eq!(classify_tlc_status("", 124), VerdictStatus::Timeout);
    }

    #[test]
    fn classify_ty_success() {
        let stdout = "Model checking complete. No errors found.";
        assert_eq!(classify_ty_status(stdout, 0), VerdictStatus::Success);
    }

    #[test]
    fn classify_ty_liveness() {
        let stdout = "Liveness property violated: <>P";
        assert_eq!(classify_ty_status(stdout, 1), VerdictStatus::Liveness);
    }

    #[test]
    fn trace_info_counts_states_and_hashes() {
        let stdout = "\
            State 1: <Init>\n\
            /\\ x = 0\n\
            /\\ y = 1\n\
            State 2: <Next>\n\
            /\\ x = 1\n\
            /\\ y = 1\n\
        ";
        let info = parse_trace_info(stdout, Tool::Tlc);
        assert_eq!(info.state_count, 2);
        assert!(info.signature.is_some());
        let sig = info.signature.clone().unwrap();
        // Sanity: re-hashing the same input gives the same signature.
        let info2 = parse_trace_info(stdout, Tool::Tlc);
        assert_eq!(info2.signature.as_deref(), Some(sig.as_str()));
    }

    #[test]
    fn trace_info_detects_stuttering() {
        let stdout = "\
            State 1: ...\n\
            stuttering at depth 3\n\
        ";
        let info = parse_trace_info(stdout, Tool::Ty);
        assert!(info.has_stuttering);
    }
}
