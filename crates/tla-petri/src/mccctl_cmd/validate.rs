// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (regression-fence tests below construct the round-1 spaced literals at
// runtime so an auto-fixer cannot rewrite them into tautologies.)

//! Library backing for `ty-mcc-validate` and the equivalent
//! `ty-mccctl spec-validate` subcommand.
//!
//! Parses `ty-mcc` stdout against an `expected.json` fixture file for
//! one MCC examination. Catches three regression classes:
//!
//! * **Spaced-keyword drift** (qualification-1 root cause). All canonical
//!   keywords route through [`crate::mcc_keywords`].
//! * **Wrong verdict** (correctness regression).
//! * **Non-canonical lines** (protocol regression — the BenchKit parser
//!   rejects unknown lines and tools have been disqualified historically
//!   for emitting extra stdout).
//!
//! Ported from `scripts/mcc_validate.py` to kill the Python cross-check
//! that produced drift between the Python forbidden-keyword list and the
//! Rust canonical constants. Single source of truth now lives in
//! [`crate::mcc_keywords`].

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde_json::Value;

use crate::examination::Examination;
use crate::mcc_keywords::{
    CANNOT_COMPUTE, DO_NOT_COMPETE, FORMULA, MAX_TOKEN_IN_PLACE, MAX_TOKEN_PER_MARKING, STATES,
    STATE_SPACE, TECHNIQUES, TRANSITIONS,
};
use crate::output::Verdict;

/// Command-line arguments for the `ty-mcc-validate` helper.
#[derive(Parser, Debug)]
#[command(
    name = "ty-mcc-validate",
    about = "End-to-end MCC validator: check ty-mcc stdout against expected.json",
    long_about = "Validates that a captured `ty-mcc` stdout transcript matches the \
                  pinned verdicts in `expected.json` for one MCC examination.\n\n\
                  Three regression classes are caught:\n  \
                  - spaced-keyword drift (qualification-1 root cause)\n  \
                  - wrong verdict (correctness regression)\n  \
                  - non-canonical lines (protocol regression)\n\n\
                  Exit 0 = PASS, exit 1 = any failure (explanation on stderr)."
)]
pub struct Cli {
    /// File containing captured stdout from a `ty-mcc` run.
    #[arg(value_name = "STDOUT_FILE")]
    pub stdout_file: PathBuf,

    /// `expected.json` fixture pinning the expected verdicts.
    #[arg(value_name = "EXPECTED_JSON")]
    pub expected_json: PathBuf,

    /// MCC examination name (e.g. `ReachabilityDeadlock`, `StateSpace`).
    ///
    /// One of the 13 MCC examinations recognised by
    /// [`crate::examination::Examination::from_name`].
    #[arg(value_name = "EXAMINATION")]
    pub examination: String,
}

/// Parsed contents of one stdout transcript.
#[derive(Debug, Default)]
struct ParsedOutput {
    /// `FORMULA <id> <verdict-or-int> TECHNIQUES <list>` lines, keyed by id.
    formulas: BTreeMap<String, String>,
    /// `STATE_SPACE <metric> <int> TECHNIQUES <list>` metrics, keyed by metric keyword.
    state_metrics: BTreeMap<String, u64>,
    /// `STATE_SPACE CANNOT_COMPUTE ...` was observed.
    state_cannot_compute: bool,
    /// Tool-level `CANNOT_COMPUTE` or `DO_NOT_COMPETE` (alone on a line).
    #[allow(dead_code)] // kept for future use; mirrors the Python script's `tool_kw`.
    tool_keyword: Option<String>,
    /// Non-canonical lines that did not match any protocol shape.
    unparsed: Vec<String>,
}

/// Entry point used by the standalone `ty-mcc-validate` binary.
pub fn run() -> ExitCode {
    execute(Cli::parse())
}

/// Entry point used by `ty-mccctl spec-validate`.
pub fn run_from<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(err) => {
            let _ = err.print();
            return ExitCode::from(u8::from(err.use_stderr()));
        }
    };
    execute(cli)
}

fn execute(cli: Cli) -> ExitCode {
    match dispatch(&cli) {
        Ok(()) => {
            println!("PASS: {} matches expected.json", cli.examination);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::from(1)
        }
    }
}

fn dispatch(cli: &Cli) -> Result<()> {
    // Validate examination name against the 13 canonical MCC examinations.
    // We don't strictly need the typed Examination once parsed, but checking
    // here gives a clear error before we touch any files.
    let _exam = Examination::from_name(&cli.examination)
        .with_context(|| format!("unknown MCC examination: {}", cli.examination))?;

    let stdout_text = fs::read_to_string(&cli.stdout_file)
        .with_context(|| format!("reading stdout file {}", cli.stdout_file.display()))?;
    let expected_text = fs::read_to_string(&cli.expected_json)
        .with_context(|| format!("reading expected json {}", cli.expected_json.display()))?;
    let expected: Value = serde_json::from_str(&expected_text)
        .with_context(|| format!("parsing expected json {}", cli.expected_json.display()))?;

    validate(&stdout_text, &expected, &cli.examination)
}

fn validate(stdout_text: &str, expected: &Value, examination: &str) -> Result<()> {
    // 1. Forbidden spaced keywords. Built at runtime so an auto-fixer
    //    cannot rewrite these literals and turn the negative assertion
    //    into a tautology. Same pattern as `output_tests::forbidden_*`.
    let sp = " ";
    let forbidden = [
        format!("CANNOT{sp}COMPUTE"),
        format!("DO{sp}NOT{sp}COMPETE"),
        format!("STATE{sp}SPACE"),
        format!("MAX{sp}TOKEN{sp}IN{sp}PLACE"),
        format!("MAX{sp}TOKEN{sp}PER{sp}MARKING"),
    ];
    let hits: Vec<&String> = forbidden
        .iter()
        .filter(|t| stdout_text.contains(*t))
        .collect();
    if !hits.is_empty() {
        bail!("FAIL: stdout contains spaced keyword(s): {hits:?}");
    }

    // 2. Parse the canonical lines. Any non-canonical line is a failure.
    let parsed = parse_stdout(stdout_text);
    if !parsed.unparsed.is_empty() {
        let mut msg =
            String::from("FAIL: stdout contains non-canonical line(s) — MCC parser will reject:\n");
        for line in parsed.unparsed.iter().take(10) {
            msg.push_str(&format!("  {line:?}\n"));
        }
        if parsed.unparsed.len() > 10 {
            msg.push_str(&format!("  ... and {} more\n", parsed.unparsed.len() - 10));
        }
        bail!(msg.trim_end().to_string());
    }
    reject_mixed_tool_keyword(&parsed)?;

    // 3. Compare against expected.
    let expected_for_exam = expected
        .get(examination)
        .ok_or_else(|| anyhow!("FAIL: expected.json has no entry for examination={examination}"))?;

    let mut failures: Vec<String> = Vec::new();

    if examination == "StateSpace" {
        if parsed.state_cannot_compute {
            failures.push("FAIL: tool reported CANNOT_COMPUTE for StateSpace".into());
        } else {
            let obj = expected_for_exam
                .as_object()
                .ok_or_else(|| anyhow!("FAIL: expected.json StateSpace entry is not an object"))?;
            for (want_key, want_val) in obj {
                let metric_key = match want_key.as_str() {
                    "states" => STATES,
                    "max_token_in_place" => MAX_TOKEN_IN_PLACE,
                    "max_token_sum" => MAX_TOKEN_PER_MARKING,
                    "edges" => TRANSITIONS,
                    other => {
                        // Python script used .upper() as the fallback for unknown keys.
                        // Preserve that here; we own both sides so this branch is rare.
                        // Allocate once into a leaked box-style: but we can compare
                        // directly with an owned String.
                        let upper = other.to_ascii_uppercase();
                        match parsed.state_metrics.get(&upper) {
                            None => failures.push(format!(
                                "FAIL: StateSpace {upper}: expected {want_val}, got None"
                            )),
                            Some(got) if Value::from(*got) != *want_val => failures.push(format!(
                                "FAIL: StateSpace {upper}: expected {want_val}, got {got}"
                            )),
                            Some(_) => {}
                        }
                        continue;
                    }
                };
                let want = want_val.as_u64().ok_or_else(|| {
                    anyhow!(
                        "FAIL: expected.json StateSpace.{want_key} is not an unsigned integer: {want_val}"
                    )
                })?;
                match parsed.state_metrics.get(metric_key) {
                    None => failures.push(format!(
                        "FAIL: StateSpace {metric_key}: expected {want}, got None"
                    )),
                    Some(got) if *got != want => failures.push(format!(
                        "FAIL: StateSpace {metric_key}: expected {want}, got {got}"
                    )),
                    Some(_) => {}
                }
            }
        }
    } else if let Some(want) = expected_for_exam.as_str() {
        // Single-formula examination — the verdict is keyed by examination name.
        match parsed.formulas.get(examination) {
            Some(got) if got == want => {}
            other => {
                failures.push(format!(
                    "FAIL: {examination}: expected {want:?}, got {other:?}"
                ));
            }
        }
    } else if let Some(map) = expected_for_exam.as_object() {
        for (formula_id, want_verdict) in map {
            let want = want_verdict.as_str().ok_or_else(|| {
                anyhow!("FAIL: expected.json {examination}.{formula_id} verdict is not a string: {want_verdict}")
            })?;
            match parsed.formulas.get(formula_id) {
                Some(got) if got == want => {}
                other => failures.push(format!(
                    "FAIL: {formula_id}: expected {want:?}, got {other:?}"
                )),
            }
        }
    } else {
        bail!(
            "FAIL: expected.json entry for {examination} has unsupported shape: {expected_for_exam}"
        );
    }

    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }
    Ok(())
}

fn reject_mixed_tool_keyword(parsed: &ParsedOutput) -> Result<()> {
    let Some(keyword) = parsed.tool_keyword.as_deref() else {
        return Ok(());
    };

    if parsed.formulas.is_empty() && parsed.state_metrics.is_empty() && !parsed.state_cannot_compute
    {
        return Ok(());
    }

    bail!("FAIL: tool-level {keyword} cannot be mixed with FORMULA or STATE_SPACE output");
}

/// Parse one stdout transcript into canonical MCC line categories.
///
/// Any line not matching one of the four canonical shapes is captured in
/// `unparsed` for strict failure reporting. Blank lines are ignored.
fn parse_stdout(text: &str) -> ParsedOutput {
    let mut out = ParsedOutput::default();
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some((id, verdict)) = parse_formula_line(line) {
            out.formulas.insert(id, verdict);
            continue;
        }
        if let Some(body) = parse_state_space_line(line) {
            match body {
                StateSpaceBody::CannotCompute => out.state_cannot_compute = true,
                StateSpaceBody::Metric(k, v) => {
                    out.state_metrics.insert(k, v);
                }
            }
            continue;
        }
        if let Some(kw) = parse_tool_keyword_line(line) {
            out.tool_keyword = Some(kw);
            continue;
        }
        out.unparsed.push(line.to_string());
    }
    out
}

/// `FORMULA <id> (TRUE|FALSE|CANNOT_COMPUTE|<int>) TECHNIQUES <list>`.
fn parse_formula_line(line: &str) -> Option<(String, String)> {
    let rest = line.strip_prefix(FORMULA)?.strip_prefix(' ')?;
    // We need three or more tokens before TECHNIQUES: id verdict TECHNIQUES <tags>.
    let tech_idx = rest.find(&format!(" {TECHNIQUES} "))?;
    let (head, tail) = rest.split_at(tech_idx);
    // The tail starts with " TECHNIQUES "; require at least one technique token.
    let after_techniques = tail
        .strip_prefix(&format!(" {TECHNIQUES} "))
        .unwrap_or_default()
        .trim();
    if after_techniques.is_empty() {
        return None;
    }
    let (id, verdict) = head.split_once(' ')?;

    if id.is_empty() || verdict.contains(' ') {
        return None;
    }
    if !is_valid_formula_verdict(verdict) {
        return None;
    }
    Some((id.to_string(), verdict.to_string()))
}

fn is_valid_formula_verdict(verdict: &str) -> bool {
    if verdict == Verdict::True.to_string()
        || verdict == Verdict::False.to_string()
        || verdict == CANNOT_COMPUTE
    {
        return true;
    }
    verdict.chars().all(|c| c.is_ascii_digit()) && !verdict.is_empty()
}

enum StateSpaceBody {
    CannotCompute,
    Metric(String, u64),
}

/// `STATE_SPACE (CANNOT_COMPUTE | STATES n | TRANSITIONS n | MAX_TOKEN_IN_PLACE n | MAX_TOKEN_PER_MARKING n) TECHNIQUES <list>`.
fn parse_state_space_line(line: &str) -> Option<StateSpaceBody> {
    let rest = line.strip_prefix(STATE_SPACE)?.strip_prefix(' ')?;
    let tech_idx = rest.find(&format!(" {TECHNIQUES} "))?;
    let (body, tail) = rest.split_at(tech_idx);
    let after_techniques = tail
        .strip_prefix(&format!(" {TECHNIQUES} "))
        .unwrap_or_default()
        .trim();
    if after_techniques.is_empty() {
        return None;
    }
    if body == CANNOT_COMPUTE {
        return Some(StateSpaceBody::CannotCompute);
    }
    let (kw, num) = body.split_once(' ')?;
    let metric = match kw {
        s if s == STATES => STATES,
        s if s == TRANSITIONS => TRANSITIONS,
        s if s == MAX_TOKEN_IN_PLACE => MAX_TOKEN_IN_PLACE,
        s if s == MAX_TOKEN_PER_MARKING => MAX_TOKEN_PER_MARKING,
        _ => return None,
    };
    let n: u64 = num.parse().ok()?;
    Some(StateSpaceBody::Metric(metric.to_string(), n))
}

/// `^(CANNOT_COMPUTE|DO_NOT_COMPETE)$` (after rstrip).
fn parse_tool_keyword_line(line: &str) -> Option<String> {
    if line == CANNOT_COMPUTE || line == DO_NOT_COMPETE {
        Some(line.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build the round-1 spaced literals at runtime so an auto-fixer
    /// cannot silently rewrite them into the canonical underscored form
    /// and turn the negative assertions below into tautologies. Same
    /// pattern as `crates/tla-petri/src/output_tests.rs`.
    fn forbidden_cannot_compute_with_space() -> String {
        format!("CANNOT{}COMPUTE", " ")
    }
    fn forbidden_state_space_with_space() -> String {
        format!("STATE{}SPACE", " ")
    }
    fn forbidden_do_not_compete_with_space() -> String {
        format!("DO{}NOT{}COMPETE", " ", " ")
    }

    #[test]
    fn canonical_single_formula_passes() {
        let stdout = "FORMULA ReachabilityDeadlock FALSE TECHNIQUES EXPLICIT\n";
        let expected = json!({ "ReachabilityDeadlock": "FALSE" });
        validate(stdout, &expected, "ReachabilityDeadlock").expect("should pass");
    }

    #[test]
    fn canonical_multi_formula_passes() {
        let stdout = "FORMULA F-00 TRUE TECHNIQUES EXPLICIT\n\
                      FORMULA F-01 FALSE TECHNIQUES EXPLICIT\n";
        let expected = json!({
            "ReachabilityFireability": {"F-00": "TRUE", "F-01": "FALSE"}
        });
        validate(stdout, &expected, "ReachabilityFireability").expect("should pass");
    }

    #[test]
    fn canonical_state_space_passes() {
        let stdout = "STATE_SPACE STATES 3 TECHNIQUES EXPLICIT\n\
                      STATE_SPACE TRANSITIONS 4 TECHNIQUES EXPLICIT\n\
                      STATE_SPACE MAX_TOKEN_IN_PLACE 1 TECHNIQUES EXPLICIT\n\
                      STATE_SPACE MAX_TOKEN_PER_MARKING 3 TECHNIQUES EXPLICIT\n";
        let expected = json!({
            "StateSpace": {"states": 3, "max_token_in_place": 1, "max_token_sum": 3}
        });
        validate(stdout, &expected, "StateSpace").expect("should pass");
    }

    #[test]
    fn state_space_cannot_compute_fails() {
        let stdout = "STATE_SPACE CANNOT_COMPUTE TECHNIQUES EXPLICIT\n";
        let expected = json!({
            "StateSpace": {"states": 3, "max_token_in_place": 1, "max_token_sum": 3}
        });
        let err = validate(stdout, &expected, "StateSpace")
            .expect_err("StateSpace CANNOT_COMPUTE must fail when metrics are expected");
        assert!(
            err.to_string()
                .contains("tool reported CANNOT_COMPUTE for StateSpace"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tool_level_cannot_compute_alone_parses() {
        // A tool-level CANNOT_COMPUTE alone on a line is a valid protocol
        // shape; it is not a formula line. Whether it satisfies expected
        // depends on the examination — here we test parser shape only.
        let stdout = "CANNOT_COMPUTE\n";
        let parsed = parse_stdout(stdout);
        assert_eq!(parsed.tool_keyword.as_deref(), Some("CANNOT_COMPUTE"));
        assert!(parsed.unparsed.is_empty());
    }

    #[test]
    fn tool_level_do_not_compete_alone_parses() {
        let stdout = "DO_NOT_COMPETE\n";
        let parsed = parse_stdout(stdout);
        assert_eq!(parsed.tool_keyword.as_deref(), Some("DO_NOT_COMPETE"));
        assert!(parsed.unparsed.is_empty());
    }

    #[test]
    fn tool_level_keyword_mixed_with_formula_fails() {
        let stdout = "FORMULA ReachabilityDeadlock FALSE TECHNIQUES EXPLICIT\n\
                      CANNOT_COMPUTE\n";
        let expected = json!({ "ReachabilityDeadlock": "FALSE" });
        let err = validate(stdout, &expected, "ReachabilityDeadlock")
            .expect_err("tool-level keyword must not be mixed with formula output");
        assert!(
            err.to_string()
                .contains("tool-level CANNOT_COMPUTE cannot be mixed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tool_level_keyword_mixed_with_state_space_fails() {
        let stdout = "STATE_SPACE STATES 3 TECHNIQUES EXPLICIT\n\
                      CANNOT_COMPUTE\n";
        let expected = json!({
            "StateSpace": {"states": 3, "max_token_in_place": 1, "max_token_sum": 3}
        });
        let err = validate(stdout, &expected, "StateSpace")
            .expect_err("tool-level keyword must not be mixed with StateSpace output");
        assert!(
            err.to_string()
                .contains("tool-level CANNOT_COMPUTE cannot be mixed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn non_canonical_line_fails() {
        let stdout = "FORMULA ReachabilityDeadlock FALSE TECHNIQUES EXPLICIT\n\
                      DEBUG: explored 17 states\n";
        let expected = json!({ "ReachabilityDeadlock": "FALSE" });
        let err = validate(stdout, &expected, "ReachabilityDeadlock")
            .expect_err("debug line should be rejected");
        assert!(
            err.to_string().contains("non-canonical line"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn wrong_verdict_fails() {
        let stdout = "FORMULA ReachabilityDeadlock TRUE TECHNIQUES EXPLICIT\n";
        let expected = json!({ "ReachabilityDeadlock": "FALSE" });
        let err = validate(stdout, &expected, "ReachabilityDeadlock")
            .expect_err("wrong verdict should fail");
        let msg = err.to_string();
        assert!(msg.contains("expected \"FALSE\""), "got: {msg}");
        assert!(msg.contains("got Some(\"TRUE\")"), "got: {msg}");
    }

    #[test]
    fn wrong_state_metric_fails() {
        let stdout = "STATE_SPACE STATES 99 TECHNIQUES EXPLICIT\n\
                      STATE_SPACE TRANSITIONS 4 TECHNIQUES EXPLICIT\n\
                      STATE_SPACE MAX_TOKEN_IN_PLACE 1 TECHNIQUES EXPLICIT\n\
                      STATE_SPACE MAX_TOKEN_PER_MARKING 3 TECHNIQUES EXPLICIT\n";
        let expected = json!({
            "StateSpace": {"states": 3, "max_token_in_place": 1, "max_token_sum": 3}
        });
        let err =
            validate(stdout, &expected, "StateSpace").expect_err("wrong STATES metric should fail");
        assert!(
            err.to_string().contains("STATES: expected 3, got 99"),
            "got: {err}"
        );
    }

    #[test]
    fn spaced_cannot_compute_fails() {
        // Build the bad input with format! so a textual fixer cannot
        // silently rewrite it into the canonical form.
        let stdout = format!("{}\n", forbidden_cannot_compute_with_space());
        let expected = json!({ "ReachabilityDeadlock": "FALSE" });
        let err = validate(&stdout, &expected, "ReachabilityDeadlock")
            .expect_err("spaced CANNOT_COMPUTE must be rejected");
        assert!(
            err.to_string().contains("spaced keyword"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn spaced_state_space_fails() {
        let stdout = format!(
            "{} STATES 3 TECHNIQUES EXPLICIT\n",
            forbidden_state_space_with_space()
        );
        let expected = json!({
            "StateSpace": {"states": 3, "max_token_in_place": 1, "max_token_sum": 3}
        });
        let err = validate(&stdout, &expected, "StateSpace")
            .expect_err("spaced STATE_SPACE must be rejected");
        assert!(
            err.to_string().contains("spaced keyword"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn spaced_do_not_compete_fails() {
        let stdout = format!("{}\n", forbidden_do_not_compete_with_space());
        let expected = json!({ "ReachabilityDeadlock": "FALSE" });
        let err = validate(&stdout, &expected, "ReachabilityDeadlock")
            .expect_err("spaced DO_NOT_COMPETE must be rejected");
        assert!(
            err.to_string().contains("spaced keyword"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unknown_examination_in_expected_fails() {
        let stdout = "FORMULA ReachabilityDeadlock FALSE TECHNIQUES EXPLICIT\n";
        let expected = json!({ "OtherExam": "TRUE" });
        let err = validate(stdout, &expected, "ReachabilityDeadlock")
            .expect_err("missing key should fail");
        assert!(
            err.to_string()
                .contains("expected.json has no entry for examination=ReachabilityDeadlock"),
            "got: {err}"
        );
    }

    #[test]
    fn formula_line_with_integer_bound_parses() {
        // UpperBounds emits integer verdicts. The parser must accept them
        // even though the validator's StateSpace path uses STATE_SPACE.
        let stdout = "FORMULA UB-00 17 TECHNIQUES EXPLICIT\n";
        let parsed = parse_stdout(stdout);
        assert_eq!(parsed.formulas.get("UB-00").map(String::as_str), Some("17"));
        assert!(parsed.unparsed.is_empty());
    }

    #[test]
    fn empty_lines_are_ignored() {
        let stdout = "\n\nFORMULA ReachabilityDeadlock FALSE TECHNIQUES EXPLICIT\n   \n\n";
        let expected = json!({ "ReachabilityDeadlock": "FALSE" });
        validate(stdout, &expected, "ReachabilityDeadlock").expect("blank lines OK");
    }

    #[test]
    fn missing_techniques_suffix_is_non_canonical() {
        let stdout = "FORMULA ReachabilityDeadlock FALSE\n";
        let expected = json!({ "ReachabilityDeadlock": "FALSE" });
        let err = validate(stdout, &expected, "ReachabilityDeadlock")
            .expect_err("missing TECHNIQUES suffix is non-canonical");
        assert!(
            err.to_string().contains("non-canonical line"),
            "unexpected error: {err}"
        );
    }
}
