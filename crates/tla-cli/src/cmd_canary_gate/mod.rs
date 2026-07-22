// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Rust entrypoint for eval/check, enumerate, API, and silent-error canary gates.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use regex::Regex;

use crate::cli_schema::{CanaryGateArgs, CanaryGateKind, CanaryGateMode};

const DEFAULT_BASELINE: &str = "tests/tlc_comparison/spec_baseline.json";

const EVAL_LIST: &str = "tests/tlc_comparison/eval_canary_specs.txt";
const EVAL_DEFAULT_TIMEOUT: u64 = 30;
const EVAL_SKIP_ENV: &str = "TY_EVAL_CANARY_SKIP";
const EVAL_WARN_ENV: &str = "TY_EVAL_CANARY_WARN";
const EVAL_TIMEOUT_ENV: &str = "TY_EVAL_CANARY_TIMEOUT";
const EVAL_TRIGGERS: &[&str] = &["crates/tla-eval/src/", "crates/tla-check/src/"];

const ENUMERATE_LIST: &str = "tests/tlc_comparison/enumerate_canary_specs.txt";
#[cfg(test)]
const ENUMERATE_SLOW_LIST: &str = "tests/tlc_comparison/enumerate_slow_canary_specs.txt";
const ENUMERATE_SKIP_ENV: &str = "TY_ENUMERATE_CANARY_SKIP";
const ENUMERATE_WARN_ENV: &str = "TY_ENUMERATE_CANARY_WARN";
const ENUMERATE_TIMEOUT_ENV: &str = "TY_ENUMERATE_CANARY_TIMEOUT";
const ENUMERATE_DEFAULT_TIMEOUT: u64 = 30;
const ENUMERATE_TRIGGERS: &[&str] = &[
    "crates/tla-check/src/enumerate/",
    "crates/tla-check/src/eval/",
    "crates/tla-check/src/compiled_guard/",
    "crates/tla-check/src/check/model_checker/",
    "crates/tla-check/src/state.rs",
    "crates/tla-check/src/state/",
    "crates/tla-ir/src/lower/",
    "crates/tla-trust-cg/src/lower.rs",
    "crates/tla-trust-cg/src/trust_ir_lower.rs",
];

const API_CANARY_DIR: &str = "tests/api_canaries";
const API_CANARIES: &[&str] = &[
    "core_translate_canary",
    "check_fingerprint_canary",
    "eval_value_canary",
];

const SILENT_ERROR_SCAN_PATHS: &[&str] = &[
    "crates/tla-check/src/check",
    "crates/tla-check/src/enumerate",
    "crates/tla-check/src/liveness",
    "crates/tla-check/src/parallel",
    "crates/tla-check/src/compiled_guard",
    "crates/tla-check/src/adaptive.rs",
    "crates/tla-check/src/error_policy.rs",
];
const SILENT_ERROR_EXEMPTION_MARKERS: &[&str] = &[
    "Part of #1433",
    "error_policy",
    "eval_required",
    "eval_speculative",
    "eval_bool_required",
    "eval_bool_speculative",
];

#[derive(Clone, Copy)]
struct CanarySet {
    label: &'static str,
    spec_list: &'static str,
    triggers: &'static [&'static str],
    skip_env: &'static str,
    warn_env: &'static str,
    timeout_env: &'static str,
    default_timeout: Option<u64>,
    fail_message: &'static str,
}

struct SilentErrorRule {
    pattern: Regex,
    description: &'static str,
}

#[derive(Debug, PartialEq, Eq)]
struct SilentErrorFinding {
    path: PathBuf,
    line: usize,
    source: String,
    description: &'static str,
}

const EVAL_SET: CanarySet = CanarySet {
    label: "eval-canary",
    spec_list: EVAL_LIST,
    triggers: EVAL_TRIGGERS,
    skip_env: EVAL_SKIP_ENV,
    warn_env: EVAL_WARN_ENV,
    timeout_env: EVAL_TIMEOUT_ENV,
    default_timeout: Some(EVAL_DEFAULT_TIMEOUT),
    fail_message: "Regression detected in eval/check canary specs.",
};

const ENUMERATE_SET: CanarySet = CanarySet {
    label: "canary-gate",
    spec_list: ENUMERATE_LIST,
    triggers: ENUMERATE_TRIGGERS,
    skip_env: ENUMERATE_SKIP_ENV,
    warn_env: ENUMERATE_WARN_ENV,
    timeout_env: ENUMERATE_TIMEOUT_ENV,
    default_timeout: Some(ENUMERATE_DEFAULT_TIMEOUT),
    fail_message: "State-count drift detected in canary specs.",
};

pub(crate) fn cmd_canary_gate(args: CanaryGateArgs) -> Result<()> {
    let plan = selected_plan(args.kind);
    let mut failures = 0usize;

    if !plan.diagnose_sets.is_empty() {
        let change_set = collect_changed_files(&args).context("read changed files")?;
        if change_set.whitespace_only_rs {
            if plan.standalone_gates.is_empty() {
                eprintln!(
                    "[canary-gate] All .rs changes are whitespace-only (fmt), skipping canaries."
                );
            } else {
                eprintln!(
                    "[canary-gate] All .rs changes are whitespace-only (fmt), skipping eval/check and enumerate canaries."
                );
            }
        } else {
            let mut ran = 0usize;
            let mut relevant = 0usize;
            for set in plan.diagnose_sets.iter().copied() {
                let set_triggered = triggered(set, &change_set.files);
                if set_triggered {
                    relevant += 1;
                }
                match run_set(set, args.mode, &change_set.files)? {
                    SetOutcome::Skipped | SetOutcome::Passed => {}
                    SetOutcome::Failed => failures += 1,
                }
                if set_triggered && !skip_enabled(set.skip_env) {
                    ran += 1;
                }
            }

            if ran == 0 && relevant == 0 {
                if plan.standalone_gates.is_empty() {
                    eprintln!("[canary-gate] No relevant changed files; skipping canaries.");
                } else {
                    eprintln!(
                        "[canary-gate] No relevant changed files for eval/check or enumerate canaries; running API/silent-error gates."
                    );
                }
            }
        }
    }

    for gate in plan.standalone_gates.iter().copied() {
        match run_standalone_gate(gate, args.mode, args.verbose)? {
            SetOutcome::Skipped | SetOutcome::Passed => {}
            SetOutcome::Failed => failures += 1,
        }
    }

    if failures > 0 {
        bail!("{failures} canary gate(s) failed");
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct ChangeSet {
    files: Vec<String>,
    whitespace_only_rs: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GitChangeScope {
    Head,
    Staged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SetOutcome {
    Skipped,
    Passed,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StandaloneGate {
    Api,
    SilentError,
}

#[derive(Clone, Copy)]
struct CanaryGatePlan {
    diagnose_sets: &'static [CanarySet],
    standalone_gates: &'static [StandaloneGate],
}

fn run_set(set: CanarySet, mode: CanaryGateMode, changed_files: &[String]) -> Result<SetOutcome> {
    if !triggered(set, changed_files) {
        return Ok(SetOutcome::Skipped);
    }
    if skip_enabled(set.skip_env) {
        eprintln!(
            "[{}] {}=1 set; skipping canary specs.",
            set.label, set.skip_env
        );
        return Ok(SetOutcome::Skipped);
    }

    let spec_list = Path::new(set.spec_list);
    if !spec_list.is_file() {
        bail!("canary spec list not found: {}", spec_list.display());
    }

    eprintln!(
        "[{}] Relevant source files changed; running canary specs...",
        set.label
    );

    let timeout = timeout_seconds(set.timeout_env, set.default_timeout)?;
    let exe = std::env::current_exe().context("resolve current exe path")?;
    let mut command = Command::new(&exe);
    command
        .arg("diagnose")
        .arg("--baseline")
        .arg(DEFAULT_BASELINE)
        .arg("--spec-list")
        .arg(set.spec_list)
        .arg("--fail-on-mismatch")
        .arg("--fail-on-non-pass")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    if let Some(timeout) = timeout {
        command.arg("--timeout").arg(timeout.to_string());
    }

    let status = command
        .status()
        .with_context(|| format!("spawn {} diagnose", exe.display()))?;
    if status.success() {
        eprintln!("[{}] All canary specs pass.", set.label);
        return Ok(SetOutcome::Passed);
    }

    match effective_mode(set, mode) {
        CanaryGateMode::Enforce => {
            eprintln!();
            eprintln!("[{}] FAIL: {}", set.label, set.fail_message);
            Ok(SetOutcome::Failed)
        }
        CanaryGateMode::Warn => {
            eprintln!();
            eprintln!("[{}] WARNING: {}", set.label, set.fail_message);
            eprintln!(
                "[{}] This is advisory - investigate before marking work complete.",
                set.label
            );
            Ok(SetOutcome::Passed)
        }
    }
}

fn selected_plan(kind: CanaryGateKind) -> CanaryGatePlan {
    let (diagnose_sets, standalone_gates) = match kind {
        CanaryGateKind::Eval => (&[EVAL_SET][..], &[][..]),
        CanaryGateKind::Enumerate => (&[ENUMERATE_SET][..], &[][..]),
        CanaryGateKind::Api => (&[][..], &[StandaloneGate::Api][..]),
        CanaryGateKind::SilentError => (&[][..], &[StandaloneGate::SilentError][..]),
        CanaryGateKind::All => (
            &[EVAL_SET, ENUMERATE_SET][..],
            &[StandaloneGate::Api, StandaloneGate::SilentError][..],
        ),
    };
    CanaryGatePlan {
        diagnose_sets,
        standalone_gates,
    }
}

fn run_standalone_gate(
    gate: StandaloneGate,
    mode: CanaryGateMode,
    verbose: bool,
) -> Result<SetOutcome> {
    match gate {
        StandaloneGate::Api => run_api_gate(mode, verbose),
        StandaloneGate::SilentError => run_silent_error_gate(mode),
    }
}

fn run_api_gate(mode: CanaryGateMode, verbose: bool) -> Result<SetOutcome> {
    eprintln!("[api-canary] Running API consumer compatibility canaries...");
    let mut passed = 0usize;
    let mut failures = Vec::new();

    for canary in API_CANARIES {
        let canary_dir = Path::new(API_CANARY_DIR).join(canary);
        if !canary_dir.is_dir() {
            eprintln!(
                "[api-canary] {canary}: FAIL (missing {})",
                canary_dir.display()
            );
            failures.push(format!(
                "{canary}: directory not found: {}",
                canary_dir.display()
            ));
            continue;
        }

        let output = Command::new("cargo")
            .arg("check")
            .arg("-p")
            .arg(canary)
            .stdin(Stdio::null())
            .output()
            .with_context(|| format!("run cargo check -p {canary}"))?;
        if output.status.success() {
            eprintln!("[api-canary] {canary}: PASS");
            passed += 1;
        } else {
            let first_error = first_error_line(&output);
            eprintln!(
                "[api-canary] {canary}: FAIL ({})",
                output
                    .status
                    .code()
                    .map(|code| format!("exit {code}"))
                    .unwrap_or_else(|| "terminated by signal".to_string())
            );
            failures.push(format!("{canary}: {first_error}"));
            if verbose {
                print_canary_output(canary, &output);
            }
        }
    }

    eprintln!(
        "[api-canary] Results: {passed} passed, {} failed",
        failures.len()
    );
    if failures.is_empty() {
        eprintln!("[api-canary] All API canaries compile.");
        return Ok(SetOutcome::Passed);
    }

    for failure in &failures {
        eprintln!("[api-canary]   - {failure}");
    }
    match mode {
        CanaryGateMode::Enforce => {
            eprintln!("[api-canary] FAIL: API consumer compatibility canaries failed.");
            Ok(SetOutcome::Failed)
        }
        CanaryGateMode::Warn => {
            eprintln!("[api-canary] WARNING: API consumer compatibility canaries failed.");
            eprintln!("[api-canary] This is advisory - investigate before marking work complete.");
            Ok(SetOutcome::Passed)
        }
    }
}

fn run_silent_error_gate(mode: CanaryGateMode) -> Result<SetOutcome> {
    eprintln!("[silent-error] Scanning model checker production code...");
    let findings = scan_silent_error_coercion(Path::new("."))?;
    if findings.is_empty() {
        eprintln!("[silent-error] PASS: no silent eval-error coercion patterns found.");
        eprintln!(
            "[silent-error] Scanned: {}",
            SILENT_ERROR_SCAN_PATHS.join(", ")
        );
        return Ok(SetOutcome::Passed);
    }

    let label = match mode {
        CanaryGateMode::Enforce => "FAIL",
        CanaryGateMode::Warn => "WARNING",
    };
    eprintln!(
        "[silent-error] {label}: {} silent eval-error coercion violation(s) found.",
        findings.len()
    );
    for finding in &findings {
        eprintln!(
            "[silent-error]   {}:{}: {}",
            finding.path.display(),
            finding.line,
            finding.description
        );
        eprintln!("[silent-error]     {}", finding.source.trim());
    }
    eprintln!(
        "[silent-error] Fix: use eval_required(), eval_speculative(), eval_bool_required(), or eval_bool_speculative()."
    );
    eprintln!(
        "[silent-error] Add 'Part of #1433:' only when the pattern has already been audited."
    );

    match mode {
        CanaryGateMode::Enforce => Ok(SetOutcome::Failed),
        CanaryGateMode::Warn => Ok(SetOutcome::Passed),
    }
}

fn scan_silent_error_coercion(root: &Path) -> Result<Vec<SilentErrorFinding>> {
    let mut files = Vec::new();
    for scan_path in SILENT_ERROR_SCAN_PATHS {
        let path = root.join(scan_path);
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            collect_rust_files(&path, &mut files)?;
        }
    }
    files.sort();

    let mut findings = Vec::new();
    for file in files {
        if is_silent_error_test_file(root, &file) {
            continue;
        }
        findings.extend(scan_silent_error_file(root, &file)?);
    }
    Ok(findings)
}

fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = fs::read_dir(dir)
        .with_context(|| format!("read directory {}", dir.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("read directory entries {}", dir.display()))?;
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, files)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(path);
        }
    }
    Ok(())
}

fn scan_silent_error_file(root: &Path, path: &Path) -> Result<Vec<SilentErrorFinding>> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let lines: Vec<&str> = content.lines().collect();
    let mut findings = Vec::new();
    let mut in_block_comment = false;
    let mut awaiting_cfg_test_item = false;
    let mut in_cfg_test_item = false;
    let mut cfg_test_depth = 0i32;

    for (idx, line) in lines.iter().enumerate() {
        let code = strip_comments_and_literals(line, &mut in_block_comment);
        let trimmed = code.trim();

        if in_cfg_test_item {
            cfg_test_depth += brace_delta(&code);
            if cfg_test_depth <= 0 && code.contains('}') {
                in_cfg_test_item = false;
            }
            continue;
        }
        if awaiting_cfg_test_item {
            let delta = brace_delta(&code);
            if delta > 0 {
                awaiting_cfg_test_item = false;
                in_cfg_test_item = true;
                cfg_test_depth = delta;
                if cfg_test_depth <= 0 {
                    in_cfg_test_item = false;
                }
            } else if trimmed.ends_with(';') {
                awaiting_cfg_test_item = false;
            }
            continue;
        }
        if trimmed.contains("#[cfg(test)]") {
            awaiting_cfg_test_item = true;
            continue;
        }

        if silent_error_line_has_exemption(line) {
            continue;
        }
        let prev_line = idx
            .checked_sub(1)
            .and_then(|prev| lines.get(prev))
            .copied()
            .unwrap_or("");
        let next_line = lines.get(idx + 1).copied().unwrap_or("");
        if silent_error_line_has_exemption(prev_line) || silent_error_line_has_exemption(next_line)
        {
            continue;
        }

        for rule in silent_error_rules() {
            if rule.pattern.is_match(&code) {
                findings.push(SilentErrorFinding {
                    path: path.strip_prefix(root).unwrap_or(path).to_path_buf(),
                    line: idx + 1,
                    source: (*line).to_string(),
                    description: rule.description,
                });
                break;
            }
        }
    }
    Ok(findings)
}

fn silent_error_rules() -> &'static [SilentErrorRule] {
    static RULES: OnceLock<Vec<SilentErrorRule>> = OnceLock::new();
    RULES.get_or_init(|| {
        vec![
            SilentErrorRule {
                pattern: Regex::new(
                    r"if\s+let\s+Ok\(.+\)\s*=\s*(?:eval_entry|eval\s*\(|crate::eval::eval)",
                )
                .expect("valid silent error regex"),
                description: "if let Ok(...) = eval - use eval_required() or eval_speculative()",
            },
            SilentErrorRule {
                pattern: Regex::new(r"eval_entry\s*\([^)]*\)\s*\.ok\(\)")
                    .expect("valid silent error regex"),
                description:
                    "eval_entry().ok() - use eval_speculative() with explicit FallbackClass",
            },
            SilentErrorRule {
                pattern: Regex::new(r"eval_entry\s*\([^)]*\)\s*\.unwrap_or\b")
                    .expect("valid silent error regex"),
                description:
                    "eval_entry().unwrap_or() - use eval_speculative() with explicit FallbackClass",
            },
            SilentErrorRule {
                pattern: Regex::new(r"eval_entry\s*\([^)]*\)\s*\.unwrap_or_default\b")
                    .expect("valid silent error regex"),
                description: "eval_entry().unwrap_or_default() - use eval_speculative()",
            },
        ]
    })
}

fn silent_error_line_has_exemption(line: &str) -> bool {
    SILENT_ERROR_EXEMPTION_MARKERS
        .iter()
        .any(|marker| line.contains(marker))
}

fn is_silent_error_test_file(root: &Path, path: &Path) -> bool {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let normalized = normalize_path(rel);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    name.ends_with("_tests.rs")
        || name == "tests.rs"
        || name.starts_with("test_")
        || normalized.starts_with("tests/")
        || normalized.contains("/tests/")
}

fn strip_comments_and_literals(line: &str, in_block_comment: &mut bool) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut idx = 0usize;
    while idx < chars.len() {
        if *in_block_comment {
            if idx + 1 < chars.len() && chars[idx] == '*' && chars[idx + 1] == '/' {
                out.push(' ');
                out.push(' ');
                idx += 2;
                *in_block_comment = false;
            } else {
                out.push(' ');
                idx += 1;
            }
            continue;
        }

        if idx + 1 < chars.len() && chars[idx] == '/' && chars[idx + 1] == '/' {
            break;
        }
        if idx + 1 < chars.len() && chars[idx] == '/' && chars[idx + 1] == '*' {
            out.push(' ');
            out.push(' ');
            idx += 2;
            *in_block_comment = true;
            continue;
        }
        if let Some(hashes) = raw_string_hashes_at(&chars, idx) {
            let start_len = hashes + 2;
            for _ in 0..start_len {
                out.push(' ');
            }
            idx += start_len;
            while idx < chars.len() {
                if raw_string_ends_at(&chars, idx, hashes) {
                    for _ in 0..=hashes {
                        out.push(' ');
                    }
                    idx += hashes + 1;
                    break;
                }
                out.push(' ');
                idx += 1;
            }
            continue;
        }
        if chars[idx] == '"' {
            out.push(' ');
            idx += 1;
            let mut escaped = false;
            while idx < chars.len() {
                let ch = chars[idx];
                out.push(' ');
                idx += 1;
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    break;
                }
            }
            continue;
        }

        out.push(chars[idx]);
        idx += 1;
    }
    out
}

fn raw_string_hashes_at(chars: &[char], idx: usize) -> Option<usize> {
    if chars.get(idx) != Some(&'r') {
        return None;
    }
    let mut cursor = idx + 1;
    let mut hashes = 0usize;
    while chars.get(cursor) == Some(&'#') {
        hashes += 1;
        cursor += 1;
    }
    if chars.get(cursor) == Some(&'"') {
        Some(hashes)
    } else {
        None
    }
}

fn raw_string_ends_at(chars: &[char], idx: usize, hashes: usize) -> bool {
    if chars.get(idx) != Some(&'"') {
        return false;
    }
    (0..hashes).all(|offset| chars.get(idx + 1 + offset) == Some(&'#'))
}

fn brace_delta(code: &str) -> i32 {
    code.chars().fold(0, |depth, ch| match ch {
        '{' => depth + 1,
        '}' => depth - 1,
        _ => depth,
    })
}

fn first_error_line(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stderr.lines().chain(stdout.lines()) {
        if line.starts_with("error") {
            return line.chars().take(120).collect();
        }
    }
    output
        .status
        .code()
        .map(|code| format!("exit {code}"))
        .unwrap_or_else(|| "terminated by signal".to_string())
}

fn print_canary_output(canary: &str, output: &Output) {
    eprintln!("[api-canary] --- {canary} stdout ---");
    eprint!("{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("[api-canary] --- {canary} stderr ---");
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    eprintln!("[api-canary] --- end {canary} ---");
}

fn effective_mode(set: CanarySet, cli_mode: CanaryGateMode) -> CanaryGateMode {
    if std::env::var_os(set.warn_env).as_deref() == Some(std::ffi::OsStr::new("1")) {
        CanaryGateMode::Warn
    } else {
        cli_mode
    }
}

fn skip_enabled(env_name: &str) -> bool {
    std::env::var_os(env_name).as_deref() == Some(std::ffi::OsStr::new("1"))
}

fn timeout_seconds(env_name: &str, default: Option<u64>) -> Result<Option<u64>> {
    let Some(raw) = std::env::var_os(env_name) else {
        return Ok(default);
    };
    let raw = raw.to_string_lossy();
    let timeout = raw
        .parse::<u64>()
        .with_context(|| format!("{env_name} must be an integer number of seconds"))?;
    Ok(Some(timeout))
}

fn triggered(set: CanarySet, changed_files: &[String]) -> bool {
    changed_files
        .iter()
        .any(|file| set.triggers.iter().any(|trigger| file.starts_with(trigger)))
}

fn collect_changed_files(args: &CanaryGateArgs) -> Result<ChangeSet> {
    if args.staged {
        return git_change_set(GitChangeScope::Staged);
    }
    if args.changed_files.is_empty() {
        return git_change_set(GitChangeScope::Head);
    }
    Ok(ChangeSet {
        files: args
            .changed_files
            .iter()
            .map(|path| normalize_path(path))
            .collect(),
        whitespace_only_rs: false,
    })
}

fn git_change_set(scope: GitChangeScope) -> Result<ChangeSet> {
    git_change_set_in(Path::new("."), scope)
}

fn git_change_set_in(repo: &Path, scope: GitChangeScope) -> Result<ChangeSet> {
    let files = git_changed_files_in(repo, scope)?;
    let rs_any = git_rs_changed_files_in(repo, scope, false)?;
    let mut rs_semantic = Vec::new();
    for file in &rs_any {
        if git_diff_has_semantic_changes_in(repo, scope, file)? {
            rs_semantic.push(file.clone());
        }
    }
    Ok(ChangeSet {
        files,
        whitespace_only_rs: !rs_any.is_empty() && rs_semantic.is_empty(),
    })
}

fn git_changed_files_in(repo: &Path, scope: GitChangeScope) -> Result<Vec<String>> {
    let args: &[&str] = match scope {
        GitChangeScope::Head => &["diff", "--name-only", "HEAD"],
        GitChangeScope::Staged => &["diff", "--cached", "--name-only"],
    };
    git_name_only_in(repo, args)
}

fn git_rs_changed_files_in(
    repo: &Path,
    scope: GitChangeScope,
    ignore_all_space: bool,
) -> Result<Vec<String>> {
    let mut args = vec!["diff"];
    if matches!(scope, GitChangeScope::Staged) {
        args.push("--cached");
    }
    if ignore_all_space {
        args.push("--ignore-all-space");
    }
    args.push("--name-only");
    if matches!(scope, GitChangeScope::Head) {
        args.push("HEAD");
    }
    args.push("--");
    args.push("*.rs");
    git_name_only_in(repo, &args)
}

fn git_diff_has_semantic_changes_in(
    repo: &Path,
    scope: GitChangeScope,
    file: &str,
) -> Result<bool> {
    let mut args = vec!["diff"];
    if matches!(scope, GitChangeScope::Staged) {
        args.push("--cached");
    }
    args.extend(["--ignore-all-space", "--quiet"]);
    if matches!(scope, GitChangeScope::Head) {
        args.push("HEAD");
    }
    args.push("--");
    args.push(file);

    let output = Command::new("git")
        .args(&args)
        .current_dir(repo)
        .output()
        .with_context(|| format!("run git {}", args.join(" ")))?;
    match output.status.code() {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        _ => bail!(
            "git {} failed\nstdout:\n{}\nstderr:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    }
}

fn git_name_only_in(repo: &Path, args: &[&str]) -> Result<Vec<String>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .with_context(|| format!("run git {}", args.join(" ")))?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect())
}

fn normalize_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::ffi::OsString;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    struct EnvVarGuard {
        name: &'static str,
        old: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: &str) -> Self {
            let old = env::var_os(name);
            crate::env_guard::set_var(name, value);
            Self { name, old }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.old {
                Some(value) => crate::env_guard::set_var(self.name, value),
                None => crate::env_guard::remove_var(self.name),
            }
        }
    }

    fn git(repo: &Path, args: &[&str]) {
        let output = Command::new(real_git())
            .args(args)
            .current_dir(repo)
            .output()
            .expect("git command should run");
        assert!(
            output.status.success(),
            "git {:?} failed\nstdout:\n{}\nstderr:\n{}",
            args,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn real_git() -> &'static str {
        if Path::new("/usr/bin/git").is_file() {
            "/usr/bin/git"
        } else {
            "git"
        }
    }

    fn init_repo() -> tempfile::TempDir {
        let repo = tempfile::tempdir().expect("temp repo");
        git(repo.path(), &["init"]);
        git(repo.path(), &["config", "user.name", "Canary Test"]);
        git(
            repo.path(),
            &["config", "user.email", "canary-test@example.com"],
        );
        repo
    }

    fn write_repo_file(repo: &Path, path: &str, contents: &str) {
        let file = repo.join(path);
        if let Some(parent) = file.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(file, contents).expect("write repo file");
    }

    fn write_executable(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create executable parent dir");
        }
        fs::write(path, contents).expect("write executable");
        let mut permissions = fs::metadata(path).expect("stat executable").permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            permissions.set_mode(permissions.mode() | 0o111);
        }
        fs::set_permissions(path, permissions).expect("chmod executable");
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn read_workspace_file(path: &str) -> String {
        fs::read_to_string(repo_root().join(path)).unwrap_or_else(|err| {
            panic!("read workspace file {path}: {err}");
        })
    }

    /// Returns `true` only if every listed workspace file is present. When any is
    /// missing it logs a SKIP and returns `false` so the caller can `return`
    /// early. Several canary-gate fixtures are NOT checked into the repo — the
    /// `.pre-commit-local.d/*.sh` hooks are local-only developer hooks, and the
    /// `tests/tlc_comparison/enumerate_*_canary_specs.txt` lists are generated —
    /// so their absence is an environment gap, not a regression, and must SKIP
    /// (not panic) the test.
    fn require_workspace_files(paths: &[&str]) -> bool {
        for path in paths {
            if !repo_root().join(path).exists() {
                eprintln!(
                    "SKIP: workspace file absent ({path}) — local-only/generated canary fixture, \
                     not checked into the repo."
                );
                return false;
            }
        }
        true
    }

    fn active_canary_specs(contents: &str) -> Vec<&str> {
        contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect()
    }

    fn install_hook_and_common(repo: &Path, hook: &str) {
        let hook_path = repo.join(hook);
        if let Some(parent) = hook_path.parent() {
            fs::create_dir_all(parent).expect("create hook parent dir");
        }
        fs::write(&hook_path, read_workspace_file(hook)).expect("write hook fixture");

        let common_path = repo.join("scripts/canary_gate_common.sh");
        if let Some(parent) = common_path.parent() {
            fs::create_dir_all(parent).expect("create common parent dir");
        }
        fs::write(
            &common_path,
            read_workspace_file("scripts/canary_gate_common.sh"),
        )
        .expect("write common fixture");
    }

    fn install_fake_cargo(fake_bin: &Path) {
        write_executable(
            &fake_bin.join("cargo"),
            r#"#!/usr/bin/env bash
set -euo pipefail
: "${CANARY_TEST_CARGO_LOG:?}"
printf 'cwd=%s\n' "$PWD" >> "$CANARY_TEST_CARGO_LOG"
printf 'target=%s\n' "${CARGO_TARGET_DIR:-}" >> "$CANARY_TEST_CARGO_LOG"
printf 'args=' >> "$CANARY_TEST_CARGO_LOG"
for arg in "$@"; do
    printf '<%s>' "$arg" >> "$CANARY_TEST_CARGO_LOG"
done
printf '\n' >> "$CANARY_TEST_CARGO_LOG"
"#,
        );
    }

    fn run_hook_with_fake_cargo(
        repo: &Path,
        hook: &str,
        fake_bin: &Path,
        cargo_log: &Path,
    ) -> Output {
        let mut paths = vec![fake_bin.to_path_buf()];
        if let Some(existing_path) = env::var_os("PATH") {
            paths.extend(env::split_paths(&existing_path));
        }
        let joined_path = env::join_paths(paths).expect("join PATH");
        Command::new("bash")
            .arg(repo.join(hook))
            .current_dir(repo)
            .env("CANARY_TEST_CARGO_LOG", cargo_log)
            .env("PATH", joined_path)
            .env_remove("CARGO_TARGET_DIR")
            .output()
            .expect("run canary hook")
    }

    fn test_canary_set(
        skip_env: &'static str,
        warn_env: &'static str,
        timeout_env: &'static str,
    ) -> CanarySet {
        CanarySet {
            label: "test-canary",
            spec_list: "tests/tlc_comparison/does-not-exist.txt",
            triggers: &["crates/tla-check/src/"],
            skip_env,
            warn_env,
            timeout_env,
            default_timeout: Some(30),
            fail_message: "test canary failed",
        }
    }

    fn assert_no_script_gate_logic(path: &str, text: &str) {
        for forbidden in [
            "diagnose_specs.py",
            "python ",
            "python3",
            "cargo check",
            "STAGED_FILES",
            " jq",
        ] {
            assert!(
                !text.contains(forbidden),
                "{path} must stay a thin Rust CLI wrapper; found {forbidden:?}"
            );
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn eval_hook_invokes_rust_cli_directly_with_worker_target() {
        if !require_workspace_files(&[".pre-commit-local.d/eval_canary_gate.sh"]) {
            return;
        }
        let repo = tempfile::tempdir().expect("temp repo");
        install_hook_and_common(repo.path(), ".pre-commit-local.d/eval_canary_gate.sh");

        let fake_bin = repo.path().join("fake-bin");
        let cargo_log = repo.path().join("cargo.log");
        install_fake_cargo(&fake_bin);

        let output = run_hook_with_fake_cargo(
            repo.path(),
            ".pre-commit-local.d/eval_canary_gate.sh",
            &fake_bin,
            &cargo_log,
        );

        assert!(
            output.status.success(),
            "hook failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        // The orchestrator-aware worker-id branch was removed; the common
        // helper now resolves to `target/user` for any non-orchestrator caller.
        assert_eq!(
            fs::read_to_string(&cargo_log)
                .expect("read cargo log")
                .lines()
                .collect::<Vec<_>>(),
            vec![
                format!("cwd={}", repo.path().display()),
                format!("target={}", repo.path().join("target/user").display()),
                "args=<run><--profile><release-canary><--bin><ty><--><canary-gate><--kind><eval><--mode><enforce><--staged>".to_string(),
            ]
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn enumerate_hook_invokes_rust_cli_directly_with_worker_target() {
        if !require_workspace_files(&[".pre-commit-local.d/enumerate_canary_gate.sh"]) {
            return;
        }
        let repo = tempfile::tempdir().expect("temp repo");
        install_hook_and_common(repo.path(), ".pre-commit-local.d/enumerate_canary_gate.sh");

        let fake_bin = repo.path().join("fake-bin");
        let cargo_log = repo.path().join("cargo.log");
        install_fake_cargo(&fake_bin);

        let output = run_hook_with_fake_cargo(
            repo.path(),
            ".pre-commit-local.d/enumerate_canary_gate.sh",
            &fake_bin,
            &cargo_log,
        );

        assert!(
            output.status.success(),
            "hook failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        // The orchestrator-aware worker-id branch was removed; the common
        // helper now resolves to `target/user` for any non-orchestrator caller.
        assert_eq!(
            fs::read_to_string(&cargo_log)
                .expect("read cargo log")
                .lines()
                .collect::<Vec<_>>(),
            vec![
                format!("cwd={}", repo.path().display()),
                format!("target={}", repo.path().join("target/user").display()),
                "args=<run><--profile><release-canary><--bin><ty><--><canary-gate><--kind><enumerate><--mode><enforce><--staged>".to_string(),
            ]
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn eval_triggers_on_tla_check_source() {
        assert!(triggered(
            EVAL_SET,
            &["crates/tla-check/src/check/model_checker.rs".to_string()]
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn enumerate_ignores_unrelated_tla_check_source() {
        assert!(!triggered(
            ENUMERATE_SET,
            &["crates/tla-check/src/cache.rs".to_string()]
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn enumerate_triggers_on_model_checker_source() {
        assert!(triggered(
            ENUMERATE_SET,
            &["crates/tla-check/src/check/model_checker/bfs/compiled_bfs_loop.rs".to_string()]
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn enumerate_triggers_on_state_representation_sources() {
        for changed_file in [
            "crates/tla-check/src/state.rs",
            "crates/tla-check/src/state/array_state.rs",
        ] {
            assert!(
                triggered(ENUMERATE_SET, &[changed_file.to_string()]),
                "{changed_file} should trigger enumerate canaries"
            );
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn enumerate_triggers_on_native_lowering_sources() {
        for changed_file in [
            "crates/tla-ir/src/lower/set_ops.rs",
            "crates/tla-trust-cg/src/lower.rs",
            "crates/tla-trust-cg/src/trust_ir_lower.rs",
        ] {
            assert!(
                triggered(ENUMERATE_SET, &[changed_file.to_string()]),
                "{changed_file} should trigger enumerate canaries"
            );
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn enumerate_canary_gate_is_bounded_and_splits_slow_specs() {
        if !require_workspace_files(&[ENUMERATE_LIST, ENUMERATE_SLOW_LIST]) {
            return;
        }
        assert_eq!(
            ENUMERATE_SET.default_timeout,
            Some(ENUMERATE_DEFAULT_TIMEOUT)
        );

        let fast = read_workspace_file(ENUMERATE_LIST);
        let fast_specs = active_canary_specs(&fast);
        assert!(
            !fast_specs.contains(&"SlidingPuzzles"),
            "SlidingPuzzles is a known slow enumerate diagnostic and must not block the fast gate"
        );

        let slow = read_workspace_file(ENUMERATE_SLOW_LIST);
        let slow_specs = active_canary_specs(&slow);
        assert!(
            slow_specs.contains(&"SlidingPuzzles"),
            "the slow-list recategorization should keep SlidingPuzzles visible for manual runs"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn all_selects_changed_file_sets_and_standalone_gates() {
        let plan = selected_plan(CanaryGateKind::All);

        assert_eq!(plan.diagnose_sets.len(), 2);
        assert_eq!(plan.diagnose_sets[0].spec_list, EVAL_LIST);
        assert_eq!(plan.diagnose_sets[1].spec_list, ENUMERATE_LIST);
        assert_eq!(
            plan.standalone_gates,
            &[StandaloneGate::Api, StandaloneGate::SilentError][..]
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn api_kind_does_not_use_spec_diagnose_sets() {
        let plan = selected_plan(CanaryGateKind::Api);

        assert!(plan.diagnose_sets.is_empty());
        assert_eq!(plan.standalone_gates, &[StandaloneGate::Api][..]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn silent_error_kind_does_not_use_spec_diagnose_sets() {
        let plan = selected_plan(CanaryGateKind::SilentError);

        assert!(plan.diagnose_sets.is_empty());
        assert_eq!(plan.standalone_gates, &[StandaloneGate::SilentError][..]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn silent_error_scan_reports_eval_entry_ok_coercion() {
        let repo = tempfile::tempdir().expect("temp repo");
        write_repo_file(
            repo.path(),
            "crates/tla-check/src/check/model_checker/example.rs",
            "fn demo() {\n    if let Ok(Value::Bool(true)) = crate::eval::eval_entry(&ctx, pred) {\n    }\n}\n",
        );

        let findings = scan_silent_error_coercion(repo.path()).expect("scan silent error coercion");

        assert_eq!(findings.len(), 1);
        assert_eq!(
            findings[0].path,
            PathBuf::from("crates/tla-check/src/check/model_checker/example.rs")
        );
        assert_eq!(findings[0].line, 2);
        assert!(
            findings[0].description.contains("if let Ok(...) = eval"),
            "{:?}",
            findings[0]
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn silent_error_scan_ignores_comments_literals_tests_and_audited_lines() {
        let repo = tempfile::tempdir().expect("temp repo");
        write_repo_file(
            repo.path(),
            "crates/tla-check/src/check/model_checker/example.rs",
            r##"//! eval_entry(ctx, expr).ok()
const DOC: &str = "eval_entry(ctx, expr).unwrap_or_default()";
const RAW: &str = r#"if let Ok(v) = eval_entry(ctx, expr)"#;
fn audited() {
    // if let Ok(v) = eval_entry(ctx, expr) {}
    /*
       eval_entry(ctx, expr).unwrap_or(false)
    */
    let _ = eval_entry(ctx, expr).ok(); // Part of #1433: legacy audited site
}

#[cfg(test)]
mod tests {
    fn allowed_in_tests() {
        if let Ok(v) = eval_entry(ctx, expr) {
            drop(v);
        }
    }
}
"##,
        );
        write_repo_file(
            repo.path(),
            "crates/tla-check/src/check/model_checker/tests.rs",
            "fn test_helper() { let _ = eval_entry(ctx, expr).unwrap_or_default(); }\n",
        );

        let findings = scan_silent_error_coercion(repo.path()).expect("scan silent error coercion");

        assert!(findings.is_empty(), "{findings:#?}");
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn silent_error_scan_reports_eval_entry_unwrap_default() {
        let repo = tempfile::tempdir().expect("temp repo");
        write_repo_file(
            repo.path(),
            "crates/tla-check/src/enumerate/example.rs",
            "fn demo() {\n    let _ = eval_entry(ctx, expr).unwrap_or_default();\n}\n",
        );

        let findings = scan_silent_error_coercion(repo.path()).expect("scan silent error coercion");

        assert_eq!(findings.len(), 1);
        assert_eq!(
            findings[0].description,
            "eval_entry().unwrap_or_default() - use eval_speculative()"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn skip_env_returns_skipped_before_spec_list_check() {
        let _lock = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock");
        let _guard = EnvVarGuard::set("TY_CANARY_GATE_TEST_SKIP", "1");
        let set = test_canary_set(
            "TY_CANARY_GATE_TEST_SKIP",
            "TY_CANARY_GATE_TEST_WARN",
            "TY_CANARY_GATE_TEST_TIMEOUT",
        );
        let changed_files = vec!["crates/tla-check/src/check/model_checker.rs".to_string()];

        let outcome = run_set(set, CanaryGateMode::Enforce, &changed_files)
            .expect("skip env should not require the spec list to exist");

        assert_eq!(outcome, SetOutcome::Skipped);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn warn_env_downgrades_enforce_mode_in_rust_gate() {
        let _lock = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock");
        let _guard = EnvVarGuard::set("TY_CANARY_GATE_TEST_WARN", "1");
        let set = test_canary_set(
            "TY_CANARY_GATE_TEST_SKIP",
            "TY_CANARY_GATE_TEST_WARN",
            "TY_CANARY_GATE_TEST_TIMEOUT",
        );

        assert_eq!(
            effective_mode(set, CanaryGateMode::Enforce),
            CanaryGateMode::Warn
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn timeout_env_is_parsed_by_rust_gate() {
        let _lock = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock");
        let _guard = EnvVarGuard::set("TY_CANARY_GATE_TEST_TIMEOUT", "not-a-number");

        let err = timeout_seconds("TY_CANARY_GATE_TEST_TIMEOUT", Some(30))
            .expect_err("invalid timeout should fail");

        assert!(
            err.to_string()
                .contains("TY_CANARY_GATE_TEST_TIMEOUT must be an integer number of seconds"),
            "{err:#}"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn shell_canary_gate_surface_stays_thin_rust_cli_wrappers() {
        if !require_workspace_files(&[
            ".pre-commit-local.d/eval_canary_gate.sh",
            ".pre-commit-local.d/enumerate_canary_gate.sh",
        ]) {
            return;
        }
        let eval_script = read_workspace_file("scripts/check_eval_canaries.sh");
        assert!(eval_script.contains("source \"$SCRIPT_DIR/canary_gate_common.sh\""));
        assert!(eval_script.contains("cargo run --profile release-canary --bin ty --"));
        assert!(eval_script.contains("canary-gate --kind eval \"$@\""));
        assert_no_script_gate_logic("scripts/check_eval_canaries.sh", &eval_script);

        let enumerate_script = read_workspace_file("scripts/check_enumerate_canaries.sh");
        assert!(enumerate_script.contains("source \"$SCRIPT_DIR/canary_gate_common.sh\""));
        assert!(enumerate_script.contains("cargo run --profile release-canary --bin ty --"));
        assert!(enumerate_script.contains("canary-gate --kind enumerate \"$@\""));
        assert_no_script_gate_logic("scripts/check_enumerate_canaries.sh", &enumerate_script);

        let api_script = read_workspace_file("scripts/check_api_canary_gate.sh");
        assert!(api_script.contains("source \"$SCRIPT_DIR/canary_gate_common.sh\""));
        assert!(api_script.contains("cargo run --profile release-canary --bin ty --"));
        assert!(api_script.contains("canary-gate --kind api"));
        assert_no_script_gate_logic("scripts/check_api_canary_gate.sh", &api_script);

        let silent_error_script = read_workspace_file("scripts/check_silent_error_coercion.sh");
        assert!(silent_error_script.contains("source \"$SCRIPT_DIR/canary_gate_common.sh\""));
        assert!(silent_error_script.contains("cargo run --profile release-canary --bin ty --"));
        assert!(silent_error_script.contains("canary-gate --kind silent-error"));
        assert_no_script_gate_logic(
            "scripts/check_silent_error_coercion.sh",
            &silent_error_script,
        );

        let quality_gate = read_workspace_file("scripts/check_code_quality_gate.sh");
        assert!(quality_gate.contains("source \"$SCRIPT_DIR/canary_gate_common.sh\""));
        assert!(
            quality_gate.contains("cargo run --profile release-canary --bin ty --"),
            "scripts/check_code_quality_gate.sh must run the Rust canary-gate directly"
        );
        assert!(quality_gate.contains("run_canary_gate --kind api --mode enforce"));
        assert!(quality_gate.contains("run_canary_gate --kind silent-error --mode enforce"));
        assert!(quality_gate.contains("rust-function-span-scan \"$@\""));
        assert!(!quality_gate.contains("scripts/check_api_canary_gate.sh"));
        assert!(!quality_gate.contains("scripts/check_silent_error_coercion.sh"));
        assert!(!quality_gate.contains("python3 scripts/check_silent_error_coercion.py"));
        let legacy_span_script = format!("rust_function_span_scan{}", ".py");
        assert!(!quality_gate.contains(&legacy_span_script));

        let eval_hook = read_workspace_file(".pre-commit-local.d/eval_canary_gate.sh");
        assert!(eval_hook.contains("ty-pre-commit-timeout: 300"));
        assert!(eval_hook.contains("source \"$COMMON\""));
        assert!(eval_hook.contains("cargo run --profile release-canary --bin ty --"));
        assert!(eval_hook.contains("canary-gate --kind eval --mode enforce --staged"));
        assert!(!eval_hook.contains("check_eval_canaries.sh"));
        assert_no_script_gate_logic(".pre-commit-local.d/eval_canary_gate.sh", &eval_hook);

        let enumerate_hook = read_workspace_file(".pre-commit-local.d/enumerate_canary_gate.sh");
        assert!(enumerate_hook.contains("ty-pre-commit-timeout: 300"));
        assert!(enumerate_hook.contains("source \"$COMMON\""));
        assert!(enumerate_hook.contains("cargo run --profile release-canary --bin ty --"));
        assert!(enumerate_hook.contains("canary-gate --kind enumerate --mode enforce --staged"));
        assert!(!enumerate_hook.contains("check_enumerate_canaries.sh"));
        assert_no_script_gate_logic(
            ".pre-commit-local.d/enumerate_canary_gate.sh",
            &enumerate_hook,
        );

        let common = read_workspace_file("scripts/canary_gate_common.sh");
        assert!(common.contains("resolve_canary_target_dir()"));
        assert!(!common.contains("cargo run"));
        assert!(!common.contains("cargo build"));
        assert_no_script_gate_logic("scripts/canary_gate_common.sh", &common);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn legacy_python_silent_error_path_is_removed() {
        assert!(
            !repo_root()
                .join("scripts/check_silent_error_coercion.py")
                .exists(),
            "silent-error policy is Rust-owned; do not restore the Python gate path"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn legacy_python_function_span_path_is_removed() {
        let legacy_span_script = format!("rust_function_span_scan{}", ".py");
        assert!(
            !repo_root()
                .join("scripts")
                .join(legacy_span_script)
                .exists(),
            "function span policy is Rust-owned; do not restore the Python gate path"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn no_python_or_jq_canary_or_supremacy_acceptance_gate_scripts() {
        for dir in ["scripts", ".pre-commit-local.d"] {
            assert_no_python_or_jq_gate_surfaces_in(dir);
        }
    }

    fn assert_no_python_or_jq_gate_surfaces_in(dir: &str) {
        let path = repo_root().join(dir);
        // An absent directory (e.g. the local-only `.pre-commit-local.d/`) trivially
        // contains no Python/JQ gate surfaces — the invariant holds vacuously.
        if !path.exists() {
            eprintln!("SKIP {dir}: directory absent (local-only); no gate surfaces to check.");
            return;
        }
        for entry in fs::read_dir(&path).unwrap_or_else(|err| panic!("read {dir} dir: {err}")) {
            let entry = entry.expect("read scripts entry");
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("script filename")
                .to_ascii_lowercase();
            let gate_like = (name.contains("canary")
                || name.contains("supremacy")
                || name.contains("silent_error_coercion"))
                && (name.contains("gate")
                    || name.starts_with("check_")
                    || name.contains("acceptance"));
            if !gate_like {
                continue;
            }

            // `name` is already `to_ascii_lowercase()`'d above, so the
            // extension comparison is effectively case-insensitive.
            #[allow(clippy::case_sensitive_file_extension_comparisons)]
            let bad_ext = name.ends_with(".py") || name.ends_with(".jq");
            assert!(
                !bad_ext,
                "{} would create a Python/JQ canary or supremacy gate surface; use the Rust CLI",
                path.display()
            );
            let text = fs::read_to_string(&path).expect("read gate script");
            for forbidden in ["python ", "python3", " jq", "jq "] {
                assert!(
                    !text.contains(forbidden),
                    "{} must not add Python/JQ canary or supremacy gate logic; found {forbidden:?}",
                    path.display()
                );
            }
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn explicit_changed_files_do_not_apply_whitespace_skip() {
        let args = CanaryGateArgs {
            kind: CanaryGateKind::Eval,
            mode: CanaryGateMode::Enforce,
            verbose: false,
            staged: false,
            changed_files: vec![PathBuf::from("crates/tla-eval/src/binding_chain.rs")],
        };

        let changes = collect_changed_files(&args).expect("collect explicit changes");

        assert_eq!(changes.files, vec!["crates/tla-eval/src/binding_chain.rs"]);
        assert!(!changes.whitespace_only_rs);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn staged_change_set_detects_whitespace_only_rust_changes() {
        let repo = init_repo();
        write_repo_file(
            repo.path(),
            "crates/tla-eval/src/binding_chain.rs",
            "pub fn value() -> i32 { 1 }\n",
        );
        git(repo.path(), &["add", "."]);
        git(repo.path(), &["commit", "-m", "initial"]);

        write_repo_file(
            repo.path(),
            "crates/tla-eval/src/binding_chain.rs",
            "pub  fn value() -> i32 { 1 }\n",
        );
        git(repo.path(), &["add", "."]);

        let changes =
            git_change_set_in(repo.path(), GitChangeScope::Staged).expect("staged changes");

        assert_eq!(changes.files, vec!["crates/tla-eval/src/binding_chain.rs"]);
        assert!(changes.whitespace_only_rs);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn staged_change_set_detects_semantic_rust_changes() {
        let repo = init_repo();
        write_repo_file(
            repo.path(),
            "crates/tla-check/src/enumerate/unified.rs",
            "pub fn value() -> i32 { 1 }\n",
        );
        git(repo.path(), &["add", "."]);
        git(repo.path(), &["commit", "-m", "initial"]);

        write_repo_file(
            repo.path(),
            "crates/tla-check/src/enumerate/unified.rs",
            "pub fn value() -> i32 { 2 }\n",
        );
        git(repo.path(), &["add", "."]);

        let changes =
            git_change_set_in(repo.path(), GitChangeScope::Staged).expect("staged changes");

        assert_eq!(
            changes.files,
            vec!["crates/tla-check/src/enumerate/unified.rs"]
        );
        assert!(!changes.whitespace_only_rs);
    }
}
